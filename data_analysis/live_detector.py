import paho.mqtt.client as mqtt
import json
import numpy as np
import joblib
import time
from collections import deque
from datetime import datetime

# ------------------- Load trained model -------------------
try:
    model = joblib.load("model/mahalanobis_simple.pkl")
    center = model["center"]
    inv_cov = model["inv_cov"]
    features = model["features"]
    print("Model loaded → mahalanobis_simple.pkl")
    print(f"Features: {features}")
except FileNotFoundError:
    exit(1)
    
# ------------------- Detector settings -------------------
window = deque(maxlen=30)  # 30 seconds smoothing
ALARM_THRESHOLD = 10.0
CLEAR_THRESHOLD = 8.0
MIN_ALARM_FRACTION = 0.60
SETTLING_TIME = 60


alarm_active = False
msg_count = 0
TP = FP = TN = FN = 0

# Track setpoints and changes
last_setpoints = None
setpoint_change_time = None


# ------------------- Feature extraction -------------------
def flatten_payload(payload):
    """Extract normalized features that are robust to setpoint changes"""
    try:
        # Get current values
        temp_mean = payload["temperature_C"]["mean"]
        ph_mean = payload["pH"]["mean"]
        rpm_mean = payload["rpm"]["mean"]

        # Get setpoints
        setpoint_temp = payload["setpoints"]["temperature_C"]
        setpoint_ph = payload["setpoints"]["pH"]
        setpoint_rpm = payload["setpoints"]["rpm"]

        # Calculate NORMALIZED errors (setpoint-invariant)
        flat = {
            "temp_error_norm": (temp_mean - setpoint_temp) / setpoint_temp,
            "ph_error_norm": (ph_mean - setpoint_ph) / setpoint_ph,
            "rpm_error_norm": (rpm_mean - setpoint_rpm) / setpoint_rpm,
            "temp_range": payload["temperature_C"]["max"]
            - payload["temperature_C"]["min"],
            "ph_range": payload["pH"]["max"] - payload["pH"]["min"],
            "rpm_range": payload["rpm"]["max"] - payload["rpm"]["min"],
            "heater_pwm": payload["actuators_avg"]["heater_pwm"],
            "motor_pwm": payload["actuators_avg"]["motor_pwm"],
            "acid_pwm": payload["actuators_avg"]["acid_pwm"],
            "base_pwm": payload["actuators_avg"]["base_pwm"],
            "acid_dose_l": payload["dosing_l"]["acid"],
            "base_dose_l": payload["dosing_l"]["base"],
        }

        # Add setpoint info for change detection
        flat["setpoints"] = (setpoint_temp, setpoint_ph, setpoint_rpm)

        return flat
    except Exception as e:
        print(f"Bad payload → skipped: {e}")
        return None


# ------------------- MQTT callbacks (MQTTv5 compatible) -------------------
def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print(f"Connected to UCL broker")
        # Test on the hardest stream
        topic = "bioreactor_sim/variable_setpoints/telemetry/summary"
        client.subscribe(topic)
        print(f"Subscribed to → {topic}")
        print("=" * 80)
        print("IMPROVED DETECTOR - Setpoint-robust with settling time")
        print("=" * 80)
    else:
        print(f"Connection failed (code {rc})")


def on_message(client, userdata, msg):
    global alarm_active, TP, FP, TN, FN, msg_count
    global last_setpoints, setpoint_change_time

    msg_count += 1

    try:
        payload = json.loads(msg.payload.decode("utf-8"))
        flat = flatten_payload(payload)
        if flat is None:
            return

        # Detect setpoint changes
        current_setpoints = flat["setpoints"]
        setpoint_changed = False

        if last_setpoints is not None:
            if last_setpoints != current_setpoints:
                setpoint_changed = True
                setpoint_change_time = time.time()
                print(f"\n{'!'*80}")
                print(f"SETPOINT CHANGE DETECTED at message {msg_count}")
                print(
                    f"Old: Temp={last_setpoints[0]:.1f}, pH={last_setpoints[1]:.1f}, RPM={last_setpoints[2]:.1f}"
                )
                print(
                    f"New: Temp={current_setpoints[0]:.1f}, pH={current_setpoints[1]:.1f}, RPM={current_setpoints[2]:.1f}"
                )
                print(f"Entering {SETTLING_TIME}s settling period...")
                print(f"{'!'*80}\n")

        last_setpoints = current_setpoints

        # Calculate Mahalanobis distance
        x = np.array([flat[f] for f in features])
        diff = x - center
        mahal_dist = np.sqrt(diff @ inv_cov @ diff)

        # Add to rolling window
        window.append(mahal_dist)

        # Check if we're in settling period after setpoint change
        in_settling = False
        if setpoint_change_time is not None:
            time_since_change = time.time() - setpoint_change_time
            if time_since_change < SETTLING_TIME:
                in_settling = True

        # Decision logic with hysteresis (only if not settling)
        if not in_settling:
            if len(window) >= 15:
                frac_high = sum(d > ALARM_THRESHOLD for d in window) / len(window)
                if frac_high >= MIN_ALARM_FRACTION:
                    alarm_active = True
                elif mahal_dist < CLEAR_THRESHOLD:
                    alarm_active = False
        else:
            # During settling, gradually allow alarm to clear if distance is low
            if mahal_dist < CLEAR_THRESHOLD:
                alarm_active = False

        # Ground truth (convert fault dict/list to string safely)
        faults_raw = payload.get("faults", {}).get("last_active", [])

        # Handle both dict format and list format
        if isinstance(faults_raw, list):
            if len(faults_raw) > 0 and isinstance(faults_raw[0], dict):
                faults_str = ", ".join(f["name"] for f in faults_raw)
            else:
                faults_str = (
                    ", ".join(str(f) for f in faults_raw) if faults_raw else "none"
                )
        else:
            faults_str = str(faults_raw) if faults_raw else "none"

        real_fault = bool(faults_raw)

        # Update counters (only if not in settling period for fair evaluation)
        if not in_settling:
            if alarm_active and real_fault:
                TP += 1
            elif alarm_active and not real_fault:
                FP += 1
            elif not alarm_active and not real_fault:
                TN += 1
            elif not alarm_active and real_fault:
                FN += 1

        # Display status
        status = "ALARM!!!" if alarm_active else "normal"
        settling_marker = " [SETTLING]" if in_settling else ""

        # Calculate precision, recall, F1 if we have data
        if (TP + FP) > 0:
            precision = TP / (TP + FP)
        else:
            precision = 0.0

        if (TP + FN) > 0:
            recall = TP / (TP + FN)
        else:
            recall = 0.0

        if (precision + recall) > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0

        print(
            f"[{msg_count:4d}] Dist={mahal_dist:6.2f} | {status:10}{settling_marker:12} | "
            f"TP:{TP:3d} FP:{FP:3d} FN:{FN:3d} TN:{TN:4d} | "
            f"P={precision:.2f} R={recall:.2f} F1={f1:.2f} | {faults_str}"
        )

    except Exception as e:
        print(f"Processing error: {e}")
        import traceback

        traceback.print_exc()


# ------------------- Start MQTT client -------------------
from config import BROKER_HOST, BROKER_PORT

client = mqtt.Client(protocol=mqtt.MQTTv5)
client.on_connect = on_connect
client.on_message = on_message

print("Starting IMPROVED live Mahalanobis anomaly detector...")
print("Features: normalized errors + ranges + actuators")
print("Waiting for faults on variable_setpoints stream...")
print("-" * 80)

try:
    client.connect(BROKER_HOST, BROKER_PORT, keepalive=60)
    client.loop_forever()
except KeyboardInterrupt:
    print("\n\nDetector stopped by user")
    print("=" * 80)
    print("FINAL STATISTICS")
    print("=" * 80)
    print(f"Total messages: {msg_count}")
    print(f"True Positives:  {TP}")
    print(f"False Positives: {FP}")
    print(f"False Negatives: {FN}")
    print(f"True Negatives:  {TN}")
    if (TP + FP) > 0:
        print(f"Precision: {TP/(TP+FP):.3f}")
    if (TP + FN) > 0:
        print(f"Recall:    {TP/(TP+FN):.3f}")
    print("=" * 80)
