import paho.mqtt.client as mqtt
import json
import pandas as pd
from datetime import datetime
import os
import sys
import numpy as np
from scipy.spatial.distance import mahalanobis

# From training: features and skewed cols
FEATURE_COLS = [
    "temp_dev",
    "ph_dev",
    "rpm_dev",  # Deviations
    "temp_min",
    "temp_max",
    "ph_min",
    "ph_max",
    "rpm_min",
    "rpm_max",  # Variability
    "heater_pwm",
    "motor_pwm",
    "acid_pwm",
    "base_pwm",  # Actuators
    "acid_dose_l",
    "base_dose_l",  # Dosing
]
SKEWED_COLS = [
    "ph_mean",
    "ph_min",
    "ph_max",
    "ph_dev",
    "acid_pwm",
    "base_pwm",
    "acid_dose_l",
    "base_dose_l",
]

# Model paths (adjust if needed)
MODEL_DIR = "./model"
MU_PATH = os.path.join(MODEL_DIR, "mu.npy")
INV_COV_PATH = os.path.join(MODEL_DIR, "inv_cov.npy")
SHIFT_VALUES_PATH = os.path.join(MODEL_DIR, "shift_values.npy")

HIGH_THRESHOLD = 1067  # Trigger alarm
LOW_THRESHOLD = 800

# Globals
mean_vector = np.load(MU_PATH)
inv_cov_matrix = np.load(INV_COV_PATH)

# Load shift values if they exist (for consistent preprocessing)
# If not available, we'll compute them on-the-fly (less ideal but workable)
try:
    shift_values = np.load(SHIFT_VALUES_PATH, allow_pickle=True).item()
    print("Loaded shift values from training")
except:
    shift_values = None
    print("WARNING: No shift values found - will compute on-the-fly")

is_anomaly = False  # Hysteresis state
tp, tn, fp, fn = 0, 0, 0, 0  # Scores
df_buffer = pd.DataFrame()
csv_filename = None

# From config
BROKER_HOST = "engf0001.cs.ucl.ac.uk"
BROKER_PORT = 1883
CONNECTION_TIMEOUT = 60
STREAM_TYPES = {
    "single_fault": "bioreactor_sim/single_fault/telemetry/summary",
    "nofaults": "bioreactor_sim/nofaults/telemetry/summary",
    "three_faults": "bioreactor_sim/three_faults/telemetry/summary",
    "variable_setpoints": "bioreactor_sim/variable_setpoints/telemetry/summary",
}


def flatten_payload(payload):
    try:
        flat = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "temp_mean": payload["temperature_C"]["mean"],
            "temp_min": payload["temperature_C"]["min"],
            "temp_max": payload["temperature_C"]["max"],
            "ph_mean": payload["pH"]["mean"],
            "ph_min": payload["pH"]["min"],
            "ph_max": payload["pH"]["max"],
            "rpm_mean": payload["rpm"]["mean"],
            "rpm_min": payload["rpm"]["min"],
            "rpm_max": payload["rpm"]["max"],
            "heater_pwm": payload["actuators_avg"]["heater_pwm"],
            "motor_pwm": payload["actuators_avg"]["motor_pwm"],
            "acid_pwm": payload["actuators_avg"]["acid_pwm"],
            "base_pwm": payload["actuators_avg"]["base_pwm"],
            "acid_dose_l": payload["dosing_l"]["acid"],
            "base_dose_l": payload["dosing_l"]["base"],
            "setpoint_temp": payload["setpoints"]["temperature_C"],
            "setpoint_ph": payload["setpoints"]["pH"],
            "setpoint_rpm": payload["setpoints"]["rpm"],
            "faults_active": (
                ", ".join(
                    [
                        f.get("type", str(f)) if isinstance(f, dict) else str(f)
                        for f in payload["faults"]["last_active"]
                    ]
                )
                if payload["faults"]["last_active"]
                else "none"
            ),
            "has_fault": int(len(payload["faults"]["last_active"]) > 0),
        }
        # Compute deviations
        flat["temp_dev"] = flat["temp_mean"] - flat["setpoint_temp"]
        flat["ph_dev"] = flat["ph_mean"] - flat["setpoint_ph"]
        flat["rpm_dev"] = flat["rpm_mean"] - flat["setpoint_rpm"]
        return flat
    except KeyError as e:
        print(f"Warning: Missing key {e}, skipping")
        return None


def preprocess_sample(flat):
    """
    Apply same log transforms as training.
    CRITICAL: Must match the training preprocessing exactly!
    """
    epsilon = 1e-6

    for col in SKEWED_COLS:
        if col in flat:
            val = flat[col]

            # Use pre-computed shift values if available
            if shift_values and col in shift_values:
                shift = shift_values[col]
            else:
                # Fallback: compute shift on-the-fly (less accurate)
                shift = min(val, 0) if val < 0 else 0

            flat[col] = np.log(val - shift + epsilon)

    features = np.array([flat[col] for col in FEATURE_COLS])
    return features


def detect_anomaly(features):
    global is_anomaly
    dist_sq = mahalanobis(features, mean_vector, inv_cov_matrix) ** 2

    # Hysteresis: once anomaly detected, needs to drop below LOW threshold to clear
    if dist_sq > HIGH_THRESHOLD:
        is_anomaly = True
    elif dist_sq < LOW_THRESHOLD:
        is_anomaly = False

    return is_anomaly, dist_sq


def update_scores(pred, true):
    global tp, tn, fp, fn
    if pred and true:
        tp += 1
    elif not pred and not true:
        tn += 1
    elif pred and not true:
        fp += 1
    else:
        fn += 1


def identify_fault(features):
    """
    Identify likely fault type based on feature contributions.
    Uses contribution analysis to see which features deviate most.
    """
    residuals = features - mean_vector
    contributions = residuals * np.dot(inv_cov_matrix, residuals)

    # Get top 3 contributing features
    top_idxs = np.argsort(np.abs(contributions))[-3:]
    top_feats = [FEATURE_COLS[i] for i in top_idxs]

    # Heuristic mapping based on known fault signatures
    if "temp_dev" in top_feats or (
        "heater_pwm" in top_feats and "temp_min" in top_feats
    ):
        return "therm_voltage_bias"
    elif "ph_dev" in top_feats or "acid_pwm" in top_feats or "base_pwm" in top_feats:
        return "ph_offset_bias"
    elif "heater_pwm" in top_feats and "temp_dev" not in top_feats:
        return "heater_power_loss"

    return "Unknown"


def save_buffer(force=False):
    global df_buffer, csv_filename
    if len(df_buffer) >= 100 or force:
        header = not os.path.exists(csv_filename)
        df_buffer.to_csv(csv_filename, mode="a", header=header, index=False)
        df_buffer = pd.DataFrame()


def on_connect(client, userdata, flags, rc):
    global csv_filename, df_buffer
    if rc == 0:
        topic = userdata["topic"]
        client.subscribe(topic)
        print(f"Subscribed to {topic}")

        stream_name = topic.split("/")[-3]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"data_{stream_name}_{timestamp}.csv"
        df_buffer = pd.DataFrame()

        print(f"Data saving to: {csv_filename}")
        print(
            f"Detection active - Thresholds: HIGH={HIGH_THRESHOLD}, LOW={LOW_THRESHOLD}"
        )
        print(f"Trained on 23,638 fault-free samples from all streams")
    else:
        print(f"Connect failed: {rc}")


def on_message(client, userdata, msg):
    global df_buffer
    try:
        payload = json.loads(msg.payload.decode())
        flat = flatten_payload(payload)
        if flat is None:
            return

        # Preprocess for model
        features = preprocess_sample(flat)

        # Detect
        pred_anomaly, dist_sq = detect_anomaly(features)
        true_has_fault = flat["has_fault"]
        update_scores(pred_anomaly, true_has_fault)

        # Optional fault ID
        if pred_anomaly:
            fault_guess = identify_fault(features)
        else:
            fault_guess = "N/A"

        # Add to buffer
        flat["dist_sq"] = dist_sq
        flat["pred_anomaly"] = int(pred_anomaly)
        flat["fault_guess"] = fault_guess
        df_buffer = pd.concat([df_buffer, pd.DataFrame([flat])], ignore_index=True)
        save_buffer()

        # Print live with color coding
        count = len(df_buffer)
        pred_status = "🔴 ANOMALY" if pred_anomaly else "✓ Normal"
        true_status = "FAULT" if true_has_fault else "OK"

        # Add warning if prediction doesn't match reality
        match_indicator = "✓" if pred_anomaly == true_has_fault else "✗"

        print(
            f"[{count:4d}] {flat['timestamp']} | "
            f"{match_indicator} Pred: {pred_status:12} (d²={dist_sq:6.2f}) | "
            f"True: {true_status:5} | Active: {flat['faults_active']:30} | "
            f"ID: {fault_guess}"
        )

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


def print_scores():
    total = tp + tn + fp + fn
    if total > 0:
        acc = (tp + tn) / total
        prec = tp / (tp + fp) if tp + fp > 0 else 0
        rec = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

        print("\n" + "=" * 60)
        print("FINAL DETECTION SCORES")
        print("=" * 60)
        print(f"True Positives  (TP): {tp:5d}  | Correctly detected faults")
        print(f"True Negatives  (TN): {tn:5d}  | Correctly identified normal")
        print(f"False Positives (FP): {fp:5d}  | False alarms")
        print(f"False Negatives (FN): {fn:5d}  | Missed faults")
        print("-" * 60)
        print(f"Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
        print(
            f"Precision : {prec:.4f}  ({prec*100:.2f}%) - Of all alarms, how many were real"
        )
        print(
            f"Recall    : {rec:.4f}  ({rec*100:.2f}%) - Of all faults, how many detected"
        )
        print(f"F1 Score  : {f1:.4f}")
        print("=" * 60)


def on_disconnect(client, userdata, rc):
    save_buffer(force=True)
    print_scores()
    print("Disconnected.")


def connect_and_listen(stream_type="single_fault", duration=None):
    if stream_type not in STREAM_TYPES:
        print(f"Invalid stream: {stream_type}")
        print(f"Available: {list(STREAM_TYPES.keys())}")
        return

    topic = STREAM_TYPES[stream_type]
    client = mqtt.Client(userdata={"topic": topic})

    client.on_connect = on_connect
    client.on_message = on_message
    client.on_disconnect = on_disconnect

    try:
        print(f"Connecting to {BROKER_HOST}:{BROKER_PORT} - {stream_type}")
        client.connect(BROKER_HOST, BROKER_PORT, CONNECTION_TIMEOUT)

        if duration:
            client.loop_start()
            import time

            time.sleep(duration)
            client.loop_stop()
        else:
            client.loop_forever()
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        client.disconnect()


if __name__ == "__main__":
    stream = sys.argv[1] if len(sys.argv) > 1 else "single_fault"
    duration = int(sys.argv[2]) if len(sys.argv) > 2 else None

    print(f"\n{'='*60}")
    print(f"MAHALANOBIS ANOMALY DETECTOR")
    print(f"{'='*60}")
    print(f"Model trained on: 23,638 fault-free samples (all streams)")
    print(f"High threshold: {HIGH_THRESHOLD} (99.9% confidence)")
    print(f"Low threshold:  {LOW_THRESHOLD} (95% confidence)")
    print(f"{'='*60}\n")

    connect_and_listen(stream, duration)
