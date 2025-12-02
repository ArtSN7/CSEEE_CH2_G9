# live_detector_supervised_fixed.py
# Fixed live detector with proper feature engineering

import paho.mqtt.client as mqtt
import json
import pandas as pd
from datetime import datetime
import os
import sys
import numpy as np
from scipy.spatial.distance import mahalanobis
from joblib import load

# ================================
# CONFIGURATION & MODEL LOADING
# ================================
MODEL_DIR = "./model"

# Load models
rf_anomaly = load(os.path.join(MODEL_DIR, "rf_anomaly_model.joblib"))
rf_type = load(os.path.join(MODEL_DIR, "rf_type_model.joblib"))
le_fault = load(os.path.join(MODEL_DIR, "fault_label_encoder.joblib"))

# Load Mahalanobis model (for dist_sq feature)
mean_vector = np.load(os.path.join(MODEL_DIR, "mu.npy"))
inv_cov_matrix = np.load(os.path.join(MODEL_DIR, "inv_cov.npy"))

# Base features (for Mahalanobis)
BASE_FEATURE_COLS = [
    "temp_dev",
    "ph_dev",
    "rpm_dev",
    "temp_min",
    "temp_max",
    "ph_min",
    "ph_max",
    "rpm_min",
    "rpm_max",
    "heater_pwm",
    "motor_pwm",
    "acid_pwm",
    "base_pwm",
    "acid_dose_l",
    "base_dose_l",
]

# Enhanced features (for RandomForest)
ENHANCED_FEATURE_COLS = BASE_FEATURE_COLS + [
    "dist_sq",
    "temp_range",
    "ph_range",
    "rpm_range",
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

# MQTT config
BROKER_HOST = "engf0001.cs.ucl.ac.uk"
BROKER_PORT = 1883
CONNECTION_TIMEOUT = 60

STREAM_TYPES = {
    "single_fault": "bioreactor_sim/single_fault/telemetry/summary",
    "nofaults": "bioreactor_sim/nofaults/telemetry/summary",
    "three_faults": "bioreactor_sim/three_faults/telemetry/summary",
    "variable_setpoints": "bioreactor_sim/variable_setpoints/telemetry/summary",
}

# Globals
tp = tn = fp = fn = 0
df_buffer = pd.DataFrame()
csv_filename = None


is_anomaly = False
FAULT_THRESHOLD = 0.75  # Higher threshold to trigger
CLEAR_THRESHOLD = 0.50  # Lower threshold to clear

def flatten_payload(payload):
    """Extract and flatten MQTT payload."""
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

        # Compute range features
        flat["temp_range"] = flat["temp_max"] - flat["temp_min"]
        flat["ph_range"] = flat["ph_max"] - flat["ph_min"]
        flat["rpm_range"] = flat["rpm_max"] - flat["rpm_min"]

        return flat
    except KeyError as e:
        print(f"Warning: Missing key {e}, skipping message")
        return None


def preprocess_sample(flat):
    """Preprocess sample to match training."""
    epsilon = 1e-6

    # Log-transform skewed columns
    for col in SKEWED_COLS:
        if col in flat:
            val = flat[col]
            shift = min(val, 0) if val < 0 else 0
            flat[col] = np.log(val - shift + epsilon)

    # Compute Mahalanobis distance
    base_features = np.array([flat[col] for col in BASE_FEATURE_COLS])
    try:
        dist_sq = mahalanobis(base_features, mean_vector, inv_cov_matrix) ** 2
    except:
        dist_sq = 0  # Handle edge cases
    flat["dist_sq"] = dist_sq

    # Build full feature vector
    features = np.array([flat[col] for col in ENHANCED_FEATURE_COLS])
    return features


def detect_anomaly(features):
    """Detect if sample is anomalous using RandomForest."""
    global is_anomaly

    pred = rf_anomaly.predict([features])[0]
    prob_fault = rf_anomaly.predict_proba([features])[0][1]

    # Apply hysteresis
    if prob_fault > FAULT_THRESHOLD:
        is_anomaly = True
    elif prob_fault < CLEAR_THRESHOLD:
        is_anomaly = False
    # Between thresholds: maintain current state

    return is_anomaly, prob_fault


def identify_fault(features):
    """Identify fault type if anomaly detected."""
    if not rf_anomaly.predict([features])[0]:
        return "N/A"

    fault_id = rf_type.predict([features])[0]
    fault_name = le_fault.inverse_transform([fault_id])[0]
    return "N/A" if fault_name == "normal" else fault_name


def update_scores(pred, true):
    """Update confusion matrix scores."""
    global tp, tn, fp, fn
    if pred and true:
        tp += 1
    elif not pred and not true:
        tn += 1
    elif pred and not true:
        fp += 1
    else:
        fn += 1


def save_buffer(force=False):
    """Save buffered data to CSV."""
    global df_buffer, csv_filename
    if len(df_buffer) >= 100 or force:
        header = not os.path.exists(csv_filename)
        df_buffer.to_csv(csv_filename, mode="a", header=header, index=False)
        df_buffer = pd.DataFrame()


def on_connect(client, userdata, flags, rc):
    """MQTT connection callback."""
    global csv_filename
    if rc == 0:
        topic = userdata["topic"]
        client.subscribe(topic)

        stream_name = topic.split("/")[-3]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"data_{stream_name}_{timestamp}.csv"

        print("=" * 70)
        print(f"Connected to {topic}")
        print(f"Data saving → {csv_filename}")
        print("Supervised RandomForest detector ACTIVE")
        print("=" * 70)
    else:
        print(f"Connection failed: {rc}")


def on_message(client, userdata, msg):
    """Process incoming MQTT message."""
    global df_buffer
    try:
        payload = json.loads(msg.payload.decode())
        flat = flatten_payload(payload)
        if flat is None:
            return

        # Preprocess and detect
        features = preprocess_sample(flat)
        pred_anomaly, prob_fault = detect_anomaly(features)
        fault_guess = identify_fault(features)

        # Update scores
        update_scores(pred_anomaly, flat["has_fault"])

        # Save to buffer
        flat["pred_anomaly"] = int(pred_anomaly)
        flat["prob_fault"] = prob_fault
        flat["fault_guess"] = fault_guess
        df_buffer = pd.concat([df_buffer, pd.DataFrame([flat])], ignore_index=True)
        save_buffer()

        # Live output
        count = len(df_buffer)
        pred_status = "🔴 FAULT" if pred_anomaly else "✓ Normal"
        true_status = "FAULT" if flat["has_fault"] else "OK"
        match = "✓" if pred_anomaly == flat["has_fault"] else "✗"

        print(
            f"[{count:4d}] {flat['timestamp']} | "
            f"{match} Pred: {pred_status:10} (p={prob_fault:.3f}) | "
            f"True: {true_status:5} | Active: {flat['faults_active']:30} | "
            f"ID: {fault_guess}"
        )

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


def print_scores():
    """Print final detection scores."""
    total = tp + tn + fp + fn
    if total == 0:
        return

    acc = (tp + tn) / total
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

    print("\n" + "=" * 70)
    print("FINAL DETECTION SCORES")
    print("=" * 70)
    print(f"True Positives  (TP): {tp:5d}  | Correctly detected faults")
    print(f"True Negatives  (TN): {tn:5d}  | Correctly identified normal")
    print(f"False Positives (FP): {fp:5d}  | False alarms")
    print(f"False Negatives (FN): {fn:5d}  | Missed faults")
    print("-" * 70)
    print(f"Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"Precision : {prec:.4f}  ({prec*100:.2f}%) - Alarm reliability")
    print(f"Recall    : {rec:.4f}  ({rec*100:.2f}%) - Fault detection rate")
    print(f"F1 Score  : {f1:.4f}")
    print("=" * 70)


def on_disconnect(client, userdata, rc):
    """MQTT disconnect callback."""
    save_buffer(force=True)
    print_scores()
    print("Disconnected.")


def connect_and_listen(stream_type="single_fault", duration=None):
    """Connect to MQTT broker and start listening."""
    if stream_type not in STREAM_TYPES:
        print(f"Invalid stream: {stream_type}")
        print(f"Available: {list(STREAM_TYPES.keys())}")
        return

    topic = STREAM_TYPES[stream_type]
    client = mqtt.Client(userdata={"topic": topic})

    client.on_connect = on_connect
    client.on_message = on_message
    client.on_disconnect = on_disconnect

    print(f"Connecting to {BROKER_HOST}:{BROKER_PORT} → {stream_type}")
    client.connect(BROKER_HOST, BROKER_PORT, CONNECTION_TIMEOUT)

    try:
        if duration:
            client.loop_start()
            import time

            time.sleep(duration)
            client.loop_stop()
        else:
            client.loop_forever()
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    finally:
        client.disconnect()


if __name__ == "__main__":
    stream = sys.argv[1] if len(sys.argv) > 1 else "single_fault"
    duration = int(sys.argv[2]) if len(sys.argv) > 2 else None

    print("\n" + "=" * 70)
    print("SUPERVISED ANOMALY DETECTOR (RandomForest)")
    print("=" * 70)
    print("Trained on all data streams with enhanced features")
    print("=" * 70 + "\n")

    connect_and_listen(stream, duration)
