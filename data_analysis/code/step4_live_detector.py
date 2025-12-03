# step4_live_detector.py
# Live detector using temporal-aware multi-label model
# Maintains rolling window of recent samples for pattern detection
# Uses exponential smoothing and hysteresis

import paho.mqtt.client as mqtt
import json
import pandas as pd
from datetime import datetime
import os
import sys
import numpy as np
from scipy.spatial.distance import mahalanobis
from joblib import load
from collections import deque
import pickle

# ================================
# CONFIGURATION
# ================================
MODEL_DIR = "./model"

# Load model
print("Loading models...")
rf_multi = load(os.path.join(MODEL_DIR, "rf_multi_temporal.joblib"))
mu_fixed = np.load(os.path.join(MODEL_DIR, "mu_fixed.npy"))
inv_cov_fixed = np.load(os.path.join(MODEL_DIR, "inv_cov_fixed.npy"))
mu_var = np.load(os.path.join(MODEL_DIR, "mu_var.npy"))
inv_cov_var = np.load(os.path.join(MODEL_DIR, "inv_cov_var.npy"))

with open(os.path.join(MODEL_DIR, "shift_values_fixed.pkl"), "rb") as f:
    shift_values = pickle.load(f)  # Use fixed for simplicity

with open(os.path.join(MODEL_DIR, "k_values_fixed.pkl"), "rb") as f:
    k_fixed = pickle.load(f)
with open(os.path.join(MODEL_DIR, "k_values_var.pkl"), "rb") as f:
    k_var = pickle.load(f)
k_heater = (k_fixed["k_heater"] + k_var["k_heater"]) / 2
k_acid = (k_fixed["k_acid"] + k_var["k_acid"]) / 2
k_base = (k_fixed["k_base"] + k_var["k_base"]) / 2

# Load feature list
with open(os.path.join(MODEL_DIR, "feature_list.txt"), "r") as f:
    ALL_FEATURES = [line.strip() for line in f]

print(f"✓ Loaded model with {len(ALL_FEATURES)} features")

# Feature groups
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

WINDOW_SIZE = 5
CORR_WINDOW = 30  # For live, use deque of size 30

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

# Thresholds
FAULT_THRESHOLD = 0.7  # Trigger alarm at 70% probability
CLEAR_THRESHOLD = 0.4  # Clear alarm when below 40%
SMOOTH_ALPHA = 0.7  # For exponential smoothing

FAULT_TYPES = ["therm_voltage_bias", "ph_offset_bias", "heater_power_loss"]

# ================================
# GLOBALS
# ================================
tp = tn = fp = fn = 0
is_anomaly = False
df_buffer = pd.DataFrame()
csv_filename = None
smoothed_prob = 0.0  # Initial smoothed probability

# Rolling windows
history_window = deque(maxlen=WINDOW_SIZE)  # For small rolling
corr_history = deque(maxlen=CORR_WINDOW)  # For correlations

# ================================
# FUNCTIONS
# ================================


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

        # Compute ranges
        flat["temp_range"] = flat["temp_max"] - flat["temp_min"]
        flat["ph_range"] = flat["ph_max"] - flat["ph_min"]
        flat["rpm_range"] = flat["rpm_max"] - flat["rpm_min"]

        return flat
    except KeyError as e:
        print(f"Warning: Missing key {e}")
        return None


def preprocess_sample(flat):
    """Apply log transforms matching training."""
    epsilon = 1e-6

    for col in SKEWED_COLS:
        if col in flat:
            shift = shift_values.get(col, 0)
            flat[col] = np.log(flat[col] + shift + epsilon)

    return flat


def compute_mahalanobis(flat, mu, inv_cov):
    """Compute Mahalanobis distance."""
    features = np.array([flat[col] for col in BASE_FEATURE_COLS])
    try:
        dist_sq = mahalanobis(features, mu, inv_cov) ** 2
    except:
        dist_sq = 0
    return dist_sq


def compute_temporal_features(flat, history, corr_hist):
    """Compute rolling window features from history."""
    if len(history) == 0:
        # First sample - initialize with zeros
        for feat in ["temp_dev", "ph_dev", "rpm_dev"]:
            flat[f"{feat}_rolling_mean"] = flat[feat]
            flat[f"{feat}_rolling_std"] = 0

        flat["dist_sq_rolling_mean"] = flat["dist_sq_min"]
        flat["dist_sq_rolling_std"] = 0
        flat["dist_sq_rolling_max"] = flat["dist_sq_min"]

        for feat in [
            "temp_dev",
            "ph_dev",
            "rpm_dev",
            "dist_sq_min",
            "heater_pwm",
            "acid_pwm",
            "base_pwm",
        ]:
            flat[f"{feat}_diff"] = 0

        flat["heater_temp_corr"] = 0
        flat["acid_ph_corr"] = 0
        flat["base_ph_corr"] = 0

        flat["heater_eff_ratio"] = 0
        flat["acid_eff_ratio"] = 0
        flat["base_eff_ratio"] = 0

        flat["heater_residual"] = 0
        flat["acid_residual"] = 0
        flat["base_residual"] = 0

        return flat

    # Convert history to DataFrame for computation
    hist_df = pd.DataFrame(list(history))

    # Rolling means and stds
    for feat in ["temp_dev", "ph_dev", "rpm_dev"]:
        flat[f"{feat}_rolling_mean"] = hist_df[feat].mean()
        flat[f"{feat}_rolling_std"] = hist_df[feat].std() if len(hist_df) > 1 else 0

    flat["dist_sq_rolling_mean"] = hist_df["dist_sq_min"].mean()
    flat["dist_sq_rolling_std"] = (
        hist_df["dist_sq_min"].std() if len(hist_df) > 1 else 0
    )
    flat["dist_sq_rolling_max"] = hist_df["dist_sq_min"].max()

    # Differences
    last_sample = history[-1]
    flat["temp_dev_diff"] = flat["temp_dev"] - last_sample["temp_dev"]
    flat["ph_dev_diff"] = flat["ph_dev"] - last_sample["ph_dev"]
    flat["rpm_dev_diff"] = flat["rpm_dev"] - last_sample["rpm_dev"]
    flat["dist_sq_diff"] = flat["dist_sq_min"] - last_sample["dist_sq_min"]
    flat["heater_pwm_diff"] = abs(flat["heater_pwm"] - last_sample["heater_pwm"])
    flat["acid_pwm_diff"] = abs(flat["acid_pwm"] - last_sample["acid_pwm"])
    flat["base_pwm_diff"] = abs(flat["base_pwm"] - last_sample["base_pwm"])

    # Cross-correlations (use corr_hist)
    corr_df = pd.DataFrame(list(corr_hist))
    flat["heater_temp_corr"] = (
        corr_df["heater_pwm"].corr(corr_df["temp_dev_diff"]) if len(corr_df) > 1 else 0
    )
    flat["acid_ph_corr"] = (
        corr_df["acid_pwm"].corr(corr_df["ph_dev_diff"]) if len(corr_df) > 1 else 0
    )
    flat["base_ph_corr"] = (
        corr_df["base_pwm"].corr(corr_df["ph_dev_diff"]) if len(corr_df) > 1 else 0
    )

    # Efficiency ratios
    epsilon = 1e-6
    flat["heater_eff_ratio"] = flat["heater_pwm"] / (
        abs(flat["temp_dev_diff"]) + epsilon
    )
    flat["acid_eff_ratio"] = flat["acid_pwm"] / (abs(flat["ph_dev_diff"]) + epsilon)
    flat["base_eff_ratio"] = flat["base_pwm"] / (abs(flat["ph_dev_diff"]) + epsilon)

    # Residuals
    flat["heater_residual"] = flat["heater_pwm"] - k_heater * abs(flat["temp_dev"])
    flat["acid_residual"] = flat["acid_pwm"] - k_acid * abs(flat["ph_dev"])
    flat["base_residual"] = flat["base_pwm"] - k_base * abs(flat["ph_dev"])

    return flat


def detect_anomaly(features):
    """Detect anomaly with multi-label model, smoothing, and hysteresis."""
    global is_anomaly, smoothed_prob

    # Get probs for each fault
    probs = rf_multi.predict_proba([features])
    probs = np.array(
        [p[0][1] for p in probs]
    )  # [prob_fault1, prob_fault2, prob_fault3]

    # Max prob as overall fault prob
    prob_fault = np.max(probs)

    # Exponential smoothing
    smoothed_prob = SMOOTH_ALPHA * prob_fault + (1 - SMOOTH_ALPHA) * smoothed_prob

    # Hysteresis
    if smoothed_prob > FAULT_THRESHOLD:
        is_anomaly = True
    elif smoothed_prob < CLEAR_THRESHOLD:
        is_anomaly = False

    return is_anomaly, smoothed_prob, probs


def identify_faults(probs):
    """Identify faults based on probs."""
    faults = [FAULT_TYPES[i] for i, p in enumerate(probs) if p > 0.5]
    return ", ".join(faults) if faults else "N/A"


def update_scores(pred, true):
    """Update confusion matrix (binary for simplicity)."""
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
    """Save buffered data."""
    global df_buffer, csv_filename
    if len(df_buffer) >= 100 or force:
        header = not os.path.exists(csv_filename)
        df_buffer.to_csv(csv_filename, mode="a", header=header, index=False)
        df_buffer = pd.DataFrame()


def on_connect(client, userdata, flags, rc):
    """MQTT connection callback."""
    global csv_filename, history_window, corr_history, smoothed_prob

    if rc == 0:
        topic = userdata["topic"]
        client.subscribe(topic)

        stream_name = topic.split("/")[-3]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"detected_{stream_name}_{timestamp}.csv"

        # Reset history and smoothing
        history_window.clear()
        corr_history.clear()
        smoothed_prob = 0.0

        print("=" * 70)
        print(f"Connected → {topic}")
        print(f"Data saving → {csv_filename}")
        print(
            f"Temporal-aware detector ACTIVE (window={WINDOW_SIZE}, corr_window={CORR_WINDOW})"
        )
        print("=" * 70)
    else:
        print(f"Connection failed: {rc}")


def on_message(client, userdata, msg):
    """Process incoming MQTT message."""
    global df_buffer, history_window, corr_history

    try:
        payload = json.loads(msg.payload.decode())
        flat = flatten_payload(payload)
        if flat is None:
            return

        # Preprocess
        flat = preprocess_sample(flat)

        # Compute dual Mahalanobis
        flat["dist_sq_fixed"] = compute_mahalanobis(flat, mu_fixed, inv_cov_fixed)
        flat["dist_sq_var"] = compute_mahalanobis(flat, mu_var, inv_cov_var)
        flat["dist_sq_min"] = min(flat["dist_sq_fixed"], flat["dist_sq_var"])

        # Add to corr_history (for larger window features)
        corr_hist_entry = {
            "heater_pwm": flat["heater_pwm"],
            "acid_pwm": flat["acid_pwm"],
            "base_pwm": flat["base_pwm"],
            "temp_dev_diff": 0,  # Will update after
            "ph_dev_diff": 0,
        }
        corr_history.append(corr_hist_entry)

        # Compute temporal features (updates diffs in-place)
        flat = compute_temporal_features(flat, history_window, corr_history)

        # Update corr_hist diffs now that they are computed
        corr_history[-1]["temp_dev_diff"] = flat["temp_dev_diff"]
        corr_history[-1]["ph_dev_diff"] = flat["ph_dev_diff"]

        # Add to small history AFTER computing
        history_window.append(
            {
                "temp_dev": flat["temp_dev"],
                "ph_dev": flat["ph_dev"],
                "rpm_dev": flat["rpm_dev"],
                "dist_sq_min": flat["dist_sq_min"],
                "heater_pwm": flat["heater_pwm"],
                "acid_pwm": flat["acid_pwm"],
                "base_pwm": flat["base_pwm"],
            }
        )

        # Extract features for model
        features = np.array([flat.get(f, 0) for f in ALL_FEATURES])

        # Detect
        pred_anomaly, smoothed_prob_fault, probs = detect_anomaly(features)
        fault_guess = identify_faults(probs)

        # Update scores (binary)
        update_scores(pred_anomaly, flat["has_fault"])

        # Save to buffer
        flat["pred_anomaly"] = int(pred_anomaly)
        flat["smoothed_prob_fault"] = smoothed_prob_fault
        flat["fault_guess"] = fault_guess
        df_buffer = pd.concat([df_buffer, pd.DataFrame([flat])], ignore_index=True)
        save_buffer()

        # Live output
        count = len(df_buffer) + (tp + tn + fp + fn)  # Approximate total
        pred_status = "🔴 FAULT" if pred_anomaly else "✓ Normal"
        true_status = "FAULT" if flat["has_fault"] else "OK"
        match = "✓" if pred_anomaly == flat["has_fault"] else "✗"

        print(
            f"[{count:4d}] {flat['timestamp']} | "
            f"{match} Pred: {pred_status:11} (p={smoothed_prob_fault:.3f}) | "
            f"True: {true_status:5} | Active: {flat['faults_active']:30} | "
            f"ID: {fault_guess}"
        )

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


def print_scores():
    """Print final scores."""
    total = tp + tn + fp + fn
    if total == 0:
        return

    acc = (tp + tn) / total
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

    print("\n" + "=" * 70)
    print("FINAL DETECTION SCORES (BINARY)")
    print("=" * 70)
    print(f"True Positives  (TP): {tp:5d}  | Correctly detected faults")
    print(f"True Negatives  (TN): {tn:5d}  | Correctly identified normal")
    print(f"False Positives (FP): {fp:5d}  | False alarms")
    print(f"False Negatives (FN): {fn:5d}  | Missed faults")
    print("-" * 70)
    print(f"Accuracy  : {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precision : {prec:.4f} ({prec*100:.2f}%) - Alarm reliability")
    print(f"Recall    : {rec:.4f} ({rec*100:.2f}%) - Fault detection rate")
    print(f"F1 Score  : {f1:.4f}")
    print("=" * 70)


def on_disconnect(client, userdata, rc):
    """MQTT disconnect callback."""
    save_buffer(force=True)
    print_scores()
    print("Disconnected.")


def connect_and_listen(stream_type="single_fault", duration=None):
    """Connect to MQTT and start listening."""
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


# ================================
# MAIN
# ================================

if __name__ == "__main__":
    stream = sys.argv[1] if len(sys.argv) > 1 else "single_fault"
    duration = int(sys.argv[2]) if len(sys.argv) > 2 else None

    print("\n" + "=" * 70)
    print("TEMPORAL-AWARE ANOMALY DETECTOR")
    print("=" * 70)
    print(f"Using {len(ALL_FEATURES)} features including temporal patterns")
    print(f"Rolling window size: {WINDOW_SIZE} samples (corr: {CORR_WINDOW})")
    print(f"Thresholds: Trigger={FAULT_THRESHOLD}, Clear={CLEAR_THRESHOLD}")
    print(f"Smoothing alpha: {SMOOTH_ALPHA}")
    print("=" * 70 + "\n")

    connect_and_listen(stream, duration)
