# step1_train_baseline.py
# Train Mahalanobis models: fixed (nofaults) + dynamic (variable_setpoints normals)
# This establishes "normal" baselines robust to setpoint changes
# Also computes proportional constants (k) for residuals

import pandas as pd
import numpy as np
from scipy.spatial.distance import mahalanobis
import os
import pickle

# ================================
# CONFIGURATION
# ================================
FIXED_CSV = (
    "./data/data_nofaults_20251203_001104.csv"  # Update with your actual filename
)
VAR_CSV = "./data/data_variable_setpoints_20251203_001146.csv"  # Update with your actual filename
MODEL_DIR = "./model"
os.makedirs(MODEL_DIR, exist_ok=True)

# Base features for Mahalanobis (focus on relations/dynamics for setpoint robustness)
BASE_FEATURE_COLS = [
    "temp_dev",
    "ph_dev",
    "rpm_dev",  # Deviations from setpoint
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
    "base_dose_l",  # Dosing volumes
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

# ================================
# FUNCTIONS
# ================================


def load_and_preprocess(csv_path, filter_normal=False):
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path, on_bad_lines="skip")
    if len(df) == 0:
        raise ValueError(f"Empty file: {csv_path}")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    # Compute deviations
    df["temp_dev"] = df["temp_mean"] - df["setpoint_temp"]
    df["ph_dev"] = df["ph_mean"] - df["setpoint_ph"]
    df["rpm_dev"] = df["rpm_mean"] - df["setpoint_rpm"]
    if filter_normal and "has_fault" in df.columns:
        df = df[df["has_fault"] == 0]
    print(f"Loaded {len(df)} normal samples")
    return df


def apply_log_transforms(df):
    epsilon = 1e-6
    shift_values = {}
    for col in SKEWED_COLS:
        if col in df.columns:
            min_val = df[col].min()
            shift = -min_val + epsilon if min_val < 0 else 0
            shift_values[col] = shift
            df[col] = np.log(df[col] + shift + epsilon)
    return df, shift_values


def compute_proportional_constants(df):
    epsilon = 1e-6
    k_heater = np.mean(df["heater_pwm"] / (np.abs(df["temp_dev"]) + epsilon))
    k_acid = np.mean(df["acid_pwm"] / (np.abs(df["ph_dev"]) + epsilon))
    k_base = np.mean(df["base_pwm"] / (np.abs(df["ph_dev"]) + epsilon))
    return {"k_heater": k_heater, "k_acid": k_acid, "k_base": k_base}


def train_mahalanobis(X, name):
    mu = np.mean(X, axis=0)
    cov = np.cov(X, rowvar=False)
    if np.linalg.det(cov) < 1e-10:
        cov += np.eye(cov.shape[0]) * 1e-6
    inv_cov = np.linalg.inv(cov)
    distances_sq = [mahalanobis(X[i], mu, inv_cov) ** 2 for i in range(len(X))]
    distances_sq = np.array(distances_sq)
    threshold_99 = np.percentile(distances_sq, 99.0)
    threshold_95 = np.percentile(distances_sq, 95.0)
    print(f"\n=== Mahalanobis ({name}) Results ===")
    print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"Thresholds: 95th={threshold_95:.2f}, 99th={threshold_99:.2f}")
    return mu, inv_cov, {"95th": threshold_95, "99th": threshold_99}


def save_baseline_model(mu, inv_cov, shift_values, thresholds, k_values, suffix):
    np.save(os.path.join(MODEL_DIR, f"mu_{suffix}.npy"), mu)
    np.save(os.path.join(MODEL_DIR, f"inv_cov_{suffix}.npy"), inv_cov)
    with open(os.path.join(MODEL_DIR, f"shift_values_{suffix}.pkl"), "wb") as f:
        pickle.dump(shift_values, f)
    with open(os.path.join(MODEL_DIR, f"thresholds_{suffix}.pkl"), "wb") as f:
        pickle.dump(thresholds, f)
    with open(os.path.join(MODEL_DIR, f"k_values_{suffix}.pkl"), "wb") as f:
        pickle.dump(k_values, f)
    print(f"✓ {suffix.capitalize()} baseline saved")


# ================================
# MAIN
# ================================

if __name__ == "__main__":
    print("=" * 70)
    print("STEP 1: TRAIN MAHALANOBIS BASELINES (FIXED + DYNAMIC)")
    print("=" * 70)

    # Fixed baseline (nofaults)
    df_fixed = load_and_preprocess(FIXED_CSV)
    df_fixed, shifts_fixed = apply_log_transforms(df_fixed)
    X_fixed = df_fixed[BASE_FEATURE_COLS].values
    mu_fixed, inv_cov_fixed, thresh_fixed = train_mahalanobis(X_fixed, "Fixed")
    k_fixed = compute_proportional_constants(df_fixed)
    save_baseline_model(
        mu_fixed, inv_cov_fixed, shifts_fixed, thresh_fixed, k_fixed, "fixed"
    )

    # Dynamic baseline (variable_setpoints normals)
    df_var = load_and_preprocess(VAR_CSV, filter_normal=True)
    df_var, shifts_var = apply_log_transforms(df_var)
    X_var = df_var[BASE_FEATURE_COLS].values
    mu_var, inv_cov_var, thresh_var = train_mahalanobis(X_var, "Dynamic")
    k_var = compute_proportional_constants(df_var)
    save_baseline_model(mu_var, inv_cov_var, shifts_var, thresh_var, k_var, "var")

    print("\n✓ STEP 1 COMPLETE!")
