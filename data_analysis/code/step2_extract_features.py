# step2_extract_features.py
# Extract statistical features from ALL data streams
# Including temporal/sequential patterns, correlations, ratios, residuals

import pandas as pd
import numpy as np
from scipy.spatial.distance import mahalanobis
import os
import pickle

# ================================
# CONFIGURATION
# ================================
CSV_FILES = [
    "./data/data_nofaults_20251203_001104.csv",
    "./data/data_single_fault_20251203_001101.csv",
    "./data/data_three_faults_20251203_001136.csv",
    "./data/data_variable_setpoints_20251203_001146.csv",
]

MODEL_DIR = "./model"
OUTPUT_FILE = "./data/features_extracted.csv"

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

# Window size for temporal features
WINDOW_SIZE = 5  # Small for rolling stats
CORR_WINDOW = 30  # Larger for correlations

# ================================
# LOAD BASELINE MODELS
# ================================

print("Loading baseline models...")
mu_fixed = np.load(os.path.join(MODEL_DIR, "mu_fixed.npy"))
inv_cov_fixed = np.load(os.path.join(MODEL_DIR, "inv_cov_fixed.npy"))
with open(os.path.join(MODEL_DIR, "shift_values_fixed.pkl"), "rb") as f:
    shift_values_fixed = pickle.load(f)
with open(os.path.join(MODEL_DIR, "k_values_fixed.pkl"), "rb") as f:
    k_fixed = pickle.load(f)

mu_var = np.load(os.path.join(MODEL_DIR, "mu_var.npy"))
inv_cov_var = np.load(os.path.join(MODEL_DIR, "inv_cov_var.npy"))
with open(os.path.join(MODEL_DIR, "shift_values_var.pkl"), "rb") as f:
    shift_values_var = pickle.load(
        f
    )  # Note: Using fixed shifts for simplicity; average if needed
with open(os.path.join(MODEL_DIR, "k_values_var.pkl"), "rb") as f:
    k_var = pickle.load(f)

print(f"✓ Loaded baselines")

# ================================
# FUNCTIONS
# ================================


def load_csv(csv_path):
    """Load and basic preprocessing."""
    print(f"\nLoading {csv_path}...")
    df = pd.read_csv(csv_path, on_bad_lines="skip")

    if len(df) == 0:
        return None

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Compute deviations
    df["temp_dev"] = df["temp_mean"] - df["setpoint_temp"]
    df["ph_dev"] = df["ph_mean"] - df["setpoint_ph"]
    df["rpm_dev"] = df["rpm_mean"] - df["setpoint_rpm"]

    # Add variability ranges
    df["temp_range"] = df["temp_max"] - df["temp_min"]
    df["ph_range"] = df["ph_max"] - df["ph_min"]
    df["rpm_range"] = df["rpm_max"] - df["rpm_min"]

    print(f"  Loaded {len(df)} samples")
    return df


def apply_log_transforms(df):
    """Apply same log transforms as baseline."""
    epsilon = 1e-6

    for col in SKEWED_COLS:
        if col in df.columns:
            shift = shift_values_fixed.get(col, 0)  # Use fixed shifts
            df[col] = np.log(df[col] + shift + epsilon)

    return df


def compute_mahalanobis_distance(row, mu, inv_cov, suffix):
    """Compute Mahalanobis distance for a single row."""
    features = row[BASE_FEATURE_COLS].values
    try:
        dist_sq = mahalanobis(features, mu, inv_cov) ** 2
    except:
        dist_sq = 0
    return dist_sq


def add_temporal_features(df):
    """Add rolling window features to capture temporal patterns."""
    print("  Computing temporal features...")

    # Rolling statistics (5-sample window)
    for col in ["temp_dev", "ph_dev", "rpm_dev"]:
        df[f"{col}_rolling_mean"] = df[col].rolling(WINDOW_SIZE, min_periods=1).mean()
        df[f"{col}_rolling_std"] = (
            df[col].rolling(WINDOW_SIZE, min_periods=1).std().fillna(0)
        )

    # Mahalanobis distance rolling stats (use min dist for robustness)
    df["dist_sq_min"] = df[["dist_sq_fixed", "dist_sq_var"]].min(axis=1)
    df["dist_sq_rolling_mean"] = (
        df["dist_sq_min"].rolling(WINDOW_SIZE, min_periods=1).mean()
    )
    df["dist_sq_rolling_std"] = (
        df["dist_sq_min"].rolling(WINDOW_SIZE, min_periods=1).std().fillna(0)
    )
    df["dist_sq_rolling_max"] = (
        df["dist_sq_min"].rolling(WINDOW_SIZE, min_periods=1).max()
    )

    # Rate of change
    df["temp_dev_diff"] = df["temp_dev"].diff().fillna(0)
    df["ph_dev_diff"] = df["ph_dev"].diff().fillna(0)
    df["rpm_dev_diff"] = df["rpm_dev"].diff().fillna(0)
    df["dist_sq_diff"] = df["dist_sq_min"].diff().fillna(0)

    # Actuator effort changes
    df["heater_pwm_diff"] = df["heater_pwm"].diff().abs().fillna(0)
    df["acid_pwm_diff"] = df["acid_pwm"].diff().abs().fillna(0)
    df["base_pwm_diff"] = df["base_pwm"].diff().abs().fillna(0)

    # New: Cross-correlations
    df["heater_temp_corr"] = (
        df["heater_pwm"]
        .rolling(CORR_WINDOW, min_periods=1)
        .corr(df["temp_dev_diff"])
        .fillna(0)
    )
    df["acid_ph_corr"] = (
        df["acid_pwm"]
        .rolling(CORR_WINDOW, min_periods=1)
        .corr(df["ph_dev_diff"])
        .fillna(0)
    )
    df["base_ph_corr"] = (
        df["base_pwm"]
        .rolling(CORR_WINDOW, min_periods=1)
        .corr(df["ph_dev_diff"])
        .fillna(0)
    )

    # New: Efficiency ratios
    epsilon = 1e-6
    df["heater_eff_ratio"] = df["heater_pwm"] / (np.abs(df["temp_dev_diff"]) + epsilon)
    df["acid_eff_ratio"] = df["acid_pwm"] / (np.abs(df["ph_dev_diff"]) + epsilon)
    df["base_eff_ratio"] = df["base_pwm"] / (np.abs(df["ph_dev_diff"]) + epsilon)

    # New: Residuals (using average k from fixed and var)
    k_heater = (k_fixed["k_heater"] + k_var["k_heater"]) / 2
    k_acid = (k_fixed["k_acid"] + k_var["k_acid"]) / 2
    k_base = (k_fixed["k_base"] + k_var["k_base"]) / 2
    df["heater_residual"] = df["heater_pwm"] - k_heater * np.abs(df["temp_dev"])
    df["acid_residual"] = df["acid_pwm"] - k_acid * np.abs(df["ph_dev"])
    df["base_residual"] = df["base_pwm"] - k_base * np.abs(df["ph_dev"])

    return df


def extract_features_from_csv(csv_path):
    """Process one CSV and return feature dataframe."""
    df = load_csv(csv_path)
    if df is None:
        return None

    # Apply log transforms
    df = apply_log_transforms(df)

    # Compute dual Mahalanobis distances
    print("  Computing Mahalanobis distances...")
    df["dist_sq_fixed"] = df.apply(
        lambda row: compute_mahalanobis_distance(row, mu_fixed, inv_cov_fixed, "fixed"),
        axis=1,
    )
    df["dist_sq_var"] = df.apply(
        lambda row: compute_mahalanobis_distance(row, mu_var, inv_cov_var, "var"),
        axis=1,
    )

    # Add temporal features
    df = add_temporal_features(df)

    # Add source stream identifier
    stream_name = os.path.basename(csv_path).split("_")[
        1
    ]  # nofaults, single, three, variable
    df["stream_source"] = stream_name

    print(f"  ✓ Extracted features for {len(df)} samples")
    return df


# ================================
# MAIN
# ================================

if __name__ == "__main__":
    print("=" * 70)
    print("STEP 2: EXTRACT STATISTICAL FEATURES FROM ALL STREAMS")
    print("=" * 70)

    all_features = []

    for csv_file in CSV_FILES:
        if not os.path.exists(csv_file):
            print(f"WARNING: {csv_file} not found, skipping...")
            continue

        df_features = extract_features_from_csv(csv_file)
        if df_features is not None:
            all_features.append(df_features)

    if len(all_features) == 0:
        print("\nERROR: No data processed!")
        exit(1)

    # Combine all features
    print("\n" + "=" * 70)
    print("COMBINING ALL FEATURES")
    print("=" * 70)

    combined = pd.concat(all_features, ignore_index=True)

    print(f"\nTotal samples: {len(combined)}")
    print(f"Normal samples: {len(combined[combined['has_fault']==0])}")
    print(f"Fault samples: {len(combined[combined['has_fault']==1])}")

    # Save to CSV
    combined.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✓ Features saved to {OUTPUT_FILE}")

    # Show feature columns
    print(f"\nTotal features: {len(combined.columns)}")
    print("\nSample of extracted features:")
    feature_cols = [
        c
        for c in combined.columns
        if not c in ["timestamp", "faults_active", "stream_source"]
    ]
    print(feature_cols[:20])

    print("\n" + "=" * 70)
    print("✓ STEP 2 COMPLETE!")
    print("=" * 70)
    print("\nNext: Run step3_train_supervised.py to train ML model on these features")
