import pandas as pd
import numpy as np
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2
import argparse
import os


CSV_FILES = [
    "./data/data_nofaults_20251201_182315.csv",
    #"./data/data_single_fault_20251201_182037.csv",
    #"./data/data_three_faults_20251201_182112.csv",
    #"./data/data_variable_setpoints_20251201_182137.csv",
]


# Recommended features based on analysis
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

# Skewed features needing transformation (from stats: high skew/kurtosis)
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


def load_and_preprocess(csv_path):
    """Load and preprocess a single CSV file."""
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path, on_bad_lines="skip")

    # Check if file has data
    if len(df) == 0:
        print(f"  WARNING: {csv_path} is empty!")
        return None, None

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Compute deviations
    df["temp_dev"] = df["temp_mean"] - df["setpoint_temp"]
    df["ph_dev"] = df["ph_mean"] - df["setpoint_ph"]
    df["rpm_dev"] = df["rpm_mean"] - df["setpoint_rpm"]

    # Select only fault-free samples BEFORE log transform
    if "has_fault" in df.columns:
        initial_count = len(df)
        df = df[df["has_fault"] == 0]
        print(f"  Loaded {len(df)} fault-free samples (out of {initial_count} total)")
    else:
        print(f"  Loaded {len(df)} samples (no fault label found)")

    if len(df) == 0:
        print(f"  WARNING: No fault-free samples in {csv_path}!")
        return None, None

    return df


def preprocess_combined(df):
    """Apply log transforms to combined dataframe."""
    # Handle skewness: log transform (add small epsilon for zeros)
    epsilon = 1e-6
    for col in SKEWED_COLS:
        if col in df.columns:
            # Shift to positive if needed (e.g., deviations can be negative)
            min_val = df[col].min()
            if min_val < 0:
                df[col] = df[col] - min_val + epsilon
            df[col] = np.log(df[col] + epsilon)

    X = df[FEATURE_COLS].values
    return X, df


def train_mahalanobis(X):
    """Train Mahalanobis distance model."""
    # Compute mean vector
    mu = np.mean(X, axis=0)

    # Compute covariance matrix
    cov = np.cov(X, rowvar=False)

    # Regularize if singular
    if np.linalg.det(cov) < 1e-10:
        print("WARNING: Covariance matrix is near-singular, adding regularization")
        cov += np.eye(cov.shape[0]) * 1e-6

    # Inverse covariance
    inv_cov = np.linalg.inv(cov)

    # ============================================
    # COMPUTE EMPIRICAL THRESHOLDS (NEW!)
    # ============================================
    print("\nComputing empirical distance distribution...")
    distances_sq = []
    for i in range(len(X)):
        dist_sq = mahalanobis(X[i], mu, inv_cov) ** 2
        distances_sq.append(dist_sq)

    distances_sq = np.array(distances_sq)

    # Use percentiles from actual fault-free data
    high_threshold = np.percentile(distances_sq, 99.9)  # 99.9th percentile
    low_threshold = np.percentile(distances_sq, 95.0)  # 95th percentile

    print(f"\n=== Training Summary ===")
    print(f"Trained on {X.shape[0]} fault-free samples")
    print(f"Feature dimension: {X.shape[1]}")
    print(f"\nDistance² Statistics (fault-free data):")
    print(f"  Min:    {distances_sq.min():.2f}")
    print(f"  Mean:   {distances_sq.mean():.2f}")
    print(f"  Median: {np.median(distances_sq):.2f}")
    print(f"  Max:    {distances_sq.max():.2f}")
    print(f"  Std:    {distances_sq.std():.2f}")
    print(f"\nEmpirical Thresholds:")
    print(f"  HIGH (99.9%): {high_threshold:.2f}")
    print(f"  LOW  (95.0%): {low_threshold:.2f}")

    # Theoretical chi-squared (for comparison)
    df = X.shape[1]
    chi2_high = chi2.ppf(0.999, df=df)
    chi2_low = chi2.ppf(0.95, df=df)
    print(f"\nTheoretical chi² thresholds (for reference):")
    print(f"  HIGH (99.9%): {chi2_high:.2f}")
    print(f"  LOW  (95.0%): {chi2_low:.2f}")

    return mu, inv_cov, high_threshold, low_threshold


def save_model(mu, inv_cov, output_dir="model"):
    """Save trained model to disk."""
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "mu.npy"), mu)
    np.save(os.path.join(output_dir, "inv_cov.npy"), inv_cov)
    print(f"\nModel saved to {output_dir}/")


if __name__ == "__main__":
    print("=== Training Mahalanobis Model on Multiple CSV Files ===\n")

    # Load all CSV files
    all_dfs = []
    for csv_file in CSV_FILES:
        if not os.path.exists(csv_file):
            print(f"WARNING: {csv_file} not found, skipping...")
            continue

        df = load_and_preprocess(csv_file)
        if df is not None:
            all_dfs.append(df)

    if len(all_dfs) == 0:
        print("\nERROR: No valid data loaded! Check your CSV file paths.")
        exit(1)

    # Combine all dataframes
    print(f"\n=== Combining {len(all_dfs)} dataframes ===")
    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"Total combined fault-free samples: {len(combined_df)}")

    # Apply preprocessing to combined data
    print("\n=== Applying log transforms ===")
    X, processed_df = preprocess_combined(combined_df)

    # Train model
    print("\n=== Training Mahalanobis Model ===")
    mu, inv_cov, high_thresh, low_thresh = train_mahalanobis(X)

    # Save model
    save_model(mu, inv_cov)

    print("\n✓ Training complete!")
    print(f"  Update your detector thresholds to:")
    print(f"  HIGH_THRESHOLD = {int(high_thresh)}")
    print(f"  LOW_THRESHOLD = {int(low_thresh)}")
