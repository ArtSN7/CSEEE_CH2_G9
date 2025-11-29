import pandas as pd
import numpy as np
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2
import argparse
import os

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
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Compute deviations
    df["temp_dev"] = df["temp_mean"] - df["setpoint_temp"]
    df["ph_dev"] = df["ph_mean"] - df["setpoint_ph"]
    df["rpm_dev"] = df["rpm_mean"] - df["setpoint_rpm"]

    # Handle skewness: log transform (add small epsilon for zeros)
    epsilon = 1e-6
    for col in SKEWED_COLS:
        if col in df.columns:
            # Shift to positive if needed (e.g., deviations can be negative)
            min_val = df[col].min()
            if min_val < 0:
                df[col] = df[col] - min_val + epsilon
            df[col] = np.log(df[col] + epsilon)

    # Select only fault-free samples (though nofaults should be clean)
    if "has_fault" in df.columns:
        df = df[df["has_fault"] == 0]

    X = df[FEATURE_COLS].values
    return X, df


def train_mahalanobis(X):
    # Compute mean vector
    mu = np.mean(X, axis=0)

    # Compute covariance matrix
    cov = np.cov(X, rowvar=False)

    # Regularize if singular
    if np.linalg.det(cov) < 1e-10:
        cov += np.eye(cov.shape[0]) * 1e-6

    # Inverse covariance
    inv_cov = np.linalg.inv(cov)

    # Degrees of freedom for chi-squared threshold
    df = X.shape[1]
    # Example thresholds (tune based on validation)
    high_threshold = chi2.ppf(0.99, df=df)
    low_threshold = chi2.ppf(0.95, df=df)

    print(f"Trained on {X.shape[0]} samples.")
    print(f"Mean vector: {mu}")
    print(f"Inverse covariance shape: {inv_cov.shape}")
    print(
        f"Suggested high/low thresholds (chi2 99%/95%): {high_threshold:.2f} / {low_threshold:.2f}"
    )

    return mu, inv_cov, high_threshold, low_threshold


def save_model(mu, inv_cov, output_dir="model"):
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "mu.npy"), mu)
    np.save(os.path.join(output_dir, "inv_cov.npy"), inv_cov)
    print(f"Model saved to {output_dir}")


if __name__ == "__main__":

    X, df = load_and_preprocess("./data/data_nofaults_20251126_091457.csv")
    mu, inv_cov, high_thresh, low_thresh = train_mahalanobis(X)
    save_model(mu, inv_cov)
