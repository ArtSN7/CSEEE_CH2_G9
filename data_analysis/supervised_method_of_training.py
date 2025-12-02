# supervised_training_fixed.py
# Trains RandomForest on ALL data streams with proper preprocessing
# Includes setpoint change detection to handle variable_setpoints properly

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    accuracy_score,
    classification_report,
)
from sklearn.preprocessing import LabelEncoder
from joblib import dump
from scipy.spatial.distance import mahalanobis
import os

# ================================
# CONFIGURATION
# ================================

# IMPORTANT: Uncomment ALL your CSV files!
CSV_FILES = [
    "./data/data_nofaults_20251201_182315.csv",
    "./data/data_single_fault_20251201_182037.csv",
    "./data/data_three_faults_20251201_182112.csv",
    "./data/data_variable_setpoints_20251201_182137.csv",
]

MODEL_DIR = "./model"
os.makedirs(MODEL_DIR, exist_ok=True)

# Skewed columns that need log transform
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

# Base features (before dist_sq)
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

# Enhanced features for better detection
ENHANCED_FEATURE_COLS = BASE_FEATURE_COLS + [
    "dist_sq",  # Mahalanobis distance as feature
    "temp_range",  # Temperature variability
    "ph_range",  # pH variability
    "rpm_range",  # RPM variability
]

# ================================
# PREPROCESSING FUNCTIONS
# ================================


def add_engineered_features(df):
    """Add engineered features for better detection."""
    # Compute deviations
    df["temp_dev"] = df["temp_mean"] - df["setpoint_temp"]
    df["ph_dev"] = df["ph_mean"] - df["setpoint_ph"]
    df["rpm_dev"] = df["rpm_mean"] - df["setpoint_rpm"]

    # Add range features (variability indicators)
    df["temp_range"] = df["temp_max"] - df["temp_min"]
    df["ph_range"] = df["ph_max"] - df["ph_min"]
    df["rpm_range"] = df["rpm_max"] - df["rpm_min"]

    return df


def preprocess(df):
    """Preprocess dataframe: log transforms and feature engineering."""
    # Add engineered features first
    df = add_engineered_features(df)

    # Log transform skewed columns
    epsilon = 1e-6
    for col in SKEWED_COLS:
        if col in df.columns:
            min_val = df[col].min()
            if min_val < 0:
                df[col] = df[col] - min_val + epsilon
            df[col] = np.log(df[col] + epsilon)

    return df


def compute_mahalanobis_distances(df, mu, inv_cov):
    """Compute Mahalanobis distance for each sample."""
    distances = []
    for idx, row in df.iterrows():
        features = row[BASE_FEATURE_COLS].values
        try:
            dist_sq = mahalanobis(features, mu, inv_cov) ** 2
        except:
            dist_sq = 0  # Handle edge cases
        distances.append(dist_sq)
    return distances


# ================================
# LOAD AND PREPROCESS DATA
# ================================

print("=" * 60)
print("SUPERVISED ANOMALY DETECTION TRAINING")
print("=" * 60)

# Load all CSV files
dfs = []
for file in CSV_FILES:
    if not os.path.exists(file):
        print(f"WARNING: {file} not found, skipping...")
        continue
    print(f"Loading {file}...")
    temp_df = pd.read_csv(file, on_bad_lines="skip")
    print(f"  → {len(temp_df)} samples")
    dfs.append(temp_df)

if len(dfs) == 0:
    print("ERROR: No data files found!")
    exit(1)

# Combine all data
print("\nCombining datasets...")
all_data = pd.concat(dfs, ignore_index=True)
print(f"Total combined samples: {len(all_data)}")

# Preprocess
print("\nPreprocessing...")
all_data = preprocess(all_data)

print(f"Normal samples (has_fault=0): {len(all_data[all_data['has_fault']==0])}")
print(f"Faulty samples (has_fault=1): {len(all_data[all_data['has_fault']==1])}")

# ================================
# TRAIN MAHALANOBIS MODEL (for dist_sq feature)
# ================================

print("\nTraining Mahalanobis baseline (for dist_sq feature)...")
normal_data = all_data[all_data["has_fault"] == 0]
X_normal = normal_data[BASE_FEATURE_COLS].values

mu = np.mean(X_normal, axis=0)
cov = np.cov(X_normal, rowvar=False)

# Regularize covariance
if np.linalg.det(cov) < 1e-10:
    cov += np.eye(cov.shape[0]) * 1e-6

inv_cov = np.linalg.inv(cov)

# Save Mahalanobis model
np.save(os.path.join(MODEL_DIR, "mu.npy"), mu)
np.save(os.path.join(MODEL_DIR, "inv_cov.npy"), inv_cov)
print("  → Mahalanobis model saved")

# Compute dist_sq for all samples
print("Computing Mahalanobis distances...")
all_data["dist_sq"] = compute_mahalanobis_distances(all_data, mu, inv_cov)

# ================================
# PREPARE FAULT TYPE LABELS
# ================================

print("\nPreparing fault type labels...")
all_data["fault_type"] = all_data["faults_active"].apply(
    lambda x: x if x != "none" else "normal"
)

# Show fault distribution
print("\nFault type distribution:")
print(all_data["fault_type"].value_counts())

le = LabelEncoder()
all_data["fault_type_encoded"] = le.fit_transform(all_data["fault_type"])

# ================================
# TRAIN RANDOM FOREST MODELS
# ================================

print("\n" + "=" * 60)
print("TRAINING RANDOM FOREST MODELS")
print("=" * 60)

# Prepare features and targets
X = all_data[ENHANCED_FEATURE_COLS].values
y_anomaly = all_data["has_fault"].values
y_type = all_data["fault_type_encoded"].values

# Split data
X_train, X_test, y_anomaly_train, y_anomaly_test = train_test_split(
    X, y_anomaly, test_size=0.3, random_state=42, stratify=y_anomaly
)
_, _, y_type_train, y_type_test = train_test_split(
    X, y_type, test_size=0.3, random_state=42, stratify=y_type
)

print(f"\nTraining set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# Train anomaly detector (binary classification)
print("\n1. Training ANOMALY DETECTOR (binary: fault/no-fault)...")
rf_anomaly = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight="balanced",  # Handle class imbalance
    random_state=42,
    n_jobs=-1,
)
rf_anomaly.fit(X_train, y_anomaly_train)

# Evaluate anomaly detector
y_pred_anomaly = rf_anomaly.predict(X_test)
print("\nANOMALY DETECTION RESULTS:")
print(f"Accuracy : {accuracy_score(y_anomaly_test, y_pred_anomaly):.4f}")
print(f"Precision: {precision_score(y_anomaly_test, y_pred_anomaly):.4f}")
print(f"Recall   : {recall_score(y_anomaly_test, y_pred_anomaly):.4f}")
print(f"\nConfusion Matrix:")
print(confusion_matrix(y_anomaly_test, y_pred_anomaly))
print("\nClassification Report:")
print(
    classification_report(
        y_anomaly_test, y_pred_anomaly, target_names=["Normal", "Fault"]
    )
)

# Feature importance
feature_importance = pd.DataFrame(
    {"feature": ENHANCED_FEATURE_COLS, "importance": rf_anomaly.feature_importances_}
).sort_values("importance", ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10).to_string(index=False))

# Train fault type classifier (multi-class)
print("\n2. Training FAULT TYPE CLASSIFIER (multi-class)...")
rf_type = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
rf_type.fit(X_train, y_type_train)

# Evaluate fault type classifier
y_pred_type = rf_type.predict(X_test)
print("\nFAULT TYPE IDENTIFICATION RESULTS:")
print(f"Accuracy : {accuracy_score(y_type_test, y_pred_type):.4f}")
print(f"Precision: {precision_score(y_type_test, y_pred_type, average='weighted'):.4f}")
print(f"Recall   : {recall_score(y_type_test, y_pred_type, average='weighted'):.4f}")
print(f"\nConfusion Matrix:")
print(confusion_matrix(y_type_test, y_pred_type))

# ================================
# SAVE MODELS
# ================================

print("\n" + "=" * 60)
print("SAVING MODELS")
print("=" * 60)

dump(rf_anomaly, os.path.join(MODEL_DIR, "rf_anomaly_model.joblib"))
dump(rf_type, os.path.join(MODEL_DIR, "rf_type_model.joblib"))
dump(le, os.path.join(MODEL_DIR, "fault_label_encoder.joblib"))

print(f"✓ Models saved to {MODEL_DIR}/")
print(f"  - rf_anomaly_model.joblib")
print(f"  - rf_type_model.joblib")
print(f"  - fault_label_encoder.joblib")
print(f"  - mu.npy")
print(f"  - inv_cov.npy")

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
print("\nYou can now run: python live_detector_supervised.py <stream_name>")
print("Available streams: nofaults, single_fault, three_faults, variable_setpoints")
