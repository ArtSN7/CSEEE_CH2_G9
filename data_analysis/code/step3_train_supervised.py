# step3_train_supervised.py
# Train supervised ML model on extracted features
# Uses multi-label for overlapping faults

import pandas as pd
import numpy as np
import pickle  # Added import
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    multilabel_confusion_matrix,
    hamming_loss,
)
from joblib import dump
import os

# ================================
# CONFIGURATION
# ================================
FEATURES_FILE = "./data/features_extracted.csv"
MODEL_DIR = "./model"

# Core features (from baseline)
CORE_FEATURES = [
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

# Statistical features
STAT_FEATURES = [
    "dist_sq_fixed", "dist_sq_var", "dist_sq_min",  # Dual + min
    "temp_range",
    "ph_range",
    "rpm_range",  # Variability
]

# Temporal features (from rolling windows)
TEMPORAL_FEATURES = [
    "temp_dev_rolling_mean",
    "temp_dev_rolling_std",
    "ph_dev_rolling_mean",
    "ph_dev_rolling_std",
    "rpm_dev_rolling_mean",
    "rpm_dev_rolling_std",
    "dist_sq_rolling_mean",
    "dist_sq_rolling_std",
    "dist_sq_rolling_max",
    "temp_dev_diff",
    "ph_dev_diff",
    "rpm_dev_diff",
    "dist_sq_diff",
    "heater_pwm_diff",
    "acid_pwm_diff",
    "base_pwm_diff",
    # New
    "heater_temp_corr",
    "acid_ph_corr",
    "base_ph_corr",
    "heater_eff_ratio",
    "acid_eff_ratio",
    "base_eff_ratio",
    "heater_residual",
    "acid_residual",
    "base_residual",
]

# All features for ML
ALL_FEATURES = CORE_FEATURES + STAT_FEATURES + TEMPORAL_FEATURES

# ================================
# LOAD DATA
# ================================

print("=" * 70)
print("STEP 3: TRAIN SUPERVISED ML MODEL")
print("=" * 70)

print(f"\nLoading {FEATURES_FILE}...")
df = pd.read_csv(FEATURES_FILE)
print(f"Loaded {len(df)} samples")

# Check for missing features
missing = [f for f in ALL_FEATURES if f not in df.columns]
if missing:
    print(f"WARNING: Missing features: {missing}")
    ALL_FEATURES = [f for f in ALL_FEATURES if f in df.columns]

print(f"\nUsing {len(ALL_FEATURES)} features for training")

# ================================
# PREPARE DATA
# ================================

print("\n" + "=" * 70)
print("DATA PREPARATION")
print("=" * 70)

# Handle any NaN/inf values
df = df.replace([np.inf, -np.inf], np.nan)
df = df.fillna(0)

# Prepare features
X = df[ALL_FEATURES].values

# Improved parsing for faults_active (handles dict-like strings)
def parse_faults(x):
    if x == 'none':
        return []
    # Clean and split
    x = x.replace("{", "").replace("}", "").replace("[", "").replace("]", "")
    parts = x.split(", ")
    names = []
    for part in parts:
        if "'name': " in part:
            name = part.split("'name': ")[1].strip("' ")
            names.append(name)
    return names

df['faults_active_list'] = df['faults_active'].apply(parse_faults)

# Dynamically extract unique fault types from data
all_faults = set()
for faults in df['faults_active_list']:
    all_faults.update(faults)
FAULT_TYPES = sorted(list(all_faults))
print(f"\nDetected fault types: {FAULT_TYPES}")

# Prepare multi-label targets
for fault in FAULT_TYPES:
    df[f"label_{fault}"] = df['faults_active_list'].apply(lambda faults: 1 if fault in faults else 0)
y_multi = df[[f"label_{fault}" for fault in FAULT_TYPES]].values

# Binary anomaly label (for eval)
y_anomaly = df["has_fault"].values

print(f"\nBinary class distribution:")
print(df["has_fault"].value_counts())
print(f"\nMulti-label distribution:")
for fault in FAULT_TYPES:
    print(f"{fault}: {df[f'label_{fault}'].sum()}")

# Train/test split
X_train, X_test, y_multi_train, y_multi_test = train_test_split(
    X, y_multi, test_size=0.3, random_state=42
)
_, _, y_anom_train, y_anom_test = train_test_split(
    X, y_anomaly, test_size=0.3, random_state=42
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# ================================
# TRAIN MULTI-LABEL CLASSIFIER
# ================================

print("\n" + "=" * 70)
print("TRAINING MULTI-LABEL FAULT CLASSIFIER")
print("=" * 70)

rf_multi = MultiOutputClassifier(
    RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
)

print("Training Random Forest...")
rf_multi.fit(X_train, y_multi_train)

# Evaluate multi-label
y_multi_pred = rf_multi.predict(X_test)
print("\n" + "=" * 70)
print("MULTI-LABEL FAULT IDENTIFICATION RESULTS")
print("=" * 70)
print(f"Hamming Loss: {hamming_loss(y_multi_test, y_multi_pred):.4f}")
print("\nConfusion Matrices per Label:")
print(multilabel_confusion_matrix(y_multi_test, y_multi_pred))
print("\nDetailed Report:")
print(classification_report(y_multi_test, y_multi_pred, target_names=FAULT_TYPES))

# Derive binary anomaly preds (any fault positive)
y_pred_anom = np.any(y_multi_pred, axis=1).astype(int)
print("\n" + "=" * 70)
print("DERIVED BINARY ANOMALY DETECTION RESULTS")
print("=" * 70)
print(f"Accuracy:  {accuracy_score(y_anom_test, y_pred_anom):.4f}")
print(f"Precision: {precision_score(y_anom_test, y_pred_anom):.4f}")
print(f"Recall:    {recall_score(y_anom_test, y_pred_anom):.4f}")
print(f"F1 Score:  {f1_score(y_anom_test, y_pred_anom):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_anom_test, y_pred_anom))

# Feature importance (average across estimators)
importances = np.mean([est.feature_importances_ for est in rf_multi.estimators_], axis=0)
feat_importance = pd.DataFrame(
    {"feature": ALL_FEATURES, "importance": importances}
).sort_values("importance", ascending=False)

print("\nTop 15 Most Important Features:")
print(feat_importance.head(15).to_string(index=False))

# ================================
# SAVE MODELS
# ================================

print("\n" + "=" * 70)
print("SAVING MODELS")
print("=" * 70)

dump(rf_multi, os.path.join(MODEL_DIR, "rf_multi_temporal.joblib"))

# Save feature list
with open(os.path.join(MODEL_DIR, "feature_list.txt"), "w") as f:
    f.write("\n".join(ALL_FEATURES))

# Also save detected FAULT_TYPES for use in step4
with open(os.path.join(MODEL_DIR, "fault_types.pkl"), "wb") as f:
    pickle.dump(FAULT_TYPES, f)

print(f"\n✓ Models saved to {MODEL_DIR}/")
print("  - rf_multi_temporal.joblib")
print("  - feature_list.txt")
print("  - fault_types.pkl")

print("\n" + "=" * 70)
print("✓ STEP 3 COMPLETE!")
print("=" * 70)
print("\nNext: Run step4_live_detector.py to test on live MQTT streams")
