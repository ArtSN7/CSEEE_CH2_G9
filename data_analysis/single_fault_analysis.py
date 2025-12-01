# find_real_threshold.py
import pandas as pd
import numpy as np
from scipy.spatial.distance import mahalanobis
import joblib

# Load your trained model
mu = np.load("model/mu.npy")
inv_cov = np.load("model/inv_cov.npy")

# Load nofaults data and apply SAME preprocessing as training
df = pd.read_csv("./data/data_single_fault_20251201_143419.csv")  # your nofaults file
df["temp_dev"] = df["temp_mean"] - df["setpoint_temp"]
df["ph_dev"] = df["ph_mean"] - df["setpoint_ph"]
df["rpm_dev"] = df["rpm_mean"] - df["setpoint_rpm"]

# Apply same log transforms (use the same code as training!)
# ... paste your training preprocessing here ...

features = df[
    [
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
].values

dists = []
for x in features:
    d2 = mahalanobis(x, mu, inv_cov) ** 2
    dists.append(d2)

print(f"Normal dist² stats:")
print(f"  95th percentile: {np.percentile(dists, 95):.1f}")
print(f"  99th percentile: {np.percentile(dists, 99):.1f}")
print(f"  99.9th percentile: {np.percentile(dists, 99.9):.1f}")
print(f"  Max during normal: {max(dists):.1f}")
