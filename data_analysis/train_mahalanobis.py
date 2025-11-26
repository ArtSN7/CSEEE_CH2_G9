import pandas as pd
import numpy as np
from numpy.linalg import pinv
import joblib


def train_function(datafile="data/data_nofaults_20251126_091457.csv"):

    df = pd.read_csv(datafile)

    df["temp_error_norm"] = (df["temp_mean"] - df["setpoint_temp"]) / df[
        "setpoint_temp"
    ]
    df["ph_error_norm"] = (df["ph_mean"] - df["setpoint_ph"]) / df["setpoint_ph"]
    df["rpm_error_norm"] = (df["rpm_mean"] - df["setpoint_rpm"]) / df["setpoint_rpm"]

    df["temp_range"] = df["temp_max"] - df["temp_min"]
    df["ph_range"] = df["ph_max"] - df["ph_min"]
    df["rpm_range"] = df["rpm_max"] - df["rpm_min"]

    features = [
        "temp_error_norm",
        "ph_error_norm",
        "rpm_error_norm",
        "temp_range",
        "ph_range",
        "rpm_range",
        "heater_pwm",
        "motor_pwm",
        "acid_pwm",
        "base_pwm",
        "acid_dose_l",
        "base_dose_l",
    ]

    X = df[features].dropna().values

    center = np.mean(
        X, axis=0
    )  # use mean as data is clean anyways - but median is an option as well ( as in pres )

    cov = np.cov(X - center, rowvar=False)
    inv_cov = pinv(cov + 1e-6 * np.eye(cov.shape[0]))

    joblib.dump(
        {"center": center, "inv_cov": inv_cov, "features": features},
        "model/mahalanobis_simple.pkl",
    )

    print("Model trained and saved → model/mahalanobis_simple.pkl")


if __name__ == "__main__":
    train_function()
