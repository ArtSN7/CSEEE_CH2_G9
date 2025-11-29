import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from datetime import datetime


# Define the features we're interested in, based on the data structure
FEATURE_COLS = [
    "temp_mean",
    "temp_min",
    "temp_max",
    "ph_mean",
    "ph_min",
    "ph_max",
    "rpm_mean",
    "rpm_min",
    "rpm_max",
    "heater_pwm",
    "motor_pwm",
    "acid_pwm",
    "base_pwm",
    "acid_dose_l",
    "base_dose_l",
]

# Deviation columns to compute
DEV_COLS = ["temp_dev", "ph_dev", "rpm_dev"]


def load_data(csv_file):
    df = pd.read_csv(csv_file)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)

    # Compute deviations
    df["temp_dev"] = df["temp_mean"] - df["setpoint_temp"]
    df["ph_dev"] = df["ph_mean"] - df["setpoint_ph"]
    df["rpm_dev"] = df["rpm_mean"] - df["setpoint_rpm"]

    return df


def compute_statistics(df, cols):
    # Basic descriptive stats
    desc = df[cols].describe()

    # Additional stats: skewness, kurtosis
    skew = df[cols].skew().to_frame(name="skewness")
    kurt = df[cols].kurtosis().to_frame(name="kurtosis")

    # Normality test (Shapiro-Wilk p-value; >0.05 suggests normal)
    normality = {}
    for col in cols:
        if len(df[col].dropna()) > 3:  # Shapiro requires at least 3 samples
            _, p = stats.shapiro(df[col].dropna())
            normality[col] = p
    norm_df = pd.Series(normality, name="shapiro_p").to_frame()

    # Combine all stats
    stats_df = pd.concat([desc.T, skew, kurt, norm_df], axis=1)
    return stats_df


def compute_z_scores(df, cols):
    z_scores = pd.DataFrame(
        stats.zscore(df[cols], nan_policy="omit"), index=df.index, columns=cols
    )
    return z_scores


def plot_distributions(df, cols, output_dir):
    for col in cols:
        plt.figure(figsize=(8, 6))
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.savefig(os.path.join(output_dir, f"{col}_hist.png"))
        plt.close()


def plot_qq_plots(df, cols, output_dir):
    for col in cols:
        plt.figure(figsize=(8, 6))
        stats.probplot(df[col].dropna(), dist="norm", plot=plt)
        plt.title(f"QQ Plot for {col}")
        plt.savefig(os.path.join(output_dir, f"{col}_qq.png"))
        plt.close()


def plot_time_series(df, cols, output_dir):
    plt.figure(figsize=(12, 8))
    df[cols].plot(
        subplots=True, layout=(len(cols) // 3 + 1, 3), figsize=(12, len(cols) * 2)
    )
    plt.suptitle("Time-Series Plots")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "time_series.png"))
    plt.close()


def plot_boxplots(df, cols, output_dir):
    for col in cols:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=df[col].dropna())
        plt.title(f"Boxplot of {col} (for Outlier Detection)")
        plt.savefig(os.path.join(output_dir, f"{col}_box.png"))
        plt.close()


def plot_correlation_heatmap(df, cols, output_dir):
    corr = df[cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.savefig(os.path.join(output_dir, "corr_heatmap.png"))
    plt.close()


def plot_pairplot(df, selected_cols, output_dir):
    sns.pairplot(df[selected_cols].dropna())
    plt.suptitle("Pairplot for Selected Features")
    plt.savefig(os.path.join(output_dir, "pairplot.png"))
    plt.close()


def main(csv_file):
    df = load_data(csv_file)

    # All numeric columns for analysis
    all_cols = FEATURE_COLS + DEV_COLS

    # Create output directory
    output_dir = f'analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    os.makedirs(output_dir, exist_ok=True)

    # Compute and print statistics
    stats_df = compute_statistics(df, all_cols)
    print("Descriptive Statistics, Skewness, Kurtosis, and Normality Test:")
    print(stats_df)
    stats_df.to_csv(os.path.join(output_dir, "statistics.csv"))

    # Compute Z-scores
    z_scores = compute_z_scores(df, all_cols)
    print("\nSample Z-Scores (first 5 rows):")
    print(z_scores.head())
    z_scores.to_csv(os.path.join(output_dir, "z_scores.csv"))

    # Plots
    plot_distributions(df, all_cols, output_dir)
    plot_qq_plots(df, all_cols, output_dir)
    plot_time_series(df, all_cols, output_dir)
    plot_boxplots(df, all_cols, output_dir)
    plot_correlation_heatmap(df, all_cols, output_dir)

    # Pairplot for a subset to avoid overload (e.g., deviations + key actuators)
    selected_for_pair = [
        "temp_dev",
        "ph_dev",
        "rpm_dev",
        "heater_pwm",
        "acid_pwm",
        "base_pwm",
    ]
    plot_pairplot(df, selected_for_pair, output_dir)

    print(f"\nAll plots and CSV outputs saved to: {output_dir}")


if __name__ == "__main__":
    main("./data/data_nofaults_20251126_091457.csv")
