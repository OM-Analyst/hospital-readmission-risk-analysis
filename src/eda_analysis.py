"""
Story 3 - Exploratory Data Analysis
Hospital Readmission Risk Analysis

Reads:
    outputs/story-2/cleaned_readmission_data.csv

Writes:
    outputs/story-3/*.csv
    outputs/story-3/*.png
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLEANED_DATA_PATH = "outputs/story-2/cleaned_readmission_data.csv"
STORY_3_OUTPUT_DIR = "outputs/story-3"

TARGET = "readmitted_30_days"

CATEGORICAL_GROUPS = [
    "age",
    "race",
    "gender",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "diabetesMed",
    "change",
    "A1Cresult",
    "max_glu_serum",
]

NUMERIC_COLS = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
]


def create_output_folder():
    os.makedirs(STORY_3_OUTPUT_DIR, exist_ok=True)


def load_cleaned_dataset():
    if not os.path.exists(CLEANED_DATA_PATH):
        raise FileNotFoundError(
            f"Cleaned dataset not found: {CLEANED_DATA_PATH}. "
            "Run src/data_cleaning.py (Story 2) first."
        )
    df = pd.read_csv(CLEANED_DATA_PATH)
    print(f"Cleaned dataset loaded. Shape: {df.shape}")
    return df


# ==============================
# Readmission rate by category
# ==============================

def readmission_rate_by_group(df, group_col):
    """Readmission rate (%) for each category in group_col."""
    summary = (
        df.groupby(group_col)[TARGET]
        .agg(encounters="count", readmitted="sum")
        .reset_index()
    )
    summary["readmission_rate_percent"] = (
        summary["readmitted"] / summary["encounters"] * 100
    ).round(2)
    summary = summary.sort_values("readmission_rate_percent", ascending=False)
    return summary


def save_readmission_rate_by_group(df):
    for col in CATEGORICAL_GROUPS:
        if col not in df.columns:
            continue
        summary = readmission_rate_by_group(df, col)
        summary.to_csv(
            os.path.join(STORY_3_OUTPUT_DIR, f"readmission_rate_by_{col}.csv"),
            index=False,
        )
    print(f"Readmission rate breakdowns saved for: {CATEGORICAL_GROUPS}")


def plot_readmission_rate_by_age(df):
    summary = readmission_rate_by_group(df, "age").sort_values("age")
    plt.figure(figsize=(8, 5))
    plt.bar(summary["age"], summary["readmission_rate_percent"], color="steelblue")
    plt.xlabel("Age Group")
    plt.ylabel("30-Day Readmission Rate (%)")
    plt.title("30-Day Readmission Rate by Age Group")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(STORY_3_OUTPUT_DIR, "readmission_rate_by_age.png"))
    plt.close()


def plot_readmission_rate_by_race(df):
    summary = readmission_rate_by_group(df, "race")
    plt.figure(figsize=(8, 5))
    plt.bar(summary["race"], summary["readmission_rate_percent"], color="darkorange")
    plt.xlabel("Race")
    plt.ylabel("30-Day Readmission Rate (%)")
    plt.title("30-Day Readmission Rate by Race")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(STORY_3_OUTPUT_DIR, "readmission_rate_by_race.png"))
    plt.close()


# ==============================
# Numeric feature analysis
# ==============================

def numeric_summary_by_target(df):
    """Mean of each numeric feature, split by readmission outcome."""
    cols = [c for c in NUMERIC_COLS if c in df.columns]
    summary = df.groupby(TARGET)[cols].mean().round(2).T
    summary.columns = ["not_readmitted_30d_mean", "readmitted_30d_mean"]
    summary = summary.reset_index().rename(columns={"index": "feature"})
    summary.to_csv(
        os.path.join(STORY_3_OUTPUT_DIR, "numeric_feature_means_by_target.csv"),
        index=False,
    )
    print("Numeric feature means by target saved.")
    return summary


def plot_time_in_hospital_distribution(df):
    plt.figure(figsize=(8, 5))
    plt.hist(
        df[df[TARGET] == 0]["time_in_hospital"],
        bins=14, alpha=0.6, label="Not readmitted <30d", color="steelblue",
        density=True,
    )
    plt.hist(
        df[df[TARGET] == 1]["time_in_hospital"],
        bins=14, alpha=0.6, label="Readmitted <30d", color="crimson",
        density=True,
    )
    plt.xlabel("Time in Hospital (days)")
    plt.ylabel("Density")
    plt.title("Time in Hospital: Readmitted vs. Not Readmitted")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(STORY_3_OUTPUT_DIR, "time_in_hospital_distribution.png"))
    plt.close()


def plot_correlation_heatmap(df):
    cols = [c for c in NUMERIC_COLS if c in df.columns] + [TARGET]
    corr = df[cols].corr()
    corr.to_csv(os.path.join(STORY_3_OUTPUT_DIR, "numeric_correlation_matrix.csv"))

    plt.figure(figsize=(9, 7))
    im = plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, label="Correlation")
    plt.xticks(range(len(cols)), cols, rotation=45, ha="right")
    plt.yticks(range(len(cols)), cols)
    plt.title("Correlation Matrix - Numeric Features & Target")
    plt.tight_layout()
    plt.savefig(os.path.join(STORY_3_OUTPUT_DIR, "correlation_heatmap.png"))
    plt.close()
    print("Correlation matrix saved.")


# ==============================
# Overall target distribution
# ==============================

def plot_target_distribution(df):
    counts = df[TARGET].value_counts().sort_index()
    labels = ["Not Readmitted <30d", "Readmitted <30d"]

    plt.figure(figsize=(5, 5))
    plt.bar(labels, counts.values, color=["steelblue", "crimson"])
    plt.ylabel("Number of Encounters")
    plt.title("30-Day Readmission Distribution")
    for i, v in enumerate(counts.values):
        plt.text(i, v + 500, str(v), ha="center")
    plt.tight_layout()
    plt.savefig(os.path.join(STORY_3_OUTPUT_DIR, "target_distribution.png"))
    plt.close()


# ==============================
# EDA summary report
# ==============================

def save_eda_summary(df):
    summary = pd.DataFrame({
        "metric": [
            "Total encounters",
            "Readmitted <30 days",
            "Readmission rate (%)",
            "Numeric features analyzed",
            "Categorical groups analyzed",
        ],
        "value": [
            len(df),
            int(df[TARGET].sum()),
            round(df[TARGET].mean() * 100, 2),
            len([c for c in NUMERIC_COLS if c in df.columns]),
            len([c for c in CATEGORICAL_GROUPS if c in df.columns]),
        ],
    })
    summary.to_csv(os.path.join(STORY_3_OUTPUT_DIR, "eda_summary.csv"), index=False)
    print("EDA summary saved.")


# ==============================
# Run Story 3 pipeline
# ==============================

def run_story_3():
    print("\nStarting Story 3...\n")

    create_output_folder()
    df = load_cleaned_dataset()

    save_readmission_rate_by_group(df)
    plot_readmission_rate_by_age(df)
    plot_readmission_rate_by_race(df)

    numeric_summary_by_target(df)
    plot_time_in_hospital_distribution(df)
    plot_correlation_heatmap(df)

    plot_target_distribution(df)
    save_eda_summary(df)

    print("\nStory 3 completed successfully!")


if __name__ == "__main__":
    run_story_3()