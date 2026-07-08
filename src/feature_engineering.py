"""
Story 4 - Feature Engineering
Hospital Readmission Risk Analysis

Reads:
    outputs/story-2/cleaned_readmission_data.csv

Writes:
    outputs/story-4/*.csv
"""

import os
import pandas as pd
import numpy as np

CLEANED_DATA_PATH = "outputs/story-2/cleaned_readmission_data.csv"
STORY_4_OUTPUT_DIR = "outputs/story-4"

TARGET = "readmitted_30_days"

# Diabetes-related medication columns present in this dataset
MEDICATION_COLUMNS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "examide", "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone",
]

AGE_MIDPOINTS = {
    "[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35, "[40-50)": 45,
    "[50-60)": 55, "[60-70)": 65, "[70-80)": 75, "[80-90)": 85, "[90-100)": 95,
}

LOW_CARDINALITY_CATEGORICALS = [
    "gender", "race", "max_glu_serum", "A1Cresult", "change", "diabetesMed",
    "diag_1_category", "medical_specialty",
]


def create_output_folder():
    os.makedirs(STORY_4_OUTPUT_DIR, exist_ok=True)


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
# Feature: numeric age
# ==============================

def add_age_numeric(df):
    """Convert age bracket (e.g. '[80-90)') to its numeric midpoint,
    then drop the original bracket column since age_numeric replaces it."""
    df["age_numeric"] = df["age"].map(AGE_MIDPOINTS)
    df = df.drop(columns=["age"])
    return df


# ==============================
# Feature: prior healthcare utilization
# ==============================

def add_prior_utilization_features(df):
    """
    Combine prior visit counts into a single utilization measure and
    flag patients with any prior inpatient stay (a strong readmission
    risk signal in this dataset).
    """
    df["total_prior_visits"] = (
        df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]
    )
    df["had_prior_inpatient_visit"] = (df["number_inpatient"] > 0).astype(int)
    return df


# ==============================
# Feature: medication activity
# ==============================

def add_medication_features(df):
    """
    num_medications_prescribed: count of diabetes drugs actually
        prescribed (value != 'No') for this encounter
    num_medications_changed: count of drugs whose dosage changed
        (value in {'Up','Down'}) - a proxy for treatment instability
    """
    present_cols = [c for c in MEDICATION_COLUMNS if c in df.columns]

    df["num_medications_prescribed"] = (df[present_cols] != "No").sum(axis=1)
    df["num_medications_changed"] = df[present_cols].isin(["Up", "Down"]).sum(axis=1)

    return df


DOSAGE_ORDER = {"No": 0, "Down": 1, "Steady": 2, "Up": 3}


def encode_medication_columns(df):
    """
    Ordinal-encode each medication column (No < Down < Steady < Up)
    so the dosage direction/intensity is usable by numeric models,
    instead of dropping or one-hot-exploding 23 extra columns.
    """
    present_cols = [c for c in MEDICATION_COLUMNS if c in df.columns]
    for c in present_cols:
        df[c] = df[c].map(DOSAGE_ORDER).fillna(0).astype(int)
    return df


TOP_N_SPECIALTIES = 10


def bucket_medical_specialty(df):
    """
    medical_specialty has ~70+ categories. Keep the most common
    TOP_N_SPECIALTIES as-is and bucket the rest as 'Other' before
    one-hot encoding, to avoid an unmanageable number of columns.
    """
    top_specialties = df["medical_specialty"].value_counts().nlargest(TOP_N_SPECIALTIES).index
    df["medical_specialty"] = df["medical_specialty"].where(
        df["medical_specialty"].isin(top_specialties), other="Other"
    )
    return df


# ==============================
# Feature: primary diagnosis category
# ==============================

def categorize_icd9(code):
    """
    Group primary diagnosis ICD-9 codes into broad clinical categories,
    following the standard grouping used for this dataset.
    """
    if pd.isna(code) or code == "Unknown":
        return "Unknown"

    code = str(code)

    if code.startswith("V") or code.startswith("E"):
        return "Other"

    try:
        value = float(code)
    except ValueError:
        return "Other"

    if 390 <= value <= 459 or value == 785:
        return "Circulatory"
    if 460 <= value <= 519 or value == 786:
        return "Respiratory"
    if 520 <= value <= 579 or value == 787:
        return "Digestive"
    if 250 <= value < 251:
        return "Diabetes"
    if 800 <= value <= 999:
        return "Injury"
    if 710 <= value <= 739:
        return "Musculoskeletal"
    if 580 <= value <= 629 or value == 788:
        return "Genitourinary"
    if 140 <= value <= 239:
        return "Neoplasms"
    return "Other"


def add_diagnosis_category(df):
    df["diag_1_category"] = df["diag_1"].apply(categorize_icd9)
    return df


# ==============================
# Encode low-cardinality categoricals
# ==============================

def one_hot_encode_categoricals(df):
    """One-hot encode small categorical fields for modeling readiness."""
    present_cols = [c for c in LOW_CARDINALITY_CATEGORICALS if c in df.columns]
    df = pd.get_dummies(df, columns=present_cols, drop_first=False)
    return df


# ==============================
# Drop columns no longer needed after engineering
# ==============================

def drop_raw_columns(df):
    """
    Drop raw diag_1/2/3 codes (replaced by diag_1_category) and
    'readmitted' (replaced by readmitted_30_days back in Story 2).
    """
    cols_to_drop = [c for c in ["diag_1", "diag_2", "diag_3", "readmitted"] if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    pd.DataFrame({"removed_column": cols_to_drop}).to_csv(
        os.path.join(STORY_4_OUTPUT_DIR, "dropped_raw_columns.csv"), index=False
    )
    print(f"Dropped raw columns: {cols_to_drop}")
    return df


# ==============================
# Feature summary reports
# ==============================

def save_new_feature_summary(df):
    new_features = [
        "age_numeric", "total_prior_visits", "had_prior_inpatient_visit",
        "num_medications_prescribed", "num_medications_changed", "diag_1_category",
    ]
    present = [c for c in new_features if c in df.columns]

    rows = []
    for c in present:
        rows.append({
            "feature": c,
            "dtype": str(df[c].dtype),
            "mean": round(df[c].mean(), 2) if pd.api.types.is_numeric_dtype(df[c]) else None,
            "unique_values": df[c].nunique(),
        })
    pd.DataFrame(rows).to_csv(
        os.path.join(STORY_4_OUTPUT_DIR, "new_feature_summary.csv"), index=False
    )
    print("New feature summary saved.")


def save_diagnosis_category_distribution(df):
    dist = df["diag_1_category"].value_counts().reset_index()
    dist.columns = ["diag_1_category", "count"]
    dist.to_csv(
        os.path.join(STORY_4_OUTPUT_DIR, "diag_1_category_distribution.csv"), index=False
    )


def save_readmission_rate_by_new_features(df, original_categorized_col_values):
    """
    Readmission rate by diag_1_category and had_prior_inpatient_visit,
    computed before one-hot encoding destroys the original column.
    """
    df_copy = df.copy()
    df_copy["diag_1_category"] = original_categorized_col_values

    for col in ["diag_1_category", "had_prior_inpatient_visit"]:
        summary = (
            df_copy.groupby(col)[TARGET]
            .agg(encounters="count", readmitted="sum")
            .reset_index()
        )
        summary["readmission_rate_percent"] = (
            summary["readmitted"] / summary["encounters"] * 100
        ).round(2)
        summary = summary.sort_values("readmission_rate_percent", ascending=False)
        summary.to_csv(
            os.path.join(STORY_4_OUTPUT_DIR, f"readmission_rate_by_{col}.csv"),
            index=False,
        )
    print("Readmission rate by new features saved.")


def save_engineering_summary(original_df, engineered_df):
    summary = pd.DataFrame({
        "metric": ["Original rows", "Original columns", "Engineered rows", "Engineered columns"],
        "value": [original_df.shape[0], original_df.shape[1],
                 engineered_df.shape[0], engineered_df.shape[1]],
    })
    summary.to_csv(os.path.join(STORY_4_OUTPUT_DIR, "engineering_summary.csv"), index=False)
    print("Engineering summary saved.")


def save_engineered_dataset(df):
    output_path = os.path.join(STORY_4_OUTPUT_DIR, "feature_engineered_data.csv")
    df.to_csv(output_path, index=False)
    print(f"Feature-engineered dataset saved: {output_path}")
    return output_path


# ==============================
# Run Story 4 pipeline
# ==============================

def run_story_4():
    print("\nStarting Story 4...\n")

    create_output_folder()
    original_df = load_cleaned_dataset()

    df = add_age_numeric(original_df.copy())
    df = add_prior_utilization_features(df)
    df = add_medication_features(df)
    df = add_diagnosis_category(df)

    diag_category_values = df["diag_1_category"].copy()

    save_new_feature_summary(df)
    save_diagnosis_category_distribution(df)
    save_readmission_rate_by_new_features(df, diag_category_values)

    df = drop_raw_columns(df)
    df = encode_medication_columns(df)
    df = bucket_medical_specialty(df)
    df = one_hot_encode_categoricals(df)

    save_engineering_summary(original_df, df)
    save_engineered_dataset(df)

    print("\nStory 4 completed successfully!")


if __name__ == "__main__":
    run_story_4()
