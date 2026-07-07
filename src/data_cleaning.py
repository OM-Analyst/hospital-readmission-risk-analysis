import os
import pandas as pd
import numpy as np

RAW_DATA_PATH = "data/diabetic_data.csv"
IDS_MAPPING_PATH = "data/IDS_mapping.csv"
STORY_2_OUTPUT_DIR = "outputs/story-2"


# ==============================
# Setup
# ==============================

def create_output_folder():
    os.makedirs(STORY_2_OUTPUT_DIR, exist_ok=True)


def load_dataset(file_path=RAW_DATA_PATH):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    df = pd.read_csv(file_path)
    print("Dataset loaded successfully.")
    print(f"Shape: {df.shape}")
    return df


def load_ids_mapping(file_path=IDS_MAPPING_PATH):
    """
    data/IDS_mapping.csv stacks three lookup tables
    (admission_type_id, discharge_disposition_id, admission_source_id)
    separated by a blank ',' row. Split them into a dict of dicts:
        {"discharge_disposition_id": {11: "Expired", ...}, ...}
    """
    raw = pd.read_csv(file_path, header=None, names=["id", "description"])

    blank_idx = raw[raw["id"].isna()].index.tolist()
    blocks, start = [], 0
    for b in blank_idx + [len(raw)]:
        block = raw.iloc[start:b].dropna()
        if len(block) > 1:
            blocks.append(block)
        start = b + 1

    mapping = {}
    for block in blocks:
        header = block.iloc[0]["id"]
        body = block.iloc[1:].copy()
        body["id"] = pd.to_numeric(body["id"], errors="coerce")
        mapping[header] = dict(zip(body["id"], body["description"]))

    return mapping


# ==============================
# Story 1 confirmation
# ==============================

def review_story_1_outputs():
    """Confirms Story 1 outputs exist before continuing."""
    expected_files = [
        "dataset_shape.csv",
        "data_types.csv",
        "missing_values_report.csv",
        "question_mark_missing_report.csv",
        "target_distribution.csv",
        "unique_values_report.csv",
        "numeric_summary_statistics.csv",
    ]

    results = [
        {"file_name": f, "exists": os.path.exists(os.path.join("outputs/story-1", f))}
        for f in expected_files
    ]
    review_df = pd.DataFrame(results)
    review_df.to_csv(os.path.join(STORY_2_OUTPUT_DIR, "story_1_review.csv"), index=False)
    print("Story 1 output review completed.")
    return review_df


# ==============================
# Cleaning steps
# ==============================

def drop_sparse_columns(df):
    """weight (96.86% missing) and payer_code (39.56% missing) are too
    sparse to impute meaningfully -> drop."""
    cols_to_drop = [c for c in ["weight", "payer_code"] if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    pd.DataFrame({"removed_column": cols_to_drop,
                  "reason": ["Excessive missingness (>39% as '?')"] * len(cols_to_drop)}) \
        .to_csv(os.path.join(STORY_2_OUTPUT_DIR, "dropped_sparse_columns.csv"), index=False)

    print(f"Dropped sparse columns: {cols_to_drop}")
    return df


def handle_question_marks(df):
    """Replace '?' with NaN, then fill the remaining categorical gaps
    with 'Unknown' rather than dropping rows (missingness here is small
    and not related to the target)."""
    before = (df == "?").sum().sum()
    df = df.replace("?", np.nan)

    fill_unknown_cols = [c for c in ["race", "medical_specialty", "diag_1", "diag_2", "diag_3"]
                         if c in df.columns]
    for c in fill_unknown_cols:
        df[c] = df[c].fillna("Unknown")

    after = (df == "?").sum().sum()
    pd.DataFrame({"metric": ["Question marks before cleaning", "Question marks after cleaning"],
                  "value": [before, after]}) \
        .to_csv(os.path.join(STORY_2_OUTPUT_DIR, "question_mark_cleaning_summary.csv"), index=False)

    print("Question mark values resolved.")
    return df


def handle_lab_test_columns(df):
    """max_glu_serum and A1Cresult are genuinely blank (not '?') when a
    test was not ordered. Treat that as its own category rather than
    dropping the column or the rows."""
    lab_cols = [c for c in ["max_glu_serum", "A1Cresult"] if c in df.columns]
    for c in lab_cols:
        df[c] = df[c].fillna("Not Tested")

    print(f"Filled lab test columns with 'Not Tested': {lab_cols}")
    return df


def remove_death_and_hospice_encounters(df, ids_mapping):
    """Remove encounters where discharge_disposition_id maps to
    Expired or Hospice, since 30-day readmission is not a meaningful
    outcome for these patients."""
    disposition_map = ids_mapping.get("discharge_disposition_id", {})
    death_hospice_ids = [
        code for code, desc in disposition_map.items()
        if isinstance(desc, str) and ("expired" in desc.lower() or "hospice" in desc.lower())
    ]

    before = len(df)
    df = df[~df["discharge_disposition_id"].isin(death_hospice_ids)]
    removed = before - len(df)

    pd.DataFrame({
        "metric": ["Rows before removal", "Rows after removal", "Rows removed (death/hospice)"],
        "value": [before, len(df), removed]
    }).to_csv(os.path.join(STORY_2_OUTPUT_DIR, "death_hospice_removal_summary.csv"), index=False)

    print(f"Removed {removed} death/hospice discharge encounters "
          f"(disposition codes: {sorted(death_hospice_ids)}).")
    return df


def remove_duplicate_patient_encounters(df):
    """71,518 unique patients across 101,766 encounters. Keep only the
    first encounter per patient_nbr to prevent the same patient
    appearing in both train and test later (leakage)."""
    before = len(df)
    df = df.sort_values("encounter_id")
    df = df.drop_duplicates(subset="patient_nbr", keep="first")
    removed = before - len(df)

    pd.DataFrame({
        "metric": ["Rows before dedup", "Rows after dedup", "Repeat encounters removed"],
        "value": [before, len(df), removed]
    }).to_csv(os.path.join(STORY_2_OUTPUT_DIR, "duplicate_patient_summary.csv"), index=False)

    print(f"Removed {removed} repeat encounters, keeping first per patient.")
    return df


def create_readmission_target(df):
    """Binarize target: <30 -> 1, {>30, NO} -> 0."""
    if "readmitted" not in df.columns:
        raise ValueError("Column 'readmitted' does not exist.")

    df["readmitted_30_days"] = (df["readmitted"] == "<30").astype(int)

    target_summary = df["readmitted_30_days"].value_counts().reset_index()
    target_summary.columns = ["readmitted_30_days", "count"]
    target_summary["percentage"] = (target_summary["count"] / target_summary["count"].sum() * 100).round(2)
    target_summary.to_csv(os.path.join(STORY_2_OUTPUT_DIR, "target_after_cleaning.csv"), index=False)

    print("30-day readmission target created.")
    return df


def remove_identifiers(df):
    """encounter_id / patient_nbr are identifiers with no predictive
    value and risk leakage if left in as features."""
    columns_to_remove = [c for c in ["encounter_id", "patient_nbr"] if c in df.columns]
    df = df.drop(columns=columns_to_remove)

    pd.DataFrame({"removed_column": columns_to_remove,
                  "reason": ["Identifier column"] * len(columns_to_remove)}) \
        .to_csv(os.path.join(STORY_2_OUTPUT_DIR, "removed_identifier_columns.csv"), index=False)

    print(f"Removed identifier columns: {columns_to_remove}")
    return df


def save_missing_values_report(df):
    missing_report = pd.DataFrame({
        "column": df.columns,
        "missing_count": df.isnull().sum().values,
        "missing_percentage": (df.isnull().sum().values / len(df) * 100).round(2)
    }).sort_values(by="missing_percentage", ascending=False)

    missing_report.to_csv(os.path.join(STORY_2_OUTPUT_DIR, "missing_values_after_cleaning.csv"), index=False)
    print("Missing values report saved.")
    return missing_report


def save_cleaning_summary(original_df, cleaned_df):
    summary = pd.DataFrame({
        "metric": ["Original rows", "Original columns", "Cleaned rows", "Cleaned columns",
                   "Rows removed", "Columns removed"],
        "value": [original_df.shape[0], original_df.shape[1], cleaned_df.shape[0], cleaned_df.shape[1],
                  original_df.shape[0] - cleaned_df.shape[0], original_df.shape[1] - cleaned_df.shape[1]]
    })
    summary.to_csv(os.path.join(STORY_2_OUTPUT_DIR, "cleaning_summary.csv"), index=False)
    print("Cleaning summary saved.")


def save_clean_dataset(df):
    output_path = os.path.join(STORY_2_OUTPUT_DIR, "cleaned_readmission_data.csv")
    df.to_csv(output_path, index=False)
    print(f"Clean dataset saved: {output_path}")
    return output_path


# ==============================
# Run Story 2 pipeline
# ==============================

def run_story_2():
    print("\nStarting Story 2...\n")

    create_output_folder()

    original_df = load_dataset()
    ids_mapping = load_ids_mapping()

    review_story_1_outputs()

    df = drop_sparse_columns(original_df.copy())
    df = handle_question_marks(df)
    df = handle_lab_test_columns(df)
    df = remove_death_and_hospice_encounters(df, ids_mapping)
    df = remove_duplicate_patient_encounters(df)
    df = create_readmission_target(df)
    df = remove_identifiers(df)

    save_missing_values_report(df)
    save_cleaning_summary(original_df, df)
    save_clean_dataset(df)

    print("\nStory 2 completed successfully!")


if __name__ == "__main__":
    run_story_2()