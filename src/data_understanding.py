from pathlib import Path

import pandas as pd


INPUT_FILE = Path("data") / "diabetic_data.csv"
OUTPUT_DIR = Path("outputs") / "story-1"


def load_dataset(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    df = pd.read_csv(file_path)
    return df


def save_dataset_shape(df: pd.DataFrame) -> None:
    shape_summary = pd.DataFrame({
        "metric": ["rows", "columns"],
        "value": [df.shape[0], df.shape[1]]
    })

    shape_summary.to_csv(OUTPUT_DIR / "dataset_shape.csv", index=False)


def save_column_types(df: pd.DataFrame) -> None:
    data_types = pd.DataFrame({
        "column": df.columns,
        "data_type": df.dtypes.astype(str).values
    })

    data_types.to_csv(OUTPUT_DIR / "data_types.csv", index=False)


def save_missing_values_report(df: pd.DataFrame) -> None:
    missing_report = pd.DataFrame({
        "column": df.columns,
        "missing_count": df.isnull().sum().values,
        "missing_percent": (df.isnull().sum().values / len(df) * 100).round(2)
    })

    missing_report = missing_report.sort_values(
        by="missing_percent",
        ascending=False
    )

    missing_report.to_csv(
        OUTPUT_DIR / "missing_values_report.csv",
        index=False
    )


def save_question_mark_report(df: pd.DataFrame) -> None:
    question_mark_counts = []

    for col in df.columns:
        count = (df[col] == "?").sum()
        percent = round((count / len(df)) * 100, 2)

        question_mark_counts.append({
            "column": col,
            "question_mark_count": count,
            "question_mark_percent": percent
        })

    question_mark_report = pd.DataFrame(question_mark_counts)

    question_mark_report = question_mark_report.sort_values(
        by="question_mark_percent",
        ascending=False
    )

    question_mark_report.to_csv(
        OUTPUT_DIR / "question_mark_missing_report.csv",
        index=False
    )


def save_target_distribution(df: pd.DataFrame) -> None:
    if "readmitted" not in df.columns:
        raise ValueError("Target column 'readmitted' not found in dataset.")

    target_distribution = (
        df["readmitted"]
        .value_counts()
        .rename_axis("readmitted")
        .reset_index(name="count")
    )

    target_distribution["percent"] = (
        target_distribution["count"] / len(df) * 100
    ).round(2)

    target_distribution.to_csv(
        OUTPUT_DIR / "target_distribution.csv",
        index=False
    )


def save_unique_values_report(df: pd.DataFrame) -> None:
    unique_report = pd.DataFrame({
        "column": df.columns,
        "unique_values": [df[col].nunique() for col in df.columns]
    })

    unique_report = unique_report.sort_values(
        by="unique_values",
        ascending=False
    )

    unique_report.to_csv(
        OUTPUT_DIR / "unique_values_report.csv",
        index=False
    )


def save_numeric_summary(df: pd.DataFrame) -> None:
    numeric_summary = df.describe().T

    numeric_summary.to_csv(
        OUTPUT_DIR / "numeric_summary_statistics.csv"
    )


def run_story_1() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset(INPUT_FILE)

    save_dataset_shape(df)
    save_column_types(df)
    save_missing_values_report(df)
    save_question_mark_report(df)
    save_target_distribution(df)
    save_unique_values_report(df)
    save_numeric_summary(df)

    print("Story 1 data understanding complete.")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print(f"Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_story_1()