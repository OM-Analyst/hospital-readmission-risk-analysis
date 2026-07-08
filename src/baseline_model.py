"""
Story 5 - Baseline Machine Learning Model
Hospital Readmission Risk Analysis

Reads:
    outputs/story-4/feature_engineered_data.csv

Writes:
    outputs/story-5/*.csv
    outputs/story-5/*.png
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, classification_report,
)

FEATURE_DATA_PATH = "outputs/story-4/feature_engineered_data.csv"
STORY_5_OUTPUT_DIR = "outputs/story-5"

TARGET = "readmitted_30_days"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def create_output_folder():
    os.makedirs(STORY_5_OUTPUT_DIR, exist_ok=True)


def load_feature_engineered_dataset():
    if not os.path.exists(FEATURE_DATA_PATH):
        raise FileNotFoundError(
            f"Feature-engineered dataset not found: {FEATURE_DATA_PATH}. "
            "Run src/feature_engineering.py (Story 4) first."
        )
    df = pd.read_csv(FEATURE_DATA_PATH)
    print(f"Feature-engineered dataset loaded. Shape: {df.shape}")
    return df


# ==============================
# Split
# ==============================

def split_features_and_target(df):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y


def train_test_split_data(X, y):
    """Stratified split preserves the ~9% positive rate in both sets."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    split_summary = pd.DataFrame({
        "set": ["train", "test"],
        "rows": [len(X_train), len(X_test)],
        "positive_rate_percent": [
            round(y_train.mean() * 100, 2), round(y_test.mean() * 100, 2)
        ],
    })
    split_summary.to_csv(os.path.join(STORY_5_OUTPUT_DIR, "train_test_split_summary.csv"), index=False)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


# ==============================
# Scale + train
# ==============================

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def train_baseline_model(X_train_scaled, y_train):
    """
    Logistic Regression with class_weight='balanced' to counter the
    ~91/9 class imbalance in readmitted_30_days - without this, a
    baseline model would just predict 'no readmission' for everyone
    and still score ~91% accuracy while catching zero real cases.
    """
    model = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
    )
    model.fit(X_train_scaled, y_train)
    print("Baseline Logistic Regression model trained.")
    return model


# ==============================
# Evaluate
# ==============================

def evaluate_model(model, X_test_scaled, y_test):
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    metrics_df = pd.DataFrame(list(metrics.items()), columns=["metric", "value"])
    metrics_df["value"] = metrics_df["value"].round(4)
    metrics_df.to_csv(os.path.join(STORY_5_OUTPUT_DIR, "baseline_model_metrics.csv"), index=False)

    print("Baseline model metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    report = classification_report(y_test, y_pred, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(
        os.path.join(STORY_5_OUTPUT_DIR, "classification_report.csv")
    )

    return y_pred, y_proba, metrics


def save_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["Actual: Not Readmitted", "Actual: Readmitted"],
        columns=["Predicted: Not Readmitted", "Predicted: Readmitted"],
    )
    cm_df.to_csv(os.path.join(STORY_5_OUTPUT_DIR, "confusion_matrix.csv"))

    plt.figure(figsize=(5, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix - Baseline Model")
    plt.xticks([0, 1], ["Not Readmitted", "Readmitted"])
    plt.yticks([0, 1], ["Not Readmitted", "Readmitted"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(STORY_5_OUTPUT_DIR, "confusion_matrix.png"))
    plt.close()
    print("Confusion matrix saved.")


def save_roc_curve(y_test, y_proba, roc_auc):
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {roc_auc:.3f})", color="steelblue")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Baseline Model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(STORY_5_OUTPUT_DIR, "roc_curve.png"))
    plt.close()
    print("ROC curve saved.")


# ==============================
# Feature importance (logistic regression coefficients)
# ==============================

def save_feature_importance(model, feature_names):
    importance = pd.DataFrame({
        "feature": feature_names,
        "coefficient": model.coef_[0],
    })
    importance["abs_coefficient"] = importance["coefficient"].abs()
    importance = importance.sort_values("abs_coefficient", ascending=False).drop(
        columns="abs_coefficient"
    )
    importance.to_csv(os.path.join(STORY_5_OUTPUT_DIR, "feature_importance.csv"), index=False)
    print("Feature importance saved.")


# ==============================
# Run Story 5 pipeline
# ==============================

def run_story_5():
    print("\nStarting Story 5...\n")

    create_output_folder()
    df = load_feature_engineered_dataset()

    X, y = split_features_and_target(df)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

    model = train_baseline_model(X_train_scaled, y_train)

    y_pred, y_proba, metrics = evaluate_model(model, X_test_scaled, y_test)
    save_confusion_matrix(y_test, y_pred)
    save_roc_curve(y_test, y_proba, metrics["roc_auc"])
    save_feature_importance(model, X.columns.tolist())

    print("\nStory 5 completed successfully!")


if __name__ == "__main__":
    run_story_5()