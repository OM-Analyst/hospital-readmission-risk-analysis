"""
Story 6 - Model Evaluation
Hospital Readmission Risk Analysis

Reads:
    outputs/story-4/feature_engineered_data.csv

Writes:
    outputs/story-6/*.csv
    outputs/story-6/*.png

Notes:
    Story 5 did not persist the trained model to disk, so this script
    rebuilds it using the identical split/scaling/training settings
    (same random_state=42, same stratify, same class_weight='balanced')
    to guarantee the exact same baseline model is being evaluated here.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_recall_curve, average_precision_score, roc_auc_score,
    precision_score, recall_score, f1_score, accuracy_score,
)

FEATURE_DATA_PATH = "outputs/story-4/feature_engineered_data.csv"
STORY_6_OUTPUT_DIR = "outputs/story-6"

TARGET = "readmitted_30_days"
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

# Thresholds to test when tuning the decision cutoff
CANDIDATE_THRESHOLDS = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]


def create_output_folder():
    os.makedirs(STORY_6_OUTPUT_DIR, exist_ok=True)


def load_feature_engineered_dataset():
    if not os.path.exists(FEATURE_DATA_PATH):
        raise FileNotFoundError(
            f"Feature-engineered dataset not found: {FEATURE_DATA_PATH}. "
            "Run src/feature_engineering.py (Story 4) first."
        )
    df = pd.read_csv(FEATURE_DATA_PATH)
    print(f"Feature-engineered dataset loaded. Shape: {df.shape}")
    return df


def rebuild_baseline_split_and_model(df):
    """Recreate the exact Story 5 split, scaling, and model."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train_scaled, y_train)

    print("Baseline model rebuilt for evaluation (matches Story 5 settings).")
    return model, X, y, X_train_scaled, X_test_scaled, y_train, y_test, scaler


# ==============================
# Compare against a naive baseline
# ==============================

def compare_against_dummy_classifier(model, X_train_scaled, X_test_scaled, y_train, y_test, model_roc_auc):
    """
    A DummyClassifier that always predicts the majority class ('not
    readmitted') would score ~91% accuracy on this data while catching
    zero actual readmissions. This shows why accuracy alone is
    misleading here, and gives a floor the real model must beat.
    """
    dummy = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
    dummy.fit(X_train_scaled, y_train)
    dummy_pred = dummy.predict(X_test_scaled)

    model_pred = model.predict(X_test_scaled)

    comparison = pd.DataFrame({
        "model": ["Dummy (majority class)", "Logistic Regression (baseline)"],
        "accuracy": [
            round(accuracy_score(y_test, dummy_pred), 4),
            round(accuracy_score(y_test, model_pred), 4),
        ],
        "recall": [
            round(recall_score(y_test, dummy_pred, zero_division=0), 4),
            round(recall_score(y_test, model_pred, zero_division=0), 4),
        ],
        "roc_auc": [0.5, round(model_roc_auc, 4)],
    })
    comparison.to_csv(os.path.join(STORY_6_OUTPUT_DIR, "dummy_vs_baseline_comparison.csv"), index=False)
    print("Dummy classifier comparison saved.")
    return comparison


# ==============================
# Cross-validation
# ==============================

def run_cross_validation(model, X, y):
    """
    5-fold stratified cross-validation on ROC-AUC to check the
    baseline's performance is stable and not a fluke of one split.
    Uses a fresh, unfitted model instance so each fold trains cleanly.
    """
    X_scaled = StandardScaler().fit_transform(X)
    cv_model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(cv_model, X_scaled, y, cv=skf, scoring="roc_auc")

    cv_results = pd.DataFrame({
        "fold": [f"fold_{i+1}" for i in range(len(scores))] + ["mean", "std"],
        "roc_auc": list(scores.round(4)) + [round(scores.mean(), 4), round(scores.std(), 4)],
    })
    cv_results.to_csv(os.path.join(STORY_6_OUTPUT_DIR, "cross_validation_results.csv"), index=False)
    print(f"5-fold CV ROC-AUC: {scores.mean():.4f} (+/- {scores.std():.4f})")
    return cv_results


# ==============================
# Precision-Recall curve
# ==============================

def save_precision_recall_curve(model, X_test_scaled, y_test):
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    avg_precision = average_precision_score(y_test, y_proba)

    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, color="darkorange",
             label=f"Precision-Recall (AP = {avg_precision:.3f})")
    baseline_rate = y_test.mean()
    plt.axhline(baseline_rate, linestyle="--", color="gray",
               label=f"No-skill baseline ({baseline_rate:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve - Baseline Model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(STORY_6_OUTPUT_DIR, "precision_recall_curve.png"))
    plt.close()

    print(f"Precision-Recall curve saved (Average Precision = {avg_precision:.4f}).")
    return y_proba, avg_precision


# ==============================
# Threshold tuning
# ==============================

def tune_decision_threshold(y_test, y_proba):
    """
    Logistic Regression defaults to a 0.5 cutoff, which is not
    necessarily optimal for an imbalanced target. Evaluate a range of
    thresholds so the best cutoff can be chosen based on the
    precision/recall tradeoff that matters for readmission screening
    (recall is usually prioritized - missing an at-risk patient costs
    more than a false alarm).
    """
    rows = []
    for t in CANDIDATE_THRESHOLDS:
        y_pred_t = (y_proba >= t).astype(int)
        rows.append({
            "threshold": t,
            "precision": round(precision_score(y_test, y_pred_t, zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred_t, zero_division=0), 4),
            "f1_score": round(f1_score(y_test, y_pred_t, zero_division=0), 4),
        })

    threshold_df = pd.DataFrame(rows)
    threshold_df.to_csv(os.path.join(STORY_6_OUTPUT_DIR, "threshold_tuning_results.csv"), index=False)

    best_row = threshold_df.loc[threshold_df["f1_score"].idxmax()]
    print(f"Best threshold by F1 score: {best_row['threshold']} "
          f"(precision={best_row['precision']}, recall={best_row['recall']}, f1={best_row['f1_score']})")

    plt.figure(figsize=(8, 5))
    plt.plot(threshold_df["threshold"], threshold_df["precision"], marker="o", label="Precision")
    plt.plot(threshold_df["threshold"], threshold_df["recall"], marker="o", label="Recall")
    plt.plot(threshold_df["threshold"], threshold_df["f1_score"], marker="o", label="F1 Score")
    plt.xlabel("Decision Threshold")
    plt.ylabel("Score")
    plt.title("Precision / Recall / F1 vs. Decision Threshold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(STORY_6_OUTPUT_DIR, "threshold_tuning_plot.png"))
    plt.close()

    return threshold_df, best_row


# ==============================
# Evaluation summary
# ==============================

def save_evaluation_summary(cv_results, avg_precision, best_threshold_row):
    cv_mean = cv_results.loc[cv_results["fold"] == "mean", "roc_auc"].values[0]
    cv_std = cv_results.loc[cv_results["fold"] == "std", "roc_auc"].values[0]

    summary = pd.DataFrame({
        "metric": [
            "5-fold CV ROC-AUC (mean)",
            "5-fold CV ROC-AUC (std)",
            "Average Precision (PR-AUC)",
            "Best threshold (by F1)",
            "Precision at best threshold",
            "Recall at best threshold",
            "F1 at best threshold",
        ],
        "value": [
            cv_mean, cv_std, round(avg_precision, 4), best_threshold_row["threshold"],
            best_threshold_row["precision"], best_threshold_row["recall"], best_threshold_row["f1_score"],
        ],
    })
    summary.to_csv(os.path.join(STORY_6_OUTPUT_DIR, "evaluation_summary.csv"), index=False)
    print("Evaluation summary saved.")


# ==============================
# Run Story 6 pipeline
# ==============================

def run_story_6():
    print("\nStarting Story 6...\n")

    create_output_folder()
    df = load_feature_engineered_dataset()

    model, X, y, X_train_scaled, X_test_scaled, y_train, y_test, scaler = \
        rebuild_baseline_split_and_model(df)

    baseline_roc_auc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])

    compare_against_dummy_classifier(model, X_train_scaled, X_test_scaled, y_train, y_test, baseline_roc_auc)
    cv_results = run_cross_validation(model, X, y)
    y_proba, avg_precision = save_precision_recall_curve(model, X_test_scaled, y_test)
    threshold_df, best_row = tune_decision_threshold(y_test, y_proba)
    save_evaluation_summary(cv_results, avg_precision, best_row)

    print("\nStory 6 completed successfully!")


if __name__ == "__main__":
    run_story_6()
