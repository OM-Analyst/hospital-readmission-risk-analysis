"""
Story 7 - Random Forest / XGBoost
Hospital Readmission Risk Analysis

Reads:
    outputs/story-4/feature_engineered_data.csv

Writes:
    outputs/story-7/*.csv
    outputs/story-7/*.png

Trains a Random Forest and an XGBoost classifier on the same
train/test split used in Story 5/6, and compares both against the
Story 5 Logistic Regression baseline.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
)
from xgboost import XGBClassifier

FEATURE_DATA_PATH = "outputs/story-4/feature_engineered_data.csv"
STORY_7_OUTPUT_DIR = "outputs/story-7"

TARGET = "readmitted_30_days"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def create_output_folder():
    os.makedirs(STORY_7_OUTPUT_DIR, exist_ok=True)


def load_feature_engineered_dataset():
    if not os.path.exists(FEATURE_DATA_PATH):
        raise FileNotFoundError(
            f"Feature-engineered dataset not found: {FEATURE_DATA_PATH}. "
            "Run src/feature_engineering.py (Story 4) first."
        )
    df = pd.read_csv(FEATURE_DATA_PATH)
    print(f"Feature-engineered dataset loaded. Shape: {df.shape}")
    return df


def split_data(df):
    """Same split settings as Story 5/6 so all models are comparable."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return X, X_train, X_test, y_train, y_test


# ==============================
# Models
# ==============================

def train_logistic_regression_baseline(X_train, X_test, y_train):
    """Re-trained here (scaled) purely so it can be compared side by
    side with the tree models on the same test set / same report."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train_scaled, y_train)
    return model, X_test_scaled


def train_random_forest(X_train, y_train):
    """
    Random Forest with class_weight='balanced' to handle the ~91/9
    imbalance. Trees don't need feature scaling, unlike Logistic
    Regression.
    """
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    print("Random Forest trained.")
    return model


def train_xgboost(X_train, y_train):
    """
    XGBoost with scale_pos_weight set to the negative/positive class
    ratio, which is XGBoost's equivalent of class_weight='balanced'.
    """
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    print("XGBoost trained.")
    return model


# ==============================
# Evaluation
# ==============================

def evaluate(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": model_name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    }
    return metrics, y_pred, y_proba


def save_model_comparison(all_metrics):
    comparison_df = pd.DataFrame(all_metrics).sort_values("roc_auc", ascending=False)
    comparison_df.to_csv(os.path.join(STORY_7_OUTPUT_DIR, "model_comparison.csv"), index=False)
    print("\nModel comparison:")
    print(comparison_df.to_string(index=False))
    return comparison_df


def save_roc_curve_comparison(results):
    """results: list of (model_name, y_test, y_proba, roc_auc)"""
    plt.figure(figsize=(7, 7))
    for name, y_test, y_proba, auc in results:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison - Logistic Regression vs. Random Forest vs. XGBoost")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(STORY_7_OUTPUT_DIR, "roc_curve_comparison.png"))
    plt.close()
    print("ROC curve comparison saved.")


def save_confusion_matrices(results_pred):
    """results_pred: list of (model_name, y_test, y_pred)"""
    fig, axes = plt.subplots(1, len(results_pred), figsize=(6 * len(results_pred), 5))
    if len(results_pred) == 1:
        axes = [axes]

    for ax, (name, y_test, y_pred) in zip(axes, results_pred):
        cm = confusion_matrix(y_test, y_pred)
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(name)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Not Readmitted", "Readmitted"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Not Readmitted", "Readmitted"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.tight_layout()
    plt.savefig(os.path.join(STORY_7_OUTPUT_DIR, "confusion_matrices_comparison.png"))
    plt.close()
    print("Confusion matrices comparison saved.")


def save_feature_importance(rf_model, xgb_model, feature_names):
    rf_importance = pd.DataFrame({
        "feature": feature_names,
        "random_forest_importance": rf_model.feature_importances_,
    }).sort_values("random_forest_importance", ascending=False)

    xgb_importance = pd.DataFrame({
        "feature": feature_names,
        "xgboost_importance": xgb_model.feature_importances_,
    }).sort_values("xgboost_importance", ascending=False)

    rf_importance.to_csv(os.path.join(STORY_7_OUTPUT_DIR, "random_forest_feature_importance.csv"), index=False)
    xgb_importance.to_csv(os.path.join(STORY_7_OUTPUT_DIR, "xgboost_feature_importance.csv"), index=False)

    top_n = 15
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    rf_top = rf_importance.head(top_n).iloc[::-1]
    axes[0].barh(rf_top["feature"], rf_top["random_forest_importance"], color="seagreen")
    axes[0].set_title("Random Forest - Top 15 Features")

    xgb_top = xgb_importance.head(top_n).iloc[::-1]
    axes[1].barh(xgb_top["feature"], xgb_top["xgboost_importance"], color="firebrick")
    axes[1].set_title("XGBoost - Top 15 Features")

    plt.tight_layout()
    plt.savefig(os.path.join(STORY_7_OUTPUT_DIR, "feature_importance_comparison.png"))
    plt.close()
    print("Feature importance comparison saved.")


# ==============================
# Run Story 7 pipeline
# ==============================

def run_story_7():
    print("\nStarting Story 7...\n")

    create_output_folder()
    df = load_feature_engineered_dataset()

    X, X_train, X_test, y_train, y_test = split_data(df)

    # Logistic Regression (from Story 5, retrained here for direct comparison)
    lr_model, X_test_scaled = train_logistic_regression_baseline(X_train, X_test, y_train)
    lr_metrics, lr_pred, lr_proba = evaluate(lr_model, X_test_scaled, y_test, "Logistic Regression")

    # Random Forest
    rf_model = train_random_forest(X_train, y_train)
    rf_metrics, rf_pred, rf_proba = evaluate(rf_model, X_test, y_test, "Random Forest")

    # XGBoost
    xgb_model = train_xgboost(X_train, y_train)
    xgb_metrics, xgb_pred, xgb_proba = evaluate(xgb_model, X_test, y_test, "XGBoost")

    all_metrics = [lr_metrics, rf_metrics, xgb_metrics]
    save_model_comparison(all_metrics)

    save_roc_curve_comparison([
        ("Logistic Regression", y_test, lr_proba, lr_metrics["roc_auc"]),
        ("Random Forest", y_test, rf_proba, rf_metrics["roc_auc"]),
        ("XGBoost", y_test, xgb_proba, xgb_metrics["roc_auc"]),
    ])

    save_confusion_matrices([
        ("Logistic Regression", y_test, lr_pred),
        ("Random Forest", y_test, rf_pred),
        ("XGBoost", y_test, xgb_pred),
    ])

    save_feature_importance(rf_model, xgb_model, X.columns.tolist())

    print("\nStory 7 completed successfully!")


if __name__ == "__main__":
    run_story_7()
