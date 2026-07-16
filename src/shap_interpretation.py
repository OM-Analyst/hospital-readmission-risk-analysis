"""
Story 8 - SHAP Interpretation
Hospital Readmission Risk Analysis

Reads:
    outputs/story-4/feature_engineered_data.csv

Writes:
    outputs/story-8/*.csv
    outputs/story-8/*.png

Explains the Story 7 XGBoost model (best ROC-AUC) using SHAP, so the
drivers of predicted readmission risk can be understood at both the
global (whole-model) and local (individual patient) level.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

FEATURE_DATA_PATH = "outputs/story-4/feature_engineered_data.csv"
STORY_8_OUTPUT_DIR = "outputs/story-8"

TARGET = "readmitted_30_days"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# SHAP on the full test set can be slow; sample for the plots but
# keep the underlying values available for the full set if needed.
SHAP_SAMPLE_SIZE = 2000


def create_output_folder():
    os.makedirs(STORY_8_OUTPUT_DIR, exist_ok=True)


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
    """Same split settings as Story 5/6/7 so this explains the same model."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return X_train, X_test, y_train, y_test


def train_xgboost(X_train, y_train):
    """Same XGBoost configuration as Story 7."""
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
    print("XGBoost trained (matches Story 7 settings).")
    return model


# ==============================
# SHAP computation
# ==============================

def compute_shap_values(model, X_test):
    """
    Sample the test set for plotting speed. TreeExplainer is exact
    for tree models, so this is not an approximation of the
    explanation itself - only the number of points visualized.
    """
    sample_size = min(SHAP_SAMPLE_SIZE, len(X_test))
    X_sample = X_test.sample(n=sample_size, random_state=RANDOM_STATE)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    print(f"SHAP values computed for {sample_size} test samples.")
    return explainer, shap_values, X_sample


# ==============================
# Global importance
# ==============================

def save_global_importance(shap_values, X_sample):
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": X_sample.columns,
        "mean_abs_shap_value": mean_abs_shap,
    }).sort_values("mean_abs_shap_value", ascending=False)

    importance_df.to_csv(
        os.path.join(STORY_8_OUTPUT_DIR, "shap_global_feature_importance.csv"), index=False
    )
    print("Global SHAP feature importance saved.")
    print(importance_df.head(10).to_string(index=False))
    return importance_df


def save_summary_plot(shap_values, X_sample):
    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(os.path.join(STORY_8_OUTPUT_DIR, "shap_summary_plot.png"), bbox_inches="tight")
    plt.close()
    print("SHAP summary (beeswarm) plot saved.")


def save_bar_plot(shap_values, X_sample):
    plt.figure()
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(os.path.join(STORY_8_OUTPUT_DIR, "shap_bar_plot.png"), bbox_inches="tight")
    plt.close()
    print("SHAP bar plot saved.")


# ==============================
# Dependence plots for top features
# ==============================

def save_dependence_plots(shap_values, X_sample, importance_df, top_n=4):
    top_features = importance_df["feature"].head(top_n).tolist()

    for feature in top_features:
        plt.figure()
        shap.dependence_plot(feature, shap_values, X_sample, show=False)
        plt.tight_layout()
        safe_name = feature.replace("/", "_")
        plt.savefig(
            os.path.join(STORY_8_OUTPUT_DIR, f"shap_dependence_{safe_name}.png"),
            bbox_inches="tight",
        )
        plt.close()

    print(f"SHAP dependence plots saved for: {top_features}")


# ==============================
# Individual patient explanation example
# ==============================

def save_example_patient_explanation(explainer, shap_values, X_sample, model):
    """
    Picks one high-risk predicted patient from the sample and saves
    the individual feature contributions behind that specific
    prediction, as an example of local interpretability.
    """
    probas = model.predict_proba(X_sample)[:, 1]
    highest_risk_idx = np.argmax(probas)

    patient_row = X_sample.iloc[highest_risk_idx]
    patient_shap = shap_values[highest_risk_idx]

    explanation_df = pd.DataFrame({
        "feature": X_sample.columns,
        "feature_value": patient_row.values,
        "shap_contribution": patient_shap,
    }).sort_values("shap_contribution", key=lambda s: s.abs(), ascending=False)

    explanation_df.to_csv(
        os.path.join(STORY_8_OUTPUT_DIR, "example_patient_explanation.csv"), index=False
    )
    print(f"\nExample high-risk patient (predicted probability = {probas[highest_risk_idx]:.4f}):")
    print(explanation_df.head(8).to_string(index=False))


# ==============================
# Run Story 8 pipeline
# ==============================

def run_story_8():
    print("\nStarting Story 8...\n")

    create_output_folder()
    df = load_feature_engineered_dataset()

    X_train, X_test, y_train, y_test = split_data(df)
    model = train_xgboost(X_train, y_train)

    explainer, shap_values, X_sample = compute_shap_values(model, X_test)

    importance_df = save_global_importance(shap_values, X_sample)
    save_summary_plot(shap_values, X_sample)
    save_bar_plot(shap_values, X_sample)
    save_dependence_plots(shap_values, X_sample, importance_df)
    save_example_patient_explanation(explainer, shap_values, X_sample, model)

    print("\nStory 8 completed successfully!")


if __name__ == "__main__":
    run_story_8()