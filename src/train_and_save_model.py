"""
Story 11 - Train and Save Final Model (for the Streamlit app)
Hospital Readmission Risk Analysis

Reads:
    outputs/story-4/feature_engineered_data.csv

Writes:
    models/xgboost_model.pkl
    models/feature_columns.json
"""

import os
import json
import joblib
import pandas as pd
from xgboost import XGBClassifier

FEATURE_DATA_PATH = "outputs/story-4/feature_engineered_data.csv"
MODELS_DIR = "models"
TARGET = "readmitted_30_days"
RANDOM_STATE = 42


def train_and_save():
    os.makedirs(MODELS_DIR, exist_ok=True)

    df = pd.read_csv(FEATURE_DATA_PATH)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    scale_pos_weight = (y == 0).sum() / (y == 1).sum()
    model = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight, eval_metric="logloss",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    model.fit(X, y)

    joblib.dump(model, os.path.join(MODELS_DIR, "xgboost_model.pkl"))
    with open(os.path.join(MODELS_DIR, "feature_columns.json"), "w") as f:
        json.dump(list(X.columns), f)

    print(f"Model trained on {len(X)} rows, {X.shape[1]} features.")
    print(f"Saved to {MODELS_DIR}/xgboost_model.pkl and {MODELS_DIR}/feature_columns.json")


if __name__ == "__main__":
    train_and_save()