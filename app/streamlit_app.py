"""
Story 11 - Streamlit Interactive Risk Prediction App (Extra)
Hospital Readmission Risk Analysis

Loads the saved XGBoost model and lets a user enter a patient's
details to get a live 30-day readmission risk prediction.

Run with:
    streamlit run app/streamlit_app.py
"""

import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = "models/xgboost_model.pkl"
FEATURE_COLUMNS_PATH = "models/feature_columns.json"

AGE_MIDPOINTS = {
    "0-10": 5, "10-20": 15, "20-30": 25, "30-40": 35, "40-50": 45,
    "50-60": 55, "60-70": 65, "70-80": 75, "80-90": 85, "90-100": 95,
}

# Discharge dispositions, excluding death/hospice codes (11,13,14,19,20,21)
# since those encounters were removed during cleaning and are outside
# what the model was trained on.
DISCHARGE_DISPOSITIONS = {
    "Discharged to home": 1,
    "Discharged/transferred to another short term hospital": 2,
    "Discharged/transferred to SNF": 3,
    "Discharged/transferred to ICF": 4,
    "Discharged/transferred to home with home health service": 6,
    "Left AMA": 7,
    "Admitted as an inpatient to this hospital": 9,
    "Still patient or expected to return for outpatient services": 12,
    "Discharged/transferred to a long term care hospital": 23,
    "Not Mapped": 25,
    "Unknown/Invalid": 26,
}

ADMISSION_TYPES = {
    "Emergency": 1, "Urgent": 2, "Elective": 3, "Newborn": 4,
    "Not Available": 5, "Trauma Center": 7, "Not Mapped": 8,
}

ADMISSION_SOURCES = {
    "Physician Referral": 1, "Clinic Referral": 2, "HMO Referral": 3,
    "Transfer from a hospital": 4, "Transfer from a Skilled Nursing Facility": 5,
    "Transfer from another health care facility": 6, "Emergency Room": 7,
    "Court/Law Enforcement": 8, "Not Available": 9,
}

DIAG_CATEGORIES = [
    "Circulatory", "Respiratory", "Digestive", "Diabetes", "Injury",
    "Musculoskeletal", "Genitourinary", "Neoplasms", "Other", "Unknown",
]

MEDICAL_SPECIALTIES = [
    "Cardiology", "Emergency/Trauma", "Family/GeneralPractice", "InternalMedicine",
    "Nephrology", "Orthopedics", "Orthopedics-Reconstructive", "Radiologist",
    "Surgery-General", "Other", "Unknown",
]

DOSAGE_MAP = {"No": 0, "Down": 1, "Steady": 2, "Up": 3}

# Medication columns the model was trained on. Only insulin and metformin
# are exposed as inputs (they carry the most weight); the rest default to
# "No" (not prescribed) to keep the form usable.
ALL_MEDICATION_COLUMNS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "examide", "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone",
]


@st.cache_resource
def load_model_and_columns():
    model = joblib.load(MODEL_PATH)
    with open(FEATURE_COLUMNS_PATH) as f:
        feature_columns = json.load(f)
    return model, feature_columns


def build_feature_row(inputs, feature_columns):
    """Build a single-row DataFrame matching the model's training columns exactly."""
    row = {col: 0 for col in feature_columns}

    row["admission_type_id"] = inputs["admission_type_id"]
    row["discharge_disposition_id"] = inputs["discharge_disposition_id"]
    row["admission_source_id"] = inputs["admission_source_id"]
    row["time_in_hospital"] = inputs["time_in_hospital"]
    row["num_lab_procedures"] = inputs["num_lab_procedures"]
    row["num_procedures"] = inputs["num_procedures"]
    row["num_medications"] = inputs["num_medications"]
    row["number_outpatient"] = inputs["number_outpatient"]
    row["number_emergency"] = inputs["number_emergency"]
    row["number_inpatient"] = inputs["number_inpatient"]
    row["number_diagnoses"] = inputs["number_diagnoses"]

    for med in ALL_MEDICATION_COLUMNS:
        row[med] = 0  # default: not prescribed
    row["insulin"] = DOSAGE_MAP[inputs["insulin"]]
    row["metformin"] = DOSAGE_MAP[inputs["metformin"]]

    row["age_numeric"] = AGE_MIDPOINTS[inputs["age_bracket"]]
    row["total_prior_visits"] = (
        inputs["number_outpatient"] + inputs["number_emergency"] + inputs["number_inpatient"]
    )
    row["had_prior_inpatient_visit"] = int(inputs["number_inpatient"] > 0)

    prescribed_meds = [inputs["insulin"], inputs["metformin"]]
    row["num_medications_prescribed"] = sum(1 for m in prescribed_meds if m != "No")
    row["num_medications_changed"] = sum(1 for m in prescribed_meds if m in ("Up", "Down"))

    def set_dummy(prefix, value):
        col = f"{prefix}_{value}"
        if col in row:
            row[col] = 1

    set_dummy("gender", inputs["gender"])
    set_dummy("race", inputs["race"])
    set_dummy("max_glu_serum", inputs["max_glu_serum"])
    set_dummy("A1Cresult", inputs["a1c_result"])
    set_dummy("change", inputs["change"])
    set_dummy("diabetesMed", inputs["diabetes_med"])
    set_dummy("diag_1_category", inputs["diag_1_category"])
    set_dummy("medical_specialty", inputs["medical_specialty"])

    return pd.DataFrame([row])[feature_columns]


def main():
    st.set_page_config(page_title="Readmission Risk Predictor", layout="wide")
    st.title("🏥 30-Day Hospital Readmission Risk Predictor")
    st.caption(
        "Enter a patient's encounter details to get a live readmission risk score "
        "from the project's XGBoost model (ROC-AUC 0.646 on held-out test data)."
    )

    model, feature_columns = load_model_and_columns()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Demographics")
        age_bracket = st.selectbox("Age Group", list(AGE_MIDPOINTS.keys()), index=6)
        gender = st.selectbox("Gender", ["Female", "Male", "Unknown/Invalid"])
        race = st.selectbox("Race", ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other", "Unknown"])

        st.subheader("Admission")
        admission_type = st.selectbox("Admission Type", list(ADMISSION_TYPES.keys()))
        admission_source = st.selectbox("Admission Source", list(ADMISSION_SOURCES.keys()))
        discharge_disposition = st.selectbox("Discharge Disposition", list(DISCHARGE_DISPOSITIONS.keys()))

    with col2:
        st.subheader("Encounter Details")
        time_in_hospital = st.slider("Time in Hospital (days)", 1, 14, 4)
        num_lab_procedures = st.slider("Number of Lab Procedures", 0, 130, 43)
        num_procedures = st.slider("Number of Procedures", 0, 6, 1)
        num_medications = st.slider("Number of Medications", 1, 80, 16)
        number_diagnoses = st.slider("Number of Diagnoses", 1, 16, 7)
        diag_1_category = st.selectbox("Primary Diagnosis Category", DIAG_CATEGORIES)
        medical_specialty = st.selectbox("Admitting Medical Specialty", MEDICAL_SPECIALTIES)

    with col3:
        st.subheader("Prior Utilization")
        number_outpatient = st.number_input("Prior Outpatient Visits (past year)", 0, 50, 0)
        number_emergency = st.number_input("Prior Emergency Visits (past year)", 0, 50, 0)
        number_inpatient = st.number_input("Prior Inpatient Visits (past year)", 0, 20, 0)

        st.subheader("Labs & Medications")
        max_glu_serum = st.selectbox("Max Glucose Serum Test", ["Not Tested", "Norm", ">200", ">300"])
        a1c_result = st.selectbox("A1C Test Result", ["Not Tested", "Norm", ">7", ">8"])
        insulin = st.selectbox("Insulin", ["No", "Steady", "Up", "Down"])
        metformin = st.selectbox("Metformin", ["No", "Steady", "Up", "Down"])
        change = st.selectbox("Medication Changed This Encounter", ["No", "Ch"])
        diabetes_med = st.selectbox("On Diabetes Medication", ["Yes", "No"])

    st.divider()

    if st.button("Predict Readmission Risk", type="primary"):
        inputs = {
            "age_bracket": age_bracket, "gender": gender, "race": race,
            "admission_type_id": ADMISSION_TYPES[admission_type],
            "admission_source_id": ADMISSION_SOURCES[admission_source],
            "discharge_disposition_id": DISCHARGE_DISPOSITIONS[discharge_disposition],
            "time_in_hospital": time_in_hospital, "num_lab_procedures": num_lab_procedures,
            "num_procedures": num_procedures, "num_medications": num_medications,
            "number_diagnoses": number_diagnoses, "diag_1_category": diag_1_category,
            "medical_specialty": medical_specialty,
            "number_outpatient": number_outpatient, "number_emergency": number_emergency,
            "number_inpatient": number_inpatient,
            "max_glu_serum": max_glu_serum, "a1c_result": a1c_result,
            "insulin": insulin, "metformin": metformin,
            "change": change, "diabetes_med": diabetes_med,
        }

        X_input = build_feature_row(inputs, feature_columns)
        risk_score = model.predict_proba(X_input)[0, 1]

        if risk_score >= 0.55:
            tier, color = "High", "🔴"
        elif risk_score >= 0.40:
            tier, color = "Medium", "🟡"
        else:
            tier, color = "Low", "🟢"

        st.subheader("Prediction Result")
        r1, r2 = st.columns(2)
        r1.metric("Predicted Readmission Risk", f"{risk_score:.1%}")
        r2.metric("Risk Tier", f"{color} {tier}")

        st.progress(min(float(risk_score), 1.0))

        st.caption(
            "This score is descriptive, not diagnostic. Use it to help prioritize "
            "discharge planning and follow-up resources - not as a sole basis for "
            "clinical decisions."
        )


if __name__ == "__main__":
    main()
