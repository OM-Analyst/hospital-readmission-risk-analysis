\\# Hospital Readmission Risk Analysis







Predictive analytics and Power BI dashboard project analyzing 30-day hospital readmission risk using the Diabetes 130-US Hospitals dataset.







\\## Project Objective







The goal of this project is to analyze hospital readmission patterns and build a predictive analytics workflow to identify patients at risk of readmission.







\\## Dataset







This project uses:







\\- `diabetic\\\_data.csv`



\\- `IDs\\\_mapping.csv`







Dataset source: UCI Diabetes 130-US Hospitals dataset.







\\## Project Structure







```text



data/



src/



outputs/



notebooks/



dashboard/



docs/



requirements.txt



README.md

```

\## Progress Log



\### Story 1 – Data Understanding \& Quality Assessment



Completed initial dataset assessment using:



\- `src/data\_understanding.py`



Generated outputs:



\- `dataset\_shape.csv`

\- `data\_types.csv`

\- `missing\_values\_report.csv`

\- `question\_mark\_missing\_report.csv`

\- `target\_distribution.csv`

\- `unique\_values\_report.csv`

\- `numeric\_summary\_statistics.csv`



Analysis completed:



\- Dataset dimensions assessment

\- Data type inspection

\- Missing value analysis

\- Question-mark placeholder analysis

\- Target variable distribution review

\- Unique value analysis

\- Numeric summary statistics generation

### Story 2 – Data Cleaning & Preprocessing

✅ Completed

Completed using:

- `src/data_cleaning.py`

Cleaning steps performed:

- Dropped `weight` (96.86% missing) and `payer_code` (39.56% missing) — too sparse to impute
- Filled `race`, `medical_specialty`, `diag_1/2/3` missing ('?') values as "Unknown"
- Filled `max_glu_serum` / `A1Cresult` blanks as "Not Tested" (test-not-ordered is informative, not noise)
- Removed 2,423 encounters where discharge disposition = Expired/Hospice (readmission not meaningful for these)
- Removed 29,353 repeat encounters, keeping first per patient (prevents patient-level leakage)
- Binarized target: `readmitted_30_days` (1 = <30 days, 0 = otherwise)
- Removed identifier columns (`encounter_id`, `patient_nbr`) post-dedup

Result: 101,766 rows × 50 cols → 69,990 rows × 47 cols, 0 missing values

Outputs saved to:

```text
outputs/story-2/
```

Primary output:

```text
outputs/story-2/cleaned_readmission_data.csv
```
### Story 3 – Exploratory Data Analysis

✅ Completed

Completed using:

- `src/eda_analysis.py`

Analysis performed:

- Readmission rate by age, race, gender, admission type, discharge disposition, admission source, diabetes medication use, A1C/glucose results
- Numeric feature comparison (readmitted vs. not readmitted)
- Correlation matrix across numeric features and target
- Distribution plots: age, race, time in hospital, target balance

Key finding: readmission rate rises steadily with age, from 1.96% (ages 0–10) to 10.79% (ages 80–90).

Outputs saved to:

```text
outputs/story-3/
```
### Story 4 – Feature Engineering

✅ Completed

Completed using:

- `src/feature_engineering.py`

New features created:

- `age_numeric` (age bracket converted to midpoint)
- `total_prior_visits`, `had_prior_inpatient_visit`
- `num_medications_prescribed`, `num_medications_changed`
- `diag_1_category` (ICD-9 grouped into 9 clinical categories)

Encoding:

- Medication columns ordinal-encoded (No < Down < Steady < Up)
- `medical_specialty` bucketed to top 10 + "Other"
- Remaining categoricals one-hot encoded

Key finding: prior inpatient visits nearly double the readmission rate (15.4% vs. 8.13%).

Result: 69,990 rows × 47 cols → 69,990 rows × 82 cols, fully numeric, 0 nulls.

Outputs saved to:

```text
outputs/story-4/
```

Primary output:

```text
outputs/story-4/feature_engineered_data.csv
```

### Story 5 – Baseline Machine Learning Model

✅ Completed

Completed using:

- `src/baseline_model.py`

Approach:

- 80/20 stratified train/test split
- StandardScaler feature scaling
- Logistic Regression with class_weight='balanced' (to handle ~91/9 class imbalance)

Baseline results:

| Metric | Value |
|---|---|
| Accuracy | 0.629 |
| Precision | 0.128 |
| Recall | 0.537 |
| F1 Score | 0.207 |
| ROC-AUC | 0.628 |

Top predictors: discharge_disposition_id, number_inpatient, age_numeric, time_in_hospital

Outputs saved to:

```text
outputs/story-5/
```
### Story 6 – Model Evaluation

✅ Completed

Completed using:

- `src/model_evaluation.py`

Evaluation performed:

- Compared baseline vs. dummy (majority-class) classifier
- 5-fold cross-validation (ROC-AUC 0.625 ± 0.004 — confirms stability)
- Precision-Recall curve (Average Precision 0.150)
- Decision threshold tuning (0.3–0.7)

Key finding: default 0.5 threshold gives the best recall (0.537); 
threshold 0.6 gives the best F1 (0.216) but lower recall (0.278) — 
tradeoff to revisit once a stronger model exists.

Outputs saved to:

```text
outputs/story-6/
```

### Story 7 – Random Forest / XGBoost

✅ Completed

Completed using:

- `src/advanced_models.py`

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| XGBoost | 0.674 | 0.140 | 0.515 | 0.221 | 0.646 |
| Random Forest | 0.669 | 0.138 | 0.512 | 0.218 | 0.640 |
| Logistic Regression | 0.629 | 0.128 | 0.537 | 0.207 | 0.628 |

Key finding: XGBoost outperforms the baseline on ROC-AUC. 
discharge_disposition_id, number_inpatient, and prior inpatient visits 
remain the strongest predictors across all models.

Outputs saved to:

```text
outputs/story-7/
```

### Story 8 – SHAP Interpretation

✅ Completed

Completed using:

- `src/shap_interpretation.py`

Explains the Story 7 XGBoost model (best-performing model) using SHAP:

- Global feature importance (mean absolute SHAP value)
- Summary and bar plots
- Dependence plots for top 4 features
- Example individual patient risk explanation

Top drivers: discharge_disposition_id, number_inpatient, time_in_hospital, age_numeric

Outputs saved to:

```text
outputs/story-8/
```

### Story 9 – Power BI Dashboard

✅ Completed

Data prepared using:

- `src/dashboard_data_prep.py`

Output: `dashboard/dashboard_data.csv` — readable, dashboard-ready table
with decoded labels, engineered features, and model risk scores.

Dashboard built in Power BI Desktop (`dashboard/readmission_dashboard.pbix`), 3 pages:

**Overview** — KPI cards (readmission rate, total encounters, high-risk count, avg predicted risk), risk tier distribution, actual readmission rate by risk tier, readmission rate by age group

**Risk Drivers** — readmission rate by diagnosis category and discharge disposition, prior inpatient visit impact, avg prior visits by outcome

**Patient Detail** — sortable patient-level risk table with conditional formatting, avg predicted risk by gender and age, risk tier slicer

Key validation: High-risk tier shows 22.9% actual readmission rate vs. 3.2% for Low — confirms the model's risk scores are meaningfully separating patients.

\## Current Status

✅ Project setup completed

✅ Story 1 completed

✅ Story 2 completed

✅ Story 3 completed

✅ Story 4 completed 

✅ Story 5 completed

✅ Story 6 completed

✅ Story 7 completed

✅ Story 8 completed

✅ Story 9 completed

🔄 Story 10 – Recommendations & Reporting next