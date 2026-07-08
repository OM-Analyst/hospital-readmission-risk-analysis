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

\## Current Status

✅ Project setup completed

✅ Story 1 completed

✅ Story 2 completed

✅ Story 3 completed

✅ Story 4 completed 

🔄 Story 5 – Baseline Machine Learning Model next