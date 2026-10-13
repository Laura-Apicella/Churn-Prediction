# 📡 Customer Churn Prediction — TIM

End-to-end machine learning pipeline to predict customer churn for TIM telecom lines, from raw data cleaning to risk segmentation for retention campaigns.


# [👉 Click here to see the whole project](https://github.com/Laura-Apicella/Churn-Prediction/blob/main/churn_prediction_TIM_FINALE_v2%20%283%29.ipynb)


## Overview

The goal is to estimate, for each customer line, the probability of terminating the contract, so that retention actions can be focused on the customers most likely to leave.

| | |
|---|---|
| **Customers analysed** | 198,537 (after removing 2,218 duplicate lines) |
| **Churn rate** | 38.7% |
| **Original features** | 56 |
| **Final features** | 53 (after engineering and selection) |
| **Split** | 80% train / 20% test (stratified) |

## Pipeline

1. **Data loading** – master table (customer features) and target table (lines that churned), semicolon-separated CSVs
2. **Target construction** – `churn = 1` if the line's `id_linea` appears in the target table
3. **Exploratory data analysis** – missing values, churn by technology / gender / age / ARPU, correlations
4. **Feature engineering** – tenure, trend and ratio features, behavioural flags, interactions
5. **Preprocessing** – median/mode imputation, scaling, ordinal and target encoding
6. **Modelling & evaluation** – 4 models compared, 5-fold cross-validation, threshold tuning, lift analysis
7. **Interpretability** – SHAP feature importance and beeswarm plots
8. **Risk segmentation & export** – per-customer churn probability and risk class

## Data quality fixes

| Issue | Fix |
|---|---|
| 2,218 duplicated `id_linea` | Dropped before the train/test split to avoid leakage |
| Informative missing values (+15pp churn) | Dedicated missing-indicator flags |
| Sentinel dates before 1990 (23.5%) | Flag + tenure computed only on valid dates |
| Structural missing ADSL data (FWA / STTH / RTG) | `flag_adsl_non_applicable` + neutral value, instead of median imputation |
| Highly skewed ADSL traffic (skewness > 400) | `log1p` transform |
| High-cardinality provinces | Target encoding with smoothing (k=20), fitted on train only |
| Technology underused by the model | `tech_churn_rate` encoding (24pp gap between STTH and FIBRA) |

## Feature engineering highlights

- **Tenure**: months since first activation / last activation
- **Usage trends**: month-over-month changes in voice and mobile traffic (IQR-clipped)
- **Competitor signals**: share of calls to other operators, a classic number-portability precursor
- **Behavioural flags**: "silent customer", "pays but doesn't use", voice usage drop, legacy technology
- **Interactions**: age × ARPU × tenure, ARPU × technology churn rate

## Models

| Model | AUC-ROC | Avg. Precision | F1 (churn) |
|---|---|---|---|
| Logistic Regression (baseline) | 0.623 | 0.497 | 0.521 |
| Random Forest | 0.656 | 0.544 | 0.544 |
| **XGBoost** | **0.662** | **0.551** | 0.552 |
| LightGBM | 0.662 | 0.551 | 0.553 |

Class imbalance is handled with `class_weight`, `scale_pos_weight` and `is_unbalance`, depending on the model.

## Results

| Metric | Value |
|---|---|
| **AUC-ROC (XGBoost, test set)** | 0.662 |
| **5-fold CV AUC** | 0.662 ± 0.001 |
| **Recall at F1-optimal threshold** | ~0.86 |
| **Lift, top 10% of customers** | ~1.65x vs. random |
| **High-risk segment** | ~12,000 customers, 76% actual churn rate |

Contacting the top 10% of customers by predicted churn probability captures about **1.65x more churners** than a random campaign of the same size. The very low CV standard deviation shows that performance is stable across folds.

### Risk segmentation

| Segment | Probability threshold |
|---|---|
| HIGH | ≥ 0.70 |
| MEDIUM | 0.45 – 0.70 |
| LOW | < 0.45 |

## Future improvements

Going beyond AUC ~0.67 would likely require external data sources:

- Call-center contact history (complaints)
- Measured vs. nominal connection quality
- History of plan changes over the last 12 months
- Presence of other TIM contracts (mobile + fixed bundles)

## Tech stack

Python · pandas · NumPy · scikit-learn · XGBoost · LightGBM · imbalanced-learn · category_encoders · SHAP · matplotlib · seaborn

---

# [👉 Click here to see the whole project](https://github.com/Laura-Apicella/Churn-Prediction/blob/main/churn_prediction_TIM_FINALE_v2%20%283%29.ipynb)




> **Note:** the datasets are not included in this repository.
