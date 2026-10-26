# Polish Company Bankruptcy Prediction

Final project for ASDS 5302 (Principles of Data Science), University of Texas at Arlington, Summer 2026

Team: Humberto Alanis, Daniel Ramirez

## Overview

Missing a company that is about to go bankrupt can cost lenders and investors a lot of money. In this project we built classification models that use financial ratios to predict whether a Polish company will go bankrupt within the next 3 years.

The main challenge is class imbalance: fewer than 5% of the companies went bankrupt. Because of this, accuracy is not a useful metric. We focused on recall, precision, ROC-AUC, and PR-AUC.

## Data

- **Source:** [Polish Companies Bankruptcy Data](https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data), UCI Machine Learning Repository
- **Size:** 10,503 companies, 64 financial ratio features (year 3 file, 3-year forecasting window)
- **Target:** bankrupt (1) or still operating (0)
- **Class balance:** 10,008 operating (95.3%) and 495 bankrupt (4.7%)

## Preprocessing

- Removed 87 duplicate rows
- Filled missing values with the median, and added indicator columns that flag which values were missing
- Scaled features with `RobustScaler`, since the ratios are heavily skewed and full of outliers
- Dropped highly correlated features to reduce multicollinearity
- Used a stratified train/test split so both sets kept the same bankruptcy rate

## Models

We trained 7 models and compared two ways of handling the imbalance: class weights, and SMOTE oversampling (30% sampling strategy).

1. Logistic regression (baseline)
2. Logistic regression with L1 (Lasso) penalty
3. L1 logistic regression + SMOTE
4. Random Forest
5. Random Forest + SMOTE
6. XGBoost
7. XGBoost + SMOTE

We also tested lowering the decision threshold from 0.50 down to 0.15 to catch more bankruptcies.

## Results

Test set: 2,084 companies, 99 of them bankrupt. Recall and precision are for the bankrupt class at the default 0.50 threshold.

| Model | ROC-AUC | PR-AUC | Recall | Precision |
|---|---|---|---|---|
| L1 logistic regression | 77.4% | 19.4% | 54.5% | 18.0% |
| Random Forest | 83.9% | 37.5% | 25.3% | 58.1% |
| Random Forest + SMOTE | 85.5% | 42.8% | 28.3% | 58.3% |
| XGBoost | 89.3% | 53.1% | 48.5% | 50.5% |
| XGBoost + SMOTE | 89.3% | 51.9% | 43.4% | 58.1% |

The baseline logistic regression did not converge and only reached 61.0% ROC-AUC, so we left it out of the table.

**Key findings**
- XGBoost performed best overall, with the highest PR-AUC.
- Lowering XGBoost's threshold to 0.15 raised recall to 80.8%, catching 80 of the 99 bankruptcies with 19.7% precision. For a bank, missing a bankruptcy costs far more than reviewing a false alarm, so this trade-off can be worth it.
- SMOTE did not always beat class weights. It improved Random Forest, but for XGBoost it mostly traded some recall for fewer false alarms.
- Missing values turned out to be useful signals. Several missing-value indicators, especially for interest coverage (Attr27) and sales growth (Attr21), ranked among the most important features.

## Repo contents

| File | Description |
|---|---|
| `polish+companies+bankruptcy+data/` | Dataset files from UCI |
| `Data_Loading_EDA_Preprocessing.ipynb` | Data loading, EDA, and preprocessing |
| `Logistic_Regression_Model.ipynb` | Baseline, L1, and L1 + SMOTE logistic regression models |
| `XGBoost:RandomForest.ipynb` | Random Forest and XGBoost models, with and without SMOTE |
