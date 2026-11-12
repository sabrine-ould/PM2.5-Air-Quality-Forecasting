# PM2.5-Air-Quality-Forecasting
PM2.5 Air Quality Forecasting Project

## Project Overview

Air pollution is an important environmental challenge, and fine particulate matter (PM2.5) is one of the key indicators used to assess air quality.

This project focuses on forecasting daily surface PM2.5 concentrations using historical air-quality observations, satellite-derived information, and meteorological variables.

The project aims to investigate and compare different forecasting approaches, with particular attention to their predictive performance, temporal generalization, and ability to generalize to unseen monitoring stations.

## Objectives

The main objectives of this project are:

- Explore and understand the available air-quality and environmental datasets.
- Perform data quality assessment, cleaning, and preprocessing.
- Investigate temporal patterns, missing observations, and potential data drift.
- Develop appropriate temporal, spatial, and spatio-temporal validation strategies.
- Implement and evaluate different PM2.5 forecasting models.
- Compare model performance using consistent evaluation protocols.
- Assess model robustness and the contribution of different feature groups.

## Data Description

The project uses three primary datasets:

| Dataset | Description |
|---|---|
| `pm25.csv` | PM2.5 observation data |
| `gee_features_daily.csv` | Daily environmental features derived from Google Earth Engine |
| `combined_gee_pm25.csv` | Combined PM2.5 observations and environmental features |

The combined dataset provides the basis for the subsequent data exploration and forecasting workflow.

## Methodology

The project follows a structured experimental workflow.

### 1. Data Exploration and Auditing

- Dataset structure and variable inspection.
- Descriptive statistical analysis.
- Missing-value and duplicate analysis.
- Outlier investigation.
- Correlation and multicollinearity analysis.
- Target distribution and station-level analysis.

### 2. Data Preprocessing

- Data quality checks and cleaning.
- Handling missing environmental observations.
- Missingness indicators for selected variables.
- Feature preparation.
- Chronological data splitting.
- Prevention of data leakage.

### 3. Temporal Analysis

- Sampling regularity and missing-date analysis.
- Autocorrelation and partial autocorrelation analysis.
- Trend and seasonal pattern investigation.
- Temporal drift analysis.

### 4. Validation Strategies

Four validation schemes are considered:

- **Temporal Cross-Validation:** evaluates performance on future observations.
- **Spatial Cross-Validation:** evaluates generalization to unseen stations.
- **Random Cross-Validation:** used as a diagnostic comparison, not for final model selection.
- **Spatio-Temporal Blocked Cross-Validation:** evaluates generalization across both time and location.

### 5. Forecasting and Model Evaluation

Different forecasting approaches will be implemented and evaluated under consistent validation conditions.

Model evaluation will include appropriate regression metrics, model comparisons, hyperparameter analysis, and robustness experiments.

## Project Organization

```text
PM25_Forecasting/
│
├── Data/
│   ├── pm25.csv
│   ├── gee_features_daily.csv
│   ├── combined_gee_pm25.csv
│   └── cleaned_pm25.csv
│
├── src/
│   ├── preprocessing.py
│   └── validation_schemes.py
│
├── Results/
│   ├── validation_splits.csv
│   ├── validation_splits.npz
│   ├── Exploratory_Data.csv
│   └── exploratory_temporal.csv
│
├── Report/
│
└── README.md
```


## Technologies and Tools

- **Programming Language:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Data Visualization:** Matplotlib, Seaborn
- **Version Control:** Git and GitHub

## Reproducibility

The project follows a consistent validation protocol. Cross-validation fold indices are saved and reused across model experiments to ensure comparable evaluation conditions.

The independent future test period is kept separate from model development and validation.

## Project Status

**Status:** In Progress

This repository contains the code, datasets organization, experimental results, and documentation developed throughout the PM2.5 forecasting project.
