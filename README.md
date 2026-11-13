# 🌍 PM2.5 Air Quality Forecasting
### Mediterranean Basin · 2016–2026

<p align="center">
  <strong>A Data Science Project for Daily Air Quality Analysis and Forecasting</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Project-Data%20Science-2563EB?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Model-MLP-7C3AED?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Task-Regression-059669?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Region-Mediterranean-EA580C?style=for-the-badge" />
</p>

---

## 📌 01 · Project Overview

Air pollution is a major environmental challenge, particularly in densely populated and urban regions. Fine particulate matter (**PM2.5**) is an important air-quality indicator due to its potential impact on human health.

This project focuses on the **analysis and forecasting of daily surface PM2.5 concentrations across the Mediterranean Basin**, using historical air-quality observations, satellite-derived information, and meteorological variables.

The project emphasizes reliable temporal analysis, leakage-free preprocessing, and rigorous model evaluation to assess forecasting performance and generalization.

## 🎯 02 · Objectives

The main objectives of this project are to:

- Explore the spatial and temporal characteristics of PM2.5 concentrations.
- Investigate missing values, duplicates, outliers, and data quality.
- Analyze temporal trends, seasonal patterns, and autocorrelation.
- Study relationships between PM2.5 and atmospheric or meteorological predictors.
- Apply preprocessing and feature engineering while preventing data leakage.
- Develop an MLP-based regression model for daily PM2.5 forecasting.
- Evaluate predictive performance using appropriate validation strategies.
- Investigate model robustness and feature-importance stability.
- Ensure that experiments are reproducible and scientifically documented.

## 📊 03 · Dataset

**Dataset:** Air Quality – Mediterranean Basin (2016–2026)

The dataset combines air-quality observations with meteorological and atmospheric variables.

### Variables overview

| Category | Variables |
|:---|:---|
| 🎯 Target | PM2.5 |
| 🌡️ Meteorological | Temperature, Pressure, Wind U, Wind V |
| 🛰️ Atmospheric | NO2, CO, O3, AOD |
| 📍 Identification | Date, Station, Latitude, Longitude |

<p align="center">
  <strong>Prediction Target</strong><br/>
  Daily surface PM2.5 concentration<br/>
  <code>µg/m³</code>
</p>

## 🧪 04 · Methodology

The project follows a structured data science workflow.

### ① Exploratory Data Analysis

- Dataset overview and descriptive statistics.
- Missing-value and outlier investigation.
- Distribution and correlation analysis.
- PM2.5 concentration patterns.

### ② Temporal Analysis

- Timestamp regularity and missing dates.
- Autocorrelation and stationarity analysis.
- Monthly, yearly, and seasonal patterns.
- Temporal drift and lead–lag relationships.

### ③ Preprocessing & Feature Engineering

- Data cleaning and quality checks.
- Missing-data treatment.
- Temporal feature construction.
- Leakage prevention and station-aware processing.

### ④ Model Development & Evaluation

The project develops and evaluates a **Multi-Layer Perceptron (MLP)** neural network for daily PM2.5 forecasting.

The model uses available atmospheric, meteorological, and justified temporal features to learn nonlinear relationships associated with PM2.5 concentrations.

## 🤖 05 · Forecasting Model

<p align="center">
  <strong>MLP — Multi-Layer Perceptron</strong>
</p>

An MLP is a feedforward artificial neural network composed of interconnected layers of neurons. It can learn nonlinear relationships between input variables and a continuous target.

For this project, the MLP is used as a regression model to estimate daily surface PM2.5 concentrations.

## 📈 06 · Evaluation Metrics

Model performance will be assessed using the following regression metrics:

| Metric | Description |
|:---:|:---|
| **MAE** | Mean Absolute Error |
| **RMSE** | Root Mean Squared Error |
| **R²** | Coefficient of Determination |

Validation and test results will be reported separately to avoid misleading performance estimates.

## 📁 07 · Project Structure

```text
PM2.5-Air-Quality-Forecasting/
│
├── Data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── Exploratory_Data.py
│   ├── Exploratory_temporal.py
│   └── ...
│
├── Results/
│   ├── Figures/
│   ├── Exploratory_Data.csv
│   ├── Exploratory_temporal.csv
│   ├── validation_splits.npz
│   └── ...
│
├── README.md
└── requirements.txt
```

<sub>Note: The structure should be updated to reflect the final repository contents.</sub>

## ⚙️ 08 · Installation

### Step 1 — Clone the repository

```bash
git clone https://github.com/sabrine-ould/PM2.5-Air-Quality-Forecasting.git
```

### Step 2 — Navigate to the project directory

```bash
cd PM2.5-Air-Quality-Forecasting
```

### Step 3 — Create a virtual environment

```bash
python -m venv venv
```

### Step 4 — Install dependencies

```bash
pip install -r requirements.txt
```

## ▶️ 09 · Execution

Run the analysis scripts from the project root directory.

**Exploratory Data Analysis**

```bash
python src/Exploratory_Data.py
```

**Temporal Analysis**

```bash
python src/Exploratory_temporal.py
```

Run the remaining preprocessing, training, and evaluation scripts according to their final names and execution order.

## 📦 10 · Outputs

The project generates structured outputs for analysis and reproducibility.

| Output | Description |
|:---|:---|
| `Exploratory_Data.csv` | Exploratory analysis results |
| `Exploratory_temporal.csv` | Temporal analysis results |
| `results.csv` | Model evaluation results |
| `hyperparameters.csv` | Hyperparameter configurations |
| `feature_importance.csv` | Feature-importance results |
| `comparisons.csv` | Experimental comparisons |
| `experiments.csv` | Experiment records |
| `ablation.csv` | Ablation-study results |
| `ranking_stability.csv` | Feature-ranking stability |

Additional figures and diagnostic outputs are organized under the `Results/` directory.

## 🔬 11 · Reproducibility

To support reproducible experiments, the project follows these practices:

- Record software and library versions.
- Fix random seeds where applicable.
- Save and reuse validation splits.
- Fit preprocessing transformations using training data only.
- Keep the future test period separate from model development.
- Document model configurations and evaluation results.

## 👥 12 · Team & Academic Information

<p align="center">
  <strong>GROUP G8</strong><br/>
  Master 1 · Data Science<br/>
  University of Algiers 1<br/>
  Academic Year 2026–2027
</p>

## 📚 13 · Acknowledgements

This project is conducted as part of the Master 1 Data Science curriculum.

---

<p align="center">
  <strong>PM2.5 AIR QUALITY FORECASTING</strong><br/>
  <sub>Data Science · Machine Learning · Environmental Analysis</sub>
</p>

---

<p align="center">
  **PM2.5 Forecasting | Data Science Project**
</p>
