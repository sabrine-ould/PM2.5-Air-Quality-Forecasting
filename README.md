# 🌍 PM2.5-Air-Quality-Forecasting — Mediterranean Basin

<p align="center">
  **A Data Science Project for Daily Air Quality Analysis and Forecasting**
</p>

**---

## 📌 Project Overview

Air pollution is a major environmental challenge, particularly in densely populated and urban regions. Fine particulate matter (PM2.5) is an important air-quality indicator due to its potential impact on human health.

This project focuses on the **analysis and forecasting of daily surface PM2.5 concentrations across the Mediterranean Basin**, using historical air-quality observations, satellite-derived information, and meteorological variables.

The project emphasizes reliable temporal analysis, leakage-free preprocessing, and rigorous model evaluation to assess forecasting performance and generalization.

## 🎯 Objectives

- Exxplore the spatial and temporal characteristics of PM2.5 concentrations.

Investigate missing values, duplicates, outliers, and data quality.

Analyze temporal trends, seasonal patterns, and autocorrelation.

Study relationships between PM2.5 and atmospheric or meteorological predictors.

Apply preprocessing and feature engineering while preventing data leakage.

Develop an MLP-based regression model for daily PM2.5 forecasting.

Evaluate predictive performance using appropriate validation strategies.

Investigate model robustness and feature-importance stability.

Ensure that experiments are reproducible and scientifically documented.

## 📊 Dataset

**Dataset:** Air Quality – Mediterranean Basin (2016–2026)

The dataset combines air-quality observations with meteorological and atmospheric variables.

| Category | Variables |
|---|---|
| Target | PM2.5 |
| Meteorological | Temperature, Pressure, Wind U, Wind V |
| Atmospheric | NO2, CO, O3, AOD |
| Identification | Date, Station, Latitude, Longitude |

**Prediction target:** Daily surface PM2.5 concentration, measured in µg/m³.

## 🧪 Methodology

The project follows a structured data science workflow:

1. **Exploratory Data Analysis**
   - Dataset overview and descriptive statistics.
   - Missing-value and outlier investigation.
   - Distribution and correlation analysis.
   - PM2.5 concentration patterns.

2. **Temporal Analysis**
   - Timestamp regularity and missing dates.
   - Autocorrelation and stationarity analysis.
   - Monthly, yearly, and seasonal patterns.
   - Temporal drift and lead–lag relationships.

3. **Preprocessing and Feature Engineering**
   - Data cleaning and quality checks.
   - Missing-data treatment.
   - Temporal feature construction.
   - Leakage prevention and station-aware processing.

4. **Model Development and Evaluation**
   The project develops and evaluates a Multi-Layer Perceptron (MLP) neural network for daily PM2.5 forecasting.

The model uses available atmospheric, meteorological, and justified temporal features to learn nonlinear relationships associated with PM2.5 concentrations.

## 🤖 Model

The assigned forecasting model for **Group G8** is:

**MLP — Multi-Layer Perceptron**

An MLP is a feedforward artificial neural network composed of interconnected layers of neurons. It can learn nonlinear relationships between input variables and a continuous target.

For this project, the MLP is used as a regression model to estimate daily surface PM2.5 concentrations.

## 📈 Evaluation

Model performance will be assessed using:

- MAE — Mean Absolute Error
- RMSE — Root Mean Squared Error
- R² — Coefficient of Determination

Validation and test results will be reported separately to avoid misleading performance estimates.

## 📁 Project Structure

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

*Note: Update this structure to match the final repository contents.*

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/sabrine-ould/PM2.5-Air-Quality-Forecasting.git
```

Move into the project directory:

```bash
cd PM2.5-Air-Quality-Forecasting
```

Create and activate a virtual environment:

```bash
python -m venv venv
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Execution

Run the analysis scripts from the project root directory:

```bash
python src/Exploratory_Data.py
```

```bash
python src/Exploratory_temporal.py
```

Run the remaining preprocessing, training, and evaluation scripts according to their final names and execution order.

## 📦 Outputs

The project generates structured outputs for analysis and reproducibility, including:

- `Exploratory_Data.csv`
- `Exploratory_temporal.csv`
- `results.csv`
- `hyperparameters.csv`
- `feature_importance.csv`
- `comparisons.csv`
- `experiments.csv`
- `ablation.csv`
- `ranking_stability.csv`

Additional figures and diagnostic outputs are organized under the `Results/` directory.

## 🔬 Reproducibility

To support reproducible experiments:

- Record the software and library versions.
- Fix random seeds where applicable.
- Save and reuse validation splits.
- Fit preprocessing transformations using training data only.
- Keep the future test period separate from model development.
- Document model configurations and evaluation results.

## 👥 Team

**Group:** G8  
**Program:** Master 1 — Data Science  
**Institution:** University of Algiers 1  
**Academic Year:** 2026–2027

## 📚 Acknowledgements

This project is conducted as part of the Master 1 Data Science curriculum.

---

<p align="center">
  **PM2.5 Forecasting | Data Science Project**
</p>
