import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from src.config import CLEAN_DATA_FILE

# ============================================================
# 1. LOAD AND PREPARE DATA
# ============================================================


def load_and_prepare_data():

    # Load cleaned dataset
    df = pd.read_csv(CLEAN_DATA_FILE)

    # Convert date column to datetime
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Remove rows with invalid dates
    df = df.dropna(subset=["date"]).copy()

    # Define target variable
    target = "pm25"

    # Sentinel-5P features
    sentinel_features = ["NO2", "CO", "O3"]

    # Other numerical features
    other_numerical_features = [
        "temperature_celsius",
        "pressure_mb",
        "wind_u",
        "wind_v",
        "AOD",
    ]

    # Combine all numerical predictors
    continuous_features = sentinel_features + other_numerical_features

    # Convert numerical columns to numeric values
    for column in continuous_features + [target]:

        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Remove rows where the target is missing
    df = df.dropna(subset=[target]).copy()

    # Sort observations chronologically
    df = df.sort_values("date").reset_index(drop=True)

    return (
        df,
        target,
        sentinel_features,
        other_numerical_features,
        continuous_features,
    )


# ============================================================
# 2. BUILD PREPROCESSING PIPELINE
# ============================================================


def build_preprocessor(sentinel_features, other_numerical_features):

    # Sentinel-5P values:
    # Median imputation followed by RobustScaler
    sentinel_value_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )

    # Create missingness indicators for Sentinel-5P variables
    sentinel_indicator_pipeline = MissingIndicator(features="all")

    # Other numerical variables:
    # Median imputation followed by RobustScaler
    other_numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )

    # Combine all preprocessing operations
    preprocessor = ColumnTransformer(
        transformers=[
            ("sentinel_values", sentinel_value_pipeline, sentinel_features),
            ("sentinel_missing", sentinel_indicator_pipeline, sentinel_features),
            ("other_numerical", other_numerical_pipeline, other_numerical_features),
        ],
        remainder="drop",
    )

    return preprocessor
