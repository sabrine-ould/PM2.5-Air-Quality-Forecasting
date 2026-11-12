
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer

#% Load cleaned dataset

df = pd.read_csv("../Data/cleaned_pm25.csv")
#%Define numerical features

numerical_features = [
    "temperature_celsius",
    "pressure_mb",
    "wind_u",
    "wind_v",
    "NO2",
    "CO",
    "O3",
    "AOD"
]

#% Check missing values

print("Missing values before imputation:")
print(df[numerical_features].isnull().sum())
#% Check temporal range

df["date"] = pd.to_datetime(df["date"], errors="coerce")

df = df.sort_values("date")

print("First date:", df["date"].min())
print("Last date:", df["date"].max())

print("Number of observations:", len(df))
print("Number of unique dates:", df["date"].nunique())

#% Temporal Split - Date boundaries

unique_dates = sorted(df["date"].dropna().unique())

n_dates = len(unique_dates)

train_end = int(n_dates * 0.70)
val_end = int(n_dates * 0.85)

train_end_date = unique_dates[train_end - 1]
val_end_date = unique_dates[val_end - 1]

print("Training ends:", train_end_date)
print("Validation ends:", val_end_date)
print("Test ends:", unique_dates[-1])
#% Create Temporal Train / Validation / Test Sets

train_df = df[df["date"] <= train_end_date].copy()

validation_df = df[
    (df["date"] > train_end_date) &
    (df["date"] <= val_end_date)
].copy()

test_df = df[df["date"] > val_end_date].copy()


# Check split sizes
print("Training shape:", train_df.shape)
print("Validation shape:", validation_df.shape)
print("Test shape:", test_df.shape)


# Check date ranges
print("\nTraining period:")
print(train_df["date"].min(), "to", train_df["date"].max())

print("\nValidation period:")
print(validation_df["date"].min(), "to", validation_df["date"].max())

print("\nTest period:")
print(test_df["date"].min(), "to", test_df["date"].max())
#% Check Missing Values in Each Split

sentinel_variables = ["NO2", "CO", "O3"]

print("TRAINING:")
print(train_df[sentinel_variables].isnull().sum())

print("\nVALIDATION:")
print(validation_df[sentinel_variables].isnull().sum())

print("\nTEST:")
print(test_df[sentinel_variables].isnull().sum())

#% Investigate Missing Values by Date

for variable in ["NO2", "CO", "O3"]:

    missing_data = train_df[
        train_df[variable].isnull()
    ]

    print(f"\n{variable}")
    print("Number of missing records:", len(missing_data))
    print("First missing date:", missing_data["date"].min())
    print("Last missing date:", missing_data["date"].max())
    print("Stations affected:", missing_data["id"].nunique())
    
#% Median Imputation + Missingness Indicators

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Sentinel-5P variables
sentinel_variables = ["NO2", "CO", "O3"]

# Create imputation pipeline
sentinel_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(
        strategy="median",
        add_indicator=True
    ))
])

# Fit on training data only
train_sentinel = sentinel_pipeline.fit_transform(
    train_df[sentinel_variables]
)

# Transform validation and test using training medians
validation_sentinel = sentinel_pipeline.transform(
    validation_df[sentinel_variables]
)

test_sentinel = sentinel_pipeline.transform(
    test_df[sentinel_variables]
)

# Get output column names
sentinel_columns = sentinel_pipeline.get_feature_names_out()

# Convert results into DataFrames
train_sentinel = pd.DataFrame(
    train_sentinel,
    columns=sentinel_columns,
    index=train_df.index
)

validation_sentinel = pd.DataFrame(
    validation_sentinel,
    columns=sentinel_columns,
    index=validation_df.index
)

test_sentinel = pd.DataFrame(
    test_sentinel,
    columns=sentinel_columns,
    index=test_df.index
)

# Check results
print("Training missing values after imputation:")
print(train_sentinel.isnull().sum())

print("\nTraining columns:")
print(train_sentinel.columns.tolist())

#% Update datasets with imputed variables

for original_df, imputed_df in [
    (train_df, train_sentinel),
    (validation_df, validation_sentinel),
    (test_df, test_sentinel)
]:
    original_df.drop(columns=sentinel_variables, inplace=True)
    original_df[imputed_df.columns] = imputed_df
        

    
    #% Check all remaining missing values

for name, dataset in [
    ("Training", train_df),
    ("Validation", validation_df),
    ("Test", test_df)
]:
    print(f"\n{name} missing values:")
    print(dataset[sentinel_columns].isnull().sum())
    print(f"\n{name}:")
    print(dataset.isnull().sum()[dataset.isnull().sum() > 0])
    print(train_df.dtypes)
    
#%% Check categorical variables

categorical_columns = df.select_dtypes(
    include=["object", "string", "category"]
).columns

print("Categorical columns:")
print(categorical_columns.tolist())

print("\nMissing values:")
print(df[categorical_columns].isnull().sum())
#%% Missingness report by variable

sentinel_variables = ["NO2", "CO", "O3"]

missing_report = pd.DataFrame({
    "Missing_Count": df[sentinel_variables].isnull().sum(),
    "Missing_Percentage": (
        df[sentinel_variables].isnull().mean() * 100
    )
})

print(missing_report)

#%% Missingness by month

df["month"] = df["date"].dt.to_period("M")

for variable in sentinel_variables:

    monthly_missing = df.groupby("month")[variable].agg(
        Total_Records="count",
        Missing_Count=lambda x: x.isnull().sum()
    )

    monthly_missing["Total_Records"] = (
        df.groupby("month").size()
    )

    monthly_missing["Missing_Percentage"] = (
        monthly_missing["Missing_Count"] /
        monthly_missing["Total_Records"] * 100
    )

    print(f"\n{variable} - Missingness by month:")
    print(monthly_missing[monthly_missing["Missing_Count"] > 0])

#%% Missingness by station

for variable in sentinel_variables:

    station_missing = df.groupby("id")[variable].agg(
        Total_Records="count",
        Missing_Count=lambda x: x.isnull().sum()
    )

    station_missing["Total_Records"] = df.groupby("id").size()

    station_missing["Missing_Percentage"] = (
        station_missing["Missing_Count"] /
        station_missing["Total_Records"] * 100
    )

    print(f"\n{variable} - Missingness by station:")
    print(station_missing[station_missing["Missing_Count"] > 0])
    
    

#%% Step 1 - Define Features and Target

target = "pm25"

# Continuous numerical features
continuous_features = [
    "temperature_celsius",
    "pressure_mb",
    "wind_u",
    "wind_v",
    "AOD",
    "NO2",
    "CO",
    "O3"
]

# Binary missingness indicators
indicator_features = [
    "missingindicator_NO2",
    "missingindicator_CO",
    "missingindicator_O3"
]

# All selected features
feature_columns = continuous_features + indicator_features

# Separate features (X) and target (y)
X_train = train_df[feature_columns].copy()
y_train = train_df[target].copy()

X_val = validation_df[feature_columns].copy()
y_val = validation_df[target].copy()

X_test = test_df[feature_columns].copy()
y_test = test_df[target].copy()

# Check dimensions
print("X_train:", X_train.shape)
print("X_val:", X_val.shape)
print("X_test:", X_test.shape)


#% Step 2 - Check Missing Values

print("\nMissing Values in Training Features:")
print(X_train.isnull().sum())

print("\nMissing Values in Validation Features:")
print(X_val.isnull().sum())

print("\nMissing Values in Test Features:")
print(X_test.isnull().sum())


#% Step 3 - Outlier Analysis (Training Only)

Q1 = X_train[continuous_features].quantile(0.25)
Q3 = X_train[continuous_features].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = (
    (X_train[continuous_features] < lower_bound) |
    (X_train[continuous_features] > upper_bound)
)

outlier_report = pd.DataFrame({
    "Outlier_Count": outliers.sum(),
    "Outlier_Percentage": outliers.mean() * 100
})

print("\nOutlier Report:")
print(outlier_report)


#% Step 4 - RobustScaler

from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
import numpy as np

robust_pipeline = Pipeline(steps=[
    ("scaler", RobustScaler())
])

# Fit on training data only
X_train_robust_cont = robust_pipeline.fit_transform(
    X_train[continuous_features]
)

# Transform validation and test
X_val_robust_cont = robust_pipeline.transform(
    X_val[continuous_features]
)

X_test_robust_cont = robust_pipeline.transform(
    X_test[continuous_features]
)

# Keep binary indicators unchanged
X_train_robust = np.column_stack([
    X_train_robust_cont,
    X_train[indicator_features].to_numpy()
])

X_val_robust = np.column_stack([
    X_val_robust_cont,
    X_val[indicator_features].to_numpy()
])

X_test_robust = np.column_stack([
    X_test_robust_cont,
    X_test[indicator_features].to_numpy()
])

# Check dimensions
print("\nRobustScaler Results:")
print("Training:", X_train_robust.shape)
print("Validation:", X_val_robust.shape)
print("Test:", X_test_robust.shape)

# Verify binary indicators remain unchanged
print("\nBinary Indicators:")
print(np.unique(X_train_robust[:, len(continuous_features):]))


#%% Step 5 - Scaling and Feature Selection Within Each Training Fold

#%% Step 5 - Scaling and Feature Selection Within Each Training Fold

import numpy as np
import pandas as pd

from functools import partial
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 1. Define feature names and number of selected features

all_feature_names = continuous_features + indicator_features

k = 8


# 2. Define preprocessing

preprocessor = ColumnTransformer(
    transformers=[
        ("continuous", RobustScaler(), continuous_features),
        ("indicators", "passthrough", indicator_features)
    ]
)


# 3. Define Mutual Information

mi_function = partial(
    mutual_info_regression,
    discrete_features=[False] * len(continuous_features)
                     + [True] * len(indicator_features),
    random_state=42
)


# 4. Build the Pipeline

mlp_pipeline = Pipeline(steps=[
    ("preprocessing", preprocessor),

    ("feature_selection", SelectKBest(
        score_func=mi_function,
        k=k
    )),

    ("model", MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=300,
        random_state=42
    ))
])


# 5. Train and evaluate each fold separately

# Create Temporal Cross-Validation splits

from sklearn.model_selection import TimeSeriesSplit

# Get dates corresponding to training rows
train_dates = pd.to_datetime(df.loc[X_train.index, "date"])

# Sort unique dates
unique_train_dates = np.sort(train_dates.unique())

# Create 5 temporal folds with a 1-day gap
tscv = TimeSeriesSplit(n_splits=5, gap=1)

cv_splits = []

for train_date_idx, val_date_idx in tscv.split(unique_train_dates):

    train_fold_dates = unique_train_dates[train_date_idx]
    val_fold_dates = unique_train_dates[val_date_idx]

    train_idx = np.where(train_dates.isin(train_fold_dates))[0]
    val_idx = np.where(train_dates.isin(val_fold_dates))[0]

    cv_splits.append((train_idx, val_idx))

print("Number of CV folds:", len(cv_splits))

fold_results = []
selected_features_by_fold = []

for fold_number, (train_idx, val_idx) in enumerate(cv_splits, start=1):

    X_fold_train = X_train.iloc[train_idx]
    y_fold_train = y_train.iloc[train_idx]

    X_fold_val = X_train.iloc[val_idx]
    y_fold_val = y_train.iloc[val_idx]

    # Clone the pipeline for each fold
    fold_model = clone(mlp_pipeline)

    # Fit all steps using training fold only
    fold_model.fit(X_fold_train, y_fold_train)

    # Predict on validation fold
    y_pred = fold_model.predict(X_fold_val)

    # Record selected features
    selector = fold_model.named_steps["feature_selection"]

    selected_features = np.array(all_feature_names)[
        selector.get_support()
    ]

    selected_features_by_fold.append(
        list(selected_features)
    )

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_fold_val, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_fold_val, y_pred)
    )

    r2 = r2_score(y_fold_val, y_pred)

    fold_results.append({
        "Fold": fold_number,
        "Selected_Features": ", ".join(selected_features),
        "Number_of_Features": len(selected_features),
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })


# 6. Display selected features and results

results_df = pd.DataFrame(fold_results)

print("\nFeature Selection and Cross-Validation Results:")
print(results_df.to_string(index=False))


# 7. Count how often each feature was selected

feature_frequency = pd.Series(
    [
        feature
        for fold_features in selected_features_by_fold
        for feature in fold_features
    ]
).value_counts()

print("\nFeature Selection Frequency:")
print(feature_frequency)


# 8. Display average performance

print("\nMean Cross-Validation Performance:")

print("MAE:", results_df["MAE"].mean())
print("RMSE:", results_df["RMSE"].mean())
print("R2:", results_df["R2"].mean())

#%% 5.3.3 Investigating PM2.5 Outliers
#%% 5.3.3 Investigating Extreme PM2.5 Observations

# Step 1: Calculate IQR thresholds using training data only
Q1 = y_train.quantile(0.25)
Q3 = y_train.quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Step 2: Identify extreme PM2.5 observations
extreme_mask = (
    (y_train < lower_bound) |
    (y_train > upper_bound)
)

# Step 3: Examine extreme observations
extreme_values = df.loc[
    y_train.index[extreme_mask],
    ["date", "sensor_name", "latitude",
     "longitude", "pm25"]
].copy()

print("Lower threshold:", lower_bound)
print("Upper threshold:", upper_bound)

print("\nNumber of extreme observations:", len(extreme_values))

print("\nNegative PM2.5 values:", (y_train < 0).sum())
print("-999 values:", (y_train == -999).sum())

print("\nHighest extreme observations:")
print(
    extreme_values
    .sort_values(by="pm25", ascending=False)
    .head(20)
)

# Step 4: Preserve extreme observations
print("\nExtreme PM2.5 observations are preserved.")
print("No removal or capping is applied.")
#%2 Apply Training IQR Thresholds to Validation and Test

# Apply the same training thresholds to validation
val_outliers = (
    (y_val < lower_bound) |
    (y_val > upper_bound)
)

# Apply the same training thresholds to test
test_outliers = (
    (y_test < lower_bound) |
    (y_test > upper_bound)
)

# Display results
print("Training outliers:", extreme_mask.sum())

print("Validation outliers:", val_outliers.sum())
print("Validation percentage:", val_outliers.mean() * 100)

print("Test outliers:", test_outliers.sum())
print("Test percentage:", test_outliers.mean() * 100)

#%% 5.3.3 IQR Capping on Input Features

# Calculate IQR bounds for each continuous feature
feature_bounds = {}

for feature in continuous_features:

    Q1 = X_train[feature].quantile(0.25)
    Q3 = X_train[feature].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    feature_bounds[feature] = (lower, upper)

print("IQR thresholds for input features:")
for feature, bounds in feature_bounds.items():
    print(feature, bounds)

# Create copies of the datasets
#Step 2 Apply Capping
X_train_capped = X_train.copy()
X_val_capped = X_val.copy()

# Apply the same training thresholds
for feature, (lower, upper) in feature_bounds.items():

    X_train_capped[feature] = (
        X_train_capped[feature].clip(lower, upper)
    )

    X_val_capped[feature] = (
        X_val_capped[feature].clip(lower, upper)
    )
    
    #Step 3 — Train both models
# Model 1: Without IQR Capping
model_without = clone(mlp_pipeline)

model_without.fit(X_train, y_train)

y_pred_without = model_without.predict(X_val)


# Model 2: With IQR Capping
model_with = clone(mlp_pipeline)

model_with.fit(X_train_capped, y_train)

y_pred_with = model_with.predict(X_val_capped)
# Calculate evaluation metrics

print("\ncomparison:")
comparison = pd.DataFrame({

    "Model": [
        "Without IQR Capping",
        "With IQR Capping"
    ],

    "MAE": [
        mean_absolute_error(y_val, y_pred_without),
        mean_absolute_error(y_val, y_pred_with)
    ],

    "RMSE": [
        np.sqrt(mean_squared_error(y_val, y_pred_without)),
        np.sqrt(mean_squared_error(y_val, y_pred_with))
    ],

    "R2": [
        r2_score(y_val, y_pred_without),
        r2_score(y_val, y_pred_with)
    ]
})

print(comparison.to_string(index=False))


#%% 5.3.4 - Step 1: Validate timestamps and station identity



