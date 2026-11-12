
#%% Import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%% step1 Load the working dataset
df = pd.read_csv("../Data/combined_gee_pm25.csv")
#%1. 3. Dataset dimensions
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

#%1. 4. Variable names
print("\nColumns:")
print(df.columns.tolist())

#%1. Dataset information
print("\n=====================Dataset information:=============")
df.info()

#% 1.6. Target variable

target = "pm25"

print("Target variable:", target)
print("Target unit: µg/m³")

#%1.7. Temporal coverage

df["date"] = pd.to_datetime(df["date"], errors="coerce")

print("Start date:", df["date"].min())
print("End date:", df["date"].max())
print("Number of unique dates:", df["date"].nunique())

#%1. 8. Spatial coverage

print("Number of stations:", df["sensor_name"].nunique())

print("Number of GEE IDs:", df["id"].nunique())

print("Minimum latitude:", df["latitude"].min())
print("Maximum latitude:", df["latitude"].max())

print("Minimum longitude:", df["longitude"].min())
print("Maximum longitude:", df["longitude"].max())

#% 1.9. Variables classification

variables = {
    "Target": ["pm25"],
    "Temporal": ["date", "datetime_utc"],
    "Identification": ["id", "sensor_name"],
    "Geographical": ["latitude", "longitude"],
    "Meteorological": [
        "temperature_celsius",
        "pressure_mb",
        "wind_u",
        "wind_v"
    ],
    "Atmospheric": ["NO2", "CO", "O3", "AOD"]
}

for category, columns in variables.items():
    print(f"\n{category}:")
    for column in columns:
        print("-", column)



#%% Step 2: Numerical variables

numerical_variables = [
    "latitude",
    "longitude",
    "temperature_celsius",
    "pressure_mb",
    "wind_u",
    "wind_v",
    "NO2",
    "CO",
    "O3",
    "AOD",
    "pm25"
]

# Summary statistics

summary = df[numerical_variables].describe().T

summary["skewness"] = df[numerical_variables].skew()
summary["kurtosis"] = df[numerical_variables].kurt()

print("\nNumerical Variables Summary:")
print(summary.round(4))
# Count categories for relevant identifiers

# Count unique stations
print("Number of unique stations:")
print(df["sensor_name"].nunique())

# Count observations for each station
print("\nObservations per station:")
print(df["sensor_name"].value_counts())


# Count unique GEE IDs
print("\nNumber of unique GEE IDs:")
print(df["id"].nunique())

# Count observations for each GEE ID
print("\nObservations per GEE ID:")
print(df["id"].value_counts())

#%% Step 3: Visualize key variables

import matplotlib.pyplot as plt

# Prepare data for visualization
df_plot = df[
    (df["pm25"] > 0) &
    (df["pm25"] < 500)
].copy()

# Variables to visualize
variables = [
    "pm25",
    "AOD",
    "NO2",
    "CO",
    "O3",
    "temperature_celsius",
    "pressure_mb",
    "wind_u",
    "wind_v"
]


# 1. Histograms

fig, axes = plt.subplots(3, 3, figsize=(15, 12))

for variable, ax in zip(variables, axes.flatten()):
    df_plot[variable].dropna().hist(
        bins=40, ax=ax, edgecolor="black"
    )
    ax.set_title(f"Histogram of {variable}")
    ax.set_xlabel(variable)
    ax.set_ylabel("Frequency")

plt.tight_layout()
plt.show()


# 2. Boxplots

fig, axes = plt.subplots(3, 3, figsize=(15, 12))

for variable, ax in zip(variables, axes.flatten()):
    df_plot.boxplot(column=variable, ax=ax)
    ax.set_title(f"Boxplot of {variable}")
    ax.set_ylabel(variable)

plt.tight_layout()
plt.show()


# 3. Categorical plot: PM2.5 by season

# Convert date to datetime
df_plot["date"] = pd.to_datetime(
    df_plot["date"], errors="coerce"
)

# Extract month
df_plot["month"] = df_plot["date"].dt.month

# Define seasons
season_mapping = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Autumn", 10: "Autumn", 11: "Autumn"
}

df_plot["season"] = df_plot["month"].map(season_mapping)

season_order = ["Winter", "Spring", "Summer", "Autumn"]

df_plot.boxplot(
    column="pm25",
    by="season",
    grid=False
)

plt.title("PM2.5 Distribution by Season")
plt.suptitle("")
plt.xlabel("Season")
plt.ylabel("PM2.5 (µg/m³)")
plt.show()

# 4. Categorical plot: PM2.5 by station
# Select the 10 stations with the most observations
top_stations = df_plot["sensor_name"].value_counts().head(10).index

station_data = df_plot[
    df_plot["sensor_name"].isin(top_stations)
]

station_data.boxplot(
    column="pm25",
    by="sensor_name",
    figsize=(14, 6),
    grid=False
)

plt.title("PM2.5 Distribution for the Top 10 Stations")
plt.suptitle("")
plt.xlabel("Station")
plt.ylabel("PM2.5 (µg/m³)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#%% Step 4:  Missing values analysis

missing_count = df.isnull().sum()
# Calculate missing percentage
missing_percentage = (missing_count / len(df)) * 100
# Create summary table
missing_summary = pd.DataFrame({
    "Missing Count": missing_count,
    "Missing Percentage (%)": missing_percentage
})

print(missing_summary.round(2))



#% 4.2. Boxplots

variables = [
    "pm25", "AOD", "NO2", "CO", "O3",
    "temperature_celsius", "pressure_mb",
    "wind_u", "wind_v"
]

for col in variables:

    values = df[col].dropna()

    plt.figure(figsize=(8, 4))
    plt.boxplot(values)

    plt.title("Boxplot of " + col)
    plt.ylabel(col)

    plt.show()


#% 4.3. IQR Outlier Detection

iqr_results = []

for col in variables:

    values = df[col].dropna()

    Q1 = values.quantile(0.25)
    Q3 = values.quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = values[
        (values < lower_bound) |
        (values > upper_bound)
    ]

    iqr_results.append({
        "Variable": col,
        "IQR Outliers": len(outliers)
    })

iqr_summary = pd.DataFrame(iqr_results)

print("\nIQR Results:")
print(iqr_summary.to_string(index=False))


#% 4.4. Z-score Outlier Detection

from scipy.stats import zscore

zscore_results = []

for col in variables:

    data = df[col].dropna()

    z_scores = zscore(data)

    outlier_count = (abs(z_scores) > 3).sum()

    zscore_results.append({
        "Variable": col,
        "Z-score Outliers": outlier_count
    })

zscore_summary = pd.DataFrame(zscore_results)

print("\nZ-score Results:")
print(zscore_summary.to_string(index=False))


#% 4.5. IQR vs Z-score Comparison

comparison = iqr_summary.merge(
    zscore_summary,
    on="Variable"
)

x = np.arange(len(comparison))
width = 0.35

plt.figure(figsize=(12, 6))

plt.bar(
    x - width/2,
    comparison["IQR Outliers"],
    width,
    label="IQR"
)

plt.bar(
    x + width/2,
    comparison["Z-score Outliers"],
    width,
    label="Z-score"
)

plt.title("Comparison of Outliers Detected by IQR and Z-score")
plt.xlabel("Variables")
plt.ylabel("Number of Potential Outliers")

plt.xticks(x, comparison["Variable"], rotation=45)
plt.legend()

plt.tight_layout()
plt.show()

#%% Isolation Forest and Local Outlier Factor

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

environmental_variables = [
    "temperature_celsius", "pressure_mb",
    "wind_u", "wind_v", "NO2", "CO", "O3", "AOD"
]

# Select environmental variables
X = df[environmental_variables].copy()

# Handle missing values
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

# Scale variables
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Isolation Forest
if_model = IsolationForest(
    contamination="auto",
    random_state=42
)

if_labels = if_model.fit_predict(X_scaled)

n_out_if = (if_labels == -1).sum()
pct_out_if = n_out_if / len(X_scaled) * 100

# Local Outlier Factor
lof_model = LocalOutlierFactor(
    n_neighbors=20,
    contamination="auto"
)

lof_labels = lof_model.fit_predict(X_scaled)

n_out_lof = (lof_labels == -1).sum()
pct_out_lof = n_out_lof / len(X_scaled) * 100

print("Isolation Forest:", n_out_if, pct_out_if)
print("LOF:", n_out_lof, pct_out_lof)


#%% Step 5 - Remove invalid PM2.5 rows


# Create cleaned dataset
df_clean = df[
    df["pm25"].notna() &
    (df["pm25"] > 0) &
    (df["pm25"] < 500)
].copy()
# Count missing PM2.5 values
missing_pm25 = df["pm25"].isna().sum()

# Count invalid PM2.5 values
missing_pm25 = df["pm25"].isna().sum()

invalid_low = (df["pm25"] <= 0).sum()
invalid_high = (df["pm25"] >= 500).sum()

df_clean = df[
    df["pm25"].notna() &
    (df["pm25"] > 0) &
    (df["pm25"] < 500)
].copy()

# Display cleaning results
print("Original dataset:", df.shape)
print("Missing PM2.5 rows:", missing_pm25)
print("PM2.5 <= 0:", invalid_low)
print("PM2.5 >= 500:", invalid_high)
print("Total rows removed:", len(df) - len(df_clean))
print("Cleaned dataset:", df_clean.shape)
#%% 6.1. Check duplicates using all columns

duplicates_all = df_clean.duplicated().sum()

print("Duplicate rows (all columns):", duplicates_all)
#%6.2. Check duplicates using key fields

key_fields = [
    "date",
    "latitude",
    "longitude",
    "sensor_name"
]

duplicates_key = df_clean.duplicated(
    subset=key_fields
).sum()

print("Duplicate rows (key fields):", duplicates_key)
#% 6.3. Check duplicates using GEE ID

gee_key = [
    "date",
    "latitude",
    "longitude",
    "id"
]

duplicates_gee = df_clean.duplicated(
    subset=gee_key
).sum()

print("Duplicate rows (GEE key):", duplicates_gee)

#%% Missing values in cleaned dataset

missing = df_clean.isnull().sum()

missing_percent = (missing / len(df)) * 100

missing_summary = pd.DataFrame({
    "Missing Values": missing,
    "Percentage (%)": missing_percent
})

print(missing_summary)
#%% 7 1 Correlation analysis

import pandas as pd

# Select environmental numerical variables
variables = [
    "pm25", "AOD", "NO2", "CO", "O3",
    "temperature_celsius", "pressure_mb",
    "wind_u", "wind_v"
]

# Pearson correlation
pearson_corr = df_clean[variables].corr(method="pearson")

print("PEARSON CORRELATION:")
print(pearson_corr.round(3))

# Spearman correlation
spearman_corr = df_clean[variables].corr(method="spearman")

print("\nSPEARMAN CORRELATION:")
print(spearman_corr.round(3))

#% 7.2Identify highly correlated predictors

predictors = variables[1:]

for method, corr in [
    ("Pearson", pearson_corr),
    ("Spearman", spearman_corr)
]:
    print(f"\n{method} correlations above 0.9:")

    found = False

    for i in range(len(predictors)):
        for j in range(i + 1, len(predictors)):
            var1 = predictors[i]
            var2 = predictors[j]
            value = corr.loc[var1, var2]

            if abs(value) > 0.9:
                print(var1, "-", var2, ":", round(value, 3))
                found = True

    if not found:
        print("No highly correlated predictor pairs found.")

#%7.3 VIF analysis

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Use rows with complete predictor values
X = df_clean[predictors].dropna()

# Add constant
X = sm.add_constant(X)

# Calculate VIF
vif_data = pd.DataFrame()
vif_data["Variable"] = predictors
vif_data["VIF"] = [
    variance_inflation_factor(X.values, i + 1)
    for i in range(len(predictors))
]

print(vif_data)


#% t-SNE visualization

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Select the 8 predictors
predictors = [
    "AOD", "NO2", "CO", "O3",
    "temperature_celsius", "pressure_mb",
    "wind_u", "wind_v"
]

# Prepare the data
X = df_clean[predictors]
y = df_clean["pm25"]

# Handle missing values
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

# Standardize the variables
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Use a sample to reduce computation time
sample_size = min(5000, len(X_scaled))

X_sample = X_scaled[:sample_size]
y_sample = y.iloc[:sample_size]

# Apply t-SNE
tsne = TSNE(
    n_components=2,
    random_state=42,
    perplexity=30
)

X_tsne = tsne.fit_transform(X_sample)

# Plot the results
plt.figure(figsize=(8, 6))

scatter = plt.scatter(
    X_tsne[:, 0],
    X_tsne[:, 1],
    c=y_sample,
    cmap="viridis",
    s=10
)

plt.colorbar(scatter, label="PM2.5")
plt.title("t-SNE Visualization of Environmental Variables")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.show()

#% UMAP visualization

import umap

# Apply UMAP
umap_model = umap.UMAP(
    n_components=2,
    random_state=42
)

X_umap = umap_model.fit_transform(X_sample)

# Plot the results
plt.figure(figsize=(8, 6))

scatter = plt.scatter(
    X_umap[:, 0],
    X_umap[:, 1],
    c=y_sample,
    cmap="viridis",
    s=10
)

plt.colorbar(scatter, label="PM2.5")
plt.title("UMAP Visualization of Environmental Variables")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.show()



#%% 8.1. PM2.5 vs Predictors

import matplotlib.pyplot as plt
import seaborn as sns

predictors = [
    "AOD", "NO2", "CO", "O3",
    "temperature_celsius", "pressure_mb",
    "wind_u", "wind_v"
]

# Random sample for faster visualization
plot_data = df_clean.sample(
    n=min(5000, len(df_clean)),
    random_state=42
)

y_limit = df_clean["pm25"].quantile(0.99)

fig, axes = plt.subplots(4, 2, figsize=(14, 18))

for ax, col in zip(axes.flatten(), predictors):

    sns.regplot(
        data=plot_data,
        x=col,
        y="pm25",
        lowess=True,
        ax=ax,
        scatter_kws={"alpha": 0.2, "s": 8},
        line_kws={"color": "red", "linewidth": 2}
    )

    ax.set_ylim(0, y_limit)
    ax.set_title(f"PM2.5 vs {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("PM2.5 (µg/m³)")

plt.tight_layout()
plt.show()

#% 8.2. Seasonal Analysis

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Extract month from date
df_clean["date"] = pd.to_datetime(df_clean["date"])
df_clean["month"] = df_clean["date"].dt.month

# Assign seasons
def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Autumn"

df_clean["season"] = df_clean["month"].apply(get_season)

# Seasonal statistics
season_order = ["Winter", "Spring", "Summer", "Autumn"]

seasonal_pm25 = df_clean.groupby("season")["pm25"].agg(
    ["count", "mean", "median", "std"]
).reindex(season_order)

print("Seasonal PM2.5 Statistics:")
print(seasonal_pm25.round(3))

# Seasonal PM2.5 Boxplot

plt.figure(figsize=(9, 5))

sns.boxplot(
    data=df_clean,
    x="season",
    y="pm25",
    order=season_order
)

plt.title("PM2.5 Distribution Across Seasons")
plt.xlabel("Season")
plt.ylabel("PM2.5 (µg/m³)")
plt.ylim(0, df_clean["pm25"].quantile(0.99))

plt.tight_layout()
plt.show()

#% 8.3. PM2.5 Differences Across Stations

# Calculate statistics for each station
station_pm25 = df_clean.groupby("sensor_name")["pm25"].agg(
    ["count", "mean", "median", "std"]
)

# Sort stations by mean PM2.5
station_pm25 = station_pm25.sort_values(
    by="mean",
    ascending=False
)

print("Station Statistics:")
print(station_pm25)

# Select stations with at least 100 observations
reliable_stations = station_pm25[
    station_pm25["count"] >= 100
]

print("\nNumber of stations with at least 100 observations:",
      len(reliable_stations))
#Top 10 Stations by Mean PM2.5

top_stations = reliable_stations.head(10).sort_values(
    by="mean",
    ascending=True
)

plt.figure(figsize=(10, 6))

plt.barh(
    top_stations.index,
    top_stations["mean"]
)

plt.title("Top 10 Stations with Highest Mean PM2.5")
plt.xlabel("Mean PM2.5 (µg/m³)")
plt.ylabel("Monitoring Station")

plt.tight_layout()
plt.show()


#%% Q8.4 - Influence of meteorology on PM2.5-AOD associations

import numpy as np
import pandas as pd

# 1. Calculate wind speed

df_clean["wind_speed"] = np.sqrt(
    df_clean["wind_u"]**2 + df_clean["wind_v"]**2
)

print("Wind speed statistics:")
print(df_clean["wind_speed"].describe())


# 2. Divide wind speed into three groups

df_clean["wind_group"] = pd.qcut(
    df_clean["wind_speed"],
    q=3,
    labels=["Low", "Medium", "High"]
)


#3. PM2.5-AOD correlations by wind speed

print("\nPM2.5-AOD correlation by wind speed:")

for level in ["Low", "Medium", "High"]:

    group = df_clean[df_clean["wind_group"] == level]

    data = group[["pm25", "AOD"]].dropna()

    pearson = data["pm25"].corr(method="pearson", other=data["AOD"])
    spearman = data["pm25"].corr(method="spearman", other=data["AOD"])

    print(f"\n{level} wind:")
    print("Observations:", len(data))
    print("Pearson:", pearson)
    print("Spearman:", spearman)


# 4. Divide temperature into three groups

df_clean["temperature_group"] = pd.qcut(
    df_clean["temperature_celsius"],
    q=3,
    labels=["Low", "Medium", "High"]
)


# 5. PM2.5-AOD correlations by temperature

print("\nPM2.5-AOD correlation by temperature:")

for level in ["Low", "Medium", "High"]:

    group = df_clean[df_clean["temperature_group"] == level]

    data = group[["pm25", "AOD"]].dropna()

    pearson = data["pm25"].corr(method="pearson", other=data["AOD"])
    spearman = data["pm25"].corr(method="spearman", other=data["AOD"])

    print(f"\n{level} temperature:")
    print("Observations:", len(data))
    print("Pearson:", pearson)
    print("Spearman:", spearman)


#% Calculate wind speed
# Check cleaned PM2.5 data

print("Shape:", df_clean.shape)
print("Minimum:", df_clean["pm25"].min())
print("Maximum:", df_clean["pm25"].max())
print("Negative or zero:", (df_clean["pm25"] <= 0).sum())
print("Values >= 500:", (df_clean["pm25"] >= 500).sum())

#%% Q9 - PM2.5 Target Distribution

import matplotlib.pyplot as plt
import seaborn as sns

print("PM2.5 descriptive statistics:")
print(df_clean["pm25"].describe())

print("\nSkewness:", df_clean["pm25"].skew())
print("Kurtosis:", df_clean["pm25"].kurtosis())

# Histogram
plt.figure(figsize=(9, 5))
sns.histplot(data=df_clean, x="pm25", bins=50, kde=True)

plt.title("Distribution of PM2.5 Concentrations")
plt.xlabel("PM2.5 (µg/m³)")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

#% Q9.2 - Seasonal Variation of PM2.5

# Extract month from date
df_clean["date"] = pd.to_datetime(df_clean["date"])

# Create season column
def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Autumn"

df_clean["season"] = df_clean["date"].dt.month.apply(get_season)

# Calculate seasonal statistics
seasonal_stats = df_clean.groupby("season")["pm25"].agg(
    ["count", "mean", "median", "std"]
)

# Display results
print("Seasonal PM2.5 Statistics:")
print(seasonal_stats)
#% Q9.2 - Seasonal PM2.5 Mean Plot

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 5))

sns.barplot(
    data=df_clean,
    x="season",
    y="pm25",
    order=["Winter", "Spring", "Summer", "Autumn"],
    estimator="mean",
    errorbar=None
)

plt.title("Average PM2.5 Concentration by Season")
plt.xlabel("Season")
plt.ylabel("Mean PM2.5 (µg/m³)")

plt.show()
#%Q9.4 - Proportion of High PM2.5 Observations

# Total number of observations
total = len(df_clean)

# Count observations above 25 and 50
above_25 = (df_clean["pm25"] > 25).sum()
above_50 = (df_clean["pm25"] > 50).sum()

# Calculate percentages
percentage_25 = (above_25 / total) * 100
percentage_50 = (above_50 / total) * 100

# Display results
print("Total observations:", total)

print("\nAbove 25 µg/m3:", above_25)
print("Percentage:", percentage_25, "%")

print("\nAbove 50 µg/m3:", above_50)
print("Percentage:", percentage_50, "%")
#%% 10 - Export Exploratory_Data.csv

#%% 5.1 - Create Exploratory_Data.csv

#%% 10 - Generate Exploratory_Data.csv

#%% STEP 10 - Create Exploratory_Data.csv

import pandas as pd
import numpy as np

from scipy.stats import zscore, pearsonr, spearmanr
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.feature_selection import mutual_info_regression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import acf, pacf, adfuller

# Use the cleaned dataset
data = df_clean.copy()

# --------------------------------------------------
# 1. Define variables and metadata
# --------------------------------------------------

variables = [
    "date", "datetime_utc", "id", "latitude", "longitude",
    "temperature_celsius", "pressure_mb", "wind_u", "wind_v",
    "NO2", "CO", "O3", "AOD", "pm25", "sensor_name"
]

predictors = [
    "AOD", "NO2", "CO", "O3",
    "temperature_celsius", "pressure_mb", "wind_u", "wind_v"
]

families = {
    "date": "Calendar",
    "datetime_utc": "Calendar",
    "id": "ID",
    "latitude": "Spatial",
    "longitude": "Spatial",
    "temperature_celsius": "Meteorological",
    "pressure_mb": "Meteorological",
    "wind_u": "Meteorological",
    "wind_v": "Meteorological",
    "NO2": "NO2",
    "CO": "CO",
    "O3": "O3",
    "AOD": "AOD",
    "pm25": "Target",
    "sensor_name": "ID"
}

units = {
    "date": "",
    "datetime_utc": "",
    "id": "",
    "latitude": "degrees",
    "longitude": "degrees",
    "temperature_celsius": "°C",
    "pressure_mb": "mb",
    "wind_u": "m/s",
    "wind_v": "m/s",
    "NO2": "μg/m3",
    "CO": "μg/m3",
    "O3": "μg/m3",
    "AOD": "dimensionless",
    "pm25": "µg/m³",
    "sensor_name": ""
}

# Numeric variables for statistical analysis
numeric_variables = [
    "latitude", "longitude", "temperature_celsius",
    "pressure_mb", "wind_u", "wind_v",
    "NO2", "CO", "O3", "AOD", "pm25"
]

# Environmental variables used for multivariate detection
environmental_variables = predictors.copy()

# --------------------------------------------------
# 2. Prepare multivariate environmental data
# --------------------------------------------------
# 2. Multivariate Isolation Forest and LOF

X_environment = data[environmental_variables].copy()

# Handle missing values
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X_environment)

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Isolation Forest
if_model = IsolationForest(
    contamination="auto",
    random_state=42
)

if_labels = if_model.fit_predict(X_scaled)

if_count = int((if_labels == -1).sum())
if_percentage = if_count / len(data) * 100

# Local Outlier Factor
lof_model = LocalOutlierFactor(
    n_neighbors=min(20, len(data) - 1),
    contamination="auto"
)

lof_labels = lof_model.fit_predict(X_scaled)

lof_count = int((lof_labels == -1).sum())
lof_percentage = lof_count / len(data) * 100

print("Multivariate Isolation Forest outliers:", if_count)
print("Multivariate Isolation Forest percentage:",
      round(if_percentage, 3), "%")

print("Multivariate LOF outliers:", lof_count)
print("Multivariate LOF percentage:",
      round(lof_percentage, 3), "%")

# --------------------------------------------------
# 3. Calculate VIF for the eight predictors only
# --------------------------------------------------

vif_data = data[environmental_variables].copy()

# Keep complete observations
vif_data = vif_data.dropna()

# Add intercept
vif_data = sm.add_constant(vif_data, has_constant="add")

vif_results = {}

for variable in environmental_variables:

    column_index = vif_data.columns.get_loc(variable)

    vif_value = variance_inflation_factor(
        vif_data.values,
        column_index
    )

    vif_results[variable] = vif_value

print("\nVIF Results:")
print("Number of observations used:", len(vif_data))

for variable, value in vif_results.items():
    print(variable, round(value, 6))
 

# --------------------------------------------------
# 4. Correlation with PM2.5
# --------------------------------------------------

target_associations = {}
target_pvalues = {}

for variable in predictors:

    pair = data[[variable, "pm25"]].dropna()

    if len(pair) > 2:
        pearson_r, pearson_p = pearsonr(
            pair[variable], pair["pm25"]
        )

        spearman_r, _ = spearmanr(
            pair[variable], pair["pm25"]
        )

        mi = mutual_info_regression(
            pair[[variable]],
            pair["pm25"],
            random_state=42
        )[0]

        target_associations[variable] = (
            f"pearson_r={pearson_r:.4f}; "
            f"spearman={spearman_r:.4f}; "
            f"mi={mi:.4f}"
        )

        target_pvalues[variable] = pearson_p

# Predictor-to-predictor correlations
corr_matrix = data[predictors].corr(method="pearson")

max_correlations = {}
correlation_partners = {}

for variable in predictors:

    correlations = corr_matrix[variable].drop(variable).dropna()

    if len(correlations) > 0:
        partner = correlations.abs().idxmax()

        max_correlations[variable] = abs(correlations[partner])
        correlation_partners[variable] = partner

# --------------------------------------------------
# 5. Temporal diagnostics
# Daily mean across all stations
# --------------------------------------------------

data["date_parsed"] = pd.to_datetime(
    data["date"], errors="coerce"
)

temporal_results = {}

for variable in numeric_variables:

    daily = (
        data.groupby("date_parsed")[variable]
        .mean()
        .sort_index()
    )

    daily = daily.dropna()

    if len(daily) < 30:
        continue

    # Create a regular daily timeline
    full_dates = pd.date_range(
        start=daily.index.min(),
        end=daily.index.max(),
        freq="D"
    )

    daily = daily.reindex(full_dates)

    # Interpolation is used only for temporal diagnostics
    daily = daily.interpolate(method="linear")

    if daily.isna().any():
        daily = daily.fillna(method="bfill").fillna(method="ffill")

    if len(daily) < 30:
        continue

    try:
        acf_values = acf(
            daily,
            nlags=min(24, len(daily) - 1),
            fft=True
        )

        pacf_values = pacf(
            daily,
            nlags=min(24, len(daily) // 2 - 1),
            method="ywm"
        )

        acf_lag1 = acf_values[1]
        acf_lag24 = (
            acf_values[24] if len(acf_values) > 24 else np.nan
        )

        pacf_lag1 = pacf_values[1]

        # First lag where absolute ACF falls below 0.2
        recommended_lags = np.nan

        for lag in range(1, len(acf_values)):
            if abs(acf_values[lag]) < 0.2:
                recommended_lags = lag
                break

        adf_p = adfuller(daily, autolag="AIC")[1]

        temporal_results[variable] = {
            "acf_lag1": acf_lag1,
            "acf_lag24": acf_lag24,
            "pacf_lag1": pacf_lag1,
            "n_lags_recommended": recommended_lags,
            "adf_p": adf_p
        }

    except Exception as error:
        print("Temporal diagnostic error for", variable, ":", error)


# --------------------------------------------------
# 6. Build the final 15-row table
# --------------------------------------------------
#%6. Build the final 15-row table

results = []

for variable in variables:

    row = {
        "model_id": "MLP_pm25",
        "model_group": "G8",
        "variable": variable,
        "family": families[variable],
        "type": (
            "datetime" if variable in ["date", "datetime_utc"]
            else "id" if variable in ["id", "sensor_name"]
            else "numeric"
        ),
        "unit": units[variable],
        "n_total": len(data),
        "n_missing": data[variable].isna().sum(),
        "pct_missing": data[variable].isna().mean() * 100,
        "n_unique": data[variable].nunique(),
        "n_duplicates_key": data.duplicated(
            subset=["date", "latitude", "longitude", "sensor_name"]
        ).sum(),
        "min": np.nan,
        "q1": np.nan,
        "median": np.nan,
        "q3": np.nan,
        "max": np.nan,
        "mean": np.nan,
        "std": np.nan,
        "skewness": np.nan,
        "kurtosis": np.nan,
        "iqr": np.nan,
        "n_out_iqr": np.nan,
        "pct_out_iqr": np.nan,
        "n_out_z": np.nan,
        "pct_out_z": np.nan,
        "n_out_if": np.nan,
        "pct_out_if": np.nan,
        "n_out_lof": np.nan,
        "pct_out_lof": np.nan,
        "n_sentinel_low": 0,
        "n_sentinel_high": 0,
        "n_removed_total": 0,
        "max_abs_corr": np.nan,
        "corr_partner": "",
        "vif": np.nan,
        "modalities": "",
        "target_assoc": "",
        "target_pval": np.nan,
        "acf_lag1": np.nan,
        "acf_lag24": np.nan,
        "pacf_lag1": np.nan,
        "n_lags_recommended": np.nan,
        "adf_p": np.nan,
        "is_leakage_suspect": False,
        "notes": []
    }

    # Descriptive statistics
    if variable in numeric_variables:

        series = data[variable].dropna()

        if len(series) > 0:

            row["min"] = series.min()
            row["q1"] = series.quantile(0.25)
            row["median"] = series.median()
            row["q3"] = series.quantile(0.75)
            row["max"] = series.max()
            row["mean"] = series.mean()
            row["std"] = series.std()
            row["skewness"] = series.skew()
            row["kurtosis"] = series.kurtosis()

            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr_value = q3 - q1

            row["iqr"] = iqr_value

            if variable not in ["latitude", "longitude"]:

                lower = q1 - 1.5 * iqr_value
                upper = q3 + 1.5 * iqr_value

                iqr_count = (
                    (series < lower) | (series > upper)
                ).sum()

                z_values = np.abs(
                    zscore(series, nan_policy="omit")
                )

                z_count = (z_values > 3).sum()

                row["n_out_iqr"] = int(iqr_count)
                row["pct_out_iqr"] = (
                    iqr_count / len(series) * 100
                )

                row["n_out_z"] = int(z_count)
                row["pct_out_z"] = (
                    z_count / len(series) * 100
                )

    # Categorical modalities
    if variable in ["id", "sensor_name"]:
        row["modalities"] = data[variable].nunique()

    # Correlation, VIF and Multivariate IF/LOF
    if variable in predictors:

        row["max_abs_corr"] = max_correlations.get(
            variable, np.nan
        )

        row["corr_partner"] = correlation_partners.get(
            variable, ""
        )

        row["vif"] = vif_results.get(variable, np.nan)

        row["target_assoc"] = target_associations.get(
            variable, ""
        )

        row["target_pval"] = target_pvalues.get(
            variable, np.nan
        )

        # Same multivariate result for each predictor
        row["n_out_if"] = if_count
        row["pct_out_if"] = if_percentage

        row["n_out_lof"] = lof_count
        row["pct_out_lof"] = lof_percentage

        row["notes"].append(
            "Multivariate IF and LOF applied to all environmental variables."
        )

    # Temporal diagnostics
    if variable in temporal_results:

        temporal = temporal_results[variable]

        row["acf_lag1"] = temporal["acf_lag1"]
        row["acf_lag24"] = temporal["acf_lag24"]
        row["pacf_lag1"] = temporal["pacf_lag1"]

        row["n_lags_recommended"] = (
            temporal["n_lags_recommended"]
        )

        row["adf_p"] = temporal["adf_p"]

        row["notes"].append(
            "Temporal diagnostics use daily mean across stations; "
            "missing dates interpolated for diagnostics only."
        )

    # PM2.5 invalid values removed before this analysis
    if variable == "pm25":

        row["n_sentinel_low"] = 727
        row["n_sentinel_high"] = 8
        row["n_removed_total"] = 735

        row["notes"].append(
            "735 invalid PM2.5 observations removed before final EDA."
        )

    # IDs and station names
    if variable in ["id", "sensor_name"]:

        row["is_leakage_suspect"] = False

        row["notes"].append(
            "Identifier: consider station-based grouping "
            "during spatial validation."
        )

    # Minimum for date and datetime variables
    if variable in ["date", "datetime_utc"]:

        row["min"] = pd.to_datetime(
            data[variable], errors="coerce"
        ).min()

    # Convert notes list to text
    row["notes"] = " ".join(row["notes"])

    # Append row
    results.append(row)


# --------------------------------------------------
# 7. Create and save CSV
# --------------------------------------------------

output_columns = [
    "model_id", "model_group", "variable", "family", "type", "unit",
    "n_total", "n_missing", "pct_missing", "n_unique",
    "n_duplicates_key", "min", "q1", "median", "q3", "max",
    "mean", "std", "skewness", "kurtosis", "iqr",
    "n_out_iqr", "pct_out_iqr", "n_out_z", "pct_out_z",
    "n_out_if", "pct_out_if", "n_out_lof", "pct_out_lof",
    "n_sentinel_low", "n_sentinel_high", "n_removed_total",
    "max_abs_corr", "corr_partner", "vif", "modalities",
    "target_assoc", "target_pval", "acf_lag1", "acf_lag24",
    "pacf_lag1", "n_lags_recommended", "adf_p",
    "is_leakage_suspect", "notes"
]

exploratory_data = pd.DataFrame(results, columns=output_columns)

exploratory_data.to_csv(
    "Exploratory_Data.csv",
    index=False
)

print("\nExploratory_Data.csv saved successfully.")
print("Shape:", exploratory_data.shape)
print("\nColumns:", len(exploratory_data.columns))
print("\nMissing values:")
print(exploratory_data.isnull().sum())
print("\nFinal results:")
print(exploratory_data.to_string(index=False))
print(df_clean["date"].min())
print(df_clean["datetime_utc"].min())

# Recalculate VIF

import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

environmental_variables = [
    "AOD", "NO2", "CO", "O3",
    "temperature_celsius", "pressure_mb",
    "wind_u", "wind_v"
]

vif_data = df_clean[environmental_variables].copy()

# Keep complete observations
vif_data = vif_data.dropna()

# Add intercept
vif_data = sm.add_constant(vif_data, has_constant="add")

# Calculate VIF
vif_results = []

for variable in environmental_variables:
    column_index = vif_data.columns.get_loc(variable)

    vif_value = variance_inflation_factor(
        vif_data.values,
        column_index
    )

    vif_results.append({
        "Variable": variable,
        "VIF": vif_value
    })

vif_table = pd.DataFrame(vif_results)

print("Number of observations used:", len(vif_data))
print(vif_table)




#%% Save cleaned dataset
#%% Save cleaned dataset for next steps

original_columns = df.columns.tolist()

df_clean[original_columns].to_csv("cleaned_pm25.csv", index=False)

print("Cleaned dataset saved successfully!")

print(pd.read_csv("../Data/cleaned_pm25.csv").shape)
print(df.shape)
