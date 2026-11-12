
#%% 5.2.1 Timestamp regularity and temporal coverage

import pandas as pd

# ---------- 1. Load and validate timestamps ----------


df = pd.read_csv("Data/cleaned_pm25.csv")

df["date"] = pd.to_datetime(df["date"], errors="coerce")

print("Invalid dates:", df["date"].isna().sum())
print("Missing station IDs:", df["id"].isna().sum())

df = df.dropna(subset=["date", "id"]).copy()

df["date"] = df["date"].dt.normalize()


# ---------- 2. Duplicate timestamps ----------

n_duplicates = df.duplicated(
    subset=["id", "date"]
).sum()

print("\nTotal duplicate timestamps:", n_duplicates)

dup_per_station = (
    df[df.duplicated(subset=["id", "date"], keep="first")]
    .groupby("id")
    .size()
)

# Remove duplicate station-date records
df = df.drop_duplicates(
    subset=["id", "date"]
).sort_values(["id", "date"])


# ---------- 3. Calculate time intervals per station ----------

df["interval_days"] = (
    df.groupby("id")["date"].diff().dt.days
)


# ---------- 4. Per-station analysis ----------

rows = []

for sid, g in df.groupby("id"):

    dates = g["date"].sort_values()

    # Time intervals
    step = dates.diff().dt.days.dropna()

    # Timestamp regularity
    median_interval = step.median() if len(step) else None
    mean_interval = round(step.mean(), 2) if len(step) else None
    std_interval = round(step.std(), 2) if len(step) > 1 else None
    min_interval = step.min() if len(step) else None
    max_interval = step.max() if len(step) else None

    # Daily sampling
    n_daily = (step == 1).sum()
    n_total = len(step)
    n_non_daily = n_total - n_daily

    daily_sampling_pct = (
        round(n_daily / n_total * 100, 2)
        if n_total else None
    )

    # Missing dates
    calendar = pd.date_range(
        start=dates.min(),
        end=dates.max(),
        freq="D"
    )

    observed_dates = pd.DatetimeIndex(dates.unique())

    missing_dates = calendar.difference(observed_dates)

    n_missing = len(missing_dates)
    n_expected = len(calendar)

    pct_missing = round(
        n_missing / n_expected * 100, 2
    )

    # Duplicate timestamps
    n_station_duplicates = int(
        dup_per_station.get(sid, 0)
    )

    # Irregular observations
    n_irregular_gaps = (step > 1).sum()

    longest_gap_days = (
        int(step.max() - 1) if len(step) else 0
    )

    rows.append({

        "id": sid,

        "n_records": len(dates),

        "first_date": dates.min().date(),
        "last_date": dates.max().date(),

        "median_interval_days": median_interval,
        "mean_interval_days": mean_interval,
        "std_interval_days": std_interval,
        "min_interval_days": min_interval,
        "max_interval_days": max_interval,

        "daily_intervals": n_daily,
        "non_daily_intervals": n_non_daily,
        "total_intervals": n_total,
        "daily_sampling_pct": daily_sampling_pct,

        "expected_dates": n_expected,
        "observed_dates": len(observed_dates),
        "n_missing_days": n_missing,
        "pct_missing_days": pct_missing,

        "n_duplicates_removed": n_station_duplicates,

        "n_irregular_gaps": n_irregular_gaps,
        "longest_gap_days": longest_gap_days
    })


# ---------- 5. Create per-station results ----------

stations = pd.DataFrame(rows)

stations = stations.sort_values(
    "n_records",
    ascending=False
)


# ---------- 6. Display duplicate timestamps ----------

print("\n--- DUPLICATE TIMESTAMPS PER STATION ---")

print("Total duplicates:", n_duplicates)

print("Duplicates per station:")

print(dup_per_station.to_string())


# ---------- 7. Display timestamp regularity and daily sampling ----------

print("\n--- TIMESTAMP REGULARITY AND DAILY SAMPLING PER STATION ---")

print(
    stations[
        [
            "id",
            "n_records",
            "median_interval_days",
            "mean_interval_days",
            "std_interval_days",
            "min_interval_days",
            "max_interval_days",
            "daily_intervals",
            "non_daily_intervals",
            "daily_sampling_pct"
        ]
    ].head(10).to_string(index=False)
)


# ---------- 8. Display missing dates per station ----------

print("\n--- MISSING DATES PER STATION ---")

print(
    stations[
        [
            "id",
            "expected_dates",
            "observed_dates",
            "n_missing_days",
            "pct_missing_days"
        ]
    ].head(10).to_string(index=False)
)


# ---------- 9. Display irregular observations per station ----------

print("\n--- IRREGULAR OBSERVATIONS PER STATION ---")

print(
    stations[
        [
            "id",
            "n_irregular_gaps",
            "longest_gap_days"
        ]
    ].head(10).to_string(index=False)
)


# ---------- 10. Summary across stations ----------

print("\n--- SUMMARY ACROSS STATIONS ---")

print("Number of stations:", len(stations))

print(
    "Median daily sampling percentage:",
    round(stations["daily_sampling_pct"].median(), 2), "%"
)

print(
    "Median missing days percentage:",
    round(stations["pct_missing_days"].median(), 2), "%"
)

print(
    "Stations with no missing days:",
    (stations["n_missing_days"] == 0).sum()
)

print(
    "Stations with irregular gaps:",
    (stations["n_irregular_gaps"] > 0).sum()
)

print(
    "Stations with duplicate timestamps:",
    (stations["n_duplicates_removed"] > 0).sum()
)

print(
    "Longest gap anywhere:",
    stations["longest_gap_days"].max(),
    "missing days"
)
print("station that complete temporal coverage within its own observation period:")
print(
    stations.loc[
        stations["n_missing_days"] == 0,
        ["id", "first_date", "last_date", "n_records"]
    ].to_string(index=False)
)

# ---------- 11. Save per-station results ----------

stations.to_csv(
    "timestamp_regularity_per_station.csv",
    index=False
)

print("\nTimestamp analysis saved successfully.")

#%% 5.2.2 Select a continuous daily time series

#%% 5.2.2 Autocorrelation and Stationarity

import matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller


# ---------- 1. Load and prepare data ----------

df["date"] = pd.to_datetime(
    df["date"], errors="coerce"
).dt.normalize()

df["pm25"] = pd.to_numeric(
    df["pm25"], errors="coerce"
)

df = df.dropna(
    subset=["date", "id", "pm25"]
).copy()

df = df.drop_duplicates(
    subset=["id", "date"]
)

df = df.sort_values(
    ["id", "date"]
).reset_index(drop=True)


# ---------- 2. Identify continuous daily segments ----------

df["interval"] = (
    df.groupby("id")["date"].diff().dt.days
)

df["new_segment"] = (
    df["interval"].ne(1)
).astype(int)

df["segment"] = (
    df.groupby("id")["new_segment"].cumsum()
)


# ---------- 3. Find the longest segment ----------

segments = (
    df.groupby(["id", "segment"])
    .agg(
        start_date=("date", "min"),
        end_date=("date", "max"),
        n_observations=("pm25", "count")
    )
    .reset_index()
)

segments = segments.sort_values(
    "n_observations",
    ascending=False
).reset_index(drop=True)

print("\n--- LONGEST CONTINUOUS DAILY SEGMENTS ---")
print(segments.head(10).to_string(index=False))


# ---------- 4. Select the longest time series ----------

best_segment = segments.iloc[0]

station_id = best_segment["id"]
segment_id = best_segment["segment"]

selected = df[
    (df["id"] == station_id) &
    (df["segment"] == segment_id)
].copy()

selected = selected.sort_values("date")

series = selected.set_index("date")["pm25"]

print("\n--- SELECTED TIME SERIES ---")

print("Station ID:", station_id)
print("Start date:", series.index.min())
print("End date:", series.index.max())
print("Number of observations:", len(series))


# ---------- 5. Check series length ----------

if len(series) < 20:

    print(
        "\nWARNING: The selected series contains fewer than "
        "20 observations. Statistical results may be unreliable."
    )

if len(series) < 5:

    raise ValueError(
        "Not enough observations for ACF/PACF analysis."
    )


# ---------- 6. ACF and PACF plots ----------

max_lags = min(30, (len(series) - 1) // 2)

fig, axes = plt.subplots(2, 1, figsize=(10, 8))

plot_acf(
    series,
    lags=max_lags,
    ax=axes[0]
)

axes[0].set_title(
    f"ACF - Station {station_id}"
)

plot_pacf(
    series,
    lags=max_lags,
    method="ywm",
    ax=axes[1]
)

axes[1].set_title(
    f"PACF - Station {station_id}"
)

plt.tight_layout()
plt.show()


# ---------- 7. Ljung-Box test ----------

lb_lag = min(10, max_lags)

ljung_box = acorr_ljungbox(
    series,
    lags=[lb_lag],
    return_df=True
)

print("\n--- LJUNG-BOX TEST ---")
print(ljung_box)

lb_pvalue = ljung_box["lb_pvalue"].iloc[0]

if lb_pvalue < 0.05:

    print(
        "Result: Significant autocorrelation is present."
    )

else:

    print(
        "Result: No significant autocorrelation detected."
    )


# ---------- 8. ADF stationarity test ----------

adf_result = adfuller(
    series,
    autolag="AIC"
)

print("\n--- ADF TEST ON ORIGINAL SERIES ---")

print("ADF Statistic:", adf_result[0])
print("p-value:", adf_result[1])
print("Number of lags:", adf_result[2])
print("Number of observations:", adf_result[3])

if adf_result[1] < 0.05:

    print(
        "Result: The series is stationary."
    )

else:

    print(
        "Result: The series is non-stationary."
    )


# ---------- 9. First differencing if necessary ----------

if adf_result[1] >= 0.05:

    print("\n--- ADF TEST AFTER FIRST DIFFERENCING ---")

    diff_series = series.diff().dropna()

    adf_diff = adfuller(
        diff_series,
        autolag="AIC"
    )

    print("ADF Statistic:", adf_diff[0])
    print("p-value:", adf_diff[1])
    print("Number of lags:", adf_diff[2])
    print("Number of observations:", adf_diff[3])

    if adf_diff[1] < 0.05:

        print(
            "Result: The differenced series is stationary."
        )

    else:

        print(
            "Result: The differenced series is still non-stationary."
        )

    # Plot the differenced series

    plt.figure(figsize=(10, 4))

    plt.plot(diff_series)

    plt.title(
        f"First-Differenced PM2.5 - Station {station_id}"
    )

    plt.xlabel("Date")
    plt.ylabel("Change in PM2.5")

    plt.tight_layout()
    plt.show()


#%Plot PM2.5 Time Series - Station 30


# Select Station 30
station = df[df["id"] == 30].copy()

# Keep the selected continuous period
station = station[
    (station["date"] >= "2025-07-01") &
    (station["date"] <= "2025-08-17")
]

station = station.sort_values("date")

# Plot
plt.figure(figsize=(12, 5))

plt.plot(
    station["date"],
    station["pm25"],
    marker="o",
    markersize=4,
    linewidth=1.5
)

plt.title("Daily PM2.5 Time Series - Station 30")
plt.xlabel("Date")
plt.ylabel("PM2.5")
plt.grid(True, alpha=0.3)

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ---------- 10. Save selected time series ----------

selected.to_csv(
    "selected_continuous_pm25_series.csv",
    index=False
)

print("\nSelected time series saved successfully.")

#%% 5.2.3 Target Drift Across Months and Years


# Convert date
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Remove missing target and dates
df = df.dropna(subset=["date", "pm25"]).copy()

# Define high PM2.5 threshold
# Example threshold: 35 µg/m³
HIGH_THRESHOLD = 35

# Extract year and month
df["year"] = df["date"].dt.year
df["year_month"] = df["date"].dt.to_period("M")

#%Monthly analysis

monthly = df.groupby("year_month").agg(
    median_pm25=("pm25", "median"),
    Q1=("pm25", lambda x: x.quantile(0.25)),
    Q3=("pm25", lambda x: x.quantile(0.75)),
    total_observations=("pm25", "count"),
    high_count=("pm25", lambda x: (x >= HIGH_THRESHOLD).sum())
)

monthly["IQR"] = monthly["Q3"] - monthly["Q1"]

monthly["high_frequency_pct"] = (
    monthly["high_count"] / monthly["total_observations"] * 100
)

print("MONTHLY TARGET DRIFT")
print(monthly.round(2))

#% Yearly analysis

yearly = df.groupby("year").agg(
    median_pm25=("pm25", "median"),
    Q1=("pm25", lambda x: x.quantile(0.25)),
    Q3=("pm25", lambda x: x.quantile(0.75)),
    total_observations=("pm25", "count"),
    high_count=("pm25", lambda x: (x >= HIGH_THRESHOLD).sum())
)

yearly["IQR"] = yearly["Q3"] - yearly["Q1"]

yearly["high_frequency_pct"] = (
    yearly["high_count"] / yearly["total_observations"] * 100
)

print("\nYEARLY TARGET DRIFT")
print(yearly.round(2))

#% Monthly plots

monthly.index = monthly.index.astype(str)

fig, axes = plt.subplots(3, 1, figsize=(13, 10))

monthly["median_pm25"].plot(ax=axes[0], marker="o")
axes[0].set_title("Monthly Median PM2.5")
axes[0].set_ylabel("Median PM2.5")
axes[0].grid(True, alpha=0.3)

monthly["IQR"].plot(ax=axes[1], marker="o")
axes[1].set_title("Monthly PM2.5 IQR")
axes[1].set_ylabel("IQR")
axes[1].grid(True, alpha=0.3)

monthly["high_frequency_pct"].plot(ax=axes[2], marker="o")
axes[2].set_title("Monthly High-Concentration Frequency")
axes[2].set_ylabel("Frequency (%)")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

#%Yearly plots

fig, axes = plt.subplots(3, 1, figsize=(10, 9))

yearly["median_pm25"].plot(ax=axes[0], marker="o")
axes[0].set_title("Yearly Median PM2.5")
axes[0].set_ylabel("Median PM2.5")
axes[0].grid(True, alpha=0.3)

yearly["IQR"].plot(ax=axes[1], marker="o")
axes[1].set_title("Yearly PM2.5 IQR")
axes[1].set_ylabel("IQR")
axes[1].grid(True, alpha=0.3)

yearly["high_frequency_pct"].plot(ax=axes[2], marker="o")
axes[2].set_title("Yearly High-Concentration Frequency")
axes[2].set_ylabel("Frequency (%)")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

#%Save results

monthly.to_csv("pm25_monthly_drift.csv")
yearly.to_csv("pm25_yearly_drift.csv")


#%% 5.2.4 Daily, Monthly and Seasonal Patterns

import pandas as pd
import matplotlib.pyplot as plt

#%% Load dataset


# Convert date
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Remove missing dates
df = df.dropna(subset=["date"]).copy()

# Define variables
variables = [
    "pm25",
    "temperature_celsius",
    "pressure_mb",
    "wind_u",
    "wind_v",
    "NO2",
    "CO",
    "O3",
    "AOD"
]

# Keep available columns
variables = [col for col in variables if col in df.columns]

# Extract time features
df["month"] = df["date"].dt.month

df["season"] = df["date"].dt.month.map({
    12: "Winter",
    1: "Winter",
    2: "Winter",
    3: "Spring",
    4: "Spring",
    5: "Spring",
    6: "Summer",
    7: "Summer",
    8: "Summer",
    9: "Autumn",
    10: "Autumn",
    11: "Autumn"
})

# Define quartile functions
def calculate_q1(x):
    return x.quantile(0.25)

def calculate_q3(x):
    return x.quantile(0.75)


#%% Daily analysis

daily = df.groupby("date").agg({
    col: [
        "median",
        calculate_q1,
        calculate_q3,
        "count"
    ]
    for col in variables
})

# Calculate IQR
for col in variables:
    daily[(col, "IQR")] = (
        daily[(col, "calculate_q3")] -
        daily[(col, "calculate_q1")]
    )

print("DAILY PATTERNS")
print(daily)

daily.to_csv("daily_patterns.csv")


#%% Monthly analysis

monthly = df.groupby("month").agg({
    col: [
        "median",
        calculate_q1,
        calculate_q3,
        "count"
    ]
    for col in variables
})

# Calculate IQR
for col in variables:
    monthly[(col, "IQR")] = (
        monthly[(col, "calculate_q3")] -
        monthly[(col, "calculate_q1")]
    )

print("\nMONTHLY PATTERNS")
print(monthly)

monthly.to_csv("monthly_patterns.csv")


#%% Seasonal analysis

season_order = ["Winter", "Spring", "Summer", "Autumn"]

seasonal = df.groupby("season").agg({
    col: [
        "median",
        calculate_q1,
        calculate_q3,
        "count"
    ]
    for col in variables
})

seasonal = seasonal.reindex(season_order)

# Calculate IQR
for col in variables:
    seasonal[(col, "IQR")] = (
        seasonal[(col, "calculate_q3")] -
        seasonal[(col, "calculate_q1")]
    )

print("\nSEASONAL PATTERNS")
print(seasonal)
#%% Display all predictor results

for col in variables:

    print("\n==========", col, "==========")

    print("MONTHLY:")
    print(
        monthly[col][["median", "calculate_q1",
                      "calculate_q3", "IQR"]].round(8)
    )

    print("\nSEASONAL:")
    print(
        seasonal[col][["median", "calculate_q1",
                       "calculate_q3", "IQR"]].round(8)
    )


seasonal.to_csv("seasonal_patterns.csv")


#%% Daily plots

fig, axes = plt.subplots(3, 3, figsize=(16, 11))

for i, col in enumerate(variables):

    ax = axes[i // 3, i % 3]

    daily[(col, "median")].plot(ax=ax)

    ax.set_title("Daily Median - " + col)
    ax.set_xlabel("Date")
    ax.set_ylabel(col)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


#%% Monthly plots

fig, axes = plt.subplots(3, 3, figsize=(16, 11))

for i, col in enumerate(variables):

    ax = axes[i // 3, i % 3]

    monthly[(col, "median")].plot(
        ax=ax,
        marker="o"
    )

    ax.set_title("Monthly Median - " + col)
    ax.set_xlabel("Month")
    ax.set_ylabel(col)
    ax.set_xticks(range(1, 13))
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


#%% Seasonal plots

fig, axes = plt.subplots(3, 3, figsize=(16, 11))

for i, col in enumerate(variables):

    ax = axes[i // 3, i % 3]

    seasonal[(col, "median")].plot(
        kind="bar",
        ax=ax
    )

    ax.set_title("Seasonal Median - " + col)
    ax.set_xlabel("Season")
    ax.set_ylabel(col)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


#%% Save results

print("\nDaily, monthly and seasonal analysis completed.")




#%% 5.2.5 Lead-Lag Relationships

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#% 1. Load dataset
df = pd.read_csv("Data/cleaned_pm25.csv")


#% 2. Prepare dates

df["date"] = pd.to_datetime(df["date"], errors="coerce")

df = df.dropna(subset=["date", "pm25"]).copy()

#% 3. Calculate wind speed

df["wind_speed"] = np.sqrt(
    df["wind_u"]**2 + df["wind_v"]**2
)

#% 4. Select variables

variables = [
    "AOD",
    "NO2",
    "CO",
    "O3",
    "wind_speed",
    "wind_u",
    "wind_v",
    "temperature_celsius"
]

#% 5. Calculate daily median

daily = df.groupby("date")[["pm25"] + variables].median()

#% 6. Include missing calendar dates

all_dates = pd.date_range(
    start=daily.index.min(),
    end=daily.index.max(),
    freq="D"
)

daily = daily.reindex(all_dates)

daily.index.name = "date"

#% 7. Calculate lead-lag correlations

max_lag = 7

results = []

for variable in variables:

    for lag in range(0, max_lag + 1):

        # Shift the variable backwards in time
        shifted_variable = daily[variable].shift(lag)

        # Match PM2.5 with the shifted variable
        paired = pd.concat(
            [daily["pm25"], shifted_variable],
            axis=1
        ).dropna()

        # Calculate Pearson correlation
        correlation = paired.iloc[:, 0].corr(
            paired.iloc[:, 1]
        )

        results.append({
            "variable": variable,
            "lag": lag,
            "correlation": correlation,
            "number_of_pairs": len(paired)
        })

#% 8. Display results

lag_results = pd.DataFrame(results)

print("LEAD-LAG CORRELATIONS")

print(
    lag_results.round(4).to_string(index=False)
)

#% 9. Find strongest correlation for each variable

strongest = (
    lag_results.loc[
        lag_results.groupby("variable")["correlation"]
        .apply(lambda x: x.abs().idxmax())
    ]
)

print("\nSTRONGEST CORRELATION FOR EACH VARIABLE")

print(strongest.round(4).to_string(index=False))

#% 10. Plot lead-lag correlations

fig, axes = plt.subplots(
    4, 2,
    figsize=(15, 16)
)

for ax, variable in zip(axes.flat, variables):

    subset = lag_results[
        lag_results["variable"] == variable
    ]

    ax.plot(
        subset["lag"],
        subset["correlation"],
        marker="o"
    )

    ax.axhline(
        0,
        color="black",
        linewidth=0.8
    )

    ax.set_title(variable)
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("Pearson correlation")

    ax.set_xticks(range(0, max_lag + 1))

    ax.grid(True)

plt.suptitle(
    "Lead-Lag Relationships between PM2.5 and Atmospheric Variables",
    fontsize=15
)

plt.tight_layout()

plt.show()



#%% STEP 10 - Create Exploratory_temporal.csv

#%% STEP 10 - Create Exploratory_temporal.csv

#%% STEP 

#%% 1. Import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import ruptures as rpt

from statsmodels.tsa.stattools import acf, pacf, adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox

from scipy.stats import linregress, mannwhitneyu

import json
import warnings

warnings.filterwarnings("ignore")


#%% 2. Project information

MODEL_ID = "mpl_mp25"
MODEL_GROUP = "G8"

DATE = "date"
ID = "id"

variables = [
    "pm25",
    "temperature_celsius",
    "pressure_mb",
    "wind_u",
    "wind_v",
    "NO2",
    "CO",
    "O3",
    "AOD"
]

# PM2.5 threshold used for exceedance statistics.
# Change this value if your project specifies another threshold.
PM25_THRESHOLD = 15


#%% 3. Load and prepare the dataset

#df = pd.read_csv("combined_gee_pm25.csv")
df = pd.read_csv("Data/cleaned_pm25.csv")

# Convert date to datetime
df[DATE] = pd.to_datetime(df[DATE], errors="coerce")

# Remove rows with missing date or station ID
df = df.dropna(subset=[DATE, ID]).copy()

# Convert variables to numeric
for variable in variables:
    df[variable] = pd.to_numeric(
        df[variable],
        errors="coerce"
    )

# Remove duplicate station-date observations
df = df.drop_duplicates(
    subset=[ID, DATE],
    keep="first"
)

# Sort observations
df = df.sort_values(
    by=[ID, DATE]
).reset_index(drop=True)

print("Dataset dimensions:", df.shape)
print("Number of stations:", df[ID].nunique())


#%% 4. Create station identifiers

station_ids = sorted(df[ID].unique())

station_mapping = {
    original_id: f"ST_{i+1:02d}"
    for i, original_id in enumerate(station_ids)
}

df["station_id"] = df[ID].map(station_mapping)

print("Station mapping created successfully.")


#%% 5. Helper functions


# 5.1 Get the longest continuous daily segment

def get_longest_continuous_segment(data, variable):

    data = data[[DATE, variable]].dropna().copy()

    data = data.sort_values(DATE)

    if data.empty:
        return data

    # Identify continuous daily segments
    groups = (
        data[DATE].diff()
        .ne(pd.Timedelta(days=1))
        .cumsum()
    )

    segments = [
        segment
        for _, segment in data.groupby(groups)
    ]

    longest_segment = max(
        segments,
        key=len
    )

    return longest_segment.copy()


# 5.2 Calculate temporal statistics

def calculate_temporal_statistics(data, variable):

    valid_data = data[
        [DATE, variable]
    ].dropna().copy()

    valid_data = valid_data.sort_values(DATE)

    results = {}

    n = len(valid_data)

    results["n_records"] = n

    if n == 0:

        results["t_min"] = np.nan
        results["t_max"] = np.nan
        results["median_sampling_interval_sec"] = np.nan
        results["pct_missing_intervals"] = np.nan

        results["acf_lag1"] = np.nan
        results["acf_lag7"] = np.nan
        results["pacf_lag1"] = np.nan
        results["ljung_box_pvalue"] = np.nan

        results["adf_pvalue"] = np.nan
        results["kpss_pvalue"] = np.nan

        return results

    dates = valid_data[DATE]
    values = valid_data[variable].to_numpy(dtype=float)

    results["t_min"] = dates.min()
    results["t_max"] = dates.max()

    # Sampling intervals

    intervals = dates.diff().dropna().dt.total_seconds()

    if len(intervals) > 0:

        results["median_sampling_interval_sec"] = (
            intervals.median()
        )

    else:

        results["median_sampling_interval_sec"] = np.nan

    # Missing daily intervals

    expected_days = (
        dates.max() - dates.min()
    ).days + 1

    observed_days = dates.dt.normalize().nunique()

    missing_days = max(
        expected_days - observed_days,
        0
    )

    results["pct_missing_intervals"] = (
        missing_days / expected_days * 100
        if expected_days > 0 else np.nan
    )

    # Use longest continuous segment for time-series tests

    segment = get_longest_continuous_segment(
        valid_data,
        variable
    )

    series = segment[variable].to_numpy(dtype=float)

    if len(series) < 2 or np.std(series) == 0:

        results["acf_lag1"] = np.nan
        results["acf_lag7"] = np.nan
        results["pacf_lag1"] = np.nan
        results["ljung_box_pvalue"] = np.nan

        results["adf_pvalue"] = np.nan
        results["kpss_pvalue"] = np.nan

        return results

    # ACF

    try:

        acf_values = acf(
            series,
            nlags=min(7, len(series) - 1),
            fft=False
        )

        results["acf_lag1"] = (
            acf_values[1]
            if len(acf_values) > 1
            else np.nan
        )

        results["acf_lag7"] = (
            acf_values[7]
            if len(acf_values) > 7
            else np.nan
        )

    except Exception:

        results["acf_lag1"] = np.nan
        results["acf_lag7"] = np.nan

    # PACF

    try:

        pacf_values = pacf(
            series,
            nlags=1,
            method="ywm"
        )

        results["pacf_lag1"] = pacf_values[1]

    except Exception:

        results["pacf_lag1"] = np.nan

    # Ljung-Box test

    try:

        if len(series) > 10:

            lb_result = acorr_ljungbox(
                series,
                lags=[10],
                return_df=True
            )

            results["ljung_box_pvalue"] = (
                lb_result["lb_pvalue"].iloc[0]
            )

        else:

            results["ljung_box_pvalue"] = np.nan

    except Exception:

        results["ljung_box_pvalue"] = np.nan

    # ADF test

    try:

        results["adf_pvalue"] = adfuller(
            series,
            autolag="AIC"
        )[1]

    except Exception:

        results["adf_pvalue"] = np.nan

    # KPSS test

    try:

        results["kpss_pvalue"] = kpss(
            series,
            regression="c",
            nlags="auto"
        )[1]

    except Exception:

        results["kpss_pvalue"] = np.nan

    return results


# 5.3 Calculate temporal trend

def calculate_trend(dates, values):

    if len(values) < 2:
        return np.nan, np.nan

    x = (
        (dates - dates.min())
        .dt.total_seconds()
        / 86400
    )

    try:

        result = linregress(
            x,
            values
        )

        return result.slope, result.pvalue

    except Exception:

        return np.nan, np.nan


# 5.4 Calculate seasonal amplitude

def calculate_seasonal_amplitude(data, variable):

    valid_data = data[
        [DATE, variable]
    ].dropna().copy()

    if valid_data.empty:
        return np.nan

    valid_data["month"] = (
        valid_data[DATE].dt.month
    )

    monthly_means = (
        valid_data.groupby("month")[variable]
        .mean()
    )

    # Require all 12 months

    if len(monthly_means) < 12:
        return np.nan

    return (
        monthly_means.max()
        - monthly_means.min()
    )


# 5.5 Calculate drift using Mann-Whitney U

def calculate_drift(data, variable):

    segment = get_longest_continuous_segment(
        data,
        variable
    )

    values = segment[variable].dropna().to_numpy()

    n = len(values)

    if n < 10:
        return np.nan, np.nan

    group_size = max(
        int(n * 0.20),
        1
    )

    first_period = values[:group_size]
    last_period = values[-group_size:]

    try:

        result = mannwhitneyu(
            first_period,
            last_period,
            alternative="two-sided"
        )

        return result.statistic, result.pvalue

    except Exception:

        return np.nan, np.nan


# 5.6 Calculate monthly missingness as JSON

def calculate_monthly_missingness(station_data, variable):

    data = station_data[
        [DATE, variable]
    ].dropna(subset=[DATE]).copy()

    data = data.sort_values(DATE)

    if data.empty:
        return "{}"

    start_date = data[DATE].min()
    end_date = data[DATE].max()

    monthly_missingness = {}

    months = pd.period_range(
        start=start_date.to_period("M"),
        end=end_date.to_period("M"),
        freq="M"
    )

    for month in months:

        month_start = max(
            month.start_time.normalize(),
            start_date
        )

        month_end = min(
            month.end_time.normalize(),
            end_date
        )

        expected_dates = pd.date_range(
            start=month_start,
            end=month_end,
            freq="D"
        )

        monthly_data = data[
            data[DATE].dt.to_period("M") == month
        ]

        observed_dates = (
            monthly_data.loc[
                monthly_data[variable].notna(),
                DATE
            ]
            .dt.normalize()
            .drop_duplicates()
        )

        missing_days = len(
            expected_dates.difference(observed_dates)
        )

        total_days = len(expected_dates)

        missing_pct = (
            missing_days / total_days * 100
            if total_days > 0
            else np.nan
        )

        monthly_missingness[str(month)] = round(
            missing_pct,
            2
        )

    return json.dumps(
        monthly_missingness,
        separators=(",", ":")
    )


# 5.7 Detect regime shifts and return their dates using PELT

def calculate_regime_break_dates(data, variable):

    valid_data = data[
        [DATE, variable]
    ].dropna().copy()

    valid_data = valid_data.sort_values(DATE)

    # Apply PM2.5 validity range

    if variable == "pm25":

        valid_data = valid_data[
            (valid_data[variable] > 0) &
            (valid_data[variable] < 500)
        ]

    if valid_data.empty:
        return []

    # Separate continuous daily segments

    groups = (
        valid_data[DATE].diff()
        .ne(pd.Timedelta(days=1))
        .cumsum()
    )

    change_dates = []

    for _, segment in valid_data.groupby(groups):

        values = segment[variable].to_numpy(dtype=float)

        # Require at least 10 observations

        if len(values) < 10:
            continue

        std = np.std(values)

        if std == 0:
            continue

        # Standardize values

        values = (
            values - np.mean(values)
        ) / std

        try:

            algo = rpt.Pelt(
                model="rbf",
                min_size=2,
                jump=1
            ).fit(values)

            breakpoints = algo.predict(
                pen=2
            )

            # Last breakpoint represents the end of the series

            for bp in breakpoints[:-1]:

                change_date = segment.iloc[bp][DATE]

                change_dates.append(
                    change_date.strftime("%Y-%m-%d")
                )

        except Exception:

            continue

    return change_dates


# 5.8 Generate automatic notes

def generate_notes(record, regime_dates):

    notes = []

    if record["n_records"] < 30:

        return "insufficient observations"

    # Persistence

    acf1 = record["acf_lag1"]

    if pd.notna(acf1) and acf1 >= 0.7:

        notes.append("strong persistence")

    # Regime shifts

    if regime_dates:

        dates_text = ", ".join(regime_dates)

        notes.append(
            f"regime shift at {dates_text}"
        )

    # Significant trend

    trend_pval = record.get(
        "trend_pval",
        np.nan
    )

    if (
        pd.notna(trend_pval)
        and trend_pval < 0.05
    ):

        notes.append("significant trend")

    if not notes:

        return "no major temporal pattern detected"

    return "; ".join(notes)


#%% 6. Generate temporal features for each station

temporal_data = []

for station in sorted(df["station_id"].unique()):

    station_df = df[
        df["station_id"] == station
    ].copy()

    station_df = station_df.sort_values(DATE)

    print("Processing station:", station)

    # Station date range

    station_start = station_df[DATE].min()
    station_end = station_df[DATE].max()

    # Number of complete calendar years elapsed

    n_cycles_observed = (
        station_end.year - station_start.year
        - (
            (station_end.month, station_end.day)
            <
            (station_start.month, station_start.day)
        )
    )

    # Monthly missingness for atmospheric variables

    no2_monthly_missingness = (
        calculate_monthly_missingness(
            station_df,
            "NO2"
        )
    )

    co_monthly_missingness = (
        calculate_monthly_missingness(
            station_df,
            "CO"
        )
    )

    o3_monthly_missingness = (
        calculate_monthly_missingness(
            station_df,
            "O3"
        )
    )

    # Process each variable

    for variable in variables:

        variable_data = station_df[
            [DATE, variable]
        ].dropna().copy()

        variable_data = variable_data.sort_values(DATE)

        # Basic temporal statistics

        stats = calculate_temporal_statistics(
            station_df,
            variable
        )

        # Trend

        trend_slope, trend_pval = calculate_trend(
            variable_data[DATE],
            variable_data[variable].to_numpy()
        )

        # Seasonal amplitude

        seasonal_amplitude = calculate_seasonal_amplitude(
            station_df,
            variable
        )

        # Diurnal amplitude is unavailable for daily data

        diurnal_amplitude = np.nan

        # Drift

        drift_statistic, drift_pvalue = calculate_drift(
            station_df,
            variable
        )

        # Regime shifts

        regime_dates = calculate_regime_break_dates(
            station_df,
            variable
        )

        regime_breaks = len(regime_dates)

        # PM2.5 exceedance statistics only

        if variable == "pm25":

            valid_pm25 = variable_data[
                (variable_data[variable] > 0) &
                (variable_data[variable] < 500)
            ][variable]

            exceedance_count = int(
                (valid_pm25 > PM25_THRESHOLD).sum()
            )

            exceedance_pct = (
                exceedance_count / len(valid_pm25) * 100
                if len(valid_pm25) > 0
                else np.nan
            )

        else:

            exceedance_count = np.nan
            exceedance_pct = np.nan

        # Recommended CV strategy

        cv_strategy = "temporal"

        # Build record

        record = {

            "model_id": MODEL_ID,

            "model_group": MODEL_GROUP,

            "station_id": station,

            "variable": variable,

            "n_records": stats["n_records"],

            "t_min": stats["t_min"],

            "t_max": stats["t_max"],

            "median_sampling_interval_sec":
                stats["median_sampling_interval_sec"],

            "pct_missing_intervals":
                stats["pct_missing_intervals"],

            "acf_lag1":
                stats["acf_lag1"],

            "acf_lag7":
                stats["acf_lag7"],

            "pacf_lag1":
                stats["pacf_lag1"],

            "ljung_box_pvalue":
                stats["ljung_box_pvalue"],

            "adf_pvalue":
                stats["adf_pvalue"],

            "kpss_pvalue":
                stats["kpss_pvalue"],

            "trend_slope":
                trend_slope,

            "trend_pval":
                trend_pval,

            "seasonal_amplitude":
                seasonal_amplitude,

            "diurnal_amplitude":
                diurnal_amplitude,

            "drift_statistic":
                drift_statistic,

            "drift_pvalue":
                drift_pvalue,

            "regime_breaks":
                regime_breaks,

            "regime_break_dates":
                json.dumps(regime_dates),

            "n_cycles_observed":
                n_cycles_observed,

            "no2_pct_missing_by_month":
                no2_monthly_missingness,

            "co_pct_missing_by_month":
                co_monthly_missingness,

            "o3_pct_missing_by_month":
                o3_monthly_missingness,

            "exceedance_count":
                exceedance_count,

            "exceedance_pct":
                exceedance_pct,

            "cv_strategy_recommended":
                cv_strategy,

            "gap_between_train_test_sec":
                86400.0

        }

        # Generate short, station-variable-specific notes

        record["notes"] = generate_notes(
            record,
            regime_dates
        )

        temporal_data.append(record)


#%% 7. Create temporal features DataFrame

temporal_df = pd.DataFrame(temporal_data)

print("\nTemporal features shape:")
print(temporal_df.shape)

print("\nFirst rows:")
print(temporal_df.head())


#%% 8. Sort the results

temporal_df = temporal_df.sort_values(
    by=["station_id", "variable"]
).reset_index(drop=True)


#%% 9. Save the output

temporal_df.to_csv(
    "Exploratory_temporal.csv",
    index=False
)

print("\nFile saved successfully:")
print("Exploratory_temporal.csv")


#%% 10. Display final results

pd.set_option(
    "display.max_columns",
    None
)

pd.set_option(
    "display.max_colwidth",
    100
)

print("\nFinal temporal features:")
print(temporal_df.head(20))

print("\nNumber of rows:", len(temporal_df))
print("Number of stations:", temporal_df["station_id"].nunique())
print("Number of variables:", temporal_df["variable"].nunique())

print("\nNotes examples:")
print(
    temporal_df[
        ["station_id", "variable", "regime_breaks", "notes"]
    ].head(20)
)

import os

print(os.getcwd())
print(os.listdir())
