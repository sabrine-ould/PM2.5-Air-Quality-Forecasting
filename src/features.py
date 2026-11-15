import numpy as np
import pandas as pd

from src.config import (
    CLEAN_DATA_FILE,
    TEMPORAL_FEATURES_5_3_4_FILE,
)

# ============================================================
# 1. LOAD AND VALIDATE CLEANED DATA
# ============================================================


def load_temporal_data():

    df = pd.read_csv(CLEAN_DATA_FILE)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    numeric_columns = [
        "latitude",
        "longitude",
        "pm25",
        "AOD",
        "NO2",
        "CO",
        "O3",
        "temperature_celsius",
        "pressure_mb",
        "wind_u",
        "wind_v",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["date", "id", "pm25"]).copy()

    df = df[(df["pm25"] > 0) & (df["pm25"] < 500)].copy()

    df["id"] = df["id"].astype(str)

    df = df.sort_values(["id", "date"]).reset_index(drop=True)

    duplicate_count = df.duplicated(subset=["id", "date"]).sum()

    if duplicate_count > 0:
        raise ValueError("Duplicate station-date records exist.")

    coordinate_counts = df.groupby("id")[["latitude", "longitude"]].apply(
        lambda x: x.drop_duplicates().shape[0]
    )

    if (coordinate_counts > 1).any():
        raise ValueError("Some station IDs have multiple coordinate pairs.")

    print("Cleaned dataset shape:", df.shape)
    print("Number of stations:", df["id"].nunique())
    print("Date range:", df["date"].min(), "to", df["date"].max())

    return df


# ============================================================
# 2. STATION SELECTION
# ============================================================


def longest_consecutive_run(series):

    valid_dates = pd.DatetimeIndex(series.index[series.notna()]).sort_values()

    if len(valid_dates) == 0:
        return 0

    differences = valid_dates.to_series().diff().dt.days

    groups = differences.ne(1).cumsum()

    run_lengths = valid_dates.to_series().groupby(groups).size()

    return int(run_lengths.max())


def select_stations(df, min_consecutive_days=30):

    station_information = []

    for station_id, group in df.groupby("id"):

        group = group.sort_values("date").set_index("date")

        full_dates = pd.date_range(
            start=group.index.min(), end=group.index.max(), freq="D"
        )

        daily_pm25 = group["pm25"].reindex(full_dates)

        longest_run = longest_consecutive_run(daily_pm25)

        station_information.append(
            {
                "id": station_id,
                "total_observations": int(group["pm25"].count()),
                "longest_consecutive_days": longest_run,
            }
        )

    station_selection = pd.DataFrame(station_information)

    retained_ids = station_selection.loc[
        station_selection["longest_consecutive_days"] >= min_consecutive_days, "id"
    ].tolist()

    if len(retained_ids) == 0:
        raise ValueError("No stations meet the consecutive-day requirement.")

    selected_df = df[df["id"].isin(retained_ids)].copy()

    print("\nSTATION SELECTION")
    print("-----------------")
    print("Stations before selection:", df["id"].nunique())
    print("Stations retained:", len(retained_ids))
    print("Stations excluded:", df["id"].nunique() - len(retained_ids))

    print(
        station_selection.sort_values("longest_consecutive_days", ascending=False)
        .head(10)
        .to_string(index=False)
    )

    return selected_df, station_selection


# ============================================================
# 3. CREATE REGULAR DAILY TIMELINE
# ============================================================


def create_daily_timeline(df):

    daily_station_data = []

    for station_id, group in df.groupby("id"):

        group = group.sort_values("date").copy()

        sensor_name = group["sensor_name"].iloc[0]
        latitude = group["latitude"].iloc[0]
        longitude = group["longitude"].iloc[0]

        full_dates = pd.date_range(
            start=group["date"].min(), end=group["date"].max(), freq="D"
        )

        group = group.set_index("date")

        daily_group = group.reindex(full_dates)

        daily_group.index.name = "date"
        daily_group = daily_group.reset_index()

        daily_group["id"] = station_id
        daily_group["sensor_name"] = sensor_name
        daily_group["latitude"] = latitude
        daily_group["longitude"] = longitude

        daily_group["observed_record"] = daily_group["pm25"].notna()

        daily_station_data.append(daily_group)

    daily_df = pd.concat(daily_station_data, ignore_index=True)

    daily_df = daily_df.sort_values(["id", "date"]).reset_index(drop=True)

    print("\nDAILY TIMELINE")
    print("--------------")
    print("Timeline shape:", daily_df.shape)
    print("Number of stations:", daily_df["id"].nunique())
    print("Observed station-days:", int(daily_df["observed_record"].sum()))
    print("Inserted missing days:", int((~daily_df["observed_record"]).sum()))

    return daily_df


# ============================================================
# 4. CHRONOLOGICAL SPLITS AND HIGH-AOD THRESHOLD
# ============================================================


def assign_temporal_splits(
    daily_df, train_end="2024-04-19", validation_end="2025-04-03"
):

    daily_df = daily_df.copy()

    train_end = pd.Timestamp(train_end)
    validation_end = pd.Timestamp(validation_end)

    daily_df["dataset_split"] = np.select(
        [daily_df["date"] <= train_end, daily_df["date"] <= validation_end],
        ["train", "validation"],
        default="test",
    )

    print("\nCHRONOLOGICAL SPLITS")
    print("--------------------")

    print(daily_df.groupby("dataset_split")["date"].agg(["min", "max", "count"]))

    return daily_df


def calculate_high_aod_threshold(daily_df):

    training_aod = daily_df.loc[
        (daily_df["dataset_split"] == "train") & daily_df["AOD"].notna(), "AOD"
    ]

    if len(training_aod) == 0:
        raise ValueError("No training-period AOD values are available.")

    threshold = training_aod.quantile(0.90)

    print("\nTraining-period AOD 90th percentile:", round(threshold, 5))

    return threshold


# ============================================================
# 5. TEMPORAL FEATURE ENGINEERING
# ============================================================


def engineer_temporal_features(daily_df, high_aod_threshold):

    engineered_station_data = []

    for station_id, group in daily_df.groupby("id"):

        group = group.sort_values("date").copy()
        group = group.set_index("date")

        # A. Derived wind speed

        group["wind_speed"] = np.sqrt(group["wind_u"] ** 2 + group["wind_v"] ** 2)

        # B. PM2.5 lags and past-only rolling statistics

        group["pm25_lag_1"] = group["pm25"].shift(1)

        past_pm25 = group["pm25"].shift(1)

        group["pm25_roll3_mean"] = past_pm25.rolling(window=3, min_periods=3).mean()

        group["pm25_roll3_std"] = past_pm25.rolling(window=3, min_periods=3).std()

        # C. AOD lags

        for lag in [1, 2, 3]:
            group[f"AOD_lag_{lag}"] = group["AOD"].shift(lag)

        # D. NO2 lag

        group["NO2_lag_1"] = group["NO2"].shift(1)

        # E. Temperature and wind lags

        group["temperature_lag_1"] = group["temperature_celsius"].shift(1)

        group["wind_speed_lag_1"] = group["wind_speed"].shift(1)

        # F. Day-to-day differences

        group["AOD_diff_1"] = group["AOD"].shift(1) - group["AOD"].shift(2)

        aod_four_day_count = (
            group["AOD"].shift(1).rolling(window=4, min_periods=4).count()
        )

        group["AOD_diff_3"] = (group["AOD"].shift(1) - group["AOD"].shift(4)).where(
            aod_four_day_count == 4
        )

        group["temperature_diff_1"] = group["temperature_celsius"].shift(1) - group[
            "temperature_celsius"
        ].shift(2)

        group["NO2_diff_1"] = group["NO2"].shift(1) - group["NO2"].shift(2)

        # G. AOD rates of change

        previous_aod_1 = group["AOD"].shift(2)
        previous_aod_3 = group["AOD"].shift(4)

        group["AOD_rate_1"] = group["AOD_diff_1"] / previous_aod_1

        group["AOD_rate_3"] = group["AOD_diff_3"] / previous_aod_3

        group[["AOD_rate_1", "AOD_rate_3"]] = group[
            ["AOD_rate_1", "AOD_rate_3"]
        ].replace([np.inf, -np.inf], np.nan)

        # H. Past-only AOD rolling statistics

        past_aod = group["AOD"].shift(1)

        group["AOD_roll3_mean"] = past_aod.rolling(window=3, min_periods=3).mean()

        group["AOD_roll3_std"] = past_aod.rolling(window=3, min_periods=3).std()

        group["AOD_roll7_mean"] = past_aod.rolling(window=7, min_periods=7).mean()

        group["AOD_roll7_std"] = past_aod.rolling(window=7, min_periods=7).std()

        group["AOD_roll7_max"] = past_aod.rolling(window=7, min_periods=7).max()

        # I. Time since last high-AOD event

        aod_segment = past_aod.isna().cumsum()

        previous_day_dates = pd.Series(
            group.index - pd.Timedelta(days=1), index=group.index
        )

        event_dates = previous_day_dates.where(past_aod >= high_aod_threshold)

        last_event_date = event_dates.groupby(aod_segment).ffill()

        group["days_since_high_AOD"] = (
            pd.Series(group.index, index=group.index) - last_event_date
        ).dt.days

        # J. Cyclical calendar features

        day_of_year = group.index.dayofyear
        month = group.index.month

        group["day_of_year_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)

        group["day_of_year_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)

        group["month_sin"] = np.sin(2 * np.pi * month / 12)

        group["month_cos"] = np.cos(2 * np.pi * month / 12)

        group = group.reset_index()

        engineered_station_data.append(group)

    engineered_df = pd.concat(engineered_station_data, ignore_index=True)

    engineered_df = engineered_df.sort_values(["id", "date"]).reset_index(drop=True)

    print("\nENGINEERED DATASET")
    print("------------------")
    print("Shape:", engineered_df.shape)

    return engineered_df


# ============================================================
# 6. CREATE FORECASTING AND NOWCASTING DATASETS
# ============================================================


def create_forecasting_datasets(engineered_df):

    # A. Next-day forecasting
    # The row date is the target date.
    # Predictors use information from the previous days only.

    forecast_features = [
        "pm25_lag_1",
        "pm25_roll3_mean",
        "pm25_roll3_std",
        "AOD_lag_1",
        "AOD_lag_2",
        "AOD_lag_3",
        "AOD_diff_1",
        "AOD_diff_3",
        "AOD_rate_1",
        "AOD_rate_3",
        "AOD_roll3_mean",
        "AOD_roll3_std",
        "AOD_roll7_mean",
        "AOD_roll7_std",
        "AOD_roll7_max",
        "NO2_lag_1",
        "NO2_diff_1",
        "temperature_lag_1",
        "temperature_diff_1",
        "wind_speed_lag_1",
        "days_since_high_AOD",
        "day_of_year_sin",
        "day_of_year_cos",
        "month_sin",
        "month_cos",
    ]

    next_day_forecast = engineered_df[
        ["date", "id", "sensor_name", "dataset_split", "pm25"] + forecast_features
    ].copy()

    next_day_forecast = next_day_forecast.dropna(subset=["pm25", "pm25_lag_1"]).copy()

    # B. Same-day nowcasting

    nowcast_features = [
        "AOD",
        "NO2",
        "CO",
        "O3",
        "temperature_celsius",
        "pressure_mb",
        "wind_u",
        "wind_v",
        "wind_speed",
        "day_of_year_sin",
        "day_of_year_cos",
        "month_sin",
        "month_cos",
    ]

    same_day_nowcasting = engineered_df[
        ["date", "id", "sensor_name", "dataset_split", "pm25"] + nowcast_features
    ].copy()

    same_day_nowcasting = same_day_nowcasting.dropna(subset=["pm25"]).copy()

    print("\nNEXT-DAY FORECASTING")
    print("--------------------")
    print("Shape:", next_day_forecast.shape)
    print("Stations:", next_day_forecast["id"].nunique())
    print(next_day_forecast["dataset_split"].value_counts())

    print("\nSAME-DAY NOWCASTING")
    print("-------------------")
    print("Shape:", same_day_nowcasting.shape)
    print("Stations:", same_day_nowcasting["id"].nunique())
    print(same_day_nowcasting["dataset_split"].value_counts())

    return next_day_forecast, same_day_nowcasting


# ============================================================
# 7. RUN COMPLETE FEATURE ENGINEERING PIPELINE
# ============================================================


def run_feature_engineering():

    print("=" * 65)
    print("5.3.4 — TEMPORAL PREPROCESSING AND FEATURE ENGINEERING")
    print("=" * 65)

    df = load_temporal_data()

    selected_df, station_selection = select_stations(df, min_consecutive_days=30)

    daily_df = create_daily_timeline(selected_df)

    daily_df = assign_temporal_splits(daily_df)

    high_aod_threshold = calculate_high_aod_threshold(daily_df)

    engineered_df = engineer_temporal_features(daily_df, high_aod_threshold)

    next_day_forecast, same_day_nowcasting = create_forecasting_datasets(engineered_df)

    # Final checks

    assert engineered_df.duplicated(["id", "date"]).sum() == 0

    assert all(
        group["date"].is_monotonic_increasing
        for _, group in engineered_df.groupby("id")
    )

    assert next_day_forecast["pm25_lag_1"].isna().sum() == 0

    # Save final engineered dataset

    TEMPORAL_FEATURES_5_3_4_FILE.parent.mkdir(parents=True, exist_ok=True)

    engineered_df.to_csv(TEMPORAL_FEATURES_5_3_4_FILE, index=False)

    print("\n" + "=" * 65)
    print("SECTION 5.3.4 COMPLETED")
    print("=" * 65)

    print("Engineered dataset:", engineered_df.shape)
    print("Forecasting dataset:", next_day_forecast.shape)
    print("Nowcasting dataset:", same_day_nowcasting.shape)

    print("Saved file:", TEMPORAL_FEATURES_5_3_4_FILE)

    return (engineered_df, next_day_forecast, same_day_nowcasting, station_selection)
