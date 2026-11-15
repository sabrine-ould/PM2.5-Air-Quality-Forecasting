import json
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit, GroupKFold, KFold

from src.config import SEED, FOLDS_DIR

# ============================================================
# 1. PREPARE DEVELOPMENT AND TEST DATASETS
# ============================================================


def prepare_development_test(data, dataset_name):

    data = data.copy()

    data["date"] = pd.to_datetime(data["date"])

    development = data[data["dataset_split"].isin(["train", "validation"])].copy()

    test = data[data["dataset_split"] == "test"].copy()

    development = development.reset_index(drop=True)
    test = test.reset_index(drop=True)

    if development.empty or test.empty:
        raise ValueError(f"{dataset_name}: development or test dataset is empty.")

    # Verify that development and test dates do not overlap.

    if development["date"].max() >= test["date"].min():
        raise ValueError(f"{dataset_name}: development and test dates overlap.")

    print(f"\n{dataset_name.upper()} DATA")
    print("-" * 45)

    print("Development shape:", development.shape)
    print("Test shape:", test.shape)

    print(
        "Development dates:", development["date"].min(), "to", development["date"].max()
    )

    print("Test dates:", test["date"].min(), "to", test["date"].max())

    print("Development stations:", development["id"].nunique())

    print("Test stations:", test["id"].nunique())

    return development, test


# ============================================================
# 2. HELPER: CONVERT FOLD INDICES TO JSON-FRIENDLY FORMAT
# ============================================================


def make_fold_record(fold_number, train_idx, validation_idx):

    return {
        "fold": int(fold_number),
        "train_indices": train_idx.tolist(),
        "validation_indices": validation_idx.tolist(),
    }


# ============================================================
# 3. TEMPORAL CROSS-VALIDATION
# ============================================================


def create_temporal_folds(data, dataset_name, n_splits=5):

    print(f"\n{dataset_name.upper()} — TEMPORAL CV")
    print("-" * 45)

    unique_dates = data["date"].drop_duplicates().sort_values().reset_index(drop=True)

    date_to_position = {date: position for position, date in enumerate(unique_dates)}

    splitter = TimeSeriesSplit(n_splits=n_splits, gap=1)

    folds = []

    for fold_number, (train_dates_idx, validation_dates_idx) in enumerate(
        splitter.split(unique_dates), start=1
    ):

        train_dates = unique_dates.iloc[train_dates_idx]
        validation_dates = unique_dates.iloc[validation_dates_idx]

        train_idx = data.index[data["date"].isin(train_dates)].to_numpy()

        validation_idx = data.index[data["date"].isin(validation_dates)].to_numpy()

        # No row may belong to both sets.

        assert len(set(train_idx) & set(validation_idx)) == 0

        # Training dates must precede validation dates.

        assert data.loc[train_idx, "date"].max() < (
            data.loc[validation_idx, "date"].min()
        )

        record = make_fold_record(fold_number, train_idx, validation_idx)

        record["train_date_start"] = str(data.loc[train_idx, "date"].min().date())

        record["train_date_end"] = str(data.loc[train_idx, "date"].max().date())

        record["validation_date_start"] = str(
            data.loc[validation_idx, "date"].min().date()
        )

        record["validation_date_end"] = str(
            data.loc[validation_idx, "date"].max().date()
        )

        record["train_rows"] = int(len(train_idx))
        record["validation_rows"] = int(len(validation_idx))

        folds.append(record)

        print(f"\nFold {fold_number}")
        print("Training rows:", len(train_idx))
        print("Validation rows:", len(validation_idx))
        print(
            "Training dates:",
            record["train_date_start"],
            "to",
            record["train_date_end"],
        )
        print(
            "Validation dates:",
            record["validation_date_start"],
            "to",
            record["validation_date_end"],
        )

    return {
        "dataset": dataset_name,
        "method": "TimeSeriesSplit",
        "n_splits": n_splits,
        "gap": 1,
        "gap_unit": "unique observed dates",
        "folds": folds,
    }


# ============================================================
# 4. SPATIAL CROSS-VALIDATION
# ============================================================


def create_spatial_folds(data, dataset_name, n_splits=5):

    print(f"\n{dataset_name.upper()} — SPATIAL CV")
    print("-" * 45)

    splitter = GroupKFold(n_splits=n_splits)

    groups = data["id"]

    folds = []

    for fold_number, (train_idx, validation_idx) in enumerate(
        splitter.split(data, groups=groups), start=1
    ):

        train_stations = set(data.iloc[train_idx]["id"].unique())

        validation_stations = set(data.iloc[validation_idx]["id"].unique())

        # No station may appear in both sets.

        assert train_stations.isdisjoint(validation_stations)

        record = make_fold_record(fold_number, train_idx, validation_idx)

        record["train_stations"] = sorted(train_stations)

        record["validation_stations"] = sorted(validation_stations)

        record["train_rows"] = int(len(train_idx))
        record["validation_rows"] = int(len(validation_idx))

        folds.append(record)

        print(f"\nFold {fold_number}")
        print("Training rows:", len(train_idx))
        print("Validation rows:", len(validation_idx))
        print("Training stations:", len(train_stations))
        print("Validation stations:", len(validation_stations))

    return {
        "dataset": dataset_name,
        "method": "GroupKFold",
        "group_column": "id",
        "n_splits": n_splits,
        "folds": folds,
    }


# ============================================================
# 5. RANDOM CROSS-VALIDATION
# ============================================================


def create_random_folds(data, dataset_name, n_splits=5, random_state=SEED):

    print(f"\n{dataset_name.upper()} — RANDOM CV")
    print("-" * 45)

    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    folds = []

    for fold_number, (train_idx, validation_idx) in enumerate(
        splitter.split(data), start=1
    ):

        assert len(set(train_idx) & set(validation_idx)) == 0

        record = make_fold_record(fold_number, train_idx, validation_idx)

        record["train_rows"] = int(len(train_idx))
        record["validation_rows"] = int(len(validation_idx))

        folds.append(record)

        print(f"\nFold {fold_number}")
        print("Training rows:", len(train_idx))
        print("Validation rows:", len(validation_idx))

    return {
        "dataset": dataset_name,
        "method": "KFold",
        "shuffle": True,
        "random_state": int(random_state),
        "n_splits": n_splits,
        "purpose": "Diagnostic comparison only",
        "folds": folds,
    }


# ============================================================
# 6. SAVE JSON FILE
# ============================================================


def save_folds_json(content, filename):

    FOLDS_DIR.mkdir(parents=True, exist_ok=True)

    output_file = FOLDS_DIR / filename

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(content, file, indent=4, ensure_ascii=False)

    print("\nSaved:", output_file)

    return output_file


# ============================================================
# 7. RUN ALL VALIDATION SCHEMES
# ============================================================


def run_validation_schemes(next_day_forecast, same_day_nowcasting):

    print("=" * 65)
    print("5.4 — VALIDATION SCHEMES")
    print("=" * 65)

    # Prepare development and independent test sets.

    forecast_dev, forecast_test = prepare_development_test(
        next_day_forecast, "forecasting"
    )

    nowcast_dev, nowcast_test = prepare_development_test(
        same_day_nowcasting, "nowcasting"
    )

    # Temporal folds.

    forecast_temporal = create_temporal_folds(forecast_dev, "forecasting")

    nowcast_temporal = create_temporal_folds(nowcast_dev, "nowcasting")

    # Spatial folds.

    forecast_spatial = create_spatial_folds(forecast_dev, "forecasting")

    nowcast_spatial = create_spatial_folds(nowcast_dev, "nowcasting")

    # Random folds.

    forecast_standard = create_random_folds(forecast_dev, "forecasting")

    nowcast_standard = create_random_folds(nowcast_dev, "nowcasting")

    # Save the three JSON files.

    temporal_content = {
        "description": (
            "Temporal cross-validation folds for forecasting " "and nowcasting."
        ),
        "forecasting": forecast_temporal,
        "nowcasting": nowcast_temporal,
    }

    spatial_content = {
        "description": ("Spatial cross-validation folds grouped by station ID."),
        "forecasting": forecast_spatial,
        "nowcasting": nowcast_spatial,
    }

    standard_content = {
        "description": ("Random KFold splits for diagnostic comparison only."),
        "forecasting": forecast_standard,
        "nowcasting": nowcast_standard,
    }

    save_folds_json(temporal_content, "folds_temporal.json")

    save_folds_json(spatial_content, "folds_spatial.json")

    save_folds_json(standard_content, "folds_standard.json")

    # Final summary.

    print("\n" + "=" * 65)
    print("VALIDATION SCHEMES COMPLETED")
    print("=" * 65)

    print("Forecasting development:", forecast_dev.shape)
    print("Forecasting test:", forecast_test.shape)

    print("Nowcasting development:", nowcast_dev.shape)
    print("Nowcasting test:", nowcast_test.shape)

    print("\nTotal fold pairs:", 30)

    print("\nTest datasets were kept outside all development folds.")

    return {
        "forecast_dev": forecast_dev,
        "forecast_test": forecast_test,
        "nowcast_dev": nowcast_dev,
        "nowcast_test": nowcast_test,
        "forecast_temporal": forecast_temporal,
        "nowcast_temporal": nowcast_temporal,
        "forecast_spatial": forecast_spatial,
        "nowcast_spatial": nowcast_spatial,
        "forecast_standard": forecast_standard,
        "nowcast_standard": nowcast_standard,
    }
