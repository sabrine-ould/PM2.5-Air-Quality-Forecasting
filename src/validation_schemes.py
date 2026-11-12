
#%% ============================================================
# SECTION 5.4: VALIDATION SCHEMES
# ============================================================


#%% Stage 1: Import libraries

import numpy as np
import pandas as pd
import os
import contextlib
import io

from sklearn.model_selection import (
    TimeSeriesSplit,
    GroupKFold,
    KFold
)


#%% Stage 2: Load data

# Import preprocessing variables
with contextlib.redirect_stdout(io.StringIO()):
    import preprocessing as prep

# Load cleaned dataset
df = pd.read_csv("../Data/cleaned_pm25.csv")

df["date"] = pd.to_datetime(df["date"])

# Retrieve prepared data
X_train = prep.X_train
X_test = prep.X_test

y_train = prep.y_train
y_test = prep.y_test

print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)

print("Training target:", y_train.shape)
print("Testing target:", y_test.shape)


#%% Stage 3: Data alignment and verification

# Check feature and target dimensions
assert len(X_train) == len(y_train)
assert len(X_test) == len(y_test)

# Check feature and target indices
if isinstance(y_train, pd.Series):
    assert X_train.index.equals(y_train.index)

if isinstance(y_test, pd.Series):
    assert X_test.index.equals(y_test.index)

# Check original dataset indices
assert df.index.is_unique

assert X_train.index.isin(df.index).all()
assert X_test.index.isin(df.index).all()

# Retrieve original rows
train_rows = df.loc[X_train.index].copy()
test_rows = df.loc[X_test.index].copy()

# Check row alignment
assert train_rows.index.equals(X_train.index)
assert test_rows.index.equals(X_test.index)

# Check no overlap between training and testing
assert len(
    X_train.index.intersection(X_test.index)
) == 0

# Check date information
assert train_rows["date"].notna().all()
assert test_rows["date"].notna().all()

# Check chronological separation
assert train_rows["date"].max() < test_rows["date"].min()

# Check station metadata
assert "sensor_name" in train_rows.columns

# Station identity must not be a predictive feature
assert "sensor_name" not in X_train.columns

print("\nData alignment verified.")

print(
    "Training period:",
    train_rows["date"].min(),
    "to",
    train_rows["date"].max()
)

print(
    "Testing period:",
    test_rows["date"].min(),
    "to",
    test_rows["date"].max()
)


#%% Stage 4: Define fold storage and configuration

# Save file in the current working directory
save_path = "validation_splits.npz"

# Change this version if the split methodology changes
config_version = "validation_schemes_v1"

# Required validation schemes
scheme_names = [
    "temporal",
    "spatial",
    "random",
    "spatiotemporal"
]

# Check whether a compatible saved file already exists
reuse_existing = False

if os.path.exists(save_path):

    try:
        with np.load(save_path) as saved_data:

            required_keys = {
                "train_row_labels",
                "config_version"
            }

            for scheme_name in scheme_names:

                for i in range(5):

                    required_keys.add(
                        f"{scheme_name}_train_{i}"
                    )

                    required_keys.add(
                        f"{scheme_name}_val_{i}"
                    )

            same_dataset = np.array_equal(
                saved_data["train_row_labels"],
                np.asarray(X_train.index)
            )

            same_configuration = (
                saved_data["config_version"].item()
                == config_version
            )

            has_all_folds = required_keys.issubset(
                set(saved_data.files)
            )

            reuse_existing = (
                same_dataset
                and same_configuration
                and has_all_folds
            )

    except (ValueError, KeyError, OSError):
        reuse_existing = False


#%% Stage 5: Load existing folds OR create new folds

all_schemes = {}

if reuse_existing:

    # Load previously saved folds
    with np.load(save_path) as saved_data:

        for scheme_name in scheme_names:

            all_schemes[scheme_name] = []

            for i in range(5):

                train_idx = saved_data[
                    f"{scheme_name}_train_{i}"
                ].copy()

                val_idx = saved_data[
                    f"{scheme_name}_val_{i}"
                ].copy()

                all_schemes[scheme_name].append(
                    (train_idx, val_idx)
                )

    print("\nExisting validation folds loaded and reused.")

else:

    print("\nCreating validation folds...")


    #%% Stage 5.1: Temporal Cross-Validation

    train_dates = pd.to_datetime(
        train_rows["date"]
    ).reset_index(drop=True)

    unique_dates = np.sort(
        train_dates.unique()
    )

    time_cv = TimeSeriesSplit(
        n_splits=5,
        gap=1
    )

    temporal_splits = []

    for fold, (train_date_idx, val_date_idx) in enumerate(
        time_cv.split(unique_dates), start=1
    ):

        fold_train_dates = unique_dates[
            train_date_idx
        ]

        fold_val_dates = unique_dates[
            val_date_idx
        ]

        # Map dates to observation positions
        train_idx = np.flatnonzero(
            train_dates.isin(
                fold_train_dates
            ).to_numpy()
        )

        val_idx = np.flatnonzero(
            train_dates.isin(
                fold_val_dates
            ).to_numpy()
        )

        temporal_splits.append(
            (train_idx, val_idx)
        )

        train_end = train_dates.iloc[
            train_idx
        ].max()

        val_start = train_dates.iloc[
            val_idx
        ].min()

        val_end = train_dates.iloc[
            val_idx
        ].max()

        print(f"\nTemporal Fold {fold}")

        print("Training:", train_dates.iloc[train_idx].min(),
              "to", train_end)

        print("Validation:", val_start, "to", val_end)

        assert train_end < val_start

        assert len(
            np.intersect1d(train_idx, val_idx)
        ) == 0

    print("\nTemporal CV: 5 folds created.")


    #%% Stage 5.2: Spatial Cross-Validation

    # Station ID is used only for grouping
    assert "sensor_name" not in X_train.columns

    train_stations = train_rows[
        "sensor_name"
    ].reset_index(drop=True)

    assert train_stations.notna().all()

    spatial_cv = GroupKFold(
        n_splits=5
    )

    spatial_splits = list(
        spatial_cv.split(
            X_train,
            groups=train_stations
        )
    )

    # Verify station separation
    for fold, (train_idx, val_idx) in enumerate(
        spatial_splits, start=1
    ):

        train_station_set = set(
            train_stations.iloc[train_idx]
        )

        val_station_set = set(
            train_stations.iloc[val_idx]
        )

        common_stations = (
            train_station_set & val_station_set
        )

        print(f"\nSpatial Fold {fold}")

        print("Training stations:",
              len(train_station_set))

        print("Validation stations:",
              len(val_station_set))

        print("Common stations:",
              len(common_stations))

        assert len(common_stations) == 0

    print("\nSpatial CV: 5 folds created.")


    #%% Stage 5.3: Random Cross-Validation

    random_cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    random_splits = list(
        random_cv.split(X_train)
    )

    for fold, (train_idx, val_idx) in enumerate(
        random_splits, start=1
    ):

        print(f"\nRandom Fold {fold}")

        print("Training observations:",
              len(train_idx))

        print("Validation observations:",
              len(val_idx))

        assert len(
            np.intersect1d(train_idx, val_idx)
        ) == 0

    print("\nRandom CV: 5 folds created.")
    print("Diagnostic only - not for final model selection.")


    #%% Stage 5.4: Spatio-temporal Blocked CV

    # Prepare time and station information
    train_dates = pd.to_datetime(
        train_rows["date"]
    ).reset_index(drop=True)

    train_stations = train_rows[
        "sensor_name"
    ].reset_index(drop=True)

    assert train_dates.notna().all()
    assert train_stations.notna().all()

    # Create chronological blocks
    unique_dates = np.sort(
        train_dates.unique()
    )

    time_cv = TimeSeriesSplit(
        n_splits=5,
        gap=1
    )

    time_blocks = list(
        time_cv.split(unique_dates)
    )

    # Create spatial station groups
    station_cv = GroupKFold(
        n_splits=5
    )

    station_blocks = list(
        station_cv.split(
            X_train,
            groups=train_stations
        )
    )

    spatiotemporal_splits = []

    # Combine temporal and spatial blocks
    for fold, (time_block, station_block) in enumerate(
        zip(time_blocks, station_blocks), start=1
    ):

        # Exact temporal training and validation dates
        train_date_positions, val_date_positions = time_block

        allowed_train_dates = unique_dates[
            train_date_positions
        ]

        val_dates = unique_dates[
            val_date_positions
        ]

        # Stations reserved for validation
        _, val_station_positions = station_block

        val_stations = train_stations.iloc[
            val_station_positions
        ].unique()

        # Training: past dates and other stations
        train_mask = (
            train_dates.isin(allowed_train_dates)
            &
            (~train_stations.isin(val_stations))
        )

        # Validation: future dates and held-out stations
        val_mask = (
            train_dates.isin(val_dates)
            &
            train_stations.isin(val_stations)
        )

        train_idx = np.flatnonzero(
            train_mask.to_numpy()
        )

        val_idx = np.flatnonzero(
            val_mask.to_numpy()
        )

        if len(train_idx) == 0 or len(val_idx) == 0:

            raise ValueError(
                f"Spatio-temporal Fold {fold} is empty."
            )

        spatiotemporal_splits.append(
            (train_idx, val_idx)
        )

        # Verify station separation
        train_station_set = set(
            train_stations.iloc[train_idx]
        )

        val_station_set = set(
            train_stations.iloc[val_idx]
        )

        common_stations = (
            train_station_set & val_station_set
        )

        # Verify chronological order
        train_end = train_dates.iloc[
            train_idx
        ].max()

        val_begin = train_dates.iloc[
            val_idx
        ].min()

        val_end = train_dates.iloc[
            val_idx
        ].max()

        print(f"\nSpatio-temporal Fold {fold}")

        print("Training observations:", len(train_idx))
        print("Validation observations:", len(val_idx))

        print("Training period ends:", train_end)

        print("Validation period:",
              val_begin, "to", val_end)

        print("Training stations:",
              len(train_station_set))

        print("Validation stations:",
              len(val_station_set))

        print("Common stations:",
              len(common_stations))

        assert train_end < val_begin
        assert len(common_stations) == 0

    print("\nSpatio-temporal CV: 5 folds created.")


    #%% Stage 5.5: Collect all validation schemes

    all_schemes = {
        "temporal": temporal_splits,
        "spatial": spatial_splits,
        "random": random_splits,
        "spatiotemporal": spatiotemporal_splits
    }


    #%% Stage 5.6: Save all fold indices once

    arrays_to_save = {
        "train_row_labels": np.asarray(X_train.index),
        "config_version": np.asarray(config_version)
    }

    for scheme_name, splits in all_schemes.items():

        assert len(splits) == 5

        for i, (train_idx, val_idx) in enumerate(splits):

            arrays_to_save[
                f"{scheme_name}_train_{i}"
            ] = train_idx

            arrays_to_save[
                f"{scheme_name}_val_{i}"
            ] = val_idx

    np.savez_compressed(
        save_path,
        **arrays_to_save
    )

    print("\nAll validation folds saved successfully.")
    print("Saved file:", save_path)


#%% Stage 6: Verify all saved folds

with np.load(save_path) as saved_data:

    # Verify dataset identity
    assert np.array_equal(
        saved_data["train_row_labels"],
        np.asarray(X_train.index)
    )

    # Verify configuration
    assert (
        saved_data["config_version"].item()
        == config_version
    )

    for scheme_name in scheme_names:

        assert len(all_schemes[scheme_name]) == 5

        for i in range(5):

            train_idx = saved_data[
                f"{scheme_name}_train_{i}"
            ]

            val_idx = saved_data[
                f"{scheme_name}_val_{i}"
            ]

            # Check non-empty folds
            assert len(train_idx) > 0
            assert len(val_idx) > 0

            # Check index boundaries
            assert train_idx.min() >= 0
            assert val_idx.min() >= 0

            assert train_idx.max() < len(X_train)
            assert val_idx.max() < len(X_train)

            # Check train-validation separation
            assert len(
                np.intersect1d(train_idx, val_idx)
            ) == 0

            # Check independent future test set
            train_original_ids = X_train.index[train_idx]
            val_original_ids = X_train.index[val_idx]

            assert len(
                train_original_ids.intersection(
                    X_test.index
                )
            ) == 0

            assert len(
                val_original_ids.intersection(
                    X_test.index
                )
            ) == 0


# Additional spatio-temporal verification
for train_idx, val_idx in all_schemes["spatiotemporal"]:

    train_dates_check = pd.to_datetime(
        train_rows["date"]
    ).reset_index(drop=True)

    train_stations_check = train_rows[
        "sensor_name"
    ].reset_index(drop=True)

    train_end = train_dates_check.iloc[
        train_idx
    ].max()

    val_start = train_dates_check.iloc[
        val_idx
    ].min()

    train_station_set = set(
        train_stations_check.iloc[train_idx]
    )

    val_station_set = set(
        train_stations_check.iloc[val_idx]
    )

    assert train_end < val_start

    assert len(
        train_station_set & val_station_set
    ) == 0


print("\nAll saved folds verified.")
print("Future test set remains independent.")


#%% Load and inspect NPZ file

import numpy as np

data = np.load("validation_splits.npz")

# Show saved arrays
print("Saved arrays:")
print(data.files)

# Show information about each fold
for key in data.files:
    arr = data[key]
    if isinstance(arr, np.ndarray) and arr.ndim == 1:
        print(key, ":", arr.shape)
        
print("\nTemporal Fold 1 - Training indices:")
print(data["temporal_train_0"])

print("\nTemporal Fold 1 - Validation indices:")
print(data["temporal_val_0"])


#%% Export validation folds to CSV

csv_rows = []

schemes = ["temporal", "spatial", "random", "spatiotemporal"]

for scheme in schemes:
    for fold in range(5):

        train_indices = arrays_to_save[f"{scheme}_train_{fold}"]
        val_indices = arrays_to_save[f"{scheme}_val_{fold}"]

        for idx in train_indices:
            csv_rows.append({
                "scheme": scheme,
                "fold": fold + 1,
                "split": "train",
                "row_position": idx
            })

        for idx in val_indices:
            csv_rows.append({
                "scheme": scheme,
                "fold": fold + 1,
                "split": "validation",
                "row_position": idx
            })

folds_df = pd.DataFrame(csv_rows)

folds_df.to_csv("validation_splits.csv", index=False)

print("CSV file saved successfully!")
print("CSV shape:", folds_df.shape)
print(folds_df.head())

#%% Stage 7: Final summary

print("\n========== VALIDATION SCHEMES SUMMARY ==========")

print("1. Temporal CV: 5 folds, TimeSeriesSplit, gap=1")

print("2. Spatial CV: 5 folds, GroupKFold, sensor_name")

print("3. Random CV: 5 folds, KFold, random_state=42")

print("4. Spatio-temporal CV: 5 folds, temporal + spatial blocking")

print("5. Fold indices: Saved once and reused")

print("6. Total folds:", sum(
    len(splits) for splits in all_schemes.values()
))

print("7. Future test set: Independent")

print("8. Station ID: Grouping only, excluded from predictive features")

print("\nSection 5.4 completed.")



