from pathlib import Path

# ==========================================
# PROJECT ROOT
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ==========================================
# RANDOM SEED
# ==========================================

SEED = 42


# ==========================================
# MAIN DIRECTORIES
# ==========================================

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
FOLDS_DIR = PROJECT_ROOT / "folds"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
REPORTS_DIR = PROJECT_ROOT / "reports"
PRESENTATION_DIR = PROJECT_ROOT / "presentation"
# ==========================================
# CREATE PROJECT DIRECTORIES
# ==========================================

DIRECTORIES = [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    NOTEBOOKS_DIR,
    SRC_DIR,
    SCRIPTS_DIR,
    FOLDS_DIR,
    RESULTS_DIR,
    FIGURES_DIR,
    REPORTS_DIR,
    PRESENTATION_DIR,
]

for directory in DIRECTORIES:
    directory.mkdir(parents=True, exist_ok=True)


# ==========================================
# RAW DATA FILES
# ==========================================

PM25_FILE = RAW_DATA_DIR / "pm25.csv"
GEE_FILE = RAW_DATA_DIR / "gee_features_daily.csv"
COMBINED_FILE = RAW_DATA_DIR / "combined_gee_pm25.csv"


# ==========================================
# PROCESSED DATA FILES
# ==========================================

CLEAN_DATA_FILE = PROCESSED_DATA_DIR / "combined_clean.csv"

STATION_QUALITY_FILE = PROCESSED_DATA_DIR / "station_quality.csv"

REGULAR_DAILY_TIMELINE_FILE = PROCESSED_DATA_DIR / "regular_daily_timeline.csv"

TEMPORAL_FEATURES_FILE = PROCESSED_DATA_DIR / "temporal_features.csv"

TEMPORAL_FEATURES_5_3_4_FILE = PROCESSED_DATA_DIR / "temporal_features_5_3_4.csv"


# ==========================================
# RESULTS FILES
# ==========================================

EXPLORATORY_DATA_FILE = RESULTS_DIR / "Exploratory_Data.csv"

EXPLORATORY_TEMPORAL_FILE = RESULTS_DIR / "Exploratory_temporal.csv"


# ==========================================
# VALIDATION FILES
# ==========================================

VALIDATION_FOLDS_FILE = FOLDS_DIR / "validation_fold_indices_5_4.npz"
