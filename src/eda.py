import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    RESULTS_DIR,
    PM25_FILE,
    GEE_FILE,
    COMBINED_FILE,
    CLEAN_DATA_FILE,
    EXPLORATORY_DATA_FILE,
    SEED,
)


def run_eda():

    # ==========================================
    # 1. LOAD DATA
    # ==========================================

    pm25 = pd.read_csv(PM25_FILE)
    gee = pd.read_csv(GEE_FILE)
    combined = pd.read_csv(COMBINED_FILE)

    # ==========================================
    # 2. EXPLORATORY DATA ANALYSIS
    # ==========================================

    # Insert the original EDA notebook code here.
    # Keep the original calculations and operations unchanged.

    # ==========================================
    # 3. SAVE EDA OUTPUTS
    # ==========================================

    # Keep the original CSV export commands here.
