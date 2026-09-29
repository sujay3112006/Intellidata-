"""
StockSense Configuration Module
Contains directory paths, constants, seed, and standardization mappings.
"""

from pathlib import Path

# Repository Root Directory
SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent

# Data Directories
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Output & Model Directories
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = ROOT_DIR / "models"
DASHBOARD_DIR = ROOT_DIR / "dashboard"

# Create directories if they do not exist
for directory in [RAW_DIR, PROCESSED_DIR, REPORTS_DIR, FIGURES_DIR, MODELS_DIR, DASHBOARD_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Shared Seed for Reproducibility
RANDOM_SEED = 42

# Category Standardization Mapping
CATEGORY_MAP = {
    'beverage': 'Beverages',
    'beverages': 'Beverages',
    'bev': 'Beverages',
    'dairy': 'Dairy',
    'dairy products': 'Dairy',
    'groceries': 'Groceries',
    'grocery': 'Groceries',
    'snacks': 'Snacks',
    'snack': 'Snacks',
    'personal care': 'Personal Care',
    'personal_care': 'Personal Care',
    'household': 'Household',
    'homecare': 'Household',
    'frozen': 'Frozen',
    'frozen foods': 'Frozen'
}

# --- Round 2: ML Demand Forecasting & Feature Engineering Constants ---
AS_OF_DATE = '2026-08-31'
FORECAST_HORIZON = 7

# Time-aware split date ranges
TRAIN_START = '2026-05-15'
TRAIN_END = '2026-07-10'
VAL_GAP_START = '2026-07-11'
VAL_GAP_END = '2026-07-17'
VAL_START = '2026-07-18'
VAL_END = '2026-08-03'
TEST_GAP_START = '2026-08-04'
TEST_GAP_END = '2026-08-10'
TEST_START = '2026-08-11'
TEST_END = '2026-08-24'
UNLABELED_START = '2026-08-25'

# Future Calendar (2026-09-01 to 2026-09-07) for AS_OF live forecast scoring
# 2026-09-01 (Tue), 09-02 (Wed), 09-03 (Thu), 09-04 (Fri - Krishna Janmashtami / regional festival, approx - verify),
# 09-05 (Sat), 09-06 (Sun), 09-07 (Mon)
FUTURE_CALENDAR = {
    '2026-09-01': {'weekend': 0, 'holiday': 0, 'festival': 0, 'local_event': 0},
    '2026-09-02': {'weekend': 0, 'holiday': 0, 'festival': 0, 'local_event': 0},
    '2026-09-03': {'weekend': 0, 'holiday': 0, 'festival': 0, 'local_event': 0},
    '2026-09-04': {'weekend': 0, 'holiday': 0, 'festival': 1, 'local_event': 0}, # approximate - verify
    '2026-09-05': {'weekend': 1, 'holiday': 0, 'festival': 0, 'local_event': 0},
    '2026-09-06': {'weekend': 1, 'holiday': 0, 'festival': 0, 'local_event': 0},
    '2026-09-07': {'weekend': 0, 'holiday': 0, 'festival': 0, 'local_event': 0},
}

# Planned promotions for next 7 days: dict of (store_id, product_id) -> promo_days / discount
# Default empty (no new planned promotions scheduled)
PLANNED_PROMO_NEXT_7 = {}
