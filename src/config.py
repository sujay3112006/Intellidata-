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
