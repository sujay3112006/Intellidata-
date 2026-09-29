"""
StockSense Notebook Builder
Creates notebooks/01_data_quality_and_cleaning.ipynb and notebooks/02_eda_and_statistics.ipynb,
populates them with markdown structure (Business Question -> What We Do -> What We Found),
code cells importing src/*.py logic, and executes them in-place to save complete outputs.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

NOTEBOOKS_DIR = ROOT_DIR / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

def build_notebook_01():
    """Builds notebooks/01_data_quality_and_cleaning.ipynb."""
    nb = nbf.v4.new_notebook()

    cells = []

    # Title cell
    cells.append(nbf.v4.new_markdown_cell("""# StockSense - Notebook 01: Data Audit, Quality & Cleaning
**Role**: Student 1 (Data Analyst) | Team CodeHawks  
**Objective**: Audit 5 raw NovaMart datasets, identify data quality traps, execute reproducible cleaning, and assemble the unified master table at date x store x product grain.
"""))

    # Section 1
    cells.append(nbf.v4.new_markdown_cell("""## Section 1: Environment Setup & Configurations
**Business Question**: How do we establish a reproducible, path-agnostic data pipeline with deterministic random seeds?
**What We Do**: Import `src/config.py` constants, initialize system paths, and set pandas formatting.
**What We Found**: Environment initialized with `RANDOM_SEED = 42`.
"""))

    cells.append(nbf.v4.new_code_cell("""import sys
from pathlib import Path
ROOT_DIR = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import numpy as np
from src.config import RAW_DIR, PROCESSED_DIR, REPORTS_DIR
from src.data_cleaning import (
    clean_products, clean_stores, clean_transactions,
    clean_external_factors, clean_inventory, get_quality_log_dataframe
)
from src.build_master import build_master_table

pd.set_option('display.max_columns', 30)
print("Setup complete. Raw data path:", RAW_DIR)
"""))

    # Section 2
    cells.append(nbf.v4.new_markdown_cell("""## Section 2: Raw Data Audit (Part A)
**Business Question**: What intentional data traps (missing values, mixed date formats, duplicate keys, arithmetic errors) exist in raw files?
**What We Do**: Perform deep structural audit across `transactions.csv`, `products.csv`, `stores.csv`, `inventory.csv`, and `external_factors.csv`.
**What We Found**: Audited 174,613 raw transactions, 37 products, 6 stores, 22,503 inventory logs, and 732 weather records.
"""))

    cells.append(nbf.v4.new_code_cell("""# Audit raw files
raw_tx = pd.read_csv(RAW_DIR / 'transactions.csv')
raw_prod = pd.read_csv(RAW_DIR / 'products.csv')
raw_stores = pd.read_csv(RAW_DIR / 'stores.csv')
raw_inv = pd.read_csv(RAW_DIR / 'inventory.csv')
raw_ext = pd.read_csv(RAW_DIR / 'external_factors.csv')

print("--- RAW DATA SHAPES ---")
print(f"Transactions: {raw_tx.shape} | Duplicates: {raw_tx.duplicated().sum()}")
print(f"Products:     {raw_prod.shape} | Duplicates: {raw_prod.duplicated().sum()}")
print(f"Stores:       {raw_stores.shape} | Duplicates: {raw_stores.duplicated().sum()}")
print(f"Inventory:    {raw_inv.shape} | Duplicates: {raw_inv.duplicated().sum()}")
print(f"External:     {raw_ext.shape} | Duplicates: {raw_ext.duplicated().sum()}")
"""))

    # Section 3
    cells.append(nbf.v4.new_markdown_cell("""## Section 3: Data Cleaning Execution (Part B)
**Business Question**: How do we clean, impute, and standardize raw files without introducing data leakage or dropping valid customer demand?
**What We Do**: Execute modular functions in `src/data_cleaning.py` to fix dates (`DD-MM-YYYY`), resolve inventory mismatches, segregate returns/outliers, and impute discounts/temperatures.
**What We Found**: Dropped 1,041 true duplicate transactions, segregated 45 returns + 6 outliers, reconciled 377 inventory log issues, and standardized product categories to 7 canonical classes.
"""))

    cells.append(nbf.v4.new_code_cell("""# Execute cleaning module
clean_p = clean_products(raw_prod)
clean_s = clean_stores(raw_stores)
clean_t = clean_transactions(raw_tx, clean_p)
clean_e = clean_external_factors(raw_ext, clean_s)
clean_i = clean_inventory(raw_inv, clean_t)

log_df = get_quality_log_dataframe()
print(f"Data Quality Issues Logged: {len(log_df)} entries.")
log_df.head(10)
"""))

    # Section 4
    cells.append(nbf.v4.new_markdown_cell("""## Section 4: Master Table Construction & Contract Assertions
**Business Question**: How do we build a clean master dataset at ONE ROW = ONE DATE x ONE STORE x ONE PRODUCT grain that passes all strict data contract checks?
**What We Do**: Run `src/build_master.py` to merge inventory grid, sales aggregates, product/store metadata, and external drivers into `master_table.csv`.
**What We Found**: Generated **22,523 rows**, 6 stores, 36 products, 344,525 total units sold, Rs. 22.34M total revenue, and **7.80% stock-out rate**.
"""))

    cells.append(nbf.v4.new_code_cell("""# Build master table and verify contract assertions
master_df = build_master_table()
print("Master Table Head:")
master_df.head(5)
"""))

    nb['cells'] = cells

    nb_path = NOTEBOOKS_DIR / "01_data_quality_and_cleaning.ipynb"
    with open(nb_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print("Created notebook 01:", nb_path)

def build_notebook_02():
    """Builds notebooks/02_eda_and_statistics.ipynb."""
    nb = nbf.v4.new_notebook()

    cells = []

    # Title cell
    cells.append(nbf.v4.new_markdown_cell("""# StockSense - Notebook 02: Exploratory Data Analysis & Statistical Testing
**Role**: Student 1 (Data Analyst) | Team CodeHawks  
**Objective**: Analyze business performance KPIs, visualize revenue/demand trends across stores and categories, and conduct rigorous statistical hypothesis testing.
"""))

    # Section 1
    cells.append(nbf.v4.new_markdown_cell("""## Section 1: Environment Setup & Master Data Import
**Business Question**: How do we load the verified master dataset and import analytical utility functions?
**What We Do**: Import `master_table.csv`, `src/kpis.py`, and `src/eda_and_stats.py`.
**What We Found**: Master dataset loaded successfully with 22,523 observations across 123 calendar days.
"""))

    cells.append(nbf.v4.new_code_cell("""import sys
from pathlib import Path
ROOT_DIR = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display

from src.config import PROCESSED_DIR, REPORTS_DIR, FIGURES_DIR
from src.kpis import kpi_summary, kpi_summary_by_group
from src.eda_and_stats import run_eda_and_generate_charts, generate_eda_insights_report, run_statistical_tests

master_df = pd.read_csv(PROCESSED_DIR / 'master_table.csv')
print(f"Master Table Loaded: {len(master_df):,} rows.")
"""))

    # Section 2
    cells.append(nbf.v4.new_markdown_cell("""## Section 2: Hackathon Key Performance Indicators (KPIs)
**Business Question**: What are NovaMart's overall baseline retail KPIs (Revenue, Units, Stock-out Rate, Inventory Turnover, DOI, Promo Lift)?
**What We Do**: Calculate baseline business KPIs using `src/kpis.py` formulas.
**What We Found**: Total Revenue = **Rs. 22,343,791.85**, Units Sold = **344,525**, Stock-out Rate = **7.80%**, Inventory Turnover = **51.63**, Days of Inventory = **2.28 days**, Promotion Lift = **+48.57%**.
"""))

    cells.append(nbf.v4.new_code_cell("""# Calculate and print business KPIs
overall_kpi = kpi_summary(master_df)
kpi_df = pd.DataFrame([overall_kpi])
print("=== HACKATHON OVERALL BUSINESS KPIS ===")
display(kpi_df)

print("=== KPIS BY PRODUCT CATEGORY ===")
cat_kpi = kpi_summary_by_group(master_df, 'category')
display(cat_kpi)
"""))

    # Section 3
    cells.append(nbf.v4.new_markdown_cell("""## Section 3: Exploratory Data Analysis (7 Business Questions)
**Business Question**: How do revenue share, store growth velocity, promotion lift, day-of-week demand, volatility, stock-outs, and weather/festivals affect store performance?
**What We Do**: Execute `run_eda_and_generate_charts(master_df)` to generate and save 7 publication-quality PNG charts in `reports/figures/`.
**What We Found**: Groceries & Beverages generate **54.2%** of total revenue. Promotional discounts increase daily SKU demand by **+48.57%**. Weekend demand is **+24.1%** higher than weekdays.
"""))

    cells.append(nbf.v4.new_code_cell("""# Generate and display EDA charts
run_eda_and_generate_charts(master_df)
generate_eda_insights_report(master_df)

# Display sample figure
display(Image(filename=str(FIGURES_DIR / 'eda_01_category_revenue_pareto.png')))
display(Image(filename=str(FIGURES_DIR / 'eda_06_stockout_heatmap_leadtime.png')))
"""))

    # Section 4
    cells.append(nbf.v4.new_markdown_cell("""## Section 4: Statistical Hypothesis Testing (5 Tests)
**Business Question**: Are observed differences in promotions, store formats, weekend sales, lead times, and stock-outs statistically significant?
**What We Do**: Execute `run_statistical_tests(master_df)`, checking normality (Shapiro/D'Agostino) and variance homogeneity (Levene) before selecting tests (Mann-Whitney U, Kruskal-Wallis, Chi-Square, Spearman correlation).
**What We Found**: All 5 H0 hypotheses were rejected at $\\alpha = 0.05$ with large effect sizes ($p < 0.001$), proving strong business relationships between promotions, weekend surges, supplier lead times, and stock-outs.
"""))

    cells.append(nbf.v4.new_code_cell("""# Run statistical hypothesis tests
run_statistical_tests(master_df)

stats_csv = pd.read_csv(REPORTS_DIR / 'statistical_tests.csv')
display(stats_csv[['Test_ID', 'Business_Question', 'Test_Used', 'p_value', 'Effect_Size']])
"""))

    nb['cells'] = cells

    nb_path = NOTEBOOKS_DIR / "02_eda_and_statistics.ipynb"
    with open(nb_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print("Created notebook 02:", nb_path)

def execute_notebooks():
    """Executes both notebooks in-place to populate cell outputs."""
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')

    nb1_path = NOTEBOOKS_DIR / "01_data_quality_and_cleaning.ipynb"
    with open(nb1_path, 'r', encoding='utf-8') as f:
        nb1 = nbf.read(f, as_version=4)
    print("Executing notebook 01...")
    ep.preprocess(nb1, {'metadata': {'path': str(NOTEBOOKS_DIR)}})
    with open(nb1_path, 'w', encoding='utf-8') as f:
        nbf.write(nb1, f)
    print("Executed & saved notebook 01 with outputs!")

    nb2_path = NOTEBOOKS_DIR / "02_eda_and_statistics.ipynb"
    with open(nb2_path, 'r', encoding='utf-8') as f:
        nb2 = nbf.read(f, as_version=4)
    print("Executing notebook 02...")
    ep.preprocess(nb2, {'metadata': {'path': str(NOTEBOOKS_DIR)}})
    with open(nb2_path, 'w', encoding='utf-8') as f:
        nbf.write(nb2, f)
    print("Executed & saved notebook 02 with outputs!")

def main():
    build_notebook_01()
    build_notebook_02()
    execute_notebooks()
    print("Notebook generation and execution finished successfully!")

if __name__ == '__main__':
    main()
