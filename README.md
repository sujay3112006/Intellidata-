# StockSense - Intellidata 2026 Hackathon

## Team Information
- **Team Name**: CodeHawks
- **Institution**: Sri Eshwar College of Engineering
- **Event**: IntelliData 2026 Data Science Hackathon ("StockSense")

## Business Problem
NovaMart Retail Pvt. Ltd. suffers from stock-outs causing lost revenue and overstocking locking capital and causing inventory expiry. StockSense delivers an end-to-end Machine Learning and Decision Intelligence pipeline to forecast 7-day demand, predict stock-out risk, and generate automated replenishment recommendations.

## Team Roles & Ownership
| Role | Student / Responsibilities |
|---|---|
| **Data Analyst** (Student 1) | Data understanding, cleaning, integration, master table build, Data Quality Report, EDA, statistics, KPI module |
| **ML Engineer** (Student 2) | Feature engineering, time-aware splitting, demand forecasting, error analysis, live 7-day forecast |
| **Decision Intelligence** (Student 3) | Operational stock-out definition, classification model, explainability (SHAP/Permutation), manager recommendation engine, Streamlit dashboard, pitch |

## Repository Structure
```
STOCKSENSE_CODEHAWKS/
  data/
    raw/            <- original raw CSV files (never edited)
    processed/      <- cleaned master table & feature matrices
  notebooks/        <- numbered Jupyter notebooks for reproducible analysis
  src/              <- reusable Python modules and automated pipeline scripts
  models/           <- serialized ML models and feature metadata
  dashboard/        <- interactive Streamlit web dashboard
  reports/          <- data quality, statistical reports, and PNG charts
    figures/        <- EDA and model diagnostic charts
  README.md
  requirements.txt
```

## Setup & Running Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/sujay3112006/Intellidata-.git
   cd STOCKSENSE_CODEHAWKS
   ```
2. Create and activate a Python virtual environment:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/Mac:
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Build master table and run analysis:
   ```bash
   python src/build_master.py
   python src/generate_reports.py
   python src/kpis.py
   python src/eda_and_stats.py
   python src/build_notebooks.py
   ```

---

## Round 1 Summary (Student 1 - Data Analyst)
- **Datasets Cleaned & Merged**: 5 raw files audited and cleaned across 123 calendar days (2026-05-01 to 2026-08-31) covering 6 stores and 36 products.
- **Master Table Grain**: `data/processed/master_table.csv` at **ONE ROW = ONE DATE x ONE STORE x ONE PRODUCT** (22,523 total rows).
- **Data Quality Score**: **99.2% Clean Data Integrity Index** (1,041 true duplicates dropped, 51 return/outlier transactions segregated, 377 inventory log mismatches reconciled).
- **Key Business Metrics**:
  - **Total Revenue**: Rs. 22,343,791.85
  - **Total Units Sold**: 344,525 units
  - **Stock-out Rate**: 7.80% across all store-day observations
  - **Inventory Turnover**: 51.63
  - **Days of Inventory (DOI)**: 2.28 days
  - **Promotion Lift**: +48.57% daily sales volume increase
- **Statistical Tests**: 5 hypothesis tests executed ($\alpha = 0.05$, all $p < 0.001$), confirming statistically significant promotion lift, store format volume variance, weekend sales surges, and lead-time stock-out risks.
- **Notebooks**: `notebooks/01_data_quality_and_cleaning.ipynb` and `notebooks/02_eda_and_statistics.ipynb` fully executed with saved outputs.
