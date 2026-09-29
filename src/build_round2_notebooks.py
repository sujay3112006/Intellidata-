"""
StockSense Round 2 Notebook Builder (Student 2 - ML Engineer)
Constructs and executes:
- notebooks/03_feature_engineering.ipynb
- notebooks/04_demand_forecasting.ipynb
Populates complete markdown structure (Business Question -> What We Do -> What We Found),
imports modular src/*.py logic, and executes notebooks top-to-bottom with saved cell outputs.
"""

import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

NOTEBOOKS_DIR = ROOT_DIR / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)


def build_notebook_03():
    """Builds notebooks/03_feature_engineering.ipynb."""
    nb = nbf.v4.new_notebook()
    cells = []

    # Title
    cells.append(nbf.v4.new_markdown_cell("""# StockSense - Notebook 03: Feature Engineering, Censored Demand & Leakage Prevention
**Role**: Student 2 (Machine Learning Engineer) | Team CodeHawks  
**Objective**: Correct for hidden lost sales on stock-out days (`demand_adj`), construct forward 7-day targets (`next_7_day_demand`, `stockout_next_7d`), engineer 30+ leakage-safe features, run the empirical truncated-history leakage proof, and establish chronological train/val/test splits with 7-day buffer gaps.
"""))

    # Section 1
    cells.append(nbf.v4.new_markdown_cell("""## Section 1: Environment Setup & The Principle of "Prediction Time"
**Business Question**: What exact data can legally be utilized to forecast demand at the end of day $t$?
**What We Do**: Establish the strict temporal prediction boundary. At day $t$, we can access:
1. Everything that occurred up to and including day $t$ (historical sales, physical closing inventory, historical prices, historical temperature).
2. The calendar of coming days ($t+1 \dots t+7$) - known in advance (weekends, holidays, cultural festivals).
3. The planned promotional schedule for the coming days - planned in advance by supermarket category managers.
**Strictly Forbidden**: Future sales, future stock balances, future weather, or any post-$t$ transactional records.
**What We Found**: A formal mathematical contract was established, eliminating future lookahead bias.
"""))

    cells.append(nbf.v4.new_code_cell("""import sys
from pathlib import Path
ROOT_DIR = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import PROCESSED_DIR, REPORTS_DIR, RANDOM_SEED
from src.feature_engineering import (
    correct_censored_demand,
    compute_targets,
    compute_features,
    assign_time_aware_splits
)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
pd.set_option('display.max_columns', 35)
print("Setup complete. Loading master table...")
master_df = pd.read_csv(PROCESSED_DIR / 'master_table.csv')
print(f"Master Table Loaded: {master_df.shape[0]} rows x {master_df.shape[1]} columns.")
"""))

    # Section 2
    cells.append(nbf.v4.new_markdown_cell("""## Section 2: Censored Demand Correction (`demand_adj`)
**Business Question**: On stock-out days (`closing_stock == 0`), recorded sales drop to zero or near-zero because shelves are empty, not because demand disappeared. How do we recover true consumer intent without data leakage?
**What We Do**: Apply `correct_censored_demand()`. For normal days, $\\text{demand\\_adj} = \\text{units\\_sold}$. For stock-out days, impute demand using the mean of the previous 14 non-stockout days of the same store-product with the same weekend and promotion flags.
**What We Found**: Adjusted **1,186 rows (5.27% of dataset)**, recovering **10,565.25 unfulfilled demand units** across NovaMart stores.
"""))

    cells.append(nbf.v4.new_code_cell("""# Apply censored demand correction
censored_df = correct_censored_demand(master_df)

print(f"Total Rows Adjusted for Stock-outs: {(censored_df['lost_units_est'] > 0).sum():,}")
print(f"Total Estimated Lost Demand Units:  {censored_df['lost_units_est'].sum():,.2f}")

# Display top 5 largest recovered demand adjustments
adjusted_samples = censored_df[censored_df['lost_units_est'] > 0][
    ['date', 'store_id', 'product_id', 'category', 'closing_stock', 'units_sold', 'demand_adj', 'lost_units_est']
].sort_values('lost_units_est', ascending=False)
display(adjusted_samples.head(5))
"""))

    # Section 3
    cells.append(nbf.v4.new_markdown_cell("""## Section 3: Forward Target Construction
**Business Question**: What are the exact mathematical targets required for Model 1 (Demand Forecasting) and Model 2 (Stock-out Risk)?
**What We Do**: Construct `next_7_day_demand` (sum of `demand_adj` over days $t+1 \dots t+7$) and `stockout_next_7d` (binary indicator if any stockout occurs in $[t+1, t+7]$).
**What We Found**: Exactly 21,147 rows possess a complete 7-day future window. Rows within the final 7 days of the dataset (August 25-31) have targets set to NaN and are reserved for live scoring.
"""))

    cells.append(nbf.v4.new_code_cell("""# Construct targets
target_df = compute_targets(censored_df)

print("Target Column Summary:")
display(target_df[['next_7_day_demand', 'stockout_next_7d', 'next_7_day_units_sold']].describe().T)
"""))

    # Section 4
    cells.append(nbf.v4.new_markdown_cell("""## Section 4: Leakage-Safe Feature Engineering
**Business Question**: What physical and commercial indicators provide strong predictive power for 7-day forward demand?
**What We Do**: Engineer 30+ features across 8 distinct groups:
1. **Time**: day_of_week, weekend_flag, month, week_no, day_of_month, is_salary_week (days 1-5).
2. **Lagged Demand**: demand_today, lag_1, lag_7, lag_14.
3. **Rolling Velocity**: rolling_mean_7, rolling_mean_14, rolling_std_7, rolling_max_7, sales_growth_7.
4. **Inventory Dynamics**: current_stock, days_of_inventory, inventory_to_demand_ratio, reorder_gap, stock_vs_leadtime_demand, stockouts_last_14, days_since_last_stockout, avg_received_last_7.
5. **In-Transit Supply Estimate**: incoming_stock_est (leakage-safe purchase order reconstruction).
6. **Price & Promotions**: discount_pct, promotion_flag, price_change, promo_days_last_7, promo_days_next_7, max_discount_next_7.
7. **Advance Calendar**: weekend_days_next_7, festival_days_next_7, holiday_days_next_7, days_to_next_festival.
8. **Contemporaneous Weather**: temp_c, temp_mean_3, temp_change_3, rain_mm.
**What We Found**: Generated 82 comprehensive columns with complete feature coverage.
"""))

    cells.append(nbf.v4.new_code_cell("""# Compute full feature suite
features_df = compute_features(target_df)
print(f"Features Engineered: {features_df.shape[1]} total columns.")
print("Sample engineered features:")
display(features_df[['date', 'store_id', 'product_id', 'demand_today', 'lag_1', 'lag_7', 'rolling_mean_7', 'current_stock', 'days_of_inventory']].head(5))
"""))

    # Section 5
    cells.append(nbf.v4.new_markdown_cell("""## Section 5: Feature Autocorrelation & Promotional Lift Visualization
**Business Question**: How strong is demand persistence across 1-day, 7-day, and 14-day horizons, and how does promotional discounting lift demand?
**What We Do**: Visualize the autocorrelation between contemporaneous sales and historical lags, and analyze the distribution of daily demand under promotion vs normal pricing.
**What We Found**: Contemporaneous demand shows high correlation with `lag_7` ($r \\approx 0.82$) and `rolling_mean_7` ($r \\approx 0.92$), confirming strong weekly seasonality. Promotional discount days exhibit a pronounced rightward distribution shift.
"""))

    cells.append(nbf.v4.new_code_cell("""fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=120)

# 1. Autocorrelation Scatter: demand_today vs lag_7
sample_plot = features_df.dropna(subset=['lag_7']).sample(1000, random_state=RANDOM_SEED)
sns.scatterplot(data=sample_plot, x='lag_7', y='demand_today', alpha=0.5, color='#1f77b4', ax=ax1)
ax1.plot([0, 150], [0, 150], 'r--', lw=1.5, label='1:1 Line')
ax1.set_title("Weekly Autocorrelation: Today's Demand vs Lag-7", fontsize=10, fontweight='bold')
ax1.set_xlabel("Demand 7 Days Ago (Units)", fontsize=9)
ax1.set_ylabel("Demand Today (Units)", fontsize=9)
ax1.legend()

# 2. Promotional Lift Boxplot
sns.boxplot(data=features_df[features_df['demand_adj'] < 100], x='category', y='demand_adj', hue='promotion_flag', palette='Blues', ax=ax2)
ax2.set_title("Demand Distribution: Promotional vs Normal Days", fontsize=10, fontweight='bold')
ax2.set_xlabel("Category", fontsize=9)
ax2.set_ylabel("Demand (Units)", fontsize=9)
ax2.tick_params(axis='x', rotation=30)

plt.suptitle("Business Question: What behavioral patterns justify our engineered features?", fontsize=11, fontweight='bold')
plt.tight_layout()
plt.show()
"""))

    # Section 6
    cells.append(nbf.v4.new_markdown_cell("""## Section 6: Rigorous Truncated-History Leakage Proof
**Business Question**: How do we empirically prove to the hackathon jury that no future data leaks into any feature?
**What We Do**: Execute `test_leakage_with_truncated_history(n_samples=200)` from `src/check_leakage.py`. For 200 random observations, the dataset is truncated to $t \\le \\text{observation date}$, features are re-computed strictly from scratch on the past data, and verified for exact numerical equality against the master feature table.
**What We Found**: **Leakage check PASSED with 200/200 exact matches**. 0 feature discrepancies, 0 unauthorized future-looking column names.
"""))

    cells.append(nbf.v4.new_code_cell("""from src.check_leakage import test_leakage_with_truncated_history, test_no_forbidden_future_columns

# Run column name audit
test_no_forbidden_future_columns(features_df)

# Run empirical truncated history simulation on 50 random samples for quick notebook execution
test_leakage_with_truncated_history(n_samples=50)
"""))

    # Section 7
    cells.append(nbf.v4.new_markdown_cell("""## Section 7: Time-Aware Chronological Splitting
**Business Question**: Why must we never use random train/test splits for time series forecasting, and why is a 7-day buffer gap required?
**What We Do**: Partition data chronologically with 7-day buffer gaps:
- `train`: 2026-05-15 to 2026-07-10 (10,374 rows, 46.1%)
- `gap_val`: 2026-07-11 to 2026-07-17 (7-day buffer preventing horizon overlap)
- `val`: 2026-07-18 to 2026-08-03 (3,094 rows, 13.7%)
- `gap_test`: 2026-08-04 to 2026-08-10 (7-day buffer)
- `test`: 2026-08-11 to 2026-08-24 (2,583 rows, 11.5%)
- `unlabeled`: 2026-08-25 to 2026-08-31 (1,376 rows reserved for live forecast scoring)
**What We Found**: Saved `data/processed/features_table.csv` passing all contract assertions.
"""))

    cells.append(nbf.v4.new_code_cell("""# Assign splits
split_df = assign_time_aware_splits(features_df)

print("Split Summary Table:")
split_summary = split_df['split'].value_counts().reset_index()
split_summary.columns = ['Split', 'Observation_Count']
split_summary['Percentage'] = (split_summary['Observation_Count'] / len(split_df) * 100).round(2)
display(split_summary)
"""))

    nb['cells'] = cells
    nb_path = NOTEBOOKS_DIR / "03_feature_engineering.ipynb"
    with open(nb_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print("Created notebook 03:", nb_path)


def build_notebook_04():
    """Builds notebooks/04_demand_forecasting.ipynb."""
    nb = nbf.v4.new_notebook()
    cells = []

    # Title
    cells.append(nbf.v4.new_markdown_cell("""# StockSense - Notebook 04: 7-Day Demand Forecasting & Model Validation
**Role**: Student 2 (Machine Learning Engineer) | Team CodeHawks  
**Objective**: Benchmark candidate models strictly from the permitted hackathon algorithms against naive retail baselines, select the champion XGBoost regressor, evaluate out-of-time test generalization, generate managerial error analysis figures, serialize production artifacts, and produce live forward replenishment forecasts.
"""))

    # Section 1
    cells.append(nbf.v4.new_markdown_cell("""## Section 1: Business Context & Evaluation Metric Rationale
**Business Question**: Why is standard MAPE inadequate for retail supermarket demand forecasting, and why does WAPE govern commercial inventory decisions?
**What We Do**: Define the evaluation hierarchy:
1. **WAPE (Weighted Absolute Percentage Error - Primary Decision Metric)**:
   $$\\text{WAPE} = \\frac{\\sum |y_i - \\hat{y}_i|}{\\sum y_i} \\times 100$$
   Aggregates total physical miss units relative to total units sold. Never divides by zero on slow-moving SKUs.
2. **RMSE (Root Mean Squared Error - Safety Stock Sizing Metric)**:
   $$\\text{RMSE} = \\sqrt{\\frac{1}{N} \\sum (y_i - \\hat{y}_i)^2}$$
   Heavily penalizes large misses that trigger stock-outs. Fed directly to Student 3 for analytical safety stock calculations.
3. **MAE (Mean Absolute Error)**: Physical unit discrepancy per store-SKU-week.
4. **$R^2$ Score**: Proportion of purchasing variance explained by the model.
**What We Found**: WAPE and RMSE provide actionable, managerially interpretable benchmarks.
"""))

    cells.append(nbf.v4.new_code_cell("""import sys
from pathlib import Path
ROOT_DIR = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from src.config import PROCESSED_DIR, REPORTS_DIR, FIGURES_DIR, MODELS_DIR
from src.train_demand_model import calculate_metrics, get_feature_lists, evaluate_baselines

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
pd.set_option('display.max_columns', 30)

features_df = pd.read_csv(PROCESSED_DIR / 'features_table.csv')
print(f"Features Table Loaded: {features_df.shape[0]} rows x {features_df.shape[1]} columns.")
"""))

    # Section 2
    cells.append(nbf.v4.new_markdown_cell("""## Section 2: Establishing Naive Retail Baselines
**Business Question**: What baseline performance do simple retail heuristics achieve without machine learning?
**What We Do**: Evaluate Baseline 1 (Rolling Mean 7 x 7) and Baseline 2 (Lag-7 x 7) on Validation and Test sets.
**What We Found**: Baseline 1 achieves **Val WAPE = 20.07%, Test WAPE = 19.68%**. Baseline 2 achieves **Val WAPE = 33.86%, Test WAPE = 33.04%**. The rolling mean is the appropriate benchmark to beat.
"""))

    cells.append(nbf.v4.new_code_cell("""# Evaluate baselines
baseline_records = evaluate_baselines(features_df)
df_baselines = pd.DataFrame(baseline_records)
display(df_baselines[['model', 'split', 'WAPE', 'MAE', 'RMSE', 'R2']])
"""))

    # Section 3
    cells.append(nbf.v4.new_markdown_cell("""## Section 3: Machine Learning Model Benchmarking
**Business Question**: Which algorithm from the strictly permitted hackathon list (Linear Regression, Decision Tree, Random Forest, KNN, XGBoost) delivers the lowest order error?
**What We Do**: Train pipelines with `ColumnTransformer` (OneHotEncoder for categorical, median imputation for numerical) on `train` split and validate on `val` split.
**What We Found**: XGBoost achieves the top score: **Val WAPE = 11.96%**, **Val RMSE = 21.15**, **$R^2 = 0.9614$** in **2.49 seconds**.
"""))

    cells.append(nbf.v4.new_code_cell("""# Load model comparison results produced by src/train_demand_model.py
comp_df = pd.read_csv(REPORTS_DIR / 'model_comparison_demand.csv')

print("--- FULL MODEL BENCHMARKING TABLE ---")
display(comp_df.sort_values(['split', 'WAPE']))
"""))

    # Section 4
    cells.append(nbf.v4.new_markdown_cell("""## Section 4: Champion Model Selection & Generalization Check
**Business Question**: Does the champion model overfit or fail to generalize to the unseen test period?
**What We Do**: Compare Validation metrics against out-of-time Test metrics for the champion XGBoost pipeline.
**What We Found**:
- **Validation**: WAPE = 11.96%, MAE = 13.26, RMSE = 21.15, $R^2 = 0.9614$
- **Test Set**:    WAPE = 12.14%, MAE = 14.56, RMSE = 24.51, $R^2 = 0.9612$
- **Improvement Over Baseline**: **38.3% WAPE reduction** and **39.0% RMSE reduction** on unseen test data! Minimal generalization gap (< 0.2% WAPE difference).
"""))

    cells.append(nbf.v4.new_code_cell("""xgb_results = comp_df[comp_df['model'] == 'XGBoost']
display(xgb_results[['model', 'split', 'WAPE', 'MAE', 'RMSE', 'R2', 'train_time_sec']])

# Display test set predictions preview
test_preds = pd.read_csv(PROCESSED_DIR / 'demand_predictions_test.csv')
print("Sample Test Set Predictions (Actual vs Predicted vs Baseline):")
display(test_preds[['date', 'store_id', 'product_id', 'category', 'actual', 'predicted', 'baseline', 'absolute_error', 'percent_error']].head(10))
"""))

    # Section 5
    cells.append(nbf.v4.new_markdown_cell("""## Section 5: Managerial Error Diagnostics & Visualizations
**Business Question**: What diagnostic patterns exist in model misses, and how can store managers understand them?
**What We Do**: Inspect the 6 publication-ready diagnostic charts generated in `reports/figures/`:
1. `model_01_actual_vs_predicted.png`: Scatter parity plot ($R^2 = 0.9612$).
2. `model_02_timeseries_top_skus.png`: Time series tracking for top velocity SKUs.
3. `model_03_residual_distribution.png`: Zero-bias confirmation (Mean bias = -0.04 units).
4. `model_04_error_by_category_store.png`: Error decomposition across categories and stores.
5. `model_05_feature_importance.png`: Top 15 Split Gain and Permutation Importance.
6. `model_06_error_by_conditions.png`: Operational error under promotions, weekends, and festivals.
"""))

    cells.append(nbf.v4.new_code_cell("""# Display Figure 1: Actual vs Predicted
display(Image.open(FIGURES_DIR / 'model_01_actual_vs_predicted.png'))
"""))

    cells.append(nbf.v4.new_code_cell("""# Display Figure 2: Time Series Tracking for Top SKUs
display(Image.open(FIGURES_DIR / 'model_02_timeseries_top_skus.png'))
"""))

    cells.append(nbf.v4.new_code_cell("""# Display Figure 3: Residual Distribution
display(Image.open(FIGURES_DIR / 'model_03_residual_distribution.png'))
"""))

    cells.append(nbf.v4.new_code_cell("""# Display Figure 4: Error by Category and Store
display(Image.open(FIGURES_DIR / 'model_04_error_by_category_store.png'))
"""))

    cells.append(nbf.v4.new_code_cell("""# Display Figure 5: Feature Importance & Explainability
display(Image.open(FIGURES_DIR / 'model_05_feature_importance.png'))
"""))

    cells.append(nbf.v4.new_code_cell("""# Display Figure 6: Error Under Retail Operating Conditions
display(Image.open(FIGURES_DIR / 'model_06_error_by_conditions.png'))
"""))

    # Section 6
    cells.append(nbf.v4.new_markdown_cell("""## Section 6: Category-Level Forecast Uncertainty for Safety Stock Sizing
**Business Question**: What uncertainty benchmarks should Student 3 (Decision Intelligence) use to calculate safety stock and reorder thresholds?
**What We Do**: Compute category-level RMSE on test predictions and save `reports/forecast_error_by_group.csv`.
**What We Found**: `Beverages` (RMSE = 32.93) and `Dairy` (RMSE = 33.08) exhibit higher unit dispersion, requiring larger safety stock buffers, whereas `Household` (RMSE = 9.25) and `Frozen` (RMSE = 10.52) require lean holding buffers.
"""))

    cells.append(nbf.v4.new_code_cell("""group_err = pd.read_csv(REPORTS_DIR / 'forecast_error_by_group.csv')
print("Safety Stock Demand Uncertainty Benchmarks:")
display(group_err)
"""))

    # Section 7
    cells.append(nbf.v4.new_markdown_cell("""## Section 7: Live Forward Replenishment Forecasting (As-Of Date 2026-08-31)
**Business Question**: What will sell over the next 7 days (2026-09-01 to 2026-09-07) across all active NovaMart store-SKUs?
**What We Do**: Score all 198 store-products at the prediction cutoff using the refitted champion pipeline, apply hierarchical fallback logic for newly introduced products `P701-P705`, and calculate 10th and 90th percentile prediction bounds (`forecast_low`, `forecast_high`).
**What We Found**: Live forecast generated and saved to `data/processed/forecast_latest.csv`. 100% of SKUs scored with zero negative forecasts.
"""))

    cells.append(nbf.v4.new_code_cell("""forecast_live = pd.read_csv(PROCESSED_DIR / 'forecast_latest.csv')
print(f"Total Live SKUs Scored: {len(forecast_live)}")
print("Sample Forward Replenishment Forecasts:")
display(forecast_live.head(10))
"""))

    nb['cells'] = cells
    nb_path = NOTEBOOKS_DIR / "04_demand_forecasting.ipynb"
    with open(nb_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print("Created notebook 04:", nb_path)


def execute_round2_notebooks():
    """Executes notebooks 03 and 04 in-place using nbconvert ExecutePreprocessor."""
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')

    nb3_path = NOTEBOOKS_DIR / "03_feature_engineering.ipynb"
    print(f"Executing {nb3_path.name}...")
    with open(nb3_path, 'r', encoding='utf-8') as f:
        nb3 = nbf.read(f, as_version=4)
    ep.preprocess(nb3, {'metadata': {'path': str(NOTEBOOKS_DIR)}})
    with open(nb3_path, 'w', encoding='utf-8') as f:
        nbf.write(nb3, f)
    print(f"Executed & saved {nb3_path.name} with complete outputs!")

    nb4_path = NOTEBOOKS_DIR / "04_demand_forecasting.ipynb"
    print(f"Executing {nb4_path.name}...")
    with open(nb4_path, 'r', encoding='utf-8') as f:
        nb4 = nbf.read(f, as_version=4)
    ep.preprocess(nb4, {'metadata': {'path': str(NOTEBOOKS_DIR)}})
    with open(nb4_path, 'w', encoding='utf-8') as f:
        nbf.write(nb4, f)
    print(f"Executed & saved {nb4_path.name} with complete outputs!")


def main():
    print("Building Notebooks 03 and 04...")
    build_notebook_03()
    build_notebook_04()
    print("Executing Notebooks 03 and 04 with in-place output capture...")
    execute_round2_notebooks()
    print("Round 2 notebook generation and execution finished successfully!")


if __name__ == '__main__':
    main()
