"""
StockSense KPI Calculation Module
Implements standard retail and supply chain Key Performance Indicators (KPIs)
strictly adhering to the hackathon specification formulas.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import numpy as np
from src.config import REPORTS_DIR, PROCESSED_DIR

def calculate_revenue(df: pd.DataFrame) -> float:
    """Total Revenue = Sum(quantity x selling_price)."""
    return float(df['revenue'].sum())

def calculate_units_sold(df: pd.DataFrame) -> int:
    """Total Units Sold = Sum(quantity)."""
    return int(df['units_sold'].sum())

def calculate_stockout_rate(df: pd.DataFrame) -> float:
    """Stock-out Rate = stock-out events / total product-store observations."""
    return float(df['stockout_flag_day'].mean())

def calculate_inventory_turnover(df: pd.DataFrame) -> float:
    """
    Inventory Turnover = Cost of Goods Sold (COGS) / Average Inventory Value.
    COGS = Sum(units_sold x cost_price)
    Average Inventory Value = Mean(closing_stock x cost_price) per active SKU
    """
    cogs = (df['units_sold'] * df['cost_price']).sum()
    avg_inv_val = (df['closing_stock'] * df['cost_price']).mean() * df[['store_id', 'product_id']].drop_duplicates().shape[0]
    if avg_inv_val == 0:
        return 0.0
    return float(cogs / avg_inv_val)

def calculate_days_of_inventory(df: pd.DataFrame) -> float:
    """
    Days of Inventory = Closing Stock / Average Daily Demand.
    Uses mean closing stock divided by average daily demand per store-product.
    """
    num_days = df['date'].nunique()
    if num_days == 0:
        return 0.0
    avg_daily_demand = df['units_sold'].sum() / num_days
    mean_closing_stock = df['closing_stock'].mean() * df[['store_id', 'product_id']].drop_duplicates().shape[0]
    if avg_daily_demand == 0:
        return 0.0
    return float(mean_closing_stock / avg_daily_demand)

def calculate_promotion_lift(df: pd.DataFrame) -> float:
    """
    Promotion Lift = (promotional daily sales - normal daily sales) / normal daily sales.
    Normalized per store-product to eliminate product mix bias.
    """
    promo_df = df[df['promotion_flag'] == 1]
    non_promo_df = df[df['promotion_flag'] == 0]

    promo_avg = promo_df.groupby(['store_id', 'product_id'])['units_sold'].mean()
    non_promo_avg = non_promo_df.groupby(['store_id', 'product_id'])['units_sold'].mean()

    aligned = pd.DataFrame({'promo': promo_avg, 'non_promo': non_promo_avg}).dropna()
    aligned = aligned[aligned['non_promo'] > 0]

    if len(aligned) == 0:
        return 0.0

    mean_promo = aligned['promo'].mean()
    mean_non_promo = aligned['non_promo'].mean()
    return float((mean_promo - mean_non_promo) / mean_non_promo)

def kpi_summary(df: pd.DataFrame) -> dict:
    """Computes overall business KPI summary dictionary."""
    return {
        'Total Revenue (INR)': calculate_revenue(df),
        'Total Units Sold': calculate_units_sold(df),
        'Stock-out Rate (%)': calculate_stockout_rate(df) * 100.0,
        'Inventory Turnover': calculate_inventory_turnover(df),
        'Days of Inventory (DOI)': calculate_days_of_inventory(df),
        'Promotion Lift (%)': calculate_promotion_lift(df) * 100.0
    }

def kpi_summary_by_group(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Computes KPIs grouped by store_id or category."""
    results = []
    for grp_val, grp_df in df.groupby(group_col):
        res = {'Group': grp_val}
        res.update(kpi_summary(grp_df))
        results.append(res)
    return pd.DataFrame(results)

def main():
    """Calculates and exports KPI summary reports."""
    master_path = PROCESSED_DIR / 'master_table.csv'
    if not master_path.exists():
        print("master_table.csv not found!")
        return

    df = pd.read_csv(master_path)
    overall_kpis = kpi_summary(df)

    print("=== OVERALL STOCKSENSE KPI SUMMARY ===")
    for k, v in overall_kpis.items():
        if 'Revenue' in k:
            print(f"{k}: Rs. {v:,.2f}")
        elif 'Rate' in k or 'Lift' in k:
            print(f"{k}: {v:.2f}%")
        elif 'Turnover' in k or 'DOI' in k:
            print(f"{k}: {v:.2f}")
        else:
            print(f"{k}: {v:,}")

    # Save overall summary CSV
    kpi_df = pd.DataFrame([overall_kpis])
    out_path = REPORTS_DIR / 'kpi_summary.csv'
    kpi_df.to_csv(out_path, index=False)
    print("Saved overall KPI summary to:", out_path)

    # Save store and category breakdown CSVs
    store_kpi = kpi_summary_by_group(df, 'store_id')
    store_kpi.to_csv(REPORTS_DIR / 'kpi_summary_by_store.csv', index=False)
    
    cat_kpi = kpi_summary_by_group(df, 'category')
    cat_kpi.to_csv(REPORTS_DIR / 'kpi_summary_by_category.csv', index=False)
    print("Saved store and category KPI breakdowns.")

if __name__ == '__main__':
    main()
