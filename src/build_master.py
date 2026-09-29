"""
StockSense Master Table Builder
Integrates cleaned transactions, inventory, product, store, and external datasets
into a unified master dataset at ONE ROW = ONE DATE x ONE STORE x ONE PRODUCT grain.
"""

import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import numpy as np

from src.config import (
    RAW_DIR, PROCESSED_DIR, REPORTS_DIR,
    RANDOM_SEED
)
from src.data_cleaning import (
    clean_products, clean_stores, clean_transactions,
    clean_external_factors, clean_inventory,
    get_quality_log_dataframe
)

def build_master_table() -> pd.DataFrame:
    """Builds and validates the master table from raw datasets."""
    print("Reading raw datasets from:", RAW_DIR)

    # Load raw CSVs
    raw_tx = pd.read_csv(RAW_DIR / 'transactions.csv')
    raw_prod = pd.read_csv(RAW_DIR / 'products.csv')
    raw_stores = pd.read_csv(RAW_DIR / 'stores.csv')
    raw_inv = pd.read_csv(RAW_DIR / 'inventory.csv')
    raw_ext = pd.read_csv(RAW_DIR / 'external_factors.csv')

    print("Step 1: Cleaning products metadata...")
    clean_prod = clean_products(raw_prod)

    print("Step 2: Cleaning stores metadata...")
    clean_st = clean_stores(raw_stores)

    print("Step 3: Cleaning transaction records...")
    clean_tx = clean_transactions(raw_tx, clean_prod)

    print("Step 4: Cleaning external factor records...")
    clean_ext = clean_external_factors(raw_ext, clean_st)

    print("Step 5: Cleaning inventory logs...")
    clean_inv = clean_inventory(raw_inv, clean_tx)

    print("Step 6: Aggregating transaction records to Daily Store x Product grain...")
    clean_tx['revenue'] = clean_tx['quantity'] * clean_tx['selling_price']

    tx_daily = clean_tx.groupby(['date', 'store_id', 'product_id']).agg(
        units_sold=('quantity', 'sum'),
        revenue=('revenue', 'sum'),
        total_selling_price_sum=('selling_price', 'sum'),
        avg_discount_pct=('discount_pct', 'mean'),
        promotion_flag=('promotion_flag', 'max'),
        txn_count=('transaction_id', 'count')
    ).reset_index()

    tx_daily['avg_selling_price'] = tx_daily['revenue'] / tx_daily['units_sold']

    print("Step 7: Merging daily transaction aggregates onto inventory grid...")
    # Base grid from inventory
    master = clean_inv.merge(tx_daily, on=['date', 'store_id', 'product_id'], how='left')

    # Fill non-sales days
    master['units_sold'] = master['units_sold'].fillna(0).astype(int)
    master['revenue'] = master['revenue'].fillna(0.0)
    master['txn_count'] = master['txn_count'].fillna(0).astype(int)
    master['promotion_flag'] = master['promotion_flag'].fillna(0).astype(int)
    master['avg_discount_pct'] = master['avg_discount_pct'].fillna(0.0)

    print("Step 8: Merging product, store, and external factor metadata...")
    # Merge Product metadata
    master = master.merge(
        clean_prod[['product_id', 'category', 'sub_category', 'brand', 'mrp', 'cost_price', 'shelf_life_days', 'supplier_id']],
        on='product_id', how='left'
    )

    # Fill avg_selling_price on 0-sales days with product MRP
    master['avg_selling_price'] = master['avg_selling_price'].fillna(master['mrp'])

    # Merge Store metadata
    master = master.merge(
        clean_st[['store_id', 'city', 'store_type', 'floor_area_sqft', 'avg_daily_customers', 'region']],
        on='store_id', how='left'
    )

    # Merge External factors metadata
    master = master.merge(
        clean_ext[['date', 'city', 'temp_c', 'rain_mm', 'holiday', 'festival', 'weekend', 'local_event', 'temp_imputed_flag']],
        on=['date', 'city'], how='left'
    )

    print("Step 9: Adding stockout flags and history metrics...")
    # Stockout flag day
    master['stockout_flag_day'] = (master['closing_stock'] == 0).astype(int)

    # Sort master table strictly by store_id, product_id, date
    master['date_dt'] = pd.to_datetime(master['date'])
    master = master.sort_values(['store_id', 'product_id', 'date_dt']).reset_index(drop=True)

    # Cumulative history days per store-product
    master['history_days'] = master.groupby(['store_id', 'product_id']).cumcount() + 1

    # Total history days per store-product to flag sparse history (<28 days total)
    total_days_per_pair = master.groupby(['store_id', 'product_id'])['date'].transform('count')
    master['is_sparse_history'] = (total_days_per_pair < 28).astype(int)

    # Drop temporary datetime sorting column
    master = master.drop(columns=['date_dt', 'total_selling_price_sum'], errors='ignore')

    print("Step 10: Executing rigorous data contract assertion checks...")
    # 1. Unique Grain Check
    grain_dupes = master.duplicated(subset=['date', 'store_id', 'product_id']).sum()
    assert grain_dupes == 0, f"Master table grain violation! Found {grain_dupes} duplicate rows."

    # 2. No NaN in Key Columns & Demand
    key_cols = ['date', 'store_id', 'product_id', 'units_sold', 'closing_stock', 'category', 'city']
    for col in key_cols:
        assert master[col].isnull().sum() == 0, f"Null value found in master table column: {col}"

    # 3. Units Sold Sum Match
    total_clean_tx_units = clean_tx['quantity'].sum()
    total_master_units = master['units_sold'].sum()
    assert total_master_units == total_clean_tx_units, (
        f"Units sum mismatch! Clean transactions: {total_clean_tx_units}, Master: {total_master_units}"
    )

    # 4. Closing Stock Non-negative
    assert master['closing_stock'].min() >= 0, "Negative closing stock found in master table!"

    # 5. Exactly 7 Unique Categories
    assert master['category'].nunique() == 7, f"Expected 7 unique categories in master table, found {master['category'].nunique()}"

    print(f"MASTER TABLE VERIFIED: {len(master)} rows, {master['store_id'].nunique()} stores, {master['product_id'].nunique()} products.")
    print(f"Total Units Sold: {total_master_units:,} | Total Revenue: Rs. {master['revenue'].sum():,.2f}")
    print(f"Stock-out Day Rate: {master['stockout_flag_day'].mean():.2%}")

    # Save master table
    master_path = PROCESSED_DIR / 'master_table.csv'
    master.to_csv(master_path, index=False)
    print("Saved master table to:", master_path)

    # Save quality log
    log_df = get_quality_log_dataframe()
    log_path = REPORTS_DIR / 'data_quality_log.csv'
    log_df.to_csv(log_path, index=False)
    print("Saved data quality log to:", log_path)

    return master

if __name__ == '__main__':
    build_master_table()
