# StockSense Data Quality & Cleaning Report

**Company**: NovaMart Retail Pvt. Ltd.  
**Author**: Student 1 (Data Analyst) | Team CodeHawks  
**Dataset Coverage**: 2026-05-01 to 2026-08-31 (123 Days) | 6 Stores | 36 Products | 22,523 Observations  

---

## 1. Executive Summary & Data Quality Score
A comprehensive automated audit was conducted on all 5 raw NovaMart datasets (`transactions.csv`, `products.csv`, `stores.csv`, `inventory.csv`, `external_factors.csv`).

- **Total Raw Transactions Processed**: 174,613 rows
- **Clean Transactions Retained**: 173,521 rows (99.37% retention rate)
- **Flagged Transactions (Returns/Outliers)**: 51 rows (0.03% moved to `flagged_rows_removed_from_demand.csv`)
- **Inventory Logs Reconciled / Fixed**: 377 rows
- **Overall Data Quality Score**: **99.2% Clean Data Integrity Index**

---

## 2. Detailed Data Quality Audit Log
The table below details every data anomaly detected, exact row count affected, action taken, and technical justification:

| File | Issue | Count | Action Taken | Justification |
|---|---|---|---|---|
| `products.csv` | Duplicate product rows | 1 | Dropped duplicate rows | Product metadata should be unique per product_id. |
| `products.csv` | Inconsistent category names & casing | 14 | Mapped to 7 standard Title Case categories using CATEGORY_MAP | Standardization is required for clean aggregation and EDA across product lines. |
| `products.csv` | Missing shelf_life_days | 1 | Imputed using median shelf life of sub_category / category | Sub-category peers share similar physical expiry characteristics. |
| `products.csv` | Missing brand name | 1 | Filled missing brand with 'Unknown' | Prevents missing values while preserving product record. |
| `stores.csv` | Store ID and text formatting | 0 | Validated store IDs and region mappings | Store metadata is clean and intact across all 6 locations. |
| `transactions.csv` | True duplicate transactions | 1,041 | Dropped identical duplicate transaction rows | Duplicate rows distort demand volume and total revenue. |
| `transactions.csv` | Mixed date formats (DD-MM-YYYY vs YYYY-MM-DD) | 520 | Parsed DD-MM-YYYY explicitly with format %d-%m-%Y | Prevents month/day swap errors during time-series modeling. |
| `transactions.csv` | Negative quantity returns | 45 | Excluded from demand calculations and moved to flagged_rows_removed_from_demand.csv | Customer returns represent reverse logistics, not consumer demand. |
| `transactions.csv` | Extreme quantity outliers (>100 units/basket) | 6 | Excluded from demand calculations and moved to flagged_rows_removed_from_demand.csv | Bulk wholesale orders warp retail store demand forecasts. |
| `transactions.csv` | Missing discount_pct values | 1,735 | Imputed: 1720 via store-product-date mean, 15 via selling_price vs MRP ratio, 0 via 0.0 fallback | Accurate discount percentages are vital for promotion lift and elasticity estimation. |
| `transactions.csv` | Promotion flag = 0 when discount_pct > 0 | 37 | Set promotion_flag = 1 | A positive discount proves that a price promotion occurred. |
| `transactions.csv` | Missing customer_id | 8,676 | Filled missing customer_id with 'WALKIN' | Preserves real sales transactions without customer registration. |
| `transactions.csv` | Missing payment_mode | 1,734 | Filled missing payment_mode with 'Unknown' | Preserves transaction records while flagging unknown tender. |
| `external_factors.csv` | Duplicate external factor rows | 6 | Dropped duplicate city-date records | Each city should have exactly one record per date. |
| `external_factors.csv` | Extreme temperature outliers (>45 C) | 3 | Set temperature to NaN for time-series interpolation | Temperatures > 45 C in Tamil Nadu winter/monsoon months are sensor glitches. |
| `external_factors.csv` | Missing city-date records in external factors | 12 | Reindexed to full 738 city-day calendar grid (6 cities x 123 days) | Ensures complete external context for every store-date observation. |
| `external_factors.csv` | Missing temperature values (including reindexed dates & outliers) | 36 | Linearly interpolated temp_c over time per city and marked temp_imputed_flag = 1 | Weather changes smoothly over adjacent days. |
| `external_factors.csv` | Missing precipitation (rain_mm) | 26 | Imputed rain_mm as 0.0 mm and marked rain_imputed_flag = 1 | Dry weather is the default baseline state on unrecorded days. |
| `inventory.csv` | Duplicate inventory records | 10 | Dropped duplicate store-product-date inventory rows | Inventory logs must contain exactly one daily closing record per item. |
| `inventory.csv` | Missing daily series gap rows | 30 | Inserted missing store-product-date daily series gap rows | Ensures uninterrupted daily time series for inventory and sales. |
| `inventory.csv` | Missing opening stock / arithmetic mismatches | 377 | Reconciled opening/closing stock with next day opening or inventory balance formula | Ensures inventory arithmetic consistency (closing = opening + received - sold). |

---

## 3. Known Data Limitations & Operational Constraints
1. **Unobserved Incoming Purchase Orders**: The raw inventory dataset tracks `received` stock on the day it arrives, but lacks an open purchase order pipeline or supplier commitment schedule.
2. **New Products & Sparse History**: Certain newly launched SKUs (e.g. `P701`-`P705`) have fewer than 28 days of historical sales records (`is_sparse_history = 1`). Cold-start strategy or sub-category grouping is required for ML forecasting.
3. **Store Footfall Aggregation**: Customer traffic is available as an `avg_daily_customers` static store metric rather than a dynamic daily hourly footfall counter.
4. **Local Event Specifics**: The `local_event` indicator in external factors is a binary flag without granular event category tags (e.g., concert, sporting event, store renovation).

---
*Report generated programmatically from `data_quality_log.csv`.*
