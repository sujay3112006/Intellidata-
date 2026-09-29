# StockSense Master Data Dictionary

**Dataset**: `data/processed/master_table.csv`  
**Primary Key / Grain**: **ONE ROW = ONE DATE x ONE STORE x ONE PRODUCT**  
**Total Rows**: 22,523 | **Date Range**: 2026-05-01 to 2026-08-31 (123 Days)  

---

## Operational Definition of Stock-Out
> **Stock-Out Operational Definition**: A store-product observation on a given calendar date is classified as a **Stock-Out Day** (`stockout_flag_day = 1`) if and only if its clean end-of-day closing stock balance (`closing_stock`) is **exactly 0**.

---

## Field Specifications

| Column Name | Data Type | Description | Source File |
|---|---|---|---|
| `date` | String (YYYY-MM-DD) | Calendar observation date | `inventory.csv` / `transactions.csv` |
| `store_id` | String (S01-S06) | Standardized unique store identifier | `stores.csv` |
| `product_id` | String (P101-P705) | Standardized unique product SKU identifier | `products.csv` |
| `units_sold` | Integer (>= 0) | Total clean non-return units sold on date | `transactions.csv` |
| `revenue` | Float (>= 0.0) | Total gross revenue generated (INR) | `transactions.csv` |
| `avg_selling_price` | Float (>= 0.0) | Effective average selling price (INR) | `transactions.csv` / `products.csv` |
| `avg_discount_pct` | Float (0-100) | Average percentage price discount applied | `transactions.csv` |
| `promotion_flag` | Integer (0 or 1) | Indicator if promotional discount was active | `transactions.csv` |
| `txn_count` | Integer (>= 0) | Total number of individual checkout transactions | `transactions.csv` |
| `opening_stock` | Integer (>= 0) | Beginning of day physical inventory count | `inventory.csv` |
| `received` | Integer (>= 0) | Replenishment inventory units delivered | `inventory.csv` |
| `closing_stock` | Integer (>= 0) | End of day physical inventory count | `inventory.csv` |
| `reorder_lvl` | Integer (>= 0) | Inventory reorder threshold level | `inventory.csv` |
| `lead_days` | Integer (>= 1) | Supplier delivery lead time in calendar days | `inventory.csv` |
| `stockout_flag_day` | Integer (0 or 1) | **Target/Flag**: 1 if closing_stock == 0, else 0 | Calculated |
| `inventory_fixed_flag` | Integer (0 or 1) | 1 if row required arithmetic/opening reconciliation | `inventory.csv` |
| `category` | String | Standardized top-level product category (7 unique) | `products.csv` |
| `sub_category` | String | Granular product sub-category classification | `products.csv` |
| `brand` | String | Product brand name | `products.csv` |
| `mrp` | Float (>= 0.0) | Maximum Retail Price (INR) | `products.csv` |
| `cost_price` | Float (>= 0.0) | Unit procurement cost price (INR) | `products.csv` |
| `shelf_life_days` | Float (>= 1.0) | Product shelf life duration in days | `products.csv` |
| `supplier_id` | String | Unique vendor supplier identifier | `products.csv` |
| `city` | String | Store location city name | `stores.csv` |
| `store_type` | String | Format (Supermarket, Hypermarket, Express) | `stores.csv` |
| `floor_area_sqft` | Integer (>= 1000) | Store retail floor space area in sq. ft. | `stores.csv` |
| `avg_daily_customers` | Integer (>= 100) | Average daily customer footfall baseline | `stores.csv` |
| `region` | String | Geographic store cluster region | `stores.csv` |
| `temp_c` | Float | Average daily ambient temperature (Celsius) | `external_factors.csv` |
| `rain_mm` | Float (>= 0.0) | Total daily rainfall in millimeters | `external_factors.csv` |
| `holiday` | Integer (0 or 1) | Public holiday indicator flag | `external_factors.csv` |
| `festival` | Integer (0 or 1) | Cultural/regional festival indicator flag | `external_factors.csv` |
| `weekend` | Integer (0 or 1) | Calendar weekend indicator (Saturday/Sunday) | `external_factors.csv` |
| `local_event` | Integer (0 or 1) | Local community/city event indicator flag | `external_factors.csv` |
| `temp_imputed_flag` | Integer (0 or 1) | 1 if temperature was interpolated | `external_factors.csv` |
| `history_days` | Integer (>= 1) | Cumulative days of history for store-product SKU | Calculated |
| `is_sparse_history` | Integer (0 or 1) | 1 if total SKU history length < 28 days | Calculated |

---
