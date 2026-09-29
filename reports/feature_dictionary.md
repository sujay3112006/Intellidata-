# StockSense Feature Dictionary

**Target Variables**: 
- `next_7_day_demand`: Cumulative censored-adjusted demand over the forward 7-day window $[t+1, t+7]$ (continuous float, units).
- `stockout_next_7d`: Binary stock-out flag indicating whether `stockout_flag_day == 1` occurs on ANY day in $[t+1, t+7]$ (1 if stock-out, 0 otherwise).

**Observation Grain**: **ONE ROW = ONE DATE ($t$) x ONE STORE x ONE PRODUCT**  
**Prediction Time**: End-of-Day at date $t$. Information after day $t$ is strictly forbidden, with the sole exception of advance calendar dates and planned promotions known ahead of time by supermarket management.

---

## 1. Feature Specifications

| Feature Name | Feature Group | Formula / Definition in Plain Words | Why It Helps the Model | Leakage-Safe? (Proof & Reason) |
|---|---|---|---|---|
| `demand_today` | Lag | Clean censored-adjusted sales on day $t$ (`demand_adj`). | Captures immediate contemporaneous daily purchase momentum. | **Yes**: Derived from day $t$ sales and inventory records. |
| `lag_1` | Lag | `demand_adj` at day $t-1$. | Measures 24-hour persistence and yesterday's customer demand. | **Yes**: Uses day $t-1$ historical sales. |
| `lag_7` | Lag | `demand_adj` at day $t-7$ (exact same day of previous week). | Accounts for weekly seasonality (e.g. Tuesday vs Tuesday patterns). | **Yes**: Uses day $t-7$ historical sales. |
| `lag_14` | Lag | `demand_adj` at day $t-14$ (same weekday two weeks prior). | Confirms bi-weekly cyclical consistency and multi-week trends. | **Yes**: Uses day $t-14$ historical sales. |
| `rolling_mean_7` | Rolling | Mean of `demand_adj` over the 7 days ending at $t$ ($[t-6, t]$). | Smooths daily noise to establish current weekly baseline velocity. | **Yes**: Window strictly ends at day $t$. |
| `rolling_mean_14` | Rolling | Mean of `demand_adj` over the 14 days ending at $t$ ($[t-13, t]$). | Establishes a medium-term baseline demand velocity anchor. | **Yes**: Window strictly ends at day $t$. |
| `rolling_std_7` | Rolling | Sample standard deviation of `demand_adj` over $[t-6, t]$. | Measures short-term demand volatility and purchasing irregularity. | **Yes**: Window strictly ends at day $t$. |
| `rolling_max_7` | Rolling | Maximum single-day `demand_adj` observed over $[t-6, t]$. | Identifies peak surge capacity requirements during promo or events. | **Yes**: Window strictly ends at day $t$. |
| `sales_growth_7` | Rolling | $\frac{\text{rolling\_mean\_7}}{\max(\text{rolling\_mean\_14}, 0.1)} - 1$. | Detects accelerating or decelerating demand trajectories. | **Yes**: Ratios derived strictly from past 7 and 14 days. |
| `current_stock` | Inventory | Clean physical closing stock balance at the end of day $t$. | Ground-truth inventory on hand before the 7-day forecast horizon begins. | **Yes**: Recorded at close of day $t$. |
| `days_of_inventory` | Inventory | $\frac{\text{current\_stock}}{\max(\text{rolling\_mean\_7}, 0.1)}$. | Measures shelf-life coverage days remaining at current velocity. | **Yes**: Closing stock at $t$ divided by past 7-day velocity. |
| `inventory_to_demand_ratio` | Inventory | $\frac{\text{current\_stock}}{\max(7 \times \text{rolling\_mean\_7}, 0.1)}$. | Ratios stock on hand directly to expected weekly volume demand. | **Yes**: Compares stock at $t$ to historical 7-day run rate. |
| `reorder_gap` | Inventory | $\text{current\_stock} - \text{reorder\_lvl}$. | Positive indicates healthy buffer; negative signals replenishment breach. | **Yes**: Uses day $t$ closing stock and contract reorder level. |
| `stock_vs_leadtime_demand`| Inventory | $\frac{\text{current\_stock}}{\max(\text{rolling\_mean\_7} \times \text{lead\_days}, 0.1)}$. | Quantifies inventory runway relative to supplier replenishment lead time. | **Yes**: Uses day $t$ stock and past sales rate. |
| `stockouts_last_14` | Inventory | Cumulative count of stock-out days ($\text{closing\_stock} == 0$) over $[t-13, t]$. | Identifies chronic replenishment bottlenecks and suppressed past demand. | **Yes**: Window strictly covers past 14 days up to $t$. |
| `days_since_last_stockout`| Inventory | Calendar days elapsed since the most recent stock-out day up to day $t$. | Differentiates stably supplied SKUs from recently stock-out prone SKUs. | **Yes**: Counts days backwards from $t$. |
| `avg_received_last_7` | Inventory | Rolling 7-day mean of supplier delivered units (`received`) over $[t-6, t]$. | Captures recent vendor delivery fulfillment activity. | **Yes**: Delivery logs strictly up to day $t$. |
| `incoming_stock_est` | Inventory | Estimated shipment in transit if stock $\le \text{reorder\_lvl}$ in past `lead_days` without delivery. | Reconstructs open purchase orders without physical PO tracking tables. | **Yes**: Looks only at past `lead_days` before day $t$. |
| `lead_days` | Inventory | Supplier contract replenishment delivery lead time (days). | Informs model how quickly supplier can respond to stock depletion. | **Yes**: Supplier catalog attribute known a priori. |
| `discount_pct` | Price / Promo | Percentage discount active on day $t$ ($0-100\%$). | Explains contemporaneous price elasticity and daily sales spikes. | **Yes**: Pricing active on day $t$. |
| `promotion_flag` | Price / Promo | Binary indicator if marketing promotion was active on day $t$. | Explains promotional demand lift on current day. | **Yes**: Recorded on day $t$. |
| `price_change` | Price / Promo | $\frac{\text{avg\_selling\_price}_t}{\text{mean\_price}_{14}} - 1$. | Detects markdown shifts relative to normal price anchor. | **Yes**: Compares day $t$ price to past 14-day mean price. |
| `promo_days_last_7` | Price / Promo | Count of promotional days over $[t-6, t]$. | Captures recent discount fatigue or promotional momentum. | **Yes**: Sum of past 7 days promo flags. |
| `promo_days_next_7` | Advance Promo | Count of scheduled promotional days in the upcoming $[t+1, t+7]$ window. | Allows model to anticipate future promotional demand surge (+48.5% lift). | **Yes**: Retail promotions are planned weeks in advance by NovaMart merchandising. |
| `max_discount_next_7` | Advance Promo | Maximum planned percentage discount scheduled across $[t+1, t+7]$. | Calibrates magnitude of anticipated promotional volume spike. | **Yes**: Pre-planned merchandising discount calendar. |
| `day_of_week` | Time | Day of week index ($0 = \text{Monday}, 6 = \text{Sunday}$). | Captures intra-week shopping cycles (e.g. Friday stock-up). | **Yes**: Derived from calendar date $t$. |
| `weekend_flag` | Time | Binary flag ($1$ if Saturday or Sunday, else $0$). | Differentiates weekend shopping footfall from weekday baseline. | **Yes**: Calendar math on date $t$. |
| `month` | Time | Calendar month ($5 = \text{May} \dots 8 = \text{August}$). | Captures seasonal shifts and weather transitions over summer. | **Yes**: Calendar math on date $t$. |
| `week_no` | Time | ISO calendar week number. | Captures multi-week seasonal trends across the 4-month span. | **Yes**: Calendar math on date $t$. |
| `day_of_month` | Time | Day of month ($1-31$). | Captures beginning-of-month and end-of-month shopping waves. | **Yes**: Calendar math on date $t$. |
| `is_salary_week` | Time | Binary flag ($1$ if `day_of_month` $\le 5$, else $0$). | Models the major Indian monthly grocery salary replenishment wave. | **Yes**: Calendar math on date $t$. |
| `festival_flag` | Time | Binary indicator if day $t$ is a cultural/regional festival. | Explains festive sweet, snack, and beverage demand spikes. | **Yes**: Calendar date $t$. |
| `holiday_flag` | Time | Binary indicator if day $t$ is a gazetted public holiday. | Captures holiday customer footfall shifts. | **Yes**: Calendar date $t$. |
| `local_event_flag` | Time | Binary indicator if local city/community event occurred on day $t$. | Explains localized footfall surges in specific store catchments. | **Yes**: Calendar date $t$. |
| `days_to_next_festival` | Advance Calendar| Days remaining until the next upcoming festival (capped at 14). | Anticipates pre-festival bulk household stock-up behaviour. | **Yes**: Standard annual cultural calendar is known years in advance. |
| `weekend_days_next_7` | Advance Calendar| Number of weekend days (Saturdays/Sundays) falling in $[t+1, t+7]$ ($1, 2$, or $3$). | Adjusts weekly demand for varying calendar weekend compositions. | **Yes**: Calendar dates are fixed and known in advance. |
| `festival_days_next_7`| Advance Calendar| Number of festival days falling in the upcoming $[t+1, t+7]$ window. | Prepares inventory for upcoming festive demand rushes. | **Yes**: Cultural festival calendar is fixed and known in advance. |
| `holiday_days_next_7` | Advance Calendar| Number of public holiday days falling in $[t+1, t+7]$. | Accounts for upcoming holiday shopping footfall. | **Yes**: Government holiday calendar is published in advance. |
| `temp_c` | Weather | Mean ambient outdoor temperature on day $t$ ($^\circ\text{C}$). | Drives beverage, ice cream, and cold dairy consumption. | **Yes**: Recorded on day $t$. |
| `temp_mean_3` | Weather | 3-day trailing mean temperature ending on day $t$ ($[t-2, t]$). | Captures multi-day heatwaves driving sustained cold beverage runs. | **Yes**: Trailing 3-day history up to day $t$. |
| `temp_change_3` | Weather | $\text{temp\_c}_t - \text{temp\_mean\_3}$. | Detects sudden temperature drops or surges. | **Yes**: Day $t$ temp minus trailing 3-day mean. |
| `rain_mm` | Weather | Daily precipitation on day $t$ (mm). | Heavy monsoon rainfall suppresses customer footfall and delays transit. | **Yes**: Recorded on day $t$. |
| `category` | Product Meta | Standardized product category (7 Title Case categories). | Controls for baseline turnover and perishability profiles. | **Yes**: Product catalog metadata. |
| `sub_category` | Product Meta | Granular product sub-category classification. | Enables fine-grained consumer preference clustering. | **Yes**: Product catalog metadata. |
| `brand` | Product Meta | Product manufacturer brand name. | Controls for brand loyalty and demand stability. | **Yes**: Product catalog metadata. |
| `mrp` | Product Meta | Maximum Retail Price (INR). | Captures price-tier elasticity and basket value impact. | **Yes**: Product catalog metadata. |
| `cost_price` | Product Meta | Unit procurement cost (INR). | Informs profit margins and inventory capital intensity. | **Yes**: Supplier catalog metadata. |
| `shelf_life_days` | Product Meta | Product expiration shelf life duration (days). | Differentiates fast-expiring Dairy/Bakery from long-life Groceries. | **Yes**: Product catalog metadata. |
| `store_type` | Store Meta | Retail format: `Hypermarket`, `Supermarket`, or `Express`. | Governs footfall scale, assortment depth, and basket size. | **Yes**: Store property metadata. |
| `floor_area_sqft` | Store Meta | Total store sales floor space (sq. ft.). | Proxy for shelf holding capacity and customer browsing space. | **Yes**: Store property metadata. |
| `avg_daily_customers` | Store Meta | Historical average daily footfall baseline. | Baseline store sales velocity driver. | **Yes**: Store profile metadata. |
| `city` | Store Meta | Store city location (Chennai, Bangalore, Hyderabad, etc.). | Regional demographic and weather clustering. | **Yes**: Store property metadata. |
| `history_days` | History | Cumulative days of history available for this store-product SKU up to $t$. | Signals model whether historical lags are stable or maturing. | **Yes**: Cumulative count up to day $t$. |
| `is_sparse_history` | History | Binary flag: $1$ if total SKU history length $< 28$ days, else $0$. | Routes new products to hierarchical fallback forecasting formula. | **Yes**: Static SKU introduction metadata. |

---

## 2. Operational Definition of Censored Demand (`demand_adj`)

In supermarket retail data, true customer demand is censored whenever shelves are empty (`closing_stock == 0`):
$$\text{Recorded Sales} \le \text{True Unobserved Consumer Demand}$$

To prevent machine learning models from learning downward-biased forecasts on stock-out items, we construct `demand_adj`:
1. **Normal In-Stock Day (`stockout_flag_day == 0`)**:
   $$\text{demand\_adj} = \text{units\_sold}$$
2. **Stock-Out Day (`stockout_flag_day == 1`)**:
   $$\text{demand\_adj} = \max\left(\text{units\_sold}, \text{Round}\left(\text{Mean}(\text{Past 14 Non-Stockout Days with Same Weekend \& Promo Flag}), 2\right)\right)$$
   *Fallback 1*: If no exact weekend/promo matches exist, average over the last 14 non-stockout days of the same store-product.  
   *Fallback 2*: If no prior non-stockout days exist (e.g. Day 1 stockout), fallback to observed `units_sold`.

This adjustment recovered **10,565.25 unfulfilled demand units** across 1,186 stock-out days without any forward temporal leakage.
