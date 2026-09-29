"""
StockSense Feature Engineering Module (Round 2 - Student 2: ML Engineer)
Constructs leakage-safe features, censored-demand adjustments, forward targets,
and time-aware validation splits for NovaMart 7-day demand forecasting.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    PROCESSED_DIR,
    RANDOM_SEED,
    AS_OF_DATE,
    FORECAST_HORIZON,
    TRAIN_START,
    TRAIN_END,
    VAL_GAP_START,
    VAL_GAP_END,
    VAL_START,
    VAL_END,
    TEST_GAP_START,
    TEST_GAP_END,
    TEST_START,
    TEST_END,
    UNLABELED_START,
    FUTURE_CALENDAR,
    PLANNED_PROMO_NEXT_7
)


def correct_censored_demand(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 2: Correct demand for hidden lost sales on stock-out days (closing_stock == 0).
    On stock-out days, recorded units_sold underestimates true consumer purchase intent.
    
    Formula:
    - Normal day (stockout_flag_day == 0): demand_adj = units_sold
    - Stock-out day (stockout_flag_day == 1):
      Look back at previous 14 non-stockout days of the same store-product matching
      both weekend_flag and promotion_flag.
      Fallback 1: previous 14 non-stockout days regardless of weekend/promo.
      Fallback 2: units_sold.
      demand_adj = max(units_sold, round(imputed_mean, 2))
    - lost_units_est = demand_adj - units_sold
    Strictly past data is used (zero leakage).
    """
    print("Applying censored-demand correction for stock-out days...")
    df = df.sort_values(['store_id', 'product_id', 'date']).reset_index(drop=True)
    
    demand_adj = np.zeros(len(df), dtype=float)
    lost_units = np.zeros(len(df), dtype=float)
    
    # Process each store-product time series independently
    for (s_id, p_id), grp_indices in df.groupby(['store_id', 'product_id'], sort=False).groups.items():
        idx_arr = grp_indices.values
        u = df.loc[idx_arr, 'units_sold'].values
        so = df.loc[idx_arr, 'stockout_flag_day'].values
        wk = df.loc[idx_arr, 'weekend'].values
        pr = df.loc[idx_arr, 'promotion_flag'].values
        n = len(u)
        
        for i in range(n):
            row_idx = idx_arr[i]
            if so[i] == 0:
                demand_adj[row_idx] = float(u[i])
                lost_units[row_idx] = 0.0
            else:
                # Look back strictly into the past
                matched_past = []
                for j in range(i - 1, -1, -1):
                    if so[j] == 0 and wk[j] == wk[i] and pr[j] == pr[i]:
                        matched_past.append(u[j])
                        if len(matched_past) == 14:
                            break
                
                if len(matched_past) > 0:
                    imp_val = float(np.mean(matched_past))
                else:
                    # Fallback 1: previous non-stockout days regardless of weekend/promo
                    fallback_past = []
                    for j in range(i - 1, -1, -1):
                        if so[j] == 0:
                            fallback_past.append(u[j])
                            if len(fallback_past) == 14:
                                break
                    if len(fallback_past) > 0:
                        imp_val = float(np.mean(fallback_past))
                    else:
                        # Fallback 2: observed units sold
                        imp_val = float(u[i])
                
                adj_val = max(float(u[i]), round(imp_val, 2))
                demand_adj[row_idx] = adj_val
                lost_units[row_idx] = adj_val - float(u[i])
                
    df['demand_adj'] = demand_adj
    df['lost_units_est'] = lost_units
    
    num_adjusted = int((df['lost_units_est'] > 0).sum())
    total_lost = float(df['lost_units_est'].sum())
    print(f"Censored demand correction completed: {num_adjusted} rows adjusted, {total_lost:.2f} total estimated lost units recovered.")
    return df


def compute_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 3: Construct forward targets looking strictly into t+1..t+7.
    - next_7_day_demand: forward 7-day sum of demand_adj
    - stockout_next_7d: 1 if any stockout occurs in next 7 days, else 0
    - next_7_day_units_sold: forward 7-day sum of raw units_sold (reference only)
    Rows with fewer than 7 future days in the series will have NaN targets.
    """
    print("Constructing forward targets (next_7_day_demand, stockout_next_7d)...")
    df = df.sort_values(['store_id', 'product_id', 'date']).reset_index(drop=True)
    
    fwd_demand = []
    fwd_stockout = []
    fwd_raw_units = []
    
    for (s_id, p_id), grp in df.groupby(['store_id', 'product_id'], sort=False):
        d_adj = grp['demand_adj'].values
        so = grp['stockout_flag_day'].values
        u = grp['units_sold'].values
        n = len(d_adj)
        
        grp_fwd_demand = np.full(n, np.nan, dtype=float)
        grp_fwd_stockout = np.full(n, np.nan, dtype=float)
        grp_fwd_raw_units = np.full(n, np.nan, dtype=float)
        
        for i in range(n):
            if i + 7 < n:
                # Horizon t+1 to t+7
                window_d = d_adj[i + 1: i + 8]
                window_so = so[i + 1: i + 8]
                window_u = u[i + 1: i + 8]
                
                grp_fwd_demand[i] = float(np.sum(window_d))
                grp_fwd_stockout[i] = 1.0 if np.any(window_so == 1) else 0.0
                grp_fwd_raw_units[i] = float(np.sum(window_u))
                
        fwd_demand.extend(grp_fwd_demand)
        fwd_stockout.extend(grp_fwd_stockout)
        fwd_raw_units.extend(grp_fwd_raw_units)
        
    df['next_7_day_demand'] = fwd_demand
    df['stockout_next_7d'] = fwd_stockout
    df['next_7_day_units_sold'] = fwd_raw_units
    
    valid_targets = df['next_7_day_demand'].notna().sum()
    print(f"Target calculation complete: {valid_targets} rows with fully observed 7-day future horizon.")
    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 4: Compute all feature groups strictly using information available at day t.
    Feature Groups:
    1. Time Features
    2. Demand Lag Features (demand_today, lag_1, lag_7, lag_14)
    3. Rolling Demand Features (rolling_mean_7, rolling_mean_14, rolling_std_7, rolling_max_7, sales_growth_7)
    4. Inventory Features (current_stock, days_of_inventory, inventory_to_demand_ratio, reorder_gap,
       stock_vs_leadtime_demand, stockouts_last_14, days_since_last_stockout, avg_received_last_7, lead_days)
    5. Incoming Stock Estimate (incoming_stock_est - leakage-safe transit order detection)
    6. Price & Promo Features (discount_pct, promotion_flag, price_change, promo_days_last_7,
       promo_days_next_7, max_discount_next_7)
    7. Calendar Ahead Features (weekend_days_next_7, festival_days_next_7, holiday_days_next_7, days_to_next_festival)
    8. Weather Features (temp_c, temp_mean_3, temp_change_3, rain_mm)
    9. Store & Product Metadata (encoded later in model pipeline)
    10. History Attributes (history_days, is_sparse_history)
    """
    print("Computing leakage-safe feature groups...")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['store_id', 'product_id', 'date']).reset_index(drop=True)
    
    # --- 1. TIME FEATURES ---
    df['day_of_week'] = df['date'].dt.dayofweek
    df['weekend_flag'] = (df['day_of_week'] >= 5).astype(int)
    df['month'] = df['date'].dt.month
    df['week_no'] = df['date'].dt.isocalendar().week.astype(int)
    df['day_of_month'] = df['date'].dt.day
    df['festival_flag'] = df['festival'].fillna(0).astype(int)
    df['holiday_flag'] = df['holiday'].fillna(0).astype(int)
    df['local_event_flag'] = df['local_event'].fillna(0).astype(int)
    df['is_salary_week'] = (df['day_of_month'] <= 5).astype(int)
    
    # Pre-build calendar lookup for days_to_next_festival & calendar ahead
    all_dates = pd.date_range(df['date'].min(), pd.to_datetime('2026-09-08'))
    festival_dates = set(df[df['festival'] == 1]['date'].dt.strftime('%Y-%m-%d').unique())
    # Add future calendar festival dates
    for f_date, f_info in FUTURE_CALENDAR.items():
        if f_info.get('festival', 0) == 1:
            festival_dates.add(f_date)
            
    # Days to next festival mapping
    days_to_fest_map = {}
    for d in all_dates:
        d_str = d.strftime('%Y-%m-%d')
        min_dist = 14
        for offset in range(1, 15):
            future_d = (d + pd.Timedelta(days=offset)).strftime('%Y-%m-%d')
            if future_d in festival_dates:
                min_dist = offset
                break
        days_to_fest_map[d_str] = min_dist
        
    df['days_to_next_festival'] = df['date'].dt.strftime('%Y-%m-%d').map(days_to_fest_map).fillna(14).astype(int)
    
    # --- GROUPBY STORE x PRODUCT FOR LAG, ROLLING, INVENTORY & TRANSIT FEATURES ---
    # Prepare result containers
    n_rows = len(df)
    demand_today = np.zeros(n_rows, dtype=float)
    lag_1 = np.full(n_rows, np.nan, dtype=float)
    lag_7 = np.full(n_rows, np.nan, dtype=float)
    lag_14 = np.full(n_rows, np.nan, dtype=float)
    
    rolling_mean_7 = np.zeros(n_rows, dtype=float)
    rolling_mean_14 = np.zeros(n_rows, dtype=float)
    rolling_std_7 = np.zeros(n_rows, dtype=float)
    rolling_max_7 = np.zeros(n_rows, dtype=float)
    sales_growth_7 = np.zeros(n_rows, dtype=float)
    
    current_stock = np.zeros(n_rows, dtype=float)
    days_of_inventory = np.zeros(n_rows, dtype=float)
    inventory_to_demand_ratio = np.zeros(n_rows, dtype=float)
    reorder_gap = np.zeros(n_rows, dtype=float)
    stock_vs_leadtime_demand = np.zeros(n_rows, dtype=float)
    stockouts_last_14 = np.zeros(n_rows, dtype=float)
    days_since_last_stockout = np.zeros(n_rows, dtype=float)
    avg_received_last_7 = np.zeros(n_rows, dtype=float)
    incoming_stock_est = np.zeros(n_rows, dtype=float)
    
    discount_pct = np.zeros(n_rows, dtype=float)
    promotion_flag = np.zeros(n_rows, dtype=int)
    price_change = np.zeros(n_rows, dtype=float)
    promo_days_last_7 = np.zeros(n_rows, dtype=float)
    promo_days_next_7 = np.zeros(n_rows, dtype=float)
    max_discount_next_7 = np.zeros(n_rows, dtype=float)
    
    weekend_days_next_7 = np.zeros(n_rows, dtype=float)
    festival_days_next_7 = np.zeros(n_rows, dtype=float)
    holiday_days_next_7 = np.zeros(n_rows, dtype=float)
    
    temp_c = np.zeros(n_rows, dtype=float)
    temp_mean_3 = np.zeros(n_rows, dtype=float)
    temp_change_3 = np.zeros(n_rows, dtype=float)
    rain_mm = np.zeros(n_rows, dtype=float)
    
    # Iterate through each store-product group
    for (s_id, p_id), grp_indices in df.groupby(['store_id', 'product_id'], sort=False).groups.items():
        idx_arr = grp_indices.values
        m = len(idx_arr)
        
        grp_dates = df.loc[idx_arr, 'date'].values
        grp_d_adj = df.loc[idx_arr, 'demand_adj'].values
        grp_closing = df.loc[idx_arr, 'closing_stock'].values
        grp_reorder_lvl = df.loc[idx_arr, 'reorder_lvl'].values
        grp_lead_days = df.loc[idx_arr, 'lead_days'].values
        grp_so = df.loc[idx_arr, 'stockout_flag_day'].values
        grp_received = df.loc[idx_arr, 'received'].values
        grp_price = df.loc[idx_arr, 'avg_selling_price'].values
        grp_disc = df.loc[idx_arr, 'avg_discount_pct'].values
        grp_promo = df.loc[idx_arr, 'promotion_flag'].values
        grp_wk = df.loc[idx_arr, 'weekend'].values
        grp_fest = df.loc[idx_arr, 'festival'].values
        grp_hol = df.loc[idx_arr, 'holiday'].values
        grp_temp = df.loc[idx_arr, 'temp_c'].values
        grp_rain = df.loc[idx_arr, 'rain_mm'].values
        
        last_so_pos = -999
        
        for i in range(m):
            idx = idx_arr[i]
            
            # --- 2. LAG FEATURES ---
            demand_today[idx] = grp_d_adj[i]
            if i >= 1:
                lag_1[idx] = grp_d_adj[i - 1]
            if i >= 7:
                lag_7[idx] = grp_d_adj[i - 7]
            if i >= 14:
                lag_14[idx] = grp_d_adj[i - 14]
                
            # --- 3. ROLLING DEMAND FEATURES (ending at day t) ---
            w7_start = max(0, i - 6)
            w14_start = max(0, i - 13)
            
            w7_demand = grp_d_adj[w7_start: i + 1]
            w14_demand = grp_d_adj[w14_start: i + 1]
            
            r_mean7 = float(np.mean(w7_demand))
            r_mean14 = float(np.mean(w14_demand))
            r_std7 = float(np.std(w7_demand, ddof=1)) if len(w7_demand) > 1 else 0.0
            r_max7 = float(np.max(w7_demand))
            
            rolling_mean_7[idx] = r_mean7
            rolling_mean_14[idx] = r_mean14
            rolling_std_7[idx] = r_std7
            rolling_max_7[idx] = r_max7
            sales_growth_7[idx] = (r_mean7 / max(r_mean14, 0.1)) - 1.0
            
            # --- 4. INVENTORY FEATURES ---
            c_stock = float(grp_closing[i])
            r_lvl = float(grp_reorder_lvl[i])
            l_days = float(grp_lead_days[i])
            
            current_stock[idx] = c_stock
            days_of_inventory[idx] = c_stock / max(r_mean7, 0.1)
            inventory_to_demand_ratio[idx] = c_stock / max(7.0 * r_mean7, 0.1)
            reorder_gap[idx] = c_stock - r_lvl
            stock_vs_leadtime_demand[idx] = c_stock / max(r_mean7 * l_days, 0.1)
            
            stockouts_last_14[idx] = float(np.sum(grp_so[w14_start: i + 1]))
            
            # Days since last stockout
            if grp_so[i] == 1:
                last_so_pos = i
                days_since_last_stockout[idx] = 0.0
            elif last_so_pos >= 0:
                days_since_last_stockout[idx] = float(i - last_so_pos)
            else:
                days_since_last_stockout[idx] = min(float(i + 1), 60.0)
                
            avg_received_last_7[idx] = float(np.mean(grp_received[w7_start: i + 1]))
            
            # --- 5. INCOMING STOCK ESTIMATE (Transit Order Detection) ---
            # Look at past lead_days up to day t: was stock <= reorder_lvl on any of those days?
            int_lead = int(max(1, l_days))
            lead_window_start = max(0, i - int_lead + 1)
            reorder_triggered_indices = [k for k in range(lead_window_start, i + 1) if grp_closing[k] <= grp_reorder_lvl[k]]
            
            if reorder_triggered_indices:
                latest_trigger_k = max(reorder_triggered_indices)
                # Check if any delivery arrived strictly since that trigger day
                received_since = np.sum(grp_received[latest_trigger_k + 1: i + 1]) if (latest_trigger_k + 1 <= i) else 0
                if received_since == 0:
                    # Order is likely in transit: estimate incoming volume as median of positive received in past 30 days
                    past30_start = max(0, i - 29)
                    past_received = grp_received[past30_start: i + 1]
                    pos_received = past_received[past_received > 0]
                    if len(pos_received) > 0:
                        incoming_stock_est[idx] = float(np.median(pos_received))
                    else:
                        all_pos = grp_received[:i + 1][grp_received[:i + 1] > 0]
                        incoming_stock_est[idx] = float(np.median(all_pos)) if len(all_pos) > 0 else 0.0
                else:
                    incoming_stock_est[idx] = 0.0
            else:
                incoming_stock_est[idx] = 0.0
                
            # --- 6. PRICE & PROMO FEATURES ---
            cur_price = float(grp_price[i])
            w14_prices = grp_price[w14_start: i + 1]
            mean_price_14 = float(np.mean(w14_prices)) if len(w14_prices) > 0 else cur_price
            
            discount_pct[idx] = float(grp_disc[i])
            promotion_flag[idx] = int(grp_promo[i])
            price_change[idx] = (cur_price / max(mean_price_14, 0.01)) - 1.0
            promo_days_last_7[idx] = float(np.sum(grp_promo[w7_start: i + 1]))
            
            # Known-in-advance promotions: next 7 days
            if i + 7 < m:
                promo_days_next_7[idx] = float(np.sum(grp_promo[i + 1: i + 8]))
                max_discount_next_7[idx] = float(np.max(grp_disc[i + 1: i + 8]))
            else:
                # Outside historical dataset (e.g. AS_OF_DATE)
                p_info = PLANNED_PROMO_NEXT_7.get((s_id, p_id), {'promo_days': 0.0, 'max_discount': 0.0})
                promo_days_next_7[idx] = float(p_info.get('promo_days', 0.0))
                max_discount_next_7[idx] = float(p_info.get('max_discount', 0.0))
                
            # --- 7. CALENDAR AHEAD FEATURES (Known in Advance) ---
            if i + 7 < m:
                weekend_days_next_7[idx] = float(np.sum(grp_wk[i + 1: i + 8]))
                festival_days_next_7[idx] = float(np.sum(grp_fest[i + 1: i + 8]))
                holiday_days_next_7[idx] = float(np.sum(grp_hol[i + 1: i + 8]))
            else:
                # Use FUTURE_CALENDAR for predictions at dataset horizon
                cur_dt = pd.to_datetime(grp_dates[i])
                f_wk_cnt = 0.0
                f_fest_cnt = 0.0
                f_hol_cnt = 0.0
                for f_step in range(1, 8):
                    f_d_str = (cur_dt + pd.Timedelta(days=f_step)).strftime('%Y-%m-%d')
                    f_info = FUTURE_CALENDAR.get(f_d_str, {'weekend': 0, 'festival': 0, 'holiday': 0})
                    f_wk_cnt += f_info.get('weekend', 0)
                    f_fest_cnt += f_info.get('festival', 0)
                    f_hol_cnt += f_info.get('holiday', 0)
                weekend_days_next_7[idx] = f_wk_cnt
                festival_days_next_7[idx] = f_fest_cnt
                holiday_days_next_7[idx] = f_hol_cnt
                
            # --- 8. WEATHER FEATURES (Past Only) ---
            temp_c[idx] = float(grp_temp[i])
            w3_start = max(0, i - 2)
            t_mean3 = float(np.mean(grp_temp[w3_start: i + 1]))
            temp_mean_3[idx] = t_mean3
            temp_change_3[idx] = float(grp_temp[i]) - t_mean3
            rain_mm[idx] = float(grp_rain[i])
            
    # Assign newly engineered columns to dataframe
    df['demand_today'] = demand_today
    df['lag_1'] = lag_1
    df['lag_7'] = lag_7
    df['lag_14'] = lag_14
    
    df['rolling_mean_7'] = rolling_mean_7
    df['rolling_mean_14'] = rolling_mean_14
    df['rolling_std_7'] = rolling_std_7
    df['rolling_max_7'] = rolling_max_7
    df['sales_growth_7'] = sales_growth_7
    
    df['current_stock'] = current_stock
    df['days_of_inventory'] = days_of_inventory
    df['inventory_to_demand_ratio'] = inventory_to_demand_ratio
    df['reorder_gap'] = reorder_gap
    df['stock_vs_leadtime_demand'] = stock_vs_leadtime_demand
    df['stockouts_last_14'] = stockouts_last_14
    df['days_since_last_stockout'] = days_since_last_stockout
    df['avg_received_last_7'] = avg_received_last_7
    df['incoming_stock_est'] = incoming_stock_est
    
    df['discount_pct'] = discount_pct
    df['promotion_flag'] = promotion_flag
    df['price_change'] = price_change
    df['promo_days_last_7'] = promo_days_last_7
    df['promo_days_next_7'] = promo_days_next_7
    df['max_discount_next_7'] = max_discount_next_7
    
    df['weekend_days_next_7'] = weekend_days_next_7
    df['festival_days_next_7'] = festival_days_next_7
    df['holiday_days_next_7'] = holiday_days_next_7
    
    df['temp_c'] = temp_c
    df['temp_mean_3'] = temp_mean_3
    df['temp_change_3'] = temp_change_3
    df['rain_mm'] = rain_mm
    
    print("Feature computation finished successfully.")
    return df


def assign_time_aware_splits(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 7: Assign chronological time-aware train/val/test splits with 7-day buffer gaps.
    Gaps prevent 7-day target horizon leakage between evaluation subsets.
    - warmup: date < TRAIN_START (first 14 days reserved for lag_14 warmup)
    - train: TRAIN_START to TRAIN_END (2026-05-15 to 2026-07-10)
    - gap_val: VAL_GAP_START to VAL_GAP_END (2026-07-11 to 2026-07-17)
    - val: VAL_START to VAL_END (2026-07-18 to 2026-08-03)
    - gap_test: TEST_GAP_START to TEST_GAP_END (2026-08-04 to 2026-08-10)
    - test: TEST_START to TEST_END (2026-08-11 to 2026-08-24)
    - unlabeled: date >= UNLABELED_START (2026-08-25 to 2026-08-31)
    """
    print("Assigning time-aware splits with 7-day buffer gaps...")
    date_str = df['date'].dt.strftime('%Y-%m-%d') if pd.api.types.is_datetime64_any_dtype(df['date']) else df['date'].astype(str)
    
    splits = np.full(len(df), 'warmup', dtype=object)
    
    splits[(date_str >= TRAIN_START) & (date_str <= TRAIN_END)] = 'train'
    splits[(date_str >= VAL_GAP_START) & (date_str <= VAL_GAP_END)] = 'gap_val'
    splits[(date_str >= VAL_START) & (date_str <= VAL_END)] = 'val'
    splits[(date_str >= TEST_GAP_START) & (date_str <= TEST_GAP_END)] = 'gap_test'
    splits[(date_str >= TEST_START) & (date_str <= TEST_END)] = 'test'
    splits[(date_str >= UNLABELED_START)] = 'unlabeled'
    
    df['split'] = splits
    
    split_counts = df['split'].value_counts().to_dict()
    print("Split row distribution:")
    for s_name, count in split_counts.items():
        print(f"  - {s_name}: {count} rows ({count / len(df) * 100:.2f}%)")
        
    return df


def predict_sparse_history_fallback(
    df_features: pd.DataFrame,
    as_of_date: str = AS_OF_DATE
) -> pd.DataFrame:
    """
    Step 6: Fallback demand forecasting strategy for SKUs with sparse history (history_days < 14).
    Formula:
    forecast = 7 * (w * own_mean_daily + (1 - w) * sub_cat_store_mean_daily)
    where w = history_days / 14.
    """
    as_of_mask = (df_features['date'].astype(str) == as_of_date)
    sparse_rows = df_features[as_of_mask & (df_features['history_days'] < 14)].copy()
    
    if len(sparse_rows) == 0:
        return pd.DataFrame()
        
    # Store-subcategory daily benchmark
    subcat_benchmarks = df_features.groupby(['store_id', 'sub_category'])['demand_adj'].mean().to_dict()
    
    fallback_records = []
    for _, row in sparse_rows.iterrows():
        s_id = row['store_id']
        p_id = row['product_id']
        sub_cat = row['sub_category']
        h_days = max(1, row['history_days'])
        w = min(1.0, h_days / 14.0)
        
        # Own mean from available history
        own_history = df_features[(df_features['store_id'] == s_id) & 
                                  (df_features['product_id'] == p_id) & 
                                  (df_features['date'].astype(str) <= as_of_date)]
        own_mean = own_history['demand_adj'].mean() if len(own_history) > 0 else 1.0
        bench_mean = subcat_benchmarks.get((s_id, sub_cat), own_mean)
        
        daily_forecast = w * own_mean + (1.0 - w) * bench_mean
        weekly_forecast = max(0.0, round(float(7.0 * daily_forecast), 2))
        
        fallback_records.append({
            'as_of_date': as_of_date,
            'store_id': s_id,
            'product_id': p_id,
            'forecast_next_7_day_demand': weekly_forecast,
            'forecast_low': max(0.0, round(weekly_forecast * 0.70, 2)),
            'forecast_high': round(weekly_forecast * 1.35, 2),
            'method': 'fallback_sparse_history',
            'confidence': 'low'
        })
        
    return pd.DataFrame(fallback_records)


def build_features_table(save_csv: bool = True) -> pd.DataFrame:
    """
    Main pipeline entry point:
    1. Loads master_table.csv
    2. Corrects censored demand
    3. Builds forward targets
    4. Computes 30+ leakage-safe features
    5. Assigns time-aware splits
    6. Validates schema and contract assertions
    7. Saves features_table.csv
    """
    master_path = PROCESSED_DIR / 'master_table.csv'
    if not master_path.exists():
        raise FileNotFoundError(f"Missing master table at {master_path}. Run src/build_master.py first.")
        
    df = pd.read_csv(master_path)
    print(f"Loaded master table: {df.shape[0]} rows x {df.shape[1]} columns.")
    
    # Execute pipeline stages
    df = correct_censored_demand(df)
    df = compute_targets(df)
    df = compute_features(df)
    df = assign_time_aware_splits(df)
    
    # Contract Assertions
    assert df.duplicated(subset=['date', 'store_id', 'product_id']).sum() == 0, "Error: Grain duplicates detected in features table!"
    assert df['split'].isna().sum() == 0, "Error: Unassigned split labels detected!"
    
    # Ensure no NaN in features for training/val/test splits (except warmup / sparse rows)
    eval_mask = df['split'].isin(['train', 'val', 'test']) & (df['is_sparse_history'] == 0)
    assert df.loc[eval_mask, 'lag_14'].isna().sum() == 0, "Error: NaN values found in lag_14 for mature training rows!"
    assert df.loc[eval_mask, 'next_7_day_demand'].isna().sum() == 0, "Error: NaN target found in labeled evaluation splits!"
    
    print("All feature engineering contract assertions PASSED.")
    
    if save_csv:
        output_path = PROCESSED_DIR / 'features_table.csv'
        df.to_csv(output_path, index=False)
        print(f"Features table successfully saved to {output_path} ({df.shape[0]} rows x {df.shape[1]} columns).")
        
    return df


if __name__ == '__main__':
    build_features_table()
