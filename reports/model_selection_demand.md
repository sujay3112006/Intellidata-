# Demand Forecasting Model Selection & Validation Report

**Author**: Student 2 (Machine Learning Engineer)  
**Hackathon**: StockSense 2026 - Team CodeHawks  
**Task**: 7-Day Forward SKU Demand Forecasting (`next_7_day_demand`)  
**Target Variable**: Cumulative censored-adjusted units sold from day $t+1$ to $t+7$  

---

## 1. Executive Summary & Key Results

We constructed a leakage-safe Machine Learning demand forecasting pipeline benchmarked against standard retail naive baselines. 
Across **22,523 master table observations**, models were evaluated using chronological, time-aware splits with strict 7-day buffer gaps.

| Model Candidate | Split | WAPE (%) | MAE (Units) | RMSE (Units) | $R^2$ Score | Training Time (s) |
|---|---|---|---|---|---|---|
| **Baseline 1 (Rolling Mean 7x7)** | Val | 20.07% | 22.24 | 40.82 | 0.8563 | < 0.01s |
| **Baseline 1 (Rolling Mean 7x7)** | Test | 19.68% | 23.59 | 40.19 | 0.8956 | < 0.01s |
| **Linear Regression** | Val | 14.33% | 15.88 | 24.75 | 0.9472 | 0.17s |
| **Decision Tree** | Val | 15.19% | 16.84 | 26.93 | 0.9375 | 0.32s |
| **KNN Regressor (k=7)** | Val | 17.69% | 19.60 | 29.87 | 0.9230 | 0.07s |
| **Random Forest** | Val | 12.67% | 14.04 | 22.46 | 0.9565 | 1.93s |
| **XGBoost (Champion)** | **Val** | **11.96%** | **13.26** | **21.15** | **0.9614** | **2.41s** |
| **XGBoost (Champion)** | **Test** | **12.14%** | **14.56** | **24.51** | **0.9612** | **2.41s** |

### Core Empirical Takeaways:
1. **Outperforming the Baseline**: The selected XGBoost model slashes **WAPE by 40.4%** and **RMSE by 48.2%** relative to the naive rolling average.
2. **Minimal Generalization Gap**: Validation WAPE (11.96%) matches Test WAPE (12.14%), proving the absence of overfitting across temporal shifts.
3. **Execution Efficiency**: Full training completes in **2.41 seconds**, perfectly suited for automated daily batch replenishment runs.

---

## 2. Evaluation Metric Selection & Business Rationale

In retail supermarket inventory optimization, choosing the correct loss and evaluation metric is critical:

1. **WAPE (Weighted Absolute Percentage Error - Primary Decision Metric)**:
   $$\text{WAPE} = \frac{\sum |y_i - \hat{y}_i|}{\sum y_i} \times 100$$
   *Business Justification*: Unlike standard MAPE, which divides by zero or explodes on slow-moving SKUs (e.g. selling 1 unit), WAPE aggregates total forecast error units relative to total units sold. A store manager immediately understands: "Across all items ordered, our order error is only 12.1%."

2. **RMSE (Root Mean Squared Error - Safety Stock Metric)**:
   $$\text{RMSE} = \sqrt{\frac{1}{N} \sum (y_i - \hat{y}_i)^2}$$
   *Business Justification*: RMSE heavily penalizes large forecast misses. A 50-unit error causes a catastrophic stock-out or massive overstock expiry, whereas ten 5-unit errors are easily absorbed by shelf buffers. Furthermore, Student 3 uses category-level RMSE as $\sigma_{\text{demand}}$ for analytical safety stock sizing.

3. **MAE (Mean Absolute Error)**:
   *Business Justification*: Expressed in physical unit counts (e.g. 14.56 units). Tells supply chain planners the average physical unit discrepancy per SKU-week.

4. **$R^2$ Score (Coefficient of Determination)**:
   *Business Justification*: Used as a sanity check. $R^2 = 0.9612$ confirms that over 96% of the variance in customer purchasing velocity is captured by our features.

---

## 3. Champion Model Architecture & Hyperparameters

The champion model is **XGBoost Regressor** wrapped in a scikit-learn `Pipeline`:
- **Categorical Preprocessing**: `OneHotEncoder(handle_unknown='ignore')` applied to `store_id`, `product_id`, `category`, `sub_category`, `store_type`, `city`.
- **Numeric Preprocessing**: `SimpleImputer(strategy='median')` handling sparse warmup lags without information destruction.
- **Tuned Hyperparameters**:
  - `n_estimators = 160`
  - `max_depth = 6` (controls tree interaction complexity, preventing leaf memorization)
  - `learning_rate = 0.08` (conservative shrinkage factor for smooth convergence)
  - `subsample = 0.85` & `colsample_bytree = 0.85` (stochastic row & feature bagging)
  - `min_child_weight = 3` (prunes unstable retail micro-splits)
  - `random_state = 42` (ensuring 100% deterministic reproducibility)

---

## 4. Feature Importance & Domain Interpretation

Analysis of split gain and test permutation importance identified the following primary drivers:

1. **`rolling_mean_7` & `lag_7` (Velocity Anchors)**: Recent contemporaneous sales velocity provides the strongest baseline for customer baseline purchase intent.
2. **`stockouts_last_14` & `demand_adj` (Censored Demand Recovery)**: Recognizing that suppressed past sales were caused by stock-outs prevents the model from under-ordering high-demand products.
3. **`category` & `mrp` (Product Elasticity)**: High-frequency perishable categories (`Dairy`, `Beverages`) show distinct cyclical turnover compared to durable goods (`Household`).
4. **`promo_days_next_7` & `discount_pct` (Promotional Lift)**: Advance promotional calendar visibility allows the model to anticipate the +48.5% promotional volume surge identified in Round 1 EDA.
5. **`weekend_days_next_7` & `festival_flag` (Calendar Drivers)**: Store footfall jumps on weekends and festive occasions, requiring pre-emptive buffer accumulation.

---

## 5. Error Diagnostics & Risk Boundary Analysis

Inspection of the worst forecast residuals revealed two primary drivers:
- **Promotion Start Days on Volatile Snack SKUs**: Minor under-predictions occur on the first day of an aggressive discount promotion (+30% price drop) due to sudden footfall spikes.
- **Sparse-History Fallback SKUs**: Products `P701-P705` have fewer than 28 days of history. By routing them to the hierarchical fallback formula (`predict_sparse_history_fallback`), we assign `confidence = 'low'` and wider prediction bounds, warning store managers to verify initial supplier purchase orders manually.
