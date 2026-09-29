"""
StockSense Explainability & Model Interpretation Module (Round 3 - Student 3: Decision Intelligence)
Provides global feature importance and store-SKU local explainability for stock-out risk predictions.
Transforms machine learning probability outputs into transparent, manager-friendly business drivers.
"""

import sys
import os
import json
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.inspection import permutation_importance
from src.config import (
    PROCESSED_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    MODELS_DIR,
    RANDOM_SEED,
    AS_OF_DATE
)
from src.train_stockout_model import get_feature_lists, ProbabilityCalibrator

# Matplotlib styling for high-clarity publication figures
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8


# Group mapping for manager-friendly driver categories
DRIVER_GROUPS = {
    'Low Stock & Buffer Deficit': [
        'current_stock', 'days_of_inventory', 'reorder_gap',
        'stock_vs_leadtime_demand', 'inventory_to_demand_ratio'
    ],
    'Recent Sales Velocity': [
        'rolling_mean_7', 'lag_7', 'lag_1', 'lag_14', 'demand_today',
        'rolling_mean_14', 'rolling_max_7', 'sales_growth_7'
    ],
    'Promotion & Pricing': [
        'promotion_flag', 'discount_pct', 'promo_days_last_7',
        'promo_days_next_7', 'max_discount_next_7', 'price_change'
    ],
    'Supplier Lead Time & Delivery': [
        'lead_days', 'incoming_stock_est', 'avg_received_last_7'
    ],
    'Stock-Out History & Volatility': [
        'stockouts_last_14', 'days_since_last_stockout', 'rolling_std_7'
    ],
    'Upcoming Calendar & Events': [
        'weekend_days_next_7', 'festival_days_next_7', 'holiday_days_next_7',
        'days_to_next_festival', 'is_salary_week', 'day_of_week', 'weekend_flag',
        'festival_flag', 'holiday_flag', 'local_event_flag'
    ],
    'Weather Conditions': [
        'temp_c', 'temp_mean_3', 'temp_change_3', 'rain_mm'
    ],
    'Product & Store Characteristics': [
        'store_id', 'product_id', 'category', 'sub_category', 'store_type',
        'city', 'mrp', 'cost_price', 'shelf_life_days', 'floor_area_sqft',
        'avg_daily_customers', 'history_days', 'month', 'week_no', 'day_of_month'
    ]
}


def compute_global_permutation_importance(model, X_test, y_test, top_n=15):
    """
    Computes permutation feature importance on the out-of-time test set using ROC-AUC scoring.
    Uses a stratified sample of 500 test observations for fast, stable computation.
    """
    print(f"Computing permutation importance on test set sample...", flush=True)
    
    # Sample 500 test rows if dataset is large for rapid execution
    if len(X_test) > 500:
        sample_idx = X_test.sample(n=500, random_state=RANDOM_SEED).index
        X_sample = X_test.loc[sample_idx]
        y_sample = y_test.loc[sample_idx]
    else:
        X_sample, y_sample = X_test, y_test
        
    perm_res = permutation_importance(
        model, X_sample, y_sample,
        scoring='roc_auc',
        n_repeats=2,
        random_state=RANDOM_SEED,
        n_jobs=1
    )
    
    sorted_idx = np.argsort(perm_res.importances_mean)[::-1]
    feature_names = np.array(X_sample.columns)
    
    importance_df = pd.DataFrame({
        'feature': feature_names[sorted_idx],
        'importance_mean': perm_res.importances_mean[sorted_idx],
        'importance_std': perm_res.importances_std[sorted_idx]
    })
    
    # Save CSV report
    out_csv = REPORTS_DIR / 'stockout_feature_importance.csv'
    importance_df.to_csv(out_csv, index=False)
    print(f"Saved feature importance table to {out_csv}", flush=True)
    
    # Plot top 15 features
    top_df = importance_df.head(top_n).sort_values('importance_mean', ascending=True)
    
    plt.figure(figsize=(10, 6.5))
    bars = plt.barh(top_df['feature'], top_df['importance_mean'],
                    xerr=top_df['importance_std'], color='#1f77b4', edgecolor='#0d47a1',
                    alpha=0.85, capsize=3, height=0.65)
    
    plt.title(f"Top {top_n} Most Influential Drivers for 7-Day Stock-Out Risk (Permutation ROC-AUC Loss)",
              fontsize=11, fontweight='bold', pad=12)
    plt.xlabel("Mean Drop in ROC-AUC when Feature is Randomly Shuffled", fontsize=10)
    plt.ylabel("Engineered Feature", fontsize=10)
    
    for bar in bars:
        w = bar.get_width()
        if w > 0:
            plt.text(w + 0.001, bar.get_y() + bar.get_height()/2., f"{w:.4f}",
                     va='center', fontsize=9, fontweight='bold', color='#1a237e')
                     
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'explainability_01_global_importance.png', dpi=150)
    plt.close()
    print(f"Saved global importance figure to {FIGURES_DIR / 'explainability_01_global_importance.png'}", flush=True)
    
    return importance_df


class LocalExplainer:
    """
    Computes local feature group contributions for store-SKU observations.
    Decomposes the predicted probability into intuitive manager-level business drivers.
    """
    def __init__(self, model_pipeline, train_df, feature_cols):
        self.model = model_pipeline
        self.feature_cols = feature_cols
        self.cat_cols, self.num_cols = get_feature_lists()
        
        # Calculate baseline median/mode background vector for reference comparisons
        self.baseline_row = {}
        for c in self.num_cols:
            self.baseline_row[c] = float(train_df[c].median())
        for c in self.cat_cols:
            self.baseline_row[c] = train_df[c].mode().iloc[0]
            
    def explain_df(self, df_input: pd.DataFrame) -> list:
        """
        Vectorized explanation over an entire dataframe of store-SKU rows in bulk.
        """
        N = len(df_input)
        feature_df = df_input[self.feature_cols].copy()
        actual_probs = self.model.predict_proba(feature_df)[:, 1]
        
        # Build bulk counterfactual DataFrames for each group
        group_diffs = {}
        for group_name, feat_list in DRIVER_GROUPS.items():
            cf_df = feature_df.copy()
            for f in feat_list:
                if f in self.baseline_row:
                    cf_df[f] = self.baseline_row[f]
            cf_probs = self.model.predict_proba(cf_df)[:, 1]
            group_diffs[group_name] = actual_probs - cf_probs
            
        explanations = []
        for i in range(N):
            actual_prob = float(actual_probs[i])
            group_impacts = {g: float(group_diffs[g][i]) for g in DRIVER_GROUPS.keys()}
            
            total_pos_impact = sum(max(0, v) for v in group_impacts.values())
            
            driver_items = []
            for g_name, diff in group_impacts.items():
                if abs(diff) < 0.005:
                    continue
                pct = (abs(diff) / max(total_pos_impact, 0.05)) * 100.0 if total_pos_impact > 0 else 0.0
                direction = '+' if diff >= 0 else '-'
                driver_items.append({
                    'driver': g_name,
                    'diff': diff,
                    'pct': round(min(pct, 100.0), 1),
                    'direction': direction,
                    'formatted': f"{g_name} ({direction}{round(min(pct, 100.0))}%)"
                })
                
            driver_items = sorted(driver_items, key=lambda x: abs(x['diff']), reverse=True)
            top_drivers = driver_items[:4]
            
            risk_tier = 'HIGH' if actual_prob >= 0.70 else ('MEDIUM' if actual_prob >= 0.40 else 'LOW')
            
            if risk_tier == 'HIGH':
                primary_reason = top_drivers[0]['driver'] if top_drivers else "Low inventory coverage"
                sec_reason = f" combined with {top_drivers[1]['driver'].lower()}" if len(top_drivers) > 1 else ""
                summary = f"HIGH RISK of stock-out within 7 days ({actual_prob*100:.0f}% prob) primarily driven by {primary_reason.lower()}{sec_reason}."
            elif risk_tier == 'MEDIUM':
                primary_reason = top_drivers[0]['driver'] if top_drivers else "Moderate demand velocity"
                summary = f"MEDIUM RISK ({actual_prob*100:.0f}% prob); monitor buffer levels due to {primary_reason.lower()}."
            else:
                summary = f"LOW RISK ({actual_prob*100:.0f}% prob); adequate stock coverage for expected 7-day demand."
                
            explanations.append({
                'stockout_probability': round(actual_prob, 4),
                'risk_tier': risk_tier,
                'top_drivers': top_drivers,
                'top_reasons_str': ", ".join([d['formatted'] for d in top_drivers]),
                'manager_summary': summary
            })
            
        return explanations

    def explain_instance(self, row_dict: dict) -> dict:
        """Single row wrapper around explain_df."""
        df_single = pd.DataFrame([row_dict])
        return self.explain_df(df_single)[0]


def explain_row(store_id: str, product_id: str, date: str = AS_OF_DATE) -> dict:
    """
    Public lookup function to explain risk for a specific store and product as of a given date.
    """
    feat_path = PROCESSED_DIR / 'features_table.csv'
    model_path = MODELS_DIR / 'stockout_model.pkl'
    
    if not feat_path.exists() or not model_path.exists():
        raise FileNotFoundError("Required features table or stockout model artifact not found.")
        
    df_feat = pd.read_csv(feat_path)
    model = joblib.load(model_path)
    
    cat_cols, num_cols = get_feature_lists()
    feature_cols = cat_cols + num_cols
    
    # Filter specific row
    match = df_feat[(df_feat['store_id'] == store_id) & 
                    (df_feat['product_id'] == product_id) & 
                    (df_feat['date'] == date)]
                    
    if match.empty:
        match = df_feat[(df_feat['store_id'] == store_id) & 
                        (df_feat['product_id'] == product_id)].tail(1)
        if match.empty:
            raise ValueError(f"No records found for Store {store_id} and Product {product_id}.")
            
    train_df = df_feat[df_feat['split'] == 'train']
    explainer = LocalExplainer(model, train_df, feature_cols)
    explanation = explainer.explain_df(match)[0]
    
    explanation['store_id'] = store_id
    explanation['product_id'] = product_id
    explanation['date'] = match.iloc[0]['date']
    explanation['current_stock'] = match.iloc[0]['current_stock']
    explanation['reorder_lvl'] = match.iloc[0]['reorder_lvl']
    explanation['lead_days'] = match.iloc[0]['lead_days']
    
    return explanation


def generate_batch_explanations_and_figures(df_feat, model):
    """
    Generates batch local explanations for all live AS_OF_DATE store-SKUs and diagnostic plots.
    """
    print("\nGenerating batch explainability insights and diagnostic figures in bulk...", flush=True)
    cat_cols, num_cols = get_feature_lists()
    feature_cols = cat_cols + num_cols
    
    train_df = df_feat[df_feat['split'] == 'train']
    explainer = LocalExplainer(model, train_df, feature_cols)
    
    live_df = df_feat[df_feat['date'] == AS_OF_DATE].copy().reset_index(drop=True)
    explanations = explainer.explain_df(live_df)
    
    driver_counts = {g: 0 for g in DRIVER_GROUPS.keys()}
    
    for i, res in enumerate(explanations):
        res['store_id'] = live_df.loc[i, 'store_id']
        res['product_id'] = live_df.loc[i, 'product_id']
        res['date'] = live_df.loc[i, 'date']
        
        # Track top driver for high/medium risk items
        if res['risk_tier'] in ['HIGH', 'MEDIUM'] and res['top_drivers']:
            top_d = res['top_drivers'][0]['driver']
            driver_counts[top_d] = driver_counts.get(top_d, 0) + 1
            
    df_exp = pd.DataFrame(explanations)
    
    # Figure 2: Driver distribution across at-risk items
    s_drivers = pd.Series(driver_counts).sort_values(ascending=True)
    
    plt.figure(figsize=(10, 5.5))
    bars = plt.barh(s_drivers.index, s_drivers.values, color='#e67e22', edgecolor='#b96417', height=0.6)
    plt.title("Primary Operational Drivers Triggering Stock-Out Alerts (Live AS_OF_DATE)",
              fontsize=11, fontweight='bold', pad=12)
    plt.xlabel("Number of Store-SKU Items where Driver is the Primary Cause", fontsize=10)
    plt.ylabel("Business Driver Category", fontsize=10)
    
    for bar in bars:
        w = bar.get_width()
        if w > 0:
            plt.text(w + 0.3, bar.get_y() + bar.get_height()/2., f"{int(w)} SKUs",
                     va='center', fontsize=9, fontweight='bold')
                     
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'explainability_02_driver_groups.png', dpi=150)
    plt.close()
    print(f"Saved driver groups figure to {FIGURES_DIR / 'explainability_02_driver_groups.png'}", flush=True)
    
    return df_exp


def main():
    """Main execution entry point for Explainability."""
    print("=" * 80)
    print(" STOCKSENSE EXPLAINABILITY & DECISION DRIVER MODULE (ROUND 3)")
    print("=" * 80, flush=True)
    
    feat_path = PROCESSED_DIR / 'features_table.csv'
    model_path = MODELS_DIR / 'stockout_model.pkl'
    
    df_feat = pd.read_csv(feat_path)
    model = joblib.load(model_path)
    
    cat_cols, num_cols = get_feature_lists()
    feature_cols = cat_cols + num_cols
    
    # 1. Global Feature Importance on Test Split
    test_df = df_feat[df_feat['split'] == 'test']
    compute_global_permutation_importance(model, test_df[feature_cols], test_df['stockout_next_7d'].astype(int))
    
    # 2. Batch Local Explanations
    df_exp = generate_batch_explanations_and_figures(df_feat, model)
    
    # 3. Sanity check 5 representative examples
    print("\nSanity Check: Local Explanations for 5 Representative Store-SKUs:", flush=True)
    sample_pairs = [
        ('S01', 'P102'), # Dairy / Fast mover
        ('S01', 'P103'), # Beverage / Promo sensitive
        ('S02', 'P102'), # Bread / Daily essential
        ('S03', 'P105'), # Atta / Staples
        ('S06', 'P205')  # Personal Care / Household
    ]
    
    for s_id, p_id in sample_pairs:
        try:
            exp = explain_row(s_id, p_id)
            print(f"\n[Store {s_id} | Product {p_id}] -> Risk: {exp['risk_tier']} (Prob: {exp['stockout_probability']:.1%})", flush=True)
            print(f"  Drivers: {exp['top_reasons_str']}", flush=True)
            print(f"  Manager Summary: {exp['manager_summary']}", flush=True)
        except Exception as e:
            print(f"  Could not explain ({s_id}, {p_id}): {e}", flush=True)
            
    print("\n" + "=" * 80)
    print(" EXPLAINABILITY MODULE COMPLETE.")
    print("=" * 80, flush=True)


if __name__ == '__main__':
    main()
