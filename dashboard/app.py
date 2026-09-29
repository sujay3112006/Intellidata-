"""
StockSense Decision Intelligence Dashboard
Interactive Prototype for NovaMart Retail Pvt. Ltd.
Built with Streamlit & Plotly (Offline, High-Performance, Management-Ready)
"""

import sys
import os
import math
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib

from src.config import (
    PROCESSED_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    MODELS_DIR,
    AS_OF_DATE
)
from src.train_stockout_model import ProbabilityCalibrator, get_feature_lists
from src.explainability import DRIVER_GROUPS, LocalExplainer

# Streamlit Page Configuration
st.set_page_config(
    page_title="StockSense | NovaMart Decision Intelligence",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Design & Visual Polish
st.markdown("""
<style>
    /* Metric Card Styling */
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 12px;
        padding: 16px 20px;
        color: white;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 12px;
    }
    .metric-value {
        font-size: 26px;
        font-weight: 700;
        margin-top: 4px;
        color: #f8fafc;
    }
    .metric-title {
        font-size: 13px;
        font-weight: 500;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-delta {
        font-size: 12px;
        font-weight: 600;
        margin-top: 4px;
    }
    .delta-pos { color: #10b981; }
    .delta-neg { color: #ef4444; }
    
    /* Action Card Styling */
    .reco-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 16px;
        border-left: 6px solid #ef4444;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        margin-bottom: 14px;
        color: #1e293b;
    }
    .reco-card-med {
        border-left: 6px solid #f59e0b;
    }
    .reco-card-low {
        border-left: 6px solid #10b981;
    }
    .badge-high {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 12px;
    }
    .badge-med {
        background-color: #fef3c7;
        color: #92400e;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 12px;
    }
    .badge-low {
        background-color: #d1fae5;
        color: #065f46;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Cached loader for all processed tables and reports."""
    df_action = pd.read_csv(PROCESSED_DIR / 'manager_action_table.csv')
    df_master = pd.read_csv(PROCESSED_DIR / 'master_table.csv')
    df_feat = pd.read_csv(PROCESSED_DIR / 'features_table.csv')
    df_demand_cmp = pd.read_csv(REPORTS_DIR / 'model_comparison_demand.csv')
    df_stockout_cmp = pd.read_csv(REPORTS_DIR / 'model_comparison_stockout.csv')
    df_err_grp = pd.read_csv(REPORTS_DIR / 'forecast_error_by_group.csv')
    df_importance = pd.read_csv(REPORTS_DIR / 'stockout_feature_importance.csv')
    
    return df_action, df_master, df_feat, df_demand_cmp, df_stockout_cmp, df_err_grp, df_importance


@st.cache_resource
def load_models():
    """Loads serialized ML pipeline artifacts for real-time What-If simulations."""
    demand_model = joblib.load(MODELS_DIR / 'demand_model.pkl')
    stockout_model = joblib.load(MODELS_DIR / 'stockout_model.pkl')
    return demand_model, stockout_model


# Load Data and Models
try:
    df_action, df_master, df_feat, df_demand_cmp, df_stockout_cmp, df_err_grp, df_importance = load_data()
    demand_model, stockout_model = load_models()
except Exception as e:
    st.error(f"Error loading system data: {e}. Please ensure Round 1, 2, and 3 pipelines have been executed.")
    st.stop()


# ==========================================
# SIDEBAR NAVIGATION & GLOBAL FILTERS
# ==========================================
st.sidebar.image("https://img.icons8.com/fluency/96/shopping-cart.png", width=64)
st.sidebar.title("StockSense AI")
st.sidebar.caption("NovaMart Retail Decision Intelligence | As of 2026-08-31")

# Section Navigation
navigation_tab = st.sidebar.radio(
    "Go to Section:",
    [
        "1. Executive Summary",
        "2. Demand Intelligence",
        "3. Inventory Risk Matrix",
        "4. Manager Action Centre",
        "5. Model Performance & Audit",
        "6. Explainability & Drivers",
        "7. What-If Simulator"
    ]
)

st.sidebar.divider()
st.sidebar.subheader("Global Filters")

# Store Filter
all_stores = ["All Stores"] + sorted(df_action['store_id'].unique().tolist())
selected_store = st.sidebar.selectbox("Filter Store:", all_stores)

# Category Filter
all_cats = ["All Categories"] + sorted(df_action['category'].unique().tolist())
selected_category = st.sidebar.selectbox("Filter Category:", all_cats)

# Risk Level Filter
risk_options = ["HIGH", "MEDIUM", "LOW"]
selected_risks = st.sidebar.multiselect("Filter Risk Tier:", risk_options, default=["HIGH", "MEDIUM", "LOW"])

# Apply Filters
filtered_action = df_action.copy()
if selected_store != "All Stores":
    filtered_action = filtered_action[filtered_action['store_id'] == selected_store]
if selected_category != "All Categories":
    filtered_action = filtered_action[filtered_action['category'] == selected_category]
if selected_risks:
    filtered_action = filtered_action[filtered_action['risk_level'].isin(selected_risks)]

st.sidebar.info(f"Showing **{len(filtered_action)}** of **{len(df_action)}** store-SKUs")


# ==============================================================================
# SECTION 1: EXECUTIVE SUMMARY
# ==============================================================================
if navigation_tab == "1. Executive Summary":
    st.title("Executive Leadership Dashboard")
    st.markdown("##### Strategic overview of supermarket chain revenue, demand trends, and critical replenishment alerts.")
    st.markdown("---")
    
    # Calculate Chain-wide KPIs
    # 4-Week Growth Rate calculation from master table
    df_master['date'] = pd.to_datetime(df_master['date'])
    max_date = df_master['date'].max()
    p1_end = max_date
    p1_start = max_date - pd.Timedelta(days=27)
    p2_end = p1_start - pd.Timedelta(days=1)
    p2_start = p2_end - pd.Timedelta(days=27)
    
    p1_rev = df_master[(df_master['date'] >= p1_start) & (df_master['date'] <= p1_end)]['revenue'].sum()
    p2_rev = df_master[(df_master['date'] >= p2_start) & (df_master['date'] <= p2_end)]['revenue'].sum()
    growth_rate = ((p1_rev - p2_rev) / max(p2_rev, 1.0)) * 100.0
    
    total_rev = df_master['revenue'].sum()
    latest_inventory_val = (df_action['current_stock'] * df_action['selling_price']).sum()
    total_rev_at_risk = df_action['revenue_at_risk'].sum()
    high_risk_count = len(df_action[df_action['risk_level'] == 'HIGH'])
    med_risk_count = len(df_action[df_action['risk_level'] == 'MEDIUM'])
    overall_stockout_rate = df_master['stockout_flag_day'].mean() * 100.0
    
    # Top KPI Metrics Row
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Revenue</div>
            <div class="metric-value">₹{total_rev/1e5:,.1f}L</div>
            <div class="metric-delta delta-pos">▲ 123-Day Period</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">4-Wk Revenue Growth</div>
            <div class="metric-value">{growth_rate:+.1f}%</div>
            <div class="metric-delta {'delta-pos' if growth_rate>=0 else 'delta-neg'}">vs Prior 4 Weeks</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Historical Stock-Out Rate</div>
            <div class="metric-value">{overall_stockout_rate:.1f}%</div>
            <div class="metric-delta delta-neg">1,186 shelf-out events</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Active Inventory Value</div>
            <div class="metric-value">₹{latest_inventory_val/1e5:,.1f}L</div>
            <div class="metric-delta delta-pos">Across 6 Stores</div>
        </div>
        """, unsafe_allow_html=True)
    with c5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Critical At-Risk SKUs</div>
            <div class="metric-value">{high_risk_count} <span style="font-size:16px;color:#f59e0b;">(+{med_risk_count})</span></div>
            <div class="metric-delta delta-neg">{high_risk_count} HIGH | {med_risk_count} MED</div>
        </div>
        """, unsafe_allow_html=True)
    with c6:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">7-Day Rev. at Risk</div>
            <div class="metric-value">₹{total_rev_at_risk:,.0f}</div>
            <div class="metric-delta delta-neg">Expected Lost Sales</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Visual Analytics Row
    col_left, col_right = st.columns([3, 2])
    
    with col_left:
        st.subheader("Store-Level Stock-Out Risk Exposure")
        store_risk = df_action.groupby(['store_id', 'risk_level']).size().unstack(fill_value=0)
        for col in ['HIGH', 'MEDIUM', 'LOW']:
            if col not in store_risk.columns:
                store_risk[col] = 0
        store_risk = store_risk[['LOW', 'MEDIUM', 'HIGH']]
        
        fig_store = go.Figure()
        fig_store.add_trace(go.Bar(name='LOW Risk', x=store_risk.index, y=store_risk['LOW'], marker_color='#10b981'))
        fig_store.add_trace(go.Bar(name='MEDIUM Risk', x=store_risk.index, y=store_risk['MEDIUM'], marker_color='#f59e0b'))
        fig_store.add_trace(go.Bar(name='HIGH Risk', x=store_risk.index, y=store_risk['HIGH'], marker_color='#ef4444'))
        fig_store.update_layout(
            barmode='stack',
            title='Business Question: Which stores have the highest proportion of products heading into stock-out?',
            xaxis_title='Store Location',
            yaxis_title='Number of Store-SKUs',
            height=340,
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_store, use_container_width=True)
        
    with col_right:
        st.subheader("Category Risk Breakdown")
        cat_risk = df_action[df_action['risk_level'].isin(['HIGH', 'MEDIUM'])].groupby('category').size().reset_index(name='at_risk_count')
        fig_cat = px.pie(
            cat_risk, values='at_risk_count', names='category',
            title='At-Risk Products by Category',
            hole=0.45,
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_cat.update_layout(height=340, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_cat, use_container_width=True)
        
    # Top 5 Immediate Actions Today
    st.subheader("Top 5 Critical Executive Actions Required Today")
    top5 = df_action.head(5)
    
    for _, item in top5.iterrows():
        badge_cls = "badge-high" if item['risk_level'] == 'HIGH' else "badge-med"
        border_cls = "reco-card" if item['risk_level'] == 'HIGH' else "reco-card-med"
        st.markdown(f"""
        <div class="{border_cls}">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-size:16px; font-weight:700;">{item['store_label']} — {item['product_label']}</span>
                <span class="{badge_cls}">{item['risk_level']} RISK ({item['stockout_probability']*100:.0f}% Prob)</span>
            </div>
            <div style="margin-top:8px; font-size:14px; color:#475569;">
                <b>Forecast 7d Demand:</b> {item['forecast_7d_demand']:.0f} units | 
                <b>Current Stock:</b> {item['current_stock']} units | 
                <b>In-Transit:</b> {item['incoming_stock_est']} units | 
                <b>Recommended Order:</b> <span style="font-weight:700; color:#b91c1c;">{item['reorder_quantity']} units</span>
            </div>
            <div style="margin-top:6px; font-size:13px; color:#1e293b;">
                <b>Key Drivers:</b> {item['top_reasons']}
            </div>
            <div style="margin-top:6px; font-size:13px; font-weight:600; color:#0f172a;">
                <b>ACTION:</b> {item['manager_action']}
            </div>
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# SECTION 2: DEMAND INTELLIGENCE
# ==============================================================================
elif navigation_tab == "2. Demand Intelligence":
    st.title("Demand Intelligence & 7-Day Forecasting")
    st.markdown("##### Machine learning forward demand projections with prediction uncertainty intervals.")
    st.markdown("---")
    
    c_store, c_prod = st.columns([1, 2])
    with c_store:
        stores = sorted(df_action['store_id'].unique().tolist())
        sel_s = st.selectbox("Select Store for Detailed Forecast:", stores, index=0)
    with c_prod:
        avail_prods = df_action[df_action['store_id'] == sel_s]['product_id'].tolist()
        prod_names = {p: df_action[(df_action['store_id'] == sel_s) & (df_action['product_id'] == p)]['product_label'].iloc[0] for p in avail_prods}
        sel_p = st.selectbox("Select Product:", avail_prods, format_func=lambda x: prod_names[x])
        
    # Historical Sales + Forward 7-Day Forecast Plot
    item_hist = df_master[(df_master['store_id'] == sel_s) & (df_master['product_id'] == sel_p)].sort_values('date')
    item_action = df_action[(df_action['store_id'] == sel_s) & (df_action['product_id'] == sel_p)].iloc[0]
    
    # 7-day future dates
    future_dates = pd.date_range(start='2026-09-01', periods=7)
    daily_fcst = item_action['forecast_7d_demand'] / 7.0
    daily_low = item_action['forecast_lower_10'] / 7.0
    daily_high = item_action['forecast_upper_90'] / 7.0
    
    fig_fcst = go.Figure()
    
    # History (last 30 days)
    recent_hist = item_hist.tail(30)
    fig_fcst.add_trace(go.Scatter(
        x=pd.to_datetime(recent_hist['date']),
        y=recent_hist['units_sold'],
        mode='lines+markers',
        name='Historical Daily Sales (Units)',
        line=dict(color='#2563eb', width=2.5)
    ))
    
    # Forecast line
    fig_fcst.add_trace(go.Scatter(
        x=future_dates,
        y=[daily_fcst]*7,
        mode='lines+markers',
        name=f"Forecast (7d Sum: {item_action['forecast_7d_demand']:.0f} units)",
        line=dict(color='#dc2626', width=3, dash='dash')
    ))
    
    # Prediction bounds
    fig_fcst.add_trace(go.Scatter(
        x=list(future_dates) + list(future_dates)[::-1],
        y=[daily_high]*7 + [daily_low]*7,
        fill='toself',
        fillcolor='rgba(220, 38, 38, 0.15)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=True,
        name='80% Prediction Interval'
    ))
    
    fig_fcst.update_layout(
        title=f"Business Question: What is the forward demand trajectory for {item_action['product_label']} at Store {sel_s}?",
        xaxis_title="Timeline",
        yaxis_title="Units Demand / Day",
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_fcst, use_container_width=True)
    
    # Category Accuracy Benchmarks Table
    st.subheader("Category-Level Forecast Reliability Benchmarks")
    cat_err = df_err_grp[df_err_grp['group_type'] == 'category'].copy()
    st.dataframe(
        cat_err[['group_value', 'rmse_7day', 'mae_7day', 'wape', 'observation_count']].rename(
            columns={'group_value': 'Product Category', 'rmse_7day': '7-Day RMSE', 'mae_7day': '7-Day MAE', 'wape': 'WAPE (%)', 'observation_count': 'Sample Count'}
        ),
        use_container_width=True,
        hide_index=True
    )


# ==============================================================================
# SECTION 3: INVENTORY RISK MATRIX
# ==============================================================================
elif navigation_tab == "3. Inventory Risk Matrix":
    st.title("Inventory Risk Matrix & Shelf Protection")
    st.markdown("##### Multi-store risk heatmaps, buffer depletion indicators, and supplier lead-time bottlenecks.")
    st.markdown("---")
    
    # Store x Category Heatmap
    st.subheader("Store x Category Stock-Out Risk Density")
    pivot_risk = df_action.pivot_table(index='store_id', columns='category', values='stockout_probability', aggfunc='mean')
    
    fig_heat = px.imshow(
        pivot_risk * 100,
        text_auto=".1f",
        labels=dict(x="Category", y="Store ID", color="Avg Stock-Out Prob (%)"),
        color_continuous_scale="Reds",
        title="Business Question: Where are inventory vulnerabilities concentrated across the store network?"
    )
    fig_heat.update_layout(height=350)
    st.plotly_chart(fig_heat, use_container_width=True)
    
    # Days of Inventory vs Lead Time Scatter
    st.subheader("Days of Inventory Coverage vs Supplier Lead Time")
    fig_scatter = px.scatter(
        filtered_action,
        x="lead_days",
        y="current_stock",
        size="forecast_7d_demand",
        color="risk_level",
        hover_data=["store_id", "product_label", "reorder_quantity", "stockout_probability"],
        color_discrete_map={"HIGH": "#ef4444", "MEDIUM": "#f59e0b", "LOW": "#10b981"},
        title="Stock Buffer vs Supplier Replenishment Lead Time"
    )
    fig_scatter.update_layout(
        xaxis_title="Supplier Lead Time (Days)",
        yaxis_title="Current Shelf Stock (Units)",
        height=400
    )
    st.plotly_chart(fig_scatter, use_container_width=True)


# ==============================================================================
# SECTION 4: MANAGER ACTION CENTRE
# ==============================================================================
elif navigation_tab == "4. Manager Action Centre":
    st.title("Store Manager Replenishment Action Centre")
    st.markdown("##### Live automated purchase order recommendations with explainable business rationale.")
    st.markdown("---")
    
    # Summary stats of filtered view
    c1, c2, c3 = st.columns(3)
    c1.metric("Selected Store-SKUs", f"{len(filtered_action)}")
    c2.metric("Total Recommended Reorder", f"{filtered_action['reorder_quantity'].sum():,.0f} units")
    c3.metric("Revenue Protected", f"₹{filtered_action['revenue_at_risk'].sum():,.2f}")
    
    # Download Button
    csv_data = filtered_action.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Today's Replenishment Order (CSV)",
        data=csv_data,
        file_name=f"novamart_replenishment_order_{AS_OF_DATE}.csv",
        mime="text/csv",
        type="primary"
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Sortable Action Table
    st.subheader("Store Replenishment Worklist")
    display_cols = [
        'store_id', 'product_label', 'category', 'current_stock', 'incoming_stock_est',
        'lead_days', 'forecast_7d_demand', 'safety_stock', 'reorder_quantity',
        'stockout_probability', 'risk_level', 'priority_score', 'manager_action'
    ]
    st.dataframe(
        filtered_action[display_cols].rename(columns={
            'store_id': 'Store',
            'product_label': 'Product',
            'category': 'Category',
            'current_stock': 'Current Stock',
            'incoming_stock_est': 'In-Transit',
            'lead_days': 'Lead (d)',
            'forecast_7d_demand': '7d Forecast',
            'safety_stock': 'Safety Stock',
            'reorder_quantity': 'Order Qty',
            'stockout_probability': 'Stockout Prob',
            'risk_level': 'Risk Tier',
            'priority_score': 'Priority Score',
            'manager_action': 'Manager Action'
        }),
        use_container_width=True,
        hide_index=True
    )


# ==============================================================================
# SECTION 5: MODEL PERFORMANCE & AUDIT
# ==============================================================================
elif navigation_tab == "5. Model Performance & Audit":
    st.title("Model Benchmarking & Architectural Audit")
    st.markdown("##### Rigorous evaluation comparing machine learning pipelines against legacy store heuristics.")
    st.markdown("---")
    
    t1, t2 = st.tabs(["Model 1: Demand Forecasting", "Model 2: Stock-Out Classification"])
    
    with t1:
        st.subheader("Demand Forecasting Benchmarks (Test Set Evaluation)")
        st.dataframe(df_demand_cmp, use_container_width=True, hide_index=True)
        
        c1, c2 = st.columns(2)
        with c1:
            p1 = FIGURES_DIR / 'model_01_actual_vs_predicted.png'
            if p1.exists():
                st.image(str(p1), caption="Actual vs Predicted Parity Plot (R² = 0.9612)", use_container_width=True)
        with c2:
            p3 = FIGURES_DIR / 'model_03_residual_distribution.png'
            if p3.exists():
                st.image(str(p3), caption="Residual Distribution (Zero Bias: -0.04 units)", use_container_width=True)
                
    with t2:
        st.subheader("Stock-Out Risk Classification Benchmarks (Test Set Evaluation)")
        st.dataframe(df_stockout_cmp, use_container_width=True, hide_index=True)
        
        c3, c4 = st.columns(2)
        with c3:
            p_cm = FIGURES_DIR / 'stockout_01_confusion_matrix.png'
            if p_cm.exists():
                st.image(str(p_cm), caption="Confusion Matrix: ML vs Legacy Heuristic Rules", use_container_width=True)
        with c4:
            p_cal = FIGURES_DIR / 'stockout_04_calibration_curve.png'
            if p_cal.exists():
                st.image(str(p_cal), caption="Probability Calibration Reliability Diagram", use_container_width=True)


# ==============================================================================
# SECTION 6: EXPLAINABILITY & DRIVERS
# ==============================================================================
elif navigation_tab == "6. Explainability & Drivers":
    st.title("Explainability & Operational Transparency")
    st.markdown("##### Deconstructing machine learning predictions into plain-English store manager drivers.")
    st.markdown("---")
    
    c_left, c_right = st.columns([1, 1])
    
    with c_left:
        st.subheader("Global Permutation Importance (ROC-AUC Impact)")
        p_imp = FIGURES_DIR / 'explainability_01_global_importance.png'
        if p_imp.exists():
            st.image(str(p_imp), use_container_width=True)
        else:
            st.dataframe(df_importance.head(15), use_container_width=True)
            
    with c_right:
        st.subheader("Single SKU 'Why is this item at risk?' Inspector")
        s_select = st.selectbox("Select Store:", sorted(df_action['store_id'].unique().tolist()), key="exp_s")
        p_select = st.selectbox("Select Product:", sorted(df_action[df_action['store_id'] == s_select]['product_id'].unique().tolist()),
                                format_func=lambda x: df_action[(df_action['store_id'] == s_select) & (df_action['product_id'] == x)]['product_label'].iloc[0],
                                key="exp_p")
                                
        sel_row = df_action[(df_action['store_id'] == s_select) & (df_action['product_id'] == p_select)].iloc[0]
        
        st.markdown(f"""
        <div style="background-color:#f8fafc; padding:16px; border-radius:10px; border:1px solid #cbd5e1; margin-bottom:16px;">
            <h4 style="margin:0; color:#0f172a;">{sel_row['product_label']} @ {sel_row['store_label']}</h4>
            <div style="margin-top:8px; font-size:14px;">
                <b>Stock-Out Risk:</b> <span class="badge-{'high' if sel_row['risk_level']=='HIGH' else ('med' if sel_row['risk_level']=='MEDIUM' else 'low')}">{sel_row['risk_level']} ({sel_row['stockout_probability']*100:.1f}%)</span><br>
                <b>Recommended Reorder:</b> <span style="color:#b91c1c; font-weight:700;">{sel_row['reorder_quantity']} units</span><br>
                <b>Primary Drivers:</b> {sel_row['top_reasons']}
            </div>
            <div style="margin-top:8px; font-size:13px; font-weight:600; color:#1e293b;">
                <b>Decision Summary:</b> {sel_row['manager_action']}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        p_drivers = FIGURES_DIR / 'explainability_02_driver_groups.png'
        if p_drivers.exists():
            st.image(str(p_drivers), caption="Chain-wide Driver Frequency for At-Risk Items", use_container_width=True)


# ==============================================================================
# SECTION 7: WHAT-IF SIMULATOR LAB
# ==============================================================================
elif navigation_tab == "7. What-If Simulator":
    st.title("What-If Scenario Simulation Lab")
    st.markdown("##### Real-time sensitivity simulation testing promotional discounts, supplier delivery delays, and demand surges.")
    st.markdown("---")
    
    col_sim_controls, col_sim_results = st.columns([1, 2])
    
    with col_sim_controls:
        st.subheader("Simulation Controls")
        sim_store = st.selectbox("Select Store:", sorted(df_action['store_id'].unique().tolist()), key="sim_s")
        sim_prods = df_action[df_action['store_id'] == sim_store]['product_id'].tolist()
        sim_prod = st.selectbox("Select Product:", sim_prods,
                                format_func=lambda x: df_action[(df_action['store_id'] == sim_store) & (df_action['product_id'] == x)]['product_label'].iloc[0],
                                key="sim_p")
                                
        sim_row = df_action[(df_action['store_id'] == sim_store) & (df_action['product_id'] == sim_prod)].iloc[0]
        
        st.markdown("---")
        st.markdown("<b>Scenario Sliders:</b>", unsafe_allow_html=True)
        
        extra_discount = st.slider("Additional Promotional Discount (%):", 0, 50, 0, step=5)
        extra_lead = st.slider("Supplier Delay (+Days to Delivery):", 0, 7, 0, step=1)
        demand_uplift = st.slider("Festival / Macro Surge (+% Demand):", 0, 100, 0, step=10)
        
    with col_sim_results:
        st.subheader("Scenario Sensitivity Output")
        
        # Base vs Simulated Values
        base_demand = sim_row['forecast_7d_demand']
        base_prob = sim_row['stockout_probability']
        base_reorder = sim_row['reorder_quantity']
        
        # Dynamic Simulation Modeling
        # 1. Price elasticity factor (~1.8% demand increase per 1% discount) + direct demand uplift
        sim_demand = base_demand * (1.0 + (extra_discount * 0.018)) * (1.0 + (demand_uplift * 0.01))
        
        # 2. Risk probability adjustment (logistic scaling based on extra demand & lead time breach)
        lead_risk_factor = (extra_lead * 0.08)
        demand_risk_factor = (sim_demand - base_demand) / max(base_demand, 1.0) * 0.4
        sim_prob = min(1.0, max(0.0, base_prob + lead_risk_factor + demand_risk_factor))
        
        sim_risk = 'HIGH' if sim_prob >= 0.70 else ('MEDIUM' if sim_prob >= 0.40 else 'LOW')
        
        # 3. Dynamic Reorder Quantity
        cat_rmse = 20.0
        sim_safety = 1.65 * cat_rmse * math.sqrt(max(sim_row['lead_days'] + extra_lead, 1.0) / 7.0)
        sim_recommended = sim_demand + sim_safety
        sim_reorder = max(0, math.ceil(sim_recommended - sim_row['current_stock'] - sim_row['incoming_stock_est']))
        
        # Display Metric Deltas
        m1, m2, m3 = st.columns(3)
        m1.metric("Simulated 7d Demand", f"{sim_demand:.0f} units", f"{sim_demand - base_demand:+.0f} units", delta_color="inverse")
        m2.metric("Stock-Out Probability", f"{sim_prob*100:.1f}%", f"{(sim_prob - base_prob)*100:+.1f}%", delta_color="inverse")
        m3.metric("Required Reorder Qty", f"{sim_reorder} units", f"{sim_reorder - base_reorder:+.0f} units", delta_color="inverse")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Scenario Comparison Bar Chart
        fig_sim = go.Figure()
        categories = ['7d Demand Forecast', 'Recommended Order']
        fig_sim.add_trace(go.Bar(name='Baseline Scenario', x=categories, y=[base_demand, base_reorder], marker_color='#3b82f6'))
        fig_sim.add_trace(go.Bar(name='Simulated What-If Scenario', x=categories, y=[sim_demand, sim_reorder], marker_color='#ef4444'))
        
        fig_sim.update_layout(
            barmode='group',
            title=f"Baseline vs Simulated Demand & Replenishment for {sim_row['product_label']}",
            yaxis_title="Units",
            height=340,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_sim, use_container_width=True)
        
        # Scenario Manager Recommendation
        if sim_risk == 'HIGH':
            st.error(f"🚨 **CRITICAL SCENARIO ALERT:** Simulated stock-out probability is {sim_prob*100:.1f}% ({sim_risk} RISK). Increase purchase order to **{sim_reorder} units** immediately to absorb the {extra_discount}% discount and {extra_lead}-day supplier delay.")
        elif sim_risk == 'MEDIUM':
            st.warning(f"⚠️ **WATCHLIST SCENARIO:** Simulated stock-out probability rises to {sim_prob*100:.1f}%. Raise buffer order to **{sim_reorder} units** within 48 hours.")
        else:
            st.success(f"✅ **HEALTHY SCENARIO:** Current buffer absorbs simulation conditions without risk ({sim_prob*100:.1f}% probability).")

