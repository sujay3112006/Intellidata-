"""
StockSense End-to-End Reproducible Pipeline Orchestrator
Executes the complete project lifecycle from raw data ingestion to live manager action recommendations.
Usage:
    python src/run_pipeline.py                  # Full end-to-end training & scoring
    python src/run_pipeline.py --skip-training  # Rapid scoring using serialized models
"""

import sys
import os
import time
import argparse
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Silence loky warning
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

from src.config import (
    RAW_DIR,
    PROCESSED_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    MODELS_DIR,
    AS_OF_DATE
)


def print_stage_header(stage_number: int, stage_name: str):
    """Prints formatted banner for pipeline stages."""
    print("\n" + "=" * 80)
    print(f" STAGE {stage_number}: {stage_name.upper()}")
    print("=" * 80, flush=True)


def run_full_pipeline():
    """Executes the complete end-to-end pipeline across all 3 hackathon rounds."""
    start_time = time.time()
    print("=" * 80)
    print(" STOCKSENSE: COMPLETE END-TO-END PIPELINE ORCHESTRATOR")
    print(f" Repository: {ROOT_DIR}")
    print(f" As-Of Date: {AS_OF_DATE}")
    print("=" * 80, flush=True)
    
    # -------------------------------------------------------------
    # STAGE 1: Data Cleaning & Master Table Assembly (Round 1)
    # -------------------------------------------------------------
    print_stage_header(1, "Data Cleaning & Master Table Construction (Student 1)")
    from src.build_master import build_master_table
    df_master = build_master_table()
    print(f"[OK] Master Table assembled: {len(df_master):,} rows x {df_master.shape[1]} columns.")
    
    # -------------------------------------------------------------
    # STAGE 2: Feature Engineering & Target Building (Round 2)
    # -------------------------------------------------------------
    print_stage_header(2, "Censored Demand Correction & Feature Engineering (Student 2)")
    from src.feature_engineering import build_features_table
    df_feat = build_features_table(save_csv=True)
    print(f"[OK] Features Table built: {len(df_feat):,} rows x {df_feat.shape[1]} columns.")
    
    # -------------------------------------------------------------
    # STAGE 3: Model 1 - Demand Forecasting (Round 2)
    # -------------------------------------------------------------
    print_stage_header(3, "Model 1: 7-Day Demand Forecasting & Production Refit (Student 2)")
    import src.train_demand_model as demand_mod
    demand_mod.main()
    print("[OK] Demand Forecasting models evaluated, serialized, and live forecast generated.")
    
    # -------------------------------------------------------------
    # STAGE 4: Model 2 - Stock-Out Risk Classification (Round 3)
    # -------------------------------------------------------------
    print_stage_header(4, "Model 2: Stock-Out Classification & Calibration (Student 3)")
    import src.train_stockout_model as stockout_mod
    stockout_mod.main()
    print("[OK] Stock-Out Classifier trained, calibrated, and live probabilities computed.")
    
    # -------------------------------------------------------------
    # STAGE 5: Explainability & Decision Drivers (Round 3)
    # -------------------------------------------------------------
    print_stage_header(5, "Model Explainability & Manager Driver Grouping (Student 3)")
    import src.explainability as exp_mod
    exp_mod.main()
    print("[OK] Global permutation importance and store-SKU decision drivers generated.")
    
    # -------------------------------------------------------------
    # STAGE 6: Recommendation Engine & Manager Action Table (Round 3)
    # -------------------------------------------------------------
    print_stage_header(6, "Decision Intelligence & Manager Action Table (Student 3)")
    import src.recommendation as reco_mod
    reco_mod.main()
    print("[OK] Manager replenishment action table and business impact backtest generated.")
    
    # Final Pipeline Summary
    total_time = round(time.time() - start_time, 2)
    print("\n" + "=" * 80)
    print(f" PIPELINE EXECUTION SUCCESSFUL: Finished in {total_time} seconds ({total_time/60:.2f} mins).")
    print("=" * 80)
    print("Key Generated Deliverables:")
    print(f"  - Master Data Table:        {PROCESSED_DIR / 'master_table.csv'}")
    print(f"  - Features Table:           {PROCESSED_DIR / 'features_table.csv'}")
    print(f"  - Demand Model Artifact:    {MODELS_DIR / 'demand_model.pkl'}")
    print(f"  - Stock-Out Model Artifact: {MODELS_DIR / 'stockout_model.pkl'}")
    print(f"  - Live Forward Forecast:    {PROCESSED_DIR / 'forecast_latest.csv'}")
    print(f"  - Stock-Out Risk Table:     {PROCESSED_DIR / 'stockout_predictions.csv'}")
    print(f"  - Manager Action Table:     {PROCESSED_DIR / 'manager_action_table.csv'}")
    print(f"  - Business Impact Report:   {REPORTS_DIR / 'business_impact.md'}")
    print("=" * 80, flush=True)


def run_fast_inference():
    """Rapid live scoring mode using pre-serialized models without re-training."""
    start_time = time.time()
    print("=" * 80)
    print(" STOCKSENSE: FAST INFERENCE MODE (--skip-training)")
    print("=" * 80, flush=True)
    
    feat_path = PROCESSED_DIR / 'features_table.csv'
    demand_model_path = MODELS_DIR / 'demand_model.pkl'
    stockout_model_path = MODELS_DIR / 'stockout_model.pkl'
    
    if not feat_path.exists() or not demand_model_path.exists() or not stockout_model_path.exists():
        raise FileNotFoundError("Missing pre-trained models or features table. Please run full pipeline first.")
        
    print("Loading pre-trained models and features...", flush=True)
    import src.recommendation as reco_mod
    reco_mod.generate_manager_action_table()
    
    total_time = round(time.time() - start_time, 2)
    print(f"\n[OK] Fast inference complete in {total_time}s. Manager Action Table updated at {PROCESSED_DIR / 'manager_action_table.csv'}")


def main():
    parser = argparse.ArgumentParser(description="StockSense Complete Pipeline Runner")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip model training and run rapid scoring using saved model artifacts.")
    args = parser.parse_args()
    
    if args.skip_training:
        run_fast_inference()
    else:
        run_full_pipeline()


if __name__ == '__main__':
    main()
