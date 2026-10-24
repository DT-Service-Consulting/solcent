#!/usr/bin/env python3
"""
residual_xgboost.py
===================
An A/B test to determine if Residual Learning (predicting the Cloud Penalty)
outperforms Standard Direct Learning for Day-Ahead Solar Forecasting.

Methodology:
- Standard: XGBoost(features) -> Target
- Residual: XGBoost(features) -> (ClearSky - Target)
            Prediction = ClearSky - XGBoost_Output
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Append path so we can import our baseline metrics and features
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "src" / "models"))

from model_baselines import metrics, FEATURES, TARGET, CLIP
from xgboost import XGBRegressor

MODEL_READY = PROJECT_ROOT / "data" / "processed" / "model_ready.csv"
EXCLUDE = [12293]

def main():
    if not MODEL_READY.exists():
        print(f"Error: {MODEL_READY} not found.")
        return

    print("Loading data...")
    df = pd.read_csv(MODEL_READY)
    
    if EXCLUDE:
        df = df[~df.system_id.isin(EXCLUDE)]
        
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Europe/Brussels")
    df = df.sort_values(["system_id", "timestamp"]).reset_index(drop=True)
    
    # We only care about daylight rows
    df = df[df["is_daylight"] == True].copy()
    
    feats = [c for c in FEATURES if c in df.columns]
    print(f"Dataset ready: {len(df):,} rows using {len(feats)} features.\n")

    # 1. Define the Physics Proxy (The "Perfect" Curve)
    # Using clearsky_ghi / 1000 as our base normalized perfect generation
    df["clear_sky_proxy"] = (df["clearsky_ghi"] / 1000.0).clip(0, CLIP)
    
    # 2. Define the Residual Target (The "Cloud Penalty")
    # If perfect sky is 1.0 and we only generate 0.4, the penalty is 0.6
    df["cloud_penalty"] = df["clear_sky_proxy"] - df[TARGET]

    # Split the dataset
    train_df = df[df["split"].isin(["train", "val"])] # Combine train/val for max data
    test_df = df[df["split"] == "test"]
    
    # Exclude rows where persistence is null to ensure a fair test set comparison
    prev = df[["system_id", "timestamp", TARGET]].copy()
    prev["timestamp"] += pd.Timedelta(days=1)
    prev = prev.rename(columns={TARGET: "persist"})
    test_df = test_df.merge(prev, on=["system_id", "timestamp"], how="left")
    test_df = test_df[test_df["persist"].notna()]

    # Extract X and y matrices
    X_train = train_df[feats]
    y_train_std = train_df[TARGET]
    y_train_res = train_df["cloud_penalty"]
    
    X_test = test_df[feats]
    y_test_actual = test_df[TARGET]
    caps_test = test_df["capacity_kwp"]
    clear_sky_test = test_df["clear_sky_proxy"]

    # Shared XGBoost Parameters (Optimized for speed and accuracy)
    xgb_params = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 7,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_jobs": -1,
        "random_state": 42,
        "tree_method": "hist"
    }

    # ==========================================
    # MODEL A: Standard Direct Learning
    # ==========================================
    print("Training Model A: Standard XGBoost (Predicting Raw Generation)...")
    model_std = XGBRegressor(**xgb_params)
    model_std.fit(X_train, y_train_std)
    
    pred_std = model_std.predict(X_test)
    pred_std = np.clip(pred_std, 0, CLIP)

    # ==========================================
    # MODEL B: Residual Learning
    # ==========================================
    print("Training Model B: Residual XGBoost (Predicting Cloud Penalty)...")
    model_res = XGBRegressor(**xgb_params)
    model_res.fit(X_train, y_train_res)
    
    # Inference: Subtract the predicted penalty from tomorrow's perfect clear-sky
    pred_penalty = model_res.predict(X_test)
    pred_res = clear_sky_test - pred_penalty
    pred_res = np.clip(pred_res, 0, CLIP)

    # ==========================================
    # EVALUATION
    # ==========================================
    metrics_std = metrics(y_test_actual, pred_std, caps_test)
    metrics_res = metrics(y_test_actual, pred_res, caps_test)
    
    print("\n" + "="*55)
    print("RESIDUAL LEARNING A/B TEST RESULTS (Test Set, Daylight)")
    print("="*55)
    print(f"{'Metric':<15} | {'Model A (Standard)':<20} | {'Model B (Residual)':<20}")
    print("-" * 55)
    
    for m in ["RMSE", "MAE", "MBE", "R2"]:
        val_a = metrics_std[m]
        val_b = metrics_res[m]
        diff = val_b - val_a
        
        # Format string nicely
        str_a = f"{val_a:.4f}"
        str_b = f"{val_b:.4f}"
        
        # Add visual indicator of who won
        if m in ["RMSE", "MAE"]:
            winner = "B wins" if val_b < val_a else "A wins"
        elif m == "R2":
            winner = "B wins" if val_b > val_a else "A wins"
        else:
            winner = "-"
            
        print(f"{m:<15} | {str_a:<20} | {str_b:<15} ({winner})")
    print("="*55)

if __name__ == "__main__":
    main()