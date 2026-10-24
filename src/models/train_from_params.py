#!/usr/bin/env python3
"""
train_from_params.py
====================
Loads pre-computed best hyperparameters from a JSON file, trains ALL models,
evaluates them on the unseen Test set, and generates a Data-Size Learning Curve
for every single model.

Inputs : data/processed/model_ready.csv
         models/pv_model/best_params.json
Outputs: models/pv_model/final_evaluation_metrics.csv
         models/pv_model/learning_curves/*_data_size_curve.png
"""

import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.base import clone

# Import shared baseline variables
from model_baselines import metrics, skill, FEATURES, TARGET, CLIP

# ----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_READY  = PROJECT_ROOT / "data" / "processed" / "model_ready.csv"
OUTPUT_DIR   = PROJECT_ROOT / "models" / "pv_model"
PARAMS_FILE  = OUTPUT_DIR / "best_params.json"
CURVES_DIR   = OUTPUT_DIR / "learning_curves"
EXCLUDE      = [12293]
# ----------------------------------------------------------------------------

def get_configured_models(params_dict):
    """Instantiates models using the pre-computed best parameters."""
    models = {
        "Linear Regression": (LinearRegression(), True)
    }
    
    if "Ridge Regression" in params_dict:
        models["Ridge Regression"] = (Ridge(**params_dict["Ridge Regression"]), True)
        
    if "Decision Tree" in params_dict:
        dt_params = params_dict["Decision Tree"].copy()
        if "random_state" not in dt_params: dt_params["random_state"] = 0
        models["Decision Tree"] = (DecisionTreeRegressor(**dt_params), True)
        
    if "Random Forest" in params_dict:
        rf_params = params_dict["Random Forest"].copy()
        if "random_state" not in rf_params: rf_params["random_state"] = 0
        if "n_jobs" not in rf_params: rf_params["n_jobs"] = -1
        models["Random Forest"] = (RandomForestRegressor(**rf_params), True)
        
    if "Gradient Boosting" in params_dict:
        gb_params = params_dict["Gradient Boosting"].copy()
        if "random_state" not in gb_params: gb_params["random_state"] = 0
        models["Gradient Boosting"] = (HistGradientBoostingRegressor(**gb_params), False)
        
    try:
        from xgboost import XGBRegressor
        if "XGBoost" in params_dict:
            xgb_params = params_dict["XGBoost"].copy()
            if "random_state" not in xgb_params: xgb_params["random_state"] = 0
            if "n_jobs" not in xgb_params: xgb_params["n_jobs"] = -1
            if "tree_method" not in xgb_params: xgb_params["tree_method"] = "hist"
            models["XGBoost"] = (XGBRegressor(**xgb_params), False)
    except ImportError:
        print("  (XGBoost not installed - skipping)")

    try:
        from lightgbm import LGBMRegressor
        if "LightGBM" in params_dict:
            lgb_params = params_dict["LightGBM"].copy()
            if "random_state" not in lgb_params: lgb_params["random_state"] = 0
            if "n_jobs" not in lgb_params: lgb_params["n_jobs"] = -1
            if "verbose" not in lgb_params: lgb_params["verbose"] = -1
            models["LightGBM"] = (LGBMRegressor(**lgb_params), False)
    except ImportError:
        print("  (LightGBM not installed - skipping)")

    return models

def generate_data_size_learning_curve(name, model, X_train, y_train, X_val, y_val):
    """
    Trains the model on 20%, 40%, 60%, 80%, and 100% of the training data
    to see how performance scales with data volume.
    """
    print(f"    -> Generating Data-Size Learning Curve for {name}...")
    fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
    train_errors = []
    val_errors = []
    
    total_rows = len(X_train)
    
    for frac in fractions:
        subset_size = int(total_rows * frac)
        
        # We take the *most recent* subset of training data to maintain temporal relevance
        # relative to the validation set that immediately follows it.
        X_sub = X_train.iloc[-subset_size:]
        y_sub = y_train.iloc[-subset_size:]
        
        clone_model = clone(model)
        clone_model.fit(X_sub, y_sub)
        
        # Evaluate on the training subset
        train_preds = np.clip(clone_model.predict(X_sub), 0, CLIP)
        train_rmse = np.sqrt(mean_squared_error(y_sub, train_preds))
        train_errors.append(train_rmse)
        
        # Evaluate on the fixed validation set
        val_preds = np.clip(clone_model.predict(X_val), 0, CLIP)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        val_errors.append(val_rmse)
        
    plt.figure(figsize=(9, 6))
    plt.plot(fractions, train_errors, label="Train Error", color="#1f77b4", marker='o', linewidth=2)
    plt.plot(fractions, val_errors, label="Validation Error", color="#ff7f0e", marker='s', linewidth=2)
    
    plt.title(f"{name} Learning Curve (Data Volume vs Error)")
    plt.xlabel("Fraction of Training Data Used")
    plt.ylabel("RMSE")
    plt.xticks(fractions, [f"{int(f*100)}%" for f in fractions])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(CURVES_DIR / f"{name.lower().replace(' ', '_')}_data_size_curve.png", dpi=120)
    plt.close()

def main():
    if not MODEL_READY.exists():
        print(f"Error: {MODEL_READY} not found.")
        return
        
    if not PARAMS_FILE.exists():
        print(f"Error: {PARAMS_FILE} not found. You must run the benchmark script first to generate parameters.")
        return
        
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CURVES_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(PARAMS_FILE, "r") as f:
        best_params = json.load(f)
        
    df = pd.read_csv(MODEL_READY)
    if EXCLUDE:
        df = df[~df.system_id.isin(EXCLUDE)]
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Europe/Brussels")
    df = df.sort_values(["system_id", "timestamp"]).reset_index(drop=True)
    feats = [c for c in FEATURES if c in df.columns]
    
    print(f"\n==================================================================")
    print(f"FINAL EVALUATION RUN | {len(df):,} rows | {df.system_id.nunique()} systems")
    print(f"==================================================================\n")

    # Split: Train, Val, Test
    tr = df[df.split == "train"]
    va = df[df.split == "val"]
    te = df[df.split == "test"]
    
    # Combined for the final fit
    tr_va = pd.concat([tr, va])

    Xtr, ytr = tr[feats], tr[TARGET]
    Xva, yva = va[feats], va[TARGET]
    Xtr_va, ytr_va = tr_va[feats], tr_va[TARGET]
    Xte, yte, cte = te[feats], te[TARGET], te["capacity_kwp"]
    
    med = Xtr.median()
    Xtr_i, Xva_i = Xtr.fillna(med), Xva.fillna(med)
    Xtr_va_i, Xte_i = Xtr_va.fillna(med), Xte.fillna(med)

    # Baselines
    prev = df[["system_id", "timestamp", TARGET]].copy()
    prev["timestamp"] += pd.Timedelta(days=1)
    prev = prev.rename(columns={TARGET: "persist"})
    te = te.merge(prev, on=["system_id", "timestamp"], how="left")
    
    mask = te["persist"].notna()               
    yte_m, cte_m = te.loc[mask, TARGET], te.loc[mask, "capacity_kwp"]
    persist = te.loc[mask, "persist"]
    
    csrc = "poa_global_fc" if "poa_global_fc" in te.columns else "clearsky_ghi"
    clearsky = (te.loc[mask, csrc] / 1000.0).clip(0, CLIP)
    rmse_persist = metrics(yte_m, persist, cte_m)["RMSE"]

    results = {}
    
    def record(name, pred, secs=np.nan):
        m = metrics(yte_m, np.clip(pred, 0, CLIP), cte_m)
        m["skill"] = skill(m["RMSE"], rmse_persist)
        m["train_s"] = secs
        results[name] = m
        print(f"  [FINAL TEST] {name:18s} RMSE: {m['RMSE']:.4f} | R2: {m['R2']:.3f} | Skill: {m['skill']:+.3f}")

    print("--- BASELINES ---")
    record("Persistence", persist.values, 0.0)
    record("Clear-sky proxy", clearsky.values, 0.0)
    print("\n--- TRAINING FROM BEST PARAMETERS ---")

    models_to_run = get_configured_models(best_params)

    for name, (model, needs_impute) in models_to_run.items():
        print(f"\nProcessing {name}...")
        t0 = time.time()
        
        # Select appropriate dataset
        X_train_curve = Xtr_i if needs_impute else Xtr
        X_val_curve = Xva_i if needs_impute else Xva
        X_train_final = Xtr_va_i if needs_impute else Xtr_va
        X_test_final = Xte_i.loc[mask.values] if needs_impute else Xte.loc[mask.values]

        # 1. Generate the Universal Learning Curve (Data Volume Scaling)
        # Note: Random Forest takes a very long time, so this may take a few minutes.
        generate_data_size_learning_curve(name, model, X_train_curve, ytr, X_val_curve, yva)

        # 2. Fit the final model on all available historical data (Train + Val)
        print(f"    -> Fitting final model on complete history...")
        model.fit(X_train_final, ytr_va)
        
        # 3. Predict on Test Data
        pred = model.predict(X_test_final)
        record(name, pred, time.time() - t0)

    res = pd.DataFrame(results).T
    res.index.name = "model"
    res.to_csv(OUTPUT_DIR / "final_evaluation_metrics.csv")
    
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print("\n================ FINAL EVALUATION SUMMARY (Test Set, Daylight) ================")
    print(res[["MAE", "RMSE", "R2", "skill", "train_s"]].to_string())
    
    best = res.drop(["Persistence", "Clear-sky proxy"], errors="ignore")["RMSE"].idxmin()
    print(f"\n🏆 WINNING MODEL: {best.upper()} 🏆\n")
    print(f"Artifacts saved -> {OUTPUT_DIR.name}/final_evaluation_metrics.csv")
    print(f"Learning Curves saved -> {CURVES_DIR.name}/")

if __name__ == "__main__":
    main()