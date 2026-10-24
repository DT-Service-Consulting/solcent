#!/usr/bin/env python3
"""
rfecv_benchmark.py (V2)
=======================
An automated RFECV (Recursive Feature Elimination with Cross-Validation) Arena.
Runs against the V2 features and outputs to rfecv_benchmark_results_v2.csv.
"""

import warnings
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.feature_selection import RFECV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.base import clone

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None

warnings.filterwarnings("ignore")

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "src" / "models"))
from model_baselines import metrics, skill, TARGET, CLIP

MODEL_READY  = PROJECT_ROOT / "data" / "processed" / "model_ready_v2.csv"
OUTPUT_DIR   = PROJECT_ROOT / "models" / "feature_analysis"

NON_FEATURES = {
    "system_id", "timestamp", "target_norm", "target_power_w", 
    "is_daylight", "is_valid", "split", "pred_persistence", 
    "pred_clearsky", "pred_model", "target_pr"
}

def get_models_for_rfecv():
    """Returns models configured for the RFECV process."""
    models = {
        "Linear Regression": (LinearRegression(), True),
        "Ridge Regression": (Ridge(alpha=10.0), True),
        "Decision Tree": (DecisionTreeRegressor(max_depth=15, min_samples_leaf=50, random_state=42), True),
        "Random Forest": (RandomForestRegressor(n_estimators=50, max_depth=15, n_jobs=-1, random_state=42), True)
    }
    
    if XGBRegressor is not None:
        models["XGBoost"] = (XGBRegressor(
            n_estimators=150, learning_rate=0.05, max_depth=7, 
            n_jobs=-1, random_state=42, tree_method="hist", 
            importance_type="gain" 
        ), False)
        
    if LGBMRegressor is not None:
        models["LightGBM"] = (LGBMRegressor(
            n_estimators=150, learning_rate=0.05, num_leaves=31,
            n_jobs=-1, random_state=42, verbose=-1,
            importance_type="gain"
        ), False)
        
    return models

def main():
    if not MODEL_READY.exists():
        print(f"Error: {MODEL_READY} not found.")
        return
        
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(MODEL_READY)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Europe/Brussels")
    df = df.sort_values(["system_id", "timestamp"]).reset_index(drop=True)
    
    df = df[df["is_daylight"] == True].copy()
    
    feats = [c for c in df.columns if c not in NON_FEATURES and df[c].dtype in ['float64', 'float32', 'int64', 'int32']]
    
    print(f"\n==================================================================")
    print(f"RFECV BENCHMARK ARENA | Starting with {len(feats)} V2 base features")
    print(f"==================================================================\n")

    tr_va = df[df.split.isin(["train", "val"])] 
    te = df[df.split == "test"]

    Xtr_va, ytr_va = tr_va[feats], tr_va[TARGET]
    Xte, yte, cte = te[feats], te[TARGET], te["capacity_kwp"]
    
    med = Xtr_va.median()
    Xtr_va_i, Xte_i = Xtr_va.fillna(med), Xte.fillna(med)

    prev = df[["system_id", "timestamp", TARGET]].copy()
    prev["timestamp"] += pd.Timedelta(days=1)
    prev = prev.rename(columns={TARGET: "persist"})
    te = te.merge(prev, on=["system_id", "timestamp"], how="left")
    
    mask = te["persist"].notna()               
    yte_m, cte_m = te.loc[mask, TARGET], te.loc[mask, "capacity_kwp"]
    persist = te.loc[mask, "persist"]
    rmse_persist = metrics(yte_m, persist, cte_m)["RMSE"]

    results = []
    models_to_run = get_models_for_rfecv()
    cv = TimeSeriesSplit(n_splits=3)

    for name, (model, needs_impute) in models_to_run.items():
        print(f"\n--- Running RFECV for {name} ---")
        t0 = time.time()
        
        X_train = Xtr_va_i if needs_impute else Xtr_va
        X_test = Xte_i.loc[mask.values] if needs_impute else Xte.loc[mask.values]
        
        selector = RFECV(
            estimator=model,
            step=1,
            cv=cv,
            scoring='neg_root_mean_squared_error',
            min_features_to_select=3,
            n_jobs=-1
        )
        
        print("  -> Executing recursive elimination...")
        selector.fit(X_train, ytr_va)
        
        optimal_num = selector.n_features_
        selected_features = list(np.array(feats)[selector.support_])
        dropped_features = list(set(feats) - set(selected_features))
        
        print(f"  -> Optimal features: {optimal_num} / {len(feats)}")
        if dropped_features:
            print(f"  -> Dropped features: {dropped_features}")
        else:
            print(f"  -> Model kept ALL features.")
            
        final_model = clone(model)
        
        if name in ["XGBoost", "LightGBM"]:
            split_idx = int(len(X_train) * 0.8)
            X_t, y_t = X_train[selected_features].iloc[:split_idx], ytr_va.iloc[:split_idx]
            X_v, y_v = X_train[selected_features].iloc[split_idx:], ytr_va.iloc[split_idx:]
            
            if name == "XGBoost":
                final_model.set_params(early_stopping_rounds=20)
                final_model.fit(X_t, y_t, eval_set=[(X_v, y_v)], verbose=False)
            elif name == "LightGBM":
                import lightgbm as lgb
                final_model.fit(X_t, y_t, eval_set=[(X_v, y_v)], callbacks=[lgb.early_stopping(20, verbose=False)])
        else:
            final_model.fit(X_train[selected_features], ytr_va)
            
        pred = final_model.predict(X_test[selected_features])
        m = metrics(yte_m, np.clip(pred, 0, CLIP), cte_m)
        m["skill"] = skill(m["RMSE"], rmse_persist)
        
        print(f"  -> Final RMSE: {m['RMSE']:.4f} | R2: {m['R2']:.3f} | Time: {time.time()-t0:.1f}s")
        
        results.append({
            "model": name,
            "optimal_features": optimal_num,
            "dropped_features": " | ".join(dropped_features) if dropped_features else "None",
            "RMSE": m["RMSE"],
            "MAE": m["MAE"],
            "R2": m["R2"],
            "skill": m["skill"]
        })

    res_df = pd.DataFrame(results).set_index("model")
    res_df = res_df.sort_values("RMSE")
    res_df.to_csv(OUTPUT_DIR / "rfecv_benchmark_results_v2.csv")
    
    print("\n================ FINAL RFECV BENCHMARK SUMMARY ================")
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print(res_df[["optimal_features", "RMSE", "R2", "dropped_features"]].to_string())
    print(f"\nResults saved to -> {OUTPUT_DIR.name}/rfecv_benchmark_results_v2.csv")

if __name__ == "__main__":
    main()