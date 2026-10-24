# Relative Path: src/models/model_baselines.py
#!/usr/bin/env python3
"""
model_baselines.py
==================
UPDATES:
  - Switched Objective Function to Quantile Regression (alpha=0.25). 
    For Home Energy Management (EMS), under-forecasting PV generation is safer 
    than over-forecasting (which leads to dead batteries and expensive peak grid imports).
  - Updated feature list to match the new PR (Performance Ratio) lags.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_READY  = PROJECT_ROOT / "data" / "processed" / "model_ready.csv"
OUTPUT_DIR   = PROJECT_ROOT / "models" / "pv_model"
TARGET       = "target_norm"
CLIP         = 1.2

FEATURES = [
    # The Heavy Hitters (Irradiance directly mapped to the panel)
    "poa_global_fc", "poa_global_fc2", "clearsky_ghi",
    
    # Time and Season Proxies (Kept the strongest, dropped the redundant math)
    "sun_elevation", "sun_azimuth", "hour_cos",
    
    # The only raw weather components that survived the Permutation check
    "dni_fc", "cloud_fc2", 
    
    # Static Hardware Metadata
    "capacity_kwp", "tilt_deg", "azimuth_deg",
    
    # History & Shading Proxies
    "lag_pr_same_slot_d2"
]
# FEATURES = [
#     # The Heavy Hitters (Irradiance directly mapped to the panel)
#     "poa_global_fc",  "clearsky_ghi",
    
#     # Time and Season Proxies (Kept the strongest, dropped the redundant math)
#     "sun_elevation", "sun_azimuth", "hour_cos",
    
#     # The only raw weather components that survived the Permutation check
#     "dni_fc", "cloud_fc", 
    
#     # Static Hardware Metadata
#     "capacity_kwp", "tilt_deg", "azimuth_deg",
    
#     # History & Shading Proxies
#     "lag_pr_same_slot_d2"
# ]

# FEATURES = [
#     # The Heavy Hitters (Irradiance mapped to the panel)
#     "poa_global_fc", "poa_global_fc2", "clearsky_ghi",
    
#     # Time and Season Proxies (Kept the strongest, dropped the redundant math)
#     "sun_azimuth", "hour_cos", 
    
#     # Weather
#     "dni_fc", "cloud_fc2",
    
#     # Static Hardware Metadata
#     "capacity_kwp", "tilt_deg", "azimuth_deg",
    
#     # History & Shading Proxies
#     "lag_pr_same_slot_d2", "lag_pr_recent_day_mean"
# ]

class MultiQuantileLGBM:
    """Wrapper to train multiple LightGBM quantile models simultaneously."""
    def __init__(self, quantiles=[0.25, 0.50, 0.90], **params):
        self.quantiles = quantiles
        self.params = params
        self.models = {}

    def fit(self, X, y, eval_set=None):
        import lightgbm as lgb
        for q in self.quantiles:
            print(f"  -> Training P{int(q*100)} model...")
            model = lgb.LGBMRegressor(objective='quantile', alpha=q, **self.params)
            if eval_set:
                model.fit(X, y, eval_set=eval_set, callbacks=[lgb.early_stopping(50, verbose=False)])
            else:
                model.fit(X, y)
            self.models[f"pred_model_p{int(q*100)}"] = model

    def predict(self, X):
        return {name: np.clip(model.predict(X), 0, CLIP) for name, model in self.models.items()}

def make_model():
    """Returns the Optuna-tuned Multi-Quantile LightGBM."""
    params = {
        'n_estimators': 800,  # Bumped slightly since we use early stopping
        'learning_rate': 0.0723,
        'num_leaves': 143,
        'max_depth': 17,
        'min_child_samples': 54,
        'subsample': 0.9325,
        'colsample_bytree': 0.5395,
        'reg_alpha': 0.0056,
        'reg_lambda': 0.5139,
        'random_state': 0,
        'n_jobs': -1
    }
    return ("multi_lgbm", MultiQuantileLGBM(quantiles=[0.25, 0.50, 0.90], **params))

def metrics(actual, pred, cap_kwp, alpha=0.5):
    a, p = np.asarray(actual, float), np.asarray(pred, float)
    err = p - a
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    mbe = float(np.mean(err))
    
    pinball = float(np.mean(np.where(p > a, (1 - alpha) * (p - a), alpha * (a - p))))
    
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((a - a.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    mae_w = float(np.mean(np.abs(err) * np.asarray(cap_kwp, float) * 1000))
    return {"MAE": mae, "RMSE": rmse, "MBE": mbe, "R2": r2, "Pinball": pinball, "MAE_W": mae_w}

def skill(rmse_model, rmse_ref):
    return 1.0 - rmse_model / rmse_ref if rmse_ref > 0 else np.nan

def main():
    if not MODEL_READY.exists():
        print(f"Error: Could not find {MODEL_READY}. Did you run make_model_ready.py?")
        return
        
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    EXCLUDE=[12293,81376]
    df = pd.read_csv(MODEL_READY)
    df = df[~df.system_id.isin(EXCLUDE)].reset_index(drop=True) 
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Europe/Brussels")
    feats = [c for c in FEATURES if c in df.columns]
    
    prev = df[["system_id", "timestamp", TARGET]].copy()
    prev["timestamp"] = prev["timestamp"] + pd.Timedelta(days=1)
    prev = prev.rename(columns={TARGET: "pred_persistence"})
    df = df.merge(prev, on=["system_id", "timestamp"], how="left")

    src_col = "poa_global_fc" if "poa_global_fc" in df.columns else "clearsky_ghi"
    df["pred_clearsky"] = (df[src_col] / 1000.0).clip(0, CLIP)

    tr, va, te = (df[df.split == s] for s in ["train", "val", "test"])
    name, model = make_model()
    
    print(f"\nTraining Multi-Quantile LightGBM on {len(tr):,} rows ...")
    
    if name == "multi_lgbm" and len(va):
        model.fit(tr[feats], tr[TARGET], eval_set=[(va[feats], va[TARGET])])
    else:
        model.fit(tr[feats], tr[TARGET])
        
    predictions = model.predict(df[feats])
    for col_name, pred_values in predictions.items():
        df.loc[:, col_name] = pred_values

    test = df[df.split == "test"].copy().dropna(subset=["pred_persistence"])
    rows = {}
    
    rows["Persistence"] = metrics(test[TARGET], test["pred_persistence"], test["capacity_kwp"], alpha=0.5)
    rows["Clear-sky proxy"] = metrics(test[TARGET], test["pred_clearsky"], test["capacity_kwp"], alpha=0.5)
    
    rows["Model (P25 - Battery)"] = metrics(test[TARGET], test["pred_model_p25"], test["capacity_kwp"], alpha=0.25)
    rows["Model (P50 - Manager)"] = metrics(test[TARGET], test["pred_model_p50"], test["capacity_kwp"], alpha=0.50)
    rows["Model (P90 - Export)"]  = metrics(test[TARGET], test["pred_model_p90"], test["capacity_kwp"], alpha=0.90)
        
    mtab = pd.DataFrame(rows).T

    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print("\n================ TEST METRICS ============")
    print(mtab[["MAE", "RMSE", "MBE", "R2", "Pinball", "MAE_W"]].to_string())

    keep = ["system_id", "timestamp", TARGET, "target_power_w", "capacity_kwp",
            "pred_persistence", "pred_clearsky"] + list(predictions.keys())
    test[keep].to_csv(OUTPUT_DIR / "predictions_test.csv", index=False)
    mtab.to_csv(OUTPUT_DIR / "metrics.csv")

if __name__ == "__main__":
    main()