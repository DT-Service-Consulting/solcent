#!/usr/bin/env python3
"""
tune_lightgbm.py
================
Uses Bayesian Optimization (Optuna) to find the absolute best hyperparameters 
for the day-ahead solar forecasting LightGBM model.
"""

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_READY  = PROJECT_ROOT / "data" / "processed" / "model_ready.csv"
OUTPUT_DIR   = PROJECT_ROOT / "models" / "pv_model"
TARGET       = "target_norm"
CLIP         = 1.2
EXCLUDE      = [12293]

# The newly optimized 13-feature list
FEATURES = [
    "poa_global_fc", "poa_global_fc2", "clearsky_ghi",
    "sun_elevation", "sun_azimuth", "hour_cos",
    "dni_fc", "cloud_fc2", 
    "capacity_kwp", "tilt_deg", "azimuth_deg",
    "lag_pr_same_slot_d2", "lag_pr_recent_day_mean"
]
# ----------------------------------------------------------------------------

def load_data():
    """Loads and splits the dataset."""
    df = pd.read_csv(MODEL_READY)
    df = df[~df.system_id.isin(EXCLUDE)].reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Europe/Brussels")
    
    # We only train/tune on valid daylight hours where actual generation is possible
    df = df[(df["is_valid"] == True) & (df["is_daylight"] == True)].copy()
    
    # Fill any straggling NaNs in the lags with 0 (for the first 2 days of a system)
    df = df.fillna(0)

    tr = df[df.split == "train"]
    va = df[df.split == "val"]
    te = df[df.split == "test"]
    
    return tr, va, te

def objective(trial, tr, va):
    """
    Optuna objective function. 
    Suggests hyperparameters, trains the model, and returns the Validation RMSE.
    """
    # 1. Define the hyperparameter search space
    param = {
        "objective": "quantile",
        "alpha": 0.25,  # Keep the P25 safety buffer constraint
        "metric": "quantile",
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
        "n_estimators": 1000, # Max trees (Early stopping will halt it before this)
        
        # Bayesian Search Bounds:
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "max_depth": trial.suggest_int("max_depth", 5, 20),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }

    # 2. Initialize and train the model
    model = lgb.LGBMRegressor(**param)
    
    model.fit(
        tr[FEATURES], tr[TARGET],
        eval_set=[(va[FEATURES], va[TARGET])],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
    )

    # 3. Evaluate on the Validation Set
    preds = np.clip(model.predict(va[FEATURES]), 0, CLIP)
    rmse = np.sqrt(mean_squared_error(va[TARGET], preds))
    
    # Optuna minimizes the returned value
    return rmse

def main():
    if not MODEL_READY.exists():
        print(f"Error: Could not find {MODEL_READY}.")
        return

    print("Loading data...")
    tr, va, te = load_data()
    print(f"Data Loaded: Train({len(tr)}), Val({len(va)}), Test({len(te)})")
    print(f"Tuning LightGBM on {len(FEATURES)} features...\n")

    # Create an Optuna study to MINIMIZE the RMSE
    study = optuna.create_study(direction="minimize", study_name="LightGBM_P25_Tuning")
    
    # Run 50 trials (Usually takes 5 to 10 minutes)
    study.optimize(lambda trial: objective(trial, tr, va), n_trials=50)

    print("\n" + "="*50)
    print("OPTIMIZATION COMPLETE")
    print("="*50)
    
    best_params = study.best_params
    print("\nBEST HYPERPARAMETERS FOUND:")
    for key, value in best_params.items():
        if isinstance(value, float):
            print(f"  {key:>20}: {value:.4f}")
        else:
            print(f"  {key:>20}: {value}")

    print("\nTraining final model using these optimal parameters...")
    
    # Rebuild the final parameters map
    final_params = {
        "objective": "quantile",
        "alpha": 0.25,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
        "n_estimators": 1000,
        **best_params
    }
    
    best_model = lgb.LGBMRegressor(**final_params)
    best_model.fit(
        tr[FEATURES], tr[TARGET],
        eval_set=[(va[FEATURES], va[TARGET])],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    
    print("\nEvaluating the BEST model on the Unseen Test Set...")
    preds = np.clip(best_model.predict(te[FEATURES]), 0, CLIP)
    
    actuals = te[TARGET].values
    rmse = np.sqrt(mean_squared_error(actuals, preds))
    mbe = np.mean(preds - actuals)
    
    ss_res = np.sum((actuals - preds)**2)
    ss_tot = np.sum((actuals - np.mean(actuals))**2)
    r2 = 1.0 - ss_res / ss_tot
    
    mae_w = np.mean(np.abs(preds - actuals) * te["capacity_kwp"] * 1000)

    print("\n" + "="*50)
    print("FINAL OPTIMIZED TEST METRICS (LightGBM P25)")
    print("="*50)
    print(f"  RMSE  : {rmse:.4f}")
    print(f"  R^2   : {r2:.4f}")
    print(f"  MBE   : {mbe:.4f}")
    print(f"  MAE_W : {mae_w:.1f} Watts")
    print("="*50)
    print("Copy the BEST HYPERPARAMETERS into `model_baselines.py`!")

if __name__ == "__main__":
    main()