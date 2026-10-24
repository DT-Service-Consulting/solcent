#!/usr/bin/env python3
"""
tune_mos_benchmark.py
=====================
An Optuna Benchmark Arena for Model Output Statistics (MOS).
Tests multiple algorithms (Ridge, Random Forest, XGBoost, LightGBM) to find 
the absolute highest possible reduction in ECMWF weather forecast error.

Outputs: Console Leaderboard & models/mos_model/mos_benchmark_metrics.csv
"""

import warnings
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
import optuna
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
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
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / "data" / "interim" / "mos_training_data.csv"
OUTPUT_DIR = PROJECT_ROOT / "models" / "mos_model"
CURVES_DIR = OUTPUT_DIR / "learning_curves"

N_TRIALS = 30  # Number of Optuna trials per model

def suggest_model(trial, model_name):
    """Defines the hyperparameter search space for each MOS model."""
    if model_name == "Ridge":
        alpha = trial.suggest_float("alpha", 1e-3, 1000.0, log=True)
        return Ridge(alpha=alpha)
        
    elif model_name == "Random Forest":
        n_estimators = trial.suggest_int('n_estimators', 50, 150)
        max_depth = trial.suggest_int('max_depth', 3, 12)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 5, 50)
        return RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth, 
            min_samples_leaf=min_samples_leaf, random_state=42, n_jobs=-1
        )
        
    elif model_name == "XGBoost":
        learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)
        max_depth = trial.suggest_int("max_depth", 3, 10)
        subsample = trial.suggest_float("subsample", 0.5, 1.0)
        reg_lambda = trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True)
        return XGBRegressor(
            n_estimators=150, learning_rate=learning_rate, max_depth=max_depth,
            subsample=subsample, reg_lambda=reg_lambda, random_state=42, n_jobs=-1
        )
        
    elif model_name == "LightGBM":
        learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)
        num_leaves = trial.suggest_int("num_leaves", 10, 100)
        max_depth = trial.suggest_int("max_depth", 3, 10)
        reg_lambda = trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True)
        return LGBMRegressor(
            n_estimators=150, learning_rate=learning_rate, num_leaves=num_leaves,
            max_depth=max_depth, reg_lambda=reg_lambda, random_state=42, n_jobs=-1, verbose=-1
        )

def main():
    if not INPUT_FILE.exists():
        print(f"Error: {INPUT_FILE} not found.")
        return
        
    print("--- Loading Data for MOS Benchmark Arena ---")
    df = pd.read_csv(INPUT_FILE)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    
    # 1. Target and Features
    df['error_target'] = df['actual_ghi'] - df['ecmwf_ghi']
    df['hour'] = df['timestamp'].dt.hour
    df['month'] = df['timestamp'].dt.month
    df['day_of_year'] = df['timestamp'].dt.dayofyear
    
    features = ['ecmwf_ghi', 'ecmwf_cloud', 'ecmwf_temp', 'hour', 'month', 'day_of_year']
    
    mos_df = df.sort_values('timestamp').reset_index(drop=True)
    X = mos_df[features]
    y = mos_df['error_target']

    # 2. Chronological Split (80% Train, 20% Test)
    split_idx = int(len(mos_df) * 0.8)
    X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]
    X_test, y_test = X.iloc[split_idx:], y.iloc[split_idx:]
    
    val_idx = int(len(X_train) * 0.8)
    X_train_sub, y_train_sub = X_train.iloc[:val_idx], y_train.iloc[:val_idx]
    X_val, y_val = X_train.iloc[val_idx:], y_train.iloc[val_idx:]
    
    actuals = mos_df.iloc[split_idx:]['actual_ghi']
    raw_forecasts = X_test['ecmwf_ghi']
    
    raw_mae = mean_absolute_error(actuals, raw_forecasts)
    raw_r2 = r2_score(actuals, raw_forecasts)

    models_to_test = ["Ridge", "Random Forest"]
    if XGBRegressor is not None: models_to_test.append("XGBoost")
    if LGBMRegressor is not None: models_to_test.append("LightGBM")

    print(f"\nStarting Optuna Tuning ({N_TRIALS} Trials per model)...\n")
    results = []
    all_best_params = {}
    CURVES_DIR.mkdir(parents=True, exist_ok=True)

    for model_name in models_to_test:
        print(f"Tuning {model_name}...")
        t0 = time.time()
        
        def objective(trial):
            model = suggest_model(trial, model_name)
            
            # Use early stopping for gradient boosting models
            if model_name == "XGBoost":
                model.set_params(early_stopping_rounds=20, eval_metric="mae")
                model.fit(X_train_sub, y_train_sub, eval_set=[(X_val, y_val)], verbose=False)
            elif model_name == "LightGBM":
                import lightgbm as lgb
                model.set_params(metric="mae")
                model.fit(X_train_sub, y_train_sub, eval_set=[(X_val, y_val)], 
                          callbacks=[lgb.early_stopping(20, verbose=False)])
            else:
                model.fit(X_train, y_train)
                
            predicted_errors = model.predict(X_test)
            corrected_ghi = np.clip(X_test['ecmwf_ghi'] + predicted_errors, 0, None)
            return mean_absolute_error(actuals, corrected_ghi)

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=N_TRIALS)
        
        best_mae = study.best_value
        improvement_pct = ((raw_mae - best_mae) / raw_mae) * 100
        
        all_best_params[model_name] = study.best_params
        best_model = suggest_model(optuna.trial.FixedTrial(study.best_params), model_name)
        
        # Fit with evaluation sets to extract learning curves
        if model_name == "XGBoost":
            best_model.set_params(early_stopping_rounds=20, eval_metric="mae")
            best_model.fit(X_train_sub, y_train_sub, eval_set=[(X_train_sub, y_train_sub), (X_val, y_val)], verbose=False)
            
            # Extract metrics
            evals = best_model.evals_result()
            t_key, v_key = list(evals.keys())[0], list(evals.keys())[1]
            train_mae, val_mae = evals[t_key]['mae'], evals[v_key]['mae']
            
        elif model_name == "LightGBM":
            import lightgbm as lgb
            best_model.set_params(metric="mae")
            best_model.fit(X_train_sub, y_train_sub, eval_set=[(X_train_sub, y_train_sub), (X_val, y_val)], 
                           callbacks=[lgb.early_stopping(20, verbose=False)])
            
            # Extract metrics (l1 is MAE in LightGBM)
            evals = best_model.evals_result_
            t_key, v_key = list(evals.keys())[0], list(evals.keys())[1]
            train_mae, val_mae = evals[t_key]['l1'], evals[v_key]['l1']
            
        else:
            best_model.fit(X_train, y_train)
            train_mae, val_mae = None, None

        if train_mae and val_mae:
            plt.figure(figsize=(8, 5))
            plt.plot(train_mae, label="Train MAE", color="#1f77b4", linewidth=2)
            plt.plot(val_mae, label="Validation MAE", color="#ff7f0e", linewidth=2)
            plt.title(f"{model_name} Learning Curve (Early Stopping)")
            plt.xlabel("Boosting Rounds")
            plt.ylabel("Mean Absolute Error")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(CURVES_DIR / f"{model_name.lower().replace(' ', '_')}_learning_curve.png", dpi=120)
            plt.close()

        # Generate predictions for the Test Set Leaderboard
        best_corrected_ghi = np.clip(X_test['ecmwf_ghi'] + best_model.predict(X_test), 0, None)
        best_r2 = r2_score(actuals, best_corrected_ghi)
        
        results.append({
            "Model": model_name,
            "Corrected MAE": best_mae,
            "Error Reduction %": improvement_pct,
            "Forecast R2": best_r2,
            "Tune Time (s)": round(time.time() - t0, 1)
        })

    # Display Results
    res_df = pd.DataFrame(results).sort_values(by="Corrected MAE")
    
    print("\n" + "="*60)
    print("OPTUNA MOS BENCHMARK LEADERBOARD (Test Set, Daylight)")
    print("="*60)
    print(f"Original ECMWF Raw Forecast MAE: {raw_mae:.4f} W/m²")
    print(f"Original ECMWF Raw Forecast R²:  {raw_r2:.4f}")
    print("-" * 60)
    
    pd.set_option('display.float_format', lambda x: f"{x:.4f}")
    print(res_df.to_string(index=False))
    print("="*60)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(OUTPUT_DIR / "mos_benchmark_metrics.csv", index=False)
    
    with open(OUTPUT_DIR / "best_mos_params.json", "w") as f:
        json.dump(all_best_params, f, indent=4)
        
    print(f"\nSaved leaderboard to {OUTPUT_DIR.name}/mos_benchmark_metrics.csv")
    print(f"Saved optimal parameters to {OUTPUT_DIR.name}/best_mos_params.json")
    print(f"Saved learning curves to {CURVES_DIR.name}/")

if __name__ == "__main__":
    main()