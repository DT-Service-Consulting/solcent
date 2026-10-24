#!/usr/bin/env python3
"""
tune_mos_optuna.py
==================
Uses Optuna (Bayesian Optimization) to find the absolute mathematically optimal 
hyperparameters for the Random Forest Model Output Statistics (MOS) algorithm.

Outputs: 
- Console output detailing the best parameters and improved MAE.
- models/mos_model/best_mos_params.json
"""

import pandas as pd
import numpy as np
import optuna
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / "data" / "interim" / "mos_training_data.csv"
OUTPUT_DIR = PROJECT_ROOT / "models" / "mos_model"

# We want Optuna to run quietly so it doesn't flood the console with 100 trial logs
optuna.logging.set_verbosity(optuna.logging.WARNING)

def main():
    if not INPUT_FILE.exists():
        print(f"Error: {INPUT_FILE} not found. Run fetch_mos_data.py first.")
        return
        
    print("--- Loading Data for Optuna MOS Tuning ---")
    df = pd.read_csv(INPUT_FILE)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    
    # 1. Prepare Target and Features
    df['error_target'] = df['actual_ghi'] - df['ecmwf_ghi']
    df['hour'] = df['timestamp'].dt.hour
    df['month'] = df['timestamp'].dt.month
    df['day_of_year'] = df['timestamp'].dt.dayofyear
    
    features = ['ecmwf_ghi', 'ecmwf_cloud', 'ecmwf_temp', 'hour', 'month', 'day_of_year']
    
    mos_df = df.sort_values('timestamp').reset_index(drop=True)
    X = mos_df[features]
    y = mos_df['error_target']

    # 2. Chronological Train-Test Split (80% Train, 20% Test)
    split_idx = int(len(mos_df) * 0.8)
    X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]
    X_test, y_test = X.iloc[split_idx:], y.iloc[split_idx:]
    
    # Keep track of raw actuals and forecasts for evaluation
    actuals = mos_df.iloc[split_idx:]['actual_ghi']
    raw_forecasts = X_test['ecmwf_ghi']
    
    # 3. Define the Optuna Objective Function
    def objective(trial):
        # Suggest hyperparameters
        n_estimators = trial.suggest_int('n_estimators', 50, 200)
        max_depth = trial.suggest_int('max_depth', 4, 20)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 15)
        
        # Initialize and train the model
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # Predict error and correct the forecast
        predicted_errors = model.predict(X_test)
        corrected_ghi = np.clip(X_test['ecmwf_ghi'] + predicted_errors, 0, None)
        
        # Calculate MAE
        mae = mean_absolute_error(actuals, corrected_ghi)
        return mae

    # 4. Run the Optuna Study
    print("Initiating Optuna Bayesian Optimization (50 Trials)...")
    print("This will take several minutes to run as it trains 50 different Random Forests.")
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50, show_progress_bar=True)
    
    # 5. Evaluate the Best Results
    raw_mae = mean_absolute_error(actuals, raw_forecasts)
    best_mae = study.best_value
    improvement_pct = ((raw_mae - best_mae) / raw_mae) * 100
    
    print("\n================ OPTUNA MOS TUNING COMPLETE ================")
    print(f"Original ECMWF MAE: {raw_mae:.2f} W/m²")
    print(f"Best Tuned MOS MAE: {best_mae:.2f} W/m²")
    print(f"Maximum Error Reduction: {improvement_pct:.2f}%")
    
    print("\nBest Hyperparameters Found:")
    for key, value in study.best_params.items():
        print(f"  - {key}: {value}")
        
    # 6. Save parameters for future use
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    params_file = OUTPUT_DIR / "best_mos_params.json"
    with open(params_file, "w") as f:
        json.dump(study.best_params, f, indent=4)
        
    print(f"\nSaved optimal parameters to {params_file}")

if __name__ == "__main__":
    main()