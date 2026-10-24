#!/usr/bin/env python3
"""
train_mos_model.py
==================
Trains the Model Output Statistics (MOS) Random Forest on the dataset generated
by fetch_mos_data.py to correct ECMWF forecast errors.

Outputs: 
- reports/weather_analysis/mos_correction_results.png
- data/processed/mos_corrected_ecmwf.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / "data" / "interim" / "mos_training_data.csv"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "weather_analysis"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

def main():
    if not INPUT_FILE.exists():
        print(f"Error: {INPUT_FILE} not found. Run fetch_mos_data.py first.")
        return
        
    print("--- Initializing MOS Random Forest Correction ---")
    df = pd.read_csv(INPUT_FILE)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    
    # 1. Prepare the Target (Actual Reality - Forecasted Reality)
    df['error_target'] = df['actual_ghi'] - df['ecmwf_ghi']
    
    # 2. Engineer Features
    df['hour'] = df['timestamp'].dt.hour
    df['month'] = df['timestamp'].dt.month
    df['day_of_year'] = df['timestamp'].dt.dayofyear
    
    # Note: We aren't using clear-sky index here because we didn't fetch clearsky_ghi 
    # from the API in the previous script to save API calls, but the RF will still 
    # find powerful correlations using the raw cloud and temp variables!
    
    features = [
        'ecmwf_ghi', 
        'ecmwf_cloud', 
        'ecmwf_temp',
        'hour', 
        'month', 
        'day_of_year'
    ]
    
    # Sort chronologically
    mos_df = df.sort_values('timestamp').reset_index(drop=True)
    print(f"Training MOS model on {len(mos_df):,} daylight records...")

    X = mos_df[features]
    y = mos_df['error_target']

    # 3. Chronological Train-Test Split (80% Train, 20% Test)
    split_idx = int(len(mos_df) * 0.8)
    X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]
    X_test, y_test = X.iloc[split_idx:], y.iloc[split_idx:]
    
    # 4. Train the Random Forest
    rf_mos = RandomForestRegressor(
        n_estimators=100, 
        max_depth=8,         
        min_samples_leaf=10, 
        random_state=42, 
        n_jobs=-1
    )
    
    print("Fitting Random Forest Model... (This might take a minute)")
    rf_mos.fit(X_train, y_train)
    
    # 5. Predict and Correct
    predicted_errors = rf_mos.predict(X_test)
    corrected_ghi = X_test['ecmwf_ghi'] + predicted_errors
    corrected_ghi = np.clip(corrected_ghi, 0, None) # Prevent negative irradiance
    
    # 6. Evaluate
    actuals = mos_df.iloc[split_idx:]['actual_ghi']
    raw_forecasts = X_test['ecmwf_ghi']
    
    raw_mae = mean_absolute_error(actuals, raw_forecasts)
    corrected_mae = mean_absolute_error(actuals, corrected_ghi)
    
    raw_r2 = r2_score(actuals, raw_forecasts)
    corrected_r2 = r2_score(actuals, corrected_ghi)
    
    improvement_pct = ((raw_mae - corrected_mae) / raw_mae) * 100
    
    print("\n--- MOS CORRECTION RESULTS (Unseen Future Data) ---")
    print(f"Original ECMWF MAE: {raw_mae:.2f} W/m²")
    print(f"Corrected GHI MAE:  {corrected_mae:.2f} W/m²")
    print(f"Absolute Improvement: {improvement_pct:.1f}% reduction in error!")
    print(f"Forecast R² jump: {raw_r2:.3f} -> {corrected_r2:.3f}")
    
    # 7. Visualization
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    
    sns.kdeplot(actuals - raw_forecasts, fill=True, color='red', label=f'Raw ECMWF Error (MAE: {raw_mae:.1f})', alpha=0.3)
    sns.kdeplot(actuals - corrected_ghi, fill=True, color='green', label=f'MOS Corrected Error (MAE: {corrected_mae:.1f})', alpha=0.4)
    
    plt.axvline(0, color='black', linestyle='--')
    plt.title('Distribution of Weather Forecast Errors (Test Set)', fontsize=14, fontweight='bold')
    plt.xlabel('Prediction Error (W/m²)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend()
    plt.xlim(-400, 400)
    plt.tight_layout()
    
    plot_path = OUTPUT_DIR / "mos_correction_results.png"
    plt.savefig(plot_path, dpi=120)
    print(f"\nSaved visualization to: {plot_path}")
    
    # 8. Export the Corrected Dataset
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    mos_df.loc[X_test.index, 'mos_corrected_ghi'] = corrected_ghi
    
    # Fill the train portion with predictions too so we have a continuous dataset
    train_predicted_errors = rf_mos.predict(X_train)
    mos_df.loc[X_train.index, 'mos_corrected_ghi'] = np.clip(X_train['ecmwf_ghi'] + train_predicted_errors, 0, None)
    
    export_path = PROCESSED_DIR / "mos_corrected_ecmwf.csv"
    mos_df[['timestamp', 'system_id', 'actual_ghi', 'ecmwf_ghi', 'mos_corrected_ghi']].to_csv(export_path, index=False)
    print(f"Saved complete MOS-corrected dataset to: {export_path}")

if __name__ == "__main__":
    main()