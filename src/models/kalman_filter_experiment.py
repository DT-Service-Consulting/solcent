#!/usr/bin/env python3
"""
kalman_filter_experiment.py
===========================
Applies a 1D Kalman Filter to track and correct sequential day-to-day bias 
in the ECMWF forecast. It tracks the bias for each specific 15-minute time slot 
across consecutive days.

Outputs: reports/weather_analysis/kalman_filter_results.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / "data" / "interim" / "mos_training_data.csv"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "weather_analysis"

class SimpleKalmanFilter:
    """A lightweight 1D Kalman Filter to track rolling bias."""
    def __init__(self, process_noise=1e-3, measurement_noise=0.1):
        self.Q = process_noise      # Process noise variance (How fast does true bias change?)
        self.R = measurement_noise  # Measurement noise variance (How noisy are the daily errors?)
        self.x = 0.0                # Initial state estimate (Assume 0 bias initially)
        self.P = 1.0                # Initial uncertainty

    def update_and_predict(self, measurement):
        # 1. Prediction Step (Bias remains the same from yesterday to today)
        x_pred = self.x
        P_pred = self.P + self.Q

        # 2. Update Step (Incorporate yesterday's actual error)
        # Kalman Gain: How much do we trust the new measurement vs our ongoing prediction?
        K = P_pred / (P_pred + self.R) 
        
        self.x = x_pred + K * (measurement - x_pred)
        self.P = (1 - K) * P_pred
        
        return self.x

def main():
    if not INPUT_FILE.exists():
        print(f"Error: {INPUT_FILE} not found. Run fetch_mos_data.py first.")
        return
        
    print("--- Initializing Kalman Filter Dynamic Bias Correction ---")
    df = pd.read_csv(INPUT_FILE)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    
    # We only care about daylight hours where GHI is significant
    df = df[df['actual_ghi'] > 10].copy()
    
    # Extract time features to group by House and Time-of-Day
    df['date'] = df['timestamp'].dt.date
    df['time_slot'] = df['timestamp'].dt.time
    
    df = df.sort_values(['system_id', 'time_slot', 'date']).reset_index(drop=True)
    
    # Target: Actual - Forecast (Positive means forecast was too low)
    df['raw_error'] = df['actual_ghi'] - df['ecmwf_ghi']
    
    print(f"Applying Kalman Filters across {df['system_id'].nunique()} systems...")
    
    # We will store the corrected GHI here
    corrected_ghis = []
    
    # Iterate through each unique house and each unique 15-min slot
    grouped = df.groupby(['system_id', 'time_slot'])
    
    for (sys_id, time_slot), group in grouped:
        # Sort chronologically by date
        group = group.sort_values('date')
        
        # Initialize a dedicated Kalman Filter for this specific time slot
        # Tuning: High R means we assume daily errors are mostly random noise and react slowly.
        kf = SimpleKalmanFilter(process_noise=1e-2, measurement_noise=100.0)
        
        corrected_values = []
        
        for index, row in group.iterrows():
            # Apply the CURRENT bias estimate to TODAY's forecast
            # (Note: For the very first day, bias estimate is 0)
            corrected_ghi = row['ecmwf_ghi'] + kf.x
            corrected_values.append(max(0, corrected_ghi)) # Clip negative irradiance
            
            # Now that today is over, update the Kalman Filter with TODAY's actual error
            # so it is ready for tomorrow's prediction.
            kf.update_and_predict(row['raw_error'])
            
        group['kf_corrected_ghi'] = corrected_values
        corrected_ghis.append(group)
        
    # Recombine everything
    result_df = pd.concat(corrected_ghis).sort_values(['system_id', 'timestamp'])
    
    # --- Evaluate the Improvement ---
    actuals = result_df['actual_ghi']
    raw_forecasts = result_df['ecmwf_ghi']
    kf_forecasts = result_df['kf_corrected_ghi']
    
    raw_mae = mean_absolute_error(actuals, raw_forecasts)
    kf_mae = mean_absolute_error(actuals, kf_forecasts)
    
    raw_r2 = r2_score(actuals, raw_forecasts)
    kf_r2 = r2_score(actuals, kf_forecasts)
    
    improvement_pct = ((raw_mae - kf_mae) / raw_mae) * 100
    
    print("\n--- KALMAN FILTER CORRECTION RESULTS ---")
    print(f"Original ECMWF MAE: {raw_mae:.2f} W/m²")
    print(f"KF Corrected MAE:   {kf_mae:.2f} W/m²")
    print(f"Absolute Improvement: {improvement_pct:.2f}% reduction in error!")
    print(f"Forecast R² shift:  {raw_r2:.4f} -> {kf_r2:.4f}")
    
    # --- Visualization ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    
    sns.kdeplot(actuals - raw_forecasts, fill=True, color='red', label=f'Raw ECMWF Error (MAE: {raw_mae:.1f})', alpha=0.3)
    sns.kdeplot(actuals - kf_forecasts, fill=True, color='blue', label=f'Kalman Filter Error (MAE: {kf_mae:.1f})', alpha=0.4)
    
    plt.axvline(0, color='black', linestyle='--')
    plt.title('Dynamic Bias Correction: Kalman Filter vs Raw ECMWF', fontsize=14, fontweight='bold')
    plt.xlabel('Prediction Error (W/m²)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend()
    plt.xlim(-400, 400)
    plt.tight_layout()
    
    plot_path = OUTPUT_DIR / "kalman_filter_results.png"
    plt.savefig(plot_path, dpi=120)
    print(f"\nSaved visualization to: {plot_path}")

if __name__ == "__main__":
    main()