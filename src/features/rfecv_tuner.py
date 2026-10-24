#!/usr/bin/env python3
"""
rfecv_tuner.py
==============
Performs Recursive Feature Elimination with Cross-Validation (RFECV) using LightGBM.
It plots a curve showing how the model's RMSE changes as features are dropped one by one,
proving exactly what the optimal number of features is.

Outputs: models/feature_analysis/rfecv_learning_curve.png
         models/feature_analysis/rfecv_selected_features.csv
"""

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import lightgbm as lgb
    from sklearn.feature_selection import RFECV
    from sklearn.model_selection import TimeSeriesSplit
except ImportError:
    print("Please install required packages: pip install lightgbm scikit-learn matplotlib")
    exit(1)

# Suppress lightgbm output to keep the console clean during hundreds of fits
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Add models directory to path so we can import our pruned FEATURES list
import sys
sys.path.append(str(PROJECT_ROOT / "src" / "models"))
from model_baselines import FEATURES

MODEL_READY  = PROJECT_ROOT / "data" / "processed" / "model_ready.csv"
OUTPUT_DIR   = PROJECT_ROOT / "models" / "feature_analysis"

NON_FEATURES = {
    "system_id", "timestamp", "target_norm", "target_power_w", 
    "is_daylight", "is_valid", "split", "pred_persistence", 
    "pred_clearsky", "pred_model"
}

def main():
    if not MODEL_READY.exists():
        print(f"Error: Cannot find {MODEL_READY}.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Loading data for RFECV...")
    df = pd.read_csv(MODEL_READY)
    
    # We only care about daylight rows for feature selection
    df = df[df["is_daylight"] == True].copy()
    
    # We use Train and Val combined for CV
    train_val_df = df[df["split"].isin(["train", "val"])].sort_values("timestamp")
    
    # Use the pruned feature list instead of all columns
    features = [c for c in FEATURES if c in df.columns]
    X = train_val_df[features]
    y = train_val_df["target_norm"]
    
    print(f"Starting RFECV on {len(features)} features...")
    print("This will train LightGBM dozens of times. Please wait...\n")

    # 1. Define the Estimator
    # We use a fast, lightweight LGBM for feature selection
    estimator = lgb.LGBMRegressor(
        n_estimators=150, 
        learning_rate=0.05,
        num_leaves=31,
        importance_type='gain', # CRITICAL: Drop based on mathematical gain, not splits
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    # 2. Define TimeSeries Cross-Validation
    # Crucial for time-series data to prevent data leakage from the future
    cv = TimeSeriesSplit(n_splits=3)

    # 3. Setup RFECV
    # We use negative root mean squared error because sklearn RFECV tries to MAXIMIZE the score
    selector = RFECV(
        estimator=estimator,
        step=1, # Drop 1 feature at a time
        cv=cv,
        scoring='neg_root_mean_squared_error',
        min_features_to_select=3,
        n_jobs=-1
    )
    
    selector.fit(X, y)
    
    # 4. Extract Results
    # Convert negative RMSE back to positive for plotting
    cv_results = -1 * selector.cv_results_['mean_test_score']
    n_features = len(features)
    
    # Create an ordered list of features based on their rank (1 = Keep, higher = dropped earlier)
    feature_ranks = pd.DataFrame({
        'Feature': features,
        'Rank': selector.ranking_
    }).sort_values(by='Rank')
    
    optimal_num = selector.n_features_
    best_rmse = cv_results.min()
    
    print("=" * 50)
    print(f"RFECV COMPLETE")
    print(f"Optimal number of features: {optimal_num}")
    print(f"Best CV RMSE: {best_rmse:.5f}")
    print("=" * 50)
    
    print("\nFeatures to KEEP (Rank 1):")
    for feat in feature_ranks[feature_ranks['Rank'] == 1]['Feature']:
        print(f"  - {feat}")
        
    print("\nFeatures that were DROPPED (Weakest first):")
    dropped = feature_ranks[feature_ranks['Rank'] > 1].sort_values(by='Rank', ascending=False)
    for _, row in dropped.iterrows():
        print(f"  - [Rank {row['Rank']}] {row['Feature']}")

    # 5. Plotting the Learning Curve
    plt.figure(figsize=(10, 6))
    
    # Plot number of features vs RMSE
    x_axis = range(1, len(cv_results) + 1)
    plt.plot(x_axis, cv_results, marker='o', color='#1f77b4', linewidth=2)
    
    # Highlight the optimal point
    plt.axvline(x=optimal_num, color='red', linestyle='--', label=f'Optimal ({optimal_num} features)')
    plt.scatter([optimal_num], [best_rmse], color='red', s=100, zorder=5)
    
    plt.title("RFECV: Number of Features vs. Model Error (RMSE)", fontsize=14)
    plt.xlabel("Number of Features Selected", fontsize=12)
    plt.ylabel("Cross-Validation RMSE (Lower is better)", fontsize=12)
    plt.xticks(x_axis)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    plot_path = OUTPUT_DIR / "rfecv_learning_curve.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    
    # Save the rankings
    csv_path = OUTPUT_DIR / "rfecv_selected_features.csv"
    feature_ranks.to_csv(csv_path, index=False)
    
    print(f"\nSaved RFECV Curve to: {plot_path.name}")
    print(f"Saved Feature Ranks to: {csv_path.name}")

if __name__ == "__main__":
    main()
