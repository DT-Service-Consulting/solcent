#!/usr/bin/env python3
"""
check_correlation.py (V2)
=========================
Focuses specifically on finding highly correlated feature pairs (Multicollinearity).
Saves a visual heatmap and prints actionable recommendations to the console.

Outputs: reports/feature_analysis/correlation_heatmap_v2.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_READY  = PROJECT_ROOT / "data" / "processed" / "model_ready_v2.csv"
OUTPUT_DIR   = PROJECT_ROOT / "reports" / "feature_analysis"

# Columns to explicitly ignore during correlation checking
IGNORE_COLS = {
    "system_id", "timestamp", "target_norm", "target_power_w", 
    "is_daylight", "is_valid", "split", "pred_persistence", 
    "pred_clearsky", "pred_model", "target_pr",
    "pred_model_p25", "pred_model_p50", "pred_model_p90"
}

def main():
    if not MODEL_READY.exists():
        print(f"Error: Cannot find {MODEL_READY}.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading dataset from {MODEL_READY.name}...")
    df = pd.read_csv(MODEL_READY)
    
    # We only care about daylight rows for weather correlation
    df = df[df["is_daylight"] == True]
    
    # Select only numeric feature columns using the exclusion list
    features = [c for c in df.columns if c not in IGNORE_COLS and df[c].dtype in ['float64', 'float32', 'int64', 'int32']]
    
    print(f"Calculating correlation matrix for {len(features)} features...")
    corr_matrix = df[features].corr()
    
    # 1. Generate the Heatmap Visualization
    plt.figure(figsize=(16, 14))
    
    # Create a mask to hide the upper triangle (makes it easier to read)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(
        corr_matrix, 
        mask=mask, 
        cmap='coolwarm', 
        vmin=-1, vmax=1, 
        annot=True, 
        fmt=".2f", 
        annot_kws={"size": 8},
        square=True,
        linewidths=.5
    )
    
    plt.title("Feature Correlation Heatmap V2 (Daylight Hours)", fontsize=16)
    plt.tight_layout()
    
    heatmap_path = OUTPUT_DIR / "correlation_heatmap_v2.png"
    plt.savefig(heatmap_path, dpi=150)
    plt.close()
    
    # 2. Extract and Print Highly Correlated Pairs
    print("\n" + "="*55)
    print("🚨 HIGHLY CORRELATED FEATURE PAIRS (Threshold: |r| > 0.85) 🚨")
    print("="*55)
    
    # Get absolute values and upper triangle to avoid duplicates (A-B and B-A)
    corr_abs = corr_matrix.abs()
    upper = corr_abs.where(np.triu(np.ones(corr_abs.shape), k=1).astype(bool))
    
    highly_correlated = []
    for column in upper.columns:
        for idx in upper.index:
            val = upper.loc[idx, column]
            if pd.notna(val) and val > 0.85:
                # Get the original (non-absolute) correlation value for printing
                orig_val = corr_matrix.loc[idx, column]
                highly_correlated.append((idx, column, orig_val))
                
    if not highly_correlated:
        print("Good news! No highly correlated pairs found above 0.85.")
    else:
        # Sort by highest correlation first
        highly_correlated.sort(key=lambda x: abs(x[2]), reverse=True)
        
        for feat1, feat2, val in highly_correlated:
            indicator = "🔴 EXTREME" if abs(val) > 0.95 else "🟠 HIGH"
            print(f"{indicator} : {feat1:>22}  &  {feat2:<22} (r = {val:+.3f})")
            
        print("\n[RECOMMENDATION]:")
        print("For every pair listed above, pick ONE feature to keep and drop the other.")
        print("Usually, you want to keep the feature that is mathematically closest to the ")
        print("target (e.g., keep 'expert_power_fc' and drop 'ghi_fc' or 'clearsky_ghi').")
        
    print("\n" + "="*55)
    print(f"Heatmap saved to: {heatmap_path}")
    print("="*55)

if __name__ == "__main__":
    main()