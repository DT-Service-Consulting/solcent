#!/usr/bin/env python3
"""
feature_importance.py
=====================
Analyzes feature redundancy and importance to reduce weather forecast uncertainty.
Outputs recommendations for features that can be safely dropped without losing
predictive power.

Outputs are saved to: models/feature_analysis/
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
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import mean_squared_error
except ImportError:
    print("Please install required packages: pip install lightgbm scikit-learn matplotlib")
    exit(1)

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
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
    
    print("Loading data...")
    df = pd.read_csv(MODEL_READY)
    
    # We only care about daylight rows where solar generation actually happens
    df = df[df["is_daylight"] == True].copy()
    
    features = [c for c in df.columns if c not in NON_FEATURES]
    print(f"Analyzing {len(features)} potential features...\n")
    
    print("-" * 50)
    print("STEP 1: THE REDUNDANCY CHECK (Correlation)")
    print("-" * 50)
    
    train_df = df[df["split"] == "train"]
    corr_matrix = train_df[features].corr().abs()
    
    # Extract the upper triangle of the matrix to avoid duplicate pairs
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find pairs with correlation > 0.90
    highly_correlated = []
    for column in upper.columns:
        for idx in upper.index:
            val = upper.loc[idx, column]
            if pd.notna(val) and val > 0.90:
                highly_correlated.append((idx, column, val))
                
    if highly_correlated:
        print("WARNING: Found highly correlated feature pairs (>0.90).")
        print("These contain redundant information. Consider dropping one from each pair:")
        for feat1, feat2, val in sorted(highly_correlated, key=lambda x: x[2], reverse=True):
            print(f"  - {feat1} & {feat2} (r = {val:.3f})")
    else:
        print("Good news: No highly correlated feature pairs found (>0.90).")

    print("\n" + "-" * 50)
    print("STEP 2: THE USAGE CHECK (LightGBM Built-in Gain)")
    print("-" * 50)
    
    val_df = df[df["split"] == "val"]
    
    # Fast baseline MSE model just for feature importance extraction
    print("Training baseline LightGBM model...")
    model = lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.05, num_leaves=31, 
        random_state=42, n_jobs=-1, verbose=-1
    )
    
    model.fit(
        train_df[features], train_df["target_norm"],
        eval_set=[(val_df[features], val_df["target_norm"])],
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
    )
    
    # Plotting Gain Importance
    fig, ax = plt.subplots(figsize=(10, 8))
    lgb.plot_importance(model, importance_type='gain', max_num_features=30, ax=ax, title="Feature Importance (Gain)")
    plt.tight_layout()
    gain_plot_path = OUTPUT_DIR / "lgbm_gain_importance.png"
    plt.savefig(gain_plot_path, dpi=100)
    plt.close()
    
    print(f"Saved LightGBM Gain Importance plot to: {gain_plot_path.name}")
    print("Look at the bottom of this plot. Features with < 1% total gain are suspects for removal.")

    print("\n" + "-" * 50)
    print("STEP 3: THE ROBUSTNESS CHECK (Permutation Importance)")
    print("-" * 50)
    
    # Permutation is extremely slow on large datasets. We sample 10,000 rows from Validation.
    sample_size = min(10000, len(val_df))
    print(f"Running Permutation Importance on a random sample of {sample_size} unseen validation rows...")
    val_sample = val_df.sample(n=sample_size, random_state=42)
    
    X_val = val_sample[features]
    y_val = val_sample["target_norm"]
    
    # Calculate baseline score
    baseline_preds = model.predict(X_val)
    baseline_rmse = np.sqrt(mean_squared_error(y_val, baseline_preds))
    print(f"Baseline Validation RMSE: {baseline_rmse:.4f}")
    
    # Run permutation
    result = permutation_importance(
        model, X_val, y_val, scoring='neg_root_mean_squared_error', 
        n_repeats=5, random_state=42, n_jobs=-1
    )
    
    # Sort and display results
    importances = pd.DataFrame({
        'Feature': features,
        'RMSE_Increase': result.importances_mean  # Note: scikit-learn returns negative metrics, but we want the absolute change
    }).sort_values(by='RMSE_Increase', ascending=False)
    
    print("\nImpact on RMSE if feature is shuffled (Higher = More Important):")
    for _, row in importances.iterrows():
        impact = row['RMSE_Increase']
        if impact <= 0:
            status = "💀 USELESS (Consider Dropping)"
        elif impact < 0.001:
            status = "⚠️ WEAK (Candidate for Dropping)"
        else:
            status = "✅ KEEPER"
            
        print(f"{row['Feature']:>25} : +{max(0, impact):.5f} RMSE  -> {status}")

    importances.to_csv(OUTPUT_DIR / "permutation_importance.csv", index=False)
    print(f"\nSaved full permutation results to: {OUTPUT_DIR.name}/permutation_importance.csv")
    print("\nNEXT STEP:")
    print("Try removing the 'USELESS' and 'WEAK' features from `FEATURES` in `model_baselines.py`")
    print("and see if your Test Set RMSE stays the same or improves!")

if __name__ == "__main__":
    main()