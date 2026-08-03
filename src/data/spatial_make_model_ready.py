#!/usr/bin/env python3
"""
spatial_make_model_ready.py
===========================
Prepares the 3x3 Spatial Grid dataset for PyTorch modeling.
We maintain the night rows because the LSTM relies on continuous time sequences.

Outputs: data/processed/spatial_model_ready.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT  = Path(__file__).resolve().parents[2]
FEATURES_FILE = PROJECT_ROOT / "data" / "interim" / "spatial_features_all_systems.csv"
OUTPUT_FILE   = PROJECT_ROOT / "data" / "processed" / "spatial_model_ready.csv"

def main():
    if not FEATURES_FILE.exists():
        print(f"ERROR: Cannot find {FEATURES_FILE}. Run spatial_build_features.py first.")
        return

    print("Loading massive spatial dataset into RAM...")
    df = pd.read_csv(FEATURES_FILE)
    
    print("Downcasting 64-bit data to 32-bit to halve memory usage...")
    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = df[float_cols].astype('float32')
    
    int_cols = df.select_dtypes(include=['int64']).columns
    df[int_cols] = df[int_cols].astype('int32')

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Europe/Brussels")
    
    # Filter for valid rows only
    m = df[df["is_valid"]].copy()
    print(f"Total valid rows (incl. night): {len(m):,}")

    m["target_norm"] = m["target_norm"].clip(0, 1.2)

    # Gap-fill spatial weather features (ffill -> bfill to prevent NaNs from crashing PyTorch)
    non_feat = {"system_id", "timestamp", "target_norm", "power_w", "is_daylight", "is_valid", "split", "capacity_kwp"}
    fill_cols = [c for c in m.columns if c not in non_feat]
    
    if m[fill_cols].isna().any().any():
        m = m.sort_values(["system_id", "timestamp"])
        m[fill_cols] = m.groupby("system_id")[fill_cols].ffill().bfill()
        
    print(f"Gap-filled {len(fill_cols)} spatial grid feature columns.")

    # Time-based split for strict temporal generalization testing
    m = m.sort_values("timestamp")
    q70, q85 = m["timestamp"].quantile(0.70), m["timestamp"].quantile(0.85)
    m["split"] = np.where(m["timestamp"] <= q70, "train",
                          np.where(m["timestamp"] <= q85, "val", "test"))

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    m.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved Spatial Model Ready file: {m.shape} -> {OUTPUT_FILE.name}")

if __name__ == "__main__":
    main()