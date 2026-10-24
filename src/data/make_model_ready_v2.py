#!/usr/bin/env python3
"""
make_model_ready.py (V2)
========================
Turn features_all_systems_v2.csv into model_ready_v2.csv for training.

UPDATES (V2):
  - Points to the new interim_v2 directory and _v2 filenames.
  - Expanded NO_FILL_PREFIXES to catch all "lag_" variables (including 
    the newly engineered lag_power_d2) to prevent incorrect gap-filling.
  - Fixed a latent Pandas indexing bug during gap-filling.
"""

from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT  = Path(__file__).resolve().parents[2]
FEATURES_FILE = PROJECT_ROOT / "data" / "interim_v2" / "features_all_systems_v2.csv"
OUTPUT_FILE   = PROJECT_ROOT / "data" / "processed" / "model_ready_v2.csv"
EXCLUDE       = [12293]
CLIP          = 1.2

# Expanded to simply "lag_" to catch lag_pr_same_slot_d2, lag_pr_recent_day_mean, and lag_power_d2
NO_FILL_PREFIXES = ("lag_",) 

def main():
    if not FEATURES_FILE.exists():
        print(f"ERROR: Cannot find {FEATURES_FILE}. Did you run build_features.py?")
        return

    df = pd.read_csv(FEATURES_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Europe/Brussels")
    n0 = len(df)
    
    if EXCLUDE:
        df = df[~df.system_id.isin(EXCLUDE)]
        print(f"excluded systems {EXCLUDE}: {n0-len(df):,} rows removed")

    m = df[df["is_valid"] & df["is_daylight"]].copy()
    print(f"valid daylight rows: {len(m):,} of {len(df):,}")

    m["target_norm"] = m["target_norm"].clip(0, CLIP)

    non_feat = {"system_id", "timestamp", "target_norm", "target_power_w",
                "is_daylight", "is_valid", "split"}
    fill_cols = [c for c in m.columns
                 if c not in non_feat and not c.startswith(NO_FILL_PREFIXES)]
    
    if m[fill_cols].isna().any().any():
        m = m.sort_values(["system_id", "timestamp"])
        m[fill_cols] = m.groupby("system_id")[fill_cols].ffill().bfill()
        
    lag_cols = [c for c in m.columns if c.startswith(NO_FILL_PREFIXES)]
    print(f"gap-filled {len(fill_cols)} feature cols; left {len(lag_cols)} lag cols unfilled")

    m = m.sort_values("timestamp")
    q70, q85 = m["timestamp"].quantile(0.70), m["timestamp"].quantile(0.85)
    m["split"] = np.where(m["timestamp"] <= q70, "train",
                          np.where(m["timestamp"] <= q85, "val", "test"))
    print("split boundaries:  train <= %s  |  val <= %s  |  test after"
          % (q70.date(), q85.date()))

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    m.to_csv(OUTPUT_FILE, index=False)
    print(f"\nsaved {len(m):,} rows x {m.shape[1]} cols -> {OUTPUT_FILE}")

if __name__ == "__main__":
    main()