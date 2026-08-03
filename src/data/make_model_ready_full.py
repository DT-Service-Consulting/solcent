# Relative Path: src/data/make_model_ready_full.py
#!/usr/bin/env python3
"""
make_model_ready_full.py
========================
Prepares the feature table for Deep Learning (TFT) by KEEPING NIGHT ROWS. 
Transformers and RNNs require perfectly continuous time steps without jumps. 
Because the night generation is a deterministic zero, we retain it so the model 
can naturally learn the 24-hour cycle.

Steps: 
  1. Keep all valid rows (night is valid; target is already 0).
  2. Clip target to physical bounds.
  3. Gap-fill non-lag features SAFELY (preserving index).
  4. Time-based split.

Input : data/interim/features_all_systems.csv
Output: data/processed/model_ready_full.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT  = Path(__file__).resolve().parents[2]
FEATURES_FILE = PROJECT_ROOT / "data" / "interim" / "features_all_systems.csv"
OUTPUT_FILE   = PROJECT_ROOT / "data" / "processed" / "model_ready_full.csv"
EXCLUDE       = [12293]
CLIP          = 1.2

# Columns that must NOT be gap-filled (filling a lag would leak future information)
NO_FILL_PREFIXES = ("lag_pr_",)

def main():
    if not FEATURES_FILE.exists():
        print(f"ERROR: Cannot find {FEATURES_FILE}. Did you run build_features.py?")
        return

    df = pd.read_csv(FEATURES_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Europe/Brussels")
    n0 = len(df)
    
    if EXCLUDE:
        df = df[~df.system_id.isin(EXCLUDE)]
        print(f"Excluded systems {EXCLUDE}: {n0-len(df):,} rows removed.")

    # KEEP ALL VALID ROWS (including night)
    m = df[df["is_valid"]].copy()
    print(f"Valid rows (incl. night): {len(m):,}  |  of which daylight: "
          f"{int(m['is_daylight'].sum()):,} ({100*m['is_daylight'].mean():.0f}%)")

    # Clip target to a physical range
    m["target_norm"] = m["target_norm"].clip(0, CLIP)

    # Gap-fill weather/geometry features within each system-time, but NOT lags
    non_feat = {"system_id", "timestamp", "target_norm", "target_power_w",
                "is_daylight", "is_valid", "split"}
    fill_cols = [c for c in m.columns
                 if c not in non_feat and not c.startswith(NO_FILL_PREFIXES)]
    
    # Apply safe index-preserving gap fill
    if m[fill_cols].isna().any().any():
        m = m.sort_values(["system_id", "timestamp"])
        m[fill_cols] = m.groupby("system_id")[fill_cols].ffill().bfill()
        
    lag_cols = [c for c in m.columns if c.startswith(NO_FILL_PREFIXES)]
    print(f"Gap-filled {len(fill_cols)} feature cols; left lag cols unfilled: {lag_cols}")

    # Time-based split (never shuffle time)
    m = m.sort_values("timestamp")
    q70, q85 = m["timestamp"].quantile(0.70), m["timestamp"].quantile(0.85)
    m["split"] = np.where(m["timestamp"] <= q70, "train",
                          np.where(m["timestamp"] <= q85, "val", "test"))
    
    print("Split boundaries:  train <= %s  |  val <= %s  |  test after"
          % (q70.date(), q85.date()))
    print(m["split"].value_counts().to_string())

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    m.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved {len(m):,} rows x {m.shape[1]} cols -> {OUTPUT_FILE}")

if __name__ == "__main__":
    main()