#!/usr/bin/env python3
"""
loso_experiment.py
==================
Tests how well the lightweight global model performs on a COMPLETELY UNSEEN
customer's house in the FUTURE, using Walk-Forward Spatio-Temporal validation.

UPDATED: 
  - Now handles the MultiQuantileLGBM dictionary output.
  - Evaluates P50 for strict accuracy (R2, RMSE) and P25 for battery safety (MBE).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import the optimized Multi-Quantile model and features
from model_baselines import make_model, metrics, skill, FEATURES, TARGET, CLIP

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_READY  = PROJECT_ROOT / "data" / "processed" / "model_ready.csv"
OUTPUT_DIR   = PROJECT_ROOT / "models" / "pv_model"

def main():
    if not MODEL_READY.exists():
        print(f"Error: Cannot find {MODEL_READY}. Did you run make_model_ready.py?")
        return
        
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    EXCLUDE=[12293]
    df = pd.read_csv(MODEL_READY)
    df = df[~df.system_id.isin(EXCLUDE)].reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Europe/Brussels")
    df = df.sort_values(["system_id", "timestamp"]).reset_index(drop=True)
    
    feats = [c for c in FEATURES if c in df.columns]
    
    print(f"\n[INFO] Running LOSO validation using {len(feats)} pruned features.\n")
    
    system_ids = df["system_id"].unique()
    n_sys = len(system_ids)

    prev = df[["system_id", "timestamp", TARGET]].copy()
    prev["timestamp"] += pd.Timedelta(days=1)
    prev = prev.rename(columns={TARGET: "pred_persistence"})
    df = df.merge(prev, on=["system_id", "timestamp"], how="left")
    src = "poa_global_fc" if "poa_global_fc" in df.columns else "clearsky_ghi"
    df["pred_clearsky"] = (df[src] / 1000.0).clip(0, CLIP)
    
    # Initialize target columns for the multi-quantile outputs
    df["pred_p50"] = np.nan
    df["pred_p25"] = np.nan

    print(f"Strict Spatio-Temporal Leave-One-System-Out: {n_sys} model fits...")
    
    # Establish a rigid chronological split (75% past / 25% future)
    time_split_date = df['timestamp'].quantile(0.75)
    print(f"Global Time Split Cutoff: {time_split_date.date()}")

    for k, held_out_sys in enumerate(system_ids, 1):
        # TRAIN: All OTHER systems, STRICTLY BEFORE the time cutoff
        train_mask = (df["system_id"] != held_out_sys) & (df["timestamp"] <= time_split_date)
        
        # TEST: The HELD OUT system, STRICTLY AFTER the time cutoff
        test_mask = (df["system_id"] == held_out_sys) & (df["timestamp"] > time_split_date)
        
        name, model = make_model()
        model.fit(df.loc[train_mask, feats], df.loc[train_mask, TARGET])
        
        # Predict returns a dictionary of quantiles
        predictions = model.predict(df.loc[test_mask, feats])
        
        df.loc[test_mask, "pred_p50"] = predictions["pred_model_p50"]
        df.loc[test_mask, "pred_p25"] = predictions["pred_model_p25"]
            
        print(f"  fit {k}/{n_sys}: trained on {df.loc[train_mask, 'system_id'].nunique()} systems (past), "
              f"predicted {held_out_sys} (future)", end="\r")

    print("\n\nPredictions ready. Evaluating Future Unseen subset...")
    
    # Evaluate ONLY on the future test segments
    ev = df[df["timestamp"] > time_split_date].dropna(subset=["pred_p50", "pred_persistence"]).copy()
    
    rows = []
    for sid, g in ev.groupby("system_id"):
        if len(g) < 100: continue
        
        # Evaluate P50 for Accuracy
        m_p50 = metrics(g[TARGET], g["pred_p50"], g["capacity_kwp"], alpha=0.50)
        # Evaluate P25 for Safety
        m_p25 = metrics(g[TARGET], g["pred_p25"], g["capacity_kwp"], alpha=0.25)
        
        rmse_p = metrics(g[TARGET], g["pred_persistence"], g["capacity_kwp"])["RMSE"]
        
        rows.append({
            "system_id": sid, 
            "n_test": len(g),
            "P50_RMSE": m_p50["RMSE"], 
            "P50_R2": m_p50["R2"], 
            "P25_MBE": m_p25["MBE"],
            "skill_vs_persist": skill(m_p50["RMSE"], rmse_p)
        })
                     
    per = pd.DataFrame(rows).sort_values("skill_vs_persist", ascending=False)
    
    # Calculate Pooled (Global) Metrics
    pooled_p50 = metrics(ev[TARGET], ev["pred_p50"], ev["capacity_kwp"], alpha=0.50)
    pooled_p25 = metrics(ev[TARGET], ev["pred_p25"], ev["capacity_kwp"], alpha=0.25)

    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print("\n=============== PER-SYSTEM UNSEEN PERFORMANCE (Spatio-Temporal) ===============")
    print(per.to_string(index=False))
    print("\n=============== POOLED UNSEEN PERFORMANCE ===============")
    print(f"  P50 RMSE (Accuracy) : {pooled_p50['RMSE']:.4f} | R2 {pooled_p50['R2']:.4f}")
    print(f"  P25 MBE (Safety)    : {pooled_p25['MBE']:.4f}  <-- Should be negative")
    
    per.to_csv(OUTPUT_DIR / "loso_per_system.csv", index=False)

if __name__ == "__main__":
    main()