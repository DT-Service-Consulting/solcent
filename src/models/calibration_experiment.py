# Relative Path: src/models/calibration_experiment.py
#!/usr/bin/env python3
"""
calibration_experiment.py
=========================
How much does a SHORT calibration period on a new home recover the skill that the
fleet model loses when forecasting an unseen home?
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneGroupOut

from model_baselines import make_model, metrics, skill, FEATURES, TARGET, CLIP

# ----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_READY  = PROJECT_ROOT / "data" / "processed" / "model_ready.csv"
OUTPUT_DIR   = PROJECT_ROOT / "models" / "pv_model"
EXCLUDE      = [12293]                    
CALIB_DAYS   = [0, 3, 7, 14, 30]          
MAX_SYSTEMS  = None                       
TAIL_SYSTEMS = [16111, 94362, 81376, 2979] # Updated with the worst performers from the LOSO run
# ----------------------------------------------------------------------------

def bias_correct(pred, resid_mean):
    """Subtract the mean calibration error (a single-number bias fix)."""
    return np.clip(pred - resid_mean, 0, CLIP)

def linear_rescale(pred_cal, actual_cal, pred_eval):
    """Fit actual ~ a + b*pred on calibration days; apply to eval predictions."""
    if len(pred_cal) < 10 or np.std(pred_cal) < 1e-6:
        return None
    b, a = np.polyfit(pred_cal, actual_cal, 1)
    return np.clip(a + b * pred_eval, 0, CLIP)

def main():
    if not MODEL_READY.exists():
        print(f"Error: Cannot find {MODEL_READY}. Did you run make_model_ready.py?")
        return
        
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(MODEL_READY)
    df = df[~df.system_id.isin(EXCLUDE)].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Europe/Brussels")
    df["day"] = df["timestamp"].dt.normalize()
    df = df.sort_values(["system_id", "timestamp"]).reset_index(drop=True)
    feats = [c for c in FEATURES if c in df.columns]

    systems = sorted(df.system_id.unique())
    if MAX_SYSTEMS:
        systems = systems[:MAX_SYSTEMS]
        df = df[df.system_id.isin(systems)]
    print(f"{len(df):,} rows | {len(systems)} systems (excluded {EXCLUDE}) | {len(feats)} features")

    df["pred_fleet"] = np.nan
    logo = LeaveOneGroupOut()
    for k, (tr, te) in enumerate(logo.split(df[feats], df[TARGET], df["system_id"]), 1):
        name, model = make_model()
        model.fit(df.iloc[tr][feats], df.iloc[tr][TARGET])
        df.iloc[te, df.columns.get_loc("pred_fleet")] = np.clip(
            model.predict(df.iloc[te][feats]), 0, CLIP)
        print(f"  fit {k}/{len(systems)} done", end="\r")
    print("\nbase fleet predictions ready.")

    curve = []
    per_sys_rmse = {}          
    for nd in CALIB_DAYS:
        per_sys = []
        for sid, g in df.groupby("system_id"):
            g = g.sort_values("timestamp")
            days = np.sort(g["day"].unique())
            if nd > 0 and len(days) <= nd + 5:
                continue
            calib_days = set(days[:nd])
            cal = g[g["day"].isin(calib_days)]
            ev = g[~g["day"].isin(calib_days)]
            if ev.empty:
                continue
            a = ev[TARGET].values
            if nd == 0:
                pred = ev["pred_fleet"].values
            else:
                resid_mean = float((cal["pred_fleet"] - cal[TARGET]).mean())
                pred_bias = bias_correct(ev["pred_fleet"].values, resid_mean)
                pred_lin = linear_rescale(cal["pred_fleet"].values, cal[TARGET].values,
                                          ev["pred_fleet"].values)
                pred = pred_lin if pred_lin is not None else pred_bias
            rmse = float(np.sqrt(np.mean((pred - a) ** 2)))
            per_sys.append({"system_id": sid, "rmse": rmse})
            per_sys_rmse.setdefault(sid, {})[nd] = rmse
        if per_sys:
            mean_rmse = np.mean([p["rmse"] for p in per_sys])
            curve.append({"calib_days": nd, "mean_rmse": mean_rmse, "n_systems": len(per_sys)})
            print(f"  calib {nd:>2}d: mean per-system RMSE = {mean_rmse:.4f} "
                  f"({len(per_sys)} systems)")

    cdf = pd.DataFrame(curve)
    base = cdf.loc[cdf.calib_days == 0, "mean_rmse"].iloc[0]
    cdf["rmse_reduction_%"] = (100 * (base - cdf["mean_rmse"]) / base).round(1)
    cdf.to_csv(OUTPUT_DIR / "calibration_curve.csv", index=False)
    print("\n=============== CALIBRATION CURVE (all unseen homes) ===============")
    print(cdf.to_string(index=False))

    tail = [s for s in TAIL_SYSTEMS if s in per_sys_rmse]
    rest = [s for s in per_sys_rmse if s not in TAIL_SYSTEMS]

    def group_curve(sids):
        out = []
        for nd in CALIB_DAYS:
            vals = [per_sys_rmse[s][nd] for s in sids if nd in per_sys_rmse[s]]
            if vals:
                out.append((nd, float(np.mean(vals))))
        return out

    print("\n=============== TAIL (atypical) vs REST (healthy) ===============")
    print(f"  tail homes: {tail}")
    for label, sids in [("TAIL (atypical)", tail), ("REST (healthy)", rest)]:
        gc = group_curve(sids)
        if not gc:
            continue
        b = gc[0][1]
        print(f"\n  {label}  ({len(sids)} systems)")
        print(f"    {'days':>5} {'mean_RMSE':>10} {'reduction_%':>12}")
        for nd, v in gc:
            print(f"    {nd:>5} {v:>10.4f} {100*(b-v)/b:>11.1f}%")
            
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(cdf["calib_days"], cdf["mean_rmse"], marker="o", lw=2, color="#1f6fb5")
    for _, r in cdf.iterrows():
        ax.annotate(f"-{r['rmse_reduction_%']:.0f}%", (r["calib_days"], r["mean_rmse"]),
                    textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    ax.set_xlabel("calibration days on the new home"); ax.set_ylabel("mean per-system RMSE (unseen)")
    ax.set_title("How a short calibration recovers forecast skill on a new home")
    ax.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "calibration_curve.png", dpi=95); plt.close()
    print(f"\nsaved -> {OUTPUT_DIR/'calibration_curve.csv'}, {OUTPUT_DIR/'calibration_curve.png'}")

if __name__ == "__main__":
    main()