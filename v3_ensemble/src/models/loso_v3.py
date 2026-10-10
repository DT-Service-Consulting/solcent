#!/usr/bin/env python3
"""
loso_v3.py
==========
Leave-One-System-Out + calibration sweep. THIS is the experiment that matches
the stated product goal: train a global model on a fleet, drop it on a rooftop
it has never seen, optionally hand it a few weeks of that roof's own data.

WHY THE OTHER ARMS DO NOT ANSWER THIS
-------------------------------------
model_ready.csv splits purely by TIME: all 33 systems appear in train, val and
test. So train_v3.py measures "known rooftop, future date" - the model has
already learned each roof's shading, orientation quirks and inverter
behaviour. A new customer's roof gives none of that. Every pooled R2 in this
project is therefore an optimistic bound on the deployed case.

WHAT THIS SCRIPT DOES
---------------------
For each system S:
  1. Train on the other 32 systems only (train+val period).
  2. Predict S over its test period. That is the ZERO-SHOT number: what a new
     customer sees on day one.
  3. Take the first N days of S's test window as CALIBRATION data (the "few
     weeks of data" in the plan), fit a correction, and score the remaining
     days. N in CALIB_DAYS.

CORRECTIONS COMPARED (all post-hoc - no refit, so they are cheap enough to run
on a real deployment)
  none      : raw global model.
  bias      : subtract the mean residual seen during calibration. One number.
  linear    : fit actual ~ a + b*pred on calibration days, apply to the rest.
  cap_emp   : re-normalise by the roof's OBSERVED capacity (p99 of calibration
              power) instead of the PVOutput nameplate capacity_kwp.

Why cap_emp is in here: nameplate capacity is inconsistent across the fleet -
per-system p99.9 of target_norm ranges 0.669 to 1.200, a 1.8x spread in what
"100% output" means. On the pooled temporal split that barely matters, because
the model memorises each roof's scale during training. On an UNSEEN roof it
cannot, so the nameplate error transfers straight into the forecast. This is
the one place empirical capacity should pay for itself.

Output: v3_ensemble/models/loso_v3_per_system.csv
        v3_ensemble/models/loso_v3_calibration.csv
        v3_ensemble/reports/loso_v3_report.txt
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as C
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval import metrics as M

import lightgbm as lgb

CALIB_DAYS = [0, 3, 7, 14, 30]
# LOSO needs one fit per held-out system, so the tree count is trimmed from
# 500 to keep 33 fits tractable. Comparisons stay internally consistent.
LOSO_PARAMS = dict(C.LGBM_PARAMS, n_estimators=300)


def load():
    path = C.MODEL_READY_V3 if C.MODEL_READY_V3.exists() else C.PARENT_MODEL_READY
    print(f"Loading {path.name} ...", flush=True)
    df = pd.read_csv(path)
    df = df[~df.system_id.isin(C.EXCLUDE_SYSTEMS)]
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["system_id", "timestamp"]).reset_index(drop=True)

    prev = df[["system_id", "timestamp", C.TARGET]].copy()
    prev["timestamp"] += pd.Timedelta(days=1)
    prev = prev.rename(columns={C.TARGET: "pred_persistence"})
    df = df.merge(prev, on=["system_id", "timestamp"], how="left")

    df = df[(df.is_valid == True) & (df.is_daylight == True)].copy()
    df["pred_persistence"] = df["pred_persistence"].fillna(0.0)
    df["day"] = df["timestamp"].dt.normalize()
    return df.fillna(0)


def corrections(cal, ev, cap_kwp, fleet_cap_ratio):
    """Return {name: corrected prediction array} for the evaluation slice."""
    out = {"none": ev["pred"].to_numpy()}
    if len(cal) < 10:
        return out

    # mean residual on the calibration window; positive = model under-forecasts
    resid = (cal[C.TARGET] - cal["pred"]).mean()
    out["bias"] = np.clip(ev["pred"].to_numpy() + resid, 0, C.CLIP)

    if cal["pred"].std() > 1e-6:
        b, a = np.polyfit(cal["pred"], cal[C.TARGET], 1)
        out["linear"] = np.clip(a + b * ev["pred"].to_numpy(), 0, C.CLIP)

    # Capacity correction, RELATIVE TO THE FLEET.
    #
    # The model is trained on target_norm = power / nameplate across many
    # roofs, so it learns the FLEET-AVERAGE relationship between nameplate and
    # real output - and that average is not 1.0 (measured fleet median 0.794:
    # a typical roof peaks at 79% of its nameplate). A new roof whose own
    # ratio is below the fleet median will be over-forecast, and vice versa.
    #
    # So the scale factor is this roof's ratio divided by the fleet's, NOT
    # nameplate/observed. An earlier version used the latter, which pushed
    # every prediction in the wrong direction and drove median R2 to -0.77.
    obs_kw = np.percentile(cal["target_power_w"], 99) / 1000.0
    if obs_kw > 0.1 and cap_kwp > 0.1 and fleet_cap_ratio > 0.1:
        scale = (obs_kw / cap_kwp) / fleet_cap_ratio
        scale = float(np.clip(scale, 0.4, 2.5))
        out["cap_emp"] = np.clip(ev["pred"].to_numpy() * scale, 0, C.CLIP)
        # capacity rescale first, then whatever bias is left over
        resid2 = (cal[C.TARGET] - cal["pred"] * scale).mean()
        out["cap_emp+bias"] = np.clip(ev["pred"].to_numpy() * scale + resid2,
                                      0, C.CLIP)
    return out


def main():
    df = load()
    base = [c for c in C.BASE_FEATURES if c in df.columns]
    ens = [c for c in C.ENSEMBLE_FEATURES if c in df.columns]
    feats = base + ens
    systems = sorted(df.system_id.unique())
    print(f"LOSO over {len(systems)} systems, {len(feats)} features: {feats}")
    print(f"calibration days: {CALIB_DAYS}\n", flush=True)

    zero_rows, calib_rows, pred_frames = [], [], []

    for i, sid in enumerate(systems, 1):
        tr = df[(df.system_id != sid) & (df.split.isin(["train", "val"]))]
        te = df[(df.system_id == sid) & (df.split == "test")].copy()
        if len(te) < 500:
            print(f"[{i}/{len(systems)}] {sid} skipped (only {len(te)} test rows)")
            continue

        # fleet capacity ratio from the TRAINING systems only (never the
        # held-out roof) - the baseline the model has effectively learned
        fleet_cap_ratio = float(np.median([
            np.percentile(g["target_power_w"], 99) / 1000.0 / g["capacity_kwp"].iloc[0]
            for _, g in tr.groupby("system_id")]))

        m = lgb.LGBMRegressor(objective="quantile", alpha=0.50,
                              **LOSO_PARAMS).fit(tr[feats], tr[C.TARGET])
        te["pred"] = np.clip(m.predict(te[feats]), 0, C.CLIP)

        y = te[C.TARGET].to_numpy()
        zero = {
            "system_id": sid, "n": len(te),
            "R2": M.r2(y, te["pred"]), "RMSE": M.rmse(y, te["pred"]),
            "MBE": M.mbe(y, te["pred"]),
            "skill": M.skill_score(y, te["pred"], te["pred_persistence"]),
            "capacity_kwp": te["capacity_kwp"].iloc[0],
            "cap_obs_kw": np.percentile(te["target_power_w"], 99) / 1000.0,
        }
        zero["cap_ratio"] = zero["cap_obs_kw"] / zero["capacity_kwp"]
        zero_rows.append(zero)

        days = np.sort(te["day"].unique())
        for nd in CALIB_DAYS:
            if nd == 0:
                cal, ev = te.iloc[:0], te
            else:
                if len(days) <= nd + 5:
                    continue
                cut = days[nd]
                cal, ev = te[te.day < cut], te[te.day >= cut]
            if len(ev) < 200:
                continue
            ye = ev[C.TARGET].to_numpy()
            for name, p in corrections(cal, ev, float(te["capacity_kwp"].iloc[0]),
                                       fleet_cap_ratio).items():
                calib_rows.append({
                    "system_id": sid, "calib_days": nd, "method": name,
                    "n_eval": len(ev), "R2": M.r2(ye, p), "RMSE": M.rmse(ye, p),
                    "MBE": M.mbe(ye, p),
                    "skill": M.skill_score(ye, p, ev["pred_persistence"].to_numpy()),
                })

        pred_frames.append(te[["system_id", "timestamp", C.TARGET,
                               "target_power_w", "capacity_kwp",
                               "pred", "pred_persistence", "day"]])
        print(f"[{i}/{len(systems)}] {sid}  zero-shot R2={zero['R2']:.4f} "
              f"skill={zero['skill']:.4f} cap_ratio={zero['cap_ratio']:.3f}",
              flush=True)

    zs = pd.DataFrame(zero_rows).sort_values("R2")
    cb = pd.DataFrame(calib_rows)
    zs.to_csv(C.MODELS_DIR / "loso_v3_per_system.csv", index=False)
    pd.concat(pred_frames, ignore_index=True).to_csv(
        C.MODELS_DIR / "loso_v3_predictions.csv", index=False)
    cb.to_csv(C.MODELS_DIR / "loso_v3_calibration.csv", index=False)

    print("\n" + "=" * 74)
    print("ZERO-SHOT: global model on a rooftop it has never seen")
    print("=" * 74)
    print(zs.to_string(index=False))
    print(f"\nmedian R2 {zs.R2.median():.4f}   mean R2 {zs.R2.mean():.4f}   "
          f"worst {zs.R2.min():.4f}   best {zs.R2.max():.4f}")
    print(f"median skill {zs.skill.median():.4f}")
    print(f"\ncap_ratio (observed p99 kW / nameplate kWp): "
          f"min {zs.cap_ratio.min():.3f}  median {zs.cap_ratio.median():.3f}  "
          f"max {zs.cap_ratio.max():.3f}")
    print("A cap_ratio far from 1.0 means the nameplate is wrong for that roof.")
    print(f"\ncorr(cap_ratio, MBE)      = {zs.cap_ratio.corr(zs.MBE):+.3f}")
    print(f"corr(cap_ratio, R2)       = {zs.cap_ratio.corr(zs.R2):+.3f}")
    print("A strong positive corr with MBE is the signature of nameplate error:")
    print("an overstated nameplate makes the global model over-forecast that roof.")

    print("\n" + "=" * 74)
    print("CALIBRATION SWEEP: does a few weeks of the new roof's data help?")
    print("=" * 74)
    piv = cb.pivot_table(index="calib_days", columns="method",
                         values="R2", aggfunc="median")
    print("\nmedian R2 across systems")
    print(piv.to_string())
    piv2 = cb.pivot_table(index="calib_days", columns="method",
                          values="skill", aggfunc="median")
    print("\nmedian skill vs persistence")
    print(piv2.to_string())

    with open(C.REPORTS_DIR / "loso_v3_report.txt", "w", encoding="utf-8") as f:
        f.write("LOSO + calibration, best v3 config\n" + "=" * 74 + "\n\n")
        f.write("features: " + ", ".join(feats) + "\n\n")
        f.write("ZERO-SHOT per system\n" + zs.to_string(index=False) + "\n\n")
        f.write("median R2 by calibration days x method\n" + piv.to_string() + "\n\n")
        f.write("median skill by calibration days x method\n" + piv2.to_string() + "\n")
    print(f"\nWrote {C.MODELS_DIR / 'loso_v3_per_system.csv'}")
    print(f"Wrote {C.REPORTS_DIR / 'loso_v3_report.txt'}")


if __name__ == "__main__":
    main()
