#!/usr/bin/env python3
"""
mos_satellite.py
================
Re-runs Model Output Statistics (MOS) against SATELLITE ground truth instead
of ERA5, and keeps ERA5 as a control arm so the difference is measured rather
than argued.

WHY RE-RUN IT
-------------
The parent project's MOS branch (train_mos_model.py, tune_mos_optuna.py,
tune_mos_benchmark.py) reported a best error reduction of 1.41%, with Ridge
going negative at -0.17%. That looked like "MOS does not work here".

It was not MOS that failed - it was the target. `actual_ghi` in
mos_training_data.csv is ERA5, a reanalysis produced by ECMWF with the same
IFS model whose forecast MOS was being asked to correct. Shared physics had
already cancelled most of the error, so there was almost nothing left to learn.

Measured over this study's window and systems:

    reference        ECMWF rRMSE   RMSE (W/m2)   bias MBE (W/m2)
    ERA5                 16.5 %        51.1           -1.5
    SARAH-3 satellite    29.0 %        86.6           +9.2
    literature           ~28.8 %         -          positive

MOS trained on ERA5 was chasing a -1.5 W/m2 bias. Against satellite truth
there is a real +9.2 W/m2 bias and 86.6 W/m2 of error to attack.

WHAT THIS SCRIPT DOES
---------------------
Target (what MOS learns to predict):

    error = truth_ghi - ecmwf_ghi          then    corrected = ecmwf_ghi + predicted_error

Two arms, identical in every other respect:
  SATELLITE : truth = EUMETSAT SARAH-3 (independent observation)
  ERA5      : truth = ERA5 (the parent's original target, as a control)

FEATURE UPGRADE OVER THE PARENT
-------------------------------
The parent used ecmwf_ghi, ecmwf_cloud, ecmwf_temp, hour, month, day_of_year,
and its own comment notes clear-sky GHI was omitted "to save API calls". That
omission matters: without clear-sky, the model cannot tell a dim winter noon
from a bright summer dawn, so it cannot express the error as a fraction of
available light. Clear-sky GHI and the clear-sky index are joined here from
model_ready_v3.csv, which already has them.

Every feature is available at forecast time - no leakage.

SPLITS
------
Uses the project's own train/val/test `split` column so results line up with
every other experiment in this tree. Splits are chronological.

Outputs: v3_ensemble/models/mos_satellite_metrics.csv
         v3_ensemble/models/mos_satellite_corrected.csv
         v3_ensemble/reports/mos_satellite_report.txt
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as C
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval import metrics as M

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

FEATURES = [
    "ecmwf_ghi", "ecmwf_cloud", "ecmwf_temp",
    "clearsky_ghi", "kt_fc",          # the parent's missing physics
    "sun_elevation",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos",
]
EPS = 1.0


def build():
    """Join forecast, both truth sources, and geometry into one table."""
    mos = pd.read_csv(C.PROJECT_ROOT / "data" / "interim" / "mos_training_data.csv")
    mos["timestamp"] = pd.to_datetime(mos["timestamp"], utc=True)
    mos = mos.dropna(subset=["system_id", "actual_ghi", "ecmwf_ghi"])
    mos["system_id"] = mos["system_id"].astype(int)
    mos = mos[~mos.system_id.isin(C.EXCLUDE_SYSTEMS)]
    mos = mos.rename(columns={"actual_ghi": "era5_ghi"})

    sat = pd.read_csv(C.DATA_INTERIM / "satellite_ghi_all.csv")
    sat["timestamp"] = pd.to_datetime(sat["time"], utc=True)
    sat = sat[["system_id", "timestamp", "sat_ghi"]].dropna(subset=["sat_ghi"])

    geo = pd.read_csv(C.MODEL_READY_V3, usecols=[
        "system_id", "timestamp", "clearsky_ghi", "sun_elevation",
        "hour_sin", "hour_cos", "doy_sin", "doy_cos", "split"])
    geo["timestamp"] = pd.to_datetime(geo["timestamp"], utc=True)

    df = mos.merge(sat, on=["system_id", "timestamp"], how="inner")
    df = df.merge(geo, on=["system_id", "timestamp"], how="inner")
    df = df[df.sun_elevation > 5].reset_index(drop=True)
    df["kt_fc"] = (df.ecmwf_ghi / (df.clearsky_ghi + EPS)).clip(0, 1.3)
    return df.sort_values(["timestamp", "system_id"]).reset_index(drop=True)


def make_models():
    m = {
        "Ridge": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=150, max_depth=12, min_samples_leaf=20,
            random_state=42, n_jobs=-1),
        "LightGBM": lgb.LGBMRegressor(
            n_estimators=400, learning_rate=0.05, num_leaves=64,
            min_child_samples=50, subsample=0.9, colsample_bytree=0.9,
            random_state=42, n_jobs=-1, verbose=-1),
    }
    if XGBRegressor is not None:
        m["XGBoost"] = XGBRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=8,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            random_state=42, n_jobs=-1)
    return m


def score(truth, pred, label, n):
    e = pred - truth
    ss_tot = ((truth - truth.mean()) ** 2).sum()
    return {
        "model": label, "n": n,
        "MAE": np.abs(e).mean(),
        "RMSE": np.sqrt((e ** 2).mean()),
        "MBE": e.mean(),
        "rRMSE_%": 100 * np.sqrt((e ** 2).mean()) / truth.mean(),
        "R2": 1 - (e ** 2).sum() / ss_tot,
    }


def run_arm(df, truth_col, arm_name, out_rows, corrected_store):
    print("\n" + "=" * 74)
    print(f"ARM: MOS trained against {arm_name}")
    print("=" * 74, flush=True)

    d = df.copy()
    d["error_target"] = d[truth_col] - d["ecmwf_ghi"]

    tr = d[d.split == "train"]
    va = d[d.split == "val"]
    te = d[d.split == "test"]
    print(f"train={len(tr):,}  val={len(va):,}  test={len(te):,}", flush=True)

    truth_te = te[truth_col].to_numpy()
    raw = te["ecmwf_ghi"].to_numpy()
    base = score(truth_te, raw, "RAW ECMWF (no MOS)", len(te))
    base["arm"] = arm_name
    base["MAE_reduction_%"] = 0.0
    base["RMSE_reduction_%"] = 0.0
    out_rows.append(base)
    print(f"  RAW ECMWF        MAE={base['MAE']:7.2f}  RMSE={base['RMSE']:7.2f}  "
          f"MBE={base['MBE']:+7.2f}  R2={base['R2']:.4f}", flush=True)

    # train on train+val (chronological, both precede test)
    fit = pd.concat([tr, va])
    for name, model in make_models().items():
        t0 = time.time()
        model.fit(fit[FEATURES], fit["error_target"])
        corrected = np.clip(raw + model.predict(te[FEATURES]), 0, None)
        r = score(truth_te, corrected, name, len(te))
        r["arm"] = arm_name
        r["MAE_reduction_%"] = 100 * (base["MAE"] - r["MAE"]) / base["MAE"]
        r["RMSE_reduction_%"] = 100 * (base["RMSE"] - r["RMSE"]) / base["RMSE"]
        r["fit_s"] = round(time.time() - t0, 1)
        out_rows.append(r)
        print(f"  {name:16s} MAE={r['MAE']:7.2f}  RMSE={r['RMSE']:7.2f}  "
              f"MBE={r['MBE']:+7.2f}  R2={r['R2']:.4f}  "
              f"MAE-red={r['MAE_reduction_%']:+6.2f}%  RMSE-red={r['RMSE_reduction_%']:+6.2f}%",
              flush=True)
        if arm_name.startswith("SARAH-3") and name == "LightGBM":
            keep = te[["system_id", "timestamp", "ecmwf_ghi", truth_col,
                       "clearsky_ghi", "kt_fc"]].copy()
            keep["ecmwf_ghi_mos"] = corrected
            corrected_store.append(keep)


def main():
    df = build()
    print(f"rows={len(df):,}  systems={df.system_id.nunique()}  "
          f"window {df.timestamp.min():%Y-%m-%d} -> {df.timestamp.max():%Y-%m-%d}")
    print(f"features ({len(FEATURES)}): {FEATURES}")

    rows, store = [], []
    run_arm(df, "sat_ghi", "SARAH-3 satellite (independent)", rows, store)
    run_arm(df, "era5_ghi", "ERA5 (parent's original target, control)", rows, store)

    res = pd.DataFrame(rows)[
        ["arm", "model", "n", "MAE", "RMSE", "MBE", "rRMSE_%", "R2",
         "MAE_reduction_%", "RMSE_reduction_%"]].round(3)
    res.to_csv(C.MODELS_DIR / "mos_satellite_metrics.csv", index=False)
    if store:
        pd.concat(store, ignore_index=True).to_csv(
            C.MODELS_DIR / "mos_satellite_corrected.csv", index=False)

    print("\n" + "=" * 74)
    print("SIDE BY SIDE")
    print("=" * 74)
    piv = res.pivot(index="model", columns="arm", values="MAE_reduction_%")
    print("\nMAE reduction from MOS (%), by truth source")
    print(piv.to_string())
    print("\nParent project reported, on ERA5: best 1.41% (LightGBM), Ridge -0.17%")

    with open(C.REPORTS_DIR / "mos_satellite_report.txt", "w", encoding="utf-8") as f:
        f.write("MOS against satellite vs ERA5 truth\n" + "=" * 74 + "\n\n")
        f.write(f"features: {', '.join(FEATURES)}\n\n")
        f.write(res.to_string(index=False) + "\n\n")
        f.write("MAE reduction (%) by truth source\n" + piv.to_string() + "\n")
    print(f"\nWrote {C.MODELS_DIR / 'mos_satellite_metrics.csv'}")
    print(f"Wrote {C.REPORTS_DIR / 'mos_satellite_report.txt'}")


if __name__ == "__main__":
    main()
