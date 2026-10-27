#!/usr/bin/env python3
"""
baselines.py
============
Reference forecasts that do not use the machine-learning model, scored on the
same test rows as the manuscript:

  persistence        output at the same slot one day earlier (the skill reference)
  smart persistence  persistence rescaled by the clear-sky ratio,
                     y(t-1d) * GHIcs(t) / GHIcs(t-1d); plain persistence where
                     GHIcs(t-1d) < 50 W/m2
  physical chain     ECMWF plane-of-array irradiance (Hay-Davies, reported tilt
                     and azimuth) -> Faiman cell temperature -> PVWatts DC model
                     (gamma = -0.4 %/K) with 14 % system losses, normalised by
                     nameplate capacity. Needs no training data, so it is a
                     zero-shot forecast for every rooftop.
  physical + scale   the physical chain times one factor per rooftop, fitted by
                     least squares on that rooftop's training split. A
                     site-calibrated physical model for known rooftops.

Scores: pooled R2, skill against persistence, RMSE and MBE over all test rows,
and the per-rooftop distribution (median, 10th percentile, worst, number of
rooftops with skill <= 0) for comparison with the LOSO results.

Output: models/revision/baselines_pooled.csv
        models/revision/baselines_per_system.csv
        reports/baselines.txt
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as C

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval import metrics as M

OUT = C.MODELS_DIR / "revision"
LOSSES = 0.14
GAMMA = -0.004


def load():
    df = pd.read_csv(C.MODEL_READY_V3, usecols=[
        "timestamp", "system_id", "target_norm", "is_daylight", "is_valid", "split",
        "clearsky_ghi", "poa_global_fc", "temp_fc", "wind_fc"])
    df = df[~df.system_id.isin(C.EXCLUDE_SYSTEMS)]
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["system_id", "timestamp"]).reset_index(drop=True)
    prev = df[["system_id", "timestamp", C.TARGET, "clearsky_ghi"]].copy()
    prev["timestamp"] += pd.Timedelta(days=1)
    prev = prev.rename(columns={C.TARGET: "pred_persistence", "clearsky_ghi": "cs_prev"})
    df = df.merge(prev, on=["system_id", "timestamp"], how="left")
    df = df[(df.is_valid == True) & (df.is_daylight == True)].copy()
    df["pred_persistence"] = df["pred_persistence"].fillna(0.0)
    return df


def add_baselines(df):
    ratio = np.where(df.cs_prev >= 50, df.clearsky_ghi / df.cs_prev.where(df.cs_prev >= 50), 1.0)
    df["pred_smart"] = np.clip(df.pred_persistence * np.nan_to_num(ratio, nan=1.0), 0, C.CLIP)

    # POA is capped at 1400 W/m2: the transposition produces rare spikes at low
    # sun (up to ~8000 W/m2 in the feature table) that no real array receives.
    poa = df.poa_global_fc.clip(lower=0, upper=1400).fillna(0).to_numpy()
    t_air = df.temp_fc.fillna(df.temp_fc.median()).to_numpy()
    # Open-Meteo serves wind_speed_10m in km/h by default (median 13.4 here);
    # the Faiman model expects m/s.
    wind = df.wind_fc.fillna(3.6).clip(lower=0).to_numpy() / 3.6
    t_cell = pvlib.temperature.faiman(poa, t_air, wind)
    dc = pvlib.pvsystem.pvwatts_dc(poa, t_cell, pdc0=1.0, gamma_pdc=GAMMA)
    df["pred_physical"] = np.clip(np.asarray(dc) * (1 - LOSSES), 0, C.CLIP)

    tr = df[df.split == "train"]
    k = tr.groupby("system_id").apply(
        lambda g: float((g.pred_physical * g[C.TARGET]).sum() / (g.pred_physical ** 2).sum()),
        include_groups=False)
    df["pred_physical_scaled"] = np.clip(df.pred_physical * df.system_id.map(k), 0, C.CLIP)
    return df, k


def scores(te, col):
    y, p = te[C.TARGET].to_numpy(), te[col].to_numpy()
    return {"MAE": M.mae(y, p), "RMSE": M.rmse(y, p), "MBE": M.mbe(y, p),
            "R2": M.r2(y, p), "skill": M.skill_score(y, p, te.pred_persistence.to_numpy())}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df, k = add_baselines(load())
    te = df[df.split == "test"]
    arms = {"persistence": "pred_persistence", "smart persistence": "pred_smart",
            "physical chain (nameplate)": "pred_physical",
            "physical chain + per-rooftop scale": "pred_physical_scaled"}

    pooled = pd.DataFrame([{"baseline": a, "systems": te.system_id.nunique(), **scores(te, c)}
                           for a, c in arms.items()])
    per = pd.DataFrame([{"baseline": a, "system_id": sid, **scores(g, c)}
                        for a, c in arms.items() for sid, g in te.groupby("system_id")])
    dist = per.groupby("baseline").agg(
        median_R2=("R2", "median"), mean_R2=("R2", "mean"),
        p10_R2=("R2", lambda s: s.quantile(0.10)), worst_R2=("R2", "min"),
        median_skill=("skill", "median"),
        failures=("skill", lambda s: int((s <= 0).sum()))).reindex(list(arms))

    pooled.to_csv(OUT / "baselines_pooled.csv", index=False)
    per.to_csv(OUT / "baselines_per_system.csv", index=False)
    text = ("Baselines on the chronological test split (all systems)\n" + "=" * 60 + "\n\n"
            + pooled.to_string(index=False, float_format=lambda x: f"{x:.4f}") + "\n\n"
            + "Per-rooftop distribution\n"
            + dist.to_string(float_format=lambda x: f"{x:.4f}") + "\n\n"
            + f"per-rooftop scale factors (physical + scale): median {k.median():.3f}, "
              f"range {k.min():.3f}-{k.max():.3f}\n")
    (C.REPORTS_DIR / "baselines.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
