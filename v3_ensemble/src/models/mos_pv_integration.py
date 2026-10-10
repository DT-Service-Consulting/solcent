#!/usr/bin/env python3
"""
mos_pv_integration.py
=====================
Stage 2. MOS demonstrably reduces irradiance error against satellite truth
(Random Forest: -6.37% MAE, -5.02% RMSE, bias +11.40 -> +3.72 W/m2). This
script asks the question that actually matters: does a better irradiance
forecast produce a better PV forecast?

The parent project answered this once already, using ERA5-trained MOS, and
found nothing (XGBoost R2 0.8035 with MOS vs 0.8029 without). That test is
worth repeating now that the correction is real rather than a correction of
a contaminated target.

AVOIDING LEAKAGE - WHY OUT-OF-FOLD MATTERS HERE
-----------------------------------------------
MOS is a pre-processing step, so its output becomes a FEATURE of the PV model.
If MOS is fitted on the training rows and then used to correct those same
rows, it has already seen their answers, and the PV model trains on
unrealistically good irradiance. The measured gain would be fictional.

So corrections are produced out-of-fold:
  - train split : 5 chronological folds; each fold is corrected by a MOS
                  fitted on the other four.
  - val / test  : corrected by a MOS fitted on the full train split.
Every row therefore receives a correction from a model that never saw it.

PROPAGATING THE CORRECTION TO 15 MINUTES
----------------------------------------
MOS operates on hourly irradiance; the PV model runs at 15 minutes. The
correction is carried across as a RATIO rather than an additive offset:

    ratio = ecmwf_ghi_mos / ecmwf_ghi          (hourly, clipped)
    ghi_fc_mos      = ghi_fc      * ratio_interpolated
    poa_global_fc_mos = poa_global_fc * ratio_interpolated

The ratio is interpolated in time across the 15-minute grid. Scaling rather
than adding keeps night-time zeros at zero and preserves the shape of the day.
POA is scaled by the same ratio, holding the ECMWF beam/diffuse split fixed -
the same approximation documented in build_features_v3.py.

ARMS
----
  A  baseline          : the best v3 config, unchanged
  B  MOS-corrected     : ghi_fc and poa_global_fc replaced by corrected versions
  C  baseline + MOS as extra features (both raw and corrected available)

Outputs: v3_ensemble/models/mos_pv_integration_metrics.csv
         v3_ensemble/reports/mos_pv_integration_report.txt
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as C
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval import metrics as M

from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb

MOS_FEATURES = [
    "ecmwf_ghi", "ecmwf_cloud", "ecmwf_temp",
    "clearsky_ghi", "kt_fc", "sun_elevation",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos",
]
EPS = 1.0
N_FOLDS = 5


def mos_model():
    """Random Forest - the arm that won stage 1. Boosted trees over-fitted the
    error target and made irradiance worse."""
    return RandomForestRegressor(n_estimators=150, max_depth=12,
                                 min_samples_leaf=20, random_state=42, n_jobs=-1)


def build_mos_table():
    mos = pd.read_csv(C.PROJECT_ROOT / "data" / "interim" / "mos_training_data.csv")
    mos["timestamp"] = pd.to_datetime(mos["timestamp"], utc=True)
    mos = mos.dropna(subset=["system_id", "ecmwf_ghi"])
    mos["system_id"] = mos["system_id"].astype(int)
    mos = mos[~mos.system_id.isin(C.EXCLUDE_SYSTEMS)]

    sat = pd.read_csv(C.DATA_INTERIM / "satellite_ghi_all.csv")
    sat["timestamp"] = pd.to_datetime(sat["time"], utc=True)
    sat = sat[["system_id", "timestamp", "sat_ghi"]].dropna(subset=["sat_ghi"])

    geo = pd.read_csv(C.MODEL_READY_V3, usecols=[
        "system_id", "timestamp", "clearsky_ghi", "sun_elevation",
        "hour_sin", "hour_cos", "doy_sin", "doy_cos", "split"])
    geo["timestamp"] = pd.to_datetime(geo["timestamp"], utc=True)

    d = mos.merge(sat, on=["system_id", "timestamp"], how="inner")
    d = d.merge(geo, on=["system_id", "timestamp"], how="inner")
    d = d[d.sun_elevation > 5].reset_index(drop=True)
    d["kt_fc"] = (d.ecmwf_ghi / (d.clearsky_ghi + EPS)).clip(0, 1.3)
    d["error_target"] = d.sat_ghi - d.ecmwf_ghi
    return d.sort_values("timestamp").reset_index(drop=True)


def out_of_fold_corrections(d):
    """Correction for every row, always from a model that never saw it."""
    d = d.copy()
    d["ghi_mos"] = np.nan

    tr = d[d.split == "train"].sort_values("timestamp")
    folds = np.array_split(tr.index.to_numpy(), N_FOLDS)
    for k, hold in enumerate(folds, 1):
        fit_idx = tr.index.difference(pd.Index(hold))
        m = mos_model().fit(d.loc[fit_idx, MOS_FEATURES], d.loc[fit_idx, "error_target"])
        d.loc[hold, "ghi_mos"] = (d.loc[hold, "ecmwf_ghi"].to_numpy()
                                  + m.predict(d.loc[hold, MOS_FEATURES]))
        print(f"   fold {k}/{N_FOLDS} done", flush=True)

    m_full = mos_model().fit(tr[MOS_FEATURES], tr["error_target"])
    rest = d.index[d.split.isin(["val", "test"])]
    d.loc[rest, "ghi_mos"] = (d.loc[rest, "ecmwf_ghi"].to_numpy()
                              + m_full.predict(d.loc[rest, MOS_FEATURES]))

    d["ghi_mos"] = d["ghi_mos"].clip(lower=0)
    d["mos_ratio"] = np.where(d.ecmwf_ghi > 5.0,
                              d.ghi_mos / d.ecmwf_ghi.clip(lower=5.0), 1.0)
    d["mos_ratio"] = d["mos_ratio"].clip(0.2, 3.0)
    return d[["system_id", "timestamp", "mos_ratio", "ghi_mos", "ecmwf_ghi", "sat_ghi"]]


def apply_to_grid(ratio_hourly):
    """Interpolate the hourly correction ratio onto the 15-minute grid and
    rescale ghi_fc / poa_global_fc."""
    pv = pd.read_csv(C.MODEL_READY_V3)
    pv["timestamp"] = pd.to_datetime(pv["timestamp"], utc=True)
    pv = pv[~pv.system_id.isin(C.EXCLUDE_SYSTEMS)]
    pv = pv.sort_values(["system_id", "timestamp"]).reset_index(drop=True)

    blocks = []
    for sid, g in pv.groupby("system_id", sort=True):
        r = ratio_hourly[ratio_hourly.system_id == sid]
        g = g.set_index("timestamp").sort_index()
        if r.empty:
            g["mos_ratio"] = 1.0
        else:
            s = r.set_index("timestamp")["mos_ratio"].sort_index()
            s = s[~s.index.duplicated()]
            g["mos_ratio"] = (s.reindex(s.index.union(g.index))
                                .interpolate("time", limit_direction="both")
                                .reindex(g.index).to_numpy())
        blocks.append(g.reset_index())
    pv = pd.concat(blocks, ignore_index=True)
    pv["mos_ratio"] = pv["mos_ratio"].fillna(1.0).clip(0.2, 3.0)

    pv["ghi_fc_mos"] = pv["ghi_fc"] * pv["mos_ratio"]
    pv["poa_global_fc_mos"] = pv["poa_global_fc"] * pv["mos_ratio"]
    return pv


class MultiQuantileLGBM:
    def __init__(self, **params):
        self.params = params
        self.models = {}

    def fit(self, X, y):
        for q in C.QUANTILES:
            self.models[q] = lgb.LGBMRegressor(objective="quantile", alpha=q,
                                               **self.params).fit(X, y)
        return self

    def predict(self, X):
        return {q: np.clip(m.predict(X), 0, C.CLIP) for q, m in self.models.items()}


def main():
    print("Building MOS table ...", flush=True)
    d = build_mos_table()
    print(f"  rows={len(d):,} systems={d.system_id.nunique()}", flush=True)

    print("Producing out-of-fold corrections ...", flush=True)
    ratio = out_of_fold_corrections(d)

    te = ratio.merge(d[["system_id", "timestamp", "split"]],
                     on=["system_id", "timestamp"], how="left")
    t = te[te.split == "test"]
    print("\nirradiance level, TEST split (out-of-fold):")
    for lbl, col in [("RAW ECMWF", "ecmwf_ghi"), ("MOS corrected", "ghi_mos")]:
        e = t[col].to_numpy() - t.sat_ghi.to_numpy()
        print(f"  {lbl:14s} MAE={np.abs(e).mean():7.2f}  "
              f"RMSE={np.sqrt((e**2).mean()):7.2f}  MBE={e.mean():+7.2f}", flush=True)

    print("\nApplying correction to the 15-minute grid ...", flush=True)
    pv = apply_to_grid(ratio)
    pv = pv[~pv.system_id.isin(C.BROKEN_SYSTEMS)]

    prev = pv[["system_id", "timestamp", C.TARGET]].copy()
    prev["timestamp"] += pd.Timedelta(days=1)
    prev = prev.rename(columns={C.TARGET: "pred_persistence"})
    pv = pv.merge(prev, on=["system_id", "timestamp"], how="left")
    pv = pv[(pv.is_valid == True) & (pv.is_daylight == True)].copy()
    pv["pred_persistence"] = pv["pred_persistence"].fillna(0.0)
    pv = pv.fillna(0)

    base = [c for c in C.BASE_FEATURES if c in pv.columns]
    ens = [c for c in C.ENSEMBLE_FEATURES if c in pv.columns]
    tr, te2 = pv[pv.split == "train"], pv[pv.split == "test"]
    print(f"PV train={len(tr):,} test={len(te2):,}", flush=True)

    swapped = ["ghi_fc_mos" if c == "ghi_fc" else
               "poa_global_fc_mos" if c == "poa_global_fc" else c
               for c in base + ens]

    arms = {
        "A baseline (best v3 config)": base + ens,
        "B MOS-corrected irradiance": swapped,
        "C baseline + MOS as extra features": base + ens + ["poa_global_fc_mos", "mos_ratio"],
    }

    rows = []
    for label, cols in arms.items():
        print(f"\n>>> {label}  ({len(cols)} features)", flush=True)
        m = MultiQuantileLGBM(**C.LGBM_PARAMS).fit(tr[cols], tr[C.TARGET])
        P = m.predict(te2[cols])
        y = te2[C.TARGET].to_numpy()
        r = {
            "arm": label, "n_features": len(cols),
            "MAE": M.mae(y, P[0.50]), "RMSE": M.rmse(y, P[0.50]),
            "MBE": M.mbe(y, P[0.50]), "R2": M.r2(y, P[0.50]),
            "skill": M.skill_score(y, P[0.50], te2.pred_persistence.to_numpy()),
            "mean_pinball": M.mean_pinball(y, P),
            "P25_P90_coverage": M.coverage(y, P[0.25], P[0.90]),
        }
        rows.append(r)
        print(f"    R2={r['R2']:.4f}  skill={r['skill']:.4f}  "
              f"pinball={r['mean_pinball']:.5f}", flush=True)

    res = pd.DataFrame(rows).round(5)
    res.to_csv(C.MODELS_DIR / "mos_pv_integration_metrics.csv", index=False)
    print("\n" + "=" * 74)
    print("DOES BETTER IRRADIANCE MAKE A BETTER PV FORECAST?")
    print("=" * 74)
    print(res.to_string(index=False))
    print("\nParent project, using ERA5-trained MOS: 0.8035 with vs 0.8029 without.")

    with open(C.REPORTS_DIR / "mos_pv_integration_report.txt", "w", encoding="utf-8") as f:
        f.write("MOS -> PV integration, satellite-trained MOS\n" + "=" * 74 + "\n\n")
        f.write(res.to_string(index=False) + "\n")
    print(f"\nWrote {C.MODELS_DIR / 'mos_pv_integration_metrics.csv'}")


if __name__ == "__main__":
    main()
