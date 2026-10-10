#!/usr/bin/env python3
"""
oracle_ceiling.py
=================
A DIAGNOSTIC, NOT A MODEL. Never ship this feature.

SHOULD THE ORACLE BE "IMPLEMENTED"? YES - AS AN INSTRUMENT.
-----------------------------------------------------------
The oracle feature is leakage by construction, so it can never appear in a
deployed forecast. What it CAN do is measure the headroom above your current
model, which is the one question hyperparameter tuning can never answer. Keep
it permanently as a ruler: after every input change, re-run it and see whether
the gap to the ceiling closed. That is why it lives in diagnostics/ behind a
guard, not in features/.

WHAT IT DOES
------------
For every row (one home, one 15-minute slot) it computes the mean ACTUAL
measured output of all the OTHER homes at that same instant, leave-one-out so
a home's own answer never leaks into its own feature:

    fleet_actual[i] = (sum of target over all homes at t) - target[i]
                      -------------------------------------------------
                                  (count at t) - 1

WHY THAT IS A CEILING
---------------------
Those 33 homes spread across Belgium are a distributed irradiance sensor
network. Their combined real output is a near-perfect record of what the sky
actually did over the country. So `fleet_actual` stands in for PERFECT
REGIONAL WEATHER KNOWLEDGE - the score you would get if someone handed you
tomorrow's true Belgian cloud cover. At forecast time (noon on D-1) that
number does not exist yet.

RESULT ON THE PARENT DATA (2026-09, LightGBM, parent's 12 features)
-------------------------------------------------------------------
    12 features, NWP forecast only .................. R2 0.8027
    + perfect regional truth ........................ R2 0.8257   (+0.023)
    perfect regional truth + geometry, NO NWP ....... R2 0.7244

Perfect regional truth buys +0.023. The third line is the striking one:
knowing the real regional weather with no forecast at all already reaches
0.72, so almost everything the 12-feature NWP model contributes is
regional-scale signal that it has already extracted.

Conclusion: the missing variance is not regional. It is the individual cloud
shadow crossing one specific roof - 1-5 km wide, lasting minutes. ECMWF
ifs025 is a 25 km grid at hourly steps and physically cannot resolve it.

HONEST CAVEAT
-------------
A fleet AVERAGE is spatial smoothing by construction, so it was never going to
capture local cloud. This proves "better regional weather data will not save
you". It does not prove "no input could" - a pyranometer on each roof would
score far higher, but is equally unavailable a day ahead. State it this way in
the write-up; the stronger claim is not supported.
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

ORACLE_COL = "fleet_actual__LEAKAGE_DIAGNOSTIC_ONLY"
MIN_FLEET = 3          # need more than this many homes reporting at a slot


def guard(feature_list):
    """Hard stop if the oracle ever reaches a shippable feature set."""
    if ORACLE_COL in feature_list:
        raise RuntimeError(
            f"{ORACLE_COL} is a leakage diagnostic and must never be trained "
            "into a deployed model. It is only valid inside oracle_ceiling.py."
        )


def add_oracle(df, target=None):
    """Leave-one-out mean of every OTHER system's actual output at the same
    timestamp. Returns df with ORACLE_COL added and thin slots dropped."""
    target = target or C.TARGET
    g = df.groupby("timestamp")[target]
    s = g.transform("sum")
    n = g.transform("count")
    df = df.copy()
    df[ORACLE_COL] = (s - df[target]) / (n - 1)
    return df[n > MIN_FLEET]


def load(path=None, drop_broken=False):
    path = path or C.MODEL_READY_V3
    if not Path(path).exists():
        path = C.PARENT_MODEL_READY
        print(f"[info] {C.MODEL_READY_V3.name} not built yet, "
              f"falling back to parent {path.name}")
    df = pd.read_csv(path)
    df = df[~df.system_id.isin(C.EXCLUDE_SYSTEMS)]
    if drop_broken:
        df = df[~df.system_id.isin(C.BROKEN_SYSTEMS)]
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df[(df.is_valid == True) & (df.is_daylight == True)].copy()
    return df.fillna(0)


def fit_score(tr, te, cols, tag):
    m = lgb.LGBMRegressor(**C.LGBM_PARAMS).fit(tr[cols], tr[C.TARGET])
    p = np.clip(m.predict(te[cols]), 0, C.CLIP)
    r = M.r2(te[C.TARGET], p)
    print(f"  {tag:52s} n_feat={len(cols):2d}  R2={r:.4f}  "
          f"RMSE={M.rmse(te[C.TARGET], p):.4f}", flush=True)
    return r


def main():
    drop_broken = "--drop-broken" in sys.argv
    df = load(drop_broken=drop_broken)
    df = add_oracle(df)

    base = [c for c in C.BASE_FEATURES if c in df.columns]
    ens = [c for c in C.ENSEMBLE_FEATURES if c in df.columns]
    guard(base)
    guard(base + ens)

    tr, te = df[df.split == "train"], df[df.split == "test"]
    print(f"\nrows train={len(tr):,} test={len(te):,}  "
          f"systems={df.system_id.nunique()}  "
          f"broken_dropped={drop_broken}")
    print(f"base features={len(base)}  ensemble features={len(ens)}\n")

    print("CEILING TEST")
    r_base = fit_score(tr, te, base, "baseline: NWP forecast only")
    r_orac = fit_score(tr, te, base + [ORACLE_COL], "+ ORACLE perfect regional truth")
    fit_score(tr, te, [ORACLE_COL, "sun_elevation", "clearsky_ghi",
                       "capacity_kwp", "tilt_deg", "azimuth_deg"],
              "ORACLE + geometry only, NO NWP")

    r_ens = r_ens_orac = None
    if ens:
        print("\nWITH THE v3 ENSEMBLE INPUT")
        r_ens = fit_score(tr, te, base + ens, "baseline + ensemble features")
        r_ens_orac = fit_score(tr, te, base + ens + [ORACLE_COL],
                               "+ ensemble + ORACLE")

    print("\n" + "-" * 74)
    print(f"headroom to perfect regional truth (before ensemble): "
          f"{r_orac - r_base:+.4f} R2")
    if r_ens is not None:
        print(f"gained by the ensemble input                        : "
              f"{r_ens - r_base:+.4f} R2")
        print(f"headroom remaining after the ensemble               : "
              f"{r_ens_orac - r_ens:+.4f} R2")
        closed = (r_ens - r_base) / (r_orac - r_base) if r_orac > r_base else float("nan")
        print(f"fraction of the regional headroom closed            : {closed:.1%}")
    print("-" * 74)
    print("Reminder: the oracle is unavailable at forecast time. These numbers "
          "size the prize, they are not achievable scores.")


if __name__ == "__main__":
    main()
