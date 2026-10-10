#!/usr/bin/env python3
"""
build_features_v3.py
====================
Merges the lagged + multi-model GHI ensemble onto the parent project's
model_ready.csv and engineers the ensemble feature block.

It does NOT rebuild the parent feature table. Every parent column (target,
split, solar geometry, POA, lags, metadata) is carried through untouched, so
a baseline trained here is directly comparable to the parent's numbers.

INTERPOLATION
-------------
Members arrive hourly; the model grid is 15-minute. We reuse the parent's
physics-aware method verbatim: convert each member to a clear-sky index
Kt = GHI / (clearsky + eps), interpolate Kt linearly in time, then rebuild
15-minute GHI as Kt * (clearsky_15min + eps). Interpolating Kt instead of raw
GHI preserves the shape of the solar curve between hourly samples.
Clear-sky GHI is taken from the parent table, so the two builds stay
bit-consistent on geometry.

POA APPROXIMATION
-----------------
Only GHI is fetched per member (fetching DNI+DHI for 12 members would triple
the API load). The plane-of-array estimate for the ensemble mean is therefore
obtained by rescaling the parent's fully transposed POA:

    poa_ens_mean = poa_global_fc * (ens_ghi_mean / ghi_fc)

This holds the ECMWF day1 beam/diffuse split fixed and rescales magnitude. It
is an approximation and is labelled as such. If it earns its place in the
final feature set, promote it to a true haydavies transposition by also
fetching direct_radiation / diffuse_radiation per member.

LEAKAGE
-------
`ghi_day0_LEAKAGE` is dropped here and never reaches the feature set.

Input : parent data/processed/model_ready.csv
        v3_ensemble/data/interim/ensemble_ghi_all.csv
Output: v3_ensemble/data/processed/model_ready_v3.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as C

EPS = 1.0          # matches the parent build exactly


def member_columns(ens):
    lag = [f"ghi_lag{d}" for d in C.LAGGED_DAYS if f"ghi_lag{d}" in ens.columns]
    multi = [f"ghi_{m}" for m in C.MULTI_MODELS if f"ghi_{m}" in ens.columns]
    return lag, multi


def interp_member(series_hourly, cs_hourly, cs_grid, grid):
    """Hourly member GHI -> 15-min GHI via clear-sky-index interpolation.

    limit_direction="both" fills the grid edges: the parent grid runs to
    23:45 local on the final day while the hourly member series stops at
    23:00 UTC, so without it the last slot of every system is NaN.
    Members are clipped at 0 - Kt interpolation can dip slightly negative
    around sunrise/sunset where the eps guard dominates.
    """
    kt = series_hourly / (cs_hourly + EPS)
    kt = (kt.reindex(kt.index.union(grid))
            .interpolate("time", limit_direction="both")
            .reindex(grid))
    out = kt.to_numpy() * (cs_grid + EPS)
    return np.clip(out, 0.0, None).astype("float32")


def build():
    if not C.ENSEMBLE_RAW.exists():
        print(f"Missing {C.ENSEMBLE_RAW}. Run fetch_ensemble.py first.")
        return

    print("Loading parent model_ready.csv ...", flush=True)
    df = pd.read_csv(C.PARENT_MODEL_READY)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["system_id", "timestamp"]).reset_index(drop=True)
    print(f"  parent rows={len(df):,} systems={df.system_id.nunique()}", flush=True)

    print("Loading ensemble members ...", flush=True)
    ens = pd.read_csv(C.ENSEMBLE_RAW)
    ens["time"] = pd.to_datetime(ens["time"], utc=True)
    lag_cols, multi_cols = member_columns(ens)
    all_members = lag_cols + multi_cols
    print(f"  members: {len(all_members)}  lagged={len(lag_cols)} multi={len(multi_cols)}", flush=True)
    print(f"  {all_members}", flush=True)

    out_blocks = []
    for sid, g in df.groupby("system_id", sort=True):
        e = ens[ens.system_id == sid]
        if e.empty:
            print(f"  [{sid}] no ensemble data, skipped", flush=True)
            continue
        e = e.drop_duplicates("time").set_index("time").sort_index()

        g = g.set_index("timestamp").sort_index()
        grid = g.index
        cs_grid = g["clearsky_ghi"].to_numpy(dtype="float64")

        # clear-sky at the member (hourly) stamps, taken from the parent grid
        cs_hourly = g["clearsky_ghi"].reindex(
            g.index.union(e.index)).interpolate("time").reindex(e.index)

        mat = np.full((len(grid), len(all_members)), np.nan, dtype="float32")
        for j, col in enumerate(all_members):
            if col not in e.columns or e[col].isna().all():
                continue
            mat[:, j] = interp_member(e[col], cs_hourly, cs_grid, grid)
            g[col] = mat[:, j]

        lag_idx = [all_members.index(c) for c in lag_cols if c in all_members]
        mul_idx = [all_members.index(c) for c in multi_cols if c in all_members]

        with np.errstate(invalid="ignore"):
            g["ens_ghi_mean"] = np.nanmean(mat, axis=1)
            g["ens_ghi_std"] = np.nanstd(mat, axis=1)
            g["ens_ghi_range"] = np.nanmax(mat, axis=1) - np.nanmin(mat, axis=1)
            g["ens_ghi_iqr"] = (np.nanpercentile(mat, 75, axis=1)
                                - np.nanpercentile(mat, 25, axis=1))
            g["ens_lag_std"] = np.nanstd(mat[:, lag_idx], axis=1) if lag_idx else np.nan
            g["ens_model_std"] = np.nanstd(mat[:, mul_idx], axis=1) if mul_idx else np.nan
            # how many members sit within +-15% of the mean
            m = g["ens_ghi_mean"].to_numpy()[:, None]
            close = np.abs(mat - m) <= (0.15 * np.abs(m) + 5.0)
            n_ok = np.isfinite(mat).sum(axis=1)
            g["ens_agree_frac"] = np.where(n_ok > 0, close.sum(axis=1) / np.maximum(n_ok, 1), 0.0)

        # THE ONE FEATURE THAT EARNS ITS PLACE: plain mean of the six day-1
        # forecasts (ECMWF lag1 + the five other centres). Multi-model
        # averaging at a single lead time works; averaging ECMWF across
        # days 2-7 does not, because the stale members drag the mean down.
        # See the measurement table in config.ENSEMBLE_FEATURES.
        fresh = [c for c in (["ghi_lag1"] + multi_cols) if c in all_members]
        fresh_idx = [all_members.index(c) for c in fresh]
        with np.errstate(invalid="ignore"):
            g["ens_mean_fresh"] = np.nanmean(mat[:, fresh_idx], axis=1)

        # normalised (clear-sky index) spread - comparable across season and
        # hour. The clearsky_ghi > 5 guard matches the parent's treatment of
        # clearsky_index_fc: without it, dividing by a near-zero clear-sky at
        # low sun blows the ratio up (1.1% of daylight rows exceeded kt_std 2.0).
        lit = cs_grid > 5
        g["ens_kt_mean"] = np.where(lit, g["ens_ghi_mean"] / (cs_grid + EPS), 0.0)
        g["ens_kt_mean"] = np.clip(g["ens_kt_mean"], 0, 1.3)
        g["ens_kt_std"] = np.clip(np.where(lit, g["ens_ghi_std"] / (cs_grid + EPS), 0.0), 0, 1.5)

        # POA rescale (see module docstring)
        ratio = np.where(g["ghi_fc"].to_numpy() > 1.0,
                         g["ens_ghi_mean"].to_numpy() / g["ghi_fc"].to_numpy().clip(1.0),
                         1.0)
        ratio = np.clip(ratio, 0.0, 5.0)
        g["poa_ens_mean"] = np.clip(g["poa_global_fc"].to_numpy() * ratio, 0.0, 1400.0)

        out_blocks.append(g.reset_index())
        print(f"  [{sid}] merged  rows={len(g):,}  "
              f"ens_ghi_mean_daylight={g.loc[g.is_daylight == True, 'ens_ghi_mean'].mean():.1f}",
              flush=True)

    out = pd.concat(out_blocks, ignore_index=True)
    out = out.drop(columns=[c for c in C.LEAKAGE_COLS if c in out.columns], errors="ignore")

    print(f"\nWriting {C.MODEL_READY_V3} ...", flush=True)
    out.to_csv(C.MODEL_READY_V3, index=False)
    print(f"Done. rows={len(out):,} cols={len(out.columns)} "
          f"systems={out.system_id.nunique()}")

    day = out[out.is_daylight == True]
    print("\nEnsemble feature sanity (daylight rows):")
    print(day[C.ENSEMBLE_FEATURES].describe().T[["mean", "std", "min", "max"]].to_string())


if __name__ == "__main__":
    build()
