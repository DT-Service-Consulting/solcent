#!/usr/bin/env python3
"""
fix_irradiance_timing.py
========================
Rebuilds the irradiance-derived columns of the parent model_ready.csv with the
correct Open-Meteo time reference (see irradiance_timing.py), without
re-downloading any weather.

RECOVERY OF THE HOURLY FORECASTS
--------------------------------
The parent build (src/features/build_features.py) stored 15-minute irradiance
as kt(t) * (cs(t) + 1), with kt interpolated between hourly stamps and
kt(stamp) = v(stamp) / (cs(stamp) + 1). At the hourly stamps themselves the
interpolation is exact and cs is identical (same pvlib location and model), so
the stored value at each on-the-hour row equals the downloaded hourly value
exactly. This script reads those rows back and rebuilds everything from them.

Columns rebuilt (ECMWF = *_fc, ICON = *_fc2):
    ghi_fc, dni_fc, dhi_fc, poa_global_fc,
    ghi_fc2, dni_fc2, dhi_fc2, poa_global_fc2,
    ghi_disagree_fc, clearsky_index_fc
Everything else (target, splits, geometry, lags, metadata, temperature, cloud,
wind) is carried over unchanged, so row sets and splits are identical.

Input : parent data/interim/features_system_<id>.csv, data/processed/model_ready.csv
Output: v3_ensemble/data/processed/model_ready_parent_timefix.csv
        (config.PARENT_MODEL_READY points here; build_features_v3.py reads it)
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as C
sys.path.insert(0, str(Path(__file__).resolve().parent))
import irradiance_timing as T

FEAT_DIR = C.PROJECT_ROOT / "data" / "interim"
SRC_MODEL_READY = C.PROJECT_ROOT / "data" / "processed" / "model_ready.csv"
OUT = C.DATA_PROC / "model_ready_parent_timefix.csv"
REBUILT = ["ghi_fc", "dni_fc", "dhi_fc", "poa_global_fc",
           "ghi_fc2", "dni_fc2", "dhi_fc2", "poa_global_fc2",
           "ghi_disagree_fc", "clearsky_index_fc"]


def rebuild_system(sid, meta):
    f = pd.read_csv(FEAT_DIR / f"features_system_{sid}.csv",
                    usecols=["timestamp", "ghi_fc", "dni_fc", "dhi_fc",
                             "ghi_fc2", "dni_fc2", "dhi_fc2", "tilt_deg", "azimuth_deg"])
    f["timestamp"] = pd.to_datetime(f["timestamp"], utc=True)
    f = f.set_index("timestamp").sort_index()
    tilt, azim = float(f.tilt_deg.iloc[0]), float(f.azimuth_deg.iloc[0])
    loc = pvlib.location.Location(float(meta.latitude), float(meta.longitude), tz="UTC")

    hourly = f[f.index.minute == 0]
    hour_ends = hourly.index
    cs_h = T.clearsky_hour_means(loc, hour_ends)

    grid = f.index
    centres = T.slot_centres(grid)
    cs_c = loc.get_clearsky(centres, model="ineichen")
    sp = loc.get_solarposition(centres)
    dni_extra = pvlib.irradiance.get_extra_radiation(centres)

    out = pd.DataFrame(index=grid)
    for sfx in ("", "2"):
        for comp in ("ghi", "dni", "dhi"):
            col = f"{comp}_fc{sfx}"
            v = hourly[col]
            ok = v.notna()
            out[col] = T.hourly_mean_to_instants(v[ok], cs_h[comp][ok], centres, cs_c[comp].to_numpy())
        poa = pvlib.irradiance.get_total_irradiance(
            surface_tilt=tilt, surface_azimuth=azim,
            solar_zenith=sp["apparent_zenith"].to_numpy(), solar_azimuth=sp["azimuth"].to_numpy(),
            dni=out[f"dni_fc{sfx}"].to_numpy(), ghi=out[f"ghi_fc{sfx}"].to_numpy(),
            dhi=out[f"dhi_fc{sfx}"].to_numpy(), dni_extra=dni_extra.to_numpy(), model="haydavies")
        out[f"poa_global_fc{sfx}"] = np.asarray(poa["poa_global"], dtype="float64")
    out["ghi_disagree_fc"] = out["ghi_fc"] - out["ghi_fc2"]
    csg = cs_c["ghi"].to_numpy()
    out["clearsky_index_fc"] = np.clip(np.where(csg > 5, out["ghi_fc"] / np.where(csg > 5, csg, 1.0), 0.0), 0, 1.3)
    out["system_id"] = sid
    return out.reset_index()


def main():
    t0 = time.time()
    meta = pd.read_csv(C.PARENT_META).drop_duplicates("system_id").set_index("system_id")
    print(f"Loading {SRC_MODEL_READY} ...", flush=True)
    mr = pd.read_csv(SRC_MODEL_READY)
    mr["timestamp"] = pd.to_datetime(mr["timestamp"], utc=True)
    systems = sorted(mr.system_id.unique())
    print(f"  rows={len(mr):,} systems={len(systems)}", flush=True)

    blocks = []
    for sid in systems:
        blocks.append(rebuild_system(sid, meta.loc[sid]))
        print(f"  [{sid}] rebuilt", flush=True)
    new = pd.concat(blocks, ignore_index=True)

    # sanity: the original values at on-the-hour rows must be the downloaded ones,
    # i.e. the new 30-min-earlier series must differ from the old in general.
    old_cols = mr[["system_id", "timestamp"] + REBUILT]
    m = old_cols.merge(new, on=["system_id", "timestamp"], suffixes=("_old", ""))
    for c in ("ghi_fc", "poa_global_fc"):
        d = (m[c] - m[c + "_old"])
        print(f"  {c}: mean change {d.mean():+.2f}  mean |change| {d.abs().mean():.2f} W/m2", flush=True)

    mr = mr.drop(columns=REBUILT).merge(new, on=["system_id", "timestamp"], how="left")
    miss = mr[REBUILT].isna().sum()
    if miss.any():
        print("  filling NaN per system (ffill/bfill):", miss[miss > 0].to_dict(), flush=True)
        mr = mr.sort_values(["system_id", "timestamp"])
        mr[REBUILT] = mr.groupby("system_id")[REBUILT].transform(lambda s: s.ffill().bfill())
    mr = mr.sort_values("timestamp")
    print(f"Writing {OUT} ...", flush=True)
    mr.to_csv(OUT, index=False)
    print(f"done rows={len(mr):,} in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
