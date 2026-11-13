#!/usr/bin/env python3
"""
fetch_era5.py
=============
ERA5 hourly GHI (preceding-hour mean) at each study system, for the reference
comparison and the ERA5 control arm of the MOS experiment.

Replaces the parent project's data/interim/mos_training_data.csv as the ERA5
source, which (a) covered 31 of the 33 systems, because it was fetched for an
older shortlist, and (b) kept only rows where ERA5 GHI exceeded 10 W/m2, a row
selection that conditions on the reference.

Output: v3_ensemble/data/interim/era5_ghi_all.csv  (system_id, time, era5_ghi)
"""

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as C
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_satellite_truth import get_json  # same retry logic, different URL below
import fetch_satellite_truth as S

ERA5_URL = "https://archive-api.open-meteo.com/v1/archive"


def main():
    force = "--force" in sys.argv
    out_file = C.DATA_INTERIM / "era5_ghi_all.csv"
    S.SAT_URL = ERA5_URL               # get_json reads the module-level URL
    S.USER_AGENT = "solar-forecasting-v3-era5"

    mr = pd.read_csv(C.PARENT_MODEL_READY_ORIG, usecols=["system_id"]).system_id.unique()
    meta = pd.read_csv(C.PARENT_META).drop_duplicates("system_id")
    meta = meta[meta.system_id.isin(mr)].sort_values("system_id")
    print(f"ERA5 GHI for {len(meta)} systems", flush=True)

    frames = []
    for i, row in enumerate(meta.itertuples(), 1):
        sid = int(row.system_id)
        cache = C.DATA_INTERIM / f"era5_ghi_{sid}.csv"
        if cache.exists() and not force:
            frames.append(pd.read_csv(cache))
            print(f"[{i}/{len(meta)}] {sid} cached", flush=True)
            continue
        j = get_json({
            "latitude": float(row.latitude), "longitude": float(row.longitude),
            "start_date": C.START_DATE, "end_date": C.END_DATE,
            "hourly": "shortwave_radiation", "models": "era5", "timezone": "UTC",
        })
        if not j or "hourly" not in j:
            print(f"[{i}/{len(meta)}] {sid} FAILED", flush=True)
            continue
        d = pd.DataFrame({"time": j["hourly"]["time"],
                          "era5_ghi": j["hourly"]["shortwave_radiation"]})
        d.insert(0, "system_id", sid)
        d.to_csv(cache, index=False)
        print(f"[{i}/{len(meta)}] {sid} rows={len(d)} coverage={100*d.era5_ghi.notna().mean():.1f}%",
              flush=True)
        frames.append(d)
        time.sleep(1.2)
    allo = pd.concat(frames, ignore_index=True)
    allo.to_csv(out_file, index=False)
    print(f"Wrote {out_file} rows={len(allo):,} systems={allo.system_id.nunique()}")


if __name__ == "__main__":
    main()
