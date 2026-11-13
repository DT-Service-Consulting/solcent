#!/usr/bin/env python3
"""
fetch_sarah_coarse.py
=====================
SARAH-3 averaged to the ECMWF 0.25-degree cell of each system.

SARAH-3 resolves 0.05 degrees, so a point value contains small-scale
variability that a 0.25-degree forecast cannot represent. Averaging SARAH-3
over the forecast cell removes that resolution mismatch from the reference
comparison (supervisor review). For each system, the 0.25-degree cell centred
on the nearest ECMWF grid point is sampled at 5 x 5 points 0.05 degrees apart
(the native SARAH-3 spacing) and the 25 hourly series are averaged.

Output: v3_ensemble/data/interim/satellite_ghi_025_all.csv
        (system_id, time, sat_ghi_025, n_points)
"""

import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as C

URL = "https://satellite-api.open-meteo.com/v1/archive"
OFFSETS = np.array([-0.10, -0.05, 0.0, 0.05, 0.10])


def get_json(params, retries=8):
    url = URL + "?" + urllib.parse.urlencode(params, doseq=True)
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "solar-forecasting-v3-sarah025"})
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            wait = min(600, 30 * attempt) if e.code == 429 else 5 * attempt
            print(f"   [HTTP {e.code}] retry in {wait}s", flush=True)
            time.sleep(wait)
        except Exception as e:
            print("   [net]", e, flush=True)
            time.sleep(5 * attempt)
    return None


def main():
    out_file = C.DATA_INTERIM / "satellite_ghi_025_all.csv"
    systems = pd.read_csv(C.PARENT_MODEL_READY_ORIG, usecols=["system_id"]).system_id.unique()
    meta = pd.read_csv(C.PARENT_META).drop_duplicates("system_id")
    meta = meta[meta.system_id.isin(systems)].sort_values("system_id")
    frames = []
    for i, row in enumerate(meta.itertuples(), 1):
        sid = int(row.system_id)
        cache = C.DATA_INTERIM / f"satellite_ghi_025_{sid}.csv"
        if cache.exists():
            frames.append(pd.read_csv(cache))
            print(f"[{i}/{len(meta)}] {sid} cached", flush=True)
            continue
        clat = np.round(float(row.latitude) * 4) / 4
        clon = np.round(float(row.longitude) * 4) / 4
        lats = [round(clat + a, 3) for a in OFFSETS for _ in OFFSETS]
        lons = [round(clon + b, 3) for _ in OFFSETS for b in OFFSETS]
        j = get_json({"latitude": ",".join(map(str, lats)), "longitude": ",".join(map(str, lons)),
                      "start_date": C.START_DATE, "end_date": C.END_DATE,
                      "hourly": "shortwave_radiation", "models": "eumetsat_sarah3", "timezone": "UTC"})
        if not j:
            print(f"[{i}/{len(meta)}] {sid} FAILED", flush=True)
            continue
        js = j if isinstance(j, list) else [j]
        mat = np.array([x["hourly"]["shortwave_radiation"] for x in js], dtype="float64")
        d = pd.DataFrame({"time": js[0]["hourly"]["time"],
                          "sat_ghi_025": np.nanmean(mat, axis=0),
                          "n_points": np.isfinite(mat).sum(axis=0)})
        d.insert(0, "system_id", sid)
        d.to_csv(cache, index=False)
        print(f"[{i}/{len(meta)}] {sid} cell ({clat},{clon}) points={len(js)}", flush=True)
        frames.append(d)
        time.sleep(20)   # 25 locations per call weigh 25 requests
    allo = pd.concat(frames, ignore_index=True)
    allo.to_csv(out_file, index=False)
    print(f"Wrote {out_file} rows={len(allo):,} systems={allo.system_id.nunique()}")


if __name__ == "__main__":
    main()
