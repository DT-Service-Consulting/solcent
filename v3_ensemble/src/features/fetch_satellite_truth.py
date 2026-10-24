#!/usr/bin/env python3
"""
fetch_satellite_truth.py
========================
Fetches INDEPENDENT satellite-observed irradiance as ground truth, replacing
the ERA5 reference in mos_training_data.csv.

WHY THIS EXISTS
---------------
`actual_ghi` in the parent project's mos_training_data.csv is ERA5. ERA5 is a
reanalysis produced by ECMWF using the same IFS model whose forecast is being
scored against it. Shared physics and shared assimilation make the forecast
agree with ERA5 far better than it agrees with reality.

Measured consequence, over this study's window and systems, on identical rows
(src/eval/reference_comparison.py; bias = forecast minus reference):

    reference       ECMWF RMSE (W/m2)   rRMSE    bias (W/m2)
    ERA5                  51.1          16.5 %      -1.5
    SARAH-3               86.6          29.0 %      +9.2

Ground-instrumented verification of IFS day-ahead GHI (Tuononen et al. 2019,
ACP 19:1985, four years at Helsinki) finds a positive bias of about +8 W/m2,
which the model attributes to cloud that is too transparent rather than cloud
that is missing. Against ERA5 that bias almost vanishes.

THE REPLACEMENT
---------------
EUMETSAT CM SAF SARAH-3: surface solar radiation retrieved from Meteosat
geostationary imagery. It is an OBSERVATION, not a model run, and shares no
physics with IFS. Served by Open-Meteo's satellite archive, no credentials
needed.

    endpoint : https://satellite-api.open-meteo.com/v1/archive
    model    : eumetsat_sarah3
    coverage : 99.2 % of 2024-04-01..2026-05-31 at 50.85N 4.35E

Note SARAH-3 is itself a retrieval with its own uncertainty (roughly 10-20
W/m2 for hourly GHI against ground pyranometers), so it is not perfect truth
either - but its errors are INDEPENDENT of the forecast being scored, which is
the property ERA5 lacks.

Output: v3_ensemble/data/interim/satellite_ghi_all.csv
"""

import sys
import time
import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as C

SAT_URL = "https://satellite-api.open-meteo.com/v1/archive"
SAT_MODEL = "eumetsat_sarah3"
USER_AGENT = "solar-forecasting-v3-satellite-truth"


def get_json(params, retries=8):
    url = SAT_URL + "?" + urllib.parse.urlencode(params, doseq=True)
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = min(300, 15 * (2 ** (attempt - 1)))
                print(f"      [429] sleeping {wait}s", flush=True)
                time.sleep(wait)
            else:
                try:
                    print("      [HTTP]", json.loads(e.read()).get("reason", "")[:150])
                except Exception:
                    print("      [HTTP]", e.code)
                if 400 <= e.code < 500 and e.code != 429:
                    return None
                time.sleep(5 * attempt)
        except Exception as e:
            print("      [net]", e, flush=True)
            time.sleep(5 * attempt)
    return None


def main():
    force = "--force" in sys.argv
    out_file = C.DATA_INTERIM / "satellite_ghi_all.csv"

    meta = pd.read_csv(C.PARENT_META).drop_duplicates("system_id")
    short = pd.read_csv(C.PARENT_SHORTLIST)
    keep = short[short.get("passed_gate", True) == True]["system_id"].unique()
    meta = meta[meta.system_id.isin(keep)]
    meta = meta[~meta.system_id.isin(C.EXCLUDE_SYSTEMS)]
    meta = meta.sort_values("system_id")

    print(f"SARAH-3 satellite ground truth for {len(meta)} systems")
    print(f"  window : {C.START_DATE} .. {C.END_DATE}")
    print(f"  model  : {SAT_MODEL}\n", flush=True)

    frames = []
    for i, row in enumerate(meta.itertuples(), 1):
        sid = int(row.system_id)
        cache = C.DATA_INTERIM / f"satellite_ghi_{sid}.csv"
        if cache.exists() and not force:
            print(f"[{i}/{len(meta)}] {sid} cached", flush=True)
            frames.append(pd.read_csv(cache))
            continue

        j = get_json({
            "latitude": float(row.latitude), "longitude": float(row.longitude),
            "start_date": C.START_DATE, "end_date": C.END_DATE,
            "hourly": "shortwave_radiation", "models": SAT_MODEL,
            "timezone": "UTC",
        })
        if not j or "hourly" not in j:
            print(f"[{i}/{len(meta)}] {sid} FAILED", flush=True)
            continue

        d = pd.DataFrame({"time": j["hourly"]["time"],
                          "sat_ghi": j["hourly"]["shortwave_radiation"]})
        d.insert(0, "system_id", sid)
        d.to_csv(cache, index=False)
        cov = 100 * d.sat_ghi.notna().mean()
        print(f"[{i}/{len(meta)}] {sid} rows={len(d)} coverage={cov:.1f}%", flush=True)
        frames.append(d)
        time.sleep(1.2)

    if not frames:
        print("nothing fetched")
        return
    allo = pd.concat(frames, ignore_index=True)
    allo.to_csv(out_file, index=False)
    print(f"\nWrote {out_file}  rows={len(allo):,}  systems={allo.system_id.nunique()}")


if __name__ == "__main__":
    main()
