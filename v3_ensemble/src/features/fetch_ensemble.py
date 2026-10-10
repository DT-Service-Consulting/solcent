#!/usr/bin/env python3
"""
fetch_ensemble.py
=================
Builds a LAGGED + MULTI-MODEL ensemble of day-ahead GHI forecasts for every
study system, backfilled across the full 2024-04-01..2026-05-31 window.

WHY THIS EXISTS
---------------
The parent project feeds the model ONE deterministic irradiance forecast
(ECMWF ifs025, _previous_day1). Every model family tried against that single
input lands at R2 0.79-0.81. An oracle test (src/diagnostics/oracle_ceiling.py)
showed that perfect *regional* weather truth is worth only +0.023 R2, so the
ceiling is the information content of the input, not the learner.

This script widens the input from 1 forecast to up to 12:

  A) LAGGED ENSEMBLE - the same ECMWF model re-run on successive days.
     _previous_day1 .. _previous_day7. The forecast for tomorrow issued
     1 day ago differs from the one issued 7 days ago; chaos makes them
     diverge. Their spread is a genuine uncertainty measurement.

  B) MULTI-MODEL ENSEMBLE - five other centres at true day-ahead lead time.
     Different physics, different errors. Their disagreement is a second,
     independent uncertainty signal.

NO LEAKAGE
----------
Every member fetched here was published BEFORE a D-1 noon issue time, so all
are legally available to a day-ahead forecast. The plain "shortwave_radiation"
field (the day0 analysis run) is NOT available at issue time. It is stored as
`ghi_day0_LEAKAGE` for reference/analysis only and is excluded from the feature
set by construction in build_features_v3.py.

Output: v3_ensemble/data/interim/ensemble_ghi_<sid>.csv  (per system)
        v3_ensemble/data/interim/ensemble_ghi_all.csv    (combined)
"""

import sys
import time
import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as C

USER_AGENT = "solar-forecasting-v3-ensemble"


def get_json(url, params, retries=8):
    """GET with exponential backoff. Honours HTTP 429 rate limits."""
    qs = urllib.parse.urlencode(params, doseq=True)
    full = f"{url}?{qs}"
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(full, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = min(300, 15 * (2 ** (attempt - 1)))
                print(f"      [429] rate limited, sleeping {wait}s "
                      f"(attempt {attempt}/{retries})", flush=True)
                time.sleep(wait)
            else:
                body = ""
                try:
                    body = json.loads(e.read()).get("reason", "")[:160]
                except Exception:
                    pass
                print(f"      [HTTP {e.code}] {body}", flush=True)
                if 400 <= e.code < 500 and e.code != 429:
                    return None          # bad request, retrying will not help
                time.sleep(5 * attempt)
        except Exception as e:
            print(f"      [net] {e}", flush=True)
            time.sleep(5 * attempt)
    return None


def fetch_lagged(lat, lon):
    """ECMWF re-runs: day0 (leakage marker) + day1..day7."""
    hourly = ["shortwave_radiation"] + [
        f"shortwave_radiation_previous_day{d}" for d in C.LAGGED_DAYS
    ]
    j = get_json(C.PREV_URL, {
        "latitude": lat, "longitude": lon,
        "start_date": C.START_DATE, "end_date": C.END_DATE,
        "timezone": "UTC", "models": C.LAGGED_MODEL,
        "hourly": ",".join(hourly),
    })
    if not j or "hourly" not in j:
        return None
    h = j["hourly"]
    out = {"time": h["time"], "ghi_day0_LEAKAGE": h.get("shortwave_radiation")}
    for d in C.LAGGED_DAYS:
        out[f"ghi_lag{d}"] = h.get(f"shortwave_radiation_previous_day{d}")
    return pd.DataFrame(out)


def fetch_multimodel(lat, lon):
    """Five other centres, all at _previous_day1 (true day-ahead lead time)."""
    j = get_json(C.PREV_URL, {
        "latitude": lat, "longitude": lon,
        "start_date": C.START_DATE, "end_date": C.END_DATE,
        "timezone": "UTC", "models": ",".join(C.MULTI_MODELS),
        "hourly": "shortwave_radiation_previous_day1",
    })
    if not j or "hourly" not in j:
        return None
    h = j["hourly"]
    out = {"time": h["time"]}
    for m in C.MULTI_MODELS:
        key = f"shortwave_radiation_previous_day1_{m}"
        if key in h:
            out[f"ghi_{m}"] = h[key]
    return pd.DataFrame(out)


def fetch_system(sid, lat, lon):
    lag = fetch_lagged(lat, lon)
    if lag is None:
        print(f"   !! lagged fetch failed for {sid}", flush=True)
        return None
    time.sleep(1.0)
    multi = fetch_multimodel(lat, lon)
    if multi is None:
        print(f"   !! multi-model fetch failed for {sid}, keeping lagged only", flush=True)
        df = lag
    else:
        df = lag.merge(multi, on="time", how="left")
    df.insert(0, "system_id", sid)
    return df


def main():
    force = "--force" in sys.argv

    meta = pd.read_csv(C.PARENT_META)
    short = pd.read_csv(C.PARENT_SHORTLIST)
    keep = short[short.get("passed_gate", True) == True]["system_id"].unique()
    meta = meta[meta.system_id.isin(keep)]
    meta = meta[~meta.system_id.isin(C.EXCLUDE_SYSTEMS)]
    meta = meta.drop_duplicates("system_id").sort_values("system_id")

    print(f"Fetching ensemble GHI for {len(meta)} systems")
    print(f"  window   : {C.START_DATE} .. {C.END_DATE}")
    print(f"  lagged   : {C.LAGGED_MODEL} day{C.LAGGED_DAYS}")
    print(f"  models   : {', '.join(C.MULTI_MODELS)}")
    print(f"  members  : up to {len(C.LAGGED_DAYS) + len(C.MULTI_MODELS)}\n", flush=True)

    frames = []
    for i, row in enumerate(meta.itertuples(), 1):
        sid = int(row.system_id)
        out = C.DATA_INTERIM / f"ensemble_ghi_{sid}.csv"
        if out.exists() and not force:
            print(f"[{i}/{len(meta)}] {sid} cached", flush=True)
            frames.append(pd.read_csv(out))
            continue
        print(f"[{i}/{len(meta)}] {sid} lat={row.latitude} lon={row.longitude}", flush=True)
        df = fetch_system(sid, float(row.latitude), float(row.longitude))
        if df is None:
            continue
        df.to_csv(out, index=False)
        cols = [c for c in df.columns if c.startswith("ghi_")]
        cov = {c: f"{100 * df[c].notna().mean():.0f}%" for c in cols}
        print(f"      rows={len(df)} coverage={cov}", flush=True)
        frames.append(df)
        time.sleep(1.5)

    if not frames:
        print("No data fetched.")
        return
    allo = pd.concat(frames, ignore_index=True)
    allo.to_csv(C.ENSEMBLE_RAW, index=False)
    print(f"\nWrote {C.ENSEMBLE_RAW}  rows={len(allo):,}  systems={allo.system_id.nunique()}")


if __name__ == "__main__":
    main()
