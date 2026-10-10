#!/usr/bin/env python3
"""
log_live_ensemble.py
====================
Daily logger for the TRUE ECMWF ensemble (51 members).

WHY A LOGGER AND NOT A BACKFILL
-------------------------------
ECMWF does not only run its model once. It runs it 51 times at the same
moment, each copy started from a microscopically different guess at today's
atmosphere. Chaos amplifies those differences, so you get 51 plausible
tomorrows. The spread between them is the gold standard for forecast
uncertainty - far better than the lagged ensemble, because every member has
the same lead time and differs only in initial conditions.

Open-Meteo serves it, confirmed live:
    ecmwf_ifs025          51 members
    icon_seamless / eu    40 members
    gfs_seamless          31 members
    bom_access_global     18 members

THE BLOCKER: the ensemble API keeps only ~93 days of history
(`past_days` is capped at 93). The study window starts 2024-04-01, so it
CANNOT be reconstructed - this is why the main v3 pipeline uses the lagged +
multi-model ensemble instead, which the previous-runs API does serve back to
2024-04-01 at 100% coverage.

The fix is to start accumulating now. Run this once a day (Task Scheduler or
cron). After ~3 months you have a clean forward-validated slice: a
genuinely out-of-sample test of whether a real 51-member ensemble beats the
12-member lagged/multi-model proxy. That is a strong thesis chapter, and it
only works if the logging starts today.

    schtasks /create /tn "solar-ens-log" /tr ^
      "F:\\solar_forecasting\\.venv\\Scripts\\python.exe ^
       F:\\solar_forecasting\\v3_ensemble\\src\\features\\log_live_ensemble.py" ^
      /sc daily /st 13:00

Output: v3_ensemble/data/interim/live_ensemble/<YYYY-MM-DD>.csv  (one per run)
"""

import sys
import json
import time
import datetime as dt
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as C

OUT_DIR = C.DATA_INTERIM / "live_ensemble"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL = "ecmwf_ifs025"
VAR = "shortwave_radiation"
FORECAST_DAYS = 3
USER_AGENT = "solar-forecasting-v3-ensemble-logger"


def get_json(params, retries=6):
    url = C.ENSEMBLE_URL + "?" + urllib.parse.urlencode(params, doseq=True)
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = min(300, 15 * 2 ** (attempt - 1))
                print(f"   [429] sleeping {wait}s", flush=True)
                time.sleep(wait)
            else:
                try:
                    print("   [HTTP]", json.loads(e.read()).get("reason", "")[:160])
                except Exception:
                    print("   [HTTP]", e.code)
                return None
        except Exception as e:
            print("   [net]", e)
            time.sleep(5 * attempt)
    return None


def main():
    stamp = dt.date.today().isoformat()
    out = OUT_DIR / f"{stamp}.csv"
    if out.exists():
        print(f"{out.name} already logged today.")
        return

    meta = pd.read_csv(C.PARENT_META)
    short = pd.read_csv(C.PARENT_SHORTLIST)
    keep = short[short.get("passed_gate", True) == True]["system_id"].unique()
    meta = meta[meta.system_id.isin(keep)]
    meta = meta[~meta.system_id.isin(C.EXCLUDE_SYSTEMS)]
    meta = meta.drop_duplicates("system_id").sort_values("system_id")

    frames = []
    for i, row in enumerate(meta.itertuples(), 1):
        sid = int(row.system_id)
        j = get_json({
            "latitude": float(row.latitude), "longitude": float(row.longitude),
            "hourly": VAR, "models": MODEL, "timezone": "UTC",
            "forecast_days": FORECAST_DAYS,
        })
        if not j or "hourly" not in j:
            print(f"[{i}/{len(meta)}] {sid} FAILED", flush=True)
            continue
        h = j["hourly"]
        members = [k for k in h if k.startswith(VAR)]
        d = pd.DataFrame({k: h[k] for k in members})
        d.insert(0, "time", h["time"])
        d.insert(0, "issued", stamp)
        d.insert(0, "system_id", sid)
        frames.append(d)
        print(f"[{i}/{len(meta)}] {sid} members={len(members)} rows={len(d)}", flush=True)
        time.sleep(1.0)

    if not frames:
        print("nothing logged")
        return
    allo = pd.concat(frames, ignore_index=True)
    allo.to_csv(out, index=False)
    print(f"\nWrote {out}  rows={len(allo):,}  systems={allo.system_id.nunique()}")
    n = len(list(OUT_DIR.glob("*.csv")))
    print(f"{n} day(s) logged so far. A useful forward-validation slice needs ~90.")


if __name__ == "__main__":
    main()
