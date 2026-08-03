# Relative Path: src/data/pvoutput_download.py
#!/usr/bin/env python3
"""
pvoutput_download.py  (v2 - captures EVERYTHING)
================================================
Resilient, RESUMABLE downloader for raw 5-minute PV history.
"""

import os
import sys
import csv
import json
import time
from datetime import date, timedelta
from pathlib import Path

import requests
import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

API_KEY   = os.environ.get("PVOUTPUT_API_KEY")
SYSTEM_ID = os.environ.get("PVOUTPUT_SYSTEM_ID")

START_DATE = date(2024, 4, 1)
END_DATE   = date(2026, 5, 31)

SHORTLIST_CSV = PROJECT_ROOT / "data" / "interim" / "pvoutput_selection" / "shortlist_final.csv"
TARGET_SYSTEM_IDS = []     

FETCH_DAILY_SUMMARY = True 
REQUESTS_PER_HOUR = 300
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "pvoutput_raw"

MAX_RETRIES   = 10
BACKOFF_BASE  = 5
BACKOFF_CAP   = 180

H_DATE, H_TIME, H_EGEN, H_EFF, H_PGEN, H_PAVG, H_NORM, H_ECONS, H_PCONS, H_TEMP, H_VOLT = range(11)
H_V_START = 11   

STATUS_COLS = ["system_id", "timestamp", "energy_gen_wh", "efficiency",
               "power_gen_w", "power_avg_w", "normalised_output",
               "energy_cons_wh", "power_cons_w", "temperature_c", "voltage_v",
               "v7", "v8", "v9", "v10", "v11", "v12"]

GO_DATE, GO_GEN, GO_EFF, GO_PEAKW, GO_PEAKT, GO_COND, GO_TMIN, GO_TMAX, GO_INSOL = \
    0, 1, 2, 5, 6, 7, 8, 9, 14

DAILY_COLS = ["system_id", "date", "generated_wh", "efficiency", "peak_power_w",
              "peak_time", "condition", "min_temp_c", "max_temp_c", "insolation_wh"]

BASE = "https://pvoutput.org/service/r2"
# ----------------------------------------------------------------------------

class PVOutputClient:
    def __init__(self, api_key, system_id, requests_per_hour=300):
        if not api_key or not system_id:
            sys.exit("ERROR: set PVOUTPUT_API_KEY and PVOUTPUT_SYSTEM_ID.")
        self.headers = {"X-Pvoutput-Apikey": str(api_key),
                        "X-Pvoutput-SystemId": str(system_id)}
        self.min_interval = 3600.0 / max(1, requests_per_hour)
        self._last_call = 0.0
        self._printed_limit = False
        self.donation_required = False

    def _throttle(self):
        wait = self.min_interval - (time.time() - self._last_call)
        if wait > 0:
            time.sleep(wait)

    def _request(self, endpoint, params):
        """Return (status, payload); retries only transient network/quota failures."""
        for attempt in range(1, MAX_RETRIES + 1):
            self._throttle()
            try:
                r = requests.get(f"{BASE}/{endpoint}", headers=self.headers,
                                 params=params, timeout=30)
            except requests.RequestException as e:
                self._last_call = time.time()
                wait = min(BACKOFF_CAP, BACKOFF_BASE * (2 ** (attempt - 1)))
                print(f"    network error ({e.__class__.__name__}) -> "
                      f"retry {attempt}/{MAX_RETRIES} in {wait}s")
                time.sleep(wait)
                continue
            self._last_call = time.time()

            if not self._printed_limit:
                lim = r.headers.get("X-Rate-Limit-Limit")
                if lim:
                    self._printed_limit = True
                    tier = "DONATION" if int(lim) >= 300 else "FREE"
                    print(f"  [quota] limit={lim}/hr ({tier} tier)")
            try:
                rem = int(r.headers.get("X-Rate-Limit-Remaining", "999"))
                reset = int(r.headers.get("X-Rate-Limit-Reset", "0"))
                if rem <= 1 and reset:
                    nap = max(0, reset - time.time()) + 5
                    print(f"  [quota] exhausted -> sleeping {nap:.0f}s")
                    time.sleep(nap)
            except (TypeError, ValueError):
                pass

            body = (r.text or "").strip()
            body_l = body.lower()

            if r.status_code == 200:
                return ("ok", body)
            if r.status_code == 400 and any(
                marker in body_l for marker in (
                    "no status", "no data", "no system or data found",
                )
            ):
                return ("nodata", "")
            if r.status_code in (401, 403) and "inaccessible" in body_l:
                self.donation_required = True
                return ("authfail", "")
            if r.status_code in (429, 503) or "rate limit" in body_l:
                reset = r.headers.get("X-Rate-Limit-Reset")
                try:
                    wait = max(0, int(reset) - time.time()) + 5 if reset else None
                except (TypeError, ValueError):
                    wait = None
                wait = wait or min(BACKOFF_CAP, BACKOFF_BASE * (2 ** (attempt - 1)))
                print(f"    quota/server busy ({r.status_code}) -> "
                      f"retry {attempt}/{MAX_RETRIES} in {wait:.0f}s")
                time.sleep(wait)
                continue
            if r.status_code == 400:
                print(f"    HTTP 400 from {endpoint} params={params}: {body[:240]}")
                return ("badrequest", None)
            wait = min(BACKOFF_CAP, BACKOFF_BASE * (2 ** (attempt - 1)))
            print(f"    HTTP {r.status_code} -> retry {attempt}/{MAX_RETRIES} "
                  f"in {wait}s: {body[:160]}")
            time.sleep(wait)
        return ("neterror", None)

    def get_day(self, sid, day):
        """5-min status for one day WITH extended channels. Returns (status, rows)."""
        status, txt = self._request("getstatus.jsp", {
            "sid1": sid, "d": day.strftime("%Y%m%d"), "h": 1, "limit": 288,
            "asc": 1, "ext": 1,
        })
        if status != "ok":
            return (status, [] if status == "nodata" else None)
        rows = []
        for rec in txt.split(";"):
            parts = rec.split(",")
            if len(parts) < 5:           
                continue

            def num(i):
                try:
                    v = parts[i].strip()
                    return None if v == "" or v.lower() == "nan" else float(v)
                except (IndexError, ValueError):
                    return None

            ts = pd.to_datetime(f"{parts[H_DATE]} {parts[H_TIME]}",
                                format="%Y%m%d %H:%M", errors="coerce")
            if pd.isna(ts):
                continue
            try:
                ts = ts.tz_localize("Europe/Brussels",
                                    nonexistent="shift_forward", ambiguous="NaT")
            except Exception:
                continue
            if pd.isna(ts):
                continue
            row = {"system_id": sid, "timestamp": ts.isoformat(),
                   "energy_gen_wh": num(H_EGEN), "efficiency": num(H_EFF),
                   "power_gen_w": num(H_PGEN), "power_avg_w": num(H_PAVG),
                   "normalised_output": num(H_NORM), "energy_cons_wh": num(H_ECONS),
                   "power_cons_w": num(H_PCONS), "temperature_c": num(H_TEMP),
                   "voltage_v": num(H_VOLT)}
            for k in range(6):           
                row[f"v{7+k}"] = num(H_V_START + k)
            rows.append(row)
        return ("ok", rows)

    def get_output_days(self, sid, df_date, dt_date):
        """Daily summary for a date range (<=150 days). Returns (status, rows)."""
        status, txt = self._request("getoutput.jsp", {
            "sid1": sid, "df": df_date.strftime("%Y%m%d"),
            "dt": dt_date.strftime("%Y%m%d"), "limit": 150, "insolation": 1,
        })
        if status != "ok":
            return (status, [] if status == "nodata" else None)
        rows = []
        for rec in txt.split(";"):
            parts = rec.split(",")
            if len(parts) <= GO_PEAKW:
                continue

            def num(i):
                try:
                    v = parts[i].strip()
                    return None if v == "" or v.lower() == "nan" else float(v)
                except (IndexError, ValueError):
                    return None

            def text(i):
                try:
                    return parts[i].strip()
                except IndexError:
                    return None

            try:
                d = pd.to_datetime(parts[GO_DATE], format="%Y%m%d").date().isoformat()
            except ValueError:
                continue
            rows.append({"system_id": sid, "date": d, "generated_wh": num(GO_GEN),
                         "efficiency": num(GO_EFF), "peak_power_w": num(GO_PEAKW),
                         "peak_time": text(GO_PEAKT), "condition": text(GO_COND),
                         "min_temp_c": num(GO_TMIN), "max_temp_c": num(GO_TMAX),
                         "insolation_wh": num(GO_INSOL)})
        return ("ok", rows)

class SystemStore:
    """Per-system CSV + JSON checkpoint with crash-safe appends and resume."""
    def __init__(self, out_dir, sid):
        self.sid = sid
        self.csv_path = out_dir / f"system_{sid}.csv"
        self.daily_path = out_dir / f"system_{sid}_daily.csv"
        self.ckpt_path = out_dir / f"system_{sid}.ckpt.json"
        self.completed = set()
        self.valid_bytes = 0
        self.daily_done = False
        self._load()

    def _save_ckpt(self):
        tmp = self.ckpt_path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump({"system_id": self.sid,
                       "completed_days": sorted(self.completed),
                       "valid_bytes": self.valid_bytes,
                       "daily_done": self.daily_done}, f)
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, self.ckpt_path)

    def _load(self):
        if self.ckpt_path.exists():
            ck = json.loads(self.ckpt_path.read_text())
            self.completed = set(ck.get("completed_days", []))
            self.valid_bytes = int(ck.get("valid_bytes", 0))
            self.daily_done = bool(ck.get("daily_done", False))
            if self.csv_path.exists():
                actual = self.csv_path.stat().st_size
                if actual > self.valid_bytes:        
                    with open(self.csv_path, "r+b") as f:
                        f.truncate(self.valid_bytes)
        elif not self.csv_path.exists():
            with open(self.csv_path, "w", newline="") as f:
                csv.writer(f).writerow(STATUS_COLS)
                f.flush(); os.fsync(f.fileno())
            self.valid_bytes = self.csv_path.stat().st_size
            self._save_ckpt()

    def write_day(self, day_str, rows):
        if rows:
            block = pd.DataFrame(rows)[STATUS_COLS].to_csv(index=False, header=False)
            with open(self.csv_path, "a", newline="") as f:
                f.write(block)
                f.flush(); os.fsync(f.fileno())
            self.valid_bytes = self.csv_path.stat().st_size
        self.completed.add(day_str)
        self._save_ckpt()

    def write_daily(self, rows):
        """Write the whole daily-summary table atomically (all-or-nothing)."""
        df = pd.DataFrame(rows, columns=DAILY_COLS) if rows else pd.DataFrame(columns=DAILY_COLS)
        tmp = self.daily_path.with_suffix(".csv.tmp")
        df.to_csv(tmp, index=False)
        with open(tmp, "rb+") as f:
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, self.daily_path)
        self.daily_done = True
        self._save_ckpt()

def daterange(start, end):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)

def load_target_ids():
    env = os.environ.get("PVOUTPUT_TARGET_SYSTEM_IDS", "")
    if env.strip():
        return [int(x) for x in env.split(",") if x.strip()]
    if TARGET_SYSTEM_IDS:
        return list(TARGET_SYSTEM_IDS)
    if SHORTLIST_CSV.exists():
        return pd.read_csv(SHORTLIST_CSV)["system_id"].astype(int).tolist()
    sys.exit(f"No target systems. Provide {SHORTLIST_CSV}, set TARGET_SYSTEM_IDS, "
             f"or PVOUTPUT_TARGET_SYSTEM_IDS.")

def fetch_daily(client, store, sid):
    """Pull the daily summary in <=150-day chunks; all-or-nothing per system."""
    rows, d = [], START_DATE
    while d <= END_DATE:
        chunk_end = min(d + timedelta(days=149), END_DATE)
        status, got = client.get_output_days(sid, d, chunk_end)
        if status == "authfail":
            return "authfail"
        if status in ("badrequest", "neterror"):
            return status
        rows.extend(got)                 
        d = chunk_end + timedelta(days=1)
    store.write_daily(rows)
    return "ok"

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    client = PVOutputClient(API_KEY, SYSTEM_ID, REQUESTS_PER_HOUR)
    ids = load_target_ids()
    all_days = list(daterange(START_DATE, END_DATE))
    print(f"{len(ids)} systems x {len(all_days)} days. Capturing ALL fields "
          f"(ext=1) + daily summary. Resumable -- safe to stop/re-run.\n")

    deferred = 0
    try:
        for n, sid in enumerate(ids, 1):
            store = SystemStore(OUTPUT_DIR, sid)
            todo = [d for d in all_days if d.isoformat() not in store.completed]
            print(f"[{n}/{len(ids)}] system {sid}: "
                  f"{len(all_days)-len(todo)} done, {len(todo)} to fetch"
                  f"{' (+daily)' if (FETCH_DAILY_SUMMARY and not store.daily_done) else ''}")
            for i, day in enumerate(todo, 1):
                status, rows = client.get_day(sid, day)
                if status == "authfail":
                    print("\nSTOPPING: donation mode not active (sid1 blocked).")
                    sys.exit(1)
                if status == "badrequest":
                    print(f"\nSTOPPING: bad request while fetching {sid} on {day}.")
                    sys.exit(1)
                if status == "neterror":
                    deferred += 1
                    print(f"    deferred {day} (retry next run)")
                    continue
                store.write_day(day.isoformat(), rows)
                if i % 50 == 0:
                    print(f"    {sid}: {i}/{len(todo)} this run")
            if FETCH_DAILY_SUMMARY and not store.daily_done:
                res = fetch_daily(client, store, sid)
                if res == "authfail":
                    print("\nSTOPPING: donation mode not active."); sys.exit(1)
                if res == "badrequest":
                    print(f"\nSTOPPING: bad daily-summary request for {sid}.")
                    sys.exit(1)
                if res == "neterror":
                    deferred += 1
                    print(f"    daily summary for {sid} deferred (retry next run)")
    except KeyboardInterrupt:
        print("\nInterrupted. Progress saved -- re-run to resume.")
        sys.exit(130)

    print("\nConsolidating ...")
    for label, suffix, key in [("5min", ".csv", "all_systems_5min.csv"),
                               ("daily", "_daily.csv", "all_systems_daily.csv")]:
        parts = []
        for sid in ids:
            p = OUTPUT_DIR / f"system_{sid}{suffix}"
            if p.exists():
                d = pd.read_csv(p)
                if len(d):
                    parts.append(d)
        if parts:
            comb = pd.concat(parts, ignore_index=True)
            subset = ["system_id", "timestamp"] if label == "5min" else ["system_id", "date"]
            comb = comb.drop_duplicates(subset)
            comb.to_csv(OUTPUT_DIR / key, index=False)
            print(f"  {label}: {len(comb)} rows -> {OUTPUT_DIR/key}")

    if deferred:
        print(f"\n{deferred} item(s) deferred (network). Re-run to fetch them.")
    else:
        print("\nAll requested data fetched. Done.")

if __name__ == "__main__":
    main()