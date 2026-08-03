#!/usr/bin/env python3
"""
pvoutput_belgium.py
===================
Download REAL measured solar production from individual households in Belgium
using the free PVOutput.org API, and assemble it into a clean dataset you can
feed straight into the modelling pipeline.

WHAT THIS GIVES YOU
-------------------
For each PV system you point it at, you get:
  * system metadata  -> capacity (kWp), location, orientation, tilt
  * a time series    -> power (W) and energy (Wh) at the system's logging interval
This solves both of your blockers at once: real household production data AND
the panel specs, with no need to own a roof or simulate one.

BEFORE YOU RUN
--------------
1. Register a free account at https://pvoutput.org (if you don't have a PV
   system, tick the "energy consumption only" box during registration).
2. Get your API key and your own System ID from Settings.
3. Put them in environment variables, or edit the CONFIG block below:
       export PVOUTPUT_API_KEY="xxxxxxxx"
       export PVOUTPUT_SYSTEM_ID="123456"     # YOUR own system id
4. Choose how to pick target systems (see TARGET_SYSTEM_IDS / search below).

RATE-LIMIT REALITY CHECK (DONATION TIER)
----------------------------------------
You are on the $15/yr DONATION plan, which gives 300 requests/hour and unlocks
reading OTHER people's public systems via sid1 (free accounts cannot do this at
all). ONE request still = ONE day for ONE system, so:
    1 system-year  ~= 365 requests  ~= ~1.2 hours
    10 systems x 1yr ~= 3,650 requests ~= ~12 hours (run overnight)
The script self-throttles to stay under 300/hr, reads the X-Rate-Limit headers
to confirm donor mode is active, and saves each system as it finishes -- so it
is safe to stop and re-run; completed systems are skipped.

The PVOutput API returns plain delimited text (not JSON). Column orders can vary
between API versions, so the script PRINTS the first raw response from each
endpoint and parses defensively. Eyeball that printout once and adjust the index
constants near the top if anything looks shifted.
"""

import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import requests
import pandas as pd

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv:
    load_dotenv()

# ----------------------------------------------------------------------------
# CONFIG  (edit here, or set the matching environment variables)
# ----------------------------------------------------------------------------
API_KEY = os.environ.get("PVOUTPUT_API_KEY")
SYSTEM_ID = os.environ.get("PVOUTPUT_SYSTEM_ID")

# Date range to download (inclusive). Keep it SMALL for your first run.
START_DATE = date(2024, 6, 1)
END_DATE   = date(2024, 6, 30)

# How you pick which households to download:
#   Option A (RELIABLE): paste system IDs you found by browsing pvoutput.org.
#       Browse https://pvoutput.org -> "Ladders"/teams -> filter by Belgium,
#       open systems, and copy the numeric id from the URL (sid=...).
#   Option B (CONVENIENT): leave this empty [] and let the script search by
#       location (see SEARCH_* below). Search parsing is more fragile, so
#       verify the printed raw output the first time.
def env_system_ids(name="PVOUTPUT_TARGET_SYSTEM_IDS"):
    raw = os.environ.get(name, "")
    return [int(s.strip()) for s in raw.split(",") if s.strip()]


TARGET_SYSTEM_IDS = env_system_ids() or [
    # 12345,
    # 67890,
]

# Used only if TARGET_SYSTEM_IDS is empty: search around a Belgian coordinate.
SEARCH_LAT = 50.8503        # Brussels
SEARCH_LON = 4.3517
SEARCH_DISTANCE_KM = 50
SEARCH_MAX_SYSTEMS = 8      # cap so your first run stays cheap
SEARCH_SEEN_DAYS = 30       # only systems active within this many days

# Donation tier = 300 requests/hour (free = 60). You are on the $15/yr donation
# plan, so 300 is correct. One request still = one system-day.
REQUESTS_PER_HOUR = 300

OUTPUT_DIR = Path("pvoutput_belgium_data")

# ----------------------------------------------------------------------------
# Column indices for the delimited responses. Adjust if the printed raw lines
# don't line up (the API has changed field order across versions).
# getstatus.jsp history record with h=1:
# date,time,Egen,efficiency,Pinst,Pavg,norm,Econs,Pcons,temp,volt
H_DATE, H_TIME, H_EGEN, H_EFF, H_PGEN, H_PAVG, H_NORM, H_ECONS, H_PCONS, H_TEMP, H_VOLT = range(11)
# getsystem.jsp first record: name,size,postcode,#panels,panelW,brand,#inv,invW,
#                             invBrand,orientation,tilt,shade,install,lat,lon,interval...
SY_NAME, SY_SIZE_W, SY_ORIENT, SY_TILT, SY_LAT, SY_LON, SY_INTERVAL = 0, 1, 9, 10, 13, 14, 15
# ----------------------------------------------------------------------------

BASE = "https://pvoutput.org/service/r2"


class PVOutputClient:
    """Tiny wrapper around the PVOutput API with auth + self-throttling."""

    def __init__(self, api_key, system_id, requests_per_hour=60):
        if not api_key or not system_id:
            sys.exit(
                "ERROR: set PVOUTPUT_API_KEY and PVOUTPUT_SYSTEM_ID first. "
                "You can put them in a .env file or environment variables."
            )
        self.headers = {
            "X-Pvoutput-Apikey": str(api_key),
            "X-Pvoutput-SystemId": str(system_id),
        }
        # Minimum seconds between requests to respect the hourly quota.
        self.min_interval = 3600.0 / max(1, requests_per_hour)
        self._last_call = 0.0
        self._printed_raw = set()   # so we only print one raw sample per endpoint
        self.donation_required = False   # set if we hit the sid1 permission wall
        self.rate_limit = None           # X-Rate-Limit-Limit (300 confirms donor)
        self.rate_remaining = None       # X-Rate-Limit-Remaining
        self._rate_reset = None          # X-Rate-Limit-Reset (unix epoch)
        self._printed_limit = False

    def _throttle(self):
        wait = self.min_interval - (time.time() - self._last_call)
        if wait > 0:
            time.sleep(wait)

    def _read_rate_headers(self, r):
        """Parse PVOutput's X-Rate-Limit-* headers; confirm donor mode; sleep if empty."""
        try:
            self.rate_limit = int(r.headers.get("X-Rate-Limit-Limit", 0)) or self.rate_limit
            self.rate_remaining = int(r.headers.get("X-Rate-Limit-Remaining", -1))
            self._rate_reset = int(r.headers.get("X-Rate-Limit-Reset", 0)) or self._rate_reset
        except (TypeError, ValueError):
            return
        if not self._printed_limit and self.rate_limit:
            self._printed_limit = True
            tier = "DONATION" if self.rate_limit >= 300 else "FREE"
            print(f"  [quota] limit={self.rate_limit}/hr ({tier} tier), "
                  f"remaining={self.rate_remaining}")
            if self.rate_limit < 300:
                print("  ! WARNING: limit < 300 -> donation mode may NOT be active yet. "
                      "Check your account page.")
        # If we've drained the bucket, wait until it resets (plus a small buffer).
        if self.rate_remaining is not None and 0 <= self.rate_remaining <= 1 and self._rate_reset:
            sleep_for = max(0, self._rate_reset - time.time()) + 5
            print(f"  [quota] exhausted -> sleeping {sleep_for:.0f}s until reset")
            time.sleep(sleep_for)

    def _get(self, endpoint, params):
        """Make one GET request, returning response text or None on a soft error."""
        self._throttle()
        url = f"{BASE}/{endpoint}"
        try:
            r = requests.get(url, headers=self.headers, params=params, timeout=30)
        except requests.RequestException as e:
            print(f"  ! network error on {endpoint}: {e}")
            return None
        finally:
            self._last_call = time.time()

        # PVOutput reports your quota in response headers. Use them to (a) confirm
        # donation mode is live (limit should read 300) and (b) wait out the hour
        # if you somehow exhaust the bucket.
        self._read_rate_headers(r)

        # Show one raw sample per endpoint so you can verify the format.
        if endpoint not in self._printed_raw:
            self._printed_raw.add(endpoint)
            print(f"  [raw {endpoint}] {r.text[:200]!r}")

        if r.status_code == 200:
            return r.text
        # 400 with "no status found" just means the system didn't report that day.
        if r.status_code == 400 and "no status" in r.text.lower():
            return None
        # The sid1 permission wall: free accounts can't read other systems.
        if r.status_code in (401, 403) and "inaccessible" in r.text.lower():
            if not self.donation_required:
                print("  ! PVOutput refused another system's data "
                      "('Inaccessible System ID').")
                print("    This requires DONATION MODE on your account "
                      "(sid1 is donor-only). Free accounts can only read "
                      "their OWN system. See https://pvoutput.org/help/donations.html")
            self.donation_required = True
            return None
        # Genuine quota/auth problems: back off, then give up on this call.
        if r.status_code in (401, 403):
            print(f"  ! auth/quota problem ({r.status_code}): {r.text[:120]} "
                  f"-- backing off 60s")
            time.sleep(60)
            return None
        print(f"  ! {endpoint} returned {r.status_code}: {r.text[:120]}")
        return None

    # --- metadata -----------------------------------------------------------
    def get_system_metadata(self, sid):
        txt = self._get("getsystem.jsp", {"sid1": sid, "ext": 0})
        if not txt:
            return None
        fields = txt.split(";")[0].split(",")

        def grab(i, cast=str):
            try:
                return cast(fields[i])
            except (IndexError, ValueError):
                return None

        size_w = grab(SY_SIZE_W, float)
        return {
            "system_id": sid,
            "name": grab(SY_NAME),
            "capacity_kwp": round(size_w / 1000.0, 3) if size_w else None,
            "orientation": grab(SY_ORIENT),
            "tilt_deg": grab(SY_TILT, float),
            "latitude": grab(SY_LAT, float),
            "longitude": grab(SY_LON, float),
            "interval_min": grab(SY_INTERVAL, float),
            "raw": txt.split(";")[0],
        }

    # --- discovery ----------------------------------------------------------
    def search_systems(self, lat, lon, distance_km, max_systems):
        """Find systems near a coordinate AND grab their free metadata.

        The search response already contains capacity, orientation and
        coordinates, so we don't need the donor-only getsystem.jsp for those.
        Observed field order (verify against the printed raw line):
          0 name, 1 size_W, 2 'Belgium <postcode>', 3 orientation,
          4 n_outputs, 5 last_output, 6 system_id, 7 panel, 8 inverter,
          9 distance_km, 10 latitude, 11 longitude
        (Tilt is NOT in the search response; it needs getsystem -> donation.)
        """
        txt = self._get("search.jsp", {
            "q": f"{distance_km}km",
            "ll": f"{lat},{lon}",
            "country": 1,
            "country_code": "be",
            "seen": SEARCH_SEEN_DAYS,
        })
        if not txt:
            return []
        out = []
        for record in txt.splitlines():
            if not record.strip():
                continue
            parts = [p.strip() for p in record.split(",")]

            def grab(i, cast=str):
                try:
                    return cast(parts[i])
                except (IndexError, ValueError):
                    return None

            sid = grab(6, int)
            if sid is None:                      # fall back: last numeric field
                for p in reversed(parts):
                    if p.isdigit():
                        sid = int(p)
                        break
            if sid is None:
                continue

            size_w = grab(1, float)
            out.append({
                "system_id": sid,
                "name": grab(0),
                "capacity_kwp": round(size_w / 1000.0, 3) if size_w else None,
                "location": grab(2),
                "orientation": grab(3),
                "panel": grab(7),
                "inverter": grab(8),
                "distance_km": grab(9, float),
                "latitude": grab(-2, float),
                "longitude": grab(-1, float),
                "tilt_deg": None,            # not available without donation
                "raw": record,
            })
            if len(out) >= max_systems:
                break
        return out

    # --- time series --------------------------------------------------------
    def get_day(self, sid, day):
        """One day of interval data for one system. day is a datetime.date."""
        txt = self._get("getstatus.jsp", {
            "sid1": sid, "d": day.strftime("%Y%m%d"), "h": 1, "limit": 288, "asc": 1,
        })
        if not txt:
            return []
        rows = []
        for record in txt.split(";"):
            parts = record.split(",")
            if len(parts) < 4:
                continue

            def num(i):
                try:
                    value = parts[i].strip()
                    if value.lower() == "nan" or value == "":
                        return None
                    return float(value)
                except (IndexError, ValueError):
                    return None

            ts = f"{parts[H_DATE]} {parts[H_TIME]}"
            rows.append({
                "system_id": sid,
                "timestamp_local": ts,
                "energy_gen_wh": num(H_EGEN),
                "power_gen_w": num(H_PGEN),
                "normalised_output": num(H_NORM),   # power / capacity (kW/kW)
                "energy_cons_wh": num(H_ECONS),
                "power_cons_w": num(H_PCONS),
                "temperature_c": num(H_TEMP),
                "voltage_v": num(H_VOLT),
            })
        return rows


def daterange(start, end):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def tidy_frame(rows):
    """Turn raw status rows into a clean, timezone-aware DataFrame."""
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp_local"],
                                     format="%Y%m%d %H:%M", errors="coerce")
    df = df.dropna(subset=["timestamp"]).drop(columns=["timestamp_local"])
    # PVOutput logs in the system's LOCAL time -> Belgian systems = Europe/Brussels.
    df["timestamp"] = (df["timestamp"]
                       .dt.tz_localize("Europe/Brussels",
                                       nonexistent="shift_forward",
                                       ambiguous="NaT"))
    df = df.dropna(subset=["timestamp"])
    # Light sanity guard only (real cleaning belongs in your preprocessing step):
    # production can't be negative.
    df.loc[df["power_gen_w"] < 0, "power_gen_w"] = pd.NA
    return df.sort_values(["system_id", "timestamp"]).reset_index(drop=True)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    client = PVOutputClient(API_KEY, SYSTEM_ID, REQUESTS_PER_HOUR)

    print(f"Throttling to ~{REQUESTS_PER_HOUR} requests/hour "
          f"({client.min_interval:.0f}s between calls).")

    # 1) Decide which systems to pull, capturing free metadata when we search.
    search_meta = []
    if TARGET_SYSTEM_IDS:
        target_ids = list(TARGET_SYSTEM_IDS)
        print(f"Using {len(target_ids)} system id(s) from TARGET_SYSTEM_IDS.")
    else:
        print(f"Searching for systems within {SEARCH_DISTANCE_KM} km of "
              f"({SEARCH_LAT}, {SEARCH_LON}) ...")
        search_meta = client.search_systems(SEARCH_LAT, SEARCH_LON,
                                             SEARCH_DISTANCE_KM, SEARCH_MAX_SYSTEMS)
        target_ids = [f["system_id"] for f in search_meta]
        print(f"Found {len(target_ids)} system(s): {target_ids}")
    if not target_ids:
        sys.exit("No target systems. Add IDs to TARGET_SYSTEM_IDS or widen the search.")

    # 2) Metadata. When we searched, the search response already gave
    #    capacity/orientation/coords for FREE. When IDs were supplied directly,
    #    fetch full specs (incl. tilt) via getsystem.jsp -- donation enables this.
    if search_meta:
        meta_df = pd.DataFrame(search_meta).drop(columns=["raw"], errors="ignore")
        meta_df.to_csv(OUTPUT_DIR / "systems_metadata.csv", index=False)
        print(f"Saved free metadata for {len(meta_df)} system(s) "
              f"-> {OUTPUT_DIR/'systems_metadata.csv'}")
    else:
        print("Fetching per-system metadata via getsystem.jsp ...")
        meta_rows = []
        for sid in target_ids:
            m = client.get_system_metadata(sid)
            if client.donation_required:
                break
            if m:
                meta_rows.append(m)
                print(f"   system {sid}: {m['capacity_kwp']} kWp, tilt {m['tilt_deg']}, "
                      f"@({m['latitude']},{m['longitude']})")
        if meta_rows:
            pd.DataFrame(meta_rows).drop(columns=["raw"], errors="ignore").to_csv(
                OUTPUT_DIR / "systems_metadata.csv", index=False)
            print(f"Saved metadata for {len(meta_rows)} system(s) "
                  f"-> {OUTPUT_DIR/'systems_metadata.csv'}")

    # 2b) PROBE: one cheap call on START_DATE (your historical start) confirms
    #     BOTH that donor access works AND that the system retains data that far
    #     back -- before committing to a long download.
    print(f"\nProbing access on system {target_ids[0]} for {START_DATE} ...")
    probe = client.get_day(target_ids[0], START_DATE)
    if client.donation_required:
        print("\nSTOPPING: your account cannot read other people's systems.")
        print("Fix: enable Donation mode at https://pvoutput.org/help/donations.html")
        print("     (confirm your account page shows it active, then re-run).")
        print("Or: switch to the free Fluvius Belgian dataset as your source.")
        sys.exit(1)
    if not probe:
        print(f"  ! No data for {START_DATE} on system {target_ids[0]}. Donor access "
              f"is fine, but this system may not retain history that far back -- "
              f"try a later START_DATE or a different system.")
    else:
        print(f"Access OK ({len(probe)} records on the probe day). Proceeding.")

    # 3) Time-series history, saved per system so progress isn't lost.
    n_days = (END_DATE - START_DATE).days + 1
    est_hours = (n_days * len(target_ids)) / max(1, REQUESTS_PER_HOUR)
    print(f"\nDownloading {n_days} day(s) x {len(target_ids)} system(s) "
          f"= ~{n_days * len(target_ids)} requests (~{est_hours:.1f} h at "
          f"{REQUESTS_PER_HOUR}/hr). Safe to stop and re-run.")
    for sid in target_ids:
        out_path = OUTPUT_DIR / f"system_{sid}.csv"
        if out_path.exists():
            print(f"System {sid}: already downloaded, skipping. "
                  f"(delete {out_path.name} to refetch)")
            continue
        rows = []
        for i, day in enumerate(daterange(START_DATE, END_DATE), 1):
            rows.extend(client.get_day(sid, day))
            if i % 10 == 0:
                print(f"  system {sid}: {i}/{n_days} days")
        df = tidy_frame(rows)
        if df.empty:
            print(f"System {sid}: no data returned (may be private or inactive).")
            continue
        df.to_csv(out_path, index=False)
        print(f"System {sid}: {len(df)} rows -> {out_path}")

    # 4) Combine everything into one modelling-ready file.
    parts = [pd.read_csv(p, parse_dates=["timestamp"])
             for p in OUTPUT_DIR.glob("system_*.csv")]
    if parts:
        combined = pd.concat(parts, ignore_index=True)
        combined.to_csv(OUTPUT_DIR / "all_systems_combined.csv", index=False)
        print(f"\nDONE. Combined dataset: {len(combined)} rows across "
              f"{combined['system_id'].nunique()} systems "
              f"-> {OUTPUT_DIR/'all_systems_combined.csv'}")


if __name__ == "__main__":
    main()