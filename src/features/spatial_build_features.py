#!/usr/bin/env python3
"""
spatial_build_features.py
=========================
Adapts the "Spatial NWP Error Correction" paper.
For every PV system, this script dynamically generates a 3x3 geographic grid 
(spaced 10km apart) around the house's center point. It then fetches weather 
for all 9 coordinates simultaneously to build a spatial bounding box.

UPDATES:
  - Dynamically handles 429 Rate Limits using API headers and exponential backoff.
  - Processes the entire shortlist (removed the [:5] slice constraint).
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import pvlib

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SHORTLIST_CSV = PROJECT_ROOT / "data" / "interim" / "pvoutput_selection" / "shortlist_final.csv"
META_CSV      = PROJECT_ROOT / "data" / "interim" / "pvoutput_selection" / "systems_metadata.csv"
RAW_DIR       = PROJECT_ROOT / "data" / "raw" / "pvoutput_raw"
OUTPUT_DIR    = PROJECT_ROOT / "data" / "interim" 

START_DATE = "2024-04-01"
END_DATE   = "2026-05-31"
TZ         = "Europe/Brussels"
FREQ       = "15min"
SPACING_KM = 10.0 # 10km spacing between grid points

# We will pull a subset of core features for the 3x3 grid to avoid memory overflow
SPATIAL_VARS = ["shortwave_radiation", "cloud_cover", "temperature_2m", "wind_speed_10m"]
RENAME = {"shortwave_radiation": "ghi", "cloud_cover": "cloud", 
          "temperature_2m": "temp", "wind_speed_10m": "wind"}

HIST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
PREV_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"

def generate_3x3_grid(lat, lon, step_km):
    """Generates 9 coordinates (3x3 grid) centered around the given lat/lon."""
    lat_step = step_km / 111.0
    # Longitude shrinks as we move away from equator
    lon_step = step_km / (111.0 * np.cos(np.radians(lat)))
    
    coords = []
    # Loop from Top-Left to Bottom-Right (-1 to 1)
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            coords.append((lat + (i * lat_step), lon + (j * lon_step)))
    return coords

def _get_json(url, params, retries=8):
    """Fetches JSON with dynamic rate-limit handling and exponential backoff."""
    for a in range(1, retries + 1):
        try: 
            r = requests.get(url, params=params, timeout=60)
        except requests.RequestException as e:
            print(f"      [Network Error] {e}. Retrying in {3 * a}s...")
            time.sleep(3 * a)
            continue
            
        if r.status_code == 200: 
            return r.json()
            
        if r.status_code == 429: 
            # Open-Meteo sometimes provides a Retry-After header. If not, use exponential backoff.
            retry_after = int(r.headers.get("Retry-After", min(120, 15 * (2 ** (a - 1)))))
            print(f"      [429 Rate Limit Hit] Pausing for {retry_after} seconds to respect API limits...")
            time.sleep(retry_after)
            continue
            
        print(f"      [API Error {r.status_code}] Retrying in {5 * a}s...")
        time.sleep(5 * a)
        
    print("      [Failed] Exhausted all API retries.")
    return None

def fetch_spatial_weather(coords, start, end):
    """Batches 9 coordinates into a single API call to return spatial tensors."""
    lats = [c[0] for c in coords]
    lons = [c[1] for c in coords]
    
    frames = []
    chunks = pd.date_range(start, end, freq="365D").tolist()
    if pd.Timestamp(chunks[-1]) < pd.Timestamp(end): chunks.append(pd.Timestamp(end))
    
    for i in range(len(chunks) - 1):
        s = pd.Timestamp(chunks[i]).strftime("%Y-%m-%d")
        e = (pd.Timestamp(chunks[i + 1]) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        if i == len(chunks) - 2: e = pd.Timestamp(end).strftime("%Y-%m-%d")
        
        # Request previous day-ahead forecast for ALL 9 coordinates simultaneously
        j = _get_json(PREV_URL, {
            "latitude": ",".join(map(str, lats)),
            "longitude": ",".join(map(str, lons)),
            "start_date": s,
            "end_date": e,
            "timezone": "UTC",
            "hourly": ",".join(v + "_previous_day1" for v in SPATIAL_VARS)
        })
        
        if not j: continue
        
        # Open-Meteo returns a list of 9 dictionaries when querying multiple locations
        if not isinstance(j, list): j = [j] 
        
        combined_chunk = None
        for grid_idx, loc_data in enumerate(j):
            h = loc_data["hourly"]
            if combined_chunk is None:
                combined_chunk = pd.DataFrame({"time": h["time"]})
                
            for v in SPATIAL_VARS:
                col_name = f"{RENAME[v]}_grid_{grid_idx}"
                combined_chunk[col_name] = h.get(v + "_previous_day1", [None] * len(h["time"]))
                
        frames.append(combined_chunk)
        # Polite buffer to prevent triggering burst rate limits
        time.sleep(1.5) 
        
    if not frames: return None
    wx = pd.concat(frames, ignore_index=True).drop_duplicates("time")
    wx["time"] = pd.to_datetime(wx["time"], utc=True).dt.tz_convert(TZ)
    return wx.set_index("time").sort_index()

def load_production(sid):
    per = RAW_DIR / f"system_{sid}.csv"
    if not per.exists(): return None
    d = pd.read_csv(per, usecols=["system_id", "timestamp", "power_gen_w"])
    if d.empty: return None
    d["ts"] = pd.to_datetime(d["timestamp"], utc=True).dt.tz_convert(TZ)
    return d.set_index("ts")["power_gen_w"].sort_index().resample(FREQ).mean()

def build_for_system(sid, meta_row):
    lat, lon = float(meta_row["latitude"]), float(meta_row["longitude"])
    cap_kwp = float(meta_row["capacity_kwp"])

    prod = load_production(sid)
    if prod is None or prod.dropna().empty: return None

    # Generate the 3x3 Bounding Box
    coords = generate_3x3_grid(lat, lon, SPACING_KM)
    
    grid = pd.date_range(pd.Timestamp(START_DATE, tz=TZ),
                         pd.Timestamp(END_DATE, tz=TZ) + pd.Timedelta(hours=23, minutes=45), freq=FREQ)
    df = pd.DataFrame(index=grid)
    df["power_w"] = prod.reindex(grid)

    # Calculate Solar Geometry and Clear-Sky features for the central point
    loc = pvlib.location.Location(lat, lon, tz=TZ)
    solpos = loc.get_solarposition(grid)
    cs = loc.get_clearsky(grid, model="ineichen")
    
    df["sun_elevation"] = solpos["apparent_elevation"].values
    df["sun_azimuth"] = solpos["azimuth"].values
    df["clearsky_ghi"] = cs["ghi"].values
    
    minute_of_day = df.index.hour * 60 + df.index.minute
    df["hour_cos"] = np.cos(2 * np.pi * minute_of_day / 1440)

    print(f"    -> Fetching 3x3 spatial grid (9 points)...")
    fc = fetch_spatial_weather(coords, START_DATE, END_DATE)
    if fc is None: return None
    
    # Interpolate hourly weather to 15-minute slots for all 9x4=36 spatial features
    for col in fc.columns:
        df[col] = fc[col].reindex(fc.index.union(grid)).interpolate("time").reindex(grid).values

    # Determine daylight using the physical clear-sky model
    df["is_daylight"] = df["clearsky_ghi"] > 5
    df.loc[~df["is_daylight"], "power_w"] = 0.0
    day = df["is_daylight"]
    df.loc[day, "power_w"] = df.loc[day, "power_w"].interpolate("time", limit=2, limit_area="inside")
    
    df["is_valid"] = df["power_w"].notna()
    df["power_w"] = df["power_w"].clip(lower=0)
    df["target_norm"] = df["power_w"] / (cap_kwp * 1000.0)

    df.insert(0, "system_id", sid)
    df["capacity_kwp"] = cap_kwp
    df = df.reset_index().rename(columns={"index": "timestamp"})
    
    return df

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(META_CSV).drop_duplicates("system_id")
    meta_by_id = {int(r["system_id"]): r for _, r in meta.iterrows()}
    
    # Process ALL systems in the shortlist
    ids = pd.read_csv(SHORTLIST_CSV)["system_id"].astype(int).tolist()
    ids = [s for s in ids if s in meta_by_id] 

    print(f"Building SPATIAL features for ALL {len(ids)} systems -> {OUTPUT_DIR}\n")
    print("This will execute roughly 105 total API calls (batching 9 coordinates per request).")
    
    built = []
    for count, sid in enumerate(ids, 1):
        outp = OUTPUT_DIR / f"spatial_features_system_{sid}.csv"
        
        # Resume capability: Skip if already built
        if outp.exists():
            print(f"[{count}/{len(ids)}] System {sid} already built. Skipping API call.")
            built.append(pd.read_csv(outp))
            continue
            
        print(f"[{count}/{len(ids)}] Processing System {sid}...")
        out = build_for_system(sid, meta_by_id[sid])
        if out is not None:
            out.to_csv(outp, index=False)
            built.append(out)
            print(f"  -> Complete. Shape: {out.shape} -> Saved to {outp.name}")
        
        # Extra buffer between completely different systems to be ultra-safe on API limits
        time.sleep(2.0)

    if built:
        allf = pd.concat(built, ignore_index=True)
        out_path = PROJECT_ROOT / "data" / "interim" / "spatial_features_all_systems.csv"
        allf.to_csv(out_path, index=False)
        print(f"\nSaved combined SPATIAL dataset: {len(allf):,} rows -> {out_path.name}")

if __name__ == "__main__":
    main()