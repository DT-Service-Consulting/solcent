#!/usr/bin/env python3
"""
build_features.py
=================
Turn the raw 5-minute PVOutput data into the model-ready 15-minute table.

UPDATES (V2):
  - Strict enforcement of a single, best-in-class NWP model (ECMWF) for the region.
  - Interpolates Clear-Sky Index (Kt) instead of raw Irradiance to preserve
    the physical curve of the sun at 15-minute intervals.
  - Converts lagged output features into Weather-Normalized Performance Ratio (PR)
    lags to prevent 48-hour-old rain from suppressing today's sunny forecast.
  - Added Belgian-optimal features (relative humidity, wind chill index).
  - Engineered physics-based POA power generation estimate (Expert Power Variable).
  - Introduced explicitly modeled T-48 raw power lag features alongside PR.
  - All outputs and generated directories route to *_v2 postfixes.
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import pvlib

# ----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Using V2 postfixes for directories and combined files
RAW_DIR       = PROJECT_ROOT / "data" / "raw" / "pvoutput_raw"
COMBINED_5MIN = PROJECT_ROOT / "data" / "raw" / "pvoutput_raw" / "all_systems_5min.csv"
META_CSV      = PROJECT_ROOT / "data" / "interim" / "pvoutput_selection" / "systems_metadata.csv"
SHORTLIST_CSV = PROJECT_ROOT / "data" / "interim" / "pvoutput_selection" / "shortlist_final.csv"
OUTPUT_DIR    = PROJECT_ROOT / "data" / "interim_v2" 

START_DATE = "2024-04-01"
END_DATE   = "2026-05-31"
TZ         = "Europe/Brussels"
FREQ       = "15min"
FETCH_BENCHMARK = True      
GAP_INTERP_SLOTS = 2        
DEFAULT_TILT = 35.0         
DEFAULT_AZIMUTH = 180.0     

# Exclusive use of ECMWF as the single best NWP model for Europe/Belgium
MODEL = "ecmwf_ifs025"
FORCE_REBUILD = False       

# Added relative_humidity_2m to support Belgian maritime weather context
WX_VARS = ["shortwave_radiation", "direct_radiation", "diffuse_radiation",
           "direct_normal_irradiance", "temperature_2m", "cloud_cover", 
           "wind_speed_10m", "relative_humidity_2m"]
RENAME = {"shortwave_radiation": "ghi", "direct_radiation": "dhi_horiz",
          "diffuse_radiation": "dhi", "direct_normal_irradiance": "dni",
          "temperature_2m": "temp", "cloud_cover": "cloud", 
          "wind_speed_10m": "wind", "relative_humidity_2m": "rh"}

HIST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
PREV_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
COMPASS = {"N":0,"NNE":22.5,"NE":45,"ENE":67.5,"E":90,"ESE":112.5,"SE":135,"SSE":157.5,
           "S":180,"SSW":202.5,"SW":225,"WSW":247.5,"W":270,"WNW":292.5,"NW":315,"NNW":337.5}
# ----------------------------------------------------------------------------

def azimuth_from_orientation(o):
    if o is None or (isinstance(o, float) and np.isnan(o)): return DEFAULT_AZIMUTH
    s = str(o).strip().upper()
    if s in COMPASS: return COMPASS[s]
    try: return float(s)          
    except ValueError: return DEFAULT_AZIMUTH

def _get_json(url, params, retries=5):
    for a in range(1, retries + 1):
        try: r = requests.get(url, params=params, timeout=60)
        except requests.RequestException as e:
            time.sleep(3 * a); continue
        if r.status_code == 200: return r.json()
        if r.status_code == 429: time.sleep(60); continue
        time.sleep(3 * a)
    return None

def fetch_weather(lat, lon, start, end, previous, model=None):
    model = model or MODEL
    url = PREV_URL if previous else HIST_URL
    suffix = "_previous_day1" if previous else ""
    frames = []
    chunks = pd.date_range(start, end, freq="365D").tolist()
    if pd.Timestamp(chunks[-1]) < pd.Timestamp(end): chunks.append(pd.Timestamp(end))
    for i in range(len(chunks) - 1):
        s = pd.Timestamp(chunks[i]).strftime("%Y-%m-%d")
        e = (pd.Timestamp(chunks[i + 1]) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        if i == len(chunks) - 2: e = pd.Timestamp(end).strftime("%Y-%m-%d")
        j = _get_json(url, {"latitude": lat, "longitude": lon, "start_date": s,
                            "end_date": e, "timezone": "UTC", "models": model,
                            "hourly": ",".join(v + suffix for v in WX_VARS)})
        if not j or "hourly" not in j: continue
        h = j["hourly"]
        d = {"time": h["time"]}
        for v in WX_VARS: d[RENAME[v]] = h.get(v + suffix, [None] * len(h["time"]))
        frames.append(pd.DataFrame(d))
    if not frames: return None
    wx = pd.concat(frames, ignore_index=True).drop_duplicates("time")
    wx["time"] = pd.to_datetime(wx["time"], utc=True).dt.tz_convert(TZ)
    return wx.set_index("time").sort_index()

def load_production(sid):
    per = RAW_DIR / f"system_{sid}_v2.csv"
    if per.exists(): d = pd.read_csv(per, usecols=["system_id", "timestamp", "power_gen_w"])
    elif COMBINED_5MIN.exists():
        d = pd.read_csv(COMBINED_5MIN, usecols=["system_id", "timestamp", "power_gen_w"])
        d = d[d["system_id"] == sid]
    else: return None
    if d.empty: return None
    d["ts"] = pd.to_datetime(d["timestamp"], utc=True).dt.tz_convert(TZ)
    return d.set_index("ts")["power_gen_w"].sort_index().resample(FREQ).mean()

def build_for_system(sid, meta_row):
    lat, lon = float(meta_row["latitude"]), float(meta_row["longitude"])
    cap_kwp = float(meta_row["capacity_kwp"])
    tilt = float(meta_row["tilt_deg"]) if pd.notna(meta_row.get("tilt_deg")) else DEFAULT_TILT
    azim = azimuth_from_orientation(meta_row.get("orientation"))

    prod = load_production(sid)
    if prod is None or prod.dropna().empty: return None

    # 1) Grid Initialization
    grid = pd.date_range(pd.Timestamp(START_DATE, tz=TZ),
                         pd.Timestamp(END_DATE, tz=TZ) + pd.Timedelta(hours=23, minutes=45), freq=FREQ)
    df = pd.DataFrame(index=grid)
    df["power_w"] = prod.reindex(grid)

    # 2) Solar geometry + clear-sky 
    loc = pvlib.location.Location(lat, lon, tz=TZ)
    solpos = loc.get_solarposition(grid)
    cs = loc.get_clearsky(grid, model="ineichen")
    
    df["sun_elevation"] = solpos["apparent_elevation"].values
    df["sun_zenith"] = solpos["apparent_zenith"].values
    df["sun_azimuth"] = solpos["azimuth"].values
    df["clearsky_ghi"] = cs["ghi"].values
    df["clearsky_dni"] = cs["dni"].values
    df["clearsky_dhi"] = cs["dhi"].values

    # 3) Day-ahead weather (PHYSICS-AWARE INTERPOLATION)
    fc = fetch_weather(lat, lon, START_DATE, END_DATE, previous=True)
    if fc is None: return None
    
    # Calculate hourly clear-sky to convert forecast irradiance to Kt (Clear-sky index)
    cs_fc = loc.get_clearsky(fc.index, model="ineichen")
    eps = 1.0 # Epsilon to prevent division by zero at night
    
    fc["kt_ghi"] = fc["ghi"] / (cs_fc["ghi"] + eps)
    fc["kt_dni"] = fc["dni"] / (cs_fc["dni"] + eps)
    fc["kt_dhi"] = fc["dhi"] / (cs_fc["dhi"] + eps)

    # Standard linear interpolation for non-geometric variables including RH
    for col in ["temp", "cloud", "wind", "rh"]:
        df[col + "_fc"] = fc[col].reindex(fc.index.union(grid)).interpolate("time").reindex(grid).values

    # Interpolate Kt, then rebuild 15-min irradiance by multiplying with 15-min clear-sky
    for var, kt_col in [("ghi", "kt_ghi"), ("dni", "kt_dni"), ("dhi", "kt_dhi")]:
        df[kt_col] = fc[kt_col].reindex(fc.index.union(grid)).interpolate("time").reindex(grid).values
        df[var + "_fc"] = df[kt_col] * (df["clearsky_" + var] + eps)

    # 4) POA irradiance calculation
    dni_extra = pvlib.irradiance.get_extra_radiation(grid)
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt, surface_azimuth=azim,
        solar_zenith=solpos["apparent_zenith"].values, solar_azimuth=solpos["azimuth"].values,
        dni=df["dni_fc"].values, ghi=df["ghi_fc"].values, dhi=df["dhi_fc"].values,
        dni_extra=dni_extra.values, model="haydavies")
    df["poa_global_fc"] = poa["poa_global"]

    # Expert Variable (Estimated POA Generation based on physics)
    # T_module approx = T_ambient + POA * (NOCT - 20) / 800 (assuming standard NOCT ~45C)
    t_mod = df["temp_fc"] + df["poa_global_fc"] * (25.0 / 800.0)
    
    # Power efficiency loss ~0.4% per degree C over standard test condition (25C)
    efficiency_loss = 1 - 0.004 * (t_mod - 25.0)
    df["expert_power_fc"] = (cap_kwp * 1000.0) * (df["poa_global_fc"] / 1000.0) * efficiency_loss.clip(lower=0)

    # Wind Chill Index (WCI) for convective cooling effect on panels
    df["wci_fc"] = (13.12 + 0.6215 * df["temp_fc"] 
                    - 11.37 * (df["wind_fc"] ** 0.16) 
                    + 0.3965 * df["temp_fc"] * (df["wind_fc"] ** 0.16))

    # 5) clear-sky index & calendar features
    df["clearsky_index_fc"] = np.where(df["clearsky_ghi"] > 5, df["ghi_fc"] / df["clearsky_ghi"], 0.0)
    df["clearsky_index_fc"] = df["clearsky_index_fc"].clip(0, 1.3)
    minute_of_day = df.index.hour * 60 + df.index.minute
    df["hour_sin"] = np.sin(2 * np.pi * minute_of_day / 1440)
    df["hour_cos"] = np.cos(2 * np.pi * minute_of_day / 1440)
    doy = df.index.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    # 6) Target calculation
    df["is_daylight"] = df["clearsky_ghi"] > 5
    df.loc[~df["is_daylight"], "power_w"] = 0.0
    day = df["is_daylight"]
    df.loc[day, "power_w"] = df.loc[day, "power_w"].interpolate("time", limit=GAP_INTERP_SLOTS, limit_area="inside")
    df["is_valid"] = df["power_w"].notna()
    df["power_w"] = df["power_w"].clip(lower=0)
    df["target_norm"] = df["power_w"] / (cap_kwp * 1000.0)

    # 7) WEATHER-NORMALIZED Lags (Performance Ratio)
    # Explicit t-48 historical PV generation lag
    df["lag_power_d2"] = df["power_w"].reindex(df.index - pd.Timedelta(days=2)).values
    
    # Using Clear-Sky GHI / 1000 as a simple clear-sky proxy for generation potential
    cs_proxy = (df["clearsky_ghi"] / 1000.0).clip(lower=0.05) 
    df["target_pr"] = df["target_norm"] / cs_proxy
    
    df["lag_pr_same_slot_d2"] = df["target_pr"].reindex(df.index - pd.Timedelta(days=2)).values
    
    daily_daytime_pr = (df.loc[df["is_daylight"], "target_pr"]
                        .groupby(df.index[df["is_daylight"]].normalize()).mean())
    daily_daytime_pr.index = daily_daytime_pr.index + pd.Timedelta(days=2)
    df["lag_pr_recent_day_mean"] = df.index.normalize().map(daily_daytime_pr)

    df.insert(0, "system_id", sid)
    df["capacity_kwp"] = cap_kwp
    df["tilt_deg"] = tilt
    df["azimuth_deg"] = azim
    df = df.reset_index().rename(columns={"index": "timestamp"})
    df.rename(columns={"power_w": "target_power_w"}, inplace=True)
    
    # Drop temporary calculation columns
    drop_cols = [c for c in df.columns if c.startswith("kt_") or c.startswith("clearsky_d")]
    df = df.drop(columns=drop_cols + ["target_pr"])
    return df

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not META_CSV.exists() or not SHORTLIST_CSV.exists():
        print(f"Required CSV files missing from {OUTPUT_DIR.parent}. Ensure selection script ran.")
        return
        
    meta = pd.read_csv(META_CSV).drop_duplicates("system_id")
    meta_by_id = {int(r["system_id"]): r for _, r in meta.iterrows()}
    ids = pd.read_csv(SHORTLIST_CSV)["system_id"].astype(int).tolist()
    ids = [s for s in ids if s in meta_by_id]

    print(f"Building features for {len(ids)} systems -> {OUTPUT_DIR}\n")
    failed = []
    
    for sid in ids:
        outp = OUTPUT_DIR / f"features_system_{sid}_v2.csv"
        if outp.exists() and not FORCE_REBUILD: continue
        out = build_for_system(sid, meta_by_id[sid])
        if out is not None:
            out.to_csv(outp, index=False)
            print(f"System {sid} saved.")
        else:
            failed.append(sid)

    built = [OUTPUT_DIR / f"features_system_{s}_v2.csv" for s in ids if (OUTPUT_DIR / f"features_system_{s}_v2.csv").exists()]
    if built:
        allf = pd.concat([pd.read_csv(p) for p in built], ignore_index=True)
        # Combine all to a v2 file
        all_systems_path = PROJECT_ROOT / "data" / "interim_v2" / "features_all_systems_v2.csv"
        allf.to_csv(all_systems_path, index=False)
        print(f"\nCombined {len(built)} systems: {len(allf)} rows -> {all_systems_path}")
    if failed:
        print(f"\n{len(failed)} system(s) failed or were skipped: {failed}")

if __name__ == "__main__":
    main()