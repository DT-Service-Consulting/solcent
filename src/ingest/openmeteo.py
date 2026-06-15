"""
openmeteo.py - Ingest LEAKAGE-SAFE day-ahead weather from Open-Meteo Previous Runs API.

Why this API (and not the Historical Forecast API):
  The Historical Forecast API stitches the *freshest* hours of each model run into a
  near-real-time series. That value does NOT exist yet at your noon-before deadline, so
  training on it leaks. The Previous Runs API returns each variable at a fixed lead time
  (`_previous_day1`, `_previous_day2`, ...) -- "the weather AS FORECAST a day or two
  earlier" -- exactly what you really have at the gate-closure issue time.

Lead-time mapping (confirm against gate-closure timing with the PM):
  day1 ~ early part of day D ; day2 ~ later part of day D.
"""
import time

import requests
import pandas as pd

PREV_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"

VARIABLES = [
    "shortwave_radiation", "direct_radiation", "diffuse_radiation",
    "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
    "temperature_2m", "wind_speed_10m",
]

MODELS = {"ecmwf": "ecmwf_ifs025", "icon": "icon_eu", "gfs": "gfs_seamless"}
LEADS = (1, 2)


def _get(params, tries=6):
    last = None
    for k in range(tries):
        try:
            last = requests.get(PREV_URL, params=params, timeout=120).json()
            if "hourly" in last:
                return last
        except requests.exceptions.RequestException as e:
            last = str(e)
        time.sleep(8 * (k + 1))  # free tier rate-limits repeated calls
    raise RuntimeError(f"Open-Meteo Previous Runs failed: {str(last)[:200]}")


def fetch_day_ahead_weather(lat, lon, start, end,
                            models=MODELS, variables=VARIABLES, leads=LEADS):
    """Hourly day-ahead-lead weather for a single point, aligned on UTC time.
    Columns look like: shortwave_radiation_previous_day1_ecmwf"""
    hourly = [f"{v}_previous_day{d}" for v in variables for d in leads]
    out = None
    for short, model in models.items():
        j = _get({"latitude": lat, "longitude": lon,
                  "start_date": f"{start:%Y-%m-%d}", "end_date": f"{end:%Y-%m-%d}",
                  "hourly": ",".join(hourly), "models": model, "timezone": "GMT"})
        df = pd.DataFrame(j["hourly"])
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.set_index("time")
        df.columns = [f"{c}_{short}" for c in df.columns]
        out = df if out is None else out.join(df, how="outer")
        print(f"  open-meteo {model} -> {df.shape[1]} cols, {df.shape[0]} hours")
        time.sleep(2)
    return out.sort_index()


if __name__ == "__main__":
    from datetime import datetime
    wx = fetch_day_ahead_weather(50.85, 4.35, datetime(2025, 4, 19), datetime(2025, 4, 21))
    print(wx.shape)
    print(wx.filter(like="shortwave_radiation_previous_day1").head())
