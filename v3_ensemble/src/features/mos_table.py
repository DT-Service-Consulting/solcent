#!/usr/bin/env python3
"""
mos_table.py
============
Hourly table for the reference comparison and the MOS experiments, built from
the DAY-AHEAD ECMWF forecast.

WHAT CHANGED AND WHY
--------------------
The earlier table (parent data/interim/mos_training_data.csv) took ECMWF GHI,
cloud and temperature from Open-Meteo's historical-forecast API without the
previous_day1 suffix. That endpoint stitches the first hours of successive
runs, i.e. a short-lead forecast, not the day-ahead forecast the PV model
uses. Here:

  ecmwf_ghi    ECMWF IFS 0.25, previous_day1 (ghi_lag1 in ensemble_ghi_all.csv)
  ecmwf_cloud  ECMWF previous_day1 cloud cover   } on-the-hour rows of the parent
  ecmwf_temp   ECMWF previous_day1 2-m temperature} feature build (exact hourly values)
  sat_ghi      SARAH-3 (satellite_ghi_all.csv)
  era5_ghi     ERA5    (era5_ghi_all.csv, fetch_era5.py)

All three irradiance series are Open-Meteo preceding-hour means on the same
end-of-hour stamps, so they are compared like for like. Geometry predictors
(clear-sky GHI, sun elevation, time of day, day of year) are evaluated for the
hour the values average: clear-sky as the mean over the hour, the others at
the hour centre (stamp - 30 min). Rows: the 33 study systems, sun elevation at
the hour centre above 5 degrees, all three irradiance series present. The
split follows the PV table: an hour belongs to the partition of its centre.

Output (cached): v3_ensemble/data/interim/mos_hourly_dayahead.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as C
sys.path.insert(0, str(Path(__file__).resolve().parent))
import irradiance_timing as T

CACHE = C.DATA_INTERIM / "mos_hourly_dayahead.csv"
EPS = 1.0
FEATURES = ["ecmwf_ghi", "ecmwf_cloud", "ecmwf_temp", "clearsky_ghi", "kt_fc",
            "sun_elevation", "hour_sin", "hour_cos", "doy_sin", "doy_cos"]


def _split_bounds():
    mr = pd.read_csv(C.PARENT_MODEL_READY, usecols=["system_id", "timestamp", "split"])
    mr["timestamp"] = pd.to_datetime(mr["timestamp"], utc=True)
    return (mr.loc[mr.split == "train", "timestamp"].max(),
            mr.loc[mr.split == "val", "timestamp"].max(),
            sorted(mr.system_id.unique()))


def build(force=False):
    if CACHE.exists() and not force:
        d = pd.read_csv(CACHE)
        d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
        return d
    train_end, val_end, systems = _split_bounds()
    meta = pd.read_csv(C.PARENT_META).drop_duplicates("system_id").set_index("system_id")

    ens = pd.read_csv(C.ENSEMBLE_RAW, usecols=["system_id", "time", "ghi_lag1"])
    ens["timestamp"] = pd.to_datetime(ens["time"], utc=True)
    sat = pd.read_csv(C.DATA_INTERIM / "satellite_ghi_all.csv")
    sat["timestamp"] = pd.to_datetime(sat["time"], utc=True)
    era = pd.read_csv(C.DATA_INTERIM / "era5_ghi_all.csv")
    era["timestamp"] = pd.to_datetime(era["time"], utc=True)

    blocks = []
    for sid in systems:
        f = pd.read_csv(C.PROJECT_ROOT / "data" / "interim" / f"features_system_{sid}.csv",
                        usecols=["timestamp", "cloud_fc", "temp_fc"])
        f["timestamp"] = pd.to_datetime(f["timestamp"], utc=True)
        f = f[f.timestamp.dt.minute == 0]
        d = (ens.loc[ens.system_id == sid, ["timestamp", "ghi_lag1"]]
             .rename(columns={"ghi_lag1": "ecmwf_ghi"})
             .merge(f.rename(columns={"cloud_fc": "ecmwf_cloud", "temp_fc": "ecmwf_temp"}),
                    on="timestamp", how="inner")
             .merge(sat.loc[sat.system_id == sid, ["timestamp", "sat_ghi"]], on="timestamp")
             .merge(era.loc[era.system_id == sid, ["timestamp", "era5_ghi"]], on="timestamp"))
        d = d.drop_duplicates("timestamp").set_index("timestamp").sort_index()
        loc = pvlib.location.Location(float(meta.loc[sid, "latitude"]),
                                      float(meta.loc[sid, "longitude"]), tz="UTC")
        centre = d.index - T.HOUR_CENTRE
        d["clearsky_ghi"] = T.clearsky_hour_means(loc, d.index, ("ghi",))["ghi"].to_numpy()
        d["sun_elevation"] = loc.get_solarposition(centre)["apparent_elevation"].to_numpy()
        local = centre.tz_convert(C.TZ)
        mod = local.hour * 60 + local.minute
        d["hour_sin"] = np.sin(2 * np.pi * mod / 1440)
        d["hour_cos"] = np.cos(2 * np.pi * mod / 1440)
        d["doy_sin"] = np.sin(2 * np.pi * local.dayofyear / 365.25)
        d["doy_cos"] = np.cos(2 * np.pi * local.dayofyear / 365.25)
        d["split"] = np.where(centre <= train_end, "train",
                              np.where(centre <= val_end, "val", "test"))
        d.insert(0, "system_id", sid)
        blocks.append(d.reset_index())
    d = pd.concat(blocks, ignore_index=True)
    d = d.dropna(subset=["ecmwf_ghi", "sat_ghi", "era5_ghi", "ecmwf_cloud", "ecmwf_temp"])
    d = d[d.sun_elevation > 5].reset_index(drop=True)
    d["kt_fc"] = (d.ecmwf_ghi / (d.clearsky_ghi + EPS)).clip(0, 1.3)
    d = d.sort_values(["timestamp", "system_id"]).reset_index(drop=True)
    d.to_csv(CACHE, index=False)
    return d


def ratio_to_grid(ratio_hourly, grid):
    """Hourly correction ratio (end-of-hour stamps) -> 15-min slot centres."""
    s = ratio_hourly.sort_index()
    s = s[~s.index.duplicated()]
    s.index = pd.DatetimeIndex(s.index).tz_convert("UTC") - T.HOUR_CENTRE
    t = T.slot_centres(grid).tz_convert("UTC")
    return (s.reindex(s.index.union(t)).interpolate("time", limit_direction="both")
             .reindex(t).to_numpy())


if __name__ == "__main__":
    d = build(force="--force" in sys.argv)
    print(f"rows={len(d):,} systems={d.system_id.nunique()} "
          f"split={d.split.value_counts().to_dict()}")
