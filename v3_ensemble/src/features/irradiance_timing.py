#!/usr/bin/env python3
"""
irradiance_timing.py
====================
Map hourly Open-Meteo irradiance onto the 15-minute PV grid with the correct
time reference.

THE PROBLEM THIS FIXES
----------------------
Open-Meteo hourly radiation variables (shortwave_radiation, direct_normal_
irradiance, diffuse_radiation, and the SARAH-3 and ERA5 equivalents) are means
over the PRECEDING hour: the value stamped 12:00 UTC is the mean over
11:00-12:00, centred on 11:30 ("Solar radiation is averaged over the past
hour", open-meteo.com/en/docs). The original build (src/features/
build_features.py in the parent project, and interp_member in
build_features_v3.py) treated each value as instantaneous at its stamp, so the
irradiance features lagged the PV target by about 30 minutes.

The PV target in slot t is the mean of the readings in [t, t+15 min), centred
on t + 7.5 min.

METHOD
------
For one hourly series v (stamps = end of the averaging hour):
  1. clear-sky mean over each hour, from 5-minute pvlib Ineichen values;
  2. hourly clear-sky index kt = v / (cs_hour_mean + eps), placed at the
     centre of its hour (stamp - 30 min);
  3. kt interpolated linearly in time to the requested instants (slot centres);
  4. irradiance = kt * (clear-sky at those instants + eps), clipped at 0.

eps = 1 W/m2 matches the original build. Cloud cover, temperature and wind are
instantaneous in Open-Meteo and are not affected.
"""

import numpy as np
import pandas as pd
import pvlib

EPS = 1.0
SLOT_CENTRE = pd.Timedelta(minutes=7.5)
HOUR_CENTRE = pd.Timedelta(minutes=30)


def clearsky_hour_means(loc, hour_ends, components=("ghi", "dni", "dhi")):
    """Mean clear-sky irradiance over (t - 1 h, t] for each end-stamp t (UTC)."""
    hour_ends = pd.DatetimeIndex(hour_ends).tz_convert("UTC")
    fine = pd.date_range(hour_ends.min() - pd.Timedelta(hours=1) + pd.Timedelta(minutes=5),
                         hour_ends.max(), freq="5min", tz="UTC")
    cs = loc.get_clearsky(fine, model="ineichen")[list(components)]
    means = cs.resample("1h", label="right", closed="right").mean()
    return means.reindex(hour_ends)


def hourly_mean_to_instants(v, cs_hour_mean, instants, cs_at_instants):
    """Hourly backward-mean series -> values at arbitrary instants via kt.

    v, cs_hour_mean : Series on the same end-of-hour UTC index
    instants        : DatetimeIndex (any tz) where values are wanted
    cs_at_instants  : array of clear-sky values at `instants`
    """
    kt = (v / (cs_hour_mean + EPS)).astype("float64")
    kt.index = pd.DatetimeIndex(kt.index).tz_convert("UTC") - HOUR_CENTRE
    kt = kt[~kt.index.duplicated()].sort_index()
    t = pd.DatetimeIndex(instants).tz_convert("UTC")
    k = (kt.reindex(kt.index.union(t))
           .interpolate("time", limit_direction="both")
           .reindex(t)
           .to_numpy())
    return np.clip(k * (np.asarray(cs_at_instants, dtype="float64") + EPS), 0.0, None)


def slot_centres(grid):
    return pd.DatetimeIndex(grid) + SLOT_CENTRE
