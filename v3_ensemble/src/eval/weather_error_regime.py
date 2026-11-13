#!/usr/bin/env python3
"""
weather_error_regime.py
=======================
Irradiance forecast error by forecast sky condition, per system (Fig. 5b/c and
Section 5.2 of the paper). Script version of the analysis previously done in
notebooks/weather_forecast_error_by_sky_condition.ipynb, now on the day-ahead
hourly table (features/mos_table.py) instead of the short-lead ECMWF series.

  forecast : ECMWF IFS day-ahead GHI (previous_day1)
  truth    : SARAH-3
  regime   : forecast clear-sky index kt_fc = ecmwf_ghi / (clear-sky hour mean + 1)
             Overcast < 0.35 <= Partly cloudy < 0.75 <= Clear (notebook bins)
  rows     : whole study window, sun elevation > 5 deg at the hour centre

Output: reports/weather_error/summary_by_system_regime.csv (same columns as before)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as C
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "features"))
import mos_table

BINS = [-0.01, 0.35, 0.75, 1.31]
LABELS = ["Overcast", "Partly cloudy", "Clear"]
EPS = 1.0


def metrics(g):
    a, f = g.sat_ghi.to_numpy(), g.ecmwf_ghi.to_numpy()
    e = f - a
    ss = ((a - a.mean()) ** 2).sum()
    kt_act = np.clip(a / (g.clearsky_ghi.to_numpy() + EPS), 0, 1.3)
    return pd.Series({
        "n": len(g), "mean_actual": a.mean(), "sd_actual": a.std(),
        "MBE": e.mean(), "MAE": np.abs(e).mean(), "RMSE": np.sqrt((e ** 2).mean()),
        "nMAE_%": 100 * np.abs(e).mean() / a.mean() if a.mean() > 0 else np.nan,
        "p90_abs_err": np.percentile(np.abs(e), 90),
        "corr": np.corrcoef(a, f)[0, 1] if len(g) > 2 and a.std() > 0 else np.nan,
        "r2": 1 - (e ** 2).sum() / ss if ss > 0 else np.nan,
        "kt_MBE": (g.kt_fc.to_numpy() - kt_act).mean(),
    })


def main():
    d = mos_table.build().copy()
    d["regime"] = pd.cut(d.kt_fc, BINS, labels=LABELS)
    per = (d.groupby(["system_id", "regime"], observed=True).apply(metrics, include_groups=False)
           .reset_index())
    out_dir = C.REPORTS_DIR / "weather_error"
    out_dir.mkdir(parents=True, exist_ok=True)
    per.to_csv(out_dir / "summary_by_system_regime.csv", index=False)
    fleet = d.groupby("regime", observed=True).apply(metrics, include_groups=False)
    piv = per.pivot(index="system_id", columns="regime", values="RMSE")[LABELS]
    cv = (piv.std() / piv.median()).rename("std/median of per-system RMSE")
    text = ("Day-ahead ECMWF GHI vs SARAH-3 by forecast sky condition (whole window)\n"
            f"systems={d.system_id.nunique()} rows={len(d):,}\n\n"
            + fleet.round(2).to_string() + "\n\n" + cv.round(3).to_string() + "\n")
    print(text)
    (C.REPORTS_DIR / "weather_error_regime.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
