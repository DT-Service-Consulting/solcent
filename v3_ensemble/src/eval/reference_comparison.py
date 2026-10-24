#!/usr/bin/env python3
"""
reference_comparison.py
=======================
Scores the same ECMWF day-ahead GHI forecast against two references over the
whole study window, on identical rows:

    ERA5     reanalysis from the same centre and model (not independent)
    SARAH-3  satellite retrieval (independent of the forecast)

Reported in the manuscript, Section 5.5 ("Reference"): RMSE, bias (MBE,
forecast minus reference, positive = over-forecast, the same convention as
mos_satellite.py and Tuononen et al. 2019) and relative RMSE for each reference, and the ratio
of the two RMSEs.

Rows are the MOS table of mos_satellite.py (hourly, sun elevation > 5 deg,
both references present), all splits.

Output: v3_ensemble/reports/reference_comparison.txt
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as C


def build():
    """Same row selection as mos_satellite.build(), without the model imports."""
    mos = pd.read_csv(C.PROJECT_ROOT / "data" / "interim" / "mos_training_data.csv")
    mos["timestamp"] = pd.to_datetime(mos["timestamp"], utc=True)
    mos = mos.dropna(subset=["system_id", "actual_ghi", "ecmwf_ghi"])
    mos["system_id"] = mos["system_id"].astype(int)
    mos = mos[~mos.system_id.isin(C.EXCLUDE_SYSTEMS)]
    mos = mos.rename(columns={"actual_ghi": "era5_ghi"})

    sat = pd.read_csv(C.DATA_INTERIM / "satellite_ghi_all.csv")
    sat["timestamp"] = pd.to_datetime(sat["time"], utc=True)
    sat = sat[["system_id", "timestamp", "sat_ghi"]].dropna(subset=["sat_ghi"])

    geo = pd.read_csv(C.MODEL_READY_V3, usecols=["system_id", "timestamp", "sun_elevation"])
    geo["timestamp"] = pd.to_datetime(geo["timestamp"], utc=True)

    df = mos.merge(sat, on=["system_id", "timestamp"], how="inner")
    df = df.merge(geo, on=["system_id", "timestamp"], how="inner")
    return df[df.sun_elevation > 5].reset_index(drop=True)


def score(truth, fc):
    e = fc - truth
    return {
        "n": len(e),
        "RMSE": float(np.sqrt((e ** 2).mean())),
        "MBE": float(e.mean()),
        "MAE": float(e.abs().mean()),
        "rRMSE_%": float(100 * np.sqrt((e ** 2).mean()) / truth.mean()),
    }


def main():
    df = build()
    rows = []
    for name, col in [("SARAH-3", "sat_ghi"), ("ERA5", "era5_ghi")]:
        rows.append({"reference": name, **score(df[col], df["ecmwf_ghi"])})
    r = pd.DataFrame(rows)
    ratio_rmse = r.RMSE.iloc[0] / r.RMSE.iloc[1]
    ratio_rrmse = r["rRMSE_%"].iloc[0] / r["rRMSE_%"].iloc[1]

    lines = [
        "ECMWF day-ahead GHI scored against two references (whole window)",
        "=" * 66,
        f"systems={df.system_id.nunique()}  rows={len(df):,}",
        "",
        r.to_string(index=False, float_format=lambda x: f"{x:.2f}"),
        "",
        f"RMSE ratio  SARAH-3 / ERA5 : {ratio_rmse:.3f}",
        f"rRMSE ratio SARAH-3 / ERA5 : {ratio_rrmse:.3f}",
    ]
    text = "\n".join(lines)
    print(text)
    (C.REPORTS_DIR / "reference_comparison.txt").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
