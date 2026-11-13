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

Rows are the day-ahead MOS table (features/mos_table.py: hourly, sun elevation
> 5 deg at the hour centre, both references present), all splits.

Output: v3_ensemble/reports/reference_comparison.txt
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as C
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "features"))
import mos_table  # day-ahead hourly table (replaces mos_training_data.csv)


def build():
    """Day-ahead ECMWF, SARAH-3 and ERA5 on identical hourly rows (mos_table.py)."""
    return mos_table.build()


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
    coarse = C.DATA_INTERIM / "satellite_ghi_025_all.csv"
    refs = [("SARAH-3", "sat_ghi"), ("ERA5", "era5_ghi")]
    if coarse.exists():
        # SARAH-3 averaged over the forecast's 0.25-degree cell (fetch_sarah_coarse.py),
        # scored on the same rows, so the resolution mismatch is removed
        c = pd.read_csv(coarse)
        c["timestamp"] = pd.to_datetime(c["time"], utc=True)
        df = df.merge(c[["system_id", "timestamp", "sat_ghi_025"]],
                      on=["system_id", "timestamp"], how="inner").dropna(subset=["sat_ghi_025"])
        refs.append(("SARAH-3, 0.25 deg cell mean", "sat_ghi_025"))
    rows = []
    for name, col in refs:
        rows.append({"reference": name, **score(df[col], df["ecmwf_ghi"])})
    r = pd.DataFrame(rows)
    ratio_rmse = r.RMSE.iloc[0] / r.RMSE.iloc[1]
    ratio_rrmse = r["rRMSE_%"].iloc[0] / r["rRMSE_%"].iloc[1]
    extra = []
    if len(r) == 3:
        extra = [f"RMSE ratio  SARAH-3 0.25 / ERA5 : {r.RMSE.iloc[2] / r.RMSE.iloc[1]:.3f}",
                 f"rRMSE ratio SARAH-3 0.25 / ERA5 : {r['rRMSE_%'].iloc[2] / r['rRMSE_%'].iloc[1]:.3f}"]
    r.to_csv(C.MODELS_DIR / "review" / "reference_comparison.csv", index=False) \
        if (C.MODELS_DIR / "review").exists() else None

    lines = [
        "ECMWF day-ahead GHI scored against two references (whole window)",
        "=" * 66,
        f"systems={df.system_id.nunique()}  rows={len(df):,}",
        "",
        r.to_string(index=False, float_format=lambda x: f"{x:.2f}"),
        "",
        f"RMSE ratio  SARAH-3 / ERA5 : {ratio_rmse:.3f}",
        f"rRMSE ratio SARAH-3 / ERA5 : {ratio_rrmse:.3f}",
    ] + extra
    text = "\n".join(lines)
    print(text)
    (C.REPORTS_DIR / "reference_comparison.txt").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
