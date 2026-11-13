#!/usr/bin/env python3
"""
reference_pv_corr.py
====================
Which irradiance reference tracks measured rooftop output better?

Hourly mean normalised PV output (15-min slots starting in [t-1h, t), labelled t,
the Open-Meteo preceding-hour convention) is correlated with SARAH-3 and ERA5
GHI at the same rooftop, at time shifts of -2..+2 h. The zero shift should give
the highest correlation for both references (timing check); the reference with
the higher correlation at zero shift is the closer description of the sky the
rooftops saw.

Output: reports/reference_pv_corr.txt
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as C


def load(name, col):
    d = pd.read_csv(C.DATA_INTERIM / name)
    d["t"] = pd.to_datetime(d["time"], utc=True)
    return d[["system_id", "t", col]].dropna()


def main():
    sat = load("satellite_ghi_all.csv", "sat_ghi")
    era = load("era5_ghi_all.csv", "era5_ghi")
    pv = pd.read_csv(C.MODEL_READY_V3, usecols=["system_id", "timestamp", C.TARGET])
    pv["t"] = pd.to_datetime(pv.timestamp, utc=True).dt.floor("h") + pd.Timedelta(hours=1)
    ph = pv.groupby(["system_id", "t"])[C.TARGET].mean().reset_index()

    lines = [f"Hourly PV vs reference GHI, systems={ph.system_id.nunique()}", ""]
    for k in (-2, -1, 0, 1, 2):
        row = [f"shift {k:+d} h:"]
        for ref, col in ((sat, "sat_ghi"), (era, "era5_ghi")):
            r = ref.assign(t=ref.t + pd.Timedelta(hours=k)).merge(ph, on=["system_id", "t"])
            row.append(f"corr(PV, {col})={np.corrcoef(r[col], r[C.TARGET])[0, 1]:.4f} (n={len(r):,})")
        lines.append("  ".join(row))
    text = "\n".join(lines) + "\n"
    print(text)
    (C.REPORTS_DIR / "reference_pv_corr.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
