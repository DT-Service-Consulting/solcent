#!/usr/bin/env python3
"""
error_distance.py
=================
Is the forecast error shared between rooftops? Equal RMSE at two rooftops does
not show that their errors are the same (supervisor review), so this script
correlates the errors themselves and plots the correlation against distance.

  irradiance : hourly day-ahead ECMWF GHI minus SARAH-3 (features/mos_table.py),
               test split, sun elevation > 5 deg
  PV         : P50 residuals of the proposed configuration on the test split
               (models/revision/preds/exp4__b_base_ens_all_systems.csv.gz)

For every pair of systems: Pearson correlation of the two error series over
common times, and the great-circle distance. Results binned by distance.

Outputs: models/review/error_distance_pairs.csv, error_distance_binned.csv,
         reports/review_error_distance.txt
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as C
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "features"))
import mos_table

OUT = C.MODELS_DIR / "review"
OUT.mkdir(parents=True, exist_ok=True)
BINS = [0, 10, 25, 50, 100, 150, 250]


def pair_table(wide, coords):
    ids = list(wide.columns)
    corr = wide.corr(min_periods=200)
    rows = []
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            la, lo = np.radians(coords.loc[a]), np.radians(coords.loc[b])
            h = (np.sin((lo.latitude - la.latitude) / 2) ** 2
                 + np.cos(la.latitude) * np.cos(lo.latitude) * np.sin((lo.longitude - la.longitude) / 2) ** 2)
            rows.append({"a": a, "b": b, "km": 6371 * 2 * np.arcsin(np.sqrt(h)), "corr": corr.loc[a, b]})
    return pd.DataFrame(rows)


def main():
    coords = (pd.read_csv(C.PARENT_META).drop_duplicates("system_id")
              .set_index("system_id")[["latitude", "longitude"]])
    lines = []
    out = []
    d = mos_table.build()
    d = d[d.split == "test"]
    d["err"] = d.ecmwf_ghi - d.sat_ghi
    irr = pair_table(d.pivot_table(index="timestamp", columns="system_id", values="err"), coords)
    irr["kind"] = "irradiance (ECMWF - SARAH-3, hourly)"
    out.append(irr)

    p = pd.read_csv(C.MODELS_DIR / "revision" / "preds" / "exp4__b_base_ens_all_systems.csv.gz")
    p["res"] = p.y - p.p50
    pv = pair_table(p.pivot_table(index="timestamp", columns="system_id", values="res"), coords)
    pv["kind"] = "PV P50 residual (15 min)"
    out.append(pv)

    pairs = pd.concat(out, ignore_index=True)
    pairs.to_csv(OUT / "error_distance_pairs.csv", index=False)
    pairs["bin"] = pd.cut(pairs.km, BINS)
    binned = (pairs.groupby(["kind", "bin"], observed=True)["corr"]
              .agg(["count", "median", "mean"]).reset_index())
    binned["bin"] = binned["bin"].astype(str)
    binned.to_csv(OUT / "error_distance_binned.csv", index=False)
    lines.append("Error correlation between rooftops against distance (test split)")
    lines.append(binned.round(3).to_string(index=False))
    for k, g in pairs.groupby("kind"):
        lines.append(f"{k}: Spearman(corr, km) = {g['corr'].corr(g.km, method='spearman'):+.3f}, "
                     f"n pairs = {len(g)}")
    text = "\n".join(lines)
    print(text)
    (C.REPORTS_DIR / "review_error_distance.txt").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
