#!/usr/bin/env python3
"""
residual_autocorr.py
====================
Residual autocorrelation of the proposed configuration (E2) at true time lags.

consolidate.py (exp2) shifts residuals by rows. The test rows are daylight
only, so a 96-row shift spans about two days, not 24 h. This script pairs each
residual with the residual of the same system exactly `lag` earlier and
reports the pooled Pearson correlation.

Input : models/revision/preds/exp4__b_base_ens_all_systems.csv.gz
        (P50 test predictions of the proposed configuration, 33 systems;
        written by consolidate.py --full-fleet --save-preds)
Output: models/revision/residual_autocorr.csv, reports/residual_autocorr.txt
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as C

PREDS = C.MODELS_DIR / "revision" / "preds" / "exp4__b_base_ens_all_systems.csv.gz"
LAGS = [("15min", 15), ("30min", 30), ("1h", 60), ("2h", 120), ("24h", 1440), ("48h", 2880)]


def main():
    d = pd.read_csv(PREDS)
    d["timestamp"] = pd.to_datetime(d.timestamp, utc=True)
    d["res"] = d.y - d.p50
    d = d[["system_id", "timestamp", "res"]]

    rows = []
    for name, minutes in LAGS:
        lagged = d.copy()
        lagged["timestamp"] = lagged.timestamp + pd.Timedelta(minutes=minutes)
        m = d.merge(lagged, on=["system_id", "timestamp"], suffixes=("", "_lag"))
        rows.append({"lag": name, "lag_min": minutes, "n_pairs": len(m),
                     "autocorr": m.res.corr(m.res_lag)})
    out = pd.DataFrame(rows)

    dest = C.MODELS_DIR / "revision" / "residual_autocorr.csv"
    out.to_csv(dest, index=False)
    rep = C.REPORTS_DIR / "residual_autocorr.txt"
    rep.write_text("Residual autocorrelation (P50, proposed configuration, test split, "
                   "pooled over systems, true time lags)\n\n" + out.to_string(index=False) + "\n",
                   encoding="utf-8")
    print(out.to_string(index=False))
    print("wrote", dest, rep)


if __name__ == "__main__":
    main()
