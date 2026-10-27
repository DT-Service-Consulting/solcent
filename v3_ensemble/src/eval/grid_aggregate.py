#!/usr/bin/env python3
"""
grid_aggregate.py   [EPSR-SCOPE]
================================
What a grid-side user sees. An aggregator or distribution system operator
does not act on one rooftop's forecast but on the sum over many. This script
aggregates the per-rooftop day-ahead P50 forecasts (in kW) and compares:

  individual   error of each rooftop's own forecast
  regional     sum over the rooftops of each of the five regions
  fleet        sum over all rooftops

Each level is scored by normalised RMSE (RMSE / installed capacity of the
reporting rooftops), skill against the aggregated persistence forecast, and
daily energy imbalance |sum over the day of (actual - forecast)| * 0.25 h,
as a share of the day's energy.

Aggregation is done per timestamp over the rooftops present in both actual
and forecast, so missing data never enters one side only.

Written for the Electric Power Systems Research submission (power-system
relevance). Optional for other journals; see paper/epsr/JOURNAL_NOTES.md.

Input:  models/revision/preds/exp4__b_base_ens_all_systems.csv.gz
        (consolidate.py --full-fleet --save-preds)
Output: models/revision/grid_aggregate.csv
        reports/grid_aggregate.txt
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as C

PRED = C.MODELS_DIR / "revision" / "preds" / "exp4__b_base_ens_all_systems.csv.gz"
DT_H = 0.25


def score(g, cap_col="cap_kw"):
    """g: frame with actual_kw, fc_kw, pers_kw, cap_kw, day per timestamp."""
    e = g.actual_kw - g.fc_kw
    e_ref = g.actual_kw - g.pers_kw
    rmse = np.sqrt((e ** 2).mean())
    rmse_ref = np.sqrt((e_ref ** 2).mean())
    daily = g.assign(e=e).groupby("day").agg(imb=("e", "sum"), energy=("actual_kw", "sum"))
    return {
        "nRMSE_%cap": 100 * rmse / g[cap_col].mean(),
        "skill": 1 - rmse / rmse_ref,
        "daily_imbalance_%energy": 100 * (daily.imb.abs() * DT_H).sum() / (daily.energy * DT_H).sum(),
        "mean_daily_imbalance_kWh": float((daily.imb.abs() * DT_H).mean()),
    }


def main():
    d = pd.read_csv(PRED)
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
    meta = pd.read_csv(C.PARENT_META, usecols=["system_id", "region", "capacity_kwp"])
    d = d.merge(meta, on="system_id", how="left")
    d["cap_kw"] = d.capacity_kwp
    d["actual_kw"] = d.y * d.cap_kw
    d["fc_kw"] = d.p50 * d.cap_kw
    d["pers_kw"] = d.persistence * d.cap_kw
    d["day"] = d.timestamp.dt.normalize()

    rows = []
    ind = [score(g) | {"system_id": sid} for sid, g in d.groupby("system_id")]
    ind = pd.DataFrame(ind)
    rows.append({"level": "individual rooftop (median of 33)", "n_units": len(ind),
                 **ind.drop(columns="system_id").median().to_dict()})

    agg_cols = ["actual_kw", "fc_kw", "pers_kw", "cap_kw"]
    for region, g in d.groupby("region"):
        a = g.groupby("timestamp")[agg_cols].sum().reset_index()
        a["day"] = a.timestamp.dt.normalize()
        rows.append({"level": f"region: {region}", "n_units": g.system_id.nunique(), **score(a)})

    a = d.groupby("timestamp")[agg_cols].sum().reset_index()
    a["day"] = a.timestamp.dt.normalize()
    rows.append({"level": "fleet (all rooftops)", "n_units": d.system_id.nunique(), **score(a)})

    r = pd.DataFrame(rows)
    out = C.MODELS_DIR / "revision"
    out.mkdir(parents=True, exist_ok=True)
    r.to_csv(out / "grid_aggregate.csv", index=False)
    text = ("Aggregated day-ahead forecast error  [EPSR-SCOPE]\n" + "=" * 60 + "\n"
            + "model: base + multi-model mean, all 33 systems, P50, test split\n\n"
            + r.to_string(index=False, float_format=lambda x: f"{x:.3f}") + "\n")
    (C.REPORTS_DIR / "grid_aggregate.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
