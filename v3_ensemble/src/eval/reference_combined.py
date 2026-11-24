#!/usr/bin/env python3
"""
reference_combined.py
=====================
Skill against a combined climatology-persistence reference. Murphy (1992,
Weather and Forecasting 7, 692-698) showed that the optimal linear combination of
climatology and persistence is more accurate than either alone, so skill against
it is lower than skill against either.

  persistence  : same 15-min slot one day earlier (as in the main results)
  climatology  : per rooftop, mean normalised output by calendar month and
                 UTC time of day, from the training split
                 (fallback: per rooftop, time of day only)
  combined     : a * climatology + (1 - a) * persistence, with one weight a in
                 [0, 1] that minimises squared error on the validation split
                 (closed form); the test split is never used to fit it

The weight here is the empirical MSE-optimal one, fitted on validation.

Scores the P50 forecast of the proposed configuration (chronological split,
models/revision/preds/exp4__b_base_ens_all_systems.csv.gz) and the zero-shot
P50 forecasts (models/consolidated_full/exp5_loso_per_system.csv) against all
three references.

Output: reports/reference_combined.txt, models/review/reference_combined.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
import config as C
import consolidate as CO


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def main():
    df = CO.load_pv()
    df["tod"] = df.timestamp.dt.hour * 60 + df.timestamp.dt.minute
    df["month"] = df.timestamp.dt.month
    tr = df[df.split == "train"]
    clim_m = tr.groupby(["system_id", "month", "tod"])[C.TARGET].mean().rename("clim_m")
    clim_t = tr.groupby(["system_id", "tod"])[C.TARGET].mean().rename("clim_t")
    df = df.join(clim_m, on=["system_id", "month", "tod"]).join(clim_t, on=["system_id", "tod"])
    df["clim"] = df.clim_m.fillna(df.clim_t).fillna(0.0)

    # one weight on climatology, MSE-optimal on validation (closed form, clipped)
    va = df[df.split == "val"]
    y, p, c = va[C.TARGET].to_numpy(), va.pred_persistence.to_numpy(), va.clim.to_numpy()
    a = float(np.clip(np.sum((y - p) * (c - p)) / np.sum((c - p) ** 2), 0, 1))
    df["combined"] = a * df.clim + (1 - a) * df.pred_persistence

    te = df[df.split == "test"][["system_id", "timestamp", C.TARGET, "pred_persistence", "clim", "combined"]]
    preds = pd.read_csv(C.MODELS_DIR / "revision" / "preds" / "exp4__b_base_ens_all_systems.csv.gz",
                        parse_dates=["timestamp"])
    m = preds.merge(te, on=["system_id", "timestamp"], how="inner")
    assert np.allclose(m.y, m[C.TARGET]), "target mismatch"
    assert len(m) == len(preds), (len(m), len(preds))

    refs = {"persistence": "pred_persistence", "climatology": "clim", "combined": "combined"}
    rows = []
    r_model = rmse(m.y, m.p50)
    for name, col in refs.items():
        per = m.groupby("system_id").apply(
            lambda g: 1 - rmse(g.y, g.p50) / rmse(g.y, g[col]), include_groups=False)
        rows.append({"setting": "seen (chronological split, proposed P50)", "reference": name,
                     "ref_RMSE": rmse(m.y, m[col]), "model_RMSE": r_model,
                     "pooled_skill": 1 - r_model / rmse(m.y, m[col]),
                     "median_rooftop_skill": float(per.median())})

    # zero-shot: per-rooftop RMSE of the LOSO P50 on the same test rows
    z = pd.read_csv(C.MODELS_DIR / "consolidated_full" / "exp5_loso_per_system.csv").set_index("system_id")
    for name, col in refs.items():
        ref_rmse = te.groupby("system_id").apply(lambda g: rmse(g[C.TARGET], g[col]), include_groups=False)
        sk = 1 - z.RMSE / ref_rmse.reindex(z.index)
        rows.append({"setting": "unseen (leave-one-system-out P50)", "reference": name,
                     "ref_RMSE": float(np.sqrt((ref_rmse ** 2).mean())), "model_RMSE": np.nan,
                     "pooled_skill": np.nan, "median_rooftop_skill": float(sk.median()),
                     "failures": int((sk <= 0).sum())})

    out = pd.DataFrame(rows)
    out_dir = C.MODELS_DIR / "review"
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "reference_combined.csv", index=False)
    text = ("Skill against persistence, climatology and their combination (Murphy, 1992)\n"
            f"weight on climatology a = {a:.3f} (MSE-optimal on validation)\n"
            f"test rows = {len(m):,}, systems = {m.system_id.nunique()}\n\n"
            + out.round(4).to_string(index=False) + "\n")
    print(text)
    (C.REPORTS_DIR / "reference_combined.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
