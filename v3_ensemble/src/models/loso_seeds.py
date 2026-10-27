#!/usr/bin/env python3
"""
loso_seeds.py
=============
Leave-one-system-out zero-shot evaluation repeated over several random seeds,
for the three hyperparameter configurations of manuscript Table 4:

    preliminary   config.LGBM_PARAMS     (RMSE-selected, squared-error search)
    temporal      config.TUNED_PARAMS[0.5] (pinball, chronological split)
    transfer      config.TRANSFER_PARAMS   (pinball, held-out systems)

LightGBM's row and column subsampling depend on the seed. A single LOSO run
therefore carries seed noise (about 0.005 in median R2 between the two
single-seed runs in the repository). Repeating the run gives each summary
statistic a mean and a spread across seeds, and the per-system results allow
paired tests between configurations (src/eval/revision_stats.py --loso).

Same protocol as consolidate.py exp5: P50 only; train on the other 32 systems
over train+val; score the held-out system's test window.

Output: models/revision/loso_seeds_per_system.csv
        models/revision/loso_seeds_summary.csv
        reports/loso_seeds.txt

Usage:  python loso_seeds.py --configs transfer,temporal,preliminary --seeds 0,1,2,3,4
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as C

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval import metrics as M

OUT = C.MODELS_DIR / "revision"
CONFIGS = {
    "preliminary": dict(C.LGBM_PARAMS),
    "temporal": dict(C.TUNED_PARAMS[0.5]),
    "transfer": dict(C.TRANSFER_PARAMS),
}


def load():
    df = pd.read_csv(C.MODEL_READY_V3)
    df = df[~df.system_id.isin(C.EXCLUDE_SYSTEMS)]
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["system_id", "timestamp"]).reset_index(drop=True)
    prev = df[["system_id", "timestamp", C.TARGET]].copy()
    prev["timestamp"] += pd.Timedelta(days=1)
    prev = prev.rename(columns={C.TARGET: "pred_persistence"})
    df = df.merge(prev, on=["system_id", "timestamp"], how="left")
    df = df[(df.is_valid == True) & (df.is_daylight == True)].copy()
    df["pred_persistence"] = df["pred_persistence"].fillna(0.0)
    return df.fillna(0)


def run(df, cols, name, params, seed):
    rows = []
    for sid in sorted(df.system_id.unique()):
        tr = df[(df.system_id != sid) & (df.split.isin(["train", "val"]))]
        te = df[(df.system_id == sid) & (df.split == "test")]
        if len(te) < 500:
            continue
        m = lgb.LGBMRegressor(objective="quantile", alpha=0.50, random_state=seed,
                              n_jobs=-1, **params).fit(tr[cols], tr[C.TARGET])
        p = np.clip(m.predict(te[cols]), 0, C.CLIP)
        y = te[C.TARGET].to_numpy()
        rows.append({"config": name, "seed": seed, "system_id": sid, "n": len(te),
                     "R2": M.r2(y, p), "RMSE": M.rmse(y, p), "MBE": M.mbe(y, p),
                     "skill": M.skill_score(y, p, te.pred_persistence.to_numpy())})
    return pd.DataFrame(rows)


def summarise(r):
    g = r.groupby(["config", "seed"])
    per_seed = pd.DataFrame({
        "median_R2": g.R2.median(), "mean_R2": g.R2.mean(),
        "p10_R2": g.R2.quantile(0.10), "worst_R2": g.R2.min(),
        "median_skill": g.skill.median(),
        "failures": g.skill.apply(lambda s: int((s <= 0).sum())),
    }).reset_index()
    agg = per_seed.drop(columns="seed").groupby("config").agg(["mean", "std", "min", "max"])
    return per_seed, agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="transfer,temporal,preliminary")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    a = ap.parse_args()
    configs = a.configs.split(",")
    seeds = [int(x) for x in a.seeds.split(",")]

    OUT.mkdir(parents=True, exist_ok=True)
    df = load()
    cols = [c for c in C.BASE_FEATURES + C.ENSEMBLE_FEATURES if c in df.columns]
    print(f"rows={len(df):,} systems={df.system_id.nunique()} features={len(cols)}", flush=True)

    out_file = OUT / "loso_seeds_per_system.csv"
    done = pd.read_csv(out_file) if out_file.exists() else pd.DataFrame()
    parts = [done] if len(done) else []
    for name in configs:
        for seed in seeds:
            if len(done) and ((done.config == name) & (done.seed == seed)).any():
                print(f"{name} seed={seed}: cached", flush=True)
                continue
            t0 = time.time()
            r = run(df, cols, name, CONFIGS[name], seed)
            parts.append(r)
            pd.concat(parts, ignore_index=True).to_csv(out_file, index=False)
            print(f"{name} seed={seed}: median R2={r.R2.median():.4f} mean={r.R2.mean():.4f} "
                  f"worst={r.R2.min():.4f} fails={int((r.skill <= 0).sum())} "
                  f"({(time.time()-t0)/60:.1f} min)", flush=True)

    allr = pd.concat(parts, ignore_index=True)
    per_seed, agg = summarise(allr)
    per_seed.to_csv(OUT / "loso_seeds_summary.csv", index=False)
    text = ("LOSO zero-shot over seeds (P50, 33 held-out systems)\n" + "=" * 60 + "\n\n"
            + per_seed.to_string(index=False, float_format=lambda x: f"{x:.4f}") + "\n\n"
            + "Across seeds (mean, std, min, max)\n"
            + agg.to_string(float_format=lambda x: f"{x:.4f}") + "\n")
    (C.REPORTS_DIR / "loso_seeds.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
