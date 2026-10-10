#!/usr/bin/env python3
"""
tune_transfer.py
================
Hyperparameter search against TRANSFER to unseen rooftops.

WHY A SECOND SEARCH
-------------------
tune_quantile.py optimised pinball loss on the validation split of the
chronological arrangement, in which every system appears in training. That
selects for predicting known rooftops on future dates. It worked:

    temporal split    inherited 0.8087  ->  tuned 0.8114   (+0.0027 R2)

and it made deployment worse:

    LOSO, unseen roof inherited 0.7232  ->  tuned 0.7062   (-0.0170 R2)
                      skill     0.3341  ->        0.3077   (-0.0264)

The tuned configuration is deeper (15-20), wider (148-247 leaves) and far
longer (1418-1999 trees against 500). That extra capacity buys roof-specific
memorisation, which is worth something when the roof is in the training set
and worth nothing when it is not. Selection pressure was pointed at the wrong
quantity.

WHAT THIS SCRIPT DOES DIFFERENTLY
---------------------------------
The inner loop mirrors deployment. Instead of holding out future dates for
known systems, it holds out whole SYSTEMS:

    systems in the train+val period  ->  K folds, split BY SYSTEM
    each trial : fit on K-1 folds of systems
                 predict the held-out fold's systems
    objective  : pinball loss pooled over held-out systems
    test split : never touched

A configuration scores well here only if it generalises across rooftops, which
is the property the deployment case needs.

SCOPE AND COST
--------------
Each trial costs K fits rather than one. To keep the search affordable it
tunes P50 only - the quantile LOSO reports - and subsamples rows during the
search, refitting the winner on the full data for evaluation. Tuning all three
quantiles this way would cost roughly three times as much and can follow if
the P50 result justifies it.

EXPECTATION, STATED IN ADVANCE
------------------------------
We expect the search to move toward stronger regularisation than the temporal
optimum: fewer leaves, shallower trees, larger min_child_samples, fewer
boosting rounds. Recording the expectation before running means a null result
stays interpretable.

OUTPUT
------
  models/tuned/best_params_transfer.json
  models/tuned/trials_transfer.csv
  models/tuned/transfer_comparison.csv     three configurations, full LOSO
  reports/tune_transfer_report.txt

Usage:
    python v3_ensemble/src/models/tune_transfer.py
    python v3_ensemble/src/models/tune_transfer.py --trials 25 --folds 3
    python v3_ensemble/src/models/tune_transfer.py --evaluate-only
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as C
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval import metrics as M

import lightgbm as lgb
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

OUT = C.MODELS_DIR / "tuned"
OUT.mkdir(parents=True, exist_ok=True)
PARAMS_FILE = OUT / "best_params_transfer.json"

ALPHA = 0.50           # LOSO reports P50
SEED = 42
SEARCH_FRAC = 0.45     # row subsample during the search only
MAX_TREES = 1500
EARLY_STOP = 40
LOG = []


def log(m=""):
    print(m, flush=True)
    LOG.append(str(m))


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
    df["day"] = df["timestamp"].dt.normalize()
    return df.fillna(0)


def system_folds(systems, k, seed=SEED):
    """Partition systems into k groups. Folds hold out whole rooftops, which
    is what makes the objective a transfer objective."""
    rng = np.random.default_rng(seed)
    s = np.array(sorted(systems))
    rng.shuffle(s)
    return [list(x) for x in np.array_split(s, k)]


def suggest(trial):
    """Ranges deliberately reach further into the regularised corner than the
    temporal search did: fewer leaves, shallower trees, and larger leaf
    minima are all reachable, because that is where transfer is expected to
    live."""
    return dict(
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        num_leaves=trial.suggest_int("num_leaves", 8, 200),
        max_depth=trial.suggest_int("max_depth", 3, 16),
        min_child_samples=trial.suggest_int("min_child_samples", 20, 800, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        subsample_freq=trial.suggest_int("subsample_freq", 0, 5),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 50.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 50.0, log=True),
    )


def tune(df, cols, folds, n_trials):
    pool = df[df.split.isin(["train", "val"])]
    if SEARCH_FRAC < 1.0:
        pool = pool.sample(frac=SEARCH_FRAC, random_state=SEED)
    log(f"search pool: {len(pool):,} rows "
        f"({SEARCH_FRAC:.0%} subsample of train+val)")
    log(f"system folds: {[len(f) for f in folds]} systems per fold\n")

    def objective(trial):
        p = suggest(trial)
        losses, n_rows = [], 0
        for hold in folds:
            tr = pool[~pool.system_id.isin(hold)]
            te = pool[pool.system_id.isin(hold)]
            m = lgb.LGBMRegressor(objective="quantile", alpha=ALPHA,
                                  n_estimators=MAX_TREES, random_state=SEED,
                                  n_jobs=-1, verbose=-1, **p)
            # early stopping on a time-tail of the training systems, so it
            # never sees the held-out rooftops
            cut = tr["timestamp"].quantile(0.88)
            fit, stop = tr[tr.timestamp <= cut], tr[tr.timestamp > cut]
            m.fit(fit[cols], fit[C.TARGET],
                  eval_set=[(stop[cols], stop[C.TARGET])],
                  eval_metric="quantile",
                  callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
            pred = np.clip(m.predict(te[cols]), 0, C.CLIP)
            losses.append(M.pinball_loss(te[C.TARGET].to_numpy(), pred, ALPHA)
                          * len(te))
            n_rows += len(te)
            trial.set_user_attr("last_iter", int(m.best_iteration_ or MAX_TREES))
        return float(np.sum(losses) / n_rows)

    sampler = optuna.samplers.TPESampler(seed=SEED, multivariate=True)
    study = optuna.create_study(direction="minimize", sampler=sampler,
                                study_name="transfer_pinball_p50")
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    log(f"search done: {n_trials} trials in {(time.time()-t0)/60:.1f} min")
    log(f"best held-out-system pinball: {study.best_value:.6f}")

    best = dict(study.best_params)
    best["n_estimators"] = int(study.best_trial.user_attrs.get("last_iter", 500))
    study.trials_dataframe().to_csv(OUT / "trials_transfer.csv", index=False)
    return best


def full_loso(df, cols, params, label):
    """Complete leave-one-system-out at the reported settings."""
    systems = sorted(df.system_id.unique())
    rows = []
    for sid in systems:
        tr = df[(df.system_id != sid) & (df.split.isin(["train", "val"]))]
        te = df[(df.system_id == sid) & (df.split == "test")]
        if len(te) < 500:
            continue
        m = lgb.LGBMRegressor(objective="quantile", alpha=ALPHA,
                              random_state=SEED, n_jobs=-1, verbose=-1,
                              **params).fit(tr[cols], tr[C.TARGET])
        pred = np.clip(m.predict(te[cols]), 0, C.CLIP)
        y = te[C.TARGET].to_numpy()
        rows.append({"config": label, "system_id": sid,
                     "R2": M.r2(y, pred), "RMSE": M.rmse(y, pred),
                     "MBE": M.mbe(y, pred),
                     "skill": M.skill_score(y, pred,
                                            te.pred_persistence.to_numpy())})
    r = pd.DataFrame(rows)
    log(f"  {label:34s} median R2={r.R2.median():.4f}  "
        f"skill={r.skill.median():.4f}  mean R2={r.R2.mean():.4f}  "
        f"worse-than-persistence={int((r.skill <= 0).sum())}/{len(r)}")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--evaluate-only", action="store_true")
    a = ap.parse_args()

    df = load()
    cols = [c for c in C.BASE_FEATURES + C.ENSEMBLE_FEATURES if c in df.columns]
    systems = sorted(df.system_id.unique())
    log(f"TRANSFER TUNING  {time.strftime('%Y-%m-%d %H:%M')}")
    log(f"rows={len(df):,} systems={len(systems)} features={len(cols)}")
    log(f"objective: pinball(P50) pooled over HELD-OUT SYSTEMS\n")

    if a.evaluate_only and PARAMS_FILE.exists():
        best = json.load(open(PARAMS_FILE))
        log("loaded existing transfer parameters")
    else:
        folds = system_folds(systems, a.folds)
        best = tune(df, cols, folds, a.trials)
        json.dump(best, open(PARAMS_FILE, "w"), indent=2)
        log(f"wrote {PARAMS_FILE}")

    log("\nTRANSFER-TUNED PARAMETERS")
    log("  " + ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                         for k, v in best.items()))

    log("\nFULL LOSO COMPARISON (test split, 33 folds)")
    inherited = {k: v for k, v in C.LGBM_PARAMS.items() if k != "verbose"}
    temporal = {k: v for k, v in C.TUNED_PARAMS[0.50].items() if k != "verbose"}
    allr = pd.concat([
        full_loso(df, cols, inherited, "inherited (RMSE, temporal)"),
        full_loso(df, cols, temporal, "tuned on temporal split"),
        full_loso(df, cols, best, "tuned on held-out systems"),
    ], ignore_index=True)
    allr.to_csv(OUT / "transfer_comparison.csv", index=False)

    log("\n" + "=" * 78)
    log("SUMMARY - median across 33 unseen rooftops")
    log("=" * 78)
    piv = allr.groupby("config").agg(
        median_R2=("R2", "median"), mean_R2=("R2", "mean"),
        p10_R2=("R2", lambda s: s.quantile(0.10)),
        min_R2=("R2", "min"), median_skill=("skill", "median"),
        fails=("skill", lambda s: int((s <= 0).sum())))
    log(piv.round(4).to_string())

    with open(C.REPORTS_DIR / "tune_transfer_report.txt", "w",
              encoding="utf-8") as f:
        f.write("\n".join(LOG))
    log(f"\nWrote {C.REPORTS_DIR / 'tune_transfer_report.txt'}")


if __name__ == "__main__":
    main()
