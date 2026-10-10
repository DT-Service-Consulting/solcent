#!/usr/bin/env python3
"""
tune_quantile.py
================
Hyperparameter search against the objective the model is actually scored on.

WHY THIS REPLACES THE INHERITED CONFIGURATION
---------------------------------------------
The parent project's search (`src/models/model_benchmark.py`) ran 20 Optuna
trials per model family and minimised validation RMSE under a squared-error
objective:

    objective: return np.sqrt(mean_squared_error(yva, preds))   # N_TRIALS = 20

Those parameters were then reused unchanged for quantile regression. That is a
mismatch in two directions. Squared error is minimised by the conditional
mean; pinball loss at level alpha is minimised by the conditional alpha-
quantile. A configuration selected to predict the mean well is not the one
that predicts the 90th percentile well - the tails are exactly where the two
objectives disagree most, and the tails are what a battery controller acts on.

Twenty trials over an eight-dimensional space is also thin.

WHAT THIS SCRIPT DOES
---------------------
One Optuna study per quantile in {0.25, 0.50, 0.90}, each minimising pinball
loss at its own level. Quantile levels are separate estimands, so they get
separate configurations rather than one compromise shared across all three.

SPLIT DISCIPLINE
----------------
Boosting needs a stopping rule, and using the same data to stop and to select
would leak. So:

    inner-train  first 85% of the training split   -> fits the trees
    inner-val    last 15% of the training split    -> early stopping only
    val          the project's validation split    -> Optuna objective
    test         untouched here                    -> never seen

Early stopping therefore never sees the Optuna objective set, and the test
split plays no part in tuning at all.

OUTPUT
------
  models/tuned/best_params_quantile.json   per-quantile parameters
  models/tuned/trials_q{25,50,90}.csv      full trial history
  models/tuned/comparison.csv              inherited vs tuned, on test
  reports/tune_quantile_report.txt

The comparison arm matters as much as the search: it answers whether correct-
objective tuning changes any conclusion, or merely tidies the methodology.

Usage:
    python v3_ensemble/src/models/tune_quantile.py              # 60 trials/quantile
    python v3_ensemble/src/models/tune_quantile.py --trials 30  # quicker
    python v3_ensemble/src/models/tune_quantile.py --evaluate-only
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
PARAMS_FILE = OUT / "best_params_quantile.json"

INNER_FRAC = 0.85          # of the training split, chronologically
EARLY_STOP = 50
MAX_TREES = 2000
SEED = 42
LOG = []


def log(m=""):
    print(m, flush=True)
    LOG.append(str(m))


# ---------------------------------------------------------------------------
def load():
    """Same preparation as consolidate.py, on the proposed configuration
    (81376 excluded) so the search tunes what the paper proposes."""
    df = pd.read_csv(C.MODEL_READY_V3)
    df = df[~df.system_id.isin(C.EXCLUDE_SYSTEMS + C.BROKEN_SYSTEMS)]
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["system_id", "timestamp"]).reset_index(drop=True)

    prev = df[["system_id", "timestamp", C.TARGET]].copy()
    prev["timestamp"] += pd.Timedelta(days=1)
    prev = prev.rename(columns={C.TARGET: "pred_persistence"})
    df = df.merge(prev, on=["system_id", "timestamp"], how="left")
    df = df[(df.is_valid == True) & (df.is_daylight == True)].copy()
    df["pred_persistence"] = df["pred_persistence"].fillna(0.0)
    return df.fillna(0)


def inner_split(tr):
    """Chronological inner split of the training data for early stopping."""
    cut = tr["timestamp"].quantile(INNER_FRAC)
    return tr[tr.timestamp <= cut], tr[tr.timestamp > cut]


def suggest(trial):
    """Search space. Ranges follow the parent study so the two are comparable,
    widened only where the parent's best value sat against a boundary
    (num_leaves 148 against a cap of 150; reg_alpha 7.81 against 10).

    The learning-rate floor is 0.01 rather than the parent's 0.001. A smoke
    test with the lower floor drove the search into a degenerate corner: it
    chose lr = 0.0073 and then consumed 1918-2000 trees, with the q90 study
    hitting the tree cap outright so early stopping never fired. At that
    learning rate the model simply has not converged inside any affordable
    budget, and the objective rewards spending the entire allowance rather
    than finding a good configuration.
    """
    return dict(
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        num_leaves=trial.suggest_int("num_leaves", 20, 255),
        max_depth=trial.suggest_int("max_depth", 4, 20),
        min_child_samples=trial.suggest_int("min_child_samples", 20, 200),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        subsample_freq=trial.suggest_int("subsample_freq", 0, 5),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 20.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
    )


def fit_one(params, alpha, Xi, yi, Xe, ye, n_estimators=MAX_TREES):
    m = lgb.LGBMRegressor(objective="quantile", alpha=alpha,
                          n_estimators=n_estimators, random_state=SEED,
                          n_jobs=-1, verbose=-1, **params)
    m.fit(Xi, yi, eval_set=[(Xe, ye)], eval_metric="quantile",
          callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
    return m


def tune_quantile(alpha, tr, va, cols, n_trials):
    """One study per quantile, scored by pinball at that quantile on val."""
    itr, iva = inner_split(tr)
    Xi, yi = itr[cols], itr[C.TARGET]
    Xe, ye = iva[cols], iva[C.TARGET]
    Xv, yv = va[cols], va[C.TARGET].to_numpy()

    def objective(trial):
        p = suggest(trial)
        m = fit_one(p, alpha, Xi, yi, Xe, ye)
        pred = np.clip(m.predict(Xv), 0, C.CLIP)
        trial.set_user_attr("best_iteration", int(m.best_iteration_ or MAX_TREES))
        return M.pinball_loss(yv, pred, alpha)

    # Seed per quantile, not globally. With one shared seed the TPE sampler is
    # deterministic and all three studies propose the same candidates, so the
    # searches are not independent - a smoke test returned byte-identical
    # parameters for P25, P50 and P90.
    sampler = optuna.samplers.TPESampler(seed=SEED + int(alpha * 100),
                                         multivariate=True)
    study = optuna.create_study(direction="minimize", sampler=sampler,
                                study_name=f"pinball_q{int(alpha*100)}")
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    mins = (time.time() - t0) / 60

    best = dict(study.best_params)
    best["n_estimators"] = int(study.best_trial.user_attrs["best_iteration"])
    log(f"  q{int(alpha*100):02d}: best pinball {study.best_value:.6f} "
        f"after {n_trials} trials ({mins:.1f} min), "
        f"n_estimators={best['n_estimators']}, lr={best['learning_rate']:.4f}")
    if best["n_estimators"] >= MAX_TREES:
        log(f"       WARNING: tree cap ({MAX_TREES}) reached - early stopping "
            "never fired, so this configuration is budget-limited rather than "
            "converged. Treat the result with caution.")

    study.trials_dataframe().to_csv(OUT / f"trials_q{int(alpha*100)}.csv", index=False)
    return best, study.best_value


# ---------------------------------------------------------------------------
def evaluate(df, cols, tuned):
    """Inherited vs tuned on the untouched test split, with and without the
    conformal recalibration of Section 4.7."""
    tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
    y = te[C.TARGET].to_numpy()
    rows = []

    def run(label, params_for):
        P, Pv = {}, {}
        for q in C.QUANTILES:
            p = dict(params_for(q))
            n = p.pop("n_estimators", 500)
            m = lgb.LGBMRegressor(objective="quantile", alpha=q, n_estimators=n,
                                  random_state=SEED, n_jobs=-1, verbose=-1, **p)
            m.fit(tr[cols], tr[C.TARGET])
            P[q] = np.clip(m.predict(te[cols]), 0, C.CLIP)
            Pv[q] = np.clip(m.predict(va[cols]), 0, C.CLIP)
        yv = va[C.TARGET].to_numpy()
        d = {q: float(np.quantile(yv - Pv[q], q)) for q in C.QUANTILES}
        Pc = {q: np.clip(P[q] + d[q], 0, C.CLIP) for q in C.QUANTILES}

        for tag, pred in (("raw", P), ("recalibrated", Pc)):
            rows.append({
                "config": label, "variant": tag,
                "MAE": M.mae(y, pred[0.50]), "RMSE": M.rmse(y, pred[0.50]),
                "R2": M.r2(y, pred[0.50]),
                "skill": M.skill_score(y, pred[0.50], te.pred_persistence.to_numpy()),
                "mean_pinball": M.mean_pinball(y, pred),
                "pinball_p25": M.pinball_loss(y, pred[0.25], 0.25),
                "pinball_p50": M.pinball_loss(y, pred[0.50], 0.50),
                "pinball_p90": M.pinball_loss(y, pred[0.90], 0.90),
                "cov_p90": float(np.mean(y <= pred[0.90])),
                "P25_P90_coverage": M.coverage(y, pred[0.25], pred[0.90]),
            })
        log(f"  {label:22s} raw pinball {rows[-2]['mean_pinball']:.5f} | "
            f"recal {rows[-1]['mean_pinball']:.5f} | "
            f"R2 {rows[-2]['R2']:.4f} | skill {rows[-2]['skill']:.4f}")

    inherited = {k: v for k, v in C.LGBM_PARAMS.items() if k != "verbose"}
    run("inherited (RMSE-tuned)", lambda q: inherited)
    run("tuned (pinball)", lambda q: tuned[str(q)])
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--evaluate-only", action="store_true")
    a = ap.parse_args()

    df = load()
    cols = [c for c in C.BASE_FEATURES + C.ENSEMBLE_FEATURES if c in df.columns]
    tr, va = df[df.split == "train"], df[df.split == "val"]
    log(f"rows train={len(tr):,} val={len(va):,} test={(df.split=='test').sum():,}")
    log(f"features ({len(cols)}): {cols}")
    log(f"systems={df.system_id.nunique()} (12293 and 81376 excluded)\n")

    if a.evaluate_only and PARAMS_FILE.exists():
        tuned = json.load(open(PARAMS_FILE))
        log("loaded existing parameters")
    else:
        log(f"TUNING - {a.trials} trials per quantile, pinball objective")
        tuned = {}
        for q in C.QUANTILES:
            best, _ = tune_quantile(q, tr, va, cols, a.trials)
            tuned[str(q)] = best
        json.dump(tuned, open(PARAMS_FILE, "w"), indent=2)
        log(f"\nwrote {PARAMS_FILE}")

    log("\nPER-QUANTILE PARAMETERS")
    for q, p in tuned.items():
        log(f"  q={q}: " + ", ".join(f"{k}={v:.4g}" if isinstance(v, float)
                                     else f"{k}={v}" for k, v in p.items()))

    log("\nEVALUATION ON TEST (never used in tuning)")
    res = evaluate(df, cols, tuned)
    res.to_csv(OUT / "comparison.csv", index=False)

    log("\n" + "=" * 78)
    log("INHERITED vs TUNED")
    log("=" * 78)
    log(res.to_string(index=False))

    raw = res[res.variant == "raw"].set_index("config")
    if len(raw) == 2:
        i, t = raw.loc["inherited (RMSE-tuned)"], raw.loc["tuned (pinball)"]
        log(f"\npinball  {i.mean_pinball:.5f} -> {t.mean_pinball:.5f} "
            f"({100*(i.mean_pinball-t.mean_pinball)/i.mean_pinball:+.2f}%)")
        log(f"R2       {i.R2:.4f} -> {t.R2:.4f}  ({t.R2-i.R2:+.4f})")
        log(f"skill    {i.skill:.4f} -> {t.skill:.4f}  ({t.skill-i.skill:+.4f})")
        log(f"P90 cov  {i.cov_p90:.4f} -> {t.cov_p90:.4f}  (nominal 0.90)")

    with open(C.REPORTS_DIR / "tune_quantile_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(LOG))
    log(f"\nWrote {C.REPORTS_DIR / 'tune_quantile_report.txt'}")


if __name__ == "__main__":
    main()
