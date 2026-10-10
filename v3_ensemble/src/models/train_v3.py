#!/usr/bin/env python3
"""
train_v3.py
===========
Trains the multi-quantile LightGBM on the v3 ensemble features and evaluates
it with the full probabilistic suite in src/eval/metrics.py.

ARMS COMPARED
-------------
  A  base                     - the parent's 12 pruned features. Anchors
                                every other arm.
  B  base + ens_mean_fresh    - adds ONE feature: the mean of the six day-1
                                forecasts. The full 10-feature ensemble block
                                was measured and rejected; see the table in
                                config.ENSEMBLE_FEATURES.
  D  base only, 81376 dropped - isolates the data fix from the feature.
  C  B, 81376 dropped         - both together. Best measured configuration.

D exists because the first run confounded the two changes: arm C beat arm A,
but running D showed the entire gain was the dropped system, not the
features. Any arm that changes two things at once needs its isolating twin.

Every arm is scored on RMSE/R2 (for continuity with the parent), and on
skill vs persistence, mean pinball, P25-P90 coverage and interval score,
reliability, and an R2 breakdown by cloud regime. Read the pinball and the
regime table, not the pooled R2 - see the docstring in src/eval/metrics.py.

Each arm is additionally reported AFTER post-hoc quantile recalibration
fitted on the val split, because the raw LightGBM quantiles under-cover at
P90 (empirical 0.85-0.87 against a nominal 0.90).

Outputs: v3_ensemble/models/v3_metrics.csv
         v3_ensemble/models/v3_predictions_test.csv
         v3_ensemble/models/v3_feature_importance.csv
         v3_ensemble/reports/v3_report.txt
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as C
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval import metrics as M

import lightgbm as lgb


class MultiQuantileLGBM:
    """Same shape as the parent's wrapper in vendor/model_baselines.py: one
    LightGBM per quantile, so P25/P50/P90 come out of a single fit() call."""

    def __init__(self, quantiles=None, **params):
        self.quantiles = quantiles or C.QUANTILES
        self.params = params
        self.models = {}

    def fit(self, X, y):
        for q in self.quantiles:
            m = lgb.LGBMRegressor(objective="quantile", alpha=q, **self.params)
            m.fit(X, y)
            self.models[q] = m
        return self

    def predict(self, X):
        return {q: np.clip(m.predict(X), 0, C.CLIP) for q, m in self.models.items()}


def load(drop_broken=False):
    path = C.MODEL_READY_V3 if C.MODEL_READY_V3.exists() else C.PARENT_MODEL_READY
    if path is C.PARENT_MODEL_READY:
        print(f"[warn] {C.MODEL_READY_V3.name} not found - falling back to the "
              f"parent table. Ensemble arms will be skipped.")
    print(f"Loading {path.name} ...", flush=True)
    df = pd.read_csv(path)
    df = df[~df.system_id.isin(C.EXCLUDE_SYSTEMS)]
    if drop_broken:
        df = df[~df.system_id.isin(C.BROKEN_SYSTEMS)]
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["system_id", "timestamp"]).reset_index(drop=True)

    # persistence reference: the same 15-min slot, one day earlier
    prev = df[["system_id", "timestamp", C.TARGET]].copy()
    prev["timestamp"] += pd.Timedelta(days=1)
    prev = prev.rename(columns={C.TARGET: "pred_persistence"})
    df = df.merge(prev, on=["system_id", "timestamp"], how="left")

    df = df[(df.is_valid == True) & (df.is_daylight == True)].copy()
    df["pred_persistence"] = df["pred_persistence"].fillna(0.0)
    return df.fillna(0)


def recalibrate(model, va, te, cols):
    """Post-hoc quantile recalibration, fitted on the VAL split only.

    LightGBM quantile regression comes out systematically under-covering at
    the tails: measured empirical coverage at the nominal 0.90 level was
    0.85-0.87 across every arm, i.e. the P90 is too aggressive and a battery
    controller trusting it would be over-confident.

    The fix is a conformal-style additive offset. On val, take the empirical
    q-quantile of the residual (y - pred_q) and add it to the test prediction:

        delta_q = quantile_q(y_val - pred_val_q)
        pred_test_q  ->  pred_test_q + delta_q

    This makes val coverage exact by construction, and test coverage close
    provided the two splits are drawn from a similar distribution. It costs
    one number per quantile and cannot overfit the test set, because val is
    never trained on.
    """
    pv = model.predict(va[cols])
    yv = va[C.TARGET].to_numpy()
    deltas = {q: float(np.quantile(yv - pv[q], q)) for q in pv}
    pt = model.predict(te[cols])
    return ({q: np.clip(p + deltas[q], 0, C.CLIP) for q, p in pt.items()}, deltas)


def run_arm(tr, te, cols, label, out, va=None):
    print(f"\n>>> {label}  ({len(cols)} features)", flush=True)
    model = MultiQuantileLGBM(**C.LGBM_PARAMS).fit(tr[cols], tr[C.TARGET])
    preds = model.predict(te[cols])

    if va is not None and len(va):
        cal, deltas = recalibrate(model, va, te, cols)
        y = te[C.TARGET].to_numpy()
        before = M.reliability(y, preds)
        after = M.reliability(y, cal)
        print("  recalibration offsets (fitted on val):",
              {q: round(v, 4) for q, v in deltas.items()}, flush=True)
        print("  coverage at nominal, before -> after:")
        for (_, b), (_, a) in zip(before.iterrows(), after.iterrows()):
            print(f"    q={b.nominal:.2f}  {b.empirical:.4f} -> {a.empirical:.4f}", flush=True)
        print(f"  mean pinball  before {M.mean_pinball(y, preds):.5f} "
              f"-> after {M.mean_pinball(y, cal):.5f}", flush=True)
        out[label + " [recalibrated]"] = {
            "report": M.summarise(
                te.assign(**{f"p{int(q * 100)}": p for q, p in cal.items()}),
                C.TARGET, {q: f"p{int(q * 100)}" for q in cal},
                ref_col="pred_persistence", label=label + " [recalibrated]"),
            "importance": pd.DataFrame(), "preds": te, "model": model}

    d = te.copy()
    for q, p in preds.items():
        d[f"p{int(q * 100)}"] = p
    rep = M.summarise(d, C.TARGET,
                      {q: f"p{int(q * 100)}" for q in preds},
                      ref_col="pred_persistence", label=label)
    M.print_report(rep, title=label)

    imp = pd.DataFrame({
        "feature": cols,
        "gain_p50": model.models[0.50].booster_.feature_importance("gain"),
    }).sort_values("gain_p50", ascending=False)
    imp["gain_pct"] = 100 * imp.gain_p50 / imp.gain_p50.sum()

    out[label] = {"report": rep, "importance": imp, "preds": d, "model": model}
    return rep["headline"]


def main():
    out = {}
    heads = []

    df = load()
    have_ens = all(c in df.columns for c in C.ENSEMBLE_FEATURES)
    base = [c for c in C.BASE_FEATURES if c in df.columns]
    ens = [c for c in C.ENSEMBLE_FEATURES if c in df.columns]

    tr = df[df.split == "train"]
    va = df[df.split == "val"]
    te = df[df.split == "test"]
    print(f"train={len(tr):,} val={len(va):,} test={len(te):,} "
          f"systems={df.system_id.nunique()}")
    print(f"base features={len(base)} ensemble features={len(ens)} -> {ens}")

    heads.append(run_arm(tr, te, base, "A base (parent 12 features)", out, va))

    if have_ens:
        heads.append(run_arm(tr, te, base + ens, "B base + ens_mean_fresh", out, va))

        df2 = load(drop_broken=True)
        tr2 = df2[df2.split == "train"]
        va2 = df2[df2.split == "val"]
        te2 = df2[df2.split == "test"]
        heads.append(run_arm(tr2, te2, base, "D base only, 81376 dropped", out, va2))
        heads.append(run_arm(tr2, te2, base + ens,
                             "C base + ens_mean_fresh, 81376 dropped", out, va2))
    else:
        print("\n[skip] ensemble arms - run fetch_ensemble.py then "
              "build_features_v3.py first.")

    summary = pd.DataFrame(heads)
    print("\n" + "=" * 74)
    print("ARM COMPARISON")
    print("=" * 74)
    print(summary.to_string())

    summary.to_csv(C.MODELS_DIR / "v3_metrics.csv")
    best = [k for k in out if not k.endswith("[recalibrated]")][-1]
    out[best]["preds"][["system_id", "timestamp", C.TARGET, "pred_persistence",
                        "p25", "p50", "p90"]].to_csv(
        C.MODELS_DIR / "v3_predictions_test.csv", index=False)
    out[best]["importance"].to_csv(
        C.MODELS_DIR / "v3_feature_importance.csv", index=False)

    with open(C.REPORTS_DIR / "v3_report.txt", "w", encoding="utf-8") as f:
        f.write("v3 ensemble experiment\n" + "=" * 74 + "\n\n")
        f.write(summary.to_string() + "\n\n")
        for label, o in out.items():
            f.write("\n" + "=" * 74 + f"\n{label}\n" + "=" * 74 + "\n")
            f.write("\nheadline\n" + o["report"]["headline"].to_string() + "\n")
            f.write("\nreliability\n" + o["report"]["reliability"].to_string(index=False) + "\n")
            f.write("\nby cloud regime\n" + o["report"]["regime"].to_string(index=False) + "\n")
            # recalibrated blocks reuse the parent arm's model, so they carry
            # no importance table of their own
            if len(o["importance"]):
                f.write("\nfeature importance\n"
                        + o["importance"].to_string(index=False) + "\n")

    print(f"\nWrote {C.MODELS_DIR / 'v3_metrics.csv'}")
    print(f"Wrote {C.REPORTS_DIR / 'v3_report.txt'}")

    if have_ens:
        print("\nEnsemble feature importance (arm B/C):")
        imp = out[best]["importance"]
        print(imp[imp.feature.isin(ens)].to_string(index=False))


if __name__ == "__main__":
    main()
