#!/usr/bin/env python3
"""
metrics.py
==========
Evaluation suite for v3. Replaces "one pooled R2" with the numbers that
actually describe a day-ahead PV forecast.

WHY POOLED R2 IS THE WRONG HEADLINE
-----------------------------------
R2 asks what fraction of the variance you explained. Most of the variance in
target_norm is the sun rising and setting and summer vs winter - both known
exactly from astronomy, needing zero forecast skill. The parent project's
clear-sky proxy scores R2 0.34-0.38 while learning nothing at all, which is
the size of that free component. So pooled R2 mostly measures whether you
know what time it is.

WHAT TO REPORT INSTEAD
----------------------
  skill_score      : improvement over the persistence baseline. Baseline-
                     relative, so the free astronomy cancels on both sides.
  pinball_loss     : the correct score for P25/P50/P90. Rewards intervals that
                     are simultaneously narrow and honest - you cannot win by
                     hedging wide.
  reliability      : of the times you said "90% chance output is below X", was
                     it actually below X 90% of the time? Calibration, not
                     sharpness.
  regime_table     : R2 and RMSE split by cloud regime. Reports the 0.85 clear
                     / 0.48 broken-cloud split instead of one blended number.
  interval_score   : width + calibration penalty for the P25-P90 band.

For a home battery controller "how sure are you" matters more than "what is
your single best guess": guess too high, the battery does not charge, and it
is empty when grid prices peak. That is what P25 is for, and pinball loss is
the metric that rewards getting it right.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# point metrics
# ---------------------------------------------------------------------------
def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def mbe(y, p):
    """Mean bias error. Positive = model under-forecasts."""
    return float(np.mean(np.asarray(y) - np.asarray(p)))


def r2(y, p):
    y, p = np.asarray(y), np.asarray(p)
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - ss_res / ss_tot)


def skill_score(y, p, p_ref):
    """Fraction of the reference forecast's RMSE removed. 0 = no better than
    the reference, 1 = perfect. The honest headline number."""
    return float(1 - rmse(y, p) / rmse(y, p_ref))


# ---------------------------------------------------------------------------
# probabilistic metrics
# ---------------------------------------------------------------------------
def pinball_loss(y, p, q):
    """Quantile (pinball) loss at level q. Lower is better.
    Under-prediction is charged q per unit, over-prediction (1-q)."""
    y, p = np.asarray(y), np.asarray(p)
    d = y - p
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def mean_pinball(y, preds):
    """Average pinball across all quantiles. The single best scalar summary of
    a probabilistic forecast. preds = {0.25: array, 0.50: array, ...}"""
    return float(np.mean([pinball_loss(y, p, q) for q, p in preds.items()]))


def reliability(y, preds):
    """Calibration check. For each quantile level q, the fraction of actuals
    that fell at or below the predicted quantile. Perfect calibration puts
    `empirical` equal to `nominal`."""
    y = np.asarray(y)
    rows = []
    for q, p in sorted(preds.items()):
        emp = float(np.mean(y <= np.asarray(p)))
        if abs(emp - q) < 0.03:
            verdict = "well calibrated"
        elif emp > q:
            verdict = "too conservative"
        else:
            verdict = "too aggressive"
        rows.append({"nominal": q, "empirical": emp, "gap": emp - q,
                     "verdict": verdict})
    return pd.DataFrame(rows)


def interval_score(y, lo, hi, alpha=0.65):
    """Winkler interval score for the P25-P90 band (alpha = 1 - coverage).
    Penalises width, plus a penalty for every actual falling outside."""
    y, lo, hi = np.asarray(y), np.asarray(lo), np.asarray(hi)
    width = hi - lo
    below = (2 / alpha) * (lo - y) * (y < lo)
    above = (2 / alpha) * (y - hi) * (y > hi)
    return float(np.mean(width + below + above))


def coverage(y, lo, hi):
    y, lo, hi = np.asarray(y), np.asarray(lo), np.asarray(hi)
    return float(np.mean((y >= lo) & (y <= hi)))


# ---------------------------------------------------------------------------
# regime and per-system breakdowns
# ---------------------------------------------------------------------------
KT_BINS = [-0.01, 0.2, 0.4, 0.6, 0.8, 2.0]
KT_LABELS = ["overcast <0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "clear >0.8"]


def regime_table(df, y_col, pred_col, kt_col="clearsky_index_fc"):
    """R2 / RMSE split by forecast cloudiness. This is where the real story is:
    a model can look strong pooled and still be near-useless in broken cloud."""
    d = df.copy()
    d["_bin"] = pd.cut(d[kt_col], KT_BINS, labels=KT_LABELS)
    rows = []
    for name, g in d.groupby("_bin", observed=True):
        if len(g) < 50:
            continue
        rows.append({
            "regime": name, "n": len(g),
            "R2": r2(g[y_col], g[pred_col]),
            "RMSE": rmse(g[y_col], g[pred_col]),
            "MBE": mbe(g[y_col], g[pred_col]),
            "sd_target": float(g[y_col].std()),
        })
    return pd.DataFrame(rows)


def per_system_table(df, y_col, pred_col, ref_col=None):
    rows = []
    for sid, g in df.groupby("system_id"):
        r = {"system_id": int(sid), "n": len(g),
             "R2": r2(g[y_col], g[pred_col]),
             "RMSE": rmse(g[y_col], g[pred_col]),
             "MBE": mbe(g[y_col], g[pred_col])}
        if ref_col:
            r["skill"] = skill_score(g[y_col], g[pred_col], g[ref_col])
        rows.append(r)
    return pd.DataFrame(rows).sort_values("R2")


def sse_share(df, y_col, pred_col, group="system_id"):
    """Which systems eat the error budget? Compares each group's share of the
    total squared error against its share of the rows. A system with 3% of the
    rows and 7% of the error is a data problem, not a modelling problem."""
    d = df.copy()
    d["_se"] = (d[y_col] - d[pred_col]) ** 2
    t = d.groupby(group).agg(n=("_se", "size"), sse=("_se", "sum"))
    t["row_share_pct"] = 100 * t.n / t.n.sum()
    t["sse_share_pct"] = 100 * t.sse / t.sse.sum()
    t["ratio"] = t["sse_share_pct"] / t["row_share_pct"]
    return t.sort_values("ratio", ascending=False)


# ---------------------------------------------------------------------------
# one-call summary
# ---------------------------------------------------------------------------
def summarise(df, y_col, preds, ref_col, kt_col="clearsky_index_fc",
              label="model"):
    """Full report for one model. preds = {0.25: col_or_array, 0.50: ...}"""
    y = df[y_col].to_numpy()
    P = {q: (df[v].to_numpy() if isinstance(v, str) else np.asarray(v))
         for q, v in preds.items()}
    p50 = P[0.50]
    ref = df[ref_col].to_numpy()

    head = pd.Series({
        "MAE": mae(y, p50), "RMSE": rmse(y, p50), "MBE": mbe(y, p50),
        "R2": r2(y, p50), "skill_vs_persistence": skill_score(y, p50, ref),
        "mean_pinball": mean_pinball(y, P),
        "P25_P90_coverage": coverage(y, P[0.25], P[0.90]),
        "P25_P90_interval_score": interval_score(y, P[0.25], P[0.90]),
    }, name=label)

    d = df.copy()
    d["_p50"] = p50
    return {
        "headline": head,
        "reliability": reliability(y, P),
        "regime": regime_table(d, y_col, "_p50", kt_col),
        "per_system": per_system_table(d, y_col, "_p50", ref_col),
        "sse_share": sse_share(d, y_col, "_p50"),
    }


def print_report(rep, title=""):
    if title:
        print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")
    print("\n-- headline --")
    print(rep["headline"].to_string())
    print("\n-- reliability (are the quantiles honest?) --")
    print(rep["reliability"].to_string(index=False))
    print("\n-- by cloud regime --")
    print(rep["regime"].to_string(index=False))
    print("\n-- worst 5 systems by R2 --")
    print(rep["per_system"].head(5).to_string(index=False))
    print("\n-- systems eating more error than their row share --")
    print(rep["sse_share"].head(5).to_string())
