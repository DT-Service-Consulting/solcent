#!/usr/bin/env python3
"""
revision_stats.py
=================
Confidence intervals and significance tests for the paired comparisons in the
manuscript. Works on the test-split predictions written by

    consolidate.py --full-fleet --save-preds     -> models/revision/preds/
    mos_pv_integration.py --save-preds           -> models/revision/preds/

Two methods, both respecting the dependence in the data:

  Day-block bootstrap   Days are resampled with replacement (all systems and
                        slots of a day move together), so within-day
                        autocorrelation and cross-system correlation are kept.
                        Each metric difference (arm B minus arm A) gets a 95 %
                        percentile interval from 2000 resamples.

  Diebold-Mariano test  On the daily mean loss differential (squared error of
                        P50, and mean pinball over P25/P50/P90), with a
                        Newey-West long-run variance. Two-sided p-value from
                        the normal distribution.

Metrics are pooled exactly as in the manuscript (R2 and skill from pooled
sums), so the point estimates reproduce the reported values.

Output: models/revision/paired_comparisons.csv
        reports/revision_stats.txt
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as C

PRED = C.MODELS_DIR / "revision" / "preds"
OUT = C.MODELS_DIR / "revision"
N_BOOT = 2000
SEED = 20261027
QS = (0.25, 0.50, 0.90)

# (label, file A, file B): difference reported as B minus A.
PAIRS = [
    ("E0 objective: pinball vs RMSE selection (33 systems)",
     "exp0__inherited_rmse_tuned_shared", "exp0__tuned_pinball_per_quantile"),
    ("E1 oracle: + regional truth vs NWP only",
     "exp1__baseline_nwp_only", "exp1__oracle_regional_truth"),
    ("E1 oracle: regional truth without NWP vs NWP only",
     "exp1__baseline_nwp_only", "exp1__oracle_geometry_no_nwp"),
    ("E3: + multi-model mean vs base (33 systems)",
     "exp3__base_only_12", "exp3__ens_mean_fresh_6_day_1_models"),
    ("E3: + mean of all 12 members vs base",
     "exp3__base_only_12", "exp3__ens_ghi_mean_all_12_members"),
    ("E3: + ensemble-mean POA vs base",
     "exp3__base_only_12", "exp3__poa_ens_mean"),
    ("E3: + lagged mean (days 1-3) vs base",
     "exp3__base_only_12", "exp3__lagged_mean_day1_3"),
    ("E3: + spread only vs base",
     "exp3__base_only_12", "exp3__spread_features_only"),
    ("E3: + all 10 ensemble features vs base",
     "exp3__base_only_12", "exp3__all_10_ensemble_features"),
    ("E4: recalibrated vs raw (33 systems, base + ens)",
     "exp4__b_base_ens_all_systems", "exp4__b_base_ens_all_systems_recalibrated"),
    ("E4 (32 systems): + multi-model mean vs base",
     "exp4__d_base_only_81376_dropped", "exp4__c_base_ens_81376_dropped_proposed"),
    ("MOS seen rooftops (33): corrected irradiance vs baseline",
     "mos_full__a_baseline", "mos_full__b_mos_corrected_irradiance"),
    ("MOS seen rooftops (33): MOS as extra features vs baseline",
     "mos_full__a_baseline", "mos_full__c_baseline_mos_as_extra_features"),
    ("MOS seen rooftops (32): corrected irradiance vs baseline",
     "mos__a_baseline", "mos__b_mos_corrected_irradiance"),
]


def load(name):
    p = PRED / f"{name}.csv.gz"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    d["day"] = pd.to_datetime(d["timestamp"], utc=True).dt.normalize()
    return d.sort_values(["system_id", "timestamp"]).reset_index(drop=True)


def pinball(y, p, q):
    e = y - p
    return np.maximum(q * e, (q - 1) * e)


def daily_sums(d):
    """Per-day sufficient statistics for pooled metrics."""
    y, p50 = d.y.to_numpy(), d.p50.to_numpy()
    f = pd.DataFrame({
        "day": d.day,
        "n": 1.0,
        "sy": y,
        "syy": y * y,
        "sse": (y - p50) ** 2,
        "sse_ref": (y - d.persistence.to_numpy()) ** 2,
        "pin": sum(pinball(y, d[f"p{int(q*100)}"].to_numpy(), q) for q in QS) / len(QS),
        "cov": ((y >= d.p25.to_numpy()) & (y <= d.p90.to_numpy())).astype(float),
    })
    return f.groupby("day").sum()


def metrics(s):
    """Pooled metrics from (weighted) daily sums."""
    n = s["n"]
    sst = s["syy"] - s["sy"] ** 2 / n
    return {
        "R2": 1 - s["sse"] / sst,
        "skill": 1 - np.sqrt(s["sse"] / n) / np.sqrt(s["sse_ref"] / n),
        "RMSE": np.sqrt(s["sse"] / n),
        "pinball": s["pin"] / n,
        "coverage": s["cov"] / n,
    }


def newey_west_se(d):
    d = np.asarray(d, dtype=float)
    T = len(d)
    u = d - d.mean()
    h = int(np.floor(T ** (1 / 3)))
    lrv = np.dot(u, u) / T
    for k in range(1, h + 1):
        w = 1 - k / (h + 1)
        lrv += 2 * w * np.dot(u[k:], u[:-k]) / T
    return np.sqrt(lrv / T)


def dm_test(diff):
    se = newey_west_se(diff)
    t = np.mean(diff) / se if se > 0 else np.nan
    return float(t), float(2 * (1 - stats.norm.cdf(abs(t))))


def compare(label, a, b, rng):
    key = ["system_id", "timestamp"]
    m = a[key].merge(b[key], on=key)
    a = a.merge(m, on=key)
    b = b.merge(m, on=key)
    sa, sb = daily_sums(a), daily_sums(b)
    sa, sb = sa.align(sb, join="inner", axis=0)
    days = len(sa)

    ma, mb = metrics(sa.sum()), metrics(sb.sum())
    A, B = sa.to_numpy(), sb.to_numpy()
    cols = list(sa.columns)
    boot = {k: [] for k in ma}
    for _ in range(N_BOOT):
        w = rng.multinomial(days, np.full(days, 1 / days)).astype(float)
        xa = dict(zip(cols, w @ A))
        xb = dict(zip(cols, w @ B))
        ra, rb = metrics(xa), metrics(xb)
        for k in boot:
            boot[k].append(rb[k] - ra[k])

    # daily loss differentials for Diebold-Mariano
    d_se = sb["sse"] / sb["n"] - sa["sse"] / sa["n"]
    d_pin = sb["pin"] / sb["n"] - sa["pin"] / sa["n"]
    dm_se, p_se = dm_test(d_se.to_numpy())
    dm_pin, p_pin = dm_test(d_pin.to_numpy())

    row = {"comparison": label, "days": days, "rows": len(m)}
    for k in ma:
        lo, hi = np.percentile(boot[k], [2.5, 97.5])
        row[f"{k}_A"] = ma[k]
        row[f"{k}_B"] = mb[k]
        row[f"d{k}"] = mb[k] - ma[k]
        row[f"d{k}_lo"] = lo
        row[f"d{k}_hi"] = hi
    row.update({"DM_sqerr": dm_se, "p_sqerr": p_se, "DM_pinball": dm_pin, "p_pinball": p_pin})
    return row


def main():
    rng = np.random.default_rng(SEED)
    rows, missing = [], []
    for label, fa, fb in PAIRS:
        a, b = load(fa), load(fb)
        if a is None or b is None:
            missing.append(label)
            continue
        rows.append(compare(label, a, b, rng))
        r = rows[-1]
        print(f"{label}\n  dR2 {r['dR2']:+.4f} [{r['dR2_lo']:+.4f}, {r['dR2_hi']:+.4f}]  "
              f"dskill {r['dskill']:+.4f} [{r['dskill_lo']:+.4f}, {r['dskill_hi']:+.4f}]  "
              f"dpinball {r['dpinball']:+.5f} [{r['dpinball_lo']:+.5f}, {r['dpinball_hi']:+.5f}]  "
              f"DM p(sq) {r['p_sqerr']:.3f}  p(pin) {r['p_pinball']:.3f}", flush=True)
    res = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "paired_comparisons.csv", index=False)

    lines = ["Paired comparisons: day-block bootstrap 95% CI and Diebold-Mariano tests",
             "=" * 76, f"bootstrap resamples: {N_BOOT}; difference = B minus A", ""]
    for _, r in res.iterrows():
        lines.append(r["comparison"])
        lines.append(f"  days={r['days']} rows={r['rows']:,}")
        for k, fmt in (("R2", "+.4f"), ("skill", "+.4f"), ("pinball", "+.5f"), ("coverage", "+.3f")):
            lines.append(f"  {k:8s} A={r[k+'_A']:.5f} B={r[k+'_B']:.5f}  "
                         f"diff {r['d'+k]:{fmt}} [{r['d'+k+'_lo']:{fmt}}, {r['d'+k+'_hi']:{fmt}}]")
        lines.append(f"  DM squared error {r['DM_sqerr']:+.2f} (p={r['p_sqerr']:.4f})   "
                     f"DM pinball {r['DM_pinball']:+.2f} (p={r['p_pinball']:.4f})")
        lines.append("")
    if missing:
        lines.append("Not computed (prediction files missing): " + "; ".join(missing))
    lines += loso_tests()
    (C.REPORTS_DIR / "revision_stats.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[-12:]))


def loso_tests():
    """Paired Wilcoxon signed-rank tests across the 33 held-out rooftops."""
    lines = ["Leave-one-system-out: paired Wilcoxon signed-rank tests over rooftops", "-" * 76]
    seeds = C.MODELS_DIR / "revision" / "loso_seeds_per_system.csv"
    if seeds.exists():
        s = pd.read_csv(seeds)
        # per-rooftop R2 averaged over seeds, so seed noise does not enter the test
        m = s.groupby(["config", "system_id"]).R2.mean().unstack(0)
        for a, b in (("temporal", "transfer"), ("preliminary", "transfer"),
                     ("preliminary", "temporal")):
            if a in m and b in m:
                d = (m[b] - m[a]).dropna()
                w = stats.wilcoxon(d)
                lines.append(f"  R2 {b} - {a}: median {d.median():+.4f}, "
                             f"{int((d > 0).sum())}/{len(d)} rooftops better, p={w.pvalue:.4f}")
    mos = C.MODELS_DIR / "loso_mos_per_system.csv"
    if mos.exists():
        r = pd.read_csv(mos)
        for k in ("R2", "skill"):
            d = r[f"B_mos_{k}"] - r[f"A_baseline_{k}"]
            w = stats.wilcoxon(d)
            lines.append(f"  MOS unseen rooftops, {k}: median {d.median():+.5f}, "
                         f"{int((d > 0).sum())}/{len(d)} better, p={w.pvalue:.4f}")
    return lines + [""]


if __name__ == "__main__":
    main()
