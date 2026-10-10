#!/usr/bin/env python3
"""
consolidate.py
==============
Single-configuration driver. Re-derives every number destined for the paper
under ONE identical setting, so no table mixes vintages.

WHY THIS EXISTS
---------------
Results accumulated across this project under three different configurations:

  group                                   objective  trees  feature build
  oracle, ablations, PV decomposition     L2         500    pre-clipping-fix
  ensemble arms, recalibration            quantile   500    post-fix
  LOSO, MOS-LOSO                          quantile   300    post-fix

None of the conclusions depend on the discrepancies - the effects are far
larger. But R2 from an L2 fit and R2 from a P50 quantile fit are not
comparable (quantile regression optimises MAE, so its R2 is structurally
lower), and a reviewer seeing 0.8027 and 0.7934 in adjacent tables will read
a contradiction that is not there.

This driver fixes the configuration once, at the top, and runs every
experiment through it.

FIXED CONFIGURATION
-------------------
  objective      : quantile, alpha in {0.25, 0.50, 0.90}; P50 reported
  n_estimators   : 500 everywhere except LOSO (see below)
  features       : post-fix model_ready_v3.csv build
  splits         : the project's chronological train/val/test columns
  excluded       : 12293 (parent decision) always; 81376 per-experiment
  reference      : SARAH-3 satellite for all irradiance-level scoring

LOSO EXCEPTION, STATED EXPLICITLY
---------------------------------
LOSO requires one fit per held-out system. At 500 trees x 33 folds x 2 arms
the run becomes impractical, so LOSO uses 300 trees. This is a computational
constraint, it is applied identically to every LOSO arm, and it must be
declared in the paper rather than buried. Experiment 0 quantifies the cost of
that choice directly so the declaration carries a number.

Also extends the calibration sweep to 45 and 60 days. Previously 30 was the
largest window tested, so the finding could only support "at least 30 days",
not "30 days is sufficient". Without longer windows the recommendation is
unbounded above.

Outputs: v3_ensemble/models/consolidated/*.csv
         v3_ensemble/reports/consolidated_report.txt
"""

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

from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb

# ---------------------------------------------------------------------------
# THE ONE CONFIGURATION
# ---------------------------------------------------------------------------
# Per-quantile parameters from the pinball-objective search. Each quantile is a
# separate estimand and gets its own configuration; see config.TUNED_PARAMS.
PARAMS = C.TUNED_PARAMS
# LOSO uses the TRANSFER configuration, selected against held-out systems
# rather than held-out time. The two objectives pull apart: the temporal
# optimum wins on known rooftops and loses on unseen ones. Using TUNED_PARAMS
# here would report deployment numbers from a configuration selected for a
# setting that is not deployment. See config.TRANSFER_PARAMS.
LOSO_PARAMS = C.TRANSFER_PARAMS
QUANTILES = C.QUANTILES                   # [0.25, 0.50, 0.90]
CALIB_DAYS = [0, 3, 7, 14, 30, 45, 60]
EPS = 1.0
ORACLE_COL = "fleet_actual__LEAKAGE_DIAGNOSTIC_ONLY"
MIN_FLEET = 3

OUT = C.MODELS_DIR / "consolidated"
OUT.mkdir(parents=True, exist_ok=True)
LOG = []


def log(msg=""):
    print(msg, flush=True)
    LOG.append(str(msg))


def banner(t):
    log("\n" + "=" * 78)
    log(t)
    log("=" * 78)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load_pv(drop_broken=False):
    df = pd.read_csv(C.MODEL_READY_V3)
    df = df[~df.system_id.isin(C.EXCLUDE_SYSTEMS)]
    if drop_broken:
        df = df[~df.system_id.isin(C.BROKEN_SYSTEMS)]
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


def fit_quantiles(tr, cols, params=None):
    """Fit one model per quantile. `params` maps quantile -> parameter dict, so
    each level is fitted with the configuration selected for it."""
    p = params or PARAMS
    return {q: lgb.LGBMRegressor(objective="quantile", alpha=q, **p[q])
            .fit(tr[cols], tr[C.TARGET]) for q in QUANTILES}


def predict_quantiles(models, X):
    return {q: np.clip(m.predict(X), 0, C.CLIP) for q, m in models.items()}


def score_p50(te, P, label, extra=None):
    y = te[C.TARGET].to_numpy()
    r = {"config": label,
         "MAE": M.mae(y, P[0.50]), "RMSE": M.rmse(y, P[0.50]),
         "MBE": M.mbe(y, P[0.50]), "R2": M.r2(y, P[0.50]),
         "skill": M.skill_score(y, P[0.50], te.pred_persistence.to_numpy()),
         "mean_pinball": M.mean_pinball(y, P),
         "P25_P90_coverage": M.coverage(y, P[0.25], P[0.90])}
    if extra:
        r.update(extra)
    return r


# ---------------------------------------------------------------------------
# EXP 0 - objective sensitivity: inherited (RMSE-tuned) vs tuned (pinball)
# ---------------------------------------------------------------------------
def exp0_objective_sensitivity(df):
    """Replaces the former tree-count sensitivity check, which existed only to
    quantify the LOSO shortcut. That shortcut is gone - LOSO now uses the same
    parameters as every other experiment - so the slot measures the question
    that remains open instead: does selecting hyperparameters under the correct
    objective change the result?"""
    banner("EXP 0 - hyperparameter objective: inherited (RMSE) vs tuned (pinball)")
    base = [c for c in C.BASE_FEATURES if c in df.columns]
    ens = [c for c in C.ENSEMBLE_FEATURES if c in df.columns]
    cols = base + ens
    d = df[~df.system_id.isin(C.BROKEN_SYSTEMS)]
    tr, te = d[d.split == "train"], d[d.split == "test"]

    inherited = {q: dict(C.LGBM_PARAMS) for q in QUANTILES}
    rows = []
    for label, p in (("inherited (RMSE-tuned, shared)", inherited),
                     ("tuned (pinball, per-quantile)", C.TUNED_PARAMS)):
        P = predict_quantiles(fit_quantiles(tr, cols, p), te[cols])
        r = score_p50(te, P, label)
        y = te[C.TARGET].to_numpy()
        for q in QUANTILES:
            r[f"pinball_p{int(q*100)}"] = M.pinball_loss(y, P[q], q)
        rows.append(r)
        log(f"  {label:32s} R2={r['R2']:.4f}  skill={r['skill']:.4f}  "
            f"pinball={r['mean_pinball']:.5f}")
    a, b = rows
    log(f"\n  delta from correcting the objective: "
        f"pinball {b['mean_pinball']-a['mean_pinball']:+.5f}  "
        f"R2 {b['R2']-a['R2']:+.4f}  skill {b['skill']-a['skill']:+.4f}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# EXP 1 - oracle ceiling, QUANTILE objective (was L2)
# ---------------------------------------------------------------------------
def exp1_oracle(df):
    banner("EXP 1 - fleet-oracle ceiling")
    base = [c for c in C.BASE_FEATURES if c in df.columns]
    ens = [c for c in C.ENSEMBLE_FEATURES if c in df.columns]

    g = df.groupby("timestamp")[C.TARGET]
    s, n = g.transform("sum"), g.transform("count")
    d = df.copy()
    d[ORACLE_COL] = (s - d[C.TARGET]) / (n - 1)
    d = d[n > MIN_FLEET]

    tr, te = d[d.split == "train"], d[d.split == "test"]
    geo = ["sun_elevation", "clearsky_ghi", "capacity_kwp", "tilt_deg", "azimuth_deg"]
    arms = {
        "baseline (NWP only)": base + ens,
        "+ ORACLE regional truth": base + ens + [ORACLE_COL],
        "ORACLE + geometry, no NWP": [ORACLE_COL] + [c for c in geo if c in d.columns],
    }
    rows = []
    for label, cols in arms.items():
        P = predict_quantiles(fit_quantiles(tr, cols), te[cols])
        r = score_p50(te, P, label, {"n_features": len(cols)})
        rows.append(r)
        log(f"  {label:30s} nf={r['n_features']:2d}  R2={r['R2']:.4f}  "
            f"skill={r['skill']:.4f}")
    gap = rows[1]["R2"] - rows[0]["R2"]
    log(f"\n  headroom to perfect regional truth: {gap:+.4f} R2")
    log("  NOTE: the oracle is unavailable at forecast time. It sizes the prize.")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# EXP 2 - residual structure + regime + per-system
# ---------------------------------------------------------------------------
def exp2_decomposition(df):
    banner("EXP 2 - error decomposition")
    base = [c for c in C.BASE_FEATURES if c in df.columns]
    ens = [c for c in C.ENSEMBLE_FEATURES if c in df.columns]
    cols = base + ens
    tr, te = df[df.split == "train"], df[df.split == "test"]
    P = predict_quantiles(fit_quantiles(tr, cols), te[cols])

    t = te.copy()
    t["pred"] = P[0.50]
    t["res"] = t[C.TARGET] - t["pred"]
    t = t.sort_values(["system_id", "timestamp"])

    log("\n-- residual autocorrelation --")
    ac = []
    for k in (1, 2, 4, 8, 96):
        v = t.res.corr(t.groupby("system_id").res.shift(k))
        ac.append({"lag_slots": k, "lag_min": k * 15, "autocorr": v})
        log(f"  lag {k:3d} slots ({k*15:5d} min): {v:.3f}")

    log("\n-- by FORECAST clear-sky index --")
    reg_fc = M.regime_table(t, C.TARGET, "pred", "clearsky_index_fc")
    log(reg_fc.round(4).to_string(index=False))

    # observation-binned regime - the reviewer question the 3-way scan flagged
    t["kt_obs"] = np.where(t.clearsky_ghi > 50,
                           t[C.TARGET] / (t.clearsky_ghi / 1000.0), np.nan)
    t["kt_obs"] = t["kt_obs"].clip(0, 1.5)
    log("\n-- by OBSERVED (PV-derived) clear-sky index --")
    reg_obs = M.regime_table(t.dropna(subset=["kt_obs"]), C.TARGET, "pred", "kt_obs")
    log(reg_obs.round(4).to_string(index=False))

    log("\n-- systems eating more error than their row share --")
    sse = M.sse_share(t, C.TARGET, "pred")
    log(sse.head(5).round(3).to_string())

    return pd.DataFrame(ac), reg_fc, reg_obs, sse.reset_index()


# ---------------------------------------------------------------------------
# EXP 3 - ensemble ablation, all on the post-fix build
# ---------------------------------------------------------------------------
def exp3_ensemble(df):
    banner("EXP 3 - ensemble ablation")
    base = [c for c in C.BASE_FEATURES if c in df.columns]
    full = [c for c in C.ENSEMBLE_FEATURES_FULL if c in df.columns]
    spread = [c for c in ["ens_kt_std", "ens_lag_std", "ens_model_std",
                          "ens_agree_frac", "ens_ghi_iqr"] if c in df.columns]

    d = df[~df.system_id.isin(C.BROKEN_SYSTEMS)]
    lag = [f"ghi_lag{i}" for i in (1, 2, 3) if f"ghi_lag{i}" in d.columns]
    d = d.copy()
    if lag:
        d["ens_mean_lag13"] = d[lag].mean(axis=1)

    arms = {
        "base only (12)": base,
        "+ ens_mean_fresh (6 day-1 models)": base + ["ens_mean_fresh"],
        "+ ens_ghi_mean (all 12 members)": base + ["ens_ghi_mean"],
        "+ poa_ens_mean": base + ["poa_ens_mean"],
        "+ spread features only": base + spread,
        "+ all 10 ensemble features": base + full,
    }
    if lag:
        arms["+ lagged mean (day1-3)"] = base + ["ens_mean_lag13"]

    tr, te = d[d.split == "train"], d[d.split == "test"]
    rows = []
    for label, cols in arms.items():
        cols = [c for c in cols if c in d.columns]
        P = predict_quantiles(fit_quantiles(tr, cols), te[cols])
        r = score_p50(te, P, label, {"n_features": len(cols)})
        rows.append(r)
        log(f"  {label:36s} nf={r['n_features']:2d} R2={r['R2']:.4f} "
            f"skill={r['skill']:.4f} pinball={r['mean_pinball']:.5f}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# EXP 4 - headline arms + recalibration
# ---------------------------------------------------------------------------
def exp4_arms_and_recal(df):
    banner("EXP 4 - headline arms + conformal recalibration")
    base = [c for c in C.BASE_FEATURES if c in df.columns]
    ens = [c for c in C.ENSEMBLE_FEATURES if c in df.columns]
    rows, recal_rows = [], []

    for label, drop_broken, cols in [
        ("A base, all systems", False, base),
        ("B base+ens, all systems", False, base + ens),
        ("D base only, 81376 dropped", True, base),
        ("C base+ens, 81376 dropped  [PROPOSED]", True, base + ens),
    ]:
        d = df[~df.system_id.isin(C.BROKEN_SYSTEMS)] if drop_broken else df
        tr, va, te = d[d.split == "train"], d[d.split == "val"], d[d.split == "test"]
        models = fit_quantiles(tr, cols)
        P = predict_quantiles(models, te[cols])
        r = score_p50(te, P, label, {"n_features": len(cols)})
        rows.append(r)
        log(f"  {label:40s} R2={r['R2']:.4f} skill={r['skill']:.4f} "
            f"pinball={r['mean_pinball']:.5f} cov={r['P25_P90_coverage']:.3f}")

        # conformal offsets fitted on VAL only
        Pv = predict_quantiles(models, va[cols])
        yv = va[C.TARGET].to_numpy()
        deltas = {q: float(np.quantile(yv - Pv[q], q)) for q in QUANTILES}
        Pc = {q: np.clip(P[q] + deltas[q], 0, C.CLIP) for q in QUANTILES}
        rc = score_p50(te, Pc, label + " [recalibrated]", {"n_features": len(cols)})
        rows.append(rc)
        y = te[C.TARGET].to_numpy()
        for q in QUANTILES:
            recal_rows.append({
                "config": label, "nominal": q, "delta": deltas[q],
                "coverage_before": float(np.mean(y <= P[q])),
                "coverage_after": float(np.mean(y <= Pc[q])),
            })
        log(f"    recalibrated: P90 coverage "
            f"{np.mean(y <= P[0.90]):.4f} -> {np.mean(y <= Pc[0.90]):.4f}  "
            f"pinball {r['mean_pinball']:.5f} -> {rc['mean_pinball']:.5f}")
    return pd.DataFrame(rows), pd.DataFrame(recal_rows)


# ---------------------------------------------------------------------------
# EXP 5 - LOSO + extended calibration sweep
# ---------------------------------------------------------------------------
def corrections(cal, ev, cap_kwp, fleet_ratio):
    out = {"none": ev["pred"].to_numpy()}
    if len(cal) < 10:
        return out
    out["bias"] = np.clip(ev["pred"].to_numpy()
                          + (cal[C.TARGET] - cal["pred"]).mean(), 0, C.CLIP)
    if cal["pred"].std() > 1e-6:
        b, a = np.polyfit(cal["pred"], cal[C.TARGET], 1)
        out["linear"] = np.clip(a + b * ev["pred"].to_numpy(), 0, C.CLIP)
    obs = np.percentile(cal["target_power_w"], 99) / 1000.0
    if obs > 0.1 and cap_kwp > 0.1 and fleet_ratio > 0.1:
        sc = float(np.clip((obs / cap_kwp) / fleet_ratio, 0.4, 2.5))
        r2 = (cal[C.TARGET] - cal["pred"] * sc).mean()
        out["cap_emp+bias"] = np.clip(ev["pred"].to_numpy() * sc + r2, 0, C.CLIP)
    return out


def exp5_loso(df):
    banner("EXP 5 - LOSO zero-shot + calibration sweep")
    base = [c for c in C.BASE_FEATURES if c in df.columns]
    ens = [c for c in C.ENSEMBLE_FEATURES if c in df.columns]
    cols = base + ens
    systems = sorted(df.system_id.unique())
    log(f"  {len(systems)} systems | calibration days {CALIB_DAYS}")

    zero, calib = [], []
    for i, sid in enumerate(systems, 1):
        tr = df[(df.system_id != sid) & (df.split.isin(["train", "val"]))]
        te = df[(df.system_id == sid) & (df.split == "test")].copy()
        if len(te) < 500:
            continue
        fleet_ratio = float(np.median([
            np.percentile(g["target_power_w"], 99) / 1000.0 / g["capacity_kwp"].iloc[0]
            for _, g in tr.groupby("system_id")]))
        m = lgb.LGBMRegressor(objective="quantile", alpha=0.50,
                              **LOSO_PARAMS).fit(tr[cols], tr[C.TARGET])
        te["pred"] = np.clip(m.predict(te[cols]), 0, C.CLIP)
        y = te[C.TARGET].to_numpy()
        cap = float(te["capacity_kwp"].iloc[0])
        obs_kw = np.percentile(te["target_power_w"], 99) / 1000.0
        zero.append({"system_id": sid, "n": len(te),
                     "R2": M.r2(y, te["pred"]), "RMSE": M.rmse(y, te["pred"]),
                     "MBE": M.mbe(y, te["pred"]),
                     "skill": M.skill_score(y, te["pred"],
                                            te.pred_persistence.to_numpy()),
                     "capacity_kwp": cap, "cap_obs_kw": obs_kw,
                     "cap_ratio": obs_kw / cap})
        days = np.sort(te["day"].unique())
        for nd in CALIB_DAYS:
            if nd == 0:
                cal, ev = te.iloc[:0], te
            else:
                if len(days) <= nd + 5:
                    continue
                cut = days[nd]
                cal, ev = te[te.day < cut], te[te.day >= cut]
            if len(ev) < 200:
                continue
            ye = ev[C.TARGET].to_numpy()
            for name, p in corrections(cal, ev, cap, fleet_ratio).items():
                calib.append({"system_id": sid, "calib_days": nd, "method": name,
                              "n_eval": len(ev), "R2": M.r2(ye, p),
                              "skill": M.skill_score(
                                  ye, p, ev.pred_persistence.to_numpy())})
        log(f"  [{i}/{len(systems)}] {sid} R2={zero[-1]['R2']:.4f} "
            f"skill={zero[-1]['skill']:.4f} cap_ratio={zero[-1]['cap_ratio']:.3f}")

    zs, cb = pd.DataFrame(zero), pd.DataFrame(calib)
    log("\n-- zero-shot distribution --")
    for m_ in ("R2", "skill", "RMSE"):
        q = zs[m_].quantile([0, .1, .25, .5, .75, .9, 1])
        log(f"  {m_:5s} min={q[0]:.4f} p10={q[.1]:.4f} p25={q[.25]:.4f} "
            f"MED={q[.5]:.4f} p75={q[.75]:.4f} p90={q[.9]:.4f} max={q[1]:.4f}")
    log(f"  worse than persistence : {(zs.skill <= 0).sum()}/{len(zs)}")
    log(f"  R2 < 0                 : {(zs.R2 < 0).sum()}/{len(zs)}")
    log(f"  cap_ratio  min={zs.cap_ratio.min():.3f} med={zs.cap_ratio.median():.3f} "
        f"max={zs.cap_ratio.max():.3f}")
    log(f"  corr(cap_ratio, MBE) = {zs.cap_ratio.corr(zs.MBE):+.3f}")

    log("\n-- calibration sweep, median skill (compare WITHIN a row only) --")
    piv = cb.pivot_table(index="calib_days", columns="method",
                         values="skill", aggfunc="median")
    log(piv.round(4).to_string())
    log("\n-- calibration sweep, median R2 --")
    piv2 = cb.pivot_table(index="calib_days", columns="method",
                          values="R2", aggfunc="median")
    log(piv2.round(4).to_string())
    log("\n-- best method per window (vs doing nothing, same rows) --")
    for nd in sorted(cb.calib_days.unique()):
        s = cb[cb.calib_days == nd]
        med = s.groupby("method").skill.median()
        if "none" not in med:
            continue
        best = med.drop("none").idxmax() if len(med) > 1 else None
        if best:
            log(f"  day {nd:2d}: best={best:14s} {med[best]:+.4f} vs none "
                f"{med['none']:+.4f}  gain {med[best]-med['none']:+.4f}")
    return zs, cb, piv, piv2


# ---------------------------------------------------------------------------
def main():
    # --only lets a single experiment be re-derived without redoing the suite.
    # E5 depends on LOSO_PARAMS, which changed when LOSO moved to the transfer
    # configuration; E0-E4 depend only on PARAMS and did not need re-running.
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None,
                    choices=["exp0", "exp1", "exp2", "exp3", "exp4", "exp5"],
                    help="re-derive a single experiment, leaving the rest")
    only = ap.parse_args().only

    t0 = time.time()
    log(f"CONSOLIDATION RUN  {time.strftime('%Y-%m-%d %H:%M')}"
        + (f"   [--only {only}]" if only else ""))
    for q, pp in PARAMS.items():
        log(f"  q={q}: n_estimators={pp['n_estimators']}, "
            f"lr={pp['learning_rate']:.4f}, leaves={pp['num_leaves']}, "
            f"depth={pp['max_depth']}")
    log(f"  LOSO (P50): TRANSFER config - depth={LOSO_PARAMS['max_depth']}, "
        f"leaves={LOSO_PARAMS['num_leaves']}, "
        f"reg_lambda={LOSO_PARAMS['reg_lambda']:.3f}, "
        f"n_estimators={LOSO_PARAMS['n_estimators']}")
    log("  (selected against held-out systems; see config.TRANSFER_PARAMS)")

    df = load_pv()
    log(f"\nrows={len(df):,} systems={df.system_id.nunique()} "
        f"features_base={len(C.BASE_FEATURES)} ensemble={C.ENSEMBLE_FEATURES}")

    # Write each experiment's output as soon as it finishes. The whole suite
    # takes a couple of hours, and an earlier run was killed partway through
    # with everything still in memory - nothing on disk, two hours lost.
    def dump(name, obj):
        obj.to_csv(OUT / f"{name}.csv", index=False)
        with open(C.REPORTS_DIR / "consolidated_report.txt", "w",
                  encoding="utf-8") as fh:
            fh.write("\n".join(LOG))
        log(f"  [saved {name}.csv]")

    if only in (None, "exp0"):
        e0 = exp0_objective_sensitivity(df)
        dump("exp0_objective_sensitivity", e0)

    if only in (None, "exp1"):
        e1 = exp1_oracle(df)
        dump("exp1_oracle", e1)

    if only in (None, "exp2"):
        ac, reg_fc, reg_obs, sse = exp2_decomposition(df)
        for n_, o_ in (("exp2_autocorr", ac),
                       ("exp2_regime_forecast_binned", reg_fc),
                       ("exp2_regime_observed_binned", reg_obs),
                       ("exp2_sse_share", sse)):
            dump(n_, o_)

    if only in (None, "exp3"):
        e3 = exp3_ensemble(df)
        dump("exp3_ensemble_ablation", e3)

    if only in (None, "exp4"):
        e4, recal = exp4_arms_and_recal(df)
        dump("exp4_arms", e4)
        dump("exp4_recalibration", recal)

    if only in (None, "exp5"):
        zs, cb, piv, piv2 = exp5_loso(df)
        dump("exp5_loso_per_system", zs)
        dump("exp5_calibration_sweep", cb)
        dump("exp5_calib_skill_pivot", piv.reset_index())
        dump("exp5_calib_r2_pivot", piv2.reset_index())

    banner("DONE")
    log(f"elapsed {(time.time()-t0)/60:.1f} min")
    log(f"CSVs -> {OUT}")
    with open(C.REPORTS_DIR / "consolidated_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(LOG))
    print(f"\nWrote {C.REPORTS_DIR / 'consolidated_report.txt'}")


if __name__ == "__main__":
    main()
