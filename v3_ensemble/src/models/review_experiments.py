#!/usr/bin/env python3
"""
review_experiments.py
=====================
Experiments added in response to the supervisor's review (paper revision,
November 2026). All run on the timing-corrected build (model_ready_v3.csv
after fix_irradiance_timing.py) with the proposed configuration: base
features + multi-model mean, all 33 systems, TUNED_PARAMS on the
chronological split and TRANSFER_PARAMS (P50) leave-one-system-out.

  ablation      features 10-12 (capacity, tilt, azimuth) and 13 (lagged
                performance ratio) removed, seen and unseen rooftops
  oracles       fleet oracle variants: national (E1), same region, three
                nearest rooftops, capacity-normalised national, and SARAH-3
                irradiance at the rooftop ("perfect weather")
  rolling       seen rooftops scored on four 90-day windows across the year,
                each model trained only on data before its window
  calibstarts   unseen-rooftop local calibration started on four dates across
                the year; the scoring window is fixed (days 60-90 after the
                start), so every calibration length is scored on the same rows
  p10           a P10 model (P25 hyperparameters) and sharpness of the bands

Outputs: v3_ensemble/models/review/*.csv, v3_ensemble/reports/review_<name>.txt

Run:  python src/models/review_experiments.py ablation oracles rolling calibstarts p10
"""

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
sys.path.insert(0, str(Path(__file__).resolve().parent))
import consolidate as CO
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "features"))
import irradiance_timing as T
import pvlib

OUT = C.MODELS_DIR / "review"
OUT.mkdir(parents=True, exist_ok=True)
META_COLS = ["capacity_kwp", "tilt_deg", "azimuth_deg"]
LAGPR = "lag_pr_same_slot_d2"
LOG = []


def log(msg=""):
    print(msg, flush=True)
    LOG.append(str(msg))


def flush(name):
    (C.REPORTS_DIR / f"review_{name}.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    LOG.clear()


def proposed_cols(df):
    return [c for c in C.BASE_FEATURES + C.ENSEMBLE_FEATURES if c in df.columns]


def score(te, P, label, extra=None):
    y = te[C.TARGET].to_numpy()
    r = {"arm": label, "n": len(te),
         "R2": M.r2(y, P[0.50]), "RMSE": M.rmse(y, P[0.50]), "MBE": M.mbe(y, P[0.50]),
         "skill": M.skill_score(y, P[0.50], te.pred_persistence.to_numpy()),
         "mean_pinball": M.mean_pinball(y, P),
         "P25_P90_coverage": M.coverage(y, P[0.25], P[0.90])}
    if extra:
        r.update(extra)
    return r


def loso_p50(df, cols, systems=None, train_mask=None, test_mask=None):
    """Zero-shot P50 per held-out system with the transfer configuration."""
    rows = []
    systems = systems or sorted(df.system_id.unique())
    for sid in systems:
        tr = df[(df.system_id != sid) & (train_mask if train_mask is not None
                                          else df.split.isin(["train", "val"]))]
        te = df[(df.system_id == sid) & (test_mask if test_mask is not None
                                          else df.split == "test")]
        if len(te) < 500:
            continue
        m = lgb.LGBMRegressor(objective="quantile", alpha=0.50, **CO.LOSO_PARAMS).fit(tr[cols], tr[C.TARGET])
        p = np.clip(m.predict(te[cols]), 0, C.CLIP)
        y = te[C.TARGET].to_numpy()
        rows.append({"system_id": sid, "R2": M.r2(y, p), "MBE": M.mbe(y, p),
                     "skill": M.skill_score(y, p, te.pred_persistence.to_numpy())})
    return pd.DataFrame(rows)


def summarise_loso(z, label):
    return {"arm": label, "median_R2": z.R2.median(), "mean_R2": z.R2.mean(),
            "p10_R2": z.R2.quantile(0.10), "worst_R2": z.R2.min(),
            "median_skill": z.skill.median(), "failures": int((z.skill <= 0).sum())}


# ---------------------------------------------------------------------------
def ablation(df):
    log("ABLATION - metadata (features 10-12) and lagged performance ratio (13)")
    full = proposed_cols(df)
    arms = {
        "full (13)": full,
        "- metadata (10-12)": [c for c in full if c not in META_COLS],
        "- lagged PR (13)": [c for c in full if c != LAGPR],
        "- metadata and lagged PR": [c for c in full if c not in META_COLS + [LAGPR]],
    }
    tr, te = df[df.split == "train"], df[df.split == "test"]
    rows, lrows, per = [], [], []
    for label, cols in arms.items():
        P = CO.predict_quantiles(CO.fit_quantiles(tr, cols), te[cols])
        r = score(te, P, label, {"n_features": len(cols)})
        rows.append(r)
        log(f"  seen   {label:28s} R2={r['R2']:.4f} skill={r['skill']:.4f} pinball={r['mean_pinball']:.5f}")
        z = loso_p50(df, cols)
        z["arm"] = label
        per.append(z)
        s = summarise_loso(z, label)
        lrows.append(s)
        log(f"  unseen {label:28s} median R2={s['median_R2']:.4f} mean={s['mean_R2']:.4f} "
            f"p10={s['p10_R2']:.4f} worst={s['worst_R2']:.4f} fail={s['failures']}")
    pd.DataFrame(rows).to_csv(OUT / "ablation_seen.csv", index=False)
    pd.DataFrame(lrows).to_csv(OUT / "ablation_unseen.csv", index=False)
    pd.concat(per).to_csv(OUT / "ablation_unseen_per_system.csv", index=False)


# ---------------------------------------------------------------------------
def sarah_on_grid(df):
    """SARAH-3 GHI (preceding-hour means) mapped to 15-min slot centres."""
    sat = pd.read_csv(C.DATA_INTERIM / "satellite_ghi_all.csv")
    sat["timestamp"] = pd.to_datetime(sat["time"], utc=True)
    meta = pd.read_csv(C.PARENT_META).drop_duplicates("system_id").set_index("system_id")
    out = []
    for sid, g in df.groupby("system_id"):
        s = sat[sat.system_id == sid].drop_duplicates("timestamp").set_index("timestamp").sat_ghi.dropna()
        loc = pvlib.location.Location(float(meta.loc[sid, "latitude"]), float(meta.loc[sid, "longitude"]), tz="UTC")
        cs_h = T.clearsky_hour_means(loc, s.index, ("ghi",))["ghi"]
        centres = T.slot_centres(pd.DatetimeIndex(g.timestamp))
        cs_c = loc.get_clearsky(centres, model="ineichen")["ghi"].to_numpy()
        v = T.hourly_mean_to_instants(s, cs_h, centres, cs_c)
        out.append(pd.DataFrame({"system_id": sid, "timestamp": g.timestamp.to_numpy(),
                                 "sat_ghi_15": v,
                                 "sat_kt_15": np.clip(np.where(cs_c > 5, v / np.where(cs_c > 5, cs_c, 1), 0), 0, 1.5)}))
    return pd.concat(out, ignore_index=True)


def oracles(df):
    log("ORACLE VARIANTS - chronological split, proposed configuration")
    meta = pd.read_csv(C.PARENT_META).drop_duplicates("system_id").set_index("system_id")
    d = df.copy()
    y = C.TARGET
    g = d.groupby("timestamp")[y]
    s, n = g.transform("sum"), g.transform("count")
    d["o_fleet"] = (s - d[y]) / (n - 1)

    # same region
    d["region"] = d.system_id.map(meta["region"])
    gr = d.groupby(["timestamp", "region"])[y]
    sr, nr = gr.transform("sum"), gr.transform("count")
    d["o_region"] = np.where(nr > 1, (sr - d[y]) / (nr - 1).clip(lower=1), np.nan)

    # three nearest rooftops (fixed by distance; mean of those reporting)
    ids = sorted(d.system_id.unique())
    lat = np.radians(meta.loc[ids, "latitude"].to_numpy())
    lon = np.radians(meta.loc[ids, "longitude"].to_numpy())
    dist = 6371 * 2 * np.arcsin(np.sqrt(
        np.sin((lat[:, None] - lat[None]) / 2) ** 2
        + np.cos(lat[:, None]) * np.cos(lat[None]) * np.sin((lon[:, None] - lon[None]) / 2) ** 2))
    wide = d.pivot_table(index="timestamp", columns="system_id", values=y)
    near = {}
    for i, sid in enumerate(ids):
        nn = [ids[j] for j in np.argsort(dist[i]) if ids[j] != sid][:3]
        near[sid] = wide[nn].mean(axis=1)
        log(f"  {sid}: nearest {nn} at {np.sort(dist[i])[1:4].round(1)} km")
    near_long = (pd.DataFrame(near).rename_axis("timestamp").reset_index()
                 .melt(id_vars="timestamp", var_name="system_id", value_name="o_near3"))
    near_long["system_id"] = near_long.system_id.astype(d.system_id.dtype)
    d = d.merge(near_long, on=["timestamp", "system_id"], how="left")

    # capacity-normalised national oracle: each rooftop scaled by its own
    # training-period P99 of normalised output, which removes nameplate error
    q99 = d[d.split == "train"].groupby("system_id")[y].quantile(0.99)
    d["_z"] = d[y] / d.system_id.map(q99)
    gz = d.groupby("timestamp")["_z"]
    d["o_fleet_capnorm"] = (gz.transform("sum") - d["_z"]) / (n - 1)

    # SARAH-3 at the rooftop
    sg = sarah_on_grid(d[["system_id", "timestamp"]])
    d = d.merge(sg, on=["system_id", "timestamp"], how="left")

    keep = (n.to_numpy() > CO.MIN_FLEET)
    d = d[keep]
    d = d.dropna(subset=["o_region", "o_near3", "sat_ghi_15"])
    log(f"  rows with every oracle defined: {len(d):,} (systems {d.system_id.nunique()})")
    base = proposed_cols(d)
    geo = ["sun_elevation", "clearsky_ghi"] + META_COLS
    arms = {
        "baseline (NWP)": base,
        "+ fleet oracle (national)": base + ["o_fleet"],
        "+ capacity-normalised fleet oracle": base + ["o_fleet_capnorm"],
        "+ same-region oracle": base + ["o_region"],
        "+ three-nearest oracle": base + ["o_near3"],
        "+ SARAH-3 at rooftop": base + ["sat_ghi_15", "sat_kt_15"],
        "fleet oracle + geometry, no NWP": ["o_fleet"] + geo,
        "three-nearest oracle + geometry, no NWP": ["o_near3"] + geo,
        "SARAH-3 + geometry, no NWP": ["sat_ghi_15", "sat_kt_15"] + geo,
    }
    tr, te = d[d.split == "train"], d[d.split == "test"]
    rows = []
    for label, cols in arms.items():
        P = CO.predict_quantiles(CO.fit_quantiles(tr, cols), te[cols])
        r = score(te, P, label, {"n_features": len(cols)})
        rows.append(r)
        log(f"  {label:42s} R2={r['R2']:.4f} skill={r['skill']:.4f} pinball={r['mean_pinball']:.5f}")
    pd.DataFrame(rows).to_csv(OUT / "oracle_variants.csv", index=False)


# ---------------------------------------------------------------------------
WINDOWS = ["2025-05-01", "2025-08-01", "2025-11-01", "2026-02-15"]


def rolling(df):
    log("ROLLING ORIGIN - seen rooftops, 90-day windows, trained only on earlier data")
    base = [c for c in C.BASE_FEATURES if c in df.columns]
    rows = []
    for s in WINDOWS:
        start = pd.Timestamp(s, tz=C.TZ).tz_convert("UTC")
        end = min(start + pd.Timedelta(days=90), df.timestamp.max() + pd.Timedelta(minutes=15))
        tr = df[df.timestamp < start - pd.Timedelta(days=1)]
        te = df[(df.timestamp >= start) & (df.timestamp < end)]
        for label, cols in (("base", base), ("proposed", proposed_cols(df))):
            P = CO.predict_quantiles(CO.fit_quantiles(tr, cols), te[cols])
            r = score(te, P, label, {"window_start": s, "window_end": str(end.date()),
                                     "train_rows": len(tr), "test_rows": len(te),
                                     "persistence_R2": M.r2(te[C.TARGET].to_numpy(), te.pred_persistence.to_numpy())})
            rows.append(r)
            log(f"  {s} -> {end.date()} {label:9s} R2={r['R2']:.4f} skill={r['skill']:.4f} "
                f"pinball={r['mean_pinball']:.5f} (train {len(tr):,}, test {len(te):,})")
    pd.DataFrame(rows).to_csv(OUT / "rolling_origin.csv", index=False)


CAL_N = [0, 7, 14, 30, 45, 60]


def calibstarts(df):
    log("CALIBRATION START DATES - unseen rooftops, fixed scoring window days 60-90")
    cols = proposed_cols(df)
    rows = []
    for s in WINDOWS:
        start = pd.Timestamp(s, tz=C.TZ).tz_convert("UTC")
        sc0, sc1 = start + pd.Timedelta(days=60), start + pd.Timedelta(days=90)
        for sid in sorted(df.system_id.unique()):
            tr = df[(df.system_id != sid) & (df.timestamp < start - pd.Timedelta(days=1))]
            te = df[(df.system_id == sid) & (df.timestamp >= start) & (df.timestamp < sc1)].copy()
            ev = te[te.timestamp >= sc0]
            if len(ev) < 200:
                continue
            fleet_ratio = float(np.median([
                np.percentile(g["target_power_w"], 99) / 1000.0 / g["capacity_kwp"].iloc[0]
                for _, g in tr.groupby("system_id")]))
            m = lgb.LGBMRegressor(objective="quantile", alpha=0.50, **CO.LOSO_PARAMS).fit(tr[cols], tr[C.TARGET])
            te["pred"] = np.clip(m.predict(te[cols]), 0, C.CLIP)
            ev = te[te.timestamp >= sc0]
            ye = ev[C.TARGET].to_numpy()
            cap = float(te["capacity_kwp"].iloc[0])
            for nd in CAL_N:
                cal = te[(te.timestamp < start + pd.Timedelta(days=nd))] if nd else te.iloc[:0]
                for name, p in CO.corrections(cal, ev, cap, fleet_ratio).items():
                    rows.append({"start": s, "system_id": sid, "calib_days": nd, "method": name,
                                 "R2": M.r2(ye, p),
                                 "skill": M.skill_score(ye, p, ev.pred_persistence.to_numpy())})
            log(f"  {s} {sid} done")
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "calibration_starts.csv", index=False)
    piv = r.pivot_table(index=["start", "calib_days"], columns="method", values="skill", aggfunc="median")
    log("\nmedian skill on the fixed scoring window\n" + piv.round(4).to_string())
    piv.reset_index().to_csv(OUT / "calibration_starts_pivot.csv", index=False)


# ---------------------------------------------------------------------------
def p10(df):
    log("P10 AND SHARPNESS - proposed configuration, chronological split")
    cols = proposed_cols(df)
    tr, va, te = df[df.split == "train"], df[df.split == "val"], df[df.split == "test"]
    params = dict(C.TUNED_PARAMS)
    params[0.10] = C.TUNED_PARAMS[0.25]          # no separate search for P10
    qs = [0.10, 0.25, 0.50, 0.90]
    models = {q: lgb.LGBMRegressor(objective="quantile", alpha=q, **params[q]).fit(tr[cols], tr[C.TARGET])
              for q in qs}
    y, yv = te[C.TARGET].to_numpy(), va[C.TARGET].to_numpy()
    rows = []
    for tag in ("raw", "recalibrated"):
        P = {q: np.clip(m.predict(te[cols]), 0, C.CLIP) for q, m in models.items()}
        if tag == "recalibrated":
            for q, m in models.items():
                dq = float(np.quantile(yv - np.clip(m.predict(va[cols]), 0, C.CLIP), q))
                P[q] = np.clip(P[q] + dq, 0, C.CLIP)
        for q in qs:
            rows.append({"version": tag, "quantile": q, "coverage": float(np.mean(y <= P[q])),
                         "pinball": M.pinball_loss(y, P[q], q)})
        rows.append({"version": tag, "quantile": "width P25-P90", "coverage": float(np.mean(P[0.90] - P[0.25]))})
        rows.append({"version": tag, "quantile": "width P10-P90", "coverage": float(np.mean(P[0.90] - P[0.10]))})
        rows.append({"version": tag, "quantile": "P10-P90 coverage", "coverage": M.coverage(y, P[0.10], P[0.90])})
    r = pd.DataFrame(rows)
    log(r.to_string(index=False))
    r.to_csv(OUT / "p10_sharpness.csv", index=False)


def main():
    todo = sys.argv[1:] or ["ablation", "oracles", "rolling", "calibstarts", "p10"]
    t0 = time.time()
    df = CO.load_pv()
    log(f"rows={len(df):,} systems={df.system_id.nunique()}")
    for name in todo:
        globals()[name](df)
        log(f"[{name}] elapsed {(time.time()-t0)/60:.1f} min")
        flush(name)


if __name__ == "__main__":
    main()
