#!/usr/bin/env python3
"""
loso_mos.py
===========
Does MOS-corrected irradiance help on a rooftop the model has NEVER SEEN?

WHY THIS IS THE INTERESTING CASE
--------------------------------
On the temporal split, satellite-trained MOS improved irradiance substantially
(-6.1% MAE, bias +11.40 -> +3.12 W/m2) and yet produced ZERO gain in the PV
forecast (R2 0.80874 -> 0.80890). The explanation: ECMWF's systematic bias is
present in both train and test, so a gradient-boosted tree simply learns split
thresholds that absorb it. MOS removed a component the model had already
neutralised for itself.

That explanation makes a falsifiable prediction. On an UNSEEN rooftop the model
cannot have absorbed that roof's particular bias, so removing it up front should
matter more. This script tests that prediction.

If MOS helps here but not on the temporal split, the story becomes precise and
useful: post-processing is redundant for roofs in the training fleet, and
valuable for new deployments - which is exactly the product case.

LEAKAGE - TWO SEPARATE CONCERNS, HANDLED SEPARATELY
---------------------------------------------------
1. TEMPORAL. MOS must never see the rows it corrects. Training rows get 5-fold
   chronological out-of-fold corrections; val/test rows are corrected by a MOS
   fitted on the train split only.

2. SPATIAL. For the scored rows - the held-out roof's test window - MOS is
   REFITTED with that system excluded entirely, then applied to it. So the
   correction reaching the evaluated predictions comes from a MOS that has
   never seen that location.

   Worth noting the second guard is conservative. MOS consumes only NWP fields,
   solar geometry and satellite irradiance - no PV generation data whatsoever -
   so it cannot learn anything about a roof's behaviour. But refitting removes
   the objection entirely, and costs one extra fit per fold.

ARMS
----
  A  baseline       : best v3 config, raw ECMWF irradiance
  B  MOS-corrected  : ghi_fc and poa_global_fc replaced by corrected versions

Both arms then get the same calibration sweep as loso_v3.py, so the question
"does MOS reduce how much local data a new roof needs?" is also answered.

Outputs: v3_ensemble/models/loso_mos_per_system.csv
         v3_ensemble/models/loso_mos_calibration.csv
         v3_ensemble/reports/loso_mos_report.txt
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as C
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval import metrics as M

from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb

MOS_FEATURES = [
    "ecmwf_ghi", "ecmwf_cloud", "ecmwf_temp",
    "clearsky_ghi", "kt_fc", "sun_elevation",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos",
]
EPS = 1.0
N_FOLDS = 5
CALIB_DAYS = [0, 3, 7, 14, 30]
# transfer configuration - selected against held-out systems, matching the
# LOSO protocol in consolidate.py (see config.TRANSFER_PARAMS)
LOSO_PARAMS = dict(C.TRANSFER_PARAMS)


def mos_model(n_estimators=120):
    return RandomForestRegressor(n_estimators=n_estimators, max_depth=12,
                                 min_samples_leaf=20, random_state=42, n_jobs=-1)


def build_mos_table():
    mos = pd.read_csv(C.PROJECT_ROOT / "data" / "interim" / "mos_training_data.csv")
    mos["timestamp"] = pd.to_datetime(mos["timestamp"], utc=True)
    mos = mos.dropna(subset=["system_id", "ecmwf_ghi"])
    mos["system_id"] = mos["system_id"].astype(int)
    mos = mos[~mos.system_id.isin(C.EXCLUDE_SYSTEMS)]

    sat = pd.read_csv(C.DATA_INTERIM / "satellite_ghi_all.csv")
    sat["timestamp"] = pd.to_datetime(sat["time"], utc=True)
    sat = sat[["system_id", "timestamp", "sat_ghi"]].dropna(subset=["sat_ghi"])

    geo = pd.read_csv(C.MODEL_READY_V3, usecols=[
        "system_id", "timestamp", "clearsky_ghi", "sun_elevation",
        "hour_sin", "hour_cos", "doy_sin", "doy_cos", "split"])
    geo["timestamp"] = pd.to_datetime(geo["timestamp"], utc=True)

    d = mos.merge(sat, on=["system_id", "timestamp"], how="inner")
    d = d.merge(geo, on=["system_id", "timestamp"], how="inner")
    d = d[d.sun_elevation > 5].reset_index(drop=True)
    d["kt_fc"] = (d.ecmwf_ghi / (d.clearsky_ghi + EPS)).clip(0, 1.3)
    d["error_target"] = d.sat_ghi - d.ecmwf_ghi
    return d.sort_values("timestamp").reset_index(drop=True)


def ratio_from(ghi_mos, ecmwf_ghi):
    r = np.where(ecmwf_ghi > 5.0, ghi_mos / np.clip(ecmwf_ghi, 5.0, None), 1.0)
    return np.clip(r, 0.2, 3.0)


def oof_corrections(d):
    """Temporally honest corrections for every row (guard 1)."""
    d = d.copy()
    d["ghi_mos"] = np.nan
    tr = d[d.split == "train"].sort_values("timestamp")
    for k, hold in enumerate(np.array_split(tr.index.to_numpy(), N_FOLDS), 1):
        fit_idx = tr.index.difference(pd.Index(hold))
        m = mos_model().fit(d.loc[fit_idx, MOS_FEATURES], d.loc[fit_idx, "error_target"])
        d.loc[hold, "ghi_mos"] = (d.loc[hold, "ecmwf_ghi"].to_numpy()
                                  + m.predict(d.loc[hold, MOS_FEATURES]))
        print(f"   oof fold {k}/{N_FOLDS}", flush=True)
    m_full = mos_model().fit(tr[MOS_FEATURES], tr["error_target"])
    rest = d.index[d.split.isin(["val", "test"])]
    d.loc[rest, "ghi_mos"] = (d.loc[rest, "ecmwf_ghi"].to_numpy()
                              + m_full.predict(d.loc[rest, MOS_FEATURES]))
    d["ghi_mos"] = d["ghi_mos"].clip(lower=0)
    d["mos_ratio"] = ratio_from(d.ghi_mos.to_numpy(), d.ecmwf_ghi.to_numpy())
    return d


def system_excluded_ratio(d, sid):
    """Guard 2: refit MOS without system `sid`, correct that system's rows."""
    fit = d[(d.split == "train") & (d.system_id != sid)]
    m = mos_model().fit(fit[MOS_FEATURES], fit["error_target"])
    g = d[d.system_id == sid]
    if g.empty:
        return None
    ghi_mos = np.clip(g.ecmwf_ghi.to_numpy() + m.predict(g[MOS_FEATURES]), 0, None)
    out = g[["system_id", "timestamp"]].copy()
    out["mos_ratio"] = ratio_from(ghi_mos, g.ecmwf_ghi.to_numpy())
    return out


def apply_ratio_to_grid(pv, ratio_hourly):
    """Interpolate an hourly correction ratio onto the 15-minute grid."""
    blocks = []
    for sid, g in pv.groupby("system_id", sort=True):
        r = ratio_hourly[ratio_hourly.system_id == sid]
        g = g.set_index("timestamp").sort_index()
        if r.empty:
            g["mos_ratio"] = 1.0
        else:
            s = r.set_index("timestamp")["mos_ratio"].sort_index()
            s = s[~s.index.duplicated()]
            g["mos_ratio"] = (s.reindex(s.index.union(g.index))
                                .interpolate("time", limit_direction="both")
                                .reindex(g.index).to_numpy())
        blocks.append(g.reset_index())
    out = pd.concat(blocks, ignore_index=True)
    out["mos_ratio"] = out["mos_ratio"].fillna(1.0).clip(0.2, 3.0)
    out["ghi_fc_mos"] = out["ghi_fc"] * out["mos_ratio"]
    out["poa_global_fc_mos"] = out["poa_global_fc"] * out["mos_ratio"]
    return out


def load_pv():
    pv = pd.read_csv(C.MODEL_READY_V3)
    pv["timestamp"] = pd.to_datetime(pv["timestamp"], utc=True)
    pv = pv[~pv.system_id.isin(C.EXCLUDE_SYSTEMS)]
    pv = pv.sort_values(["system_id", "timestamp"]).reset_index(drop=True)
    prev = pv[["system_id", "timestamp", C.TARGET]].copy()
    prev["timestamp"] += pd.Timedelta(days=1)
    prev = prev.rename(columns={C.TARGET: "pred_persistence"})
    pv = pv.merge(prev, on=["system_id", "timestamp"], how="left")
    pv = pv[(pv.is_valid == True) & (pv.is_daylight == True)].copy()
    pv["pred_persistence"] = pv["pred_persistence"].fillna(0.0)
    pv["day"] = pv["timestamp"].dt.normalize()
    return pv.fillna(0)


def corrections(cal, ev, cap_kwp, fleet_cap_ratio):
    out = {"none": ev["pred"].to_numpy()}
    if len(cal) < 10:
        return out
    resid = (cal[C.TARGET] - cal["pred"]).mean()
    out["bias"] = np.clip(ev["pred"].to_numpy() + resid, 0, C.CLIP)
    if cal["pred"].std() > 1e-6:
        b, a = np.polyfit(cal["pred"], cal[C.TARGET], 1)
        out["linear"] = np.clip(a + b * ev["pred"].to_numpy(), 0, C.CLIP)
    obs_kw = np.percentile(cal["target_power_w"], 99) / 1000.0
    if obs_kw > 0.1 and cap_kwp > 0.1 and fleet_cap_ratio > 0.1:
        scale = float(np.clip((obs_kw / cap_kwp) / fleet_cap_ratio, 0.4, 2.5))
        resid2 = (cal[C.TARGET] - cal["pred"] * scale).mean()
        out["cap_emp+bias"] = np.clip(ev["pred"].to_numpy() * scale + resid2, 0, C.CLIP)
    return out


def main():
    print("Building MOS table ...", flush=True)
    d = build_mos_table()
    print(f"  rows={len(d):,} systems={d.system_id.nunique()}", flush=True)

    print("Out-of-fold corrections (guard 1) ...", flush=True)
    d = oof_corrections(d)

    pv_raw = load_pv()
    base = [c for c in C.BASE_FEATURES if c in pv_raw.columns]
    ens = [c for c in C.ENSEMBLE_FEATURES if c in pv_raw.columns]
    feats = base + ens
    swapped = ["ghi_fc_mos" if c == "ghi_fc" else
               "poa_global_fc_mos" if c == "poa_global_fc" else c for c in feats]

    print("Applying OOF ratio to the 15-minute grid ...", flush=True)
    pv_oof = apply_ratio_to_grid(pv_raw, d[["system_id", "timestamp", "mos_ratio"]])

    systems = sorted(pv_raw.system_id.unique())
    print(f"\nLOSO over {len(systems)} systems\n", flush=True)

    zero_rows, calib_rows = [], []
    for i, sid in enumerate(systems, 1):
        te_raw = pv_oof[(pv_oof.system_id == sid) & (pv_oof.split == "test")]
        if len(te_raw) < 500:
            print(f"[{i}/{len(systems)}] {sid} skipped")
            continue

        # guard 2: MOS refitted without this system, used for its scored rows
        r_ex = system_excluded_ratio(d, sid)
        te_ex = apply_ratio_to_grid(
            pv_raw[(pv_raw.system_id == sid) & (pv_raw.split == "test")],
            r_ex if r_ex is not None else pd.DataFrame(columns=["system_id", "timestamp", "mos_ratio"]))

        tr = pv_oof[(pv_oof.system_id != sid) & (pv_oof.split.isin(["train", "val"]))]
        fleet_cap_ratio = float(np.median([
            np.percentile(g["target_power_w"], 99) / 1000.0 / g["capacity_kwp"].iloc[0]
            for _, g in tr.groupby("system_id")]))

        row = {"system_id": sid, "n": len(te_raw)}
        per_arm_te = {}
        for arm, cols, te in (("A_baseline", feats, te_raw),
                              ("B_mos", swapped, te_ex)):
            m = lgb.LGBMRegressor(objective="quantile", alpha=0.50,
                                  **LOSO_PARAMS).fit(tr[cols], tr[C.TARGET])
            t = te.copy()
            t["pred"] = np.clip(m.predict(t[cols]), 0, C.CLIP)
            y = t[C.TARGET].to_numpy()
            row[f"{arm}_R2"] = M.r2(y, t["pred"])
            row[f"{arm}_RMSE"] = M.rmse(y, t["pred"])
            row[f"{arm}_MBE"] = M.mbe(y, t["pred"])
            row[f"{arm}_skill"] = M.skill_score(y, t["pred"], t["pred_persistence"].to_numpy())
            per_arm_te[arm] = t

        zero_rows.append(row)
        print(f"[{i}/{len(systems)}] {sid}  baseline R2={row['A_baseline_R2']:.4f}  "
              f"MOS R2={row['B_mos_R2']:.4f}  "
              f"delta={row['B_mos_R2'] - row['A_baseline_R2']:+.4f}", flush=True)

        for arm, t in per_arm_te.items():
            days = np.sort(t["day"].unique())
            for nd in CALIB_DAYS:
                if nd == 0:
                    cal, ev = t.iloc[:0], t
                else:
                    if len(days) <= nd + 5:
                        continue
                    cut = days[nd]
                    cal, ev = t[t.day < cut], t[t.day >= cut]
                if len(ev) < 200:
                    continue
                ye = ev[C.TARGET].to_numpy()
                for name, p in corrections(cal, ev, float(t["capacity_kwp"].iloc[0]),
                                           fleet_cap_ratio).items():
                    calib_rows.append({
                        "system_id": sid, "arm": arm, "calib_days": nd,
                        "method": name, "R2": M.r2(ye, p),
                        "skill": M.skill_score(ye, p, ev["pred_persistence"].to_numpy()),
                    })

    zs = pd.DataFrame(zero_rows)
    cb = pd.DataFrame(calib_rows)
    zs.to_csv(C.MODELS_DIR / "loso_mos_per_system.csv", index=False)
    cb.to_csv(C.MODELS_DIR / "loso_mos_calibration.csv", index=False)

    zs["delta_R2"] = zs.B_mos_R2 - zs.A_baseline_R2
    zs["delta_skill"] = zs.B_mos_skill - zs.A_baseline_skill

    print("\n" + "=" * 74)
    print("ZERO-SHOT ON AN UNSEEN ROOF: does MOS-corrected irradiance help?")
    print("=" * 74)
    print(zs[["system_id", "A_baseline_R2", "B_mos_R2", "delta_R2",
              "A_baseline_skill", "B_mos_skill", "delta_skill"]]
          .sort_values("delta_R2").to_string(index=False))
    print(f"\nmedian R2   : baseline {zs.A_baseline_R2.median():.4f}  "
          f"MOS {zs.B_mos_R2.median():.4f}  delta {zs.delta_R2.median():+.4f}")
    print(f"mean   R2   : baseline {zs.A_baseline_R2.mean():.4f}  "
          f"MOS {zs.B_mos_R2.mean():.4f}  delta {zs.delta_R2.mean():+.4f}")
    print(f"median skill: baseline {zs.A_baseline_skill.median():.4f}  "
          f"MOS {zs.B_mos_skill.median():.4f}  delta {zs.delta_skill.median():+.4f}")
    print(f"systems improved by MOS: {(zs.delta_R2 > 0).sum()} / {len(zs)}")
    print(f"median |MBE|: baseline {zs.A_baseline_MBE.abs().median():.4f}  "
          f"MOS {zs.B_mos_MBE.abs().median():.4f}")

    print("\n" + "=" * 74)
    print("CALIBRATION SWEEP - does MOS reduce the data a new roof needs?")
    print("=" * 74)
    piv = cb.pivot_table(index="calib_days", columns=["arm", "method"],
                         values="skill", aggfunc="median")
    print("\nmedian skill vs persistence")
    print(piv.round(4).to_string())

    with open(C.REPORTS_DIR / "loso_mos_report.txt", "w", encoding="utf-8") as f:
        f.write("LOSO with MOS-corrected irradiance\n" + "=" * 74 + "\n\n")
        f.write(zs.to_string(index=False) + "\n\n")
        f.write("median skill by calibration days x arm x method\n")
        f.write(piv.round(4).to_string() + "\n")
    print(f"\nWrote {C.MODELS_DIR / 'loso_mos_per_system.csv'}")
    print(f"Wrote {C.REPORTS_DIR / 'loso_mos_report.txt'}")


if __name__ == "__main__":
    main()
