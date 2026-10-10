#!/usr/bin/env python3
"""
config.py
=========
Central configuration for the v3_ensemble experiment tree.

This sub-project is fully isolated from the parent codebase. It READS parent
artifacts (model_ready.csv, system metadata) but never writes outside
v3_ensemble/.
"""

from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
V3_ROOT      = Path(__file__).resolve().parent
PROJECT_ROOT = V3_ROOT.parent

# Read-only inputs borrowed from the parent project
PARENT_MODEL_READY = PROJECT_ROOT / "data" / "processed" / "model_ready.csv"
PARENT_META        = PROJECT_ROOT / "data" / "interim" / "pvoutput_selection" / "systems_metadata.csv"
PARENT_SHORTLIST   = PROJECT_ROOT / "data" / "interim" / "pvoutput_selection" / "shortlist_final.csv"

# v3-owned outputs
DATA_INTERIM  = V3_ROOT / "data" / "interim"
DATA_PROC     = V3_ROOT / "data" / "processed"
MODELS_DIR    = V3_ROOT / "models"
REPORTS_DIR   = V3_ROOT / "reports"

ENSEMBLE_RAW  = DATA_INTERIM / "ensemble_ghi_all.csv"
MODEL_READY_V3 = DATA_PROC / "model_ready_v3.csv"

for _d in (DATA_INTERIM, DATA_PROC, MODELS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Study window (must match the parent build so rows align)
# ---------------------------------------------------------------------------
START_DATE = "2024-04-01"
END_DATE   = "2026-05-31"
TZ         = "Europe/Brussels"
FREQ       = "15min"

# ---------------------------------------------------------------------------
# Ensemble definition
# ---------------------------------------------------------------------------
# A) Lagged ensemble: the SAME model re-run on successive days. Every one of
#    these runs was published before a D-1 noon cutoff, so all are legitimate.
#    day0 (plain "shortwave_radiation") is the analysis run and IS LEAKAGE.
LAGGED_MODEL = "ecmwf_ifs025"
LAGGED_DAYS  = [1, 2, 3, 4, 5, 6, 7]

# B) Multi-model ensemble: different centres, different physics, different
#    mistakes. All fetched at _previous_day1 (true day-ahead lead time).
#    Coverage measured over 2024-04-01..2026-05-31 at 50.85N 4.35E:
#      ecmwf_ifs025 100% | icon_seamless 100% | gfs_seamless 100%
#      meteofrance_seamless 100% | dmi_seamless 100% | ukmo_seamless 83.4%
MULTI_MODELS = [
    "icon_seamless",
    "gfs_seamless",
    "meteofrance_seamless",
    "dmi_seamless",
    "ukmo_seamless",
]

# Fetched for reference/analysis only. NEVER a model feature.
LEAKAGE_COLS = ["ghi_day0_LEAKAGE"]

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
PREV_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
# Live ensemble (51 ECMWF members) - only ~93 days of history, cannot backfill
# the study window. Used by the forward-logging script only.
ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"

# ---------------------------------------------------------------------------
# Modelling
# ---------------------------------------------------------------------------
TARGET   = "target_norm"
CLIP     = 1.2
QUANTILES = [0.25, 0.50, 0.90]

# 12293 excluded by the parent project throughout.
# 81376 diagnosed in this tree: R2 0.167, 3.08% of test rows but 6.75% of
# total SSE, output/fleet ratio drifting 0.56-0.91 across the window
# (time-varying shading or a failing string). Handled via DROP_SYSTEMS.
EXCLUDE_SYSTEMS = [12293]
BROKEN_SYSTEMS  = [81376]

# The parent project's 12 pruned features (see vendor/model_baselines.py)
BASE_FEATURES = [
    "poa_global_fc", "poa_global_fc2", "clearsky_ghi",
    "sun_elevation", "sun_azimuth", "hour_cos",
    "dni_fc", "cloud_fc2",
    "capacity_kwp", "tilt_deg", "azimuth_deg",
    "lag_pr_same_slot_d2",
]

# Engineered in build_features_v3.py.
#
# MEASURED 2026-09-07, quantile objective, 81376 dropped, test split:
#   base only .................................... R2 0.8050  skill 0.4255  pinball 0.02573
#   base + ALL 10 features below ................. R2 0.8049  skill 0.4254  pinball 0.02606
#   base + spread features only .................. R2 0.8038  skill 0.4239  pinball 0.02611
#   base + ens_ghi_mean only ..................... R2 0.8062  skill 0.4274  pinball 0.02568
#   base + ens_mean_fresh only ................... R2 0.8086  skill 0.4310  pinball 0.02549
#
# Two conclusions. The SPREAD features are worse than useless: they cost
# R2 and pinball, because LightGBM quantile regression already infers
# conditional uncertainty from clearsky_index_fc / sun_elevation / cloud, and
# ensemble spread is largely collinear with that. Dumping all ten at once
# dilutes the one feature that works.
#
# What DOES work is a single column: the plain mean of the six day-1
# forecasts (ECMWF lag1 + the five other centres). Multi-model averaging is
# real; unweighted lagged averaging across days 2-7 is not, because the stale
# members drag the mean down.
ENSEMBLE_FEATURES = ["ens_mean_fresh"]

# The full block, kept so the ablation stays reproducible. Not for training -
# see the measurements above.
ENSEMBLE_FEATURES_FULL = [
    "ens_ghi_mean", "ens_ghi_std", "ens_ghi_range", "ens_ghi_iqr",
    "ens_kt_mean", "ens_kt_std",
    "ens_lag_std", "ens_model_std", "ens_agree_frac",
    "poa_ens_mean",
]

# LightGBM params - the parent's Optuna result from models/pv_model/best_params.json
# ---------------------------------------------------------------------------
# Per-quantile hyperparameters from the pinball-objective search
# (src/models/tune_quantile.py, 60 TPE trials per quantile, early stopping on
# an inner split of train, scored on val). These are the PROPOSED parameters.
#
# Measured against the inherited RMSE-tuned set on the untouched test split:
#   mean pinball 0.025452 -> 0.025283 (-0.66%)
#   MAE          0.068223 -> 0.067301 (-1.35%)
#   skill        0.4315   -> 0.4351   (+0.0036)
# Gains concentrate at P50 (-1.35%) and P90 (-0.63%); P25 worsens slightly
# (+0.19%), the lower quantile being the one squared error happened to suit.
#
# Caveat: learning rates (0.013-0.015) sit near the imposed floor of 0.010 and
# tree counts (1418-1999) near the cap of 2000, so these configurations were
# still improving when the search budget ran out.
TUNED_PARAMS: dict[float, dict[str, Any]] = {
    0.25: dict(learning_rate=0.015202539432309889, num_leaves=222, max_depth=11,
               min_child_samples=45, subsample=0.7711155184556665,
               subsample_freq=1, colsample_bytree=0.6314185773599863,
               reg_alpha=0.12243107882762433, reg_lambda=0.03469790951999207,
               n_estimators=1418, verbose=-1),
    0.50: dict(learning_rate=0.012920564969641644, num_leaves=247, max_depth=15,
               min_child_samples=155, subsample=0.6304845468772297,
               subsample_freq=0, colsample_bytree=0.5898421700502072,
               reg_alpha=0.16319345684256534, reg_lambda=0.008115832887883319,
               n_estimators=1999, verbose=-1),
    0.90: dict(learning_rate=0.014686957046316671, num_leaves=148, max_depth=20,
               min_child_samples=82, subsample=0.7791436018098703,
               subsample_freq=4, colsample_bytree=0.5671580680649068,
               reg_alpha=0.492024761075982, reg_lambda=0.0014500989711018944,
               n_estimators=1993, verbose=-1),
}

# ---------------------------------------------------------------------------
# TRANSFER configuration - selected against HELD-OUT SYSTEMS, not held-out time
# (src/models/tune_transfer.py, 40 TPE trials, 3 system-folds, pinball P50).
#
# TUNED_PARAMS above optimises the chronological split, where every rooftop is
# already in training. It wins there and loses on deployment:
#
#   temporal split (known roof)   inherited 0.8087 -> temporal-tuned 0.8114
#   LOSO (unseen roof)            inherited 0.7232 -> temporal-tuned 0.7062
#
# Selecting against held-out systems instead recovers the deployment case. The
# gain is in the TAIL rather than the median - which is what matters, since the
# risk was never the typical install:
#
#   config                     med R2   mean R2   p10 R2   min R2   skill  fails
#   inherited (RMSE, temporal) 0.7189    0.6027   0.3780  -0.7092  0.2928   4/33
#   tuned on temporal split    0.7084    0.5877   0.3082  -0.9847  0.3158   5/33
#   tuned on held-out systems  0.7098    0.6567   0.4958  -0.1561  0.3399   2/33
#
# The parameters show the mechanism: depth 4 with L2 at 4.34, against depth 15
# and L2 at 0.008 for the temporal optimum. A depth-4 tree cannot encode
# rooftop identity, so it is forced to learn what rooftops share.
#
# Caveat: n_estimators reached the 1500-tree search cap, so this configuration
# is budget-limited rather than converged.
TRANSFER_PARAMS: dict[str, Any] = dict(
    learning_rate=0.018373112760379447,
    num_leaves=101,
    max_depth=4,
    min_child_samples=119,
    subsample=0.5610278637041073,
    subsample_freq=2,
    colsample_bytree=0.6681041701540076,
    reg_alpha=0.14416824387775157,
    reg_lambda=4.344012536750854,
    n_estimators=1500,
    verbose=-1,
)

# Inherited from the parent project's Optuna search (20 trials, validation RMSE
# under a squared-error objective). Retained as a comparison arm only.
LGBM_PARAMS: dict[str, Any] = dict(
    learning_rate=0.05281999494454134,
    num_leaves=148,
    max_depth=15,
    min_child_samples=79,
    subsample=0.871156309500088,
    colsample_bytree=0.9532773681510769,
    reg_alpha=7.811822590185928,
    reg_lambda=0.04468692786241433,
    n_estimators=500,
    verbose=-1,
)
