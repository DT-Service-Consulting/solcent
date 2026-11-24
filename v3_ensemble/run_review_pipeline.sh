#!/usr/bin/env bash
# Re-derives every result after the two review fixes (irradiance timing,
# day-ahead reference/MOS table) and runs the review experiments.
# Prerequisites (already run): fix_irradiance_timing.py, fetch_era5.py,
# fetch_sarah_coarse.py, build_features_v3.py.
# Usage: bash v3_ensemble/run_review_pipeline.sh   (from the repository root)
set -u
PY=".venv/Scripts/python.exe -W ignore"
V=v3_ensemble
LOG=$V/reports/review_pipeline.log
step() {
  echo "=== $(date '+%F %T') START $*" | tee -a "$LOG"
  if $PY "$@" >> "$LOG" 2>&1; then
    echo "=== $(date '+%F %T') OK    $*" | tee -a "$LOG"
  else
    echo "=== $(date '+%F %T') FAIL  $*" | tee -a "$LOG"
  fi
}
step $V/src/features/mos_table.py --force
step $V/src/models/consolidate.py --full-fleet --save-preds
step $V/src/eval/residual_autocorr.py
step $V/src/models/mos_satellite.py
step $V/src/eval/reference_comparison.py
step $V/src/eval/reference_pv_corr.py
step $V/src/eval/reference_combined.py
step $V/src/eval/weather_error_regime.py
step $V/src/models/mos_pv_integration.py --full-fleet --save-preds
step $V/src/models/mos_pv_integration.py --save-preds
step $V/src/models/loso_mos.py
step $V/src/models/baselines.py
step $V/src/eval/grid_aggregate.py
step $V/src/eval/revision_stats.py
step $V/src/eval/error_distance.py
step $V/src/models/review_experiments.py p10 ablation oracles rolling calibstarts
step $V/src/models/loso_seeds.py --configs transfer --seeds 0,1,2,3,4
step $V/src/models/loso_seeds.py --configs temporal,preliminary --seeds 0,1,2
# revision_stats reads the seed results, so it runs again at the end
step $V/src/eval/revision_stats.py
echo "=== $(date '+%F %T') PIPELINE DONE" | tee -a "$LOG"
