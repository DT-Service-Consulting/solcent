# Solcent — Day-Ahead Solar Forecasting for Belgium (Approach 1)

Probabilistic, capacity-normalized **day-ahead** solar generation forecasting for the
Belgian grid via NWP post-processing. One forecast per day at the gate-closure issue
time, covering all 96 quarter-hour slots of the next day (single-shot, whole-day).

See the companion documents: *Project Scope v1.1* and *Implementation Plan v1.2*.

## Repository layout
```
solar-dayahead/
  configs/        protocol.yaml, features.yaml, model.yaml
  src/
    ingest/       elia.py, openmeteo.py
    dataset/      build_daily_blocks.py
    features/     clearsky.py, calendar.py, lags.py, nwp.py, daylight.py, sat.py
    models/       baselines.py, gbm.py, quantile.py
    eval/         metrics.py, plots.py, calibration.py
    pipeline/     daily_forecast.py
  data/           raw/  interim/  processed/      (git-ignored)
  models/         saved model artifacts            (git-ignored)
  reports/        evaluation report, figures
  notebooks/      EDA
  tests/
```

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run the data spike
```bash
python src/dataset/build_daily_blocks.py        # pulls a demo week -> data/processed/solcent_dataset.csv
python notebooks/explore.py                      # initial EDA -> stats + figures in reports/figures/
```
Edit the dates in `build_daily_blocks.py` and pull in monthly slices for the full
2024-01 .. 2026-05 range. Each entry-point script puts `src/` on the path itself, so you
can run files directly. Data is stored as CSV (read back with
`pd.read_csv(path, index_col=0, parse_dates=True)`).

## Two rules this project never breaks
1. **Leakage-safe inputs.** Weather features come from the Open-Meteo *Previous Runs* API
   (day-ahead lead, `_previous_day1/2`) — NOT the Historical Forecast API, whose stitched
   near-real-time values do not exist at the noon-before deadline and would leak.
2. **Honest target.** Predict capacity-normalized load factor; reconstruct it from
   `measured` and the capacity known at issue time. Beat persistence, clear-sky, raw-NWP,
   and Elia's own `dayaheadforecast` baseline, on daylight-only metrics.

## Status
Phase 1-2 (ingestion) implemented and tested. Features partially implemented; models,
evaluation, and the operational pipeline are documented skeletons for later phases.
