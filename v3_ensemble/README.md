# solcent — day-ahead rooftop PV forecasting, Belgium

Code and derived results for the study *"Where the Ceiling Lies in Day-Ahead
Rooftop PV Forecasting: An Input-Limited Diagnosis Across a Belgian Residential
Fleet."*

Tag `v1.0-paper` corresponds to the results reported in the manuscript.

---

## What this study found

Day-ahead forecasting for individual rooftops is limited by the resolution of
numerical weather prediction, not by the choice of machine-learning model.
Four independent lines of evidence:

| finding | number |
|---|---|
| Perfect *regional* weather knowledge is worth almost nothing | +0.023 R² |
| Forecast error is near-identical at every rooftop | 3.3% spread across 31 sites |
| Widening the NWP input from 1 to 12 members barely helps | +0.004 R² |
| Correcting irradiance error does not propagate to power | ~0 on seen **and** unseen roofs |

Of 86 W/m² total irradiance error, only ~8 W/m² is systematic. The remainder is
cloud placement and timing — clouds a few kilometres wide, lasting minutes, on a
25 km hourly grid.

---

## Layout

```
v3_ensemble/
├── config.py                      paths, feature sets, the three hyperparameter configurations
├── src/
│   ├── features/
│   │   ├── fetch_ensemble.py      12-member lagged + multi-model NWP backfill
│   │   ├── fetch_satellite_truth.py  EUMETSAT SARAH-3 reference irradiance
│   │   ├── build_features_v3.py   merge + engineer onto the base feature table
│   │   └── log_live_ensemble.py   daily 51-member ECMWF ensemble logger
│   ├── models/
│   │   ├── tune_quantile.py       hyperparameter search, pinball objective, temporal split
│   │   ├── tune_transfer.py       hyperparameter search, held-out SYSTEMS
│   │   ├── consolidate.py         all six experiments under one configuration
│   │   ├── mos_satellite.py       MOS at irradiance level, satellite vs ERA5 truth
│   │   ├── mos_pv_integration.py  does the correction propagate to power?
│   │   └── loso_mos.py            MOS on unseen rooftops
│   ├── diagnostics/oracle_ceiling.py   headroom instrument (leakage-guarded)
│   ├── eval/metrics.py            pinball, reliability, regime, SSE share
│   └── vendor/                    verbatim copies of upstream scripts
├── models/consolidated/*.csv      every number in the results section
├── models/tuned/*.json            selected hyperparameters
└── notebooks/                     weather-error analysis by sky condition
```

`paper/figures/*.py` regenerate the manuscript figures.
`paper/verify_references.py` re-checks every citation against Crossref,
DataCite and OpenAlex.

---

## Reproducing

```bash
python v3_ensemble/src/features/fetch_ensemble.py          # ~20 min, resumable
python v3_ensemble/src/features/fetch_satellite_truth.py   # ~15 min, resumable
python v3_ensemble/src/features/build_features_v3.py       # ~10 min
python v3_ensemble/src/models/tune_quantile.py             # ~2.5 h
python v3_ensemble/src/models/tune_transfer.py             # ~1 h
python v3_ensemble/src/models/consolidate.py               # ~2 h
python v3_ensemble/src/models/consolidate.py --only exp5   # just the LOSO half
```

Derived results are committed, so the tables can be checked without re-running
anything.

---

## Three hyperparameter configurations, and why

| configuration | selected on | use |
|---|---|---|
| `LGBM_PARAMS` | validation RMSE, temporal split | inherited baseline, comparison only |
| `TUNED_PARAMS` | pinball, temporal split | fleet operation — rooftops seen in training |
| `TRANSFER_PARAMS` | pinball, **held-out systems** | deployment — unseen rooftops |

The two objectives disagree. The configuration that wins on the chronological
split loses on unseen rooftops, because depth and tree count buy rooftop-specific
memorisation that is worthless on a rooftop the model has never seen. Selecting
against held-out *systems* instead produces a much shallower model (depth 4,
L2 = 4.34) whose benefit lands in the tail rather than the median: the worst
rooftop improves from R² −0.66 to −0.16, and failures halve from 4/33 to 2/33.

---

## Data availability

Analysis code and derived results are in this repository. The underlying
datasets differ in what may be redistributed, and only what the licences allow
is included here.

- **Weather forecasts** — Open-Meteo archive, CC BY 4.0. Upstream providers
  carry their own terms. Retrieval scripts with documented endpoints are
  provided rather than the fields themselves.
- **Satellite irradiance** — EUMETSAT CM SAF SARAH-3,
  [10.5676/EUM_SAF_CM/SARAH/V003](https://doi.org/10.5676/EUM_SAF_CM/SARAH/V003).
- **PV generation** — published by individual owners on PVOutput.org. Bulk
  access is a commercial service and the terms do not grant redistribution
  rights, so the raw series are **not** included. System identifiers, the
  retrieval script and the feature schema are, which allows reconstruction by
  any user holding their own PVOutput access.

This is a real limit on exact reproducibility rather than a formality, and it is
one reason fleet metadata problems are hard to audit across studies.

---

## Note on GPU

LightGBM's OpenCL backend was measured at **3× slower** than CPU on this
workload — 47.4 s against 15.8 s for 300 trees. The GPU histogram kernel
accelerates work proportional to rows × features, and at 13 features there is
too little to parallelise to recover the kernel-launch overhead. Do not expect
GPU acceleration to help on similarly narrow feature sets.
