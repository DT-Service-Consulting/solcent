"""
daily_forecast.py - Single-shot operational cycle (one forecast per day).

At the issue time, pull the latest pre-gate NWP run covering all of day D, build the
96-slot feature block through the SAME code path used in training, predict P10/P50/P90
load factor, force night slots to zero, sort/clip, and scale by capacity to MW.

This is a documented skeleton -- it composes the feature and model modules once they
are trained (Phases 7-8).
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def run_daily_forecast(issue_time, capacity_at_issue, models, feature_fn):
    """
    Pseudocode of the operational cycle:

        nwp   = fetch_latest_nwp(before=issue_time, covers=day_D)   # one run, all slots
        feats = feature_fn(nwp, lags, capacity_at_issue)            # 96-row block
        from models.quantile import predict_quantiles
        lf    = predict_quantiles(models, feats)                    # dict alpha->array
        for a in lf:                                                # night -> 0
            lf[a][~feats["daylight"].values] = 0.0
        mw    = {a: lf[a] * capacity_at_issue for a in lf}
        return mw                                                   # 96 rows: P10/P50/P90
    """
    raise NotImplementedError("Wire up once feature + quantile models are trained.")
