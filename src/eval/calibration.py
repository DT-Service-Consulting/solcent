"""Calibration diagnostics: are the quantiles honest?"""
import numpy as np


def reliability_table(y, qpreds):
    """For each predicted quantile alpha, the empirical fraction of y below it.
    Well-calibrated => empirical ~ nominal alpha."""
    y = np.asarray(y)
    return {a: float(np.mean(y <= np.asarray(q))) for a, q in sorted(qpreds.items())}


# TODO (Phase 8): PIT histogram, reliability diagram plot, daily-reset handling for
# the diurnal zero-generation discontinuity.
