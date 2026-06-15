"""Point and probabilistic metrics. Report DAYLIGHT-ONLY as the headline."""
import numpy as np


def mae(y, yhat):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))


def rmse(y, yhat):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))


def mbe(y, yhat):
    return float(np.mean(np.asarray(yhat) - np.asarray(y)))


def skill_score(model_error, reference_error):
    """1 - model/reference. >0 means the model beats the reference."""
    return 1.0 - model_error / reference_error


def pinball(y, q, alpha):
    e = np.asarray(y) - np.asarray(q)
    return float(np.mean(np.maximum(alpha * e, (alpha - 1) * e)))


def crps_from_quantiles(y, qpreds):
    """Approximate CRPS as the mean pinball loss across the predicted quantiles."""
    return float(np.mean([pinball(y, q, a) for a, q in qpreds.items()]))


def interval_coverage(y, lo, hi):
    """Fraction of truth inside [lo, hi]. For P10-P90 this should be ~0.8."""
    y, lo, hi = map(np.asarray, (y, lo, hi))
    return float(np.mean((y >= lo) & (y <= hi)))


def sharpness(lo, hi):
    return float(np.mean(np.asarray(hi) - np.asarray(lo)))
