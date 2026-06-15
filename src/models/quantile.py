"""Probabilistic day-ahead model - one LightGBM per quantile (pinball loss)."""
import numpy as np


def train_quantile_models(X, y, quantiles, params):
    """Train one model per quantile. Same X, y; only `alpha` differs."""
    import lightgbm as lgb
    models = {}
    for a in quantiles:
        m = lgb.LGBMRegressor(objective="quantile", alpha=a, **params)
        m.fit(X, y)
        models[a] = m
    return models


def predict_quantiles(models, X):
    """Predict each quantile and SORT per row so P10 <= P50 <= P90 (no crossing)."""
    alphas = sorted(models)
    preds = np.vstack([models[a].predict(X) for a in alphas])
    preds = np.sort(preds, axis=0).clip(0.0, 1.0)
    return {a: preds[i] for i, a in enumerate(alphas)}
