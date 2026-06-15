"""Deterministic day-ahead model (P50) - LightGBM gradient-boosted trees."""
# TODO (Phase 7): tune via rolling-origin CV; inspect feature importance over lead time.


def train_point_model(X, y, params):
    import lightgbm as lgb
    model = lgb.LGBMRegressor(objective="regression", **params)
    model.fit(X, y)
    return model


def predict_load_factor(model, X):
    return model.predict(X).clip(0.0, 1.0)
