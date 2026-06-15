"""Admissible lag features: same quarter-hour on D-1 and D-7 (known at issue time)."""
import pandas as pd


def add_lags(series, lags=("1D", "7D")):
    """Return a DataFrame of shifted copies. Only use lags available at issue time."""
    return pd.DataFrame({f"lag_{L}": series.shift(freq=L) for L in lags})
