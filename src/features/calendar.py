"""Calendar and Fourier seasonal features."""
import numpy as np
import pandas as pd


def calendar_features(index, fourier_orders=3):
    idx = index.tz_convert("UTC")
    doy = np.asarray(idx.dayofyear)
    out = pd.DataFrame(index=index)
    out["hour"] = idx.hour
    out["quarter_hour"] = idx.hour * 4 + idx.minute // 15
    out["weekday"] = idx.weekday
    out["month"] = idx.month
    for k in range(1, fourier_orders + 1):
        out[f"doy_sin{k}"] = np.sin(2 * np.pi * k * doy / 365.25)
        out[f"doy_cos{k}"] = np.cos(2 * np.pi * k * doy / 365.25)
    return out
