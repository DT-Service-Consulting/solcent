"""Baseline forecasts. Beating raw-NWP and Elia's own day-ahead band is the real test."""
import pandas as pd


def persistence(load_factor, freq="1D"):
    """Yesterday's same-slot load factor."""
    return load_factor.shift(freq=freq)


def smart_persistence(clearsky_index, clearsky_ghi, capacity):
    """Persist the clear-sky index and reapply it to today's clear-sky GHI."""
    ci_yesterday = clearsky_index.shift(freq="1D")
    return ci_yesterday  # multiply by clearsky_ghi / scale to power downstream


def raw_nwp_to_power(clearsky_index_forecast, capacity, derate=0.9):
    """Convert forecast clear-sky index directly to power, no ML. The key benchmark."""
    return (clearsky_index_forecast.clip(0, 1) * capacity * derate)


def elia_baseline(df):
    """Elia's own published day-ahead band (already in ods032)."""
    return df[["dayaheadforecast", "dayaheadconfidence10", "dayaheadconfidence90"]]
