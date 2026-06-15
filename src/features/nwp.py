"""Assemble NWP features, including multi-model ensemble mean & spread (uncertainty)."""
import pandas as pd


def multimodel_mean_spread(df, var, lead, models=("ecmwf", "icon", "gfs")):
    """Mean and std across NWP models for one variable/lead.
    Expects columns named like {var}_previous_day{lead}_{model}."""
    cols = [f"{var}_previous_day{lead}_{m}" for m in models
            if f"{var}_previous_day{lead}_{m}" in df.columns]
    sub = df[cols]
    return sub.mean(axis=1), sub.std(axis=1)


def assemble_nwp_features(df, variables, leads=(1, 2), models=("ecmwf", "icon", "gfs")):
    """Build per-model columns plus ensemble mean/spread for each variable & lead."""
    out = pd.DataFrame(index=df.index)
    for v in variables:
        for d in leads:
            mean, spread = multimodel_mean_spread(df, v, d, models)
            out[f"{v}_d{d}_mean"] = mean
            out[f"{v}_d{d}_spread"] = spread
    return out
