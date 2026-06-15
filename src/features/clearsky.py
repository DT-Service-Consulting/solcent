"""Clear-sky GHI and clear-sky index (the workhorse normalization), via pvlib."""
import pandas as pd


def clearsky_ghi(index, lat=50.85, lon=4.35, model="ineichen"):
    """Clear-sky GHI (W/m^2) for a tz-aware DatetimeIndex."""
    import pvlib
    loc = pvlib.location.Location(lat, lon, tz="UTC")
    cs = loc.get_clearsky(index.tz_convert("UTC"), model=model)
    return cs["ghi"]


def clearsky_index(ghi, cs_ghi, clip=(0.0, 1.2)):
    """Clear-sky index = forecast GHI / clear-sky GHI, clipped. Night -> 0."""
    ci = ghi / cs_ghi.replace(0, pd.NA)
    return ci.fillna(0.0).clip(*clip)
