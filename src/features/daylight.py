"""Leakage-free daylight mask: depends only on date & location (known in advance)."""


def daylight_mask(index, lat=50.85, lon=4.35, min_ghi=5.0):
    """Boolean Series: True where the sun is up (clear-sky GHI above a small floor).
    Driven by solar geometry, NOT a fixed clock -- Belgian day length swings 8h<->16h."""
    import pvlib
    loc = pvlib.location.Location(lat, lon, tz="UTC")
    cs = loc.get_clearsky(index.tz_convert("UTC"))
    return cs["ghi"] > min_ghi
