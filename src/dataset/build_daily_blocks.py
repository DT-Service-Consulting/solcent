"""
build_daily_blocks.py - Join Elia generation + day-ahead weather into one aligned table.

Single-shot structure: each day D contributes one issue and a block of slots. First-pass
spike is hourly + single point; move to 15-min + a Belgian grid once it works.

Leakage notes baked in:
  - weather columns come from the Previous Runs API (day-ahead lead) -> safe inputs
  - dayaheadforecast / confidence10 / confidence90 kept as the Elia BASELINE to beat
  - load_factor here uses the per-row monitoredcapacity for simplicity. For strict
    leakage-safety, later substitute the capacity value KNOWN AT ISSUE TIME.
"""
import sys
import pathlib
from datetime import datetime

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # put src/ on path
from ingest import elia, openmeteo  # noqa: E402

LAT, LON = 50.85, 4.35  # Brussels (single-point spike)


def build(start, end, out_path="data/processed/solcent_dataset.csv"):
    print("Fetching Elia ods032 (Belgium)...")
    el = elia.fetch_ods032(start, end, region="Belgium")
    el_h = (el[["measured", "dayaheadforecast", "dayaheadconfidence10",
                "dayaheadconfidence90", "monitoredcapacity"]]
            .resample("1h").mean(numeric_only=True))
    el_h["load_factor"] = el_h["measured"] / el_h["monitoredcapacity"]  # 0..1

    print("Fetching Open-Meteo day-ahead weather (Previous Runs API)...")
    wx = openmeteo.fetch_day_ahead_weather(LAT, LON, start, end)

    df = el_h.join(wx, how="inner")
    df.index.name = "datetime_utc"

    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out)  # read back: pd.read_csv(path, index_col=0, parse_dates=True)
    print(f"\nSaved {df.shape[0]} rows x {df.shape[1]} cols -> {out_path}")
    return df


if __name__ == "__main__":
    # Demo week. For the full project: datetime(2024,1,1) .. datetime(2026,5,31),
    # pulled in monthly slices to stay polite to the APIs.
    # build(datetime(2025, 4, 19), datetime(2025, 4, 25))
    build(datetime(2024,1,1), datetime(2026, 5, 31))
