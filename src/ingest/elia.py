"""
elia.py - Ingest Belgian solar generation from Elia Open Data (dataset ods032).

ods032 stores one row per (region, 15-min timestamp): ~14 regional rows per timestamp
(the provinces) plus a 'Belgium' national aggregate. We keep the 'Belgium' rows.

Key fields:
  measured            actual generation (MW)                  -> target numerator
  monitoredcapacity   installed capacity Elia tracks          -> target denominator
  loadfactor          = measured / monitoredcapacity * 100  (PERCENT; for EDA only)
  dayaheadforecast    Elia's own day-ahead P50 (MW)            -> baseline to beat
  dayaheadconfidence10/90  Elia's own day-ahead P10/P90        -> baseline band
  dayahead11hforecast (+conf)  the 11:00 cut-off variant       -> gate-closure evidence
"""
import time
from datetime import datetime, timedelta

import requests
import pandas as pd

ODS_URL = "https://opendata.elia.be/api/explore/v2.1/catalog/datasets/ods032/records"

FIELDS = [
    "datetime", "region", "resolutioncode",
    "measured", "monitoredcapacity", "loadfactor",
    "dayaheadforecast", "dayaheadconfidence10", "dayaheadconfidence90",
    "dayahead11hforecast", "dayahead11hconfidence10", "dayahead11hconfidence90",
]


def _month_chunks(start, end):
    """Yield [chunk_start, chunk_end) month boundaries so the API offset stays small."""
    cur = datetime(start.year, start.month, 1)
    last = end + timedelta(days=1)
    while cur < last:
        nxt = datetime(cur.year + (cur.month == 12), (cur.month % 12) + 1, 1)
        yield max(cur, start), min(nxt, last)
        cur = nxt


def _get(params, tries=6):
    for k in range(tries):
        try:
            r = requests.get(ODS_URL, params=params, timeout=60)
            if r.status_code == 200:
                j = r.json()
                if "results" in j:
                    return j
        except requests.exceptions.RequestException:
            pass  # transient timeout / reset (portal is flaky under load)
        time.sleep(3 * (k + 1))
    raise RuntimeError("Elia ods032 request failed after retries")


def fetch_ods032(start, end, region="Belgium"):
    """Fetch ods032 for `region` over [start, end] (end day inclusive), 15-min.
    Returns a DataFrame indexed by UTC datetime."""
    rows = []
    for c0, c1 in _month_chunks(start, end):
        where = (f"datetime >= '{c0:%Y-%m-%d}' AND datetime < '{c1:%Y-%m-%d}' "
                 f"AND region='{region}'")
        offset = 0
        while True:
            j = _get({"where": where, "select": ",".join(FIELDS),
                      "order_by": "datetime", "limit": 100, "offset": offset})
            res = j["results"]
            if not res:
                break
            rows.extend(res)
            offset += len(res)
            if offset >= j.get("total_count", 0):
                break
        print(f"  elia {c0:%Y-%m} -> {len(rows)} rows so far")
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    return df.set_index("datetime").sort_index()


if __name__ == "__main__":
    df = fetch_ods032(datetime(2025, 4, 19), datetime(2025, 4, 21))
    print(df.shape)
    print(df[["measured", "dayaheadforecast", "monitoredcapacity", "loadfactor"]].head())
