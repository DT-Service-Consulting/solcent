# Relative Path: src/data/pvoutput_select_homes.py
#!/usr/bin/env python3
"""
pvoutput_select_homes.py  (v2)
==============================
Pick a STRATIFIED, QUALITY-CHECKED set of Belgian PV systems.
"""

import os
import sys
import time
import math
from datetime import date, timedelta
from pathlib import Path

import requests
import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

API_KEY   = os.environ.get("PVOUTPUT_API_KEY")
SYSTEM_ID = os.environ.get("PVOUTPUT_SYSTEM_ID")

START_DATE = date(2024, 4, 1)
END_DATE   = date(2026, 5, 31)

REGION_ANCHORS = [
    ("Coast-NW (West Flanders)", 51.21, 2.93, 40),   
    ("Central (Brussels)",       50.85, 4.35, 40),   
    ("North (Antwerp)",          51.22, 4.40, 40),   
    ("East (Liege)",             50.63, 5.57, 45),   
    ("Ardennes-SE",              49.68, 5.81, 70),   
    ("Ardennes-SE",              50.00, 5.72, 70),   
]
MAX_PER_ANCHOR   = 30
SEARCH_SEEN_DAYS = 90

GATE_MIN_GEN_DAYS_FRAC = 0.60
GATE_MIN_MEAN_PEAK_W   = 100
WINDOW_EDGE_DAYS       = 45

MISMATCH_FLAG_PCT  = 0.10   
RATIO_FLAG_ABOVE   = 1.4    
YIELD_LOW_KWH_KWP  = 2.0    
YIELD_HIGH_KWH_KWP = 5.0    

TARGET_PER_STRATUM       = 4
SELECT_ORIENTATION_CLASSES = ["South", "East/West"]

REQUESTS_PER_HOUR = 300
OUTPUT_DIR = PROJECT_ROOT / "data" / "interim" / "pvoutput_selection"

O_DATE, O_GEN, O_EFF, O_PEAK = 0, 1, 2, 5
SY_NAME, SY_SIZE_W, SY_POSTCODE, SY_NPANELS, SY_PANEL_W, SY_PANEL_BRAND, \
SY_NINV, SY_INV_W, SY_INV_BRAND, SY_ORIENT, SY_TILT, SY_SHADE, \
SY_INSTALL, SY_SLAT, SY_SLON = range(15)

BASE = "https://pvoutput.org/service/r2"
# ----------------------------------------------------------------------------

def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlmb/2)**2
    return 2 * r * math.asin(math.sqrt(a))

def orientation_class(orient):
    if not orient:
        return "Unknown"
    o = orient.strip().upper()
    if o in {"S", "SSE", "SSW"}:
        return "South"
    if o in {"E", "ESE", "SE", "W", "WSW", "SW"}:
        return "East/West"
    return "Other"

class PVOutputClient:
    def __init__(self, api_key, system_id, requests_per_hour=300):
        if not api_key or not system_id:
            sys.exit("ERROR: set PVOUTPUT_API_KEY and PVOUTPUT_SYSTEM_ID.")
        self.headers = {
            "X-Pvoutput-Apikey": str(api_key),
            "X-Pvoutput-SystemId": str(system_id),
        }
        self.min_interval = 3600.0 / max(1, requests_per_hour)
        self._last_call = 0.0
        self._printed_raw = set()
        self.donation_required = False
        self._printed_limit = False

    def _throttle(self):
        wait = self.min_interval - (time.time() - self._last_call)
        if wait > 0:
            time.sleep(wait)

    def _get(self, endpoint, params):
        self._throttle()
        try:
            r = requests.get(f"{BASE}/{endpoint}", headers=self.headers,
                             params=params, timeout=30)
        except requests.RequestException as e:
            print(f"  ! network error on {endpoint}: {e}")
            return None
        finally:
            self._last_call = time.time()

        if not self._printed_limit:
            lim = r.headers.get("X-Rate-Limit-Limit")
            if lim:
                self._printed_limit = True
                tier = "DONATION" if int(lim) >= 300 else "FREE"
                print(f"  [quota] limit={lim}/hr ({tier} tier)")
                if int(lim) < 300:
                    print("  ! donation mode may not be active -- check your account.")

        if endpoint not in self._printed_raw:
            self._printed_raw.add(endpoint)
            print(f"  [raw {endpoint}] {r.text[:160]!r}")

        if r.status_code == 200:
            return r.text
        if r.status_code == 400 and ("no status" in r.text.lower()
                                     or "no system or data" in r.text.lower()):
            return None
        if r.status_code in (401, 403) and "inaccessible" in r.text.lower():
            if not self.donation_required:
                print("  ! 'Inaccessible System ID' -> needs Donation mode (sid1 is "
                      "donor-only). See https://pvoutput.org/help/donations.html")
            self.donation_required = True
            return None
        print(f"  ! {endpoint} {r.status_code}: {r.text[:100]}")
        return None

    def search_systems(self, lat, lon, distance_km, max_systems):
        txt = self._get("search.jsp", {
            "q": f"{distance_km}km", "ll": f"{lat},{lon}",
            "country": 1, "country_code": "be", "seen": SEARCH_SEEN_DAYS,
        })
        if not txt:
            return []
        out = []
        for record in txt.splitlines():
            if not record.strip():
                continue
            parts = [p.strip() for p in record.split(",")]

            def grab(i, cast=str):
                try:
                    return cast(parts[i])
                except (IndexError, ValueError):
                    return None

            sid = grab(6, int)
            if sid is None:
                continue
            size_w = grab(1, float)
            location = grab(2) or ""
            if "belgium" not in location.lower():
                continue
            out.append({
                "system_id": sid,
                "name": grab(0),
                "capacity_kwp": round(size_w / 1000.0, 3) if size_w else None,
                "location": location,
                "orientation": grab(3),
                "latitude": grab(-2, float),
                "longitude": grab(-1, float),
            })
            if len(out) >= max_systems:
                break
        return out

    def get_system(self, sid):
        """Full system metadata (donor-only). Returns all useful getsystem fields."""
        txt = self._get("getsystem.jsp", {"sid1": sid, "ext": 0})
        if not txt:
            return {}
        f = txt.split(";")[0].split(",")

        def num(i):
            try:
                return float(f[i])
            except (IndexError, ValueError):
                return None

        def text(i):
            try:
                return f[i].strip()
            except IndexError:
                return None

        npan, ppw = num(SY_NPANELS), num(SY_PANEL_W)
        return {
            "system_size_w": num(SY_SIZE_W),
            "postcode": text(SY_POSTCODE),
            "num_panels": npan,
            "panel_power_w": ppw,
            "panel_array_w": (npan * ppw) if (npan and ppw) else None,
            "panel_brand": text(SY_PANEL_BRAND),
            "num_inverters": num(SY_NINV),
            "inverter_size_w": num(SY_INV_W),
            "inverter_brand": text(SY_INV_BRAND),
            "tilt_deg": num(SY_TILT),
            "shade": text(SY_SHADE),
            "install_date": text(SY_INSTALL),
        }

    def get_output(self, sid, df_date, dt_date, limit=150):
        txt = self._get("getoutput.jsp", {
            "sid1": sid, "df": df_date.strftime("%Y%m%d"),
            "dt": dt_date.strftime("%Y%m%d"), "limit": limit,
        })
        if not txt:
            return []
        days = []
        for rec in txt.split(";"):
            parts = rec.split(",")
            if len(parts) <= O_PEAK:
                continue

            def num(i):
                try:
                    v = parts[i].strip()
                    return None if v == "" or v.lower() == "nan" else float(v)
                except (IndexError, ValueError):
                    return None

            try:
                d = pd.to_datetime(parts[O_DATE], format="%Y%m%d").date()
            except ValueError:
                continue
            days.append({"date": d, "generated_wh": num(O_GEN),
                         "efficiency": num(O_EFF), "peak_w": num(O_PEAK)})
        return days

def probe_quality(client, cand):
    span = (END_DATE - START_DATE).days
    if span <= 149:
        windows = [(START_DATE, END_DATE)]
    else:
        windows = [
            (START_DATE, START_DATE + timedelta(days=149)),
            (END_DATE - timedelta(days=149), END_DATE),
        ]
    seen = {}
    for df_d, dt_d in windows:
        for row in client.get_output(cand["system_id"], df_d, dt_d):
            seen[row["date"]] = row
        if client.donation_required:
            break
    days = list(seen.values())
    if not days:
        return {"n_probe_days": 0, "gen_days_frac": 0.0, "mean_peak_w": 0.0,
                "mean_daily_kwh": 0.0, "start_covered": False, "end_covered": False}
    dates = [d["date"] for d in days]
    peaks = [d["peak_w"] for d in days if d["peak_w"] is not None]
    gens  = [d["generated_wh"] for d in days if d["generated_wh"] is not None]
    n_gen = sum(1 for g in gens if g and g > 0)
    return {
        "n_probe_days": len(days),
        "gen_days_frac": round(n_gen / len(days), 3) if days else 0.0,
        "mean_peak_w": round(sum(peaks) / len(peaks), 1) if peaks else 0.0,
        "mean_daily_kwh": round((sum(gens) / len(gens)) / 1000.0, 2) if gens else 0.0,
        "start_covered": min(dates) <= START_DATE + timedelta(days=WINDOW_EDGE_DAYS),
        "end_covered":   max(dates) >= END_DATE - timedelta(days=WINDOW_EDGE_DAYS),
    }

def passes_gate(row):
    return bool(
        row.get("capacity_kwp")
        and row.get("latitude") is not None and row.get("longitude") is not None
        and row["start_covered"] and row["end_covered"]
        and row["gen_days_frac"] >= GATE_MIN_GEN_DAYS_FRAC
        and row["mean_peak_w"] >= GATE_MIN_MEAN_PEAK_W
    )

def quality_score(row):
    if not row.get("capacity_kwp"):
        return 0.0
    cap_w = row["capacity_kwp"] * 1000.0
    peak_ratio = min(1.0, row["mean_peak_w"] / (0.3 * cap_w)) if cap_w else 0.0
    return round(row["gen_days_frac"] * peak_ratio, 3)

def review_flags(row):
    """Advisory flags (never auto-reject). Returns a list of short reason strings."""
    flags = []
    cap = row.get("capacity_kwp")
    inv = row.get("inverter_size_w")
    if cap and inv:
        ratio = (cap * 1000.0) / inv
        row["pv_inverter_ratio"] = round(ratio, 2)
        if ratio > RATIO_FLAG_ABOVE:
            flags.append(f"clipping?(ratio {ratio:.2f})")
    else:
        row["pv_inverter_ratio"] = None
    sy = row.get("specific_yield")
    if sy is not None:
        if sy < YIELD_LOW_KWH_KWP:
            flags.append(f"low-yield({sy:.2f})")
        elif sy > YIELD_HIGH_KWH_KWP:
            flags.append(f"high-yield({sy:.2f})")
    pa, ss = row.get("panel_array_w"), row.get("system_size_w")
    if pa and ss:
        m = (pa - ss) / ss
        row["capacity_mismatch_pct"] = round(m * 100, 1)
        if abs(m) > MISMATCH_FLAG_PCT:
            flags.append(f"capacity-mismatch({m*100:+.0f}%)")
    else:
        row["capacity_mismatch_pct"] = None
    return flags

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    client = PVOutputClient(API_KEY, SYSTEM_ID, REQUESTS_PER_HOUR)

    candidates = {}
    for region, lat, lon, radius in REGION_ANCHORS:
        print(f"\nSearching {region}  (r={radius} km) ...")
        found = client.search_systems(lat, lon, radius, MAX_PER_ANCHOR)
        print(f"  {len(found)} candidate(s)")
        for c in found:
            candidates.setdefault(c["system_id"], c)
    if not candidates:
        sys.exit("No candidates found. Check donation mode / widen the search.")
    print(f"\n{len(candidates)} unique candidate system(s) total.")

    for c in candidates.values():
        if c["latitude"] is None or c["longitude"] is None:
            c["region"] = "Unknown"
        else:
            c["region"] = min(
                REGION_ANCHORS,
                key=lambda a: haversine_km(c["latitude"], c["longitude"], a[1], a[2]),
            )[0]
        c["orientation_class"] = orientation_class(c["orientation"])

    print(f"\nProbing + fetching full metadata for {len(candidates)} candidates "
          f"(~3 calls each)...")
    META_KEYS = ["system_size_w", "postcode", "num_panels", "panel_power_w",
                 "panel_array_w", "panel_brand", "num_inverters", "inverter_size_w",
                 "inverter_brand", "tilt_deg", "shade", "install_date"]
    rows = []
    for i, c in enumerate(candidates.values(), 1):
        c.update(probe_quality(client, c))
        info = client.get_system(c["system_id"])
        if client.donation_required:
            sys.exit("\nSTOPPING: donation mode not active -- cannot read other systems.")
        for k in META_KEYS:
            c[k] = info.get(k)
        c["specific_yield"] = (round(c["mean_daily_kwh"] / c["capacity_kwp"], 2)
                               if c.get("capacity_kwp") else None)
        c["passed_gate"] = passes_gate(c)
        c["quality_score"] = quality_score(c)
        c["review_flag"] = "; ".join(review_flags(c))   
        rows.append(c)
        if i % 10 == 0:
            print(f"  done {i}/{len(candidates)}")

    df = pd.DataFrame(rows)

    df["stratum"] = df["region"] + " | " + df["orientation_class"]
    df = df.sort_values(["stratum", "passed_gate", "quality_score"],
                        ascending=[True, False, False])
    df["rank_in_stratum"] = (df[df["passed_gate"]]
                             .groupby("stratum").cumcount() + 1)
    shortlist = df[df["passed_gate"]
                   & df["orientation_class"].isin(SELECT_ORIENTATION_CLASSES)
                   & (df["rank_in_stratum"] <= TARGET_PER_STRATUM)]

    cols = ["system_id", "name", "region", "orientation", "orientation_class",
            "capacity_kwp", "system_size_w", "num_panels", "panel_power_w",
            "tilt_deg", "latitude", "longitude", "n_probe_days",
            "gen_days_frac", "mean_peak_w", "mean_daily_kwh", "specific_yield",
            "inverter_size_w", "pv_inverter_ratio", "capacity_mismatch_pct",
            "review_flag", "start_covered", "end_covered", "passed_gate",
            "quality_score", "rank_in_stratum", "stratum"]
    df.to_csv(OUTPUT_DIR / "all_candidates.csv", columns=cols, index=False)
    shortlist.to_csv(OUTPUT_DIR / "shortlist.csv", columns=cols, index=False)

    meta_cols = ["system_id", "name", "region", "orientation", "orientation_class",
                 "capacity_kwp", "system_size_w", "num_panels", "panel_power_w",
                 "panel_array_w", "panel_brand", "num_inverters", "inverter_size_w",
                 "inverter_brand", "tilt_deg", "shade", "install_date", "postcode",
                 "latitude", "longitude", "pv_inverter_ratio", "capacity_mismatch_pct"]
    df.to_csv(OUTPUT_DIR / "systems_metadata.csv", columns=meta_cols, index=False)

    flagged = shortlist[shortlist["review_flag"].astype(str).str.len() > 0]
    print("\n--- Homes flagged for MANUAL REVIEW (advisory only, not dropped) ---")
    if flagged.empty:
        print("  none -- all shortlisted homes look clean.")
    else:
        print(f"  {len(flagged)} of {len(shortlist)} shortlisted homes flagged:")
        for _, r in flagged.iterrows():
            print(f"  sid {int(r['system_id']):>7}  {r['region']:<26}  "
                  f"{r['capacity_kwp']:>5} kWp  yield={r['specific_yield']}  "
                  f"ratio={r['pv_inverter_ratio']}  ->  {r['review_flag']}")
        print("  Check these on pvoutput.org; drop+backfill any you consider atypical.")

    print(f"\n--- Shortlist fill by stratum (target {TARGET_PER_STRATUM} each) ---")
    regions = list(dict.fromkeys(a[0] for a in REGION_ANCHORS))
    classes = ["South", "East/West", "Other"]
    passed = df[df["passed_gate"]]
    for reg in regions:
        line = [f"{reg:28s}"]
        for cl in classes:
            n = len(passed[(passed["region"] == reg) & (passed["orientation_class"] == cl)])
            sel = " (target)" if cl in SELECT_ORIENTATION_CLASSES else " (skip)  "
            mark = "OK " if n >= TARGET_PER_STRATUM else ("-- " if n == 0 else "lo ")
            line.append(f"{cl}:{n:2d} {mark}")
        print("  " + "  ".join(line))
    print(f"\n{len(passed)} systems passed the gate; "
          f"{len(shortlist)} selected into the shortlist "
          f"(orientation classes: {SELECT_ORIENTATION_CLASSES}).")
    print(f"Files -> {OUTPUT_DIR/'shortlist.csv'}, {OUTPUT_DIR/'all_candidates.csv'}, "
          f"and {OUTPUT_DIR/'systems_metadata.csv'}")

if __name__ == "__main__":
    main()