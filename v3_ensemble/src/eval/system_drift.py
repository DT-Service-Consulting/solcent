#!/usr/bin/env python3
"""
system_drift.py
===============
Checks whether a rooftop's output drifts relative to the rest of the fleet
over the study window. A static capacity error gives a constant ratio; a
time-varying defect (growing shade, a failing string) gives a ratio that moves.

For each system and calendar month:

    ratio = sum(target_norm of the system) / sum(fleet-median target_norm)

over the same daylight, valid timestamps. The fleet median excludes the
system itself. The spread of the monthly ratio (max/min) is reported per
system, so system 81376 can be compared with the rest of the fleet.

Reported in the manuscript, Section 5.4 (exclusion of system 81376).

Output: v3_ensemble/reports/system_drift.txt
        v3_ensemble/models/system_drift_monthly.csv
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as C

MIN_ROWS = 300   # months with fewer daylight rows for a system are skipped


def main():
    df = pd.read_csv(C.MODEL_READY_V3, usecols=[
        "timestamp", "system_id", "target_norm", "is_daylight", "is_valid"])
    df = df[(df.is_daylight == True) & (df.is_valid == True)]
    df = df[~df.system_id.isin(C.EXCLUDE_SYSTEMS)]
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["month"] = df["timestamp"].dt.strftime("%Y-%m")

    wide = df.pivot_table(index="timestamp", columns="system_id",
                          values="target_norm")
    month = pd.Series(wide.index.strftime("%Y-%m"), index=wide.index)

    rows = []
    for sid in wide.columns:
        own = wide[sid]
        fleet = wide.drop(columns=sid).median(axis=1)
        ok = own.notna() & fleet.notna()
        g = pd.DataFrame({"own": own[ok], "fleet": fleet[ok], "month": month[ok]})
        for m, gm in g.groupby("month"):
            if len(gm) >= MIN_ROWS and gm.fleet.sum() > 0:
                rows.append({"system_id": sid, "month": m, "n": len(gm),
                             "ratio": gm.own.sum() / gm.fleet.sum()})
    r = pd.DataFrame(rows)
    r.to_csv(C.MODELS_DIR / "system_drift_monthly.csv", index=False)

    s = r.groupby("system_id").ratio.agg(["min", "max", "count"])
    s["max_over_min"] = s["max"] / s["min"]
    s = s.sort_values("max_over_min", ascending=False)

    lines = [
        "Monthly output relative to the fleet median",
        "=" * 60,
        s.head(10).to_string(float_format=lambda x: f"{x:.3f}"),
        "",
        f"fleet median of max/min ratio: {s.max_over_min.median():.3f}",
    ]
    if 81376 in s.index:
        x = s.loc[81376]
        rank = list(s.index).index(81376) + 1
        lines.append(f"81376: monthly ratio {x['min']:.2f}-{x['max']:.2f} "
                     f"(max/min {x.max_over_min:.2f}, rank {rank} of {len(s)})")

    # Season-matched check: monthly ratios move with sun angle for every
    # system, so compare the same calendar months across years. April-May is
    # the only period present in all three years, and 2026 is the test window.
    r["year"] = r.month.str[:4].astype(int)
    am = r[r.month.str[5:].isin(["04", "05"])]
    p = am.pivot_table(index="system_id", columns="year", values="ratio", aggfunc="mean")
    chg = (p[2026] / p[[2024, 2025]].mean(axis=1)).sort_values()
    lines += [
        "",
        "April-May ratio, 2026 (test window) relative to mean of 2024 and 2025",
        "-" * 60,
        chg.head(5).to_string(float_format=lambda v: f"{v:.3f}"),
        f"fleet median: {chg.median():.3f}  (n={chg.notna().sum()})",
    ]
    if 81376 in p.index:
        y = p.loc[81376]
        lines.append(f"81376 April-May ratio: 2024 {y[2024]:.3f}, "
                     f"2025 {y[2025]:.3f}, 2026 {y[2026]:.3f}")
    text = "\n".join(lines)
    print(text)
    (C.REPORTS_DIR / "system_drift.txt").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
