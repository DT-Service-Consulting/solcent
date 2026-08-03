# Relative Path: src/data/backfill_shortlist.py
#!/usr/bin/env python3
"""
backfill_shortlist.py
=====================
Remove homes you've decided to drop, and replace each one with the next-best
PASSING home from the SAME stratum (region x orientation), so your shortlist
stays balanced and the same size.
"""

from pathlib import Path
import pandas as pd

# ----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "interim" / "pvoutput_selection"
DROP_IDS = [54648, 73268]                 # the system_ids you decided to drop

# Prefer replacements with NO review flag; only fall back to a flagged one
# (with a warning) if a stratum has no clean spare left.
AVOID_FLAGGED_REPLACEMENTS = True
OUTPUT_NAME = "shortlist_final.csv"
# ----------------------------------------------------------------------------

def main():
    allc = pd.read_csv(DATA_DIR / "all_candidates.csv")
    short = pd.read_csv(DATA_DIR / "shortlist.csv")

    short_ids = set(short["system_id"])
    chosen = set(short_ids)              # tracks everyone currently selected
    final = short.copy()

    print(f"Starting shortlist: {len(short)} homes")
    print(f"Dropping: {DROP_IDS}\n")

    for sid in DROP_IDS:
        if sid not in short_ids:
            print(f"  ! {sid} is not in the shortlist -- skipping.")
            continue

        row = short[short["system_id"] == sid].iloc[0]
        stratum = row["stratum"]
        # remove the dropped home
        final = final[final["system_id"] != sid]
        chosen.discard(sid)

        # candidate replacements: passed, same stratum, not already chosen, not dropped
        pool = allc[(allc["passed_gate"])
                    & (allc["stratum"] == stratum)
                    & (~allc["system_id"].isin(chosen))
                    & (~allc["system_id"].isin(DROP_IDS))]

        if AVOID_FLAGGED_REPLACEMENTS:
            clean = pool[pool["review_flag"].fillna("").astype(str).str.len() == 0]
            use = clean if not clean.empty else pool
            flagged_fallback = clean.empty and not pool.empty
        else:
            use = pool
            flagged_fallback = False

        if use.empty:
            print(f"  - dropped {sid} ({stratum}) -> NO replacement available in this "
                  f"stratum; shortlist will be one smaller here. Hand-pick if needed.")
            continue

        repl = use.sort_values("quality_score", ascending=False).iloc[0]
        chosen.add(repl["system_id"])
        final = pd.concat([final, repl.to_frame().T], ignore_index=True)

        note = "  (no clean spare -> took a FLAGGED one)" if flagged_fallback else ""
        print(f"  - dropped {sid} ({stratum})")
        print(f"      replaced with {int(repl['system_id'])} '{repl['name']}' "
              f"score={repl['quality_score']} cap={repl['capacity_kwp']}kWp{note}")
        if flagged_fallback:
            print(f"      ! its flag: {repl['review_flag']}")

    final = final.sort_values(["region", "orientation_class", "quality_score"],
                              ascending=[True, True, False]).reset_index(drop=True)
    final.to_csv(DATA_DIR / OUTPUT_NAME, index=False)

    # Report
    print(f"\nFinal shortlist: {len(final)} homes -> {DATA_DIR / OUTPUT_NAME}")
    print("\nPer region:")
    print(final["region"].value_counts().to_string())
    print("\nStill-flagged homes in final shortlist (review or accept):")
    fl = final[final["review_flag"].fillna("").astype(str).str.len() > 0]
    if fl.empty:
        print("  none")
    else:
        for _, r in fl.iterrows():
            print(f"  {int(r['system_id'])}  {r['region']:<26} -> {r['review_flag']}")

    ids = ",".join(str(int(s)) for s in final["system_id"])
    print("\nPaste this into pvoutput_belgium.py TARGET_SYSTEM_IDS "
          "(or PVOUTPUT_TARGET_SYSTEM_IDS):")
    print(ids)

if __name__ == "__main__":
    main()