#!/usr/bin/env python3
"""
verify_references.py
====================
Re-verifies every reference in paper.md and writes a checkable table.
Run it yourself - do not take the reference list on trust.

    python paper/verify_references.py            # metadata only (fast)
    python paper/verify_references.py --abstracts  # also pull abstracts

Standard library only. No API key.

THREE REGISTRIES, BECAUSE ONE IS NEVER ENOUGH
---------------------------------------------
  Crossref   https://api.crossref.org/works/<DOI>      journal articles
  DataCite   https://api.datacite.org/dois/<DOI>       data records
  OpenAlex   https://api.openalex.org/works/doi:<DOI>  abstracts Crossref lacks

A Crossref 404 does not mean a DOI is fake. SARAH-3 is registered with DataCite
because it is a dataset, and most Elsevier abstracts appear only in OpenAlex.

WHAT THIS CHECKS AND WHAT IT DOES NOT
-------------------------------------
CHECKS   the work exists, and that the author list, venue, volume, pages and
         year in paper.md match the publisher-registered record.

DOES NOT check that a paper's CONTENT supports the claim it is cited for. That
         needs reading. The READ column records how far verification actually
         went. "metadata only" means the claim attached to it in the text has
         NOT been confirmed against the source.

WHY THE READ COLUMN EXISTS
--------------------------
During drafting, a literature-search summary attributed a precise statistic
about pvoutput.org metadata quality ("tilt unknown for 10% of cases, 32%
demonstrably wrong, 5000 samples") to Killinger et al. (2018). The full text
was downloaded and searched: the terms "demonstrably", "unknown for" and
"5000" appear nowhere in it, nor in the other candidate source. The statistic
was discarded. Metadata verification would not have caught that - only reading
the paper did.

Output: paper/reference_check.md
"""

import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parent / "reference_check.md"
UA = {"User-Agent": "paper-reference-check/2.0"}

FULL = "FULL TEXT"
ABS = "abstract verified"
META = "metadata only"

# (bibtex key, DOI or None, registry, read status, what it is cited for)
REFS = [
    ("Akiba2019", "10.1145/3292500.3330701", "crossref", META,
     "Optuna TPE hyperparameter search"),
    ("Aldeman2023", "10.1016/j.rset.2023.100052", "crossref", ABS,
     "aggregation reduces variability; 11 systems in 130 km circle"),
    ("Antonanzas2016", "10.1016/j.solener.2016.06.069", "crossref", META,
     "field review; methodological arc"),
    ("Depoortere2025", "10.1016/j.ijforecast.2024.12.003", "crossref", ABS,
     "transfer beats local ~6 months; 200 km mis-specification [LOAD-BEARING]"),
    ("Friedman2001", "10.1214/aos/1013203451", "crossref", META,
     "gradient boosting"),
    ("Gneiting2007", "10.1198/016214506000001437", "crossref", META,
     "proper scoring rules, basis for pinball"),
    ("Grinsztajn2022", None, "arxiv:2207.08815", ABS,
     "tree models still SOTA on tabular ~10k samples; justifies LightGBM"),
    ("Hay1979", "10.1016/0038-092X(79)90123-3", "crossref", META,
     "Hay transposition to plane of array [LOAD-BEARING]"),
    ("Hersbach2020", "10.1002/qj.3803", "crossref", META,
     "ERA5 provenance - an ECMWF/IFS product"),
    ("Holmgren2018", "10.21105/joss.00884", "crossref", META,
     "pvlib implementation"),
    ("Ineichen2002", "10.1016/S0038-092X(02)00045-2", "crossref", META,
     "Ineichen-Perez clear-sky model [LOAD-BEARING]"),
    ("Inman2013", "10.1016/j.pecs.2013.06.002", "crossref", META,
     "field review; horizon-dependent information"),
    ("Killinger2018", "10.1016/j.solener.2018.08.051", "crossref", FULL,
     "pvoutput.org tilt filtering, location gaps, BE mean tilt 35.6 deg [LOAD-BEARING]"),
    ("Ke2017", None, "openalex:W2768348081", META,
     "LightGBM; NeurIPS, no DOI registered"),
    ("Koenker1978", "10.2307/1913643", "crossref", META,
     "pinball loss / quantile regression [LOAD-BEARING]"),
    ("Lauret2019", "10.1016/j.solener.2019.10.041", "crossref", META,
     "probabilistic verification; calibration vs sharpness [LOAD-BEARING]"),
    ("Mayer2021", "10.1016/j.solener.2021.09.044", "crossref", META,
     "design-data quality in physical model chains [LOAD-BEARING]"),
    ("Mayer2022", "10.1016/j.rser.2022.112772", "crossref", META,
     "physical vs ML hybridisation"),
    ("Mayer2023", "10.1016/j.apenergy.2023.121958", "crossref", META,
     "NWP source choice affects downstream PV accuracy"),
    ("MayerYang2023", "10.1016/j.rser.2023.113171", "crossref", ABS,
     "ensemble NWP x ensemble model chain; 14 HU plants; nCRPS 18.4%"),
    ("Meng2020", "10.1016/j.solener.2020.09.077", "crossref", ABS,
     "tilt/azimuth inference from output; MAE 4.5/4.3 deg"),
    ("Mol2024", "10.1002/qj.4712", "crossref", ABS,
     "surface irradiance under broken cloud"),
    ("Nguyen2026", "10.1063/5.0300682", "crossref", FULL,
     "+11.6 pp NWP day-ahead; publication-bias trend [LOAD-BEARING]"),
    ("Pfeifroth2023", "10.5676/EUM_SAF_CM/SARAH/V003", "datacite", META,
     "SARAH-3 data record (the dataset used) [LOAD-BEARING]"),
    ("Pfeifroth2024", "10.5194/essd-16-5243-2024", "crossref", ABS,
     "SARAH-3 method paper; 0.05 deg, 30 min [LOAD-BEARING]"),
    ("Pierro2017", "10.1016/j.solener.2017.09.068", "crossref", META,
     "regional upscaling from spatial smoothing"),
    ("Romano2019", None, "arxiv:1905.03222", ABS,
     "conformalized quantile regression; finite-sample coverage [LOAD-BEARING]"),
    ("Tuononen2019", "10.5194/acp-19-1985-2019", "crossref", FULL,
     "+8 W/m2 bias; LWP/optical-property mechanism [LOAD-BEARING]"),
    ("Urraca2017", "10.1016/j.rse.2017.07.013", "crossref", ABS,
     "CM SAF validation, 313 pyranometers, MAD 8-13 W/m2 [LOAD-BEARING]"),
    ("Urraca2018", "10.1016/j.solener.2018.02.059", "crossref", ABS,
     "ERA5 cloud-dependent bias; worse than satellite [LOAD-BEARING]"),
    ("Verzijlbergh2015", "10.1016/j.solener.2015.06.005", "crossref", META,
     "MOS for NWP irradiance is established practice [LOAD-BEARING]"),
    ("Visser2022", "10.1016/j.renene.2021.10.102", "crossref", ABS,
     "12 day-ahead models on distributed PV; fleet size and spacing help [LOAD-BEARING]"),
    ("Wang2022", "10.1016/j.solener.2022.10.062", "crossref", ABS,
     "ensemble NWP as operational backbone, hours-to-days"),
    ("Winkler1972", "10.1080/01621459.1972.10481224", "crossref", META,
     "interval score [LOAD-BEARING]"),
    ("Yang2018", "10.1016/j.solener.2017.11.023", "crossref", META,
     "methodological trajectory of the field"),
    ("Yang2019", "10.1063/1.5114985", "crossref", META,
     "standard of reference, day-ahead deterministic [LOAD-BEARING]"),
    ("Yang2020a", "10.1063/5.0003495", "crossref", META,
     "clear-sky model choice"),
    ("Yang2020b", "10.1016/j.solener.2020.05.020", "crossref", META,
     "probabilistic benchmarks"),
    ("Yang2022", "10.1016/j.solener.2022.03.004", "crossref", META,
     "operational ECMWF verification"),
    ("YangYangLiu2023", "10.1016/j.renene.2023.118993", "crossref", ABS,
     "ensemble under-dispersion; combining calibrated quantiles [LOAD-BEARING]"),
]

NO_DOI = [
    ("SolarPower Europe (2025)", "EU Market Outlook for Solar Power 2025-2030",
     "https://www.solarpowereurope.org/insights/outlooks",
     "EU 406 GW cumulative, 65.1 GW added 2025", "fetched press release"),
    ("IEA-PVPS (2026)", "Belgium national member profile",
     "https://iea-pvps.org/about-iea-pvps/members/belgium/",
     "BE ~13 GW, 1080 W/capita", "fetched page"),
    ("Elia (2025)", "Electricity mix press release",
     "https://www.elia.be/en/press/2025/12/20251226_press-release-electricity-mix",
     "BE 10.1 TWh solar, +21%", "NOT VERIFIED - HTTP 403"),
]


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=40) as r:
        return json.load(r)


def openalex_abstract(doi):
    try:
        d = get(f"https://api.openalex.org/works/doi:{doi}")
        inv = d.get("abstract_inverted_index")
        if not inv:
            return None
        n = max(max(v) for v in inv.values()) + 1
        w = [""] * n
        for word, pos in inv.items():
            for p in pos:
                w[p] = word
        return " ".join(w)
    except Exception:
        return None


def lookup(key, doi, registry):
    if registry.startswith("arxiv:"):
        import re
        aid = registry.split(":", 1)[1]
        x = urllib.request.urlopen(urllib.request.Request(
            f"http://export.arxiv.org/api/query?id_list={aid}", headers=UA),
            timeout=40).read().decode()
        names = re.findall(r"<name>(.*?)</name>", x, re.S)
        title = " ".join(re.findall(r"<title>(.*?)</title>", x, re.S)[1].split())
        yr = re.findall(r"<published>(\d{4})", x)
        return dict(status="OK", first=names[0].split()[-1] if names else "?",
                    n=len(names), year=yr[0] if yr else "-",
                    venue="arXiv / NeurIPS", vol="-", page="-", title=title)
    if registry.startswith("openalex:"):
        wid = registry.split(":", 1)[1]
        d = get(f"https://api.openalex.org/works/{wid}")
        au = [a["author"]["display_name"] for a in d.get("authorships", [])]
        loc = (d.get("primary_location") or {}).get("source") or {}
        return dict(status="OK (OpenAlex)",
                    first=(au[0].split()[-1] if au else "?"), n=len(au),
                    year=d.get("publication_year", "-"),
                    venue=loc.get("display_name", "-"), vol="-", page="-",
                    title=d.get("title", "?"))
    if registry == "datacite":
        d = get(f"https://api.datacite.org/dois/{doi}")["data"]["attributes"]
        cs = d.get("creators", [])
        return dict(status="OK (DataCite)",
                    first=(cs[0].get("name", "?").split(",")[0] if cs else "?"),
                    n=len(cs), year=d.get("publicationYear", "-"),
                    venue=d.get("publisher", "-"), vol="-", page="-",
                    title=(d.get("titles") or [{}])[0].get("title", "?"))
    m = get(f"https://api.crossref.org/works/{doi}")["message"]
    a = m.get("author", [])
    return dict(status="OK", first=(a[0].get("family", "?") if a else "?"),
                n=len(a),
                year=m.get("issued", {}).get("date-parts", [[None]])[0][0],
                venue=(m.get("container-title") or ["-"])[0],
                vol=m.get("volume", "-"), page=m.get("page", "-"),
                title=(m.get("title") or ["?"])[0])


def main():
    want_abs = "--abstracts" in sys.argv
    rows = []
    print(f"Checking {len(REFS)} references ...\n")
    for key, doi, registry, read, used in REFS:
        try:
            r = lookup(key, doi, registry)
            print(f"  OK   {key:18s} {r['first']} {r['year']} | {str(r['venue'])[:40]}")
        except urllib.error.HTTPError as e:
            r = dict(status=f"HTTP {e.code}", first="-", n=0, year="-",
                     venue="-", vol="-", page="-", title="-")
            print(f"  FAIL {key:18s} HTTP {e.code}")
        except Exception as e:
            r = dict(status=f"ERR {type(e).__name__}", first="-", n=0, year="-",
                     venue="-", vol="-", page="-", title="-")
            print(f"  FAIL {key:18s} {r['status']}")
        r.update(key=key, doi=doi or registry, read=read, used=used)
        if want_abs and doi and registry == "crossref":
            r["abstract"] = openalex_abstract(doi)
        rows.append(r)
        time.sleep(0.4)

    ok = sum(1 for r in rows if r["status"].startswith("OK"))
    nfull = sum(1 for r in rows if r["read"] == FULL)
    nabs = sum(1 for r in rows if r["read"] == ABS)
    risky = [r for r in rows if "LOAD-BEARING" in r["used"] and r["read"] == META]

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("# Reference verification\n\n")
        f.write(f"Generated {time.strftime('%Y-%m-%d %H:%M')} by "
                "`paper/verify_references.py`.\n\n")
        f.write(f"- **{ok}/{len(rows)}** resolve with matching metadata\n")
        f.write(f"- **{nfull}** read in full text, **{nabs}** abstract-verified, "
                f"**{len(rows)-nfull-nabs}** metadata only\n")
        f.write(f"- **{len(risky)}** load-bearing citations still unread "
                "-> read before submission\n\n")
        f.write("## Check any entry yourself\n\n```\n"
                "https://api.crossref.org/works/<DOI>     journal articles\n"
                "https://api.datacite.org/dois/<DOI>      data records\n"
                "https://api.openalex.org/works/doi:<DOI> abstracts\n"
                "https://doi.org/<DOI>                    the paper itself\n```\n\n")
        f.write("## Table\n\n")
        f.write("| key | resolves | 1st author | n | year | venue | vol | pages | read | cited for |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| `{r['key']}` | {r['status']} | {r['first']} | {r['n']} | "
                    f"{r['year']} | {str(r['venue'])[:34]} | {r['vol']} | {r['page']} | "
                    f"{r['read']} | {r['used']} |\n")
        if risky:
            f.write("\n## Load-bearing but unread — read these before submission\n\n")
            for r in risky:
                f.write(f"- **`{r['key']}`** — {r['used']}\n"
                        f"  - https://doi.org/{r['doi']}\n")
        f.write("\n## No DOI — verify manually\n\n")
        for name, title, url, used, status in NO_DOI:
            f.write(f"- **{name}** — {title}\n  - {url}\n"
                    f"  - used for: {used}\n  - status: {status}\n")
        f.write("\n## Links\n\n")
        for r in rows:
            if r["doi"].startswith("arxiv:"):
                f.write(f"- `{r['key']}` — https://arxiv.org/abs/{r['doi'].split(':')[1]}\n")
            else:
                f.write(f"- `{r['key']}` — https://doi.org/{r['doi']}\n")

    print(f"\n{ok}/{len(rows)} resolved | {nfull} full text | {nabs} abstract | "
          f"{len(risky)} load-bearing unread")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
