# Journal-specific choices in this manuscript

This manuscript was prepared for **Electric Power Systems Research (EPSR, Elsevier)**.
Some content exists only because of EPSR's scope or limits. This file lists every such
choice, so the paper can be moved to another journal without confusion.

Four labels are used:

| Label | Meaning | Where it is marked |
|---|---|---|
| `[EPSR-SCOPE]` | Text added to show a link to power-system operation (EPSR scope requirement). | In the `.tex` files, wrapped in `\ifepsr ... \else ... \fi` |
| `[EPSR-LIMIT]` | Content cut or moved because of the 7000-word limit. | This file only (the full text survives elsewhere) |
| `[ELSEVIER-FORMAT]` | Formatting required by Elsevier journals, not only EPSR. | This file only |
| `[GENERAL]` | Improvements that any journal would expect. Keep them for every journal. | This file only |

## How to switch to another journal

1. In `main.tex`, change `\epsrtrue` to `\epsrfalse`. Every `[EPSR-SCOPE]` passage then
   switches to its neutral wording (or disappears), and the text still reads correctly.
2. Find every passage with `grep -rn "EPSR-SCOPE" sections/`.
3. Check the `[EPSR-LIMIT]` list below. If the new journal allows more words, restore
   the items you want.
4. If the new journal is not Elsevier, replace the `[ELSEVIER-FORMAT]` items.
5. Change `\journal{...}` in `main.tex` and the target journal in the cover letter.

## [EPSR-SCOPE] passages (toggled by `\ifepsr`)

| File | Passage | EPSR wording | Neutral wording (`\epsrfalse`) |
|---|---|---|---|
| `sections/00_abstract.tex` | first sentence | "...household energy management, aggregator bidding and distribution-grid operation" | "...household energy management and aggregator bidding" |
| `sections/01_introduction.tex` | paragraph 2 | DSO reverse power flow and feeder congestion; balance-responsible parties and imbalance cost | "Aggregators bid distributed capacity into day-ahead markets." |
| `sections/06_discussion.tex` | end of Section 6.2 | "a grid operator or aggregator should screen new installations..." | "new installations should be screened for implausible metadata." |
| `sections/06_discussion.tex` | Limitations, "Grid value" bullet | bullet present | bullet removed |
| `sections/07_conclusions.tex` | last paragraph, first words | "For grid operators and aggregators, a newly connected rooftop..." | "A newly connected rooftop..." |

| `sections/05_results.tex` | Section 5.4, paragraph "Aggregated forecasts" | paragraph present (regional and fleet nRMSE, daily imbalance) | paragraph removed |
| `supplementary.tex` | Section "Aggregated forecast error" | section present | delete the section by hand (the supplementary has no toggle) |

The grid-facing result comes from `v3_ensemble/src/eval/grid_aggregate.py` (also tagged
`[EPSR-SCOPE]`). For a forecasting or solar journal it is optional.

## [EPSR-LIMIT] items cut for the 7000-word limit

The full original text is in `paper/paper.md` (the long draft).

| Item | Status in this manuscript | Where the content is now |
|---|---|---|
| Nomenclature (about 400 words) | removed; symbols defined in the text | `paper/paper.md`, section "Nomenclature" |
| Section 1.6, paper structure | removed | `paper/paper.md` |
| Detailed literature review (Sections 2.1 to 2.6 of the draft) | shortened to about 730 words | `paper/paper.md`, Section 2 |
| Fig. 6, ensemble ablation chart | dropped (same data as Table 3) | `paper/figures/fig6_ensemble_ablation.pdf` |
| Table 4 of the draft, headline arms A to D | text in Section 5.4 | Supplementary Table S1 |
| Table 5 of the draft, proposed configuration | dropped | Methods text, Supplementary Table S2 |
| Table 6 of the draft, MOS at power level | text in Section 5.5 | `paper/paper.md` |
| Table 7 of the draft, known vs unseen rooftop | text in Section 5.6 | `paper/paper.md` |
| Table 9 of the draft, calibration sweep | Fig. 9 only | Supplementary Tables S4 and S5 |
| Table 10 of the draft, positioning against published work | about 200 words in Section 5.7 | Supplementary Table S6 |
| Shading result | removed from Section 3 | Supplementary Section S5 |
| GPU test, run times | removed from Section 4 | Supplementary Section S7 |
| Section 6.1 of the draft, "four routes" | removed (summarised in Conclusions) | `paper/paper.md` |

## [ELSEVIER-FORMAT] items

- Document class `elsarticle` with `epsr.sty` and Harvard citations (`elsarticle-harv.bst`).
- Highlights: 3 to 5 bullets, at most 85 characters each (`sections/00_highlights.tex`).
- Keywords: 1 to 7, single words preferred (`main.tex`).
- Abstract: at most 250 words.
- Declarations: data availability, CRediT, competing interest, generative AI
  (`sections/08_declarations.tex`).
- Cover letter must state the article length (EPSR).

## [GENERAL] improvements (keep for every journal)

| Priority | Item | Status |
|---|---|---|
| 1 | Bootstrap confidence intervals, Diebold-Mariano tests, several LOSO seeds | done: `src/eval/revision_stats.py` (Methods 4.6, CIs and p-values in Section 5, Supplementary); `src/models/loso_seeds.py` (Table 4 seed mean and spread) |
| 2 | System 81376: full fleet (33 systems) is the primary result; exclusion is a disclosed sensitivity analysis | done: E0, E3, E4 and MOS rerun on 33 systems (`consolidate.py --full-fleet`, `mos_pv_integration.py --full-fleet`) |
| 3 | Smart persistence and physical model-chain baselines | done: `src/models/baselines.py`; Methods 4.1, Section 5.6, Supplementary |
| 5 | Check the claims cited on metadata only (Mayer 2021, 2022, 2023; Lauret 2019; Yang 2019; Verzijlbergh 2015); fill in CRediT and AI declaration | to do by the authors |

Priority 4 (grid-facing metric) is `[EPSR-SCOPE]`, listed above.
