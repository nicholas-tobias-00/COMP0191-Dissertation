# Feature Engineering — Summary (04)

Synthesis of the feature-engineering phase. Companion to `fch4_drivers_and_features_review.md` (the driver review) and `F01_results.md` (the ablation).

## What was done

1. **Driver review** (`fch4_drivers_and_features_review.md`) — literature + NWFP data audit identifying the missing dominant driver: **animals in the flux footprint** (Felber 2015: cows ×100 over bare-soil flux).
2. **Management-event features** (`src/features/build_management_features.py` → `data/Hourly/management_features.csv`) — hourly exp-decay time-since-event channels (fertN, manure, cut, lime, cultiv) at site + tower-area scope. Field→catchment mapping = complete Appendix D table from `NWFP_UG_Design_Develop.pdf` (Tower 4 = Catchment 4 = {NW005, NW006}; Tower 9 = Catchment 9 = {NW013, NW039}); D-28.
3. **F-01 ablation** (`F01_feature_ablation_RFm.ipynb`) — cumulative P1–P6 on RFm, Towers 4 & 9, R-02 harness, SHAP.

## Headline result

**Livestock density is the #1 FCH₄ predictor at Tower 4** — confirming the review's central hypothesis.

- SHAP: `_lsu` (livestock units) ranks **first** (mean|SHAP| 28.2), ~2× FCO₂ and ~3× soil temperature; two of the top-four features are animal-derived.
- Adding **P1 livestock** lifts Tower 4 short-gap R² from **+0.156 → +0.256** (Δ+0.10), the largest single jump in the whole gap-filling programme.

## The arc of the whole project (R² at Tower 4, short gaps)

| Stage | Tower 4 short-gap R² | What changed |
|---|---|---|
| R-02 (met-only, no FC) | ≈ −0.13 | realistic met-only floor (D-22) |
| 03b R-02-CO2 (+ gap-filled FCO₂) | +0.156 | CO₂-flux proxy recovered |
| **F-01 + livestock** | **+0.256** | animal-footprint signal added |

The two decisive levers were **FCO₂** (an ecosystem-activity proxy) and **livestock** (the animal source) — exactly what the source-attribution analysis predicted; meteorology and the remaining biophysical groups (wind/moisture/soil-temp/chemistry) add little.

## Honest caveats (from `F01_results.md`)

- **Diminishing returns beyond livestock**: P3–P6 are flat; the recoverable signal is mostly animal + FCO₂.
- **Management features (P2) overfit** — mildly at Tower 4, **catastrophically at the data-poor Tower 9** (R² → −0.86), due to small training sets + management-timing distribution shift (Red-farmlet arable conversion). The cumulative design lets that damage persist; a non-cumulative ablation and pruned management set are the fixes.
- **Tower 9 is data-bound** (2,288 train rows): even livestock marginally hurts; needs regularization/more data, not more features.
- **Livestock is daily-resolution** (no GPS collars, unlike Felber 2015) and **footprint is approximate** (own-catchment + wind features; no site geometry).
- **Upper-bound inheritance**: BASE still includes observed FCO₂ at gap points (D-22 caveat from 03b).

## Recommended next steps

1. Prune management to 2–3 tower-specific recency channels; re-run as **leave-one-group-in** ablation.
2. **Carry livestock features into the forecasting phase** (`05_benchmarking`) — lagged livestock is a legitimate, high-value, non-co-failed predictor.
3. Regularize / pool data for Tower 9.
4. (Optional) proper flux-footprint model if site geometry is obtained.

*Source: `fch4_drivers_and_features_review.md`, `F01_results.md`, `src/features/build_management_features.py`, `results/benchmarks.csv` (F-01), `results/f01_shap_tower4.csv`. Decisions D-27, D-28.*
