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

## F-02 follow-up (done) — stocking density + pruned management + pooling

See `F02_results.md`. Three results:
1. **Pruned management fixes the F-01 overfit** — Tower 9 management went from −0.86 (F-01, 12 cols) to **+0.01…+0.04** (F-02, 2 tower-specific cols, leave-one-group-in).
2. **Stocking density (LSU/ha, Appendix D areas) pays off in a pooled multi-area model** — pooling T2(2018)+T4+T9 with density-normalised livestock lifts **Tower 9 to R² ≈ +0.21…+0.29** (best in the project; vs pooled-count +0.09/+0.18 and solo ≈ 0). Density is provably inert for single-tower / equal-area cases (Cat 4 = Cat 9 = 7.75 ha) — it only helps when areas differ (T2 = 6.65 ha), which the pool provides.
3. **Livestock stays the top per-tower driver** (Tower 4 Δ+0.10).

## F-03 follow-up (done) — partial pooling across all towers

See `F03_results.md`. Compared solo / full-pool / **partial-pool** (pooled + tower-indicator dummies), all with stocking density, evaluated on all three towers:
- **Partial pooling ≥ full pooling at every tower** — the recommended default.
- **Tower 9 rescued** (solo ≈ 0 → pooled ≈ +0.29; partial ≈ full).
- **Tower 2 benefits most from the tower dummy** (partial −0.245 vs full −0.301 short-gap) — it's the most "different" tower; the dummy gives it its own baseline. Still negative (D-15 split, not the model).
- **Tower 4 (data-rich) protected** — the dummy keeps partial ≈ solo, avoiding the small full-pool dip.

**Method notes (recorded in `F03_results.md §4–§6`):** every predictor is *shared* (one response shape across catchments); only the tower dummy is tower-specific (per-tower level). Pooling adds *rows* not features (T9: 5,387 → 12,558); a tower is predicted from its *own* features but the learned rule draws on all towers (borrowing strength, not leakage). **R² is scored strictly per tower** (own test gaps, own baseline, identical held-out points across variants). Legitimate, literature-backed: partial pooling/multilevel (Gelman & Hill 2007), Mixed-Effects RF (Hajjem 2014), and the EC-flux upscaling paradigm — FLUXCOM (Jung 2020), **UpCH4 (McNicol 2023, CH4-specific)**, Tramontana 2016, Liang 2019.

## F-04 follow-up (done) — re-testing R-03's SWC/TS lags

See `F04_results.md`. Added R-03's SWC/TS 1–4 week lags (D-23) back into the density + partial-pooling models:
- **Lags do NOT help Tower 9** (Δ ≈ −0.00) — R-03's `RF_lag` advantage doesn't transfer, because FCO₂ + density + pooling already encode the slow soil-moisture/temperature memory the lags proxied (**redundant** on the rich base).
- **Lags help the weakest-base tower most** — Tower 2 partial-pool Δ **+0.116** (best Tower 2 yet, still negative); Tower 4 marginally at medium/long gaps.
- **Lesson:** feature value is context-dependent — a feature decisive on a weak base (R-03) can be redundant on a strong one. Keep lags (cheap, help T2/T4-long-gaps), but pooling+density+FCO₂ is the Tower 9 lever (D-31).

## F-05 follow-up (done) — pruned field-event (management) features on the rich base

See `F05_results.md`. Re-added the pruned tower-specific management (cut + manure recency, dropped after F-02) onto the F-04 config:
- **Small, non-harmful bump** (overall-median Δ: Tower 2 +0.013, Tower 4 +0.012, Tower 9 +0.005) — largest at the weaker-base towers, negligible at strong Tower 9.
- **Same pattern as F-04 lags: redundant on the rich base** (FCO₂ + density + pooling already encode most of what fertiliser/cut events signal). Keep it (cheap, never hurts) but it is **not a lever** (D-32).
- **Feature engineering exhausted for hand-crafted features** — P1–P6, pooling, lags, management all tested; returns plateaued. Standard config = partial pool + density + lags + pruned management.

## F-06 follow-up (done) — REddyProc-style met gap-filling + GPP (D-33)

See `F06_results.md`. Prompted by the NWFP/REddyProc EC report. We had always **mean-imputed** the met drivers; `src/data/reddyproc_pipeline.py` gap-fills them properly (interp + mean-diurnal-course) and adds **GPP/Reco** (nighttime Lloyd-Taylor partitioning of CO₂ flux).
- **Met-fill beats mean-imputation** (overall Δ +0.017…+0.076, largest at coverage-poor Tower 2) — the **first input-fix improvement**, and **not** redundant (it fixes inputs rather than adding info the base already had).
- **GPP adds more** — **Tower 9 → +0.335 (new project best)**, Tower 4 +0.163, Tower 2 −0.045 (best yet). GPP = real productivity driver, beats the crude SWIN×TA proxy (P5).
- **New best config:** partial pool + density + lags + pruned management + **gap-filled met drivers + GPP/Reco**.
- u*-threshold filtering produced (reported separately; CH4 ebullition caveat — not applied to R²).

## F-07 (Tower 2 only) — the broken evaluation, fixed (D-34)

See `F07_results.md`. Tower 2's −16.9/−0.045 was a **broken evaluation**: its CH4 spans only Oct 2017–Jun 2019, with cattle in 2018 (FCH4≈42) but **none in 2019** (FCH4≈2) — so the D-15 year split trained the high-flux regime and tested the near-zero one. Gap-filling is interpolation → the correct evaluation is a **full-period gap-CV**.
- **RFm pooled, full-period CV = +0.519 (best in project, exceeds 0.5)**; solo +0.394.
- **MDS stays −0.49** (livestock-blind) → **RFm beats MDS by ~1.0 R² unit** — the clearest "improvement over MDS".
- Caveat: Tower 2's high R² = unusually discriminable livestock-on/off regime; **not directly comparable to 4/9's ~0.3**. Implication: re-evaluate 4/9 under the same full-period CV for consistency.

## F-08 (all towers) — EC-tower vs external (SMS/MET) sensor sourcing (D-35)

See `F08_results.md`. The project sources almost every driver from the **EC tower**, switching to the external catchment network only for soil moisture (D-18) — but soil *temperature* used a cross-tower **EC proxy** (`TS_1_1_1 [Tower 9]`, D-16) despite a well-correlated (r≈0.98) per-catchment external twin. Built a parallel external-sourced layer (`build_sms_met_dataset.py` → `consolidated_hourly_SMS_MET.csv` + `reddyproc_processed_SMS_MET.csv`; originals untouched) swapping all overlapping drivers to external, and re-evaluated **all three towers under full-period gap-CV** (EC vs EXT; harness validated, EC T2 solo = 0.395 = F-07).
- **External sourcing is essentially neutral for the RF** — pooled RFm gains a small, consistent **+0.012–0.014 at every tower** (EXT pool: T2 0.490 / T4 0.376 / T9 0.364), never hurts despite Site-level met + r≈0.17 wind. Same "redundant on the rich base" pattern as F-04/F-05.
- **Per-catchment soil-temperature fix vindicated** (net-positive pooled at all towers) → adopt it for spatial consistency with soil moisture.
- **Biggest result = the EC baseline under full-period gap-CV:** re-evaluating 4/9 with interpolation-style CV raises **T4 +0.163→+0.362**, T9 +0.335→+0.350 → all three towers now consistent **≈0.35–0.49, each ≈0.6–1.0 over MDS**.
- **Lesson:** external sourcing is a *consistency/robustness* improvement (adopt external soil temp; prefer external met on coverage grounds), **not a new accuracy lever**.

## Recommended next steps

1. **Adopt partial pooling + stocking density (+ SWC/TS lags)** (D-30/D-31) as the standard multi-tower configuration; carry into forecasting (`05_benchmarking`) as a global model with per-tower effects.
2. Keep **pruned tower-specific management**; drop site-level + fertN_rate channels.
3. Optionally replace one-hot tower dummies with **continuous tower descriptors** (fenced area, soil type) to generalise to unseen catchments.
4. **Tower 2 split redesign** (D-15) is now Tower 2's limiting factor — pooling has done what it can.
5. (Optional) proper flux-footprint model if site geometry is obtained.

*Source: `fch4_drivers_and_features_review.md`, `F01_results.md`, `src/features/build_management_features.py`, `results/benchmarks.csv` (F-01), `results/f01_shap_tower4.csv`. Decisions D-27, D-28.*
