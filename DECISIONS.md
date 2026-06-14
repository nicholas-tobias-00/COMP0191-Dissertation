# DECISIONS.md
_Log every methodological choice here — one entry per decision. Never delete entries; add a "superseded by D-XX" note if reversed._

---

## Log

### D-01 — 2026-06-12 — Infrastructure
**Decision:** Compile all annual CSV slices into single multi-year files (`data/Compiled/`) before analysis, rather than loading per-year files inline in each notebook.  
**Rationale:** Avoids repeating glob+concat boilerplate; compiled files are deduplicated and have consistent datetime parsing applied once.  
**Alternatives considered:** Load Consolidated files directly in each notebook via a shared utility function.

---

### D-02 — 2026-06-12 — Data
**Decision:** Transform livestock weight and condition score data from wide format to long format (`livestock_weight_long.csv`, `livestock_condition_score_long.csv`).  
**Rationale:** Long format required for time-series joining against environmental covariates by date; wide format has hundreds of sparse date columns.  
**Alternatives considered:** Keep wide format and melt on-the-fly downstream.

---

### D-03 — 2026-06-12 — Data
**Decision:** Keep livestock location data in wide format (rows = animals, columns = dates).  
**Rationale:** Used primarily to derive field-occupancy counts (already in `Animal_location_counts_*.csv`); wide format preserves structure for individual-animal lookups.  
**Alternatives considered:** Melt to long — deferred until a downstream use case requires it.

---

### D-04 — 2026-06-12 — Modelling
**Decision:** Use temporal cross-validation exclusively (e.g. train 2018–2021, test 2022–2023). No random train/test splits permitted anywhere in the pipeline.  
**Rationale:** Random splits cause data leakage in time series — autocorrelation means future information contaminates training. EC CH₄ data is strongly non-stationary and seasonal; leakage would produce optimistically biased metrics that don't reflect real forecasting performance.  
**Alternatives considered:** k-fold CV — rejected because it violates temporal ordering.

---

### D-05 — 2026-06-12 — Modelling
**Decision:** Benchmark RF and XGBoost against LSTM, TFT, ARIMA, and a persistence/seasonal mean baseline under identical experimental conditions. No architecture is assumed superior in advance.  
**Rationale:** Zeng et al. (2023) demonstrated that simple linear models outperform transformer architectures on 9 standard benchmarks, attributing failure to permutation-invariant self-attention destroying temporal ordering. The comparative question for EC CH₄ at farm scale is genuinely open. Irvin et al. (2021) and Kim et al. (2020) show RF/XGBoost dominate gap-filling but this has never been tested for multi-step forecasting.  
**Alternatives considered:** Start with deep learning only — rejected; start with tree-based only — rejected. Rigorous benchmarking is a core project deliverable.

---

### D-06 — 2026-06-12 — Modelling
**Decision:** Uncertainty quantification via quantile ML or conformal prediction is a non-negotiable structural requirement, not an optional add-on.  
**Rationale:** Irvin et al. (2021) documented that raw ML uncertainty estimates are systematically underestimated across all 17 FLUXNET-CH4 benchmark sites. A scenario analysis module producing point predictions without calibrated intervals is not actionable for farm management decisions. Oulaid et al. (2025) validated quantile ML specifically at NWFP for soil moisture.  
**Alternatives considered:** Bootstrap confidence intervals — viable but more computationally expensive; Monte Carlo dropout — requires deep learning architecture.

---

### D-07 — 2026-06-12 — Modelling
**Decision:** Apply SHAP (SHapley Additive exPlanations) for global and local interpretability across all model types.  
**Rationale:** Buzacott et al. (2024) demonstrated that raw predictive accuracy masks highly entangled multi-driver CH₄ dependencies that only SHAP decomposition reveals. Partridge et al. (2024) applied SHAP at NWFP successfully. RQ4 specifically requires decoding interactions between continuous environmental variables and discrete management interventions.  
**Alternatives considered:** Permutation importance only — cheaper but provides no local explanations; LIME — less stable than SHAP for tree-based models.

---

### D-08 — 2026-06-12 — Data
**Decision:** Use ERA5 reanalysis data as fallback for meteorological driver variables when local NWFP sensors fail or have gaps.  
**Rationale:** Zhu et al. (2023a) validated ERA5 substitution specifically for UK managed pasture EC gap-filling and confirmed key environment-flux responses are preserved. NWFP EC data has persistent sensor gaps; without a fallback, large continuous gaps cannot be gap-filled or used as forecasting inputs.  
**Alternatives considered:** Simple interpolation for sensor gaps — insufficient for gaps >12 days (Zhu et al. finding).

---

### D-09 — 2026-06-12 — Scope
**Decision:** The primary prediction target is ecosystem-scale EC CH₄ flux (half-hourly, from `greenhouse.csv`), not animal-scale GreenFeed measurements.  
**Rationale:** EC captures the integrated ecosystem signal from enteric fermentation, soil processes, and manure simultaneously. GreenFeed captures intermittent individual animal breath samples only (Partridge et al. 2024). The EC forecasting question is entirely unanswered; the GreenFeed question has been partially addressed (r=0.619 with Gradient Boosting). The project novelty rests on this distinction.  
**Alternatives considered:** GreenFeed as target — already explored by Partridge et al.; not novel.

---

### D-10 — 2026-06-12 — Scope
**Decision:** Build toward a "digital shadow" (unidirectional predictive model + scenario analysis interface), not a full bidirectional digital twin.  
**Rationale:** Purcell & Neubauer (2023) established through systematic review that true bidirectional digital twins are rare even in well-resourced agricultural deployments; most implementations are digital shadows. Fakeye et al. (2024) confirmed that what-if scenario simulation is the critically absent capability. A digital shadow with scenario analysis is both achievable and novel.  
**Alternatives considered:** Full digital twin — out of scope for a single MSc dissertation; monitoring dashboard only — insufficient novelty.

---

### D-11 — 2026-06-12 — Scope (RESOLVED)
**Decision:** Build **three separate models**, one per EC tower / ecosystem: Tower 2, Tower 4, Tower 9. Each tower represents a distinct field/ecosystem at NWFP — they are not redundant sensors of the same location. All three `FCH4_1_1_1 [Tower N]` columns are retained in `greenhouse_hourly.csv` and `consolidated_hourly.csv`. Models are trained and evaluated independently.  
**Rationale:** EDA confirmed all three towers are operationally active (Tower 4 = 44.6% valid; Tower 9 = 25.6%; Tower 2 = 12.1%, 1,675-day gap May 2019–Jan 2024). The three towers are distinct spatial units with different management, land cover, and livestock exposure — merging them or treating one as "primary" conflates the three ecosystems. Tower 2's sparse coverage is a data constraint for *that model*, not a reason to deprioritise it.  
**Alternatives considered:** Tower 4 only — leaves two ecosystems entirely unmodelled. Aggregate all tower flux into a single target — scientifically unjustified; footprints do not overlap.  
**Implication:** Temporal splits, gap-filling replications (R-01 through R-04), benchmarking, and SHAP analyses are all run per tower. Tower 4 is a natural starting point given its coverage, but all three towers are co-equal deliverables.

---

### D-12 — 2026-06-12 — Data
**Decision:** Resample all compiled data to a common 1-hour resolution and outer-join into a single `consolidated_hourly.csv` via `src/data/consolidate_hourly.py`. Sub-hourly data (15-min measurements, 30-min greenhouse) is aggregated by hourly mean; daily data (livestock counts) is upsampled by forward-filling the midnight value for up to 23 hours within the same day (`ffill(limit=23)`).  
**Rationale:** A common temporal index is required before any feature-target alignment or model training. Hourly is the coarsest granularity that preserves the diurnal signal present in EC CH₄ flux without inflating the dataset size. The `ffill(limit=23)` rule ensures a missing day stays NaN across all 24 of its hours rather than inheriting the previous day's value — preserving the "no gap-filling" invariant for daily-resolution sources.  
**Alternatives considered:** Keep data at native resolutions and align lazily in each notebook — rejected because it duplicates alignment logic and risks inconsistency. Resample to daily — loses diurnal signal needed for EC modelling.

---

### D-13 — 2026-06-12 — Data
**Decision:** Apply a physical plausibility filter to `FCH4_1_1_1 [Tower N]` columns before any model training, rejecting values outside [−500, 3000] nmol m⁻² s⁻¹ as a preliminary bound (to be tightened against site-specific literature).  
**Rationale:** Section 6 EDA found Tower 4 FCH4 has a mean of 15,420 nmol m⁻² s⁻¹ despite a p99 of only 548 — extreme outliers are pulling the mean to a physically impossible value. Irvin et al. (2021) Table 1 reports site means of 2–150 nmol m⁻² s⁻¹ across 17 wetland sites; agricultural managed grasslands are unlikely to exceed ~1000 nmol m⁻² s⁻¹ under any conditions. Leaving these values in will dominate model training and distort gap-filling metrics.  
**Alternatives considered:** Rolling z-score filter — adaptive but harder to justify with domain knowledge; SSITC flag filtering only — quality flags may not capture all physically implausible instrument artefacts.  
**Note:** SSITC-based filtering (retain flags 0 and 1, reject 2) is applied first; plausibility filter is a second pass. Final bounds to be confirmed at the start of `03_gap_filling`.

---

### D-14 — 2026-06-12 — Data *(REVISED 2026-06-13)*
**Original decision (superseded):** Download ERA5 `ssrd` as the SW_IN predictor — believed SW_IN was absent.  
**Revised decision:** `SWIN_1_1_1 [Tower N]` columns ARE present in `consolidated_hourly.csv` at ~52% availability (Tower 4). The EDA pattern search used `SW_IN_` (underscore after IN) but the actual column name uses `SWIN_` (no underscore). ERA5 is **not required as a blocker** for any replication; it remains a useful optional gap-filler for the ~48% of missing SWIN hours if predictor completeness becomes a limiting factor.  
**Implication:** R-01 was run with `SWIN_1_1_1 [Tower 4]` as the SW_IN predictor at ~52% availability. ERA5 download is deferred to `04_feature_engineering` as an optional enhancement.

---

### D-15 — 2026-06-12 — Modelling
**Decision:** Tower 2 requires a custom temporal split, independent of the standard 2018–2021 / 2022–2023 / 2024 split (D-04) applied to Towers 4 and 9.  
**Rationale:** Tower 2 has a 1,675-day sensor gap (May 2019–Jan 2024). Section 6 confirmed 0% valid FCH4 in both the test (2022–23) and held-out (2024) windows of the standard split — applying it would leave Tower 2 with no evaluable test set.  
**Proposed split for Tower 2:** Use pre-gap data (2018–May 2019) for training; post-gap data (Jan 2024+, once 2024 data is downloaded) as the test set. If the post-gap window is too short (< 6 months), apply leave-one-season-out CV within the pre-gap window only.  
**Note:** Towers 4 and 9 are unaffected (D-04 applies). Final Tower 2 split design to be confirmed at the start of Tower 2 modelling.

---

### D-16 — 2026-06-13 — Data
**Decision:** Use `TS_1_1_1 [Tower 9]` (71% available) as the soil temperature predictor for all three tower models, rather than the co-located `TS_1_1_1 [Tower N]` sensors.  
**Rationale:** Tower 4 soil temperature (`TS_1_1_1 [Tower 4]`) is only 9.6% available — unusable as a predictor. Tower 2 soil temperature is similarly sparse (~5%). Tower 9 has `TS_1_1_1` at 71% availability and `TS_3_1_1` at 71% — making it the only usable soil temperature source for all three models. Given that the NWFP catchments share the same underlying geology and are geographically proximate, Tower 9's soil temperature is a reasonable cross-tower proxy.  
**Alternatives considered:** ERA5 land surface temperature — valid fallback if Tower 9 TS ever becomes unavailable; spatial interpolation from Tower 9 to Tower 2/4 positions — unnecessary given geographic proximity.

---

### D-17 — 2026-06-13 — Results
**R-01 outcome (Tower 4 FCH4, test 2022–2023, 5 permutations, Catchment 4 SM corrected):**  
RF median R²=0.145 (RMSE=121.2, MAE=62.4 nmol m⁻² s⁻¹); XGBoost median R²=0.067 (RMSE=126.4, MAE=71.0).  
**Interpretation:** Much lower than Irvin et al. (2021) RF benchmark of R²=0.79 across 17 wetland sites. Expected for three reasons: (1) managed temperate grassland has far lower and more episodic CH₄ fluxes than wetlands (~10× lower signal); (2) inter-annual variability in grassland management (stocking, cutting) creates non-stationarity that a static 4-year training window cannot capture fully; (3) the predictor set lacks some Irvin predictors (notably soil temperature from the correct co-located sensor). These results establish a realistic grassland-specific baseline rather than a failure to replicate — the methodology is faithful to Irvin even if site characteristics differ fundamentally.

---

### D-18 — 2026-06-13 — Data (SPATIAL ALIGNMENT RULE)
**Decision:** Each tower model must use only predictor data from its own spatially matched catchment. The mapping is Tower N = Catchment N (confirmed by user):
- Tower 2 model → `[Catchment 2]` soil moisture, Tower 2 met sensors
- Tower 4 model → `[Catchment 4 After  2013/08/13]` soil moisture (note exact column name), Tower 4 met sensors
- Tower 9 model → `[Catchment 9]` soil moisture, Tower 9 met sensors

**Never average soil moisture across catchments from different towers.** Each catchment has distinct management (stocking, cutting, fertiliser events) that directly drives CH₄ flux at that tower — cross-catchment averaging contaminates the feature with irrelevant management signals from other fields.  
**Rationale:** An early R-01 run incorrectly used the average of Catchments 5–8, 11–13 (chosen for high coverage) rather than Catchment 4 (56% coverage). This artificially boosted training SM availability to 85% and produced inflated R² of 0.209; the correct Catchment 4 SM gives RF R²=0.145 (D-17).  
**Exception:** Soil temperature (`TS_1_1_1`) from the co-located sensor is preferred, but cross-tower use is permitted when the co-located sensor is unavailable (e.g., Tower 4 TS = 9.6% → use Tower 9 TS at 71%, documented in D-16). Soil temperature varies far less across the site than soil moisture does under contrasting management.

---

### D-19 — 2026-06-13 — Results (R-01 multi-tower extension)
**R-01 extended to all three towers.** Median results across 5 permutations:

| Tower | Model | R² | RMSE | MAE | n_train | Split |
|-------|-------|----|------|-----|---------|-------|
| Tower 4 | RF | +0.144 | 121.3 | 62.5 | 7,714 | 2022–2023 |
| Tower 4 | XGBoost | +0.086 | 126.5 | 70.7 | 7,714 | 2022–2023 |
| Tower 9 | RF | −0.027 | 123.5 | 58.8 | 3,981 | 2022–2023 |
| Tower 9 | XGBoost | −0.089 | 128.0 | 62.6 | 3,981 | 2022–2023 |
| Tower 2 | RF | −16.9 | 147.9 | 116.4 | 2,985 | Jan–May 2019 (D-15) |
| Tower 2 | XGBoost | −55.9 | 264.9 | 220.7 | 2,985 | Jan–May 2019 (D-15) |

**Tower 9 near-null R²:** Training data is 48% smaller than Tower 4 (3,981 vs 7,714 rows). The 2022–2023 test distribution differs from 2018–2021, causing high permutation variance (RF R² range: −0.035 to +0.127). Adding management event features and more training data (2024 download) should help.

**Tower 2 split failure:** The D-15 custom split trains on 2018 (all seasons) but evaluates on Jan–May 2019 (winter/spring only). This seasonal mismatch causes the model to predict summer-level fluxes during the low-flux winter/spring test period → catastrophically negative R². The 2018/2019 split is unsuitable. Redesign needed: leave-one-season-out CV within the pre-gap window, or post-gap evaluation on downloaded 2024 data.

**Implication:** Tower 2 R-01 numbers must not be compared against Irvin's benchmark. They reflect split design failure, not model capability. Tower 4 is the only apples-to-apples comparison; Tower 9 is weakly informative.

---

_[Add new entries below this line]_
