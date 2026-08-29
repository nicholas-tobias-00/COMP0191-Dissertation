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

### D-20 — 2026-06-14 — Methods
**Decision:** Implement MDS (Marginal Distribution Sampling) in Python rather than calling the REddyProc R package.  
**Rationale:** REddyProc is an R package; calling it from Python in a Jupyter notebook requires rpy2 which adds a non-trivial dependency and platform-specific setup. The MDS algorithm is fully specified in Reichstein et al. (2005) and re-stated in Zhu et al. (2023a): for each gap position, search ±7/14/28/91-day windows for observations with the same hour (±1h), similar TA (±2.5°C), and similar SW (±50 W/m²) for daytime. The Python implementation in `mds_fill_batch()` replicates this algorithm exactly. Fill rate = 100% for all scenarios at both towers, indicating sufficient temporal depth in the 2018–2024 series.  
**Alternatives considered:** REddyProc via rpy2 — correct but fragile; a simpler mean-diurnal-cycle baseline — too far from the paper's methodology.

---

### D-21 — 2026-06-14 — Data
**Decision:** driver_m for R-02 adds PPFD (`PPFD_1_1_1 [Tower N]`), NETRAD (`RN_1_1_1 [Tower N]`), precipitation (`Precipitation (mm) [Catchment N ...]`), and soil heat flux (`SHF_1_1_1 [Tower N]`) compared to the R-01 feature set. The full driver_m set (11 meteorological variables + 4 cyclical AUX = 15 features) follows Zhu et al. Table 2 for managed pastures.  
**Rationale:** Directly replicates Zhu et al. driver_m specification. All four new columns were confirmed present in `consolidated_hourly.csv` before implementation.  
**Impact:** driver_m training sets are smaller than driver₃ due to stricter dropna: Tower 4 = 7,285 rows (driver_m) vs 10,862 rows (driver₃); Tower 9 = 2,288 vs 4,048.

---

### D-22 — 2026-06-14 — Methods (IMPORTANT METHODOLOGICAL DISTINCTION)
**Decision:** LE (latent heat), H (sensible heat), and FC (CO₂ flux) are deliberately excluded from R-02 driver sets, even though R-01 included them.  
**Rationale:** In real gap-filling, LE, H, and FC are measured by the same EC system as FCH4. If the EC instrument fails (creating a CH4 gap), LE/H/FC are also unavailable — they co-fail with the target. Using them as predictors in R-01 was methodologically incorrect for a realistic gap-filling scenario. Zhu et al. correctly restrict drivers to meteorological variables measured by independent sensors (SW, TA, VPD, etc.) that remain available during EC instrument failures. This explains why R-02 RFm R² (≈−0.10 to −0.13) is lower than R-01 RF R² (+0.086 to +0.144): R-01 inadvertently "cheated" by using co-failed variables that carry strong information about FCH4.  
**Implication:** R-01 results should be interpreted as an upper bound on gap-filling accuracy under an unrealistic feature assumption. R-02 results are the realistic benchmark. Forecasting (R-05+) will use lagged versions of LE/H/FC as valid features since they come from earlier time steps, not the same gap period.

---

### D-23 — 2026-06-14 — Data (R-03 lag feature design)
**Decision:** Use SWC (soil moisture at 10 cm, catchment-matched per D-18) and TS (Tower 9 proxy per D-16) as the lag variables in R-03, with lags at 168h, 336h, 504h, and 672h (1–4 weeks at hourly resolution).  
**Rationale:** Kim et al. (2020) lag water table height (WTH) at 1–4 weeks, exploiting the delayed hydrological response of wetland CH₄ emissions to precipitation/drainage. NWFP has no WTH sensor. SWC captures soil saturation state (the immediate precursor to WTH variation in managed grassland) and TS captures the thermal driving force for methanogenesis — together they are the closest NWFP analog of Kim's WTH lags. The 1–4 week range is taken directly from Kim's specifications.  
**Outcome:** Lag features improved RF at Tower 9 (RF_lag R²=+0.152 vs RF R²=+0.129 for short gaps) confirming site-level hydrological memory. Lag features did not improve RF at Tower 4, suggesting SWC/TS lags carry weaker predictive signal there than WTH did at Kim's wetland sites.  
**Alternatives considered:** DOY-based lag (e.g., 7-day rolling mean) — smoother but less mechanistic; no lags — baseline comparison (RF model).

---

### D-24 — 2026-06-14 — Methods (R-03 SVM/ANN hyperparameters)
**Decision:** SVR: `kernel='rbf', C=1.0, epsilon=0.1, gamma='scale'`. MLPRegressor: `hidden_layer_sizes=(100, 50), activation='relu', max_iter=500, early_stopping=True, n_iter_no_change=20, random_state=42`. StandardScaler applied before both.  
**Rationale:** Kim et al. (2020) used R's `kernlab` (SVM) and `neuralnet` (ANN) packages with 2 hidden layers and RBF kernel — the closest sklearn equivalents are the parameters above. `early_stopping=True` prevents ANN overfitting on small training sets; `gamma='scale'` adapts the RBF bandwidth to the feature variance. These are sklearn defaults closest to Kim's documented hyperparameters; no grid search performed.  
**Outcome:** SVM showed systematic negative MBE (≈−22 nmol m⁻² s⁻¹ at Tower 4, ≈−22 at Tower 9), indicating C=1.0 may be under-regularised for the NWFP flux range. ANN performed best at medium/long gaps at Tower 4 but collapsed at Tower 9 xlong (R²=−0.518, small-sample artefact). Hyperparameter search for SVM noted as a future improvement.  
**Alternatives considered:** C=10 for SVM — may correct underprediction bias; larger ANN architectures — more parameters than training rows at Tower 9 would worsen overfitting.

---

### D-25 — 2026-06-16 — Data (FCO₂ quality control)
**Decision:** Apply a two-pass QC to CO₂ flux (`FC_1_1_1 [Tower N]`) before using or reconstructing it: SSITC flag ∈ {0,1}, then a physical plausibility filter of **[−100, 100] µmol m⁻² s⁻¹**.
**Rationale:** After SSITC filtering, FC's 1st–99th percentile is ≈ [−28, +26] µmol m⁻² s⁻¹ but a tail of gross instrument spikes remains (|FC| up to ~3×10⁵; ~125–190 points beyond ±100 per tower). Managed-grassland NEE rarely exceeds a few tens of µmol m⁻² s⁻¹, so [−100, 100] is a generous bound that removes only clear artefacts. Mirrors the FCH₄ plausibility approach (D-13).
**Used by:** `src/data/fco2_gapfill.py` (the 03b CO₂-augmentation experiment).

---

### D-26 — 2026-06-16 — Methods (CO₂-augmented gap-filling experiment, 03b)
**Decision:** Build a `03b_gap_filling_CO2` experiment that (1) reconstructs FCO₂ from meteorological-only drivers using the R-02 RFm approach, then (2) re-runs R-01/R-02/R-03 with the **observed-where-available** gap-filled FCO₂ as a CH₄ feature. FCO₂ reconstruction is precomputed once to `data/Hourly/fco2_gapfilled.csv` (Towers 2/4/9); the three notebooks load it. Results tagged `R-01-CO2`/`R-02-CO2`/`R-03-CO2`.
**Rationale:** D-22 established that LE/H/FC co-fail with FCH₄, so excluding them gives the realistic-but-poor benchmark. This experiment tests the converse: if FCO₂ is *reconstructed from independent met drivers* it becomes available during a gap, converting a co-failed variable into a usable predictor. FCO₂ itself gap-fills well (RFm test R² ≈ 0.745/0.746 at Towers 4/9; 0.197 at Tower 2 with only 2018 data).
**Outcome:** Adding gap-filled FCO₂ to the met-only RFm (R-02-CO2) moves Tower 4 from negative to **positive** R² (vs-gap −0.128 → +0.156; m-gap −0.160 → +0.111) while the no-FC controls (RF3, MDS) are unchanged — a clean causal demonstration that **FC is the single most informative FCH₄ predictor**. R-03-CO2 ANN reaches +0.12…+0.17 at Tower 4 (best model overall). For models that already had raw FC (R-01, R-03 trees), QC'ing it removes a spurious co-artefact signal, so short-gap RF drops. Tower 9 gains little; Tower 2 improves but stays negative (split design, D-19).
**Caveat (chosen design):** "observed-where-available" means FCO₂ is the real observed value at FCH₄-gap points, re-introducing the co-observation issue (D-22) — so 03b results are an **upper bound**, not operational. The strict operational variant would use `FC_recon` everywhere; deferred. Lagged FCO₂ is a legitimate (non-co-failed) feature for forecasting (R-05+).

**Addendum (2026-07-01) — does REddyProc-gap-filled met (F-06/D-33) improve the FCO2 reconstruction itself?** Small one-off check (temporary script, not wired into the pipeline — `fco2_gapfilled.csv` untouched): re-ran the exact same RFm reconstruction methodology (same QC, same train/test years, same hyperparameters) but swapped the raw/mean-imputed `driver_m` columns for the REddyProc-gap-filled (`__f`) equivalents from `reddyproc_processed.csv`. **Result: Tower 2 jumps from 0.197 → 0.564 (+0.367); Towers 4/9 are essentially unchanged (0.745→0.729, 0.746→0.747).** Same "redundant on the rich base" pattern seen throughout the project (F-04/F-05) — Towers 4/9 already had adequate met coverage so better inputs barely matter, but Tower 2's original reconstruction leaned on crude mean-imputation over its much sparser met record, and fixing that input quality closes most of the gap. This mirrors F-06's own finding (met-fill helped CH4 gap-filling most at coverage-poor Tower 2) at a larger effect size, since FCO2 reconstruction is more input-sensitive than CH4 gap-filling. Not adopted as a pipeline change (03b is a closed experiment) — recorded for reference only.

---

### D-27 — 2026-06-16 — Features (livestock footprint, P1)
**Decision:** Build livestock features from own-catchment head counts (`cattle_/sheep_/lamb_Catchment N`, Tower N = Catchment N, D-18; shed/housed columns excluded): per-species counts, a combined **LSU** (cattle 1.0, sheep 0.1, lamb 0.05), a grazing-presence binary, and 24 h / 7 d lags.
**Rationale:** At a grazed pasture the EC CH₄ signal is dominated by animals in the footprint (Felber et al. 2015: ×100 over bare-soil flux); this was the dominant missing driver across R-01→03b. Counts are 100 % populated with real presence variation (~32 % of hours at Tower 4).
**Outcome:** Validated — `_lsu` is the **#1 SHAP feature** at Tower 4 (mean|SHAP| 28.2, ~2× FCO₂); adding P1 lifts Tower 4 short-gap R² +0.156 → +0.256. Tower 9 (data-poor) does not benefit.
**Caveat:** livestock counts are **daily** (no GPS collars, unlike Felber) and "footprint" is approximated by own-catchment + wind features (no site geometry available).

---

### D-28 — 2026-06-16 — Features (management events, P2) + spatial mapping
**Decision:** Build hourly management-event features (`src/features/build_management_features.py` → `data/Hourly/management_features.csv`) as exponential-decay time-since-event recency per channel (fertN +rate, manure, cut, lime, cultiv; τ = 14/30/21/90/30 d) at **site-level** and **tower-area** scope. Field→catchment mapping = **complete 15-catchment table from `NWFP_UG_Design_Develop.pdf`, Appendix D** (see `CATCHMENT_FIELDS` in the script). Tower management area = its own catchment (D-18): **Tower 4 = Catchment 4 = {NW005 Bottom Burrows, NW006 Burrows}**; **Tower 9 = Catchment 9 = {NW013 Dairy South, NW039 Dairy Corner}**. Tower 2 = Red farmlet (arable from 2019) — deferred.
**Revision (2026-06-16):** an initial draft scoped Tower 4 to the *whole Green farmlet* {NW005/6/9/16/17/45/46/47} — but Appendix D shows those span Catchments 4/5/6/12/13. Corrected to Catchment 4 only (events 495 → 124). Tower 9 was already correct. Re-ran; **conclusions unchanged** (livestock still #1; P2 still weakest/overfitting). Appendix D also provides per-catchment fenced areas (Cat 4 = 7.75 ha, Cat 9 = 7.75 ha) for future stocking-density features.
**Rationale:** Slurry/fertiliser/cutting cause transient CH₄/N₂O pulses; the user guide provides the only available spatial structure (no geometry/area files exist).
**Outcome:** As implemented (12 columns), management features **overfit**: mild R² loss at Tower 4 and a **collapse at Tower 9** (R² → −0.86) driven by small training sets + management-timing distribution shift (Red-farmlet conversion). **Recommendation:** prune to 2–3 tower-specific recency channels and use a non-cumulative (leave-one-group-in) ablation; drop site-level + `fertN_rate`.

---

### D-29 — 2026-06-16 — Features (stocking density) + pooling (F-02)
**Decision:** Add **stocking-density** features — LSU/ha and per-species head/ha — using per-catchment fenced areas from `NWFP_UG_Design_Develop.pdf` Appendix D (`CATCHMENT_AREA_HA`; Cat 4 = Cat 9 = 7.75 ha). Prune management features to **tower-specific cut + manure recency** only (F-01's 12-col set overfit, D-28). Use **leave-one-group-in** ablation (BASE vs BASE+single group) instead of cumulative.
**Rationale & key caveat:** EC flux is areal, so stocking *density* is the physically correct unit. But density = LSU/area is a constant rescale, and RFm is invariant to monotonic rescaling of a single feature — so density is **inert for single-tower (and equal-area T4/T9-pooled) models**. It only adds information when catchments of *different* area share one model.
**Outcome:** Demonstrated by a **pooled T2(2018)+T4+T9 RFm**: density-normalised livestock lifts **Tower 9 to R² ≈ +0.21…+0.29 across all gap lengths** (vs pooled-count +0.09/+0.18 and solo BASE ≈ 0) — the **best Tower 9 result in the project**. Tower 4 also improves modestly with density. Pruned management now **helps** Tower 9 (+0.01…+0.04) vs F-01's −0.86 collapse — confirms the overfit diagnosis. Pooling is a viable fix for data-poor towers; adopt pooled+density going forward.

---

### D-30 — 2026-06-16 — Modelling (partial pooling, F-03)
**Decision:** Adopt **partial pooling** as the standard multi-tower configuration: one RFm trained on Towers 2+4+9 stacked (generic feature names, stocking-density livestock) **plus tower-indicator dummies** (`is_t2/is_t4/is_t9`) — shared relationships, tower-specific level. Compared against full pooling (no ID) and solo per-tower.
**Rationale:** Full pooling forces one relationship on all towers; partial pooling lets each keep its own baseline where it genuinely differs (standard hierarchical / random-intercept idea). Requested as the principled refinement after the F-02 full-pool result. **Shared-feature assumption:** every predictor is shared (one response *shape* learned across catchments); only the tower dummy is tower-specific (per-tower *level*). Strongest for met drivers, engineered for livestock (density ÷ area), weakest for soil/land-use features at Tower 2 (arable). At prediction a tower uses its own feature values, but the learned relationship/RF leaf averages draw on all towers' rows (borrowing strength via more *rows*, not more features; not leakage).
**Evaluation:** pooling changes *training* only; **R² is scored strictly per tower** on that tower's own masked test gaps, with that tower's own test-subset mean as the baseline; identical held-out points (fixed seed) across solo/full/partial; no leakage (test 2022–23 vs train 2018–21).
**Legitimacy:** established technique — partial pooling/multilevel (Gelman & Hill 2007), global forecasting models (Montero-Manso & Hyndman 2021), Mixed-Effects RF (Hajjem 2014); in-domain it is the EC-flux upscaling paradigm (FLUXCOM/Jung 2020; **UpCH4/McNicol 2023 = CH4-specific**; Tramontana 2016; Liang 2019). Site-dummy aids within-site, not unseen-site, prediction → use transferable covariates for generalisation. **Full shared-feature table, per-tower evaluation protocol, and references recorded in `F03_results.md §4–§6`.**
**Outcome (median R²):** Partial ≥ full pooling at **every** tower. Tower 9 rescued (solo ≈ 0 → pooled ≈ +0.29; partial ≈ full). Tower 4 (data-rich) neutral — the dummy **protects** it (partial ≈ solo, avoids the small full-pool dip). **Tower 2 benefits most from the dummy** (partial −0.245 vs full −0.301 short; −0.179 vs −0.230 overall) — it is the most "different" tower (Red→arable, 6.65 ha). Tower 2 still negative: pooling cuts its error ~3–4× but cannot fix the D-15 seasonal-mismatch split. **Recommendation:** use partial pooling + density into forecasting; optionally replace one-hot dummies with continuous tower descriptors (area, soil) to generalise to unseen catchments.

---

### D-31 — 2026-06-16 — Features (R-03 lags re-tested on the rich base, F-04)
**Decision/finding:** Re-added R-03's SWC/TS 1–4 week lags (D-23) to the F-03 density + partial-pooling models and tested ± lags on Towers 2/4/9 (`F04_lags_partial_pooling_RFm.ipynb`).
**Outcome:** The R-03 `RF_lag` advantage **does not transfer to Tower 9** (Δ ≈ −0.00) — once the base already has gap-filled FCO₂ + stocking density + pooling, those features already encode the slow soil-moisture/temperature memory the lags proxied, so the lags are **redundant** for Tower 9. Lags instead help the **weakest-base tower most**: Tower 2 partial-pool Δ **+0.116** (−0.179 → −0.062 overall, best Tower 2 yet, still negative); Tower 4 marginally at medium/long gaps (l 0.028 → 0.052). **Lesson:** feature value is context-dependent — a feature decisive on a weak base can be redundant on a strong one. **Recommendation:** keep SWC/TS lags in the standard set (cheap, help T2/T4-long-gaps, never materially hurt), but recognise pooling+density+FCO₂ — not lags — is the Tower 9 lever. Standard config into forecasting: **partial pooling + stocking density + SWC/TS lags**.

---

### D-32 — 2026-06-16 — Features (pruned management re-tested on the rich base, F-05)
**Finding:** Re-added the pruned tower-specific management features (cut + manure recency, D-28) to the F-04 partial-pool + density + lags config (`F05_management_partial_pooling_RFm.ipynb`; Tower 2 management added to the precompute, Catchment 2 = {NW002}).
**Outcome:** Management gives a **small, non-harmful bump** — overall-median Δ: Tower 2 +0.013, Tower 4 +0.012, Tower 9 +0.005 — largest at the weaker-base towers, negligible at the strong Tower 9. **Same pattern as F-04 lags (D-31): redundant on the rich base** — FCO₂ + density + pooling already encode most ecosystem-state signal that fertiliser/cut events drive. **Recommendation:** keep pruned management in the standard "kitchen-sink" set (partial pool + density + lags + pruned management) — cheap, marginally positive, never hurts — but it is **not a lever**. The decisive levers remain FCO₂, livestock density, and pooling.

---

### D-33 — 2026-06-16 — Methods (Python REddyProc-style pipeline, F-06)
**Decision:** Prompted by the NWFP/REddyProc EC processing report (RPubs 970790): we had never gap-filled the *meteorological drivers* (all models mean-imputed them; SWIN ~52–75% present). Built `src/data/reddyproc_pipeline.py` (Python, no R) → `data/Hourly/reddyproc_processed.csv`: (A) met-driver gap-fill (linear interp ≤2 h + mean-diurnal-course, expanding window) → 100% coverage, diurnal cycle preserved; (B) pragmatic binned-plateau u*-threshold (simplification of Papale 2006 MPT); (C/D) nighttime Lloyd-Taylor partitioning → GPP/Reco. Tested vs mean-imputation on the F-05 config (`F06_reddyproc_pipeline_RFm.ipynb`).
**Outcome — first addition since pooling/density/FCO₂ that genuinely helps:** **met-fill beats mean-imputation** (overall Δ +0.017…+0.076, largest at coverage-poorest Tower 2), and **GPP adds more on top** (Tower 9 metfill +0.287 → **+gpp +0.335 = new project best**; Tower 4 +0.163; Tower 2 −0.045, best yet). **Why it worked where lags/management (D-31/D-32) didn't:** met-fill *fixes the inputs* and GPP is a *new* biophysical driver (productivity/substrate, beats the crude SWIN×TA proxy) — neither is redundant with FCO₂+density+pooling.
**Recommendation:** adopt met-fill + GPP as standard. **New best config: partial pool + density + lags + pruned management + gap-filled met drivers + GPP/Reco.** Carry into forecasting. **Caveats:** pragmatic simplifications of REddyProc's bootstrap u*/partitioning (documented); u*-filtering NOT applied to CH4 R² (ebullition caveat) — reported separately (T4/T9 ≈ 9.3–9.6k nighttime hrs flagged).

---

### D-34 — 2026-06-25 — Evaluation (Tower 2 fix, F-07, TOWER 2 ONLY)
**Finding:** Tower 2's catastrophic results (R-01 RF = −16.9; F-06 = −0.045) were a **broken evaluation, not a data/model failure.** Tower 2 EC CH4 exists only **Oct 2017–Jun 2019** (grassland; analyser relocated to Tower 9 Jul 2019 at the Red-farmlet arable conversion). Catchment 2 had ~10 cattle in 2018 (**FCH4 ≈ 42**) but **zero livestock in early 2019** (**FCH4 ≈ 2**). The **D-15 year split** (train all-2018 / test Jan–May 2019) trains the high-flux/livestock regime and tests the near-zero/no-livestock regime → predicts the wrong level → catastrophic R². Gap-filling is **interpolation**, so the correct evaluation is a **full-period gap-CV** (mask calendar gaps anywhere across 2017–2019, fill from surrounding data).
**Decision/Result (`F07_tower2_evaluation_RFm.ipynb`, F-06 feature set):** Under full-period gap-CV, **RFm solo = +0.394, RFm pooled = +0.519** (median; per-scenario up to +0.66) — Tower 2's **best result in the project, exceeding R² ≈ 0.5** — recovered from −16.9. **MDS stays at −0.49** (livestock-blind: 2018-with-cattle and 2019-without look identical on SW/TA), so **RFm beats MDS by ~1.0 R² unit** — the project's clearest "improvement over MDS" (the supervisor-endorsed framing). Pooling adds +0.13 over solo (D-30 reaffirmed).
**Caveats:** Tower 2's high R² reflects an unusually *discriminable* near-binary livestock-on/off regime (large between-regime variance) — **not directly comparable to Towers 4/9's ~0.3** (continuous grazing). Flaw fixed vs the prototype (calendar gaps not valid-point blocks; lowered pooled overall 0.625→0.519). **Implication:** the year split is inappropriate for single-regime-per-year Tower 2; consider re-evaluating Towers 4/9 under the same full-period gap-CV for consistency before forecasting.

---

### D-35 — 2026-06-25 — Data/Methods (EC-vs-external driver sourcing audit, F-08)
**Finding (sourcing audit):** The NWFP runs **two independent sensor networks**: the EC flux-tower instruments (`greenhouse.csv`, `[Tower N]`) and a separate external network — one central MET station (`[Site]`) plus per-catchment SMS stations (`[Catchment N]`), described in `NWFP_UG_MET_Data.pdf` / `NWFP_UG_SMS_Data.pdf`. **Seven variables overlap.** The project's *de-facto* rule is **"prefer the co-located EC sensor; switch to external only when EC coverage is unusable."** Under that rule, of the overlapping variables actually used in the models, **EC is used for air temperature, shortwave radiation, wind speed, and soil temperature; external is used only for soil moisture** (D-18, tower SWC ~5–10%) and precipitation (no EC twin). RH→VPD (EC-derived); wind direction dropped.
**Inconsistency flagged:** soil *moisture* switched to the external **per-catchment** sensor (D-18) when the EC sensor was sparse, but soil *temperature* switched to a cross-tower **EC proxy** (`TS_1_1_1 [Tower 9]` for all towers, D-16) — even though a **per-catchment external** soil-temperature sensor (`Soil Temperature @ 15cm Depth (oC) [Catchment N]`) exists and correlates **r≈0.98** with the proxy. The per-catchment external twin would be the spatially-faithful, internally-consistent choice.
**Availability/agreement (verified):** per-catchment external twins exist only for soil temperature, soil moisture, precipitation; air temp / solar / wind / RH / WD are **Site-level only** (single station). EC↔external Pearson r: soil temp **0.98**, solar **0.98**, air temp **0.92–0.94** (Tower 2 only **0.28** — its EC air-temp sensor looks faulty), **wind speed 0.17** (not interchangeable; different mast height/location). External coverage is much higher for several (Site solar/air-temp ~99% vs EC 52–78%).
**Decision:** Run **F-08** (`F08_external_sensors_RFm.ipynb`) — a parallel "external-sourced" data layer (`consolidated_hourly_SMS_MET.csv`, `reddyproc_processed_SMS_MET.csv`, built by `src/data/build_sms_met_dataset.py`, **without** touching the existing files) that swaps **all** overlapping drivers to external (soil temp→per-catchment; air temp/solar/wind→Site; wind km/h→m/s ÷3.6; VPD kept EC, no twin). Evaluate **all three towers under full-period gap-CV** (the F-07 methodology, D-34) so EC-baseline and external are compared under one identical harness — this also folds in the D-34 "re-evaluate 4/9 under full-period CV" action. Report per-tower R² (EC vs EXT, solo & pooled, vs MDS) and recommend a sourcing policy for forecasting. Cross-ref D-16, D-18, D-33, D-34.
**Outcome (`results/f08_summary.csv`; harness validated — EC Tower-2 solo = 0.395 = F-07):** (1) **External sourcing is essentially neutral for the RF** — partially-pooled RFm gains a small, consistent **+0.012–0.014 at every tower** (EXT pool: T2 0.490, T4 0.376, T9 0.364 vs EC pool 0.478/0.362/0.350); solo is mixed and tiny (−0.010…+0.002). Same "redundant on the rich base" pattern as F-04/F-05 — and notably the swap **never hurt the pooled model** despite injecting site-level met and a wind series correlating only r≈0.17. (2) **Per-catchment soil-temperature fix vindicated** — the bundled swap (which replaces the Tower-9 proxy with each tower's own-catchment external sensor) is net-positive for the pooled model at all three towers → **adopt it** (removes the D-16/D-18 inconsistency, spatially faithful, costs nothing). (3) **Biggest result = the EC baseline under full-period gap-CV:** re-evaluating 4/9 with interpolation-style CV (vs F-06 year-split) **raises T4 +0.163→+0.362** and T9 +0.335→+0.350, so **all three towers now sit at a consistent ≈0.35–0.49, each beating MDS by ≈0.6–1.0** — the cleanest cross-tower picture in the project. **Recommendation:** adopt external per-catchment soil temperature; treat met sourcing as a wash (prefer external on operational/coverage grounds, ~99% vs 52–78%); carry full-period-gap-CV as the consistent evaluation. External sourcing is a **consistency/robustness** improvement, **not a new accuracy lever**. Full write-up: `F08_results.md`. benchmarks 2855 rows (90 F-08).

---

### D-36 — 2026-06-26 — Forecasting phase scope (05_benchmarking)
**Decision:** Scope the forecasting phase (the project's novel contribution). Full scope: `notebooks/05_benchmarking/forecasting_scope.md`. Key choices (user, this session):
- **Two task tracks:** (A) **hourly** nowcast, horizons {1,6,12,24,48} h; (B) **daily-mean**, horizons {1,3,7,14} d. Direct multi-horizon.
- **Driver-conditional:** future exogenous drivers supplied (future met = weather-forecast/scenario, initially **observed-met perfect-forecast proxy** = optimistic upper bound; livestock/management = planned). Serves the digital shadow (07).
- **Train on gap-filled, evaluate on observed:** train/AR features use the F-06/F-08 gap-filled continuous CH₄ (new precompute `fch4_gapfilled.csv`); metrics scored only on genuinely observed timestamps.
- **Leak-free constraint (critical):** forecasting removes the concurrent-**FCO₂** lever (D-22/D-26) — FCO₂/GPP/Reco are EC fluxes unknown at forecast time → **lagged-only**. Expect **materially lower R²** than gap-filling; lead with **skill vs persistence/seasonal baseline** (improvement-over-baseline = the supervisor framing, analogous to improvement-over-MDS).
- **Inherited:** model roster (D-05), temporal CV (D-04), partial pooling (D-30), F-06/F-08 feature base incl. external per-catchment soil temp (D-35).
**Data reality (verified):** forecasting test targets = **Towers 4 & 9** (2022–2023 valid CH₄ T4 76%/51%, T9 44%/61%). **Tower 2 cannot use the standard split** (no CH₄ post-Jun-2019) — see D-37 for the rolling-origin flip. **Held-out 2024 still empty** (index runs to 2025-01-02 but 2024 FCH₄ = 0% valid all towers) — final held-out benchmark blocked until 2024 EC fluxes are downloaded. Cross-ref D-04, D-05, D-15, D-30, D-35.

---

### D-37 — 2026-06-26 — Forecasting execution: data source, Tower-2 test mechanism, CV (FC-01)
**Decisions (user, this session):**
- **Primary data = External SMS/MET** (`consolidated_hourly_SMS_MET.csv` + `reddyproc_processed_SMS_MET.csv`, D-35): per-catchment external soil temp + Site-level met (matches how a real weather-forecast/scenario supplies drivers — site-level, not per-tower) + ~99% coverage. F-08 showed external sourcing is accuracy-neutral, so this is chosen on consistency/robustness grounds.
- **Tower 2 flipped to a TEST target** via **expanding-window rolling-origin within 2017–2019** (2 folds: train ≤2018-06 → test 2018-H2; train ≤2018-12 → test 2019-H1; donor = Tower 4, the only tower with pre-2020 CH₄). Test spans both regimes (2018 cattle / 2019 none). **T2 R² is degenerate** on its near-zero 2019 variance → report **RMSE/MAE/skill-vs-persistence** as primary.
- **CV / leakage rules:** Towers 4/9 train(target-time ≤ 2021)/test 2022–2023 (single fit, sliding origin); horizon buffer (no target leakage); **train on gap-filled, evaluate on OBSERVED only**; **leak-free** — FCO₂/GPP/Reco lagged-only, never at t+h; future met = **perfect-forecast proxy** (observed met) = optimistic upper bound.
**Precompute:** `src/models/gapfill_rfm.py` (shared F-08 gap-filler), `build_fch4_gapfilled.py` → `fch4_gapfilled.csv` (continuous CH₄), `build_forecasting_matrix.py` → `forecast_features.csv` (ar_ origin + fx_ future-exog, 210k rows).
**FC-01 result (`B01_baselines_and_ML.ipynb`, RF/XGB + persistence/climatology, both tracks):** **ML beats persistence at almost every horizon** — RF hourly skill +0.08…+0.25 (RMSE reduction), daily up to **+0.37 at 14 d** (T4) / +0.32 (T9). **Forecasting R² low but positive** (hourly 0.02–0.15; daily 0.15–0.30) — *skill-vs-baseline is the metric*, not absolute R² (open system, FCO₂ lever gone). **Honest caveats:** 1-day daily persistence is unbeatable (RF loses at T4 d1); RF's edge over the *climatology* baseline is modest (+0.02–0.15); **RF > untuned XGB**; T2 forecastable under rolling-origin (+0.08…+0.24 skill, R² caveated). Full write-up: `B01_results.md`. benchmarks +108 FC-01 rows (`track`/`horizon`/`skill_*` columns added). Cross-ref D-04, D-05, D-30, D-36.

---

### D-38 — 2026-06-26 — Forecasting Stage 2: deep learning (FC-02)
**Decision:** Hand-rolled, **pure-PyTorch** DL forecasters (`src/models/forecasting_dl.py`) — no darts/pytorch-forecasting — in a **native seq2seq** form (encoder over CH₄+drivers+lagged-flux history; decoder over **known-future drivers only** → multi-horizon), partial-pooled across towers, evaluated on the **same observed points** as FC-01. Roster: **DLinear** (Zeng-2023 decomposition-linear), **LSTM** seq2seq, **LSTM+VSN** (variable-selection gate → native importance, used in I-01). **GPU enabled:** upgraded torch **2.6.0+cu124 → 2.11.0+cu128** (`pip --user`, isolated; RTX 5070/sm_120 verified; sklearn/xgboost intact; torchvision pin warning benign as it's unused; rollback `torch==2.6.0`). Origin **stride=6** (hourly) to bound memory.
**Result (`B02_results.md`, `fc02_summary.csv`, 81 FC-02 rows):** **model complexity does NOT pay off — the Zeng-2023 finding, confirmed for open-system grassland CH₄.** **Hourly: RF/XGB win** (skill +0.11–0.25, positive R²); the DL models trail with **negative R²** at T4/T9 (beat persistence but not the test mean). **Daily: DLinear (one linear layer) is competitive with / beats RF** (T9 d3 0.235 vs 0.196); LSTM/VSN worst on the short daily series. **Exception: Tower 2 hourly — LSTM beats RF** (skill 0.375 @ h1 vs 0.080) via its strong livestock-on/off autoregressive regime. **Recommendation:** production forecasters = **RF (hourly) + DLinear (daily)**; keep LSTM for Tower 2 / as a complexity baseline. The dissertation value is the **honest "simpler wins" benchmark**, not making DL win. Cross-ref D-05, D-36, D-37.

---

### D-39 — 2026-06-26 — Forecasting feature importance (I-01)
**Decision:** Cross-model importance harness (`I01_feature_importance.ipynb`, `06_interpretability_uq/`) on Track-A hourly, Tower-4 main split: **permutation importance** (grouped by feature family, per horizon) for RF + LSTM — the model-agnostic comparator; **SHAP** TreeExplainer on RF (reuse F-01 pattern); **VSN-native** gate weights from LSTM+VSN. Not a benchmark metric → separate CSVs/figures, not `benchmarks.csv`.
**Findings (`I01_results.md`):** (1) **Importance shifts with horizon** — RF: recent **CH₄ history dominates at h=1** (ΔRMSE 12.8) and **decays**, while **planned livestock+management grows to dominate at h=48** (8.8), met/seasonality also rising. "What you can plan matters more the further ahead you forecast." (2) **SHAP: `fx_lsu_dens` (livestock density) is the #1 forecasting feature** (mean|SHAP| 31.8, ~3.5× the next) — echoes F-01's gap-filling result; **the project's livestock-is-dominant thesis carries into forecasting.** (3) **RF vs LSTM use features very differently** — the LSTM's CH₄-memory importance **collapses to ≈0 by h=24** and it over-relies on future met (ΔRMSE ~40), whereas RF blends memory + planned drivers — a plausible mechanism for the trees' edge (D-38). (4) VSN corroborates CH₄+met+FCO₂ (more diffuse). Permutation and SHAP agree on livestock + CH₄-memory. Cross-ref D-27 (livestock), F-01 SHAP, D-38.

---

### D-40 — 2026-06-26 — Forecasting Stage 3: uncertainty quantification (FC-03)
**Decision:** Calibrated 90% prediction intervals for the production forecasters via three paradigms (`U01_uncertainty.ipynb`, `06_interpretability_uq/`): **split-conformal** (hand-rolled, model-agnostic; RF hourly + DLinear daily; calibrated on 2021, **Mondrian per horizon**), **quantile XGBoost** (`reg:quantileerror` α=0.05/0.5/0.95), **LSTM-pinball** (new `LSTMQuantile` + `pinball_loss` in `forecasting_dl.py`). Metrics: **PICP@90% / MPIW / pinball**, Towers 4/9, per horizon; UQ metrics → `fc03_uq_summary.csv` + `picp/mpiw/pinball` columns in benchmarks (54 FC-03 rows).
**Result (`U01_results.md`):** **calibrated but wide — the spikes are irreducibly uncertain.** **Conformal most reliable** (mean PICP 0.87–0.88 ≈ nominal, by construction) but **widest** (RF MPIW ~240). **Quantile-XGB = best calibration–sharpness trade-off** (PICP ~0.84, sharpest MPIW ~153, best pinball 13.0) → decision-useful default. **LSTM-pinball under-covers** (0.62–0.82) → drop. Intervals are **wide (~150–260 nmol)** and **even they miss the biggest spikes** (fan chart: 600–1520 nmol events burst the band) — the slight under-coverage is the spike tail; coverage degrades with horizon for quantile models, conformal stays stable. **Recommendation:** report **conformal** bands for guaranteed coverage + **quantile-XGB** for sharper bands; for the digital shadow (07) ship forecasts with conformal 90% bands (width = spike-risk signal). The UQ **quantifies that the uncertainty lives in the spike tail → motivates spike-aware modelling next.** Cross-ref D-36/37/38; conformal = Vovk/Lei split-conformal.

---

### D-41 — 2026-06-30 — Enriched-feature forecasting reruns + Round-1 HPO (B-03 / B-04)
**Decision:** Productionise the `NWFP_T9_Dataset_Structure.md` feature engineering across **all towers** and re-run the two forecasting benchmarks: **B-03** (= FC-01 trees) and **B-04** (= FC-02 DL). New builder `src/features/build_forecasting_matrix_v2.py` emits `forecast_features_v2.csv` (hourly, +7 future-exog `fx_`: wind-direction sin/cos, is_daytime, 3-sensor SHF mean, is_growing/is_winter, days_since_grazing) and `forecast_daily_v2.csv` (daily, guide aggregations: TA min/max, precip-sum, **external** soil daily lags{7,14,21,28}/rolling{7,14} per D-35, circular daily WD, days_since_grazing, calendar) + daily AR (`ar_ch4_dlag*`+`drm7`+**lagged-only** `ar_fc_dlag1`). B-03/B-04 are **additive clones** of B01/B02 (CV/eval/baselines unchanged); only shared-code change = backward-compatible `forecasting_dl.load_matrix(path=None)`. **Acceptance (user) = skill + best-achievable R²**, R²~0.5 a stretch not a gate; **bounded iteration = features (Round 0) + one HPO round (Round 1)** only.
**Result (`b03_b04_results.md`, `b03_summary.csv`, `b04_summary.csv`; 108 B03 + 81 B04 rows):** **enriched features lift the TREE forecasters; the DL is unmoved.** B-03 daily best R²: **T4 0.263→0.362, T9 0.304→0.388** (mean daily ΔR² RF +0.118 / XGB +0.166; Round-0 features ~+0.08, Round-1 daily HPO — RF leaf10/max-features0.5, XGB depth2/lr0.02/400 — a further ~+0.05). Daily skill-vs-persistence grows with horizon (T4 −0.07→+0.42, T9 +0.11→+0.37). Hourly barely moves (ΔR² +0.02/+0.04, best ≈0.15). **B-04 DL flat** (DLinear best daily T4 0.333→0.337, T9 0.326→0.292) — the seq2seq 28-day lookback already encodes the soil/TA history the new daily lags add → redundant for DL (re-confirms D-38 "simpler wins"). **Verdict vs the 0.5 target:** best daily forecasting R² now **≈0.36–0.39**, a real **+0.08–0.10 over FC-01** but **short of 0.5** (expected — leak-free forecasting is harder than the gap-filling ceiling 0.36–0.49; Zhu-2023a floor <0.1). Reaching ≥0.4–0.5 would need deferred levers (target transform, coarser/cumulative eval, spike-aware hurdle). **Production = enriched trees on the daily track** (RF/XGB on `forecast_daily_v2.csv`) + RF hourly; DLinear unchanged. Cross-ref D-35 (external soil), D-36/37 (FC-01), D-38 (DL), D-39 (livestock #1).

---

### D-43 — 2026-06-30 — spike-aware two-stage hurdle model (B-06) — NEGATIVE for daily; mixed for hourly
**Decision:** Test a **two-stage occurrence × magnitude hurdle architecture** as the structural alternative to the B-05 target transform (D-42). Per tower, freeze a **q90 spike threshold** on `y_observed` training years (2018–2021) — T4 hourly/daily q90 ≈ 79/71 nmol, T9 ≈ 110/116 nmol. Fit (a) a **spike classifier** (`RandomForestClassifier` / `XGBClassifier`, B-03 hyperparameters, `class_weight="balanced"` / `scale_pos_weight`) and (b) **two magnitude regressors** — one trained only on non-spike rows, one only on spike rows — then combine via **soft probability blend**: `P(spike)·spike_model(x) + (1−P(spike))·base_model(x)` (correct decomposition of `E[y|x]`, chosen by user over hard classify-then-route). Same partial pooling (T2+T4+T9 + dummies, D-30), same CV, same B-03 enriched features, **no new HPO** (bounded-iteration norm, D-41). Both hourly (Track A) and daily (Track B) tracks. Threshold at q90, not q95 (daily Tower-9 q95 gives only 23 training spike rows — too thin). New metric columns in benchmarks: `precision`/`recall`/`f1` (mirroring how D-40 added `picp`/`mpiw`/`pinball`).
**Result (`B06_spike_hurdle.ipynb`, `b06_summary.csv`, 162 B06 rows; `b06_results.md`):** **NEGATIVE for the daily (production) track; mixed-to-positive for hourly Tower 4.** Daily best R² (T4/T9): Hurdle-RF 0.253/0.130 vs B-03-RF 0.357/0.388; Hurdle-XGB 0.179/−0.280 vs B-03-XGB 0.362/0.324 — large degradation at every horizon. Hourly Tower 4: Hurdle-RF **beats** plain RF at all 5 horizons (mean ΔR² +0.041), Hurdle-XGB similarly (+0.024); hourly Tower 9: RF nearly flat (+0.009), XGB negative (−0.043). **Mechanism:** the classifier's **precision is low** (daily 0.25–0.42; hourly 0.30–0.56) while recall is high (0.58–0.93) — many false-positive "spike" predictions blend the noisy spike-only regressor into non-spike test points, inflating non-spike RMSE by more than the spike RMSE is reduced. Conditional RMSE: daily T9 XGB example — spike RMSE −47 nmol (win) vs non-spike RMSE +41 nmol (loss); since ~90% of rows are non-spike the net aggregate R² collapses. The daily spike-only regressor is also fit on ~100 pooled training rows — borderline for stability. **Conclusion:** the low-precision classifier is the bottleneck (more predictive spike features needed, not more tuning); a hard-classify-then-route variant would worsen non-spike collateral damage, not fix it. B-03 remains the production forecaster. Both the target-transform (D-42) and architecture-split (D-43) attacks on the spike problem have now been tried and documented negative for the daily track. Hourly Tower 4's small gain (+0.02–0.04 R²) does not justify a second production path. Cross-ref D-41 (HPO), D-42 (arcsinh negative), D-40 (spike tail = irreducible UQ uncertainty).

---

### D-42 — 2026-06-30 — arcsinh target transform (B-05) — NEGATIVE result
**Decision:** Test whether an **`arcsinh` target transform** (chosen over `log` because CH₄ flux is signed, ≈ −1559…+6161; ~linear near 0, ~log in tails) lifts daily forecasting R² by compressing the episodic spikes that dominate squared error. Same enriched-tree pipeline as B-03 (`B05_asinh_ML.ipynb`), trees fit on `arcsinh(y)`, back-transformed with **Duan smearing** (sinh over sampled training residuals) to remove the Jensen bias; metrics in original nmol space; baselines untransformed.
**Result (`b05_summary.csv`, 108 B05 rows):** **does NOT help — slightly worse than identity.** Naive `sinh` back-transform was badly biased (daily R² collapsed to 0.13–0.21, MBE ≈ −20 nmol). With Duan smearing the bias is fixed and B-05 recovers to **daily best R² T4 0.337 / T9 0.347 — still below B-03's 0.362 / 0.388.** A scale sweep `arcsinh(y/c)` only converges *up* to identity as `c→∞` (the transform weakens). **Mechanism:** R² is scored in original units where the spikes dominate the variance, so compressing them in training trades spike accuracy (which R² rewards most) for bulk accuracy — a net loss. **Conclusion:** target transforms are a dead end for original-space R² on this signal; **B-03 remains the production config.** Logged as a documented negative result (kept in benchmarks, clearly flagged). The spike problem needs a *structural* attack (two-stage hurdle), not a monotonic squashing. Cross-ref D-41, D-40 (spikes are the irreducible-uncertainty tail).

---

### D-44 — 2026-06-30 — spike-classifier diagnostics + recency features + early-warning threshold analysis (B-07)
**Decision:** Follow-up to B-06 (D-43), whose mechanism diagnosis pointed to the spike classifier's **low precision** (not low recall) as the bottleneck. Three-part bounded investigation (`B07_spike_diagnostics.ipynb`): (1) **diagnose** the classifier's false positives/negatives at representative horizon/tower combos (daily h=1/14, hourly h=1/24, Towers 4/9, both algos) against context features (precip, days-since-grazing, growing-season flag); (2) **add leak-free recency/clustering features** — `ar_days_since_spike`, `ar_spike_count_<w>`, `ar_rolling_max_<w>` (daily w=7/28, hourly w=24/168) computed causally from each tower's gap-filled CH₄ series against the same frozen q90 thresholds as B-06 — and retest the **full B-06 harness** (classifier + 2 magnitude regressors + soft blend, plus plain RF/XGB) with them added, **built and tested regardless of the diagnostic outcome** (user decision, one bounded empirical test, no new HPO — D-41 norm); (3) **precision-recall threshold analysis** on the daily classifier, reframing it as a standalone "elevated-emission-risk" early-warning signal at a recall≥0.8 operating point, independent of the regression R².
**Result (`b07_results.md`, `b07_summary.csv`, 108 B07 regression rows + 8 early-warning operating points; 55,928 diagnostic rows):** **Diagnostic — false positives are context-indistinguishable from true positives.** Daily-track FP rows have nearly identical mean precip (0.59 vs TP's 0.60), days-since-grazing (33.9 vs 34.0), and growing-season flag (0.92 vs 0.96) to true positives — the classifier has correctly learned "growing season + recent grazing → elevated risk" (FP rate 0% Nov–Mar → 21–38% Jun–Aug), but that signal covers roughly twice as many quiet days as spike days, so it cannot discriminate further with available features. **Recency features (verified leak-free) — marginal, inconsistent, do not flip the B-06 verdict.** Plain RF/XGB daily R² moves by <0.02 in either direction depending on tower/horizon (no consistent sign). Hurdle+recency daily classifier precision rises modestly (0.25–0.42 → 0.26–0.44) but Hurdle R² remains mixed and **stays below B-03 at every daily tower/horizon** — short-horizon Hurdle-RF nudges up slightly (T4 h=1 +0.005, T9 h=1 +0.015), longer horizons and Hurdle-XGB move both directions with no pattern. **Early-warning analysis — the one positive, narrow finding:** at a recall≥0.8 operating point, RF/XGB catch ~81% of true elevated-emission days at precision 0.28–0.43 (e.g. T4 RF h=1: recall 0.806, precision 0.425) — a usable farm-management screening trade-off, decoupled from the regression benchmark. **Conclusion:** B-05 (transform), B-06 (hurdle architecture), and B-07 (diagnostics + recency features) have now all been tried and documented negative/marginal for daily forecasting R² — the spike events appear driven by information not present in the current `ar_`/`fx_` feature set (plausibly sub-daily turbulence/wind conditions, not captured by daily aggregates). **B-03 remains the production forecaster.** The early-warning framing is retained as a standalone decision-support artefact for potential Phase 07 use. Cross-ref D-43 (hurdle mechanism), D-42 (transform negative), D-41 (HPO/feature norm).

---

### D-45 — 2026-07-01 — filling the D-05 model-roster gaps: SARIMAX (B-03a) and full TFT (B-03b)
**Decision:** The original model roster (D-05) was persistence/seasonal-mean → **ARIMA** → RF/XGBoost → LSTM/**TFT** → SARIMAX. ARIMA/SARIMAX was never implemented in any notebook, and a full TFT was explicitly de-scoped at FC-02 time in favour of `LSTM_VSN` (`forecasting_scope.md`: "Full TFT/N-HiTS de-scoped → VSN supplies native importance", D-38). User requested both be added as small experiments on B-03's data/CV/horizons (`forecast_features_v2.csv`/`forecast_daily_v2.csv`, Towers 4/9 main split test 2022–2023, Tower 2 expanding folds), named **B-03a**/**B-03b**. **B-03a:** per-tower solo SARIMAX (no panel-pooling equivalent for D-30) with a small SHAP-informed exogenous set (I-01's top non-AR drivers: `fx_lsu_dens`, wind speed, VPD, USTAR, PPFD + seasonality proxies), bounded AIC order grid (`d=1` fixed, `p∈{1,2}`, `q∈{0,1}` → **(2,1,1)** won everywhere), walk-forward evaluated via `statsmodels`' `append(refit=False)` + `get_forecast(steps=H)` (ARIMA's natural multi-step usage, vs. the rest of the project's one-model-per-horizon design). **B-03b:** canonical TFT (Lim et al. 2021) hand-rolled in pure PyTorch (`src/models/forecasting_dl.py`, new `TFT`/`GRN`/`VSN`/`InterpretableMultiHeadAttention` classes) — Variable Selection Networks, static covariate encoders (4 context vectors), LSTM encoder-decoder with static-initialised state, gated locality enhancement, static enrichment, interpretable (shared-value) multi-head self-attention with causal masking, gated position-wise feed-forward — every architectural component present, `d_model=32`/`n_heads=4`/30 epochs (modest sizing, bounded-iteration norm D-41, no new HPO). Reuses `run_track`/`build_windows`/`_eval_rows` unchanged.
**Result (`b03a_b03b_results.md`, `b03a_summary.csv`/`b03b_summary.csv`, 27+45 rows):** **SARIMAX negative beyond h=1; TFT negative until fixed — B-03 remains unambiguously production either way.** Daily R² (towers 4/9, h=1→14): B-03 RF **0.372→0.306**; B-03a SARIMAX 0.326→**-0.177** (competitive only at h=1, collapses by h=7; MASE 1.06–1.13, worse than persistence at every horizon); B-03b TFT (original) **-0.967→-0.730** (negative at every single horizon/tower/track, MASE 1.03–1.79 — the single worst model result in the entire forecasting phase). **TFT result was independently verified, not taken at face value** — a manual training run confirmed clean loss convergence (standardized MSE 0.85→0.09 over 30 epochs, ~91%-equivalent training fit) and sanely-scaled test predictions (no NaNs, no degenerate output, mean/std close to actual) — ruling out an implementation bug. The actual mechanism is **overfitting**: test correlation is weakly positive (r=0.27) but a handful of large overconfident spike-mispredictions (e.g. predicted 355.8 vs actual 31.4) drag the squared-error-based R² deeply negative.
**Fix (user-requested follow-up, same session):** added optional `weight_decay`/`val_data`/`patience` params to `train_model()` (backward-compatible, off by default — AdamW replaces Adam but is equivalent when `weight_decay=0`; existing DLinear/LSTM/LSTM_VSN calls in B02/B04 unaffected). Retrained TFT on Towers 4/9 with `weight_decay=1e-3` + early stopping on a **held-out 2021 validation year** (train 2018–2020, mirrors FC-03/U-01's existing precedent of reserving 2021 for calibration), patience=5. **First attempt hit a real bug**: the validation-loss check ran as one *unbatched* forward pass over the full validation set — with TFT's O(T²) attention over L+H=216 timesteps, an unbatched multi-thousand-window batch allocates a multi-GB attention-score tensor, which hit a 1800s nbconvert timeout (confirmed via `nvidia-smi`: 100% GPU util, near-exhausted 12GB memory, no crash — genuinely too slow, not hung). **Fixed by batching the validation pass** the same way `predict()` already batches inference (16s/epoch after the fix, vs. exceeding a 30-minute budget before). **Result: the fix worked.** Daily R² went from **-0.836→-1.078 (T4) / -0.623→-0.856 (T9), negative everywhere, to +0.247→+0.255 (T4) / +0.097→+0.106 (T9), positive everywhere** — MASE dropped from 1.03–2.01 (worse than persistence throughout) to 0.65–1.23 (beats persistence from h=3 onward at T4, h=7 onward at T9). Hourly moved the same direction more modestly (R² -0.17…-0.02 → +0.004…+0.039). **TFT-Reg still sits well below B-03's trees at every horizon** (0.10–0.26 vs 0.27–0.39) — the fix converts TFT from "single worst model in the project" to "genuinely reasonable, non-competitive forecaster," not into a production contender.
**Conclusion:** this sharpens D-38's "simpler wins" pattern rather than just restating it — the more components a model has, the more it needs regularisation/data/tuning to avoid overfitting under a bounded compute budget, and applying that regularisation reverses the sign of the result entirely (a useful methodological finding in itself: "complex model underperforms" should be checked for overfitting before being read as an architecture verdict). The original D-05 model roster (persistence/seasonal-mean, ARIMA, RF/XGBoost, LSTM/TFT, SARIMAX) is now **fully populated** with documented results; B-03 remains production; no further algorithm-search experiments are warranted on this feature set. The regularisation recipe (`weight_decay`+`val_data`+`patience` in `train_model()`) is retained in the shared module for future use. Cross-ref D-05 (original roster), D-38 (TFT de-scoping/LSTM-VSN substitute), D-41 (bounded-iteration norm), D-22 (feature realism > algorithm choice).

---

### D-46 — 2026-07-01 — Feature-addition scoping (fertiliser/AR/seasonality), other-catchment data rejected, long-range (2030) scenario feasibility + candidate climate dataset
**Feature-addition considerations (no code changes, discussion only):**
- **Fertiliser recency** — `build_management_features.py` already computes a `fertN` channel (recency + rate-weighted magnitude, τ=14d), but it is **not** in `forecast_features_v2.csv`/`forecast_daily_v2.csv` — it was part of F-01's original 12-column set that **overfit** (D-28: "collapse at Tower 9") and was pruned to cut+manure-only in F-02 (D-29/D-32). Re-adding it would be revisiting a specifically-tested-and-reverted decision, not adding something new.
- **Weekly AR mean** — already present (`ar_ch4_drm7`, 7-day rolling mean, daily track). A longer window (2-4wk) would be cheap but low-expected-payoff, per the established lags/management pattern (D-31/D-32: helps weak-base towers marginally, redundant on the rich base).
- **Explicit season/week-of-year calendar features** (4-season flags, `week_of_year`) — **not recommended**: `fx_DOY_sin`/`fx_DOY_cos` (daily) already encode position-in-year continuously: a tree model (RF/XGB, the actual production algorithm) can already split this into any granularity it needs, so discretising into seasons or week-bins can only *lose* resolution relative to what's already there, not add information. (A `day_of_week` feature, targeting weekly *operational* rhythms rather than annual seasonality, would be a genuinely different and currently-absent signal — not evaluated here.)
**Other-catchment data (beyond Towers 2/4/9) — rejected as a lever.** Considered and dismissed: (1) no FCH4 target exists at any other NWFP catchment, so there is no supervised signal to pool toward (partial pooling, D-30, requires a target at every pooled unit); (2) EC flux is footprint-local (D-18's spatial-alignment rule — never mix catchment-specific inputs across towers — applies with equal force to importing *other* catchments' local features); (3) the one genuinely farm-wide signal (regional weather) is already captured via the Site-level external MET network (D-35). Spatial upscaling to uninstrumented catchments (FLUXCOM-style) is a legitimate but *different* research question from this project's temporal forecasting task, and would need its own validation data this project doesn't have either.
**Long-range (~2030) scenario feasibility — scoping note, not yet executed.** Established that a 2030 projection is categorically a **scenario projection** (Phase 07), not a forecast (B-03's domain) — no real weather/livestock plan reaches that far. Requirements identified: (1) persist a frozen B-03 model artifact (currently fit-and-discard per notebook run — no saved model exists yet); (2) make the feature-build pipeline accept synthetic/scenario driver overrides, not just historical replay; (3) a livestock/management assumption (necessarily hypothetical this far out); (4) a climate-scenario driver source (see below); (5) a strategy for AR/CH4-history features at a horizon with no real recent observations (climatological seeding or self-referential rollout — the latter risks compounding error across a multi-year projection, unlike B-03's current 1-14-day design where errors don't get to accumulate); (6) a driver-range/extrapolation check (RF/XGB do not extrapolate beyond training-leaf values — verify 2030 scenario driver values against the 2018-2021 training range) and a partial-dependence sanity check on key drivers before trusting scenario output; (7) B-08 (driver-realism sensitivity, queued) becomes directly load-bearing here, not just a side experiment, since it quantifies model sensitivity to synthetic/degraded drivers; (8) explicit framing in any write-up that 2030 output is conditional-on-scenario, not a prediction, given the training window ends in 2021 (a 9+ year extrapolation gap, larger than anything else in this project).
**Candidate climate-scenario dataset identified**: Semenov, Senapati, Coleman & Collins (2025), "A dataset of large ensemble of CMIP6-based transient climate scenarios for impact assessment in Great Britain," *Data in Brief*, DOI 10.1016/j.dib.2025.111695 (Rothamsted Research-authored; Zenodo DOI 10.5281/zenodo.14040993, CC BY). **Contents:** daily Tmin/Tmax/rainfall/solar radiation, 26 GB sites, 2020-2090, 5 CMIP6 GCMs × 2 SSPs (2-4.5/5-8.5), 100 realizations/scenario via LARS-WG 8.0 stochastic downscaling (preserves variability/extremes, not just a smoothed mean trend). **Directly answers the climate-driver half of requirement (4) above.** Caveats before use: (a) unconfirmed whether North Wyke/Devon is among the 26 sites, or a nearest-site proxy would be needed; (b) covers only 4 of B-03's ~11 daily driver variables (`fx_TA_min/max` and `fx_PRECIP_sum` map directly, `fx_SWIN_mean`≈solar radiation; **missing** wind speed, VPD, USTAR, soil temp/moisture, SHF, wind direction — these would need a separate treatment for a 2030 run); (c) the 2020-2021 overlap with this project's existing training/test window is a free validation opportunity (compare simulated-baseline vs. real observed NWFP weather before trusting the out-of-sample 2030 realizations); (d) near-term (2030) climate-shift signal is likely modest relative to natural year-to-year variability already sampled in 2018-2023, which partially — not fully — eases the extrapolation-range concern in (6) above relative to a more distant target year. Does not address the livestock/management assumption (3) or the AR-history/compounding-error issue (5). Cross-ref D-30 (pooling), D-18 (spatial alignment), D-35 (external MET), D-41 (bounded iteration), the queued B-08 plan.

---

### D-48 — 2026-07-01 — root-cause fix: unfiltered USTAR/VPD outliers corrupting met-driver gap-filling, all towers (F-09)
**Trigger:** user-observed anomaly (a visually obvious spike in Tower 2's gap-filled FCH4 during a period with zero real observations, also noted at Towers 4/9). **Decision:** investigate before fixing (explicitly requested over documenting-only). **Root cause, confirmed via SHAP TreeExplainer on the pooled RFm gap-filler**: for a representative Tower-2 July-2019 row (100% gap-filled, no real anchor), **`USTAR_0_0_1` contributes +244 nmol and `WS_0_0_1` +107 nmol of a total +380 nmol lift** — over 92% of the spike from two features. Tower 2's raw `USTAR_0_0_1` has **zero real observations in July 2019** and was **never quality/plausibility-filtered anywhere in the pipeline** (unlike FCH4 `[-500,3000]` D-13, FCO2 `[-100,100]` D-25) — it contains 722 historical readings above 5 m/s, max **1039.9 m/s** (physically impossible), giving a ~10x mean(2.450)/median(0.233) gap diagnostic of severe contamination. `reddyproc_pipeline.py`'s `mdc_gapfill()` last-resort fallback uses the **arithmetic mean**, which only fires for extended blackouts with no nearby real data — exactly where a contaminated mean does the most damage; confirmed the actual `USTAR_0_0_1 [Tower 2]__f` feature is flat at 2.468 for July 2019, identical to the series-wide mean, i.e. a non-seasonal fallback constant, not a real signal. A broader audit (all met columns, all 3 towers) found the same contamination pattern in **VPD** too (1.8-12.6% of readings > 15 kPa, worst at Tower 4, max 36-41 kPa — implausible given VPD is bounded by saturation vapour pressure); PPFD/WS/SHF were checked and are clean.
**Fix:** added `plausibility_filter()` to `reddyproc_pipeline.py` (USTAR bounded `[0,3]` m/s, VPD `[0,15]` kPa, applied before gap-filling; reused automatically by `build_sms_met_dataset.py` since it imports the same `mdc_gapfill`), and changed `mdc_gapfill()`'s last-resort fallback from mean to **median** (robust to outliers by construction; the rolling-window MDC step itself was left as mean, matching literature convention, since it's far less vulnerable given it always has a local mostly-real window to average over).
**Validation:** regenerated `reddyproc_processed.csv`/`_SMS_MET.csv`/`consolidated_hourly_SMS_MET.csv`/`fch4_gapfilled.csv`/`forecast_features.csv`/`forecast_features_v2.csv`/`forecast_daily_v2.csv`. **Tower 2's spike is fully resolved** (325-413 nmol → 2.9-27.7 nmol, back in line with its real Jan-May-2019 baseline of 2-20). **Tower 4's 2024 blackout resolved** (mean 227.6 → 26.6, back within its 2017-2023 real range of 16-80). **Tower 9 improved but less completely** (10-207 → 10-160, smaller effect, not further investigated).
**Scope discovery — not a narrow boundary effect.** Comparing AR-feature means across the **entire 2018-2023 evaluated window** (not just the isolated blackout periods) shows the impact is systematic: Tower 2 AR-mean 71.9-72.2→**20.2**, Tower 4 54.6→**31.1**, Tower 9 79.2-79.4→**36.6-36.7**. **Corrected AR-feature means now sit close to each tower's real observed CH4 mean** (T9: 36.6 vs 36.0 real; T4: 31.1 vs 29.2-29.8 real) — exactly what a correctly-functioning autoregressive feature should look like; before the fix every tower's AR feature was inflated 1.5-2x above its own real distribution — independent corroborating evidence the fix is correct, beyond resolving the originally-observed spike. **Consequence: AR/CH4-history (one of the most important predictors in the forecasting phase, I-01) was systematically biased throughout the whole 2018-2023 window for all three towers, for every one of B01 through B08 — the current `benchmarks.csv` numbers for the entire forecasting phase are stale.** Most likely direction on re-run is neutral-to-positive (a less-biased predictor should be more, not less, informative), but this is a plausibility argument, not a measured result.
**Decision on scope (user-directed):** log the fix now; **stage the full B01-B08 re-run as a separate, later task** rather than executing it in this session. B-08 (driver-realism, queued) should wait for that re-run rather than building on stale AR features. Cross-ref D-13/D-25 (existing plausibility-filter precedent), D-30 (pooling — affects all pooled towers identically), D-33/D-35 (the met-gapfilling/external-sourcing pipeline this patches), D-36 (forecasting precompute chain), I-01 (AR/CH4-history importance).

---

### D-49 — 2026-07-02 — F-09 fix re-run: B-03/B-03a/B-03b on corrected data + F-09a gap-filling re-verification
**Decision:** Per D-48's staged plan, re-ran the highest-priority subset (B-03, B-03a, B-03b — user-scoped, not the full B01-B08 sweep yet) against the corrected post-fix data files, plus a lightweight standalone re-verification of the gap-filling task itself (not just its forecasting downstream). **B-03 (RF/XGB, production):** small, consistent gains — T4 RF daily h=1: 0.357→0.365, h=14: 0.270→0.280 (before D-48 fix → after); T9 RF h=14: 0.342→0.359. No qualitative change, production model unaffected in ranking. **B-03a (SARIMAX): major, qualitative reversal.** Daily R² (T4/T9 mean, h=1→14): before **0.326→-0.177** (collapsed by h=7, worse than persistence past h=1) → after **0.416→0.284**, i.e. h=1 *also* improved (0.326→0.416) and h=3/7/14 flipped from strongly negative to positive (0.31/0.25/0.24 at T4, 0.31/0.30/0.33 at T9); MASE flips from >1 (worse than persistence) to <1 from h=3 onward at both towers. **This reverses the original D-45 conclusion that "SARIMAX collapses beyond h=1"** — under corrected AR/exogenous features the same `(2,1,1)` order stays competitive at every horizon tried. **B-03b (TFT/TFT-Reg): mixed, no clean directional story** (unlike SARIMAX) — daily R² moved in *both* directions depending on tower/track: TFT-Reg T4 daily declined slightly (0.247→0.255 before → 0.191→0.199 after, still solidly positive) while T4 hourly (track A) got *worse* (0.016→0.008 before → -0.076→-0.185 after, flipping positive-to-negative); T9 daily improved (0.097→0.106 → 0.189→0.217) while T9 hourly stayed marginal (0.012→0.039 → 0.006→0.057). Original (unregularised) TFT also mixed: T4 daily less negative (-1.078→-0.836 → -0.658→-0.348, an improvement) but T9 daily *more* negative (-0.856→-0.623 → -1.214→-1.036, a regression) and T2 both tracks got worse. **No overfitting-adjacent explanation identified for the sign flips — most plausibly reflects noise sensitivity of the TFT's many components to the altered feature distributions, not a directional bug**; not further investigated given TFT/TFT-Reg were never production candidates (D-45).
**F-09a (`results/f09a_summary.csv`, standalone script, NOT part of `F08_external_sensors_RFm.ipynb`):** the original F-08 notebook could not be re-run for direct before/after comparison — it timed out twice (1800s, then 3600s) even before this fix, given its full EC×EXT × solo×pool × 3-tower × 5-scenario × 5-rep grid (300+ individual 500-tree RF fits). **Per explicit user instruction, the original notebook/`f08_summary.csv`/its `benchmarks.csv` rows were left untouched** (confirmed via `git diff` — no changes) rather than risk overwriting the historical pre-fix D-35 baseline; a separate, reduced-scope script reusing F-08's exact gap-CV methodology verbatim (EXT variant only, `RFm_pool` only — the recommended config — 2 reps instead of 5) was built and run instead, completing in ~20 min. **Result: real gap-filling accuracy (recovering masked FCH4 observations under full-period gap-CV) genuinely improved, not just shifted downstream AR-feature statistics** — EXT/RFm_pool overall median R² (across all 5 gap-scenarios): T2 0.490→**0.574**, T4 0.376→**0.402**, T9 0.364→**0.418**, all three towers up. This is independent corroborating evidence (beyond D-48's AR-mean-plausibility check) that the plausibility-filter fix improves the underlying gap-filler, not only the forecasting features derived from it.
**B-04 also re-run (same session, extending this entry):** Daily R² (T4/T9 mean, h=1→14), before→after: **DLinear improved cleanly at every horizon** (0.314→0.363, 0.237→0.310, 0.202→0.249, 0.164→0.192) — joins RF/XGB/SARIMAX/gap-filling in the "uniformly improved" group, consistent with it using the AR/CH4-history feature directly. **LSTM and LSTM_VSN show the same horizon-inconsistent pattern as TFT/TFT-Reg**: slightly worse at h=1/3 (LSTM -0.182→-0.209, -0.091→-0.152) but substantially better at h=7/14 (LSTM -0.207→0.044, -0.346→0.076, flipping to positive; LSTM_VSN -0.303→-0.156, -0.410→-0.051, still negative but much less so). Ranking/recommendation unchanged — DLinear remains the DL baseline, still below RF/XGB/SARIMAX at every horizon. Detail in `b03_b04_results.md` addendum.
**Still pending:** B01, B02, B05, B06, B07 remain un-rerun on corrected data (stale, D-48) — user has not yet prioritised these; B-08 (D-47, driver-realism) still queued behind the full re-run. `b03a_b03b_results.md`/`b03_b04_results.md` addenda should be read alongside this entry for the superseded SARIMAX conclusion and the B-04 DL comparison. Cross-ref D-45 (original pre-fix B-03a/B-03b results, now partially superseded for SARIMAX), D-41 (original pre-fix B-03/B-04 results), D-48 (the fix itself and its scope discovery).

---

### D-50 — 2026-07-02 — Supervisor-meeting scoping: outlier handling, pattern-fidelity metric, feature correlation, diurnal/livestock timing, daily-over-hourly priority
**Context:** Follow-up ideas from a supervisor meeting (progress reported as strong). Five items, discussion/scoping only — no code changes in this entry.
1. **Box-Cox / winsorization / outlier correction for gap-filling — audit executed (`d50_met_outlier_audit.csv`), two new confirmed contamination sources found.** We already do hard truncation (D-48's `plausibility_filter`: USTAR→[0,3] m/s, VPD→[0,15] kPa; earlier FCH4 [-500,3000] D-13, FCO2 [-100,100] D-25) plus a robust-median fallback. **Box-Cox remains assessed as low-value and not planned** (RF gap-filler is transform-invariant). Extended D-48's audit to the remaining ~9 met/soil columns (`data/Hourly/consolidated_hourly.csv`, all 3 towers, conservative UK-plausible bounds) — **found two genuine, previously-missed contamination sources**: (a) **`WS_0_0_1` (wind speed) has the identical fault signature to USTAR** — Tower 2 max 1370.6 m/s, Tower 4 max 1277.2 m/s (physically impossible; UK storm-force gusts rarely exceed ~50 m/s), and **100% of Tower 2's 312 WS>40 readings also have USTAR>3** — same sonic anemometer, same fault, confirming D-48's own SHAP finding that WS was the #2 contributor (+107 nmol) to the original spike. The `reddyproc_pipeline.py` comment claiming "PPFD/WS/SHF were clean" is **superseded — was incomplete** (checked differently, missed the tail). (b) **`TA_0_0_1` (air temperature), Tower 2 only** — 3,227 readings (6.8%) clustered tightly at −39.4 to −39.6°C (std=1.28), occurring in **both January and August** 2022–2023 — a real −39°C August reading in lowland Devon is impossible, this is a fixed sensor-fault/error-code value, not weather. Towers 4/9 are clean (0% beyond bound) — Tower-2-specific. **Not contamination, false alarm from an overly strict bound**: SWIN/PPFD showed large "beyond-bound"/mean-median-ratio numbers, but this is normal diurnal structure (roughly half of all hourly readings are night, giving small near-zero/slightly-negative values by construction) not a data-quality problem — the huge mean/median ratio diagnostic that flagged USTAR does **not** transfer to solar-radiation-type variables, since they're expected to be extremely right-skewed. PPFD's negative tail is negligible (7-8 readings site-wide beyond −30). RN/SHF/precipitation/soil temp/soil moisture: all clean, zero readings beyond bound at every tower. **Recommendation: add `WS_0_0_1: (0,40)` and `TA_0_0_1: (-20,40)` to `MET_PLAUS` in `reddyproc_pipeline.py`, same mechanism as USTAR/VPD** — not applied this session (would trigger another full data-regeneration + re-run cascade, and the user has explicitly paused further B-notebook reruns pending more reading); flagged as the next data-correctness fix when reruns resume.
2. **Pattern-fidelity metric (recreate the system's temporal pattern, not just point-accuracy) — still not implemented, scoping only.** Directly explains tension already observed in the project: R² is squared-error-based and penalizes spike-timing misses harshly even when a model captures the general shape correctly — this is part of why B-05's transform (D-42) and B-06's hurdle model (D-43) scored poorly on R² despite the hurdle classifier having genuinely useful recall (B-07, D-44's early-warning framing was built specifically to salvage that mismatch). **Candidate pattern-fidelity metrics for a future addition**: cross-correlation at lag 0 (does predicted vs actual move together, independent of exact magnitude), dynamic time warping (DTW) distance (tolerant of small timing shifts), or a spectral/periodogram comparison (does the model reproduce the diurnal/seasonal cycle's dominant frequencies). Not yet implemented — a bounded implementation (most naturally as an addition to `src/evaluation/metrics.py`, applied retroactively to existing B-03 predictions) is a candidate next task.
3. **Feature correlation / multicollinearity check — executed (`d50_feature_correlation.csv`, `d50_feature_vif.csv`).** Correlation matrix + VIF computed on all 41 `fx_`/`ar_` columns in `forecast_daily_v2.csv`. **Confirms substantial, structurally-expected multicollinearity**: 44 feature pairs with |r|≥0.8, 25/41 features with VIF>10 (worst: `fx_SWC_roll14` VIF≈13,060, `fx_SWC_roll7` VIF≈6,791) — almost entirely within the soil-moisture/soil-temperature lag+rolling-window family (`fx_SWC_lag7/14/21/28`, `fx_SWC_roll7/14`, and the `fx_TS_*` equivalents), which are smoothly-varying transforms of just two underlying slow-moving series (SWC, TS), plus expected pairs (`fx_TA_mean`/`fx_TA_max`/`fx_TA_min`; `fx_DOY_cos`/`fx_is_growing`, r=−0.90; `ar_ch4` lags vs `ar_ch4_drm7`). **Not a data bug and not actionable as-is**: RF/XGB (production) are largely robust to multicollinearity — trees just alternate credit among correlated features rather than losing accuracy — but this **does mean SHAP-based individual-lag importance (I-01) should be read as "this family of soil-history features matters," not "this exact lag matters more than its neighbour."** No feature removal recommended (the lag/rolling ladder was intentionally built for F-01 richness); flagged as an interpretation caveat for I-01/any future write-up, not a fix.
4. **Diurnal cycle and livestock time-of-day behaviour in forecasting.** Diurnal cycle **is already captured** on the hourly track (`is_daytime` flag + hour-of-day sin/cos, part of the D-41 enrichment). **Livestock behaviour by time-of-day is NOT captured and is a genuine, likely-unfixable-with-current-data limitation**: `fx_lsu_dens`/`fx_graze` are built from daily location-census counts (D-27/D-28), so there is no signal on whether cattle are grazing at 2am vs 2pm — sub-daily livestock activity/collar data was never available (part of why R-04/GreenFeed was dropped earlier as animal-scale rather than EC-scale data). No action planned absent new data.
5. **Prioritise daily over hourly going forward.** User direction — daily is already the production track (B-03 daily RF/XGB) and D-49's re-run was already scoped daily-first; future reporting/experiment effort should continue weighting daily over hourly accordingly.
**Status:** items 1 and 3 executed this session (diagnostics only, no fix/retrain applied — item 1's WS/TA fix is staged, not applied, per the user's explicit pause on further rerun cascades); items 2, 4, 5 remain scoping/no-code. Cross-ref D-42 (transform, negative), D-43 (hurdle, negative for daily), D-44 (early-warning reframing), D-48 (existing outlier-filter precedent and audit methodology this extends), D-27/D-28 (livestock feature construction), I-01 (SHAP importance, now caveated by the VIF finding).
**Correction (2026-07, D-51):** item 1's flagged WS/TA contamination was later found to **not reach the production pipeline** at all (see D-51) — the urgency framing above ("next data-correctness fix when reruns resume") is superseded; it remains worth fixing eventually for the raw EC-tower data's own honesty, but is not a pending production bug.

---

### D-51 — 2026-07-04 — outlier-correction technique comparison (F-09b): winsorization/Hampel filter vs hard truncation, and a correction to D-50's scoping
**Decision:** Before applying D-50's staged WS/TA hard-truncation fix, tested whether winsorization or a dedicated outlier-correction algorithm (Hampel/rolling-MAD despiking) are better alternatives, using the two D-50-confirmed contamination cases (`WS_0_0_1`, `TA_0_0_1` @ Tower 2) as real ground truth. Standalone script (`f09b_outlier_technique_check.py`, scratchpad, same precedent as F-09a) — no changes to `reddyproc_pipeline.py`/`gapfill_rfm.py`/`build_sms_met_dataset.py`/any `data/Hourly/*.csv`, all reuse via unmodified imports plus a runtime-only self-restoring `MET_PLAUS` monkeypatch for the truncation config.
**Scoping finding (corrects D-50):** D-50's audit ran against the raw EC-tower file; `build_sms_met_dataset.py` (D-35) already swaps `WS_0_0_1`/`TA_0_0_1` to external Site-station readings for all towers before gap-filling — verified the production EXT dataset's WS/TA are **already clean** (max 32.2 m/s, range −7.24…32.1°C). **D-50's confirmed contamination does not reach the production pipeline** (unlike USTAR/VPD, which stay EC-sourced with no external twin — why D-48's fix mattered). D-50's "next data-correctness fix" framing is downgraded accordingly.
**Result 1 (does each technique clean the series?):** hard truncation and winsorization **both fully resolve the contamination by construction**. **Hampel filter is a genuine, partial failure**: WS improves ~90% (1370→~135 m/s) but doesn't fully clear the physical bound; TA at Tower 2 is **essentially untouched at either window size (25h or 169h)** — the fault block (median run 11.5h, mean 159.6h, max ~1805h/~75 days, spanning both Jan and Aug) is long/internally-consistent enough that the rolling local median/MAD become dominated by the fault value itself, exactly as pre-registered/predicted before running.
**Result 2 (does it matter downstream?):** **null result** — gap-filling R² is statistically indistinguishable across every config at every tower/scenario (within ~0.01 R², `results/f09b_gapfill_summary.csv`), including the fully uncorrected baseline vs. clean production. A materially different outcome from D-48's USTAR/VPD fix, which visibly moved AR-feature means and fixed clear spurious spikes. Two candidate (undistinguished) explanations: (1) WS/TA's contamination carries less predictive leverage than USTAR's did (D-48's own SHAP found USTAR contributed 244 of a 380 nmol spurious spike vs WS's 107); (2) this calendar-gap CV harness (masks ≤288h chunks within an otherwise-available domain) may not exercise the *last-resort mean/median fallback* failure mode that made USTAR/VPD damaging, which only fires during genuine multi-month real blackouts — a caveat on this harness generally, not just this result. **Caveat on result strength**: `N_REPS=2` gives an impression of "within noise," not a rigorous variance estimate.
**Recommendation:** no production change needed (WS/TA already clean there); if the EC-sourced WS/TA are ever used directly, prefer truncation or winsorization over Hampel for this contamination profile (both fully resolve it; Hampel doesn't). D-50's flagged fix remains worth doing eventually but is not urgent. Flag the calendar-gap-CV-may-under-detect-long-blackout-damage caveat for any future contamination audit. Full write-up: `notebooks/04_feature_engineering/F09b_results.md`. Cross-ref D-48 (USTAR/VPD fix and its audit methodology), D-50 (the contamination this corrects/extends), D-35 (the Site-station swap that makes the scoping finding true).

---

### D-52 — 2026-07-05 — Long-range climate scenario dataset arrived: `data/Simulated Climate Data/` (North Wyke confirmed, site coverage resolved; not a literal copy of the cited paper)
**Trigger:** user obtained the actual CMIP6-based climate scenario files, plus the Semenov, Senapati, Coleman & Collins (2024) *Data in Brief* paper (DOI 10.1016/j.dib.2024.110709) originally identified as a candidate dataset in D-46. Investigated both against each other before use.
**Site-coverage question resolved (D-46 caveat a, now closed):** `data/Simulated Climate Data/NWWG.st` gives site coordinates lat 50.77, lon −3.90, altitude 177m — an exact match to North Wyke Farm Platform. Confirmed visually too: the site map (`RFF.26sites in GB.PNG`) plots "NW" in Devon, exactly where NWFP sits. **North Wyke is directly one of the 26 sites — no nearest-site proxy needed.**
**Important finding — the delivered files are NOT the literal published Zenodo dataset the paper describes.** The paper documents: 1000 stochastic yearly samples per site, for 3 discrete time-slices (baseline 1985–2015, "2030"=2021–2040, "2050"=2041–2060), 2 SSPs (SSP2-4.5, SSP5-8.5), one `.st` metadata file per scenario. What's actually on disk (verified by reading the files directly) is a **different, related product**: the folder's own `Readme.txt` (Oct 2023, project-internal, labeled "RFF" — almost certainly Rothamsted's *Resilient Farming Futures* programme, the same grant credited in the paper's acknowledgments) describes a **continuous transient version** instead — one unbroken 2020–2090 trajectory (71 years, confirmed via row counts: 25,915 = 71×365, LARS-WG's fixed 365-day calendar with no leap days) per {GCM × SSP × realization}, only one shared `.st` file (`NWWG.st`), and — per the files themselves, not even mentioned in the Readme — **3 SSPs, not 2**: `ssp126` (SSP1-2.6, low-emission, not documented anywhere in this folder's own paperwork), `ssp245`, `ssp585`. 100 realizations per {GCM × SSP} (1500 files) plus a 100-realization baseline file (`NWWG.dat`, not literally 1000 as the paper states) = 1501 total files. **The paper remains the right reference for methodology** (LARS-WG 8.0 downscaling approach, the 5-GCM selection rationale, SSP definitions/radiative-forcing framing, site-selection criteria) — it is just not a literal manifest of these specific files. Two larger internal docs (`RFF_MOHC_CMIP6_climate_model_scorecard.docx`, `RFF_MOHC_climate_projections_analysis.pdf`) exist in the folder but were not opened this session (out of scope for this pass).
**Variables confirmed (D-46 caveat b, unchanged):** `NWWG.st`'s `[FORMAT]` section confirms exactly `YEAR, JDAY, MIN (TMIN °C), MAX (TMAX °C), RAIN (mm/day), RAD (solar radiation, MJ/m²/day)` — 4 climate variables, nothing else. **Still missing, as D-46 already flagged: wind speed, VPD/humidity, USTAR, soil temperature/moisture, SHF, wind direction** — all of B-03's other daily drivers. This gap is unchanged by the file discrepancy above.
**Sanity check (data legitimacy):** compared TMAX 2020s vs 2080s for one GCM (HadGEM3-GC31-LL) across all 3 SSPs — warming by the 2080s: ssp126 +1.0°C, ssp245 +1.7°C, ssp585 +3.1°C. Monotonic in the physically-correct direction (more warming under higher emissions) — good evidence the data is legitimate and correctly labeled.
**Net effect on D-46's long-range/Phase-07 feasibility scoping:** upgrades requirement (4) from "candidate, unconfirmed coverage" to "confirmed, in hand, richer than expected" (71 continuous years and 3 SSPs vs. the originally-assumed 2 discrete snapshots and 2 SSPs) — but requirement (4)'s variable-coverage gap (only 4 of ~11 daily drivers) is confirmed to persist exactly as scoped, and is the immediate open question (see below). D-46's other requirements (frozen model artifact, scenario-capable feature pipeline, livestock/management assumption, AR-history strategy, extrapolation-range check, B-08 driver-realism) are all still outstanding, unaffected by this arrival.
**Decision on missing drivers (resolved this session): climatology via historical-day resampling, not a Copernicus CMIP6 pull.** The user raised pulling raw CMIP6 GCM output directly from the Copernicus Climate Data Store (the same source the paper itself used) for wind/VPD/USTAR/soil/SHF. Weighed against I-01's actual SHAP ranking (`I01_results.md`): `fx_WS`/`fx_VPD` sit in the lowest reported tier (mean|SHAP| 1.8–4.1, ~8–17x below `fx_lsu_dens`'s 31.8 and below even `ar_ch4_lag48`); **USTAR does not appear in the top-15 at all**. Given (a) these variables are low-importance to begin with, (b) raw CMIP6 is coarse-grid (~1.25°×1.9°, not site-specific) and would need its own bias-correction against the observed NWFP baseline to be locally meaningful — non-trivial effort — and (c) Phase 07 has much larger unresolved pieces (no frozen model artifact yet, no scenario-capable feature pipeline, no livestock assumption), **Copernicus is skipped for now**. **Chosen method: historical-day resampling ("climatology" in the richer sense, not a flat mean)** — for each missing variable, per tower, per future scenario day, sample an actual historical day's value from the same time-of-year (day-of-year window) in the real 2018–2023 observed record, rather than using a single fixed mean-per-day. This preserves realistic day-to-day variability/noise (unlike a deterministic per-day mean, which would produce an artificially smooth series) while still carrying no future trend — appropriate since none of these variables (USTAR, soil temp/moisture, SHF; wind/VPD to a lesser extent) have a strong climate-change-driven trend at decadal scale relative to their own day-to-day weather variability. Conceptually the same mechanism as `reddyproc_pipeline.py`'s `mdc_gapfill()` last-resort fallback tier (D-48's hourly-climatology fill), applied prospectively instead of retrospectively. **Not yet implemented** — this closes the design question, not the code; building the actual resampling function is future work, alongside D-46's other outstanding Phase-07 requirements. Cross-ref D-46 (original scoping + caveats), D-35 (external MET sourcing, the same site-vs-EC distinction relevant to any climate-scenario driver swap-in), D-48 (the `mdc_gapfill` mechanism this borrows from), I-01 (the SHAP evidence this decision rests on).

---

### D-53 — 2026-07-05 — recursive 365-day daily rollout backtest (B-09): does autoregressive forecasting compound error?
**Decision:** Direct empirical test of D-46 requirement 5 (AR/CH4-history strategy for a horizon with no real recent observations — "self-referential rollout risks compounding error across a multi-year projection"), and a direct operationalisation of `FORECASTING_LEARNINGS.md`'s lesson ("always validate a rollout mechanism against a real held-out window before trusting it on a genuinely blind future window"). Single fixed anchor (**2021-12-16**, Tower 4), one continuous 365-day recursive chain per model — not a walk-forward evaluation across many origins (which never lets error compound past a few days). Every model fit fresh on data strictly ≤ anchor (no reuse of B03/B03a/B03b/B04's existing ≤2021-12-31 results, which would leak real 2021-12-17…12-31 observations into "training"). Perfect-foresight `fx_` drivers throughout (isolates the recursive-mechanism question from driver-forecast realism, B-08/D-47's separate scope). **Committed scope: SARIMAX, RandomForest, XGBoost, LightGBM, DLinear, LSTM — Tower 4 only** (user-confirmed, given TFT's known instability D-45/D-48 and TabPFN's uninstalled/unresearched context-length fit make them poor uses of bounded session time). New shared module `src/models/recursive_rollout.py`: tree-model AR-recompute-and-append loop (reuses `build_forecasting_matrix_v2.py`'s exact shift/rolling math, spot-checked to 6 decimal places against the real precomputed columns), a new DL slide-forward harness (none existed anywhere in `forecasting_dl.py` — confirmed its `run_track()` is purely fixed-window batch), day-of-year-windowed climatology and chain-persistence baselines, and lead-time-binned evaluation (days 1-7/8-30/31-90/91-180/181-270/271-365 — the M5-lesson analogue of "don't blend across the hierarchy," applied to lead-time-within-the-chain instead of store/category).
**Result 1 — the 1-7 day bin is a small-sample artifact, not a real signal.** Only 3 real `y_observed` days fall in that bin (the target window's real-data gaps happen to land right after the anchor); catastrophic-looking R² there (e.g. DLinear -31, SARIMAX -5.6) reflects n=3 noise, not an immediate mechanism failure — the actual predicted values in that window are physically reasonable and not wildly divergent.
**Result 2 — recursive rollout does not inevitably collapse; it holds up for a multi-month stretch.** In the trustworthy bins (91-180, 181-270; n=88-90), **every model except LSTM achieves positive R² and MASE<1**, beating both a chain-persistence and a day-of-year climatology baseline roughly 3-9 months into the 365-day chain. This is more encouraging than D-46's original a-priori "compounding error will dominate" framing.
**Result 3 — universal late-window degradation (days 271-365, ≈Sept-Dec 2022), but likely NOT primarily a recursion artifact.** Every model, including the two baselines, degrades in the final quarter. Critically, **SARIMAX shows the same pattern despite having zero recursive self-feedback** (its forecast comes from one `get_forecast(steps=365)` call propagating the frozen anchor's Kalman state analytically, never re-consuming its own noisy output) — a model with no feedback loop degrading identically suggests the more likely cause is late-2022 being intrinsically harder to predict for this tower/season (grazing/weather variance), not compounding AR-feedback error specifically. Not fully disentangled — a future test with a different anchor date would be needed to separate "day 271-365 of any chain" from "late-2022 specifically."
**Result 4 — model ranking (trustworthy bins): SARIMAX and LightGBM most consistently reliable** (MASE 0.86-0.96 in both stable bins); **DLinear achieves the single best R² of any model (0.417, bin 181-270)** but is the least stable early on (real, not small-sample-artifact, poor MASE 2.3-2.9 through day 90, n=19-50); **LSTM is the weakest model throughout**; RF/XGB track each other in the middle of the pack.
**Caveats:** N_REPS=1 (single anchor/chain, not a distribution); ground truth itself is only ~48.9% true-hourly-observed in the target window (D-51-style caveat, stated throughout); tree track uses `forecast_daily_v2.csv` directly while DL track resamples `forecast_features_v2.csv` (an existing B-03-vs-B-04 feature-source asymmetry, not new here).
**Recommendation:** recursive daily rollout is a viable candidate for future long-range work, not a dead end — SARIMAX or LightGBM are the best-evidenced single choices on this test; DLinear's long-horizon strength (best R² at 181-270) is worth a second look once its early-window instability is addressed. The late-window degradation should be flagged in any future long-range design regardless of its exact cause. Full write-up: `notebooks/05_benchmarking/b09_results.md`. Stretch items (TFT, Tower 9, TabPFN-prep) not attempted this session — genuinely deferred, not silently dropped. Cross-ref D-46 (requirement 5, the question this directly tests), D-47 (B-08, the separate driver-realism question this does not address), D-52 (climate-scenario data, the eventual consumer of whichever rollout design this informs), D-45/D-48 (TFT instability, why it's deferred).
**Addendum — multi-anchor extension (2018-2022) revises Results 3/4 above.** Re-ran the identical rollout anchored at Dec-16 of 5 different years (136s total). **The late-window degradation is NOT universal** — only 2018 and 2021 show it; 2019/2020 stay flat-to-positive at 271-365 (SARIMAX, a zero-feedback structural control, even peaks there in 2020) — it is a year-specific seasonal-echo effect (the anchor's own December flux level relative to the rest of that particular year), not a universal "day 271-365 always fails" law. **DLinear's single-anchor "best R² anywhere" (0.417) was a fluke, not robust** — multi-anchor mean R² is -4.752 (worst of all models, driven by a catastrophic 2018 result). **Revised ranking: XGB is the most robust model** (only positive multi-anchor mean R², 0.003; mean MASE 0.968), LightGBM close behind (-0.014/0.978); SARIMAX/RF land near persistence on average; LSTM consistently weak. This supersedes the single-anchor recommendation above (SARIMAX/LightGBM/DLinear) — **use XGB if choosing one model**. Full revision: `notebooks/05_benchmarking/b09_results.md` §3-5.

---

### D-54 — 2026-07-06 — recursive-rollout improvements (B-10): ensemble, blended AR, H=1 DL retrain
**Decision:** Direct follow-up to D-53/B-09's modest result — tested 3 concrete improvement ideas, each reusing B-09's `recursive_rollout.py` machinery (extended backward-compatibly, not replaced): **(1) blended AR** — `tree_rollout` gained optional `alpha`/`clim_series` params (`alpha=1.0,clim_series=None` reproduces B-09's exact original chain, verified bit-for-bit, max abs diff 0.0); tested alpha∈{1.0,0.5,0.0} blending recursive memory with day-of-year climatology at append-time. **(2) ensemble** — unweighted and MASE-weighted (weights frozen from B-09's own multi-anchor MASE, not re-derived here — avoids circularity) mean of RF+XGB+LightGBM+SARIMAX. **(3) H=1 DL retrain** — confirmed zero new shared-module code needed (`forecasting_dl.make_windows(ser,L,H,stride=1)` already accepts H directly, bypassing `TRACKS`/`build_windows`); retrained DLinear/LSTM natively for single-step-ahead instead of reusing the H=14 model's first output. Single-anchor (2021-12-16) smoke test first, then the full 5-anchor (2018-2022) sweep — mandatory per B-09's own "don't trust a single anchor" lesson; verdicts below are from the 5-anchor sweep.
**Result 1 — blended AR fails, cleanly and monotonically.** Pure recursion (alpha=1.0) beats every blend for all three tree models (e.g. XGB: 0.003 > -0.050 > -0.114 mean R² at alpha=1.0/0.5/0.0) — recursion is already extracting the available signal; diluting it with climatology only hurts.
**Result 2 — ensemble is a modest, genuine win.** Unweighted ensemble mean R²=0.012, MASE-weighted 0.011 — both beat the best individual model (XGB alone, 0.003), the best mean R² found across B-09+B-10 combined, though MASE ticks up marginally (0.975 vs 0.968). Unweighted and MASE-weighted variants are nearly identical (B-09's frozen weights cluster tightly, 0.24-0.26 each).
**Result 3 — H=1 DL retrain is mixed, model-dependent.** LSTM-H1 improves consistently over B-09's H=14-truncated LSTM (R² -0.364 vs -0.438, MASE 1.073 vs 1.104). DLinear-H1 is worse on R² (-1.729 vs -1.460) despite marginally better MASE — DLinear's instability (already flagged in D-53's addendum as anchor-dependent, not robust) persists under H=1 retraining. Not a uniform DL fix; single-anchor smoke test showed the opposite ranking for DLinear (another instance of DLinear's single-anchor behaviour not generalizing).
**Verdict:** none of the three ideas closes the R² gap to a genuinely good result (best mean R² still only 0.012); the spike-blindness from B-05/B-06/B-07/B-09 is nudged, not fixed. **Recommendation: use the unweighted RF+XGB+LightGBM+SARIMAX ensemble** if deploying a single recursive daily rollout — cheapest genuine improvement found. Do not use blended AR. H=1 retraining is not a blanket DL fix. Full write-up: `notebooks/05_benchmarking/b10_results.md`. Single-anchor smoke-test rows appended to `benchmarks.csv` (102 rows, `replication="B10"`); multi-anchor sweep is a diagnostic check only, not appended (same precedent as B-09). Cross-ref D-53 (the baseline this improves on), D-46 (long-range scoping this precursor work supports).

---

### D-55 — 2026-07-06 — monthly-resolution rollout + downscale-to-daily (B-11): does coarser resolution help, and does the gain survive downscaling?
**Decision:** Direct follow-up to D-53/D-54 (B-09/B-10) — tests the M5-hierarchy lesson (coarser aggregates score better) at monthly resolution, then downscales back to daily for a direct comparison against B-09/B-10. New sibling module `src/features/build_forecasting_matrix_monthly.py` resamples `forecast_daily_v2.csv` up to monthly (sum for precip; mean for 13 `fx_` columns; min/max for TA; fraction-of-month for binary flags; month-end value for the `days_since_grazing` counter; DOY sin/cos recomputed fresh from month-of-year, not resampled; soil lags/rolls re-lagged at monthly windows; new `ar_ch4_mlag{1,2,3}`) into `forecast_monthly_v2.csv`. `recursive_rollout.py` extended (not replaced) with `ar_features_for_month`, `monthly_rollout`, `moy_climatology`, `downscale_monthly_to_daily`. **Model roster: SARIMAX + RF/XGB/LightGBM only** — DLinear/LSTM scoped out (only ~90 monthly rows per tower, too thin for a from-scratch DL window regime, a flagged stretch item not silently dropped). **Monthly anchor = last fully complete month before the daily anchor** (e.g. 2021-11-01 for the 2021-12-16 daily anchor) — avoids leaking post-anchor days into "training"; 13 months forecast per anchor, spanning the same range as B-09/B-10's 365-day window. **Downscaling = hybrid-calibration**: reuses the corresponding B-09 daily chain (same anchor/tower/model) as the within-month *shape* template, recentered so its monthly mean matches the independently-derived monthly prediction (`daily_synth[d] = daily_template[d] - mean(daily_template over month) + monthly_pred[month]`) — verified exact by construction (max abs diff 0.0 for every model). Single-anchor smoke test first, then the 5-anchor (2018-2022) sweep, per B-09's own lesson; verdicts below are from the 5-anchor sweep.
**Result 1 — monthly-native evaluation is a real, substantial improvement**, confirming the M5-hierarchy prediction. Mean R²/MASE across 5 anchors: LightGBM 0.156/0.724 (vs 0.014/0.978 daily), XGB 0.147/0.722 (vs 0.003/0.968 daily), RF 0.050/0.813 (vs -0.067/1.024 daily) — all substantially better than B-09's daily-native numbers. SARIMAX is the one exception (monthly R²=-0.120, worse than its daily -0.039).
**Result 2 — the improvement does NOT survive downscaling back to daily.** Downscaled-to-daily mean R²/MASE: XGB -0.000/0.980, LightGBM -0.016/0.989, RF -0.064/1.023, SARIMAX -0.244/1.104 — essentially unchanged from (RF/XGB/LightGBM) or worse than (SARIMAX) B-09's own daily-native numbers. By bin, the late-window bin (271-365) genuinely improves (all four models flip from clearly negative, e.g. -0.51 to -0.59, to slightly positive, 0.05-0.06) but the short/mid bins (1-7, 8-30) get *worse* on average across 5 anchors than B-09's originals.
**Mechanism (not a bug)**: the downscaling method reuses the daily template's own within-month *shape* unchanged and only recenters the *monthly mean* — any within-month spike-miss error is inherited unchanged from the daily template, since the monthly model never sees individual days. Where B-09's daily models already capture the seasonal trend reasonably, the correction adds little; where the monthly model's own seasonal fit disagrees, it can subtract signal (visible in the 1-7/8-30 regression). The late-window improvement is the one place this clearly helps, because B-09's daily models are weakest there (the D-53-addendum seasonal-echo effect) and an independent monthly bias correction has real room to add value.
**Recommendation:** monthly resolution is genuinely better for evaluation/reporting where monthly granularity suffices (e.g. digital-shadow scenario aggregates); do not expect it to fix daily-level spike-blindness via this downscaling method — if daily output is required, B-10's ensemble (D-54, mean R²=0.012) remains the best available daily-resolution result. Full write-up: `notebooks/05_benchmarking/b11_results.md`. Single-anchor smoke-test rows appended to `benchmarks.csv` (24 rows, `replication="B11"`); multi-anchor sweep is a diagnostic check only, not appended (same precedent as B-09/B-10). Cross-ref D-53/B-09 (source of the daily downscaling templates), D-54/B-10 (the best daily-native result this does not beat).

---

### D-56 — 2026-07-06 — combined ensemble + monthly-downscale (B-12): executed despite low expected payoff, and why
**Decision:** Direct follow-up combining D-54/B-10's winning idea (unweighted ensemble of RF+XGB+LightGBM+SARIMAX) with D-55/B-11's monthly-rollout-plus-downscaling framework — build the ensemble at monthly resolution, downscale to daily using **B-10's own daily ensemble chain** (not each individual model's own chain, as B-11 did) as the shape template. User explicitly asked to proceed despite B-11's finding that the downscaling mechanism this inherits mostly cancels its own gain, to get a concrete number rather than relying on inference. No new shared-module code — reuses `recursive_rollout.py`'s existing functions unmodified, hardcoding B-10/B-11's winning configs as frozen constants (no new grid). Single-anchor (2021-12-16) smoke test first, then the 5-anchor (2018-2022) sweep, per this project's now well-established norm.
**Result 1 — single-anchor result looks like a clear win.** Overall n-weighted R²/MASE: B12 0.075/1.002 vs B10's daily ensemble alone -0.034/1.122. By bin, B-12 clearly beats B-10 in 271-365 (0.066 vs -0.436) and 8-30 (-0.052 vs -0.200).
**Result 2 — the multi-anchor (2018-2022) sweep reverses this.** Mean R²/MASE across 5 anchors: B10_daily_ensemble_original 0.012/0.975, B12_monthly_ensemble_downscaled **-0.011/0.993** — B-12 is now slightly *worse* than B-10 alone. By bin, short-lead bins (1-7, 8-30) and the late bin (271-365) genuinely improve, but mid-range bins (91-180, 181-270) get worse, netting to a small negative overall.
**Mechanism**: identical to B-11's own finding (D-55) — the downscaling step inherits the ensemble's own within-month shape errors unchanged, correcting only the coarse monthly bias; averaging this across 5 different years erodes the single-anchor gain. **This is the third instance this session of a single-anchor result being reversed by the multi-anchor sweep** (after B-09's DLinear and, less dramatically, B-10's H=1-retrain single-anchor read) — reinforcing this project's own headline methodological lesson.
**Recommendation:** do not deploy B-12's combined configuration — B-10's daily ensemble alone (mean R²=0.012) remains the best available daily-resolution recursive-rollout result, is simpler (one model family, not two), and performs at least as well on average. The exercise was still worth running: a clean, concrete confirmation rather than an inference, and a instructive third demonstration of the single-anchor-vs-multi-anchor reversal pattern. Full write-up: `notebooks/05_benchmarking/b12_results.md`. Single-anchor smoke-test rows appended to `benchmarks.csv` (12 rows, `replication="B12"`); multi-anchor sweep is a diagnostic check only, not appended (same precedent as B-09/B-10/B-11). Cross-ref D-54/B-10 (source of the ensemble idea), D-55/B-11 (source of the downscaling mechanism and its explanation). **This closes the B-09→B-12 recursive-rollout experiment sequence** — B-10's unweighted ensemble is the final recommended daily-resolution configuration.

---

### D-57 — 2026-07-06 — B-13: TFT and TabPFN for recursive rollout, plus DLinear/LSTM chain-plot extension
**Decision:** Fills in B-09's two remaining stretch items (TFT deferred for instability D-45/D-48; TabPFN deferred, unresearched) against the same B-09→B-12 recursive-rollout backtest, Tower 4, plus extends the actual/gap-filled/predicted chain-plot visualization (introduced for B-10's tree models across all 3 towers x 5 anchors) to DLinear/LSTM. **TFT**: reuses D-45's exact regularized recipe (`d_model=32, n_heads=4, weight_decay=1e-3, patience=5`), one adaptation — validation carve-out shortened from a full held-out year to the **last 90 days before the anchor** (early anchors, e.g. 2018, don't have a spare year of data). **TabPFN**: installed `tabpfn`/`tabpfn-time-series` (confirmed via research this is **not autoregressive** — one forward pass predicts the full 365-day horizon from a context+future-covariates dataframe, closer to SARIMAX's one-shot `get_forecast` than to any rollout loop); new `recursive_rollout.tabpfn_forecast()`. **Local inference mode** (user-confirmed) required a one-time browser license acceptance at ux.priorlabs.ai producing a `TABPFN_TOKEN` (stored in gitignored `.env`, never committed) — local mode's first-use model download is gated behind this even though inference then runs entirely on the local GPU with no per-call data transmission. Context uses real `y_observed` (gaps as NaN, TabPFN handles missing values internally) rather than `y_gapfilled`, deliberately avoiding the diffuse gap-filler-optimism caveat that applies to every other model's training target.
**Result 1 (Part A, DLinear/LSTM plots)**: 30 new plots, consistent with B-09's documented DLinear/LSTM behaviour — no new finding, pure visualization completeness.
**Result 2 (TFT)**: no reproduction of the original unregularized-TFT catastrophic-failure pathology — sane, non-NaN predictions. Multi-anchor mean R²=-0.237, mean MASE=1.055 — **the best deep-learning model in the entire B-09–B13 sequence** (beats LSTM -0.438, DLinear -1.460) but still behind every tree/SARIMAX model. Directly confirms D-45's regularization fix is a general pattern, not a one-off patch for B-03b's specific setup.
**Result 3 (TabPFN) — the headline finding of this session**: multi-anchor mean R²=-0.006 (competitive, between LightGBM and XGB), mean MASE=**0.862 — the best of any model tested across the entire B-09–B13 sequence**, beating B-10's ensemble (0.975) and XGB (0.968), with every one of its 5 per-anchor MASE values comfortably below 1. Also the only model with positive mean R² in both late-window bins (181-270, 271-365), where most other models show the seasonal-echo degradation (D-53's addendum). **Achieved with zero training or hyperparameter search** — pure in-context learning from the context+future-covariates data supplied at inference time.
**Recommendation:** B-10's ensemble (D-56) remains the headline R² recommendation — unchanged. TabPFN earns a standing mention as a genuine alternative/complementary choice, especially where MASE/robustness or avoiding a training pipeline matters — not a replacement recommendation, but not a footnote either. TFT confirms D-45's regularization-reverses-architecture-verdicts point generalizes. A TabPFN+tree ensemble is a natural but unexecuted follow-up (flagged, not pursued, to avoid re-opening the closed B-09→B-12 sequence). Full write-up: `notebooks/05_benchmarking/b13_results.md`. No `benchmarks.csv` rows appended (single-anchor smoke test only, same "diagnostic not ledger" precedent as every B-09–B12 multi-anchor sweep). Cross-ref D-45 (TFT recipe reused), D-53/54/55/56 (the baseline this compares against).
**Addendum — full 3-tower x 5-year TabPFN grid** (user asked for the same breadth every other model got, cheap to add at ~1-2s/anchor): Tower 4 mean R²/MASE (-0.006/0.862) reproduces exactly, confirming internal consistency. Tower 9 (4/5 anchors usable — 2018 anchor has 0% real test-window coverage): mean R²=-0.264, MASE=0.925, same qualitative pattern as Tower 4 (spot-checked visually: tracks the seasonal high-flux window, misses the largest spike). **Tower 2 is usable for only 1/5 anchors** (2018; 2019-2022 all have 0% real test-window coverage, the same severe data-scarcity issue already documented for the B-10 chain-plot extension) — its single MASE=0.243 reflects an unusually weak persistence baseline in that one narrow window, not a meaningful evaluation, and should not be read as "TabPFN performs exceptionally at Tower 2."

---

### D-58 — 2026-07-06 — B-14: GridSearchCV hyperparameter tuning for recursive rollout (retroactive log; CV-picked configs don't reliably transfer to rollout)
**Decision:** Re-opens the B-09→B-13 "closed" sequence (D-53-D-57) for one bounded hyperparameter-tuning round, prompted by noticing that every tree-model hyperparameter in B-10 (D-54) traces to D-41's single manual HPO pass, never a systematic search. **Design:** literal `GridSearchCV` for RF (16 combos)/XGB (36)/LightGBM (36) with a 3-fold walk-forward expanding-window CV (train ≤2019/val 2020, train ≤2020/val 2021, train ≤2021/val 2022) scored by one-step R², reusing B-10's exact pooled T2/T4/T9 training construction; SARIMAX gets a widened AIC-order grid (p∈{1,2,3}, q∈{0,1,2}, not literal GridSearchCV — a different, pre-existing selection method); DLinear/LSTM get a manual 12-combo grid with early-stopping validation (deprioritized, never completed — `b14_dl_grid.csv` was never generated). Winning hyperparameters were then plugged into B-10's exact 5-anchor (2018-2022) rollout mechanism for the real verdict, per this project's now-standard "one-step CV is a proxy, rollout is the real target" design.
**This entry was written after B-14a (tree/SARIMAX tuning + rollout validation) had already been executed but never logged** — written up here alongside D-59/B-15 once the gap was noticed. Two implementation issues were found and are logged honestly rather than silently fixed: **(1)** B-14's "tuned ensemble" (`Ensemble_tuned_trees`) was built as a 3-model mean (RF+XGB+LightGBM only) — it omitted SARIMAX, unlike B-10's actual 4-model ensemble baseline, so the original B-14-vs-B-10 ensemble comparison was not apples-to-apples (fixed in B-15, D-59, which builds the correct 4-model composition). **(2)** `compile_b14_results.py` originally compared tuned models against a hand-typed, stale B-10 baseline (R²=0.012 for the ensemble) rather than a value recomputed from `results/b10_ensemble_multi_anchor.csv` — and a first attempt at that recomputation used a different (pooled, not per-anchor) aggregation formula that silently disagreed with D-54's own published numbers; both are now fixed to use the established convention (n-weighted R²/MASE per anchor, then a simple mean across the 5 anchors — verified to reproduce D-54's exact published XGB=0.003/LightGBM=-0.014/SARIMAX=-0.039/RF=-0.067 baseline figures).
**Result 1 — CV-picked hyperparameters mostly don't transfer to rollout, but there is one real exception.** 5-anchor mean R²: `LightGBM_tuned`=**0.006** (beats B-10's own untuned LightGBM, -0.014, by a wide margin, and is close to B-10's ensemble, 0.012), `XGB_tuned`=0.002 (essentially unchanged from B-10's XGB, 0.003), `RF_tuned`=-0.072 (slightly worse than B-10's RF, -0.067), `SARIMAX_widened`=-0.054 (worse than B-10's SARIMAX, -0.039). **The 3-model tuned ensemble scores R²=-0.005**, underperforming B-10's 4-model ensemble baseline (0.012) — but this comparison conflates two different things (worse tuning *and* a missing 4th model), so it should not be read as "tuning made things worse" in isolation.
**Recommendation:** B-10's unweighted 4-model ensemble (D-54, R²=0.012) remains the best-validated production configuration. LightGBM's CV-tuned config is a genuinely useful finding on its own (best single tree model found in the B-09-B14 sequence) and directly motivated B-15's follow-up question — does scoring by rollout performance directly (rather than one-step CV) find an even better config, and can a properly-composed 4-model ensemble built from tuned components beat B-10 outright? Full write-up: `notebooks/05_benchmarking/b14_results.md`. No `benchmarks.csv` rows appended (diagnostic multi-anchor sweep, same precedent as every B-09-B13 multi-anchor sweep). Cross-ref D-41 (source of the never-grid-searched hyperparameters), D-54 (the baseline), D-59 (B-15, the rollout-based follow-up).

---

### D-59 — 2026-07-06 — B-15: direct rollout-based hyperparameter tuning (does scoring by the real 365-day metric, not one-step CV, find better hyperparameters?)
**Decision:** Direct follow-up to D-58/B-14 — instead of scoring hyperparameter combos by one-step CV R² (a proxy that D-58 showed doesn't reliably transfer), score each combo by its own actual 365-day rollout performance (fit → `tree_rollout()` → `bin_metrics()` → n-weighted mean R²). Manual grid bracketing B-10/B-14's values (RF 9 / XGB 12 / LightGBM 12 = 33 combos), searched at anchor 2021-12-16, with a 2-anchor stability check (top-3 shortlisted combos per model re-scored at anchor 2019-12-16, a differently-behaved year per D-53's addendum) — winner = **combined rank** (mean of n-weighted R² across both anchors), not the 2021 search alone. Winners validated on the full 5-anchor (2018-2022) sweep with the **correct 4-model ensemble** (RF+XGB+LightGBM+SARIMAX), fixing D-58/B-14's 3-model-only composition. SARIMAX/DLinear/LSTM excluded from the grid (SARIMAX already re-selects its own AIC order every anchor; DL's per-combo retrain cost breaks the "cheap grid" premise) — flagged as deferred stretch items, not silently dropped.
**Implementation bugs found and fixed before finalizing these numbers (a Sonnet second-opinion review of the already-executed pipeline, requested by the user):** **(1)** the first implementation computed the 2021+2019 combined-rank score but then discarded it in favor of the 2021-only rank via a dead-code overwrite — confirmed by direct recomputation from the already-saved `b15_rollout_grid_search.csv`/`b15_stability_check.csv` that this silently picked the *worst* of the 3 shortlisted RF combos (avg R²=-0.059 vs the true best -0.040) and a suboptimal LightGBM combo (avg R²=0.028 vs the true best 0.045); fixed, and Stage 2 (5-anchor validation) was re-run with the corrected winners (RF: max_features=0.3/min_samples_leaf=20; LightGBM: num_leaves=7/min_child_samples=20/learning_rate=0.05; XGB unaffected, its winner was already correct either way). **(2)** the 3-way comparison table (`compile_b15_results.py`) originally left two prose "interpretation" sections as literal unfilled placeholder text (`[result placeholder: ...]`) rather than a computed verdict — fixed to compute the B-14-vs-B-15 per-family win/loss/tie and production recommendation programmatically. **(3)** model names differ across B-10/B-14/B-15's raw CSVs (e.g. B-10's `SARIMAX` vs B-14's `SARIMAX_widened`), which broke the merge into 11 sparse rows instead of one row per model family — fixed via a normalized `family()` mapping, with raw per-source names kept as secondary columns for traceability (composition differences, e.g. 3-model vs 4-model ensembles, remain visible). **(4)** the same stale-baseline and aggregation-formula issues found in D-58 applied here too and are fixed identically (per-anchor-then-mean, matching D-54's published figures).
**Result (corrected numbers) — mixed on tuning method, but a genuine single-model win.** Per-family 5-anchor mean R²: **Ensemble** B-10=0.012 (best), B-15 4-model=0.007, B-14 3-model=-0.005; **LightGBM** B-15=**0.017** (best single model found across the entire B-09-B15 sequence, beating B-10's own ensemble), B-14=0.006, B-10=-0.014; **XGB** B-10=0.003, B-14=0.002, B-15=-0.009 (the more-rigorous 2-anchor selection did not generalize better here than B-14's CV pick); **RF** B-10=-0.067, B-14=-0.072, B-15=-0.078 (worse still — the combined-rank selection, though more principled, does not guarantee the best 5-anchor outcome from only 2 anchors); **SARIMAX** unchanged between B-14/B-15 (both re-run the same per-anchor AIC search). Rollout-based tuning beats CV-based tuning on 3 of 5 families (Ensemble, LightGBM, and a tie on SARIMAX) but loses on 2 (XGB, RF) — no uniform winner between the two tuning methods.
**Recommendation:** B-10's unweighted 4-model ensemble (D-54, R²=0.012) remains the best-validated production configuration — neither CV-based (B-14) nor rollout-based (B-15) tuning produced a *better ensemble*. However, B-15's rollout-tuned LightGBM alone is the best single model found in the whole B-09-B15 sequence, ahead of B-10's own ensemble — an ensemble built around this specific tuned LightGBM (rather than the equal-weight 4-model mean, which dilutes its gain with the still-weak RF/XGB/SARIMAX components) is a natural, unexecuted follow-up, flagged here rather than pursued (would re-open this now-closed side-thread again). **This closes the B-14/B-15 hyperparameter-tuning side-thread** — the original B-09→B-12 recursive-rollout sequence (D-53-D-56) and B-13's model additions (D-57) remain the standing recommendations, unchanged in their conclusion that B-10's ensemble is the best daily-resolution configuration, now additionally confirmed robust to two independent tuning attempts. Full write-up: `notebooks/05_benchmarking/b15_results.md`. No `benchmarks.csv` rows appended (diagnostic multi-anchor sweep). Cross-ref D-58/B-14 (the CV-based predecessor this responds to), D-54/B-10 (the baseline both fail to beat), D-53 (source of the "don't over-trust few anchors" lesson, which applies here to the tuning method's own anchor count, not just to reporting).
**Addendum — cross-tower generalization check (T2/T9), prompted by noticing B-14/B-15's tuning and evaluation were Tower-4-only throughout (training pools T2+T4+T9, but scoring never left Tower 4).** Real `y_observed` coverage check first: Tower 2 usable at only 1/5 anchors (2018, 27.9%; 2019-2022 all 0%), Tower 9 at 4/5 (2019-2022, 49-74%; 2018 is 0%) — Tower 2 too data-scarce for its own tuning search (reconfirms B-13's independent finding), Tower 9 good enough. **Part A (cross-tower eval of T4-tuned winners):** T4-tuned LightGBM — the single best model on Tower 4 (R²=0.017) — is Tower 9's *worst* model (R²=-0.388); T4-tuned hyperparameters do not transfer, and can actively hurt. Tower 2 (single usable anchor) is deeply negative across every model, SARIMAX catastrophically so (-3.396). **Part B (independent Tower-9 tuning):** a full 33-combo search + 2-anchor stability check scored on Tower 9 picks genuinely different winners (e.g. XGB: max_depth=2/lr=0.02/min_child_weight=10 vs Tower 4's max_depth=3/lr=0.01/min_child_weight=20) — but validated across Tower 9's 4 usable anchors, the outcome is **nearly identical to just reusing Tower 4's config** (all deltas within ±0.03 R², e.g. ensemble -0.228 T9-tuned vs -0.226 T4-tuned-on-T9). **Tower 9's poor performance is not primarily a hyperparameter problem** — points to something more structural (feature set, driver availability, or genuinely different flux dynamics) that this bounded grid cannot fix. One partial explanation surfaced: anchor 2020 is a catastrophic whole-anchor outlier for Tower 9 (R² -0.79 to -1.38 across every model, vs positive 0.01-0.19 at 2019/2021) — consistent with D-53's "anchor-specific, not universal" degradation pattern, not further diagnosed here. **Recommendation:** Tower 4 keeps B-15's tuned LightGBM/B-10's ensemble; Tower 9 gets no benefit from further tuning (reuse either config, least-bad option is the 4-model ensemble at R²≈-0.23); Tower 2 has too little real evaluation data for any reliable conclusion — treat as open pending more held-out data. Chain plots: `results/figures/b15_chains/T{2,9}_anchor*.png` (75 new plots, including a T4-tuned-vs-T9-tuned side-by-side for Tower 9). Full write-up: `notebooks/05_benchmarking/b15_results.md` (Addendum section — merged from a separate `b15_results_t2_t9.md` draft into the single results doc, per B10's one-notebook-one-results-doc convention). Cross-ref D-57 (B-13's own independent Tower-2-scarcity finding, reconfirmed here), D-53 (source of the anchor-specific-degradation pattern).

---

### D-60 — 2026-07-06 — descope the B01/B02/B05-B07 full rerun on D-48-corrected data (backlog cleanup)
**Decision:** D-48 flagged the entire forecasting-phase `benchmarks.csv` (FC-01 through B03b) as stale pending re-run after fixing the USTAR/VPD plausibility-filter bug. A scoped rerun (B-03, B-03a, B-03b only, logged under D-48) already happened — the full B01/B02/B05-B07 rerun was left "still pending" in CONTEXT.md's backlog. **Descoped, not deferred:** the production model (B-03, RF/XGB enriched trees) was part of that scoped rerun and showed only small, non-ranking-changing gains from the data fix (T4 RF daily h=1 0.357→0.365, h=14 0.270→0.280; T9 RF h=14 0.342→0.359) — the fix's real-world impact on the actual best-performing model is modest. B01 (baselines, superseded by B-03's enriched features), B02 (DL, already established as underperforming trees by a wide margin, D-38), B05 (asinh transform, already a clean NEGATIVE finding — R² is scored in original units where spikes dominate variance regardless of AR-feature-mean shifts, D-42), B06 (spike hurdle, already NEGATIVE for the daily/production track by a wide margin, >0.1 R² at T4, >0.25 at T9, D-43), and B07 (recency features + diagnostics, already marginal/inconsistent and explicitly does not flip B-06's verdict, D-44) are all **structural/architectural findings, not close calls** — a data shift of the same modest magnitude B-03 experienced is very unlikely to reverse any of them. **Recommendation:** do not spend further session time re-running B01/B02/B05-B07; treat B-03/B-03a/B-03b's already-completed scoped rerun as sufficient validation that the production conclusion (B-03 = best, D-41) is robust to the D-48 fix. **B-08 (driver-realism sensitivity, D-47) is no longer blocked waiting on this rerun** and can proceed directly. Cross-ref D-48 (the fix and the scoped B-03/B-03a/B-03b rerun this decision builds on), D-41/D-42/D-43/D-44 (the original findings whose robustness this argues for), D-47 (B-08, now unblocked).

---

### D-61 — 2026-07-06 — I-02: feature importance (native/SHAP/LIME) for the recursive-rollout models (B-10 + B-13), all 3 towers, full 5-anchor sweep
**Decision:** Fresh interpretability experiment for the B-09→B-15 recursive-rollout sequence — none of I-01's original feature-importance work (D-39) touched these models, since I-01 targeted the earlier FC-01/FC-02 windowed-forecast harness. **Explicitly not used as design precedent** per the user's instruction (I-01 is considered redundant from prior experiments and was left untouched, not deleted). Covers all 8 B-10/B-13 models (RF, XGB, LightGBM, SARIMAX, Ensemble_unweighted, Ensemble_MASEweighted, TFT, TabPFN), all 3 towers (T2/T4/T9, including T2 despite its known severe data scarcity, per explicit user choice), full 5-anchor (2018-2022) sweep. Three importance families: native (whatever a model exposes "for free" — tree `.feature_importances_`, SARIMAX coefficients, TFT VSN gate weights, TabPFN permutation importance as its only substitute), SHAP (TreeExplainer for RF/XGB/LightGBM; additive combination for the ensembles), and LIME (local, per-instance only, 6 representative days per anchor/tower). New shared module `src/interpretability/importance.py`. **SARIMAX/TFT/TabPFN excluded from KernelSHAP/LIME** — SARIMAX already has an exact closed-form linear-effect view via its coefficients (a KernelExplainer pass would be redundant and require re-running a 365-step `get_forecast` per perturbation sample, computationally intractable at this scale); TFT/TabPFN are architecturally mismatched with row-wise tabular explainers (TFT needs a full L-day encoder window per prediction; TabPFN is a one-shot whole-horizon forecast, not a per-row predictor) — stated scope limitations, not silent omissions.
**Result: `fx_lsu_dens` (livestock density) is the dominant driver, confirmed independently by every method that can see it** (RF/XGB native, RF/XGB/LightGBM global SHAP, SARIMAX coefficients at all 3 towers, TabPFN permutation importance at T4/T9) — directly reconfirms this project's central thesis (first established at F-01/F-02, D-27/D-30) from the recursive-rollout models themselves, never previously tested. **New finding: `fx_lsu_dens`'s SHAP importance grows with lead time, not shrinks** (mean |SHAP| at Tower 4: 7.6 at bin 1-7 → 38.4 at bin 181-270) — plausible mechanism: AR features degrade as the rollout's own compounding predictions dilute real recent history, while the exogenous livestock-density signal never degrades, becoming relatively more informative at long lead times. This is a plausible partial explanation for why B-15's tuned LightGBM (which leans harder on exogenous drivers relative to AR memory) outperforms the untuned ensemble. **TabPFN's driver ranking differs sharply at Tower 2** (temperature/soil-lag features dominate, `fx_lsu_dens` absent from top-4) — read as a symptom of Tower 2's known data scarcity (TabPFN's context uses real `y_observed` only, and T2 has real data in just 1/5 anchors) rather than a genuine ecosystem difference, consistent with this project's standing Tower-2 caution (B-13, B-15). **TFT's VSN channel importance is reported as a genuine scope limitation** — encoder/decoder channels are indexed generically (not mapped to real feature names), a real gap flagged plainly rather than glossed over.
**Recommendation:** no change to any production model — this is a diagnostic/interpretability layer on top of already-validated models. Full write-up: `notebooks/06_interpretability_uq/I02_feature_importance_rollout.ipynb`, `I02_results.md`. No `benchmarks.csv` rows (not a point-forecast benchmark). Cross-ref D-27/D-30 (original livestock-density finding), D-53/D-57 (source models), D-59 (B-15's cross-tower finding, for which this experiment's lead-time result is a plausible partial explanation).

---

### D-62 — 2026-07-06 — U-02: quantile-ML + conformal uncertainty for the recursive-rollout models (B-10 + B-13), all 3 towers, full 5-anchor sweep
**Decision:** Fresh UQ experiment for the same B-09→B-15 recursive-rollout models, prompted directly by the dissertation's own stated goal (uncertainty quantification for scenario analysis) and by B-14/B-15 establishing that further point-forecast tuning has hit its ceiling. **Explicitly not used as precedent: I-01's sibling, U-01 (D-40)**, which targeted the earlier FC-01/FC-02 harness and is considered redundant/superseded (left untouched, not deleted, per user instruction); `pinball`/`picp`/`mpiw` were freshly written into `src/evaluation/metrics.py` rather than migrated from U-01's inline versions. Per-model quantile mechanism: RF via the quantile-regression-forest trick on the already-fitted point model (no retraining); XGB/LightGBM via 3 separately-fit quantile-objective models (q=0.05/0.5/0.95, same hyperparameters otherwise, no new HPO); SARIMAX via `get_forecast().conf_int()` (essentially free); TFT's native quantile head deferred as a real architecture change out of scope, conformal-wrapping its point-forecast chain instead; TabPFN's native quantile output via `tabpfn-time-series`'s own `quantiles=` parameter (genuine library support). Conformal calibration: leave-one-anchor-out, per lead-time bin (reusing `bin_metrics`'s 6 bins, given the rollout's well-documented heteroscedasticity by lead time) — new `tree_rollout_quantile`, `RFQuantileAdapter`, `MultiModelQuantileAdapter`, `sarimax_quantile`, `conformal_margins_by_bin`, `lead_time_bin` added to `src/models/recursive_rollout.py`.
**Result: conformal calibration works, consistently, across every model type, at T4/T9** — every model converges to 0.88-0.90 PICP after calibration regardless of its raw (pre-calibration) coverage, which ranged from badly overconfident (RF/XGB/LightGBM raw PICP 0.35-0.50 — a concrete demonstration that a model trained to predict a quantile does not automatically produce a calibrated one) to already-reasonable (SARIMAX 0.83-0.92, TabPFN 0.72-0.87 raw, without any calibration at all). **After calibration, the Ensemble and RF have the sharpest (lowest-pinball) intervals at both T4 and T9** — despite RF being the *worst*-calibrated model raw, once corrected its underlying point accuracy carries through to a sharp, honest interval; TabPFN has the worst calibrated pinball at Tower 9 (13.0) despite good raw coverage, consistent with B-13's own finding that TabPFN's strength is bulk-error robustness (MASE) rather than interval sharpness specifically. **Tower 2 cannot support conformal calibration at all** (all conformal columns NaN — leave-one-anchor-out calibration needs real residuals from other anchors, and T2 has real data in only 1/5 anchor windows) — reported honestly as a data-availability limit, not hidden.
**Three real bugs found and fixed before finalizing these numbers**, illustrating the value of the Sonnet-style review habit applied proactively this time: (1) `tabpfn_forecast`'s quantile columns were matched by string (`str(0.05)`) against the library's actual float-keyed columns (`0.05`), silently producing 100% NaN raw TabPFN intervals; (2) `Ensemble_MASEweighted` was accidentally computed identically to `Ensemble_unweighted` (both used a plain 1/4 average) — fixed to use B-09's frozen MASE weights, and the full sweep was killed and re-run rather than reported with corrupted data; (3) a fan-chart visualization bug (caught via direct user inspection of the generated figures, not automated testing) decided calibrated-vs-raw once per whole chain rather than per day, silently showing blank gaps for any chain with a genuine mix of calibrated/uncalibrated bins even where a valid raw interval existed — fixed to fall back day-by-day, re-verified against both the case that exposed it (Tower 4/2022/SARIMAX) and a genuine no-fallback case (TFT, which has no raw quantile mechanism to fall back to at all).
**Recommendation:** use conformal-calibrated intervals, not raw quantile output, for any tree-based model in production — raw tree quantiles are demonstrably overconfident. The calibrated Ensemble or RF give the sharpest intervals at T4/T9, consistent with B-10's ensemble already being the point-forecast recommendation. SARIMAX/TabPFN are reasonable fallbacks where a calibration step isn't available. Tower 2 cannot currently support calibrated uncertainty estimates — an open question pending more held-out data, same as every other Tower-2 caveat in this project. Full write-up: `notebooks/06_interpretability_uq/U02_uncertainty_rollout.ipynb`, `U02_results.md`. 120 fan-chart figures: `results/figures/u02_fancharts/`. No `benchmarks.csv` rows (interval-calibration metrics, a different family from point-forecast accuracy, same exclusion precedent as the original FC-03/U-01). Cross-ref D-40 (U-01, not used as precedent), D-53/D-57 (source models), D-58/D-59 (B-14/B-15's tuning conclusions, unchanged by this UQ layer), I-02/D-61 (this experiment's companion).
**Addendum — TFTQuantile: giving TFT a native quantile head, closing its one blank-gap limitation.** Prompted by direct user inspection of the fan charts (not automated testing) surfacing that TFT was the one model with a genuine total-blank failure mode (no raw interval to fall back on when calibration was unavailable for a bin — e.g. Tower 9's 2021 anchor in its last ~90 days). New `TFTQuantile` class in `forecasting_dl.py`: architecturally identical to the point `TFT` class (its VSN/GRN/attention body is agnostic to output head width), with exactly the two edits the codebase's own `LSTMQuantile`-vs-`LSTMSeq2Seq` precedent already established (output head widened to `nn.Linear(d_model, nq)`; final `.squeeze(-1)` dropped, retaining the quantile axis). **Not wired into `build_model`** — constructed directly, mirroring `LSTMQuantile`'s own precedent. `pinball_loss`/`train_quantile`/`predict_quantile` were already generic over any 3-arg `(enc,dec,static)` model emitting `(B,H,Q)` (verified by reading their implementations before touching anything) — reused verbatim, with `train_quantile` gaining the same optional `weight_decay`/`val_data`/`patience` early-stopping parameters `train_model` already has (D-45), defaults preserving `LSTMQuantile`'s one existing caller's (U-01's) exact behavior. New `dl_rollout_quantile` in `recursive_rollout.py` mirrors `tree_rollout_quantile`'s already-verified median-feedback contract. **Backward compatibility verified, not assumed**: exhaustive grep confirmed the only TFT callers in the repo are `B03b_tft.ipynb`, `B13_tft_tabpfn.ipynb`, and `i02_multi_anchor_tower.py` (confirmed `B10_daily_improvements.ipynb` does not use TFT at all, despite being named in the request); `git diff` confirmed the existing `TFT` class, `build_model`, `LSTMQuantile`, `pinball_loss`, `predict_quantile`, and `dl_rollout` are completely untouched, line-by-line. A numeric before/after comparison of I-02's TFT VSN weights showed different values between runs — traced to a **pre-existing** property of this codebase (TFT's initial weights were never seeded; `torch.manual_seed` is only called inside `train_model`/`train_quantile`, after model construction) rather than a regression from this change. **Result:** TFT's raw PICP (0.83 T4, 0.85 T9, 0.76 T2) puts it alongside SARIMAX/TabPFN in the "already reasonably calibrated without conformal adjustment" group, not the overconfident-tree group — since the tree models here also use a from-scratch quantile mechanism, this points to the *training objective* (pinball loss vs the tree-specific quantile tricks used here), not architecture complexity, as what drives calibration quality. TFT's calibrated pinball is nonetheless the *worst* of any model at Tower 4 (11.21) — giving it a quantile head fixed its coverage gap but not its sharpness, an honestly-reported finding consistent with TFT's standing status in this project (solid, not class-leading, D-45/D-57). Cross-ref D-45 (source of the early-stopping recipe reused here), D-57 (B-13, TFT's original rollout introduction).

---

### D-63 — 2026-07-07 — U-03: does U-02's conformal calibration hold up under distribution shift?
**Decision:** Direct follow-up to U-02 (D-62), prompted by the pivot from "long-horizon forecasting" to **scenario simulation** for the Phase-07 digital-shadow work (2x-livestock, CMIP6-2050-climate scenarios). A codebase review (via Explore agent) confirmed U-02's leave-one-anchor-out conformal calibration is a vanilla split-conformal symmetric additive margin, and that **no exchangeability discussion exists anywhere in `U02_results.md` or D-62** — a real, previously undocumented gap, since split-conformal's coverage guarantee only holds if the test point is exchangeable with the calibration data, an assumption a genuine future scenario violates by construction. **User-confirmed scope (via AskUserQuestion): test calibration specifically under distribution shift**, not alternative calibration methods or closing Tower 2's calibration gap (both explicitly deferred). Since no real FCH4 exists for a hypothetical scenario, "coverage under a genuine future scenario" cannot be empirically validated — only diagnosed — so the experiment was split into two honestly distinct parts: **Part A** (real ground truth, cheap re-analysis) and **Part B** (no ground truth, explicit sensitivity diagnostic). Both parts reuse `recursive_rollout.py`'s existing quantile-rollout/calibration machinery completely unmodified.
**Part A result: no clear evidence, within the range of historical shift actually observed (2018–2022), that conformal calibration degrades under distribution shift.** For each of the 5 anchors × 3 towers, a shift score (mean |z-score| of `fx_lsu_dens`/`fx_TA_mean`/`fx_PRECIP_sum`/`fx_SWIN_mean` against the other 4 anchors' pooled distribution — the exact reference set U-02's own calibration used) was correlated against U-02's already-computed conformal PICP. Correlation was essentially null at Tower 4 (−0.166, n=5) and *positive* at Tower 9 (+0.562, n=4 — the opposite direction from the feared degradation, though not statistically robust at this sample size). Even the single most-shifted real anchor (2020, Tower 4, shift score 1.99, driven by an anomalous cold year) showed conformal PICP (0.895) in line with the T4 average. **Caveat stated plainly: this only clears the bar of "ordinary year-to-year variation" — the shift magnitudes tested are far smaller than a genuine 2x-livestock or 2050-climate scenario, and this result cannot be extrapolated to claim calibration survives a real future scenario shift.**
**Part B result: a clean, structural split by model family in response to a synthetic `fx_lsu_dens` (livestock density) perturbation swept from 1.0× (real) to 3.0× on Tower 4/anchor 2021, crossing Tower 4's training-seen maximum at ≈1.29×.** RF/XGB/LightGBM show a strongly decelerating, near-flat response (mean predicted FCH4 rises only 6–8% across the full 3× sweep) — the tree-extrapolation-ceiling signature predicted by this session's literature review, confirmed empirically on this project's own models for the first time: RF/XGB/LightGBM split on leaf boundaries fit to the training range and cannot extrapolate a trend past it. **SARIMAX shows the opposite failure mode**: a near-perfectly linear, unbounded response (+122%, ≈15.8 nmol per 0.5× step) — by 3.0× its mean prediction (76.9) is more than double Tower 4's real historical mean (33.5), extrapolated from a coefficient with no structural guarantee of validity that far beyond its fitted range. TFT sits in between (+29.9%), a real but also-decelerating response, plausibly via a different (non-leaf-based) saturation mechanism. **Every "calibrated" width recorded in this test is an explicitly-flagged mechanical application of U-02's real, frozen anchor-2021 margins to a deliberately non-exchangeable input — not a validated interval.**
**Recommendation for Phase 07:** do not apply U-02's conformal margins to genuine scenario predictions as validated intervals — they remain sound for held-out historical-regime evaluation only. The extrapolation-ceiling problem is now confirmed as a real, structural issue for this project's specific models (not merely a literature-review concern), directly motivating the detrend-and-residual or hybrid process+ML approach already flagged in the deep-research prompt sent this session. Any future scenario-uncertainty treatment needs either a better-extrapolating model/architecture or an explicit, separately-justified interval-widening rule for out-of-range scenario inputs — not a bare reuse of U-02's historical margins. Full write-up: `notebooks/06_interpretability_uq/U03_uncertainty_shift_robustness.ipynb`, `U03_results.md`. New figures: `results/figures/u03_fancharts/` (8 highlighted-anchor fan charts + 1 response-curve plot). No `benchmarks.csv` rows (diagnostic/robustness analysis, same exclusion precedent as U-01/U-02). Cross-ref D-62 (U-02, the calibration machinery this tests), D-61 (I-02, `fx_lsu_dens`'s dominant-driver finding that motivates using it as the scenario knob here).
**Addendum — Part B expanded from 1 anchor/tower to 5 anchors × 2 towers, prompted by the user directly questioning the single-anchor scope.** The original Part B pilot (Tower 4, anchor 2021 only) was exactly the narrow scope this project has repeatedly found unreliable (B-09/B-10/B-12's single-anchor reversals) — the user caught this and asked for the same diagnostic across both well-covered towers (T4, T9; T2 excluded — zero usable conformal margins anywhere in `u02_summary.csv`, same reasoning as Part A's own T2 exclusion) and the full 2018–2022 anchor range. New `u03_extrapolation_stress_test_multi.py` reuses the pilot's fit/rollout logic verbatim (verified: the 2021/Tower-4 slice of the 1,500-row extended sweep reproduced RF/XGB/LightGBM/SARIMAX bit-for-bit against the pilot's saved output, max abs diff = 0.0 across 150 rows; only TFT differed, up to 28 nmol — confirmed as the **already-documented** pre-existing non-determinism from this same D-62 addendum, TFT's weights are never seeded before construction, not a new bug). **The pilot's magnitudes were understated, sometimes by 3×**: mean %-change (1.0×→3.0× `fx_lsu_dens`) across all 10 anchor/tower cases is RF +23.4% (pilot: +8.3%), SARIMAX +150.3% (pilot: +122.1%, but ranging 58.6–379.8% across cases — highly anchor-dependent). **What survives the expansion: SARIMAX extrapolates further than every tree model and further than TFT in all 10/10 cases — the one fully robust ordering claim.** **What does not survive: TFT is not reliably "in between" trees and SARIMAX** — it exceeds the tree-model maximum in only 4/10 cases; TFT and the trees are better described as broadly comparable (both meaningfully more muted than SARIMAX), not a clean three-tier ordering. Revised recommendation: SARIMAX is the highest-risk model for naive scenario extrapolation, since its degree of overshoot is itself highly anchor-dependent (58–380%), meaning even the choice of which historical period's coefficients to use materially changes a scenario projection's scale. New files: `u03_extrapolation_stress_test_multi.py`, `u03_response_curve_plot_multi.py`, `results/u03_extrapolation_stress_test_multi.csv`, `results/u03_pct_change_summary.csv`, 10 per-anchor-tower response-curve plots + 1 robustness-summary scatter plot in `results/figures/u03_fancharts/`.
**Second addendum — Part B expanded a second time, from 5 models/2 towers to the genuinely complete 8 models × 3 towers (T2 included), prompted by the user catching that the first expansion had repeated the same narrowing mistake at smaller scale** ("I don't think U03 has been completed for all models, towers, and years?", then "tower 2 should also be included... always include tower 2"). `u03_extrapolation_stress_test_multi.py` rewritten to add TabPFN (via `rr.tabpfn_forecast(..., quantiles=...)`, `fx_lsu_dens` perturbed in `future_covariates`, local-mode inference reusing the already-provisioned `TABPFN_TOKEN`) and both ensembles (post-hoc weighted combination of the already-perturbed RF/XGB/LightGBM/SARIMAX outputs, zero extra fitting cost, identical definition to U-02's). 3,600 rows (5 anchors × 3 towers × 5 multipliers × 6 bins × 8 models), zero TabPFN skips, verified via a reduced-scope smoke test before committing to the full ~35-minute run. **Genuine, non-bug finding surfaced by including Tower 2: its `fx_lsu_dens` is exactly 0.0 for the entire 365-day rollout window in 4 of 5 anchors (2019–2022)** — multiplying zero by any factor is still zero, so the livestock-extrapolation diagnostic is structurally degenerate there; only anchor 2018 has any real (small) signal. Tower 2 is reported in the raw data but excluded from the headline comparison table and robustness plot, since averaging in four by-construction-zero rows would misleadingly drag every model's summary toward zero for a reason unrelated to model behavior. **New headline finding: the production-recommended B-10 ensemble is not immune to the extrapolation problem.** Despite being 75% tree-weighted, both ensembles show +49–50% mean overshoot (1.0×→3.0× `fx_lsu_dens`, Towers 4/9), more than double the pure tree-mean (~23%) — SARIMAX's 25% weight is not diluted away by the other three members' flat response. **TabPFN is the least predictable model tested** (range −4.9% to +90.1%, the only model that sometimes decreases under more livestock) — a zero-shot foundation model showing erratic direction-of-effect under covariate perturbation, itself a notable result. **SARIMAX's robust-ordering claim strengthens further**: it is now confirmed as the maximum of all 8 models (not just the 5 checked previously) in 10/10 T4/T9 cases. Revised Phase-07 recommendation: do not reuse B-10's ensemble unmodified for scenario extrapolation without addressing its SARIMAX-inherited risk (reweight for scenario runs, or drop SARIMAX from the scenario-specific configuration while keeping it for historical-regime forecasting where it remains competitive); any Tower-2 scenario work needs an explicit livestock-baseline assumption independent of the 2019–2022 fitting window, which records essentially no grazing there. New/updated files: `u03_extrapolation_stress_test_multi.py` (rewritten), `u03_response_curve_plot_multi.py` (rewritten, 8-model palette), `results/u03_extrapolation_stress_test_multi.csv` (3,600 rows), `results/u03_pct_change_summary.csv`, 15 per-anchor-tower response-curve plots (all 3 towers) + 1 robustness-summary scatter plot (T4/T9 only) in `results/figures/u03_fancharts/`.

---

### D-64 — 2026-07-07 — S-01: first Phase 07 scenario-simulation worked example (level-residual hybrid)
**Decision:** First bounded worked example for Phase 07 (the dissertation's actual novel deliverable, "PLANNED, not started" since project inception), proving the full scenario-simulation mechanism end-to-end. Builds directly on D-46 (candidate CMIP6 dataset scoped) and D-52 (dataset arrival confirmed North-Wyke-matched; missing-driver strategy already decided: historical-day resampling via `rr.doy_climatology()`, not a raw Copernicus pull) and on U-03/D-63 (already answered D-46's extrapolation-range-check requirement directly on the candidate models — **user-confirmed B-08 is superseded for Phase 07's purposes** by this finding, though it remains separately available for the point-forecast track). A deep-research literature pass this session (user-provided) independently confirmed North Wyke coverage and the persistent 4-variable CMIP6 gap, and recommended a **level–residual hybrid** architecture: an explicit trend/level model carries the extrapolation (climate + livestock axes), a tree ensemble corrects only the residual (near-stationary, sidesteps the tree-extrapolation-ceiling U-03 found). **User-confirmed: parametric trend (Ridge) for this pass, not a mechanistic process model** — SPACSYS (already validated at North Wyke, Wu et al. 2016, including an animal-growth module) is logged as a stronger future research direction, not attempted here. Also adopted from the deep-research pass: monotonic constraints on livestock density in XGB/LightGBM, and **dropping `fx_USTAR_mean`/`fx_SHF_mean` entirely** (true EC-tower turbulence quantities with no climate-scenario-product source at all, unlike WS/VPD/soil moisture which at least have an effort-heavy external source).
**A real, previously-unflagged unit mismatch was caught and fixed during this session's data verification** (via Explore agent, before this plan was written): the CMIP6 dataset's `RAD` column is solar radiation in MJ m⁻² day⁻¹ (a daily-integrated total), while this project's `fx_SWIN_mean` is W/m² (a daily mean) — a genuine unit conversion (×1e6/86400 ≈ 11.574) is required, not a direct join; D-52 had confirmed the variable's presence but not this unit distinction. Verified via sanity check: a summer-peak RAD (~19.4 MJ/m²/day) converts to ~224 W/m², a winter value (~2.3) to ~27 W/m², both physically plausible for UK solar radiation.
**Method:** `src/features/build_scenario_drivers.py` (new) loads the North Wyke CMIP6 `.dat` files (confirmed tab-delimited, `YEAR JDAY MIN MAX RAIN RAD`), builds an ensemble-mean day-of-year climatology (all 5 GCMs × 100 realizations, 2041–2060 window — "the 2050s," ~10,000 samples per day-of-year, 31s load time for 500 files), derives `fx_TA_mean`/`fx_SWIN_mean`, and historical-day-resamples every other driver via `rr.doy_climatology()` (reused unmodified). `src/models/scenario_hybrid.py` (new) fits a Ridge trend model **once** on the full pooled real historical record (8,772 rows, T2+T4+T9 — deliberately a single fit, not per-anchor, the specific fix for U-03's SARIMAX-instability finding) on `fx_TA_mean`/`fx_lsu_dens`/`fx_DOY_sin/cos` + tower dummies (scaled coefficient for `fx_lsu_dens` = 27.0, by far the largest — reconfirms I-02's livestock-dominance finding from the trend model itself), then RF/XGB/LightGBM (B-10's exact hyperparameters, no new HPO) on the residual, with a monotonic constraint on `fx_lsu_dens` for XGB/LightGBM — **verified directly via a synthetic 0–10 sweep**, not assumed (XGB/LightGBM strictly non-decreasing, min step 0.0000; RF, with no native sklearn monotonic-constraint support, is non-monotonic as expected). Also implemented `dissimilarity_index()`, a lightweight from-scratch Python analogue of Meyer & Pebesma (2021)'s Area of Applicability (nearest-neighbour distance to training data in scaled feature space, Tukey-IQR-fence threshold derived from the training data's own leave-one-out distances) — explicitly labeled as "in the spirit of," not a port of the R `CAST` package. Frozen model artifacts (trend + 3 residual models + imputer) persisted via `joblib` for the first time in this project (closes D-46 requirement 1).
**Scenario tested:** SSP2-4.5, ensemble-mean, 2041–2060, Tower 4, livestock multipliers {1.0×, 2.0×, 3.0×} on the day-of-year climatology of `fx_lsu_dens` (a naive multiplier, explicitly flagged as a known simplification per the literature's own caution against naive column-scaling — user-confirmed scope for this first pass).
**Result — baseline sanity check passes**: the 1.0× scenario's predicted annual mean (29.35) matches Tower 4's real historical mean (29.95) closely, confirming pipeline coherence before trusting the perturbed cases. **The hybrid design measurably fixes U-03's flattening finding**: the same nominal 3× livestock sweep that gave RF/XGB/LightGBM only +21–23% in U-03's raw-tree-only test gives **+138.2%** here (29.35→69.90), because the trend component now carries the extrapolation instead of the trees clipping it. **A genuine, non-obvious finding from the AOA check: "2× livestock" under this construction is NOT automatically out-of-distribution** — the 2.0× scenario's `fx_lsu_dens` peaks at 4.95, inside Tower 4's own training envelope (max 5.65), because scaling a *smoothed climatology* caps peaks well below what a single real raw grazing day can reach (0% AOA-flagged); **only the 3.0× scenario genuinely exceeds the envelope** (max 7.43, 5.5% AOA-flagged). This is a real methodological finding, not a null result to bury: a "2× livestock" scenario built by scaling a smoothed climatology is meaningfully milder than one built by scaling raw daily values (as U-03 tested) — the two constructions are not interchangeable, and this changes whether a nominally-identical "2×" label is even out-of-distribution. **The climate axis is not the extrapolation risk in this scenario**: SSP2-4.5's 2050s daily-mean temperatures (5.56–17.29°C) sit comfortably inside Tower 4's real historical range (−4.71 to 24.05°C) — day-to-day weather variability exceeds the multi-decade climate-mean shift, exactly as D-46 anticipated. Seasonal pattern is physically sensible (JJA peak, consistent with known CH₄ seasonality).
**Recommendation:** treat S-01 as the proof-of-mechanism, not the final Phase-07 output. Explicit caveats carried into every downstream use: parametric-not-mechanistic trend (SPACSYS logged as future work); USTAR/SHF dropped (different feature set than B-10's production model); 9 of ~11 drivers are historical-day-resampled, not real future weather; livestock scenario is a naive multiplier; U-02/U-03's conformal intervals are NOT attached to this output (per U-03's own standing recommendation, only ever valid for in-AOA points — re-check per output); the AOA check's own limitation (full-feature-space dilution, cross-checked against the simpler "scenario max vs training max" comparison) is itself worth carrying forward as a lesson for any future OOD-flagging attempt. Full write-up: `notebooks/07_scenario_analysis/S01_first_scenario.ipynb`, `s01_results.md`. New files: `src/features/build_scenario_drivers.py`, `src/models/scenario_hybrid.py`, `results/s01_scenario_summary.csv`, `results/models/s01_*.joblib`, `results/figures/s01_scenario_comparison.png`. No `benchmarks.csv` rows (scenario-simulation output, different metric family, same exclusion precedent as U-01/U-02/U-03). Cross-ref D-46, D-52 (data/design foundations), D-63 (extrapolation-ceiling finding this directly addresses), D-61 (I-02, the livestock-dominance finding reconfirmed here from the trend model itself).
**Addendum — extended to all 3 towers + per-model breakdown, prompted by direct user request** ("ensure that you output figures... do so for all towers and models"). No re-fitting needed — one pooled hybrid model (trend + 3 residual trees) already serves all towers via the tower dummies, so the extension only re-ran the scenario/prediction step per tower (T2, T9 added alongside T4). **Tower 2 is genuinely different, confirmed as a real finding not a pipeline artifact**: its own historical `fx_lsu_dens` maximum (0.71) is ~7-8x smaller than T4/T9's (4.99/5.65) — even a 3x scenario multiplier only reaches 2.14, still below T4/T9's own 1x baseline, and the AOA check never flags Tower 2 as out-of-envelope at any multiplier tested (T4/T9 both get flagged at 3x, 5.5%/6.0%) — directly consistent with U-03's own finding that T2's `fx_lsu_dens` is exactly zero throughout the rollout window in 4/5 anchor years, now reconfirmed from the scenario-construction side. T2's baseline-reconstruction sanity check also shows the largest gap (predicted 23.31 vs real 19.39, a documented consequence of T2's known data sparsity, not a new problem). **Per-model breakdown surfaced a genuine, verified-not-assumed architectural finding**: a full monotonic sweep (0-10, not just the min-step check) shows XGB and LightGBM's residual prediction is **completely flat** across the entire `fx_lsu_dens` range for a representative row (e.g. XGB: 8.077 at all 21 sampled points) — confirmed in the actual scenario runs too (`residual_XGB`/`residual_LightGBM` near-identical across all 3 multipliers at every tower). This is architecturally coherent given B-10's exact shallow-tree hyperparameters (XGB `max_depth=2`, LightGBM `num_leaves=7`) combined with the monotonic constraint and the trend model already absorbing the primary livestock signal — a shallow constrained tree has little residual variance left to split on. **RF (no native monotonic-constraint support) is the only residual model showing real sensitivity to livestock density** (e.g. T4: -4.36 -> -8.36 -> -9.68 across 1x/2x/3x). Net effect: for two of the three tree models, essentially 100% of the livestock-driven scenario response flows through the trend component, not the residual — the level-residual split isn't just designed to divide responsibility this way, it demonstrably does. New figures (4, replacing the single-tower `s01_scenario_comparison.png`): `results/figures/s01_scenario_comparison_all_towers.png` (grouped bar, 3 towers x 3 multipliers), `s01_per_model_breakdown.png` (trend vs. RF/XGB/LightGBM, one panel per tower), `s01_seasonal_all_towers.png` (seasonal small multiples), `s01_aoa_ensemble_disagreement.png` (AOA flag % + ensemble disagreement by tower/multiplier). `results/s01_scenario_summary.csv` now 9 rows (3 towers x 3 multipliers) with per-model residual columns added.

---

### D-65 — 2026-07-07 — expand `bin_metrics` with RMSE/WAPE/Correlation, rerun B-10 + B-13 into one table
**Decision:** Prompted by two user questions in the same conversation — is correlation commonly used in time-series forecasting evaluation, and does B-10→B-13 track MAE/RMSE. Direct inspection confirmed `bin_metrics()` — the single shared evaluation function every B-09→B-15 experiment (and I-02/U-02/U-03/S-01 by extension) calls — only computed R²/MAE/MASE, narrower than the earlier B01→B07 phase's full roster (D-44b). Added `correlation()` (Pearson r) to `src/evaluation/metrics.py` — genuinely new to the project (a repo-wide grep found zero prior uses of any correlation function) but not a novel idea here: it formalizes the same informal check already used once to diagnose the original unregularized TFT (D-45, r=0.27 despite deeply negative R²). Extended `bin_metrics()` to compute RMSE/WAPE/Correlation alongside the existing R²/MAE/MASE — confirmed purely additive-safe via a full repo grep (all 15+ existing call sites across the B-09→B-15 sequence access the returned DataFrame by column name only, none positionally).
**A discovery made while planning the rerun**: neither B-10's nor B-13's original multi-anchor script was ever committed (`git log --all --diff-filter=A` found nothing) — both were "ad-hoc, not committed" per this project's own stated precedent going back to B-09; only their output CSVs survived. Reconstructed both from their fully-documented methodology (hyperparameters read directly from the committed single-anchor notebooks `B10_daily_improvements.ipynb`/`B13_tft_tabpfn.ipynb`) in a new, **committed** script (`notebooks/05_benchmarking/b10_b13_rerun_multi_anchor.py`) — closing that reproducibility gap rather than repeating it, matching the newer B-14/B-15 precedent of committing multi-anchor scripts. Preserved a subtle detail that would otherwise have silently broken reproduction: B-13's TFT block uses a different `y_true` source (`fdl.tower_series`'s resampled `Y`) than the trees/SARIMAX/TabPFN blocks (`forecast_daily_v2.csv`'s `y_observed` directly) — kept exactly as each original used it.
**Verification, before trusting any new number: reproduction confirmed exact across all 5 anchors for 7 of 8 models.** RF, XGB, LightGBM, SARIMAX, both ensembles, and TabPFN all reproduce R²/MAE/MASE **bit-for-bit** against the existing published CSVs. **TFT differs**, exactly as expected from the already-documented pre-existing non-determinism (D-62 addendum: TFT's initial weights are never seeded before construction) — median |R² difference| a modest 0.13, with one extreme outlier concentrated in the 3-point 1-7 day bin, consistent with this project's own established "1-7 day bin is a small-sample artifact" finding (D-53), not a new instability.
**Result — the singular table (Tower 4, 5-anchor n-weighted-per-anchor-then-mean, matching the established aggregation convention):**

| Model | R² | RMSE | MAE | MASE | WAPE | Correlation |
|---|---|---|---|---|---|---|
| RF | −0.067 | 51.54 | 34.08 | 1.024 | 1.028 | 0.402 |
| XGB | 0.003 | 51.23 | 33.08 | 0.968 | 0.964 | 0.372 |
| LightGBM | −0.014 | 51.37 | 33.25 | 0.978 | 0.978 | 0.384 |
| SARIMAX | −0.039 | 52.39 | 35.23 | 1.038 | 1.047 | 0.379 |
| **Ensemble_unweighted** | **0.012** | **50.96** | 33.18 | 0.975 | 0.977 | 0.396 |
| Ensemble_MASEweighted | 0.011 | 50.96 | 33.17 | 0.975 | 0.977 | 0.396 |
| TFT (this rerun's draw) | −0.568 | 54.60 | 33.67 | 1.045 | 1.050 | 0.329 |
| TabPFN | −0.006 | 54.19 | 30.46 | **0.862** | 0.860 | 0.391 |

**New finding, invisible with the old metric set: TabPFN's best-in-sequence MASE (0.862) does not extend to RMSE** — its RMSE (54.19) is second-worst, close to TFT's, clearly worse than every tree model (~51-52). MASE/MAE weight every error linearly; RMSE squares errors and is dominated by TabPFN's worst individual misses. **TabPFN's real strength is consistency relative to persistence, not small worst-case errors** — a materially more complete characterization than "best MASE" alone implied, and a concrete reason to track more than one error metric per model going forward. **Correlation is uniformly weak-to-moderate (0.33-0.40) across every model, including the standing recommendation** (Ensemble_unweighted highest at 0.396, r²≈0.16) — a humbling, consistent-with-everything-else-this-project-has-found context: even the best model isn't tracking the true pattern strongly by this measure either. RMSE mostly agrees with MASE on ranking (Ensemble best on both) — the TabPFN divergence is the one place the new metrics change the picture, not a wholesale reordering.
**Explicit caveat on the TFT row:** it is a real, honestly-computed result from this specific rerun, not fabricated, but should not be read as superseding the originally-published −0.237 mean R² (D-57) — both are correct draws of an unseeded model; the correct takeaway remains D-62's own: TFT's point estimate carries real run-to-run uncertainty the other 7 models don't.
**Recommendation:** no change to the standing production recommendation (B-10's Ensemble_unweighted remains best on R² and now also RMSE). Full write-up: `notebooks/05_benchmarking/b10_b13_metrics_rerun.md`. New files: `src/evaluation/metrics.py` (+`correlation()`), `src/models/recursive_rollout.py` (`bin_metrics` extended, additive only), `notebooks/05_benchmarking/b10_b13_rerun_multi_anchor.py`, `results/b10_b13_rerun_summary.csv` (240 rows), `results/b10_b13_rerun_table.csv`. No changes to any existing B-09/B-11/B-12/B-14/B-15 file or CSV — scoped to B-10 and B-13 only, as requested. No `benchmarks.csv` rows (a metrics backfill + verification exercise on already-logged results, same precedent as D-44b). Cross-ref D-44b (the earlier metrics backfill this extends the spirit of), D-45 (the informal correlation check this formalizes), D-54/D-57 (B-10/B-13's original numbers, reproduced here), D-62 (TFT's documented non-determinism).

**Addendum — 2026-07-07 — extended to all 3 towers (T2/T4/T9).** The above was Tower-4-only. The
user flagged this as the same narrow-scope pattern already seen twice this session (U-03, S-01) and
asked that "always run for everything" become a standing project convention — added to `CLAUDE.md`
as a new "Full coverage by default" rule (default every experiment's *final* result to all 3 towers
and the full relevant model roster; single-tower/single-anchor runs are fine as smoke tests but must
not be presented as the finished result). `b10_b13_rerun_multi_anchor.py` was extended to loop
`TOWERS=[2,4,9]`, reusing the pooled-fit-once-per-anchor pattern already established in U-02/U-03/S-01
(RF/XGB/LightGBM/TFT fit once per anchor on pooled T2+T4+T9 data, rolled out separately per tower;
SARIMAX/TabPFN refit per tower as always). Output grew from 240 to 720 rows (`results/b10_b13_rerun_summary.csv`).
Tower 4's numbers reproduce exactly (unchanged from the original run, confirmed by diff); Tower 2/9
have no prior multi-anchor CSV to check bit-for-bit (originals were T4-only), so trust rests on
reusing the already-verified fit/rollout code path — Tower 2's near-total-NaN pattern (only the 2018
anchor has real coverage) matches the already-documented U-02/U-03/S-01 finding, not a new anomaly.
**All-tower pooled headline (per-anchor n-weighted mean across bins and all towers present, then mean
across anchors) is substantially worse than the T4-only numbers on R² but comparable-to-better on
MASE**: Ensemble_unweighted R² 0.012→−0.165, MASE 0.975→0.918. Every model's R² drops once T9 (a
consistently harder tower across this whole project) and T2 (data-scarce, degenerate outside 2018)
are included — model ranking is essentially unchanged (Ensemble/TabPFN still best on MASE, SARIMAX/TFT
still worst), confirming this is a scope-of-evaluation effect, not a different conclusion about which
model is best. A new tower×year×model breakdown table was added to `b10_b13_metrics_rerun.md` and
`results/b10_b13_rerun_table_by_tower_year.csv` (true nested `MultiIndex`: tower as parent column,
anchor_year as parent row). New files: `results/b10_b13_rerun_table_all_towers.csv`,
`results/b10_b13_rerun_table_by_tower_year.csv`. `BEST_RESULTS.md` Section 3 and `CONTEXT.md` updated
to cite the all-tower headline going forward; the original Tower-4-only table is kept in
`b10_b13_metrics_rerun.md` for continuity with existing citations, not removed.

**Second addendum — 2026-07-08 — secondary metric vs. gap-filled target.** User's own idea, from a
live discussion: also score every chain against `y_gapfilled` (dense/continuous) alongside the
primary `y_observed`-target metric, since real observations are sparse — especially Tower 2 (816
real data-points summed across all models/anchors/bins vs. 14,600 possible; T4 10,560; T9 7,200).
**This is an explicit, bounded departure from D-36/D-37's "train on gap-filled, evaluate on
observed" convention for one secondary/exploratory check, not a redefinition of it** — user-confirmed
scope. Real circularity risk, stated up front in every downstream table: `y_gapfilled` seeds
`history_init` (the pre-anchor AR memory every rollout builds forward from) and is itself a pooled
RFm gap-filler's output trained on met/soil features overlapping RF/XGB/LightGBM's own forecast
features, so agreement can partly reflect "forecaster resembles gap-filler," not real skill.
`bin_metrics()` needed no change (already generic over `y_true`) — `b10_b13_rerun_multi_anchor.py`
was extended to call it a second time per already-computed chain, plus a per-bin `real_frac` (fraction
of that bin's days that were also real-observed) so the caveat is backed by numbers. Full 3-tower ×
5-anchor × 8-model coverage (user-confirmed), as an **extension only** — verified the 4 existing
observed-target output files are unaffected (non-TFT rows of `b10_b13_rerun_summary.csv` bit-for-bit
identical after rerun; TFT rows differ only from its already-documented unseeded non-determinism,
consistent with the first addendum's own TFT reproduction check; the 3 derived observed-target tables
were never touched by the script and are untouched files, not regenerated).
**Findings**: (1) Tower 2's coverage problem is fully unlocked (14,600/14,600 vs. 816/14,600), and its
gap-filled-target numbers, while mostly negative R², are broadly consistent with — not contradicting
— its already-documented harder/data-scarce status, not degenerate the way its observed-target numbers
were. (2) RMSE/MAE/WAPE/MASE all *improve* under the gap-filled target (e.g. Ensemble_unweighted MASE
0.918→0.751) but R² gets *worse* (0.012→−0.189 at Tower 4-equivalent all-tower scale) — mechanistic,
not contradictory: `y_gapfilled` is smoother (lower variance) than `y_observed`, so the same
(smaller) absolute error is a larger share of a smaller total variance, which is exactly what R²
penalizes. (3) Model ranking mostly holds — Ensemble_unweighted stays at or near the top on R² in
both metrics (reinforcing the standing recommendation), SARIMAX/TFT remain the two worst in both —
**except TabPFN, which drops from best-R² (observed-target, 1st of 8) to 6th of 8 (gap-filled-target)**,
the one real ranking disagreement, flagged as a reason not to over-read TabPFN's observed-target R²
advantage as fully robust. New files: `results/b10_b13_rerun_summary_vs_gapfilled.csv`,
`results/b10_b13_rerun_table_vs_gapfilled_all_towers.csv`,
`results/b10_b13_rerun_table_vs_gapfilled_by_tower_year.csv`. New section in
`b10_b13_metrics_rerun.md` ("Secondary metric: scored against gap-filled target"). No new D-number
(secondary/exploratory, bounded to B-10/B-13 rerun only per user's explicit scope choice — no
retrofit into U-02/U-03/S-01/I-02 this round). `BEST_RESULTS.md` gets a one-line pointer only (not a
new quick-reference row) — this does not supersede or compete with the all-tower observed-target
headline above.

**Third addendum — 2026-07-08 — DLinear/LSTM model-roster extension (closes the `b10_chains` figure
gap).** DLinear and LSTM had rollout chain *figures* extended to all 3 towers × 5 anchors during
B-13's Part A (`results/figures/b10_chains/T*_anchor*_{DLinear,LSTM}.png`) but that extension never
saved a `bin_metrics()` summary — there was no full-coverage evaluation table for these two models
with the D-65 metric set. Reconstructed from `B09_recursive_rollout.ipynb`'s exact Section-4 recipe
(track B, `L=28/H=14`, pooled T2+T4+T9 training with **no** validation carve-out — deliberately not
TFT's regularized recipe — per-tower rollout via the already-generic `rr.dl_rollout`, `y_true` from
`fdl.tower_series(...)["Y"]`). **Verification produced a genuinely new, sharper finding that
refines D-62's addendum**: LSTM reproduces bit-for-bit exactly against the published
`b09_multi_anchor_summary.csv` in all 5 anchors; DLinear matches exactly in 4 of 5 (only the very
first anchor processed in the run, 2018, differs). Root cause: `fdl.train_model()` calls
`torch.manual_seed()` **after** `fdl.build_model()` already constructed and randomly initialized the
model — so only whichever model is built *first in the whole process, before any prior
`torch.manual_seed()` call*, gets non-deterministic weights; every later model/anchor lands on fully
deterministic initialization because an earlier `train_model()` call already fixed the global
PyTorch RNG. This is almost certainly the same mechanism behind TFT's own documented
non-determinism (D-62) — TFT is typically the only/first torch model built in its script — and
implies seeding once at the very top of a script would make every model in it exactly reproducible
(noted as a real, easy future fix; not applied, to avoid silently changing already-published
numbers without being asked). **Result: DLinear/LSTM are drastically worse than every one of the 8
models in the primary tables** (all-tower R²: DLinear −5.057, LSTM −1.357, vs. Ensemble_unweighted's
−0.165 best) — directly confirming, not contradicting, D-53/D-54's finding that these two were
correctly excluded from B-10's ensemble, now shown to hold at all 3 towers, not just Tower 4.
Reported in their own standalone tables (not merged into the primary 8-model tables), because doing
so surfaced a **separate, real discovery**: TFT's row in the currently-saved
`results/b10_b13_rerun_summary.csv` has independently drifted to a **third** random draw since the
primary all-tower/Tower-4-only tables above were built (current file's Tower-4 TFT R²=−0.232 vs.
−0.568 cited in those tables) — the same non-determinism, now manifesting as a staleness mismatch
between sibling artifacts rather than a single-run caveat. Flagged plainly, not resolved this pass
(would require deciding whether to regenerate the primary tables, a separate decision from this
one). New files: `notebooks/05_benchmarking/b10_b13_dl_extension.py`,
`results/b10_b13_dl_extension_summary.csv` (180 rows), `results/b10_b13_dl_extension_table_all_towers.csv`,
`_table_tower4.csv`, `_table_by_tower_year.csv`. New section in `b10_b13_metrics_rerun.md`
("Model-roster extension: DLinear + LSTM").

**Follow-up same day — DLinear/LSTM added to the gap-filled-target secondary-metric table too.**
`b10_b13_dl_extension.py` extended with the same secondary `y_gapfilled`-target `bin_metrics()` call
(+`real_frac`) already added to `b10_b13_rerun_multi_anchor.py`, so DLinear/LSTM now have rows in
the "Secondary metric" section's all-tower comparison table alongside the other 8 models. **Found a
further, concrete illustration of DLinear's instability while doing this**: rerunning the script
redrew the 2018 anchor's DLinear weights (the already-identified non-deterministic "cold start"
case) into a genuine numerical divergence this time — MAE up to ~7,545 nmol m⁻² s⁻¹ at Tower 9,
physically implausible for a series spanning roughly −1,559 to +6,161 — which, scored against the
low-variance `y_gapfilled` target, drove the pooled R² to −6576.7. Reported plainly with the
outlier explained rather than silently smoothed over, alongside the 4-anchor (excluding 2018)
alternative (R²=−7.035) for an interpretable number — still clearly the worst model either way.
LSTM's gap-filled numbers (R²=−3.769) are unaffected by this, consistent with its own confirmed
determinism. New files: `results/b10_b13_dl_extension_summary_vs_gapfilled.csv`,
`results/b10_b13_dl_extension_table_all_towers_vs_gapfilled.csv`.

---

### D-66 — 2026-07-09 — TabICLv2 added to the B-10/B-13 recursive-rollout sequence
**Decision:** Added **TabICLv2** (`tabicl` package, `TabICLForecaster` class) — a tabular foundation
model released Feb 2026 (ICML 2026), the first version of TabICL with regression support (v1 was
classification-only). Its own documentation describes it as "heavily inspired by TabPFN-TS," so the
integration mirrors `rr.tabpfn_forecast()`'s exact block structure: **per-tower, per-anchor, never
pooled** (the `predict_df` API has no static-covariate/pooling support, same limitation as TabPFN),
`hist_target = y_observed` (real data only, never `y_gapfilled`, same rationale as TabPFN),
covariates = the full `FX_B` daily `fx_`-prefixed column set (not the narrower 8-column `EXOG_B`
SARIMAX uses). New `rr.tabicl_forecast()` (`src/models/recursive_rollout.py`, additive only) and a
new standalone sibling script `notebooks/05_benchmarking/b10_b13_tabicl_extension.py` (mirrors
`b10_b13_dl_extension.py`'s minimal-file-output pattern) — no edits to `b10_b13_rerun_multi_anchor.py`,
`b10_b13_dl_extension.py`, or any historical B-13 notebook. Assigned a new decision number (not a
further D-65 addendum) because this introduces a genuinely new model to the project, unlike
DLinear/LSTM which closed an existing gap for models already in scope — matches the precedent that
TFT/TabPFN's own original addition got its own number (D-57).

**API contract verified empirically before writing the real integration** (documentation was
incomplete on this point): `TabICLForecaster.predict_df(context_df, future_df=...)` takes covariates
as plain extra columns on both dataframes — identical convention to `tabpfn_forecast()`'s own
`context_df`/`future_df` construction — but returns a DataFrame with a `(item_id, timestamp)`
MultiIndex (string timestamps) and **always** includes the default quantile grid `[0.1..0.9]`
alongside `target`, regardless of the `quantiles=` argument (unlike `tabpfn_forecast`, which only
computes quantiles when explicitly asked). **Local-only inference confirmed**: downloads a
Hugging Face Hub checkpoint once (cached thereafter — ~33s first call incl. download+GPU warmup,
~24s on a cached call at realistic scale), no token/API key required (unlike TabPFN's
`TABPFN_TOKEN`). Installed via `python -m pip install "tabicl[forecast]"` — caught and fixed a
`pip`/`python` environment mismatch in this repo along the way (bare `pip` resolved to a different
Python environment than `python` itself; `python -m pip` is now the safer invocation to use going
forward for any future dependency installs in this repo).

**Full 3-tower × 5-anchor sweep completed in under 30 seconds total** — dramatically cheaper than
every other model in this sequence (TabPFN/SARIMAX take tens of seconds to minutes per anchor;
TabICLv2 took ~1-4s per tower/anchor combination).

**A genuine, already-precedented limitation surfaced immediately, not a new bug**: Tower 9's 2018
and 2019 anchors have zero real `y_observed` values in their entire pre-anchor history (715/1080
pre-anchor rows, 0 non-null). Since `hist_target` never falls back to `y_gapfilled`, the model has
nothing real to condition on and produces a degenerate near-flat forecast (~0.0 for the whole
365-day window). TabPFN — using the identical `hist_target = y_observed` convention — shows the
exact same flat-zero pattern for these same two (tower, anchor) combinations, confirmed by direct
comparison against `results/b10_b13_rerun_chains.csv`. A shared, already-accepted limitation of the
"real-data-only, no gap-filled fallback" design at Tower 9's data-scarce early anchors, not specific
to this integration.

**Result — full coverage, both target metrics:** all-tower pooled R²=−1.929 (observed), −4.472
(gap-filled); RMSE 64.80/40.75; MASE 1.424/1.418; Correlation 0.256/0.205. **TabICLv2 underperforms
every model in the sequence except DLinear** — its observed R² sits between LSTM (−1.357) and
DLinear (−5.057), well behind TFT (−0.363, reconciled 2026-07-09 — see addendum below) and well
behind the standing recommendation (Ensemble_unweighted, −0.165). Given its
dramatically lower compute cost, this is treated as a genuinely informative negative result, not a
wasted effort — a zero-shot foundation model this cheap being this uncompetitive is itself worth
knowing, and is consistent with TabPFN's own more moderate but still sub-ensemble showing
(all-tower observed R²=−0.122) — one-shot foundation-model forecasters as a class are not yet
closing the gap to the tuned tree ensemble on this task.

**Scope, explicitly bounded (matches TabPFN's own staged-rollout precedent — its presence in
I-02/U-02/U-03 was each added later, as separate numbered work, not part of its original B-13
integration):** point-forecast integration into the B-10/B-13 rerun sequence only. No quantile
wiring this pass (flagged as a likely-easy follow-on given `TabICLForecaster`'s native quantile
support — richer out-of-the-box than TabPFN's original integration needed). No retrofit into
I-02/U-02/U-03. No new HPO — `TabICLForecaster()` used with its own defaults (zero-shot foundation
model, matching TabPFN's own "zero training/HPO" precedent).

New files: `notebooks/05_benchmarking/b10_b13_tabicl_extension.py`, `src/models/recursive_rollout.py`
(+`tabicl_forecast()`), `results/b10_b13_tabicl_extension_summary.csv` (90 rows),
`..._summary_vs_gapfilled.csv` (90 rows), `..._chains.csv` (5,475 rows), plus the derived all-tower/
per-tower/tower-year table CSVs (both target metrics). New section in `b10_b13_metrics_rerun.md`
("Model-roster extension: TabICLv2"); TabICLv2 also added as a new row to the existing "Secondary
metric" section's all-tower gap-filled comparison table. `BEST_RESULTS.md`/`CONTEXT.md` updated with
a one-line pointer, not promoted into any quick-reference table (does not beat the standing
recommendation). Cross-ref D-57 (TFT/TabPFN's original addition, the precedent for "new model = new
decision number"), D-65 (the metric set and secondary-metric convention this reuses), D-53/D-54
(DLinear's own precedent for a cheap-but-uncompetitive model still being worth reporting honestly).

**Same-day follow-up — figures + consolidated chains now a standing convention.** User-prompted:
`results/b10_b13_full_chains.csv` updated to include TabICLv2 (11 models, zero missing predictions).
New **committed** `notebooks/05_benchmarking/b10_b13_chain_plots.py` replaces the ad-hoc,
uncommitted plotting process every prior `results/figures/b10_chains/` figure came from (confirmed
lost via repo-wide search — same "ad-hoc, not committed" pattern this project has repeatedly had to
retroactively fix). Regenerated all 165 figures (11 models × 3 towers × 5 anchors); found and closed
two real coverage gaps in the process — **TFT had only 1 of 15 figures**, **TabICLv2 had none**
(TabPFN was initially assumed missing too but was already at full 15/15 coverage — corrected before
acting on the wrong assumption). Spot-checked regenerated deterministic-model figures (RF) against
the archived originals — identical underlying trajectory/data, pixel dimensions differ slightly
since the original lost script's exact `figsize` was never recorded. New standing rule added to
`CLAUDE.md` ("Forecasting work: always generate figures and keep the consolidated chains CSV
current") so this doesn't recur for future model additions.

**Second same-day follow-up — TFT staleness reconciled across `b10_b13_metrics_rerun.md`
(2026-07-09).** User-flagged: "I see its currently a little bit inconsistent." Root cause (already
partly documented above and in the D-65 second addendum's own note): TFT's row in
`results/b10_b13_rerun_summary.csv` redraws every time `b10_b13_rerun_multi_anchor.py` runs (its
initial weights are never seeded, D-62 addendum), and that script was rerun several times this
session for unrelated purposes (the gap-filled secondary metric, the raw-chains export) — each rerun
silently moved TFT further from the numbers already published in this document's tables, without any
of them being reconciled at the time. Fixed by recomputing every TFT-only row directly from the
currently-live `results/b10_b13_rerun_summary.csv` / `..._vs_gapfilled.csv` (every other model's row
in these files reconfirmed bit-for-bit unchanged first) and propagating into: the 5 underlying CSVs
(`b10_b13_rerun_table_all_towers.csv`, `..._table.csv` [Tower-4-only], `..._table_by_tower_year.csv`,
`..._table_vs_gapfilled_all_towers.csv`, `..._table_vs_gapfilled_by_tower_year.csv` — TFT rows only,
all other rows left untouched) and every markdown table/prose reference in
`b10_b13_metrics_rerun.md` (all-tower summary, Tower-4-only table, tower×year breakdown ×5 rows,
gap-filled all-tower table, gap-filled tower×year breakdown ×5 rows, the "Findings" section's
RMSE/MASE ranking claims, the "Explicit caveat" section, the DLinear/LSTM merged-table note, and this
D-66 entry's own stale citation above). **Corrected headline numbers**: all-tower R²=−0.363 (was
−0.565), Tower-4-only R²=−0.228 (was −0.568) — a real ranking shift, since TFT now sits much closer
to SARIMAX (−0.360 all-tower) than before, and on Tower-4-only MASE, TFT (1.014) is no longer the
single worst model — SARIMAX (1.038) is. Also fixed, found during this pass: the TabICLv2 verdict
paragraph's claim that TabICLv2 (−1.929) "sits between TFT (−0.565) and LSTM (−1.357)" was
mathematically wrong even under the old TFT number (−1.929 is not between −0.565 and −1.357) —
corrected to "sits between LSTM (−1.357) and DLinear (−5.057)." `BEST_RESULTS.md`'s two TFT rows
(Section 3) updated to match. **This does not change the standing recommendation**
(Ensemble_unweighted remains best on R²/RMSE in every table) — it only corrects TFT's own reported
position, which was already flagged as carrying real run-to-run uncertainty (D-62 addendum) and
remains so; a future rerun of `b10_b13_rerun_multi_anchor.py` for any purpose will redraw TFT again
and reopen this same staleness gap unless the propagation step above is repeated at that time.

**Third same-day follow-up — a real point-estimate bug in `tabicl_forecast()` found and fixed
(2026-07-10), prompted by the user's own skepticism.** After being told TabICLv2 (all-tower observed
R²=−1.929) ranked far below TabPFN (R²=−0.122) despite the two being architecturally close
("heavily inspired by TabPFN-TS," per TabICLv2's own docs), the user pushed back directly:
"I am skeptical how TabICLv2 is the exact opposite of TabPFN here." Investigation (comparing
`tabicl_forecast()`'s and `tabpfn_forecast()`'s input construction line-by-line, confirmed
byte-identical, then inspecting `TabICLForecaster.predict_df()`'s raw output columns directly)
found the actual bug: `tabicl_forecast()` extracted its point forecast from `TabICLForecaster`'s
`target` column, which uses the library's default `point_estimate='mean'`. CH4 flux is heavily
right-skewed and spike-dominated (this project's own recurring "MASE<1 alongside near-zero/negative
R² = spike-tail signature" finding, D-44b) — a **mean**-based point estimate is dragged far above
the typical value by the model's own upper-tail belief. Direct evidence, same context/covariates,
only the extracted column differing: Tower 2/anchor 2018, `target` (mean) column mean=30.7 vs. true
evaluation-window mean=2.98, vs. the model's own **median** (`0.5`) quantile column mean=6.3 — much
closer. Separately confirmed covariates were genuinely being used (perturbing `future_df`'s `fx_`
columns 3x measurably shifted the forecast, from a `target` mean of 30.7 to 186) — ruling out a
covariate-plumbing bug; this was purely a point-estimate-choice bug. A 4-anchor/tower spot check
swapping `target`→`0.5` before the full fix confirmed R² improved by ~1-4 points and MASE dropped
below 1 (beat persistence) in 3/4 cases, vs. never for `target`.

**Fix**: `tabicl_forecast()`'s point-only branch (`quantiles=None`) now returns the `0.5` quantile
column instead of `target` (`src/models/recursive_rollout.py` — the only code change; TabICLv2's
own default quantile grid `[0.1..0.9]` is always present regardless of the `quantiles=` argument, so
`0.5` is always available with no extra request needed). Full 3-tower × 5-anchor sweep rerun
(~10 seconds total, still dramatically cheaper than every other model).

**Corrected result — reverses the original finding almost entirely**: all-tower observed R²=−0.329
(was −1.929), MASE=0.928 (was 1.424). TabICLv2 now **beats SARIMAX (−0.360) and TFT (−0.363) on R²**,
and its MASE is 4th-best of all 10 models in the sequence (beating LightGBM, RF, TFT, SARIMAX),
comfortably below 1.0 — the pre-fix number never beat persistence at all. It remains behind TabPFN
(R²=−0.122, MASE=0.855) and the standing recommendation (Ensemble_unweighted, R²=−0.165,
MASE=0.918), so **the standing recommendation is unchanged** — but TabICLv2 is now a genuinely
competitive mid-pack model at a fraction of the compute cost of everything ahead of it, not a
near-bottom one. Tower 9's 2019 anchor is the one row unaffected by the fix (already-documented
zero-real-context degenerate case, D-66 original entry — both mean and median collapse near 0 when
there is no real signal to condition on). Updated: `b10_b13_metrics_rerun.md` ("Model-roster
extension: TabICLv2" section and the "Secondary metric" all-tower table's TabICLv2 row),
`results/b10_b13_tabicl_extension_summary.csv` + all 6 derived table CSVs (all-tower/per-tower ×
observed/gap-filled, tower×year), `results/b10_b13_tabicl_extension_chains.csv`. Per the standing "always generate figures and keep
the consolidated chains CSV current" convention: `results/b10_b13_full_chains.csv`'s `TabICLv2`
column was refreshed from the corrected chains (sanity check: its overall mean moved from 35.3 to
10.2, now closely matching TabPFN's own 10.2 — direct evidence the two models' predictions are now
comparably scaled, not the biased-high mean output from before) and
`notebooks/05_benchmarking/b10_b13_chain_plots.py` was rerun, regenerating all 165 figures including
TabICLv2's 15. `BEST_RESULTS.md`/`CONTEXT.md` updated to match.

---

### D-44b — 2026-06-30 — additional point-forecast metrics: WAPE, MASE, sMAPE, MAPE (backfilled B01–B07)
**Decision:** Track 4 more point-forecast metrics alongside the existing RMSE/MAE/R²/MBE, centralised in a new
shared module `src/evaluation/metrics.py` (`rmse/mae/r2/mbe/wape/mase/smape/mape/full_metrics`), imported by
every forecasting notebook (`sys.path.insert(0,"../../src")`; `forecasting_dl.py` imports it directly for
B02/B04). **FCH4 is a signed flux that crosses zero** (uptake periods → flux near/below 0, range ≈
−1559…+6161), which makes **MAPE mathematically unstable** (division by near-zero/zero actuals) — confirmed
empirically post-backfill: MAPE values of 280–420% on FC-01's hourly persistence/RF rows. **WAPE**
(`Σ|y−ŷ|/Σ|y|`, aggregates before dividing) and **MASE** (test-set relative-MAE form:
`MAE(model)/MAE(persistence)`, scaled against the same out-of-sample persistence baseline already used for
`skill_persist` rather than Hyndman-Koehler's in-sample naive, to stay consistent with this project's existing
baseline convention) are the recommended primary additions; sMAPE/MAPE are computed for completeness but
flagged unstable on this data (`mape()` returns `(value, n_excluded)`, filtering `|y|<1` nmol rows; sMAPE uses
an epsilon floor) — **MASE is the recommended "watch out" indicator**: MASE>1 = worse than naive persistence
(hard fail); MASE comfortably <1 alongside a near-zero/negative R² is the spike-tail signature this project
has repeatedly hit (B-05/B-06's mechanism) and a fast way to flag "fine on bulk error, unreliable on spikes."
**Scope (user-confirmed):** backfilled across all 7 forecasting-phase benchmark notebooks (FC-01/B01, FC-02/B02,
B03, B04, B05, B06, B07 — all re-executed via nbconvert, zero rows lost, total benchmarks.csv still 3665 rows).
**Excluded:** FC-03/U-01 (uncertainty quantification) — its rows are interval-calibration metrics
(PICP/MPIW/pinball), a different metric family from point-forecast accuracy; the same models' point accuracy is
already covered by FC-01/B-03/B-04. **Deferred:** the gap-filling phase (R-01 through F-08, 11 notebooks,
several with expensive 5×4/5×5 bootstrap loops) — staged for a later pass per the user's explicit choice to
prioritise the forecasting phase first, given remaining timeline (deadline 1 Sept). Cross-ref D-37 (skill vs
baseline framing), D-40 (UQ metrics precedent for additive benchmarks.csv columns).

---

### D-67 — 2026-07-10 — F-10: extended feature engineering (livestock species, land-use regime,
### catchment flow, fertilizer richness, bonus liveweight density) — Stage 1 build + signal check

**Decision:** After D-66 showed the whole B-09→B-15 recursive-rollout sequence has converged to
roughly the same ceiling (R²≈0, MASE≈0.85–0.98) regardless of model architecture/HPO, the user
pivoted the search for improvement from models to features — directly supported by this project's
own history (B-03/B-04's enriched features lifted point-forecast R² by +0.08–0.10, more than any
model change anywhere in this project). Four leads were named (livestock-type granularity,
fertilizer richness, catchment flow instrumentation, a possible Tower-2 land-use regime shift) and
confirmed real via research before any code was written:

- **`forecast_daily_v2.csv`'s zero fertilizer/management columns** — even `mgmt_t{t}_cut_recency`/
  `manure_recency` (F-05/D-32-validated small-positive on the old hourly harness) never reached
  the daily matrix.
- **Livestock species identity discarded, not missing** — `gapfill_rfm.py: frame()` sums
  `cattle_Catchment N`/`sheep_Catchment N`/`lamb_Catchment N` (present through
  `consolidated_hourly_SMS_MET.csv`) into one `lsu_dens` scalar; F-01's own SHAP ablation already
  found `cattle_Catchment 4` carries independent signal (#4 top feature) before this collapse.
  `livestock_weight_long.csv`/per-animal location files exist, used by zero scripts.
- **Catchment flume instrumentation (17 parameters/catchment, incl. `Flow (l/s)` at 87-91%
  coverage — better than SWC's own coverage) is fully ingested through `consolidated_hourly.csv`
  but never reaches any forecasting feature matrix** — an oversight (`build_forecasting_matrix_v2.py`
  selects features via 2 hardcoded literal constants that never match flow/chemistry names), not a
  documented exclusion (no `DECISIONS.md` entry ever decided against it).
- **Tower 2 (field NW002) underwent a confirmed, genuine permanent-pasture→arable conversion**
  starting 2019-09-09 (first `Plough`)/2019-10-02 (first wheat drill), continuing as a cereal
  rotation through 2024; raw livestock headcounts at Catchment 2 are genuinely 0.0 (not imputed)
  from 2019 onward. Independently re-confirmed Towers 4/9's own fields (NW005/NW006, NW013/NW039)
  show **zero** arable operations across 2017-2024 — Tower-2-only regime shift. See D-68 for the
  separate documentation-reconciliation entry this motivated.

**Stage 1 (this entry): build + cheap signal check, all 3 towers.** New, purely additive files
(nothing existing edited): `src/features/build_bodyweight_density.py` (bonus family (e) — a
last-observed-carried-forward per-animal location×weight join; feasibility confirmed empirically,
location files genuinely resolve to real NWFP field codes, not just shed labels, contrary to the
original planning-time uncertainty), `src/features/build_forecasting_matrix_v3.py` (reads
`forecast_daily_v2.csv` read-only + raw `load_ext()`/`management_features.csv`/
`Field_Event_Data_Format_1.csv`, left-merges 18 new columns, writes `data/Hourly/
forecast_daily_v3.csv`, 8,772 rows × 66 cols), `notebooks/04_feature_engineering/
f10_signal_check.py` (Stage 1 ablation), `F10_extended_features.ipynb` + `F10_results.md`.

**Two real implementation bugs caught and fixed during verification, not after:**
1. `fx_is_arable`'s first implementation used `build_management_features.classify()=="cultiv"` as
   its trigger — too broad, also fires on routine grassland renovation (a single `Chain harrow` at
   Tower 4 in 2022, `Chain harrow`/clover-blend `Grass seeding (overseeding)` at Tower 9),
   producing false arable flips at both towers that contradicted the very field-record check that
   motivated the feature. Fixed to a narrower, empirically-verified trigger (literal `Plough`, or a
   Drill/Broadcast-Seed operation whose `Application` names an actual cereal crop —
   wheat/oat/barley/bean, not grass/clover) — confirmed `Plough`/cereal-drilling only ever occurs
   at NW002/NW003/NW004/NW015/NW019/NW047, never at Towers 4/9's own fields. **Result: Tower 2
   flips 2019-09-09, Towers 4/9 never flip (`fx_is_arable.sum()==0` unconditionally)** — matches
   the direct field-event check exactly.
2. `fx_flow_lag{7,14,21,28}` were built on the raw hourly-indexed frame before resampling to daily,
   so `.shift(L)` shifted by L *hours*, producing wholly-NaN columns. Fixed by resampling to daily
   first (matching `build_forecasting_matrix_v2.py`'s own pattern) — confirmed non-NaN, 87-92%
   coverage at all 3 towers post-fix.

Every family's verification checklist item passed after these fixes: exact LSU-linearity identity
(<1e-9) confirming the species split is a lossless refinement of `fx_lsu_dens`; `fx_flow_roll7`
matches a manual recompute; a spot-checked fertiliser event's `fx_mgmt_fertN_recency` shows the
correct τ=14 exponential decay with no leakage/off-by-one.

**Stage 1 signal check (cheap, bounded, single-seed leave-one-group-in RF ablation — B-03/B-10's
exact daily-track hyperparameters, no new HPO): none of the 5 families clear the pre-registered
go/no-go bar (ΔR²>+0.01 consistent across towers/horizons, or a top-10 SHAP rank that survives a
collinearity check).** Harness sanity-checked first: `BASE`'s own R² (T4 h=1=0.365, h=14=0.280;
T9 h=14=0.359) reproduces `BEST_RESULTS.md`'s published B-03 numbers almost exactly. `fx_cattle_dens`/
`fx_total_liveweight_dens` draw real SHAP attention (#1/#2 of 18 new columns) but `fx_cattle_dens`
correlates with the existing `fx_lsu_dens` at r=0.972 — a follow-up `SWAP_species_for_lsu` test
(replace rather than add) shows a genuine tower-specific pattern (+0.008/+0.019 R² at Tower 4,
where I-02 already found livestock density dominant; −0.013/−0.021 at Tower 9) that nets to
~−0.002 once both towers are weighted equally — not a consistent win. `BASE+ALL` (all 18 columns
at once) is worse than `BASE` everywhere, consistent with this project's repeated "weak features
stacked can hurt" pattern. Management richness did **not** reproduce D-28's Tower-9 collapse this
time (plausibly because the richer v2/v3 base feature set no longer relies on management signal as
heavily) but showed no positive signal either — neutral, not actively harmful.

**Verdict: Stage 2 (B-16 full point-forecast/recursive-rollout retrain) is NOT triggered this
round** — per the plan's own explicit rule, families that don't clear the bar are reported as null
results and dropped, not carried forward. `BEST_RESULTS.md` is unchanged. This is an honest null
result, not a wasted round — two real bugs were caught (see above), and `fx_is_arable` is a
genuine, now-correctly-implemented mechanistic feature with real documentation/interpretability
value (D-68) even though it doesn't move a direct-forecast R² (expected, given it's largely
redundant with the `is_t2`/`is_t4`/`is_t9` dummies already in every model).

**Caveats, stated plainly:** this is a single-seed, no-CV, direct-h-forecast smoke test, not a
final word — flow's null result doesn't rule out its mechanistic hypothesis (a direct h-day-ahead
forecast with only mean+lag/roll features may not be the right lens for a cumulative, event-driven
waterlogging signal); the species/liveweight tower-specific pattern is a genuine finding worth
remembering if a tower-specific (not pooled) model is ever tried; Tower 2 could not be evaluated
directly in this check (zero real 2022-2023 rows, the same well-documented data-scarcity finding
as everywhere else in this project).

New files: `src/features/build_bodyweight_density.py`, `src/features/build_forecasting_matrix_v3.py`
(+`data/Hourly/forecast_daily_v3.csv`, `data/Hourly/bodyweight_density.csv`),
`notebooks/04_feature_engineering/f10_signal_check.py`, `F10_extended_features.ipynb`,
`F10_results.md`, `results/f10_signal_check_{summary,shap,deltas}.csv`. **No existing file edited**
(`build_forecasting_matrix_v2.py`, `gapfill_rfm.py`, `build_management_features.py`, every
B01-B15/I01-I02/U01-U03/S01 artifact untouched). Cross-ref D-28/D-29/D-30 (management-overfit
precedent), D-61/I-02 (livestock dominance, motivating the species-disaggregation test), D-46
(fertiliser recency's prior scoping), D-41 (bounded-iteration norm — one round of build + one round
of signal check, no iterative re-tuning given the mixed/null result).

**Addendum — Stage 2b: recursive-rollout confirmation (2026-07-10), run despite Stage 1's null
result, per direct user instruction.** User: "run the rollout test (recursive forecasting). The
point of this experiment is to test on forecasting performance improvements, no gap-filing." Stage
1's signal check only exercised a direct, non-autoregressive point forecast (B-03-style); the
actual model/HPO ceiling that motivated pivoting from models to features in the first place lives
specifically in the recursive rollout (B-09→B-15), a genuinely different failure mode (error
compounding, spike-blindness) a one-shot forecast never exercises — so a point-forecast null result
doesn't necessarily transfer, and was worth checking directly rather than assumed.

New file `notebooks/05_benchmarking/b16_recursive_rollout_v3.py` (committed) — reuses
`b10_b13_rerun_multi_anchor.py`'s exact methodology (same hyperparameters, same unmodified
`tree_rollout`/`bin_metrics` from `recursive_rollout.py`, same SARIMAX order-search) for 6 configs
(`BASE` = v2's feature set, plus each of the 5 families added individually) × all 3 towers × all 5
anchors × the full RF/XGB/LightGBM/SARIMAX ensemble. SARIMAX fit once per (anchor, tower), reused
across all 6 configs (its `EXOG_B` set is unaffected by any new family). One implementation note,
not a bug: this script had to be run via the explicit anaconda Python path
(`/c/ProgramData/anaconda3/python.exe`), not the bash shell's default `python` (which resolved to a
different, package-incomplete environment, `ModuleNotFoundError: xgboost`) — the same class of
pip/python environment mismatch already documented in D-66, confirming that lesson generalizes
beyond `pip install` to plain script execution in this environment too.

**Sanity check passed first**: `BASE`'s `Ensemble_unweighted` reproduces `BEST_RESULTS.md`'s
published all-tower headline almost exactly (R²=−0.1652 vs. published −0.165, MASE=0.9169 vs.
0.918). **Result: none of the 5 families beat `BASE` on the ensemble** (all-tower R²: BASE −0.165,
species −0.167, arable −0.169, mgmt −0.169, bodyweight −0.170, flow −0.172) — species comes
closest (ΔR²≈−0.001, essentially noise) but does not improve on it. Checked individually for RF,
XGB, and LightGBM too (not just the ensemble) — same null pattern holds on every single tree model.
Per-tower breakdown for species (the closest contender) replicates Stage 1's exact tower-specific
pattern: small gain at Tower 4 (R² +0.0054, the tower I-02 already found livestock-density-dominant)
offset by small losses at Towers 2 and 9 — twice-replicated, not a fluke, but not a net win either.

**Final verdict: two independent forecasting evaluations (point-forecast and recursive-rollout,
both scoped to forecasting only — the gap-filling pipeline was never touched by any part of F-10)
agree that none of the 5 new feature families improve forecasting performance.**
`BEST_RESULTS.md` is unchanged; B-10's `Ensemble_unweighted` remains the standing recommendation.
New files: `notebooks/05_benchmarking/b16_recursive_rollout_v3.py`,
`results/b16_recursive_rollout_v3_summary.csv` (3,240 rows), `results/b16_recursive_rollout_v3_chains.csv`.

**Second addendum — `BASE+ALL` follow-up (2026-07-10), closing a gap the user asked about.** The
original Stage 2b sweep tested `BASE` + each of the 5 families individually but omitted a
`BASE+ALL` (all 18 columns stacked) config — an oversight, not a deliberate scope decision (Stage
1's point-forecast check *did* include `BASE+ALL`). New script
`notebooks/05_benchmarking/b16_recursive_rollout_v3_all.py` fills this in cheaply, reusing the
already-fitted SARIMAX chains from the first Stage 2b run (its `EXOG_B` set is unaffected by any
family, so refitting would be pure waste) and refitting only RF/XGB/LightGBM on the full
18-column set — ~5 minutes instead of another ~28. **Result: `BASE+ALL` (R²=−0.168) still loses to
`BASE` (−0.165), but lands in the middle of the pack** — better than 4 of the 5 individual
families (arable, mgmt, bodyweight, flow), worse only than the species family alone. **This is a
genuinely different pattern than Stage 1's point-forecast check**, where stacking all 18 columns
was clearly the *worst* config by a wide margin (~0.04–0.05 R² gap to `BASE`) — here the stacking
penalty is much milder and not the single worst outcome. Both harnesses still agree on the
headline (nothing beats `BASE`), but not on the specific shape of how excess features hurt —
worth remembering that a failure mode confirmed on one evaluation harness doesn't automatically
describe the other's behavior. New files:
`notebooks/05_benchmarking/b16_recursive_rollout_v3_all.py`,
`results/b16_recursive_rollout_v3_all_{summary,chains}.csv` (also appended into the main
`b16_recursive_rollout_v3_summary.csv`, now 3,780 rows / 7 configs).

**Third addendum — MASE re-read + foundation models (2026-07-10), materially changes the
"final verdict" above.** Two follow-ups, both user-requested:

1. **MASE prioritized as the primary forecasting metric going forward** (new `CLAUDE.md` standing
   convention — CH4's spike-tail behavior repeatedly destabilizes R², MASE (error relative to
   naive persistence) is far more robust to it). Re-reading the tree-ensemble table under MASE:
   **`BASE+species` (MASE=0.9161) is actually marginally the best config**, edging out `BASE`
   (0.9169) — a real, if small, reversal of the R²-led framing above (which had `BASE+species`
   losing by ΔR²=−0.0014). Every other family remains worse than `BASE` on both metrics.
2. **TabPFN and TabICLv2 tested across all 7 configs** (new script
   `b16_foundation_models_v3.py`, zero-shot, no retraining needed per config), per the user's
   request to show "all models from SARIMAX to TabICLv2." **Result: unlike the trees, both
   foundation models show real, substantial, broadly-consistent gains from several families** —
   verified not a single-tower/single-anchor artifact. Headline: **`TabPFN+species`
   (MASE=0.840, R²=−0.084) is the best single-model result in the entire B-09→B-15 sequence**,
   beating the standing `Ensemble_unweighted` recommendation (MASE=0.917) outright. `TabICLv2`'s
   best config is `BASE+ALL` (MASE=0.871, R²=−0.155), also a large improvement over its own
   `BASE` (0.928/−0.329). `fx_is_arable` shows exactly zero effect on either model (numerically
   identical to `BASE`) — expected, not a bug: it's a constant flag within nearly every per-tower
   rollout window, giving neither model anything to condition on.

**The "Final verdict: ... none of the 5 new feature families improve forecasting performance"
statement above is superseded by this addendum — it was true for trees/SARIMAX/ensemble only, not
for the foundation models.** TFT/DLinear/LSTM are being tested next (required building a new
hourly-track matrix, `forecast_features_v3.csv`, via `build_forecasting_matrix_v3_hourly.py`,
since these models read the hourly track which F-10's original scope never touched) before the
final, complete verdict across all 11 models is written up. New files:
`notebooks/05_benchmarking/b16_foundation_models_v3.py`,
`results/b16_foundation_models_v3_summary.csv` (1,260 rows), `src/features/
build_forecasting_matrix_v3_hourly.py`, `data/Hourly/forecast_features_v3.csv`.

**Fourth addendum — final all-11-model verdict + gap-filled secondary metric (2026-07-10).**
TFT/DLinear/LSTM tested the same way (new `b16_dl_models_v3.py`; `forecasting_dl.py` needed **zero
code changes** — it already auto-detects `fx_`-prefixed columns into a module-level `FX` list read
at call time by `tower_series()`, so the script just reassigns that list per config before each
training/rollout call). Ran in ~7 minutes (much cheaper than the tree sweep). **Result: TFT shows
the same pattern as the foundation models, and it's dramatic** — `BASE` alone (MASE=1.063) loses to
naive persistence; `BASE+ALL` (MASE=0.941) beats it, a genuine flip. LSTM/DLinear also improve with
extra features (species/bodyweight respectively) but remain far behind every other model in
absolute terms, consistent with their well-documented instability (D-53/D-54).

**Complete picture, best config per model, MASE-ranked**: TabPFN+species (0.840) > TabICLv2+ALL
(0.871) > Ensemble_unweighted+species (0.916) ≈ Ensemble_MASEweighted+species (0.916) >
XGB+ALL (0.919) > LightGBM/BASE (0.939) > TFT+ALL (0.941) > RF+species (0.964) > SARIMAX/BASE
(0.974) > LSTM+species (1.086) > DLinear+bodyweight (1.374). **Trees/SARIMAX show no meaningful
gain from any family; every attention-based or foundation model (TFT, TabPFN, TabICLv2, and to a
lesser extent LSTM/DLinear) shows real, often large, gains** — the opposite conclusion a
tree-only smoke test would have suggested, and exactly why the user's push to test "all models"
mattered.

**Headline recommendation: `TabPFN+species` (MASE=0.840, R²=−0.084) — a new best single-model
result for the whole B-09→B-15 sequence, beating the standing `Ensemble_unweighted` recommendation
(MASE=0.918/R²=−0.165) outright, at near-zero adoption cost** (TabPFN is zero-shot/no-training; the
species family is only 3 extra columns). Promoted to `BEST_RESULTS.md`.

**Gap-filled secondary metric added for every model's own best config** (per user request, matching
the established D-65 second-addendum pattern and the `b10_b13_metrics_rerun.md` table style):
tree/ensemble chains recomputed from already-saved chains (no refit, new script
`b16_recursive_rollout_v3_gapfilled.py`); TabPFN/TabICLv2/TFT/DLinear/LSTM scripts extended to score
both targets inline and rerun. **One real bug caught while writing the tree-recompute script**: the
saved chains CSV has each config's rows appended separately (a UNION of columns across configs), so
naively grouping by (tower, anchor) and reading a config's column mixed in NaN-prediction rows from
other configs' rows and crashed `r2_score` — fixed by filtering to `.notna()` rows for the specific
column being scored before calling `bin_metrics`. **Result, corrected after direct user
questioning of the initial (wrong) "doesn't change the ranking" claim: the gap-filled/observed
comparison splits cleanly in two, and the ranking DOES flip for the headline comparison.** Trees/
SARIMAX/both ensembles score *better* on gap-filled than observed (e.g. `Ensemble_unweighted`:
0.916→0.749 MASE) — because these models are fit by directly regressing onto `y_gapfilled` as
their training label, so the circularity risk already flagged for this secondary metric
("agreement can partly reflect forecaster resembles gap-filler") works directly in their favor.
TabPFN/TabICLv2/TFT/LSTM/DLinear score *worse* on gap-filled (e.g. TabPFN: 0.840→0.944 MASE) —
their targets are `y_observed`, so they get no such boost, and this is the ordinary variance-
normalization artifact (D-65). **Consequence: under gap-filled scoring, the OLD standing
recommendation (`Ensemble_unweighted`, MASE=0.749) beats the NEW "winner" (`TabPFN+species`,
MASE=0.944) by a wide margin** — the full ranking flips, not just narrows. The observed-target
ranking remains the one to trust (D-36/D-37's "train on gap-filled, evaluate on observed"
convention — `y_observed` is the intended validation target, specifically because it isn't
inflated by this circularity), but the "TabPFN+species is unambiguously the new best" framing
from the third addendum was too strong: it is best **on the primary, convention-endorsed metric
only**, and this must be stated plainly, not left as an implied unconditional claim. Within-model
(species vs. that model's own `BASE`) the two targets DO agree on direction for TabPFN
specifically (0.854→0.840 observed, 0.949→0.944 gap-filled) — the feature-family finding itself is
robust to target choice; it's specifically the across-model "who's best overall" comparison that
isn't. New files: `notebooks/05_benchmarking/b16_dl_models_v3.py`,
`b16_recursive_rollout_v3_gapfilled.py`; `results/b16_dl_models_v3_summary.csv` (3,780 rows, both
targets), `results/b16_recursive_rollout_v3_summary_vs_gapfilled.csv` (7,560 rows), `results/
b16_full_comparison_{all_models,vs_gapfilled}.csv`, `results/
b16_final_table_vs_gapfilled_best_config.csv`. Full write-up:
`notebooks/04_feature_engineering/F10_results.md`.

---

### D-68 — 2026-07-10 — documentation reconciliation: Tower 2's land-use regime shift was already
### correctly identified (D-28/D-30/D-34), later write-ups (D-63/D-64) re-described it as generic
### "data sparsity" without cross-referencing the earlier, correct explanation

**Decision:** A colleague's review (relayed by the user) suggested Tower 2's `fx_lsu_dens=0`
finding (U-03/D-63: "exactly 0.0 for the entire 365-day rollout window in 4 of 5 anchors") might
reflect a real land-use regime shift (permanent pasture→arable) rather than plain data scarcity.
**Investigation confirmed this is correct — and, further, that it was already correctly identified
earlier in this project, then lost in translation at a later stage:**

- **`DECISIONS.md` D-28** (2026-06-16) already states: *"Tower 2 = Red farmlet (arable from 2019) —
  deferred"* and *"management-timing distribution shift (Red-farmlet conversion)."*
- **D-30** (2026-06-16): *"weakest for soil/land-use features at Tower 2 (arable)... Tower 2
  benefits most from the dummy... it is the most 'different' tower (Red→arable, 6.65 ha)."*
- **D-34** (2026-06-25), most explicit: *"Tower 2 EC CH4 exists only Oct 2017–Jun 2019 (grassland;
  analyser relocated to Tower 9 Jul 2019 at the Red-farmlet arable conversion). Catchment 2 had ~10
  cattle in 2018 (FCH4 ≈ 42) but zero livestock in early 2019 (FCH4 ≈ 2)."*
- **D-63** (2026-07-07, U-03) re-describes the identical fact purely as *"T2's known data
  sparsity"* / *"records essentially no grazing there,"* with no cross-reference back to D-28/D-30/
  D-34's already-correct causal explanation.

**Direct field-record confirmation this session** (F-10, D-67): field NW002 ("Great Field",
6.65 ha, Catchment 2) shows a clean transition — last silage/grazing event 2019-08-21, first
`Plough` 2019-09-09, first `Drill Seed 'Wheat (Crusoe)'` 2019-10-02, continuing as a wheat/oats/
beans rotation through 2024 (`Field_Event_Data_Format_1.csv`). Raw livestock headcounts at
Catchment 2 are genuinely `0.0` (not `NaN`, not imputed) from 2019 onward — `fx_lsu_dens=0` at
Tower 2 is a real reading of a real regime change, not a data gap. Independently confirmed Towers
4/9's own fields (NW005/NW006, NW013/NW039) show zero arable operations across the whole 2017-2024
record — this is a Tower-2-only shift, not a general pattern.

**This is a documentation cross-reference fix, not a new empirical finding — no numbers change.**
D-63's and D-64's "data sparsity" framing was not wrong (Tower 2's real `y_observed` coverage is
genuinely thin), it was simply incomplete — the *cause* of the livestock-related symptom
specifically was already on record at D-28/D-30/D-34 and should have been cited. Going forward,
Tower 2's `fx_lsu_dens=0` behavior (and any future scenario-analysis or interpretability work
touching it) should be read as reflecting a **genuine land-use regime shift**, not merely sparse
telemetry — a mechanistically meaningful distinction for the methods chapter (Tower 2's field has
no enteric/dung CH4 source at all post-2019, not just under-recorded livestock). F-10 (D-67) built
a corresponding `fx_is_arable` feature for this reason, even though it did not move a direct-
forecast R² in Stage 1's signal check (expected — it's a coarse, mostly-static per-tower flag,
largely redundant with the `is_t2`/`is_t4`/`is_t9` dummies already present in every model; its
value is interpretive/mechanistic, not predictive-accuracy).

**No files changed beyond this entry** — `U03_results.md`/`s01_results.md` are not retroactively
edited (matches this project's append-only decision-log convention: corrections are logged forward,
not rewritten into prior entries). Cross-ref D-28, D-30, D-34, D-63, D-64, D-67 (F-10's
`fx_is_arable` feature).

---

### D-69 — 2026-07-13 — S-02: driver-reconstruction feasibility — proxy models for CMIP6's missing
### scenario variables (preparation, not yet integrated)

**Decision:** The user proposed a genuinely new idea, confirmed via research to have never been
considered in D-52/D-64 (which only weighed a raw Copernicus CMIP6 pull vs. the historical-day-
climatology-resampling that was actually adopted): train small proxy models predicting the
variables CMIP6 doesn't provide (`fx_WS_mean`, `fx_VPD_mean`, `fx_PPFD_mean`, `fx_RN_mean`,
`fx_TS_mean`, `fx_SWC_mean`) from the 4 it does (`Tmin, Tmax, Rain, RAD`), using real historical
NWFP data to fit/validate, then applying the trained relationship to the actual simulated future
driver trajectory — more scenario-responsive than climatology, which ignores how extreme a given
future day's available drivers actually are. `fx_USTAR_mean`/`fx_SHF_mean` stay out of scope
(D-64's reasoning for dropping them is physical/data-availability based, not statistical).

**Pre-registered feasibility argument** (given to the user before building anything, per their
explicit request): D-50's own correlation matrix showed a mixed picture — strong for soil temp
(`TA`-`TS` r=0.742), moderate for PPFD/RN (r=0.48–0.56), weak for wind speed/VPD (r=−0.11 to 0.35,
also both already in I-01's lowest SHAP tier). **User chose to attempt all 6 variables anyway**
(not a narrower pilot), for a complete, honest picture.

**New notebook**: `notebooks/07_scenario_analysis/preparation/
S02_driver_reconstruction_feasibility.ipynb` — reuses `src/data/fco2_gapfill.py`'s exact
architecture (RF regressor, `n_estimators=500, min_samples_leaf=5, random_state=42`, calendar-based
train/test split, D-26) adapted to daily/6-target/pooled-across-towers (D-30's partial-pooling
default), plus `build_scenario_drivers.load_towers()`/`load_cmip6_climatology()` and
`recursive_rollout.doy_climatology()` and `scenario_hybrid.dissimilarity_index()` — all reused,
none reimplemented. Train 2018–2021, test 2022–2023 (D-04). Climatology baseline built the same
way `build_scenario_drivers.py` already does it (per-tower `doy_climatology()` on that tower's own
training history) — the critical, otherwise-missing comparison: does a trained proxy actually beat
what's already being done, not just "does it have positive R²."

**Results — real, not pre-judged, with two genuinely surprising reversals of the pre-registered
expectation:**

| Variable | RF R² | Climatology R² | Winner |
|---|---|---|---|
| `fx_PPFD_mean` | **0.507** | 0.290 | RF |
| `fx_RN_mean` | **0.449** | 0.354 | RF |
| `fx_WS_mean` | **0.363** | 0.038 | RF (largest relative win) |
| `fx_VPD_mean` | −0.000 | −0.315 | RF (both weak — climatology is just worse) |
| `fx_SWC_mean` | −0.662 | **−0.441** | Climatology |
| `fx_TS_mean` | −1.257 | **−1.043** | Climatology |

1. **Wind speed is the strongest relative RF win, despite being the correlation-evidence's weakest
   candidate (r=−0.11 to 0.31).** Confirms the reasoning flagged before running anything: linear
   Pearson r misses nonlinear/interaction structure a tree model can exploit — pre-judging
   feasibility from correlation alone would have wrongly written this one off.
2. **Soil temperature and soil moisture fail for BOTH methods** (strongly negative R²), despite
   `TS` having the *strongest* linear correlation with `TA` (r=0.742) of any candidate — the
   opposite of what the pre-registered argument expected. Root-caused with a quick, honest
   follow-up check (not pre-planned, done live after seeing the anomaly): **test-period
   (2022–2023) variance is roughly half of training-period (2018–2021) variance for these two
   variables at Tower 4** (`fx_TS_mean` std 3.56→1.53; `fx_SWC_mean` std 6.96→3.57) — a real
   train/test distributional shift that makes R² (which divides by test-set variance) punishing for
   *any* predictor here, not evidence the underlying TA–TS relationship is actually weak.
3. **Extrapolation check (reusing `dissimilarity_index()`, unmodified): 100% of 2041–2060 SSP2-4.5
   scenario days are flagged outside the real historical training envelope, at all 3 towers.**
   Stronger than "some risk" — every proxy model, including the genuine winners, would be applied
   entirely outside its validated range if used for real scenario projection as currently
   constructed. Plausibly partly an artifact of the CMIP6 ensemble-mean's own smoothing (averaged
   across 500 GCM×realization files, so it has less day-to-day texture than any single real year)
   rather than purely a climate-shift signal — but a real, serious caveat regardless.

**Verdict: PPFD, RN, and WS show genuine, validated within-envelope skill over climatology and are
real candidates for a follow-up integration decision** (only after addressing the 100%-
extrapolation caveat — e.g. re-testing against individual GCM/realization trajectories, which
would have real day-to-day texture, rather than the smoothed ensemble mean). VPD shows no real
skill either way. TS/soil moisture are not good candidates for this specific approach as built —
though the variance-shift root cause suggests the underlying idea isn't necessarily disproven for
soil variables, just under-tested here.

**Explicitly a preparation/feasibility pass — nothing wired into production.**
`src/features/build_scenario_drivers.py` and `notebooks/07_scenario_analysis/
S01_first_scenario.ipynb` are untouched; adopting any winning proxy model is a deliberate, separate
follow-up decision, not assumed from this pass alone. No new HPO (`fco2_gapfill.py`'s exact
hyperparameters reused verbatim). New files: `notebooks/07_scenario_analysis/preparation/
S02_driver_reconstruction_feasibility.ipynb`, `results/s02_driver_reconstruction_summary.csv`
(per-tower detail), `results/s02_driver_reconstruction_pooled.csv` (all-tower verdict). Cross-ref
D-50 (correlation evidence), D-52 (the climatology baseline this compares against), D-64
(USTAR/SHF's separate, unrevisited exclusion), D-26 (`fco2_gapfill.py`'s architecture precedent).

### D-70 — 2026-07-14 — S-03: driver-availability ablation — isolating scenario-mode
### feature-degradation cost from extrapolation cost

**Decision:** Supervisor request (Prof. Paul Harris, relayed by the user): isolate how much
forecasting accuracy is lost purely from **not having access to real-time sensor variables that a
CMIP6 climate scenario can never supply** — distinct from two effects already tested, both of which
conflate this question with something else. **U-03/D-63** tests calibration/model robustness under
an out-of-envelope `fx_lsu_dens` perturbation — real historical anchors, but the shock is an extreme
covariate value, not a feature-set change. **S-01/D-64** builds the real scenario pipeline but
evaluates on unscored 2041–2060 data, so it cannot separate "the drivers are degraded" from "2050 is
out of the training envelope." S-03 fixes this: test data stays real and historical (same 2018–2022
anchors, same 3 towers as B-10/D-65) — only the feature set changes.

**Design:** Model 1 = B-10's unweighted RF+XGB+LightGBM+SARIMAX ensemble, full feature set —
**not rerun**, read directly from the existing D-65 tables. Model 2 = same architecture/
hyperparameters/ensemble, two variants on a **24-column degraded set** (imported directly from
`build_scenario_drivers.py`'s `RESAMPLED_COLS`+`DROPPED_COLS` — S-01's own production list, not
retyped, resolving the PPFD/RN ambiguity flagged going in: both are already in `RESAMPLED_COLS`,
S-01 already treats them the same as WS/VPD/SWC/TS). User-confirmed the list also includes wind
direction and grazing features (S-01's own treatment). `fx_lsu_dens` (the scenario lever) and all AR
features (real recent history genuinely exists at a historical anchor, unlike a genuinely blind 2050
future) stay real/untouched in both variants — the design choice that keeps this experiment
isolating driver-availability cost specifically.
- **Variant A (removal)**: the 24 columns dropped entirely, never seen in training or rollout.
- **Variant B (resample)**: same columns real in training (identical to Model 1); only the
  rollout-time/test-window values are day-of-year-climatology-resampled via `rr.doy_climatology()`.
  **One deliberate, necessary deviation from S-01's own call**: `doy_climatology()`'s history is
  restricted to **pre-anchor-only** data (`dft.loc[:anchor, col]`), not S-01's full-record call
  (correct for S-01, which has no real anchor — it projects from the end of history to 2050; using
  the full record here would leak a historical anchor's own future values into its own test-window
  climatology). Verified via assertion on every substitution call.

Both column lists are **independently customizable parameters** (`remove_cols`/`resample_cols` in
`s03_driver_availability_ablation.py`'s `main()`, both defaulting to the 24-column list), per direct
user request, so a follow-up sensitivity check (e.g. resampling only soil variables) is a one-line
notebook call with a distinct `run_label`, not a script edit. Full 3-tower × 5-anchor × 2-variant
coverage; no new HPO; TFT/TabPFN out of scope (Model 1/2 are specifically B-10's 4-model
architecture). Scored via `bin_metrics()` (unmodified) against both `y_observed` (primary) and
`y_gapfilled` (secondary, D-65's addendum convention, with `real_frac`).

**Headline result — genuinely surprising, reported plainly:** neither degraded variant costs
accuracy relative to Model 1, pooled across all 3 towers — if anything, both modestly beat it.
`Ensemble_unweighted` pooled: MASE 0.918 (Model 1) → 0.926 (Variant A) → **0.892 (Variant B)**; R²
−0.165 → −0.108 → **−0.089**. Variant B (resample) beats Model 1 on MASE for all 6 model rows and on
R² for 5 of 6; Variant A (removal) beats Model 1 on R² for every model but is a near-wash on MASE.
Per-tower: Variant B wins/ties at T4 and T9 on both metrics; Variant A narrowly wins at T2. Plausible
explanation (not directly measured in this pass): many of the 24 degraded columns (USTAR, wind,
VPD, PPFD/RN) were already flagged low-SHAP-importance in I-01/I-02 — dropping or smoothing
low-signal, noisy sensor inputs may reduce overfitting in a 365-day recursive rollout more than it
costs real signal, consistent with (not contradicting) `fx_lsu_dens`'s already-established dominance.

**Secondary gap-filled-target metric disagrees for Variant A specifically, stated plainly rather
than smoothed over**: under the observed target Variant A looks roughly competitive; under the
gap-filled target it looks clearly worse than both Model 1 and Variant B across every model (e.g.
`Ensemble_unweighted` R² −0.189 → −1.027). Plausible read: `y_gapfilled` is itself a pooled RFm
gap-filler's output trained on met/soil features Variant A's models never see, so Variant A's
predictions diverge further from the gap-filler's own output space specifically — a circularity
artifact (D-65's own established caveat for this secondary metric), not necessarily a real accuracy
difference.

**Practical implication for Phase 07**: this particular, isolated driver-availability cost looks
small — the documented scenario risk (U-03/S-01) concentrates in extrapolation and SARIMAX's
unbounded response, not in losing these 24 specific sensor channels. This does not mean scenario
forecasting overall is risk-free — only that this specific, isolated cost is small.

**Chain figures generated and merged** (per direct user request, extending the standing
`b10_b13_chain_plots.py`/`b10_b13_full_chains.csv` convention to this ablation): 12 new
variant-suffixed model columns (`{model}_S03_A_removal`/`{model}_S03_B_resample`) merged into
`results/b10_b13_full_chains.csv` via a left join on `(date, tower, anchor_year)` — verified row
count unchanged (5,475 → 5,475), only new columns added. 180 new figures in
`results/figures/b10_chains/` (12 models × 15 tower/anchor combos); `MODEL_COLORS` in
`b10_b13_chain_plots.py` extended with 12 new entries (muted tints of each base model's own hue).

**Purely additive** — `B10_daily_improvements.ipynb`, `S01_first_scenario.ipynb`, S-02's notebook,
`build_scenario_drivers.py`, `scenario_hybrid.py`, `recursive_rollout.py`,
`b10_b13_rerun_multi_anchor.py`, and every existing D-65 results CSV are untouched (verified via
`git status`). New files: `notebooks/07_scenario_analysis/s03_driver_availability_ablation.py`,
`compile_s03_results.py`, `S03_driver_availability_ablation.ipynb`, `s03_results.md`;
`results/s03_summary.csv`, `s03_summary_vs_gapfilled.csv`, `s03_chains.csv`,
`s03_table_all_towers.csv`, `s03_table_by_tower.csv`, `s03_table_vs_gapfilled_all_towers.csv`,
`s03_table_vs_gapfilled_by_tower.csv`. Cross-ref D-63 (U-03, the distribution-shift test this
disentangles from), D-64 (S-01, the extrapolation test this disentangles from), D-65 (Model 1's
source numbers, `bin_metrics()`/chain-figure conventions reused), D-69 (S-02, the excluded
proxy-reconstruction alternative), D-52 (the RESAMPLED_COLS/DROPPED_COLS list this reuses).

**Addendum (2026-07-15): model-roster extension — TFT/TabPFN/DLinear/LSTM/TabICLv2 were missing, a
real scope gap, not a deliberate exclusion.** S-03's original design explicitly scoped Model 1/2 to
"B-10's 4-model architecture" — true when written, but the project's standing model roster had grown
to 11 models by B-13/D-66 (TFT/TabPFN, then DLinear/LSTM/TabICLv2), and S-03 was never extended to
match. Caught only after the user asked directly whether S-03 covered TabPFN/TabICLv2 — it did not,
and the user flagged this as a mistake to rectify. Fixed via a new script
(`s03_model_roster_extension.py`) running the exact same two variants against all 5 remaining
models: TabPFN/TabICLv2 (zero-shot, per-tower/anchor, mirrors `b16_foundation_models_v3.py`) and
TFT/DLinear/LSTM (pooled per anchor, hourly Track B, mirrors `b16_dl_models_v3.py`/
`b10_b13_dl_extension.py`, exact D-45/B-13a and B-09 recipes, no new HPO). Of the 24 daily degraded
columns, 12 have a verified direct hourly analogue (traced to `build_forecasting_matrix_v2.py`'s
source — the daily columns are literally `.resample("D").mean()` of these hourly series); the
remaining 12 (the SWC/TS daily lag/rolling ladder) have no hourly equivalent at all and are
explicitly out of scope for the hourly DL track, not silently dropped. Smoke-tested (1 anchor, all 3
towers/models) before the full 5-anchor sweep — all results non-degenerate before committing.

**Result refines, rather than confirms, the original finding**: on MASE, Variant B (resample) beats
or ties Model 1 for 9 of 11 models (only TFT is genuinely worse); Variant A (removal) is much more
mixed across the extended roster (clear win for SARIMAX/TabPFN/DLinear, flat-to-worse for
RF/XGB/LightGBM/both ensembles/TFT/LSTM/TabICLv2). **TFT is a genuine reversal**: R² gets measurably
worse under Variant B specifically (-0.363 → -0.492), the opposite direction from every other model
tested — flagged as a hypothesis (attention's possible sensitivity to the resampling-boundary
discontinuity) for future interpretability work, not a proven mechanism. Practical implication
updated: the "driver-availability cost is small" finding holds for the production-recommended
ensemble and most of the extended roster, but not uniformly — does not transfer to TFT.

A latent bug was caught and fixed while merging: `b10_b13_chain_plots.py`'s `plot_chain()` selected
the ground-truth column via `model in ("TFT","DLinear","LSTM")` verbatim, which silently missed every
variant-suffixed column this addendum adds (e.g. `TFT_S03_A_removal`), always falling back to the
wrong ground-truth series for those 6 new columns. Fixed to strip the `_S03_*` suffix before the
check; verified visually on a regenerated figure before trusting the full 495-figure regeneration.

Canonical files extended in place (row counts verified unchanged where expected, backed up before
writing): `results/s03_summary.csv`/`s03_summary_vs_gapfilled.csv` (1,080→1,980 rows),
`results/s03_chains.csv` (10,950 rows, +5 model columns +`y_true_tft`),
`results/b10_b13_full_chains.csv` (5,475 rows, +10 variant-suffixed columns). `compile_s03_results.py`
extended to build Model 1's numbers for the 5 new models from the raw per-bin summary files
(`b10_b13_dl_extension_summary*.csv`, `b10_b13_tabicl_extension_summary*.csv`) rather than each
family's differently-shaped pre-built table — verified to reproduce the previously-published
`b10_b13_rerun_table_all_towers.csv` bit-for-bit before adopting this approach. Full write-up:
`s03_results.md`'s "Addendum: model-roster extension" section.

---

### D-71 — 2026-07-15 — is chain-persistence a valid MASE baseline for a seasonal series? Extended B-09's climatology baseline to full coverage to check

**Question raised (user, live discussion):** this project's MASE denominator throughout (D-37) is
chain-persistence — the anchor day's real value, held flat for the full 365-day rollout. For a
series with real seasonality (FCH4), Hyndman & Koehler's own MASE recommendation is to scale
against a *seasonal-naive* baseline instead, since a flat hold ignores season entirely and can be
trivially beaten at long lead times by any model that merely tracks the seasonal cycle. This
project already has a seasonal-mean analogue — `rr.doy_climatology()` (day-of-year mean, ±7-day
window, from strictly pre-anchor real `y_observed` history) — but it had only ever been computed
for a single tower/anchor, informally, inside B-09's original smoke test (D-53), never extended to
full coverage or used to rescale B-10/B-13's headline MASE.

**Fix:** new script `b10_b13_climatology_baseline.py` (committed) — reproduces B-09's exact
climatology recipe for all 3 towers × 5 anchors (2018–2022), merges the result into
`b10_b13_full_chains.csv` as a new `Climatology` column, then reruns `rr.bin_metrics()` (unmodified)
per (tower, anchor, model) with `y_persist=Climatology` instead of `persistence` — recomputing MASE
only (R²/RMSE/MAE/WAPE/Correlation are baseline-independent, unchanged) for the full 11-model B-10/
B-13 roster. No models refit — reuses the predictions already in `b10_b13_full_chains.csv`. One
real, expected NaN case: Tower 9 has zero pre-anchor real `y_observed` history for its 2018/2019
anchors (confirmed directly), so climatology is undefined there (730/5,475 rows) — handled by
falling back to `y_persist=None` for those two (tower, anchor) combos rather than crashing on
sklearn's NaN-intolerant `mean_absolute_error`.

**Result — genuinely surprising, and the opposite of the motivating hypothesis: pooled across all
3 towers, climatology is the *weaker* baseline, not the harder one.** Directly checked both
baselines' own MAE against real `y_true`, pooled (n-weighted): persistence MAE=37.50 vs
**climatology MAE=43.79** — climatology's own forecast error is *higher* than flat persistence's.
Every model's MASE therefore looks numerically better when scaled against climatology (range
0.797–1.265) than against persistence (0.855–1.460) for all 11 models — not because any model
forecasts better, but because the climatology denominator is bigger for everyone equally.

**Per-tower breakdown shows this isn't uniform, and the reason likely isn't a coincidence**: at
**Tower 2**, climatology genuinely is the harder baseline (matches the original hypothesis) —
e.g. RF's MASE goes 0.346 (vs persistence) → 0.672 (vs climatology), *worse*-looking under
climatology. At **Towers 4 and 9** (paradoxically the towers with *more* real-data coverage),
climatology is the weaker baseline in most anchors — e.g. Tower 9 RF: 0.944 → 0.767, *better*-
looking under climatology. Plausible explanation (not proven): FCH4 is spike-dominated (D-44b's
established finding), and a ±7-day day-of-year window averaged over only a handful of real
historical years per tower is itself a noisy estimate that a few extreme spike days can distort —
so "climatology" here is a low-sample, high-variance seasonal estimate, not the smooth stable
curve the term usually implies, and it doesn't reliably outperform simply holding one real
anchor-day value flat.

**Practical implication: this is a reason to keep, not abandon, the project's standing
persistence-scaled MASE convention (D-37)** — not merely for cross-table consistency (the original
reason), but because the empirical alternative this project actually has available
(`doy_climatology`) turns out not to be a more reliable baseline given how sparse/spiky the real
FCH4 record is. The climatology-scaled MASE is retained as a secondary, exploratory comparison
column (not a replacement), consistent with this project's habit of adding secondary metrics
alongside rather than instead of the primary one (cf. the `y_gapfilled` secondary-target
convention, D-65's addendum).

**Files added:** `notebooks/05_benchmarking/b10_b13_climatology_baseline.py` (committed).
`results/b10_b13_full_chains.csv` extended in place (+1 `Climatology` column, 5,475 rows, row count
verified unchanged; backed up before writing). New: `results/b10_b13_climatology_mase_summary.csv`
(990 rows, raw per-bin/anchor/tower/model), `results/b10_b13_climatology_mase_table_all_towers.csv`
(pooled comparison, persistence vs. climatology MASE + R², all 11 models),
`results/b10_b13_climatology_mase_table_by_tower.csv` (per-tower breakdown). No `benchmarks.csv`
rows (same exclusion precedent as every other diagnostic/ablation pass this session — not a
point-forecast/interval-calibration benchmark in its own right). Cross-ref D-37 (the original
persistence convention), D-53 (climatology's first, single-anchor appearance), D-65 (the
`bin_metrics()`/multi-anchor aggregation conventions reused here).

**Follow-up same day — fairness caveat identified and fixed.** User noticed the comparison wasn't
apples-to-apples: `Climatology` was built from real `y_observed` history only (B-09's original
recipe), while `persistence`'s single anchor value comes from `y_gapfilled` (dense, can itself be a
gap-filler's smoothed model output on days the anchor wasn't a real observation) — so part of
climatology's apparent weakness could be "real vs. gap-filled data source," not "flat vs. seasonal."
Added `b10_b13_climatology_gf_baseline.py`: identical `doy_climatology()` recipe, sourced from
`y_gapfilled` instead, producing a second new column `Climatology_gf`.

**Refined result: climatology is still the weaker baseline pooled, even on a fair (gap-filled-vs-
gap-filled) basis — but the gap narrows substantially, and reverses at Tower 2.** Pooled MAE against
real `y_true`: persistence 37.50, climatology (y_observed-basis) 43.79, **climatology (y_gapfilled-
basis) 40.74** — roughly halfway between the two, still worse than persistence but much less so.
Tower 2 flips outright: climatology-gf (MAE 25.86) clearly *beats* persistence (51.94) there —
stronger than the original y_observed-basis climatology (37.26) at the same tower. Towers 4/9 still
favor persistence under either climatology variant. Conclusion stands (persistence remains the
better-justified primary MASE denominator, pooled), but the fair comparison shows climatology's
disadvantage is smaller than the first pass suggested and tower-dependent, not a blanket result.

Files added: `notebooks/05_benchmarking/b10_b13_climatology_gf_baseline.py` (committed).
`results/b10_b13_full_chains.csv` extended again in place (+1 `Climatology_gf` column, 42 cols
total, row count unchanged). New: `results/b10_b13_climatology_gf_mase_summary.csv`,
`_table_all_towers.csv`, `_table_by_tower.csv`. `results/b10_b13_mase_baseline_comparison.csv`
(the per-bin deep-dive file) extended to carry all three MASE variants side by side
(`MASE_persistence`, `MASE_climatology_obs`, `MASE_climatology_gf`). `results/figures/b09_chains/`
(165 figures) regenerated with the third baseline overlaid.

---

### D-72 — 2026-07-15 — gap-filled-target/-context ablation for the DL family (DLinear/LSTM/TFT/TabPFN/TabICLv2)

**Decision:** The B-09→B-15 sequence already has a project-wide convention (D-36/D-37): tree models
(RF/XGB/LightGBM) and SARIMAX train their target on `y_gapfilled` (dense, continuous) and are
evaluated against real `y_observed`. The DL family was the odd one out -- DLinear/LSTM/TFT's training
loss was masked to real `y_observed` days only (`mask = np.isfinite(y)` in `forecasting_dl.train_model`,
~45-55% dense depending on tower), even though gap-filled values already feed their AR/encoder
history; TabPFN/TabICLv2 went further and explicitly rejected `y_gapfilled` as context, with each
function's docstring (`recursive_rollout.py`) stating this was deliberate, "avoiding the diffuse
globally-trained-gap-filler optimism flagged for every other model's training target." This
experiment (user-initiated, live discussion) tests that prior rejection empirically: does extending
the tree/SARIMAX convention to the DL family help, hurt, or mostly just inflate apparent skill via
gap-filler mimicry (`y_gapfilled` is itself an RFm regressor's output over met/soil/livestock/mgmt
features that substantially overlap the forecasters' own `fx_` driver set -- a real, previously-named
circularity risk, not a new one)?

**Design** (user-confirmed): evaluation always stays on real `y_observed` (`y_true`/`y_true_tft`,
unchanged) -- only the *fitting* process changes. TFT's validation split (early stopping) also uses
`y_gapfilled`, consistent with its training target (both are "fitting," not final evaluation).
TabPFN/TabICLv2's historical context swaps to `y_gapfilled` too, on the same reasoning (context is
the in-context analogue of training data, not a held-out test set). Full 3-tower x 5-anchor
(2018-2022) coverage for all 5 models per CLAUDE.md's "full coverage by default" rule. Tree
models/SARIMAX are out of scope (already use this convention).

**Implementation:** additive `y_source="observed"|"gapfilled"` param added to `forecasting_dl.
make_windows()`/`build_windows()` (default reproduces prior output bit-for-bit, verified);
`train_model()` needed no changes (masking/loss already generic over whichever `y` array the windows
dict carries). Three new sibling scripts, each mirroring its non-gf predecessor's exact recipe with
only the target/context source changed: `b10_b13_dl_gf_extension.py` (DLinear_gf/LSTM_gf),
`b10_b13_tft_gf_extension.py` (TFT_gf), `b10_b13_foundation_gf_extension.py` (TabPFN_gf/TabICLv2_gf).

**Result -- dense supervision helps the weakest models most, is a wash for the strongest, and a
real (if modest) win for both foundation models' previously-rejected choice.** Pooled (all
towers/anchors) MASE / R2, original -> gf:
- **DLinear: 1.460 -> 1.059 MASE (-0.401), -2.068 -> -0.540 R2 (+1.528)** -- the largest gain by far,
  on this project's single most unstable/worst-performing model (D-53's "worst model" finding).
- **LSTM: 1.151 -> 1.098 MASE (-0.053), -1.357 -> -0.686 R2 (+0.671)** -- clear, smaller gain.
- **TFT: 0.972 -> 1.005 MASE (+0.033), -0.363 -> -0.439 R2 (-0.076)** -- essentially a wash, marginally
  worse pooled -- TFT already had the most-regularized recipe in the roster (D-45's `weight_decay`/
  `patience` regime), so dense-but-noisier-in-effect supervision doesn't help it the way it helps the
  weaker/less-regularized DLinear/LSTM.
- **TabPFN: 0.855 -> 0.829 MASE (-0.026), -0.122 -> 0.001 R2 (+0.123)** -- modest but real gain,
  crossing R2 into positive territory pooled. The "gap-filler optimism" concern that motivated
  TabPFN's original y_observed-only context choice does not show up as a dominant effect here --
  evaluation is still against real y_observed throughout, so this is a genuine (if small) improvement,
  not just agreement-with-the-gap-filler.
- **TabICLv2: 0.930 -> 0.891 MASE (-0.039), -0.330 -> -0.060 R2 (+0.270)** -- similar modest, real gain.

By-tower breakdown (`b10_b13_gf_ablation_table_by_tower.csv`) shows the DLinear/LSTM gain holds at
all 3 towers (largest at Tower 4); TFT is roughly flat-to-slightly-worse at all 3; TabPFN/TabICLv2
are mixed at Tower 2 (small MASE regression) but consistently improve at Towers 4/9.

**Recommendation:** the DL family's original design choice (masked-to-real loss for DLinear/LSTM/TFT,
observed-only context for TabPFN/TabICLv2) was NOT uniformly the right call -- it left real
performance on the table for the weaker/less-regularized models (DLinear especially) and for both
foundation models. It WAS approximately right for TFT specifically. Given this, prefer the
`_gf`-trained DLinear/LSTM/TabPFN/TabICLv2 over their originals if picking a single variant per model
going forward; TFT's choice between variants is a toss-up (roughly equal, use the original for
simplicity). Does not change the overall B-09-B15 ranking conclusion (TFT/TabPFN remain the
strongest models on MASE either way) -- this closes a design-choice question, not a model-selection
one. Cross-ref D-36/D-37 (the original convention this generalizes), D-53/D-54 (DLinear's
already-documented instability, which this substantially mitigates but does not eliminate), D-65
(the `bin_metrics()`/multi-anchor aggregation conventions reused here).

Files added (all committed): `notebooks/05_benchmarking/b10_b13_dl_gf_extension.py`,
`b10_b13_tft_gf_extension.py`, `b10_b13_foundation_gf_extension.py`, `b10_b13_gf_merge.py`,
`b10_b13_gf_comparison_table.py`. `src/models/forecasting_dl.py` extended (`y_source` param,
additive, default-preserving). `results/b10_b13_full_chains.csv` extended in place (+5 `_gf`
columns, 47 cols total, 5,475 rows unchanged; backed up first to
`b10_b13_full_chains_backup_pre_gf.csv`). New: `b10_b13_{dl,tft,foundation}_gf_extension_
{summary,summary_vs_gapfilled,chains}.csv`, `b10_b13_gf_ablation_{combined_summary,pooled,
table_all_towers,by_tower,table_by_tower}.csv`. `results/figures/b10_chains/` regenerated (598
figures total, +75 new `_gf` figures); one spot-checked (Tower 4/anchor 2021, DLinear_gf and
TabPFN_gf) before trusting the full batch. No `benchmarks.csv` rows -- matches every prior
diagnostic/ablation precedent in this sequence (a design-choice ablation, not a point-forecast
benchmark in its own right).

### D-73 -- 2026-07-18 -- IMP-01: opens a revisited, more thorough gap-filling/imputation phase (`08_imputation_revisited/`)

**Decision:** user requested a fresh branch of work revisiting gap-filling in more depth, via a
5-step plan: (1) visualise the feature space and missing values, (2) explore imputation
algorithms, (3) quantify imputation uncertainty, (4) introduce distributional shifts and repeat
1-3, (5) fit predictors with that logic and test against masked data. Scoped with the user
up front on three axes: **feature-space scope = full** (every raw measured column relevant to a
tower, not just the FCH4 target -- broader than any prior gap-filling replication, R-01 through
F-09b, which all curated narrow predictor sets); **relation to prior work = a fresh methodology
going forward, but prior results (R-01-F-09b, best R² 0.40-0.57) are not overwritten/deleted**,
kept as-is and citable; **step 4's distributional shift = standalone synthetic shifts** (not a
reuse of Phase 07's CMIP6/livestock-multiplier scenario machinery), deferred to when step 4 is
actually reached.

**This entry covers step 1 (IMP-01) only.** New reusable helper
`src/features/tower_feature_space.py` assembles the full per-tower column set (69/67/67 cols at
T2/T4/T9) via the existing D-18 spatial-alignment rule (`[Tower N]` + `[Catchment N]` +
`{species}_Catchment N`), tagged into 9 variable families (EC_flux, QC_flag, EC_met, EC_soil,
EC_fetch, Catchment_water, Catchment_soil, Precipitation, Livestock) for analysis/plotting.
New notebook `IMP01_missingness_landscape.ipynb`: missingness inventory, NaN%-by-family bar
charts, a weekly temporal-missingness raster, pooled gap-length distributions per family,
co-missingness clustering (Pearson corr of the missing-indicator matrix), and seasonal/diurnal
missingness patterns -- all 3 towers, full feature space, no narrow-scope shortcuts (CLAUDE.md
"full coverage by default").

**Key findings (full detail: `notebooks/08_imputation_revisited/IMP01_results.md`):**
- **Missingness is strongly block-structured, not independent per-column or MCAR.** Co-missingness
  clustering reveals consistent groups across all 3 towers: a large EC/met/turbulence block
  (near-perfectly co-missing, consistent with shared logger/power failures), a separate footprint
  (EC_fetch) cluster, a separate catchment water-quality cluster, a separate tower-side soil-probe
  (TS/SWC) cluster, and livestock (always complete, 0.0% missing at every tower). This argues for
  block-aware/multivariate imputers in step 2, not naive per-column mean/median fill.
- **Gap lengths are strongly bimodal at every family/tower**: most gaps are a single missing hour,
  but EC_soil specifically has gaps up to ~2,500-2,800 days (multi-year blackouts) -- no single
  imputation method suits both regimes; long blackouts need a proxy/pooled source (as D-16's
  Tower-9-TS-proxy and F-08's external sourcing already do for specific cases) or an explicit
  "not reliably imputable" flag.
- **Real seasonal (May-September elevated) and diurnal (EC_fetch dips in daytime hours) missingness
  patterns** at all 3 towers -- further evidence against MCAR, in favor of imputers that can
  condition on season/time-of-day.
- **Tower-2-specific structural difference**: FCH4/CH4 break out of the main EC co-missingness
  block and cluster instead with soil-probe/precipitation variables -- unlike Towers 4/9, where
  FCH4 sits inside the main EC cluster. Consistent with Tower 2 already being flagged
  project-wide as structurally different (D-15, F-07, D-68) but a new, specific mechanism
  (independent CH4-analyzer failure history) not previously documented.
- Raw (pre-QC) FCH4 NaN%: T2 88.5%, T4 56.5%, T9 75.0% -- consistent with, but numerically
  distinct from, the previously-reported QC-filtered valid% figures (T2 12.1%, T4 44.6%, T9
  25.6%) since these are different filter stages of the same underlying data.

**Files added:** `src/features/tower_feature_space.py`,
`notebooks/08_imputation_revisited/IMP01_missingness_landscape.ipynb`,
`notebooks/08_imputation_revisited/IMP01_results.md`,
`results/imp01_missingness_inventory.csv`, `results/imp01_gap_length_summary.csv`,
`results/figures/imp01_missingness/` (8 figures). No `benchmarks.csv` rows -- this is an EDA/
diagnostic step, not a point-forecast or gap-filling accuracy benchmark.

**Next:** step 2 (imputation algorithms) -- explore candidate imputers informed directly by
this notebook's findings (block structure, bimodal gaps, MAR/MNAR seasonal signal, Tower 2's
FCH4-specific mechanism).

### D-74 -- 2026-07-19 -- F-11: SAITS gap-filling evaluated, tested and not adopted

**Decision:** user asked to (1) retrieve the current best-recorded gap-filling result, then (2)
plan, evaluate feasibility of, and (if feasible) implement SAITS (Self-Attention-based Imputation
for Time Series, Du et al. 2023, via the `pypots` package) as a candidate gap-filler for FCH4,
under `notebooks/04_feature_engineering/F11_SAITS_Implementation.ipynb`.

**Part 1 answer (research only, no new experiment):** current best is the partial-pooled,
external-sourced RFm under full-period gap-CV (D-35/D-49): R² T2=0.574, T4=0.402, T9=0.418
(`results/f09a_summary.csv`, `BEST_RESULTS.md` §1).

**Environment fix required:** `pypots` was not installed; installing it surfaced a real blocker
unrelated to SAITS itself -- `pypots.imputation.__init__` eagerly imports every bundled model
(including `TimeLLM`, which needs `transformers` -> `torchvision`), and the environment's
`torchvision` (0.21.0+cu124) was ABI-mismatched against `torch` (2.11.0+cu128), crashing the
import. Fixed by reinstalling `torchvision` from the matching cu128 index (`0.26.0+cu128`) --
no `torch` version change, and fixed the project's separately-broken `pytorch-lightning` import
as a bonus side effect.

**Methodology (deliberate, stated deviations from RFm's own evaluation):** reused the exact same
`insert_calendar_gaps` held-out-timestamp generator F08 used (identical `SCENARIOS`/`MASK_FRAC`/
`DOMAIN`/seed) so every scored point matches RFm's evaluation point-for-point. But RFm retrains a
fresh model per `(tower, scenario, rep)` (75 cheap fits); SAITS is a deep model, so this
experiment instead trains **one pooled SAITS model** (T2+T4+T9 + tower one-hot channels, matching
RFm's own pooled config) on the **union** of all 25 held-out sets per tower excluded from
training -- zero leakage, but a harder training regime for SAITS than any single RFm run gets
(RFm's per-scenario retrain still had the other 4 scenarios' held-out points available). Feature
set = RFm's EXT-variant channels minus the hand-engineered SWC/TS lag columns (redundant once
SAITS sees raw temporal context via 336h/24h-stride windowing). One architecture, no HPO sweep
(`n_layers=2, d_model=128, epochs=100, patience=10`).

**Phase 1 smoke test (Tower 4 solo, scenario `m`, 1 rep) -- GO:** 35.9s wall time, R²=0.027,
RMSE/MAE in the same physical range as RFm's own numbers (confirms scaling/inverse-transform
correctness); loss still falling at the 20-epoch smoke-test cap, not a red flag.

**Phase 2 full pooled run (all 3 towers x 5 scenarios) result -- SAITS loses at every tower, by
a wide margin, and reproduces closely on an independent rerun:**

| Tower | SAITS median R² | RFm champion R² | delta |
|---|---|---|---|
| T2 | 0.027-0.034 | 0.574 | ~-0.55 |
| T4 | -0.003-0.000 | 0.402 | ~-0.40 |
| T9 | -0.019--0.016 | 0.418 | ~-0.44 |

Full pooled fit took 300.5s (early-stopped epoch 97, best epoch 87) -- compute was never the
constraint. **Diagnosis (not a mystery, not a bug):** every single scenario/tower row has a large
negative MBE (-9 to -35 nmol m-2 s-1) -- SAITS systematically under-predicts. Most likely
compounding causes: (1) FCH4 is far sparser (25-45% valid even before any masking) than SAITS's
design target of a mostly-dense multivariate series with occasional gaps, and the union-mask
design (excluding ~35% of T4's domain from the one training fit) compounds this specifically in
the target channel while covariates stay dense; (2) FCH4's heavy right-skew/spike-dominated
distribution (this project's recurring "MASE<1 alongside near-zero/negative R² = spike-tail
signature," D-44b) means a symmetric-loss model regresses toward the typical/low value rather
than reconstructing rare high-flux events; (3) RFm's structural advantages (explicit high-flux
tree splits on `lsu_dens`/season, hand-engineered 672h lags) aren't automatically replicated by a
small 2-layer attention encoder trained on ~3,900 pooled windows without HPO.

**Outcome: not adopted.** RFm (D-35/D-49) remains the standing gap-filling recommendation -- no
change to `BEST_RESULTS.md` §1. Logged as a legitimate tested-and-rejected alternative (mirroring
B-03a/F-09b's handling elsewhere in this project), not a technical failure -- the pipeline runs
correctly and cheaply (~5 minutes for the full 3-tower run), and the negative result is
diagnosable, not mysterious. 15 rows tagged `F-11` in `results/benchmarks.csv`. See
`notebooks/04_feature_engineering/F11_SAITS_Implementation.ipynb`, `F11_results.md`,
`results/f11_summary.csv`.

### D-75 -- 2026-07-20 -- F-12: bidirectional (lead) soil lags tested for RFm, not adopted

**Decision:** user noted that RFm's champion feature set (`gapfill_rfm.py`) only ever uses
backward-only soil lag features (`swc_l{lag}`/`ts_l{lag}` via `.shift(lag)`, positive lag only),
despite the upstream met/soil driver gap-filling (`reddyproc_pipeline.py`) already using a
bidirectional, centered expanding ±7/14/28/60-day window, and F-11's SAITS being inherently
bidirectional (unmasked self-attention). RFm itself had never been tested with forward-looking
soil context. User asked to devise and run an experiment testing this, choosing (when offered a
2-arm vs. 3-arm design) the more mechanistic **3-arm ablation**:

- **Arm A (baseline)** -- current backward-only lags, rerun fresh in a new notebook for a clean
  same-run comparison point (prior reruns of this protocol have shown small cross-run drift, e.g.
  the F-08 vs F-09a ~0.01-0.014 discrepancy).
- **Arm B (bidir)** -- Arm A's features plus new forward lags `swc_f{lag}`/`ts_f{lag}` via
  `.shift(-lag)`, same `LAG_HOURS=[168,336,504,672]`. Tests whether adding future context helps.
- **Arm C (leadonly)** -- forward lags replacing the backward ones, same feature *count* as Arm A.
  Isolates whether *direction* matters vs. just having more temporal context of any kind.

Everything else (met/fc/AUX/lsu_dens/graze/mgmt/gpp/reco channels, partial pooling T2+T4+T9 +
tower dummies, EXT sourcing, full-period gap-CV) held identical across arms -- a pure ablation on
the swc/ts lag block only. Reused F-08's exact `insert_calendar_gaps`/`dom_mask`/`mets`/
`med_metrics` harness verbatim and `gapfill_rfm.py`'s `load_ext`/`cfg`/`ts_col_for`/`feat_list`/
`frame`/`fit`/`TOWERS`/`LAG_HOURS`/`DUM` via import (never edited -- it's imported live by the
production precompute `build_fch4_gapfilled.py`). Arms B/C are additive-clone `frame_bidir`/
`frame_leadonly` functions, changing only the lag block.

**Data-leakage checks, added specifically at the user's request:** (1) programmatic assertion
that no arm's feature list references the target or any FCH4-derived column -- the lead features
are built strictly from the external per-catchment soil moisture/temperature series; (2) a
permanent runtime assertion inside `run_rf`, checked on every single fit (not a one-off check),
confirming no held-out timestamp survives into that fit's training partition before the pooled
concat. Both passed on all 90 fits. Separately documented (not leakage, a scope caveat): the
forward lag features use soil-sensor readings genuinely observed *after* the reconstructed
timestamp, which is legitimate for gap-filling an already-recorded historical archive but would
**not** be legitimate in a live forecasting deployment -- a gap-filling-only finding.

**Operational incident and fix:** the first attempt at the full F-08 `N_REPS=5` protocol (225
fits, ~3.45h extrapolated from a smoke test) was killed by the environment after ~2h20m with
**zero progress saved**, because `nbconvert --to notebook --execute --inplace` only writes the
output file once the entire run finishes -- a silent all-or-nothing risk for any long notebook
run. Fixed by rebuilding with `N_REPS=2` (90 fits, ~83 min extrapolated, matching F-09a's own
precedent for reducing reps under time pressure while keeping full tower/scenario/arm coverage)
and a cell-by-cell execution driver (`nbclient.NotebookClient`, one `.execute_cell()` call per
cell inside a persistent kernel, `nbformat.write()` after every cell) so the notebook file is
checkpointed to disk continuously -- worst-case loss on another kill is one (arm, tower) pair
(~9 min), not the whole run. This second attempt completed cleanly in 5,261s (~87.7 min).
**General lesson for future long-running notebook executions in this project: prefer the
checkpointed `nbclient` cell-by-cell pattern over a single `nbconvert --inplace` call whenever
expected wall time exceeds roughly an hour.**

**Result: mixed/null, not adopted.** Arm A reproduces the recorded champion exactly at all 3
towers (R² 0.574/0.402/0.418), confirming no regression was introduced by the arms' shared reuse
of `frame_baseline`. Arms B and C both gain a small, tied +0.008 at Tower 4 only, while both
regress at Tower 2 (-0.017 to -0.019) and Tower 9 (-0.010 to -0.011) -- no arm beats the champion
on balance across towers, and the ±0.008-0.019 deltas sit inside the noise band already documented
for this exact protocol (the F-08/F-09a discrepancy was ~0.01-0.014 at T4/T9 alone), widened
further here by the reduced rep count. Feature-importance diagnostic (`native_importance_tree`,
one clean refit per arm): lead columns are **not ignored** by the RF -- real, non-trivial
importance mass in both Arm B (0.099) and Arm C (0.123), comparable to Arm A's backward-lag mass
(0.107) -- ruling out a "leads get ignored" explanation for the null result. The more likely
explanation: soil moisture/temperature autocorrelate strongly in both time directions, so forward
lags carry largely redundant, not novel, information relative to backward lags at the same
offsets.

**Outcome: not adopted.** RFm (D-35/D-49) remains the standing gap-filling recommendation -- no
change to `BEST_RESULTS.md` §1 numbers (one-line confirmed-null-result note added instead, per the
F-09b precedent). Logged as a legitimate tested-and-rejected alternative, not a technical failure.
45 rows tagged `F-12` in `results/benchmarks.csv`. See
`notebooks/04_feature_engineering/F12_bidirectional_soil_lags_RFm.ipynb`, `F12_results.md`,
`results/f12_summary.csv`.

### D-76 -- 2026-07-21 -- F-11 follow-up: testing SAITS's diagnosed failure modes, real gains found (still not adopted)

**Decision:** direct follow-up to D-74 (same F-11 experiment). User asked to discuss why SAITS
lost so badly, then to "test all of this out" against the three diagnosed causes (union-mask
sparsity, FCH4's spike-dominated skew, no HPO). Five levers tested in
`F11_SAITS_Implementation.ipynb`, staged cheapest/most-diagnostic first, each stage's continuation
decided by its own result. All runs seeded (`torch.manual_seed`) after confirming D-74's original
result had drifted run-to-run from unseeded init.

**Infrastructure note (most of the elapsed time, not experiment design):** this session's biggest
cost was repeated background-job interruptions, not code problems -- a stuck run had to be killed
after 10+ hours of no progress (likely a machine sleep/resume cycle corrupting the CUDA context,
confirmed via a fresh-process GPU health check afterward), a subsequent retry hit a real
`CUDA error: unknown error` during `load_state_dict` (transient corruption from that same kill,
resolved by simply retrying in a fresh process), and multiple runs' background-task tracking was
silently dropped by session restarts unrelated to the code (confirmed via direct Windows process
inspection each time, not assumed). None of these were bugs in the experiment itself.

**Results:**
- **EXP_B (per-scenario retraining, attacks sparsity):** confirmed directionally but small --
  flips T4/T9 from negative to positive R² (+0.02 to +0.05), T2 flat. A real contributing cause,
  not the dominant one.
- **EXP_C (solo vs. pooled structural probe):** genuinely noise-level -- flipped winner between
  two independent full reruns (solo won first, pooled won second, by a ~0.001 margin each time).
  Locked in solo for the remaining stages as a defensible choice, not a "winning" one.
- **EXP_D (spike-weighted loss, attacks the skew):** custom `SpikeWeightedMAE` (upweights points
  by `1 + |target|` in standardized units) in place of SAITS's default MAE. **The single largest
  lever in the whole experiment** -- reproduced in both full reruns regardless of solo/pooled
  structure, roughly 5-6x'd R² on its own. Confirms the spike-dominated skew (this project's
  recurring D-44b pattern) was the dominant failure mode, not sparsity or pooling.
- **EXP_E (bigger model, `n_layers=3, d_model=256`):** a further consistent, real gain on top.
- **Final confirmation (solo + SpikeWeightedMAE + bigger model, all 5 scenarios, not just the
  cheap probe scenario):**

| Tower | Final median R² | Original naive baseline (D-74) | RFm champion | Gap now (was) |
|---|---|---|---|---|
| T2 | 0.192 | 0.02-0.03 | 0.574 | -0.38 (was -0.55) |
| T4 | 0.225 | -0.03 to 0.03 | 0.402 | -0.18 (was -0.43) |
| T9 | 0.110 | -0.02 to 0.04 | 0.418 | -0.31 (was -0.44) |

**Outcome: still not adopted -- RFm (D-35/D-49) remains the standing gap-filling recommendation,
no `BEST_RESULTS.md` change** -- SAITS loses at every tower even after this pass. But the gap
narrowed substantially (T4's more than halved), a qualitatively different picture from D-74's
naive baseline. **Not uniform**, reported honestly rather than smoothed over: T2's `l` (288h)
scenario actually went negative (R²=-0.073, MBE=+36.3 -- the one case with a large *positive*
bias, opposite the systematic under-prediction seen everywhere else). If SAITS is revisited again,
the priority is the loss objective (a proper quantile/focal-style loss) and a real architecture
search -- that's where the confirmed gains came from; sparsity fixes and the solo/pooled
structural choice are lower priority. 15 rows tagged `F-11-phase4` in `results/benchmarks.csv`.
See D-76, `notebooks/04_feature_engineering/F11_SAITS_Implementation.ipynb`, `F11_results.md` §8,
`results/f11_phase4_final_summary.csv`.

### D-77 -- 2026-07-22/23 -- `03c_gap_filling_revisited`: self-contained reproduction notebook + a real `mdc_gapfill` fix (new champion R²)

**Decision:** built `notebooks/03c_gap_filling_revisited/temp_gap_filing_exploration.ipynb`, a
fully self-contained (zero `src/` imports -- every function replicated inline, explicit user
constraint) reproduction of the project's best-validated gap-filling result, starting from
`data/Hourly/consolidated_hourly.csv` with its own EDA, FCO2 reconstruction, external-sourcing
build, met/soil gap-filling, u*/GPP-Reco partitioning, management/livestock features, and the full
F-08/F-09a gap-CV harness. Two real bugs caught and fixed while rebuilding independently of
`src/`: a `classify()` substring-collision bug ("inorganic fertiliser" contains "organic
fertiliser" as a literal substring, misclassifying inorganic-N events as manure -- fixed by
restoring the guard clause present in production; negligible R² impact but a real correctness
bug) and an `N_REPS` mismatch (first full run undershot the target -- root-caused to
`BEST_RESULTS.md`'s numbers coming from F-09a's reduced-scope `N_REPS=2` re-check, not F-08's
original `N_REPS=5`; switching to 2 gave an exact match, confirming this was legitimate Monte
Carlo variance from rep count, not a bug).

**A genuine, validated fix found via this independent rebuild:** `mdc_gapfill()`'s interpolation
step used a flat `limit=2` (hours) for every driver, appropriate for variables with strong diurnal
structure (its mean-diurnal-course fallback covers the rest) but not for low-diurnal-structure
variables (soil moisture, soil temperature, TA, VPD, WS) where a short interpolation cutoff
forces a weaker MDC/median fallback unnecessarily often. **Fix:** extended `limit=288` (12 days)
specifically for `{TA_0_0_1, VPD_0_0_1, Soil Moisture @ 10cm Depth (%), Soil Temperature @ 15cm
Depth (oC), WS_0_0_1}` via a new `LONG_INTERP_VARS` set, leaving every other driver's `limit=2`
unchanged. Validated end-to-end (not just at the driver level) by rerunning the full FCH4
gap-CV harness, all 3 towers: **R² improved at every tower** (T2 0.574->0.576, T4 0.402->0.404,
T9 0.418->0.426) -- **new standing champion numbers**, `BEST_RESULTS.md` §1 updated. Model caching
infrastructure was also added during this rebuild (`_model_cache/`, MD5 hash of feature list +
training bytes + RF hyperparameters as key) after discovering `RandomForestRegressor(n_jobs=-1)`
is **not** bit-reproducible across separate process runs even with `random_state` fixed (confirmed
via a controlled two-process test) -- the one uncached RF (FCO2 reconstruction) was switched to
`n_jobs=1`, which fixed it and dropped full-notebook rerun time from ~25-40 min to ~3.5 min.

**Outcome: adopted.** New standing gap-filling champion: **R² T2=0.576, T4=0.404, T9=0.426**
(partial-pooled external-sourced RFm, full-period gap-CV -- methodology unchanged from
D-35/D-49, only the upstream feature gap-filling improved). See D-77,
`notebooks/03c_gap_filling_revisited/temp_gap_filing_exploration.ipynb`.

### D-78 -- 2026-07-23/24 -- Extended exploration on the D-77 base: UQ, six more models, lag/lead expansion -- champion unchanged, all additive

**Decision:** continuation of D-77's rebuild, in a separate working copy
(`temp_gap_filing_exploration copy.ipynb`), per an explicit user request to (1) add UQ, (2) test
more models (BI-LSTM/TabPFN/TabICL/SAITS explicitly named, plus recommendations), (3) expand
lag/lead features for both covariates and the target -- evaluated/criticized before building
anything (plan mode). User overrode two of the critical-review recommendations (drop SAITS and
soil-lag re-expansion given F-11/F-12 precedent) with an explicit instruction to keep them anyway,
since the D-77 feature fix changes the base being tested against. Governing principle throughout,
stated explicitly by the user: **strictly additive -- nothing overwrites the champion
(`RFm_pool`/`frame`/`FEATURES`/`fit`) or its cached results**; every new model/variant reports via
its own separately-labeled comparison table. Full 3-tower coverage from the start for every part
(user's explicit choice over single-tower smoke-test staging).

**Phase A -- UQ via Area of Applicability (validated, additive to production):** replicated this
project's own `scenario_hybrid.py` dissimilarity-index formula inline (`StandardScaler`-scaled
nearest-neighbour distance, Tukey IQR-fence threshold from training data's own leave-one-out
distances) rather than the PCA-distance idea floated earlier in conversation, for consistency with
the established S-01/S-03/U-03 convention. Caught and fixed two real bugs before trusting the
result: missing imputation before scaling/distance (propagated NaN through everything) and
validation-set leakage (the initial validation query points were themselves present, unmasked, in
the training matrix, since production trains on all real data with no gaps withheld -- fixed with
a separate leakage-safe training matrix for validation only, leaving the production-application
step's unrestricted matrix correctly as-is). **Result: weak but real positive correlation with
error** (Pearson +0.11 to +0.16, Spearman +0.15 to +0.19, consistent direction at all 3 towers;
flagged points show 50-80% higher mean error) -- reported honestly as weak, not oversold.

**Phase B -- six additional models tested, full 3-tower, none beat the champion outright:**

| Model | T2 | T4 | T9 | Note |
|---|---|---|---|---|
| LightGBM | 0.522 | **0.410** | 0.422 | Edges champion at T4 only (+0.006) |
| XGBoost | 0.551 | 0.349 | 0.369 | Loses everywhere |
| TabPFN | 0.459 | 0.401 | 0.402 | Loses everywhere; extremely slow (~3.7h for 60 folds -- see operational note below) |
| TabICL | 0.558 | **0.423** | 0.364 | Edges champion at T4 only (+0.019); ~2 min for the same 60 folds |
| SAITS | 0.358 | 0.293 | 0.285 | F-11's own best config (solo, per-scenario retrain, `SpikeWeightedMAE`, `n_layers=3/d_model=256`), rebuilt on D-77's corrected features -- **substantially improved vs. F-11's own numbers** (was 0.192/0.225/0.110) but still loses everywhere |
| BI-LSTM | 0.237 | 0.155 | 0.146 | Custom 2-layer bidirectional LSTM, self-supervised masked-imputation training task (own random ~20% additional masking + spike-weighted loss, same 336h/24h windowing as SAITS); loses everywhere, weakest of all six |

**Operational note (real cost, not a bug):** TabPFN's 60-fold run genuinely took ~3.7h (confirmed
via a controlled rerun after the first attempt hit a 90-minute per-cell nbconvert timeout with
zero visible progress -- `nbconvert --inplace` does not stream cell `print()` output to its own
log, only writes the finished notebook once at the end, so a slow fold is indistinguishable from a
hang without instrumentation). Fixed by adding per-fold progress logging to `run_model()` and,
separately, **result-level caching for every Phase B/C model** (skip-if-already-in-
`model_comparison.csv`) after discovering that `nbconvert --execute` unconditionally reruns every
cell on every invocation -- without this, each new phase added to the notebook would silently
re-pay the full cost of every previous slow phase on every subsequent full-notebook rerun.

**Phase C -- lag/lead feature expansion, both negative:**

| Experiment | T2 | T4 | T9 | Note |
|---|---|---|---|---|
| Soil-lag bidir (F-12's Arm B, rebuilt) | 0.561 | 0.410 | 0.415 | Reproduces F-12's null result almost exactly on D-77's corrected features |
| Soil-lag leadonly (F-12's Arm C, rebuilt) | 0.564 | 0.412 | 0.411 | Same as above |
| Target (FCH4) lag/lead (`target_lag{1,24,168}`/`target_lead{1,24,168}`, new) | 0.495 | 0.329 | 0.353 | Never tested before this session; clear regression, larger than any other variant tested |

Target lag/lead required a genuinely new leakage-safety pattern, not just a feature addition: the
target series is masked at the **current fold's own held-out timestamps first**, and
`target_lag{h}`/`target_lead{h}` are derived via `.shift()` from that already-masked series --
otherwise a held-out point deep inside a contiguous gap could have its lag/lead feature reference
a *neighbouring* held-out point's true value (unlike the covariate lags, which have no such
circularity since covariates are never the modelling target). A runtime assertion checks this on
every fold (mirroring F-12's own held-out/training-overlap check) -- passed cleanly throughout, no
leakage found. The regression is plausibly explained by FCH4's own sparsity (12-45% observed
depending on tower): after fold-masking, `target_lag`/`target_lead` are frequently NaN too, so the
RF leans on a sparse, often-mean-imputed feature block rather than gaining real signal.

**Outcome: not adopted (Phases A/B/C all additive, no `BEST_RESULTS.md` change beyond D-77's own
fix).** RFm (D-77) remains the standing gap-filling recommendation at every tower. TabICL and
LightGBM are the only challengers that beat it anywhere, both only at Tower 4
(+0.019/+0.006) -- not enough to unseat the champion project-wide. The UQ layer (Phase A) is
validated and additive to the existing production pipeline. See D-78,
`notebooks/03c_gap_filling_revisited/temp_gap_filing_exploration copy.ipynb`,
`notebooks/03c_gap_filling_revisited/_data/model_comparison.csv`,
`notebooks/03c_gap_filling_revisited/_data/soil_lag_results.csv`,
`notebooks/03c_gap_filling_revisited/_data/target_laglead_results.csv`.

### D-79 -- 2026-08-02 -- `03c_gap_filling_revisited/temp_gap_filling_pipeline.ipynb`: MDS literature fix, HyperImpute, all-model production fill, TICA/UMAP feedback-feature line (D5-D7, all negative), TabICL row-cap bagging (D8, one real gain) -- TabICL-solo now edges the RF champion at 2 of 3 towers

**Decision:** a long, multi-part session on a third, parallel notebook
(`temp_gap_filling_pipeline.ipynb`, a tidied condensed rebuild, distinct from D-77/D-78's
`temp_gap_filing_exploration[ copy].ipynb`; its own champion reproduces the same 0.576/0.404/0.426
numbers). Work fell into two arcs: (1) closing out known gaps in the reference-baseline table and
production tooling, and (2) a new "feed dimensionality-reduction/uncertainty signal back into the
model" experiment line (D5-D8), run additively on top of the champion feature set throughout.

**MDS literature-correct fix (ported from a separate audit notebook, `temp_mds.ipynb`):** the
production `mds_fill_batch` had three real bugs vs. the actual Reichstein (2005)/REddyProc
algorithm -- an hour-of-day restriction wrongly applied to every case instead of only the final
fallback, a missing intermediate SW-only-lookup case, and a crude fixed-window fallback instead of
the real expanding mean-diurnal-course search. Fixed (3-case hierarchy), and confirmed **Tower
2/4/9 are the literal same physical sites as Zhu et al. (2023a)'s ROTH_HS/PP/HSC farmlets**, not
merely analogous ecosystems (farmlet colour-coding + Tower 2's 2018-cattle/2019-no-cattle regime
shift matching their documented April-2019 management change, independently confirmed). This
enabled a genuine metric-definition discovery: this project's standing `r2_score` (unbounded
below) diverges sharply from Zhu et al.'s own stated R² (squared Pearson r via OLS regression,
bounded [0,1]) **specifically for MDS** (+0.30 gap for the old implementation, still +0.12 after
the fix) but not for any RF/TabICL/MICE model (+0.001 to +0.05 gap, essentially none) -- both old
and new MDS land close to Zhu et al.'s published ~0.03-0.05 figure once measured their way,
validating the reconstruction on the literal same sites. `mets()`/`med_metrics()` now report
`R2_OLS`/`OLS_slope` alongside the standing sklearn R² notebook-wide.

**HyperImpute added as a third imputation baseline** (same champion `FEATURES`, van der Schaar
lab's per-column-AutoML chained-equations imputer vs. MICE's one fixed `BayesianRidge`): **R²=
0.509/0.336/0.354, a 0.25-0.43 jump over MICE (0.081/0.118/0.107) on identical inputs** -- the
single largest "same features, smarter model" gain in this table, closing most of the remaining
gap to the champion (within 0.07 R² everywhere).

**Production gap-filled series generalized from champion-only to all 16 models this notebook has
evaluated** (`_data/fch4_gapfilled_all_models.csv`, long format, 1,669,680 rows, domain-restricted
per tower) -- reused cell 159's existing no-holdout production-fit pattern rather than building
new machinery; one real bug fixed along the way (TabICL's unbatched whole-domain prediction hit a
21.6GB CUDA OOM for D4's wide feature set -- fixed via 2,000-row prediction batching).

**D5-D8: a new "feed derived signal back into the model" experiment line, evaluated honestly
including three straight negative results before one narrow positive:**

| # | What was tested | Result |
|---|---|---|
| D5 | Fold-safe consensus TICA/UMAP/t-SNE feature *selection* (100%-stable 6-variable core found) + leak-free environmental-KNN features, additive on RF/TabICL | KNN features: wash/mild negative (RF -0.007 to -0.012, TabICL near-neutral). **Separately found: TabICL-solo beats TabICL-pooled at every tower** (+0.005 to +0.118, tracks inverse domain size) -- TabICL's fixed 10,000-row context cap means pooling dilutes it, unlike RF; adopted as this notebook's standing TabICL default going forward |
| D6 | Supervisor's idea: feed TICA's *transformed components* (not just loadings) + native per-point model-uncertainty width (RF tree-spread / TabICL quantile API) back in as features, full 4-arm ablation | TICA-as-feature: wash. Uncertainty-as-feature: **actively harmful**, severely so for TabICL at 2 of 3 towers (-0.35 to -0.40 R²) -- plausible cause: leak-free inner-CV widths are noisy on TabICL's already data-constrained context |
| D7 | TabICL-only (RF treated as exhausted): drop D5's 2 least-reliable features, swap in TICA components in their place, and ensemble already-computed RF+TabICL+HyperImpute predictions (no new RF fitting) | Feature changes: flat/slightly negative everywhere. Ensembling: TabICL-solo alone still wins at 2 of 3 towers; only Tower 4 gains from an ensemble (+0.017) |
| D8 | TabICL-only: native hyperparameter sweep (`n_estimators`/`norm_methods`/`feat_shuffle_method`/`outlier_threshold`) + row-cap bagging (k independent random-subsample fits, averaged) | Hyperparameter sweep: clean null, defaults already well-chosen. **Bagging: a real, mechanistically-explained gain specific to Tower 4** (+0.012-0.013 R², plateaus by k=5-8) -- the one tower whose domain most exceeds the 10,000-row context cap (~18% coverage); flat/negative at T2 (already fits under the cap) and T9 (intermediate coverage) |

**Headline finding, not yet operationally adopted:** **TabICL-solo on the plain champion
`FEATURES` beats the RF champion at Tower 2 (0.676 vs. 0.576, +0.100) and Tower 4 (0.428 vs.
0.404, +0.024), and is within noise at Tower 9 (0.423 vs. 0.426, -0.003)** -- the first result in
either `03c_gap_filling_revisited` notebook to genuinely beat RFm at more than one tower. Flagged
as a validated **benchmark** result, not yet an adopted production swap: no UQ/production-fill
tooling exists for this exact TabICL configuration yet (all of that infrastructure is RF-based),
and TabICL's fixed context cap/GPU dependency are real operational differences from RF worth
weighing before a production switch.

**Operational note (not a modelling finding):** `_model_cache/` (RF's MD5-keyed joblib cache) had
silently grown to **166 GB** over the project's iterative history -- confirmed via inspection to be
a pure, fully-regenerable speed cache (zero unique data, gitignored, untracked) rather than
anything a stale-entry accumulation across abandoned experiment variants. Deleted outright, then a
full top-to-bottom rerun of all 314 cells (cold cache, ~8h36m, zero errors) rebuilt it to a leaner
**77 GB** -- the genuine footprint of everything the *current* code needs. Added `_LIVE_CACHE_KEYS`
tracking + a `prune_stale_cache()` utility (run as this notebook's final cell, safe only after a
genuine full top-to-bottom pass) so future stale entries from superseded experiments get swept
automatically instead of accumulating forever.

**Outcome: strictly additive, RFm remains the production-adopted config at every tower** (its own
numbers unchanged: T2/T4/T9 = 0.576/0.404/0.426). MDS's corrected baseline and HyperImpute now sit
alongside it in the reference table; TabICL-solo is flagged in `BEST_RESULTS.md` as a validated
benchmark-best at 2 of 3 towers, worth a real production/UQ-tooling investment if pursued further.
See D-79, `notebooks/03c_gap_filling_revisited/temp_gap_filling_pipeline.ipynb`,
`notebooks/03c_gap_filling_revisited/temp_mds.ipynb`,
`notebooks/03c_gap_filling_revisited/summary.md` §12-18,
`notebooks/03c_gap_filling_revisited/_data/{reference_baseline_met_only,fch4_gapfilled_all_models,
d5_tabicl_solo_results,d6_results,d7_results,d8_hp_results,d8_bagging_results}.csv`.

---

### D-80 -- 2026-08-02 -- Forecasting: does D-79's improved gap-filling help the standing forecasting champion? (no) -- plus a MASE baseline convention change (persistence -> climatology)

**Decision:** direct follow-up to D-79, testing whether TabICL-solo's gap-filling improvement
(beats RFm at Tower 2/4 on the gap-filling benchmark itself) carries through to the downstream
forecasting phase. Built the wide-format `fch4_gapfilled_tabicl.csv` (drop-in schema match to
`fch4_gapfilled.csv`) and its `forecast_daily_v2/v3_tabicl.csv` daily-matrix siblings (reusing
`build_forecasting_matrix_v2.py`/`v3.py`'s guide-feature functions unchanged, since none of that
logic depends on which model produced the gap-fill), then reran the standing champion
(`TabPFN+species`, F-10/D-67) against the revised data.

**First attempt was a null result by construction, caught before being reported as real.** The
standing `tabpfn_forecast()`/`tabicl_forecast()` (`recursive_rollout.py`) deliberately use
`y_observed` (not `y_gapfilled`) as historical context -- confirmed by direct inspection that
model MAE/R² were **bit-identical** between the RF-sourced and TabICL-sourced reruns, at every
tower/anchor/bin. Mechanistically correct: neither model's inputs (`y_observed` + `fx_` guide
covariates) depend on which gap-filler produced `y_gapfilled` at all -- only the MASE
denominator's anchor value did, which is why an initial (wrong) reading looked like a real
improvement before this was checked.

**The actual right test: D-72 (2026-07-15) already found TabPFN/TabICLv2 genuinely improve when
given `y_gapfilled` (not `y_observed`) as context, but never combined that with F-10's feature
families or tested it against a second gap-fill source.** Built two new sibling scripts
(`b16_foundation_models_v3_gf.py`, `b16_foundation_models_v3_tabicl_gf.py`) combining both: F-10's
7-config feature sweep, gap-filled context, both gap-fill sources. This time predictions
genuinely differed between sources (confirmed before the full sweep). **Result: TabICL-sourced
context makes forecasting *worse*, consistently, across every config and both models** --
TabPFN_gf (BASE+species): RF-sourced 0.840 vs. TabICL-sourced 0.893 (persistence-scored); same
direction under climatology-scored MASE (0.722 vs. 0.769). TabICLv2_gf shows the same pattern,
larger in magnitude (0.882 -> 0.984 persistence-scored). Plausible mechanism: TabICL's per-tower-
solo, fixed-context-cap architecture (D-79 S15.5) optimizes point accuracy on scattered held-out
gaps, not necessarily the smoothness/internal-consistency of the resulting daily series -- RF's
gap-fill may be a worse point-accuracy estimator but a more usable dense AR *context* for a
downstream in-context forecaster. **Outcome: the standing forecasting champion is unchanged**
(`TabPFN+species`, RF-sourced gap-filling) -- D-79's better gap-filling is a real, useful result
for gap-filling itself, but does not transfer to forecasting, and combining D-72's gf-context
finding with F-10's feature families doesn't beat D-72's own original number either, so no new
champion emerged from this search.

**Separately, a real methodology change, user-directed:** questioned whether `chain_persistence()`
is a valid MASE baseline at all. D-71 (2026-07-15) had found persistence's own error is
empirically *lower* than climatology's pooled (independently re-verified here from
`results/b10_b13_climatology_mase_table_all_towers.csv` -- confirmed again, 11/11 models score
"better vs. climatology," meaning climatology's own MAE is the higher of the two everywhere).
**User's explicit override: a baseline's conceptual validity matters more than which one happens
to have lower error, and holding one anchor-day value flat across an entire 365-day rollout,
blind to seasonality, is not a real naive forecaster regardless of its error magnitude.**
Climatology (`rr.doy_climatology()`) adopted as the new MASE denominator going forward
(`CLAUDE.md` updated). Recomputed the full B-09->B-15 headline directly from
`results/b10_b13_full_chains.csv`'s already-saved raw per-day predictions (no model refitting
needed -- the file already carries a `Climatology` column from D-71) via `rr.bin_metrics()`,
unmodified, just swapping `y_persist`:

| Rank | Model | MASE (climatology) | R² |
|---|---|---|---|
| 1 | TabPFN_gf | 0.722 | −0.008 |
| 2 | TabPFN | 0.724 | −0.081 |
| 3 | TabICLv2 | 0.771 | −0.286 |
| 4 | TabICLv2_gf | 0.774 | −0.054 |
| 5-6 | Ensemble (both variants) | 0.803 | −0.19 |
| 7 | XGB | 0.805 | −0.220 |
| 8 | LightGBM | 0.817 | −0.234 |
| 9 | TFT | 0.834 | −0.345 |
| 10 | RF | 0.842 | −0.270 |
| 11 | TFT_gf | 0.844 | −0.409 |
| 12 | SARIMAX | 0.875 | −0.376 |
| 13-16 | DLinear_gf/LSTM_gf/LSTM/DLinear | 0.916-1.232 | worst |

(BASE config only -- this table predates F-10's feature families.) **The ranking is essentially
unchanged by the convention switch** -- foundation models still win by a wide margin,
ensembles/trees mid-pack, DL worst -- this is closer to a uniform rescaling (climatology being an
easier baseline to beat than persistence, pooled) than a reshuffle.

**Full table completed (same session, no reruns needed) -- 55 model/config combinations, every
family this project has tested, rescaled to climatology.** Realized `bin_metrics()` always saves
raw `MAE` (not just the `MASE` ratio) alongside `n`/`tower`/`anchor_year`/`bin` in every existing
summary CSV -- so `MASE_climatology = MAE_model / MAE_climatology` is pure arithmetic on
already-saved files (merge in the climatology baseline's own MAE per tower/anchor/bin, no model
ever needs to refit or re-predict). Covers the tree/SARIMAX F-10 sweep (`BASE+ALL` only -- the one
config carried to full rollout for that family, per the Stage-1 signal check's null result),
DL (TFT/DLinear/LSTM, all 7 configs), and both foundation-model context variants (TabPFN/TabICLv2,
observed- and gap-filled-context, all 7 configs). **Confirmed headline: `TabPFN+BASE+species`
(0.715) is effectively tied with `TabPFN+BASE+bodyweight` (0.715)** for best overall -- both
comfortably ahead of the next cluster (`TabPFN_gf` variants, 0.716-0.722). Full ranking shape
matches the persistence-scored one exactly (foundation models -> ensembles/trees -> TFT/SARIMAX ->
LSTM -> DLinear, worst). Source: `results/b09_b16_climatology_mase_full_table.csv`.

**Outcome:** `BEST_RESULTS.md` §3 headline updated (0.840 -> 0.715); `CLAUDE.md`'s MASE section
updated to state climatology as the standing denominator. Per this project's own "apply going
forward" convention (CLAUDE.md), the many other persistence-scored MASE figures throughout
`BEST_RESULTS.md`/`DECISIONS.md`'s forecasting sections were **not** retroactively rewritten --
only the headline. See D-79, D-71, D-72, D-67,
`notebooks/05_benchmarking/b16_foundation_models_v3_{gf,tabicl,tabicl_gf}.py`,
`src/data/build_fch4_gapfilled_tabicl.py`,
`src/features/build_forecasting_matrix_{v2,v3}_tabicl.py`,
`results/b16_foundation_models_v3_{gf,tabicl,tabicl_gf}_summary.csv`,
`results/b09_b16_climatology_mase_{recompute,headline,full_table}.csv`.

---

### D-81 -- 2026-08-03 -- TabPFN v2-vs-v3 A/B: does the TS-finetuned v3 checkpoint actually beat the plain generic v2 checkpoint?

**Decision:** the standing champion (`TabPFN+species`) already used TabPFN v3 by default --
confirmed by direct package inspection: `tabpfn_time_series` (1.2.0) LOCAL mode silently resolves
to a TS-finetuned v3 checkpoint (`tabpfn-v3-regressor-v3_20260506_timeseries.ckpt`, "TabPFN-TS-3
ship config from the TabPFN-3 paper") whenever no `model_path` override is given, and every call
site in this project (`recursive_rollout.py::tabpfn_forecast()`) passes none. User confirmed
(AskUserQuestion) the actual ask given this: an **explicit v2-vs-v3 A/B**, forcing the OLD,
generic tabular v2 checkpoint (`tabpfn-v2-regressor.ckpt` -- what the original TabPFN-TS paper's
approach ran, before any TS-specific finetuning existed) and comparing against the existing v3
results, to test whether v3's finetuning provides a real, decisive edge.

**Implementation, additive only:** `tabpfn_forecast()` gained one new optional kwarg
(`tabpfn_model_config=None`, forwarded to `TabPFNTSPipeline` only when given -- default preserves
byte-identical behavior for every existing caller) plus a small `tabpfn_v2_model_config()` helper.
Confirmed via `TabPFNRegressor.create_default_for_version()` that v2 and v3 share identical
`n_estimators=8, softmax_temperature=0.9` defaults in this installed package version, so
`model_path` alone is sufficient for a fair, complete "v2 as intended" comparison -- no other
regressor kwargs need to differ. Two new sibling scripts
(`b16_foundation_models_v3_tabpfnv2.py`, `_tabpfnv2_gf.py`), same 7-config x 3-tower x 5-anchor
sweep as the standing champion's own script, TabICLv2 deliberately excluded (irrelevant to a
TabPFN-version question, avoids an unnecessary rerun).

**Smoke test caught a real "looks dramatic but isn't representative" trap.** At Tower 4/anchor
2021/BASE+species, the 1-7-day lead-time bin showed R²=-24.926 (v2) vs. -0.887 (v3) -- a huge,
alarming-looking gap that would have suggested v3 is decisively better. **At full 3-tower x
5-anchor scope this washes out completely** -- confirms this project's standing "verify at full
scope before concluding" lesson (S-01/U-03/B-10 all hit variants of this before), this time at
the smoke-test-vs-full-sweep boundary specifically rather than single-tower-vs-all-towers.

**Full result (climatology-scored, D-80 convention, all 28 model/config combinations): v2 and v3
are essentially tied.** Deltas (v2 minus v3) range only ±0.004-0.008 across every config, both
directions -- far smaller than any other effect found this session (the TabICL-sourced-context
effect, D-80, moved MASE by 0.05-0.10 by comparison). Headline config (`BASE+species`, observed
context): TabPFN(v3)=0.715 vs. TabPFN_v2=0.720 -- v3 marginally better, gap tiny. Curiosity: the
single best cell in the whole 28-row table is actually `TabPFN_v2/BASE+ALL` at **0.712**,
marginally beating the standing champion's 0.715 -- but given the entire table spans only
0.712-0.728, this is within noise, not a robust new champion. Under gap-filled context, the
pattern is mixed (v3_gf usually marginally better, v2_gf wins at 2/7 configs) -- same
noise-level-tie picture.

**Outcome: no change to the standing champion or to any project convention.** TabPFN v3's
TS-specific finetuning does not show a clear, decisive advantage over the plain generic v2
checkpoint on this task -- contrary to what "newer model" framing alone would suggest, but
consistent with v2 and v3 sharing identical `n_estimators`/`softmax_temperature` defaults (the
finetuning difference is entirely in the checkpoint weights, and apparently doesn't move this
particular spike-dominated, small-sample forecasting task much either way). Not worth switching
away from v3 (still the marginal favorite at the champion config, and the environment's own
default), but also not worth citing v3 as a proven upgrade over v2 for this task specifically.
See D-80, D-67, `src/models/recursive_rollout.py` (`tabpfn_v2_model_config`),
`notebooks/05_benchmarking/b16_foundation_models_v3_tabpfnv2{,_gf,_vs_v3_compare}.py`,
`results/b16_foundation_models_v3_tabpfnv2{,_gf}_summary.csv`,
`results/b16_foundation_models_v3_tabpfnv2_vs_v3_compare.csv`.

---

### D-82 -- 2026-08-06 -- S-04 analyzed: a completed-but-undocumented realization-level/SSP5-8.5
scenario trajectory (built 2026-07-15/16) was sitting unwritten -- analysis closes S-01's queued
extensions 1 and 2

**Context:** a full repo-familiarization pass found that `s04_trajectory_2050.py` and
`s04_daily_top3_2050.py` (committed 2026-07-15/16 as `777cf89`/`ea6530f`, both commit messages
naming "S04") had already run to completion -- `s04_trajectory_realizations.csv` (234,000 rows,
primary hybrid, both SSPs), `s04_trajectory_realizations_b10benchmark.csv` (28,080 rows, B-10
diagnostic benchmark, both SSPs), `s04_aoa_by_year.csv` (4,680 rows), `s04_daily_top3_2050.csv`
(5.1M rows) and 180 chain figures in `results/figures/s04_chains/` all existed, complete and
non-partial (row counts verified against the expected combinatorial totals) -- but neither commit
touched `DECISIONS.md`, `BEST_RESULTS.md`, or updated `CONTEXT.md`'s "queued next" list, which still
read S-01 as current and listed "extend to SSP5-8.5" / "realization-level spread" as outstanding.
This was a real end-of-session documentation gap (CLAUDE.md's own session-workflow requirement),
not a re-run of anything -- **S-04's design already delivers exactly what those two queued items
asked for**: SSP2-4.5 AND SSP5-8.5, full realization scale (500/SSP, 5 GCMs x 100, for the primary
hybrid; a stratified 10-realization subset for the B-10 diagnostic benchmark, a tracked deadline-
driven scope cut from an originally-approved 20), and a real annual 2025-2050 transient trajectory
(not a single ensemble-mean climatological snapshot). New `s04_analysis.py` (committed) runs
read-only against this existing output -- no new model fitting.

**Result 1 -- realization-level spread is real but small, and narrows in relative terms as
livestock stress increases.** Pooled across 26 years, the p10-p90 band is ~1.3-4.8% of the mean at
every tower; absolute band width barely changes with the livestock multiplier while the mean
triples, so the relative band *shrinks* (Tower 4: 3.2% of mean at 1x -> 1.3% at 3x). Confirms S-01's
Finding 5 ("climate axis is not the extrapolation risk") at full realization scale rather than a
single point estimate.

**Result 2 -- SSP2-4.5 vs SSP5-8.5 divergence is real, grows toward 2050 as physically expected, but
stays under 1% of the mean even by 2050** (e.g. Tower 4: +0.26% early-window -> +0.64% late-window).
The emission-scenario choice is a minor lever on this project's headline numbers next to the
livestock multiplier.

**Result 3 -- AOA-flagged extrapolation risk does not trend upward across 2025-2050** (no evidence
climate drift alone pushes the scenario further out-of-envelope over the horizon tested) **but sits
materially higher in absolute terms than S-01's own reported numbers, at every multiplier including
the unchanged 1x baseline.** S-01 (single 2041-2060 ensemble-mean snapshot) found 0% flagged at
1x/2x everywhere and only 5.5-6.0% at 3x (T4/T9 only), 0% at every multiplier for T2. Here, using
real transient annual weather rather than a smoothed ensemble-mean composite, **1x baseline is
already flagged 9-15% of days at every tower**, rising to 17-20% at 3x (T4/T9). Read as confirmation
of S-01's own Finding 7 caveat (the AOA check can be diluted/sensitive to construction method) --
smoothing (S-01's ensemble-mean approach) measurably suppresses the flagged rate relative to genuine
transient weather variability (S-04's), independent of the livestock question entirely. **A
genuinely new, non-obvious finding, not just a scope extension**: how "out-of-distribution" a
scenario looks depends materially on whether it's built from a smoothed climatological composite or
real transient weather -- AOA numbers from S-01 and S-04 are not directly comparable to each other.

**Result 4 -- S-01's central finding (the level-residual hybrid fixes U-03's tree-only extrapolation
plateau) holds and is reinforced across the full 26-year x both-SSP trajectory, not just a single
snapshot.** Matched on the shared 10-realization stratified subset (fair, same-inputs comparison),
pooled across both SSPs/26 years, 1x->3x response: hybrid +38.6%/+156.4%/+120.3% (T2/T4/T9) vs. the
B-10 diagnostic ensemble's own +20.4%/+76.6%/+62.0% -- roughly 2x the diagnostic ensemble's response
at T4/T9, consistent in direction and magnitude with S-01 vs. U-03's original single-snapshot
comparison (+138%/+105% hybrid vs. U-03's trees-alone +21-23%; here the diagnostic ensemble's own
+63-77% sits between U-03's trees-alone and both-ensembles figures as expected, since it's the same
4-model RF+XGB+LightGBM+SARIMAX mix SARIMAX's own +150% extrapolation pulls upward). **Genuinely
interesting reversal at baseline**: at 1x (no livestock stress), the diagnostic tree/SARIMAX
ensemble actually predicts *higher* FCH4 than the hybrid at T4/T9 (T4: 27.85 vs 25.66) -- the
hybrid's advantage is specifically in the *rate of change* under scenario stress, not the absolute
baseline level.

**Outcome:** `BEST_RESULTS.md` section 6 updated (S-04 now the current entry, S-01 kept as the
architecture reference). `CONTEXT.md`'s stale "queued next: SSP5-8.5, realization-level spread"
line superseded -- both delivered. **Remaining open items, unchanged from S-01's own list**: a
self-consistent mechanistic livestock-scenario construction (grazing-recency features still not
adjusted jointly with the density multiplier), and SPACSYS as the trend/level component if time
permits before the 1 Sept deadline. See D-64 (S-01, the architecture this extends),
D-63 (U-03, the tree-only extrapolation-plateau finding this reproduces at full trajectory scale),
D-70 (S-03, the driver-availability ablation this is distinct from),
`notebooks/07_scenario_analysis/s04_trajectory_2050.py`, `s04_daily_top3_2050.py`,
`s04_analysis.py`, `s04_results.md`, `results/s04_trajectory_realizations*.csv`,
`results/s04_aoa_by_year.csv`, `results/s04_{trajectory_summary,ssp_divergence,
realization_spread,aoa_trend,aoa_early_vs_late,hybrid_vs_benchmark_raw,hybrid_vs_benchmark_summary,
hybrid_vs_benchmark_response}.csv`, `results/figures/s04_summary/*.png`,
`results/figures/s04_chains/` (180 figures, pre-existing).

---

### D-83 -- 2026-08-08 -- S-03 brought up to speed: climatology MASE (D-80), TabICL-sourced data
(D-79), full 11-model roster wired into the notebook -- plus an unplanned, more consequential
finding about TabICL as a forecasting *training target* specifically

**Context:** user asked for S-03 (D-70, the driver-availability ablation) to be updated three
ways: (1) MASE baseline switched to climatology, matching D-80's project-wide convention change
(persistence -> climatology) that postdates S-03's original run by three weeks; (2) rerun on the
latest imputed data (TabICL-sourced gap-filling, D-79); (3) full model roster (TabPFN/TabICLv2/DL)
-- which already existed as a standalone addendum script (`s03_model_roster_extension.py`,
2026-07-15) but had never actually been wired into `S03_driver_availability_ablation.ipynb`
itself.

**Implementation.** `s03_driver_availability_ablation.py` and `s03_model_roster_extension.py` both
gained a `climatology_baseline()` helper (day-of-year climatology from strictly pre-anchor real
`y_observed`, same recipe as `b10_b13_climatology_baseline.py`) used in place of
`rr.chain_persistence()` for every MASE/RMSSE call; the persistence series is still saved in chain
output for reference, just no longer the denominator. Both scripts also gained a `daily_csv`
parameter so Variant A/B can run against `forecast_daily_v2_tabicl.csv` (confirmed via direct diff
to be schema-identical to `forecast_daily_v2.csv` -- only `y_gapfilled`/`ar_ch4_*` differ, every
`fx_` driver and `y_observed` are byte-identical).

**A design tension resolved along the way, not left implicit.** S-03's "Model 1" was always read
from an existing, never-rerun table (D-65, RF-sourced, persistence-scored). Swapping only Variant
A/B to TabICL-sourced data while leaving Model 1 on the old table would reintroduce exactly the
data-source/feature-availability conflation S-03 exists to avoid (isolating one axis of variation
at a time is the entire point of this experiment). Fixed by recomputing Model 1 too, wherever a
TabICL-sourced daily file makes that possible, via the SAME code path as the real ablation
(`remove_cols=[]`/`resample_cols=[]`/`degraded_cols=[]` collapses both variants to an identical
full-feature/no-substitution config) -- not a separately-written "similar" computation. TFT/
DLinear/LSTM read the hourly `forecast_features_v2.csv`, which has no TabICL-sourced sibling
anywhere in this project (`build_forecasting_matrix_v2_tabicl.py`'s own docstring: "hourly does
not depend on the gap-filled CH4 series at all") -- these 3 stay RF-sourced throughout (Model 1
*and* Variant A/B), an unavoidable data-availability limit, not a gap introduced here. Their
Variant A/B predictions are unaffected by either requested change (climatology is scoring-only;
there is no TabICL data to swap to), so they were rescored from the already-saved chains
(`s03_model_roster_extension_chains_dl.csv`) rather than retrained for a metric-only update.

New committed orchestrator script, `s03_climatology_tabicl_update.py` (~21 min full run, both
halves smoke-tested first): (1) tree/SARIMAX/ensemble Variant A/B on TabICL data, (2) the same
architecture's Model-1-equivalent on TabICL data, (3-4) the same pair for TabPFN/TabICLv2, (5)
climatology-rescoring of the existing DL chains, (6) assembling the full 11-model Model1/VariantA/
VariantB tables for both targets (`results/s03_table_all_towers_climatology_tabicl.csv` /
`_by_tower_climatology_tabicl.csv`). Wired into `S03_driver_availability_ablation.ipynb` as new
cells (loads the pre-computed tables, does not re-execute the 21-minute sweep inline, matching the
project's established script-does-the-heavy-lifting pattern); notebook re-executed clean end to
end via `nbconvert`.

**Result 1 (the two requested changes): the original driver-availability finding replicates
qualitatively unchanged.** For every one of the 11 models, Variant B (resample) still beats or is
within noise of Model 1 on MASE, and the removal/resample ordering matches the original
persistence-scored, RF-sourced run (TFT remains the one genuine reversal; TabICLv2 remains a close
wash). Driver degradation itself still does not cost material accuracy under either the
climatology-baseline switch or the TabICL data swap -- that headline is robust to both changes
tested here.

**Result 2 (unplanned, and more consequential than either requested change): TabICL-sourced
gap-filling is a substantially worse *training target* for the tree/SARIMAX/ensemble family,
well beyond D-80's earlier (softer) finding.** D-80 found TabICL-sourced *context* makes
TabPFN/TabICLv2 forecasting modestly worse (MASE +0.05 to +0.10, persistence-scored). Here,
RF/XGB/LightGBM/SARIMAX/the 2 ensembles -- which train directly ON `y_gapfilled` as their fitting
target, not just condition on it -- show a much larger absolute MASE increase (roughly 1.3-2.5x
the original RF-sourced numbers, e.g. RF Model1 climatology-MASE 2.270 on TabICL data vs. an
apples-to-apples ~0.84-1.06 range on RF-sourced data checked directly in this session) at all 3
towers, not just the towers with sparse real coverage (T4, the best-covered tower, shows the
single largest absolute jump). **Mechanism, confirmed directly (not inferred)**: TabICL's
gap-filled CH4 series sits at a substantially different mean level than RF's at every tower (T2:
54.2 vs. 19.4; T4: 33.7 vs. 30.0; T9: 72.2 vs. 36.7 -- vs. real `y_observed` means of 30.8/29.5/
36.1 respectively) -- training a regression target directly on a differently-calibrated series
propagates that miscalibration into every prediction, a much larger effect than merely using it as
historical *context* for a zero-shot foundation model (which is invariant to this, since
TabPFN/TabICLv2 always condition on real `y_observed`, never `y_gapfilled`). TFT/DLinear/LSTM (no
TabICL data available at all) are entirely unaffected, confirming the effect is specific to models
that fit `y_gapfilled` directly.

**Outcome: independently confirms and extends D-80's own conclusion** ("the standing forecasting
champion is unchanged -- D-79's better gap-filling is a real, useful result for gap-filling
itself, but does not transfer to forecasting"). It now also does not transfer to the tree/SARIMAX/
ensemble family, and the failure mode there is considerably more severe than the
foundation-model-context case D-80 already flagged. **RF-sourced gap-filling remains the right
choice for every model family this project forecasts with** -- no change to the standing
recommendation, but a sharper, independently-derived reason why for the tree/SARIMAX/ensemble
family specifically. No `BEST_RESULTS.md` change (S-03 remains a diagnostic result, not a
production-config input, per D-70's own framing). See D-80, D-79, D-70, D-71, D-37,
`notebooks/07_scenario_analysis/s03_driver_availability_ablation.py`,
`s03_model_roster_extension.py`, `s03_climatology_tabicl_update.py`,
`S03_driver_availability_ablation.ipynb`, `s03_results.md` (second addendum),
`results/s03_table_all_towers_climatology_tabicl.csv`, `s03_table_by_tower_climatology_tabicl.csv`.

**Addendum -- 2026-08-13 -- correction to an explanatory (not tabular) error, caught by direct
user question ("are you sure this is correct? refer back to B16 and B15").** When later asked why
this table's `Ensemble_unweighted` MASE (1.454) looked so much worse than the standing B16/F-10
result (0.809, `results/b09_b16_climatology_mase_full_table.csv`, `BASE+ALL` config), an initial
verbal explanation used that 0.809 number as an "isolate the metric-convention effect alone"
reference point -- **invalid**: 0.809 runs on F-10's v3/`BASE+ALL` feature set, while this entire
addendum's harness (confirmed directly in `s03_driver_availability_ablation.py`) defaults to the
older v2 feature set throughout, so that comparison silently changed two variables (feature set
AND metric convention) instead of one. **The table above and every number in it is unaffected and
correct** -- only the verbal decomposition was wrong. Corrected via a clean, purely arithmetic
same-feature-set (v2) isolation, no new model calls: since `climatology_baseline()` depends only
on real `y_observed` (confirmed byte-identical between the RF-sourced and TabICL-sourced daily
files), its own MAE can be backed out of the already-computed TabICL-sourced+climatology table
and reapplied to the RF-sourced+persistence file's own saved MAE (mirrors D-80's own rescoring
technique exactly). Result, `Ensemble_unweighted`: RF-sourced+persistence 0.936/0.900
(VarA/VarB) -> RF-sourced+climatology 0.736/0.717 (climatology alone genuinely improves it, as
D-80 predicts, and lands in the same healthy ballpark as B16's own 0.809 -- different feature set,
same order of magnitude) -> TabICL-sourced+climatology 1.346/1.250 (climatology held fixed; the
real, isolated TabICL-sourced-data effect). **Confirms the headline finding is not just intact but
better evidenced** -- the 1.3-2.5x degradation is entirely the TabICL-sourced-data swap, not a
contradiction of B16's real (unaffected) production-relevant Ensemble result. Full table (all 6
tree/SARIMAX/ensemble models) in `results/s03_rf_sourced_climatology_isolation.csv`;
`s03_results.md`'s Result-2 section updated with the corrected isolation table and an explicit
cross-reference to B16's own number, to prevent the same confusion for future readers.

---

### D-84 -- 2026-08-08 -- S-05: TabICLv2 + S-03's Variant A + F-10's species split, run as a
10-year transient CMIP6 trajectory with independent per-species livestock multipliers

**Context:** live-discussion follow-up to D-83/S-03. User asked whether S-03's Variant A
(driver-removal) could be run further out than S-03's own 365-day horizon. Key realization,
corrected mid-discussion: TabICLv2 is a one-shot foundation model (same architecture as
`tabpfn_forecast()` -- a single forward pass over however long a target window is given), not a
recursive day-by-day rollout, so extending its horizon doesn't carry a tree/SARIMAX rollout's
compounding-error risk. Initial framing ("we don't have 10 years of real driver data") was wrong
and corrected by the user: `data/Simulated Climate Data/` (Semenov et al. CMIP6/LARS-WG, 2020-2090,
5 GCMs x 3 SSPs x 100 realizations) is exactly the dataset `build_transient_scenario_drivers.py`
already draws from for S-04, and S-03's Variant A feature set (10 columns: TA/SWIN/PRECIP/DOY/
season/livestock) is almost exactly what that CMIP6 data can supply directly -- a clean fit, not a
new data-engineering problem.

**Scope decision, user-confirmed after empirically measuring cost (not estimated).** User wanted
S-05 scoped like S-04 (multi-GCM/realization/SSP sweep) AND wanted independent per-species
livestock multipliers (Option B: cattle/sheep/lamb each scaled separately, 27 combos, not S-01/
S-04's single shared scalar) -- explicitly chosen over the cheaper shared-multiplier option
(Option A, 3 combos) to isolate species-specific effects. A single 10-year `tabicl_forecast()`
call was timed directly before committing to a scope: ~3.8s cold. At S-04's full scale (3 towers x
2 SSPs x 5 GCMs x 100 realizations x 27 combos = 81,000 calls), that would be ~85 hours --
infeasible against the 1 Sept deadline. Presented 4 concrete scope options with measured/
extrapolated wall-clock times via `AskUserQuestion`; user selected **10 realizations/GCM, both
SSPs, full 27 combos** (~8.5h estimated, mirrors S-04's own precedent of cutting realization count
for its B-10 diagnostic when it hit the same problem). Actual runtime: **2.54 hours** (steady-state
per-call rate ~1.2-1.3s, faster than the cold-call estimate) -- 8,100 calls, 0 skipped/failed.

**Implementation, additive only.** New sibling module
`src/features/build_transient_scenario_drivers_species.py` (imports from, does not modify,
`build_transient_scenario_drivers.py`): `FX_A_SPECIES` (13 cols) = S-03's own `FX_A` (10 cols) +
F-10's 3 species-density columns (`fx_cattle_dens`/`fx_sheep_dens`/`fx_lamb_dens`) -- exactly the
"BASE+species" config behind the standing `TabPFN+species` forecasting champion (D-67), applied to
Variant A's narrower BASE. Independent per-species multipliers scale each species' own day-of-year
climatology base (built from the FULL real historical record, matching S-04's own convention, not
S-03's pre-anchor-only restriction -- no leakage risk for a genuinely blind trajectory);
`fx_lsu_dens` is rebuilt as the exact LSU-weighted sum (`1.0*cattle + 0.1*sheep + 0.05*lamb`) under
every combo, preserving F-10's own construction identity. New script
`notebooks/07_scenario_analysis/s05_trajectory_10yr.py` anchors each tower at its own last real
`y_observed` date (T4/T9: 2023-12-29; T2: 2019-05-31, its usual data-scarce anchor -- matches
S-01/S-04's end-of-real-data convention, not S-03's mid-history anchors, since there's no real
future to leak from for a genuinely blind trajectory), precomputes each tower's AOA threshold ONCE
(not per scenario call, which `scenario_hybrid.dissimilarity_index()`'s own naive per-call
recomputation would have made a real cost at 8,100 calls).

**Result 1: species response is highly asymmetric -- cattle dominates far beyond its own
LSU-weight share.** Holding the other two species at 1x, tripling cattle ALONE roughly triples
predicted FCH4 at T4 (+205.6%) and T9 (+195.6%); sheep/lamb stay under 25% even at 3x at every
tower. Cattle contributes 74-88% of real historical `fx_lsu_dens` at these towers, but its share of
the *response* is even higher than that -- the model has learned cattle-specific signal beyond
what the single aggregate `fx_lsu_dens` feature alone could carry, the entire point of F-10's
species split, now shown to hold at CMIP6-scenario scale, not just F-10's own real-historical-
anchor signal check. Tower 2 is muted across every species (max +2.3%) -- consistent with its
known different-regime status (D-18 lineage), not a new limitation.

**Result 2: joint (all-3-species) scaling is close to additive at the best-covered tower, with one
real exception.** T4's response to scaling all three species to 3x together is almost exactly the
sum of each species' individual marginal effect (-0.2% synergy). T9 shows a real +8.8%
super-additive effect. T2's -8.1% is not read as real -- its absolute deltas are too small (0.1-0.5
nmol) for a percentage synergy figure to mean much there.

**Result 3, the most consequential correction this session: realization/GCM spread is small once
correctly isolated, but a first-pass metric mirroring S-04's own pooling convention was
misleading, and this was caught and fixed before being reported, not left as a headline number.**
Pooling year+realization+GCM together (S-04 Finding 1's own convention) gave 32-69% band-width-as-
%-of-mean -- an order of magnitude larger than S-04's 1-5%. Investigated directly rather than taken
at face value: a single fixed (GCM, realization, SSP) already ranges 9.98-15.46 across its own 10
years at T4 -- the pooled number was dominated by genuine year-to-year weather variability, not by
which of the 50 weather sequences was drawn (per-GCM means, pooled across realizations/years, sit
within 13.11-13.29 of each other -- essentially identical). **Isolating the realization/GCM axis
alone (fixed year) gives 2.4-6.6%** -- consistent with S-04's own finding. Read as a genuine,
TabICLv2-specific architectural difference, not a contradiction: S-04's hybrid has an explicit
smooth Ridge trend that damps year-to-year weather noise; TabICLv2 has no such backbone and is
directly, much more strongly sensitive to which specific year's weather it's asked to predict
from. Both `s05_realization_spread_pooled.csv` (conflated) and `_isolated.csv` (corrected) are
kept, not just the corrected one, so the correction itself is visible in the record.

**Result 4: AOA extrapolation risk is high in absolute level (62-68% flagged at every tower) but
flat across the 10-year horizon** -- no evidence risk grows with horizon length, matching S-04's
own Finding 3. The high absolute level vs. S-04's 9-15% is the concrete confirmation of S-04's own
"AOA can be diluted by many in-range dimensions" hypothesis: S-05's feature space is deliberately
narrow (13 columns, Variant A's whole point) vs. S-04's ~40+, so it dilutes less. Second
independent confirmation (S-04 -> S-05) that AOA's absolute flagged-% depends heavily on
feature-space breadth -- any AOA number from this project should be read with its feature-space
dimensionality stated alongside it, not compared raw across experiments.

**Result 5: SSP2-4.5 vs SSP5-8.5 divergence stays under 1% of the mean at every tower/window** --
same order of magnitude as S-04, the SSP choice remains a minor lever relative to the livestock
question.

**Outcome: no change to any standing recommendation** (TabPFN+species remains the forecasting
champion; RF-sourced gap-filling remains correct per D-83; B-10's ensemble / the S-01 hybrid remain
the standing scenario-architecture choices) -- this is a diagnostic extension of S-03/F-10's own
findings to a longer horizon and a species-disaggregated livestock lever, not a production-config
change. Genuinely new, reportable findings: cattle-dominant species asymmetry (beyond its own
LSU-weight share), near-additive joint scaling with one real T9 exception, and the
realization-vs-year-variability decomposition (a methodological lesson worth carrying into any
future TabICLv2-based scenario work: pool year and realization separately, don't conflate them
into one "spread" number). See D-70, D-79, D-80, D-83, D-67, D-64,
`src/features/build_transient_scenario_drivers_species.py`,
`notebooks/07_scenario_analysis/s05_trajectory_10yr.py`, `s05_analysis.py`, `s05_results.md`,
`S05_species_trajectory.ipynb`, `results/s05_trajectory_realizations.csv`.

---

### D-85 -- 2026-08-08 -- S-05 extended to 2050 + full daily chains saved for every scenario point

**Context:** direct follow-up to D-84, same session. Two user requests, addressed together: (1)
extend S-05's horizon from a fixed 10 years to **2050**, matching S-04's own endpoint; (2) save
**full daily chains** for every one of the 8,100 calls, not just the annual mean the original run
kept -- motivated by a smaller interim check (18-call representative subset, saved separately) that
confirmed the within-year seasonal pattern looked physically sensible (T4 baseline: clear summer
peak ~18 nmol m⁻² s⁻¹ in June, winter trough ~3 nmol in December, no negative predictions) and the
user wanted that same daily-resolution view available for the complete grid, not just a subset.

**Cost, measured before committing (not assumed cheap just because it sounds incremental).** A
single 27-year call (T4/T9's new horizon: anchor 2023-12-29 to 2050) was timed directly at 4.07s,
vs. ~1.2-1.3s for the original 10-year call -- the horizon extension, not the daily-chain save,
is what dominates the new cost (T2's horizon is longer still: 2020-2050, 31 years). At the full
8,100-call grid that's ~9h, not the original ~2.5h. Since compute was already the bottleneck
either way, the daily-chain save was folded into the same run rather than done as a separate pass
(I/O is negligible next to inference time) -- storage was measured too: 83,767,500 daily rows
across the full grid, ~6GB as CSV (impractical), ~0.8-1GB estimated as Parquet (pyarrow already
installed, 7.5x compression measured directly on the smaller subset). **Actual runtime: 5.44h**
(faster than the 9h estimate), 0 failed calls, 1.25GB Parquet (written incrementally via
`pyarrow.parquet.ParquetWriter`, never holding the full 83.8M rows in memory).

**New script `s05_trajectory_2050.py`** (sibling to, does not modify, `s05_trajectory_10yr.py` --
both kept, this one is now canonical): same 8,100-call grid, years computed per-tower from anchor
to 2050 instead of a fixed `N_YEARS=10`, same AOA/climatology-base machinery reused unchanged.
Smoke-tested (1 tower, 1 SSP, 1 GCM, 2 realizations, 3 combos, both output formats verified) before
the full run, per the project's own standing convention. Reproducibility spot-checked directly
against the original 10-year run's overlapping year (T4/2024/baseline/ACCESS-ESM1-5/realization1):
9.981 (original) vs. 9.974 (this run) -- 0.07% apart, ordinary GPU inference variance, not drift.

**Result: every finding from the 10-year version (D-84) replicates at the extended horizon, several
more clearly than before.** Cattle-dominance holds and if anything strengthens (T4 3x-alone:
+205.6% -> +214.5%; T9: +195.6% -> +186.4%). Joint-vs-additive holds (T4: -0.2% -> -0.6% synergy;
T9's real super-additive effect: +8.8% -> +9.1%). Realization/GCM spread, correctly isolated, stays
in the same small range (2.4-6.6% -> 2.5-7.6%) -- the pooled-vs-isolated distinction from D-84
remains essential at the longer horizon, since pooling more years together means more year-to-year
weather variability to conflate with realization choice if not separated. **AOA's flatness over
time is now confirmed far more strongly** -- stable within ~1 percentage point across the *entire*
27-31-year horizon (not just the original 10 years) at every tower. **One genuine new pattern that
the 10-year window was too short to show**: SSP2-4.5 vs SSP5-8.5 divergence now visibly grows from
the early to the late window (e.g. T4: +0.09% -> +0.77%), matching S-04's own "widens toward end of
century" finding much more directly -- T9 shows a direction-inconsistent late-window number
(+0.09% -> -0.75%), flagged rather than smoothed over.

**Outcome: no change to any standing recommendation** -- this extends and strengthens D-84's own
diagnostic findings, doesn't overturn or supersede them methodologically (both the 10-year and
2050-horizon outputs are kept on disk and referenced in `s05_results.md`, not one overwriting the
other). New capability going forward: the full daily-resolution Parquet file
(`s05_daily_chains_2050.parquet`) means any future question about within-year seasonal shape, at
any of the 8,100 scenario points, can be answered by querying already-computed output rather than
rerunning the model. See D-84, D-70, D-67,
`notebooks/07_scenario_analysis/s05_trajectory_2050.py`, `s05_analysis_2050.py`,
`s05_daily_chains_subset.py`, `s05_results.md`'s "Update: extended to 2050" section,
`S05_species_trajectory.ipynb`, `results/s05_trajectory_realizations_2050.csv`,
`results/s05_daily_chains_2050.parquet`.

---

### D-86 -- 2026-08-08 -- S-05 extended to farming-practice scenarios: grazing timing and
fertilizer schedule, run as two separate baseline-livestock experiments

**Context:** same-session follow-up to D-84/D-85. User asked to extend S-05 beyond the
livestock-density axis to farming-practice management levers -- specifically grazing practices and
fertilizer addition, both explicitly named. Before building anything, expectations were stated
directly (not fitted after the fact): both axes expected to show a smaller effect than livestock's
cattle result, per F-01/F-04/F-05's repeated "redundant on the rich base" finding for management
features on real historical data -- but the two axes were expected to differ from EACH OTHER:
grazing timing is directly tied to livestock presence (the #1 driver throughout this project's
history), while fertilizer's stronger mechanistic link is to N2O, which this project doesn't
model, not CH4 specifically. The user confirmed this framing before any code was written.

**Design, resolved before implementation.** Unlike livestock density (a continuous quantity
directly scalable by a multiplier), both target features are DERIVED, not directly scalable:
`fx_grazing_active`/`fx_days_since_grazing` come from a presence pattern (any species density > 0
that day); `fx_mgmt_fertN_recency`/`fx_mgmt_fertN_rate` come from a discrete event list run through
an exponential-decay function (`recency_series()`, tau=14 days, `build_management_features.py`).
Resolved by reusing both real construction functions UNCHANGED, applied to synthetic scenario
inputs instead of real data, rather than reimplementing either:
- **Grazing**: the real day-of-year species-density climatology (already built for
  `FX_A_SPECIES`) is phase-shifted at the season edges -- first half of the year sampled at
  `doy+shift_days`, second half at `doy-shift_days` -- extending the shoulder seasons (earlier
  turnout, later housing) without needing per-tower/species boundary-detection logic. Re-derives
  `fx_grazing_active`/`fx_days_since_grazing` from the same shifted presence pattern via
  `days_since_grazing()` (`build_forecasting_matrix_v2.py`), reused directly.
- **Fertilizer**: real per-tower fertN event history (`Field_Event_Data_Format_1.csv`) summarized
  into a "typical year" template (event count, DOY range, mean rate) rather than replaying one
  arbitrary specific year -- checked directly this session: T4 ~8.25 events/yr (DOY 82-234, mean
  127 kg/ha), T9 ~4/yr (DOY 87-204, mean 90 kg/ha), T2 ~5/yr (DOY 55-268, mean 137 kg/ha). Levels
  scale the template's rate or event count; `recency_series()` run over real pre-anchor events +
  the synthetic future schedule, same decay function the real columns use.

Both held at BASELINE (1x/1x/1x) livestock and run as two SEPARATE experiments (not stacked onto
the 27-combo livestock grid) -- matches the project's "isolate one axis at a time" convention (same
reasoning as S-05's own species-marginal-response design) and keeps compute tractable: 900 calls
per axis (3 towers x 2 SSPs x 5 GCMs x 10 realizations x 3 levels), at the current canonical 2050
horizon. Both smoke-tested (1 tower, 1 SSP, all 5 GCMs, 1 realization, all 3 levels) before the
full run -- smoke test alone was informative enough to preview the eventual finding (grazing showed
a clear, monotonic +8.7%/+18.9% pattern; fertilizer showed a muted +1%/+0.9%), confirmed at full
scale. **Actual runtime: ~51 minutes per axis (~1.7h combined)**, 0 skipped/failed calls, 0 NaN.

**Result: both stated priors confirmed, cleanly, not adjusted after seeing the numbers.**

- **Grazing timing shows a real, substantial, monotonic effect at every tower**: T4 14.81 -> 17.61
  nmol (+18.9% at +4 weeks), T9 30.67 -> 35.93 (+17.2%), T2 muted in magnitude but same monotonic
  direction (+4.1%). A genuine, previously-untested management lever.
- **Fertilizer schedule shows a small effect, inconsistent in DIRECTION across towers**: T2/T9 show
  increased application frequency *decreasing* predicted FCH4 (-4.1%/-2.2%), T4 shows the opposite
  (+1.2%), every magnitude under 5%. Read as genuinely weak/noise-level -- extends F-01/F-04/F-05's
  standing "redundant on the rich base" finding from real-data feature importance to scenario
  response, the first time that finding has been tested this way.
- **AOA side-finding**: grazing's AOA-flagged-% is both higher in absolute level (76-88% vs.
  fertilizer's 59-68%) and grows monotonically with the shift level (T4: 76.0%->82.0%->88.0%) --
  extending the season genuinely pushes the scenario further from the real training distribution, a
  sensible pattern fertilizer's schedule changes don't show as cleanly.

**Outcome: no change to any standing recommendation** -- both are diagnostic scenario extensions.
Genuinely useful for the dissertation regardless of direction: grazing-season length joins
livestock density as a real, substantial management lever worth reporting; fertilizer's null result
is itself a legitimate, now twice-confirmed (real-data feature importance AND scenario response)
finding, not a failed experiment. See D-84, D-85, D-70, D-67, F-01, F-04, F-05,
`src/features/build_transient_scenario_drivers_practices.py`,
`notebooks/07_scenario_analysis/s05_practices_trajectory.py`, `s05_practices_analysis.py`,
`s05_results.md`'s "Second update" section, `S05_species_trajectory.ipynb`,
`results/s05_practices_grazing.csv`, `results/s05_practices_fertilizer.csv`.

**Addendum, same day: daily-resolution figures for all 3 scenario families, made naming-consistent,
extended to both SSPs.** User asked for daily-chain figures (full horizon/single-year zoom/
monthly-smoothed) "similar to" an earlier livestock-axis check, for all three families -- livestock
already had every daily prediction saved (`s05_daily_chains_2050.parquet`), so its figures were
free (pyarrow predicate-pushdown query, ~0.6s/slice); grazing/fertilizer never had daily chains
saved (only `annual_mean`), so a small representative subset was rerun (`s05_practices_daily_
chains_subset.py`, 3 towers x 1 GCM/realization x each axis's 3 levels). User then flagged a real
naming inconsistency (livestock's files didn't match grazing/fertilizer's `s05_practices_{axis}_
daily_*` pattern) -- fixed by renaming livestock's script/figures to `s05_livestock_daily_*`
(`s05_daily_chains_2050_plots.py` -> `s05_livestock_daily_chains_plots.py`; nothing was tracked in
git yet, so a clean rename, not a git-history-preserving move). User then asked for a second SSP,
correctly anticipating (confirmed directly, not assumed) that realization/GCM choice barely moves
the result (S-05's own isolated-realization-spread finding, 2.4-7.6%) but SSP is a real, separate
axis worth showing both sides of -- livestock's SSP5-8.5 was free (requery the existing full grid);
grazing/fertilizer's subset script was extended to loop both SSPs (18 calls/axis now, ~60s each,
re-run). Applied the SSP-labeling fix to BOTH SSPs uniformly (regenerated ssp245's files with the
`_ssp245` suffix too, not just adding `_ssp585`) rather than leaving one SSP implicit and the other
explicit -- final convention: `s05_{axis}_daily_{view}_{ssp}.png`, `axis` in {livestock,
practices_grazing, practices_fertilizer}, `view` in {full_horizon, zoom2035, monthly_smoothed},
`ssp` always explicit, 18 figures total. See `s05_livestock_daily_chains_plots.py`,
`s05_practices_daily_chains_subset.py`, `s05_practices_daily_chains_plots.py`, `s05_results.md`'s
file inventory (updated to match).

---

### D-87 -- 2026-08-08 -- Streamlit digital-shadow interface (Objective 6) dropped from scope

**Decision:** the Streamlit interactive interface named in the project's own Objective 6
("Digital shadow interface (Streamlit) with scenario analysis and uncertainty visualisation") is
dropped from scope, user-directed, given the 1 Sept deadline. Explicitly scoped narrowly: this
drops the INTERFACE deliverable only -- the underlying digital-shadow substance (scenario
simulation, management/climate levers, uncertainty quantification once attached) is exactly what
S-01/S-03/S-04/S-05 have spent this project building, and none of that work is affected or
devalued by this decision. The interface would have been a packaging/presentation layer on top of
already-complete analysis, not a prerequisite for it.

**Context for the call:** with Phase 07's analysis substance now genuinely rich (validated
end-to-end mechanism, both SSPs, three independent management levers tested, extrapolation risk
quantified, full daily-resolution output) but UQ intervals still unattached to any of it (flagged
repeatedly, not yet built) and the deadline fixed, building a full interactive interface would
trade dissertation-writing/UQ time for a presentation layer with lower marginal value against the
actual assessment criteria. A reasonable call given the constraint, not a quality judgment on the
original objective.

**Outcome:** Objective 6 in `CONTEXT.md` marked dropped (not silently deleted -- the original
wording is kept, struck through, with this decision cited), matching this project's own
established convention for scope changes (e.g. B-08's "confirmed superseded" treatment, D-70's
"not a production-config change" framing) -- decisions are recorded, not erased. No other
methodology, finding, or standing recommendation in this project is affected. If a lightweight
static summary of the scenario results is wanted later (e.g. a rendered notebook or a simple
static page, well short of a full Streamlit app), that remains a cheap, separate option -- not
precluded by this decision, just not the originally-scoped interactive interface. See CONTEXT.md's
Objectives section.

---

### D-88 -- 2026-08-10 -- U-04: UQ recalibrated for the current forecasting champion
(TabPFN+species, TabICLv2), closing the gap U-02 left behind

**Context:** direct follow-up to the UQ discussion this session. Checking exact dates surfaced a
real, undernoticed gap: U-02 (D-62, 2026-07-06) built leave-one-anchor-out conformal calibration
on `forecast_daily_v2.csv` with an 8-model roster -- but **TabICLv2 joined the roster 3 days
later** (D-66, 2026-07-09), **F-10's species-disaggregated features landed 4 days after that**
(D-67, 2026-07-10), and the standing forecasting champion became **TabPFN+species** off the back
of F-10 (reconfirmed under the climatology-MASE convention at D-80). U-02's "TabPFN" conformal
interval is calibrated for a feature configuration this project no longer recommends; TabICLv2 has
never had UQ built for it at all. User named this "Option A" of a two-part plan (A: close the
current-pipeline gap; B: build on it for scenario-analysis UQ) -- this decision covers A.

**Scope, user-confirmed via explicit choice (not assumed).** Two options were presented: recalibrate
the full 11-model roster (matches CLAUDE.md's "full coverage by default" convention, but requires
retraining RF/XGB/LightGBM/SARIMAX/TFT/DLinear/LSTM -- expensive) vs. champion-focused (TabPFN +
TabICLv2 only -- both zero-shot with native quantile support already, no retraining). User chose
champion-focused, reasoning that the other 6 models' feature configuration never changed, so their
existing U-02 numbers remain valid as-is -- only the two models whose production config actually
changed needed rebuilding.

**Implementation, deliberately minimal-diff for direct comparability.** New script
`u04_champion_uq.py`: same 3-tower x 5-anchor sweep, same quantiles (0.05, 0.5, 0.95), same
`rr.conformal_margins_by_bin()` leave-one-anchor-out calibration, same PICP/MPIW/pinball metrics.
`evaluate_stage()` is **imported unmodified from `u02_multi_anchor_tower.py`** rather than
reimplemented -- only `fit_stage` differs (TabPFN/TabICLv2 zero-shot rollout on `forecast_daily_v3
.csv`'s `BASE+species` config, instead of U-02's pooled tree/SARIMAX/TFT fitting on `v2.csv`) -- so
U-02 and U-04's numbers are directly comparable, not just similarly-computed. `BASE+species`
construction imported in spirit from `b16_foundation_models_v3.py`'s own `FAMILIES`/`BASE_FX`
logic (`SPECIES_COLS` = `["fx_cattle_dens","fx_sheep_dens","fx_lamb_dens"]`), not retyped
independently. Smoke-tested in two stages: 1 tower/1 anchor first (caught, correctly, that
leave-one-anchor-out calibration produces all-NaN conformal columns with only 1 anchor available --
expected, not a bug, since there's no "other anchor" left to pool residuals from), then 1 tower/2
anchors to confirm the calibration path itself works (PICP 0.84-0.86, sane). **Full run: 25
seconds** -- no model training at all, both models are zero-shot.

**Result 1: calibration converges to ~0.89-0.90 PICP at T4/T9**, replicating U-02's own headline
finding ("every model converges to ~0.88-0.90 regardless of raw coverage") on the new champion
feature config. **Tower 2 still cannot support calibration** (all-NaN conformal columns) -- the
exact same pre-existing limitation U-02 already documented ("Tower 2 cannot support calibration at
all -- real `y_observed` in only 1/5 anchor windows"), confirmed to persist independent of the
feature-set change (T2's real coverage ends May 2019 regardless of which columns are used).

**Result 2, the actual finding, not guaranteed in advance: species enrichment improved point
accuracy without materially changing calibration quality.** Direct old-vs-new comparison, TabPFN:
T4 conformal MPIW 148.44 (U-02, BASE-only) vs. 149.52 (U-04, BASE+species); T9 190.31 vs. 188.93;
pinball essentially unchanged at both towers (10.56/10.56, 13.00/12.86). Read as mechanistically
sensible, not surprising once seen: conformal margins track residual *distribution*, and D-80's
own numbers show the species-config MASE gain over BASE was modest, not dramatic -- a modest
point-accuracy change shouldn't be expected to move interval width much either. Still a genuinely
new, directly-measured result -- "does closing the UQ gap change the answer" could have gone
either way and didn't get assumed.

**Outcome:** the UQ gap for the two models that actually need it (TabPFN+species, the standing
champion; TabICLv2, its closest zero-shot competitor) is closed. No recalibration needed for the
other 6 U-02 models -- their feature configuration never changed. This is the validated foundation
"Option B" (scenario-analysis UQ, AOA-stratified) builds on next -- same
`conformal_margins_by_bin()`/leave-one-anchor-out machinery, extended to scenario points rather
than real historical anchors. See D-62, D-63, D-66, D-67, D-80,
`notebooks/06_interpretability_uq/u04_champion_uq.py`, `U04_results.md`,
`results/u04_chains.csv`, `results/u04_summary.csv`.

**Addendum, same day: U-04 fancharts built.** `u04_fanchart_plots.py`, same visual convention as
`u02_fanchart_plots.py`, all 30 (tower x anchor x model) combinations, `results/figures/
u04_fancharts/`. U-04 had shipped without figures despite U-01/U-02/U-03 all having them --
flagged directly (checking actual file locations surfaced the gap), not left unaddressed.

---

### D-89 -- 2026-08-10 -- U-05: scenario-analysis UQ ("Option B"), built on U-04's validated method
but on S-05's own architecture, with an AOA-stratified two-tier margin resolved empirically

**Context:** direct follow-up to D-88 (U-04, "Option A"), completing the two-part plan. Key
design realization stated before building: U-04's calibration (on `forecast_daily_v3.csv`'s
`BASE+species`, 52 columns) is NOT reusable for S-05, which uses a deliberately narrower
`FX_A_SPECIES` (13 columns, S-03's Variant A + species) -- a different feature space is a
genuinely different model with different error characteristics, S-03's whole ablation exists to
demonstrate exactly that. This decision repeats U-04's *method* on the correct architecture rather
than reusing its numbers.

**Implementation, Steps 1-2 (calibration set).** New script `u05_scenario_uq.py`: TabICLv2
zero-shot, `FX_A_SPECIES`, 5 real historical anchors (2018-2022) x 3 towers, same quantiles/
leave-one-anchor-out `conformal_margins_by_bin()`/`evaluate_stage()` (imported unmodified from
`u02_multi_anchor_tower.py` a third time) as U-02/U-04. **Full run: 9 seconds.**

**A real bug caught by smoke-testing before the full run, not shipped.** First version computed
each tower's AOA nearest-neighbour training set from the unrestricted full historical record --
correct for S-05's own AOA check (its scenario dates are genuinely future, never overlapping real
data) but wrong here, since U-05 tests on real historical anchors: a test point can be a literal
row already inside its own unrestricted training set, giving distance-to-self = 0 for every point.
Caught directly (`aoa_dist` was uniformly 0.0 in the 2-anchor smoke test) before the full run,
diagnosed correctly (not just "a bug," the specific leakage mechanism was identified), and fixed
by restricting the AOA training set to pre-anchor-only data, recomputed fresh per anchor -- this
project's standing no-leakage convention, applied here for the first time to an AOA computation
rather than a model-fitting one.

**Step 3, the actual design question from the original plan -- resolved empirically, exactly as
promised, not assumed either way.** Does |residual| correlate with AOA-flagged status in real
historical data? **Result: weak raw linear correlation (pooled Pearson r=0.146) but a real,
substantial categorical difference** -- out-of-AOA residuals are ~48% larger than in-AOA, pooled
(46.03 vs. 31.14, n=529/1,793). T4 (+49%) and T9 (+39%) both confirm the same direction cleanly;
T2 shows a reversed small difference but on n=4 out-of-AOA points, not trusted. **This resolves the
plan's Level-1-vs-Level-2 fork with a genuine third option neither fully anticipated**: weak linear
correlation rules out fitting a smooth continuous widening function (Level 2 as originally
sketched), but the real categorical gap rules out ignoring AOA entirely (which would leave Level 1
as the only "safe" fallback). Landed on a **two-tier margin, interpolated continuously by each
point's own `aoa_flagged_pct`** (0-100%, not a hard cutoff) between the in-AOA and out-of-AOA base
margins -- smoother than a strict binary split, without overclaiming a distance-response
relationship the correlation check doesn't support.

**Step 4 (apply to S-05's existing output -- zero new model calls, pure post-processing).** Joins
the two-tier margin onto `s05_trajectory_realizations_2050.csv` (229,500 rows), `s05_practices_
grazing.csv`/`s05_practices_fertilizer.csv` (25,500 each) against each row's own already-saved
`aoa_flagged_pct`. **A second real issue caught and fixed before finalizing**: the first Step-4
implementation applied one flat per-tower margin regardless of whether a given row was itself
flagged -- didn't actually use Step 3's finding. Fixed to genuinely interpolate per-row.
**T2 is forced to NaN throughout Step 4**, gated on Step 2's own `conformal_mpiw` (not merely
whether AOA-stratified residuals exist for T2) -- T2 already fails proper leave-one-anchor-out
calibration (only 1 anchor has real ground truth) and its own out-of-AOA sample is n=4; a cruder
fallback standard for T2 here would have quietly contradicted the same finding U-02/U-04 already
established, so it wasn't allowed to.

**Result: calibration converges to ~0.88-0.89 PICP at T4/T9** (0.8907/0.8821), matching U-02/U-04's
pattern a third time. **The calibrated interval is genuinely very wide, stated plainly**: roughly
±94-100% of the mean in-AOA, ±139-140% out-of-AOA (T4: 93.9%->139.6%; T9: 100.4%->139.2%) --
consistent with, not contradicting, U-01's original "intervals are wide... CH4 carries large
aleatoric uncertainty" finding (D-40), now confirmed on the scenario architecture specifically.
T2 remains uncalibratable -- a third independent confirmation of the same structural limitation.

**Figures**: `u05_fanchart_plots.py` (15 figures, U-02/U-04's fanchart convention, TabICLv2 on
`FX_A_SPECIES`, real anchors -- AOA-flagged days marked directly on the chain, a visual addition
not present in U-02/U-04's fancharts). `u05_trajectory_with_uq_plots.py` -- the actual deliverable:
S-05's livestock-baseline trajectory with the two-tier interval overlaid, kept **visibly separate**
from S-05's own realization-spread band (a different uncertainty source entirely -- weather/GCM
draw variability, not predictive/model uncertainty), per this session's own pooled-vs-isolated
realization-spread lesson (D-85) -- not merged into one band despite the temptation to simplify.

**Outcome: scenario-analysis UQ is no longer missing.** S-05's livestock/grazing/fertilizer
outputs now carry a calibrated, AOA-aware interval (`u05_{axis}_with_uq.csv`), not just a point
estimate. No change to any standing forecasting/scenario recommendation -- this is UQ
infrastructure, not a new finding about the scenario levers themselves. See D-88, D-70, D-84-D-86,
D-62, `notebooks/06_interpretability_uq/u05_scenario_uq.py`, `u05_fanchart_plots.py`,
`u05_trajectory_with_uq_plots.py`, `U05_results.md`, `results/u05_livestock_with_uq.csv`.

---

### D-90 -- 2026-08-10 -- U-06: Conformalized Quantile Regression (CQR) fixes the spike-coverage
failure U-04/U-05's own fancharts revealed visually

**Context:** user observation on U-04/U-05's fancharts, checked directly rather than taken on
faith: "a lot of spikes are still beyond the interval." Confirmed and quantified against U-04's
chains: overall PICP≈0.89 looked fine, but **75% of the top-10%-magnitude (spike) days fell
entirely outside the interval, vs. 3.3% for the bottom 90%**. Mechanism: split-conformal's flat,
symmetric per-bin margin (`median +/- constant`) guarantees only *average* coverage across a bin,
and was hitting its ~90% target almost entirely by nailing the easy majority of low-flux days
while systematically failing the rare high-magnitude ones -- a distinction the aggregate PICP
number alone can never reveal. Not a new problem for this project (U-01/D-40 already flagged "even
wide intervals miss the biggest spikes"; B-05/B-06's point-forecast-level fixes both came back
negative) but not yet addressed at the UQ-calibration level specifically.

**A second empirical check, run before committing to any fix (matching this session's own standing
"measure before deciding" practice).** Does the model's own raw (uncalibrated) quantile spread
already track spike magnitude better than the median does? **Yes**: raw `q95-q05` spread widens
1.3-1.8x on spike days vs. normal days, and on spike days the raw q95 ALONE already sits close to
(TabPFN: 182.7 vs. actual 193.1) or exceeds (TabICLv2: ~360 vs. 193.1) the actual spike value --
while the median (~35) massively undershoots. This directly motivated CQR over alternatives (a
smooth AOA-distance-widening function, or magnitude-binned margins) -- the raw quantiles already
carry real, usable adaptive signal the flat margin was throwing away entirely.

**Implementation: Conformalized Quantile Regression (Romano et al. 2019).** Nonconformity score =
`max(q05-y_true, y_true-q95)` instead of split-conformal's `|y_true-median|`; calibrated interval =
`[q05-margin, q95+margin]` (asymmetric-capable) instead of `[median-margin, median+margin]`.
**`rr.conformal_margins_by_bin()` reused completely unchanged** -- already generic over whatever
nonconformity-score array it receives; CQR only changes what's computed and how the margin is
applied, not the calibration function itself (the fourth reuse of this one function across
U-02/U-04/U-05/U-06). Same leave-one-anchor-out structure, same bins, same 3-tower coverage.
**No new model calls anywhere** -- pure recalibration of U-04's/U-05's already-saved chains.

**A real bug caught and fixed before reporting, not shipped.** First aggregation pass showed T2 at
`0.0000` PICP instead of the expected `NaN`. Checked directly against the per-bin rows (correctly
NaN) before concluding anything -- the bug was in the new script's own `wavg()` aggregator, missing
the exact all-NaN guard `u02_multi_anchor_tower.py`'s own `wavg()` already carries and documents
in its own comment (pandas silently sums an all-NaN column to `0.0`, not `NaN`). Fixed by
replicating that guard verbatim rather than rediscovering the fix independently.

**Result: spike coverage roughly triples, at an honest, stated cost, not hidden.**

| Model / data | Spike coverage (old->CQR) | Normal-day coverage (old->CQR) | Spike interval width (old->CQR) |
|---|---|---|---|
| TabICLv2 (U-04, BASE+species) | 24.3% -> **79.7%** | 96.7% -> 83.7% | 183 -> 394 |
| TabPFN (U-04, BASE+species) | 24.3% -> **57.2%** | 96.7% -> 88.4% | 177 -> 248 |
| TabICLv2 (U-05, FX_A_SPECIES) | 22.1% -> **79.3%** | 96.1% -> 82.5% | 185 -> 405 |

**TabICLv2 benefits substantially more than TabPFN** (~80% vs. ~57% spike coverage) -- consistent
with the raw-quantile check above (TabICLv2's raw q95 already exceeds actual spike values on
average; TabPFN's sits just short). Normal-day coverage drops from ~96-97% to ~83-88% (still
comfortably above an 80% practical floor); spike intervals roughly double in width -- the honest
price of actually covering the events that matter, rather than a flat interval that only looks
good on average by ignoring them. **Aggregate (non-spike-specific) PICP stays similar to U-04/U-05's
original numbers** -- CQR doesn't change the headline PICP much, because that number was never the
real problem; it changes *where* the coverage comes from, invisible to the headline alone.

**Outcome: CQR should replace the symmetric-margin approach as this project's standing UQ method**
going forward, given the fix's magnitude and that it costs nothing extra (same calibration
function, same already-available native-quantile inputs). **Not yet applied to S-05's actual
scenario trajectories** -- U-05's Step 4 output used a %-of-mean margin from the old symmetric
calibration; extending CQR there would need S-05's scenario points to have raw daily q05/q95 saved,
which the current daily-chains-subset scripts don't currently request -- flagged as a small, cheap,
concrete next step, not yet built. This reframes what "the UQ gap is closed" meant from D-88/D-89:
those closed "no interval exists"; this closes "the interval exists but silently fails on the days
that matter most," invisible to the PICP headline number alone. See D-88, D-89, D-40, D-42, D-43,
`notebooks/06_interpretability_uq/u06_cqr_recalibration.py`, `u06_cqr_comparison_plots.py`,
`U06_results.md`, `results/u06_u04_cqr_summary.csv`, `results/u06_spike_coverage_U04.csv`.

---

### D-91 -- 2026-08-10 -- U-07: livestock-density-stratified CQR -- thinner margins where
livestock presence is smaller, a much cleaner stratifier than AOA distance turned out to be

**Context:** direct user question on U-06's output, checked empirically before building anything
(matching this session's own standing practice): "can't the margin be thinner for instances where
livestock presence is smaller?" U-06's CQR already lets the model's own raw q05/q95 respond to
livestock density implicitly (it's a model input), but the additive calibration margin layered on
top was only binned by lead-time (U-04) or lead-time x AOA-flagged status (U-05) -- never by the
covariate actually driving the heteroscedasticity most strongly.

**Checked directly, and the signal is much stronger than U-05's AOA-distance check**:
`corr(|residual|, fx_lsu_dens) = 0.43-0.45` (vs. AOA-distance's weak 0.09-0.15), residuals ~3.2x
larger on above-median-LSU days (51.0 vs. 15.9-16.2 mean |residual|). `fx_cattle_dens` correlates
almost identically (0.427) -- consistent with S-05's own cattle-dominance finding; sheep/lamb
correlate at noise level (0.03-0.05). Used `fx_lsu_dens` directly (standard, interpretable
aggregate), not a species-specific variable.

**Implementation.** Same CQR machinery as U-06 -- only the bin KEY changes, from lead-time-bin
alone to lead-time-bin x LSU-tertile (e.g. `"31-90_low"`). `conformal_margins_by_bin()` needed
ZERO code changes -- already generic over arbitrary dict keys, the fifth reuse of this one function
across U-02/U-04/U-05/U-06/U-07. Tertile boundaries (1/3, 2/3 quantiles of `fx_lsu_dens`) computed
from LEAVE-IN (calibration) anchors only per test fold, never the held-out anchor -- same
no-leakage discipline as every other leave-one-anchor-out step this project uses (the same class
of bug U-05 caught and fixed for its own AOA computation, applied correctly here from the start).
No new model calls -- recalibrates U-04's/U-05's already-saved chains a second time.

**Result: low-LSU intervals are 29-46% the width of high-LSU intervals -- a genuine win-win, not a
trade-off.** TabPFN (U-04): low MPIW=84.3 vs. high=293.7 (29%). TabICLv2 (U-04): 177.3 vs. 419.8
(42%). TabICLv2 (U-05): 206.1 vs. 447.4 (46%). PICP stays reasonably consistent across tiers
(0.76-0.83) -- each tier calibrated against its own now-homogeneous residual distribution, not one
pooled distribution compromising between very different regimes the way the flat margin had to.

**Verified this doesn't trade away U-06's spike-coverage fix, not assumed**: spike days (top 10%
magnitude) average 3.2x higher `fx_lsu_dens` than normal days (3.30 vs. 1.04) -- the "high" tier
substantially captures the spike population with its OWN dedicated, appropriately-wide calibration
set, not diluted by mixing in low-LSU days the way the single pooled CQR margin was. Genuine
improvement on both axes simultaneously: tighter where the model and the real driver both say
confidence should be high, still wide where it shouldn't be.

**Outcome: LSU-density stratification should be the standing UQ method going forward, layered on
top of U-06's CQR** -- costs nothing extra (same calibration function, same already-available
`fx_lsu_dens` input). A broader lesson worth carrying forward: a domain-meaningful covariate
directly tied to the outcome's own magnitude (0.43 correlation) was a substantially cleaner
stratifier than a generic distance-to-training-data metric (0.09-0.15) -- worth checking for a
covariate like this FIRST, before reaching for a more abstract distance-based one, if this pattern
recurs. Not yet applied to S-05's actual scenario trajectories -- same standing caveat as U-06
(would need raw daily q05/q95 saved for scenario points). See D-90, D-89, D-88,
`notebooks/06_interpretability_uq/u07_lsu_stratified_cqr.py`, `u07_lsu_cqr_comparison_plot.py`,
`U07_results.md`, `results/u07_u04_lsu_cqr_summary.csv`.

**Addendum -- 2026-08-10 -- full-coverage figures.** Original comparison plot showed one
representative chain (T4/TabICLv2) only; underlying `lsu_cqr_margin` numbers were already computed
for the full champion roster (T2/T4/T9 x TabPFN+TabICLv2 for U-04, T2/T4/T9 x TabICLv2 for U-05)
but never plotted or tabulated per combination. Extended `u07_lsu_cqr_comparison_plot.py` to loop
all (tower, model, data_label) combinations -- 6 figures now saved (T4/T9 x TabPFN/TabICLv2 for
U-04, T4/T9 x TabICLv2 for U-05). T2 explicitly logged as skipped, not silently omitted: 0 of its
rows have a valid `lsu_cqr_margin` in either summary, because T2's base split-conformal calibration
(U-04/U-05's own Step 2) already failed before CQR or LSU-stratification were applied -- same
finding already established in U-04/U-05, not new. Full-roster MPIW low-as-%-of-high range is
26-59% (T9/TabPFN best at 26%, T4/TabICLv2-U05 weakest at 59%) -- wider than the original two-point
29-46% estimate but the direction/magnitude of the effect holds at every tower and model it can be
computed for.

### D-92 -- 2026-08-10 -- S-05 + UQ: U-06/U-07's CQR calibrations attached to actual scenario
trajectories, closing the last gap between scenario analysis and UQ

**Context:** flagged repeatedly since U-06 (D-90) as a small, cheap, not-yet-built next step, then
directly confirmed by the user ("let's do (1)") after I framed it as the natural close-out task
before shifting to write-up given the 1 Sept deadline. U-06/U-07's calibrations had only ever
touched U-04/U-05's historical calibration chains -- S-05's own sweeps (`s05_trajectory_2050.py`,
`s05_practices_trajectory.py`) only ever saved point predictions, so Objectives 4 (UQ) and 5
(scenario analysis) never actually connected in the output.

**Method, two new scripts, zero new calibration fitting (pure inference + post-processing):**
`s05_uq_daily_chains_subset.py` reruns S-05's existing small representative subset (3 towers x 2
SSPs x 3 levels/combos/axis = 18 calls/axis x 3 axes = 54 calls, 2050 horizon) requesting
`quantiles=(0.05,0.5,0.95)` from `tabicl_forecast()` instead of a point prediction -- confirmed
free (4.3s/call, same as point-only) since TabICL always computes an internal quantile grid
regardless of the argument. Full 54-call run: ~2.5 min, 0 failures. `s05_uq_cqr_apply.py` attaches
U-06's flat CQR margin and U-07's LSU-tier CQR margin (both from U-05's FX_A_SPECIES-architecture
summaries, not U-04's -- D-89's feature-space finding) via a (tower, lead-bin[, LSU-tier]) lookup,
margins pooled across U-05's 5 anchor years since scenario points have no anchor year of their own.

**Two explicit extrapolation assumptions, stated not hidden:** (1) lead times beyond 365 days hold
the widest calibrated bin's margin flat -- no calibration evidence exists past ~1 year for a
horizon running to 2050, and this likely UNDERSTATES true uncertainty at year 20+ (error should
plausibly grow with lead time, not plateau) -- read far-horizon CQR bands as a floor, not a
ceiling. (2) grazing/fertilizer axes reuse the livestock-architecture margins despite their own
extra covariates -- same approximation U-05's own Step 4 already made applying its margins
uniformly across all three axes.

**Result: works cleanly, >99% coverage at T4/T9** (T2 0% -- same pre-established degeneracy as
every prior UQ step, not new). One genuine thin spot: Tower 4's days-1-7 x mid-LSU-tier cell has
zero calibration samples across all 5 U-05 anchors (surfaced as NaN, not filled in) -- a real
data-sparsity limit inherited from U-07's own summary. Verified directly: zero interval inversions
anywhere; CQR correctly TIGHTENS the model's own raw quantile spread on average at T4 (raw MPIW
572.6 vs. U-06 514.6 vs. U-07 523.4), consistent with U-06's original historical finding.

**Outcome: this closes the experimental UQ-for-scenarios arc (U-04 through U-07, now connected to
S-05).** No further "not yet applied to S-05" caveat remains outstanding. Figures:
`results/figures/s05_uq_cqr/s05_uq_cqr_{livestock,grazing,fertilizer}.png`. Data:
`results/s05_{livestock,grazing,fertilizer}_with_cqr.csv`. See D-91, D-90, D-89, D-88,
`notebooks/07_scenario_analysis/s05_uq_daily_chains_subset.py`, `s05_uq_cqr_apply.py`,
`s05_uq_cqr_plots.py`, `s05_results.md`.

**Addendum -- 2026-08-10 -- Tower 2 + second SSP added to D-92's figures.** User question ("where
is tower 2 and the other ssp?") caught two real gaps in the first draft: `s05_uq_cqr_plots.py` had
hardcoded `TOWERS = [4, 9]` (T2 silently excluded from the loop entirely, not even attempted) and
`SSP = "ssp245"` (single value, no loop) despite both T2 and ssp585 already being present in the
underlying `s05_*_with_cqr.csv` data (no new model calls needed -- purely a plotting-scope fix).
Fixed: all 3 towers now plotted in every figure (T2 shows its raw `[q05,q95]` band, which is valid
-- TabICL forecasts fine for T2 -- with an explicit on-figure annotation that no calibrated CQR
band exists, rather than being omitted); both SSPs now plotted, filename always carries the SSP
suffix (matching the naming convention the rest of S-05's daily-chain figures already use). 6
figures total (was 3): `s05_uq_cqr_{livestock,grazing,fertilizer}_{ssp245,ssp585}.png`.

---

### D-93 -- 2026-08-10 -- B-16 round 2: TICA embeddings + static AR-lag features for TabICLv2 --
both negative, combining them is actively worse, BASE+species remains champion

**Context:** user revisited forecasting, focused specifically on Track B (TabICLv2, zero-shot)
rather than Track A's trained trees (explicit choice via AskUserQuestion). Wanted to test TICA
embeddings and expanded autocorrelation/lag features. Before building, worked through two real
design forks with the user rather than picking silently:

1. **Dynamic vs. static AR-lag features.** `tabpfn_forecast()`/`tabicl_forecast()` are single-shot
   (one call produces the whole 365-day chain; no day-by-day recursion like Track A's
   `tree_rollout()`), so a genuinely dynamic lag feature would need a new day-by-day (or
   block-recursive) roller -- costed at ~30h+ (daily) or ~75-90min (30-day blocks) for a full
   sweep vs. seconds for a static, frozen-at-anchor design. User caught a real weakness in the
   static design directly ("does that mean May/June forecasts still use December's AR features?")
   -- confirmed yes, explicit acknowledged tradeoff, user chose to proceed with static anyway
   ("use static AR features if no other option").
2. **Does S-05 (scenario projection) already have this problem?** Checked directly: S-05 uses the
   identical single-shot architecture (one `tabicl_forecast()` call per (tower, scenario-combo),
   producing the full 27-31-year chain at once) and currently has ZERO explicit AR-lag columns at
   all in `FX_A_SPECIES`. A block-recursive design that's cheap for a 365-day evaluation window
   becomes ~30x more expensive over S-05's multi-decade horizon (untenable at full-sweep scale) --
   flagged a climatology-anchored (not model-prediction-chained) alternative as a cheaper, more
   portable idea for later, but not built this round (out of scope, user chose to focus back on
   TabICL/static features for now).

**Method, single consolidated notebook** (`B16_tica_static_ar_features.ipynb`, user-requested
format for easy rerun/tweaking) **-- TabICLv2 only, all 3 towers x 5 anchors (2018-2022), 4
configs (60 calls total)**:
- **`+static_ar`**: 10 pre-anchor summary features (mean/std7,14,28d, max28d, days-since-spike,
  trend14d, autocorr7) of the CH4 target -- future window frozen at the anchor's own value (stated
  tradeoff above); **historical context uses genuinely time-varying rolling stats** (a real bug
  caught by the smoke test: an earlier version broadcast the frozen scalar across history too,
  giving TabICL zero-variance history columns that broke an internal feature-deduplication filter,
  `IndexError: boolean index did not match indexed array` -- fixed via
  `historical_static_ar_frame()`, vectorized rolling stats using only data strictly before each
  historical day).
- **`+tica`**: reuses D-79's exact TICA method (`temp_gap_filing_exploration copy.ipynb`, gap-filling)
  verbatim -- generalized eigenvalue problem on symmetrized lag-covariance matrices, from scratch
  (no `deeptime` dependency). Adapted `tau=7` (1 week) for daily data (D-79 used `tau=24` hours on
  hourly data); 3 components; input = 43 continuous drivers (excludes calendar cyclical encodings
  and binary flags, same exclusion logic D-79 used to avoid a trivial tau-alignment artifact). Fit
  leak-free per anchor on pre-anchor history only. **Smoke test caught a second real bug**: with 43
  collinear continuous drivers (e.g. `fx_SWC_lag7/14/21/28/roll7/14` are near-linear transforms of
  one signal), the instantaneous covariance matrix `C0` was not positive definite --
  `scipy.linalg.eigh`'s generalized solve failed outright. Fixed with a small relative ridge
  regularization on `C0`'s diagonal (standard TICA/PCA stabilization, matches deeptime/PyEMMA's own
  shrinkage handling).

**Result: clean, consistent negative -- BASE+species remains the standing champion.**

| Config | MASE (climatology) | R2 | vs. baseline |
|---|---|---|---|
| BASE+species (baseline) | 0.7353 | -0.1070 | -- |
| +tica | 0.7348 | -0.0937 | -0.0005 (noise) |
| +static_ar | 0.7358 | -0.0924 | +0.0005 (noise) |
| +static_ar+tica | **0.7603** | -0.1559 | **+0.025 (clearly worse)** |

TICA alone replicates D-79's own "wash" finding from gap-filling, this time on a genuinely
different task (extrapolation, not interpolation) -- the prior transferred directionally (still a
wash) even though the task differs. Static AR features alone are also a wash, consistent with the
stated expectation (TabICLv2 already sees raw `y_observed` history natively, so explicit derived
summary statistics add little). **Combining both is clearly worse, consistent across all 3
towers** (T2/T4/T9 all degrade) -- matches this project's own recurring "stacking too many feature
families hurts more than any one family helps" pattern (D-67's original `BASE+ALL` finding).

**Outcome:** no new champion. Both feature families are negative results worth keeping on record.
`BASE+species` (MASE=0.715 in the original 5-anchor headline table, D-80) remains production. Full
write-up in the notebook's own Section 9 (Verdict). Files:
`notebooks/05_benchmarking/B16_tica_static_ar_features.ipynb`,
`results/b16_tica_static_ar_{chains,summary}.csv`.

---

### D-94 -- 2026-08-10 -- B-16 round 4: pooled vs. solo for TabPFN/TabICLv2 on the forecasting
task specifically -- splits by model, TabPFN gets a small real gain, TabICLv2 replicates the
gap-filling wash

**Context:** direct follow-up user question -- pooling was adopted for Track A (RF/XGB/LightGBM,
DL) from gap-filling's F-02/F-03 finding, and Track B (TabPFN/TabICLv2) was left solo based on a
gap-filling-only precedent (D-79/D5: "TabICL-solo beats TabICL-pooled at every tower"). That
precedent used a DIFFERENT API (`TabICLRegressor`/`TabPFNRegressor`, plain sklearn-style tabular
regressors, pooled via simple row-stacking) than what the forecasting champion actually runs on
(`TabICLForecaster`/`TabPFNTSPipeline`, genuine time-series-native forecasters) -- never re-tested
on the forecasting task with the champion's own architecture until now.

**Verified before building, not assumed:** both `TabICLForecaster.predict_df()` and
`TabPFNTSPipeline.predict_df()` natively support an `item_id` column for genuine multi-series panel
input (direct inspection of both signatures/docstrings), then smoke-tested (~2-5s for a pooled
call covering all 3 towers, actually faster than 3 solo calls since the GPU batches all items
together).

**Method** (`B16_pooled_vs_solo.ipynb`): `BASE+species` + `is_t2/is_t4/is_t9` tower dummies as
covariates (matching Track A's own pooling convention). One pooled call per (anchor, model) covers
all 3 towers at once -- 10 calls total (5 anchors x 2 models), cheaper than solo's 30. **Solo
baseline reused from U-04's already-saved `u04_chains.csv`** (same exact config/anchors/models,
D-88) rather than rerun, rescored via the same `bin_metrics()` + climatology-MASE convention for a
direct comparison. **Two real bugs caught by the smoke test/full traceback, not silently patched:**
(1) the pooled/batched multi-item_id call path does not tolerate the same partial NaN in covariates
that solo calls handle fine internally (`sklearn`'s own array validation inside TabICL's batch
dispatch raised `Input contains NaN` even though these exact tower/anchor combos succeed solo) --
fixed with explicit mean-imputation (context-only, leak-free, same convention as Track A's
`SimpleImputer`). (2) Tower 9's two earliest anchors (2018, 2019) have empty pre-anchor history
(its real analyser data only starts Feb 2020) -- `doy_climatology()`'s global-mean fallback
silently produces NaN on an empty series, poisoning `mean_absolute_error` downstream -- fixed with
an explicit skip-and-log guard (same pre-established T9 data-scarcity class as everywhere else in
this project, not new).

**Result, splits by model:**

| Model | Solo MASE (climatology) | Pooled MASE | Delta |
|---|---|---|---|
| TabICLv2 | 0.7353 | 0.7355 | +0.0002 (noise) |
| TabPFN | 0.7166 | **0.7138** | **-0.0028 (pooled better)** |

**TabICLv2** is a wash either way -- directionally consistent with D-79's gap-filling finding
(TabICL prefers solo, context-cap dilution) but far smaller in magnitude than D-79's own +0.005 to
+0.118 gap; plausible mechanism: the dilution effect matters less for a temporally-coherent 365-day
rollout window than for gap-filling's dense full-record pooled context. **TabPFN beats solo at all
3 towers, consistently** (T2: 0.4477 vs. 0.4502; T4: 0.7600 vs. 0.7628, R2 crosses from slightly
negative to slightly positive; T9: 0.6658 vs. 0.6687) -- small but real and not driven by one
tower. **This was never tested by D-79 at all** (its pooled-vs-solo comparison only covered TabICL)
-- a genuinely new finding, not a contradiction of prior work.

**Outcome:** a small, real, TabPFN-specific case for pooling -- not large enough to force an
immediate champion switch given the 1 Sept deadline, but a concrete, cheap (no retraining, same
call cost) improvement lead if forecasting is revisited again. `BASE+species` (solo) remains the
currently-documented champion. Files: `notebooks/05_benchmarking/B16_pooled_vs_solo.ipynb`,
`results/b16_pooled_{chains,vs_solo_summary}.csv`.

---

### D-95 -- 2026-08-10 -- S05-T2: does pooling rescue Tower 2's muted livestock-scenario response?
No -- exact 0.0pp effect for both TabICLv2 and TabPFN, a decisive negative result

**Context:** direct follow-up to the Tower-2-muted-response discussion (cattle 3x = only +1.8-2.3%
at T2 vs. +186-215% at T4/T9, s05_results.md). User's own hypothesis: T2's real `fx_lsu_dens`
never exceeds ~0.71 (T4/T9's own 1x baseline is ~5), so the zero-shot model solo-per-tower has no
historical livestock->CH4 covariation to learn from at T2 -- would pooling T2's context with T4/T9's
real (livestock-rich) history let the model borrow their learned cattle sensitivity when projecting
T2's scenario? Directly motivated by precedent: pooling is exactly what rescued Tower 9 in
gap-filling (F-02/F-03, solo~0 -> pooled +0.29 R2).

**Method** (`s05_t2_pooled_test.py`): pooled context = T2+T4+T9's real pre-anchor history
(`item_id`-tagged, the pooling itself); T2 gets its real scenario future (full 2050 horizon, 3
combos x 2 SSPs); T4/T9 get a minimal 30-day placeholder future window at their own baseline
scenario purely to satisfy the API (confirmed directly, not assumed: `TabICLForecaster`'s
multi-item_id dispatcher requires every context `item_id` to also appear in `future_df`, or it
raises a `KeyError` deep in its batch dispatch -- caught via a smoke test before the full run, not
mid-sweep). T4/T9's own predictions are discarded; the pooling benefit (if any) comes purely from
their real rows being present in context, not from their future window. Both TabICLv2 (compared
against T2's already-recorded solo numbers, same GCM/realization draw) and TabPFN (compared
against a freshly-computed solo baseline, since S-05 never used TabPFN before -- no prior record
existed) tested, closing the same "only tested one model" gap D-94 flagged.

**Result: exactly 0.0 percentage points of difference, both models, all 4 comparisons** --

| Model | Solo (cattle 3x) | Pooled (cattle 3x) | Solo (all-species 3x) | Pooled (all-species 3x) |
|---|---|---|---|---|
| TabICLv2 | +1.8% | +1.8% | +4.2% | +4.2% |
| TabPFN | +11.0-11.2% | +11.0-11.2% | +17.2-17.5% | +17.2-17.5% |

Not just "small" -- an EXACT match to the decimal across both SSPs for both models. **Mechanistic
read**: this "pooling" only shares context ROWS within one batched inference call -- it is not
retraining shared parameters the way Track A's RF/XGB pooling is (every split in a pooled tree is
genuinely informed by all towers' rows together). A zero-shot in-context forecaster's output for a
given series still appears driven almost entirely by that series' OWN history in the query; other
towers being present in the same batch does not change what the model outputs for Tower 2 at all.
This also retroactively explains why D-94's own pooling gain (TabPFN, ~0.4% MASE in the *standard*
evaluation) was so small to begin with -- likely a minor batching/numerical effect, not genuine
cross-series transfer, and that effect vanishes entirely once tested against a genuine
extrapolation (T2 has never had 3x cattle, in reality or in the model's own experience).

**Secondary finding, incidental to the main question**: TabPFN's own SOLO response to livestock
scaling at T2 (+11-17.5%) is meaningfully larger than TabICLv2's (+1.8-4.2%) -- a real model-choice
difference in how muted T2's sensitivity is, independent of pooling.

**Outcome: pooling does not rescue Tower 2's livestock-scenario response, for either champion
model.** Closes the debate opened alongside the muted-T2-response discussion -- the "maybe pooling
fixes it" possibility is now closed empirically, not left open. T2's muted response should continue
to be read as a genuine model-extrapolation limit (T2's arable regime gives the model nothing to
learn a livestock relationship from, and simply providing more context rows from other towers does
not transfer one in) rather than a fixable data-availability gap. Files:
`notebooks/07_scenario_analysis/s05_t2_pooled_test.py`,
`results/s05_t2_{pooled_trajectory,tabpfn_solo_trajectory,pooled_vs_solo_compare}.csv`.

**Addendum -- 2026-08-10 -- figure added.** User asked "where are the figures tho?" — the initial
D-95 build was numeric-only (comparison table + CSVs), no plot. Added
`s05_t2_pooled_plot.py`: solo (dashed) vs. pooled (solid) annual-mean trajectories, both models,
all 3 combos, ssp245, full 2050 horizon — the direct visual of the "exactly 0.0pp" finding. Solo
and pooled lines overlap completely for both models across the entire 30-year horizon, confirming
the numeric result visually. `results/figures/s05_t2_pooled/s05_t2_pooled_vs_solo_ssp245.png`.

**Second addendum -- 2026-08-10 -- daily-resolution figures matching the project's standard
livestock-figure format.** User asked for the same format as
`s05_livestock_daily_full_horizon_ssp585.png` (stacked panels, thin daily lines, one condition per
panel) rather than the annual-mean version above. Required a rerun preserving daily chains (the
first D-95 pass only saved annual aggregates) -- `run_pooled()`/`run_solo_tabpfn()` now return
`(annual_df, daily_df)`, same underlying calls, no new compute. New script
`s05_t2_pooled_daily_plot.py`: 4 stacked panels (TabICLv2 solo/pooled, TabPFN solo/pooled), both
SSPs, full 2050 horizon -- TabICLv2 solo pulled directly from the existing
`s05_daily_chains_2050.parquet` (no new calls needed for that half). **Solo/pooled panels remain
visually indistinguishable at daily resolution**, for both models -- confirms the 0.0pp finding
holds day-by-day, not just in annual aggregate.
`results/figures/s05_t2_pooled/s05_t2_pooled_daily_full_horizon_{ssp245,ssp585}.png`,
`results/s05_t2_pooled_daily_chains.csv`, `results/s05_t2_tabpfn_solo_daily_chains.csv`.

### D-96 -- 2026-08-13 -- Fix: F-10/D-67's "vs. y_gapfilled" secondary MASE metric was never
rescored under D-80's climatology convention -- it silently mixed target AND baseline at once

**Context:** while retracing "what is the best model, how was it evaluated, what is y_true"
(direct user request), quoted F-10's gapfilled-target secondary metric (`TabPFN+species`
MASE=0.944 gapfilled, per `b16_final_table_vs_gapfilled_best_config.csv`/`F10_results.md`)
alongside the observed-target climatology-scored headline (0.715). **User caught the problem
immediately**: "the MASE for y_gapfilled is weird... but perhaps its because climatology refers
to the wrong variable as well here?" Checked the actual code, not just the table.

**Root cause, confirmed via direct code inspection:** `b16_foundation_models_v3.py` (and every
sibling `b16_*_gf`/`b16_recursive_rollout_v3_gapfilled.py`/`b16_dl_models_v3.py` script) computes
the gapfilled-target row via `rr.bin_metrics(y_gf, yp, target_dates, anchor, y_persist=persist)`
where `persist = chain_persistence(anchor_val, N_DAYS)` and `anchor_val = dft.loc[anchor,
"y_gapfilled"]` -- the OLD (D-37) chain-persistence baseline, held flat from the anchor's
gap-filled value. D-80's climatology rescoring (`mase_climatology()` in
`temp_forecasting_pipeline.ipynb`) was only ever built and applied with `target_filter="observed"`
as its sole mode -- it never touched the gapfilled-target secondary metric at all. So **0.715
(observed, climatology) and 0.944 (gapfilled, persistence) differ in both target AND baseline
convention simultaneously** -- not an isolated target-only comparison, despite being presented
side by side as one. A second, independent problem: the persistence baseline's anchor value is
itself always `y_gapfilled`-sourced regardless of which target it scores against -- the exact
apples-to-oranges pattern D-71's follow-up ("fairness fix", `Climatology_gf`) already identified
and fixed for a *different* comparison (persistence vs. climatology on the observed target); that
fix was never propagated into the foundation-model gapfilled-secondary-metric scripts.

**Fix (`b16_gapfilled_climatology_fix.py`, pure arithmetic, zero new model calls):** built a
`Climatology_gf` baseline -- day-of-year climatology (D-80's exact recipe, +/-7-day window) built
from `y_gapfilled` history instead of `y_observed`, reusing D-71's already-approved
"baseline sourced from the same series as the target" principle
(`b10_b13_climatology_gf_baseline.py`) -- then scored it against `y_gapfilled` truth (the genuinely
fair combination neither prior script computed: D-71's `Climatology_gf` was scored against
`y_observed` truth; F-10's gapfilled-target rows used persistence, not climatology, as the
baseline). Recomputed `MASE_climatology_gf = MAE_model / MAE_climatology_gf` from already-saved
per-bin MAE columns across the full 11-model roster (foundation + trees/SARIMAX/ensembles + DL),
all 3 towers, all 5 anchors, all 7 F-10 feature configs.

**Result: the qualitative conclusion survives, the numbers move.**

| Model (own best-known config) | Old (persistence, mismatched) | New (Climatology_gf, fair) |
|---|---|---|
| TabPFN+species | 0.944 | **0.906** |
| TabICLv2+species | -- | **0.938** |
| Ensemble_unweighted+species | 0.749 | **0.665** |

**Ensemble_unweighted still beats TabPFN+species by a wide margin under the corrected, fair
metric** (0.665 vs. 0.906) -- the same direction and similar magnitude as before, so F-10's
original "ranking flips under the gapfilled target" finding was directionally right even though
its own number was computed on a mismatched baseline. Full 11-model best-config table (fair
metric): Ensemble_MASEweighted (0.665) and Ensemble_unweighted (0.665) top, then XGB (0.670),
LightGBM (0.684), RF (0.702), SARIMAX (0.831), TabPFN (0.903, best config `BASE+bodyweight` not
`BASE+species`), TabICLv2 (0.922), TFT (1.013), LSTM (1.122), DLinear (1.502, worst). R2 columns
are unaffected by this fix (unchanged from the original table) since R2 doesn't depend on the MASE
baseline at all -- confirms the fix touched only what it should have.

**No change to any standing recommendation** -- `TabPFN+species` (observed target, climatology
MASE=0.715) remains the champion per D-36/D-37's authoritative-target convention; this fix only
corrects the secondary/diagnostic gapfilled-target comparison column, which was already flagged as
non-authoritative and carries the same circularity caveat as before (trees/ensembles train
directly on `y_gapfilled`, so they retain a structural advantage on this metric regardless of
baseline choice). **Standing lesson, same shape as D-83's addendum earlier this session**: when a
project convention changes (D-80's persistence->climatology switch), every script that computes
that metric needs to be checked, not just the one script that motivated the change -- secondary/
diagnostic tables are exactly where a stale convention hides longest. Files:
`notebooks/05_benchmarking/b16_gapfilled_climatology_fix.py`,
`results/b16_climatology_gf_baseline_v3.csv`, `results/b16_gapfilled_climatology_fix_all_configs.csv`,
`results/b16_gapfilled_climatology_fix_best_config.csv`.

### D-97 -- 2026-08-13 -- Supervisor feedback (3 items): livestock scenario plausibility, fertilizer
units clarity, annual CH4 generation for visualization

**Context:** three items of feedback from a supervisor meeting on S-05's scenario analysis: (1) is
3x livestock plausible given real catchment area -- research actual maximum livestock capacity and
redesign around max-capacity/baseline/half-baseline; (2) elaborate fertilizer practices -- what are
the units, what's the difference between rate and frequency; (3) generate annual methane generation
for visualization once the above are resolved.

**Item 1 -- livestock ladder redesigned, plausibility concern confirmed quantitatively.** The old
1x/2x/3x multiplier scales the smoothed day-of-year climatology curve, not raw data. Checked
directly against each catchment's own real historical instantaneous max (T2=4.51, T4=4.99,
T9=5.65 LSU/ha, `forecast_daily_v3.csv`): at T4/T9 the old 3x (7.44/7.74 LSU/ha) exceeded the
highest single day this catchment has EVER recorded by 30-50%, sustained for a full smoothed year
-- while at T2, 3x (2.13) undershot both that catchment's own historical spike and external
literature benchmarks. Same flat multiplier was simultaneously too extreme for T4/T9, too mild for
T2. Pulled external grounding (UK grassland stocking literature: typical conventional 1.5-2.5
GLU/ha, >3 GLU/ha only under "very best conditions"; NVZ 170 kg N/ha/yr manure cap, ~2 GLU/ha by
common convention; AHDB rotational-grazing case study reaching 2.4 LSU/ha). **User-confirmed
redesign** (AskUserQuestion): 4 absolute-anchored levels -- `half` (0.5x), `baseline` (1x,
unmodified), `lit_ceil` (uniform 3.0 LSU/ha literature ceiling), `own_max` (each tower's own real
historical peak) -- x 2 families (`all_species`, `cattle_alone`, mirroring D-84's cattle-dominance
axis) = 7 combos, multipliers solved by bisection so each level's climatology peak hits its target
exactly (not assumed). Full 3-tower x 2-SSP x 5-GCM x 10-realization sweep (2,100 calls, 6,901s).
**Result: monotonic, sensible response, cattle-dominance reconfirmed** (`all_species` vs.
`cattle_alone` within a few points of each other at every level, e.g. T4 own_max +128.9% vs.
+132.4%) -- no change to any standing conclusion, this redesign fixes scenario-level plausibility,
not the species-dominance finding. Fully additive: new module
(`build_transient_scenario_drivers_livestock_v2.py`) and new output
(`s05_practices_livestock_v2.csv`) alongside the untouched original 27-combo sweep.

**Item 2 -- fertilizer units fix, found a real data-labeling issue.** `Application_rate_per_ha` is
kg PRODUCT/ha, not kg NITROGEN/ha, and the `fertN` channel (`build_management_features.py`'s
`classify()`, used project-wide) tags any inorganic fertiliser event regardless of nitrogen
content -- 31% of "fertN"-tagged events site-wide are 0%-N products (P/K/S/Mg-only, e.g. Triple
Superphosphate). The original S-05 fertilizer template (D-86) inherited this: `mean_rate` was
kg-product/ha averaged over ALL events including non-nitrogen ones. **Fix, scenario-scope only**
(user-confirmed, `build_management_features.py` itself left untouched given the Sept 1 deadline and
~15 prior experiments depending on it): recomputed the template as true kg N/ha
(`Application_rate_per_ha x N-content-%/100`, parsed from `Application_Info`), restricted to N%>0
events. Corrected template: T2 3.3 true-N events/yr @ 51.9 kg N/ha (was 5 @ 137 kg product/ha), T4
7.4 @ 43.6 (was 8 @ 127), T9 1.55 @ 26.1 (was 4 @ 90) -- T9's frequency drops the most, since most
of its "fertiliser" events turn out not to be nitrogen at all. **Units now stated plainly: rate =
kg true N/ha per application; frequency = applications/year** -- directly answers the supervisor's
question, and the two were already independently-scalable axes in the existing `FERT_LEVELS`
design, just never clearly labeled. **Rerun result: unchanged conclusion, corrected numbers** --
still <5% effect, still sign-inconsistent (T2 −4.1%/−0.5%, T4 +1.1%/+0.8%, T9 0.0%/−0.1% for
frequency/rate) -- the original "redundant on the rich base" finding holds. Original
(pre-correction) files backed up as `*_PRE_D97_UNITS_FIX.csv`, not deleted, since this rerun
overwrites the fixed-filename originals rather than sitting alongside them the way the livestock
redesign does.

**Item 3 -- annual CH4 generation, standard unit conversion, applied to all 3 finalized axes.**
`annual_mean` (mean daily FCH4 flux, nmol CH4 m⁻² s⁻¹) converted to kg CH4 ha⁻¹ yr⁻¹ via
`flux x 1e-9 (mol) x 16.04 (g/mol CH4) x 1e-3 (kg/g) x 10,000 (m²/ha) x 31,536,000 (s/yr) =
flux x 5.0584` (verified arithmetically), plus total catchment mass via each tower's real fenced
area (T2=6.65, T4=7.75, T9=7.75 ha). Exact (no approximation) since `annual_mean` is already the
equally-weighted mean of 365 daily chain values. Applied to livestock_v2, grazing, and
(corrected-units) fertilizer axes -- e.g. T4 baseline livestock = 72.2 kg CH4/ha/yr (559 kg/yr
total catchment), rising to 165-168 kg CH4/ha/yr (1,280-1,300 kg/yr total) at `own_max`.

**Operational note, not a methods issue:** two of the three background sweeps this session
appeared to hang (zero output for 10-20+ minutes) and were killed/restarted unnecessarily twice
before the real causes were found: (a) genuine GPU contention from launching two local-inference
jobs concurrently once, and (b) monitoring the wrong file after redirecting a script's stdout to a
separate log rather than letting the task tool capture it, and (c) grep-piped live-tail buffering
delaying visible output despite the process being healthy (confirmed via CPU-time/GPU-utilization
checks, not just waiting). No science was affected -- all final results came from clean, verified
runs -- but ~30-40 minutes of wall-clock time were lost to false-alarm troubleshooting. Lesson: when
backgrounding a script, let the task tool capture stdout directly (don't redirect to a separate
file) and verify a "stuck" job via CPU time / GPU utilization before killing it, not just elapsed
time with no visible output.

**Files:** `src/features/build_transient_scenario_drivers_livestock_v2.py`,
`notebooks/07_scenario_analysis/s05_livestock_v2_trajectory.py`,
`s05_livestock_v2_daily_chains_subset.py`, `s05_livestock_v2_daily_chains_plots.py`,
`s05_annual_ch4_generation.py`; `src/features/build_transient_scenario_drivers_practices.py`
(edited in place, D-97 units fix); `results/s05_practices_livestock_v2.csv`,
`s05_livestock_v2_summary.csv`, `s05_fertilizer_corrected_summary.csv`,
`s05_annual_ch4_{livestock_v2,grazing,fertilizer}.csv` + `_summary.csv`; figures in
`results/figures/s05_summary/s05_livestock_v2_*.png` and `results/figures/s05_annual_ch4/`.
Full detail: `notebooks/07_scenario_analysis/s05_results.md`.

### D-98 -- 2026-08-13 -- Additive test: corrected fertN amount + a genuinely new frequency
feature, added to each of the 3 processes' own best-performing model

**Context:** direct user follow-up to D-97's units fix and the earlier "is fertN actually used"
discussion. Question: does the CORRECTED fertN feature (true kg N/ha, not the old product-mass
figure) plus a real frequency signal (which never existed as a real-data feature anywhere in this
project -- only as an S-05 scenario-template parameter) help, if added to each process's own
champion? Explicitly scoped **additive-only**: new feature file, new test scripts, zero edits to
any production pipeline (`gapfill_rfm.py`, `build_management_features.py`, the F-10 forecasting
configs, or S-05's driver scripts).

**New feature, built once, shared across all 3 tests** (`src/features/
build_fertN_amount_freq_features.py` -> `data/Hourly/fertN_amount_freq_features.csv`):
- `fertN_amount_t{N}` -- same recency-decay mechanism as the existing `mgmt_fertN_rate`
  (`exp(-days_since_event/14) x magnitude`), but magnitude is now true kg N/ha (D-97's
  N-content-%-adjusted correction, applied here to the REAL-DATA pipeline, not just the S-05
  scenario template) and restricted to true-nitrogen events only.
- `fertN_freq_t{N}` -- **new**, no precedent: trailing-365-day count of true-N application events,
  computed per real timestamp from real event history.
Verified against D-97's known per-tower event counts (T2=24, T4=54, T9=11 true-N events site-wide
in-catchment) before use.

**Process 1 -- gap-filling.** User redirected mid-run: test TabICL-solo only (D-79's
benchmark-best model), not RF -- RF's own run (kept on disk for reference, not deleted) already
showed a consistent degradation at all 3 towers; TabICL was run to check whether that's an
RF-specific overfitting artifact or holds across a structurally different model family
(foundation/in-context, not a fitted tree ensemble).

| Tower | RF champion | RF+fertN | RF delta | TabICL champion | TabICL+fertN | TabICL delta |
|---|---|---|---|---|---|---|
| T2 | 0.576 | 0.556 | -0.020 | 0.676 | 0.539 | **-0.137** |
| T4 | 0.404 | 0.373 | -0.031 | 0.428 | 0.376 | -0.052 |
| T9 | 0.426 | 0.382 | -0.044 | 0.423 | 0.308 | **-0.115** |

**Result: degrades gap-filling at every tower, for BOTH models -- TabICL degrades even more than
RF.** This is not an RF-specific overfitting quirk (F-01's original explanation for excluding raw
fertN) -- the corrected, richer feature (true N amount + genuine frequency) still hurts, and hurts
a completely different model family more. **Reinforces F-01's exclusion decision with new
evidence, on the corrected feature, across two structurally different models** -- fertN
(amount+frequency) should stay excluded from gap-filling, full stop, not just "the old broken
version of it should be excluded."

**Process 2 -- forecasting (`TabPFN+species` champion).** MASE_climatology 0.7150 -> 0.7142
(observed target) -- a 0.08% improvement, noise-level. By tower: T2/T9 marginally better, T4
marginally worse. **No meaningful effect** -- consistent with `BASE+mgmt`'s own small-but-real
gain over plain `BASE` (0.724->0.719, checked directly this session) being diminishing-to-zero
once species density is already in the model, not additive on top of it.

**Process 3 -- S-05 scenario architecture (`TabICLv2` + `FX_A_SPECIES`).** MASE_climatology 0.7588
-> 0.7591 -- essentially flat, a tiny regression. T2/T4 marginally worse, T9 marginally better.
**No meaningful effect**, same direction/magnitude as Process 2.

**Overall reading:** the "redundant on the rich base" pattern this project has now found
repeatedly (F-01/F-04/F-05 on real historical importance, D-86/D-97 on scenario response) extends
a further two ways here: (a) to real forecast accuracy specifically, not just feature-importance
diagnostics or scenario-response magnitude, and (b) with a twist for gap-filling specifically --
there it's not neutral/redundant, it's actively harmful, confirmed now on the corrected feature
and across two model families rather than resting on F-01's original (product-mass, RF-only)
finding. **No change to any standing recommendation** -- neither production pipeline gains
anything from adding this feature; gap-filling's existing exclusion is reinforced, not
reconsidered.

**Files:** `src/features/build_fertN_amount_freq_features.py`,
`data/Hourly/fertN_amount_freq_features.csv`;
`notebooks/04_feature_engineering/d98_fertN_amount_freq_gapfill_test.py` (RF, superseded per
user direction, kept for reference), `d98_fertN_amount_freq_gapfill_tabicl_test.py` (TabICL,
headline); `notebooks/05_benchmarking/d98_fertN_amount_freq_forecast_test.py`;
`notebooks/07_scenario_analysis/d98_fertN_amount_freq_s05_test.py`; `results/
d98_gapfill_fertN_amount_freq_final_summary.csv` (RF vs. TabICL comparison table),
`d98_forecast_fertN_amount_freq_summary.csv`, `d98_s05_fertN_amount_freq_summary.csv`.

---

### D-99 -- 2026-08-17 -- D9: confidence-gated self-training second pass for TabICL, a
"prompting strategy for TFMs" experiment -- negative

**Context:** the user's own idea, given directly as feedback: once TabICL's uncertainty bounds are
available, do a second pass -- set a confidence-interval-width threshold, treat held-out points
predicted within that bound as "successfully predicted," move them into the training set with
their own prediction as a pseudo-label, and re-run prediction for the remaining points to see if
they improve. Explicitly requested additive-only. Three design questions were resolved with the
user before building (`AskUserQuestion`): (1) the width gates *promotion into training data only*,
never exposed to the model as an input column -- keeps this mechanistically distinct from D6's
already-negative "uncertainty as a feature" result, not a repeat of it; (2) a single second pass,
not iterated to convergence; (3) the width threshold swept as percentile bands (narrowest 25% / 50%
/ 75% of each fold's own width distribution) rather than one fixed nmol cutoff.

**Location and mechanism:** new cells (§19, `temp_gap_filling_pipeline.ipynb`, cells 314-317,
inserted before the notebook's final model-cache-maintenance cell, which was renumbered §19 -> §20
as the only non-additive edit -- a heading-text change, not a logic change). TabICL-solo only (per
D5.5's pooling-hurts-TabICL finding), champion `FEATURES` (unchanged since D1), all 3 towers x 5
scenarios x `N_REPS`, reusing `TabICLRegressor`'s native `output_type="quantiles"` call (D6.2) for
the interval width. Per fold: pass-1 fit -> point prediction + 90% interval width -> promote the
narrowest-width points into the training pool using their OWN pass-1 prediction as pseudo-label
(never the true target, so the method is legitimate for real production gap-filling, not just this
benchmark) -> pass-2 refit on real+pseudo rows -> predict only the remaining (non-promoted) points
-> hybrid output = pass-1 for promoted points, pass-2 for the rest.

**Execution note, stated plainly:** a first attempt ran the full notebook top-to-bottom via
`nbconvert --execute --inplace` in the background: after ~50 minutes with no cell-level progress it
became clear this was going to take 2-4+ hours, because -- unlike D1-D8's `_model_cache/`-backed RF
fits -- HyperImpute (~83 min historically, invoked twice) and every TabICL fit in D1-D8 (`fit_tabicl`
explicitly never caches, "every call fits fresh") recompute from scratch on every full run, none of
which D9 actually depends on. Killed that run and instead extracted the exact, verbatim source of
only the cells D9's dependency chain needs (data loading through `d_all`/`FEATURES`/`fit_tabicl`/
`FCH4_EXTREME_THRESHOLDS`/`D6_ALPHAS`, cells 0-45 + 115 + 266 + 282, plus D9's own 3 cells) into a
standalone script, executed that (~15 min total), and used the **`baseline` arm's exact reproduction
of D5.5's champion numbers (0.676/0.428/0.423) as the safety check** that this shortcut didn't
silently diverge from the real notebook's state. It didn't -- verified bit-for-bit. The real
production checkpoint files (`_data/d9_checkpoints/`, `_data/d9_*.csv`) were written to their real
locations by this run, and the notebook's own D9 cell outputs were populated with the real captured
stdout -- so a future genuine top-to-bottom `nbconvert` execution will hit these checkpoints and
reproduce the identical numbers near-instantly, closing the loop honestly without requiring the
multi-hour full rerun to *validate* this specific addition.

**Result -- negative-to-neutral, and the pre-registered risk confirmed empirically, not just
theorized.** Full-set hybrid vs. pass-1-only (operational headline): T2 degrades monotonically as
the promoted band widens (-0.006 / -0.011 / **-0.017** at p25/p50/p75); T4 has a small genuine edge
at p25/p50 (+0.002 / **+0.005**) that flips negative by p75 (-0.005); T9 is noise throughout (0.000
/ -0.002 / +0.002). The cleaner remaining-points-only comparison (pass-2 vs. what pass-1 alone
would have scored on the identical non-promoted points) shows the same pattern.

**Spike-stratum breakdown is the sharpest finding**, and it's exactly what was flagged *before*
running, not an after-the-fact rationalization: narrow-interval (promoted) points should skew
toward non-spike/near-mean values (temp_mds.ipynb section 17's regression-to-the-mean finding), so
pass 2's enriched training pool could end up MORE mean-biased, not less. Confirmed at T2 (-0.002 /
-0.003 / -0.007) and **clearly at T9** (-0.016 / -0.039 / **-0.053**, worsening with band width) --
T4 is the one exception, improving at p25/p50 (+0.010) before the same reversal hits at p75.
Promotion rates landed exactly on 25.0/50.0/75.0% at every tower, confirming the percentile-gating
mechanism worked as designed.

**Outcome: not adopted, no change to the standing gap-filling champion (D-79's TabICL-solo,
0.676/0.428/0.423) or the production RFm champion (D-77, 0.576/0.404/0.426).** Joins D5-D8's own
mostly-flat/negative pattern in this notebook's TabICL-refinement arc (D8's row-cap bagging remains
the one adoptable positive finding from that whole line) rather than breaking it. See D-77, D-78,
D-79, `notebooks/03c_gap_filling_revisited/summary.md` section 19,
`notebooks/03c_gap_filling_revisited/temp_gap_filling_pipeline.ipynb` (section 19, cells 314-317),
`results/` n/a (all D9 outputs live under the notebook's own `_data/` per this notebook's
established convention) -- `_data/d9_raw_predictions.csv`, `_data/d9_summary.csv`,
`_data/d9_promoted_pct.csv`.

### D-100 -- 2026-08-17 -- Delta-method bias correction for Phase 07 scenario outputs (S-01/S-04/S-05),
anchoring model predictions to real historical means

**Context:** S-01's own baseline-reconstruction sanity check (`s01_results.md` Finding 1) already
documented that the 1.0x (no-perturbation) scenario doesn't reconstruct each tower's real historical
mean exactly (T2 +20.2%, T4 -2.0%, T9 +9.2%) and judged this small enough to trust the perturbed-
scenario comparison as-is -- **this task does not treat that as a newly-discovered bug**; it applies
a more rigorous, standard climate-impact-modelling correction on top of that same known, previously-
accepted gap: trust the model for the SHAPE of the change, anchor the LEVEL to the real observed
mean (`corrected = real_mean + (predicted_scenario - predicted_baseline_1x)`). Explicitly additive-
only, both raw and corrected numbers reported side by side, nothing overwritten or hidden.

**S-01/S-04 (same frozen model, same bias, one offset transfers directly):** per-tower bias offset
re-derived from `s01_scenario_summary.csv` (T2 +3.91, T4 -0.60, T9 +3.39), applied identically to
S-04 (reuses S-01's exact joblib artifacts, no retraining, so the same bias mechanism applies
unchanged). S-04's % change recomputed as a PAIRED comparison within each (tower, ssp, gcm,
realization, year) group against that same group's corrected 1x value, not against the single S-01
snapshot, since S-04's own realization/year structure is a more statistically sound denominator.

| Tower | Raw 3x % (S-01 snapshot) | Corrected | Raw 3x % (S-04 pooled, both SSPs) | Corrected |
|---|---|---|---|---|
| T2 | +33.8% | **+40.7%** | +38.5-38.7% | **+48.1-48.5%** |
| T4 | +138.2% | **+135.4%** | +156.0-156.8% | **+152.4-153.2%** |
| T9 | +104.7% | **+114.4%** | +120.1-120.5% | **+133.1-133.7%** |

A modest, expected-magnitude correction (single-digit-to-low-teens percentage points) -- consistent
with S-01's own already-documented, already-accepted gap being real but small.

**S-05 (TabICLv2 + `FX_A_SPECIES`, S-03's Variant A + species split -- S-05's current, latest
architecture, never checked before this task per direct user instruction to scope the check to
this specific config): a much larger, structurally different finding.** S-05 is a genuinely
different model family (zero-shot foundation model, not a fitted Ridge+tree hybrid), so S-01's
small gap was explicitly not assumed to transfer -- checked fresh, from source. **Result: the
1x baseline underpredicts the real historical mean by 40-80% at every tower, in the same direction
throughout** (checked against both `y_gapfilled`, matching S-01/S-04's own convention, and
`y_observed`, this project's authoritative target -- consistent under both): T2 -69.9%(gf)/-81.1%
(obs), T4 -52.4%/-51.6%, T9 -40.8%/-39.8%. This is not the same modest gap recurring in a new
model -- it is 2-4x larger in magnitude and always in the undershoot direction, never overshoot.

**Applying the same delta-method correction to S-05 roughly HALVES several of this project's most
prominently reported headline percentages:**

| Result | Raw | Corrected |
|---|---|---|
| Cattle-3x-alone, T4 (main species table) | +213.9% | **+101.8%** |
| Cattle-3x-alone, T9 | +187.3% | **+110.4%** |
| `own_max` all-species (D-97 redesigned ladder), T4 | +128.8% | **+61.2%** |
| `own_max` all-species, T9 | +142.8% | **+84.2%** |
| `own_max` all-species, T2 | +57.9% | **+16.8%** |

**Explicit caution, stated plainly, not glossed over**: this correction should be trusted less than
S-01/S-04's. Delta-method's core assumption is that the model's own SHAPE of response is reliable
even when its LEVEL isn't -- that assumption gets progressively less defensible the larger the
absolute bias is relative to the signal, and a 40-80% underprediction is a large bias by any
standard. The qualitative finding (cattle dominates far beyond its LSU-weight share) is unaffected
in direction -- it still clearly holds under either raw or corrected numbers -- but the exact
MAGNITUDE quoted for it should now be read as genuinely uncertain between the raw and corrected
figures, not settled at either one. **This affects prior headline citations directly** (e.g. the
"+214.5%" cattle-dominance figure already cited in `BEST_RESULTS.md`, `DECISIONS.md` D-85, and the
just-rewritten report Chapter 7) -- flagged for follow-up, not silently left inconsistent.

**No retraining anywhere** -- purely a post-hoc arithmetic correction on already-computed
predictions, fully additive (new `_bias_corrected` files, originals untouched).

**Files:** `notebooks/07_scenario_analysis/d100_bias_correction_s01_s04.py`,
`d100_bias_check_s05.py`; `results/s01_scenario_summary_bias_corrected.csv`,
`s04_trajectory_realizations_bias_corrected.csv`, `s04_trajectory_summary_bias_corrected.csv`,
`s05_trajectory_realizations_2050_bias_corrected.csv`,
`s05_practices_livestock_v2_bias_corrected.csv`.

### D-101 -- 2026-08-18 -- S-06: bias-correcting the CMIP6/LARS-WG driver data itself (not just the
model's output, D-100) -- full rerun of every S-05 process against corrected drivers

**Context:** direct follow-up to the S-05 AOA discussion (baseline scenario flagged 63-68% of days
as outside the training envelope even with no livestock perturbation at all). User's own hypothesis,
checked rather than assumed: does the simulated climate driver data itself need preprocessing
before AOA looks this bad?

**Diagnosis, same-calendar-year comparison (2020-2024 real vs. simulated, ruling out a
baseline-period-mismatch explanation before concluding anything):**
- **Precipitation: simulated ~4.3x wetter than real** (real mean 0.74mm/day vs. sim 3.16mm/day;
  median 12x higher; max 6x higher). Traced two contributing causes: (a) a genuine, confirmed
  construction bug in `fx_PRECIP_sum` (`.resample("D").sum()` silently treats missing hours as 0mm
  rather than excluding them -- 23% of hourly readings are missing -- but correcting this only
  explains ~20% of the gap, not the bulk of it); (b) the larger remainder is a real property of the
  CMIP6/LARS-WG driver data itself, most likely coarse-grid (~1.25x1.9 degree) site-representativity
  rather than a fixable data-quality issue on the real side -- confirmed by holding calendar years
  identical on both sides, which rules out "different multi-decade baseline period" as the
  explanation.
- **Temperature: simulated 2-3.5C COOLER than real** (TA_min: real 9.78 vs. sim 7.65; TA_max: real
  18.04 vs. sim 14.63) -- the opposite direction from precipitation, and the opposite of what an
  earlier, methodologically weaker (mismatched-years) comparison had suggested.
- **Radiation: smaller mismatch** (~8% low on the mean), real's own max (689 W/m^2 as a 24h mean) is
  itself a likely sensor-artifact outlier, flagged but not pursued further this pass.

**Decision: rescale the driver, not just flag it as a limitation.** User's own reasoning, endorsed:
un-bias-corrected raw GCM/downscaled output fed directly into an impact model is the methodologically
*weaker* choice in climate-impact practice, not the safer one -- local bias-correction to the site's
own observed climatology is the standard expected step. Convention (standard practice, not
improvised): **additive** correction for temperature (interval-scale -- multiplying temperature
isn't physically meaningful; `corrected = simulated + (real_mean - sim_mean)`), **multiplicative**
correction for precipitation/radiation (non-negative, ratio-scale, skewed -- an additive shift risks
negative rainfall on dry days; `corrected = simulated * (real_mean / sim_mean)`).

**Implementation, fully additive.** Correction factors computed **per-GCM** (pooling realizations
1-10 and a 2020-2030 near-term window -- deliberately early, so the correction targets each GCM's
own near-term bias without absorbing the genuine future-warming signal it's supposed to preserve),
against NWFP's own full real historical record (Tower 4, best coverage of the 3 towers -- one
site-level reference used since all 3 towers share one CMIP6 file and real inter-tower climate
differences at a single farm don't plausibly explain anything at this magnitude). All 5 GCMs show
consistent bias direction/magnitude (TA_min offset +1.8 to +2.7C, TA_max +2.6 to +3.6C, RAIN ratio
0.225-0.240, SWIN ratio 1.03-1.08) -- reassuring, not GCM-specific noise.

**Bias-corrected `.dat` files** (identical format to the originals) written to a new directory,
`data/Simulated Climate Data Bias Corrected/` (100 files: 5 GCMs x 2 SSPs x 10 realizations,
matching S-05's own `stratified_realizations(10)` scope) -- `src/features/
build_bias_corrected_cmip6.py`. A drop-in-compatible `load_transient_years()` reads from the new
directory (`build_transient_scenario_drivers_s06.py`), and every S-05 runner script is reused
**completely unmodified** via a single monkey-patch of the module-level `load_transient_years`
name on the already-imported `s05_practices_trajectory` module (Python resolves global names from
the defining module's own namespace at call time, not import time -- the same technique already
validated for D-98's `TOWERS`/`SSPS` override) -- zero duplication of `run_axis()`/
`build_livestock_frame()`/etc.

**Smoke-tested before the full run** (1 tower, 1 SSP, 1 realization, all 3 axes): confirmed the
pipeline runs correctly end-to-end and AOA flagged-% moves in the expected direction, by a modest
amount -- baseline ~63%->60%, `own_max__cattle_alone` ~51%->47%. **Expected to be partial, not a
full fix, and it is**: AOA is computed across all 13 `FX_A_SPECIES` dimensions, and only the 4
climate drivers were corrected -- the livestock-density dimensions (which the elevated scenario
levels deliberately push into unusual territory) are untouched by this correction and still
contribute real, legitimate extrapolation flags.

**Full sweep launched** (3 towers x 2 SSPs x 5 GCMs x 10 realizations, all 7 livestock-ladder
combos + 3 grazing + 3 fertilizer levels -- matching S-05's exact coverage, ~3.5h total) --
results/full write-up pending completion, to follow as an addendum. **No change to S-05's own
files or conclusions** -- this is a parallel, fully additive companion body of work (S-06), not a
replacement.

**Files:** `src/features/build_bias_corrected_cmip6.py`,
`build_transient_scenario_drivers_s06.py`; `notebooks/07_scenario_analysis/s06_master_runner.py`;
`data/Simulated Climate Data Bias Corrected/` (100 files); `results/s06_bias_correction_factors.csv`.

### D-102 -- 2026-08-18 -- I-03: interpretability recalibrated for the current forecasting champion
(TabPFN+species) -- closes the same "predates the champion" gap U-04 already closed for UQ

**Context:** user concern, checked and confirmed rather than assumed: I-02 (D-61, 2026-07-06), this
project's only comprehensive rollout-forecasting interpretability pass, predates TabICLv2 joining
the roster (D-66, 2026-07-09) and F-10's species-disaggregated features (D-67, 2026-07-10) by 3-4
days -- both of which produced the standing champion, TabPFN+species (MASE=0.715 climatology-scored,
D-80). I-02's SHAP/permutation numbers, the ones currently written into Chapter 6, were computed on
the OLD 8-model roster and OLD (BASE-only) feature set. Confirmed via direct grep that no I-03/I-04
equivalent existed anywhere in this project prior to this entry -- unlike UQ, which got U-04 as an
explicit champion-recalibration pass, interpretability had never received the same treatment.

**Scope (champion-focused, mirrors U-04's precedent, user-directed):** TabPFN only -- "the best
forecasting model... the model that was ingested for S-03" -- confirmed via S-03's own 11-model
table (TabPFN MASE=0.855, the lowest/best of all models tested there, including TabICLv2 at 0.930).
TabICLv2 not covered this pass -- flagged as a cheap, natural follow-up (same zero-shot cost
profile), not executed.

**Method: unchanged from I-02's own TabPFN treatment** (`i02_multi_anchor_tower.py` lines ~277-304)
-- permutation importance (TabPFN's only available substitute for a native signal; architecturally
mismatched with SHAP's row-wise framework per I-02's own reasoning, not revisited). One baseline
zero-shot rollout + one single-shuffle permutation per feature column (seeded by anchor year,
matching I-02 exactly -- not upgraded to n_repeats>1, keeping old-vs-new numbers comparable) per
(anchor, tower). Only the feature set changed: BASE+species, 52 `fx_` columns from
`forecast_daily_v3.csv` (F-10's actual champion config, construction identical to
`u04_champion_uq.py`'s `fit_stage_champion`). Same 3-tower x 5-anchor sweep as I-02/U-02/U-04.
Smoke-tested (1 tower, 1 anchor, 3 features) before the full 795-call run. **Actual runtime: ~22
minutes**, zero training.

**Headline: `fx_lsu_dens` still dominates, and the new species split's real driver is isolated for
the first time.** Overall ranking (mean importance, pooled): `fx_lsu_dens` 1.1456 (#1),
`fx_cattle_dens` 0.8043 (#2), `fx_grazing_active` 0.3750 (#3), `fx_total_liveweight_dens` 0.2807
(#4) -- confirms I-02's original livestock-dominance finding survives on the actual champion
config. **New finding I-02 could not have made** (it predates species features entirely):
`fx_sheep_dens` (0.0143) and `fx_lamb_dens` (0.0320) rank near the bottom of all 52 features, an
order of magnitude below `fx_cattle_dens` -- F-10/D-67's species-disaggregation gain is
concentrated entirely in the cattle component, not a generic benefit of splitting by species. This
independently corroborates S-05's scenario-projection cattle-dominance finding (cattle tripling
~triples FCH4 at T4/T9; sheep/lamb stay under 25% even at 3x) via a completely different method
(real-anchor permutation importance vs. scenario dose-response) -- convergent evidence, not
circular.

**Tower 2 confirms its livestock-blindness a fourth independent time.** T2's top-10 importance list
contains zero livestock features at all -- top-ranked are `fx_TA_max`/`fx_TA_mean`/`fx_TS_lag28`/
`fx_SWIN_mean` (meteorological/soil-temperature only). Consistent with U-03 (rollout stress test),
S-01 (AOA/climatology-max check), and S05-T2/D-95 (pooling test) -- now a settled, cross-method
characteristic of that catchment, not a single-method artifact. T4/T9 both show the identical
top-3 ordering (`fx_lsu_dens` > `fx_cattle_dens` > `fx_grazing_active`).

**Practical implications:** (1) the interpretability gap this project carried since D-66/D-67 is
closed for the model that actually needs it; TabICLv2 remains open. (2) I-02's headline finding is
confirmed, not overturned, and sharpened with a mechanism I-02 structurally could not see. (3)
Chapter 6's current SHAP/permutation write-up (based on I-02) should be read as accurate in
direction but built on a superseded feature configuration -- worth a note or a light update citing
I-03 for the champion-specific claim. See D-102,
`notebooks/06_interpretability_uq/i03_champion_interpretability.py`, `I03_results.md`,
`results/i03_tabpfn_species_importance*.csv`.

---

### D-103 -- 2026-08-19 -- Recalculating R2_OLS (scipy/Zhu-style) for D-78's five non-TabPFN
challenger models -- ranking unchanged, TabPFN itself skipped by explicit user scope

**Context:** direct user follow-up to the earlier scipy-vs-sklearn R² discussion (`temp_mds.ipynb`
section 13.4/13.5, ported into `BEST_RESULTS.md` -- `R2_OLS`, the squared-Pearson-r/Zhu et al.
(2023a) convention, bounded [0,1], vs. this project's standing `sklearn.metrics.r2_score`,
unbounded below). Section 13.5 had already extended that comparison to Mean/MDS/MICE/RFm-family/
D1-D4, but never to D-78's six-model challenger comparison (LightGBM/XGBoost/TabPFN/TabICL/SAITS/
BI-LSTM, `temp_gap_filing_exploration copy.ipynb`). Checked before building anything (user's own
explicit request): confirmed no raw (actual, predicted) pairs exist anywhere on disk for these six
models -- `run_model()`'s per-fold `mets()` call only ever fed an aggregate `med_metrics()`
reduction into `model_comparison.csv` (`R2`/`RMSE`/`MAE`/`MBE` only), and this notebook's own
`mets()` predates the `R2_OLS` addition entirely (a genuinely different, older 4-value function than
`temp_gap_filling_pipeline.ipynb`'s). No free recompute was possible -- a real refit was required.
**Scope, per explicit user direction:** skip TabPFN (already documented ~3.7h for its 60-fold
sweep, and a clear, unambiguous loser on sklearn R² already) -- recalculate for the other five.

**Method: additive-only**, zero edits to `model_comparison.csv` or any existing notebook cell. A
standalone script (verbatim-extracted setup chain -- data loading through `FEATURES`/`frame()`/
`fit_lgbm`/`fit_xgb`/`fit_tabicl`/the SAITS+BI-LSTM helper functions, cells 0-52/56/103/107/110/113
of `temp_gap_filing_exploration copy.ipynb` -- plus new capture wrappers around each model's exact
original fitting logic) computes `R2_OLS` via the same `scipy.stats.linregress`-based formula as
`temp_gap_filling_pipeline.ipynb`'s `mets()`, alongside a **built-in verification checkpoint**: each
model's recomputed sklearn R² must reproduce `model_comparison.csv`'s original number, or the run
flags MISMATCH rather than silently trusting a possibly-diverged harness. Champion R² hardcoded
(0.576/0.404/0.426) rather than recomputed, since it's informational-only in this context and
recomputing it would mean rerunning the full RF gap-CV harness for no reason.

**Two real operational failures hit and fixed along the way, not just a straightforward run:**
1. The first full-notebook-style attempt used `jupyter nbconvert --execute --inplace` on the whole
   notebook, on the (wrong) assumption that D1-D8/HyperImpute/MICE would mostly hit existing
   checkpoints/caches the way `temp_gap_filling_pipeline.ipynb`'s D5-D9 sections do. They don't --
   HyperImpute alone is ~83min per the project's own D-78 notes, invoked twice, and every TabICL/
   LightGBM/XGBoost fit in that notebook's D1-D8 equivalent sections also refits from scratch on
   every run, none of which D-103 depends on. Killed after confirming (via `Get-Process` CPU
   accounting) it would cost 2-4+ hours for work this experiment doesn't need; replaced with the
   verbatim-extracted-dependency-chain approach described above (~15 min for setup).
2. The extracted script itself silently hung twice, for two unrelated reasons, both diagnosed via
   direct OS-level inspection (`Get-Process`/`nvidia-smi`) rather than assumption: **(a)** a
   `plt.show()` call inside the extracted EDA cells opened a real interactive matplotlib GUI window
   (confirmed via `MainWindowTitle` = "Figure 1") that blocked the headless script indefinitely with
   near-zero CPU use -- fixed with `matplotlib.use("Agg")` forced before any other import. **(b)**
   after that fix, SAITS's repeated per-scenario/per-tower model creation on GPU was never
   explicitly freed between fits (`del model`, `gc.collect()`, `torch.cuda.empty_cache()` all
   missing) -- confirmed via `nvidia-smi` climbing to 11.6/12.2GB VRAM by the 6th-7th SAITS fit,
   causing severe thrashing (a single fit's wall-clock time went from ~70-90s to effectively stalled
   for hours while CPU accounting showed almost no active computation). Fixed by adding explicit
   GPU-memory cleanup after every SAITS/BI-LSTM fit, plus per-(model,tower) checkpointing
   (`_data/d100_checkpoints/`, matching this project's own established checkpoint-and-resume
   convention) so a third interruption would not repay already-finished work. Total real compute
   once both fixes were in place: ~50 minutes.

**Result -- verification passed for 4 of 5 models exactly; SAITS showed small, expected
neural-net-training non-determinism, not a bug:**

| Model | Tower | R2_sklearn | vs. original | R2_OLS | Champion R2 |
|---|---|---|---|---|---|
| LightGBM | 2 / 4 / 9 | 0.522 / 0.410 / 0.422 | OK / OK / OK | 0.583 / 0.418 / 0.425 | 0.576 / 0.404 / 0.426 |
| XGBoost | 2 / 4 / 9 | 0.551 / 0.349 / 0.369 | OK / OK / OK | 0.619 / 0.380 / 0.375 | " |
| TabICL | 2 / 4 / 9 | 0.558 / 0.423 / 0.364 | OK / OK / OK | 0.587 / 0.425 / 0.368 | " |
| SAITS | 2 / 4 / 9 | 0.341 / 0.275 / 0.263 | MISMATCH (orig 0.358/0.293/0.285) | 0.562 / 0.289 / 0.304 | " |
| BI-LSTM | 2 / 4 / 9 | 0.237 / 0.155 / 0.146 | OK / OK / OK | 0.280 / 0.204 / 0.189 | " |

SAITS's ~0.017-0.023 R2 gap from the original is attributable to `pygrinder.mcar()`'s validation-set
masking depending on NumPy's *global* RNG state (not fully pinned by `torch.manual_seed()` alone),
which differs between this standalone script's execution history and the original notebook's --
expected for a re-trained deep model in a different harness, not a capture-logic bug. All other four
models verified bit-exact.

**Outcome: ranking unchanged, consistent with the sklearn-R2 story.** Every model's R2_OLS sits
meaningfully above its sklearn R2 (same direction section 13.5 already found for the RF/TabICL/MDS/
MICE family), but LightGBM and TabICL remain the only two that edge the champion, and only at Tower
4 (0.418/0.425 vs. champion's 0.404 under OLS too) -- SAITS/BI-LSTM stay clearly behind at every
tower under either metric. This extends section 13.5's own "R2-definition sensitivity is essentially
an MDS-specific phenomenon" finding to two more model families (SAITS/BI-LSTM) without changing its
conclusion. **No `BEST_RESULTS.md` champion change** -- purely a metric-completeness fill for D-78's
own table. See D-78, D-79,
`notebooks/03c_gap_filling_revisited/_d100_ols_runner_setup.py`,
`notebooks/03c_gap_filling_revisited/_d100_ols_runner_capture.py`,
`notebooks/03c_gap_filling_revisited/_data/d100_ols_recalc_summary.csv`,
`notebooks/03c_gap_filling_revisited/_data/d100_ols_recalc_raw_predictions.csv` (446,400 rows),
`notebooks/03c_gap_filling_revisited/_data/d100_checkpoints/`.

### D-104 -- 2026-08-19 -- S-06 only: `lit_ceil` corrected to 2.5 LSU/ha (UK regulatory ceiling),
replacing D-97's 3.0 synthesized estimate -- S-05 untouched

**Context:** D-97's `lit_ceil` level (3.0 LSU/ha) was the author's own synthesized round number,
set just above several converging but indirect sources (a ScienceDirect Topics tertiary aggregator,
an AHDB case study, NVZ regulation) -- already flagged as not a single paper's literal figure when
first questioned. User found a stronger, directly-citable primary source: **UK Countryside
Stewardship regulation, Annex 8** (non-SDA land) states a hard ceiling of **2.5 livestock units/ha**
("do not stock more than... 2.5 livestock units (LU) per hectare on non-SDA land... on average over
the year"). North Wyke (lowland Devon) is non-SDA, so 2.5 is the applicable figure.

**Scoped to S-06 only, per direct user instruction** -- D-97's original `lit_ceil` (target=3.0,
S-05's already-documented result) is left untouched, not retroactively invalidated. New multipliers
solved by the same bisection method (`build_transient_scenario_drivers_livestock_v2_s06.py`, a
thin S-06-only override -- `half`/`baseline`/`own_max` unaffected, reused unchanged): T2 all-species
m=3.5062/cattle-alone m=3.8016, T4 m=1.0097/1.0102, T9 m=0.9690/0.9593.

**Genuine finding, not smoothed over: Tower 9's own real baseline climatology peak (2.58 LSU/ha)
already exceeds 2.5** -- its solved multiplier is <1.0, meaning `lit_ceil` now sits BELOW
`baseline` for T9 specifically (confirmed in the rerun: baseline=19.21 vs. lit_ceil=18.76-18.77),
not above it as originally intended. T9 is, on real historical evidence, already operating above
the UK non-SDA regulatory stocking ceiling -- a real result about this specific catchment, not a
construction artifact (T2/T4 still show `lit_ceil` > `baseline` as originally intended, T4 only
marginally so under the tighter ceiling: 15.09->15.21/15.22).

**Rerun scope**: only the 2 affected combos (`lit_ceil__all_species`, `lit_ceil__cattle_alone`),
full coverage (3 towers x 2 SSPs x 5 GCMs x 10 realizations, 600 calls, ~34 min) plus the daily-
chains subset (12 calls) -- `half`/`baseline`/`own_max` unaffected, not rerun. Corrected rows
merged into the existing S-06 output files (row counts confirmed unchanged: 59,500 annual /
434,350 daily), superseding the 3.0-based S-06 `lit_ceil` rows in place. Affected figures
regenerated (`s06_livestock_v2_*` daily-chain set, `s06_annual_ch4_livestock_v2`).

**Files:** `src/features/build_transient_scenario_drivers_livestock_v2_s06.py`,
`notebooks/07_scenario_analysis/s06_lit_ceil_fix.py`; `results/
s05_practices_s06_livestock_v2.csv` (updated in place), `s06_livestock_v2_daily_chains_subset.csv`
(updated in place), `s06_annual_ch4_livestock_v2*.csv` (regenerated).

### D-105 -- 2026-08-19 -- New fertiliser scenario level `reg_cap`, regulatory-grounding check found
and fixed a real double-counting bug in the interim (never reached production, S-05/S-06 model
outputs unaffected)

**Context:** user asked whether the fertiliser scenario's amounts (T2/T4/T9's `+50% rate`/`+50%
frequency` levels) are grounded in real nitrogen regulation, mirroring the livestock ceiling's
gov.uk grounding (D-104). Checked directly against gov.uk ("Using nitrogen fertilisers in nitrate
vulnerable zones"): **NVZ N-max for grassland = 300 kg N/ha/yr, +40kg (340 total) for grassland cut
>=3 times/yr.**

**Investigation surfaced a real bug, but it was in chat-only arithmetic, not in any pipeline code.**
Naively multiplying `FERTN_TEMPLATE`'s pooled n_events x mean_rate (T4: 7.4 x 43.6 = 323 kg N/ha/yr)
and comparing it to the 300 cap made it look like T4 already exceeded NVZ regulation -- surprising
for a UKRI research platform, and the user pushed back. Root cause, confirmed directly against
`Field_Event_Data_Format_1.csv`: **Catchment 4 (and 9) are each TWO independently-fertilised
sub-fields** (T4: NW005 Bottom Burrows 1.26ha + NW006 Burrows 6.49ha; T9: NW013 Dairy South 6.45ha +
NW039 Dairy Corner 1.30ha), and `FERTN_TEMPLATE` pools both fields' events into one catchment-level
frequency/rate with no area-weighting -- overstating the true catchment-average annual N loading by
roughly 2x at T4/T9 (T2 is a single field, NW002, unaffected). Correctly area-weighted (total N mass
/ true catchment area): T2=155.8, T4=145.5, T9=27.5 kg N/ha/yr (typical year) -- and **no real field,
in any real year 2017-2024, ever exceeds ~245 kg N/ha/yr**, comfortably under the 300 cap. NWFP does
not, and never has, breached NVZ regulation; the "exceeds 300" claim was wrong.

**Important scope note: this double-counting was never inside `synthetic_fertN_events()` or any
scenario/pipeline code** -- `FERT_LEVELS`'s existing `historical`/`plus50pct_rate`/`plus50pct_freq`
levels generate individual per-event synthetic entries (matching real per-event magnitudes, which
themselves aren't inflated), not an aggregated annual total; the pooled-catchment convention itself
mirrors `build_management_features.py`'s project-wide `CATCHMENT_FIELDS` design (an EC tower's
footprint integrates signal across its whole catchment, not one sub-field) and is not itself a bug.
The flawed "n_events x mean_rate = annual total" step existed only in the chat's ad hoc regulatory-
comparison arithmetic. **No rerun was needed to fix that bug** -- only the write-up claim.

**New level added anyway, per direct user request**: `reg_cap` -- rate scaled (frequency held at
historical, matching `plus50pct_rate`'s axis-isolation convention) so the TRUE area-weighted typical-
year N loading hits exactly 300 kg N/ha/yr per tower. Needed its own module
(`build_transient_scenario_drivers_fertN_regcap.py`) rather than a plain new `FERT_LEVELS` entry,
because every existing level is one multiplier applied uniformly across all 3 towers, whereas 300 is
an absolute target each tower sits a very different distance from (same reasoning as D-97/D-104's
`lit_ceil`/`own_max`). Solved directly (linear in rate_mult, no bisection needed): **T2 x1.926, T4
x2.062, T9 x10.897** -- T9's real fertiliser use is so low it needs ~11x to reach the same absolute
ceiling that T2/T4 reach at ~2x, a genuinely interesting contrast on its own. Monkey-patch pattern
(same technique as D-104): `synthetic_fertN_events_regcap()` delegates to the ORIGINAL function
(captured at import time, before patching, to avoid self-recursion) for every level except `reg_cap`.

**Result: negligible effect, consistent with D-97/D-98's standing "redundant on the rich base"
finding** -- even scaled all the way to the regulatory ceiling: T2 -0.9%/-0.7% (S-05/S-06), T4
+1.5%/+1.7%, T9 +0.1%/+0.1%. Notably, T9's effect stays negligible even though its per-event rate is
scaled to ~284 kg N/ha per single application (a large, likely out-of-AOA magnitude) -- reinforces
that fertiliser schedule is not a meaningful CH4 lever in this model, now anchored at a real
regulatory ceiling rather than an arbitrary +50%, not just a narrower conclusion.

**Full coverage**: added to BOTH S-05 (real drivers) and S-06 (bias-corrected drivers), all 3
towers, both SSPs, full 5-GCM x 10-realization sweep (300 calls each, ~1075s) plus daily-chains
subset (12 calls). Additive only -- existing `historical`/`plus50pct_rate`/`plus50pct_freq` rows
untouched, row counts confirmed (34,000 annual S-06 total, 248,200 daily-chains total each). Fixed
the same mixed-timestamp-format bug as D-104 in both `s0{5,6}_practices_daily_chains_plots.py`
(`pd.to_datetime(..., format="mixed")`) after merging old and newly-rerun rows.

**Files:** `src/features/build_transient_scenario_drivers_fertN_regcap.py` (new),
`notebooks/07_scenario_analysis/s05_s06_fertN_regcap_fix.py` (new); `results/
s05_practices_fertilizer.csv`, `s05_practices_s06_fertilizer.csv`,
`s05_practices_fertilizer_daily_chains_subset.csv`, `s06_practices_fertilizer_daily_chains_subset.csv`
(all updated in place, additive); `s05_practices_daily_chains_plots.py`,
`s06_practices_daily_chains_plots.py` (edited: `reg_cap` added to the fertilizer AXES entry, mixed-
format timestamp fix); all 12 `s0{5,6}_practices_fertilizer_daily_*` figures regenerated.

### D-106 -- 2026-08-19 -- B17/B18: direct pooled, recency-aware, event-corrected TabPFN
supersedes the B16-style long-horizon numerical benchmark; final equal-weight gain is exploratory

**Scope and comparison rule:** user requested a precise B16→B18 improvement account for TabPFN
only. All numbers below use the same primary observed-target, day-of-year-climatology-scaled MASE
convention (D-80/CLAUDE.md), all 3 towers and all 5 16-December anchors. There are 2,127 observed
evaluation points across 9 tower-anchor blocks; empty target blocks remain in the raw 15-chain
coverage but do not invent evaluation values. Strict candidates see `y_observed` only on or before
the anchor and known-future `fx_*` drivers; post-anchor observed/gap-filled methane and methane AR
features are not supplied. R² values here are global across the evaluated points, not averages of
per-bin R². TabICL is deliberately outside this decision's scope.

**Naming clarification:** B16's `BASE+ALL` means all available covariate families, not “all
species.” The genuine species split is `BASE_species_37` (`fx_cattle_dens`, `fx_sheep_dens`,
`fx_lamb_dens`). On the reconstructed same-protocol comparison, the strongest B16-style checkpoint
is TabPFN-TS v2 `BASE+ALL` (MASE=0.7123); genuine-species TabPFN-TS v3 is 0.7154. The latter was the
previously documented `TabPFN+species` champion within the original B16 configuration framing.

**Main architectural improvement (B16→B17):** replace the one-shot `TabPFN-TS` wrapper with the
generic `TabPFNRegressor` fitted directly to pooled observed daily rows at each anchor. The direct
model uses all 52 `fx_*` predictors plus three tower indicators, calendar year, and days since
2010-01-01, returns the predictive median, and uses the best screened seed (137). This lowers MASE
from 0.7123 to 0.6958 (−0.0165, −2.31%), MAE 30.669→30.073, RMSE 61.704→60.708, and raises global
R² 0.146→0.173. This wrapper→direct-regression change supplies roughly three quarters of the total
B16→B18 MASE reduction; it is the substantive improvement, not a feature-name relabelling.

**B18 refinements:** (1) moderate forgetting of stale history -- the latest 1,095 days gives
MASE=0.6930, while a 1,460-day tower-robust variant gives 0.6934; 730 days discards too much and
1,825 days trends back toward all-history. (2) A leakage-safe, tower-specific pre-anchor p95 event
classifier plus base/normal/spike TabPFN regressors. Hard gates and full probability mixtures are
negative; the useful form is deliberately conservative, adding only 25% of the predicted
spike-minus-base excess. This is the **best single/gated B18 forecast**: MASE=0.6924, MAE=29.857,
RMSE=59.530, R²=0.205, bias=−9.402. (3) A fixed equal mean of that p95 correction, the 1,095-day
raw forecast, and the 1,460-day tower-robust forecast is the **exploratory numerical champion**:
MASE=0.6908, MAE=29.821, RMSE=60.019, R²=0.192, bias=−10.452.

**Total measured change:** B16-style v2 `BASE+ALL` 0.7123→B18 equal mean 0.6908: −0.0215 MASE
(−3.01%), MAE −0.848 (−2.76%), RMSE −1.684 (−2.73%), and R² +0.046. B17 direct all-history→B18
contributes only −0.0050 MASE (−0.72%). Component gains overlap and are not additive: the recency,
event, and ensemble rows are alternative forecasts evaluated on the same observations.

**What was tested and rejected rather than silently omitted:** 42 antecedent/interaction features
(all-feature TabPFN MASE=0.7040), seasonal experts (~0.738-0.742), recency replication (~0.697),
tower-month target normalization (completed safe fallback 0.7673), hard spike gating, and
tower-adaptive model switching. A same-data tower oracle looks attractive (0.6892), but
leave-one-block-out/forward tower selection degrades to ~0.698-0.699 and is not adopted. The added
antecedent variables improve event-ranking AUROC in one p90 comparison but not flux regression.

**Spike limitation remains decisive:** the p95 classifier has AUROC=0.876, average precision=0.302,
and Brier=0.0537, but calibrated hard-gate recall is only 4.5%. Using pre-anchor tower-specific p95
labels, the conservative forecast scores MASE=3.285/MAE=176.5 on 130 spikes versus MASE=0.524/
MAE=20.3 on 1,997 non-spikes; nearly all spike error is underprediction. B18 improves ordinary and
some large-error behaviour but does not solve spike magnitude or approach the requested MASE<0.25.

**Uncertainty/claim strength:** block bootstrap (10,000 resamples of the 9 evaluable blocks) for
the p95 single/gated forecast versus B17 gives ΔMASE=−0.0034, 95% interval [−0.0129, 0.0043],
P(B18 better)=0.784 -- directionally favourable but not independently conclusive. Comparisons with
B16-style v2 and genuine-species v3 are clearer (intervals wholly below zero; P=0.998/0.985). The
fixed three-forecast mean is retained as the lowest exploratory score, but LOBO-estimated pair/
triple weights score 0.6925-0.6936; do not claim the final ~0.0016 ensemble increment as a validated
breakthrough. Use the p95 result when one interpretable single/gated model is required.

**Decision and integration boundary:** update the long-horizon forecasting benchmark to the B18
equal three-TabPFN mean (explicitly labelled exploratory) and record the p95-corrected forecast as
the best single/gated result. This does **not** silently replace the model inside I-03, U-04, S-03,
or Phase-07 scenario pipelines; those outputs remain analyses of the prior B16-style architecture
until separately recalibrated. No `.tex` file was changed.

**Files:** `notebooks/05_benchmarking/B18_direct_structure.py`, `B18_spike_models.py`,
`B18_monthly_normalization_fallback.py`, `B18_evaluate_and_plot.py`, `B18_blend_validation.py`,
`B18_final_triple_chain.py`; `results/b18_direct_structure_chains.csv`,
`b18_spike_model_chains.csv`, `b18_final_triple_blend_chains.csv`,
`b18_candidate_registry.csv`, `b18_champion_block_bootstrap.csv`,
`b18_champion_spike_metrics.csv`; `results/figures/b18_chains_final/` (15 B15-style figures);
`report/Outlines/B17_forecasting_experiment_results.md`,
`report/Outlines/B18_forecasting_experiment_results.md`.

### D-107 -- 2026-08-20 -- Latest TabICL-solo gap-filling figures use native hourly,
per-tower six-month windows and raw native UQ

The report-facing TabICL gap-filling chain must represent the actual D5.5 benchmark architecture,
not reuse or relabel the earlier pooled TabICL UQ artifact. Section 19.2 of
`notebooks/03c_gap_filling_revisited/temp_gap_filling_pipeline.ipynb` therefore refits
`TabICLRegressor(random_state=42)` separately for Towers 2/4/9 on the champion `FEATURES`, using
mean imputation and the same fixed random sample capped at 10,000 real observations. It exports
each tower's latest six calendar months at native hourly resolution. Solid black is observed
FCH4; blue dotted model means and pale-blue bands are drawn only at genuinely missing target
hours. Tower 2 consequently shows Dec 2018-Jun 2019, while Towers 4/9 show Jun-Dec 2023.

The band is the model's own q05-q95 output and is explicitly **raw/uncalibrated**: no empirical
coverage claim is made and the older calibrated/pool-based endpoints are not borrowed. Exact
hourly point and q05/q50/q95 outputs are persisted in
`_data/latest_tabicl_uq_6month_chains.csv`; provenance and row/coverage counts are in
`_figures/latest_tabicl_uq_hourly_manifest.json`. The three canonical figures live in `_figures/`
with identical report copies at `report/Figures/ch4_latest_tabicl_uq_hourly_T{2,4,9}.png`.
This is figure/raw-output tooling only: no benchmark score changed, TabICL is not silently
production-adopted, and no `.tex` file was changed.

### D-108 -- 2026-08-20 -- B18 integration into Phase 07 (I-03b/U-08/S-03b-d/U-05b-07b/S-06b):
the direct-regression architecture genuinely carries over, once correctly adapted

D-106 (B18) explicitly did NOT propagate into I-03/U-04/S-03/Phase-07 scenario work -- those stayed
on the prior B16-style architecture pending separate recalibration. This closes that gap via a
6-phase additive plan, user-directed ("plan this thoroughly... ensure it is all additive"),
executed in full.

**Phase 1 (I-03b) / Phase 2 (U-08):** interpretability/UQ recalibrated for B18's actual champion
(direct TabPFN regression + p95 spike-gate, `BASE_ALL_52`, full point-forecast feature space --
no restriction needed here). Both reconfirm the standing thesis under the new architecture:
`fx_lsu_dens` still #1 (1.746, up from I-03's 1.1456), Tower 2 livestock-blind a 5th time; UQ
converges to ~0.89-0.90 PICP a 5th/6th time, T2 still uncalibratable. No surprises -- see
`I03b_results.md`, `U08_results.md`.

**Phase 3 (S-03b/c/d) -- the gate, and where the real work was.** B18's `BASE_ALL_52` cannot run in
scenario mode (flow/antecedent-driver columns have no constructible value for a synthetic 2050
future) -- S-05/S-06 use `FX_A_SPECIES` (13 cols, S-03/D-70's scenario-safe set) instead. Three
rounds of real-anchor backtesting (5 anchors x 3 towers, climatology MASE) were needed to find the
config that's BOTH validated AND actually deployable in S-05/S-06's real pipeline:
- S-03b: pooled-across-towers `Direct_TabICLv2_raw` (tower-dummy statics + trend) beat the current
  production `tabicl_forecast()` TS-wrapper by **4.4% MASE** (0.726 vs 0.760) -- but pooling
  assumes one shared training cutoff across towers, which doesn't exist in S-05/S-06 (each tower
  has its own `tower_anchor()`, the last real-data date for that tower specifically).
- S-03c: the faithful adaptation -- solo per-tower fits, no trend feature -- barely beat control
  (0.755 vs 0.760, +0.6%, noise-level). Most of S-03b's margin came from pooling and/or trend, not
  the architecture change alone.
- S-03d: isolated the cause -- solo per-tower **+ trend feature** recovers most of the margin
  (0.738 vs 0.760, **+2.79%**). Trend (`b17_days_since_2010`) was the real driver. Its training
  range (2,557-5,480 days, ~2017-2025) is 2.7x short of the 2050 horizon (~14,600) -- a genuine
  extrapolation risk, checked DIRECTLY (not assumed): a sanity fit queried 27-47 years past its
  training max **saturates to a bounded, plausible value (~8.3 nmol)** rather than exploding, unlike
  SARIMAX's known explosive extrapolation (D-63/U-03). Confirmed safe to use.
- **Locked-in config: solo per-tower `Direct_TabICLv2` regression on `FX_A_SPECIES + b17_days_
  since_2010`, no pooling, no spike-gate** (single point-config, user-directed, to bound Phase 6's
  compute). See `S03b_results.md`.

**Phase 4 (U-05b) / Phase 5 (U-06b/U-07b):** scenario UQ rebuilt for the FINAL locked config (the
file was initially built against S-03b's pooled config, then corrected to match S-03d before Phase
6 ran). AOA-residual correlation replicates almost exactly (r=0.128-0.139 vs U-05's 0.146;
out-of-AOA residuals 48-54% larger vs U-05's ~48%). CQR spike-fix triples coverage again (U08:
0.266->0.775; U05b: 0.266->**0.901**, close to the full-feature champion). LSU-stratified
margins replicate the width-ratio pattern (27.9%/40.7% low-as-%-of-high, within U-07's 26-59%
range). See `U08_results.md` (already updated to the corrected architecture), `U06b_U07b_results.md`.

**Phase 6 -- production replication of S-06's core grid (livestock ladder + grazing + fertilizer,
bias-corrected drivers), new architecture.** `s06b_direct_regression_engine.run_axis_b18()`: same
signature/behaviour as `s05_practices_trajectory.run_axis()`, but the model is fit ONCE per tower
(not once per call, unlike the TS-wrapper which has no separate fit/predict) and reused across
every (ssp, gcm, realization, level) combo -- 3 fits total for the whole livestock grid, not 2,100.
Actual runtime **~2.2h for the full 3-axis grid** (livestock 2,100 calls/80min, grazing 900/45min,
fertilizer 900/36min), vs. S-05/S-06's original ~6-9h.

**Bug caught by direct user question ("do the scenarios follow gov.uk regulations?"):** the first
production run reused `s05_livestock_v2_trajectory.build_livestock_frame` unchanged, which imports
`multiplier_for` directly from the base module (D-97's original, uncorrected 3.0 LSU/ha) -- it
never had D-104's correction (2.5 LSU/ha, UK Countryside Stewardship Annex 8) applied, unlike the
original `s06_lit_ceil_fix.py`'s targeted patch for the OLD architecture. Fixed via the identical
scoped-rerun-and-merge pattern (`s06b_lit_ceil_fix.py`): reran only the 2 `lit_ceil` combos with
`multiplier_for_s06`, merged into `s06b_practices_s06b_livestock_v2.csv` in place (59,500 -> 59,500
rows, clean). `reg_cap` (D-105, NVZ N-max) intentionally excluded from this pass, matching D-105's
own single-realization-addendum precedent -- not run for S-06b yet, flagged as a follow-up.

**Result: every S-06 headline finding replicates cleanly under the new architecture** (T9/T4/T2
pooled annual-mean `nmol m-2 s-1` by level, ssp245):

| Tower | baseline | lit_ceil (all/cattle) | own_max (all/cattle) |
|---|---:|---:|---:|
| T2 | 5.48 | 5.73 / 5.37 | 10.89 / 8.64 |
| T4 | 14.31 | 14.44 / 14.43 | 34.02 / 33.27 |
| T9 | 18.22 | 17.74 / 17.75 | 51.54 / 52.40 |

- T9 `lit_ceil` still sits BELOW baseline (-2.6%, matches old -2.3%) -- T9's real historical
  stocking density still exceeds the UK regulatory ceiling, independent of architecture.
- T4 `lit_ceil` margin still thin-positive (+0.8-0.9%, matches old +0.8-0.9% almost exactly).
- Grazing +4wk: T4 +19.8% (S-06 old, bias-corrected: +18.5%), T9 +19.9% (S-06 old: +18.6%) --
  same real, management-actionable lever, same magnitude. (Correction, post-hoc audit: the
  figures first quoted here, +18.9%/+17.2%, were S-05's ORIGINAL raw-driver numbers, not S-06's
  bias-corrected ones -- comparing S-06b against S-05 conflated the architecture change with the
  driver-correction change. The correct apples-to-apples S-06-vs-S-06b comparison, re-derived
  directly from `results/s05_practices_s06_grazing.csv` vs `s06b_practices_s06b_grazing.csv`, is
  actually a slightly closer match: +18.5%->+19.8% (T4), +18.6%->+19.9% (T9).)
- Fertilizer: still a null result (<8%, sign-inconsistent across towers, same as old <5%).
- Cattle dominance: `own_max__cattle_alone` remains ~as high as `own_max__all_species` at T4/T9
  (T9 cattle-alone actually edges ahead, 187.6% vs 182.9% gain) -- reconfirmed a new way.

**Not yet done (explicit follow-ups, not gaps in this pass's own scope):** `reg_cap` fertiliser
level for S-06b; U-05b's Step 4 (attach calibration to the NEW S-06b outputs specifically, mirroring
D-92 -- the file currently only has Steps 1-3); report text in `report/TODO.tex`'s Ch4-6 still
describes the OLD TabICLv2/`tabicl_forecast()` architecture and has not been updated to reference
S-06b.

**Files:** `notebooks/06_interpretability_uq/i03b_champion_interpretability_b18.py`,
`i03b_plots.py`, `u08_champion_uq_b18.py`, `u08_fanchart_plots.py`,
`u05b_scenario_uq_b18.py`, `u05b_fanchart_plots.py`, `u06b_u07b_cqr_b18.py`, `u06b_u07b_plots.py`;
`notebooks/07_scenario_analysis/s03b_driver_availability_b18.py`, `s03c_solo_notrend_check.py`,
`s03d_solo_trend_check.py`, `s06b_direct_regression_engine.py`, `s06b_master_runner.py`,
`s06b_lit_ceil_fix.py`, `s06b_livestock_v2_daily_chains_subset.py`,
`s06b_practices_daily_chains_subset.py`, `s06b_livestock_v2_daily_chains_plots.py`,
`s06b_practices_daily_chains_plots.py`, `s06b_annual_ch4_generation.py`;
`results/s06b_practices_s06b_{livestock_v2,grazing,fertilizer}.csv`,
`results/s06b_annual_ch4_*.csv`; 66 new figures across
`results/figures/{i03b_*,u08_fancharts,u05b_fancharts,u06_cqr,u07_lsu_cqr,s06b_summary,
s06b_annual_ch4}/`.

### D-109 -- 2026-08-29 -- Additive canonical workflow layer and repository README

The repository had accumulated a complete but difficult-to-navigate experiment history: promoted
methods, negative controls, plotting utilities, and superseded notebooks lived together in the
numbered stage directories, while no root `README.md` identified the current runnable path. A
second issue was that `/report/` was broadly ignored, so the newly split chapter and appendix
sources would not be included in a normal Git update.

**Decision:** preserve every historical experiment in place and add `workflows/latest/` as the
canonical navigation layer. Do not clone or rename the large notebooks into scattered `latest_*`
copies, because those copies would immediately create two potential sources of truth. The new
layer contains short ordered runbooks, a machine-readable `manifest.json`, and a lightweight
validator. It distinguishes production references, benchmark-best methods, supporting tools, and
validation-only experiments. The root `README.md` points to this layer and to `BEST_RESULTS.md`,
`CONTEXT.md`, `DECISIONS.md`, the data dictionary, and the compiled report.

The manifest records 6 stages and 44 current entry points: data preparation; TabICLv2-solo gap
filling and calibrated UQ; the three-component direct TabPFN forecast; B18-specific
interpretability/CQR; S-06b Direct TabICLv2 scenario projection; and the dissertation build. The
validator confirms that all referenced source files and runbooks exist without executing expensive
models. All 39 referenced Python entry points were separately parsed successfully.

**Git boundary:** `.gitignore` receives narrow additive exceptions for current report chapter
sources, split appendices, report outlines, and report-facing figures. Large local data,
`results/figures`, historical report backups, caches, and generated CSV archives remain excluded.
No modelling output or champion score changes under this decision.

**Files:** `README.md`; `workflows/latest/{README.md,manifest.json,validate.py,01_data_preparation.md,
02_gap_filling.md,03_forecasting.md,04_interpretability_uq.md,05_scenario_projection.md,
06_report.md}`; `.gitignore`.
