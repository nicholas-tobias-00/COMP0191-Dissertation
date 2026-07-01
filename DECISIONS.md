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
