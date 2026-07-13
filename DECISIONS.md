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
