# CONTEXT.md
_Read at the start of every session. Update "Current status" and "Next task" at the end of each session._

---

## Project

**Title:** AI for Agriculture: Towards Digital Twins for Methane Emissions Forecasting and Scenario Analysis  
**Module:** COMP0191 — MSc AI for Sustainable Development, UCL  
**Student:** Nicholas Tobias (ucabnt1@ucl.ac.uk)  
**Supervisor:** Prof. Paul Harris, Rothamsted Research  
**Data source:** North Wyke Farm Platform (NWFP), Rothamsted Research, Devon UK  
**EC data period:** 2018–present (7+ years of half-hourly measurements)

---

## Research aim

> Develop and evaluate multiple ML approaches for CH₄ flux forecasting at NWFP and demonstrate integration with a digital shadow architecture with scenario analysis and uncertainty quantification.

**Critical gap this project fills:** No prior study has applied ML-based temporal *forecasting* to EC CH₄ flux from a managed temperate grassland. Existing ML work on EC CH₄ is confined to gap-filling (within-distribution interpolation), a fundamentally different task. NWFP has 7+ years of untouched half-hourly EC data.

## Hypotheses

1. EC-based CH₄ prediction will be more grounded and accurate than IPCC Tier 1 static emission factors.
2. The ML-based digital shadow will produce multi-step forecasts with statistically distinguishable, interpretable predictions under contrasting management (known) and climate (unknown) scenarios.

## Research questions

| ID | Question |
|---|---|
| RQ1 | What ML approaches exist for CH₄ flux prediction in agricultural/grassland contexts; do any address ecosystem-scale EC data; how does gap-filling differ from multi-step forecasting as a modelling constraint? |
| RQ2 | How do statistical baselines, tree-based ensembles, and deep learning compare for half-hourly EC CH₄ forecasting under temporal variability and non-stationarity? |
| RQ3 | How transferable are findings from adjacent domains (wetland gap-filling, animal-scale prediction) to managed grassland EC forecasting? |
| RQ4 | How can XAI (SHAP) and UQ (quantile ML / conformal prediction) decode interactions between continuous environmental variables and discrete management interventions? |
| RQ5 | What structural requirements are needed to integrate forecasting models into a digital shadow with "what-if" scenario analysis at farm scale? |

---

## Objectives

1. Systematic literature review on time-series forecasting, quantile ML, EC sensor methodology, agricultural DT frameworks.
2. Acquire, preprocess, and document NWFP EC CH₄ data (2018–present): gap imputation, QC, feature engineering from meteorological and soil variables.
3. Benchmarking pipeline: LSTM, TFT, RF, XGBoost vs persistence and seasonal mean baselines; temporal cross-validation to prevent leakage.
4. XAI (permutation importance, SHAP) + UQ (quantile ML or conformal prediction) for calibrated prediction intervals.
5. Synthetic management and climate scenario generation; in-sample and out-of-sample evaluation.
6. ~~Digital shadow interface (Streamlit) with scenario analysis and uncertainty visualisation.~~
   **Dropped from scope (D-87, 2026-08-08, user-directed)**, given the 1 Sept deadline — the
   underlying digital-shadow substance (scenario simulation, management/climate levers, UQ once
   attached) is exactly what S-01/S-03/S-04/S-05 already deliver; only the interactive interface
   packaging layer is dropped, not the analysis itself.

---

## Methodology overview

**Three targets — one model per tower/ecosystem:**

| Tower | Column | Valid data | Ecosystem / field |
|---|---|---|---|
| Tower 2 | `FCH4_1_1_1 [Tower 2]` | 12.1% | Field/ecosystem 2 |
| Tower 4 | `FCH4_1_1_1 [Tower 4]` | 44.6% | Field/ecosystem 4 |
| Tower 9 | `FCH4_1_1_1 [Tower 9]` | 25.6% | Field/ecosystem 9 |

Each tower measures a distinct spatial unit. The deliverable is **three separate forecasting models**, trained and evaluated independently. Tower 2's sparse coverage (1,675-day gap May 2019–Jan 2024) is a real data constraint for that model, not a reason to deprioritise it.

Quality flag pattern: `FCH4_SSITC_TEST_1_1_1 [Tower N]` (0=best, 1=ok, 2=reject). Note: `CH4_1_1_1` is mole fraction (nmol/mol) not flux — do not confuse.  
**Best starting point for modelling:** `data/Hourly/consolidated_hourly.csv` — all sources on a common 1h DatetimeIndex.  
**Features (shared across towers):** Soil moisture + flow from `measurements.csv`; management events from `Field_Event_Data_Format_1.csv`; livestock location counts from `Animal_location_counts_*.csv`. Feature–tower spatial alignment TBD during feature engineering.

**Temporal split (applied independently per tower):**
- Train: 2018–2021 | Test: 2022–2023 | Held-out: 2024
- Tower 2's 1,675-day gap (May 2019–Jan 2024) means its effective training window differs — evaluate what remains within each split window before running Tower 2 models.

**Evaluation metrics:**
- Regression: RMSE, MAE
- Uncertainty: Coverage Probability, Interval Score
- Interpretability: SHAP plausibility vs domain knowledge

**Model ladder:**
1. Persistence / seasonal mean (baseline)
2. ARIMA
3. Random Forest, XGBoost / Gradient Boosting
4. LSTM
5. Temporal Fusion Transformer (TFT)

**Key methodological commitments** (see DECISIONS.md):
- Temporal cross-validation only — no random splits
- UQ via quantile ML or conformal prediction is non-negotiable (Irvin et al. 2021 showed raw ML uncertainty is systematically underestimated)
- SHAP for driver interpretation
- ERA5 reanalysis as fallback when local sensors fail (Zhu et al. 2023)

---

## Data summary

All data lives in `data/` (gitignored). Use `data/Hourly/` as the starting point for modelling.

| Layer | Path | How to regenerate |
|---|---|---|
| Raw annual slices | `data/Consolidated/` | Download from NWFP portal |
| Multi-year compiled | `data/Compiled/` | Run `notebooks/01_data_compilation/` |
| **1-hour consolidated** | **`data/Hourly/`** | **`python src/data/consolidate_hourly.py`** |

**Hourly outputs (primary modelling inputs):**

| File | Rows | Cols | NaN% | Notes |
|---|---|---|---|---|
| `greenhouse_hourly.csv` | 61,345 | 147 | 30.6% | FCH4 + CO₂ + H + LE + met, all towers |
| `measurements_hourly.csv` | 70,153 | 239 | 49.9% | Flow + soil moisture per catchment |
| `livestock_hourly.csv` | 70,129 | 63 | 0.0% | Head counts per location, all species |
| `consolidated_hourly.csv` | 70,153 | 449 | 39.4% | All sources outer-joined |

**Data notes:**
- In `data/Compiled/`: quality-flag string columns (`"Acceptable"`/`"Not set"`) and `"Quality Last Modified"` timestamp columns are present — filter before use.
- In `data/Hourly/`: non-numeric columns are already dropped by `consolidate_hourly.py`.
- EC data has persistent sensor gaps (especially Tower 2) — ERA5 fallback validated for UK managed pastures (Zhu et al. 2023a, D-08).

---

## Repository layout

```
notebooks/
  01_data_compilation/    COMPLETE — compiles Consolidated -> Compiled (23 files)
  02_eda/                 COMPLETE — full EDA + Section 6 modelling readiness; figures in results/figures/
  03_gap_filling/         COMPLETE — R-01 through R-03 replications + gap_filling_summary.md (R-04 dropped)
  03b_gap_filling_CO2/    COMPLETE — R-01/02/03-CO2: gap-filled FCO2 as a CH4 feature (D-26)
  04_feature_engineering/ PLANNED
  05_benchmarking/        PLANNED
  06_interpretability_uq/ IN PROGRESS -- I-01/I-02, U-01/U-02/U-03
  07_scenario_analysis/   IN PROGRESS -- S-01/S-02/S-03
  08_imputation_revisited/ IN PROGRESS -- IMP-01 done (step 1 of 5: missingness viz)
  03c_gap_filling_revisited/ D-77/D-78 -- fully self-contained (no src/) gap-filling reproduction
    + a real mdc_gapfill fix (new champion R2 0.576/0.404/0.426) + extended exploration (UQ,
    6 additional models, lag/lead expansion) -- see temp_gap_filing_exploration[ copy].ipynb
    D-79 -- parallel notebook temp_gap_filling_pipeline.ipynb: literature-correct MDS fix,
    HyperImpute baseline, all-16-model production fill, TICA/UMAP feedback-feature line (D5-D8) --
    TabICL-solo now beats RFm champion at T2/T4 (benchmark, not yet production-adopted)
src/
  data/
    consolidate_hourly.py COMPLETE — resamples all data to 1h; writes data/Hourly/
  features/               aggregation, lag construction, quality filtering
  models/                 model wrappers
  evaluation/             metrics, plotting
results/                  benchmarks.csv (append-only) + figures/
prompts/                  session templates
DECISIONS.md
```

---

## Key prior work at NWFP

| Paper | Relevance |
|---|---|
| Partridge et al. (2024) | Gradient Boosting on NWFP **GreenFeed** cattle CH₄ (r=0.619) — most comparable prior work, but animal-scale not EC |
| Cardenas et al. (2022) | CO₂ EC flux at NWFP — same Tower 2 infrastructure |
| Oulaid et al. (2025) | Quantile ML for soil moisture at NWFP — direct UQ methodology reference, same site |
| Fakeye et al. (2024) | Farm-scale DT framework at NWFP — proposes CH₄ module as named gap |

---

## Current status

- **Phase:** Gap-filling replication phase COMPLETE (R-01 through R-03, Towers 4+9). R-04 dropped (GreenFeed is animal-scale, not EC flux — not a valid gap-filling comparison).
- **Completed:**
  - `01_data_compilation` — 23 compiled files in `data/Compiled/`
  - `02_eda` — full EDA + Section 6 modelling readiness check; figures in `results/figures/`
  - `src/data/consolidate_hourly.py` — `data/Hourly/consolidated_hourly.csv` (70,153 rows × 449 cols, 39.4% NaN)
  - `03_gap_filling/R01_Irvin2021_RF_XGBoost.ipynb` — R-01 complete, all three towers (see below)
  - `03_gap_filling/R01_results.md` — detailed per-tower results, interpretation, and next steps
  - `03_gap_filling/R02_Zhu2023a_RF_MDS.ipynb` — R-02 complete, Towers 4 and 9
  - `03_gap_filling/R02_results.md` — detailed per-tower results, MDS vs RF gap-length analysis
  - `03_gap_filling/R03_Kim2020_RF_ANN_SVM_MDS_PCA.ipynb` — R-03 complete, Towers 4 and 9
  - `03_gap_filling/R03_results.md` — detailed per-tower results, model comparison, Kim findings tested
  - `03_gap_filling/gap_filling_summary.md` — three-way synthesis (datasets/columns, R² evaluation, metrics, root-cause analysis)
  - `03_gap_filling/gap_filling_flowcharts_and_features.md` — per-replication process flowcharts + full feature dictionary (columns, descriptions, custom-vs-raw)
  - `src/data/fco2_gapfill.py` + `data/Hourly/fco2_gapfilled.csv` — RFm reconstruction of FCO2 from met drivers (Towers 2/4/9); recon test R²≈0.745/0.746 (T4/T9), 0.20 (T2)
  - `03b_gap_filling_CO2/` — R-01/02/03-CO2 notebooks + R0X_CO2_results.md + co2_augmented_summary.md (CO2-augmentation experiment, D-25/D-26)
  - `04_feature_engineering/` — `fch4_drivers_and_features_review.md` (driver review), `F01_feature_ablation_RFm.ipynb` + `F01_results.md` + `feature_engineering_summary.md`; `src/features/build_management_features.py` (D-27/D-28)
- **Key EDA findings:**
  - FCH4 flux range (Tower 4, QC-filtered): mean 33.5, range −1559 to +6161 nmol m⁻² s⁻¹
  - Tower 2: 12.1% valid (1,675-day gap May 2019–Jan 2024); Tower 4: 44.6%; Tower 9: 25.6%
  - Soil moisture: 15 catchments, best 83% availability; Flow: best 84%
  - Livestock: mean 82 cattle / 143 sheep / 139 lambs per day (2017–2025)
- **Modelling readiness findings (Section 6 — `consolidated_hourly.csv`):**
  - Index: 70,153 continuous hourly timestamps — no gaps, no duplicates ✓
  - 11 near-constant columns to drop; 438 usable columns
  - **SWIN present as `SWIN_1_1_1 [Tower N]`** (~52%): EDA pattern `SW_IN_` missed it; ERA5 not required for R-01 (D-14 revised)
  - **Tower 4 FCH4 extreme outliers**: applied [-500, 3000] plausibility filter in R-01 (D-13)
  - **LE correct column**: `LE_1_1_1 [Tower 4]` (74%) — used in R-01; `LE_SSITC_TEST_*` is the quality flag
  - **Tower 4 soil temp only 9.6%**: using `TS_1_1_1 [Tower 9]` (71%) as proxy (D-16)
  - **Tower 2 split**: 0% valid FCH4 in both test (2022–23) and held-out (2024) windows — standard split inapplicable, custom split needed (D-15)
  - **Held-out 2024 empty for all towers**: data currently ends Jan 2024; 2024 data download needed to use this window
  - New figures: `hourly_nan_distribution.png`, `hourly_gap_length_distribution.png`, `hourly_fch4_distributions.png`
- **Spatial alignment rule (confirmed):** Tower N = Catchment N. Tower 2 ↔ Catchment 2, Tower 4 ↔ Catchment 4 (column: `Catchment 4 After  2013/08/13`), Tower 9 ↔ Catchment 9. Each model must use only the soil moisture column for its own catchment — never average across catchments from other towers (D-18).
- **R-01 results (5 permutations, median):**
  - Tower 4 (test 2022–2023, n_train=7,714): RF R²=+0.144, RMSE=121.3, MAE=62.5; XGB R²=+0.086, RMSE=126.5, MAE=70.7
  - Tower 9 (test 2022–2023, n_train=3,981): RF R²=−0.027, RMSE=123.5, MAE=58.8; XGB R²=−0.089, RMSE=128.0, MAE=62.6
  - Tower 2 (D-15 custom, train 2018 / test Jan–May 2019, n_train=2,985): RF R²=−16.9, XGB R²=−55.9 — split design failure (seasonal mismatch; see D-19)
  - Full details in `notebooks/03_gap_filling/R01_results.md`
- **R-02 results (5 reps × 5 scenarios, median):**
  - **Tower 4** (n_train_driver3=10,862, n_train_driverm=7,285): RF3 R²≈−0.13, RFm R²≈−0.13, MDS R²≈−0.20, XGBm R²≈−0.35 (median across scenarios)
  - **Tower 9** (n_train_driver3=4,048, n_train_driverm=2,288): RFm R²≈−0.10 (best), RF3≈−0.16, MDS deteriorates to −0.58 for l (12-day) gaps, XGBm≈−0.14
  - All R² negative: consistent with Zhu's finding of R² < 0.10 at managed pastures; our values worse because LE/H/FC excluded (D-22) vs R-01 which included them
  - Paper's main finding confirmed: RF > MDS for long (288h) gaps at Tower 9 (RF3 −0.182 vs MDS −0.584)
  - MDS nearly unbiased (MBE ≈ 0); ML models show 10–40 nmol m⁻² s⁻¹ positive bias (Tower 4)
  - Full details in `notebooks/03_gap_filling/R02_results.md`
- **R-03 results (5 reps × 4 scenarios, median):**
  - **Tower 4**: RF best at short gaps (R²=+0.136); **ANN best at medium/long/xlong** (R²=+0.091/+0.077/+0.057); RF_lag slightly worse than RF; RF_PCA7 better than RF_lag at medium/long (reverses Kim's PCA finding); SVM underperforms (R²≈0, strong negative MBE)
  - **Tower 9**: RF_lag best at short/medium (R²=+0.152/+0.160); RF_PCA7 best at long/xlong (R²=+0.111/+0.056); ANN catastrophic at xlong (R²=−0.518 — small-sample artefact); MDS worst at all scenarios
  - Kim's RF≥ANN finding partially confirmed; lag feature finding confirmed at T9, not at T4; PCA-degrades-ML finding NOT confirmed (site-specific at NWFP)
  - 240 R-03 rows in benchmarks.csv (470 total)
  - Full details in `notebooks/03_gap_filling/R03_results.md`
- **Cross-replication synthesis (`gap_filling_summary.md`):** Headline R² driven by feature realism (D-22), not algorithm: R-01/R-03 include LE/H/FC → Tower 4 ~+0.14; R-02 excludes them → all negative. Realistic met-only ceiling is near-zero/negative. Algorithm choice is not the bottleneck; management-event features are the next lever.
- **CO₂-augmentation experiment (`03b_gap_filling_CO2/`, D-26):** FCO₂ reconstructs from met at R²≈0.75 (T4/T9). Adding gap-filled FCO₂ to met-only RFm (R-02-CO2) moves Tower 4 negative→**positive** (vs −0.128→+0.156; m −0.160→+0.111) with RF3/MDS controls unchanged — proving **FC is the key FCH₄ predictor** (confirms D-22). R-03-CO2 ANN reaches +0.12–0.17 at T4 (best overall). Caveat: observed-FC-at-gaps = upper bound, not operational. Full results in `co2_augmented_summary.md`.
- **Feature-engineering ablation (`04_feature_engineering/F01`, D-27/D-28):** P1 **livestock is the #1 FCH₄ driver** at Tower 4 — `_lsu` is the top SHAP feature (28.2, ~2× FCO₂); +P1 lifts Tower 4 short-gap R² **+0.156 → +0.256** (biggest single jump in the programme). Confirms Felber 2015 / the driver review. Beyond livestock, diminishing returns; P2 management (12-col cumulative) overfit. Full results in `F01_results.md`.
- **F-02 stocking density + pooling (`04/F02`, D-29):** Pruned management (2 tower-specific cols) **fixes the F-01 overfit** (Tower 9 −0.86 → +0.01…+0.04). **Stocking density (LSU/ha, Appendix D areas) pays off in a pooled T2+T4+T9 model: Tower 9 → R² ≈ +0.29** (best in project; vs pooled-count +0.18, solo ≈ 0). Density is inert single-tower (Cat 4 = Cat 9 = 7.75 ha) — only helps with different-area catchments in the pool (T2 = 6.65 ha). Full results in `F02_results.md`.
- **F-03 partial pooling (`04/F03`, D-30):** **Partial pooling (pooled + tower-indicator dummies) ≥ full pooling at every tower** — the recommended default. Keeps Tower 9 rescue (≈+0.29); **Tower 2 benefits most from the dummy** (partial −0.245 vs full −0.301 short-gap); Tower 4 protected. Tower 2 still negative (D-15 split). Full results in `F03_results.md`.
- **F-04 R-03 lags re-tested (`04/F04`, D-31):** Adding SWC/TS 1–4 wk lags **does NOT help Tower 9** (R-03's RF_lag advantage doesn't transfer — FCO₂+density+pooling already encode that memory). Helps **weakest-base towers**: Tower 2 partial Δ +0.116 (best T2 yet, still neg.), Tower 4 marginal at long gaps. Lesson: feature value is context-dependent. Full results in `F04_results.md`.
- **F-05 management re-tested (`04/F05`, D-32):** pruned tower-specific management gives a **small, non-harmful bump** (Δ +0.005…+0.013) — **redundant on the rich base**, same as lags.
- **F-06 REddyProc-style met gap-fill + GPP (`04/F06`, D-33):** prompted by NWFP/REddyProc EC report. We had always **mean-imputed** met drivers; `src/data/reddyproc_pipeline.py` gap-fills them (interp + mean-diurnal-course → 100%, diurnal preserved) + adds **GPP/Reco** (nighttime Lloyd-Taylor). **First addition since pooling that genuinely helps:** met-fill > mean-impute (Δ +0.017…+0.076, largest at coverage-poor T2); **GPP adds more → Tower 9 +0.335 (NEW PROJECT BEST)**, T4 +0.163, T2 −0.045 (best yet). Not redundant (fixes inputs + new productivity driver). **New best config = partial pool + density + lags + pruned mgmt + gap-filled met + GPP.** Full results in `F06_results.md`.
- **F-07 Tower 2 evaluation fix (`04/F07`, D-34, TOWER 2 ONLY):** Tower 2's −16.9/−0.045 was a **broken evaluation**, not data/model failure. Tower 2 CH4 = Oct2017–Jun2019; 2018 had cattle (FCH4≈42), 2019 none (FCH4≈2) → D-15 year split trains high-flux/tests near-zero → catastrophic. **Fix = full-period gap-CV** (gap-filling is interpolation). Result: **RFm pooled +0.519 (best in project, exceeds 0.5)**, solo +0.394; **MDS stays −0.49 (livestock-blind) → RFm beats MDS by ~1.0 R² unit** (clearest "improvement over MDS"). Caveat: Tower 2's high R² = discriminable livestock-on/off regime, not directly comparable to 4/9. Full results in `F07_results.md`.
- **F-08 EC-vs-external sensor sourcing (`04/F08`, D-35, COMPLETE):** built a parallel external-sourced data layer (`consolidated_hourly_SMS_MET.csv`, `reddyproc_processed_SMS_MET.csv` via `src/data/build_sms_met_dataset.py` — originals untouched) swapping all overlapping drivers to external (soil temp→per-catchment; air temp/solar/wind→Site, wind ÷3.6). Re-evaluated **all three towers under full-period gap-CV** (EC vs EXT, solo/pooled/MDS); harness validated (EC T2 solo = 0.395 = F-07). **Findings:** (1) external sourcing is **essentially neutral** for RFm — pooled gains a small, consistent **+0.012–0.014 at every tower** (never hurts, despite Site-level met + r≈0.17 wind); "redundant on the rich base" again. (2) The **per-catchment soil-temperature fix is vindicated** (net-positive pooled at all towers) → adopt it (removes the D-16/D-18 inconsistency). (3) **Biggest result = EC baseline under full-period gap-CV:** re-evaluating 4/9 with interpolation-style CV raises **T4 +0.163→+0.362**, T9 +0.335→+0.350 → **all three towers now consistent ≈0.35–0.49, each ≈0.6–1.0 over MDS**. External sourcing = consistency/robustness improvement, not a new accuracy lever. Full write-up: `F08_results.md`. benchmarks 2855 rows (90 F-08).
- **Supervisor steer (18 June mtg, Harris + Varma):** high R² not feasible for an open system → **goal = improvement over MDS, not absolute R²**. Deadline **1 Sept 2026**.
- **Next phase:** Forecasting (`05_benchmarking`) — partially-pooled global model carrying livestock+FCO₂+density+lags+mgmt+gap-filled met+GPP. (F-08 settles the EC-vs-external sensor-sourcing question first.)

## Replications

| ID | Paper | Target metrics | Status | Notebook |
|---|---|---|---|---|
| R-01 | Irvin et al. (2021) — FLUXNET-CH4 RF/XGBoost gap-filling | Paper: RF R²=0.79, XGB~0.65–0.67 (17 wetland sites). **T4: RF R²=0.144, XGB R²=0.086; T9: RF R²=−0.027, XGB R²=−0.089; T2: RF R²=−16.9 (split design failure)**. See R01_results.md. | complete | `03_gap_filling/R01_Irvin2021_RF_XGBoost.ipynb` |
| R-02 | Zhu et al. (2023a) — UK managed pastures gap-filling | RFR beats MDS for gaps >12 days; ERA5 validated. **T4: RFm R²≈−0.13; T9: RFm R²≈−0.10 (best); MDS −0.58 for l-gaps (T9). All methods negative R². Paper finding confirmed.** | complete | `03_gap_filling/R02_Zhu2023a_RF_MDS.ipynb` |
| R-03 | Kim et al. (2020) — RF vs ANN vs SVM vs MDS + PCA | RF best short; ANN best medium/long (T4); RF_lag best short/medium (T9); PCA degrades NOT confirmed. Full results in R03_results.md. | complete | `03_gap_filling/R03_Kim2020_RF_ANN_SVM_MDS_PCA.ipynb` |
| ~~R-04~~ | ~~Partridge et al. (2024) — NWFP GreenFeed Gradient Boosting~~ | Dropped — GreenFeed is animal-scale breath sampling, not EC flux; not a valid gap-filling comparison. | dropped | — |

Synthesis across R-01–R-03: `notebooks/03_gap_filling/gap_filling_summary.md`.

Each replication is run **per tower** (Tower 2, 4, 9 independently). Start with Tower 4 (best coverage), then extend to Towers 9 and 2 in that order. Log per-tower results in `results/benchmarks.csv`.  
_Update Status to `in-progress` / `complete` / `abandoned` as work proceeds._

---

## Next task

**Replications (R-01–R-03) + CO₂-aug (03b) + feature-eng (F-01…F-07) done; R-04 dropped. F-08 (EC-vs-external sensor sourcing, D-35) in progress.** Key arc (Tower 4 short gaps): met-only ≈ −0.13 → +FCO₂ +0.156 → +livestock **+0.256**. Livestock = #1 driver. **F-02: pooling T2+T4+T9 with stocking-density livestock → Tower 9 R² ≈ +0.29 (best in project).** Next steps, in priority order:

0. **F-08 done (`04/F08`, D-35):** EC-vs-external sourcing settled — external sourcing is accuracy-neutral for RFm (pooled +0.01 everywhere, never hurts); **adopt per-catchment external soil temperature** (removes D-16/D-18 inconsistency); under full-period gap-CV all three towers sit at ≈0.35–0.49 (T4 +0.163→+0.362). **Feature/sourcing engineering now fully exhausted.**
1. **Forecasting (`05_benchmarking`) — the project's novel contribution. SCOPED (D-36) + Stage 0–1 DONE (D-37).** Data=External SMS/MET; (A) hourly {1,6,12,24,48}h + (B) daily {1,3,7,14}d; driver-conditional; train-on-gap-filled/eval-on-observed; leak-free (FCO₂/GPP lagged-only). T4/T9 test 2022–23; **T2 flipped to a test target** via rolling-origin within 2017–2019.
   - **Precompute built:** `src/models/gapfill_rfm.py`, `build_fch4_gapfilled.py`→`fch4_gapfilled.csv`, `build_forecasting_matrix.py`→`forecast_features.csv`.
   - **FC-01 done (`B01`, `B01_results.md`):** RF/XGB + persistence/climatology. **RF beats persistence at almost every horizon** (hourly skill +0.08…+0.25; daily up to +0.37@14d). R² low but positive → **skill-vs-baseline is the metric**. Caveats: 1-day daily persistence unbeatable; RF's edge over climatology modest; T2 R² degenerate (use RMSE/skill).
   - **FC-02 done — DL benchmark (`B02`, `B02_results.md`, D-38):** hand-rolled pure-PyTorch seq2seq (DLinear/LSTM/LSTM-VSN) on the **RTX 5070 GPU** (torch upgraded 2.6→2.11+cu128). **Model complexity does NOT pay off — Zeng-2023 confirmed:** hourly **RF/XGB win** (DL negative R²); daily **DLinear ≈/beats RF**; LSTM only wins at **Tower 2 hourly** (strong AR regime). Production = RF (hourly) + DLinear (daily).
   - **I-01 done — feature importance (`06_interpretability_uq/I01`, `I01_results.md`, D-39):** permutation (grouped, per-horizon) + SHAP + VSN. **Importance shifts with horizon** (CH₄-history short → planned livestock/mgmt + met long); **livestock density = #1 SHAP feature** (carries the project thesis into forecasting); RF blends memory+drivers while LSTM drops memory (explains the trees' edge). benchmarks 3044 rows (81 FC-02).
   - **FC-03 done — UQ (`06/U01`, `U01_results.md`, D-40):** 90% intervals via conformal + quantile-XGB + LSTM-pinball. **Calibrated but wide** — conformal most reliable (PICP≈0.88) but widest; **quantile-XGB best trade-off** (sharpest, best pinball); LSTM-pinball under-covers (drop). Intervals ~150–260 nmol and **even they miss the biggest spikes** → uncertainty lives in the spike tail. benchmarks 3098 (54 FC-03).
   - **B-03 / B-04 done — enriched-feature reruns (`B03`/`B04`, `b03_b04_results.md`, D-41):** productionised the `NWFP_T9_Dataset_Structure.md` features across all towers (`src/features/build_forecasting_matrix_v2.py` → `forecast_features_v2.csv` + `forecast_daily_v2.csv`): wind-direction sin/cos, TA min/max, external-soil daily lags/rolling, days_since_grazing, expanded calendar, lagged-only FCO₂ (daily). **Enriched features lift the TREES, not the DL.** B-03 daily best R²: **T4 0.263→0.362, T9 0.304→0.388** (mean daily ΔR² RF +0.118 / XGB +0.166; features ~+0.08 then a Round-1 daily HPO ~+0.05). B-04 DLinear flat (lookback already saw that history). Best daily forecasting R² now **≈0.36–0.39** — real +0.08–0.10 gain, still short of the 0.5 stretch (physical ceiling). benchmarks 3287 (108 B03 + 81 B04). Production = **enriched trees on the daily track**.
   - **B-05 arcsinh target transform — NEGATIVE (`B05`, D-42):** tested asinh(y) spike-compression with Duan-smearing back-transform; **does not beat B-03** (daily best R² T4 0.337 / T9 0.347 vs B-03 0.362 / 0.388). R² is scored in original units where spikes dominate variance → squashing them is a net loss. Target transforms = dead end; the spike problem needs a **structural** attack (hurdle), not a transform. Kept in benchmarks (flagged). benchmarks 3395 (108 B05).
   - **B-06 done — spike-aware hurdle model (`B06`, `b06_results.md`, D-43):** two-stage occurrence × magnitude (q90 threshold, soft P(spike) blend, no new HPO). **NEGATIVE for the daily (production) track** (Hurdle-RF daily best R² T4 0.253 / T9 0.130 vs B-03 0.362 / 0.388; Hurdle-XGB even negative at T9). Mixed for hourly: **Tower 4 hourly Hurdle-RF/XGB beats plain RF/XGB at every horizon** (mean ΔR² +0.04/+0.02), T9 flat/negative. Root cause: low classifier precision (0.25–0.56) → false-positive blends inflate non-spike RMSE more than spike RMSE is reduced. **Both the transform (D-42) and architecture-split (D-43) attacks on the spike problem have been tried and documented negative for daily forecasting. B-03 remains the production config.** benchmarks 3557 rows (162 B06).
   - **B-07 done — spike diagnostics + recency features + early-warning analysis (`B07`, `b07_results.md`, D-44):** diagnosed B-06's classifier — false positives are **context-indistinguishable from true positives** on precip/days-since-grazing/growing-season (the classifier already learned "growing season + recent grazing → risk," but that covers ~2× as many quiet days as spike days). Added leak-free recency/clustering features (`ar_days_since_spike`, `ar_spike_count_<w>`, `ar_rolling_max_<w>`, verified causal) and retested the full B-06 harness regardless of the diagnostic — **marginal/inconsistent (<0.02 R² either direction), does not flip the B-06 verdict**, daily R² stays below B-03 at every tower/horizon. Genuine positive: at a recall≥0.8 operating point the classifier supports a standalone **early-warning** framing (~81% of elevated-emission days caught at precision 0.28–0.43) — retained for possible Phase 07 use. **B-05/B-06/B-07 (transform, architecture, diagnostics+features) now all exhausted and negative/marginal for daily R². B-03 remains production.** benchmarks 3665 rows (108 B07).
   - **Metrics backfill done — WAPE/MASE/sMAPE/MAPE (D-44b):** new shared `src/evaluation/metrics.py`
     (`full_metrics()`), imported by all forecasting notebooks (`forecasting_dl.py` for B02/B04). Backfilled
     across **all 7 forecasting-phase benchmarks** (FC-01, FC-02, B03–B07; re-executed, 3665 rows, zero lost).
     **MAPE confirmed unstable on this signed/near-zero flux data** (280–420% on FC-01 hourly rows) — **MASE is
     the recommended primary "watch out" indicator** (test-set relative-MAE vs persistence; >1 = worse than
     naive, hard fail); watch for MASE comfortably <1 alongside near-zero/negative R² — that divergence is the
     spike-tail signature (B-05/B-06 mechanism). FC-03/U-01 (UQ) excluded — different metric family
     (interval calibration, not point accuracy). Gap-filling phase (R-01–F-08) backfill **deferred** — staged
     for later per user's explicit staging choice, given timeline (deadline 1 Sept).
   - **Model-roster gaps filled — B-03a (SARIMAX) + B-03b (full TFT), D-45:** the original D-05 roster
     (persistence/seasonal-mean → ARIMA → RF/XGBoost → LSTM/TFT → SARIMAX) had two never-implemented rungs —
     ARIMA was scoped but never built; TFT was explicitly de-scoped at FC-02/D-38 in favour of `LSTM_VSN`. Both
     added on B-03's exact data/CV/horizons. **B-03a SARIMAX** (`B03a_arima.ipynb`, solo per-tower, SHAP-informed
     exogenous set, order (2,1,1), walk-forward via `append(refit=False)`): competitive only at h=1 (daily R²
     0.33) then **collapses negative by h=7–14** (MASE 1.06–1.13, worse than persistence). **B-03b TFT**
     (`B03b_tft.ipynb`, canonical architecture — VSN/GRN/static-encoders/interpretable-attention, new `TFT`
     class in `forecasting_dl.py`) **original run: negative R² at every horizon/tower/track** (daily -0.73 to
     -0.97, MASE 1.03–1.79) — the single worst model result in the whole forecasting phase. Result was
     independently verified (not taken at face value) — manual training-loop check confirmed clean loss
     convergence and sanely-scaled predictions, ruling out an implementation bug; mechanism = **overfitting**
     (weak positive test correlation r=0.27, dragged deeply negative by occasional large overconfident
     spike-mispredictions). **Fixed** (user follow-up, same session): added `weight_decay`/`val_data`/`patience`
     to `train_model()` (backward-compatible), retrained with weight_decay=1e-3 + early stopping on a held-out
     2021 validation year — hit and fixed a real bug along the way (unbatched validation forward pass caused a
     multi-GB attention tensor and a 30-min timeout; batching it like `predict()` already does brought it to
     16s/epoch). **Fix worked**: daily R² flipped from -0.73…-1.08 (negative everywhere) to **+0.10…+0.26**
     (positive everywhere), MASE from 1.03–2.01 to 0.65–1.23 (beats persistence from h=3–7 onward) — still well
     below B-03's trees (0.27–0.39) but no longer broken. **Model-roster question now fully closed** — every
     D-05 rung has a documented result; B-03 remains unambiguously production either way. benchmarks 3719+18
     rows (+27 B03a, +45 B03b incl. TFT-Reg). Full write-up: `b03a_b03b_results.md`.
   - **Feature/scope discussion (D-46):** fertiliser recency confirmed pruned (not re-added, D-28/D-32 already
     tested this); weekly AR mean already present (`ar_ch4_drm7`); explicit season/week-of-year calendar flags
     **not recommended** (redundant with `fx_DOY_sin/cos` for a tree model). **Other-catchment data (beyond
     2/4/9) rejected** — no FCH4 target exists elsewhere on the farm, so no pooling benefit; EC flux is
     footprint-local (D-18); farm-wide weather already captured via Site MET (D-35). **Long-range (~2030)
     scenario feasibility scoped** (not yet executed) — this is categorically Phase-07 scenario work, not B-03
     forecasting; requires a frozen model artifact, a scenario-capable feature pipeline, a livestock/management
     assumption, a climate-scenario driver source, an AR-history strategy for a horizon with no real recent
     data, an extrapolation-range check (RF/XGB don't extrapolate past training-leaf values), and explicit
     "conditional on scenario" framing given the training window ends 2021 (9+ year gap to 2030). **Candidate
     climate dataset found:** Semenov et al. (2025, Rothamsted, *Data in Brief*, DOI 10.1016/j.dib.2025.111695) —
     CMIP6-based daily Tmin/Tmax/rainfall/solar radiation, 26 GB sites, 2020–2090, 5 GCMs × 2 SSPs, 100
     realizations/scenario (LARS-WG downscaling), Zenodo/CC BY. Covers 4 of B-03's ~11 daily drivers (missing
     wind/VPD/USTAR/soil/SHF); North Wyke site-coverage unconfirmed; 2020–2021 overlap is a free validation
     check against real NWFP weather before trusting 2030 output.
   - ⚠⚠ **CRITICAL — F-09 data-quality fix (D-48): `benchmarks.csv` for the ENTIRE forecasting phase (FC-01
     through B03b) is now stale, pending re-run.** Investigated a user-observed spike in Tower 2's gap-filled
     FCH4 (also present at T4/T9) via SHAP attribution on the pooled RFm gap-filler — root cause: raw
     `USTAR_0_0_1` (and `VPD_0_0_1`) were never quality/plausibility-filtered anywhere in the pipeline (unlike
     FCH4/FCO2, D-13/D-25); USTAR contains readings up to 1039.9 m/s (physically impossible), and
     `reddyproc_pipeline.py`'s `mdc_gapfill()` used a mean-based fallback for extended blackouts, which a
     contaminated mean corrupts badly. **Fixed**: added `plausibility_filter()` (USTAR `[0,3]` m/s, VPD `[0,15]`
     kPa) + changed the fallback to median. Regenerated `reddyproc_processed*.csv`/`fch4_gapfilled.csv`/
     `forecast_features*.csv`/`forecast_daily_v2.csv` — data files are current. **Validated**: T2's spike
     resolved (325-413→2.9-27.7 nmol), T4's 2024 blackout resolved (mean 227.6→26.6). **But the impact is much
     bigger than the isolated spikes** — AR-feature means across the *entire* 2018-2023 evaluated window shifted
     systematically (T2: 71.9→20.2, T4: 54.6→31.1, T9: 79.2→36.6), now much closer to each tower's real observed
     CH4 mean (a strong correctness signal) but meaning **every forecasting benchmark (B01-B08) trained/tested
     on a biased AR feature and needs re-running** before its numbers can be trusted. Full write-up:
     `notebooks/04_feature_engineering/F09_results.md`.
   - **D-49 update (2026-07-02): B-03/B-03a/B-03b re-run on corrected data is done** (user-prioritised slice
     of the D-48 re-run). B-03 (production RF/XGB): small consistent gains, no ranking change. **B-03a
     (SARIMAX): major reversal — the original "collapses beyond h=1" conclusion is superseded.** Corrected
     daily R² (T4/T9 mean) 0.416→0.284 across h=1→14 (was 0.326→-0.177), now competitive with B-03's trees at
     every horizon, MASE<1 from h=3 onward. B-03b (TFT/TFT-Reg): mixed, direction-inconsistent movement across
     towers/tracks, no clean story either way — verdict unchanged (still non-competitive with B-03). Also ran a
     lightweight standalone **F-09a** gap-filling re-check (reusing F-08's exact gap-CV methodology, reduced
     scope, NOT touching the original `F08_external_sensors_RFm.ipynb`/`f08_summary.csv` per explicit
     instruction) — confirms real gap-filling accuracy improved (EXT/RFm_pool median R²: T2 0.490→0.574, T4
     0.376→0.402, T9 0.364→0.418), independent corroboration the D-48 fix is correct. See D-49,
     `notebooks/05_benchmarking/b03a_b03b_results.md` addendum, `results/f09a_summary.csv`.
   - **B-04 (DLinear/LSTM/LSTM_VSN) also re-run (2026-07-02, D-49):** DLinear improved cleanly at every
     horizon (daily R² 0.314→0.363 down to 0.164→0.192, h=1→14) — joins the "uniformly improved" group.
     LSTM/LSTM_VSN show the same horizon-inconsistent pattern as TFT (slightly worse short horizons,
     substantially better long horizons, LSTM flips negative→positive by h=14). Ranking unchanged — DLinear
     stays the DL baseline, still below RF/XGB/SARIMAX everywhere. See `b03_b04_results.md` addendum.
   - **B-09 (2026-07-05, D-53): recursive 365-day daily rollout backtest — does autoregressive
     forecasting compound error?** Direct empirical test of D-46 requirement 5. Single anchor
     (2021-12-16, Tower 4), one continuous 365-day recursive chain per model (SARIMAX, RF, XGB,
     LightGBM, DLinear, LSTM), every model fit fresh ≤ anchor. **The 1-7 day bin is a small-sample
     artifact** (only 3 real ground-truth days) — not a real "recursion fails immediately" signal.
     **Recursion does not inevitably collapse**: in the trustworthy 91-180/181-270 bins (n=88-90),
     every model except LSTM beats both persistence and climatology (positive R², MASE<1) — more
     encouraging than the original a-priori compounding-error concern. **Multi-anchor extension
     (2018-2022, D-53 addendum) revised the single-anchor picture substantially**: late-window
     degradation is year-specific (only 2018/2021 show it), not universal; **DLinear's striking
     single-anchor R² (0.417) was a fluke** (multi-anchor mean -4.752, worst of all models);
     **XGB is the most robust model** (only positive mean R², 0.003; mean MASE 0.968), LightGBM
     close behind. TFT/Tower 9/TabPFN-prep deferred as stretch items, not attempted. See D-53,
     `notebooks/05_benchmarking/b09_results.md`.
   - **B-10 (2026-07-06, D-54): recursive-rollout improvements — does anything fix the
     spike-blindness?** Three ideas tested on top of B-09's machinery, verdicts from the 5-anchor
     (2018-2022) sweep: **(1) blended AR fails cleanly** — pure recursion (alpha=1.0) always beats
     blending with climatology, for all three tree models. **(2) ensemble is a modest genuine win**
     — unweighted mean of RF+XGB+LightGBM+SARIMAX gets R²=0.012 (best of B-09+B-10 combined),
     beating best-individual XGB's 0.003, though MASE ticks up marginally (0.975 vs 0.968).
     **(3) H=1 DL retrain is mixed** — helps LSTM (R² -0.364 vs -0.438) but hurts DLinear (-1.729
     vs -1.460), not a uniform DL fix. None of the three closes the R² gap to a genuinely good
     result. **Recommendation: use the unweighted 4-model ensemble** if deploying one recursive
     rollout. See D-54, `notebooks/05_benchmarking/b10_results.md`.
   - **B-11 (2026-07-06, D-55): monthly-resolution rollout + downscale-to-daily — does coarser
     resolution help, and does the gain survive downscaling?** New `forecast_monthly_v2.csv` (via
     `build_forecasting_matrix_monthly.py`), SARIMAX+RF/XGB/LightGBM only (DL scoped out, too
     little monthly data). **Monthly-native evaluation is a real, substantial improvement** —
     confirms the M5-hierarchy prediction (LightGBM mean R²=0.156 vs 0.014 daily; XGB 0.147 vs
     0.003 daily). **But this does NOT survive downscaling back to daily** — downscaled mean R²
     (XGB -0.000, LightGBM -0.016) is essentially unchanged from B-09's own daily numbers, because
     the hybrid-calibration downscaling method reuses the daily template's own within-month shape
     unchanged and only corrects the coarse monthly bias (verified exact by construction). The one
     genuine win: the late-window bin (271-365) improves consistently across all four models. See
     D-55, `notebooks/05_benchmarking/b11_results.md`.
   - **B-12 (2026-07-06, D-56): combined ensemble + monthly-downscale — executed despite low
     expected payoff.** Combines B-10's ensemble with B-11's monthly-downscale framework. Single
     anchor looked like a clear win (R²=0.075 vs B-10 alone's -0.034) but the **5-anchor sweep
     reverses this** (mean R²: B-10 alone 0.012 vs B-12 -0.011) — the third single-anchor-vs-
     multi-anchor reversal this session (after B-09's DLinear, B-10's H1-retrain read). **B-10's
     daily ensemble alone remains the best recursive-rollout result and the recommendation** — this
     closes the B-09→B-12 experiment sequence. See D-56, `notebooks/05_benchmarking/b12_results.md`.
   - **B-13 (2026-07-06, D-57): TFT and TabPFN for recursive rollout, plus DLinear/LSTM chain-plot
     extension.** Fills B-09's remaining stretch items. **TFT**: D-45's regularization recipe
     (adapted validation window: last 90 days before anchor, not a full year) generalizes cleanly
     — no reproduction of the original catastrophic instability; mean R²=-0.237, MASE=1.055, the
     best DL model in the B-09-B13 sequence, still behind trees/SARIMAX. **TabPFN — headline
     finding**: a zero-shot foundation model (`tabpfn-time-series`, one-shot 365-day forecast, NOT
     autoregressive, local GPU inference) achieves mean MASE=**0.862, the best of any model tested
     this session** (beats B-10's ensemble at 0.975), with competitive mean R²=-0.006, using zero
     training/HPO. B-10's ensemble remains the headline R² recommendation, but TabPFN earns a
     standing mention as a genuine alternative. See D-57, `notebooks/05_benchmarking/
     b13_results.md`.
   - **B-14 (2026-07-06, D-58): GridSearchCV hyperparameter tuning for recursive rollout (retroactive
     log).** Re-opens the closed B-09→B-13 sequence for one bounded tuning round — literal `GridSearchCV`
     for RF/XGB/LightGBM (3-fold walk-forward CV, one-step R²), widened SARIMAX AIC-order grid, plugged
     into B-10's exact 5-anchor rollout mechanism for the real verdict. **Result**: CV-picked hyperparameters
     mostly don't transfer to rollout, except LightGBM (tuned mean R²=0.006, beats B-10's own untuned
     LightGBM at -0.014); the 3-model tuned ensemble (RF+XGB+LightGBM, no SARIMAX — a composition mismatch
     vs B-10's 4-model baseline, fixed in B-15) scores R²=-0.005. B-10's ensemble (D-54, R²=0.012) remains
     the best-validated configuration. See D-58, `notebooks/05_benchmarking/b14_results.md`.
   - **B-15 (2026-07-06, D-59): direct rollout-based hyperparameter tuning.** Follow-up to B-14 — scores
     hyperparameter combos by their own 365-day rollout R² instead of one-step CV, with a 2-anchor
     (2021+2019) combined-rank selection and the correct 4-model ensemble (RF+XGB+LightGBM+SARIMAX).
     **Result**: no uniform winner between CV-tuning and rollout-tuning (rollout-tuning wins on
     Ensemble/LightGBM, loses on XGB/RF), but **B-15's rollout-tuned LightGBM (mean R²=0.017) is the best
     single model found across the entire B-09→B-15 sequence**, ahead of B-10's own ensemble (0.012) —
     though the equal-weight 4-model ensemble dilutes this gain (R²=0.007), so B-10's ensemble remains the
     best-validated **ensemble** configuration. **This closes the B-14/B-15 hyperparameter-tuning
     side-thread** — B-10's ensemble stands, now confirmed robust to two independent tuning attempts. A
     rebalanced ensemble weighted toward the tuned LightGBM is a flagged, unexecuted follow-up. See D-59,
     `notebooks/05_benchmarking/b15_results.md`.
   - **B-15 addendum (2026-07-06, D-59): cross-tower generalization check (T2/T9).** B-14/B-15's tuning
     was Tower-4-only throughout (training pools T2+T4+T9, scoring never left T4). **T4-tuned LightGBM
     (T4's best model, R²=0.017) is T9's *worst* model (R²=-0.388)** — doesn't transfer. An independent
     tuning search scored on T9 (usable at 4/5 anchors; T2 usable at only 1/5, too scarce to tune) picks
     genuinely different hyperparameters but the validated outcome is **nearly identical to reusing T4's
     config** (within ±0.03 R²) — **T9's poor performance isn't primarily a tuning problem**, something
     more structural is limiting it. Anchor 2020 is a catastrophic whole-anchor outlier for T9 (R²
     -0.79 to -1.38), a partial explanation not fully diagnosed. T2 (1 usable anchor) too data-scarce for
     any reliable conclusion. 75 new chain plots in `results/figures/b15_chains/`. See D-59 addendum,
     `notebooks/05_benchmarking/b15_results_t2_t9.md`.
   - **F-09b (2026-07-04, D-51): outlier-correction technique comparison (winsorization/Hampel vs hard
     truncation) for the D-50-flagged WS/TA contamination.** Scoping finding first: D-50's contamination
     lives only in the raw EC-tower data — `build_sms_met_dataset.py`'s Site-station swap (D-35) means
     production's WS/TA are **already clean**, so D-50's fix is **not urgent for production** (downgrades
     that entry's framing). Tested anyway on the EC-tower-sourced series as real ground truth: hard
     truncation and winsorization both fully resolve the contamination; **Hampel filter only partially
     fixes WS and doesn't touch Tower 2's long stuck-sensor TA fault at all** (predicted in advance, then
     confirmed). Downstream gap-filling R² is a **null result** — indistinguishable across every config,
     even fully uncorrected, unlike D-48's USTAR/VPD fix which clearly moved things — candidate explanation
     is this calendar-gap CV harness may not exercise the long-blackout fallback failure mode that made
     USTAR/VPD damaging (untested caveat, flagged for any future contamination audit). See D-51,
     `notebooks/04_feature_engineering/F09b_results.md`.
   - **B-09→B-15 recursive-rollout sequence is now closed.** Final recommendation: B-10's unweighted
     ensemble (RF+XGB+LightGBM+SARIMAX, daily resolution) is the best available configuration
     (mean R²=0.012 across 5 anchors) — B-12 confirmed combining it with B-11's monthly-downscale
     framework does not improve on it, and B-14/B-15 confirmed neither CV-based nor rollout-based
     hyperparameter tuning produces a better ensemble (though B-15 did find a better single model,
     tuned LightGBM at R²=0.017 — flagged as an unexecuted rebalanced-ensemble follow-up).
   - **B01/B02/B05-B07 full rerun on D-48-corrected data — DESCOPED (D-60), not pending.** B-03 (the
     production model) was already rerun under the D-48 fix (D-48 addendum) and showed only small,
     non-ranking-changing gains (T4 RF daily h=1 0.357→0.365, h=14 0.270→0.280; T9 RF h=14
     0.342→0.359) — the fix's real-world impact on the best-performing model is modest. B01/B02/B05/
     B06/B07 are all already-established structural/architectural findings (DL underperforms trees,
     asinh transform is a dead end, hurdle architecture hurts daily R² by a wide margin, recency
     features are marginal) — a data shift of similar modest magnitude is very unlikely to reverse
     any of them. Do not spend further session time re-running these.
   - **I-02 (2026-07-06, D-61): feature importance (native/SHAP/LIME) for the recursive-rollout
     models.** Fresh experiment, not based on I-01 (different harness, explicitly not used as
     precedent, left untouched). Covers all 8 B-10/B-13 models, all 3 towers, full 5-anchor sweep.
     New `src/interpretability/importance.py`. **`fx_lsu_dens` (livestock density) confirmed as the
     dominant driver by every method that can see it** (native + SHAP across RF/XGB/LightGBM,
     SARIMAX coefficients at all 3 towers, TabPFN permutation importance at T4/T9) — reconfirms this
     project's central thesis from the rollout models themselves. **New finding: its SHAP importance
     grows with lead time**, not shrinks (7.6→38.4 mean|SHAP| from bin 1-7 to 181-270 at T4) — AR
     features degrade as the rollout's own predictions dilute real history, while the exogenous
     livestock signal doesn't. TabPFN's driver ranking differs sharply at Tower 2 (read as a
     data-scarcity symptom, not a real ecosystem difference). See D-61,
     `notebooks/06_interpretability_uq/I02_feature_importance_rollout.ipynb`, `I02_results.md`.
   - **U-02 (2026-07-06, D-62): quantile-ML + conformal uncertainty for the recursive-rollout
     models.** Fresh experiment, not based on U-01 (different harness, left untouched). RF via
     quantile-forest trick, XGB/LightGBM via 3 quantile-objective fits, SARIMAX via `conf_int()`,
     TabPFN via its own native `quantiles=` parameter, **TFT via a new `TFTQuantile` class**
     (added in a follow-up round after the initial conformal-only-wrap version left it as the one
     model with a genuine blank-gap failure mode — see D-62 addendum; verified fully
     backward-compatible with `B03b_tft.ipynb`/`B13_tft_tabpfn.ipynb`/`i02_multi_anchor_tower.py`,
     the only other TFT callers in the repo). Leave-one-anchor-out conformal calibration per
     lead-time bin. **Result: calibration works consistently across every model at T4/T9** — all
     converge to ~0.88-0.90 PICP regardless of raw coverage (tree models raw PICP 0.35-0.50, badly
     overconfident; SARIMAX/TabPFN/TFT all already reasonable raw, 0.72-0.92 — the training
     objective, not architecture complexity, appears to drive calibration quality). **Calibrated
     Ensemble/RF are sharpest (lowest pinball)** at both towers; TFT's calibrated pinball is the
     worst at T4 despite its good coverage. **Tower 2 cannot support calibration at all** (all
     conformal columns NaN, consistent with its known data scarcity) — raw intervals only, reported
     as low-confidence. Four real bugs caught and fixed before finalizing: a TabPFN quantile
     column-matching bug (silently gave 100% NaN raw intervals), an
     Ensemble_MASEweighted-identical-to-unweighted bug, a fan-chart whole-chain-fallback bug
     (caught via direct user inspection of the figures) that hid valid raw intervals behind
     spurious blank gaps, and TFT's original no-raw-quantile limitation (closed via `TFTQuantile`).
     See D-62,
     `notebooks/06_interpretability_uq/U02_uncertainty_rollout.ipynb`, `U02_results.md`, 120
     fan-chart figures in `results/figures/u02_fancharts/`.
   - **U-03 (2026-07-07, D-63): does U-02's conformal calibration hold up under distribution
     shift?** Direct follow-up to U-02, prompted by the pivot from "long-horizon forecasting" to
     **scenario simulation** for Phase 07 (2x-livestock, CMIP6-2050-climate). Found U-02's
     calibration never discussed the split-conformal exchangeability assumption — a real gap once
     scenario inputs are, by construction, not exchangeable with historical calibration data.
     **Part A (real ground truth, all 3 towers x 5 anchors):** no clear evidence that conformal
     PICP degrades with historical distribution shift (corr(shift score, PICP): T4 −0.166 n=5, T9
     +0.562 n=4 — small samples, no degradation signal either way); caveat — the shift magnitudes
     tested (max ≈2.0, one anomalous weather year) are far smaller than a genuine future scenario,
     so this does not certify calibration survives real scenario shift.
     **Part B (no ground truth, diagnostic only) — expanded twice after the user caught two
     successive scope gaps** (first "I don't think U03 has been completed for all models, towers,
     and years?", then "tower 2 should also be included... always include tower 2") **to its final
     scope: all 8 U-02/B-10/B-13 models x all 3 towers x all 5 anchors**, matching U-02's own
     coverage exactly. Sweeping `fx_lsu_dens` (livestock density) 1.0×→3.0× (Towers 4/9, 10 usable
     cases; Tower 2 structurally degenerate for this test — its `fx_lsu_dens` is exactly 0.0 for
     the entire rollout window in 4/5 anchors, a genuine data finding, not a bug) produced a clean
     structural split: **RF/XGB/LightGBM plateau** (mean +21–23%, the tree-extrapolation-ceiling
     signature), **TFT/TabPFN form a broadly comparable, muted-but-noisier cluster** (+26%/+30%
     mean; TabPFN ranges −4.9% to +90.1%, the least predictable model, sometimes inverting
     direction), **both ensembles sit in a distinct elevated tier** (+49–50% mean — B-10's
     production-recommended ensemble is NOT immune to this problem despite being 75% tree-weighted,
     since SARIMAX's 25% weight isn't diluted away), and **SARIMAX is the clear outlier** (+150%
     mean, 59–380% range, the maximum of all 8 models in 10/10 cases — the one fully robust ordering
     claim). **Recommendation: do not reuse U-02's conformal margins as validated intervals for
     genuine scenario predictions; do not reuse B-10's ensemble unmodified for scenario
     extrapolation without addressing its SARIMAX-inherited risk; any Tower-2 scenario needs an
     explicit livestock baseline independent of the 2019-2022 fitting window.** Directly motivates
     the detrend-and-residual/hybrid process+ML approach already flagged in this session's
     literature discussion. See D-63 (+ two addenda),
     `notebooks/06_interpretability_uq/U03_uncertainty_shift_robustness.ipynb`, `U03_results.md`,
     `results/u03_extrapolation_stress_test_multi.csv`, `results/figures/u03_fancharts/` (24
     figures).
   - **U-04 (2026-08-10, D-88): UQ recalibrated for the current forecasting champion (TabPFN+
     species, TabICLv2), closing a gap U-02 left behind.** U-02 (2026-07-06) predates TabICLv2
     joining the roster (D-66, 3 days later) and F-10's species features (D-67, 4 days later) — its
     "TabPFN" interval is calibrated for a feature config this project no longer recommends;
     TabICLv2 has never had UQ at all. User-confirmed scope: champion-focused (TabPFN+TabICLv2
     only, both zero-shot with native quantile support) over the full 11-model roster, since the
     other 6 U-02 models' feature config never changed. New script `u04_champion_uq.py` reuses
     U-02's `evaluate_stage()` unmodified (only `fit_stage` differs) on `forecast_daily_v3.csv`'s
     `BASE+species` config — **25-second runtime**, no retraining needed. **Result: calibration
     converges to ~0.89-0.90 PICP at T4/T9**, matching U-02's own headline finding on the new
     config; **T2 still cannot support calibration** (same pre-existing limitation, confirmed
     independent of the feature-set change). **The actual finding**: species enrichment improved
     point accuracy without materially changing calibration quality (TabPFN conformal MPIW/pinball
     essentially unchanged old-vs-new at both T4/T9) — mechanistically sensible, not assumed in
     advance. Foundation for "Option B" (scenario-analysis UQ, AOA-stratified, queued next). See
     D-88, `notebooks/06_interpretability_uq/u04_champion_uq.py`, `U04_results.md`. **Addendum,
     same day**: U-04 shipped without fancharts despite U-01/U-02/U-03 all having them — flagged
     directly, `u04_fanchart_plots.py` built, all 30 (tower×anchor×model) combinations,
     `results/figures/u04_fancharts/`.
   - **U-05 (2026-08-10, D-89): scenario-analysis UQ ("Option B"), built on U-04's method but on
     S-05's own architecture (FX_A_SPECIES, not U-04's BASE+species — different feature space,
     different model).** New script `u05_scenario_uq.py`: TabICLv2 zero-shot, 5 real anchors × 3
     towers, same leave-one-anchor-out conformal machinery, **9-second runtime**. A real leakage
     bug caught by smoke-testing (AOA training set was unrestricted, giving every test point a
     literal distance-to-self of 0 — fixed to pre-anchor-only, recomputed per anchor). **Step 3
     resolved the plan's design question empirically**: |residual| vs. AOA-flagged status shows a
     weak raw correlation (r=0.146) but a real, substantial categorical gap (out-of-AOA residuals
     ~48% larger, pooled) — landed on a **two-tier margin interpolated continuously by each point's
     own aoa_flagged_pct**, a genuine third option between the plan's original Level 1/Level 2.
     **Applied to S-05's existing livestock/grazing/fertilizer outputs with zero new model calls**
     (pure post-processing join) — a second bug (flat per-tower margin not actually using Step 3's
     finding) caught and fixed before finalizing. **Calibration converges to ~0.88-0.89 PICP at
     T4/T9**, matching U-02/U-04 a third time; T2 stays uncalibratable (third confirmation).
     **Interval is genuinely wide, stated plainly**: ±94-100% of mean in-AOA, ±139-140%
     out-of-AOA — consistent with U-01's original "large aleatoric uncertainty" finding (D-40).
     15 calibration fancharts + 1 applied-trajectory figure (keeps the calibrated interval visibly
     separate from realization spread, per D-85's own lesson about not merging uncertainty
     sources). No change to any standing recommendation — UQ infrastructure, not a new scenario
     finding. See D-89, D-88, `notebooks/06_interpretability_uq/u05_scenario_uq.py`,
     `U05_results.md`.
   - **U-06 (2026-08-10, D-90): CQR fixes the spike-coverage failure U-04/U-05's own fancharts
     revealed visually.** User observation, checked directly: "a lot of spikes are still beyond
     the interval." Confirmed and quantified — overall PICP≈0.89 looked fine, but **75% of the
     top-10%-magnitude days fell entirely outside the interval, vs. 3.3% for the bottom 90%**
     (split-conformal's flat symmetric margin only guarantees *average* coverage). A second check
     before building anything: raw q95 already sits close to/exceeds actual spike values (TabPFN
     182.7 vs. 193.1; TabICLv2 ~360 vs. 193.1) while the median (~35) massively undershoots —
     directly motivating **Conformalized Quantile Regression** (nonconformity =
     `max(q05-y_true, y_true-q95)`, interval = `[q05-margin, q95+margin]`) over a smooth
     AOA-distance function or magnitude-binned margins. `rr.conformal_margins_by_bin()` reused
     completely unchanged (4th reuse across U-02/U-04/U-05/U-06) — **no new model calls**, pure
     recalibration of already-saved chains. A bug caught before reporting (T2 showing 0.0 instead
     of NaN — missing the same all-NaN aggregation guard U-02's own `wavg()` already documents,
     fixed by replicating it). **Result: spike coverage roughly triples** (TabICLv2: 24.3%→79.7%
     U-04, 22.1%→79.3% U-05; TabPFN: 24.3%→57.2%), at the honest cost of normal-day coverage
     dropping to ~83-88% (still >80%) and spike intervals roughly doubling. TabICLv2 benefits more
     than TabPFN, consistent with its raw q95 already exceeding actual spikes. **CQR should replace
     the symmetric-margin approach as the standing UQ method going forward** — not yet applied to
     S-05's actual scenario trajectories (would need raw daily q05/q95 saved for scenario points,
     a small flagged next step). See D-90, D-88, D-89, D-40,
     `notebooks/06_interpretability_uq/u06_cqr_recalibration.py`, `U06_results.md`.
   - **U-07 (2026-08-10, D-91): livestock-density-stratified CQR — thinner margins where
     livestock presence is smaller, a much cleaner stratifier than AOA distance turned out to be.**
     Direct user question on U-06's output, checked empirically before building: "can't the margin
     be thinner where livestock presence is smaller?" **Signal much stronger than U-05's AOA-
     distance check**: `corr(|residual|, fx_lsu_dens)=0.43-0.45` (vs. AOA's weak 0.09-0.15),
     residuals ~3.2x larger on above-median-LSU days. `fx_cattle_dens` correlates almost
     identically (0.427, consistent with S-05's cattle-dominance finding); sheep/lamb at noise
     level. Same CQR machinery as U-06 — only the bin key changes to lead-time × LSU-tertile;
     `conformal_margins_by_bin()` needed zero code changes (5th reuse across U-02/U-04–U-07).
     Tertile boundaries from calibration anchors only, no-leakage discipline maintained. **Result:
     low-LSU intervals are 29-46% the width of high-LSU intervals** (TabPFN: 84.3 vs. 293.7 nmol)
     — **a genuine win-win, not a trade-off**: verified directly that spike days (3.2× higher
     `fx_lsu_dens` than normal days) still get their own dedicated, appropriately-wide calibration
     in the "high" tier, not diluted the way the single pooled CQR margin was. **Should be the
     standing UQ method going forward, layered on U-06's CQR.** See D-91, D-90,
     `notebooks/06_interpretability_uq/u07_lsu_stratified_cqr.py`, `U07_results.md`. **Addendum
     (full-roster figures):** comparison plot extended from 1 chain to the full champion roster
     (T4/T9 × TabPFN/TabICLv2 for U-04, T4/T9 × TabICLv2 for U-05 — T2 explicitly logged as skipped,
     0% valid margin, not silently dropped). Low-as-%-of-high MPIW range widens to **26-59%** across
     all 6 combinations (still the same direction/magnitude everywhere).
   - **S-05 + UQ (2026-08-10, D-92): U-06/U-07's CQR calibrations attached to S-05's ACTUAL scenario
     trajectories, closing the last standing gap between scenario analysis and UQ.** Two new
     scripts, zero new calibration fitting: `s05_uq_daily_chains_subset.py` reruns S-05's existing
     18-call/axis representative subset (livestock+grazing+fertilizer, 54 calls total, 2050
     horizon) requesting `quantiles=(0.05,0.5,0.95)` instead of a point prediction — **confirmed
     free** (4.3s/call, same as point-only; TabICL always computes an internal quantile grid
     regardless) — full run **~2.5 min**, 0 failures. `s05_uq_cqr_apply.py` attaches U-05's own
     FX_A_SPECIES-architecture CQR/LSU-CQR margins via a (tower, lead-bin[, LSU-tier]) lookup,
     pooled across U-05's 5 anchor years. Two explicit extrapolation assumptions (stated, not
     hidden): lead times beyond 365 days hold the widest calibrated bin's margin flat (likely
     **understates** true uncertainty at year 20+, since error should plausibly grow not plateau —
     read far-horizon bands as a floor); grazing/fertilizer axes reuse the livestock-architecture
     margins despite extra covariates (same approximation U-05's own Step 4 already made). **Result:
     works cleanly, >99% coverage at T4/T9** (T2 0%, pre-established degeneracy, not new); one
     genuine thin spot (T4 days-1-7 × mid-LSU-tier, zero calibration samples, surfaced as NaN).
     Verified directly: zero interval inversions, CQR correctly tightens the model's own raw
     quantile spread on average at T4 (raw MPIW 572.6 vs. U-06 514.6 vs. U-07 523.4). **This closes
     the U-04→U-07 UQ arc's last open caveat — no "not yet applied to S-05" gap remains.** See D-92,
     D-91, D-90, D-89, `notebooks/07_scenario_analysis/s05_uq_daily_chains_subset.py`,
     `s05_uq_cqr_apply.py`, `s05_uq_cqr_plots.py`, `s05_results.md`.
   - **B-16 round 2 (2026-08-10, D-93): TICA embeddings + static AR-lag features for TabICLv2 —
     both negative, combining them is actively worse, `BASE+species` remains champion.** User
     revisited forecasting focused on Track B (TabICLv2 zero-shot) specifically, requested as a
     single consolidated notebook. Worked through two real design forks before building: (1)
     dynamic vs. static AR-lag features — TabPFN/TabICLv2 are single-shot (no day-by-day
     recursion), so genuine dynamic lags would need a new roller costed at ~30h+ (daily) or
     ~75-90min (30-day blocks) for a full sweep; user caught the static design's real weakness
     directly ("does that mean May/June forecasts still use December's AR features?" — yes,
     confirmed) but chose to proceed with static anyway given the cost gap. (2) checked whether
     S-05 already has this problem — confirmed S-05 uses the identical single-shot architecture
     and has zero AR-lag columns at all; a climatology-anchored (not prediction-chained)
     alternative was flagged as cheaper/more portable for later but not built this round.
     **Result**: TICA alone (MASE=0.7348) and static AR alone (0.7358) both land within noise of
     `BASE+species`'s baseline (0.7353) — TICA replicates D-79's own gap-filling "wash" finding in
     a different task; static AR adds little since TabICLv2 already sees raw history natively.
     **Combining both is clearly worse (0.7603, +0.025), consistent across all 3 towers** — matches
     this project's recurring "stacking too many feature families hurts" pattern (D-67's
     `BASE+ALL` finding). Two real implementation bugs caught by the notebook's own smoke test
     before the full sweep (not discovered mid-sweep): a zero-variance-history-column bug from
     over-broadcasting the static features, and a non-positive-definite TICA covariance matrix
     from 43 collinear drivers (fixed with ridge regularization). See D-93,
     `notebooks/05_benchmarking/B16_tica_static_ar_features.ipynb`.
   - **B-16 round 4 (2026-08-10, D-94): pooled vs. solo for TabPFN/TabICLv2 on forecasting
     specifically — splits by model, TabPFN gets a small real gain, TabICLv2 replicates the
     gap-filling wash.** Direct follow-up user question — pooling was adopted for Track A from
     gap-filling's F-02/F-03 finding, Track B stayed solo based on a gap-filling-only precedent
     (D-79) that used a DIFFERENT API (sklearn-style `TabICLRegressor`, not the TS-native
     `TabICLForecaster` the champion actually runs on) — never re-tested on forecasting with the
     real architecture. Verified before building: both `TabICLForecaster`/`TabPFNTSPipeline`
     natively support an `item_id` column for genuine multi-series panel input (checked via direct
     inspection, then smoke-tested — ~2-5s for a pooled call covering all 3 towers, actually faster
     than 3 solo calls). Solo baseline reused from U-04's `u04_chains.csv` (same exact config, not
     rerun). Two real bugs caught by the smoke test/traceback: pooled/batched calls don't tolerate
     the same partial NaN solo calls handle fine (fixed with mean-imputation); Tower 9's two
     earliest anchors have empty pre-anchor history, silently NaN-poisoning the climatology
     baseline (fixed with a skip-and-log guard). **Result: TabICLv2 pooled ≈ solo (0.7355 vs.
     0.7353, noise — direction matches D-79 but far smaller). TabPFN pooled beats solo at all 3
     towers** (0.7138 vs. 0.7166 overall; T4 R² crosses from slightly negative to slightly
     positive) — small but real and consistent, never tested by D-79 at all. Not large enough to
     force an immediate champion switch given the deadline, but a concrete, cheap improvement lead.
     See D-94, `notebooks/05_benchmarking/B16_pooled_vs_solo.ipynb`.
   - **B-17/B-18 TabPFN improvement programme (2026-08-19, D-106): new long-horizon numerical
     benchmark, with the B16→B18 gain explicitly decomposed.** Same observed-target,
     climatology-MASE convention, all 3 towers × 5 anchors (2,127 observed evaluation points across
     9 evaluable tower-anchor blocks); strict candidates use observed methane history only through
     each anchor plus known-future `fx_*` drivers, never future methane. **The main improvement was
     architectural:** replacing B16's one-shot TabPFN-TS wrapper (`BASE+ALL`, MASE=0.7123) with a
     generic direct pooled TabPFN v2 regressor over observed daily rows, all 52 `fx_*` predictors,
     tower indicators, year, and elapsed time reduced MASE to 0.6958 (B17, −2.31%). B18 then added
     moderate recency (1,095-day raw MASE=0.6930; 1,460-day tower-robust=0.6934), a conservative p95
     event correction (25% of predicted spike excess; **best single/gated MASE=0.6924**, RMSE=59.53,
     R²=0.205), and a fixed equal mean of those three complementary forecasts (**exploratory best
     MASE=0.6908**, MAE=29.821, R²=0.192). Total same-protocol improvement from B16-style v2 to B18:
     −0.0215 MASE (−3.01%); B17→B18 contributes only −0.0050, so the wrapper→direct-regression change
     is the substantive step. **Caveat retained:** the ensemble's extra gain is not independently
     stable under block-wise model/weight selection; p95 spike magnitude remains the dominant
     failure (MASE=3.285 vs. 0.524 non-spike). Rich antecedent features, seasonal experts,
     recency replication, tower-month normalisation, hard spike gates, and tower-adaptive switching
     were all tested and rejected. Raw chains, all 15 B15-style figures, bootstrap, and full tables
     are saved under `results/b18_*` / `results/figures/b18_*`; see
     `report/Outlines/B18_forecasting_experiment_results.md`. **Benchmark change only:** I-03/U-04
     and Phase-07 scenario outputs remain tied to the prior B16-style architecture and have not been
     silently reinterpreted as B18.
   - **S05-T2 (2026-08-10, D-95): does pooling rescue Tower 2's muted livestock-scenario response?
     No — exactly 0.0pp difference, both TabICLv2 and TabPFN, a decisive negative result.** Direct
     follow-up to T2's muted-response finding (cattle 3× = only +1.8-2.3% at T2 vs. +186-215% at
     T4/T9) — tested whether pooling T2's context with T4/T9's real livestock-rich history (the
     exact mechanism that rescued Tower 9 in gap-filling, F-02/F-03) lets the model borrow their
     learned cattle sensitivity. Confirmed the API mechanics first (every context `item_id` must
     also appear in `future_df`, or `TabICLForecaster` raises a `KeyError` — caught via smoke test,
     not mid-sweep); T4/T9 get a minimal 30-day placeholder future purely to satisfy this, their
     own predictions discarded. **Result: exactly 0.0 percentage points of difference for both
     models, every combo/SSP** (TabICLv2: +1.8%→+1.8%, +4.2%→+4.2%; TabPFN: +11.0%→+11.0%,
     +17.2%→+17.2%) — not just small, an exact match to the decimal. **Mechanistic read**:
     `item_id`-based pooling shares context rows within one batched call, not fitted parameters —
     unlike Track A's trees (where every split is genuinely informed by all towers together), a
     zero-shot forecaster's output for one series stays driven almost entirely by that series' own
     history; other towers being present doesn't move it. Retroactively explains why D-94's own
     pooling gain was so small — likely a minor batching effect, not real cross-series transfer,
     and it vanishes entirely under a genuine extrapolation test. **T2's muted response should be
     read as a genuine model-extrapolation limit, not a fixable data-availability gap** — the
     "maybe pooling fixes it" question is now closed empirically. Secondary finding: TabPFN's own
     solo T2 response (+11-17.5%) is meaningfully larger than TabICLv2's (+1.8-4.2%), independent
     of pooling. See D-95, `notebooks/07_scenario_analysis/s05_t2_pooled_test.py`, `s05_results.md`
     ("Update (S05-T2, D-95)").
   - **Delta-method bias correction, all of Phase 07 (2026-08-17, D-100).** S-01's own Finding 1
     (1x baseline doesn't exactly reconstruct real historical mean, 9-20% gap) already accepted as
     small — applied a rigorous climate-impact-modelling correction on top anyway (anchor LEVEL to
     real mean, trust model for SHAPE of change), reported additively alongside raw numbers. **S-01
     (T2 +33.8%→+40.7%, T4 +138.2%→+135.4%, T9 +104.7%→+114.4%) and S-04 (same frozen model, same
     offset transfers directly) shift modestly**, as expected from S-01's own small gap. **S-05
     (TabICLv2 + `FX_A_SPECIES`, never checked before, a structurally different zero-shot model)
     turned out very different: 40-80% underprediction of the real historical mean at every tower**
     (T2 -69.9%, T4 -52.4%, T9 -40.8% vs `y_gapfilled`; consistent under `y_observed` too) —
     2-4x larger than S-01/S-04's gap, always an undershoot. Correcting it **roughly HALVES the
     cattle-dominance headline** (T4 3x-alone +213.9%→+101.8%, T9 +187.3%→+110.4%). Dominance
     *direction* unaffected; exact *magnitude* now genuinely uncertain between raw/corrected —
     flagged, not resolved. **Affects prior citations directly** (BEST_RESULTS.md's "+214.5%",
     D-85, report Chapter 7) — updated where touched this pass, flagged for follow-up elsewhere.
     No retraining, fully additive (`_bias_corrected` files, originals untouched). See D-100,
     `notebooks/07_scenario_analysis/d100_bias_correction_s01_s04.py`, `d100_bias_check_s05.py`.
   - **I-03 (2026-08-18, D-102): interpretability recalibrated for the then-current B16-style forecasting champion
     (TabPFN+species) — closes the same "predates the champion" gap U-04 already closed for UQ.**
     I-02 (D-61) predates TabICLv2 (D-66) and F-10's species features (D-67) by days — its SHAP/
     permutation results, currently in Chapter 6, were computed on the old 8-model roster and old
     (BASE-only) feature set, never on the model this project actually recommends. **Scope
     (champion-focused, user-directed): TabPFN only** — "the best forecasting model... the model
     that was ingested for S-03" (confirmed via S-03's own table: TabPFN MASE=0.855, lowest/best of
     all 11 models tested there). TabICLv2 not covered, flagged as a cheap follow-up. **Method
     unchanged from I-02's own TabPFN treatment** (permutation importance, TabPFN's only available
     substitute for a native signal) — only the feature set changed to BASE+species (52 `fx_`
     columns, `forecast_daily_v3.csv`, F-10's actual champion config). Same 3-tower × 5-anchor
     sweep, ~22-minute runtime, zero training. **Result: `fx_lsu_dens` dominance confirmed on the
     actual champion** (mean importance 1.1456, #1 of 52). **New finding I-02 could not make**:
     `fx_cattle_dens` is a clear #2 (0.8043) while `fx_sheep_dens`/`fx_lamb_dens` rank near the
     bottom (0.0143/0.0320) — F-10's species-disaggregation gain is concentrated entirely in the
     cattle component, independently corroborating S-05's scenario cattle-dominance finding via a
     completely different method. **Tower 2 shows zero livestock features in its top 10** (all
     TA/TS/SWIN) — a fourth independent confirmation of T2's livestock-blindness (after U-03, S-01,
     S05-T2/D-95). See D-102, `notebooks/06_interpretability_uq/i03_champion_interpretability.py`,
     `I03_results.md`.
   - **S-01 (2026-07-07, D-64): first Phase 07 scenario-simulation worked example (level-residual
     hybrid), all 3 towers, all 3 residual models broken out individually.** Phase 07 moves from
     "PLANNED, not started" to a proven end-to-end mechanism. B-08 confirmed superseded for Phase
     07's purposes by U-03 (user-confirmed). Architecture (deep-research-informed, user-scoped to a
     parametric — not mechanistic — trend for this pass; SPACSYS, already validated at North Wyke,
     logged as future work): a Ridge trend model (fit **once** on the full pooled real record, not
     per-anchor — the fix for U-03's SARIMAX-instability finding) carries the climate+livestock
     extrapolation; RF/XGB/LightGBM (B-10's exact hyperparameters, monotonic constraint on
     livestock density for XGB/LightGBM) correct only the residual; `fx_USTAR_mean`/`fx_SHF_mean`
     dropped entirely (no climate-scenario-product source at all). Caught and fixed a real unit
     mismatch (CMIP6 `RAD` is MJ/m²/day, this project's `fx_SWIN_mean` is W/m² — verified conversion
     via a physical sanity check). **Scenario**: SSP2-4.5 ensemble-mean (5 GCMs × 100 realizations),
     2041–2060 ("the 2050s"), all 3 towers × {1×, 2×, 3×} livestock multipliers on the day-of-year
     climatology of `fx_lsu_dens`. **Result: the hybrid measurably fixes U-03's flattening** — the
     same 3× sweep that gave trees-alone only +21–23% (U-03) gives **+138% at T4, +105% at T9**
     here, since the trend now carries the extrapolation. **Genuine, non-obvious finding**: a "2×
     livestock" scenario built by scaling a *smoothed climatology* is meaningfully milder than one
     built by scaling raw daily values (U-03's method) — only 3× genuinely exceeds the training
     envelope (Area-of-Applicability check, a lightweight from-scratch Python implementation in the
     spirit of Meyer & Pebesma 2021), confirmed at T4/T9 (5.5%/6.0% flagged) but **never at Tower 2**
     (its own livestock baseline is ~7-8× smaller than T4/T9's, directly consistent with U-03's own
     T2 finding). **Per-model breakdown finding**: a full monotonic sweep shows XGB/LightGBM's
     residual correction is completely flat with respect to livestock density — for two of three
     tree models, ~100% of the scenario response flows through the trend, not the residual; only RF
     (no native monotonic-constraint support) shows real residual sensitivity. Frozen model artifact
     persisted for the first time in this project (closes D-46 requirement 1). Explicitly a
     proof-of-mechanism, not a final output — every caveat (parametric-not-mechanistic trend,
     9/11 drivers historical-day-resampled, naive livestock multiplier, U-02/U-03 intervals NOT
     attached) carried forward. See D-64 (+ addendum),
     `notebooks/07_scenario_analysis/S01_first_scenario.ipynb`, `s01_results.md`,
     `results/s01_scenario_summary.csv`, `results/figures/s01_*.png` (4 figures).
   - **D-65 (2026-07-07): `bin_metrics()` extended with RMSE/WAPE/Correlation** (was R²/MAE/MASE
     only, narrower than the B01-B07 phase's full roster). B-10 + B-13 reconstructed and rerun
     with the fuller metric set (their original multi-anchor scripts were never committed — this
     one is, closing that gap). **Reproduction confirmed bit-for-bit for 7/8 models at Tower 4**
     (TFT differs due to its already-documented unseeded-init non-determinism). **Addendum: extended
     to all 3 towers (T2/T4/T9)** per the new "full coverage by default" CLAUDE.md convention (this
     exact narrow-scope gap had already recurred 3 times this session: U-03, S-01, and this rerun).
     **All-tower pooled headline is substantially worse on R² than the T4-only number but comparable
     on MASE** (Ensemble_unweighted R² 0.012→−0.165, MASE 0.975→0.918) — driven by Tower 9 being
     consistently harder and Tower 2 being largely degenerate outside 2018; model ranking is
     unchanged (Ensemble/TabPFN still best on MASE, SARIMAX/TFT still worst). **New finding: TabPFN's
     best-in-sequence MASE (0.862 T4-only / 0.855 all-tower) does not extend to RMSE** (second-worst
     in both tables) — its strength is consistency vs. persistence, not small worst-case errors.
     Correlation uniformly weak (0.26-0.40) across every model/scope. No change to the standing
     recommendation (B-10's Ensemble_unweighted best on R² and RMSE in both tables). A new
     tower×year×model breakdown table added. See D-65 (+ addendum),
     `notebooks/05_benchmarking/b10_b13_metrics_rerun.md`, `results/b10_b13_rerun_table.csv`,
     `results/b10_b13_rerun_table_all_towers.csv`, `results/b10_b13_rerun_table_by_tower_year.csv`.
     **Second addendum (2026-07-08, user's own idea): secondary/exploratory metric scored against
     `y_gapfilled` instead of `y_observed`** — a deliberate, bounded departure from D-36/D-37's
     "train on gap-filled, evaluate on observed" convention for this one check, with a real
     circularity caveat (`y_gapfilled` seeds the AR history and shares features with the
     forecasters), stated up front. Unlocks Tower 2's coverage completely (816→14,600 of 14,600) but
     **R² gets worse while RMSE/MASE improve** — a variance-normalization artifact of scoring against
     a smoother target, not a contradiction. Model ranking mostly holds (Ensemble_unweighted stays
     top-tier either way) except **TabPFN drops from best-R² to 6th of 8** — the one real
     disagreement, worth flagging as a reason not to over-read TabPFN's observed-target R² edge.
     Bounded to the B-10/B-13 rerun only (no retrofit into U-02/U-03/S-01/I-02 this round, user-
     confirmed); no new D-number (logged as a further D-65 addendum). See
     `notebooks/05_benchmarking/b10_b13_metrics_rerun.md` ("Secondary metric" section),
     `results/b10_b13_rerun_summary_vs_gapfilled.csv`,
     `results/b10_b13_rerun_table_vs_gapfilled_all_towers.csv`,
     `results/b10_b13_rerun_table_vs_gapfilled_by_tower_year.csv`.
     **Third addendum (2026-07-08): DLinear/LSTM model-roster extension** — closes the gap between
     the `b10_chains` figures (existed for all 3 towers) and their evaluation metrics (didn't).
     Reconstructed from `B09_recursive_rollout.ipynb`'s exact recipe; confirms D-53/D-54's finding at
     full coverage (both drastically worse than every other model — all-tower R²: DLinear −5.06,
     LSTM −1.36 — correctly excluded from B-10's ensemble). **Also produced a sharper, generalized
     version of D-62's TFT non-determinism finding**: LSTM reproduces bit-for-bit exactly every
     time; DLinear only differs on the very first anchor processed in a run — root cause is
     `torch.manual_seed()` being called *after* model construction in `fdl.train_model()`, so only
     the first torch model built in a process (before any prior seed call) gets non-deterministic
     initial weights — likely the same mechanism behind TFT's own non-determinism, and implies
     seeding once at the top of a script would fix it (not applied, to avoid silently changing
     already-published numbers). **Separately discovered while doing this**: TFT's row in the
     currently-saved `b10_b13_rerun_summary.csv` had independently drifted to a third random draw
     since the primary all-tower/Tower-4-only tables were built — **reconciled 2026-07-09** (D-66
     second same-day addendum): every TFT-citing table/CSV in `b10_b13_metrics_rerun.md` updated to
     the live draw (all-tower R²=−0.363, Tower-4-only R²=−0.228), no longer stale. See
     `notebooks/05_benchmarking/b10_b13_metrics_rerun.md` ("Model-roster extension" section).
     **Same-day follow-up**: DLinear/LSTM added to the gap-filled-target secondary-metric table too
     — DLinear's pooled R² there (−6576.7) is dominated by a genuine numerical divergence in the
     2018 anchor's non-deterministic draw (MAE up to ~7,545 nmol m⁻² s⁻¹, physically implausible),
     amplified by the low-variance gap-filled target; excluding that anchor gives R²=−7.035, still
     worst but interpretable — reported plainly rather than smoothed over.
   - **D-66 (2026-07-09): TabICLv2 added to the B-10/B-13 sequence.** New tabular foundation model
     (ICML 2026, "heavily inspired by TabPFN-TS" per its own docs) — new sibling script
     (`b10_b13_tabicl_extension.py`), mirroring TabPFN's per-tower/per-anchor/never-pooled
     integration; API contract (context_df/future_df covariate convention, always-present default
     quantile grid) verified empirically before writing any real code, not assumed from incomplete
     docs. **Full 3-tower × 5-anchor sweep in under 30 seconds** — dramatically cheaper than every
     other model in the sequence. **Corrected 2026-07-10** (user-prompted skepticism caught a real
     bug: `tabicl_forecast()` was extracting `TabICLForecaster`'s **mean**-based `target` column
     rather than the median, badly biased high on this heavy-tailed, spike-dominated flux
     distribution — fixed to use the `0.5` quantile column instead). **Corrected result: all-tower
     R²=−0.329 observed, −0.886 gap-filled** — now beats SARIMAX (−0.360) and TFT (−0.363), MASE
     (0.928) is 4th-best of 10 models, still behind TabPFN (−0.122) and the standing recommendation
     (Ensemble_unweighted, −0.165) but no longer near-bottom. Shares TabPFN's exact Tower-9/2019
     degenerate-forecast limitation (zero real `y_observed` pre-anchor → flat ~0.0 prediction,
     unaffected by the point-estimate fix — not a new bug). New decision number (not a D-65
     addendum), matching the D-57 precedent that a genuinely new model gets its own number. See
     D-66, `notebooks/05_benchmarking/b10_b13_metrics_rerun.md` ("Model-roster extension: TabICLv2").
   - **D-67/D-68 (2026-07-10): F-10, extended feature engineering — Stage 1 (build + signal check),
     reopening the `04_feature_engineering` phase.** User pivoted the search for improvement from
     models to features, given the whole B-09→B-15 sequence has converged to roughly the same
     ceiling regardless of model/HPO. Built 5 new feature families onto a new
     `forecast_daily_v3.csv` (additive clone of v2, nothing existing edited): livestock species
     disaggregation (`fx_cattle_dens`/`sheep`/`lamb`, exact-lossless refinement of `fx_lsu_dens`),
     a land-use regime flag (`fx_is_arable`, Tower 2 flips 2019-09-09, Towers 4/9 never flip),
     catchment flow (`fx_flow_mean`+lags/rolls, 87-92% coverage), previously-unused fertilizer/
     management recency columns, and a bonus liveweight-density feature. **Two real implementation
     bugs caught and fixed during verification**: `fx_is_arable`'s first trigger was too broad
     (fired on routine grassland renovation, not just genuine arable conversion) and produced false
     positives at Towers 4/9; `fx_flow_lag*` was wholly NaN from an hourly-vs-daily indexing bug.
     **Stage 1's cheap signal check (leave-one-group-in RF ablation, all 3 towers, h∈{1,14}) found
     none of the 5 families clear the pre-registered go/no-go bar** — reported honestly as a null
     result (D-31/D-32/F-01-P2 precedent). `fx_cattle_dens`/liveweight density draw real SHAP
     attention but a follow-up swap test shows a tower-specific pattern (helps T4, hurts T9) that
     nets to ~zero, not a consistent win. **Stage 2b (recursive-rollout confirmation) run anyway
     per direct user instruction** ("test forecasting performance improvements, no gap-filling") —
     the real B-10 ensemble, all 3 towers, all 5 anchors, 6 configs. `BASE` reproduced the
     published headline almost exactly (R²=−0.1652 vs. −0.165); **none of the 5 families beat it**
     on the ensemble or any individual tree model, same tower-specific species pattern replicated
     independently. **Follow-up `BASE+ALL` run** (all 18 columns stacked, cheaply reusing the
     already-fitted SARIMAX chains): still loses to `BASE` but lands mid-pack, unlike Stage 1's
     point-forecast check where stacking everything was clearly the worst config — the two
     harnesses agree on the headline (nothing beats `BASE`) but not on the exact shape of the
     stacking penalty. **MASE promoted to the primary forecasting metric** (new `CLAUDE.md`
     convention — CH4's spike-tail behavior repeatedly destabilizes R²); re-read under MASE,
     `BASE+species` is actually marginally the best tree-ensemble config (0.9161 vs `BASE`'s
     0.9169), a small reversal of the earlier R²-led framing. **Stage 2c/2d: tested "all models
     from SARIMAX to TabICLv2" per user request** (TabPFN/TabICLv2 — zero-shot, cheap; TFT/DLinear/
     LSTM — required a new hourly-track matrix, `forecast_features_v3.csv`, since `forecasting_dl.py`
     needed zero code changes to consume it). **Result: the tree-only finding does NOT generalize —
     every attention-based/foundation model shows real, often large, gains from feature families**
     (TFT's `BASE` loses to persistence, MASE=1.063; `BASE+ALL` beats it, MASE=0.941). **New
     observed-target best: `TabPFN+species`, MASE=0.840/R²=−0.084 — beats the standing B-10
     ensemble recommendation outright on this metric, at near-zero adoption cost. Promoted to
     `BEST_RESULTS.md`.** Gap-filled secondary metric added for every model's best config too (per
     user request, D-65 second-addendum pattern) — **caught and corrected an overstated initial
     claim here after direct user questioning**: the gap-filled/observed comparison does NOT
     uniformly make every model worse — it splits cleanly by whether the model's own training
     target is `y_gapfilled` (trees/SARIMAX/ensembles score *better* on gap-filled, a circularity
     artifact from directly regressing onto that series) or `y_observed` (TabPFN/TabICLv2/TFT/
     LSTM/DLinear score worse, no such boost). **Consequence: under gap-filled scoring the OLD
     standing ensemble (MASE=0.749) beats the NEW winner (`TabPFN+species`, MASE=0.944) by a wide
     margin — the full ranking flips.** The observed-target ranking remains the one to trust
     (D-36/D-37 convention specifically favors it, since it isn't inflated by that circularity),
     but "TabPFN+species is the new best" is true on that metric only, stated plainly rather than
     unconditionally. Neither track touched the gap-filling pipeline at all — F-10 was scoped to
     forecasting only throughout. **D-68 (separate,
     documentation-only)**: reconciles D-63/D-64's "Tower 2 data sparsity" framing with D-28/D-30/
     D-34's already-correct "Red-farmlet arable conversion" explanation — the same underlying fact,
     just not cross-referenced later; no numbers change. See D-67 (+ 4 addenda), D-68,
     `notebooks/04_feature_engineering/F10_results.md`.
   - **D-69 (2026-07-13): S-02, driver-reconstruction feasibility (preparation, not yet
     integrated).** New idea, confirmed never considered in D-52/D-64: train proxy models
     predicting CMIP6's 6 missing scenario variables (`fx_WS_mean`, `fx_VPD_mean`, `fx_PPFD_mean`,
     `fx_RN_mean`, `fx_TS_mean`, `fx_SWC_mean`) from the 4 it provides (`Tmin/Tmax/Rain/RAD`), using
     real historical NWFP data, then compare against the current climatology-resampling baseline
     (D-52) — more scenario-responsive in principle, since climatology ignores how extreme a given
     future day's available drivers actually are. Correlation evidence (D-50) predicted a mixed
     picture; **user chose to attempt all 6 anyway** for a complete picture. New notebook
     `notebooks/07_scenario_analysis/preparation/S02_driver_reconstruction_feasibility.ipynb`
     (`fco2_gapfill.py`'s exact RF architecture, D-26, reused). **Two genuine surprises reversing
     the pre-registered expectation**: wind speed (weakest correlation, r=−0.11 to 0.31) is the
     *strongest* RF win (R²=0.363 vs. climatology's 0.038) — linear correlation missed nonlinear
     structure RF could exploit. Soil temperature (*strongest* correlation, r=0.742) fails for
     *both* methods — root-caused to a real train/test variance shift (test-period std roughly
     half of training-period std at Tower 4), not evidence the underlying relationship is weak.
     **PPFD/RN/WS show genuine validated skill over climatology; VPD is a wash; TS/soil moisture
     lose.** Extrapolation check (reusing S-01's `dissimilarity_index()`): **100% of 2041–2060
     SSP2-4.5 scenario days fall outside the training envelope at all 3 towers** — a serious
     caveat for any winning model, plausibly partly an artifact of the CMIP6 ensemble-mean's own
     smoothing. **Explicitly a preparation pass — `build_scenario_drivers.py`/`S01_first_scenario.ipynb`
     untouched**, adopting any winning proxy is a deliberate separate follow-up. See D-69,
     `notebooks/07_scenario_analysis/preparation/S02_driver_reconstruction_feasibility.ipynb`.
   - **Next (in order, superseded 2026-08-06 by D-82/S-04 — see below): (1) 07 scenario analysis,
     extending S-01** — ~~SSP5-8.5, realization-level (not just ensemble-mean) spread~~ (DONE, D-82),
     a self-consistent mechanistic livestock-scenario construction, and (if time permits) the
     SPACSYS process-model route for the trend/level component. B-08 remains available separately
     for the point-forecast track but is not on this critical path. Deferred: coarser/cumulative
     eval; gap-filling-phase metrics backfill. Backlog: ERA5; chase 2024 held-out EC data. If S-02's
     PPFD/RN/WS candidates are pursued further, address the 100%-extrapolation caveat first (e.g.
     test against individual GCM/realization trajectories, not just the ensemble mean).
   - **D-70 (2026-07-14): S-03, driver-availability ablation — isolates scenario-mode
     feature-degradation cost from extrapolation cost (supervisor request, Prof. Paul Harris).**
     Distinct from U-03/D-63 (distribution shift on real anchors) and S-01/D-64 (real scenario
     pipeline but no ground truth at 2050) — S-03 holds test data real/historical (same 2018-2022
     anchors as B-10/D-65) and only changes the feature set. Model 1 = B-10's full-feature ensemble
     (not rerun, read from D-65's tables). Model 2 = same architecture, two variants on a 24-column
     degraded set (`RESAMPLED_COLS+DROPPED_COLS`, imported from `build_scenario_drivers.py`, not
     retyped — resolves the PPFD/RN ambiguity: both already in `RESAMPLED_COLS`, no real
     contradiction with S-02): **Variant A (removal)** drops the columns entirely; **Variant B
     (resample)** keeps them real in training but climatology-resamples the rollout-time values
     (pre-anchor-only history — a deliberate fix vs. S-01's own full-record call, necessary since
     this experiment evaluates on real historical anchors, not a genuinely blind future). Both
     column lists are independently customizable script parameters, per direct user request.
     **Genuinely surprising headline, reported plainly: neither degraded variant costs accuracy
     pooled across all 3 towers — both modestly beat Model 1.** `Ensemble_unweighted` pooled MASE:
     0.918 (Model 1) → 0.926 (Variant A) → **0.892 (Variant B)**; R²: −0.165 → −0.108 → **−0.089**.
     Plausible explanation: many of the 24 degraded columns were already low-SHAP-importance
     (I-01/I-02) — smoothing/dropping noisy low-signal inputs may reduce overfitting in a 365-day
     rollout more than it costs real signal. Secondary gap-filled-target metric disagrees for
     Variant A specifically (looks clearly worse there) — flagged as a likely circularity artifact
     (diverges further from the gap-filler's own feature space), not necessarily a real accuracy
     loss. Practical implication: this isolated driver-availability cost is small — the documented
     scenario risk (U-03/S-01) concentrates in extrapolation/SARIMAX, not in losing these 24 sensor
     channels. Chain figures generated and merged into `b10_b13_full_chains.csv`/`b10_chains` figure
     set per direct user request (180 new figures, 12 new variant-suffixed columns, verified
     additive). See D-70, `notebooks/07_scenario_analysis/s03_results.md`.
   - **D-70 addendum (2026-07-15): model-roster extension — TFT/TabPFN/DLinear/LSTM/TabICLv2 were
     missing from S-03, a real scope gap caught only after the user asked directly** ("does S-03
     have TabPFN and TabICL??"). S-03's original scope ("B-10's 4-model architecture") was true when
     written but never revisited once the project's standing roster grew to 11 models (B-13/D-66).
     New `s03_model_roster_extension.py` runs the same two variants against all 5 remaining models
     (TabPFN/TabICLv2 zero-shot per-tower/anchor mirroring `b16_foundation_models_v3.py`; TFT/
     DLinear/LSTM pooled hourly Track-B mirroring `b16_dl_models_v3.py`, exact existing recipes, no
     new HPO) — smoke-tested before the full 5-anchor sweep. **Result refines the original finding
     rather than confirming it uniformly**: on MASE, Variant B (resample) beats/ties Model 1 for 9 of
     11 models; Variant A (removal) is much more mixed on the extended roster (clear win only for
     SARIMAX/TabPFN/DLinear). **TFT is a genuine reversal** — R² gets measurably *worse* under
     Variant B (-0.363→-0.492), opposite every other model, flagged as an untested attention-
     sensitivity hypothesis for future interpretability work. Also caught and fixed a latent bug in
     `b10_b13_chain_plots.py` (ground-truth column selection silently broke for variant-suffixed
     TFT/DLinear/LSTM columns). Canonical files extended in place: `s03_summary*.csv` (1,080→1,980
     rows), `s03_chains.csv`/`b10_b13_full_chains.csv` (row counts unchanged, new columns added),
     `compile_s03_results.py` extended to all 11 models (verified bit-for-bit against the
     previously-published Model-1 table before adopting the new aggregation path). 495 total chain
     figures now in `results/figures/b10_chains/`. See `s03_results.md`'s "Addendum: model-roster
     extension" section.
   - **D-71 (2026-07-15): is chain-persistence a valid MASE baseline for a seasonal series?**
     User-raised concern: MASE's flat-persistence denominator (D-37) ignores seasonality entirely,
     so Hyndman & Koehler's own MASE convention would recommend a seasonal-naive baseline instead —
     this project already has one (`rr.doy_climatology()`) but it had only ever run for a single
     tower/anchor inside B-09's original smoke test (D-53). Extended to full coverage (3 towers × 5
     anchors) via new `b10_b13_climatology_baseline.py`, then reran `bin_metrics()` for all 11 B-10/
     B-13 models with climatology in place of persistence as MASE's denominator. **Result reverses
     the motivating hypothesis**: pooled, climatology is the *weaker* baseline (own MAE 43.79 vs.
     persistence's 37.50 against real `y_true`) — likely because FCH₄'s spike-dominated record
     (D-44b) makes a ±7-day day-of-year average, built from only a handful of real historical years
     per tower, a noisy estimate rather than a stable seasonal curve. Per-tower breakdown shows this
     isn't uniform: Tower 2 matches the original hypothesis (climatology genuinely harder), but
     Towers 4/9 show the reversal. **Practical implication: reinforces keeping persistence as the
     primary MASE denominator** (D-37) — not just for cross-table consistency, but because the
     available seasonal alternative isn't empirically more reliable given how sparse the real data
     is. Climatology-scaled MASE retained as a secondary comparison column, not a replacement. See
     `results/b10_b13_climatology_mase_table_all_towers.csv`/`_by_tower.csv`.
     **Follow-up same day**: user flagged a fairness gap — climatology was built from real
     `y_observed` while persistence's anchor value comes from `y_gapfilled`. Added a second,
     gap-filled-basis climatology variant (`b10_b13_climatology_gf_baseline.py`) for a fair
     comparison. Refined result: climatology-gf narrows the gap substantially (pooled MAE 40.74 vs.
     persistence's 37.50, vs. the original 43.79) and **reverses at Tower 2** (climatology-gf clearly
     wins there) — conclusion (keep persistence as primary) still stands pooled, but is tower-
     dependent, not uniform. `results/figures/b09_chains/` (165 figures) now show all 3 baselines.
   - **D-73 (2026-07-18): IMP-01 opens a revisited, more thorough gap-filling/imputation phase
     (`08_imputation_revisited/`), step 1 of a 5-step plan (viz → algorithms → UQ →
     distributional shift → masked-data predictor testing).** User-scoped: full feature space
     (every raw measured column per tower, not just FCH4 — broader than any prior gap-filling
     replication), a fresh methodology going forward without overwriting prior R-01–F-09b
     results, and standalone synthetic distributional shifts for step 4 (not a reuse of Phase
     07's CMIP6 machinery), deferred until step 4 is reached. New `src/features/
     tower_feature_space.py` assembles the full per-tower column set (69/67/67 cols at T2/T4/T9)
     via the existing D-18 spatial-alignment rule, tagged into 9 variable families. **IMP-01
     (`IMP01_missingness_landscape.ipynb`, all 3 towers): missingness is strongly block-structured
     (not MCAR)** — co-missingness clustering shows a shared EC/met/turbulence outage block, a
     separate footprint (EC_fetch) cluster, a separate catchment water-quality cluster, a
     separate tower-side soil-probe cluster, and always-complete livestock (0.0% missing every
     tower). **Gap lengths are strongly bimodal** (mostly 1-hour gaps, but EC_soil reaches
     ~2,500–2,800-day blackouts) — no single imputer suits both regimes. **Real seasonal
     (May–Sept elevated) and diurnal (EC_fetch dips daytime) missingness patterns** confirm
     MAR/MNAR structure, not MCAR. **Tower-2-specific finding: FCH4/CH4 break out of the main EC
     co-missingness block** (cluster with soil-probe/precipitation instead) unlike Towers 4/9 —
     a new, specific mechanism (independent CH4-analyzer failure history) consistent with Tower
     2's already-known structural differences (D-15/F-07/D-68). No `benchmarks.csv` rows (EDA/
     diagnostic step). See D-73, `notebooks/08_imputation_revisited/IMP01_results.md`.
   - **D-74 (2026-07-19): F-11, SAITS gap-filling — evaluated, tested, not adopted.** User asked
     first for the current best-recorded gap-filling result (answered: partial-pooled
     external-sourced RFm under full-period gap-CV, R² T2=0.574/T4=0.402/T9=0.418, D-35/D-49),
     then to plan/evaluate/implement SAITS (`pypots`) as a candidate replacement, under
     `notebooks/04_feature_engineering/F11_SAITS_Implementation.ipynb`. **Environment fix
     required first**: `pypots` wasn't installed; installing it surfaced an unrelated blocker
     (`pypots.imputation.__init__` eagerly imports `TimeLLM`, which needs `transformers` →
     `torchvision`, and the installed `torchvision` was ABI-mismatched against `torch`) — fixed
     by reinstalling `torchvision` to a matching cu128 build, no `torch` version change, also
     incidentally fixed the project's separately-broken `pytorch-lightning` import.
     **Methodology**: reused F08's exact `insert_calendar_gaps` held-out timestamps for a
     point-for-point-comparable evaluation, but trained **one pooled SAITS model** (vs RFm's 75
     per-scenario-per-rep retrains) on the **union** of all 25 held-out sets excluded from
     training — a deliberate, harder-for-SAITS compute-bounding choice, stated explicitly.
     Phase 1 smoke test (Tower 4/scenario `m`) was GO (35.9s, sane RMSE/MAE units). **Full pooled
     run, all 3 towers × 5 scenarios: SAITS loses at every tower by a wide margin** (median R²
     T2≈0.03, T4≈0.00, T9≈−0.02 vs. RFm's 0.574/0.402/0.418), reproduced closely on an
     independent rerun. **Diagnosis**: every row shows a large negative MBE (systematic
     under-prediction) — most likely FCH4's baseline sparsity (25–45% valid even before masking,
     worse than SAITS's dense-series design target) compounded by the union-mask's extra target
     sparsity, plus FCH4's spike-dominated right-skew (this project's recurring "MASE<1 alongside
     near-zero/negative R²" pattern, D-44b) pulling a symmetric-loss model toward the typical/low
     value rather than the flux tail. **Not adopted — RFm (D-35/D-49) remains the standing
     recommendation**, no change to `BEST_RESULTS.md` §1. A legitimate tested-and-rejected
     alternative (mirrors B-03a/F-09b's handling), not a technical failure — pipeline runs
     correctly and cheaply (~5 min for the full run). 15 rows tagged `F-11` in
     `results/benchmarks.csv`. See D-74, `notebooks/04_feature_engineering/
     F11_SAITS_Implementation.ipynb`, `F11_results.md`, `results/f11_summary.csv`.
   - **D-75 (2026-07-20): F-12, bidirectional (lead) soil lags for RFm — tested, not adopted.**
     User asked whether RFm's gap-filler already uses bidirectional context (it doesn't — its
     `swc_l*`/`ts_l*` lags are backward-only via `.shift(lag)`, unlike the upstream REddyProc met
     gap-filling's centered window or F-11's SAITS) and to devise an experiment testing it. Ran a
     controlled 3-arm ablation on only the swc/ts lag block (everything else identical to the
     champion config): **Arm A** baseline (backward-only, rerun fresh), **Arm B** bidir (backward
     + new forward lags via `.shift(-lag)`), **Arm C** leadonly (forward lags replacing backward,
     same feature count). **Operational note**: a first attempt at the full F-08 `N_REPS=5`
     protocol (225 fits, ~3.45h) was killed by the environment after ~2h20m with zero progress
     saved (`nbconvert --inplace` only writes the file once the whole run finishes) — rebuilt with
     `N_REPS=2` (90 fits) and a cell-by-cell `nbclient` driver that checkpoints the notebook to
     disk after every cell; this completed cleanly in ~88 min. Leakage checks (feature-purity +a
     permanent per-fit assertion that no held-out timestamp survives into its training partition)
     passed on all 90 fits. **Result: mixed/null.** Arm A reproduces the champion exactly
     (0.574/0.402/0.418). Arms B/C gain a marginal, noise-level +0.008 at T4 only, while both
     regress at T2 (−0.017 to −0.019) and T9 (−0.010 to −0.011) — no arm beats the champion on
     balance. Feature-importance diagnostic confirms leads aren't ignored by the RF (real,
     non-trivial importance mass in both directions) — the null result is genuine redundancy
     between forward/backward soil-lag information, not a "leads get ignored" artifact. **Not
     adopted** — RFm (D-35/D-49) remains the standing recommendation, no `BEST_RESULTS.md` change
     (one-line null-result note added instead, per the F-09b precedent). 45 rows tagged `F-12` in
     `results/benchmarks.csv`. See D-75, `notebooks/04_feature_engineering/
     F12_bidirectional_soil_lags_RFm.ipynb`, `F12_results.md`, `results/f12_summary.csv`.
   - **D-76 (2026-07-21): F-11 follow-up — testing SAITS's diagnosed failure modes finds real
     gains, still not adopted.** Direct continuation of D-74 in the same notebook: user asked to
     discuss why SAITS lost so badly, then to test all the diagnosed fix ideas. Five levers tested,
     staged cheapest/most-diagnostic first, all seeded (D-74's original run had drifted from
     unseeded `torch` init). **Most of the elapsed session time was infrastructure friction, not
     experiment design** — a stuck run was killed after 10+ hours with zero progress (likely a
     machine sleep/resume cycle corrupting the CUDA context, confirmed healthy via a fresh-process
     check afterward), a retry hit a genuine `CUDA error: unknown error` during
     `load_state_dict` (transient corruption from that kill, resolved by simply retrying fresh),
     and background-task tracking was silently dropped by session restarts multiple times
     (confirmed via direct Windows process inspection each time, not assumed). **Results**:
     per-scenario retraining (fixes union-mask sparsity) — confirmed directionally, small
     (flips T4/T9 positive, +0.02–0.05); solo-vs-pooled structure — genuinely noise-level, flipped
     winner between two independent full reruns; **spike-weighted loss (custom `SpikeWeightedMAE`,
     attacks FCH4's spike-dominated skew) — by far the largest lever, ~5–6x'd R² on its own,
     reproduced in both reruns regardless of structure**; bigger model (`n_layers=3, d_model=256`)
     — a further consistent real gain on top. **Full 5-scenario confirmation (solo +
     SpikeWeightedMAE + bigger model): T2 0.192, T4 0.225, T9 0.110** (vs. D-74's naive baseline
     0.02–0.03 and RFm's 0.574/0.402/0.418) — SAITS still loses everywhere but the gap narrowed
     substantially (T4's more than halved, -0.43→-0.18). Not uniform, reported honestly: T2's `l`
     (288h) scenario went negative (R²=-0.073, MBE=+36.3, the one case with a large *positive*
     bias). **Still not adopted** — RFm (D-35/D-49) remains standing, no `BEST_RESULTS.md` change.
     If revisited, priority is the loss objective and a real architecture search, not further
     sparsity/pooling work. 15 rows tagged `F-11-phase4` in `results/benchmarks.csv`. See D-76,
     `notebooks/04_feature_engineering/F11_SAITS_Implementation.ipynb`, `F11_results.md` §8,
     `results/f11_phase4_final_summary.csv`.
   - **D-77 (2026-07-22/23): `03c_gap_filling_revisited` — fully self-contained (zero `src/`
     imports) gap-filling reproduction notebook, and a real, adopted `mdc_gapfill` fix.** Built
     `temp_gap_filing_exploration.ipynb` from scratch (hourly data → EDA → FCO2 reconstruction →
     external sourcing → met/soil gap-filling → u*/GPP-Reco → management/livestock → full F-08/
     F-09a gap-CV harness), catching two real bugs along the way: a `classify()` substring-collision
     bug ("inorganic fertiliser" matched as "organic fertiliser," fixed by restoring production's
     guard clause) and an `N_REPS` mismatch (`BEST_RESULTS.md`'s numbers came from F-09a's
     `N_REPS=2` re-check, not F-08's original 5 — confirmed legitimate rep-count variance, not a
     bug, once matched). **Genuine fix found**: `mdc_gapfill()`'s flat 2h interpolation cutoff was
     too short for low-diurnal-structure drivers (soil moisture/temperature, TA, VPD, WS) —
     extended to 288h for those 5 variables only (`LONG_INTERP_VARS`). Validated end-to-end via the
     full FCH4 gap-CV harness: **R² improved at every tower, new standing champion** T2 0.574→
     **0.576**, T4 0.402→**0.404**, T9 0.418→**0.426**. Also added model-fit caching
     (`_model_cache/`, MD5-keyed) after discovering `RandomForestRegressor(n_jobs=-1)` is not
     bit-reproducible across separate process runs (fixed via `n_jobs=1` for the one uncached RF),
     dropping full-notebook rerun time from ~25-40 min to ~3.5 min. `BEST_RESULTS.md` §1 updated.
     See D-77, `notebooks/03c_gap_filling_revisited/temp_gap_filing_exploration.ipynb`.
   - **D-78 (2026-07-23/24): extended exploration on the D-77 base — UQ, six more models, lag/lead
     expansion — champion unchanged, all additive, all logged in a separate working copy.**
     Continuation of D-77 in `temp_gap_filing_exploration copy.ipynb`, per explicit user request
     (UQ; more models — BI-LSTM/TabPFN/TabICL/SAITS named explicitly; expanded lag/lead for
     covariates and the target), critically evaluated in plan mode first. User overrode two
     critical-review recommendations (keep SAITS and soil-lag re-expansion despite F-11/F-12
     precedent, since the feature base changed at D-77) and set the governing rule: strictly
     additive, nothing overwrites the champion, full 3-tower coverage throughout.
     **Phase A (UQ, validated)**: Area-of-Applicability dissimilarity index (replicated inline from
     `scenario_hybrid.py`), applied to the production gap-filled series — weak but real positive
     correlation with error (Pearson +0.11 to +0.16 at all 3 towers), after catching and fixing two
     real bugs (missing imputation before scaling, and validation-set leakage from unmasked
     training data).
     **Phase B (6 additional models, full 3-tower)** — none beat the champion outright: LightGBM
     (T4 0.410, +0.006) and TabICL (T4 0.423, +0.019) edge the champion at Tower 4 only; XGBoost
     (0.551/0.349/0.369) and TabPFN (0.459/0.401/0.402, and extremely slow — ~3.7h for 60 folds)
     lose everywhere; SAITS (0.358/0.293/0.285, F-11's own best config rebuilt on D-77's features)
     substantially improved vs. F-11's original numbers (was 0.192/0.225/0.110) but still loses
     everywhere; BI-LSTM (0.237/0.155/0.146, a custom self-supervised windowed bidirectional-LSTM
     imputer) is weakest of all six. Result-level caching added for every Phase B/C model after
     discovering `nbconvert --execute` reruns every cell on every invocation regardless of prior
     success — without it, each new phase silently re-paid the full cost of every earlier slow
     phase on every subsequent rerun.
     **Phase C (lag/lead expansion)** — both negative: soil-lag bidir/leadonly (F-12's arms,
     rebuilt on D-77's features) reproduce F-12's null result almost exactly (T2 0.561-0.564, T4
     0.410-0.412, T9 0.411-0.415); target (FCH4) lag/lead (`target_lag{1,24,168}`/
     `target_lead{1,24,168}`, genuinely new, never tested) regresses clearly
     (0.495/0.329/0.353) — required a new leakage-safety pattern (mask the target at the current
     fold's own held-out timestamps *first*, derive lag/lead from that masked series, runtime
     assertion on every fold) since a held-out point's neighbour can itself be held out in the
     same contiguous gap. **Outcome: not adopted, no `BEST_RESULTS.md` change beyond D-77's own
     fix** — RFm (D-77) remains standing at every tower. See D-78,
     `notebooks/03c_gap_filling_revisited/temp_gap_filing_exploration copy.ipynb`,
     `notebooks/03c_gap_filling_revisited/_data/{model_comparison,soil_lag_results,
     target_laglead_results}.csv`.
   - **D-79 (2026-08-02): a third, parallel notebook (`temp_gap_filling_pipeline.ipynb`) —
     literature-correct MDS fix, HyperImpute, all-16-model production fill, and a TICA/UMAP
     feedback-feature line (D5-D8) that found TabICL-solo beats the RF champion at 2 of 3
     towers.** Reproduces the same 0.576/0.404/0.426 champion. **MDS fix** (ported from a separate
     audit notebook, `temp_mds.ipynb`): 3 real bugs fixed (literature-correct 3-case hierarchy),
     confirmed Towers 2/4/9 are the literal same sites as Zhu et al. (2023a)'s ROTH_HS/PP/HSC
     farmlets, and found this project's `r2_score` diverges sharply from Zhu et al.'s own
     OLS/Pearson-r² convention specifically for MDS (+0.30 gap, old; +0.12, fixed) but not for any
     RF/TabICL/MICE model (≤0.05) — both MDS versions now land close to Zhu et al.'s own published
     ~0.03-0.05 figure once measured their way. **HyperImpute** (AutoML-per-column imputer, same
     features as MICE): R²=0.509/0.336/0.354, a 0.25-0.43 jump over MICE on identical inputs.
     **Production fill generalized to all 16 models** this notebook has evaluated (was
     champion-only), one real CUDA-OOM bug fixed along the way (TabICL prediction batching).
     **D5-D8: three straight negative "feed derived signal back into the model" results, then one
     real positive.** D5: consensus TICA/UMAP/t-SNE feature *selection* is clean and stable, but
     downstream environmental-KNN features are a wash; separately found **TabICL-solo beats
     TabICL-pooled at every tower** (its fixed 10,000-row context cap makes pooling actively
     dilutive, unlike RF) — adopted as the standing TabICL default going forward. D6 (supervisor's
     idea): TICA components + native model-uncertainty width fed back as features — TICA is a
     wash, uncertainty is **actively harmful**, severely so for TabICL at 2 of 3 towers (-0.35 to
     -0.40 R²). D7 (TabICL-only, RF treated as exhausted): dropping D5's 2 least-reliable
     features / swapping in TICA components — both flat; ensembling already-computed RF+TabICL+
     HyperImpute predictions helps only at Tower 4 (+0.017). D8 (TabICL-only): native
     hyperparameter sweep is a clean null; **row-cap bagging (k independent random-subsample fits,
     averaged) gives a real, mechanistically-explained gain specific to Tower 4** (+0.012-0.013 R²,
     plateaus by k=5-8) — the one tower whose domain most exceeds the context cap. **Headline
     result: TabICL-solo on the plain champion `FEATURES` beats RFm at T2 (0.676 vs. 0.576) and T4
     (0.428 vs. 0.404), ties at T9** — the first result in either `03c_gap_filling_revisited`
     notebook to beat RFm at more than one tower. Flagged as a validated **benchmark**, not yet
     production-adopted. D-107 subsequently added exact-config production-refit/chart tooling:
     native-hourly latest-six-month chains for all three towers with TabICL's raw q05-q95 bands,
     persisted point/quantile output, and report copies. The bands are uncalibrated, so this closes
     figure/raw-output provenance but does not by itself promote TabICL to the adopted production
     gap-filler or establish interval coverage.
     **Operational note:** `_model_cache/` had silently grown to 166 GB over the project's
     iterative history (confirmed pure, fully-regenerable RF joblib cache, zero unique data) —
     deleted and rebuilt via a full top-to-bottom rerun (314 cells, cold cache, ~8h36m, zero
     errors) to a leaner 77 GB; added automatic stale-entry pruning (`prune_stale_cache()`, safe
     only after a genuine full rerun) so it won't silently balloon again. **Outcome: strictly
     additive, RFm remains production-adopted at every tower** — `BEST_RESULTS.md` §1 now flags
     TabICL-solo as benchmark-best at T2/T4. See D-79,
     `notebooks/03c_gap_filling_revisited/temp_gap_filling_pipeline.ipynb`,
     `notebooks/03c_gap_filling_revisited/temp_mds.ipynb`,
     `notebooks/03c_gap_filling_revisited/summary.md` §12-18.
   - **D-82 (2026-08-06): S-04 analyzed — a completed-but-undocumented realization-level/SSP5-8.5
     scenario trajectory (built 2026-07-15/16) closes S-01's two queued extensions.** A
     repo-familiarization pass found `s04_trajectory_2050.py`/`s04_daily_top3_2050.py` had already
     run to completion (commits `777cf89`/`ea6530f`, both naming "S04") — 234,000-row primary-hybrid
     sweep + 28,080-row B-10 diagnostic-benchmark sweep + 4,680-row AOA table + 5.1M-row daily
     top-3-model chains + 180 chain figures, all complete, both SSPs — but neither commit updated
     `DECISIONS.md`/`BEST_RESULTS.md`, and `CONTEXT.md` still listed "extend S-01: SSP5-8.5,
     realization-level spread" as outstanding. New read-only `s04_analysis.py` (no new model
     fitting) summarizes the existing output. **S-01's central finding holds and is reinforced
     across the full 26-year × both-SSP trajectory**: 1×→3× livestock, hybrid response
     +38.6%/+156.4%/+120.3% (T2/T4/T9) vs. the B-10 diagnostic ensemble's own
     +20.4%/+76.6%/+62.0% — roughly 2× the diagnostic ensemble's response at T4/T9, matching
     S-01/U-03's original single-snapshot comparison. **Genuinely new finding**: the
     transient/realization-level AOA check flags materially more days than S-01's smoothed
     ensemble-mean snapshot did, at every multiplier including the unchanged 1× baseline (9–15%
     here vs. 0% in S-01 at 1×/2×) — scenario-construction method (smoothed composite vs. real
     transient weather) measurably changes how out-of-distribution a scenario looks, independent of
     the livestock question; S-01 and S-04's AOA numbers are not directly comparable. SSP2-4.5 vs
     SSP5-8.5 divergence is real and grows toward 2050 as expected but stays under 1% of the mean
     throughout; realization-level spread itself is small (1–5% of the mean) and narrows in
     relative terms as livestock stress increases. **Remaining open (unchanged from S-01): a
     self-consistent mechanistic livestock-scenario construction, and SPACSYS for the trend/level
     component if time permits before the 1 Sept deadline.** `BEST_RESULTS.md` §6 updated (S-04 now
     current, S-01 kept as architecture reference). See D-82, D-64,
     `notebooks/07_scenario_analysis/s04_trajectory_2050.py`, `s04_daily_top3_2050.py`,
     `s04_analysis.py`, `s04_results.md`.
   - **D-83 (2026-08-08): S-03 (driver-availability ablation) brought up to speed with D-80's
     climatology-MASE convention and D-79's TabICL-sourced gap-filling, and its model-roster
     addendum finally wired into the notebook itself (previously only ever run as a standalone
     script).** `s03_driver_availability_ablation.py`/`s03_model_roster_extension.py` gained a
     `climatology_baseline()` MASE denominator (D-80) and a `daily_csv` parameter for the
     TabICL-sourced sibling file. Model 1 was recomputed (not left on the old persistence-scored/
     RF-sourced table) wherever TabICL data makes that possible, so Model 1 and Variant A/B stay
     apples-to-apples; TFT/DLinear/LSTM have no TabICL-sourced hourly data anywhere in this project
     and were rescored from their existing chains, not retrained. **The original driver-availability
     finding replicates qualitatively unchanged under both switches** (degrading drivers still
     doesn't cost material accuracy — Variant B still beats/ties Model 1 for most models, TFT is
     still the one reversal). **Unplanned finding, more consequential than the two requested
     changes**: switching the tree/SARIMAX/ensemble family's *training target* (not just
     TabPFN/TabICLv2's context, which D-80 already found only mildly hurt by this) to TabICL-sourced
     `y_gapfilled` causes a much larger absolute MASE increase (~1.3-2.5x) at every tower — traced to
     TabICL's gap-filled series sitting at a substantially different mean level than RF's
     (confirmed by direct comparison in the notebook). Independently confirms and sharpens D-80's own
     conclusion that D-79's better gap-filling does not transfer to forecasting — RF-sourced
     gap-filling remains the right target source for every model family here. See D-83, D-80, D-79,
     D-70, `notebooks/07_scenario_analysis/s03_climatology_tabicl_update.py`,
     `results/s03_table_all_towers_climatology_tabicl.csv`, `s03_results.md`'s second addendum.
   - **D-84 (2026-08-08): S-05 — TabICLv2 + S-03's Variant A + F-10's species split, run as a
     10-year transient CMIP6 trajectory with independent per-species livestock multipliers.**
     Follow-up to D-83: since TabICLv2 is one-shot (not a recursive rollout), its S-03 horizon
     extends past 365 days without compounding-error risk, and Variant A's 10-column feature set
     (TA/SWIN/PRECIP/DOY/season/livestock) is almost exactly what `data/Simulated Climate Data/`
     (S-04's own CMIP6 source) already supplies. Scoped like S-04 (3 towers x 2 SSPs x 5 GCMs x 10
     realizations/GCM, user-confirmed after empirically timing a single call and presenting 4
     scope options) x **27 independent per-species livestock multiplier combos** (cattle/sheep/
     lamb each scaled separately, user's explicit choice over a cheaper shared-multiplier option)
     = 8,100 calls, **2.54h actual runtime**. **Headline: cattle dominates the FCH4 response far
     beyond its own LSU-weight share** (tripling cattle alone ~triples predicted FCH4 at T4/T9;
     sheep/lamb stay under 25% even at 3x) — F-10's species-split feature earns its place in a
     scenario context, not just on real historical anchors. Joint 3-species scaling is close to
     additive at T4 (-0.2% synergy), a real +8.8% super-additive effect at T9. **A genuine
     methodological correction happened mid-analysis, not swept under the rug**: a first-pass
     "realization spread" metric mirroring S-04's own pooling convention gave 32-69% of the mean
     (10x too high) — traced directly to year-to-year weather variability within a decade being
     conflated with realization/GCM choice; isolating the latter alone gives 2.4-6.6%, consistent
     with S-04. AOA flagged-% is high (62-68%, vs S-04's 9-15%) but flat over the horizon — second
     confirmation that AOA's absolute level depends on feature-space breadth (S-05's 13-dim space
     dilutes less than S-04's ~40+). No change to any standing recommendation. See D-84, D-83, D-70,
     D-67, `notebooks/07_scenario_analysis/s05_trajectory_10yr.py`, `s05_analysis.py`,
     `s05_results.md`, `S05_species_trajectory.ipynb`.
   - **D-85 (2026-08-08): S-05 extended to 2050 + full daily chains saved for every scenario
     point.** Same-session follow-up to D-84: (1) horizon extended from a fixed 10 years to 2050
     (matching S-04's endpoint; T4/T9 now 27 years, T2 31 years); (2) full daily chains saved for
     all 8,100 calls, not just annual_mean, folded into the same run since the horizon extension
     (not the daily save) dominates the new cost — measured directly (a single 27-year call: 4.07s
     vs. ~1.2-1.3s for the original 10-year call) before committing to the ~9h estimate. **Actual
     runtime 5.44h**, 0 failed calls, 83,767,500 daily rows written incrementally to Parquet
     (1.25GB). Reproducibility spot-checked against the original run's overlapping year: 9.981 vs.
     9.974 — 0.07% apart, ordinary GPU variance. **Every D-84 finding replicates, several more
     clearly**: cattle-dominance holds/strengthens (T4 3x-alone +205.6%→+214.5%); joint-vs-additive
     holds; realization/GCM spread stays in the same small isolated range (2.4-6.6%→2.5-7.6%); AOA
     flatness now confirmed across the full 27-31-year horizon, not just 10 years. **New pattern
     the 10-year window was too short to show**: SSP2-4.5 vs SSP5-8.5 divergence now visibly grows
     from early to late window, matching S-04's own "widens toward end of century" finding. No
     change to any standing recommendation — both the 10-year and 2050-horizon outputs are kept on
     disk, neither overwrites the other. See D-85, D-84,
     `notebooks/07_scenario_analysis/s05_trajectory_2050.py`, `s05_analysis_2050.py`,
     `s05_results.md`'s "Update: extended to 2050" section.
   - **D-86 (2026-08-08): S-05 extended to farming-practice scenarios — grazing timing and
     fertilizer schedule, two separate baseline-livestock experiments.** User named both levers
     explicitly. Priors stated before building anything (per F-01/F-04/F-05's "redundant on the
     rich base" finding): both smaller than livestock's cattle effect, but grazing tied directly to
     livestock presence (expect real effect) vs. fertilizer's weaker CH4-specific mechanistic link
     (expect muted). Both DERIVED features (not directly scalable) reconstructed by reusing their
     real-data construction functions unchanged — grazing phase-shifts the real day-of-year species
     climatology at season edges, re-deriving `fx_grazing_active`/`fx_days_since_grazing` via
     `days_since_grazing()`; fertilizer builds a per-tower "typical year" event template (T4: ~8.25
     events/yr, DOY 82-234, mean 127 kg/ha) scaled by rate/frequency, run through
     `recency_series()` (tau=14 decay). 900 calls/axis (3 towers x 2 SSPs x 5 GCMs x 10
     realizations x 3 levels), 2050 horizon, smoke-tested first. **Actual runtime ~51 min/axis
     (~1.7h combined)**, 0 failed calls. **Both priors confirmed cleanly**: grazing shows a real,
     monotonic effect at every tower (T4 +18.9% at +4wk, T9 +17.2%); fertilizer shows a small,
     sign-inconsistent effect across towers (T2/T9 negative, T4 positive, all under 5%) — extends
     F-01/F-04/F-05's finding from real-data feature importance to scenario response. AOA
     side-finding: grazing's flagged-% grows monotonically with shift level (T4: 76%→88%),
     fertilizer's doesn't move as cleanly. No change to any standing recommendation. See D-86, D-85,
     D-84, `notebooks/07_scenario_analysis/s05_practices_trajectory.py`, `s05_practices_analysis.py`,
     `s05_results.md`'s "Second update" section. **Addendum, same day**: daily-resolution figures
     built for all 3 scenario families (livestock free via the existing full parquet; grazing/
     fertilizer via a small rerun, `s05_practices_daily_chains_subset.py`), naming made consistent
     across all three (`s05_{axis}_daily_{view}_{ssp}.png`), extended to both SSPs (18 figures
     total) — see `s05_livestock_daily_chains_plots.py`, `s05_practices_daily_chains_plots.py`.
   - **D-87 (2026-08-08): Streamlit digital-shadow interface (Objective 6) dropped from scope,
     user-directed**, given the 1 Sept deadline. Scoped narrowly to the INTERFACE only — the
     underlying digital-shadow substance (scenario simulation, management/climate levers, UQ once
     attached) is exactly what S-01/S-03/S-04/S-05 already deliver; none of that work is affected.
     Objective 6 marked dropped in this file's Objectives section (struck through, not deleted,
     matching this project's own convention of recording scope changes rather than erasing them).
     See D-87.
   - ⚠ **Held-out 2024 still empty** (2024 FCH₄ = 0% valid all towers) — final held-out benchmark blocked until 2024 EC fluxes are downloaded; test on 2022–2023 meanwhile.
2. **Use partial pooling (D-30) as the multi-tower default** — pooled global model + tower-indicator (or continuous tower descriptors); rescues data-poor towers while protecting data-rich ones.
3. **Tower 2 split redesign** (D-15/D-19) — also lets Tower 2 be a proper pooled/test member.
4. **(Optional) Operational FCO₂ variant** — re-run 03b with `FC_recon` everywhere (strict, leak-free).
5. **ERA5 driver_era** (D-14); **SVM C-search** (R-03); validate Tower-9 pooled-density gain on 2024 once downloaded.

---
_Last updated: 2026-08-20 (D-108: B18 integration into Phase 07 — I-03b/U-08/S-03b-d/U-05b-07b/S-06b.
D-106 (B18) explicitly did not propagate into I-03/U-04/S-03/Phase-07 scenario work; this closes
that gap via a 6-phase additive plan (user-directed). I-03b/U-08 reconfirm the standing thesis
(fx_lsu_dens still #1, UQ converges to ~0.89-0.90 PICP again) under B18's actual champion. The real
work was S-03b/c/d (the gate): B18's own feature set can't run in scenario mode, and three rounds
of real-anchor backtesting were needed to find a config that's both validated AND deployable —
**locked-in: solo per-tower `Direct_TabICLv2` regression on FX_A_SPECIES + a trend feature
(+2.79% MASE vs. the old TS-wrapper, extrapolation-to-2050 safety confirmed directly, not
assumed)**. U-05b-07b rebuilt scenario UQ for this config (AOA/CQR findings replicate almost
exactly). Phase 6 replicated S-06's full core grid (livestock+grazing+fertilizer, bias-corrected
drivers) on the new architecture in ~2.2h (vs. the original ~6-9h) — every S-06 headline finding
reproduces cleanly (T9 still exceeds the regulatory stocking ceiling, grazing +4wk still ~+20%,
fertilizer still null, cattle dominance reconfirmed). A real bug (D-104's lit_ceil correction not
carried over into the new pipeline) was caught by a direct user question and fixed via the same
scoped rerun-and-merge pattern as the original fix. 66 new figures generated across all 6 phases,
matching every established format (I-03/U-04/U-05/U-06/U-07/S-05/S-06 conventions). Not yet done:
`reg_cap` for S-06b, U-05b's D-92-style attach-to-outputs step, report text update. See D-108,
`BEST_RESULTS.md` §4/5/6, `S03b_results.md`, `I03b_results.md`, `U08_results.md`,
`U06b_U07b_results.md`.)_

_Previously updated: 2026-08-19 (D-106: B17/B18 TabPFN-only forecasting improvement programme completed,
all 3 towers × 5 anchors. The direct pooled TabPFN architecture supplies most of the B16→B18 gain
(MASE 0.7123→0.6958); recency, conservative p95 event correction, and equal-weight averaging lower
the exploratory numerical best to 0.6908 (best single/gated=0.6924). Total B16-style-v2→B18 change:
−0.0215 MASE/−3.01%. Block-wise validation does not establish the ensemble's final increment as
stable, and spike magnitude remains the limiting regime. Benchmark updated; downstream I-03/U-04/
Phase-07 integrations remain on the prior architecture. See D-106, `BEST_RESULTS.md`, and
`report/Outlines/B18_forecasting_experiment_results.md`.)_

_Previously updated: 2026-08-19 (D-105: new fertiliser scenario level `reg_cap` — rate scaled so the true,
area-weighted typical-year N loading hits exactly the UK NVZ N-max for grassland (300 kg N/ha/yr,
gov.uk), added to both S-05 and S-06, all 3 towers. A regulatory-grounding check surfaced a real
double-counting bug (T4/T9 are each two independently-fertilised sub-fields; `FERTN_TEMPLATE`'s
pooled n_events x mean_rate overstates the true catchment-average loading ~2x there) — but the bug
lived only in chat-only regulatory-comparison arithmetic, never in production/scenario code, so no
rerun was needed to fix it; the 3 existing fertiliser levels are untouched. Result at the new level:
negligible effect even at the regulatory ceiling (T2 -0.9%/-0.7%, T4 +1.5%/+1.7%, T9 +0.1%/+0.1%,
S-05/S-06), reinforcing the standing "fertiliser is not a meaningful CH4 lever" finding. See D-105,
`s05_results.md`'s addendum section for full detail.)_

_Previously updated: 2026-08-18 (D-102: I-03 — interpretability recalibrated for the then-current
B16-style forecasting champion, TabPFN+species, closing the same "predates the champion" gap U-04
already closed for UQ.
I-02 (D-61) predates TabICLv2/F-10's species features by days; I-03 reruns I-02's own TabPFN
permutation-importance method, unchanged, on the then-current champion's BASE+species config
(`forecast_daily_v3.csv`, 52 `fx_` columns), full 3-tower × 5-anchor sweep, ~22 minutes, zero
training. **`fx_lsu_dens` dominance confirmed on the real champion** (#1 of 52, mean importance
1.1456). **New finding I-02 could not make**: `fx_cattle_dens` is a clear #2 (0.8043) while
`fx_sheep_dens`/`fx_lamb_dens` rank near the bottom — F-10's species-split gain is concentrated
entirely in cattle, independently corroborating S-05's scenario cattle-dominance finding via a
completely different method. **Tower 2 has zero livestock features in its top 10** — a fourth
independent confirmation of its livestock-blindness (after U-03, S-01, S05-T2/D-95). TabICLv2 not
covered this pass, flagged as a follow-up. See "Current status" bullets above, D-102,
`notebooks/06_interpretability_uq/i03_champion_interpretability.py`, `I03_results.md`.)_

_Previously updated: 2026-08-17 (D-100: delta-method bias correction across all of Phase 07 (S-01/S-04/
S-05) — S-01's own already-accepted baseline-reconstruction gap (Finding 1, 9-20%) corrected via a
standard climate-impact-modelling technique (anchor level to real historical mean, trust the model
for the shape of the change), applied additively (raw + corrected reported side by side, nothing
overwritten). **S-01/S-04 shift modestly** (T2 +33.8%→+40.7%, T4 +138.2%→+135.4%, T9
+104.7%→+114.4% at S-01; S-04 pooled trajectory shifts similarly), as expected from an already-small
gap. **S-05 (TabICLv2 + `FX_A_SPECIES`, never checked before this task, a structurally different
zero-shot model) turned out far worse: 40-80% underprediction of the real historical mean at every
tower** (checked against both `y_gapfilled` and `y_observed`, consistent under both) — 2-4x larger
than S-01/S-04's gap, always an undershoot. Correcting it **roughly halves the cattle-dominance
headline** (T4 3×-alone +213.9%→+101.8%, T9 +187.3%→+110.4%) — the dominance finding's direction is
completely unaffected, but its exact magnitude is now genuinely uncertain pending further
investigation, not resolved by this correction alone. **Directly affects prior citations**
(BEST_RESULTS.md's "+214.5%", D-85, the report's Chapter 7) — updated where touched this pass,
explicitly flagged (not silently left stale) everywhere else. See "Current status" bullets above,
D-100, `notebooks/07_scenario_analysis/d100_bias_correction_s01_s04.py`, `d100_bias_check_s05.py`.)_

_Previously updated: 2026-08-10 (D-95: S05-T2 — does pooling rescue Tower 2's muted livestock-scenario
response? No — exactly 0.0pp difference, both TabICLv2 and TabPFN. Direct follow-up to T2's
muted-response finding (cattle 3× only +1.8-2.3% at T2 vs. +186-215% at T4/T9) — tested whether
pooling T2's context with T4/T9's real livestock-rich history (the mechanism that rescued Tower 9
in gap-filling, F-02/F-03) transfers a learned cattle sensitivity in. Confirmed API mechanics first
(every context `item_id` must appear in `future_df` or the call raises a `KeyError`). **Result:
exactly 0.0 percentage points of difference for both models, every combo/SSP** — not just small,
an exact decimal match. Mechanistic read: `item_id`-based pooling shares context rows within one
batched call, not fitted parameters — a zero-shot forecaster's output for one series stays driven
almost entirely by that series' own history regardless of what else is in the batch. Retroactively
explains why D-94's own pooling gain was so small (likely batching noise, not real transfer) and
closes the "maybe pooling fixes it" question empirically — **T2's muted response is a genuine
model-extrapolation limit, not a fixable data gap.** See "Current status" bullets above, D-95,
`notebooks/07_scenario_analysis/s05_t2_pooled_test.py`.)_

_Previously updated: 2026-08-10 (D-94: B-16 round 4 — pooled vs. solo for TabPFN/TabICLv2 on
forecasting specifically. Direct follow-up question after D-93 — pooling was adopted for Track A from
gap-filling's F-02/F-03 finding, Track B stayed solo based on a gap-filling-only precedent (D-79)
that used a different API than the champion's own architecture, never re-tested on forecasting.
Verified before building: both `TabICLForecaster`/`TabPFNTSPipeline` natively support an `item_id`
column for genuine multi-series panel input. Solo baseline reused from U-04's `u04_chains.csv`, not
rerun. Two real bugs caught before/during the sweep: pooled/batched calls don't tolerate the same
partial NaN solo calls handle fine (fixed with mean-imputation); Tower 9's two earliest anchors
have empty pre-anchor history, silently NaN-poisoning the climatology baseline (fixed with a
skip-and-log guard). **Result: TabICLv2 pooled ≈ solo (noise, direction matches D-79 but far
smaller). TabPFN pooled beats solo at all 3 towers** (0.7138 vs. 0.7166 overall) — small but real
and consistent, never tested by D-79 at all (TabPFN wasn't part of that comparison). Not large
enough to force an immediate champion switch given the deadline, but a concrete, cheap improvement
lead. See "Current status" bullets above, D-94,
`notebooks/05_benchmarking/B16_pooled_vs_solo.ipynb`.)_

_Previously updated: 2026-08-10 (D-93: B-16 round 2 — TICA embeddings + static AR-lag features for
TabICLv2, both negative, combining them actively worse, `BASE+species` remains champion. User
revisited forecasting focused on Track B specifically (single consolidated notebook, requested for
easy rerun/tweaking). Worked through two design forks before building: static vs. dynamic AR-lag
features given Track B's single-shot (non-recursive) architecture — user caught the static
design's real staleness weakness directly, chose to proceed anyway given the ~30h+ cost of a
genuine day-by-day recursive alternative; and confirmed S-05 has the identical single-shot
architecture with zero AR-lag columns, so a future climatology-anchored alternative was flagged
but not built this round. **Result: TICA (MASE=0.7348) and static AR (0.7358) both land within
noise of baseline (0.7353)** — TICA replicates D-79's own "wash" finding from gap-filling in a
different task; **combining both is clearly worse (0.7603), consistent across all 3 towers** —
matches the project's recurring "stacking too many feature families hurts" pattern. Two real bugs
caught by the notebook's own smoke test before the full sweep: a zero-variance-history-column bug,
and a non-positive-definite TICA covariance matrix (fixed with ridge regularization). See "Current
status" bullets above, D-93, `notebooks/05_benchmarking/B16_tica_static_ar_features.ipynb`.)_

_Previously updated: 2026-08-10 (D-92: S-05 + UQ — U-06/U-07's CQR calibrations attached to S-05's
ACTUAL scenario trajectories, closing the last standing gap between scenario analysis and UQ. Two new
scripts, zero new calibration fitting: `s05_uq_daily_chains_subset.py` reruns S-05's existing
18-call/axis subset (54 calls, 2050 horizon) requesting quantiles instead of a point prediction —
confirmed free (4.3s/call, same as point-only), ~2.5 min total, 0 failures. `s05_uq_cqr_apply.py`
attaches U-05's FX_A_SPECIES-architecture margins via a (tower, lead-bin[, LSU-tier]) lookup pooled
across U-05's 5 anchors. Two explicit extrapolation assumptions: lead times beyond 365 days hold
the widest bin's margin flat (likely understates true uncertainty at year 20+ — read far-horizon
bands as a floor); grazing/fertilizer axes reuse the livestock-architecture margins. **Result:
works cleanly, >99% coverage at T4/T9** (T2 0%, pre-established degeneracy); one genuine thin spot
(T4 days-1-7 × mid-LSU-tier, zero calibration samples). Verified: zero interval inversions, CQR
correctly tightens the model's own raw quantile spread on average at T4 (raw MPIW 572.6 vs. U-06
514.6 vs. U-07 523.4). **This closes the U-04→U-07 UQ arc's last open caveat.** See "Current
status" bullets above, D-92, `notebooks/07_scenario_analysis/s05_uq_daily_chains_subset.py`,
`s05_uq_cqr_apply.py`, `s05_uq_cqr_plots.py`, `s05_results.md`.)_

_Previously updated: 2026-08-10 (D-91: U-07 — livestock-density-stratified CQR, thinner margins
where livestock presence is smaller. Direct user question on U-06's output, checked empirically
before building: "can't the margin be thinner where livestock presence is smaller?" Signal much
stronger than U-05's AOA-distance check: corr(|residual|, fx_lsu_dens)=0.43-0.45 (vs. AOA's weak
0.09-0.15), residuals ~3.2x larger on above-median-LSU days. fx_cattle_dens correlates almost
identically (0.427, consistent with S-05's cattle-dominance finding). Same CQR machinery as U-06,
only the bin key changes to lead-time × LSU-tertile — conformal_margins_by_bin() needed zero code
changes (5th reuse across U-02/U-04–U-07). **Result: low-LSU intervals are 29-46% the width of
high-LSU intervals** (TabPFN: 84.3 vs. 293.7 nmol) — a genuine win-win, not a trade-off: verified
spike days (3.2× higher fx_lsu_dens) still get their own dedicated, appropriately-wide calibration
in the "high" tier. Should be the standing UQ method going forward, layered on U-06's CQR. See
D-91, D-90, `notebooks/06_interpretability_uq/u07_lsu_stratified_cqr.py`, `U07_results.md`.)_

_Previously updated: 2026-08-10 (D-90: U-06 — Conformalized Quantile Regression (CQR) fixes the
spike-coverage failure U-04/U-05's own fancharts revealed visually. User observation, checked
directly: "a lot of spikes are still beyond the interval." Confirmed and quantified — overall
PICP≈0.89 looked fine, but 75% of top-10%-magnitude days fell entirely outside the interval, vs.
3.3% for the bottom 90% (split-conformal's flat symmetric margin only guarantees average
coverage). A pre-build check found raw q95 already sits close to/exceeds actual spike values
while the median massively undershoots — motivating CQR (nonconformity =
`max(q05-y_true, y_true-q95)`, interval = `[q05-margin, q95+margin]`) over alternatives.
`conformal_margins_by_bin()` reused unchanged — no new model calls, pure recalibration. A bug
caught before reporting (T2 showing 0.0 instead of NaN, same all-NaN aggregation footgun U-02
already documents, fixed by replicating its guard). **Result: spike coverage roughly triples**
(TabICLv2: 24.3%→79.7% U-04, 22.1%→79.3% U-05; TabPFN: 24.3%→57.2%), at the honest cost of
normal-day coverage dropping to ~83-88% and spike intervals roughly doubling. **CQR should replace
the symmetric-margin approach as the standing UQ method going forward** — not yet applied to
S-05's actual scenario trajectories (flagged as a small next step). See "Current status" bullets
above, D-90, D-88, D-89, `notebooks/06_interpretability_uq/u06_cqr_recalibration.py`,
`U06_results.md`.)_

_Previously updated: 2026-08-10 (D-89: U-05 — scenario-analysis UQ ("Option B"), built on U-04's method
but on S-05's own architecture (FX_A_SPECIES, 13 cols — not U-04's BASE+species, 52 cols; a
different feature space is a genuinely different model). TabICLv2 zero-shot, 5 real anchors × 3
towers, same leave-one-anchor-out conformal machinery as U-02/U-04 — 9-second runtime. A real
leakage bug caught by smoke-testing (AOA training set was unrestricted, giving every test point a
literal distance-to-self of 0 — fixed to pre-anchor-only, recomputed per anchor). **Step 3 resolved
the plan's design question empirically**: |residual| vs. AOA-flagged status shows weak raw
correlation (r=0.146) but a real, substantial categorical gap (out-of-AOA residuals ~48% larger,
pooled) — landed on a two-tier margin interpolated continuously by each point's own
`aoa_flagged_pct`, a genuine third option between the original plan's Level 1/Level 2. Applied to
S-05's existing livestock/grazing/fertilizer outputs with zero new model calls (a second bug — a
flat per-tower margin not actually using Step 3's finding — caught and fixed before finalizing).
**Calibration converges to ~0.88-0.89 PICP at T4/T9**, matching U-02/U-04 a third time; T2 stays
uncalibratable (third confirmation). **Interval is genuinely wide, stated plainly**: ±94-100% of
mean in-AOA, ±139-140% out-of-AOA — consistent with U-01's original "large aleatoric uncertainty"
finding (D-40). 15 calibration fancharts + 1 applied-trajectory figure, kept visibly separate from
realization spread per D-85's own lesson. No change to any standing recommendation. See "Current
status" bullets above, D-89, D-88, `notebooks/06_interpretability_uq/u05_scenario_uq.py`,
`U05_results.md`.)_

_Previously updated: 2026-08-10 (D-88: U-04 — UQ recalibrated for the current forecasting champion
(TabPFN+species, TabICLv2), closing a gap U-02 left behind. U-02 (2026-07-06) predates TabICLv2
(D-66, 3 days later) and F-10's species features (D-67, 4 days later) — its "TabPFN" interval is
calibrated for a superseded feature config; TabICLv2 has never had UQ at all. User-confirmed scope
(Option A of a two-part plan, B being scenario-analysis UQ next): champion-focused (TabPFN+
TabICLv2, both zero-shot with native quantile support) over the full 11-model roster, since the
other 6 U-02 models' feature config never changed. New script `u04_champion_uq.py` reuses U-02's
`evaluate_stage()` unmodified on `forecast_daily_v3.csv`'s `BASE+species` config — 25-second
runtime, no retraining. **Calibration converges to ~0.89-0.90 PICP at T4/T9**, matching U-02's own
finding on the new config; T2 still can't support calibration (confirmed independent of the
feature-set change). **Species enrichment improved point accuracy without materially changing
calibration quality** (TabPFN conformal MPIW/pinball essentially unchanged old-vs-new) — sensible,
not assumed in advance. See "Current status" bullets above, D-88, D-62, D-66, D-67,
`notebooks/06_interpretability_uq/u04_champion_uq.py`, `U04_results.md`.)_

_Previously updated: 2026-08-08 (D-87: Streamlit digital-shadow interface (Objective 6) dropped from
scope, user-directed, given the 1 Sept deadline. Scoped narrowly to the INTERFACE only — the
underlying digital-shadow substance (scenario simulation, management/climate levers, UQ once
attached) is exactly what S-01/S-03/S-04/S-05 already deliver; none of that analysis work is
affected or devalued. Objective 6 marked dropped in this file's Objectives section (struck
through, not deleted, matching this project's own convention of recording scope changes rather
than erasing them — e.g. B-08's "confirmed superseded" treatment). Same-day addendum to D-86:
daily-resolution figures built for all 3 scenario families (livestock, grazing, fertilizer),
naming made consistent (`s05_{axis}_daily_{view}_{ssp}.png`), extended to both SSPs (18 figures
total). See D-87, D-86, `s05_livestock_daily_chains_plots.py`, `s05_practices_daily_chains_plots.py`.)_

_Previously updated: 2026-08-08 (D-86: S-05 extended to farming-practice scenarios — grazing timing and
fertilizer schedule, two separate baseline-livestock experiments, same-session follow-up to D-85.
User named both levers explicitly. Priors stated before building anything (per F-01/F-04/F-05's
"redundant on the rich base" finding): both expected smaller than livestock's cattle effect, but
grazing tied directly to livestock presence (expect real effect) vs. fertilizer's weaker CH4-
specific mechanistic link (expect muted) — confirmed, not adjusted after seeing the numbers. Both
DERIVED features (not directly scalable) reconstructed by reusing their real-data construction
functions unchanged: grazing phase-shifts the real day-of-year species climatology at season edges,
re-deriving `fx_grazing_active`/`fx_days_since_grazing` via `days_since_grazing()`; fertilizer
builds a per-tower "typical year" event template (T4: ~8.25 events/yr, DOY 82-234, mean 127 kg/ha)
scaled by rate/frequency, run through `recency_series()` (tau=14 decay). 900 calls/axis (3 towers x
2 SSPs x 5 GCMs x 10 realizations x 3 levels), 2050 horizon, smoke-tested first. **Actual runtime
~51 min/axis (~1.7h combined)**, 0 failed calls. **Grazing shows a real, monotonic effect at every
tower** (T4 +18.9% at +4wk, T9 +17.2%); **fertilizer shows a small, sign-inconsistent effect**
(T2/T9 negative, T4 positive, all under 5%) — extends F-01/F-04/F-05's finding from real-data
feature importance to scenario response. AOA side-finding: grazing's flagged-% grows monotonically
with shift level (T4: 76%→88%), fertilizer's doesn't move as cleanly. No change to any standing
recommendation. See "Current status" bullets above, D-86, D-85,
`notebooks/07_scenario_analysis/s05_practices_trajectory.py`, `s05_practices_analysis.py`,
`s05_results.md`'s "Second update" section.)_

_Previously updated: 2026-08-08 (D-85: S-05 extended to 2050 + full daily chains saved for every
scenario point, same-session follow-up to D-84. Horizon extended from a fixed 10 years to 2050
(T4/T9: 27 years, T2: 31 years, matching S-04's endpoint); full daily chains saved for all 8,100
calls, folded into the same run since the horizon extension — not the daily save — dominates the
new cost (measured: a single 27-year call takes 4.07s vs. ~1.2-1.3s for the original 10-year call).
Actual runtime 5.44h (estimated ~9h), 0 failed calls, 83,767,500 daily rows written incrementally
to Parquet (1.25GB). Reproducibility spot-checked: 9.981 (original) vs. 9.974 (this run) — 0.07%
apart, ordinary GPU variance. **Every D-84 finding replicates, several more clearly**: cattle
dominance holds/strengthens (T4 3x-alone +205.6%→+214.5%); joint-vs-additive holds; realization/
GCM spread stays in the same small isolated range (2.4-6.6%→2.5-7.6%); AOA flatness now confirmed
across the full 27-31-year horizon. **New pattern the 10-year window was too short to show**:
SSP2-4.5 vs SSP5-8.5 divergence now visibly grows from early to late window, matching S-04's own
"widens toward end of century" finding. No change to any standing recommendation — both horizon
versions kept on disk. See "Current status" bullets above, D-85, D-84,
`notebooks/07_scenario_analysis/s05_trajectory_2050.py`, `s05_analysis_2050.py`,
`s05_results.md`.)_

_Previously updated: 2026-08-08 (D-84: S-05 — TabICLv2 + S-03's Variant A + F-10's species split, run
as a 10-year transient CMIP6 trajectory with independent per-species livestock multipliers.
Follow-up to D-83: TabICLv2 is one-shot (not a recursive rollout), so its S-03 horizon extends
past 365 days without compounding-error risk, and Variant A's 10-column feature set is almost
exactly what `data/Simulated Climate Data/` (S-04's own CMIP6 source) already supplies. Scoped
like S-04 (3 towers x 2 SSPs x 5 GCMs x 10 realizations/GCM) x 27 independent per-species
livestock multiplier combos (user's explicit choice over a cheaper shared-multiplier option) =
8,100 calls, 2.54h actual runtime. **Headline: cattle dominates the FCH4 response far beyond its
own LSU-weight share** (tripling cattle alone ~triples predicted FCH4 at T4/T9; sheep/lamb stay
under 25% even at 3x). Joint 3-species scaling is close to additive at T4, a real +8.8%
super-additive effect at T9. **A genuine methodological correction happened mid-analysis**: a
first-pass "realization spread" metric mirroring S-04's own pooling convention gave 32-69% of the
mean (10x too high) — traced to year-to-year weather variability being conflated with realization/
GCM choice; isolating the latter alone gives 2.4-6.6%, consistent with S-04. AOA flagged-% is high
(62-68%, vs S-04's 9-15%) but flat over the horizon — second confirmation that AOA's absolute
level depends on feature-space breadth. No change to any standing recommendation. See "Current
status" bullets above, D-84, D-83, `notebooks/07_scenario_analysis/s05_trajectory_10yr.py`,
`s05_analysis.py`, `s05_results.md`.)_

_Previously updated: 2026-08-08 (D-83: S-03 driver-availability ablation brought up to speed with D-80's
climatology-MASE convention and D-79's TabICL-sourced gap-filling, and its model-roster addendum
finally wired into the notebook itself. Model 1 recomputed (not left on the old persistence-scored/
RF-sourced table) wherever TabICL data makes that possible, so Model 1 and Variant A/B stay
apples-to-apples; TFT/DLinear/LSTM (no TabICL-sourced hourly data exists anywhere in this project)
rescored from their existing chains, not retrained. **The original driver-availability finding
replicates qualitatively unchanged under both switches.** **Unplanned, more consequential finding**:
switching the tree/SARIMAX/ensemble family's training target (not just TabPFN/TabICLv2's context,
which D-80 found only mildly hurt) to TabICL-sourced `y_gapfilled` causes a much larger absolute
MASE increase (~1.3-2.5x) at every tower — traced to TabICL's gap-filled series sitting at a
substantially different mean level than RF's. Sharpens D-80's own conclusion: D-79's better
gap-filling does not transfer to forecasting for this family either, more severely than the
foundation-model-context case D-80 already flagged. See "Current status" bullets above, D-83, D-80,
D-79, `notebooks/07_scenario_analysis/s03_climatology_tabicl_update.py`, `s03_results.md`.)_

_Previously updated: 2026-08-06 (D-82: S-04 analyzed — a completed-but-undocumented realization-level/
SSP5-8.5 scenario trajectory, built 2026-07-15/16 but never written up, closes S-01's two queued
extensions. New read-only `s04_analysis.py` summarizes the existing `s04_trajectory_2050.py`/
`s04_daily_top3_2050.py` output (234,000-row primary-hybrid sweep + 28,080-row B-10
diagnostic-benchmark sweep, both SSP2-4.5/SSP5-8.5, full realization scale, annual 2025-2050 — no
new model fitting). **S-01's central finding holds and is reinforced at full trajectory scale**:
1×→3× livestock, hybrid response +38.6%/+156.4%/+120.3% (T2/T4/T9) vs. the B-10 diagnostic
ensemble's own +20.4%/+76.6%/+62.0%. **New finding**: the transient/realization-level AOA check
flags materially more days than S-01's smoothed ensemble-mean snapshot did, at every multiplier
including the unchanged 1× baseline (9–15% here vs. 0% in S-01) — scenario-construction method
measurably changes how out-of-distribution a scenario looks, independent of the livestock question.
SSP2-4.5/SSP5-8.5 divergence is real, grows toward 2050, stays under 1% of the mean throughout;
realization-level spread is small (1–5% of the mean) and narrows as livestock stress increases.
Remaining open: a self-consistent mechanistic livestock-scenario construction, SPACSYS if time
permits. See "Current status" bullets above, D-82,
`notebooks/07_scenario_analysis/s04_analysis.py`, `s04_results.md`.)_

_Previously updated: 2026-08-02 (D-79: a third, parallel notebook,
`03c_gap_filling_revisited/temp_gap_filling_pipeline.ipynb`, reproduced the same 0.576/0.404/0.426
champion and did four things: (1) ported a literature-correct, 3-bug-fixed MDS reconstruction from
a separate audit notebook (`temp_mds.ipynb`), which also confirmed Towers 2/4/9 are the literal
same sites as Zhu et al. (2023a)'s ROTH_HS/PP/HSC farmlets and surfaced a genuine R²
metric-definition divergence specific to MDS (this project's `r2_score` vs. Zhu et al.'s own
OLS/Pearson-r² convention); (2) added HyperImpute as a third imputation baseline (R²=0.51/0.34/
0.35, dramatically ahead of MICE on identical features); (3) generalized the production
gap-filled series from champion-only to all 16 models evaluated; (4) ran a new "feed derived
signal back into the model" experiment line (D5-D8) — three straight negative results (env-KNN
features, D5; TICA-components/model-uncertainty features, D6; TabICL feature-drop/TICA-swap, D7)
before one real, narrow positive (TabICL row-cap bagging helps specifically at Tower 4, D8).
**Along the way, found the first result in either `03c_gap_filling_revisited` notebook to beat
the RF champion at more than one tower: TabICL-solo (champion `FEATURES`, trained per-tower not
pooled — its fixed 10,000-row context cap makes pooling actively dilutive) reaches T2=0.676,
T4=0.428, T9=0.423 vs. RFm's 0.576/0.404/0.426** — flagged as a validated benchmark result in
`BEST_RESULTS.md` §1, not yet production-adopted (no UQ/production-fill tooling exists for this
TabICL config yet). Separately, `_model_cache/` (RF's joblib cache) had silently grown to 166 GB
over the project's history — deleted, then rebuilt cleanly to 77 GB via a full top-to-bottom
rerun (314 cells, cold cache, ~8h36m, zero errors), with automatic stale-entry pruning added going
forward. **Outcome: RFm remains the production-adopted config at every tower, unchanged.** See
"Current status" bullets above, D-79,
`notebooks/03c_gap_filling_revisited/temp_gap_filling_pipeline.ipynb`,
`notebooks/03c_gap_filling_revisited/summary.md` §12-18.)_

_Previously updated: 2026-07-24 (D-77/D-78: `03c_gap_filling_revisited` — a fully self-contained (zero
`src/` imports) rebuild of the gap-filling pipeline found and fixed a real bug in `mdc_gapfill()`
(flat 2h interpolation cutoff too short for low-diurnal-structure drivers — soil moisture/
temperature, TA, VPD, WS — extended to 288h for those 5 only). **New standing champion R²: T2
0.576 (was 0.574), T4 0.404 (was 0.402), T9 0.426 (was 0.418)** — `BEST_RESULTS.md` §1 updated
(D-77). A large follow-up exploration on this corrected base (D-78, separate working copy,
`temp_gap_filing_exploration copy.ipynb`, strictly additive throughout) then tested: an
Area-of-Applicability UQ layer (validated, weak-but-real error correlation +0.11 to +0.16, ready
for production use); six additional models full 3-tower (LightGBM/TabICL edge the champion at
Tower 4 only, +0.006/+0.019; XGBoost/TabPFN/SAITS/BI-LSTM all lose everywhere — SAITS notably
improved substantially vs. F-11's original numbers on the corrected features but still short);
and lag/lead feature expansion (soil-lag bidir/leadonly reproduce F-12's null result almost
exactly on the corrected base; target/FCH4 lag-lead, genuinely new, regresses clearly and required
a new per-fold leakage-safe masking pattern). **Outcome: RFm (D-77) remains the standing
gap-filling recommendation at every tower — nothing in D-78 beat it outright.** See "Current
status" bullets above, D-77, D-78,
`notebooks/03c_gap_filling_revisited/temp_gap_filing_exploration.ipynb` (+ ` copy.ipynb`).)_

_Previously updated: 2026-07-21 (D-76: F-11 follow-up — testing SAITS's diagnosed failure modes from
D-74 finds real gains, still not adopted. Five levers tested (seeding, per-scenario retraining,
solo-vs-pooled, spike-weighted loss, bigger model), staged cheapest/most-diagnostic first. Most
elapsed time was infrastructure friction (a stuck run killed after 10+ hours with zero progress,
likely machine-sleep-induced CUDA corruption; a subsequent transient CUDA error resolved by
retrying fresh; background-task tracking dropped by session restarts multiple times), not
experiment design. **Spike-weighted loss (attacks FCH4's spike-dominated skew) was by far the
largest lever** — ~5–6x'd R² on its own, reproduced across reruns regardless of pooling structure
(which turned out to be noise-level, flipping winner between reruns). Full 5-scenario confirmation
(solo + spike-weighted loss + bigger model): T2 0.192, T4 0.225, T9 0.110 — still loses to RFm's
0.574/0.402/0.418 everywhere, but the gap narrowed substantially from D-74's naive baseline
(T4 -0.43→-0.18). **Not adopted** — RFm (D-35/D-49) remains standing, no `BEST_RESULTS.md` change.
See "Current status" bullets above and `F11_results.md` §8 for full detail.)_

_Previously updated: 2026-07-20 (D-75: F-12, bidirectional/lead soil lags tested for RFm — 3-arm ablation
(backward-only baseline / bidir / leads-only) on just the swc/ts lag block, everything else fixed
at the champion config. First `N_REPS=5` attempt (225 fits) was killed by the environment after
~2h20m with zero progress saved; rebuilt with `N_REPS=2` (90 fits) and a checkpointed `nbclient`
driver, completed in ~88 min. Leakage checks (feature-purity + per-fit held-out/training-overlap
assertion) passed on all 90 fits. **Result: mixed/null** — Arm A reproduces the champion exactly;
Arms B/C gain a noise-level +0.008 at T4 only, regress at T2/T9. Feature importance confirms leads
are used by the RF (real importance mass), so the null is genuine redundancy, not an ignored-
feature artifact. **Not adopted** — RFm remains standing, no `BEST_RESULTS.md` numeric change. See
"Current status" bullets above and `notebooks/04_feature_engineering/F12_results.md` for full
detail.)_

_Previously updated: 2026-07-19 (D-74: F-11, SAITS gap-filling evaluated via `pypots` — required an
unrelated `torchvision`/`torch` ABI-mismatch environment fix first (reinstalled `torchvision` to a
matching cu128 build). Reused F08's exact `insert_calendar_gaps` held-out timestamps for a
point-for-point comparison against the RFm champion (R² T2=0.574/T4=0.402/T9=0.418), but trained
one pooled SAITS model on the union of all 25 held-out sets (compute-bounded, stated explicitly)
instead of RFm's 75 per-scenario retrains. Full 3-tower × 5-scenario run: **SAITS loses at every
tower by a wide margin** (median R² ≈0.03/0.00/−0.02), reproduced on an independent rerun.
Diagnosed via a consistent negative MBE (systematic under-prediction) to FCH4's baseline sparsity
(worse than SAITS's dense-series design target) plus its spike-dominated right-skew. **Not
adopted** — RFm remains the standing gap-filling recommendation, no `BEST_RESULTS.md` change. See
"Current status" bullets above and `notebooks/04_feature_engineering/F11_results.md` for full
detail.)_

_Previously updated: 2026-07-18 (D-73: IMP-01 opens a revisited, more thorough gap-filling/imputation
phase (`08_imputation_revisited/`) — step 1 of 5 (viz → algorithms → UQ → distributional shift →
masked-data predictor testing). Full feature space (every raw measured column per tower, not just
FCH4), reusing D-18's spatial-alignment rule via new `src/features/tower_feature_space.py`.
Missingness is strongly block-structured (not MCAR) — co-missingness clustering shows a shared
EC/met/turbulence outage block, a separate footprint cluster, a separate catchment water-quality
cluster, a separate soil-probe cluster, and always-complete livestock. Gap lengths are strongly
bimodal (mostly 1-hour, but EC_soil reaches ~2,500-2,800-day blackouts). Real seasonal/diurnal
missingness patterns confirm MAR/MNAR structure. Tower-2-specific finding: FCH4/CH4 break out of
the main EC co-missingness block, unlike Towers 4/9. Prior gap-filling results (R-01-F-09b) not
overwritten. See "Current status" bullets above and `notebooks/08_imputation_revisited/
IMP01_results.md` for full detail.)_

_Previously updated: 2026-07-15 (D-71: is chain-persistence a valid MASE baseline for a seasonal series?
Extended B-09's `doy_climatology()` baseline from a single tower/anchor to full 3-tower × 5-anchor
coverage and reran `bin_metrics()` for all 11 B-10/B-13 models with climatology as MASE's
denominator instead of persistence. Result reverses the motivating hypothesis: pooled, climatology
is the *weaker* baseline (own MAE 43.79 vs. persistence's 37.50) — reinforces keeping persistence as
the primary MASE denominator (D-37) rather than switching. See "Current status" bullets above.)_

_Previously updated: 2026-07-15 (D-70 addendum: S-03's model-roster extension — TFT/TabPFN/DLinear/LSTM/
TabICLv2 were missing from the driver-availability ablation, a real scope gap caught only after the
user asked directly whether S-03 covered TabPFN/TabICLv2. Fixed via `s03_model_roster_extension.py`
(smoke-tested, then full 5-anchor sweep); result refines rather than confirms the original finding
— Variant B (resample) beats/ties Model 1 on MASE for 9/11 models, but TFT is a genuine reversal on
R² (-0.363→-0.492 under resampling). Also fixed a latent ground-truth-column bug in
`b10_b13_chain_plots.py` that would have broken for the new variant-suffixed DL columns. See
"Current status" bullets above and `s03_results.md`'s addendum section for full detail.)_

_Previously updated: 2026-07-10 (D-67, F-10: new standing recursive-rollout best is `TabPFN+species`,
MASE=0.840/R²=−0.084, beating B-10's ensemble outright — see "Current status" bullets above and
`BEST_RESULTS.md` §3 for full detail. MASE is now this project's primary forecasting metric,
CLAUDE.md.)_

_Updated: 2026-08-29 (D-109: repository standardisation. Added an entirely additive
`workflows/latest/` layer with six ordered stage runbooks, a 44-entry machine-readable manifest,
and a validator; added a root `README.md`; and exposed only current report sources, outlines,
appendices, and report-facing figures through narrow `.gitignore` exceptions. Historical
experiments remain untouched and no best-result metric changed. The validator passes, every new
Markdown link resolves, and all 39 referenced Python entry points parse. A future Git commit must
remain scoped to this canonical layer and current report materials because unrelated user-owned
result changes remain in the working tree.)_

_Previously updated: 2026-07-09 (D-66 second same-day addendum: TFT staleness reconciled across
`b10_b13_metrics_rerun.md` -- several unrelated reruns of `b10_b13_rerun_multi_anchor.py` (TFT is
unseeded, D-62) had let TFT's published tables drift from its live CSV values without ever being
fixed. Recomputed every TFT row from the live summary CSVs and propagated into all 5 underlying
result CSVs and every markdown table/prose reference (all-tower, Tower-4-only, tower×year x2,
gap-filled tables, Findings section, TabICLv2 verdict paragraph, BEST_RESULTS.md). Corrected
headline: all-tower R²=−0.363 (was −0.565), Tower-4-only R²=−0.228 (was −0.568) -- TFT now sits
much closer to SARIMAX than before; standing recommendation (Ensemble_unweighted) unchanged. See
D-66. D-66 (corrected 2026-07-10): TabICLv2 -- a new tabular foundation model (ICML 2026, "heavily inspired by TabPFN-TS") -- added to the B-10/B-13 sequence via a new sibling script mirroring TabPFN's per-tower/per-anchor/never-pooled integration; API contract verified empirically before coding. Full 3-tower x 5-anchor sweep in ~10 seconds -- dramatically cheaper than every other model here. A real point-estimate bug was found and fixed after user skepticism ("I am skeptical how TabICLv2 is the exact opposite of TabPFN here"): `tabicl_forecast()` was extracting a mean-based point column rather than the median, badly biased high on this heavy-tailed flux distribution. Corrected result: all-tower R²=-0.329 observed (was -1.929), -0.886 gap-filled (was -4.472) -- now beats SARIMAX (-0.360) and TFT (-0.363), MASE (0.928) 4th-best of 10 models, still behind TabPFN (-0.122) and the standing recommendation (Ensemble_unweighted, -0.165) but no longer near-bottom -- the best accuracy-per-compute-second result in the sequence. Shares TabPFN's own already-documented Tower-9/2019 degenerate-forecast limitation (zero real y_observed pre-anchor -> flat ~0.0 prediction, unaffected by the fix), confirmed not a new bug. `results/b10_b13_full_chains.csv` and all 165 `b10_chains` figures regenerated with corrected predictions. See D-66, `notebooks/05_benchmarking/b10_b13_metrics_rerun.md` ("Model-roster extension: TabICLv2"). D-65 third addendum: DLinear/LSTM model-roster extension closes the `b10_chains`-figures-vs-metrics gap, confirms D-53/D-54 at full coverage, and produces a sharper generalized version of D-62's TFT non-determinism finding (root cause: `torch.manual_seed()` runs after model construction, so only the first torch model built in a process is unseeded) -- also surfaced a separate TFT staleness discovery (third random draw in the live summary CSV vs. the cited tables), flagged not resolved. See `notebooks/05_benchmarking/b10_b13_metrics_rerun.md` ("Model-roster extension" section). D-65 second addendum: secondary/exploratory metric scored against `y_gapfilled` instead of `y_observed` (user's own idea) -- deliberate, caveated departure from D-36/D-37's "train on gap-filled, evaluate on observed" convention for this one bounded check. Unlocks Tower 2's coverage fully (816→14,600 of 14,600) but **R² gets worse while RMSE/MASE improve** -- a variance-normalization artifact (smoother target -> smaller total variance -> same absolute error penalized more by R²), not a contradiction. Ranking mostly holds (Ensemble_unweighted stays top-tier either way) except **TabPFN drops from best-R² (observed) to 6th of 8 (gap-filled)** -- the one real disagreement. Bounded to B-10/B-13 only, no retrofit into U-02/U-03/S-01/I-02. See `notebooks/05_benchmarking/b10_b13_metrics_rerun.md`, `results/b10_b13_rerun_summary_vs_gapfilled.csv`. D-65 (first addendum, 2026-07-07): `bin_metrics()` extended with RMSE/WAPE/Correlation, alongside D-64's S-01 first scenario result. B-10+B-13 reconstructed and rerun with the fuller metric set (their original multi-anchor scripts were never committed -- this one is, closing that gap) -- **reproduction confirmed bit-for-bit for 7/8 models at Tower 4** (TFT differs, already-documented non-determinism). **Addendum: extended to all 3 towers (T2/T4/T9)** per the new "full coverage by default" CLAUDE.md convention -- all-tower pooled R² is substantially worse than the T4-only headline (Ensemble_unweighted 0.012→−0.165) while MASE holds/improves (0.975→0.918), driven by Tower 9 being consistently harder and Tower 2 largely degenerate outside 2018; model ranking unchanged. **New finding: TabPFN's best-in-sequence MASE (0.862 T4-only / 0.855 all-tower) does not extend to RMSE** (second-worst in both) -- its strength is consistency vs. persistence, not small worst-case errors; correlation is uniformly weak (0.26-0.40) across every model/scope. No change to the standing recommendation. See D-65 (+ addendum), `notebooks/05_benchmarking/b10_b13_metrics_rerun.md`, `results/b10_b13_rerun_table_all_towers.csv`, `results/b10_b13_rerun_table_by_tower_year.csv`. D-64: S-01, the first Phase 07 scenario-simulation worked example — Phase 07 moves from "PLANNED, not started" to a proven end-to-end mechanism. Level-residual hybrid (Ridge trend, fit once on the full pooled record, carries the climate+livestock extrapolation; RF/XGB/LightGBM with a monotonic constraint on livestock density correct only the residual; USTAR/SHF dropped entirely) built on D-46/D-52's data scoping and directly informed by U-03's extrapolation-ceiling finding plus a user-provided deep-research literature pass. Extended to **all 3 towers and all 3 residual models individually**, per direct user request. **Result: the hybrid measurably fixes U-03's flattening** (+138%/+105% at T4/T9 for a 3x livestock sweep, vs. U-03's trees-alone +21-23%). **Genuine finding: a "2x" scenario built by scaling a smoothed climatology is milder than one built by scaling raw values** (U-03's method) — only 3x exceeds the training envelope (Area-of-Applicability check), and Tower 2 never does (its own livestock baseline is far smaller, consistent with U-03's own T2 finding). **Per-model finding: XGB/LightGBM's residual correction is completely flat with respect to livestock density** (verified via a full monotonic sweep) — for 2 of 3 tree models, ~100% of the scenario response flows through the trend, not the residual; only RF shows real residual sensitivity. A frozen model artifact was persisted for the first time in this project. See D-64 (+ addendum), `notebooks/07_scenario_analysis/S01_first_scenario.ipynb`, `s01_results.md`, `results/figures/s01_*.png`. Builds on D-46/D-52/D-63/D-61. **Next: extend S-01 (SSP5-8.5, realization-level spread, mechanistic livestock scenario, possibly SPACSYS for the trend component)** — B-08 confirmed superseded for Phase 07's purposes, remains available separately for the point-forecast track.)_
