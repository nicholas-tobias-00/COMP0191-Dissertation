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
6. Digital shadow interface (Streamlit) with scenario analysis and uncertainty visualisation.

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
   - **Next (in order): (1) 07 scenario analysis, extending S-01** — SSP5-8.5, realization-level
     (not just ensemble-mean) spread, a self-consistent mechanistic livestock-scenario construction,
     and (if time permits) the SPACSYS process-model route for the trend/level component. B-08
     remains available separately for the point-forecast track but is not on this critical path.
     Deferred: coarser/cumulative eval; gap-filling-phase metrics backfill. Backlog: ERA5; chase
     2024 held-out EC data. If S-02's PPFD/RN/WS candidates are pursued further, address the
     100%-extrapolation caveat first (e.g. test against individual GCM/realization trajectories,
     not just the ensemble mean).
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
   - ⚠ **Held-out 2024 still empty** (2024 FCH₄ = 0% valid all towers) — final held-out benchmark blocked until 2024 EC fluxes are downloaded; test on 2022–2023 meanwhile.
2. **Use partial pooling (D-30) as the multi-tower default** — pooled global model + tower-indicator (or continuous tower descriptors); rescues data-poor towers while protecting data-rich ones.
3. **Tower 2 split redesign** (D-15/D-19) — also lets Tower 2 be a proper pooled/test member.
4. **(Optional) Operational FCO₂ variant** — re-run 03b with `FC_recon` everywhere (strict, leak-free).
5. **ERA5 driver_era** (D-14); **SVM C-search** (R-03); validate Tower-9 pooled-density gain on 2024 once downloaded.

---
_Last updated: 2026-07-21 (D-76: F-11 follow-up — testing SAITS's diagnosed failure modes from
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

_Previously updated: 2026-07-09 (D-66 second same-day addendum: TFT staleness reconciled across
`b10_b13_metrics_rerun.md` -- several unrelated reruns of `b10_b13_rerun_multi_anchor.py` (TFT is
unseeded, D-62) had let TFT's published tables drift from its live CSV values without ever being
fixed. Recomputed every TFT row from the live summary CSVs and propagated into all 5 underlying
result CSVs and every markdown table/prose reference (all-tower, Tower-4-only, tower×year x2,
gap-filled tables, Findings section, TabICLv2 verdict paragraph, BEST_RESULTS.md). Corrected
headline: all-tower R²=−0.363 (was −0.565), Tower-4-only R²=−0.228 (was −0.568) -- TFT now sits
much closer to SARIMAX than before; standing recommendation (Ensemble_unweighted) unchanged. See
D-66. D-66 (corrected 2026-07-10): TabICLv2 -- a new tabular foundation model (ICML 2026, "heavily inspired by TabPFN-TS") -- added to the B-10/B-13 sequence via a new sibling script mirroring TabPFN's per-tower/per-anchor/never-pooled integration; API contract verified empirically before coding. Full 3-tower x 5-anchor sweep in ~10 seconds -- dramatically cheaper than every other model here. A real point-estimate bug was found and fixed after user skepticism ("I am skeptical how TabICLv2 is the exact opposite of TabPFN here"): `tabicl_forecast()` was extracting a mean-based point column rather than the median, badly biased high on this heavy-tailed flux distribution. Corrected result: all-tower R²=-0.329 observed (was -1.929), -0.886 gap-filled (was -4.472) -- now beats SARIMAX (-0.360) and TFT (-0.363), MASE (0.928) 4th-best of 10 models, still behind TabPFN (-0.122) and the standing recommendation (Ensemble_unweighted, -0.165) but no longer near-bottom -- the best accuracy-per-compute-second result in the sequence. Shares TabPFN's own already-documented Tower-9/2019 degenerate-forecast limitation (zero real y_observed pre-anchor -> flat ~0.0 prediction, unaffected by the fix), confirmed not a new bug. `results/b10_b13_full_chains.csv` and all 165 `b10_chains` figures regenerated with corrected predictions. See D-66, `notebooks/05_benchmarking/b10_b13_metrics_rerun.md` ("Model-roster extension: TabICLv2"). D-65 third addendum: DLinear/LSTM model-roster extension closes the `b10_chains`-figures-vs-metrics gap, confirms D-53/D-54 at full coverage, and produces a sharper generalized version of D-62's TFT non-determinism finding (root cause: `torch.manual_seed()` runs after model construction, so only the first torch model built in a process is unseeded) -- also surfaced a separate TFT staleness discovery (third random draw in the live summary CSV vs. the cited tables), flagged not resolved. See `notebooks/05_benchmarking/b10_b13_metrics_rerun.md` ("Model-roster extension" section). D-65 second addendum: secondary/exploratory metric scored against `y_gapfilled` instead of `y_observed` (user's own idea) -- deliberate, caveated departure from D-36/D-37's "train on gap-filled, evaluate on observed" convention for this one bounded check. Unlocks Tower 2's coverage fully (816→14,600 of 14,600) but **R² gets worse while RMSE/MASE improve** -- a variance-normalization artifact (smoother target -> smaller total variance -> same absolute error penalized more by R²), not a contradiction. Ranking mostly holds (Ensemble_unweighted stays top-tier either way) except **TabPFN drops from best-R² (observed) to 6th of 8 (gap-filled)** -- the one real disagreement. Bounded to B-10/B-13 only, no retrofit into U-02/U-03/S-01/I-02. See `notebooks/05_benchmarking/b10_b13_metrics_rerun.md`, `results/b10_b13_rerun_summary_vs_gapfilled.csv`. D-65 (first addendum, 2026-07-07): `bin_metrics()` extended with RMSE/WAPE/Correlation, alongside D-64's S-01 first scenario result. B-10+B-13 reconstructed and rerun with the fuller metric set (their original multi-anchor scripts were never committed -- this one is, closing that gap) -- **reproduction confirmed bit-for-bit for 7/8 models at Tower 4** (TFT differs, already-documented non-determinism). **Addendum: extended to all 3 towers (T2/T4/T9)** per the new "full coverage by default" CLAUDE.md convention -- all-tower pooled R² is substantially worse than the T4-only headline (Ensemble_unweighted 0.012→−0.165) while MASE holds/improves (0.975→0.918), driven by Tower 9 being consistently harder and Tower 2 largely degenerate outside 2018; model ranking unchanged. **New finding: TabPFN's best-in-sequence MASE (0.862 T4-only / 0.855 all-tower) does not extend to RMSE** (second-worst in both) -- its strength is consistency vs. persistence, not small worst-case errors; correlation is uniformly weak (0.26-0.40) across every model/scope. No change to the standing recommendation. See D-65 (+ addendum), `notebooks/05_benchmarking/b10_b13_metrics_rerun.md`, `results/b10_b13_rerun_table_all_towers.csv`, `results/b10_b13_rerun_table_by_tower_year.csv`. D-64: S-01, the first Phase 07 scenario-simulation worked example — Phase 07 moves from "PLANNED, not started" to a proven end-to-end mechanism. Level-residual hybrid (Ridge trend, fit once on the full pooled record, carries the climate+livestock extrapolation; RF/XGB/LightGBM with a monotonic constraint on livestock density correct only the residual; USTAR/SHF dropped entirely) built on D-46/D-52's data scoping and directly informed by U-03's extrapolation-ceiling finding plus a user-provided deep-research literature pass. Extended to **all 3 towers and all 3 residual models individually**, per direct user request. **Result: the hybrid measurably fixes U-03's flattening** (+138%/+105% at T4/T9 for a 3x livestock sweep, vs. U-03's trees-alone +21-23%). **Genuine finding: a "2x" scenario built by scaling a smoothed climatology is milder than one built by scaling raw values** (U-03's method) — only 3x exceeds the training envelope (Area-of-Applicability check), and Tower 2 never does (its own livestock baseline is far smaller, consistent with U-03's own T2 finding). **Per-model finding: XGB/LightGBM's residual correction is completely flat with respect to livestock density** (verified via a full monotonic sweep) — for 2 of 3 tree models, ~100% of the scenario response flows through the trend, not the residual; only RF shows real residual sensitivity. A frozen model artifact was persisted for the first time in this project. See D-64 (+ addendum), `notebooks/07_scenario_analysis/S01_first_scenario.ipynb`, `s01_results.md`, `results/figures/s01_*.png`. Builds on D-46/D-52/D-63/D-61. **Next: extend S-01 (SSP5-8.5, realization-level spread, mechanistic livestock scenario, possibly SPACSYS for the trend component)** — B-08 confirmed superseded for Phase 07's purposes, remains available separately for the point-forecast track.)_
