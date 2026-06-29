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
  06_interpretability_uq/ PLANNED
  07_scenario_analysis/   PLANNED
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
   - **Next: spike-aware modelling** (two-stage hurdle: occurrence→magnitude — the agreed priority, plan in `woolly-sparking-plum.md`); then **07 scenario analysis** (digital shadow). Optional backlog: target asinh transform + exog lags; ensembling/HPO; realistic-driver sensitivity + ERA5; chase 2024 held-out EC data.
   - ⚠ **Held-out 2024 still empty** (2024 FCH₄ = 0% valid all towers) — final held-out benchmark blocked until 2024 EC fluxes are downloaded; test on 2022–2023 meanwhile.
2. **Use partial pooling (D-30) as the multi-tower default** — pooled global model + tower-indicator (or continuous tower descriptors); rescues data-poor towers while protecting data-rich ones.
3. **Tower 2 split redesign** (D-15/D-19) — also lets Tower 2 be a proper pooled/test member.
4. **(Optional) Operational FCO₂ variant** — re-run 03b with `FC_recon` everywhere (strict, leak-free).
5. **ERA5 driver_era** (D-14); **SVM C-search** (R-03); validate Tower-9 pooled-density gain on 2024 once downloaded.

---
_Last updated: 2026-06-25 (F-07 Tower-2-only evaluation fix: F07_tower2_evaluation_RFm.ipynb + F07_results.md; D-34 added; benchmarks.csv 2765 rows. Finding: Tower 2's −16.9 was a broken year-split; full-period gap-CV gives RFm pooled +0.519 (best in project, beats MDS by ~1.0). Supervisor steer: optimise improvement-over-MDS, not absolute R²; deadline 1 Sept 2026. Next: forecasting (05), optionally re-eval 4/9 under full-period CV first.)_
