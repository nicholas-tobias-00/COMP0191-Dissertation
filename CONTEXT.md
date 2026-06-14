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
  03_gap_filling/         PLANNED — R-01 through R-04 replications
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

- **Phase:** Gap-filling replications (R-01 complete all towers, R-02 next)
- **Completed:**
  - `01_data_compilation` — 23 compiled files in `data/Compiled/`
  - `02_eda` — full EDA + Section 6 modelling readiness check; figures in `results/figures/`
  - `src/data/consolidate_hourly.py` — `data/Hourly/consolidated_hourly.csv` (70,153 rows × 449 cols, 39.4% NaN)
  - `03_gap_filling/R01_Irvin2021_RF_XGBoost.ipynb` — R-01 complete, all three towers (see below)
  - `03_gap_filling/R01_results.md` — detailed per-tower results, interpretation, and next steps
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
  - Tower 4 is the only tower with positive R²; Tower 9 is near-null; Tower 2 D-15 split needs redesign
  - Full details in `notebooks/03_gap_filling/R01_results.md`
- **Next phase:** R-02 gap-filling replication (Kim et al. 2020) — Towers 4 and 9

## Replications

| ID | Paper | Target metrics | Status | Notebook |
|---|---|---|---|---|
| R-01 | Irvin et al. (2021) — FLUXNET-CH4 RF/XGBoost gap-filling | Paper: RF R²=0.79, XGB~0.65–0.67 (17 wetland sites). **T4: RF R²=0.144, XGB R²=0.086; T9: RF R²=−0.027, XGB R²=−0.089; T2: RF R²=−16.9 (split design failure)**. See R01_results.md. | complete | `03_gap_filling/R01_Irvin2021_RF_XGBoost.ipynb` |
| R-02 | Kim et al. (2020) — RF vs ANN vs SVM vs MDS + PCA | RF best; PCA degrades; lags matter more for CH₄ | planned | `03_gap_filling/` |
| R-03 | Zhu et al. (2023a) — UK managed pastures gap-filling | RFR beats MDS for gaps >12 days; ERA5 validated | planned | `03_gap_filling/` |
| R-04 | Partridge et al. (2024) — NWFP GreenFeed Gradient Boosting | r=0.619, RMSE=51.8 g/day (animal-scale baseline) | planned | `03_gap_filling/` |

Each replication is run **per tower** (Tower 2, 4, 9 independently). Start with Tower 4 (best coverage), then extend to Towers 9 and 2 in that order. Log per-tower results in `results/benchmarks.csv`.  
_Update Status to `in-progress` / `complete` / `abandoned` as work proceeds._

---

## Next task

**R-01 is complete for all three towers.** Next steps:

1. **R-02 replication** — Kim et al. (2020): RF vs ANN vs SVM + lag features on Tower 4 FCH4. Notebook: `notebooks/03_gap_filling/R02_Kim2020_RF_SVM_ANN.ipynb`. Extend to Tower 9 after Tower 4.
2. **Tower 2 split redesign** — D-15 custom split (2018 train / Jan–May 2019 test) causes seasonal mismatch (R²=−16.9). Redesign using leave-one-season-out CV within pre-gap window, or download 2024 data for a genuine post-gap test set.
3. **Tower 9 feature investigation** — near-null R² (−0.027 RF) may reflect training-data sparsity (3,981 vs 7,714 rows). Adding management event features in R-02 may improve this.
4. **ERA5 for SWIN gap-fill** — `SWIN_1_1_1 [Tower 4]` is only ~52% available; ERA5 `ssrd` would improve predictor coverage but is not a blocker (D-14).

---
_Last updated: 2026-06-13 (R-01 extended to all three towers; R01_results.md created; D-19 added)_
