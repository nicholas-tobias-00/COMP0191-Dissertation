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

**Primary target:** Half-hourly EC CH₄ flux (`greenhouse.csv` — column to be confirmed during EDA)  
**Complementary features:** Soil moisture, temperature, rainfall from `measurements.csv`; management events from `Field_Event_Data_Format_1.csv`; livestock location counts from `Animal_location_counts_*.csv`

**Temporal split (indicative):**
- Train: 2018–2021
- Test: 2022–2023
- Held-out: 2024

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

All data lives in `data/` (gitignored).

| Location | Contents |
|---|---|
| `data/Consolidated/` | Raw annual CSVs from NWFP portal |
| `data/Compiled/` | Merged multi-year files from `notebooks/01_data_compilation/` |

**Key compiled files:**

| File | Description | Frequency |
|---|---|---|
| `greenhouse.csv` | EC fluxes: CH₄, CO₂, H₂O, H, LE (Tower 2, 2018–present) | 30-min |
| `measurements.csv` | Water flow + soil moisture across ≤15 catchments | 15-min |
| `livestock_weight_long.csv` | Cattle + sheep + lamb weighings | Event |
| `livestock_condition_score_long.csv` | Body condition scores | Event |
| `Animal_location_counts_*.csv` | Head-count per field per species | Daily |
| `Field_Event_Data_Format_1.csv` | Fertiliser, spraying, reseeding events | Event |

**Data quality notes:**
- `greenhouse_` and `measurements_` columns carry sibling quality-flag columns (`"Acceptable"`, `"Not set"`). Always filter before analysis.
- EC data has persistent sensor gaps — ERA5 reanalysis substitution validated for this (Zhu et al. 2023a).
- High sparsity in sensor columns is normal.

---

## Repository layout

```
notebooks/
  01_data_compilation/    COMPLETE — compiles Consolidated → Compiled (23 files)
  02_eda/                 IN PROGRESS — skeleton only
  03_gap_filling/         PLANNED — replications + gap-filling baseline
  04_feature_engineering/ PLANNED
  05_benchmarking/        PLANNED
  06_interpretability_uq/ PLANNED
  07_scenario_analysis/   PLANNED
src/
  data/                   loaders for compiled CSVs
  features/               aggregation, lag construction, quality filtering
  models/                 model wrappers
  evaluation/             metrics, plotting
results/                  benchmarks.csv (append-only) + figures
prompts/                  session templates
DECISIONS.md
REPLICATIONS.md
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

- **Phase:** EDA
- **Completed:** Data compilation pipeline (`01_data_compilation`) — all 23 compiled files verified
- **In progress:** `02_eda` — skeleton only (two `head()` calls on raw files; needs full analysis on compiled data)
- **Next phase:** Gap-filling replications (`03_gap_filling`) — start with Irvin et al. (2021) RF benchmark

## Next task

> _[One sentence — update this before committing at end of each session]_

---
_Last updated: 2026-06-12_
