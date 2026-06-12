# REPLICATIONS.md
_One entry per paper replication. Update status as work progresses. Results go in `results/benchmarks.csv`._

---

## Status legend

| Status | Meaning |
|---|---|
| `planned` | Identified in literature review; not started |
| `in-progress` | Notebook exists; implementation underway |
| `complete` | Results obtained; deviations documented |
| `abandoned` | Dropped — reason noted |

---

## Index

| ID | Paper | Relevance | Status | Notebook |
|---|---|---|---|---|
| R-01 | Irvin et al. (2021) — FLUXNET-CH4 gap-filling benchmark | Core CH₄ EC methodology; RF/XGBoost reference | planned | `notebooks/03_gap_filling/` |
| R-02 | Kim et al. (2020) — ML comparison with PCA inputs | RF vs ANN vs SVM vs MDS; lagged features insight | planned | `notebooks/03_gap_filling/` |
| R-03 | Zhu et al. (2023a) — Gap-filling in challenging ecosystems | UK managed pastures included; ERA5 validation | planned | `notebooks/03_gap_filling/` |
| R-04 | Partridge et al. (2024) — ML for NWFP cattle CH₄ | Direct prior work at NWFP; establishes baseline to beat | planned | `notebooks/03_gap_filling/` |

---

## Entries

---

### R-01 — Irvin et al. (2021)

**Full citation:**
> Irvin, J., et al. (2021). Gap-filling eddy covariance methane fluxes: Comparison of machine learning model predictions and uncertainties at FLUXNET-CH4 wetlands. *Agricultural and Forest Meteorology*, 308-309, 108528. https://doi.org/10.1016/j.agrformet.2021.108528

**What is being replicated:**
The benchmark ML gap-filling pipeline for EC CH₄ flux using RF and XGBoost, evaluated against MDS (marginal distribution sampling), LASSO, and ANN baselines. Specifically: (1) the artificial gap scenario construction method, (2) the cross-validation protocol, (3) the uncertainty calibration approach, and (4) the finding that soil temperature is the most important predictor. This is the reference method that the NWFP forecasting pipeline must beat to claim novelty.

**Notebook:** `notebooks/03_gap_filling/`

**Status:** planned

**Target metrics (from paper):**
- R² ~0.7–0.8 across 17 FLUXNET-CH4 wetland sites
- RF and XGBoost superior to ANN and penalised regression across all biomes
- Soil temperature most important predictor
- Raw ML uncertainty systematically underestimated — calibration corrections required

**Our results:**
_[Fill in once run]_

**Deviations from paper:**
- Dataset: NWFP TOWER 2 (managed temperate grassland, single site) vs 17 FLUXNET-CH4 wetland sites
- Ecosystem type: managed beef cattle grassland vs natural/semi-natural wetlands
- Data period: 2018–present vs multi-site, multi-period
- Objective: test transferability of their method to NWFP; not expecting identical performance

**Notes:**
Open-source Python code released by authors — use as starting point. Primary benchmark paper for the whole gap-filling literature. The uncertainty underestimation finding directly motivates D-06 (mandatory UQ).

---

### R-02 — Kim et al. (2020)

**Full citation:**
> Kim, Y., et al. (2020). Gap-filling approaches for eddy covariance methane fluxes: A comparison of three machine learning algorithms and a traditional method with principal component analysis. *Global Change Biology*, 26(3), 1499–1518. https://doi.org/10.1111/gcb.14845

**What is being replicated:**
The four-way comparison of RF vs ANN vs SVM vs MDS for EC CH₄ gap-filling, with and without PCA input reduction. Key finding to test: (1) RF outperforms other methods; (2) lagged environmental variables improve CH₄ prediction substantially more than CO₂; (3) PCA *degrades* ML performance despite reducing input complexity.

**Notebook:** `notebooks/03_gap_filling/`

**Status:** planned

**Target metrics (from paper):**
- RF outperforms ANN, SVM, and MDS at all 5 sites
- PCA input reduction degrades ML performance relative to full feature set
- Lagged variables more important for CH₄ than CO₂

**Our results:**
_[Fill in once run]_

**Deviations from paper:**
- Dataset: NWFP single site vs 5 AmeriFlux/AsiaFlux wetland and rice paddy sites
- Will add XGBoost/Gradient Boosting to comparison (not in original paper)
- Will use temporal CV (not leave-one-site-out, since we have one site)

**Notes:**
The lagged features finding is directly actionable for feature engineering: build lagged versions of soil temperature, rainfall, and soil moisture as candidate inputs. The PCA finding means we should NOT use PCA for dimensionality reduction in the final pipeline.

---

### R-03 — Zhu et al. (2023a)

**Full citation:**
> Zhu, S., et al. (2023). Gap-filling carbon dioxide, water, energy, and methane fluxes in challenging ecosystems: Comparing between methods, drivers, and gap-lengths. *Agricultural and Forest Meteorology*, 332, 109365. https://doi.org/10.1016/j.agrformet.2023.109365

**What is being replicated:**
The Random Forest gap-filling approach applied to UK managed pastures (the most directly comparable ecosystem to NWFP). Specifically: (1) RFR vs MDS performance across different gap lengths (short <12 days, medium 12–30 days, long >30 days); (2) ERA5 reanalysis substitution as fallback for failed local sensors; (3) verification that key environment-flux responses are preserved in gap-filled data.

**Notebook:** `notebooks/03_gap_filling/`

**Status:** planned

**Target metrics (from paper):**
- RFR outperforms MDS for gaps >12 days
- ERA5 substitution validated — key env-flux responses preserved
- MDS still preferred for short (<12 day) CO₂ gaps

**Our results:**
_[Fill in once run]_

**Deviations from paper:**
- Dataset: NWFP Tower 2 CH₄ vs their European managed pasture sites
- Will use ERA5 as primary fallback strategy (not optional validation)
- NWFP-specific management events not available in their dataset

**Notes:**
This paper validates ERA5 substitution (D-08) and directly confirms that RFR is the appropriate gap-filling approach for the exact ecosystem type at NWFP. Completing this replication produces the gap-filled `greenhouse.csv` CH₄ column needed for all downstream forecasting work.

---

### R-04 — Partridge et al. (2024)

**Full citation:**
> Partridge, T., Li, B., Alhnaity, B., & Meng, Q. (2024). Utilizing Machine Learning to Understand and Predict Methane Emissions in Cattle Farming with Farm-Scale Environmental and Biological Variables. In *Proc. ICCA 2024*. https://doi.org/10.1109/ICCA62237.2024.10927962

**What is being replicated:**
The Gradient Boosting + SHAP pipeline applied to NWFP data, using farm environmental and biological variables to predict GreenFeed-measured cattle CH₄ (g/day and g/kg live weight gain). This establishes the **performance ceiling for animal-scale prediction at NWFP** and serves as the direct prior work baseline that the EC-based pipeline must contextualise against.

**Notebook:** `notebooks/03_gap_filling/`

**Status:** planned

**Target metrics (from paper):**
- Gradient Boosting: r=0.619, RMSE=51.8 g/day
- g/kg live weight gain: r=0.562, RMSE=65.9
- XAI/SHAP applied to quantify driver contributions

**Our results:**
_[Fill in once run]_

**Deviations from paper:**
- We use EC flux data (not GreenFeed) — this is an intentional and critical distinction (see D-09)
- Our target variable is ecosystem-scale flux (nmol m⁻² s⁻¹ or equivalent), not g/day per animal
- Results will be fundamentally incomparable in absolute terms; the replication establishes methodological familiarity with NWFP data and SHAP workflow

**Notes:**
This is the most important paper to understand before starting: it shows what ML on NWFP data looks like, which variables matter (animal weight, breed, season, weather), and what r~0.6 looks like in practice for this site. The SHAP analysis is directly replicable. This paper motivates why EC is the better measurement paradigm (D-09): r=0.619 at animal scale leaves substantial room for improvement, and the EC signal captures processes GreenFeed cannot.

---

_[Copy the entry template above for each new replication]_
