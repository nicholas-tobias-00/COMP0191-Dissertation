# FCH₄ Drivers & Methods — Literature Review and Feature-Ingestion Plan

**Purpose:** explain why gap-filling R² is stuck near zero, what the literature says actually drives EC CH₄ flux (especially over grazed pasture), and which features to ingest next at NWFP. Feeds `04_feature_engineering`.

**Date:** 2026-06-16

---

## 1  Why R² is stuck — the source-attribution insight

The replications (R-01→R-03) and the CO₂-augmentation (03b) converge on one conclusion: **the limiting factor is feature availability, not algorithm.** The deeper reason is *what* an EC CH₄ signal over grazed pasture is made of:

> **EC CH₄ flux at a grazed-pasture tower = a small, slowly-varying soil flux + large, episodic spikes whenever animals are in the flux footprint.**

Felber et al. (2015) quantified this directly with GPS-collared cows + a footprint model: **CH₄ fluxes with cows in the footprint were up to two orders of magnitude larger than the bare-ecosystem flux** (mean ≈ 423 g CH₄ head⁻¹ d⁻¹). The animal term is enteric (breath/eructation) and dwarfs the soil term.

Implication for our models:
- Meteorology (SW, TA, VPD, …) describes the **soil** process reasonably but is **blind to the animal process**, which is the dominant variance. Hence met-only R² ≈ 0.
- FCO₂ helped (03b) because it is a weak proxy for ecosystem state/activity — but it is still not the animal signal.
- The R² metric is dominated by a handful of animal-driven spikes; a model that nails the soil baseline but misses spikes scores **negative** R². So part of the "low R²" is a **metric + target-distribution** artefact, not pure model failure (see §4).

**The single highest-value missing feature is therefore: are animals in the footprint, and how many?**

---

## 2  What the literature says drives EC CH₄

| Driver | Evidence | Available at NWFP? |
|---|---|---|
| **Animals in footprint (enteric)** | Felber 2015: dominant, ×100 over soil flux at grazed pasture | **Yes** — `cattle_/sheep_/lamb_Catchment N` (daily) |
| **Soil temperature** | Irvin 2021 (our R-01): most important predictor across 17 FLUXNET-CH4 sites; controls methanogenesis | Yes — `TS_*` + per-catchment `Soil Temperature @ 15cm` |
| **Water table / soil moisture (anaerobiosis)** | Irvin 2021, Kim 2020 (our R-03): water-table lags key at wet sites; wetness → methanogenesis, dryness → methanotrophic uptake | Yes — `Soil Moisture @ 10cm [Catchment N]`, precip |
| **Management: slurry/manure, fertiliser, grazing, cutting** | Zhu 2023b: farm management is "a key unresolved challenge causing abrupt, unanticipated flux changes"; excreta/slurry add labile C+N | Yes — `Field_Event_Data_Format_1.csv` |
| **Productivity / GPP / NEE** | FLUXNET-CH4 syntheses: GPP couples substrate supply to methanogenesis | Partly — FC/NEE (added in 03b); GPP derivable |
| **Turbulence / friction velocity** | Quality + footprint extent; USTAR gates valid EC | Yes — `USTAR_*` |
| **Wind direction / footprint geometry** | Felber 2015: attribution depends entirely on which source area is upwind | **Yes** — `WD_0_0_1 [Tower N]` |

Methodological notes from the same literature:
- **Decision-tree ensembles perform best** for gap-filling; **ANN comparable when given the full predictor set** (Irvin 2021 — matches our R-03 ANN result). 
- **Raw ML uncertainties are underestimated** and must be calibrated (Irvin 2021 → our D-06 UQ requirement).

---

## 3  Prioritised feature-ingestion plan (NWFP-specific)

All columns below were confirmed present in `consolidated_hourly.csv`.

### P1 — Footprint-weighted livestock density  ⭐ highest expected impact
The dominant missing driver. Build from livestock counts × wind direction:
- **Base:** `cattle_Catchment N`, `sheep_Catchment N`, `lamb_Catchment N` for the tower's own catchment (Tower N = Catchment N, D-18). Convert head-count → stocking density (head ha⁻¹) and a **grazing-presence indicator**.
- **Footprint weighting:** gate by `WD_0_0_1 [Tower N]` — weight each upwind catchment's animal count by whether that field lies in the upwind sector (even a coarse 8-sector mask is a strong start; a Kljun/Kormann–Meixner footprint model is the rigorous version).
- **Caveat:** livestock counts are **daily** (CH₄ is hourly) and there are no NWFP GPS collars (unlike Felber), so within-day animal position is unresolved — the feature captures presence/density, not sub-daily movement. Still expected to be the biggest single lever.

### P2 — Management-event features
From `Field_Event_Data_Format_1.csv` (fertiliser, slurry, spraying, reseeding, cutting):
- **Time-since-event** with exponential decay (e.g. `exp(-Δt/τ)`) per event type — captures the transient flux pulse after slurry/fertiliser/cutting.
- **Event magnitude** (N rate, slurry volume) where recorded.

### P3 — Wind direction & footprint quality (standalone)
- `sin/cos(WD)` cyclical encoding; an "own-field-in-footprint" indicator; `USTAR`-based footprint-validity flag. Useful even before full P1 weighting.

### P4 — Antecedent moisture / wetness memory
- Rolling precipitation sums (7/30-day) from `Precipitation (mm) [Catchment N]`; antecedent soil-moisture means and the SWC/TS lags already trialled in R-03 (extend horizons). Wetness ⇒ anaerobic ⇒ methanogenesis; dryness ⇒ uptake.

### P5 — Better soil temperature & productivity
- Replace the cross-tower Tower 9 TS proxy (D-16) with per-catchment `Soil Temperature @ 15cm [Catchment N]` where coverage allows. 
- Add a GPP/productivity proxy (from NEE partitioning of FC, or seasonal phenology) — substrate supply for methanogens.

### P6 — Soil/water chemistry (exploratory)
- From `measurements.csv`: nitrate/ammonium, dissolved O₂, conductivity — indicators of redox/anaerobic state that modulate CH₄ production vs oxidation.

---

## 4  Method & evaluation recommendations

The bottleneck is features, but the **bimodal source structure** (soil baseline vs animal spikes) also argues for changes to model framing, target, and metric:

1. **Keep gradient-boosted trees as the workhorse** (RF/XGBoost/LightGBM) — best for tabular, interaction-rich gap-filling. Use ANN/MLP as a strong comparator (it already won at Tower 4 in R-03/03b).
2. **Consider a two-part / hurdle model:** (a) classifier "animals in footprint / high-emission regime?", then (b) magnitude regressor — matches the physical two-source structure better than one monolithic regressor.
3. **Reconsider target & metric.** The spiky, heavy-tailed target makes plain R² fragile:
   - Evaluate on a **signed-log / asinh-transformed** flux, and/or model soil-baseline and animal-spike components separately.
   - Report **RMSE, MAE, NSE, and regime-stratified scores** alongside R², and consider spike-capture metrics. A negative R² with good baseline RMSE is a metric artefact, not necessarily a bad model.
4. **UQ is mandatory (D-06):** quantile regression / conformal prediction — raw ML CH₄ uncertainties are known to be underestimated (Irvin 2021).
5. **Forecasting phase (later):** once features are rich, LSTM/TFT for temporal structure — but feature richness will matter more than architecture.

---

## 5  Concrete next experiment

Add **P1 (footprint-weighted livestock) + P2 (management events)** to the current best configuration (R-02-CO2 RFm / R-03 ANN, Towers 4 & 9) and measure the R²/RMSE delta against the 03b baseline. Hypothesis: the livestock feature moves Tower 4 decisively positive and, unlike met/FCO₂, also helps the animal-driven spike regime that currently dominates the residuals.

Order of build: P3 (cheap, enables P1) → P1 → P2 → P4 → P5 → P6.

> **Update (F-01 done):** this experiment ran — see `F01_results.md` / `feature_engineering_summary.md`. Result: **livestock (P1) is the #1 SHAP driver** at Tower 4 and lifts short-gap R² +0.156 → +0.256, confirming the hypothesis. Management (P2) as implemented overfit (collapsed data-poor Tower 9); prune and re-run leave-one-group-in next.

---

## Sources

- Irvin, J. et al. (2021). *Gap-filling eddy covariance methane fluxes: comparison of machine learning model predictions and uncertainties at FLUXNET-CH4 wetlands.* Agric. Forest Meteorol. — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0168192321002124) · [NERC open archive](https://nora.nerc.ac.uk/id/eprint/530810/)
- Felber, R., Münger, A., Neftel, A., Ammann, C. (2015). *Eddy covariance methane flux measurements over a grazed pasture: effect of cows as moving point sources.* Biogeosciences 12, 3925–3940 — [open access PDF](https://bg.copernicus.org/articles/12/3925/2015/bg-12-3925-2015.pdf)
- Cardenas, L. et al. (2022). *CO₂ fluxes from three different temperate grazed pastures using eddy covariance.* — [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S004896972201912X)
- Soil GHG budget of intensively managed grazing systems (2020) — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0168192320300629)
- Project replications: `../03_gap_filling/gap_filling_summary.md`, `../03b_gap_filling_CO2/co2_augmented_summary.md` (Zhu 2023a = R-02, Kim 2020 = R-03).
