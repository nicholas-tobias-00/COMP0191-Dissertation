# Report outline and evidence audit

**Status:** working editorial brief for the first Markdown draft  
**Scope:** review of `report/Outlines`, current result summaries and supporting literature  
**Constraint:** do not transfer this text into the LaTeX report until the argument and evidence are agreed

## 1. Editorial verdict

The eight chapter outlines already describe a coherent dissertation, but the report will be stronger if it is framed as one connected evaluation rather than five partly separate investigations. The central question is:

> Can pretrained tabular models turn a severely incomplete farm-scale eddy-covariance methane record into useful gap fills, multi-step forecasts, and bounded what-if sensitivities?

This framing connects the three modelling stages:

1. gap filling reconstructs missing historical observations;
2. recursive forecasting tests whether the reconstructed data support prospective prediction; and
3. scenario analysis uses the fitted response surface for conditional sensitivity experiments.

The final stage is not a validated forecast of future methane emissions. It is a model-conditional what-if analysis, limited by covariate shift, high out-of-applicability rates, model-form uncertainty and the absence of future ground truth.

The strongest empirical story is not simply that one model wins. It is that pretrained tabular models are unusually competitive under severe missingness and small effective sample sizes, while their point predictions, uncertainty estimates and scenario responses still require different kinds of validation.

## 2. Recommended research-question structure

The current five questions can be retained, but three points should be corrected:

- RQ1 currently contains several questions and risks becoming a literature-review checklist.
- RQ4 promises to “decode interactions”, although no SHAP-interaction or equivalent interaction analysis has been completed.
- RQ5 asks about structural requirements that are discussed conceptually rather than evaluated in a controlled experiment.

A tighter four-question structure would be:

1. **Task and evidence gap:** How do methane gap filling, forecasting and scenario simulation differ, and what evidence is missing for managed temperate grasslands?
2. **Predictive performance:** How do statistical, ensemble, sequence and pretrained tabular models compare for gap filling and recursive multi-step forecasting under severe target missingness?
3. **Drivers and uncertainty:** Which variables drive the selected models, and how do predictive uncertainty and interval reliability vary across towers, emission magnitudes and livestock regimes?
4. **Conditional scenarios:** What methane responses do the selected model and bias-corrected climate trajectories imply under plausible livestock, grazing and fertiliser interventions, and where do applicability limits prevent strong interpretation?

If the five-question structure is retained, revise RQ4 from “decode interactions” to “identify influential drivers and characterise uncertainty across environmental and management regimes”. Treat the structural requirements in RQ5 as design lessons derived from the pipeline, not as experimentally proven requirements.

## 3. Evidence-safe headline findings

### 3.1 Data and missingness

- The consolidated hourly table contains approximately 70,153 rows and 449 columns.
- The full feature matrix is about 39.4% missing, but this is not the relevant headline for the target. The methane-flux target is missing for roughly 56–88% of timestamps depending on tower.
- Approximate valid target coverage is 12.1% at Tower 2, 44.6% at Tower 4 and 25.6% at Tower 9.
- Tower 2 includes an approximately 1,675-day gap from May 2019 to January 2024 associated with land-use and analyser relocation. It should not be described as a routine sensor outage.
- The source record is multi-year and half-hourly, but it is not “seven years of continuous half-hourly flux data”. The modelling table harmonises half-hourly flux, higher-frequency environmental measurements and daily management records to an hourly index.

### 3.2 Gap filling

Under the held-out gap-cross-validation protocol, the TabICL-solo configuration achieved:

| Tower | R² | OLS/correlation R² | RMSE | MAE |
|---|---:|---:|---:|---:|
| T2 | 0.676 | 0.686 | 77.60 | 30.03 |
| T4 | 0.428 | 0.429 | 99.18 | 41.57 |
| T9 | 0.423 | 0.430 | 110.01 | 50.48 |

The production random-forest model achieved hourly held-out R² values of 0.576, 0.404 and 0.426 at T2, T4 and T9. TabICL-solo therefore improves two towers and is effectively tied at T9, while the random forest remains the operationally complete production pipeline.

The outline currently labels the range 0.40–0.58 as daily-resolution performance. That range is hourly random-forest performance. The corresponding daily random-forest medians across the five gap scenarios were approximately 0.698, 0.504 and 0.485. Hourly and daily results must not be compared without an explicit change-of-resolution caveat.

For comparison with Zhu et al., use the OLS/correlation form of R² because that paper reports a squared-correlation/linear-regression statistic. Describe Zhu et al. as a published same-site comparator, not a performance ceiling: the feature set, years and validation design differ.

Production fills are in-sample reconstructions. Accuracy claims must come only from held-out gap cross-validation.

The available hourly uncertainty results compare quantile random forest and an earlier TabICL configuration. They show roughly nominal pooled coverage after calibration, but interval width did not reliably increase with real blackout length. These intervals have not yet been rerun for the later TabICL-solo accuracy champion, so the report must not imply that champion accuracy and reported UQ belong to one identical fitted model.

### 3.3 Recursive forecasting

The standing selected champion is TabPFN with species-aware management features:

- climatology-scaled MASE: **0.7150**;
- aggregate R²: **−0.0382**.

The negative R² and the MASE below one are not contradictory. The model improves mean absolute error relative to the seasonal-climatology baseline while still failing to explain aggregate variance under the selected R² calculation. Both results should be reported.

Several TabPFN configurations are almost tied. A bodyweight configuration achieved MASE 0.7149, and a later fertiliser configuration achieved 0.7142 but was not adopted because the gain was negligible and R² worsened. Therefore use “standing selected champion among near-tied TabPFN variants”, not “beats every model and configuration”.

Within this experiment, the broad ordering is:

> pretrained tabular foundation models > locally fitted ensembles > tested sequence models

This is a project-specific empirical result, not a general law. The explanation that pretraining compensates for missingness is plausible but has not been isolated by an ablation. Present it as an interpretation or hypothesis. Zeng et al. support the importance of simple baselines in long-horizon forecasting, but they do not directly establish the tree-versus-deep-learning mechanism proposed here.

The forecasting chapter should incorporate the newer uncertainty results:

- ordinary conformal calibration brought aggregate coverage close to 90% at T4 and T9, while T2 could not be calibrated robustly;
- flat intervals with acceptable aggregate coverage missed about three quarters of the highest-magnitude days;
- conformalised quantile regression improved spike coverage to roughly 79% for TabICLv2 and 57% for TabPFN, with a cost in normal-day coverage;
- livestock-stratified intervals were substantially narrower in low-livestock regimes, but tier-specific coverage remained imperfect.

This supports a more useful conclusion than “coverage reached 90%”: aggregate coverage can conceal failure exactly where emissions are largest.

### 3.4 Interpretability

The completed permutation-importance analysis is for the TabPFN species-aware champion. Its leading variables are:

1. livestock-unit density;
2. cattle density;
3. active grazing;
4. total liveweight;
5. growing-season status.

Sheep and lamb densities rank much lower. Tower 2 differs, with meteorological variables dominating its leading features. This heterogeneity should be retained rather than forcing one global driver story.

The result converges with earlier native-importance, SHAP-magnitude and statistical analyses, but no formal SHAP-interaction analysis has been identified. It supports statements about influential drivers, not decoded causal interactions.

The chapter outline currently attributes permutation-importance results to TabICLv2. That analysis has not been executed for TabICLv2. Either describe the completed result as TabPFN-specific and acknowledge the model mismatch with the scenario engine, or run a separate TabICLv2 importance analysis before making a cross-model claim.

### 3.5 Scenario analysis

The final S-06 files exist and should replace the mixture of S-04/S-05 preliminary results in the outline. The scenario engine uses TabICLv2 with the reduced `FX_A_SPECIES` feature set. Its climatology-scaled MASE is approximately 0.7588, compared with approximately 0.7405 for the corresponding fuller TabICLv2 configuration: a deterioration of about 0.0183, or 2.5%. The exact full-model comparator must be named consistently because a nearby `TabICLv2+ALL` result is 0.7385.

Key S-06 response summaries under SSP2-4.5 are:

| Intervention | T2 | T4 | T9 |
|---|---:|---:|---:|
| halve all livestock | −15.2% | −26.2% | −33.9% |
| raise all livestock to literature ceiling | +25.4% | +19.9% | +13.5% |
| raise all livestock to tower-specific historical maximum | +53.4% | +125.5% | +149.8% |
| extend grazing by four weeks | +3.8% | +18.6% | +18.6% |

Fertiliser-rate and fertiliser-frequency changes were small and sign-inconsistent across towers. They should be reported as a near-null model response, not omitted because the result is less narratively convenient.

The Tower 2 level definition in the outline is incorrect. The value 2.13 was the peak of an older smoothed three-times scenario, not the tower's historical maximum. The implemented Tower 2 historical maximum is 4.511 livestock units per hectare, above the literature ceiling of 3.0. The implemented historical maxima are 4.511, 4.987 and 5.652 for T2, T4 and T9.

SSP effects were generally small relative to management perturbations, but “below 1% throughout” is too strong: Tower 2 reaches roughly 2.9% divergence late in the horizon. Use a tower- and horizon-qualified statement.

The scenario climate required material bias correction: simulated rainfall was roughly four times too high before correction, temperature was around 2–3.5 °C too cool, and radiation was modestly low. Report the correction method and cite an appropriate climate-bias-correction source.

Applicability is the main constraint. Baseline out-of-applicability rates remained around 61–65% in the livestock scenarios and were also high for grazing and fertiliser experiments. Scenario outputs therefore describe conditional sensitivity of the fitted model, not validated future methane emissions.

The reported value above 99% in the scenario UQ workflow measures **interval availability**, not empirical coverage. Coverage cannot be calculated because future outcomes do not yet exist. Do not label it PICP or coverage.

## 4. Chapter-by-chapter drafting brief

### Chapter 1 — Introduction

Keep the global-to-local funnel, but make the quantitative context precise:

- agriculture contributed about 49% of UK methane emissions in 2024;
- livestock systems contribute about 32% of anthropogenic methane globally;
- use the appropriate biogenic methane global-warming potentials if numerical GWP values are included;
- describe North Wyke as representative of temperate high-rainfall livestock systems and soils common in Western Britain, not of all Western Europe.

Replace “continuous seven-year record” with “multi-year monitoring record with substantial missingness”. Replace the incorrect daily R² range. State novelty as the outcome of the structured review and targeted top-up search, rather than an unprovable universal first:

> No study identified in the structured review and targeted search evaluated ML-based multi-step forecasting of ecosystem-scale eddy-covariance methane flux in a managed temperate grassland.

End the chapter with one pipeline-level contribution and a short list of empirical contributions. Avoid presenting every notebook output as an independent contribution.

### Chapter 2 — Background and Related Work

Organise the chapter around the distinction between:

- retrospective gap filling;
- prospective multi-step forecasting; and
- conditional scenario simulation.

Fakeye et al. support farm digital-twin architecture, updating, forecasting, scenario design and uncertainty. They do not directly identify methane forecasting as a missing North Wyke module. Purcell et al. support the broader claim that implementation of agricultural simulation and digital-twin frameworks remains limited.

Use Zeng et al. only for the narrower point that sophisticated sequence models should be benchmarked against strong simple alternatives. Introduce TabPFN, TabICL and TabICLv2 with primary sources. Add sources for SPACSYS and climate bias correction rather than relying on similarly named but unrelated bibliography entries.

### Chapter 3 — Data

Define target coverage and whole-matrix missingness separately. Explain why Tower 2 is structurally different. Include a compact data-provenance table giving source resolution, aggregation rule, units, coverage and intended role.

Verify units for vapour-pressure deficit, photosynthetic photon flux density and soil heat flux against the source guides before drafting them as facts. Explain how target-derived features are prevented from leaking information across gaps and forecast cut-offs.

### Chapter 4 — Gap-Filling

Separate three questions:

1. Which model performs best under the common held-out gap protocol?
2. Which pipeline is used for production reconstructions and why?
3. How reliable are its uncertainty intervals across towers and gap lengths?

Keep metrics at one temporal resolution within each table. Make the OLS-R² comparison with Zhu explicit. State that fertiliser additions degraded gap-filling performance. Treat the failure of interval width to track blackout length as a substantive negative result.

### Chapter 5 — Forecasting

Lead with the recursive rollout design and the climatology-scaled MASE definition. State why MASE is primary and R² remains diagnostically important. Show the near-tie among leading TabPFN configurations before naming the selected champion.

Add the spike-coverage and livestock-stratified UQ results. Include a leakage and circularity audit for features derived from target history, gap-filled values or management records. Avoid claiming statistical significance unless formal tests are added.

### Chapter 6 — Scenario Projection

Use one final S-06 scenario definition throughout. Present interpretability as motivation for the tested intervention axes, not causal proof that manipulating those variables will produce the modelled response.

Describe bias correction, scenario levels, transition timing, aggregation and applicability checks before showing response percentages. Report livestock, grazing and fertiliser results, including near-null effects. Use “conditional response” or “model sensitivity”, not “future prediction”.

### Chapter 7 — Discussion

Structure the discussion around four claims:

1. pretrained tabular models were strongest under this project's data constraints;
2. good aggregate error does not guarantee reliable extreme-event uncertainty;
3. management variables dominate many fitted responses, but importance is not causality;
4. high scenario covariate shift limits policy interpretation.

The main limitation is not merely sample size. It is the combination of non-random missingness, site heterogeneity, dependence on reconstructed targets, recursive error propagation, model-form uncertainty and absent future ground truth.

### Chapter 8 — Conclusion

Answer each final research question explicitly in one paragraph. Separate demonstrated findings from interpretations and recommended next steps. Avoid “significant” unless it denotes a tested statistical result; use “substantial”, “small” or the reported effect size otherwise.

## 5. Bibliography audit

`references_mendeley.bib` is useful because it contains several sources absent from the original report bibliography, including TabICL, TabICLv2, Hoo and Semenov. It is not ready to merge unchanged.

Two citation keys are duplicated:

- `Purcell2023` refers to two different publications;
- `Zhu2023` refers to two different publications.

The keys must be made unique before use. The Hoo entry is dated 2026, while the outline currently says 2025. The file does not contain a suitable SPACSYS reference, and its `Wu2022` entry concerns GAN-based data imputation rather than SPACSYS. A primary SPACSYS source and a primary climate-bias-correction source still need to be added.

The original `references.bib` also lacks primary entries for TabPFN, TabICL, TabICLv2, Hoo, Semenov and the implemented bias-correction method. During drafting, use citation-key placeholders only where a clean, unique key has been verified.

## 6. Source-backed contextual claims for the draft

Useful authoritative sources for the introductory context include:

- [Defra, Agriculture in the United Kingdom 2025: agri-environment](https://www.gov.uk/government/statistics/agriculture-in-the-united-kingdom-2025/chapter-11-agri-environment)
- [FAO, enteric methane](https://www.fao.org/in-action/enteric-methane/en/)
- [IPCC AR6 WGIII Chapter 2 supplementary material](https://www.ipcc.ch/report/ar6/wg3/downloads/report/IPCC_AR6_WGIII_Chapter02_SM.pdf)
- [Rothamsted, North Wyke Farm Platform](https://www.rothamsted.ac.uk/national-capability/north-wyke-farm-platform)
- [Defra, structure of the agricultural industry](https://www.gov.uk/government/statistics/agriculture-in-the-united-kingdom-2025/chapter-2-structure-of-industry)
- [Defra, English regional agricultural profiles](https://www.gov.uk/government/statistics/agricultural-facts-england-regional-profiles/agricultural-facts-summary)

The literature discussion should prefer primary papers and repository records, including:

- [Fakeye et al., farm-scale digital-twin framework](https://repository.rothamsted.ac.uk/item/99369/towards-a-framework-for-farm-scale-digital-twin)
- [Purcell and Neubauer, digital twins in agriculture](https://repositum.tuwien.at/handle/20.500.12708/139747)
- [Zhu et al., methane gap filling at North Wyke](https://www.sciencedirect.com/science/article/pii/S016819232300059X)
- [Irvin et al., gap-filling benchmark repository record](https://escholarship.org/uc/item/2jd650hp)

## 7. Unresolved items before the LaTeX transfer

1. Confirm the dissertation word limit and required front-matter structure from the current programme handbook or Moodle specification.
2. Decide whether to retain five research questions or adopt the tighter four-question structure.
3. Decide whether the production random forest or the held-out TabICL-solo champion is the principal gap-filling model in the narrative.
4. Reconcile the exact full-feature TabICLv2 comparator for the scenario feature-ablation cost.
5. Either run TabICLv2 interpretability or state clearly that the completed importance analysis belongs to TabPFN.
6. Do not attach the older hourly UQ results to the later TabICL-solo champion without a rerun.
7. Verify the data-dictionary units and document all target-derived feature safeguards.
8. Repair duplicate bibliography keys and add missing primary sources before citations are moved into LaTeX.

## 8. Drafting rules

- Report metric name, temporal resolution, validation design and model configuration together.
- Do not mix preliminary S-04/S-05 scenario values with final S-06 results.
- Distinguish prediction-interval availability from empirical coverage.
- Distinguish feature importance from causal effect.
- Distinguish held-out validation from in-sample reconstruction.
- Distinguish climatology-scaled MASE from persistence-scaled or tower-level MASE variants.
- Qualify all scenario outputs as model-conditional sensitivities.
- Preserve negative and null results when they constrain the interpretation.
