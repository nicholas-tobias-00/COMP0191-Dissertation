# Initial report draft

> **Working status.** This is a prose-first Markdown draft derived from the chapter outlines and the evidence audit. It deliberately retains qualifications, citation prompts and unresolved decisions. It is not yet formatted for LaTeX and should not be treated as the final numerical record. The proposed consolidation to four research questions is provisional.

## Provisional title

**From incomplete observations to bounded what-if analysis: pretrained tabular models for methane flux modelling in managed temperate grassland**

## Abstract

Methane mitigation in livestock agriculture requires evidence at the scale at which environmental conditions and management decisions interact. Eddy-covariance systems can provide continuous ecosystem-scale methane-flux measurements, but the resulting records are sparse, heterogeneous and interrupted by long periods of missing data. This study examines whether modern pretrained tabular models can turn such a record into useful historical reconstructions, prospective forecasts and conditional scenario analyses. The case study uses the North Wyke Farm Platform in south-west England, combining methane flux, meteorological, soil, vegetation and livestock-management data from three instrumented farmlets on a common hourly index.

The study distinguishes three tasks that are often conflated: retrospective gap filling, recursive multi-step forecasting and model-based what-if simulation. Statistical models, tree ensembles, sequence models and pretrained tabular foundation models were evaluated with task-appropriate held-out protocols. For gap filling, TabICL-solo achieved held-out R² values of 0.676, 0.428 and 0.423 at Towers 2, 4 and 9, improving on the operational random forest at two towers and matching it at the third. For recursive forecasting, the selected species-aware TabPFN configuration achieved a climatology-scaled MASE of 0.715, although its aggregate R² remained slightly negative. This combination shows that it reduced absolute error relative to seasonal climatology without reliably explaining variance. Aggregate prediction-interval coverage also concealed poor performance on the largest methane-flux days, motivating magnitude- and livestock-aware uncertainty analysis.

Management variables—particularly livestock-unit density, cattle density, grazing activity and liveweight—were influential in the selected forecasting model. Bias-corrected climate and management experiments consequently produced substantial livestock sensitivities, especially at Towers 4 and 9, but fertiliser effects were small and inconsistent. High out-of-applicability rates and the absence of future ground truth mean that these outputs are conditional model sensitivities, not validated projections. The study concludes that pretrained tabular models are promising for small, incomplete environmental datasets, while operational use requires explicit separation of reconstruction, prediction and scenario evidence, together with validation that targets extreme events and covariate shift.

# 1. Introduction

## 1.1 Methane mitigation needs farm-scale evidence

Methane is a short-lived but powerful greenhouse gas, making reductions in methane emissions an important route to near-term climate mitigation. Agriculture accounts for a large share of this burden. Livestock systems contribute about 32% of anthropogenic methane emissions globally, primarily through enteric fermentation and manure management [CITATION: FAO enteric methane]. In the United Kingdom, agriculture contributed approximately 49% of national methane emissions in 2024 [CITATION: Defra, Agriculture in the United Kingdom 2025]. These figures make livestock management a central mitigation problem, but national inventories alone cannot show how emissions respond to local weather, soil conditions, grazing activity and herd composition.

This local evidence is especially relevant in the United Kingdom, where permanent grassland occupies a large fraction of agricultural land and where pasture-based livestock systems are economically and environmentally important. The North Wyke Farm Platform in south-west England provides a rare research setting in which environmental monitoring and farm-management records are collected together across approximately 63 hectares of temperate, high-rainfall grassland [CITATION: Rothamsted North Wyke Farm Platform]. Its soils and production system are relevant to livestock areas of Western Britain, although one platform cannot represent all grassland agriculture in Britain or Western Europe.

Eddy covariance offers an ecosystem-scale view of methane exchange, avoiding some of the spatial limitations of chambers or animal-level measurements. Its apparent continuity is deceptive, however. Quality-control filtering, analyser downtime, weak turbulence, changes in land use and instrument relocation can remove a large proportion of the observations. At North Wyke, valid methane-flux coverage differs markedly among the three analysed towers, and Tower 2 contains a multi-year interruption. The resulting modelling problem is therefore not conventional prediction from a clean, regularly sampled time series. It is learning from a heterogeneous record in which the response variable is absent for much of the nominal monitoring period.

## 1.2 Three different modelling tasks

This dissertation separates three tasks that answer different scientific questions.

**Gap filling** estimates methane flux at missing timestamps inside a historical record. It may use information from both sides of a missing block and is evaluated by hiding observations whose true values are known. Its purpose is reconstruction.

**Forecasting** predicts methane flux beyond an historical cut-off. Future target values are unavailable at prediction time, so evaluation must reproduce this information constraint. In a recursive forecast, previous predictions may become inputs to later steps, allowing errors to propagate. Its purpose is prospective prediction.

**Scenario simulation** changes selected future drivers and observes the response of a fitted model. Because the counterfactual outcomes are unobserved, a scenario cannot be validated in the same way as a historical forecast. Its purpose is conditional sensitivity analysis, not direct proof of a management intervention's causal effect.

Keeping these tasks separate is essential. A model that interpolates a historical gap well may not remain accurate in a recursive forecast, and a model with acceptable historical forecast error may still extrapolate unreliably under an intervention that moves its inputs away from the training distribution.

## 1.3 Research gap

Previous work has demonstrated machine-learning gap filling for ecosystem fluxes, including methane at North Wyke [CITATION: Zhu et al. 2023; Irvin et al.]. Agricultural digital-twin research also identifies forecasting, updating, scenario design and uncertainty as important components of decision-support systems [CITATION: Fakeye et al.; Purcell and Neubauer]. Yet these bodies of work do not by themselves establish a validated pipeline from sparse ecosystem-scale methane observations to multi-step forecasts and bounded management scenarios.

No study identified in the structured review and targeted top-up search evaluated machine-learning-based multi-step forecasting of ecosystem-scale eddy-covariance methane flux in a managed temperate grassland. The recent development of pretrained tabular foundation models offers a plausible way to address this small-data setting. Unlike locally trained deep sequence models, these models bring information learned from many synthetic or real tabular tasks, although whether that advantage persists under environmental missingness and temporal rollout must be tested rather than assumed.

## 1.4 Aim and research questions

The aim is to evaluate whether pretrained tabular models can support a connected methane-modelling pipeline under severe target missingness, while identifying where validation evidence is insufficient for operational or policy claims.

The working research questions are:

1. How do methane gap filling, forecasting and scenario simulation differ, and what evidence is missing for managed temperate grasslands?
2. How do statistical, ensemble, sequence and pretrained tabular models compare for gap filling and recursive multi-step forecasting under severe target missingness?
3. Which variables drive the selected models, and how do predictive uncertainty and interval reliability vary among towers, emission magnitudes and livestock regimes?
4. What methane responses do the selected model and bias-corrected climate trajectories imply under plausible livestock, grazing and fertiliser interventions, and where do applicability limits prevent strong interpretation?

The contribution is both empirical and methodological. Empirically, the work benchmarks modern tabular foundation models against established alternatives at a data-rich but target-sparse farm platform. Methodologically, it shows why reconstruction accuracy, recursive forecast skill, interval reliability and scenario applicability must be evaluated separately. The resulting system is best understood as a digital shadow: it updates and analyses a physical system but does not yet form a fully validated, bidirectional digital twin.

# 2. Background and Related Work

## 2.1 Measuring and modelling methane flux

Methane flux in grazed systems is shaped by processes operating at different scales. Enteric emissions depend on animal number, species, liveweight, diet and activity. Soil methane exchange and transport respond to water status, temperature and microbial conditions. Atmospheric turbulence and footprint geometry determine which parts of a field contribute to an eddy-covariance observation. A farm-scale model must therefore combine continuous environmental variables with discrete or slowly changing management records.

Eddy covariance estimates vertical gas exchange from high-frequency covariance between turbulent wind and gas concentration. It provides a spatially integrated flux over a changing footprint, but observations are removed when physical or quality-control assumptions are violated. Missingness is consequently structured rather than purely random. Weather conditions, low turbulence and operational changes can affect both whether a measurement exists and the process being measured. Randomly deleting isolated observations is therefore an inadequate evaluation of a gap-filling method intended for long real-world outages.

## 2.2 Gap filling is retrospective reconstruction

Environmental gap filling has traditionally used marginal distribution sampling, interpolation, regression and process-informed methods. Machine-learning alternatives include random forests, gradient-boosted trees, recurrent neural networks and specialised imputation architectures. Tree ensembles are attractive because they learn nonlinearities, tolerate mixed feature types and require comparatively little preprocessing. Sequence models can represent temporal dependence directly, but their parameter counts and data requirements can become disadvantages when valid target observations are scarce.

Zhu et al. evaluated machine-learning methods for methane-flux gap filling at North Wyke and provide the closest published comparator for the present work [CITATION: Zhu et al. 2023]. Their methane results were weak relative to other flux targets, reinforcing the difficulty of this response. Direct numerical comparisons still require care: validation gaps, predictor availability, study years, temporal aggregation and even the definition of R² can differ. In this dissertation, the comparison is therefore contextual rather than a claim that one study supplies a fixed ceiling that a new model has surpassed.

A defensible gap-filling evaluation should reproduce realistic missing-block lengths, prevent target leakage across the held-out interval and report performance by site as well as in aggregate. It should also separate benchmark performance from production reconstruction. Once a model is trained on all available observations to fill actual gaps, the missing values have no ground truth and cannot supply a new accuracy statistic.

## 2.3 Forecasting is an information-constrained task

Forecasting differs because information after the forecast origin must not enter the model. One-step validation can overstate usefulness when the intended deployment requires a long horizon. Recursive rollout is more demanding: the model predicts the next value or block and then advances using information available at that stage, including its own earlier predictions where required.

Strong baselines are indispensable. A seasonal climatology can perform well when environmental signals are persistent, while persistence and autoregressive models expose whether a complex learner adds value beyond short-term continuity. Zeng et al. showed that simple linear models can outperform transformer architectures on several long-term time-series benchmarks [CITATION: Zeng et al.]. Their result does not prove that deep learning is unsuitable for methane, but it warns against treating architectural sophistication as evidence of skill.

The model families considered here span statistical forecasting, random forests and boosting, ensembles, sequence models and pretrained tabular foundation models. TabPFN learns a prior over tabular prediction tasks through pretraining and performs in-context inference on a new dataset [CITATION: primary TabPFN paper]. TabICL and TabICLv2 extend the foundation-model approach to larger or more flexible tabular settings [CITATION: Qu et al. 2025; Qu et al. 2026]. Their relevance to methane forecasting is empirical: they may be effective where the number of valid observations is small relative to the feature space, but they do not remove the need for time-aware validation.

## 2.4 Interpretability and uncertainty

Feature importance can reveal which inputs a model uses, but it cannot establish that changing an input will cause the predicted outcome. Correlated livestock variables are a particular challenge: livestock-unit density, cattle density and liveweight encode overlapping aspects of herd presence. Native tree importance, permutation importance and SHAP magnitude answer related but non-identical questions and can distribute importance differently among correlated variables.

Uncertainty evaluation also requires more than nominal aggregate coverage. Quantile models estimate conditional intervals directly, while conformal calibration uses held-out residuals to target finite-sample marginal coverage under exchangeability assumptions. Environmental time series strain those assumptions through autocorrelation, seasonality and distribution shift. A 90% interval can attain 90% coverage overall while failing on the high-emission events that matter most for mitigation. Coverage should therefore be stratified by tower, target magnitude and management regime, and reported together with interval width.

## 2.5 From digital shadow to scenario analysis

Agricultural digital twins are commonly described as linked representations of physical systems that support monitoring, updating, prediction and what-if analysis. Reviews find substantial conceptual activity but limited implementation and validation at farm scale [CITATION: Purcell and Neubauer]. The North Wyke framework proposed by Fakeye et al. includes data integration, forecasting, scenario design and uncertainty [CITATION: Fakeye et al.]. It does not, however, directly demonstrate the methane-forecasting component evaluated in this dissertation.

The present pipeline remains a digital shadow because information flows mainly from farm observations into an analytical representation. It does not automatically enact management decisions or continuously validate scenario outcomes. This distinction sets an appropriate standard for the scenario chapter: a useful shadow can expose model sensitivities and data gaps without claiming the causal or operational maturity of a full digital twin.

# 3. Data

## 3.1 Study site and sources

The North Wyke Farm Platform comprises self-contained pasture-based livestock systems with detailed environmental and management monitoring. This study uses three methane-flux towers, denoted T2, T4 and T9. The source data combine eddy-covariance methane flux, meteorology, soil measurements, vegetation indicators and records of livestock and farm operations. Source frequencies differ: flux observations are nominally half-hourly, several environmental sensors operate at sub-hourly resolution, and many management variables are daily or event based. All sources were aligned to a common hourly index using variable-appropriate aggregation and forward-filling rules.

The consolidated table contains approximately 70,153 hourly rows and 449 columns. About 39.4% of the complete feature matrix is missing, but target availability is much lower and more variable. Valid methane-flux coverage is approximately 12.1% at T2, 44.6% at T4 and 25.6% at T9, corresponding to target missingness of roughly 88%, 57% and 75%. The distinction matters: saying only that “the data are 39% missing” understates the problem faced by the supervised models.

Tower 2 contains an approximately 1,675-day interruption between May 2019 and January 2024 associated with land-use and analyser relocation. It is treated separately in interpretation because its observation regime, history and effective training size differ from the other towers.

## 3.2 Harmonisation and quality control

Flux data were filtered using the platform's quality-control information before modelling. Environmental and management sources were timestamp-normalised and mapped to tower or farmlet where appropriate. Continuous variables were aggregated by statistics consistent with their physical meaning; event and state variables were converted to hourly indicators or carried forward only where their meaning justified persistence.

The final methods chapter should include a data-provenance table listing each feature family, source system, native resolution, units, hourly transformation, coverage and role in the model. Units for vapour-pressure deficit, photosynthetic photon flux density and soil heat flux require final checking against the source guides. This is preferable to reproducing hundreds of column names in the main text; the full catalogue can be supplied as supplementary material.

Leakage control is central. Any rolling, lagged or target-derived feature used in gap filling must be constructed without exposing the held-out target block. Forecast features must use only information available at the forecast origin or supplied exogenously for the future period. Gap-filled target values should not be silently treated as observations when fitting or evaluating forecasts. The final implementation description will document these safeguards explicitly.

## 3.3 Exploratory findings

Methane flux is heterogeneous across towers, strongly right-skewed and punctuated by high-magnitude events. Seasonal environmental cycles are visible, but management changes do not align perfectly with them. Livestock variables are discrete or piecewise constant at hourly scale, whereas meteorological and soil variables evolve continuously. This mixture motivates tabular models capable of nonlinear response surfaces without requiring an extremely long complete sequence.

Exploratory correlations are descriptive rather than causal. Herd variables co-vary, environmental variables are seasonally confounded, and tower identity captures unmeasured differences in footprint and management. These constraints motivate tower-stratified evaluation and later comparison of several importance methods.

# 4. Gap-Filling

## 4.1 Evaluation design

Gap-filling models were evaluated by withholding known methane observations in contiguous blocks designed to resemble realistic outages. Training and feature construction excluded the held-out targets, and predictions were compared with their observed values. Performance was reported separately for each tower using R², an OLS/correlation form of R² where comparison with prior work required it, RMSE and MAE. Daily summaries were calculated separately and were not pooled with hourly metrics.

The model comparison included the operational random forest, boosted-tree alternatives, sequence or imputation architectures and TabICL variants. This distinction matters because the best benchmark model and the most complete production pipeline are not currently identical. The random forest has the established reconstruction and tooling workflow, whereas a later TabICL-solo evaluation produced the strongest held-out accuracy at two towers.

## 4.2 Point-prediction results

Table 4.1 gives the full standing comparison, together with the principal statistical and imputation floors. Values are median hourly sklearn R² across the five artificial-gap scenarios.

| Model | T2 | T4 | T9 | Role |
|---|---:|---:|---:|---|
| training-set mean | −0.003 | −0.001 | −0.001 | trivial floor |
| corrected MDS | −0.023 | −0.113 | −0.073 | literature baseline |
| RF, meteorology only | 0.052 | 0.036 | 0.059 | raw-driver baseline |
| MICE | 0.081 | 0.118 | 0.107 | multivariate imputer |
| HyperImpute | 0.509 | 0.336 | 0.354 | AutoML imputer |
| operational RFm | 0.576 | 0.404 | **0.426** | production champion |
| LightGBM | 0.522 | 0.410 | 0.422 | boosted tree |
| XGBoost | 0.551 | 0.349 | 0.369 | boosted tree |
| TabPFN | 0.459 | 0.401 | 0.402 | foundation model |
| TabICL, pooled | 0.558 | 0.423 | 0.364 | foundation model |
| SAITS | 0.341* | 0.275* | 0.263* | sequence imputer |
| bidirectional LSTM | 0.237 | 0.155 | 0.146 | sequence model |
| TabICL-solo | **0.676** | **0.428** | 0.423 | point champion |

The recalculated SAITS row does not reproduce an older summary, so it remains flagged pending artefact reconciliation. The wider result is nevertheless clear: MDS and MICE were weak; HyperImpute recovered much of the available signal; the tree and foundation models formed the leading group; and both tested sequence architectures trailed that group.

Table 4.2 reports the principal error metrics for the expanded final roster. The TabICL-solo values are medians of each metric across scenarios, so the R², RMSE and MAE medians need not originate from the same individual gap scenario.

| Model | Tower | R² | OLS R² | RMSE | MAE | MBE |
|---|---:|---:|---:|---:|---:|---:|
| operational RFm | T2 | 0.576 | 0.601 | 75.0 | 31.2 | — |
| operational RFm | T4 | 0.404 | 0.409 | 100.3 | 42.8 | — |
| operational RFm | T9 | 0.426 | 0.430 | 107.1 | 49.3 | — |
| LightGBM | T2 | 0.522 | 0.583 | 82.74 | 34.78 | 6.36 |
| LightGBM | T4 | 0.410 | 0.418 | 100.88 | 44.34 | 2.77 |
| LightGBM | T9 | 0.422 | 0.425 | 109.02 | 51.60 | 2.31 |
| XGBoost | T2 | 0.551 | 0.619 | 89.48 | 40.74 | 5.28 |
| XGBoost | T4 | 0.349 | 0.380 | 107.74 | 49.79 | 3.86 |
| XGBoost | T9 | 0.369 | 0.375 | 112.28 | 57.10 | 2.58 |
| TabPFN | T2 | 0.459 | — | 89.02 | 30.61 | 3.73 |
| TabPFN | T4 | 0.401 | — | 99.36 | 42.40 | 3.12 |
| TabPFN | T9 | 0.402 | — | 107.97 | 49.74 | 5.26 |
| TabICL, pooled | T2 | 0.558 | 0.587 | 76.62 | 30.82 | −0.59 |
| TabICL, pooled | T4 | 0.423 | 0.425 | 101.70 | 41.97 | 2.13 |
| TabICL, pooled | T9 | 0.364 | 0.368 | 108.55 | 50.59 | 3.65 |
| SAITS* | T2 | 0.341 | 0.562 | 92.30 | 36.72 | −2.74 |
| SAITS* | T4 | 0.275 | 0.289 | 104.41 | 42.27 | −5.11 |
| SAITS* | T9 | 0.263 | 0.304 | 119.33 | 54.34 | −3.10 |
| bidirectional LSTM | T2 | 0.237 | 0.280 | 93.99 | 33.65 | −0.49 |
| bidirectional LSTM | T4 | 0.155 | 0.204 | 115.73 | 49.07 | 3.10 |
| bidirectional LSTM | T9 | 0.146 | 0.189 | 127.50 | 54.29 | −2.99 |
| TabICL-solo | T2 | **0.676** | **0.686** | 77.60 | **30.03** | 5.78 |
| TabICL-solo | T4 | **0.428** | **0.429** | **99.18** | **41.57** | 2.54 |
| TabICL-solo | T9 | 0.423 | **0.430** | 110.01 | 50.48 | 7.02 |

The metric columns do not always select the same winner. At T2 the operational RF has lower RMSE than TabICL-solo, whereas TabICL-solo has higher R² and lower MAE. At T9 the operational RF retains a small R², RMSE and MAE advantage even though the OLS R² is nearly identical.

At daily resolution, median random-forest R² across five held-out gap scenarios was approximately 0.698 at T2, 0.504 at T4 and 0.485 at T9. These numbers are higher than or similar to the hourly values partly because daily aggregation suppresses high-frequency noise. They should not be used to imply that the model improved simply by changing the label on the same metric.

The OLS/correlation R² values provide the closest comparison with Zhu et al. The present results are materially stronger than the weak methane performance reported in that same-site study, but differences in predictors, periods and validation blocks prevent a controlled attribution of the improvement to model architecture alone.

Adding detailed fertiliser variables did not improve the gap-filling models. R² declined for both the random forest and TabICL configurations. This negative result suggests that greater feature detail can introduce sparsity, noise or unstable correlations without adding recoverable target information.

The developmental ablations reinforce that conclusion. Species-density and livestock-memory variants gave small T2 gains but no consistent three-tower improvement; nearest-target and wide lag/lead features degraded performance; TICA components were neutral; uncertainty width used as an input was harmful; and TabICL hyperparameter changes were within noise. Row-cap bagging was the main narrow exception, improving T4 from 0.428 to approximately 0.440 before plateauing. Averaging RF and TabICL predictions raised T4 R² to 0.445 but reduced the stronger TabICL result at T2.

## 4.3 Uncertainty and production use

Quantile random forest and TabICL interval estimates achieved approximately nominal pooled hourly coverage. Their uncalibrated results were:

| Model | Tower | n | 90% PICP | MPIW | corr(width, absolute error) |
|---|---:|---:|---:|---:|---:|
| quantile RFm | T2 | 12,384 | 0.941 | 162.70 | 0.644 |
| quantile RFm | T4 | 48,707 | 0.919 | 197.15 | 0.493 |
| quantile RFm | T9 | 28,189 | 0.898 | 213.67 | 0.504 |
| TabICL quantiles | T2 | 12,384 | 0.918 | 141.70 | 0.590 |
| TabICL quantiles | T4 | 48,707 | 0.901 | 184.25 | 0.509 |
| TabICL quantiles | T9 | 28,189 | 0.893 | 220.82 | 0.482 |

After split-conformal correction, individual gap-scenario coverage ranged from 0.876 to 0.912 for quantile RFm and from 0.888 to 0.910 for TabICL. Daily conformal PICP was 0.866, 0.923 and 0.861 at T2, T4 and T9, with corresponding widths of 104.72, 161.52 and 124.67.

Two cautions prevent a simple claim of successful uncertainty quantification. First, the available interval experiment belongs to an earlier TabICL configuration rather than the later TabICL-solo point-prediction champion. Second, interval width did not consistently increase with the length of real missing blocks. A narrow interval across a long blackout can therefore reflect the model's inability to represent epistemic uncertainty rather than genuine confidence.

The final production reconstruction should consequently retain provenance for every filled value: model identity, training period, gap length, interval method and applicability indicators. Reconstructed values are useful for downstream summaries, but they are not new observations. The held-out experiment—not agreement inside an unobserved production gap—supplies the evidence for accuracy.

# 5. Forecasting

## 5.1 Forecast design and baselines

The forecasting experiment advances from historical anchor dates into held-out future periods. Models receive only information available at each forecast step, and recursive designs reuse predicted quantities where required. Multiple anchors reduce dependence on a single favourable or adverse year. Seasonal climatology is the primary scaling baseline for MASE, while persistence and statistical models provide additional points of comparison.

The principal metric is climatology-scaled mean absolute error. A value below one means that the model's absolute error is lower than the seasonal-climatology error. R² is retained because it measures a different property: whether predictions reproduce variance around the observed mean. Both are required to avoid treating one form of skill as complete performance.

The comparison spans SARIMAX, random forests, gradient boosting, ensembles, tested deep sequence models, TabPFN and TabICL variants. All headline comparisons use the same rollout and aggregation protocol. Tower-level tables based on a persistence-scaled MASE are kept separate because they are not numerically interchangeable with the all-tower climatology-scaled result.

## 5.2 Forecast results

Table 5.1 gives the complete eleven-model observed-target comparison. Lower MASE, RMSE and WAPE are better; higher R² and correlation are better.

| Model | Selected configuration | MASE | R² | RMSE | WAPE | correlation |
|---|---|---:|---:|---:|---:|---:|
| TabPFN | BASE + bodyweight | **0.7149** | −0.095 | 56.16 | **0.880** | 0.356 |
| TabICLv2 | BASE + ALL | 0.7385 | −0.143 | 56.61 | 0.905 | 0.302 |
| ensemble, MASE weighted | BASE | 0.8019 | −0.163 | **52.25** | 0.991 | 0.379 |
| ensemble, unweighted | BASE | 0.8020 | −0.162 | 52.25 | 0.991 | 0.379 |
| XGBoost | BASE + ALL | 0.8038 | −0.201 | 52.30 | 0.988 | 0.364 |
| TFT | BASE + ALL | 0.8120 | −0.237 | 55.78 | 1.009 | 0.236 |
| LightGBM | BASE | 0.8166 | −0.214 | 52.83 | 1.014 | 0.371 |
| random forest | BASE | 0.8411 | −0.241 | 52.91 | 1.045 | **0.380** |
| SARIMAX | BASE | 0.8741 | −0.329 | 54.35 | 1.082 | 0.348 |
| LSTM | BASE + species | 0.9521 | −0.651 | 62.32 | 1.154 | 0.275 |
| DLinear | BASE + bodyweight | 1.1872 | −1.751 | 61.99 | 1.519 | 0.277 |

The ranking is metric dependent. TabPFN minimises climatology-scaled absolute error, but the ensembles have lower RMSE and the random forest has the highest correlation by a small margin. Every model has negative aggregate R², so none reliably reproduces total variance under this pooling convention.

The selected species-aware TabPFN configuration achieved MASE 0.7150 with aggregate R² −0.0382 under the current climatology summary. Thus it reduced absolute error by about 28.5% relative to the climatology denominator while failing to explain aggregate variance. The bodyweight and species values come from slightly different consolidated summaries; the final table must use one aggregation convention consistently.

Selection was not based on a large gap to every alternative. TabPFN with bodyweight achieved MASE 0.7149, and an experimental fertiliser configuration achieved 0.7142 but had worse R². These differences are too small to support a claim of decisive superiority among feature variants. The species-aware model was retained as the standing champion because it offered a defensible balance of accuracy, interpretability and alignment with the scenario questions.

When the same model families were scored against the gap-filled target, the ranking changed: the weighted and unweighted ensembles achieved MASE 0.666, RMSE about 25.4 and correlation about 0.52; XGBoost, LightGBM and random forest followed at MASE 0.677–0.703; TabPFN and TabICLv2 fell to 0.903 and 0.922. This does not make the reconstructed target more authoritative. It indicates that locally fitted ensembles follow its smoother response surface more closely, whereas the foundation models lead on the sparse observed target.

Within the common observed-target experiment, pretrained tabular foundation models outperformed the locally fitted ensemble and tested sequence models on the primary MASE metric. TabICLv2 configurations reached MASE around 0.74, leading ensemble and boosting configurations around 0.80, TFT 0.812, LSTM 0.952 and DLinear 1.187. This ranking supports pretrained tabular models for this dataset, but it does not establish that tabular models dominate deep learning in all environmental forecasting. A missingness or sample-size ablation would be needed to test the proposed mechanism directly.

Tower-level species-aware TabPFN results further show the heterogeneity:

| Tower | n | R² | OLS R² | RMSE | MAE | persistence MASE | WAPE | correlation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| T2 | 102 | −0.068 | 0.003 | 17.47 | 12.44 | 0.229 | 0.910 | 0.300 |
| T4 | 1,320 | −0.080 | 0.218 | 54.90 | 31.11 | 0.883 | 0.869 | 0.366 |
| T9 | 900 | −0.171 | 0.082 | 61.30 | 37.95 | 0.897 | 0.909 | 0.281 |

The persistence-scaled MASE in this table is not interchangeable with the climatology-scaled headline. T2 has the smallest evaluation set and weakest basis for calibration; its small RMSE does not imply strong trajectory agreement, as shown by OLS R² near zero.

## 5.3 Forecast uncertainty

Initial uncertainty estimates were generated for random forest, XGBoost, LightGBM, SARIMAX, both ensembles, TFT and TabPFN. Table 5.2 reports weighted results on the common T4/T9 calibration rows; T2 could not support robust conformal calibration.

| Model | raw PICP | raw MPIW | conformal PICP | conformal MPIW | conformal pinball |
|---|---:|---:|---:|---:|---:|
| ensemble, unweighted | 0.732 | 78.82 | 0.890 | **145.28** | **10.53** |
| ensemble, MASE weighted | 0.728 | 77.98 | 0.892 | 145.31 | **10.53** |
| random forest | 0.379 | 47.18 | 0.886 | 145.77 | **10.53** |
| LightGBM | 0.489 | 55.22 | **0.894** | 149.79 | 10.70 |
| XGBoost | 0.485 | 56.36 | 0.891 | 148.67 | 10.74 |
| SARIMAX | 0.871 | 156.52 | 0.892 | 158.65 | 11.04 |
| TFT | 0.839 | 145.63 | **0.894** | 162.89 | 11.49 |
| TabPFN | 0.807 | 120.31 | 0.892 | 165.42 | 11.55 |

Calibration largely equalised marginal PICP by widening the most under-covering models. It did not make the interval systems equivalent: width and pinball loss remained different.

Champion-specific evaluation gave conformal PICP/MPIW of 0.898/149.5 for TabPFN at T4 and 0.889/188.9 at T9. TabICLv2 gave 0.895/154.7 at T4 and 0.894/195.3 at T9. Aggregate results still concealed an operationally important failure: the flat intervals covered only 0.239 of TabPFN spikes and 0.248 of TabICLv2 spikes.

Conformalised quantile regression improved spike coverage to 0.797 for TabICLv2 and 0.572 for TabPFN. The gain reduced normal-day coverage from 0.967 to 0.837 and 0.884 respectively. Livestock-stratified calibration further showed that uncertainty was conditional on management state:

| Model | livestock tier | PICP | MPIW | pinball |
|---|---|---:|---:|---:|
| TabPFN | low | 0.865 | 92.27 | 5.29 |
| TabPFN | mid | 0.854 | 119.59 | 6.54 |
| TabPFN | high | 0.840 | 300.07 | 20.29 |
| TabICLv2 | low | 0.840 | 194.07 | 7.81 |
| TabICLv2 | mid | 0.815 | 261.79 | 10.90 |
| TabICLv2 | high | 0.826 | 428.89 | 23.80 |

Intervals were substantially wider and less sharp in high-livestock regimes, while coverage remained below the nominal target in every aggregated tier.

These results change the interpretation of forecasting skill. A single aggregate coverage figure is insufficient for methane decision support because the loss associated with missing an emission spike differs from that of a typical day. Future calibration should target conditional validity across tower, season, magnitude and management state, while acknowledging that increasingly fine strata reduce calibration sample size.

# 6. Scenario Projection

## 6.1 Drivers used by the forecasting model

Permutation importance for the selected species-aware TabPFN model ranked livestock-unit density first, followed by cattle density, grazing activity and total liveweight. Growing-season status was the leading non-livestock variable among the first five. Sheep and lamb densities were much less influential. The ordering broadly agrees with earlier feature-importance and SHAP-magnitude analyses, increasing confidence that the result is not unique to one explanation method.

The global ranking hides tower heterogeneity. At T2, meteorological predictors rather than livestock variables occupied the leading positions. This could reflect real differences in farmlet response, the long observation interruption, a smaller effective sample or a shift in the measurement footprint. The analysis cannot distinguish these explanations conclusively.

Feature importance is not a causal estimate. Correlated herd variables can substitute for one another, and a variable can be predictive because it indexes season or unmeasured management. No formal SHAP-interaction analysis was completed, so the results identify influential drivers but do not decode environmental-management interactions. The completed permutation analysis is also specific to TabPFN; applying its ranking to the TabICLv2 scenario engine is a modelling judgement that should either be acknowledged or checked with a TabICLv2-specific analysis.

## 6.2 Scenario construction

The scenario-driver ablation changed model rankings substantially. Under the first reduced-driver Model-1 comparison, TabPFN remained strongest, while the locally fitted tree models deteriorated sharply:

| Model | R² | RMSE | MAE | MASE | WAPE | correlation |
|---|---:|---:|---:|---:|---:|---:|
| TabPFN | **−0.122** | **56.12** | **33.14** | **0.733** | **0.899** | **0.358** |
| TabICLv2 | −0.330 | 58.05 | 34.95 | 0.782 | 0.988 | 0.255 |
| TFT | −0.363 | 56.59 | 35.62 | 0.841 | 1.045 | 0.292 |
| LSTM | −1.357 | 63.22 | 41.06 | 0.956 | 1.268 | 0.212 |
| SARIMAX | −1.416 | 65.63 | 49.54 | 1.108 | 1.516 | 0.332 |
| XGBoost | −1.611 | 71.87 | 58.50 | 1.297 | 1.706 | 0.250 |
| DLinear | −2.068 | 64.53 | 47.52 | 1.265 | 1.638 | 0.237 |
| ensemble, MASE weighted | −2.101 | 75.37 | 63.22 | 1.455 | 1.972 | 0.298 |
| ensemble, unweighted | −2.104 | 75.31 | 63.16 | 1.454 | 1.972 | 0.299 |
| LightGBM | −2.682 | 83.59 | 70.59 | 1.593 | 2.122 | 0.255 |
| random forest | −7.051 | 99.11 | 87.77 | 2.270 | 3.121 | 0.190 |

This table is a model-selection stage rather than the final S-06 result. The final scenario workflow uses the reduced FX_A_SPECIES feature set with TabICLv2. This configuration achieved climatology-scaled MASE 0.7588 and R² −0.1708. Corresponding fuller TabICLv2 configurations achieved MASE 0.7385 for BASE + ALL and 0.7405 for BASE + species. Adding final fertiliser-v2 inputs produced MASE 0.7591 and R² −0.1837, so they did not improve the engine.

Future environmental drivers were derived from climate-model trajectories and adjusted against the historical site record. Before correction, simulated precipitation was approximately four times too high, temperature was around 2–3.5 °C too cool and incoming short-wave radiation was modestly low. Additive temperature and multiplicative precipitation and radiation corrections were applied by climate model [CITATION: primary bias-correction method]. The magnitude of these corrections is itself a limitation: scenario outputs depend on both the methane model and the correction mapping.

Management experiments changed livestock level, grazing duration or fertiliser management while holding the remainder of the scenario design fixed. Livestock levels included half of the baseline, a literature ceiling of 3 livestock units per hectare and tower-specific historical maxima. The implemented maxima were 4.511, 4.987 and 5.652 at T2, T4 and T9. An earlier peak value of 2.13 at T2 belonged to a smoothed three-times scenario and was not the historical maximum.

Scenarios transition progressively rather than jumping instantaneously, and results are compared with a matched scenario baseline. Applicability diagnostics assess whether future covariates remain within the region represented by historical training data. Prediction intervals can be generated for most scenario rows, but their availability is not empirical coverage because the future target is unobserved.

## 6.3 Conditional scenario responses

Table 6.1 gives the principal livestock responses under SSP2-4.5.

| Intervention | T2 | T4 | T9 |
|---|---:|---:|---:|
| halve all livestock | −15.2% | −26.2% | −33.9% |
| literature ceiling | +25.4% | +19.9% | +13.5% |
| tower-specific historical maximum | +53.4% | +125.5% | +149.8% |

The fitted response is nonlinear and tower specific. Reducing livestock lowers predicted methane at all towers, with the largest relative decrease at T9. Raising livestock to the historical maximum produces much larger responses at T4 and T9 than at T2. Cattle-only historical-maximum scenarios produce similarly large changes at T4 and T9, consistent with cattle density's high permutation importance and the much lower importance assigned to sheep and lamb variables.

Extending the grazing season by four weeks increased predicted methane by about 3.8% at T2 and 18.6% at both T4 and T9. In contrast, increasing fertiliser frequency or rate by 50% produced small, sign-inconsistent changes: slightly negative at T2, about one percent positive at T4 and close to zero at T9. The near-null result may reflect a genuinely weak conditional signal, insufficient fertiliser representation or confounding in the historical data. The model cannot distinguish these possibilities.

Differences between SSP2-4.5 and SSP5-8.5 were generally much smaller than the management effects over the analysed horizon. They were not below one percent in every case: late-horizon T2 divergence approached 2.9%. A defensible conclusion is therefore that management perturbations dominated climate-pathway differences in this model and horizon, not that climate had no effect.

## 6.4 Applicability and interpretation

Baseline out-of-applicability rates remained around 61–65% for the livestock experiments. Historical or baseline rates were also high in the grazing and fertiliser workflows. Bias correction improved individual climate variables but did not move the joint scenario distribution fully into the historical support.

This prevents a causal or predictive interpretation of the response percentages. The model has learned associations in an observational record and is being evaluated at many covariate combinations that are weakly represented by that record. The percentages answer a bounded question: *how does this fitted TabICLv2 response surface change when the specified inputs are perturbed under the implemented scenario construction?* They do not establish how methane emissions will actually change under a farm intervention.

The scenario workflow reports prediction intervals for more than 99% of applicable output rows. That statistic measures interval availability, not prediction-interval coverage. Coverage requires observed future methane and cannot yet be calculated. Scenario uncertainty should therefore combine interval width, applicability status, climate-model spread and sensitivity to modelling choices, while clearly stating which components remain unquantified.

# 7. Discussion

## 7.1 Pretrained tabular models were effective under target scarcity

The most consistent empirical result is the competitiveness of pretrained tabular foundation models. TabICL-solo improved held-out gap filling at two of three towers, and TabPFN and TabICLv2 led the recursive forecast comparison. This pattern is compatible with the hypothesis that pretraining supplies a useful inductive bias when the local dataset has many predictors but few complete target observations.

The experiments do not isolate that mechanism. Tabular models also differ from sequence models in preprocessing, parameter fitting, context construction and access to engineered temporal features. A controlled ablation varying observed sample size and missing-block structure would be needed to attribute the ranking specifically to pretraining or missingness tolerance. The appropriate conclusion is project-specific: pretrained tabular models were the strongest tested family under this evaluation design.

## 7.2 Reconstruction skill did not guarantee forecast skill

Gap-filling R² was positive at all towers, whereas recursive forecast R² was slightly negative even for the selected model. This is not an inconsistency in the analysis; it demonstrates the difference between interpolation-like reconstruction and forward prediction. A gap filler can use contemporaneous environmental context and observations around a missing block. A forecast must operate beyond an information boundary and may accumulate error through rollout.

The forecast MASE of 0.715 nevertheless indicates improvement over seasonal climatology in absolute error. R² and MASE answer different questions, so neither should be suppressed. For a decision-support application, the result suggests utility for reducing typical error but insufficient evidence for reproducing the full amplitude and timing of methane variability.

## 7.3 Aggregate uncertainty metrics hid high-emission failures

The uncertainty analysis shows why marginal coverage can be misleading. Intervals that appeared close to nominal overall performed poorly on the largest methane-flux days. Conditional quantile and livestock-stratified methods improved alignment between band width and emission regime, but no method attained nominal coverage uniformly.

This matters scientifically and operationally. High-emission events can contribute disproportionately to cumulative budgets and mitigation priorities. An interval system that is reliable on ordinary days but misses peaks can support an overly confident assessment. Future work should consider weighted calibration objectives, block-aware conformal methods and validation that explicitly reserves extreme events, while avoiding strata so small that calibration becomes unstable.

## 7.4 Management importance did not establish intervention causality

Livestock variables dominated the global TabPFN importance ranking, and the TabICLv2 scenarios responded strongly to livestock perturbations. This agreement makes the livestock axis a reasonable object of sensitivity analysis. It does not convert observational association into a causal dose-response curve. Herd variables are correlated with season, forage availability, animal movement and unrecorded management decisions. The models may also extrapolate when several species variables are altered together.

The contrasting fertiliser result is informative. More detailed fertiliser variables degraded gap-filling performance, and fertiliser scenarios produced near-zero or inconsistent changes. This could mean that direct fertiliser information adds little predictive signal at the available resolution, or that its effect is mediated through vegetation and soil processes not captured by the statistical model. A process-based model or a designed intervention would be needed to distinguish these explanations.

## 7.5 Limits of the scenario evidence

Scenario analysis is the least validated stage of the pipeline. Its uncertainty includes at least four layers: climate-model and emissions-pathway uncertainty, bias-correction uncertainty, methane-model uncertainty and uncertainty about responses outside the historical covariate support. The current interval workflow represents only part of this stack. High out-of-applicability rates show that many future rows cannot be treated as ordinary in-distribution predictions.

The scenario results are still useful when framed correctly. They identify where the fitted model is sensitive, expose tower heterogeneity, show that cattle-related interventions dominate its livestock response, and reveal that the tested fertiliser perturbations have little effect. They also diagnose what a more mature digital twin would require: denser verified methane observations, harmonised management records, prospective intervention data, model ensembles, applicability monitoring and a mechanism for updating uncertainty as new outcomes arrive.

## 7.6 Generalisability and future work

North Wyke is a uniquely instrumented temperate high-rainfall livestock platform, but three towers at one site cannot represent all grassland systems. Generalisation should be tested at independent farms with different soils, climates, stocking practices and sensor regimes. Cross-site validation would reveal whether pretrained tabular models transfer beyond the local tower identities or depend on site-specific correlations.

Near-term methodological priorities are to rerun uncertainty estimation for the final gap-filling champion, execute interpretability for the TabICLv2 scenario model, test sensitivity to gap-filling choices in the downstream forecast, and quantify scenario spread across climate and methane models. Prospective evaluation is ultimately decisive: predictions and intervention responses should be registered before outcomes are observed and then assessed as new seasons accumulate.

# 8. Conclusion

This dissertation evaluated a connected but deliberately separated pipeline for methane gap filling, recursive forecasting and scenario sensitivity at the North Wyke Farm Platform. The literature review established that these tasks have different information constraints and validation standards, and that multi-step ecosystem-scale methane forecasting remains weakly represented in managed temperate grassland research.

For the predictive comparison, pretrained tabular models were the strongest tested family. TabICL-solo achieved held-out gap-filling R² of 0.676, 0.428 and 0.423 at the three towers. Species-aware TabPFN achieved climatology-scaled forecast MASE of 0.715, while its slightly negative R² showed that improvement over a baseline did not amount to complete reproduction of observed variability.

Interpretability analyses identified livestock-unit density, cattle density, grazing activity and liveweight as leading global drivers, with a different meteorological pattern at T2. Uncertainty analyses showed that near-nominal aggregate coverage masked failure on high-emission days. These findings answer the driver-and-uncertainty question more cautiously than a single global explanation: model behaviour varies by tower and regime, and importance remains associational.

The final scenarios implied substantial livestock responses at T4 and T9, a smaller response at T2, moderate grazing-duration effects and near-null fertiliser effects. Management perturbations generally dominated the difference between the two tested climate pathways. However, high out-of-applicability rates, large climate bias corrections and absent future outcomes prevent these values from being interpreted as validated emissions forecasts or causal treatment effects.

The principal contribution is therefore not an autonomous methane digital twin. It is an evidence-graded digital shadow that demonstrates where pretrained tabular models add predictive value and where the chain from observation to decision remains weak. A credible next generation should preserve the task separation developed here, validate extreme-event uncertainty, monitor applicability and close the loop with prospective farm interventions and new methane observations.

## Citation and revision notes

- Replace all `[CITATION: ...]` prompts only after duplicate keys in `references_mendeley.bib` have been repaired and missing primary sources have been added.
- Confirm the programme word limit before expanding the literature review or methods detail.
- Reconcile the exact full TabICLv2 comparator used to calculate the scenario feature-reduction cost.
- Decide whether the final report adopts the four consolidated research questions or maps this prose back to the original five.
- Do not move the numerical claims into LaTeX until they have been checked against the final result artefacts listed in the evidence audit.
