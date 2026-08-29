# B18 forecasting improvement experiment

## Purpose and scope

B18 tested the improvement opportunities identified after the B16/B17 forecasting review, prioritising TabPFN and TabICL. All B18 work is additive: no B16/B17 output, report chapter, or `.tex` file was changed.

The experiment covered:

1. direct-model feature ablations;
2. pooled target normalisation by tower and tower-month;
3. recent-history windows, recency replication, and seasonal experts;
4. leakage-safe antecedent environmental and management features;
5. tower-adaptive model selection;
6. percentile-defined spike classification and two-stage forecasting;
7. conservative blends of the leading forecasts.

The primary evaluation remains the established observed-target, climatology-scaled MASE. Forecasts use observed methane only before each anchor and known future `fx_*` covariates; no future methane value is supplied as a predictor. There are 2,127 observed evaluation points across nine tower-anchor blocks. The gap-filled-target analysis is a sensitivity analysis with 2,938 points, using the same established climatology scaling.

## Headline result

The requested MASE below 0.25 was not achieved. The best exploratory B18 score is **MASE 0.6908**, an equal-weight mean of three TabPFN forecasts:

- the p95 event model with a conservative 25% spike-excess correction;
- the direct TabPFN model trained on the most recent 1,095 days;
- the direct TabPFN model trained on the most recent 1,460 days with tower-robust target scaling.

This improves on the reproduced B17 direct TabPFN champion (MASE 0.6958) by 0.0050 MASE, or approximately 0.72%. The improvement is real numerically but small. The equal-weight ensemble was evaluated after the component sweep, so it should be described as the best **exploratory** B18 result, not as an independently confirmed generalisation gain.

The best interpretable single/gated B18 forecast is the p95 TabPFN event correction at **MASE 0.6924**. Its block-bootstrap comparison with B17 favours B18 in 78.4% of resamples, but the 95% interval for the MASE difference crosses zero (-0.0129 to 0.0043). Therefore B18 has not established a statistically stable improvement over B17 with only nine evaluable blocks.

## Main model comparison

Metrics below are computed globally over the observed evaluation points, rather than averaging per-bin R² values.

| Model or strategy | MASE | MAE | RMSE | R² | Bias |
|---|---:|---:|---:|---:|---:|
| B18 equal three-forecast TabPFN mean | **0.6908** | **29.821** | 60.019 | 0.192 | -10.452 |
| B18 TabPFN p95 + 25% event excess | 0.6924 | 29.857 | **59.530** | **0.205** | -9.402 |
| B18 TabPFN recent 1,095-day window | 0.6930 | 29.976 | 60.323 | 0.184 | -10.826 |
| B18 TabPFN recent 1,460-day, tower-robust | 0.6934 | 29.882 | 60.485 | 0.179 | -11.128 |
| B17 direct TabPFN all-history reproduction | 0.6958 | 30.073 | 60.708 | 0.173 | -11.756 |
| B16-style TabPFN-TS v2 BASE+ALL | 0.7123 | 30.669 | 61.704 | 0.146 | -10.874 |
| B16 genuine-species TabPFN-TS v3 | 0.7154 | 30.890 | 62.228 | 0.131 | -12.184 |
| Best complete B18 TabICL: 1,460-day, tower-robust | 0.7191 | 30.864 | 60.470 | 0.180 | -8.351 |

The event-corrected single forecast has slightly worse MASE than the final mean but better RMSE and R². This is consistent with the event correction reducing a few large errors while leaving the median absolute scaled error pattern largely unchanged.

## What improved performance

### Recent-history restriction

Restricting direct TabPFN training to the latest 1,095 days improved MASE from 0.6958 to 0.6930. A 1,460-day window with tower-robust scaling produced MASE 0.6934. The 730-day window was too short (MASE 0.6980 raw), while 1,825 days moved back towards the all-history result. This supports modest temporal non-stationarity, but not aggressive forgetting.

### Conservative event correction

The p95 classifier achieved AUROC 0.876, average precision 0.302, and Brier score 0.0537. Despite useful rank discrimination, the calibrated hard gate had only 4.5% recall at 41.2% precision. Hard gating and full probability mixing generally worsened observed-target MASE. The only robustly useful form was a small correction: adding 25% of the predicted spike-minus-base excess reduced MASE to 0.6924 and RMSE to 59.53.

The result argues for treating spike probability as a weak correction signal rather than allowing a classifier to switch completely between normal and spike regressors.

### Conservative averaging

An equal mean of the p95 correction, 1,095-day raw model, and 1,460-day tower-robust model reached MASE 0.6908. A two-model equal mean reached 0.6911. Exhaustive aggregate pair weighting found an essentially flat optimum near 50/50 (MASE 0.6910), so the gain is not dependent on a precise coefficient.

However, leave-one-block-out estimated pair weights scored 0.6925, forward-only weights scored 0.6934, and leave-one-block-out three-model weights scored 0.6936. The appropriate interpretation is modest variance reduction, with insufficient blocks to validate weight optimisation.

## What did not improve performance

### Richer antecedent features

Forty-two leakage-safe antecedent and interaction variables were added, including prior precipitation accumulations; rolling and changing soil moisture, soil temperature, and air temperature; VPD, PPFD and flow histories; freeze/thaw and precipitation events; and wet/warm/grazing interactions.

Adding all antecedent variables increased direct TabPFN observed-target MASE to 0.7040. The same features slightly improved p90 spike-classification AUROC to 0.861, but this did not translate into better flux regression. The likely problem is the ratio between dimensionality and sparse observed targets rather than complete absence of signal.

### Target normalisation

Tower z-scoring was competitive at MASE 0.6942, while basic tower-robust scaling ranged from 0.6934 to 0.6961 depending on the history window. Tower-month scaling failed for six early cases because Tower 9 had no pre-anchor observed target from which to estimate a tower center. Those failures were preserved, and a separate leakage-safe pooled fallback completed all blocks. The completed tower-month approach was substantially worse: TabPFN MASE 0.7673 and TabICL MASE 0.7873.

### Seasonal experts and recency replication

Season-specific models were poor (approximately MASE 0.738-0.742 for the principal TabPFN variants). Replicating recent observations also failed to improve the all-history baseline (approximately MASE 0.697). Both approaches fragment or distort an already sparse training sample.

### Tower-adaptive model switching

Selecting a model separately for each tower appears attractive retrospectively: the optimistic same-data tower oracle reaches MASE 0.6892. It does not generalise. Leave-one-block-out same-tower selection scores 0.6985, global leave-one-block-out selection scores 0.6991, and forward same-tower selection scores 0.6985. The available tower histories are too short and uneven for reliable tower-specific model choice.

### TabICL

The best complete B18 TabICL variant used a 1,460-day window and tower-robust target scaling, with MASE 0.7191. It remains materially weaker than direct TabPFN. The B18 event version of TabICL also underperformed, with its base regressor at MASE 0.7308. TabICL is therefore not the preferred foundation model for this forecasting phase.

## Tower-level performance

For the exploratory equal-weight B18 forecast:

| Tower | Observed n | MASE | MAE | RMSE | R² | Bias |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 102 | 0.4080 | 12.923 | 19.830 | -0.060 | 4.398 |
| 4 | 1,320 | 0.7532 | 30.545 | 61.392 | 0.169 | -11.520 |
| 9 | 705 | 0.6149 | 30.910 | 61.275 | 0.215 | -10.601 |

Tower 2 has the lowest MASE but only 102 observed evaluation points and negative R². Tower 4 supplies most of the evaluation sample and remains the main source of error. Towers 4 and 9 retain a sizeable negative bias, showing continued underprediction of large positive methane fluxes.

## Spike versus non-spike result

For the interpretable p95 event-corrected forecast, using tower-specific thresholds estimated only from pre-anchor observations:

| Class | n | MASE | MAE | RMSE | R² | Bias |
|---|---:|---:|---:|---:|---:|---:|
| Spike | 130 | 3.2849 | 176.484 | 210.122 | -2.280 | -176.484 |
| Non-spike | 1,997 | 0.5237 | 20.312 | 30.007 | 0.198 | 1.475 |

The forecast is useful during ordinary conditions but still fails to recover spike magnitude. Almost all spike error is underprediction. This explains why MASE remains far above 0.25 even when the classifier discriminates events reasonably well: identifying a likely event is easier than estimating its methane amplitude from the available covariates.

## Gap-filled-target sensitivity

Across all available gap-filled evaluation values, the final three-forecast mean has MASE 0.6045, MAE 22.387, RMSE 40.692, and R² 0.262. The p95 event-corrected single forecast has MASE 0.5983. The p95 soft mixture is best for the gap-filled target at MASE 0.5884, but its observed-target MASE is 0.7069. Optimising against the smoother gap-filled series therefore changes the preferred method and does not improve the primary observed-target result.

This reinforces the earlier interpretation of TFT and LSTM: gap-filled targets provide an easier, smoother goal post, but that advantage cannot be assumed to transfer to sparse observed fluxes.

## Robustness and uncertainty

The block bootstrap resampled the nine evaluable tower-anchor blocks 10,000 times. For the p95 event-corrected single forecast:

| Comparator | B18 minus comparator MASE | 95% interval | P(B18 better) | Block wins |
|---|---:|---:|---:|---:|
| B18 1,095-day TabPFN | -0.0006 | [-0.0088, 0.0073] | 0.568 | 5/9 |
| B17 direct TabPFN | -0.0034 | [-0.0129, 0.0043] | 0.784 | 5/9 |
| B18 TabICL | -0.0266 | [-0.0493, -0.0042] | 0.993 | 6/9 |
| B16-style TabPFN-TS v2 | -0.0198 | [-0.0323, -0.0056] | 0.998 | 7/9 |
| B16 genuine-species TabPFN-TS v3 | -0.0230 | [-0.0458, -0.0020] | 0.985 | 7/9 |

B18 is convincingly better than the older B16-style and TabICL controls, but not convincingly better than the strongest B17 direct forecast or the simple B18 recency model.

## Recommended report claim

A defensible statement is:

> B18 reduced the best observed-target forecasting MASE from 0.6958 to 0.6908 through a conservative ensemble of recent-history, tower-normalised, and event-corrected TabPFN forecasts. The improvement was small and was not independently stable under block-wise weight or model selection. Spike magnitude remained the dominant failure mode, with pre-anchor p95 events reaching MASE 3.28 compared with 0.52 for non-spike observations.

The p95 event-corrected forecast should be used when a single interpretable model is required. The equal three-forecast mean may be presented as the best exploratory B18 result, explicitly retaining the uncertainty qualification above.

## B18 artifacts

### Reproducible scripts

- `notebooks/05_benchmarking/B18_direct_structure.py`
- `notebooks/05_benchmarking/B18_spike_models.py`
- `notebooks/05_benchmarking/B18_monthly_normalization_fallback.py`
- `notebooks/05_benchmarking/B18_evaluate_and_plot.py`
- `notebooks/05_benchmarking/B18_blend_validation.py`
- `notebooks/05_benchmarking/B18_final_triple_chain.py`

### Raw chains and principal tables

- `results/b18_direct_structure_chains.csv`
- `results/b18_spike_model_chains.csv`
- `results/b18_monthly_fallback_chains.csv`
- `results/b18_final_triple_blend_chains.csv`
- `results/b18_direct_structure_summary.csv`
- `results/b18_spike_model_summary.csv`
- `results/b18_spike_classification_metrics.csv`
- `results/b18_candidate_registry.csv`
- `results/b18_tower_adaptive_summary.csv`
- `results/b18_blend_validation_summary.csv`
- `results/b18_champion_block_bootstrap.csv`
- `results/b18_champion_spike_metrics.csv`
- `results/b18_final_triple_blend_summary.csv`

### Figures

- `results/figures/b18_chains_final/`: 15 B15-style plots for the exploratory equal three-forecast result.
- `results/figures/b18_chains/`: 15 B15-style plots for the p95 event-corrected single forecast.
- `results/figures/b18_chains_blend/`: 15 B15-style plots for the equal two-forecast blend.
- `results/figures/b18_summary/`: candidate ranking, tower comparison, and block comparison figures.

All chain legends follow the B15 convention exactly: `Gap-filled FCH4`, `Actual FCH4 (observed)`, and `TabPFN_v2 (predicted)`.
