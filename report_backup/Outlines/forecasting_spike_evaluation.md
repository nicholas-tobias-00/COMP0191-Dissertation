# TabPFN + Species Forecast Performance on Spikes and Non-Spikes

## Purpose and model selection

This note evaluates whether the selected species-aware TabPFN forecast performs differently on high-methane events and ordinary days. The evaluated predictions are the saved daily median forecasts from `results/u04_chains.csv`, filtered to `model == "TabPFN"`. This run uses the `BASE+species` feature configuration across towers T2, T4 and T9 and forecast anchors 2018–2022.

Strictly by pooled climatology-scaled MASE, `TabPFN_v2 + BASE+ALL` is marginally lower than species-aware TabPFN (0.7121 versus 0.7150). Species-aware TabPFN is used here because it is the documented interpretability champion, has the stronger pooled R² of the near-tied TabPFN configurations, and has a saved daily prediction chain. The U04 median chain is a later probabilistic run of the same model/configuration and can differ slightly from the deterministic headline run whose raw chain was not saved.

## Spike definition and evaluation protocol

A spike is defined retrospectively as an observed daily methane value at or above the 90th percentile for that tower. Tower-specific thresholds prevent the higher-flux towers from determining a single global cutoff. This is the same definition used in the U06 conditional-coverage analysis.

The split is based on observed outcomes and is therefore a diagnostic stratification, not a deployable spike detector. Point metrics use every row with finite `y_true` and median prediction. Climatology-scaled MASE and RMSSE use the project's established tower/anchor/horizon-bin aggregation and a day-of-year climatology computed from strictly pre-anchor observed history with a ±7-day window. T9 anchor combinations without pre-anchor observed history cannot contribute to climatology-scaled metrics.

| Tower | 90th-percentile threshold | Non-spikes | Spikes |
|---|---:|---:|---:|
| T2 | 23.25 | 91 | 11 |
| T4 | 86.39 | 1,188 | 132 |
| T9 | 112.53 | 810 | 90 |
| **Total** | tower-specific | **2,089** | **233** |

## Pooled point-forecast performance

| Metric | Non-spikes | Spikes |
|---|---:|---:|
| Observed mean | 14.41 | **188.84** |
| Predicted mean | 14.84 | **32.69** |
| Bias, prediction minus observation | +0.43 | **-156.15** |
| MAE | 19.38 | **156.15** |
| RMSE | 27.91 | **191.03** |
| R² | 0.036 | **-1.976** |
| Correlation | 0.329 | **0.131** |
| OLS R², squared correlation | 0.108 | **0.017** |
| WAPE | 0.869 | 0.827 |
| MAE / within-regime target SD | 0.682 | **1.410** |
| Underprediction rate | 46.6% | **100.0%** |

The mean spike prediction captures only 17.3% of the mean observed spike magnitude. All 233 spike observations are underpredicted. WAPE appears marginally lower on spikes only because its denominator grows with observed magnitude; absolute error is approximately eight times larger than on non-spike days.

## Baseline-scaled and uncertainty performance

| Metric | Non-spikes | Spikes |
|---|---:|---:|
| Climatology MASE | **0.659** | **1.001** |
| Climatology RMSSE | **0.699** | **0.995** |
| Persistence MASE | 0.911 | 0.909 |
| Persistence RMSSE | 0.942 | 0.926 |
| Native 90% interval coverage | 84.3% | **50.2%** |
| Mean native interval width | 107.15 | 183.75 |
| Mean native pinball loss | 6.32 | **49.95** |

The headline climatology-MASE advantage is concentrated on non-spike days. On spike days the model is effectively tied with seasonal climatology, with MASE approximately equal to one. Native intervals widen by about 72% on spikes, but coverage still falls from 84.3% to 50.2%, while pinball loss rises almost eightfold. Existing U06 conformalized-quantile results improve TabPFN spike coverage to approximately 57%, but this remains well below the nominal 90% target.

## Tower-level results

Each cell reports `non-spike / spike`.

| Tower | MAE | R² | Correlation | Climatology MASE | Native 90% coverage |
|---|---:|---:|---:|---:|---:|
| T2 | 11.53 / 27.06 | -0.181 / -9.354 | 0.052 / 0.106 | 0.357 / **1.357** | 0.802 / 0.818 |
| T4 | 17.11 / 155.49 | 0.103 / -2.097 | 0.350 / 0.143 | 0.700 / **0.990** | 0.905 / 0.530 |
| T9 | 23.59 / 172.91 | -0.052 / -2.631 | 0.312 / **0.001** | 0.625 / **0.962** | 0.758 / 0.422 |

T2 contains only 11 spike observations, so its conditional estimates are unstable. At T9, the near-zero spike correlation indicates essentially no ability to track variation in spike magnitude. Conditional R² is also affected by the restricted range within a percentile-defined subset, but the large negative bias, high absolute error and event-detection failure show that the result is not merely an R² artefact.

## Spike-event detection

For this diagnostic, a predicted spike occurs when the median forecast exceeds the same tower-specific threshold used to label observed spikes.

| Quantity | Value |
|---|---:|
| True positives | 1 |
| False positives | 3 |
| False negatives | 232 |
| True negatives | 2,086 |
| Precision | 25.0% |
| Recall | **0.43%** |
| Specificity | 99.86% |
| F1 | **0.008** |

The median forecast almost never signals a spike. Its high specificity is obtained by predicting virtually every day as non-spike and should not be interpreted as useful event-detection skill.

## Percentile sensitivity

The qualitative conclusion does not depend on selecting the 90th percentile. Increasing the threshold makes the amplitude and coverage failures more pronounced.

| Spike definition | Spike n | Observed mean | Predicted mean | MAE | R² | Native coverage |
|---|---:|---:|---:|---:|---:|---:|
| Top 20%, tower-specific P80 | 465 | 128.64 | 30.02 | 99.00 | -0.947 | 62.4% |
| Top 10%, tower-specific P90 | 233 | 188.84 | 32.69 | 156.15 | -1.976 | 50.2% |
| Top 5%, tower-specific P95 | 117 | 257.33 | 34.79 | 222.54 | -3.500 | 35.9% |

Observed spike magnitude rises sharply from P80 to P95, whereas the mean prediction remains near 30–35. This is strong evidence of amplitude compression or spike blindness.

## Interpretation for the report

Species-aware TabPFN reduces typical absolute error relative to seasonal climatology, but this aggregate advantage does not extend to high-emission events. Its median forecast tracks ordinary days moderately and is nearly unbiased across the bottom 90% of observations. On the upper decile it systematically regresses toward the centre of the target distribution, underpredicts every event, loses trajectory agreement and provides intervals with severe undercoverage. MASE near one on spikes means that even the model's limited absolute-error advantage disappears precisely in the regime most relevant to emission peaks.

This result should temper any claim that the selected forecast is uniformly useful. It supports a narrower conclusion: TabPFN improves typical-day absolute error but does not reliably reproduce, detect or quantify methane spikes. Future work should evaluate spike-weighted objectives, hurdle or mixture formulations, tail-aware losses, prospective event thresholds derived from training data only, and conditional calibration designed explicitly for high-emission regimes.

## Source artefacts

- [`results/u04_chains.csv`](../../results/u04_chains.csv): daily TabPFN and TabICLv2 quantile chains used for point and native-interval metrics.
- [`results/u06_spike_coverage_U04.csv`](../../results/u06_spike_coverage_U04.csv): existing spike/non-spike conformal coverage analysis.
- [`results/u06_u04_cqr_summary.csv`](../../results/u06_u04_cqr_summary.csv): conformalized quantile results.
- [`data/Hourly/forecast_daily_v3.csv`](../../data/Hourly/forecast_daily_v3.csv): historical observed series used to reconstruct pre-anchor day-of-year climatology.
- [`src/models/recursive_rollout.py`](../../src/models/recursive_rollout.py): day-of-year climatology and forecast evaluation implementation.

