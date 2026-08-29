# B17 — Foundation-model forecasting improvement experiments

## Purpose and integrity constraints

B17 tested whether the fixed-origin 365-day methane-flux forecasting component could be improved, prioritising TabPFN and TabICL. All B17 work was additive: no B16 result, report source, shared model module, or earlier output was changed.

The primary evaluation remains the observed-target, lead-bin-weighted climatology MASE used for the latest B16 comparison. The evaluated protocol uses observed FCH4 history available on or before each 16 December anchor and the established known-future `fx_*` driver convention. Post-anchor `y_observed`, `y_gapfilled`, and methane autoregressive features were not supplied to strict candidates. A separate realised-FCO2 sensitivity test was explicitly labelled conditional and did not improve performance.

An integrity assertion corrected the configuration-name drift found in some later B16 notebooks:

- `BASE_34`: 34 predictors.
- Genuine `BASE_species_37`: 34 base predictors plus three species-density predictors.
- `BASE_ALL_52`: all 52 `fx_*` predictors.

## Experiments completed

B17 generated and evaluated the following additive experiment families:

1. A nine-configuration feature screen for TabPFN-TS v3 and TabICL-TS v2, plus the three principal TabPFN-TS v2 controls.
2. Recent-context windows of 730, 1,095, and 1,460 days.
3. Pre-anchor-only covariate imputation, climatology-residual targets, prediction quantiles, and corrected multi-tower time-series pooling.
4. Direct pooled and tower-specific `TabPFNRegressor` and `TabICLRegressor` forecasting, distinct from the one-shot time-series wrappers.
5. Self-fed 30-, 90-, and 180-day forecast chunks.
6. Pre-anchor target interpolation and day-of-year target densification.
7. TabPFN ensemble size, softmax temperature, averaging convention, random seed, and robust target-transform tests; corresponding TabICL ensemble-size, seed, and transform tests.
8. Fixed and leave-block-out ensembles with XGB and other foundation candidates.

The raw direct-regression champion uses pooled observed training rows from all towers, the 52 `fx_*` predictors, three tower indicators, calendar year, and days since 1 January 2010. It fits a generic TabPFN v2 regressor independently at each anchor and returns the predictive median.

## Main result

| Candidate | Observed MASE | MAE | RMSE | R² | Bias |
|---|---:|---:|---:|---:|---:|
| **B17 direct pooled TabPFN v2, seed 137** | **0.6958** | **30.073** | 60.708 | 0.173 | -11.756 |
| B17 direct pooled TabPFN v2, signed-log target | 0.6960 | 30.075 | 61.019 | 0.165 | -12.922 |
| B17 direct pooled TabPFN v2, asinh target | 0.6961 | 30.088 | 60.697 | 0.174 | -12.299 |
| B17 direct pooled TabPFN v2, seed 42 | 0.6985 | 30.144 | 60.920 | 0.168 | -12.288 |
| **Best B17 TabICL: direct pooled, asinh target** | **0.7089** | 30.592 | 61.750 | 0.145 | -12.444 |
| TabPFN-TS v2, `BASE_ALL_52` | 0.7123 | 30.669 | 61.704 | 0.146 | -10.874 |
| Genuine TabPFN-TS v3, `BASE_species_37` | 0.7154 | 30.890 | 62.228 | 0.131 | -12.184 |

Thus B17 reduces observed-target MASE by 0.0165 (2.31%) relative to the strongest B16-style TabPFN-TS v2 checkpoint and by 0.0196 (2.74%) relative to genuine TabPFN-TS v3 + species. The best TabICL result improves substantially over the approximately 0.735–0.741 time-series TabICL range, but does not displace TabPFN.

The requested aspirational MASE below 0.25 was not reached. Reaching 0.25 would require approximately a further 64% reduction in the climatology-scaled absolute error from the B17 champion. No honest fixed-origin candidate approached that range.

## Uncertainty and model selection

A 10,000-resample tower–anchor block bootstrap gave:

| Comparison, B17 minus comparator | Delta MASE | 95% interval | P(B17 better) |
|---|---:|---:|---:|
| Versus TabPFN-TS v2 `BASE_ALL_52` | -0.01647 | [-0.03071, 0.00003] | 0.974 |
| Versus genuine TabPFN-TS v3 `BASE_species_37` | -0.01963 | [-0.03835, -0.00326] | 0.994 |
| Seed 137 versus direct seed 42 | -0.00271 | [-0.00756, 0.00233] | 0.857 |

The architectural change to direct pooled regression is therefore supported more strongly than the exact seed choice. The seed-137 result is the numerical champion, but its small advantage over direct seed 42 is not itself decisive.

Leave-one-block-out selection and blending did not improve observed MASE:

| Method | Observed MASE | RMSE | R² |
|---|---:|---:|---:|
| Raw B17 champion | **0.6958** | 60.708 | 0.173 |
| Fixed mean of five direct TabPFN variants | 0.6961 | 60.764 | 0.172 |
| Leave-block-out champion + XGB blend | 0.6974 | **59.918** | **0.195** |
| Leave-block-out model selection | 0.7005 | 61.165 | 0.161 |

The XGB blend improves RMSE and R² but slightly worsens the primary MASE, so it is retained only as a secondary-metric diagnostic rather than promoted as the champion.

## Tower and horizon behaviour

Observed-target performance differs markedly by tower:

| Tower | n | MASE | MAE | RMSE | R² |
|---:|---:|---:|---:|---:|---:|
| 2 | 102 | 0.4073 | 12.713 | 19.548 | -0.030 |
| 4 | 1,320 | 0.7561 | 30.633 | 61.707 | 0.161 |
| 9 | 705 | 0.6246 | 31.536 | 62.724 | 0.178 |

The weakest lead ranges are days 1–7 (MASE 0.9795, but only 19 observations), days 31–90 (0.7618), and days 181–270 (0.7916). Days 91–180 are strongest among well-populated ranges at MASE 0.5624.

## Spike limitation

The remaining error is concentrated in high-flux events:

| Definition | Class | n | MASE | MAE | RMSE | Bias |
|---|---|---:|---:|---:|---:|---:|
| Above observed 90th percentile (95.14) | Spike | 200 | 2.743 | 145.856 | 180.994 | -145.856 |
| Below 90th percentile | Non-spike | 1,927 | 0.483 | 18.056 | 25.845 | 2.162 |
| Above observed 95th percentile (157.17) | Spike | 101 | 3.858 | 213.106 | 241.388 | -213.106 |
| Below 95th percentile | Non-spike | 2,026 | 0.538 | 20.948 | 31.054 | -1.719 |

Every 90th-percentile spike is underpredicted on average, as shown by bias equalling negative MAE in that subset. This is the clearest empirical obstacle to a very low aggregate MASE: performance outside spikes is already materially better, while the sparse extreme events remain poorly predictable from the available daily drivers.

## Gap-filled-target sensitivity

Against available gap-filled targets, using the same frozen climatology scaling convention, the champion obtains n = 2,938, MASE = 0.6142, MAE = 22.832, RMSE = 41.580, R² = 0.229, and bias = -14.926. This easier target improves the headline error, but still does not approach MASE 0.25. These numbers are a target sensitivity analysis, not evidence that future gap-filled FCH4 was used as an input.

## Negative and near-null findings

- The genuine 37-feature species configuration was not the best input set; TabPFN-TS v2 with all 52 predictors remained the strongest one-shot wrapper.
- A 1,460-day window improved TabPFN-TS v2 only from 0.7123 to 0.7117.
- Climatology-residual targets worsened MASE to approximately 0.88–0.91.
- Quantiles above 0.5 worsened MASE despite reducing some mean underprediction.
- Corrected time-series pooling was neutral or worse in the new strict rerun.
- Self-fed chunks did not beat the one-shot wrapper; the best 180-day result was 0.7122.
- Larger 16- and 32-member TabPFN ensembles did not beat the default-size seed sweep.
- Realised future FCO2 did not provide an upper-bound gain; the direct TabPFN conditional result was 0.7032.

## Interpretation and recommendation

The B17 direct pooled TabPFN v2 model is the new numerical forecasting champion under the established driver-conditional benchmark. The defensible claim is a modest but measurable improvement, not a breakthrough to MASE 0.25. Its strongest evidence is the comparison with genuine TabPFN v3 + species; the comparison with TabPFN-TS v2 + all drivers remains borderline at the upper bootstrap limit.

Future gains are unlikely to come from more feature subsets, context lengths, or foundation inference settings alone. The next scientifically motivated work would require new predictors that lead methane spikes, denser observed target coverage, or a separately validated event model. Any further tuning should use nested temporal or site-held-out validation because only nine tower–anchor blocks currently contribute to the primary score.

## B17 artifacts

Core raw outputs:

- `results/b17_foundation_screen_chains.csv`
- `results/b17_context_target_chains.csv`
- `results/b17_direct_recursive_chains.csv`
- `results/b17_direct_tuning_chains.csv`

Final evaluation tables:

- `results/b17_candidate_registry.csv`
- `results/b17_champion_metrics_by_tower.csv`
- `results/b17_champion_metrics_by_block.csv`
- `results/b17_champion_metrics_by_horizon.csv`
- `results/b17_champion_spike_metrics.csv`
- `results/b17_champion_block_bootstrap.csv`
- `results/b17_crossfit_ensemble_summary.csv`

Figures:

- Fifteen tower–anchor chain figures: `results/figures/b17_chains/`
- Candidate, tower, and block summaries: `results/figures/b17_summary/`

The chain figures follow the requested convention: B17 prediction in blue, observed FCH4 in black, and gap-filled FCH4 in black dotted lines.
