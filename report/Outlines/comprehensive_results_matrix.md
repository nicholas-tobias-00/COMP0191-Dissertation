# Comprehensive results matrix

> **Working status.** This document is the numerical source sheet for Chapters 4–6. It separates standing comparisons from developmental ablations and superseded runs. No result should be moved into LaTeX without preserving its target, temporal resolution, aggregation rule and baseline definition.

## 1. Metric and protocol rules

- Gap-filling point metrics are hourly medians across the five artificial-gap scenarios unless a table explicitly says daily. The scenarios are 1 hour, 4 hours, 32 hours, 288 hours and mixed length.
- Gap-filling R² is sklearn R² unless labelled OLS R². OLS R² is squared correlation and is the closest comparison with Zhu et al.
- Forecast MASE in the final comparison is scaled to seasonal climatology. Older tables scaled MASE to persistence and must not be numerically merged with it.
- Forecast point results are evaluated against observed targets unless labelled gap-filled target.
- PICP is empirical interval coverage, MPIW is mean prediction-interval width, and pinball loss is lower-is-better quantile loss.
- Scenario outputs have no future ground truth. Interval availability and Area of Applicability diagnostics are not PICP.

# 2. Gap-Filling

## 2.1 Floors, imputers and standing models

The first table shows median hourly R² on the common artificial-gap protocol. The first six rows document the build-up from trivial/statistical filling to the operational random forest. Later rows are the expanded final model roster.

| Model or experiment | T2 R² | T4 R² | T9 R² | Status |
|---|---:|---:|---:|---|
| training-set mean | −0.003 | −0.001 | −0.001 | trivial floor |
| MDS, corrected three-case hierarchy | −0.023 | −0.113 | −0.073 | literature baseline |
| RF, meteorology only | 0.052 | 0.036 | 0.059 | raw-driver baseline |
| MICE with Bayesian ridge | 0.081 | 0.118 | 0.107 | multivariate imputer |
| HyperImpute | 0.509 | 0.336 | 0.354 | AutoML imputer |
| RFm partial-pooled champion | 0.576 | 0.404 | 0.426 | operational champion |
| LightGBM | 0.522 | 0.410 | 0.422 | expanded roster |
| XGBoost | 0.551 | 0.349 | 0.369 | expanded roster |
| TabPFN | 0.459 | 0.401 | 0.402 | expanded roster |
| TabICL, pooled comparison | 0.558 | 0.423 | 0.364 | expanded roster |
| SAITS | 0.341* | 0.275* | 0.263* | sequence imputer |
| bidirectional LSTM | 0.237 | 0.155 | 0.146 | sequence model |
| TabICL-solo | **0.676** | **0.428** | 0.423 | held-out point champion |

*The recalculated SAITS values do not reproduce an older summary that reported 0.358, 0.293 and 0.285. Treat the recalculated row as provisional until the artefact mismatch is resolved.

The corresponding error metrics are:

| Model | Tower | RMSE | MAE | MBE | OLS R² |
|---|---:|---:|---:|---:|---:|
| MDS | 2 | 143.0 | 45.1 | — | see corrected MDS audit |
| MDS | 4 | 134.7 | 52.7 | — | see corrected MDS audit |
| MDS | 9 | 144.0 | 63.2 | — | see corrected MDS audit |
| MICE | 2 | 132.6 | 59.6 | — | — |
| MICE | 4 | 114.6 | 55.9 | — | — |
| MICE | 9 | 130.4 | 61.7 | — | — |
| HyperImpute | 2 | 93.9 | 37.0 | — | — |
| HyperImpute | 4 | 107.0 | 47.2 | — | — |
| HyperImpute | 9 | 116.8 | 53.4 | — | — |
| RFm champion | 2 | 75.0 | 31.2 | — | 0.601 |
| RFm champion | 4 | 100.3 | 42.8 | — | 0.409 |
| RFm champion | 9 | 107.1 | 49.3 | — | 0.430 |
| LightGBM | 2 | 82.74 | 34.78 | 6.36 | 0.583 |
| LightGBM | 4 | 100.88 | 44.34 | 2.77 | 0.418 |
| LightGBM | 9 | 109.02 | 51.60 | 2.31 | 0.425 |
| XGBoost | 2 | 89.48 | 40.74 | 5.28 | 0.619 |
| XGBoost | 4 | 107.74 | 49.79 | 3.86 | 0.380 |
| XGBoost | 9 | 112.28 | 57.10 | 2.58 | 0.375 |
| TabPFN | 2 | 89.02 | 30.61 | 3.73 | — |
| TabPFN | 4 | 99.36 | 42.40 | 3.12 | — |
| TabPFN | 9 | 107.97 | 49.74 | 5.26 | — |
| TabICL, pooled | 2 | 76.62 | 30.82 | −0.59 | 0.587 |
| TabICL, pooled | 4 | 101.70 | 41.97 | 2.13 | 0.425 |
| TabICL, pooled | 9 | 108.55 | 50.59 | 3.65 | 0.368 |
| SAITS* | 2 | 92.30 | 36.72 | −2.74 | 0.562 |
| SAITS* | 4 | 104.41 | 42.27 | −5.11 | 0.289 |
| SAITS* | 9 | 119.33 | 54.34 | −3.10 | 0.304 |
| bidirectional LSTM | 2 | 93.99 | 33.65 | −0.49 | 0.280 |
| bidirectional LSTM | 4 | 115.73 | 49.07 | 3.10 | 0.204 |
| bidirectional LSTM | 9 | 127.50 | 54.29 | −2.99 | 0.189 |
| TabICL-solo | 2 | 77.60 | 30.03 | 5.78 | 0.686 |
| TabICL-solo | 4 | 99.18 | 41.57 | 2.54 | 0.429 |
| TabICL-solo | 9 | 110.01 | 50.48 | 7.02 | 0.430 |

The TabICL-solo row is the median of each metric across gap scenarios; the R², RMSE, MAE and MBE medians need not come from the same individual scenario.

## 2.2 Robustness to gap length

Median R² by gap scenario is shown below. Each cell is T2 / T4 / T9.

| Model | 1 h | 4 h | 32 h | 288 h | mixed |
|---|---|---|---|---|---|
| LightGBM | .522 / .448 / .447 | .663 / .410 / .406 | .646 / .441 / .422 | .033 / .238 / .177 | .470 / .291 / .422 |
| XGBoost | .551 / .389 / .445 | .671 / .349 / .369 | .598 / .363 / .383 | −.647 / −.062 / .060 | .143 / .228 / .333 |
| TabPFN | .459 / .455 / .443 | .704 / .401 / .402 | .684 / .456 / .543 | .092 / .214 / .190 | .456 / .346 / .350 |
| TabICL, pooled | .558 / .461 / .459 | .722 / .435 / .364 | .698 / .423 / .467 | .218 / .233 / .231 | .466 / .332 / .330 |
| SAITS | .510 / .408 / .378 | .219 / .319 / .285 | .659 / .248 / .338 | .055 / .207 / −.139 | .358 / .293 / .258 |
| bidirectional LSTM | .423 / .011 / .146 | .237 / .138 / .150 | .546 / .265 / .167 | .148 / .155 / .072 | .189 / .175 / .112 |
| TabICL-solo | .676 / .521 / .514 | .727 / .465 / .423 | .687 / .428 / .459 | .057 / .206 / −.102 | .428 / .291 / .227 |

The 288-hour experiment is the clearest weakness. Several models become negative at one tower, including TabICL-solo at T9. Overall medians therefore should not be read as evidence of equal reliability at every blackout duration.

## 2.3 Daily aggregation and coverage sensitivity

The operational RFm champion has hourly R² of 0.576, 0.404 and 0.426. When held-out predictions are aggregated to daily resolution, the unweighted full-scope R² values are 0.698, 0.504 and 0.485.

| Daily rule | T2 | T4 | T9 |
|---|---:|---:|---:|
| unweighted full-scope R² | 0.698 | 0.504 | 0.485 |
| days with at least 50% hourly coverage | 0.638 | 0.446 | 0.586 |
| hour-count-weighted daily R² | 0.721 | 0.528 | 0.498 |

The sensitivity range should be reported because daily aggregation can be affected by the number of observed hours contributing to each day.

## 2.4 Feature, pooling and refinement ablations

Important developmental findings are:

- Replacing aggregate livestock density with species densities changed RF R² to 0.593, 0.404 and 0.428.
- Species and grazing-memory features changed RF R² to 0.601, 0.395 and 0.425: a T2 gain with no general improvement.
- MDS-to-RFm improvement came from both model class and engineered features. Meteorology-only RF reached only 0.052–0.059 R², while the full RFm champion reached 0.404–0.576.
- Nearest-target and wide lag/lead feature sets degraded accuracy. The revised 164-feature RF achieved 0.543, 0.262 and 0.293.
- Fertiliser additions reduced gap-filling R² for both RF and TabICL.
- TICA components were essentially neutral. Feeding model interval width back as an input reduced RF R² and severely reduced TabICL at T2 and T9.
- TabICL feature dropping and TICA replacement were flat or negative.
- Native TabICL hyperparameter variants remained within −0.025 to +0.019 R² of the default.
- Row-cap bagging improved only T4, from 0.428 to approximately 0.440–0.441, and plateaued by five to eight bags.
- Confidence-gated TabICL self-training was neutral or harmful overall; its largest gain was approximately +0.005 R² at T4.

Prediction averaging did not produce a general replacement for TabICL-solo:

| Tower | RF | TabICL | HyperImpute | RF + TabICL | all three |
|---|---:|---:|---:|---:|---:|
| T2 | 0.576 | **0.676** | 0.509 | 0.646 | 0.621 |
| T4 | 0.404 | 0.428 | 0.336 | **0.445** | **0.445** |
| T9 | **0.426** | 0.423 | 0.354 | 0.432 | 0.417 |

The T4 ensemble gain is real but tower-specific.

## 2.5 Gap-filling uncertainty

Uncalibrated 90% interval results:

| Model | Tower | n | PICP | MPIW | corr(width, absolute error) |
|---|---:|---:|---:|---:|---:|
| quantile RFm | 2 | 12,384 | 0.941 | 162.70 | 0.644 |
| quantile RFm | 4 | 48,707 | 0.919 | 197.15 | 0.493 |
| quantile RFm | 9 | 28,189 | 0.898 | 213.67 | 0.504 |
| TabICL quantiles | 2 | 12,384 | 0.918 | 141.70 | 0.590 |
| TabICL quantiles | 4 | 48,707 | 0.901 | 184.25 | 0.509 |
| TabICL quantiles | 9 | 28,189 | 0.893 | 220.82 | 0.482 |

Split-conformal calibration moved most gap-scenario coverages toward 0.90. Across individual tower/scenario cells, calibrated QRF coverage ranged from 0.876 to 0.912 and calibrated TabICL coverage from 0.888 to 0.910. Calibration sometimes narrowed already over-covering intervals and widened under-covering intervals.

Daily conformal results were:

| Tower | calibration days | test days | PICP | MPIW | naive-normal PICP |
|---|---:|---:|---:|---:|---:|
| T2 | 112 | 112 | 0.866 | 104.72 | 0.866 |
| T4 | 430 | 430 | 0.923 | 161.52 | 0.935 |
| T9 | 245 | 245 | 0.861 | 124.67 | 0.910 |

The gap-filling intervals predate the TabICL-solo point champion, and interval width did not reliably grow with real blackout length. They cannot yet be presented as calibrated uncertainty for that exact champion.

# 3. Forecasting

## 3.1 Full observed-target model comparison

The final comparison contains eleven model rows. Lower MASE, RMSE and WAPE are better; higher R² and correlation are better. Values below use each row's named selected configuration.

| Model | Selected config | MASE climatology | R² | RMSE | WAPE | correlation |
|---|---|---:|---:|---:|---:|---:|
| TabPFN | BASE + bodyweight | **0.7149** | −0.0953 | 56.16 | 0.880 | 0.356 |
| TabICLv2 | BASE + ALL | 0.7385 | −0.1426 | 56.61 | 0.905 | 0.302 |
| ensemble, MASE weighted | BASE | 0.8019 | −0.1627 | **52.25** | 0.991 | 0.379 |
| ensemble, unweighted | BASE | 0.8020 | −0.1625 | 52.25 | 0.991 | 0.379 |
| XGBoost | BASE + ALL | 0.8038 | −0.2010 | 52.30 | **0.988** | 0.364 |
| TFT | BASE + ALL | 0.8120 | −0.2372 | 55.78 | 1.009 | 0.236 |
| LightGBM | BASE | 0.8166 | −0.2140 | 52.83 | 1.014 | 0.371 |
| random forest | BASE | 0.8411 | −0.2411 | 52.91 | 1.045 | **0.380** |
| SARIMAX | BASE | 0.8741 | −0.3285 | 54.35 | 1.082 | 0.348 |
| LSTM | BASE + species | 0.9521 | −0.6513 | 62.32 | 1.154 | 0.275 |
| DLinear | BASE + bodyweight | 1.1872 | −1.7510 | 61.99 | 1.519 | 0.277 |

The table demonstrates metric-dependent rankings. TabPFN has the lowest MASE, while the ensembles have lower RMSE and random forest has the highest correlation by a small margin. No final model has positive aggregate R².

The standing interpretability champion uses TabPFN with species features. Under the current climatology aggregation it achieves MASE 0.7150 and R² −0.0382, effectively tied with the bodyweight configuration on MASE but better on R². A fertiliser extension gives MASE 0.7142 and R² −0.0418; the MASE change is negligible and was not adopted.

## 3.2 Evaluation against the gap-filled target

The same model families behave differently when evaluation includes the gap-filled target:

| Model | MASE gap-filled | R² gap-filled | RMSE gap-filled | WAPE gap-filled | correlation gap-filled |
|---|---:|---:|---:|---:|---:|
| ensemble, MASE weighted | **0.666** | −0.196 | **25.38** | **0.704** | 0.522 |
| ensemble, unweighted | 0.666 | **−0.190** | 25.39 | 0.705 | 0.523 |
| XGBoost | 0.677 | −0.584 | 25.87 | 0.725 | 0.477 |
| LightGBM | 0.689 | −0.480 | 26.11 | 0.728 | 0.496 |
| random forest | 0.703 | −0.593 | 25.85 | 0.756 | **0.529** |
| SARIMAX | 0.831 | −1.002 | 29.21 | 0.919 | 0.463 |
| TabPFN | 0.903 | −0.657 | 34.90 | 0.814 | 0.240 |
| TabICLv2 | 0.922 | −0.784 | 35.27 | 0.841 | 0.210 |
| TFT | 1.021 | −2.179 | 36.10 | 1.103 | 0.147 |
| LSTM | 1.122 | −2.724 | 39.79 | 1.188 | 0.185 |
| DLinear | 1.502 | −6.792 | 42.59 | 1.653 | 0.182 |

This reversal is not evidence that gap-filled targets are more truthful. It shows that locally fitted ensembles track the smoother reconstruction more closely, whereas foundation models perform best on the sparse observed target under climatology-scaled MASE.

## 3.3 Feature-set sensitivity within foundation and deep models

Selected climatology-MASE results from the configuration sweep are:

| Model/configuration | MASE | R² |
|---|---:|---:|
| TabPFN v2, BASE + ALL | 0.7121 | −0.0496 |
| TabPFN, BASE + bodyweight | 0.7149 | −0.0619 |
| TabPFN, BASE + species | 0.7150 | −0.0382 |
| TabPFN, BASE + ALL | 0.7171 | −0.0459 |
| TabPFN, BASE | 0.7238 | −0.0812 |
| TabICLv2, BASE + ALL | 0.7385 | −0.1134 |
| TabICLv2, BASE + species | 0.7405 | −0.1355 |
| TabICLv2, BASE + bodyweight | 0.7466 | −0.1615 |
| TabICLv2, BASE | 0.7702 | −0.2858 |
| TFT, BASE + ALL | 0.8120 | −0.2600 |
| TFT, BASE + species | 0.8398 | −0.3676 |
| TFT, BASE | 0.8928 | −0.4308 |
| LSTM, BASE + species | 0.9521 | −0.7110 |
| LSTM, BASE | 0.9696 | −1.4950 |
| DLinear, BASE + bodyweight | 1.1872 | −1.9042 |
| DLinear, BASE | 1.2456 | −2.9749 |

TabPFN feature variants are close, TabICLv2 benefits more clearly from the full feature set, and management-feature choice does not rescue LSTM or DLinear.

## 3.4 Tower-level species-aware TabPFN metrics

These tower rows use persistence-scaled MASE and must not be averaged into the climatology-MASE table.

| Tower | n | R² | OLS R² | RMSE | MAE | persistence MASE | WAPE | correlation | nMAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T2 | 102 | −0.068 | 0.003 | 17.47 | 12.44 | 0.229 | 0.910 | 0.300 | 0.686 |
| T4 | 1,320 | −0.080 | 0.218 | 54.90 | 31.11 | 0.883 | 0.869 | 0.366 | 0.460 |
| T9 | 900 | −0.171 | 0.082 | 61.30 | 37.95 | 0.897 | 0.909 | 0.281 | 0.528 |

T2's low error magnitude partly reflects a different target distribution and only 102 evaluation rows. Its very low OLS R² shows that low absolute error does not imply trajectory agreement.

## 3.5 All-model uncertainty comparison

The first uncertainty experiment calibrated eight model families on the common T4/T9 test rows. T2 lacked sufficient calibration support.

| Model | n | raw PICP | raw MPIW | raw pinball | conformal PICP | conformal MPIW | conformal pinball |
|---|---:|---:|---:|---:|---:|---:|---:|
| ensemble, unweighted | 2,217 | 0.732 | 78.82 | 10.47 | 0.890 | **145.28** | **10.53** |
| ensemble, MASE weighted | 2,217 | 0.728 | 77.98 | 10.48 | 0.892 | 145.31 | **10.53** |
| random forest | 2,217 | 0.379 | 47.18 | 12.23 | 0.886 | 145.77 | **10.53** |
| LightGBM | 2,217 | 0.489 | 55.22 | 11.38 | **0.894** | 149.79 | 10.70 |
| XGBoost | 2,217 | 0.485 | 56.36 | 11.38 | 0.891 | 148.67 | 10.74 |
| SARIMAX | 2,217 | 0.871 | 156.52 | 11.17 | 0.892 | 158.65 | 11.04 |
| TFT | 2,217 | 0.839 | 145.63 | 10.92 | **0.894** | 162.89 | 11.49 |
| TabPFN | 2,217 | 0.807 | 120.31 | 11.18 | 0.892 | 165.42 | 11.55 |

Conformal calibration largely equalised marginal coverage, but it did so by widening under-covering intervals. The ensembles and random forest achieved the lowest conformal pinball loss in this first comparison.

Champion-specific U04 results were:

| Model | Tower | n | raw PICP | conformal PICP | conformal MPIW | conformal pinball |
|---|---:|---:|---:|---:|---:|---:|
| TabPFN | 4 | 1,318 | 0.867 | 0.898 | 149.52 | 10.56 |
| TabPFN | 9 | 899 | 0.724 | 0.889 | 188.93 | 12.86 |
| TabICLv2 | 4 | 1,318 | 0.964 | 0.895 | 154.71 | 10.63 |
| TabICLv2 | 9 | 899 | 0.771 | 0.894 | 195.33 | 13.04 |

## 3.6 Extreme-event and livestock-conditional uncertainty

Marginal calibration failed on high-magnitude observations:

| Model | spike n | old spike coverage | CQR spike coverage | normal n | old normal coverage | CQR normal coverage |
|---|---:|---:|---:|---:|---:|---:|
| TabPFN | 222 | 0.239 | 0.572 | 1,995 | 0.967 | 0.884 |
| TabICLv2 | 222 | 0.248 | 0.797 | 1,995 | 0.967 | 0.837 |

CQR greatly improved spike coverage, especially for TabICLv2, but reduced normal-day coverage and widened intervals.

Livestock-stratified CQR produced:

| Model | LSU tier | n | PICP | MPIW | pinball |
|---|---|---:|---:|---:|---:|
| TabPFN | low | 855 | 0.865 | 92.27 | 5.29 |
| TabPFN | mid | 466 | 0.854 | 119.59 | 6.54 |
| TabPFN | high | 875 | 0.840 | 300.07 | 20.29 |
| TabICLv2 | low | 855 | 0.840 | 194.07 | 7.81 |
| TabICLv2 | mid | 466 | 0.815 | 261.79 | 10.90 |
| TabICLv2 | high | 875 | 0.826 | 428.89 | 23.80 |

The strong increase in width and pinball loss with livestock tier quantifies regime-dependent uncertainty. Coverage remains below the nominal 0.90 target in every aggregated tier.

# 4. Scenario Projection

## 4.1 Reduced-driver model comparison

Before the final scenario configuration was selected, models were rerun under a reduced future-driver setting. The Model-1 observed-target comparison was:

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

This stage is not the final S-06 engine result. It demonstrates that driver availability can alter the ranking and degrade locally fitted tree models severely.

The final S-06 engine uses TabICLv2 with FX_A_SPECIES:

| Configuration | climatology MASE | R² |
|---|---:|---:|
| TabICLv2 BASE + ALL | 0.7385 | approximately −0.11 to −0.14, depending aggregation |
| TabICLv2 BASE + species | 0.7405 | −0.1355 |
| TabICLv2 FX_A_SPECIES | 0.7588 | −0.1708 |
| TabICLv2 FX_A_SPECIES + fertiliser v2 | 0.7591 | −0.1837 |

Adding the final fertiliser variables did not improve the scenario engine.

## 4.2 Final conditional responses

SSP2-4.5 livestock responses:

| Intervention | T2 | T4 | T9 |
|---|---:|---:|---:|
| halve all livestock | −15.2% | −26.2% | −33.9% |
| all livestock to literature ceiling | +25.4% | +19.9% | +13.5% |
| all livestock to historical maximum | +53.4% | +125.5% | +149.8% |
| cattle alone to historical maximum | +36.7% | +128.4% | +154.8% |
| extend grazing by four weeks | +3.8% | +18.6% | +18.6% |

Fertiliser changes were small and inconsistent:

| Intervention | T2 | T4 | T9 |
|---|---:|---:|---:|
| fertiliser frequency +50% | −3.5% | +1.2% | approximately 0% |
| fertiliser rate +50% | −0.4% | +0.9% | −0.1% |

The SSP2-4.5 versus SSP5-8.5 difference was small relative to management perturbations, but T2 reached approximately 2.9% divergence late in the horizon.

## 4.3 Climate correction, applicability and uncertainty diagnostics

- Raw simulated precipitation was approximately 4.3 times too wet.
- Temperature was approximately 2–3.5 °C too cool.
- Incoming short-wave radiation was approximately 8% low.
- Per-GCM corrections were additive for temperature and multiplicative for precipitation and radiation.
- Baseline Area-of-Applicability failure was approximately 61–65% for livestock scenarios, approximately 74–83% for grazing histories, and approximately 52–64% for fertiliser histories.
- Prediction intervals were generated for more than 99% of eligible scenario rows. This is interval availability, not empirical coverage.

These diagnostics are part of the results, not optional caveats. They limit the strength of every scenario-response percentage.

# 5. Source hierarchy for drafting

Standing curated sources:

- results/latest/gap_filling/data/model_comparison.csv
- results/latest/gap_filling/data/tabicl_solo_champion_full_metrics.csv
- results/latest/gap_filling/data/prediction_intervals_summary.csv
- results/latest/gap_filling/data/conformal_calibration_summary.csv
- results/b16_full_metrics_outline.csv
- results/temp_forecasting_pipeline_master_table.csv
- results/latest/forecasting/data/tabpfn_species_champion_full_metrics.csv
- results/u02_summary.csv
- results/u04_summary.csv
- results/u06_spike_coverage_U04.csv
- results/u07_u04_lsu_cqr_summary.csv
- results/d98_forecast_fertN_amount_freq_summary.csv
- results/d98_s05_fertN_amount_freq_summary.csv
- results/s05_practices_s06_livestock_v2.csv
- results/s05_practices_s06_grazing.csv
- results/s05_practices_s06_fertilizer.csv

Developmental context:

- notebooks/03c_gap_filling_revisited/summary.md
- results/all_experiments_by_gaplength.md
- results/b09_b16_climatology_mase_full_table.csv
- results/s03_table_all_towers_climatology_tabicl.csv

Where a developmental summary conflicts with a curated final table, the conflict must be stated and resolved rather than silently choosing the more favourable number.
