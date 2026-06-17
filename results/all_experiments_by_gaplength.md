# All experiments — best model by gap length (median R²)

Best configuration and its median R² at each artificial-gap length, per experiment and tower. `—` = that gap length not run in that experiment. R-01/R-01-CO2 used empirical (mixed-length) gaps, so they have a single value (shown separately).

Gap lengths: 1 h, 4 h, 32 h (~1.3 d), 288 h (12 d), 768 h (32 d), mixed.


## Tower 4

| Experiment | 1h | 4h | 32h | 288h | 768h | mixed |
|---|---|---|---|---|---|---|
| R-01 — Irvin RF/XGB (empirical ~40% gaps) | RF +0.14 (empirical) | — | — | — | — | — |
| R-01-CO2 — Irvin +FCO2 (empirical) | RF +0.05 (empirical) | — | — | — | — | — |
| **R-02** Zhu met-only | RFm -0.13 | RFm -0.10 | RF3 -0.10 | RFm -0.11 | — | RF3 -0.15 |
| **R-03** Kim +lags/PCA | RF +0.14 | — | ANN +0.09 | ANN +0.08 | ANN +0.06 | — |
| **R-02-CO2** Zhu +FCO2 | RFm +0.16 | RFm +0.08 | RFm +0.11 | RFm +0.03 | — | RFm -0.12 |
| **R-03-CO2** Kim +FCO2 | ANN +0.16 | — | ANN +0.17 | ANN +0.16 | ANN +0.12 | — |
| **F-01** feat ablation | P1_livestock +0.26 | P1_livestock +0.16 | P1_livestock +0.19 | BASE +0.03 | — | P5_soiltemp_prod -0.06 |
| **F-02** density+pool | ALL +0.26 | ALL +0.16 | POOLED_density +0.24 | wind +0.05 | — | ALL -0.02 |
| **F-03** partial pool | solo +0.25 | partial_pool +0.15 | partial_pool +0.24 | partial_pool +0.03 | — | solo -0.09 |
| **F-04** +SWC/TS lags | solo +0.25 | solo_lags +0.16 | partial_lags +0.24 | partial_lags +0.05 | — | solo_lags -0.05 |

## Tower 9

| Experiment | 1h | 4h | 32h | 288h | 768h | mixed |
|---|---|---|---|---|---|---|
| R-01 — Irvin RF/XGB (empirical ~40% gaps) | RF -0.03 (empirical) | — | — | — | — | — |
| R-01-CO2 — Irvin +FCO2 (empirical) | RF -0.03 (empirical) | — | — | — | — | — |
| **R-02** Zhu met-only | RFm -0.09 | RFm -0.09 | RFm -0.09 | RFm -0.16 | — | RFm -0.07 |
| **R-03** Kim +lags/PCA | RF_lag +0.15 | — | RF_lag +0.16 | RF_PCA7 +0.11 | RF_PCA7 +0.06 | — |
| **R-02-CO2** Zhu +FCO2 | RFm -0.03 | RFm -0.01 | RFm +0.00 | RFm -0.05 | — | RFm -0.07 |
| **R-03-CO2** Kim +FCO2 | RF +0.12 | — | RF_lag +0.11 | RF_PCA7 +0.06 | SVM +0.00 | — |
| **F-01** feat ablation | BASE -0.03 | BASE -0.01 | BASE +0.00 | BASE -0.05 | — | P1_livestock -0.02 |
| **F-02** density+pool | POOLED_density +0.29 | POOLED_density +0.22 | POOLED_density +0.29 | POOLED_density +0.21 | — | POOLED_density +0.25 |
| **F-03** partial pool | partial_pool +0.29 | partial_pool +0.22 | full_pool +0.29 | full_pool +0.21 | — | partial_pool +0.25 |
| **F-04** +SWC/TS lags | partial +0.29 | partial_lags +0.25 | partial_lags +0.30 | partial +0.20 | — | partial +0.25 |

## Tower 2

| Experiment | 1h | 4h | 32h | 288h | 768h | mixed |
|---|---|---|---|---|---|---|
| R-01 — Irvin RF/XGB (empirical ~40% gaps) | RF -16.94 (empirical) | — | — | — | — | — |
| R-01-CO2 — Irvin +FCO2 (empirical) | RF -4.85 (empirical) | — | — | — | — | — |
| **F-03** partial pool | partial_pool -0.24 | partial_pool -0.13 | partial_pool -0.22 | partial_pool -0.20 | — | solo -0.14 |
| **F-04** +SWC/TS lags | partial_lags -0.07 | partial_lags -0.06 | partial_lags -0.10 | partial_lags -0.07 | — | partial_lags -0.02 |
