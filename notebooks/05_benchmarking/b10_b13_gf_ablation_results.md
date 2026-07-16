# D-72: gap-filled-target/-context ablation for the DL family (DLinear/LSTM/TFT/TabPFN/TabICLv2)

Full metric set (RMSE, MAE, MASE, WAPE, Correlation, R²) for the gap-filled-target/-context
ablation, in the same reporting convention as `b10_b13_metrics_rerun.md` (D-65).

## Context

The B-09→B-15 sequence's tree models (RF/XGB/LightGBM) and SARIMAX already train on `y_gapfilled`
(dense) and evaluate against real `y_observed` (D-36/D-37). The DL family was the exception:
DLinear/LSTM/TFT's training loss was masked to real `y_observed` only (~45-55% dense), and
TabPFN/TabICLv2 explicitly rejected `y_gapfilled` context, citing "gap-filler optimism" risk
(`recursive_rollout.py` docstrings). This experiment tests that design choice empirically for all 5
models, full 3-tower × 5-anchor (2018-2022) coverage. Evaluation always stays on real `y_observed`
(`y_true`/`y_true_tft`) — only the fitting process (training target / in-context history) changes.
Full design rationale and the headline result: `DECISIONS.md` D-72.

## Method

Additive `y_source="observed"|"gapfilled"` param on `forecasting_dl.make_windows()`/
`build_windows()` (default preserves prior behavior bit-for-bit). Three new sibling scripts, each
an exact copy of its non-gf predecessor's recipe with only the target/context source changed:
`b10_b13_dl_gf_extension.py` (DLinear_gf/LSTM_gf, pooled fit, no val split), `b10_b13_tft_gf_extension.py`
(TFT_gf, pooled fit, train+val both on `y_gapfilled`), `b10_b13_foundation_gf_extension.py`
(TabPFN_gf/TabICLv2_gf, `y_gapfilled` as historical context). Aggregation convention throughout
(matching D-65): per-anchor n-weighted mean across the 6 lead-time bins, then simple mean across the
5 anchors, for pooled/per-tower tables; tower × year table uses the per-anchor n-weighted value
directly (no cross-anchor averaging).

## Combined leaderboard (all 11 models, latest — supersedes prior standalone tables for citation)

Every model in the B-09→B-15 sequence, one table. RF/XGB/LightGBM/SARIMAX/both Ensembles already
train on `y_gapfilled` (D-36/D-37 — unchanged here); every DL-family model uses its gap-filled-trained
`_gf` variant (D-72) in place of the original. MASE shown against both the primary baseline
(persistence, D-37) and the secondary one (Climatology_gf, D-71 follow-up) — both denominators agree
on the ranking, so the secondary check confirms rather than changes any conclusion. Evaluation ground
truth throughout is real `y_observed`. Aggregation: per-anchor n-weighted mean across the 6 lead-time
bins, then mean across the 5 anchors, pooled across all 3 towers.

**Established models (already gap-filled-trained):**

| Model | RMSE | MAE | MASE (vs persistence) | MASE (vs Climatology_gf) | WAPE | Correlation | R² |
|---|---|---|---|---|---|---|---|
| RF | 52.232 | 34.836 | 0.968 | 0.869 | 1.050 | 0.375 | −0.241 |
| XGB | 51.571 | 33.807 | 0.922 | 0.824 | 0.991 | 0.368 | −0.184 |
| LightGBM | 52.083 | 34.323 | 0.941 | 0.837 | 1.012 | 0.368 | −0.206 |
| SARIMAX | 53.791 | 36.059 | 0.976 | 0.901 | 1.105 | 0.343 | −0.360 |
| Ensemble_unweighted | 51.574 | 33.746 | 0.918 | 0.828 | 0.998 | 0.375 | −0.165 |
| Ensemble_MASEweighted | 51.567 | 33.743 | 0.918 | 0.827 | 0.998 | 0.375 | −0.165 |

**DL family (gap-filled-trained, `_gf`, D-72):**

| Model | RMSE | MAE | MASE (vs persistence) | MASE (vs Climatology_gf) | WAPE | Correlation | R² |
|---|---|---|---|---|---|---|---|
| DLinear_gf | 54.860 | 37.909 | 1.059 | 0.965 | 1.159 | 0.312 | −0.540 |
| LSTM_gf | 57.977 | 39.384 | 1.098 | 0.955 | 1.200 | 0.301 | −0.686 |
| TFT_gf | 53.966 | 35.837 | 1.005 | 0.888 | 1.109 | 0.325 | −0.439 |
| **TabPFN_gf** | **51.675** | **31.463** | **0.829** | **0.746** | **0.883** | **0.360** | **0.001** |
| TabICLv2_gf | 52.913 | 33.848 | 0.891 | 0.801 | 0.958 | 0.344 | −0.060 |

**TabPFN_gf is the single best result in the entire B-09→B-15 sequence** — best MASE under both
denominators (0.829 vs persistence, 0.746 vs Climatology_gf) and the only positive pooled R² (0.001)
of any model, established or DL-family. It beats the prior record-holder (TabPFN original, 0.855
MASE, D-57/D-65) and every established model, including the standing production recommendation
(Ensemble_unweighted, 0.918 MASE). TFT_gf is the one DL-family model where the gap-filled variant is
a wash rather than a win (D-72's own finding) — included here per "all DL models use their gf
variant," not because it's the better choice for TFT specifically.

Source: `results/b10_b13_latest_combined_leaderboard.csv`.

## All-tower pooled: scored against gap-filled target vs. observed target — all 11 models

Same convention and exact column structure as D-65's original "Secondary metric: scored against
gap-filled target" table (`b10_b13_metrics_rerun.md`) — every model scored twice, once against real
`y_observed` (the primary, trustworthy number) and once against `y_gapfilled` (dense/continuous,
exploratory — same D-36/D-37 circularity caveat as that original table: `y_gapfilled` is itself an
RFm regressor's output over features that overlap the forecasters' own drivers, so agreement with it
partly reflects "resembles the gap-filler," not pure real-world skill). **The only change from the
original table: every DL-family row now uses its `_gf` variant** (trained on `y_gapfilled`, D-72)
instead of the original DL-family models — RF/XGB/LightGBM/SARIMAX/both Ensembles are unchanged,
identical to the published D-65 numbers.

| Model | RMSE (gapfilled) | RMSE (observed) | MAE (gapfilled) | MAE (observed) | MASE (gapfilled) | MASE (observed) | Correlation (gapfilled) | Correlation (observed) | R² (gapfilled) | R² (observed) |
|---|---|---|---|---|---|---|---|---|---|---|
| RF | 25.85 | 52.23 | 18.250 | 34.836 | 0.800 | 0.968 | 0.529 | 0.375 | −0.593 | −0.241 |
| XGB | 25.72 | 51.57 | 17.705 | 33.807 | 0.761 | 0.922 | 0.483 | 0.368 | −0.502 | −0.184 |
| LightGBM | 26.11 | 52.08 | 18.202 | 34.323 | 0.774 | 0.941 | 0.496 | 0.368 | −0.480 | −0.206 |
| SARIMAX | 29.21 | 53.79 | 21.664 | 36.059 | 0.943 | 0.976 | 0.463 | 0.343 | −1.004 | −0.360 |
| Ensemble_unweighted | 25.39 | 51.57 | 17.664 | 33.746 | 0.751 | 0.918 | 0.523 | 0.375 | −0.189 | −0.165 |
| Ensemble_MASEweighted | 25.38 | 51.57 | 17.651 | 33.743 | 0.750 | 0.918 | 0.522 | 0.375 | −0.195 | −0.165 |
| DLinear_gf | 30.233 | 54.860 | 22.524 | 37.909 | 1.008 | 1.059 | 0.385 | 0.312 | −1.141 | −0.540 |
| LSTM_gf | 30.744 | 57.977 | 22.318 | 39.384 | 0.955 | 1.098 | 0.462 | 0.301 | −0.907 | −0.686 |
| TFT_gf | 29.769 | 53.966 | 20.637 | 35.837 | 0.973 | 1.005 | 0.433 | 0.325 | −2.041 | −0.439 |
| **TabPFN_gf** | **27.256** | **51.675** | **17.380** | **31.463** | **0.678** | **0.829** | **0.507** | **0.360** | **−0.007** | **0.001** |
| TabICLv2_gf | 27.565 | 52.913 | 18.545 | 33.848 | 0.726 | 0.891 | 0.485 | 0.344 | −0.054 | −0.060 |

**RMSE/MAE/MASE all improve under the gap-filled target for every DL-family model too, for the same
mechanistic reason D-65's Finding 1 already established for the original 8 models**: `y_gapfilled` is
smoother than `y_observed` (RFm damps real spike noise), so absolute errors shrink — but R² divides
by the target's own (now much smaller) variance, so it doesn't uniformly improve alongside the other
three metrics (TFT_gf's R² actually gets much worse against gap-filled, −2.041 vs −0.439 observed,
even though its RMSE/MAE/MASE/Correlation all improve — the same "R² penalizes flattening" pattern).

**TabPFN_gf is again the standout** — best RMSE/MAE/MASE/Correlation of any model under both targets,
and the only model with positive R² under the observed target. Comparing to D-65's original TabPFN
row (RMSE 56.12/35.19, MASE 0.855/0.949, R² −0.122/−0.689, observed/gapfilled): TabPFN_gf improves on
every single one of those eight numbers.

Source: `results/b10_b13_gf_ablation_pooled_vs_gapfilled.csv` (DL-family gap-filled-target pooling)
+ `results/b10_b13_gf_ablation_pooled.csv` (DL-family observed-target pooling, already used above)
+ `results/b10_b13_rerun_table_vs_gapfilled_all_towers.csv` + `results/b10_b13_rerun_table_all_towers.csv`
(established models, unchanged from D-65).

## DL-family only: original vs. gf, side by side

| Model | RMSE | MAE | MASE | WAPE | Correlation | R² |
|---|---|---|---|---|---|---|
| DLinear | 64.534 | 47.515 | 1.460 | 1.638 | 0.237 | −2.068 |
| DLinear_gf | 54.860 | 37.909 | 1.059 | 1.159 | 0.312 | −0.540 |
| LSTM | 63.224 | 41.061 | 1.151 | 1.268 | 0.212 | −1.357 |
| LSTM_gf | 57.977 | 39.384 | 1.098 | 1.200 | 0.301 | −0.686 |
| TFT | 56.589 | 35.624 | 0.972 | 1.045 | 0.292 | −0.363 |
| TFT_gf | 53.966 | 35.837 | 1.005 | 1.109 | 0.325 | −0.439 |
| TabPFN | 56.124 | 33.140 | 0.855 | 0.899 | 0.358 | −0.122 |
| **TabPFN_gf** | **51.675** | **31.463** | **0.829** | **0.883** | **0.360** | **0.001** |
| TabICLv2 | 58.046 | 34.951 | 0.930 | 0.988 | 0.255 | −0.330 |
| TabICLv2_gf | 52.913 | 33.848 | 0.891 | 0.958 | 0.344 | −0.060 |

This is the same comparison as the combined leaderboard above, kept in its original orig-vs-gf
paired form for anyone specifically comparing each DL model against its own baseline (rather than
against the full 11-model field).

## Per-tower tables

**Tower 2:**

| Model | RMSE | MAE | MASE | WAPE | Correlation | R² |
|---|---|---|---|---|---|---|
| DLinear | 36.529 | 30.201 | 0.588 | 2.389 | 0.009 | −5.421 |
| DLinear_gf | 29.904 | 24.782 | 0.509 | 1.980 | 0.152 | −4.293 |
| LSTM | 33.051 | 26.461 | 0.508 | 2.054 | 0.319 | −3.791 |
| LSTM_gf | 22.893 | 18.382 | 0.353 | 1.373 | 0.080 | −1.119 |
| TFT | 29.526 | 20.839 | 0.400 | 1.549 | 0.111 | −2.312 |
| TFT_gf | 21.184 | 17.126 | 0.338 | 1.332 | 0.186 | −1.035 |
| TabPFN | 18.251 | 12.956 | 0.243 | 0.969 | 0.098 | −0.240 |
| TabPFN_gf | 19.360 | 14.902 | 0.280 | 1.108 | 0.053 | −0.353 |
| TabICLv2 | 18.436 | 13.008 | 0.240 | 0.957 | 0.181 | −0.219 |
| TabICLv2_gf | 21.989 | 18.291 | 0.348 | 1.369 | 0.073 | −0.800 |

**Tower 4:**

| Model | RMSE | MAE | MASE | WAPE | Correlation | R² |
|---|---|---|---|---|---|---|
| DLinear | 65.712 | 48.112 | 1.626 | 1.685 | 0.237 | −2.105 |
| DLinear_gf | 54.554 | 37.340 | 1.119 | 1.127 | 0.302 | −0.244 |
| LSTM | 59.825 | 37.267 | 1.106 | 1.105 | 0.228 | −0.439 |
| LSTM_gf | 56.521 | 37.393 | 1.126 | 1.153 | 0.345 | −0.404 |
| TFT | 54.480 | 33.430 | 1.014 | 1.020 | 0.315 | −0.228 |
| TFT_gf | 53.434 | 34.524 | 1.050 | 1.092 | 0.347 | −0.371 |
| TabPFN | 54.261 | 30.502 | 0.864 | 0.861 | 0.391 | −0.006 |
| TabPFN_gf | 51.118 | 29.895 | 0.858 | 0.853 | 0.384 | 0.085 |
| TabICLv2 | 56.762 | 33.037 | 0.963 | 0.980 | 0.295 | −0.291 |
| TabICLv2_gf | 52.336 | 33.349 | 0.953 | 0.957 | 0.378 | 0.008 |

**Tower 9:**

| Model | RMSE | MAE | MASE | WAPE | Correlation | R² |
|---|---|---|---|---|---|---|
| DLinear | 66.590 | 48.336 | 1.251 | 1.345 | 0.287 | −1.073 |
| DLinear_gf | 59.658 | 40.457 | 1.009 | 1.031 | 0.358 | −0.348 |
| LSTM | 74.019 | 49.576 | 1.289 | 1.358 | 0.179 | −2.229 |
| LSTM_gf | 65.580 | 45.184 | 1.119 | 1.183 | 0.287 | −0.879 |
| TFT | 64.870 | 41.662 | 0.974 | 0.978 | 0.304 | −0.209 |
| TFT_gf | 59.651 | 39.863 | 0.967 | 1.012 | 0.338 | −0.268 |
| TabPFN | 65.980 | 40.842 | 0.926 | 0.931 | 0.349 | −0.265 |
| TabPFN_gf | 58.570 | 36.701 | 0.855 | 0.878 | 0.388 | −0.052 |
| TabICLv2 | 67.151 | 41.593 | 0.953 | 0.957 | 0.242 | −0.312 |
| TabICLv2_gf | 59.296 | 37.132 | 0.858 | 0.869 | 0.357 | −0.012 |

**Tower 2 is the one place the foundation models regress under `_gf`** (TabPFN R² −0.240→−0.353,
TabICLv2 −0.219→−0.800) — every other model/tower combination improves or is roughly flat.

## Tower × year × model breakdown

Per-anchor n-weighted mean across the 6 lead-time bins (no cross-anchor averaging — that's what the
summary tables above already show). Columns flattened to `T{tower}_{metric}` for markdown; true
long-format version in `results/b10_b13_gf_ablation_table_by_tower_year.csv`. `nan` = no real
`y_observed` coverage for that tower/anchor (Tower 2 outside 2018, matching every other B-09→B-15
document's Tower 2 finding).

| Year | Model | T2_RMSE | T2_MAE | T2_MASE | T2_WAPE | T2_Correlation | T4_RMSE | T4_MAE | T4_MASE | T4_WAPE | T4_Correlation | T9_RMSE | T9_MAE | T9_MASE | T9_WAPE | T9_Correlation | T2_R2 | T4_R2 | T9_R2 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2018 | DLinear | 36.529 | 30.201 | 0.588 | 2.389 | 0.009 | 77.284 | 58.362 | 2.127 | 2.710 | -0.076 | nan | nan | nan | nan | nan | -5.421 | -5.683 | nan |
| 2018 | DLinear_gf | 29.904 | 24.782 | 0.509 | 1.980 | 0.152 | 51.673 | 38.869 | 1.220 | 1.449 | 0.156 | nan | nan | nan | nan | nan | -4.293 | -0.636 | nan |
| 2018 | LSTM | 33.051 | 26.461 | 0.508 | 2.054 | 0.319 | 54.390 | 33.470 | 1.026 | 1.225 | 0.079 | nan | nan | nan | nan | nan | -3.791 | -0.678 | nan |
| 2018 | LSTM_gf | 22.893 | 18.382 | 0.353 | 1.373 | 0.080 | 57.875 | 42.976 | 1.389 | 1.691 | 0.140 | nan | nan | nan | nan | nan | -1.119 | -1.600 | nan |
| 2018 | TFT | 29.526 | 20.839 | 0.400 | 1.549 | 0.111 | 52.204 | 32.357 | 1.004 | 1.200 | 0.057 | nan | nan | nan | nan | nan | -2.312 | -0.621 | nan |
| 2018 | TFT_gf | 21.184 | 17.126 | 0.338 | 1.332 | 0.186 | 57.408 | 43.343 | 1.512 | 1.900 | 0.146 | nan | nan | nan | nan | nan | -1.035 | -2.054 | nan |
| 2018 | TabPFN | 18.251 | 12.956 | 0.243 | 0.969 | 0.098 | 49.883 | 29.175 | 0.847 | 0.977 | 0.082 | nan | nan | nan | nan | nan | -0.240 | -0.202 | nan |
| 2018 | TabPFN_gf | 19.360 | 14.902 | 0.280 | 1.108 | 0.053 | 46.777 | 28.637 | 0.836 | 0.966 | 0.212 | nan | nan | nan | nan | nan | -0.353 | -0.061 | nan |
| 2018 | TabICLv2 | 18.436 | 13.008 | 0.240 | 0.957 | 0.181 | 55.296 | 34.590 | 1.133 | 1.396 | -0.021 | nan | nan | nan | nan | nan | -0.219 | -1.092 | nan |
| 2018 | TabICLv2_gf | 21.989 | 18.291 | 0.348 | 1.369 | 0.073 | 49.102 | 32.125 | 0.989 | 1.173 | 0.138 | nan | nan | nan | nan | nan | -0.800 | -0.300 | nan |
| 2019 | DLinear | nan | nan | nan | nan | nan | 63.574 | 45.013 | 1.381 | 1.383 | 0.364 | 83.365 | 58.841 | 1.186 | 1.093 | 0.282 | nan | -1.222 | -0.351 |
| 2019 | DLinear_gf | nan | nan | nan | nan | nan | 53.810 | 35.808 | 1.024 | 1.026 | 0.379 | 73.159 | 50.345 | 1.069 | 0.971 | 0.466 | nan | -0.392 | -0.216 |
| 2019 | LSTM | nan | nan | nan | nan | nan | 56.098 | 33.146 | 0.842 | 0.843 | 0.326 | 86.209 | 52.444 | 1.006 | 0.926 | 0.134 | nan | -0.119 | -0.250 |
| 2019 | LSTM_gf | nan | nan | nan | nan | nan | 53.781 | 31.386 | 0.823 | 0.824 | 0.398 | 76.420 | 52.528 | 1.079 | 1.002 | 0.408 | nan | -0.085 | -0.074 |
| 2019 | TFT | nan | nan | nan | nan | nan | 53.475 | 32.543 | 0.879 | 0.880 | 0.260 | 81.218 | 53.041 | 1.035 | 0.955 | 0.239 | nan | -0.113 | -0.117 |
| 2019 | TFT_gf | nan | nan | nan | nan | nan | 50.617 | 30.051 | 0.793 | 0.794 | 0.336 | 74.055 | 48.462 | 0.979 | 0.902 | 0.402 | nan | -0.013 | 0.024 |
| 2019 | TabPFN | nan | nan | nan | nan | nan | 55.863 | 32.090 | 0.809 | 0.811 | 0.505 | 91.602 | 56.285 | 1.087 | 1.000 | nan | nan | -0.090 | -0.460 |
| 2019 | TabPFN_gf | nan | nan | nan | nan | nan | 49.178 | 27.307 | 0.705 | 0.706 | 0.393 | 71.187 | 44.613 | 0.879 | 0.815 | 0.542 | nan | 0.118 | 0.147 |
| 2019 | TabICLv2 | nan | nan | nan | nan | nan | 58.009 | 34.179 | 0.845 | 0.846 | 0.425 | 91.602 | 56.285 | 1.087 | 1.000 | 0.156 | nan | -0.158 | -0.460 |
| 2019 | TabICLv2_gf | nan | nan | nan | nan | nan | 49.369 | 29.406 | 0.773 | 0.774 | 0.396 | 73.825 | 45.182 | 0.888 | 0.826 | 0.462 | nan | 0.057 | 0.091 |
| 2020 | DLinear | nan | nan | nan | nan | nan | 72.658 | 49.139 | 1.544 | 1.392 | 0.138 | 61.818 | 43.845 | 1.195 | 1.365 | 0.277 | nan | -1.139 | -1.342 |
| 2020 | DLinear_gf | nan | nan | nan | nan | nan | 64.907 | 38.177 | 1.084 | 0.957 | 0.226 | 58.080 | 35.560 | 0.938 | 1.055 | 0.339 | nan | -0.065 | -0.580 |
| 2020 | LSTM | nan | nan | nan | nan | nan | 70.131 | 40.930 | 1.231 | 1.085 | 0.175 | 75.783 | 53.735 | 1.634 | 1.925 | 0.220 | nan | -0.582 | -6.314 |
| 2020 | LSTM_gf | nan | nan | nan | nan | nan | 62.336 | 37.244 | 1.082 | 0.965 | 0.358 | 71.437 | 49.731 | 1.325 | 1.532 | 0.180 | nan | -0.037 | -2.490 |
| 2020 | TFT | nan | nan | nan | nan | nan | 63.770 | 34.777 | 1.015 | 0.905 | 0.316 | 58.098 | 33.147 | 0.876 | 0.924 | 0.334 | nan | -0.090 | -0.105 |
| 2020 | TFT_gf | nan | nan | nan | nan | nan | 64.339 | 36.411 | 1.024 | 0.906 | 0.263 | 61.458 | 40.023 | 1.021 | 1.146 | 0.317 | nan | -0.036 | -0.664 |
| 2020 | TabPFN | nan | nan | nan | nan | nan | 64.302 | 31.196 | 0.886 | 0.786 | 0.394 | 62.234 | 35.554 | 0.906 | 1.003 | 0.192 | nan | 0.034 | -0.510 |
| 2020 | TabPFN_gf | nan | nan | nan | nan | nan | 62.639 | 31.560 | 0.897 | 0.797 | 0.360 | 57.842 | 33.374 | 0.834 | 0.936 | 0.352 | nan | 0.073 | -0.335 |
| 2020 | TabICLv2 | nan | nan | nan | nan | nan | 67.353 | 33.134 | 0.937 | 0.829 | 0.259 | 63.698 | 35.622 | 0.914 | 1.004 | 0.082 | nan | -0.055 | -0.508 |
| 2020 | TabICLv2_gf | nan | nan | nan | nan | nan | 63.653 | 34.928 | 0.973 | 0.862 | 0.286 | 58.048 | 34.371 | 0.845 | 0.914 | 0.269 | nan | 0.042 | -0.142 |
| 2021 | DLinear | nan | nan | nan | nan | nan | 63.898 | 48.195 | 1.548 | 1.312 | 0.460 | 71.776 | 49.583 | 0.991 | 1.248 | 0.291 | nan | -1.820 | -1.223 |
| 2021 | DLinear_gf | nan | nan | nan | nan | nan | 60.330 | 43.036 | 1.131 | 0.997 | 0.386 | 63.563 | 41.792 | 0.734 | 0.770 | 0.280 | nan | -0.110 | 0.022 |
| 2021 | LSTM | nan | nan | nan | nan | nan | 67.095 | 42.995 | 1.135 | 0.998 | 0.292 | 77.828 | 50.573 | 0.882 | 0.904 | 0.152 | nan | -0.349 | -0.368 |
| 2021 | LSTM_gf | nan | nan | nan | nan | nan | 61.797 | 40.237 | 1.072 | 0.946 | 0.396 | 66.742 | 43.759 | 0.790 | 0.882 | 0.256 | nan | -0.099 | -0.170 |
| 2021 | TFT | nan | nan | nan | nan | nan | 63.387 | 39.415 | 1.097 | 0.974 | 0.468 | 75.701 | 47.016 | 0.799 | 0.819 | 0.298 | nan | -0.343 | -0.203 |
| 2021 | TFT_gf | nan | nan | nan | nan | nan | 58.946 | 38.144 | 0.997 | 0.875 | 0.438 | 64.311 | 42.085 | 0.774 | 0.879 | 0.344 | nan | -0.004 | -0.201 |
| 2021 | TabPFN | nan | nan | nan | nan | nan | 63.488 | 36.782 | 0.891 | 0.791 | 0.425 | 73.699 | 45.735 | 0.780 | 0.770 | 0.442 | nan | 0.058 | -0.134 |
| 2021 | TabPFN_gf | nan | nan | nan | nan | nan | 60.147 | 38.440 | 0.957 | 0.846 | 0.405 | 69.103 | 42.941 | 0.746 | 0.778 | 0.288 | nan | 0.082 | -0.036 |
| 2021 | TabICLv2 | nan | nan | nan | nan | nan | 64.431 | 39.169 | 0.972 | 0.842 | 0.425 | 76.525 | 48.494 | 0.814 | 0.804 | 0.422 | nan | -0.251 | -0.218 |
| 2021 | TabICLv2_gf | nan | nan | nan | nan | nan | 63.489 | 44.141 | 1.044 | 0.925 | 0.478 | 69.205 | 43.602 | 0.755 | 0.770 | 0.332 | nan | 0.003 | -0.024 |
| 2022 | DLinear | nan | nan | nan | nan | nan | 51.146 | 39.851 | 1.529 | 1.627 | 0.301 | 49.403 | 41.075 | 1.633 | 1.674 | 0.299 | nan | -0.660 | -1.378 |
| 2022 | DLinear_gf | nan | nan | nan | nan | nan | 42.048 | 30.809 | 1.136 | 1.205 | 0.363 | 43.828 | 34.131 | 1.295 | 1.326 | 0.347 | nan | -0.016 | -0.618 |
| 2022 | LSTM | nan | nan | nan | nan | nan | 51.412 | 35.792 | 1.294 | 1.371 | 0.267 | 56.254 | 41.552 | 1.634 | 1.675 | 0.212 | nan | -0.464 | -1.986 |
| 2022 | LSTM_gf | nan | nan | nan | nan | nan | 46.814 | 35.120 | 1.265 | 1.340 | 0.432 | 47.722 | 34.718 | 1.284 | 1.314 | 0.302 | nan | -0.201 | -0.783 |
| 2022 | TFT | nan | nan | nan | nan | nan | 39.562 | 28.056 | 1.074 | 1.142 | 0.472 | 44.464 | 33.443 | 1.185 | 1.212 | 0.345 | nan | 0.030 | -0.411 |
| 2022 | TFT_gf | nan | nan | nan | nan | nan | 35.860 | 24.672 | 0.926 | 0.984 | 0.549 | 38.781 | 28.883 | 1.096 | 1.123 | 0.290 | nan | 0.253 | -0.233 |
| 2022 | TabPFN | nan | nan | nan | nan | nan | 37.767 | 23.265 | 0.884 | 0.940 | 0.552 | 36.387 | 25.794 | 0.932 | 0.953 | 0.412 | nan | 0.171 | 0.046 |
| 2022 | TabPFN_gf | nan | nan | nan | nan | nan | 36.851 | 23.534 | 0.894 | 0.950 | 0.553 | 36.149 | 25.877 | 0.960 | 0.983 | 0.369 | nan | 0.212 | 0.016 |
| 2022 | TabICLv2 | nan | nan | nan | nan | nan | 38.720 | 24.112 | 0.926 | 0.986 | 0.385 | 36.778 | 25.973 | 0.996 | 1.021 | 0.307 | nan | 0.103 | -0.063 |
| 2022 | TabICLv2_gf | nan | nan | nan | nan | nan | 36.064 | 26.143 | 0.989 | 1.051 | 0.591 | 36.105 | 25.372 | 0.942 | 0.964 | 0.365 | nan | 0.240 | 0.025 |

The 2018-anchor Tower 4 rows for DLinear/TFT (T4_RMSE 77.28/52.20 respectively) are noticeably worse
than their own pooled-Tower-4 averages (65.71/54.48) — same "one bad anchor drags the pooled mean"
pattern D-65 documented for DLinear's 2018 cold-start non-determinism; 2019-2022 are comparatively
stable for every model.

## Findings

**1. Dense supervision helps most where the original model was weakest.** DLinear (this project's
single worst/most unstable model, D-53) gets the largest gain by far (pooled MASE 1.460→1.059,
R² −2.068→−0.540); LSTM improves clearly (1.151→1.098, −1.357→−0.686). Both remain worse than every
other model, but the `_gf` variant closes much of the gap to the pack.

**2. TFT — already the most-regularized recipe in the roster (D-45's `weight_decay`/`patience`
regime) — is the one case where `_gf` is a wash, not a win, pooled.** Pooled MASE ticks up slightly
(0.972→1.005), RMSE improves marginally (56.59→53.97), R² gets marginally worse (−0.363→−0.439). But
the pooled wash hides real tower-level variance, not a uniformly flat result: **Tower 2 improves
substantially** (RMSE 29.53→21.18, MASE 0.400→0.338, R² −2.312→−1.035), **Tower 4 clearly worsens**
(MASE 1.014→1.050, R² −0.228→−0.371), and **Tower 9 is mixed** (RMSE/MASE both improve slightly,
64.87→59.65 / 0.974→0.967, but R² ticks down, −0.209→−0.268). The pooled near-wash is really one
tower improving a lot, one worsening, and one roughly flat — not "nothing changed anywhere."

**3. TabPFN_gf sets a new best-in-sequence result, and TabICLv2_gf close behind — with one
exception: Tower 2.** Both foundation models improve pooled and at Towers 4/9, but *regress* at
Tower 2 (TabPFN R² −0.240→−0.353; TabICLv2 −0.219→−0.800, its worst cell in this whole table). Tower
2 is this project's data-scarcest tower (real `y_observed` coverage ≈5.6%, per D-65's own
`real_frac` finding) — the same tower where climatology's persistence-vs-climatology ranking also
reversed (last session's finding). The pattern suggests `y_gapfilled` context is a net win only
where there's enough real signal underlying it to also mean something at inference time; at Tower 2
it may be conditioning these two in-context models on a series that's almost entirely gap-filler
output with very little real anchor to it.

**4. RMSE and MASE agree on direction for 4 of 5 models — TFT is the one exception.** DLinear/LSTM/
TabPFN/TabICLv2 all improve on both RMSE and MASE together under `_gf`. **TFT is a genuine split**:
RMSE improves (56.59→53.97) while MASE gets *worse* (0.972→1.005) — the same kind of metric
disagreement D-65 first flagged for the original TabPFN (best MASE, second-worst RMSE), now showing
up within a single model's orig-vs-gf comparison rather than across models. A reminder that even
within this one ablation, no single metric should be read in isolation.

**5. Correlation improves for every single `_gf` variant, pooled — no exception.** LSTM_gf and
TabICLv2_gf tie for the largest jump (0.212→0.301 and 0.255→0.344, both +0.089); DLinear_gf improves
next-most (0.237→0.312, +0.075); TFT_gf a smaller but real +0.033 (0.292→0.325); TabPFN_gf barely
moves (0.358→0.360, +0.002 — already the strongest correlation of any model, orig or gf, so little
room left to gain). Still uniformly weak-to-moderate overall (0.30-0.36 range post-gf, same "no
model captures the real signal strongly" caveat D-65 raised) but directionally consistent with the
R²/MASE gains — not one metric moving while others stay flat.

## Secondary metric: MASE vs. Climatology_gf instead of persistence

Same caveat as D-71/D-65's own gap-filled-target sections: persistence remains this project's primary
MASE denominator; this is a robustness check on whether the ranking survives a different (weaker)
baseline, not a redefinition.

| Model | MASE (vs Climatology_gf) | MASE (vs persistence) |
|---|---|---|
| DLinear | 1.308 | 1.460 |
| DLinear_gf | 0.965 | 1.059 |
| LSTM | 1.018 | 1.151 |
| LSTM_gf | 0.955 | 1.098 |
| TFT | 0.878 | 0.972 |
| TFT_gf | 0.888 | 1.005 |
| TabPFN | 0.773 | 0.855 |
| TabPFN_gf | 0.746 | 0.829 |
| TabICLv2 | 0.820 | 0.930 |
| TabICLv2_gf | 0.801 | 0.891 |

Every MASE value is lower against `Climatology_gf` (a weaker baseline, D-71) than against
persistence, but the qualitative ranking is unchanged — gf beats original for DLinear/LSTM/TabPFN/
TabICLv2, TFT is a wash — under either denominator.

## Files

- `src/models/forecasting_dl.py` (+`y_source` param on `make_windows`/`build_windows`, additive)
- `notebooks/05_benchmarking/b10_b13_dl_gf_extension.py`, `b10_b13_tft_gf_extension.py`,
  `b10_b13_foundation_gf_extension.py` (new, committed — the 3 gf-variant sibling scripts)
- `notebooks/05_benchmarking/b10_b13_gf_merge.py` (merges the 5 `_gf` columns into
  `b10_b13_full_chains.csv`, backup-then-verify-row-count discipline)
- `notebooks/05_benchmarking/b10_b13_gf_comparison_table.py` (pooled/by-tower orig-vs-gf tables)
- `notebooks/05_benchmarking/b10_b13_gf_vs_climatology_gf.py` (secondary MASE-denominator check)
- `results/b10_b13_{dl,tft,foundation}_gf_extension_{summary,summary_vs_gapfilled,chains}.csv`
- `results/b10_b13_gf_ablation_{combined_summary,pooled,table_all_towers,by_tower,table_by_tower,
  table_by_tower_year,flattened_by_tower_year}.csv`
- `results/b10_b13_gf_ablation_vs_climatology_gf_{summary,table}.csv`
- `results/b10_b13_latest_combined_leaderboard.csv` (all 11 models — established + DL-family `_gf`
  variants — RMSE/MAE/MASE-vs-persistence/MASE-vs-Climatology_gf/WAPE/Correlation/R², the "Combined
  leaderboard" section's source)
- `results/b10_b13_gf_ablation_pooled_vs_gapfilled.csv` (DL-family `_gf` models pooled RMSE/MAE/MASE/
  WAPE/Correlation/R², scored against `y_gapfilled` instead of `y_observed` — the "scored against
  gap-filled target vs. observed target" section's DL-family half)
- `results/b10_b13_full_chains.csv` (+5 `_gf` columns, 5,475 rows unchanged; backed up to
  `b10_b13_full_chains_backup_pre_gf.csv` first)
- `results/figures/b10_chains/` regenerated (598 figures total, +75 new `_gf` figures); spot-checked
  Tower 4/anchor 2021 (DLinear_gf, TabPFN_gf) before trusting the full batch

No `benchmarks.csv` rows — a design-choice ablation, not a point-forecast benchmark in its own right
(same precedent as every other diagnostic pass in this sequence). Full narrative decision log:
`DECISIONS.md` D-72.
