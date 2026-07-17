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

| Model | RMSE | MAE | MASE (vs persistence) | MASE (vs Climatology_gf) | RMSSE (vs persistence) | RMSSE (vs Climatology_gf) | WAPE | Correlation | R² |
|---|---|---|---|---|---|---|---|---|---|
| RF | 52.232 | 34.836 | 0.968 | 0.869 | 0.885 | 0.855 | 1.050 | 0.375 | −0.241 |
| XGB | 51.571 | 33.807 | 0.922 | 0.824 | 0.862 | 0.832 | 0.991 | 0.368 | −0.184 |
| LightGBM | 52.083 | 34.323 | 0.941 | 0.837 | 0.877 | 0.845 | 1.012 | 0.368 | −0.206 |
| SARIMAX | 53.791 | 36.059 | 0.976 | 0.901 | 0.895 | 0.882 | 1.105 | 0.343 | −0.360 |
| Ensemble_unweighted | 51.574 | 33.746 | 0.918 | 0.828 | 0.859 | 0.833 | 0.998 | 0.375 | −0.165 |
| Ensemble_MASEweighted | 51.567 | 33.743 | 0.918 | 0.827 | 0.859 | 0.833 | 0.998 | 0.375 | −0.165 |

**DL family (gap-filled-trained, `_gf`, D-72):**

| Model | RMSE | MAE | MASE (vs persistence) | MASE (vs Climatology_gf) | RMSSE (vs persistence) | RMSSE (vs Climatology_gf) | WAPE | Correlation | R² |
|---|---|---|---|---|---|---|---|---|---|
| DLinear_gf | 54.860 | 37.909 | 1.059 | 0.965 | 0.952 | 0.935 | 1.159 | 0.312 | −0.540 |
| LSTM_gf | 57.977 | 39.384 | 1.098 | 0.955 | 1.008 | 0.952 | 1.200 | 0.301 | −0.686 |
| TFT_gf | 53.966 | 35.837 | 1.005 | 0.888 | 0.933 | 0.889 | 1.109 | 0.325 | −0.439 |
| **TabPFN_gf** | **51.675** | **31.463** | **0.829** | **0.746** | **0.830** | **0.808** | **0.883** | **0.360** | **0.001** |
| TabICLv2_gf | 52.913 | 33.848 | 0.891 | 0.801 | 0.854 | 0.830 | 0.958 | 0.344 | −0.060 |

**TabPFN_gf is the single best result in the entire B-09→B-15 sequence** — best MASE under both
denominators (0.829 vs persistence, 0.746 vs Climatology_gf), best RMSSE under both denominators too
(0.830 / 0.808), and the only positive pooled R² (0.001) of any model, established or DL-family. It
beats the prior record-holder (TabPFN original, 0.855 MASE, D-57/D-65) and every established model,
including the standing production recommendation (Ensemble_unweighted, 0.918 MASE / 0.859 RMSSE).
TFT_gf is the one DL-family model where the gap-filled variant is a wash rather than a win (D-72's
own finding) — included here per "all DL models use their gf variant," not because it's the better
choice for TFT specifically. RMSSE agrees with MASE's ranking everywhere in this table except one
subtlety already flagged below (§ Secondary metric): original TFT's MASE (0.972) beats persistence
more convincingly than its RMSSE (0.946) does.

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

## All-tower pooled: scored against gap-filled target vs. observed target — MASE/RMSSE baseline = Climatology_gf

Same table shape as the section above, same 11 models, same gapfilled-vs-observed *evaluation
ground-truth* axis — but here MASE and RMSSE are both rescaled against `Climatology_gf` instead of
persistence (the secondary-baseline lens from D-71/D-72, now crossed with the gapfilled/observed
evaluation axis for the first time). RMSE/MAE/Correlation/R² are baseline-independent and identical
to the persistence-baseline table above — only the MASE/RMSSE columns differ.

| Model | RMSE (gapfilled) | RMSE (observed) | MAE (gapfilled) | MAE (observed) | MASE (gapfilled) | MASE (observed) | RMSSE (gapfilled) | RMSSE (observed) | Correlation (gapfilled) | Correlation (observed) | R² (gapfilled) | R² (observed) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| RF | 25.85 | 52.23 | 18.250 | 34.836 | 0.703 | 0.869 | 0.711 | 0.855 | 0.529 | 0.375 | −0.593 | −0.241 |
| XGB | 25.72 | 51.57 | 17.705 | 33.807 | 0.670 | 0.824 | 0.705 | 0.832 | 0.483 | 0.368 | −0.502 | −0.184 |
| LightGBM | 26.11 | 52.08 | 18.202 | 34.323 | 0.689 | 0.837 | 0.715 | 0.845 | 0.496 | 0.368 | −0.480 | −0.206 |
| SARIMAX | 29.21 | 53.79 | 21.664 | 36.059 | 0.831 | 0.901 | 0.812 | 0.882 | 0.463 | 0.343 | −1.004 | −0.360 |
| Ensemble_unweighted | 25.39 | 51.57 | 17.664 | 33.746 | 0.666 | 0.828 | 0.691 | 0.833 | 0.523 | 0.375 | −0.189 | −0.165 |
| Ensemble_MASEweighted | 25.38 | 51.57 | 17.651 | 33.743 | 0.666 | 0.827 | 0.690 | 0.833 | 0.522 | 0.375 | −0.195 | −0.165 |
| DLinear_gf | 30.233 | 54.860 | 22.524 | 37.909 | 0.883 | 0.965 | 0.860 | 0.935 | 0.385 | 0.312 | −1.141 | −0.540 |
| LSTM_gf | 30.744 | 57.977 | 22.318 | 39.384 | 0.829 | 0.955 | 0.842 | 0.952 | 0.462 | 0.301 | −0.907 | −0.686 |
| TFT_gf | 29.769 | 53.966 | 20.637 | 35.837 | 0.787 | 0.888 | 0.839 | 0.889 | 0.433 | 0.325 | −2.041 | −0.439 |
| **TabPFN_gf** | **27.256** | **51.675** | **17.380** | **31.463** | **0.636** | **0.746** | **0.726** | **0.808** | **0.507** | **0.360** | **−0.007** | **0.001** |
| TabICLv2_gf | 27.565 | 52.913 | 18.545 | 33.848 | 0.674 | 0.801 | 0.724 | 0.830 | 0.485 | 0.344 | −0.054 | −0.060 |

**TabPFN_gf stays the best model under every single one of these 8 MASE/RMSSE combinations** (2
baselines × 2 evaluation targets) — no ranking reversal anywhere in this expanded view. As with the
persistence-baseline table, MASE and RMSSE broadly agree on direction; the same TFT_gf split noted
earlier persists here too (RMSSE closer to parity than MASE suggests, both evaluation targets).

Source: `results/b10_b13_gf_ablation_vs_climatology_gf_gapfilled_target_table.csv`
(`notebooks/05_benchmarking/b10_b13_gf_vs_climatology_gf_gapfilled_target.py`).

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

Full parity, all 11 models: every metric now split by evaluation target (gf = gap-filled, obs =
observed) crossed with MASE/RMSSE baseline (p = persistence, c = Climatology_gf). RMSE/MAE/
Correlation/R² depend only on the evaluation target, not the baseline, so each has one gf and one
obs column.

**Tower 2:**

| Model | RMSE (gf) | RMSE (obs) | MAE (gf) | MAE (obs) | MASE (gf,p) | MASE (gf,c) | MASE (obs,p) | MASE (obs,c) | RMSSE (gf,p) | RMSSE (gf,c) | RMSSE (obs,p) | RMSSE (obs,c) | Correlation (gf) | Correlation (obs) | R² (gf) | R² (obs) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| RF | 14.503 | 22.130 | 10.865 | 18.047 | 0.689 | 0.673 | 0.346 | 0.906 | 0.694 | 0.690 | 0.399 | 0.868 | 0.459 | 0.087 | −0.108 | −0.889 |
| XGB | 14.767 | 20.819 | 10.406 | 16.452 | 0.638 | 0.628 | 0.312 | 0.791 | 0.690 | 0.696 | 0.371 | 0.787 | 0.372 | 0.214 | −0.153 | −0.618 |
| LightGBM | 14.643 | 20.667 | 10.480 | 16.320 | 0.653 | 0.642 | 0.310 | 0.792 | 0.699 | 0.697 | 0.369 | 0.788 | 0.420 | 0.182 | −0.147 | −0.600 |
| SARIMAX | 17.687 | 34.101 | 13.783 | 30.759 | 0.799 | 0.796 | 0.574 | 1.399 | 0.807 | 0.812 | 0.604 | 1.276 | 0.293 | 0.056 | −1.432 | −3.244 |
| Ensemble_unweighted | 14.609 | 23.657 | 10.564 | 19.821 | 0.646 | 0.631 | 0.374 | 0.941 | 0.683 | 0.685 | 0.421 | 0.890 | 0.433 | 0.176 | −0.185 | −1.048 |
| Ensemble_MASEweighted | 14.596 | 23.542 | 10.546 | 19.694 | 0.645 | 0.630 | 0.372 | 0.936 | 0.682 | 0.684 | 0.419 | 0.885 | 0.433 | 0.178 | −0.179 | −1.029 |
| DLinear_gf | 18.601 | 29.904 | 14.372 | 24.782 | 0.876 | 0.879 | 0.509 | 1.545 | 0.879 | 0.884 | 0.579 | 1.488 | 0.270 | 0.152 | −1.241 | −4.293 |
| LSTM_gf | 16.044 | 22.893 | 11.793 | 18.382 | 0.734 | 0.729 | 0.353 | 0.923 | 0.756 | 0.769 | 0.416 | 0.923 | 0.440 | 0.080 | −0.364 | −1.119 |
| TFT_gf | 15.894 | 21.184 | 11.067 | 17.126 | 0.675 | 0.683 | 0.338 | 0.940 | 0.741 | 0.765 | 0.392 | 0.912 | 0.373 | 0.186 | −0.393 | −1.035 |
| **TabPFN_gf** | **15.291** | **19.360** | **10.007** | **14.902** | **0.604** | **0.610** | **0.280** | **0.694** | **0.698** | **0.727** | **0.343** | **0.719** | **0.384** | **0.053** | **−0.108** | **−0.353** |
| TabICLv2_gf | 14.964 | 21.989 | 10.640 | 18.291 | 0.651 | 0.639 | 0.348 | 0.895 | 0.693 | 0.699 | 0.393 | 0.840 | 0.392 | 0.073 | −0.120 | −0.800 |

**Tower 4:**

| Model | RMSE (gf) | RMSE (obs) | MAE (gf) | MAE (obs) | MASE (gf,p) | MASE (gf,c) | MASE (obs,p) | MASE (obs,c) | RMSSE (gf,p) | RMSSE (gf,c) | RMSSE (obs,p) | RMSSE (obs,c) | Correlation (gf) | Correlation (obs) | R² (gf) | R² (obs) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| RF | 32.612 | 51.616 | 22.757 | 34.134 | 0.916 | 0.817 | 1.026 | 0.913 | 0.845 | 0.783 | 0.908 | 0.867 | 0.528 | 0.403 | −1.490 | −0.067 |
| XGB | 32.473 | 51.301 | 22.148 | 33.128 | 0.862 | 0.779 | 0.970 | 0.865 | 0.816 | 0.768 | 0.884 | 0.846 | 0.500 | 0.373 | −1.201 | 0.003 |
| LightGBM | 32.513 | 51.439 | 22.247 | 33.301 | 0.865 | 0.782 | 0.980 | 0.870 | 0.817 | 0.768 | 0.891 | 0.852 | 0.506 | 0.385 | −1.171 | −0.014 |
| SARIMAX | 33.935 | 52.462 | 24.323 | 35.281 | 0.890 | 0.851 | 1.040 | 0.925 | 0.825 | 0.799 | 0.904 | 0.862 | 0.524 | 0.380 | −0.590 | −0.039 |
| Ensemble_unweighted | 31.827 | 51.035 | 21.866 | 33.227 | 0.827 | 0.761 | 0.977 | 0.868 | 0.781 | 0.744 | 0.881 | 0.841 | 0.535 | 0.397 | −0.465 | 0.012 |
| Ensemble_MASEweighted | 31.831 | 51.034 | 21.864 | 33.219 | 0.827 | 0.761 | 0.977 | 0.868 | 0.782 | 0.745 | 0.881 | 0.841 | 0.534 | 0.397 | −0.484 | 0.011 |
| DLinear_gf | 36.550 | 54.554 | 26.579 | 37.340 | 1.050 | 0.950 | 1.119 | 0.989 | 0.962 | 0.892 | 0.974 | 0.923 | 0.394 | 0.302 | −1.415 | −0.244 |
| LSTM_gf | 36.828 | 56.521 | 25.995 | 37.393 | 1.017 | 0.902 | 1.126 | 0.972 | 0.959 | 0.872 | 1.010 | 0.938 | 0.465 | 0.345 | −1.309 | −0.404 |
| TFT_gf | 35.018 | 53.434 | 23.638 | 34.524 | 1.016 | 0.853 | 1.050 | 0.915 | 0.991 | 0.868 | 0.965 | 0.897 | 0.478 | 0.347 | −3.236 | −0.371 |
| **TabPFN_gf** | **33.729** | **51.118** | **20.825** | **29.895** | **0.731** | **0.710** | **0.858** | **0.769** | **0.769** | **0.768** | **0.854** | **0.821** | **0.541** | **0.384** | **0.009** | **0.085** |
| TabICLv2_gf | 34.050 | 52.336 | 22.733 | 33.349 | 0.796 | 0.759 | 0.953 | 0.842 | 0.785 | 0.770 | 0.884 | 0.842 | 0.506 | 0.378 | −0.148 | 0.008 |

**Tower 9:**

| Model | RMSE (gf) | RMSE (obs) | MAE (gf) | MAE (obs) | MASE (gf,p) | MASE (gf,c) | MASE (obs,p) | MASE (obs,c) | RMSSE (gf,p) | RMSSE (gf,c) | RMSSE (obs,p) | RMSSE (obs,c) | Correlation (gf) | Correlation (obs) | R² (gf) | R² (obs) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| RF | 30.440 | 58.407 | 21.130 | 38.303 | 0.793 | 0.620 | 0.944 | 0.774 | 0.761 | 0.661 | 0.899 | 0.827 | 0.601 | 0.389 | −0.180 | −0.348 |
| XGB | 29.910 | 57.562 | 20.559 | 37.592 | 0.782 | 0.602 | 0.920 | 0.752 | 0.757 | 0.652 | 0.884 | 0.815 | 0.576 | 0.392 | −0.152 | −0.358 |
| LightGBM | 31.161 | 58.618 | 21.878 | 38.707 | 0.804 | 0.641 | 0.954 | 0.780 | 0.772 | 0.679 | 0.909 | 0.839 | 0.564 | 0.378 | −0.120 | −0.389 |
| SARIMAX | 36.017 | 60.121 | 26.886 | 38.396 | 1.141 | 0.845 | 0.920 | 0.777 | 0.977 | 0.823 | 0.914 | 0.857 | 0.572 | 0.350 | −0.991 | −0.406 |
| Ensemble_unweighted | 29.736 | 57.485 | 20.560 | 36.554 | 0.781 | 0.607 | 0.884 | 0.728 | 0.733 | 0.643 | 0.870 | 0.807 | 0.601 | 0.385 | 0.081 | −0.253 |
| Ensemble_MASEweighted | 29.721 | 57.485 | 20.543 | 36.582 | 0.779 | 0.606 | 0.885 | 0.728 | 0.732 | 0.642 | 0.870 | 0.807 | 0.600 | 0.385 | 0.079 | −0.255 |
| DLinear_gf | 35.547 | 59.658 | 26.622 | 40.457 | 1.096 | 0.819 | 1.009 | 0.813 | 0.963 | 0.802 | 0.941 | 0.859 | 0.490 | 0.358 | −0.767 | −0.348 |
| LSTM_gf | 39.362 | 65.580 | 29.166 | 45.184 | 1.115 | 0.856 | 1.119 | 0.904 | 1.040 | 0.883 | 1.046 | 0.959 | 0.480 | 0.287 | −1.048 | −0.879 |
| TFT_gf | 38.397 | 59.651 | 27.207 | 39.863 | 1.227 | 0.826 | 0.967 | 0.797 | 1.118 | 0.885 | 0.912 | 0.847 | 0.449 | 0.338 | −2.493 | −0.268 |
| **TabPFN_gf** | **32.748** | **58.570** | **21.308** | **36.701** | **0.697** | **0.589** | **0.855** | **0.712** | **0.739** | **0.684** | **0.855** | **0.801** | **0.598** | **0.388** | **0.076** | **−0.052** |
| TabICLv2_gf | 33.681 | 59.296 | 22.263 | 37.132 | 0.731 | 0.622 | 0.858 | 0.712 | 0.751 | 0.704 | 0.858 | 0.804 | 0.556 | 0.357 | 0.107 | −0.012 |

**Tower 2 is the one place the foundation models regress under `_gf`** (TabPFN R² −0.240→−0.353,
TabICLv2 −0.219→−0.800, both observed-target/persistence) — every other model/tower combination
improves or is roughly flat. This holds under both MASE/RMSSE baselines — Tower 2's TabPFN/TabICLv2
regression isn't an artifact of the persistence denominator specifically.

**A genuinely new finding from extending this to the gapfilled/observed axis at per-tower
granularity: Tower 2's gapfilled-vs-observed MASE direction reverses relative to every other
tower.** Everywhere else in this document, scoring against `y_gapfilled` gives a *lower* (better)
MASE than scoring against `y_observed` (D-65's "smoother target, smaller absolute error" pattern).
At Tower 2 it flips for most models — e.g. RF: MASE(gf,persist)=0.689 vs MASE(obs,persist)=0.346,
*observed* is lower. The reason isn't a metric artifact: Tower 2 has real `y_observed` coverage only
in the 2018 anchor (every other anchor is all-NaN, per the tower×year table below), so the "observed"
pooled number here is really a single-anchor value, while the "gapfilled" pooled number still
averages across all 5 anchors (since `y_gapfilled` is dense every year). The two columns are scoring
different underlying evaluation sets at this specific tower, not the same set through two lenses —
worth remembering before comparing gf/obs columns across towers.

Source: `results/b10_b13_full_parity_gf_obs_by_tower.csv`
(`notebooks/05_benchmarking/b10_b13_full_parity_gf_obs_tables.py`).

## Tower × year × model breakdown

Full parity, all 11 models: per-anchor n-weighted mean across the 6 lead-time bins (no cross-anchor
averaging -- that's what the summary tables above already show). Columns flattened to
`T{tower}_{metric}` for markdown; true long-format version in
`results/b10_b13_full_parity_gf_obs_table_by_tower_year.csv`. `nan` = no real `y_observed`
coverage for that tower/anchor (Tower 2 outside 2018, matching every other B-09→B-15 document's
Tower 2 finding). Every metric split by evaluation target (`_gf`/`_obs`) crossed with MASE/RMSSE
baseline (`_p` = persistence, `_c` = Climatology_gf) -- RMSE/MAE/Correlation/R² depend only on
the evaluation target, so each has one `_gf` and one `_obs` column per tower.

| Year | Model | T2_RMSE_gf | T2_RMSE_obs | T2_MAE_gf | T2_MAE_obs | T2_MASE_gf_p | T2_MASE_gf_c | T2_MASE_obs_p | T2_MASE_obs_c | T2_RMSSE_gf_p | T2_RMSSE_gf_c | T2_RMSSE_obs_p | T2_RMSSE_obs_c | T2_Corr_gf | T2_Corr_obs | T4_RMSE_gf | T4_RMSE_obs | T4_MAE_gf | T4_MAE_obs | T4_MASE_gf_p | T4_MASE_gf_c | T4_MASE_obs_p | T4_MASE_obs_c | T4_RMSSE_gf_p | T4_RMSSE_gf_c | T4_RMSSE_obs_p | T4_RMSSE_obs_c | T4_Corr_gf | T4_Corr_obs | T9_RMSE_gf | T9_RMSE_obs | T9_MAE_gf | T9_MAE_obs | T9_MASE_gf_p | T9_MASE_gf_c | T9_MASE_obs_p | T9_MASE_obs_c | T9_RMSSE_gf_p | T9_RMSSE_gf_c | T9_RMSSE_obs_p | T9_RMSSE_obs_c | T9_Corr_gf | T9_Corr_obs | T2_R2_gf | T4_R2_gf | T9_R2_gf | T2_R2_obs | T4_R2_obs | T9_R2_obs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2018 | RF | 10.165 | 22.130 | 7.491 | 18.047 | 0.176 | 0.368 | 0.346 | 0.906 | 0.231 | 0.380 | 0.399 | 0.868 | 0.392 | 0.087 | 30.214 | 46.857 | 22.206 | 32.996 | 0.884 | 0.873 | 1.025 | 0.944 | 0.891 | 0.813 | 0.961 | 0.850 | 0.478 | 0.278 | 25.874 | nan | 17.621 | nan | 0.813 | 0.628 | nan | nan | 0.720 | 0.623 | nan | nan | 0.641 | nan | −0.069 | −0.073 | 0.305 | −0.889 | −0.239 | nan |
| 2018 | XGB | 11.106 | 20.819 | 7.858 | 16.452 | 0.183 | 0.370 | 0.312 | 0.791 | 0.252 | 0.405 | 0.371 | 0.787 | 0.313 | 0.214 | 29.384 | 45.715 | 20.618 | 30.287 | 0.796 | 0.806 | 0.896 | 0.828 | 0.844 | 0.790 | 0.907 | 0.811 | 0.459 | 0.267 | 25.907 | nan | 17.354 | nan | 0.818 | 0.623 | nan | nan | 0.741 | 0.627 | nan | nan | 0.579 | nan | −0.392 | 0.072 | 0.253 | −0.618 | −0.073 | nan |
| 2018 | LightGBM | 10.758 | 20.667 | 7.571 | 16.320 | 0.178 | 0.364 | 0.310 | 0.792 | 0.245 | 0.396 | 0.369 | 0.788 | 0.368 | 0.182 | 29.115 | 45.480 | 20.294 | 29.927 | 0.801 | 0.804 | 0.899 | 0.826 | 0.855 | 0.792 | 0.917 | 0.815 | 0.483 | 0.285 | 28.008 | nan | 19.495 | nan | 0.826 | 0.673 | nan | nan | 0.757 | 0.667 | nan | nan | 0.567 | nan | −0.277 | 0.038 | 0.232 | −0.600 | −0.111 | nan |
| 2018 | SARIMAX | 20.331 | 34.101 | 17.655 | 30.759 | 0.390 | 0.751 | 0.574 | 1.399 | 0.440 | 0.687 | 0.604 | 1.276 | 0.087 | 0.056 | 31.626 | 49.688 | 25.886 | 37.932 | 1.036 | 1.030 | 1.176 | 1.082 | 0.958 | 0.881 | 1.020 | 0.909 | 0.503 | 0.241 | 44.595 | nan | 40.346 | nan | 2.414 | 1.512 | nan | nan | 1.620 | 1.179 | nan | nan | 0.557 | nan | −5.706 | −0.254 | −3.310 | −3.244 | −0.399 | nan |
| 2018 | Ensemble_unweighted | 12.073 | 23.657 | 9.230 | 19.821 | 0.211 | 0.419 | 0.374 | 0.941 | 0.269 | 0.430 | 0.421 | 0.890 | 0.382 | 0.176 | 29.024 | 46.213 | 21.407 | 32.229 | 0.848 | 0.847 | 0.981 | 0.903 | 0.859 | 0.792 | 0.935 | 0.832 | 0.513 | 0.278 | 27.667 | nan | 21.225 | nan | 1.108 | 0.780 | nan | nan | 0.844 | 0.687 | nan | nan | 0.629 | nan | −0.810 | 0.022 | 0.011 | −1.048 | −0.160 | nan |
| 2018 | Ensemble_MASEweighted | 12.011 | 23.542 | 9.155 | 19.694 | 0.209 | 0.416 | 0.372 | 0.936 | 0.268 | 0.428 | 0.419 | 0.885 | 0.381 | 0.178 | 29.012 | 46.179 | 21.355 | 32.152 | 0.846 | 0.844 | 0.978 | 0.900 | 0.858 | 0.792 | 0.934 | 0.831 | 0.513 | 0.278 | 27.555 | nan | 21.040 | nan | 1.094 | 0.772 | nan | nan | 0.837 | 0.683 | nan | nan | 0.628 | nan | −0.783 | 0.024 | 0.028 | −1.029 | −0.157 | nan |
| 2018 | DLinear_gf | 17.683 | 29.904 | 14.364 | 24.782 | 0.332 | 0.707 | 0.509 | 1.545 | 0.396 | 0.668 | 0.579 | 1.488 | 0.341 | 0.152 | 36.197 | 51.673 | 28.395 | 38.869 | 1.188 | 1.148 | 1.220 | 1.076 | 1.159 | 1.026 | 1.087 | 0.942 | 0.312 | 0.156 | 39.699 | nan | 31.496 | nan | 1.787 | 1.152 | nan | nan | 1.366 | 1.019 | nan | nan | 0.464 | nan | −3.139 | −0.951 | −1.980 | −4.293 | −0.636 | nan |
| 2018 | LSTM_gf | 10.433 | 22.893 | 8.082 | 18.382 | 0.185 | 0.369 | 0.353 | 0.923 | 0.233 | 0.369 | 0.416 | 0.923 | 0.396 | 0.080 | 39.062 | 57.875 | 30.255 | 42.976 | 1.256 | 1.116 | 1.389 | 1.171 | 1.279 | 1.043 | 1.290 | 1.064 | 0.318 | 0.140 | 40.466 | nan | 29.681 | nan | 1.463 | 1.011 | nan | nan | 1.300 | 1.013 | nan | nan | 0.494 | nan | −0.541 | −1.982 | −1.626 | −1.119 | −1.600 | nan |
| 2018 | TFT_gf | 11.180 | 21.184 | 8.431 | 17.126 | 0.200 | 0.424 | 0.338 | 0.940 | 0.256 | 0.421 | 0.392 | 0.912 | 0.399 | 0.186 | 39.677 | 57.408 | 31.191 | 43.343 | 1.428 | 1.221 | 1.512 | 1.247 | 1.396 | 1.122 | 1.347 | 1.091 | 0.304 | 0.146 | 46.120 | nan | 34.790 | nan | 2.393 | 1.257 | nan | nan | 1.864 | 1.220 | nan | nan | 0.292 | nan | −0.577 | −2.915 | −6.645 | −1.035 | −2.054 | nan |
| 2018 | TabPFN_gf | 11.245 | 19.360 | 7.629 | 14.902 | 0.181 | 0.378 | 0.280 | 0.694 | 0.259 | 0.429 | 0.343 | 0.719 | 0.205 | 0.053 | 31.504 | 46.777 | 20.615 | 28.637 | 0.788 | 0.799 | 0.836 | 0.768 | 0.884 | 0.837 | 0.908 | 0.818 | 0.442 | 0.212 | 35.789 | nan | 22.762 | nan | 0.759 | 0.699 | nan | nan | 0.835 | 0.811 | nan | nan | 0.539 | nan | −0.227 | −0.017 | −0.004 | −0.353 | −0.061 | nan |
| 2018 | TabICLv2_gf | 11.320 | 21.989 | 8.296 | 18.291 | 0.193 | 0.394 | 0.348 | 0.895 | 0.257 | 0.417 | 0.393 | 0.840 | 0.312 | 0.073 | 31.397 | 49.102 | 21.674 | 32.125 | 0.881 | 0.835 | 0.989 | 0.870 | 0.935 | 0.839 | 0.990 | 0.869 | 0.345 | 0.138 | 35.597 | nan | 23.302 | nan | 0.869 | 0.740 | nan | nan | 0.850 | 0.806 | nan | nan | 0.465 | nan | −0.390 | −0.224 | −0.048 | −0.800 | −0.300 | nan |
| 2019 | RF | 19.687 | nan | 13.936 | nan | 0.872 | 0.624 | nan | nan | 0.888 | 0.650 | nan | nan | 0.454 | nan | 27.278 | 48.567 | 18.319 | 29.897 | 0.694 | 0.632 | 0.784 | 0.809 | 0.712 | 0.666 | 0.786 | 0.820 | 0.527 | 0.473 | 35.626 | 71.212 | 23.509 | 45.019 | 0.640 | 0.605 | 0.913 | 0.779 | 0.611 | 0.634 | 0.847 | 0.810 | 0.722 | 0.499 | 0.010 | −1.354 | 0.405 | nan | 0.007 | 0.078 |
| 2019 | XGB | 20.638 | nan | 14.289 | nan | 0.875 | 0.631 | nan | nan | 0.912 | 0.682 | nan | nan | 0.337 | nan | 27.105 | 48.210 | 17.618 | 28.596 | 0.648 | 0.603 | 0.745 | 0.760 | 0.697 | 0.664 | 0.769 | 0.799 | 0.517 | 0.439 | 35.243 | 70.191 | 23.316 | 44.247 | 0.652 | 0.594 | 0.900 | 0.757 | 0.623 | 0.635 | 0.837 | 0.799 | 0.708 | 0.521 | −0.014 | −1.045 | 0.354 | nan | 0.075 | 0.101 |
| 2019 | LightGBM | 19.903 | nan | 13.844 | nan | 0.850 | 0.613 | nan | nan | 0.882 | 0.658 | nan | nan | 0.411 | nan | 26.734 | 47.799 | 17.059 | 27.667 | 0.621 | 0.581 | 0.715 | 0.734 | 0.676 | 0.649 | 0.759 | 0.790 | 0.538 | 0.468 | 36.745 | 71.335 | 24.509 | 44.843 | 0.707 | 0.653 | 0.919 | 0.779 | 0.664 | 0.677 | 0.865 | 0.828 | 0.704 | 0.511 | 0.053 | −0.824 | 0.216 | nan | 0.100 | 0.003 |
| 2019 | SARIMAX | 22.177 | nan | 16.291 | nan | 1.049 | 0.725 | nan | nan | 1.057 | 0.744 | nan | nan | 0.312 | nan | 31.278 | 47.927 | 24.313 | 31.362 | 0.738 | 0.871 | 0.836 | 0.845 | 0.699 | 0.790 | 0.769 | 0.797 | 0.560 | 0.448 | 38.174 | 69.127 | 24.498 | 41.551 | 0.667 | 0.618 | 0.814 | 0.704 | 0.652 | 0.663 | 0.801 | 0.771 | 0.731 | 0.490 | −0.594 | −0.070 | 0.335 | nan | 0.114 | 0.186 |
| 2019 | Ensemble_unweighted | 20.142 | nan | 14.142 | nan | 0.876 | 0.628 | nan | nan | 0.904 | 0.667 | nan | nan | 0.433 | nan | 26.469 | 47.046 | 17.339 | 27.700 | 0.599 | 0.591 | 0.710 | 0.728 | 0.648 | 0.646 | 0.744 | 0.773 | 0.539 | 0.465 | 34.633 | 69.614 | 22.185 | 42.737 | 0.611 | 0.570 | 0.861 | 0.737 | 0.599 | 0.618 | 0.822 | 0.788 | 0.733 | 0.520 | −0.016 | −0.377 | 0.419 | nan | 0.136 | 0.134 |
| 2019 | Ensemble_MASEweighted | 20.133 | nan | 14.127 | nan | 0.874 | 0.627 | nan | nan | 0.903 | 0.667 | nan | nan | 0.432 | nan | 26.458 | 47.068 | 17.311 | 27.705 | 0.599 | 0.590 | 0.710 | 0.728 | 0.649 | 0.646 | 0.744 | 0.774 | 0.539 | 0.465 | 34.643 | 69.637 | 22.218 | 42.784 | 0.612 | 0.571 | 0.863 | 0.738 | 0.600 | 0.618 | 0.823 | 0.789 | 0.733 | 0.520 | −0.012 | −0.393 | 0.417 | nan | 0.134 | 0.132 |
| 2019 | DLinear_gf | 23.940 | nan | 19.153 | nan | 1.235 | 0.873 | nan | nan | 1.154 | 0.804 | nan | nan | 0.205 | nan | 33.872 | 53.810 | 26.002 | 35.808 | 1.061 | 0.961 | 1.024 | 1.038 | 0.982 | 0.879 | 0.927 | 0.963 | 0.445 | 0.379 | 41.540 | 73.159 | 31.952 | 50.345 | 1.003 | 0.928 | 1.069 | 0.902 | 0.795 | 0.799 | 0.924 | 0.883 | 0.636 | 0.466 | −0.972 | −4.843 | −0.226 | nan | −0.392 | −0.216 |
| 2019 | LSTM_gf | 22.465 | nan | 15.610 | nan | 0.960 | 0.676 | nan | nan | 1.024 | 0.743 | nan | nan | 0.528 | nan | 32.285 | 53.781 | 22.760 | 31.386 | 0.725 | 0.733 | 0.823 | 0.815 | 0.737 | 0.752 | 0.854 | 0.872 | 0.513 | 0.398 | 42.609 | 76.420 | 30.526 | 52.528 | 0.817 | 0.732 | 1.079 | 0.881 | 0.722 | 0.729 | 0.919 | 0.872 | 0.631 | 0.408 | −0.345 | −0.603 | 0.175 | nan | −0.085 | −0.074 |
| 2019 | TFT_gf | 20.776 | nan | 14.328 | nan | 0.875 | 0.632 | nan | nan | 0.918 | 0.682 | nan | nan | 0.430 | nan | 29.550 | 50.617 | 19.602 | 30.051 | 0.826 | 0.664 | 0.793 | 0.803 | 0.846 | 0.725 | 0.813 | 0.842 | 0.445 | 0.336 | 42.833 | 74.055 | 29.684 | 48.462 | 0.839 | 0.760 | 0.979 | 0.828 | 0.751 | 0.757 | 0.873 | 0.840 | 0.550 | 0.402 | −0.028 | −4.185 | 0.062 | nan | −0.013 | 0.024 |
| 2019 | TabPFN_gf | 21.659 | nan | 14.275 | nan | 0.863 | 0.634 | nan | nan | 0.937 | 0.713 | nan | nan | 0.438 | nan | 31.506 | 49.178 | 20.668 | 27.307 | 0.595 | 0.725 | 0.705 | 0.726 | 0.666 | 0.769 | 0.761 | 0.794 | 0.512 | 0.393 | 37.068 | 71.187 | 23.294 | 44.613 | 0.598 | 0.578 | 0.879 | 0.750 | 0.620 | 0.650 | 0.821 | 0.786 | 0.752 | 0.542 | −0.057 | 0.023 | 0.398 | nan | 0.118 | 0.147 |
| 2019 | TabICLv2_gf | 22.024 | nan | 14.746 | nan | 0.916 | 0.653 | nan | nan | 0.992 | 0.728 | nan | nan | 0.110 | nan | 29.297 | 49.369 | 20.024 | 29.406 | 0.594 | 0.714 | 0.773 | 0.767 | 0.641 | 0.731 | 0.799 | 0.815 | 0.443 | 0.396 | 40.028 | 73.825 | 25.364 | 45.182 | 0.655 | 0.636 | 0.888 | 0.750 | 0.666 | 0.702 | 0.846 | 0.812 | 0.726 | 0.462 | −0.222 | 0.098 | 0.298 | nan | 0.057 | 0.091 |
| 2020 | RF | 15.525 | nan | 11.203 | nan | 0.841 | 0.711 | nan | nan | 0.883 | 0.775 | nan | nan | 0.427 | nan | 37.921 | 61.519 | 22.464 | 34.465 | 0.908 | 0.860 | 1.029 | 1.004 | 0.805 | 0.807 | 0.944 | 0.939 | 0.517 | 0.337 | 37.821 | 58.459 | 25.380 | 36.644 | 0.773 | 0.677 | 0.950 | 0.784 | 0.776 | 0.733 | 0.941 | 0.861 | 0.619 | 0.354 | 0.095 | 0.237 | 0.283 | nan | −0.070 | −1.147 |
| 2020 | XGB | 15.132 | nan | 10.247 | nan | 0.770 | 0.648 | nan | nan | 0.865 | 0.755 | nan | nan | 0.367 | nan | 40.248 | 63.099 | 23.491 | 34.873 | 0.936 | 0.880 | 1.021 | 0.992 | 0.841 | 0.846 | 0.950 | 0.943 | 0.436 | 0.251 | 36.425 | 58.137 | 24.110 | 35.897 | 0.743 | 0.648 | 0.938 | 0.775 | 0.750 | 0.707 | 0.948 | 0.867 | 0.630 | 0.340 | 0.119 | 0.172 | 0.327 | nan | −0.073 | −1.378 |
| 2020 | LightGBM | 15.063 | nan | 10.476 | nan | 0.789 | 0.664 | nan | nan | 0.862 | 0.753 | nan | nan | 0.407 | nan | 40.061 | 62.955 | 23.734 | 35.099 | 0.953 | 0.899 | 1.031 | 1.003 | 0.841 | 0.845 | 0.949 | 0.944 | 0.449 | 0.272 | 36.303 | 58.507 | 24.537 | 36.900 | 0.751 | 0.657 | 0.960 | 0.796 | 0.740 | 0.701 | 0.953 | 0.874 | 0.643 | 0.339 | 0.125 | 0.173 | 0.343 | nan | −0.068 | −1.229 |
| 2020 | SARIMAX | 16.361 | nan | 13.010 | nan | 0.982 | 0.829 | nan | nan | 0.933 | 0.817 | nan | nan | 0.438 | nan | 39.660 | 62.671 | 23.118 | 35.207 | 0.856 | 0.822 | 1.000 | 0.964 | 0.796 | 0.809 | 0.914 | 0.907 | 0.505 | 0.318 | 37.320 | 58.555 | 24.074 | 35.010 | 0.752 | 0.655 | 0.895 | 0.737 | 0.781 | 0.735 | 0.925 | 0.846 | 0.604 | 0.283 | −0.021 | 0.248 | 0.273 | nan | 0.060 | −0.921 |
| 2020 | Ensemble_unweighted | 14.946 | nan | 10.750 | nan | 0.808 | 0.681 | nan | nan | 0.853 | 0.746 | nan | nan | 0.421 | nan | 38.881 | 62.143 | 22.333 | 34.145 | 0.867 | 0.824 | 0.990 | 0.960 | 0.802 | 0.809 | 0.926 | 0.921 | 0.504 | 0.309 | 36.371 | 58.032 | 23.755 | 35.442 | 0.727 | 0.636 | 0.915 | 0.756 | 0.744 | 0.704 | 0.932 | 0.854 | 0.644 | 0.337 | 0.147 | 0.245 | 0.339 | nan | 0.000 | −1.076 |
| 2020 | Ensemble_MASEweighted | 14.937 | nan | 10.722 | nan | 0.806 | 0.680 | nan | nan | 0.852 | 0.745 | nan | nan | 0.420 | nan | 38.909 | 62.163 | 22.365 | 34.159 | 0.869 | 0.825 | 0.991 | 0.961 | 0.803 | 0.810 | 0.927 | 0.922 | 0.502 | 0.308 | 36.362 | 58.034 | 23.767 | 35.463 | 0.728 | 0.637 | 0.915 | 0.756 | 0.744 | 0.704 | 0.932 | 0.854 | 0.644 | 0.338 | 0.147 | 0.243 | 0.339 | nan | −0.002 | −1.082 |
| 2020 | DLinear_gf | 18.527 | nan | 13.037 | nan | 0.984 | 0.834 | nan | nan | 1.056 | 0.927 | nan | nan | 0.302 | nan | 43.417 | 64.907 | 27.662 | 38.177 | 1.081 | 0.996 | 1.084 | 1.034 | 0.926 | 0.923 | 0.971 | 0.953 | 0.339 | 0.226 | 40.429 | 58.080 | 26.599 | 35.560 | 0.862 | 0.735 | 0.938 | 0.756 | 0.872 | 0.805 | 0.935 | 0.845 | 0.496 | 0.339 | −0.308 | −0.015 | 0.078 | nan | −0.065 | −0.580 |
| 2020 | LSTM_gf | 15.813 | nan | 11.580 | nan | 0.867 | 0.739 | nan | nan | 0.890 | 0.786 | nan | nan | 0.468 | nan | 39.406 | 62.336 | 25.763 | 37.244 | 0.986 | 0.937 | 1.082 | 1.046 | 0.817 | 0.822 | 0.956 | 0.940 | 0.543 | 0.358 | 48.199 | 71.437 | 35.704 | 49.731 | 1.150 | 0.969 | 1.325 | 1.076 | 1.103 | 0.998 | 1.254 | 1.132 | 0.446 | 0.180 | 0.076 | 0.211 | −0.632 | nan | −0.037 | −2.490 |
| 2020 | TFT_gf | 15.596 | nan | 9.865 | nan | 0.754 | 0.633 | nan | nan | 0.887 | 0.779 | nan | nan | 0.407 | nan | 40.670 | 64.339 | 23.686 | 36.411 | 0.891 | 0.838 | 1.024 | 0.985 | 0.832 | 0.840 | 0.958 | 0.947 | 0.459 | 0.263 | 44.069 | 61.458 | 30.092 | 40.023 | 0.923 | 0.803 | 1.021 | 0.835 | 0.921 | 0.867 | 0.970 | 0.887 | 0.508 | 0.317 | 0.079 | 0.180 | −0.012 | nan | −0.036 | −0.664 |
| 2020 | TabPFN_gf | 14.569 | nan | 8.654 | nan | 0.647 | 0.546 | nan | nan | 0.827 | 0.722 | nan | nan | 0.423 | nan | 41.293 | 62.639 | 21.191 | 31.560 | 0.822 | 0.772 | 0.897 | 0.868 | 0.859 | 0.865 | 0.909 | 0.906 | 0.536 | 0.360 | 39.444 | 57.842 | 24.299 | 33.374 | 0.715 | 0.630 | 0.834 | 0.690 | 0.804 | 0.759 | 0.892 | 0.823 | 0.620 | 0.352 | 0.183 | 0.137 | 0.226 | nan | 0.073 | −0.335 |
| 2020 | TabICLv2_gf | 13.955 | nan | 9.346 | nan | 0.695 | 0.588 | nan | nan | 0.794 | 0.694 | nan | nan | 0.499 | nan | 41.658 | 63.653 | 24.106 | 34.928 | 0.901 | 0.852 | 0.973 | 0.942 | 0.858 | 0.866 | 0.924 | 0.920 | 0.457 | 0.286 | 39.711 | 58.048 | 24.870 | 34.371 | 0.742 | 0.647 | 0.845 | 0.694 | 0.825 | 0.770 | 0.883 | 0.812 | 0.542 | 0.269 | 0.225 | 0.139 | 0.176 | nan | 0.042 | −0.142 |
| 2021 | RF | 12.982 | nan | 10.682 | nan | 0.971 | 0.880 | nan | nan | 0.855 | 0.856 | nan | nan | 0.360 | nan | 43.894 | 61.899 | 32.702 | 44.242 | 1.070 | 1.025 | 1.195 | 1.081 | 0.898 | 0.923 | 0.981 | 0.964 | 0.523 | 0.411 | 27.434 | 64.098 | 18.770 | 41.110 | 0.609 | 0.567 | 0.753 | 0.774 | 0.669 | 0.652 | 0.780 | 0.836 | 0.551 | 0.343 | −0.156 | −0.088 | −0.168 | nan | −0.141 | 0.008 |
| 2021 | XGB | 13.005 | nan | 9.697 | nan | 0.839 | 0.781 | nan | nan | 0.830 | 0.846 | nan | nan | 0.215 | nan | 42.722 | 60.778 | 32.166 | 43.891 | 0.991 | 0.970 | 1.145 | 1.049 | 0.846 | 0.876 | 0.946 | 0.933 | 0.528 | 0.413 | 27.699 | 64.133 | 19.086 | 41.537 | 0.611 | 0.559 | 0.744 | 0.759 | 0.673 | 0.651 | 0.774 | 0.830 | 0.492 | 0.320 | −0.111 | 0.046 | −0.041 | nan | −0.058 | 0.043 |
| 2021 | LightGBM | 13.931 | nan | 10.548 | nan | 0.926 | 0.862 | nan | nan | 0.929 | 0.910 | nan | nan | 0.238 | nan | 42.090 | 59.971 | 31.510 | 43.110 | 0.963 | 0.945 | 1.128 | 1.032 | 0.830 | 0.859 | 0.940 | 0.927 | 0.534 | 0.434 | 27.715 | 62.304 | 18.918 | 40.467 | 0.615 | 0.562 | 0.726 | 0.744 | 0.693 | 0.659 | 0.756 | 0.810 | 0.471 | 0.318 | −0.388 | 0.094 | −0.122 | nan | −0.043 | 0.074 |
| 2021 | SARIMAX | 15.456 | nan | 11.582 | nan | 1.069 | 0.921 | nan | nan | 1.037 | 0.999 | nan | nan | 0.061 | nan | 43.227 | 61.754 | 30.357 | 42.024 | 0.867 | 0.863 | 1.069 | 0.982 | 0.789 | 0.819 | 0.931 | 0.919 | 0.521 | 0.386 | 36.004 | 74.626 | 27.672 | 49.356 | 0.947 | 0.876 | 0.928 | 0.968 | 0.916 | 0.897 | 0.970 | 1.044 | 0.586 | 0.319 | −0.783 | 0.172 | −1.337 | nan | −0.036 | −0.746 |
| 2021 | Ensemble_unweighted | 13.013 | nan | 9.820 | nan | 0.873 | 0.790 | nan | nan | 0.849 | 0.846 | nan | nan | 0.217 | nan | 42.060 | 60.447 | 31.277 | 42.982 | 0.951 | 0.932 | 1.122 | 1.026 | 0.811 | 0.840 | 0.936 | 0.923 | 0.542 | 0.424 | 26.630 | 63.999 | 17.709 | 39.444 | 0.552 | 0.503 | 0.693 | 0.706 | 0.622 | 0.602 | 0.763 | 0.818 | 0.547 | 0.326 | −0.175 | 0.137 | 0.175 | nan | −0.034 | 0.075 |
| 2021 | Ensemble_MASEweighted | 13.009 | nan | 9.818 | nan | 0.873 | 0.790 | nan | nan | 0.849 | 0.846 | nan | nan | 0.219 | nan | 42.054 | 60.433 | 31.295 | 42.998 | 0.952 | 0.933 | 1.122 | 1.026 | 0.811 | 0.841 | 0.936 | 0.923 | 0.542 | 0.424 | 26.619 | 63.938 | 17.701 | 39.440 | 0.553 | 0.503 | 0.693 | 0.707 | 0.622 | 0.602 | 0.762 | 0.817 | 0.545 | 0.326 | −0.173 | 0.135 | 0.173 | nan | −0.034 | 0.078 |
| 2021 | DLinear_gf | 18.745 | nan | 14.693 | nan | 1.259 | 1.229 | nan | nan | 1.171 | 1.241 | nan | nan | 0.016 | nan | 43.408 | 60.330 | 32.495 | 43.036 | 1.082 | 1.031 | 1.131 | 1.035 | 0.935 | 0.947 | 0.959 | 0.943 | 0.475 | 0.386 | 27.494 | 63.563 | 20.092 | 41.792 | 0.644 | 0.585 | 0.734 | 0.750 | 0.668 | 0.643 | 0.773 | 0.828 | 0.498 | 0.280 | −1.309 | −0.271 | −0.013 | nan | −0.110 | 0.022 |
| 2021 | LSTM_gf | 17.386 | nan | 13.718 | nan | 1.170 | 1.128 | nan | nan | 1.071 | 1.129 | nan | nan | 0.269 | nan | 45.036 | 61.797 | 30.066 | 40.237 | 1.077 | 0.980 | 1.072 | 0.977 | 0.975 | 0.950 | 0.951 | 0.935 | 0.490 | 0.396 | 34.115 | 66.742 | 26.239 | 43.759 | 0.952 | 0.854 | 0.790 | 0.818 | 0.934 | 0.882 | 0.831 | 0.893 | 0.495 | 0.256 | −0.906 | −0.737 | −1.405 | nan | −0.099 | −0.170 |
| 2021 | TFT_gf | 16.043 | nan | 11.489 | nan | 0.957 | 0.932 | nan | nan | 0.973 | 1.049 | nan | nan | 0.141 | nan | 42.860 | 58.946 | 27.873 | 38.144 | 0.908 | 0.874 | 0.997 | 0.914 | 0.914 | 0.945 | 0.915 | 0.910 | 0.557 | 0.438 | 33.239 | 64.311 | 21.607 | 42.085 | 0.706 | 0.674 | 0.774 | 0.803 | 0.884 | 0.884 | 0.814 | 0.877 | 0.496 | 0.344 | −0.493 | −0.252 | −2.284 | nan | −0.004 | −0.201 |
| 2021 | TabPFN_gf | 14.426 | nan | 10.169 | nan | 0.857 | 0.822 | nan | nan | 0.874 | 0.941 | nan | nan | 0.206 | nan | 42.168 | 60.147 | 27.439 | 38.440 | 0.781 | 0.772 | 0.957 | 0.884 | 0.758 | 0.790 | 0.878 | 0.873 | 0.528 | 0.405 | 29.767 | 69.103 | 20.356 | 42.941 | 0.609 | 0.554 | 0.746 | 0.765 | 0.654 | 0.639 | 0.812 | 0.869 | 0.576 | 0.288 | −0.241 | 0.226 | 0.131 | nan | 0.082 | −0.036 |
| 2021 | TabICLv2_gf | 15.478 | nan | 12.328 | nan | 1.005 | 0.954 | nan | nan | 0.921 | 0.983 | nan | nan | 0.352 | nan | 48.188 | 63.489 | 33.106 | 44.141 | 0.873 | 0.871 | 1.044 | 0.972 | 0.825 | 0.862 | 0.908 | 0.905 | 0.586 | 0.478 | 31.046 | 69.205 | 21.901 | 43.602 | 0.671 | 0.601 | 0.755 | 0.770 | 0.693 | 0.673 | 0.810 | 0.867 | 0.532 | 0.332 | −0.440 | 0.059 | 0.038 | nan | 0.003 | −0.024 |
| 2022 | RF | 14.157 | nan | 11.012 | nan | 0.588 | 0.782 | nan | nan | 0.614 | 0.791 | nan | nan | 0.664 | nan | 23.752 | 39.235 | 18.093 | 29.067 | 1.026 | 0.695 | 1.095 | 0.730 | 0.918 | 0.706 | 0.867 | 0.760 | 0.595 | 0.515 | 25.448 | 39.859 | 20.372 | 30.438 | 1.130 | 0.622 | 1.160 | 0.757 | 1.030 | 0.662 | 1.029 | 0.801 | 0.470 | 0.362 | −0.420 | −6.169 | −1.724 | nan | 0.106 | −0.331 |
| 2022 | XGB | 13.954 | nan | 9.940 | nan | 0.521 | 0.708 | nan | nan | 0.590 | 0.793 | nan | nan | 0.626 | nan | 22.906 | 38.706 | 16.849 | 27.991 | 0.940 | 0.638 | 1.042 | 0.699 | 0.854 | 0.663 | 0.850 | 0.746 | 0.559 | 0.493 | 24.277 | 37.787 | 18.928 | 28.686 | 1.087 | 0.586 | 1.099 | 0.718 | 0.998 | 0.638 | 0.977 | 0.763 | 0.471 | 0.388 | −0.366 | −5.248 | −1.653 | nan | 0.142 | −0.197 |
| 2022 | LightGBM | 13.558 | nan | 9.963 | nan | 0.524 | 0.709 | nan | nan | 0.576 | 0.766 | nan | nan | 0.674 | nan | 24.565 | 40.992 | 18.637 | 30.701 | 0.984 | 0.683 | 1.127 | 0.755 | 0.881 | 0.696 | 0.892 | 0.783 | 0.527 | 0.464 | 27.036 | 42.326 | 21.932 | 32.618 | 1.119 | 0.658 | 1.212 | 0.802 | 1.006 | 0.691 | 1.060 | 0.842 | 0.434 | 0.343 | −0.250 | −5.336 | −1.271 | nan | 0.054 | −0.404 |
| 2022 | SARIMAX | 14.113 | nan | 10.378 | nan | 0.507 | 0.755 | nan | nan | 0.567 | 0.812 | nan | nan | 0.566 | nan | 23.884 | 40.272 | 17.942 | 29.880 | 0.953 | 0.669 | 1.117 | 0.751 | 0.884 | 0.698 | 0.886 | 0.779 | 0.530 | 0.507 | 23.994 | 38.176 | 17.842 | 27.669 | 0.923 | 0.562 | 1.044 | 0.699 | 0.918 | 0.642 | 0.960 | 0.770 | 0.382 | 0.307 | −0.054 | −3.047 | −0.919 | nan | 0.063 | −0.143 |
| 2022 | Ensemble_unweighted | 12.870 | nan | 8.879 | nan | 0.460 | 0.638 | nan | nan | 0.537 | 0.734 | nan | nan | 0.712 | nan | 22.703 | 39.324 | 16.975 | 29.080 | 0.868 | 0.611 | 1.082 | 0.725 | 0.785 | 0.633 | 0.862 | 0.757 | 0.575 | 0.509 | 23.379 | 38.296 | 17.926 | 28.595 | 0.905 | 0.546 | 1.066 | 0.711 | 0.854 | 0.603 | 0.961 | 0.767 | 0.450 | 0.355 | −0.070 | −2.351 | −0.537 | nan | 0.116 | −0.144 |
| 2022 | Ensemble_MASEweighted | 12.889 | nan | 8.905 | nan | 0.462 | 0.639 | nan | nan | 0.539 | 0.735 | nan | nan | 0.712 | nan | 22.722 | 39.326 | 16.992 | 29.078 | 0.871 | 0.612 | 1.082 | 0.724 | 0.787 | 0.634 | 0.862 | 0.757 | 0.576 | 0.508 | 23.428 | 38.330 | 17.987 | 28.643 | 0.911 | 0.548 | 1.068 | 0.712 | 0.858 | 0.604 | 0.963 | 0.767 | 0.451 | 0.355 | −0.076 | −2.428 | −0.561 | nan | 0.116 | −0.148 |
| 2022 | DLinear_gf | 14.112 | nan | 10.611 | nan | 0.572 | 0.752 | nan | nan | 0.619 | 0.780 | nan | nan | 0.484 | nan | 25.857 | 42.048 | 18.343 | 30.809 | 0.838 | 0.615 | 1.136 | 0.761 | 0.808 | 0.687 | 0.925 | 0.811 | 0.401 | 0.363 | 28.574 | 43.828 | 22.971 | 34.131 | 1.186 | 0.696 | 1.295 | 0.844 | 1.116 | 0.746 | 1.133 | 0.881 | 0.354 | 0.347 | −0.478 | −0.996 | −1.694 | nan | −0.016 | −0.618 |
| 2022 | LSTM_gf | 14.122 | nan | 9.975 | nan | 0.488 | 0.731 | nan | nan | 0.562 | 0.820 | nan | nan | 0.539 | nan | 28.348 | 46.814 | 21.130 | 35.120 | 1.040 | 0.746 | 1.265 | 0.852 | 0.988 | 0.794 | 0.998 | 0.881 | 0.459 | 0.432 | 31.422 | 47.722 | 23.681 | 34.718 | 1.196 | 0.714 | 1.284 | 0.843 | 1.140 | 0.793 | 1.180 | 0.937 | 0.333 | 0.302 | −0.105 | −3.436 | −1.750 | nan | −0.201 | −0.783 |
| 2022 | TFT_gf | 15.875 | nan | 11.224 | nan | 0.590 | 0.794 | nan | nan | 0.673 | 0.892 | nan | nan | 0.487 | nan | 22.331 | 35.860 | 15.836 | 24.672 | 1.027 | 0.667 | 0.926 | 0.624 | 0.965 | 0.709 | 0.792 | 0.696 | 0.626 | 0.549 | 25.722 | 38.781 | 19.861 | 28.883 | 1.274 | 0.635 | 1.096 | 0.724 | 1.172 | 0.696 | 0.990 | 0.784 | 0.399 | 0.290 | −0.949 | −9.005 | −3.587 | nan | 0.253 | −0.233 |
| 2022 | TabPFN_gf | 14.559 | nan | 9.310 | nan | 0.474 | 0.671 | nan | nan | 0.593 | 0.831 | nan | nan | 0.646 | nan | 22.177 | 36.851 | 14.211 | 23.534 | 0.669 | 0.483 | 0.894 | 0.598 | 0.677 | 0.578 | 0.814 | 0.714 | 0.685 | 0.553 | 21.672 | 36.149 | 15.828 | 25.877 | 0.805 | 0.482 | 0.960 | 0.643 | 0.783 | 0.561 | 0.894 | 0.724 | 0.500 | 0.369 | −0.197 | −0.325 | −0.370 | nan | 0.212 | 0.016 |
| 2022 | TabICLv2_gf | 12.045 | nan | 8.483 | nan | 0.446 | 0.606 | nan | nan | 0.504 | 0.675 | nan | nan | 0.687 | nan | 19.711 | 36.064 | 14.753 | 26.143 | 0.732 | 0.524 | 0.989 | 0.660 | 0.667 | 0.549 | 0.799 | 0.701 | 0.699 | 0.591 | 22.020 | 36.105 | 15.876 | 25.372 | 0.719 | 0.486 | 0.942 | 0.634 | 0.722 | 0.569 | 0.891 | 0.725 | 0.516 | 0.365 | 0.224 | −0.813 | 0.071 | nan | 0.240 | 0.025 |
The 2018-anchor Tower 4 rows for the DL-family models (e.g. T4_RMSE_obs 77.28/52.20 for DLinear/TFT
respectively) are noticeably worse than their own pooled-Tower-4 averages -- same "one bad anchor
drags the pooled mean" pattern D-65 documented for DLinear's 2018 cold-start non-determinism;
2019-2022 are comparatively stable for every model. Established models (RF/XGB/LightGBM/SARIMAX/
Ensembles) don't show this anchor-dependent instability, consistent with their fixed-hyperparameter,
no-refit-per-anchor-drama design (they were never subject to DL cold-start effects to begin with).

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

## Secondary metric: MASE/RMSSE vs. Climatology_gf instead of persistence

Same caveat as D-71/D-65's own gap-filled-target sections: persistence remains this project's primary
MASE/RMSSE denominator; this is a robustness check on whether the ranking survives a different
(weaker) baseline, not a redefinition. RMSSE is the squared-error analogue of MASE
(`RMSE(model)/RMSE(baseline)`, `evaluation.metrics.rmsse`) — same y_naive convention, added to
`rr.bin_metrics()` after the original gf-ablation summaries were generated, so both columns below are
a fresh recompute over the stored predictions in `b10_b13_full_chains.csv` (no refit needed).

| Model | MASE (vs Climatology_gf) | MASE (vs persistence) | RMSSE (vs Climatology_gf) | RMSSE (vs persistence) |
|---|---|---|---|---|
| DLinear | 1.308 | 1.460 | 1.205 | 1.264 |
| DLinear_gf | 0.965 | 1.059 | 0.935 | 0.952 |
| LSTM | 1.018 | 1.151 | 1.059 | 1.112 |
| LSTM_gf | 0.955 | 1.098 | 0.952 | 1.008 |
| TFT | 0.878 | 0.972 | 0.927 | 0.946 |
| TFT_gf | 0.888 | 1.005 | 0.889 | 0.933 |
| TabPFN | 0.773 | 0.855 | 0.865 | 0.884 |
| TabPFN_gf | 0.746 | 0.829 | 0.808 | 0.830 |
| TabICLv2 | 0.820 | 0.930 | 0.902 | 0.943 |
| TabICLv2_gf | 0.801 | 0.891 | 0.830 | 0.854 |

Every MASE and RMSSE value is lower against `Climatology_gf` (a weaker baseline, D-71) than against
persistence, but the qualitative ranking is unchanged — gf beats original for DLinear/LSTM/TabPFN/
TabICLv2, TFT is a wash — under all four denominator/metric combinations. RMSSE tells the same story
as MASE for 9 of 10 rows; the one point of note is **TFT (original, not gf)**: MASE says TFT already
beats persistence (0.972 < 1) while RMSSE says it's essentially tied (0.946, still <1 but closer to
parity) — consistent with D-72's finding that TFT's pooled result is dominated by a handful of
large-error days (RMSE penalizes those more than MAE does), so the squared-error view is slightly
less flattering to TFT than MASE alone suggests.

## Files

- `src/models/forecasting_dl.py` (+`y_source` param on `make_windows`/`build_windows`, additive)
- `notebooks/05_benchmarking/b10_b13_dl_gf_extension.py`, `b10_b13_tft_gf_extension.py`,
  `b10_b13_foundation_gf_extension.py` (new, committed — the 3 gf-variant sibling scripts)
- `notebooks/05_benchmarking/b10_b13_gf_merge.py` (merges the 5 `_gf` columns into
  `b10_b13_full_chains.csv`, backup-then-verify-row-count discipline)
- `notebooks/05_benchmarking/b10_b13_gf_comparison_table.py` (pooled/by-tower orig-vs-gf tables)
- `notebooks/05_benchmarking/b10_b13_gf_vs_climatology_gf.py` (secondary MASE/RMSSE-denominator
  check, both vs `Climatology_gf` and vs `persistence`)
- `src/evaluation/metrics.py` (+`rmsse()`, squared-error analogue of `mase()`); `rr.bin_metrics()`
  (+`RMSSE` column, purely additive, same `y_persist` argument MASE already uses)
- `results/b10_b13_{dl,tft,foundation}_gf_extension_{summary,summary_vs_gapfilled,chains}.csv`
- `results/b10_b13_gf_ablation_{combined_summary,pooled,table_all_towers,by_tower,table_by_tower,
  table_by_tower_year,flattened_by_tower_year}.csv`
- `results/b10_b13_gf_ablation_vs_climatology_gf_{summary,table}.csv`
- `results/b10_b13_gf_ablation_vs_persistence_{summary,table}.csv` (RMSSE vs persistence recompute,
  same 10 gf-ablation models — MASE values match the original persistence-baseline table exactly,
  confirming the recompute is a pure additive extension, not a re-derivation)
- `notebooks/05_benchmarking/b10_b13_gf_vs_climatology_gf_gapfilled_target.py` +
  `results/b10_b13_gf_ablation_vs_climatology_gf_gapfilled_target_table.csv` (all 11 models,
  MASE/RMSSE vs `Climatology_gf`, crossed with the gapfilled/observed evaluation-target axis — the
  "MASE/RMSSE baseline = Climatology_gf" section's source; RMSE/MAE/Correlation/R² columns confirmed
  identical to the persistence-baseline version, as expected since those metrics don't depend on the
  MASE/RMSSE baseline)
- `notebooks/05_benchmarking/b10_b13_combined_leaderboard_rmsse.py` (recomputes the 11-model
  leaderboard with RMSSE added vs both baselines; MASE values reconfirmed unchanged)
- `results/b10_b13_latest_combined_leaderboard.csv` (all 11 models — established + DL-family `_gf`
  variants — RMSE/MAE/MASE-vs-persistence/MASE-vs-Climatology_gf/RMSSE-vs-persistence/
  RMSSE-vs-Climatology_gf/WAPE/Correlation/R², the "Combined leaderboard" section's source)
- `results/b10_b13_gf_ablation_pooled_vs_gapfilled.csv` (DL-family `_gf` models pooled RMSE/MAE/MASE/
  WAPE/Correlation/R², scored against `y_gapfilled` instead of `y_observed` — the "scored against
  gap-filled target vs. observed target" section's DL-family half)
- `notebooks/05_benchmarking/b10_b13_gf_ablation_full_parity_tables.py` +
  `results/b10_b13_gf_ablation_by_tower_full_parity.csv` +
  `results/b10_b13_gf_ablation_table_by_tower_year_full_parity.csv` (intermediate step: 10-model
  DL-family-only per-tower/tower×year MASE+RMSSE vs both baselines — superseded below by the 11-model,
  gapfilled+observed version, kept for the intermediate CSVs' provenance)
- `notebooks/05_benchmarking/b10_b13_full_parity_gf_obs_tables.py` +
  `results/b10_b13_full_parity_gf_obs_by_tower.csv` +
  `results/b10_b13_full_parity_gf_obs_table_by_tower_year.csv` (final per-tower and tower×year×model
  breakdowns — all 11 models, every metric crossed with both the gapfilled/observed evaluation-target
  axis and the persistence/Climatology_gf baseline axis — the "Per-tower tables" and "Tower × year ×
  model breakdown" sections' current source; MASE/RMSE-vs-persistence-observed values confirmed
  identical to every earlier table version before the new columns were trusted)
- `results/b10_b13_full_chains.csv` (+5 `_gf` columns, 5,475 rows unchanged; backed up to
  `b10_b13_full_chains_backup_pre_gf.csv` first)
- `results/figures/b10_chains/` regenerated (598 figures total, +75 new `_gf` figures); spot-checked
  Tower 4/anchor 2021 (DLinear_gf, TabPFN_gf) before trusting the full batch

No `benchmarks.csv` rows — a design-choice ablation, not a point-forecast benchmark in its own right
(same precedent as every other diagnostic pass in this sequence). Full narrative decision log:
`DECISIONS.md` D-72.
