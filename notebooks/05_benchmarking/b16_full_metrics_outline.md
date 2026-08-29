# B-16 full metrics outline — full 11-model roster, both targets, all metrics (D-96)

Consolidated, correctly-baselined forecasting metrics across every model this project has
benchmarked in the recursive-rollout sequence (B-09→B-16), covering both evaluation targets
(`y_observed`, `y_gapfilled`) and the full metric set (MASE, R², RMSE, WAPE, Correlation). Built to
answer "what is the latest, correct picture of every forecasting metric" after D-96 found the
gap-filled-target MASE column had never been rescored under D-80's climatology convention.

**Generator:** `notebooks/05_benchmarking/b16_full_metrics_outline.py` (pure arithmetic on
already-saved per-bin MAE/R²/RMSE/WAPE/Correlation columns — no new model calls). **Data:**
`results/b16_full_metrics_outline.csv`.

## Conventions

- **MASE baseline: climatology (D-80), matched to each column's own target** — `MAE_climatology`
  (built from `y_observed` history) for the observed column, `MAE_climatology_gf` (D-96, built from
  `y_gapfilled` history) for the gapfilled column. This is the first table in the project where
  both MASE columns use a baseline that's actually consistent with the target it's scoring.
- **Aggregation: single-stage n-weighted mean per (model, config)**, matching D-80's own
  `mase_climatology()` exactly (`temp_forecasting_pipeline.ipynb` cell 16 — the function that
  produced the officially-cited 0.715 champion number). Deliberately reused rather than the
  two-stage anchor-then-mean convention used elsewhere in this project
  (`b16_recursive_rollout_v3_gapfilled.py`), so this table's MASE (obs) column reproduces the
  published headline bit-for-bit instead of introducing a third aggregation convention.
- **Config selection: each model's own best config on the observed-target climatology MASE**
  (the metric that determines champion status, D-36/D-37/D-80) — the gapfilled-side columns are
  read off that *same* config, not re-optimized per target, so each row is a genuine
  same-model/same-config, both-targets comparison.
- R²/RMSE/WAPE/Correlation are baseline-independent (they don't depend on the MASE
  denominator choice at all) — pulled directly as n-weighted means from the same raw per-bin files.

## Table — MASE-ranked (primary metric)

| Model | Best config | MASE (obs) | MASE (gf) | R² (obs) | R² (gf) | RMSE (obs) | RMSE (gf) | WAPE (obs) | WAPE (gf) | Corr (obs) | Corr (gf) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **TabPFN** | BASE+bodyweight | **0.715** | 0.903 | −0.095 | −0.657 | 56.16 | 34.90 | 0.880 | 0.814 | 0.356 | 0.240 |
| TabICLv2 | BASE+ALL | 0.739 | 0.922 | −0.143 | −0.784 | 56.61 | 35.27 | 0.905 | 0.841 | 0.302 | 0.210 |
| Ensemble_MASEweighted | BASE | 0.802 | 0.666 | −0.163 | −0.196 | 52.25 | 25.38 | 0.991 | 0.704 | 0.379 | 0.522 |
| Ensemble_unweighted | BASE | 0.802 | 0.666 | −0.162 | −0.190 | 52.25 | 25.39 | 0.991 | 0.705 | 0.379 | 0.523 |
| XGB | BASE+ALL | 0.804 | 0.677 | −0.201 | −0.584 | 52.30 | 25.87 | 0.988 | 0.725 | 0.364 | 0.477 |
| TFT | BASE+ALL | 0.812 | 1.021 | −0.237 | −2.179 | 55.78 | 36.10 | 1.009 | 1.103 | 0.236 | 0.147 |
| LightGBM | BASE | 0.817 | 0.689 | −0.214 | −0.480 | 52.83 | 26.11 | 1.014 | 0.728 | 0.371 | 0.496 |
| RF | BASE | 0.841 | 0.703 | −0.241 | −0.593 | 52.91 | 25.85 | 1.045 | 0.756 | 0.380 | 0.529 |
| SARIMAX | BASE | 0.874 | 0.831 | −0.329 | −1.002 | 54.35 | 29.21 | 1.082 | 0.919 | 0.348 | 0.463 |
| LSTM | BASE+species | 0.952 | 1.122 | −0.651 | −2.724 | 62.32 | 39.79 | 1.154 | 1.188 | 0.275 | 0.185 |
| DLinear | BASE+bodyweight | 1.187 | 1.502 | −1.751 | −6.792 | 61.99 | 42.59 | 1.519 | 1.653 | 0.277 | 0.182 |

n = 2,322 (weighted total across 3 towers × 5 anchors × 6 lead-time bins) per row.

## Reading notes

- **Champion: `TabPFN`** on the metric this project treats as authoritative (MASE, observed
  target, climatology baseline). `TabPFN+bodyweight` and `TabPFN+species` are a near-exact tie
  (0.715 both, `+bodyweight` ahead by <0.001) — within noise, consistent with the standing
  `TabPFN+species` recommendation (D-67).
- **Ensemble wins on the gapfilled column** (0.666) — this is the training-target circularity
  effect discussed at D-65/D-67/D-96 (trees/SARIMAX/ensembles are fit by regressing directly onto
  `y_gapfilled`), not a competing model recommendation. Read the observed column as authoritative.
- Every column here uses a baseline convention that's actually consistent with its own target —
  the first time that's been true across the whole roster in one place. Prior tables
  (`b16_final_table_vs_gapfilled_best_config.csv`, `F10_results.md`'s original secondary-metric
  table) mixed conventions on the gapfilled side; superseded by this table for that reason, kept
  for history.

## Sources

`results/b16_foundation_models_v3_summary.csv` (TabPFN, TabICLv2), `results/
b16_recursive_rollout_v3_summary_vs_gapfilled.csv` (RF/XGB/LightGBM/SARIMAX/Ensembles), `results/
b16_dl_models_v3_summary.csv` (TFT/DLinear/LSTM), `results/_today_climatology_baseline.csv`
(observed-target climatology baseline, D-80), `results/b16_climatology_gf_baseline_v3.csv`
(gapfilled-target climatology baseline, D-96). Cross-reference: `BEST_RESULTS.md` §3,
`notebooks/04_feature_engineering/F10_results.md`, `DECISIONS.md` D-67/D-80/D-96.
