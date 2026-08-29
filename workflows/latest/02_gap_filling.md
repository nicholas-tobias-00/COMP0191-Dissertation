# 2. Gap filling

## Promoted methods

- **Production reference:** pooled RFm with external Site MET, REddyProc-style drivers, management,
  stocking density, and lagged soil variables.
- **Benchmark-best:** TabICLv2-solo fitted separately per tower with 30 predictors, mean predictor
  imputation, at most 10,000 observed context rows, and seed 42.

The full experimental record remains
`notebooks/03c_gap_filling_revisited/temp_gap_filling_pipeline.ipynb`. Its historical `temp_` name
is not a status marker; promotion is recorded in `BEST_RESULTS.md`.

## Recreate the benchmark-best continuous series

```powershell
python src/data/build_fch4_gapfilled_tabicl.py
```

## Uncertainty products

Native, uncalibrated hourly intervals and report chains:

```powershell
python notebooks/03c_gap_filling_revisited/export_latest_tabicl_uq_chains.py
```

Exact-config split-conformal calibration and calibrated plots:

```powershell
python notebooks/03c_gap_filling_revisited/export_tabiclv2_solo_calibrated_uq.py
```

The calibrated script reuses persisted D5.5 folds by default. `--full-cqr` performs the slower
native-endpoint rerun.

Observed-only temporal diagnostics used in the appendix:

```powershell
python notebooks/03c_gap_filling_revisited/export_observed_tsa_appendix.py
```

Gap-CV remains the evaluation protocol. Report whether an R2 value is sklearn R2 or the bounded
OLS/squared-Pearson convention; they are not interchangeable.

