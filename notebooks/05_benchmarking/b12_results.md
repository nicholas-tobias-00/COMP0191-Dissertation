# B-12 — combined: B-10's ensemble + B-11's monthly rollout/downscale

**Notebook:** `B12_combined.ipynb` (single-anchor smoke test, 2021-12-16 daily / 2021-11-01
monthly). **Results:** `results/b12_summary.csv` (single-anchor), `results/b12_multi_anchor.csv`
(5-anchor extension, 2018–2022 — the headline verdict). No new shared-module code — reuses
`recursive_rollout.py`'s existing `tree_rollout`/`monthly_rollout`/`downscale_monthly_to_daily`
unmodified, hardcoding B-10's and B-11's winning configurations as frozen constants.

**Status: executed** (user confirmed proceeding after B-10/B-11 completed with time to spare,
despite B-11's finding that the monthly-downscaling mechanism this combines with mostly cancels
out its own gain — see D-55).

Combines B-10's winning idea (unweighted ensemble of RF+XGB+LightGBM+SARIMAX, D-54) with B-11's
monthly-rollout-plus-downscaling framework (D-55): build the ensemble at **monthly** resolution,
then downscale it to daily using **B-10's own daily ensemble chain** (not each individual model's
own chain, as B-11 did) as the within-month shape template.

## 0. Downscaling exactness check

Confirmed exact by construction, as in B-11: max abs diff = 0.0000000000 between the downscaled
daily series' monthly mean and the input monthly ensemble prediction, for every month.

## 1. Single-anchor result (2021-12-16): looks like a clear win

| Model | Overall R² (n-weighted) | Overall MASE (n-weighted) |
|---|---|---|
| **B12_monthly_ensemble_downscaled** | **0.075** | **1.002** |
| B10_daily_ensemble_original | -0.034 | 1.122 |

By bin (R²): B-12 beats B-10 in 271-365 (0.066 vs -0.436), 8-30 (-0.052 vs -0.200), and roughly
ties in 1-7 and 181-270; B-10 is slightly ahead in 31-90 and 91-180. **At this one anchor, B-12
looks like the best recursive-rollout configuration found across B-09/B-10/B-11/B-12 combined.**

## 2. Multi-anchor (2018-2022) extension: reverses the single-anchor conclusion

Per this project's own repeated lesson (already demonstrated twice this session — B-09 §3 on
DLinear, B-10 §3 on the H=1 retrain single-anchor read), the single-anchor result above is not
trustworthy on its own. Extended to the same 5-anchor sweep (`b12_multi_anchor.py`, ~112s):

| Model | Mean R² (5 anchors) | Mean MASE (5 anchors) |
|---|---|---|
| **B10_daily_ensemble_original** | **0.012** | **0.975** |
| B12_monthly_ensemble_downscaled | -0.011 | 0.993 |

**The multi-anchor result reverses §1's finding** — B-12 is now slightly *worse* than B-10's daily
ensemble alone, not better. By bin (mean R² across 5 anchors):

| Bin | B10 (daily ensemble) | B12 (monthly ensemble, downscaled) |
|---|---|---|
| 1-7 | -3.292 | **-1.986** |
| 8-30 | -1.006 | **-0.740** |
| 31-90 | -0.046 | -0.039 |
| 91-180 | **0.001** | -0.152 |
| 181-270 | **0.232** | 0.190 |
| 271-365 | 0.011 | **0.072** |

The short-lead bins (1-7, 8-30) and the late-window bin (271-365) genuinely improve, consistent
with B-11's own bin-level finding — but the mid-range bins (91-180, 181-270) get worse, and the
net effect across all bins/anchors is a small negative.

## 3. Why — the same mechanism B-11 already identified, now confirmed a second time

The downscaling step inherits the ensemble's own within-month shape errors unchanged; only the
coarse monthly bias gets corrected. Averaging this effect across 5 different years erodes the
apparent single-anchor gain, exactly as B-11 (D-55) found for the individual per-model downscaled
chains. **This is itself a third demonstration of this project's own headline methodological
lesson this session: don't trust a single-anchor backtest, even one that looks clean and
well-motivated.**

## 4. Recommendation

- **B-10's daily ensemble alone remains the best available daily-resolution recursive-rollout
  result** (mean R²=0.012) — combining it with B-11's monthly-downscaling framework does not
  improve on it, and slightly hurts on average across 5 anchor years.
- **Do not deploy B-12's combined configuration** in place of B-10's simpler ensemble — it is not
  simpler (requires building and maintaining both a daily and a monthly model per forecast), and
  does not perform better.
- **The exercise was still worth running**: it is a clean, concrete confirmation (not just an
  inference from B-10+B-11 separately) that this combination doesn't help, and a third instance of
  the single-anchor-vs-multi-anchor reversal pattern that has recurred throughout this session —
  worth keeping in mind for any future single-anchor forecasting result in this project.

## 5. Caveats

- Single-anchor smoke test numbers showed minor run-to-run variation between an earlier prototype
  script run (R²=0.082) and the notebook's own executed run (R²=0.075) for the monthly-ensemble
  configuration specifically — both point the same direction and both are reversed by the
  multi-anchor sweep, so this doesn't affect the conclusion, but is a reminder that tree ensembles
  fit with `n_jobs=-1` can show small non-deterministic variation run-to-run even with a fixed
  `random_state` (LightGBM/XGBoost histogram-based training under multi-threading is a known source
  of this). B-10's daily ensemble baseline was exactly reproducible across both runs.
- Same ground-truth and single-calendar-date-anchor caveats as B-09/B-10/B-11.

## Files / scope

New: `notebooks/05_benchmarking/B12_combined.ipynb`, `results/b12_summary.csv` (single-anchor),
`results/b12_multi_anchor.csv` (5-anchor). No new shared-module code, no new production files
modified — reuses `recursive_rollout.py`'s existing functions unmodified throughout.

*Source: `B12_combined.ipynb`, `results/b12_summary.csv`, `results/b12_multi_anchor.csv`.
Cross-ref D-54/B-10 (source of the ensemble idea and its daily-resolution baseline), D-55/B-11
(source of the downscaling mechanism and the mechanism explanation in §3).*
