# D-71: is chain-persistence a valid MASE baseline for a seasonal series?

**Addendum (2026-07-15, same day): fairness fix.** The original climatology baseline was built from
real `y_observed` history, while `persistence`'s single anchor value comes from `y_gapfilled` — not
an apples-to-apples comparison. A second, gap-filled-basis climatology variant (`Climatology_gf`)
was added the same day to isolate "flat vs. seasonal" from "real vs. gap-filled data source." See
"Gap-filled-basis follow-up" below; both variants are kept, neither replaces the other.

## Context

This project's MASE denominator throughout (D-37) is chain-persistence — the anchor day's real
value, held flat for the full 365-day rollout. Hyndman & Koehler's own MASE convention recommends
scaling against a *seasonal-naive* baseline for series with real seasonality, since a flat hold
ignores season entirely and can be trivially beaten at long lead times by any model that merely
tracks the seasonal cycle. FCH4 clearly has real seasonality, raising a direct, user-posed question:
is persistence actually a fair/valid MASE baseline here, or is this project's headline MASE an
artificially easy number?

This project already has a seasonal-mean analogue — `rr.doy_climatology()` (day-of-year mean,
±7-day window, from strictly pre-anchor history) — but it had only ever been computed for a single
tower/anchor, informally, inside B-09's original smoke test (D-53), never extended to full coverage
or used to rescale B-10/B-13's headline metrics.

## Method

`notebooks/05_benchmarking/b10_b13_climatology_baseline.py` (new, committed) reproduces B-09's exact
climatology recipe — `hist = dft.loc[:anchor - 1 day, "y_observed"].dropna()`,
`rr.doy_climatology(hist, target_dates, window=7)` — for all 3 towers × 5 anchors (2018–2022),
merges the result into `results/b10_b13_full_chains.csv` as a new `Climatology` column, then reruns
`rr.bin_metrics()` (unmodified) per (tower, anchor, model) with `y_persist=Climatology` instead of
`persistence` — recomputing MASE only for the full 11-model B-10/B-13 roster. **No models are
refit** — this reuses the predictions already sitting in `b10_b13_full_chains.csv`; R²/RMSE/MAE/
WAPE/Correlation are baseline-independent and mathematically unchanged, verified identical between
the persistence-scaled and climatology-scaled raw summary files before trusting only the new MASE
column.

**One real, expected NaN case, not a bug**: Tower 9 has zero pre-anchor real `y_observed` history
for its 2018 and 2019 anchors (confirmed directly — `n_hist_obs=0` for both), so climatology's
`global_mean` fallback (`np.nanmean` of an empty array) is itself undefined. Handled by falling
back to `y_persist=None` for those two (tower, anchor) combos, since sklearn's
`mean_absolute_error` raises on NaN input rather than silently propagating it (unlike plain numpy).

## Baseline coverage: why climatology isn't available everywhere persistence is

Real `y_observed` coverage differs sharply by tower, which determines both (a) how many bins have
any real data to score against at all, and (b) whether enough *pre-anchor* history exists to build
a climatology curve:

| Tower | First real `y_observed` | Last real `y_observed` | n |
|---|---|---|---|
| 2 | 2017-10-06 | 2019-05-31 | 404 |
| 4 | 2017-10-07 | 2023-12-29 | 1,585 |
| 9 | **2020-02-13** | 2023-12-29 | 909 |

Non-null MASE cells per (tower, anchor), out of 66 possible (11 models × 6 bins):

| Tower | 2018 | 2019 | 2020 | 2021 | 2022 |
|---|---|---|---|---|---|
| 2 | 44 (persist=clim) | 0 | 0 | 0 | 0 |
| 4 | 55 | 55 | 66 | 66 | 33 |
| 9 | 0 | **44 (persist) / 0 (clim)** | 66 | 44 | 44 |

Tower 2's real record ends mid-2019, so every anchor from 2019 onward has zero real data in its
forecast window for *either* baseline. Tower 9's real record only starts 2020-02-13: its 2018 anchor
has no real data for either baseline; its **2019 anchor is the one genuine "climatology-only" gap**
— real data exists later in that window (persistence can score against it), but climatology needs
real history *before* the 2019-12-16 anchor, and there is none yet.

## All-tower summary (pooled, both climatology variants)

Aggregated via this project's established convention (per-anchor n-weighted mean across the 6
lead-time bins, then simple mean across the 5 anchors):

| Model | MASE (persistence) | MASE (climatology, `y_observed`-basis) | MASE (climatology, `y_gapfilled`-basis) | R² |
|---|---|---|---|---|
| RF | 0.968 | 0.837 | 0.869 | −0.241 |
| XGB | 0.922 | 0.797 | 0.824 | −0.184 |
| LightGBM | 0.941 | 0.804 | 0.837 | −0.206 |
| SARIMAX | 0.976 | 0.882 | 0.901 | −0.360 |
| Ensemble_unweighted | 0.918 | 0.796 | 0.828 | −0.165 |
| Ensemble_MASEweighted | 0.918 | 0.795 | 0.827 | −0.165 |
| TFT | 0.972 | 0.841 | 0.878 | −0.363 |
| TabPFN | 0.855 | 0.733 | 0.773 | −0.122 |
| DLinear | 1.460 | 1.265 | 1.308 | −2.068 |
| LSTM | 1.151 | 0.956 | 1.018 | −1.357 |
| TabICLv2 | 0.930 | 0.782 | 0.820 | −0.330 |

Source: `results/b10_b13_climatology_mase_table_all_towers.csv`,
`results/b10_b13_climatology_gf_mase_table_all_towers.csv`.

### Full metric comparison (RMSE/MAE/WAPE/Correlation/R² are baseline-invariant — shown once; MASE shown for all 3 baselines)

Same style as `b10_b13_metrics_rerun.md`'s gap-filled-vs-observed comparison tables, adapted to
what actually varies here: RMSE/MAE/WAPE/Correlation/R² do not depend on the MASE baseline at all
(confirmed identical between the persistence-scaled and climatology-scaled raw summary files), so
each appears once; only MASE changes across the three columns.

**All-tower pooled:**

| Model | RMSE | MAE | MASE (persistence) | MASE (climatology, obs) | MASE (climatology, gf) | WAPE | Correlation | R² |
|---|---|---|---|---|---|---|---|---|
| RF | 52.23 | 34.84 | 0.968 | 0.837 | 0.869 | 1.050 | 0.375 | −0.241 |
| XGB | 51.57 | 33.81 | 0.922 | 0.797 | 0.824 | 0.991 | 0.368 | −0.184 |
| LightGBM | 52.08 | 34.32 | 0.941 | 0.804 | 0.837 | 1.012 | 0.368 | −0.206 |
| SARIMAX | 53.79 | 36.06 | 0.976 | 0.882 | 0.901 | 1.105 | 0.343 | −0.360 |
| **Ensemble_unweighted** | 51.57 | 33.75 | **0.918** | 0.796 | 0.828 | 0.998 | **0.375** | **−0.165** |
| Ensemble_MASEweighted | 51.57 | 33.74 | 0.918 | 0.795 | 0.827 | 0.998 | 0.375 | −0.165 |
| TFT | 56.59 | 35.62 | 0.972 | 0.841 | 0.878 | 1.045 | 0.292 | −0.363 |
| TabPFN | 56.12 | 33.14 | 0.855 | 0.733 | 0.773 | 0.899 | 0.358 | −0.122 |
| DLinear | 64.53 | 47.52 | 1.460 | 1.265 | 1.308 | 1.638 | 0.237 | −2.068 |
| LSTM | 63.22 | 41.06 | 1.151 | 0.956 | 1.018 | 1.268 | 0.212 | −1.357 |
| TabICLv2 | 58.05 | 34.95 | 0.930 | 0.782 | 0.820 | 0.988 | 0.255 | −0.330 |

Source: `results/b10_b13_climatology_full_metrics_all_towers.csv`.

**Tower 2:**

| Model | RMSE | MAE | MASE (persistence) | MASE (climatology, obs) | MASE (climatology, gf) | WAPE | Correlation | R² |
|---|---|---|---|---|---|---|---|---|
| RF | 22.13 | 18.05 | 0.346 | 0.672 | 0.906 | 1.347 | 0.087 | −0.889 |
| XGB | 20.82 | 16.45 | 0.312 | 0.580 | 0.791 | 1.205 | 0.214 | −0.618 |
| LightGBM | 20.67 | 16.32 | 0.310 | 0.580 | 0.792 | 1.204 | 0.182 | −0.600 |
| SARIMAX | 34.10 | 30.76 | 0.574 | 1.055 | 1.399 | 2.286 | 0.056 | −3.244 |
| **Ensemble_unweighted** | 23.66 | 19.82 | **0.374** | 0.698 | 0.941 | 1.464 | 0.176 | **−1.048** |
| Ensemble_MASEweighted | 23.54 | 19.69 | 0.372 | 0.693 | 0.936 | 1.455 | 0.178 | −1.029 |
| TFT | 29.53 | 20.84 | 0.400 | 0.801 | 1.041 | 1.549 | 0.111 | −2.312 |
| TabPFN | 18.25 | 12.96 | 0.243 | 0.444 | 0.608 | 0.969 | 0.098 | −0.240 |
| DLinear | 36.53 | 30.20 | 0.588 | 1.125 | 1.642 | 2.389 | 0.009 | −5.421 |
| LSTM | 33.05 | 26.46 | 0.508 | 0.952 | 1.362 | 2.054 | 0.319 | −3.791 |
| TabICLv2 | 18.44 | 13.01 | 0.240 | 0.419 | 0.566 | 0.957 | 0.181 | −0.219 |

**Tower 4:**

| Model | RMSE | MAE | MASE (persistence) | MASE (climatology, obs) | MASE (climatology, gf) | WAPE | Correlation | R² |
|---|---|---|---|---|---|---|---|---|
| RF | 51.62 | 34.13 | 1.026 | 0.885 | 0.913 | 1.030 | 0.403 | −0.067 |
| XGB | 51.30 | 33.13 | 0.970 | 0.844 | 0.865 | 0.966 | 0.373 | 0.003 |
| LightGBM | 51.44 | 33.30 | 0.980 | 0.846 | 0.870 | 0.980 | 0.385 | −0.014 |
| SARIMAX | 52.46 | 35.28 | 1.040 | 0.903 | 0.925 | 1.049 | 0.380 | −0.039 |
| **Ensemble_unweighted** | **51.04** | 33.23 | **0.977** | 0.845 | 0.868 | 0.979 | 0.397 | **0.012** |
| Ensemble_MASEweighted | 51.03 | 33.22 | 0.977 | 0.845 | 0.868 | 0.979 | 0.397 | 0.011 |
| TFT | 54.48 | 33.43 | 1.014 | 0.874 | 0.897 | 1.020 | 0.315 | −0.228 |
| TabPFN | 54.26 | 30.50 | 0.864 | 0.760 | 0.778 | 0.861 | 0.391 | −0.006 |
| DLinear | 65.71 | 48.11 | 1.626 | 1.364 | 1.418 | 1.685 | 0.237 | −2.105 |
| LSTM | 59.83 | 37.27 | 1.106 | 0.947 | 0.969 | 1.105 | 0.228 | −0.439 |
| TabICLv2 | 56.76 | 33.04 | 0.963 | 0.829 | 0.844 | 0.980 | 0.295 | −0.291 |

**Tower 9:**

| Model | RMSE | MAE | MASE (persistence) | MASE (climatology, obs) | MASE (climatology, gf) | WAPE | Correlation | R² |
|---|---|---|---|---|---|---|---|---|
| RF | 58.41 | 38.30 | 0.944 | 0.767 | 0.774 | 1.001 | 0.389 | −0.348 |
| XGB | 57.56 | 37.59 | 0.920 | 0.746 | 0.752 | 0.971 | 0.392 | −0.358 |
| LightGBM | 58.62 | 38.71 | 0.954 | 0.777 | 0.780 | 1.005 | 0.378 | −0.389 |
| SARIMAX | 60.12 | 38.40 | 0.920 | 0.813 | 0.777 | 0.993 | 0.350 | −0.406 |
| **Ensemble_unweighted** | 57.49 | 36.55 | **0.884** | 0.717 | 0.728 | 0.922 | **0.385** | **−0.253** |
| Ensemble_MASEweighted | 57.49 | 36.58 | 0.885 | 0.718 | 0.728 | 0.924 | 0.385 | −0.255 |
| TFT | 64.87 | 41.66 | 0.974 | 0.761 | 0.809 | 0.978 | 0.304 | −0.209 |
| TabPFN | 65.98 | 40.84 | 0.926 | 0.709 | 0.789 | 0.931 | 0.349 | −0.265 |
| DLinear | 66.59 | 48.34 | 1.251 | 0.995 | 1.024 | 1.345 | 0.287 | −1.073 |
| LSTM | 74.02 | 49.58 | 1.289 | 1.003 | 1.027 | 1.358 | 0.179 | −2.229 |
| TabICLv2 | 67.15 | 41.59 | 0.953 | 0.720 | 0.806 | 0.957 | 0.242 | −0.312 |

Source: `results/b10_b13_climatology_full_metrics_tower2.csv`, `_tower4.csv`, `_tower9.csv`.

**Reading these against RMSE/Correlation/R² (which don't move at all)**: at Tower 2, every model's
MASE roughly doubles or triples going persistence→climatology-gf while RMSE/R²/Correlation stay
fixed — visual confirmation that climatology is a *harder* denominator there. At Towers 4/9, MASE
drops by a similar margin under either climatology variant while RMSE/R²/Correlation again don't
move — confirming the opposite: climatology is the *easier* denominator at those two towers. The
non-MASE columns are the fixed reference point that makes clear the MASE swing is entirely about
the baseline, not the model's actual fit to the data.

**Every model's MASE looks numerically better under either climatology variant, pooled — this is
not evidence any model forecasts better.** It reflects the climatology baselines' own accuracy being
worse than persistence's, pooled (see next section) — dividing by a bigger denominator flatters
every model equally.

**Model ranking is largely stable across all three baselines.** Ranked by MASE (best→worst):
TabPFN stays #1 under every baseline; SARIMAX/LSTM/DLinear stay the bottom 3, in the same order,
under every baseline. Only the middle pack reshuffles: TabICLv2 moves from 5th (persistence) to 2nd
(climatology, either basis) — the model that benefits most from the switch — while both ensembles
slip slightly from 2nd to 4th/5th.

## The actual finding: which baseline is more accurate against real flux?

This is the number that actually explains the table above — each baseline's own MAE against real
`y_true`, pooled (n-weighted across all towers/anchors where climatology is defined):

| Baseline | Pooled MAE vs. real `y_true` |
|---|---|
| Persistence (flat anchor-day hold) | **37.50** |
| Climatology, `y_observed`-basis (original) | 43.79 |
| Climatology, `y_gapfilled`-basis (fairness-fixed) | 40.74 |

**Climatology is the weaker baseline pooled, under either basis** — the opposite of the motivating
hypothesis (seasonal-naive should beat flat persistence for a seasonal series). About half of the
`y_observed`-basis gap closes once the data-source asymmetry is removed (43.79 → 40.74), but it does
not fully close.

### Per (tower, anchor) breakdown

| Tower | Anchor | n | MAE (persistence) | MAE (climatology, obs) | MAE (climatology, gf) |
|---|---|---|---|---|---|
| 2 | 2018 | 102 | 51.94 | 37.26 | **25.86** |
| 4 | 2018 | 263 | 32.73 | 37.10 | 36.95 |
| 4 | 2019 | 229 | 40.11 | 38.05 | 40.13 |
| 4 | 2020 | 263 | 34.84 | 37.39 | 35.95 |
| 4 | 2021 | 341 | 42.82 | 44.10 | 43.06 |
| 4 | 2022 | 224 | 26.49 | 40.70 | 39.55 |
| 9 | 2020 | 254 | 38.29 | 65.97 | 46.22 |
| 9 | 2021 | 180 | 55.12 | 47.75 | 53.15 |
| 9 | 2022 | 271 | 27.07 | 42.57 | 39.87 |

**Tower 2 is the one clear reversal — climatology genuinely is the harder, better baseline there**,
matching the original hypothesis, and more so on the fair (`y_gapfilled`-basis) version than the
original. **Towers 4 and 9 favor persistence in most anchors, under either climatology variant** —
paradoxically the two towers with *more* real-data coverage.

### Per-tower MASE breakdown, all three baselines (all 11 models)

| Tower | Model | MASE (persistence) | MASE (climatology, obs) | MASE (climatology, gf) |
|---|---|---|---|---|
| 2 | RF | 0.346 | 0.672 | 0.906 |
| 2 | XGB | 0.312 | 0.580 | 0.791 |
| 2 | LightGBM | 0.310 | 0.580 | 0.792 |
| 2 | SARIMAX | 0.574 | 1.055 | 1.399 |
| 2 | Ensemble_unweighted | 0.374 | 0.698 | 0.941 |
| 2 | Ensemble_MASEweighted | 0.372 | 0.693 | 0.936 |
| 2 | TFT | 0.400 | 0.801 | 1.041 |
| 2 | TabPFN | 0.243 | 0.444 | 0.608 |
| 2 | DLinear | 0.588 | 1.125 | 1.642 |
| 2 | LSTM | 0.508 | 0.952 | 1.362 |
| 2 | TabICLv2 | 0.240 | 0.419 | 0.566 |
| 4 | RF | 1.026 | 0.885 | 0.913 |
| 4 | XGB | 0.970 | 0.844 | 0.865 |
| 4 | LightGBM | 0.980 | 0.846 | 0.870 |
| 4 | SARIMAX | 1.040 | 0.903 | 0.925 |
| 4 | Ensemble_unweighted | 0.977 | 0.845 | 0.868 |
| 4 | Ensemble_MASEweighted | 0.977 | 0.845 | 0.868 |
| 4 | TFT | 1.014 | 0.874 | 0.897 |
| 4 | TabPFN | 0.864 | 0.760 | 0.778 |
| 4 | DLinear | 1.626 | 1.364 | 1.418 |
| 4 | LSTM | 1.106 | 0.947 | 0.969 |
| 4 | TabICLv2 | 0.963 | 0.829 | 0.844 |
| 9 | RF | 0.944 | 0.767 | 0.774 |
| 9 | XGB | 0.920 | 0.746 | 0.752 |
| 9 | LightGBM | 0.954 | 0.777 | 0.780 |
| 9 | SARIMAX | 0.920 | 0.813 | 0.777 |
| 9 | Ensemble_unweighted | 0.884 | 0.717 | 0.728 |
| 9 | Ensemble_MASEweighted | 0.885 | 0.718 | 0.728 |
| 9 | TFT | 0.974 | 0.761 | 0.809 |
| 9 | TabPFN | 0.926 | 0.709 | 0.789 |
| 9 | DLinear | 1.251 | 0.995 | 1.024 |
| 9 | LSTM | 1.289 | 1.003 | 1.027 |
| 9 | TabICLv2 | 0.953 | 0.720 | 0.806 |

Source: `results/b10_b13_climatology_mase_table_by_tower.csv`,
`results/b10_b13_climatology_gf_mase_table_by_tower.csv`.

Reading Tower 2's rows against Towers 4/9 makes the reversal visible directly in the models' own
MASE, not just the baselines' raw MAE: at Tower 2, every model's MASE goes *up* under either
climatology variant (a harder bar); at Towers 4/9, every model's MASE goes *down* (an easier bar).

## Gap-filled-basis follow-up: fairness fix

**Concern raised (user, live discussion):** the original `Climatology` column was built from real
`y_observed` history only (B-09's own recipe), while `persistence`'s single anchor value is drawn
from `y_gapfilled` — dense, and can itself be a gap-filler's smoothed model output on days the
anchor wasn't a real observation. So part of climatology's apparent weakness could be "real vs.
gap-filled data source," not "flat vs. seasonal" — not an apples-to-apples comparison.

**Fix:** `notebooks/05_benchmarking/b10_b13_climatology_gf_baseline.py` (new, committed) — identical
`doy_climatology()` recipe, sourced from `y_gapfilled` instead. Produces a second new column,
`Climatology_gf`, alongside (not replacing) the original `Climatology` column. Because
`y_gapfilled` is dense (no real-data gaps), there are **zero NaN rows** in `Climatology_gf` — unlike
the original, which had 730 NaN rows at Tower 9's data-scarce early anchors.

**Result: climatology-gf narrows the gap substantially but doesn't fully close it, pooled — and
reverses at Tower 2.** Pooled MAE: persistence 37.50, climatology-obs 43.79, **climatology-gf
40.74** — roughly halfway between the two. At Tower 2, climatology-gf (MAE 25.86) clearly *beats*
persistence (51.94) — an even stronger seasonal-baseline win than the original `y_observed`-basis
version (37.26) at the same tower. Towers 4/9 still favor persistence under either climatology
variant.

## Findings

**1. The motivating hypothesis does not hold pooled, on either climatology variant.** Contrary to
Hyndman & Koehler's general seasonal-naive recommendation, this project's available seasonal
baseline (`doy_climatology`) is *less* accurate than flat persistence when pooled across all 3
towers — plausible explanation: FCH4's spike-dominated record (D-44b) makes a ±7-day day-of-year
average, built from only a handful of real historical years per tower, a noisy estimate rather than
a stable seasonal curve.

**2. Roughly half of the original gap was a data-source artifact, not a seasonal-vs-flat effect —
fixed, not just noted.** Scoring climatology from the same `y_gapfilled` series persistence draws
its anchor value from closes about half the gap (43.79 → 40.74 pooled MAE). The remaining gap is a
genuine "flat beats seasonal-mean" result on this data, not measurement of two different underlying
series.

**3. Tower 2 is a real, consistent exception — under every variant tested.** Climatology (either
basis) is the *harder*, more accurate baseline at Tower 2, and the gap-filled-basis version is even
stronger there than the original. This is the one tower where the standard "seasonal-naive is
harder to beat" intuition holds.

**4. Model ranking is materially unaffected by the choice of baseline.** Top (TabPFN) and bottom
3 (SARIMAX/LSTM/DLinear) are identical under all three baselines; only mid-pack ordering shifts
(TabICLv2 benefits most from switching off persistence).

**5. Practical implication: this reinforces, rather than undermines, keeping persistence as the
primary MASE denominator (D-37).** Not merely for cross-table consistency with this project's
existing convention, but because the empirically available seasonal alternative isn't more reliable
given how sparse and spike-dominated the real FCH4 record is, pooled — though Tower 2 is a
documented, real exception worth carrying forward if any future work reports per-tower MASE
choices. Climatology-scaled MASE (both variants) is retained as a secondary comparison column, not
a replacement — consistent with this project's habit of adding secondary metrics alongside rather
than instead of the primary one (cf. the `y_gapfilled` secondary-target convention, D-65's
addendum).

## Figures

`results/figures/b09_chains/` (165 figures: 11 models × 3 towers × 5 anchors,
`notebooks/05_benchmarking/b09_chain_plots.py`, new, committed) — actual/gap-filled/predicted chains
with **all three baselines overlaid**: persistence (dashed gray), climatology `y_observed`-basis
(dotted dim gray), climatology `y_gapfilled`-basis (dash-dot slate gray). Separate directory from
`results/figures/b10_chains/` (which doesn't plot any baseline) rather than modifying those figures
in place. Spot-checked: Tower 4/2018/RF visibly shows why climatology loses there (real flux spikes
to 360, climatology smooths to a moderate summer bump); Tower 9/2018/RF confirms both climatology
columns correctly render as blank (no line) for the zero-pre-anchor-history case, persistence still
shows as the flat dashed line.

## Files

- `notebooks/05_benchmarking/b10_b13_climatology_baseline.py` (new, committed) — builds the
  `y_observed`-basis climatology chains, merges into `b10_b13_full_chains.csv`, recomputes MASE
  for all 11 models, builds the pooled/by-tower comparison tables.
- `notebooks/05_benchmarking/b10_b13_climatology_gf_baseline.py` (new, committed) — same recipe,
  `y_gapfilled`-basis; reuses the first script's aggregation helpers directly (no duplication).
- `notebooks/05_benchmarking/b09_chain_plots.py` (new, committed) — the 3-baseline chain figures.
- `results/b10_b13_full_chains.csv` — extended in place, +2 columns (`Climatology`,
  `Climatology_gf`), 5,475 rows, row count verified unchanged at each step; backed up before each
  write.
- `results/b10_b13_climatology_mase_summary.csv` (990 rows, raw per-bin/anchor/tower/model,
  `y_observed`-basis), `results/b10_b13_climatology_gf_mase_summary.csv` (990 rows, `y_gapfilled`-
  basis).
- `results/b10_b13_climatology_mase_table_all_towers.csv` / `_by_tower.csv` (pooled/by-tower
  comparison, `y_observed`-basis), `results/b10_b13_climatology_gf_mase_table_all_towers.csv` /
  `_by_tower.csv` (all three baselines side by side).
- `results/b10_b13_mase_baseline_comparison.csv` (990 rows, per-bin deep-dive file: `tower,
  anchor_year, model, bin, n, R2, RMSE, MAE, WAPE, Correlation, MASE_persistence,
  MASE_climatology_obs, MASE_climatology_gf`).
- `results/b10_b13_climatology_full_metrics_all_towers.csv`, `_tower2.csv`, `_tower4.csv`,
  `_tower9.csv` (pooled + per-tower full-metric tables: RMSE/MAE/WAPE/Correlation/R² shown once
  each alongside all 3 MASE variants — the "Full metric comparison" tables above).
- `results/figures/b09_chains/` (165 figures).

No `benchmarks.csv` rows (a methodology/baseline-validity check, not a point-forecast or
interval-calibration benchmark in its own right — same exclusion precedent as every other
diagnostic/ablation pass this session). Cross-ref D-37 (the original persistence convention), D-53
(climatology's first, single-anchor appearance), D-65 (the `bin_metrics()`/multi-anchor aggregation
conventions reused here).
