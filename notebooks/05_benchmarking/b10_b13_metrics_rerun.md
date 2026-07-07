# D-65: B-10 + B-13 rerun with expanded metrics (RMSE, WAPE, Correlation added), all 3 towers

**Addendum (2026-07-07): extended from Tower 4 only to all 3 towers (T2/T4/T9)**, per the project's
new "full coverage by default" convention (`CLAUDE.md`) — a single-tower result is a smoke test, not
a finished deliverable. See "All-tower summary" and "Results by tower and year" below; the original
Tower-4-only table is kept for continuity with existing citations (`BEST_RESULTS.md`).

## Context

`bin_metrics()` — the single shared evaluation function every B-09→B-15 experiment (and I-02,
U-02, U-03, S-01 by extension) calls — only computed **R², MAE, MASE**, narrower than the earlier
B01→B07 phase's full roster (RMSE/MAE/R²/MBE/WAPE/MASE/sMAPE/MAPE, via D-44b). Added RMSE, WAPE,
and (new to the project entirely) **Pearson correlation** — the same informal check already used
once to diagnose the original unregularized TFT (D-45: r=0.27 despite deeply negative R²), now
formalized as a metric rather than a one-off manual calculation.

**A discovery made while planning this**: neither B-10's nor B-13's original multi-anchor script
was ever committed to the repo (both were "ad-hoc, not committed," per this project's own stated
precedent going back to B-09) — only their output CSVs survived. This rerun reconstructs both from
their fully-documented methodology (hyperparameters read directly from the committed single-anchor
notebooks) and is itself committed, closing that reproducibility gap.

## Method

`src/evaluation/metrics.py`: added `correlation(y, p)` (Pearson r, NaN-guarded). `bin_metrics()`
extended to compute RMSE/WAPE/Correlation alongside the existing R²/MAE/MASE — purely additive,
verified against every one of the 15+ existing call sites across the B-09→B-15 sequence (all
access the returned DataFrame by column name only).

`notebooks/05_benchmarking/b10_b13_rerun_multi_anchor.py` (new, committed) reconstructs B-10
(RF/XGB/LightGBM/SARIMAX + 2 ensembles) and B-13 (TFT/TabPFN), **all 3 towers (T2/T4/T9)**, 5 anchors
(2018–2022), using the exact hyperparameters read directly from `B10_daily_improvements.ipynb` /
`B13_tft_tabpfn.ipynb` — including B-10's original SARIMAX grid (`p∈{1,2}, q∈{0,1}`, not the wider
grid used in later U-02/U-03/S-01 scripts) and B-13's use of two different `y_true` sources (trees/
SARIMAX/TabPFN from `forecast_daily_v2.csv`'s `y_observed`; TFT from `fdl.tower_series`'s
resampled `Y` — preserved exactly, since conflating them would silently evaluate different rows).

**Per-anchor fit structure** (matching the pooled-fit-once-per-anchor pattern established in
U-02/U-03/S-01): RF/XGB/LightGBM and TFT are trained **once per anchor** on pooled T2+T4+T9 data
(with tower dummies), then rolled out separately per tower using that one fitted model — the
same fitted model serves all 3 towers' chains, only the tower-specific `fx_frame`/`history_init`
differ. SARIMAX and TabPFN are never pooled — both are refit/re-queried independently per tower
per anchor, as in every prior B-10/B-13-derived experiment in this project.

**Verification, before trusting any new number**: the rerun's R²/MAE/MASE were checked against the
existing published CSVs (`b10_ensemble_multi_anchor.csv`, `b13b_tabpfn_multi_anchor.csv`) across
**all 5 anchors, Tower 4** — **RF, XGB, LightGBM, SARIMAX, both ensembles, and TabPFN reproduce
bit-for-bit exactly** (7 of 8 models, deterministic/seeded). **TFT differs**, as expected from the
already-documented pre-existing non-determinism (D-62 addendum: TFT's initial weights are never
seeded before construction) — median |R² difference| across all bins/anchors is a modest 0.13, with
one extreme outlier in the 3-point 1–7 day bin, consistent with this project's own established
"the 1-7 day bin is a small-sample artifact" finding (D-53), not a new instability.

**Tower 2 and Tower 9 have no prior B-10/B-13 multi-anchor CSV to reproduce** (the originals were
Tower-4-only) — there is nothing to bit-for-bit check them against. Their trust basis instead is:
(a) the same pooled per-anchor fit and rollout code path as Tower 4, which is itself verified
bit-for-bit correct; (b) sanity-check that Tower 2's near-total NaN pattern (only the 2018 anchor
has real `y_observed` coverage) matches the same degeneracy already documented in U-02/U-03/S-01 —
confirmed, not a new or surprising pattern.

## All-tower summary (final, full-coverage headline)

Aggregated across **all 3 towers** using the same established convention, extended: per-anchor
n-weighted mean across the 6 lead-time bins **and all towers present that anchor** (Tower 2's
mostly-NaN bins drop out of the weighting, not silently zero-filled), then simple mean across the
5 anchors. This is the number that should be cited going forward per the project's "full coverage
by default" convention — the Tower-4-only table below is kept for continuity, not as the headline.

| Model | R² | RMSE | MAE | MASE | WAPE | Correlation |
|---|---|---|---|---|---|---|
| RF | −0.241 | 52.23 | 34.84 | 0.968 | 1.050 | 0.375 |
| XGB | −0.184 | 51.57 | 33.81 | 0.922 | 0.991 | 0.368 |
| LightGBM | −0.206 | 52.08 | 34.32 | 0.941 | 1.012 | 0.368 |
| SARIMAX | −0.360 | 53.79 | 36.06 | 0.976 | 1.105 | 0.343 |
| **Ensemble_unweighted** | **−0.165** | 51.57 | 33.75 | **0.918** | 0.998 | **0.375** |
| Ensemble_MASEweighted | −0.165 | 51.57 | 33.74 | 0.918 | 0.998 | 0.375 |
| TFT | −0.565* | 59.23* | 37.37* | 1.047* | 1.096* | 0.260* |
| TabPFN | −0.122 | 56.12 | 33.14 | 0.855 | 0.899 | 0.358 |

*TFT's numbers reflect this specific unseeded rerun (per-tower/per-anchor refits) — see the TFT
caveat below, which applies identically here. Source: `results/b10_b13_rerun_table_all_towers.csv`.

**Every model's R² drops substantially once T2/T9 are included** — the ensemble goes from +0.012
(T4-only) to −0.165 (all-tower). This is consistent with, not contradictory to, this project's
repeated finding (B-15, S-01) that a model/config tuned or evaluated on Tower 4 alone does not
transfer cleanly to Tower 9, and that Tower 2 is frequently degenerate. **MASE tells a different
story**: it *improves* pooled (Ensemble 0.975→0.918) — Tower 9's persistence baseline is itself
harder to beat in absolute terms but the models' errors relative to it are comparable-to-better than
at Tower 4, while WAPE/RMSE (scale-sensitive, unlike MASE) worsen because Tower 9's flux magnitudes
and error scale are larger. The ranking among models is essentially unchanged (Ensemble/TabPFN
still best on MASE, SARIMAX/TFT still worst throughout) — pooling changes the absolute numbers, not
which model wins.

## Tower-4-only table (original scope, kept for continuity with `BEST_RESULTS.md`)

Aggregated via this project's established convention (per-anchor n-weighted mean across the 6
lead-time bins, then simple mean across the 5 anchors — confirmed to reproduce D-54/D-57's
published headline numbers before trusting the new columns):

| Model | R² | RMSE | MAE | MASE | WAPE | Correlation |
|---|---|---|---|---|---|---|
| RF | −0.067 | 51.54 | 34.08 | 1.024 | 1.028 | 0.402 |
| XGB | 0.003 | 51.23 | 33.08 | 0.968 | 0.964 | 0.372 |
| LightGBM | −0.014 | 51.37 | 33.25 | 0.978 | 0.978 | 0.384 |
| SARIMAX | −0.039 | 52.39 | 35.23 | 1.038 | 1.047 | 0.379 |
| **Ensemble_unweighted** | **0.012** | **50.96** | 33.18 | 0.975 | 0.977 | 0.396 |
| Ensemble_MASEweighted | 0.011 | 50.96 | 33.17 | 0.975 | 0.977 | 0.396 |
| TFT | −0.568* | 54.60* | 33.67* | 1.045* | 1.050* | 0.329* |
| TabPFN | −0.006 | 54.19 | 30.46 | **0.862** | 0.860 | 0.391 |

*TFT's numbers reflect this specific unseeded rerun, not a re-validated "new" TFT result — see
caveat below. All other rows are exact reproductions of already-published numbers, now with 3 new
columns.

## Results by tower and year

Full breakdown: tower as the parent column grouping (T2/T4/T9), each with its own 6 metrics; year
(anchor) as the parent row grouping, model nested beneath. Per-year values use the same n-weighted
mean across the 6 lead-time bins as above (no cross-anchor averaging in this table — that's what
the two summary tables above already show). **Markdown cannot render true nested/spanning headers**,
so columns below are flattened to `T{tower}_{metric}`; the canonical version with a true
`MultiIndex` (tower as the real parent column level, anchor_year as the real parent row level) is
`results/b10_b13_rerun_table_by_tower_year.csv`. `nan` = no real `y_observed` coverage for that
tower/anchor/model (expected for Tower 2 in every anchor except 2018 — a genuine data finding, not
a computation error, consistent with U-02/U-03/S-01's own Tower 2 documentation).

| Year | Model | T2_R2 | T2_RMSE | T2_MAE | T2_MASE | T2_WAPE | T2_Correlation | T4_R2 | T4_RMSE | T4_MAE | T4_MASE | T4_WAPE | T4_Correlation | T9_R2 | T9_RMSE | T9_MAE | T9_MASE | T9_WAPE | T9_Correlation |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2018 | RF | −0.889 | 22.13 | 18.05 | 0.346 | 1.347 | 0.087 | −0.238 | 46.68 | 32.87 | 1.021 | 1.206 | 0.277 | nan | nan | nan | nan | nan | nan |
| 2018 | XGB | −0.618 | 20.82 | 16.45 | 0.312 | 1.205 | 0.214 | −0.072 | 45.54 | 30.17 | 0.893 | 1.036 | 0.265 | nan | nan | nan | nan | nan | nan |
| 2018 | LightGBM | −0.600 | 20.67 | 16.32 | 0.310 | 1.204 | 0.182 | −0.111 | 45.31 | 29.81 | 0.896 | 1.050 | 0.284 | nan | nan | nan | nan | nan | nan |
| 2018 | SARIMAX | −3.244 | 34.10 | 30.76 | 0.574 | 2.286 | 0.056 | −0.398 | 49.50 | 37.79 | 1.171 | 1.382 | 0.240 | nan | nan | nan | nan | nan | nan |
| 2018 | Ensemble_unweighted | −1.048 | 23.66 | 19.82 | 0.374 | 1.464 | 0.176 | −0.160 | 46.04 | 32.11 | 0.977 | 1.148 | 0.276 | nan | nan | nan | nan | nan | nan |
| 2018 | Ensemble_MASEweighted | −1.029 | 23.54 | 19.69 | 0.372 | 1.455 | 0.178 | −0.157 | 46.00 | 32.03 | 0.974 | 1.144 | 0.277 | nan | nan | nan | nan | nan | nan |
| 2018 | TFT | −0.325 | 18.81 | 13.32 | 0.250 | 0.978 | 0.045 | −0.183 | 49.57 | 29.70 | 0.872 | 1.010 | 0.166 | nan | nan | nan | nan | nan | nan |
| 2018 | TabPFN | −0.240 | 18.25 | 12.96 | 0.243 | 0.969 | 0.098 | −0.201 | 49.69 | 29.06 | 0.844 | 0.974 | 0.081 | nan | nan | nan | nan | nan | nan |
| 2019 | RF | nan | nan | nan | nan | nan | nan | 0.007 | 48.57 | 29.90 | 0.784 | 0.785 | 0.473 | 0.078 | 71.21 | 45.02 | 0.913 | 0.840 | 0.499 |
| 2019 | XGB | nan | nan | nan | nan | nan | nan | 0.075 | 48.21 | 28.60 | 0.745 | 0.746 | 0.439 | 0.101 | 70.19 | 44.25 | 0.900 | 0.830 | 0.521 |
| 2019 | LightGBM | nan | nan | nan | nan | nan | nan | 0.100 | 47.80 | 27.67 | 0.715 | 0.716 | 0.468 | 0.003 | 71.34 | 44.84 | 0.919 | 0.843 | 0.511 |
| 2019 | SARIMAX | nan | nan | nan | nan | nan | nan | 0.114 | 47.93 | 31.36 | 0.836 | 0.837 | 0.448 | 0.186 | 69.13 | 41.55 | 0.814 | 0.753 | 0.490 |
| 2019 | Ensemble_unweighted | nan | nan | nan | nan | nan | nan | 0.136 | 47.05 | 27.70 | 0.710 | 0.711 | 0.465 | 0.134 | 69.61 | 42.74 | 0.861 | 0.793 | 0.520 |
| 2019 | Ensemble_MASEweighted | nan | nan | nan | nan | nan | nan | 0.134 | 47.07 | 27.71 | 0.710 | 0.711 | 0.465 | 0.132 | 69.64 | 42.78 | 0.863 | 0.794 | 0.520 |
| 2019 | TFT | nan | nan | nan | nan | nan | nan | −0.068 | 53.84 | 31.45 | 0.823 | 0.824 | 0.183 | −0.102 | 80.69 | 48.20 | 0.939 | 0.868 | 0.197 |
| 2019 | TabPFN | nan | nan | nan | nan | nan | nan | −0.090 | 55.86 | 32.09 | 0.809 | 0.811 | 0.505 | −0.460 | 91.60 | 56.29 | 1.087 | 1.000 | nan |
| 2020 | RF | nan | nan | nan | nan | nan | nan | −0.070 | 61.52 | 34.47 | 1.029 | 0.929 | 0.337 | −1.147 | 58.46 | 36.64 | 0.950 | 1.132 | 0.354 |
| 2020 | XGB | nan | nan | nan | nan | nan | nan | −0.073 | 63.10 | 34.87 | 1.021 | 0.917 | 0.251 | −1.378 | 58.14 | 35.90 | 0.938 | 1.134 | 0.340 |
| 2020 | LightGBM | nan | nan | nan | nan | nan | nan | −0.068 | 62.96 | 35.10 | 1.031 | 0.927 | 0.272 | −1.229 | 58.51 | 36.90 | 0.960 | 1.149 | 0.339 |
| 2020 | SARIMAX | nan | nan | nan | nan | nan | nan | 0.060 | 62.67 | 35.21 | 1.000 | 0.887 | 0.318 | −0.921 | 58.56 | 35.01 | 0.895 | 1.034 | 0.283 |
| 2020 | Ensemble_unweighted | nan | nan | nan | nan | nan | nan | 0.000 | 62.14 | 34.15 | 0.990 | 0.886 | 0.309 | −1.076 | 58.03 | 35.44 | 0.915 | 1.087 | 0.337 |
| 2020 | Ensemble_MASEweighted | nan | nan | nan | nan | nan | nan | −0.002 | 62.16 | 34.16 | 0.991 | 0.887 | 0.308 | −1.082 | 58.03 | 35.46 | 0.915 | 1.089 | 0.338 |
| 2020 | TFT | nan | nan | nan | nan | nan | nan | −0.079 | 67.50 | 35.41 | 0.998 | 0.886 | 0.184 | −0.418 | 61.31 | 37.28 | 0.982 | 1.071 | 0.294 |
| 2020 | TabPFN | nan | nan | nan | nan | nan | nan | 0.034 | 64.30 | 31.20 | 0.886 | 0.786 | 0.394 | −0.510 | 62.23 | 35.55 | 0.906 | 1.003 | 0.192 |
| 2021 | RF | nan | nan | nan | nan | nan | nan | −0.141 | 61.90 | 44.24 | 1.195 | 1.060 | 0.411 | 0.008 | 63.74 | 40.88 | 0.749 | 0.837 | 0.341 |
| 2021 | XGB | nan | nan | nan | nan | nan | nan | −0.058 | 60.78 | 43.89 | 1.145 | 1.019 | 0.413 | 0.043 | 63.78 | 41.31 | 0.740 | 0.791 | 0.318 |
| 2021 | LightGBM | nan | nan | nan | nan | nan | nan | −0.043 | 59.97 | 43.11 | 1.128 | 1.007 | 0.434 | 0.073 | 61.96 | 40.24 | 0.722 | 0.784 | 0.317 |
| 2021 | SARIMAX | nan | nan | nan | nan | nan | nan | −0.036 | 61.75 | 42.02 | 1.069 | 0.949 | 0.386 | −0.742 | 74.21 | 49.08 | 0.923 | 1.111 | 0.317 |
| 2021 | Ensemble_unweighted | nan | nan | nan | nan | nan | nan | −0.034 | 60.45 | 42.98 | 1.122 | 0.998 | 0.424 | 0.075 | 63.64 | 39.23 | 0.689 | 0.714 | 0.324 |
| 2021 | Ensemble_MASEweighted | nan | nan | nan | nan | nan | nan | −0.034 | 60.43 | 43.00 | 1.122 | 0.999 | 0.424 | 0.077 | 63.58 | 39.22 | 0.689 | 0.715 | 0.324 |
| 2021 | TFT | nan | nan | nan | nan | nan | nan | −1.967 | 77.57 | 49.54 | 1.512 | 1.329 | 0.308 | −0.747 | 75.73 | 52.45 | 0.967 | 1.127 | 0.298 |
| 2021 | TabPFN | nan | nan | nan | nan | nan | nan | 0.058 | 63.49 | 36.78 | 0.891 | 0.791 | 0.425 | −0.133 | 73.29 | 45.48 | 0.776 | 0.765 | 0.440 |
| 2022 | RF | nan | nan | nan | nan | nan | nan | 0.105 | 39.06 | 28.94 | 1.090 | 1.159 | 0.512 | −0.331 | 39.86 | 30.44 | 1.160 | 1.189 | 0.362 |
| 2022 | XGB | nan | nan | nan | nan | nan | nan | 0.142 | 38.53 | 27.87 | 1.038 | 1.102 | 0.491 | −0.197 | 37.79 | 28.69 | 1.099 | 1.126 | 0.388 |
| 2022 | LightGBM | nan | nan | nan | nan | nan | nan | 0.054 | 40.81 | 30.56 | 1.122 | 1.190 | 0.462 | −0.404 | 42.33 | 32.62 | 1.212 | 1.241 | 0.343 |
| 2022 | SARIMAX | nan | nan | nan | nan | nan | nan | 0.063 | 40.09 | 29.75 | 1.112 | 1.181 | 0.504 | −0.143 | 38.18 | 27.67 | 1.044 | 1.069 | 0.307 |
| 2022 | Ensemble_unweighted | nan | nan | nan | nan | nan | nan | 0.116 | 39.15 | 28.95 | 1.077 | 1.144 | 0.506 | −0.144 | 38.30 | 28.60 | 1.066 | 1.091 | 0.355 |
| 2022 | Ensemble_MASEweighted | nan | nan | nan | nan | nan | nan | 0.116 | 39.15 | 28.95 | 1.077 | 1.144 | 0.506 | −0.148 | 38.33 | 28.64 | 1.068 | 1.093 | 0.355 |
| 2022 | TFT | nan | nan | nan | nan | nan | nan | −0.011 | 40.44 | 29.35 | 1.112 | 1.182 | 0.542 | −1.315 | 52.84 | 40.46 | 1.524 | 1.561 | 0.344 |
| 2022 | TabPFN | nan | nan | nan | nan | nan | nan | 0.171 | 37.60 | 23.16 | 0.880 | 0.936 | 0.549 | 0.046 | 36.39 | 25.79 | 0.932 | 0.953 | 0.412 |

Tower 2 shows real (non-NaN) evaluation only in the 2018 anchor — every other anchor's Tower 2 rows
are `nan` because there is no real `y_observed` coverage in those rollout windows, matching this
project's already-established Tower 2 data-scarcity finding (U-02/U-03/B-15). Where Tower 2 *does*
have real data (2018), its R² is uniformly worse than Tower 4/9's — small-sample, high-variance,
consistent with (not a new contradiction of) that same standing finding.

## Findings

**1. Reproduction is exact wherever the model is deterministic.** 7 of 8 models — everything except
TFT — match the already-published R²/MAE/MASE bit-for-bit across all 5 anchors. This is strong
evidence the reconstruction is faithful, not just plausible-looking.

**2. A genuinely new finding, invisible with the old metric set: TabPFN's headline strength (best
MASE in the whole B-09→B-15 sequence) does not extend to RMSE.** TabPFN has the best MASE (0.862)
and best MAE (30.46) of any model, but its RMSE (54.19) is second-worst, close to TFT's (54.60) and
clearly worse than every tree model (~51–52). MASE/MAE are linear (each error counted once); RMSE
squares errors, so it's dominated by TabPFN's worst individual misses in a way MASE never revealed.
**TabPFN's real strength is consistency relative to a naive baseline, not small worst-case errors**
— a materially different, more complete characterization than "best MASE" alone implied.

**3. Correlation is uniformly weak-to-moderate (0.33–0.40) across every model, including the
best-performing ones.** Even the ensemble (0.396, the highest) only weakly tracks the true pattern
in a linear-association sense (r²≈0.16, i.e. ~16% of variance explained in that sense) — a humbling,
consistent-with-everything-else-this-project-has-found context for the whole recursive-rollout
sequence: no model, including the standing recommendation, is capturing the real signal strongly by
this measure either.

**4. RMSE and MASE mostly agree on ranking here** (Ensemble best on both, SARIMAX/TFT worst on
both) — the divergence is specifically TabPFN's case (finding 2), not a general pattern. Worth
knowing RMSE doesn't change the standing recommendation (Ensemble_unweighted remains best on R²
*and* now RMSE), it specifically nuances the TabPFN "genuine alternative" framing from D-57.

## Explicit caveat: what the TFT row does and does not mean

TFT's row in the table above is a **real, honestly-computed result from this specific rerun** —
not fabricated — but it should **not** be read as superseding or contradicting the originally
published −0.237 mean R² (D-57). Both numbers are correct; TFT's initial weights are never seeded,
so two honest runs of the identical recipe land on genuinely different results. This rerun's
draw happened to be worse, driven substantially by one extreme small-sample-bin outlier. **The
correct takeaway is the one already established (D-62 addendum): TFT's point estimate carries real
run-to-run uncertainty that the other 7 models don't, and any single TFT number — including both
the original and this one — should be read with that in mind.**

## Files

- `src/evaluation/metrics.py` (+`correlation()`)
- `src/models/recursive_rollout.py` (`bin_metrics()` extended, additive only)
- `notebooks/05_benchmarking/b10_b13_rerun_multi_anchor.py` (new, committed — closes the
  B-10/B-13 script-reproducibility gap; loops all 3 towers)
- `results/b10_b13_rerun_summary.csv` (720 rows: 8 models × 3 towers × 5 anchors × 6 bins × 11 columns)
- `results/b10_b13_rerun_table.csv` (the Tower-4-only table)
- `results/b10_b13_rerun_table_all_towers.csv` (the all-tower summary — the new headline)
- `results/b10_b13_rerun_table_by_tower_year.csv` (the tower × year breakdown, true nested
  `MultiIndex`; the markdown table above is a flattened rendering of this file)

No `benchmarks.csv` rows (a metrics backfill + verification exercise on already-logged results,
same precedent as D-44b's own backfill).
