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

| Model | RMSE | MAE | MASE | WAPE | Correlation | R² |
|---|---|---|---|---|---|---|
| RF | 52.23 | 34.84 | 0.968 | 1.050 | 0.375 | −0.241 |
| XGB | 51.57 | 33.81 | 0.922 | 0.991 | 0.368 | −0.184 |
| LightGBM | 52.08 | 34.32 | 0.941 | 1.012 | 0.368 | −0.206 |
| SARIMAX | 53.79 | 36.06 | 0.976 | 1.105 | 0.343 | −0.360 |
| **Ensemble_unweighted** | 51.57 | 33.75 | **0.918** | 0.998 | **0.375** | **−0.165** |
| Ensemble_MASEweighted | 51.57 | 33.74 | 0.918 | 0.998 | 0.375 | −0.165 |
| TFT | 59.23* | 37.37* | 1.047* | 1.096* | 0.260* | −0.565* |
| TabPFN | 56.12 | 33.14 | 0.855 | 0.899 | 0.358 | −0.122 |

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

| Model | RMSE | MAE | MASE | WAPE | Correlation | R² |
|---|---|---|---|---|---|---|
| RF | 51.54 | 34.08 | 1.024 | 1.028 | 0.402 | −0.067 |
| XGB | 51.23 | 33.08 | 0.968 | 0.964 | 0.372 | 0.003 |
| LightGBM | 51.37 | 33.25 | 0.978 | 0.978 | 0.384 | −0.014 |
| SARIMAX | 52.39 | 35.23 | 1.038 | 1.047 | 0.379 | −0.039 |
| **Ensemble_unweighted** | **50.96** | 33.18 | 0.975 | 0.977 | 0.396 | **0.012** |
| Ensemble_MASEweighted | 50.96 | 33.17 | 0.975 | 0.977 | 0.396 | 0.011 |
| TFT | 54.60* | 33.67* | 1.045* | 1.050* | 0.329* | −0.568* |
| TabPFN | 54.19 | 30.46 | **0.862** | 0.860 | 0.391 | −0.006 |

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

| Year | Model | T2_RMSE | T2_MAE | T2_MASE | T2_WAPE | T2_Correlation | T4_RMSE | T4_MAE | T4_MASE | T4_WAPE | T4_Correlation | T9_RMSE | T9_MAE | T9_MASE | T9_WAPE | T9_Correlation | T2_R2 | T4_R2 | T9_R2 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2018 | RF | 22.13 | 18.05 | 0.346 | 1.347 | 0.087 | 46.68 | 32.87 | 1.021 | 1.206 | 0.277 | nan | nan | nan | nan | nan | −0.889 | −0.238 | nan |
| 2018 | XGB | 20.82 | 16.45 | 0.312 | 1.205 | 0.214 | 45.54 | 30.17 | 0.893 | 1.036 | 0.265 | nan | nan | nan | nan | nan | −0.618 | −0.072 | nan |
| 2018 | LightGBM | 20.67 | 16.32 | 0.310 | 1.204 | 0.182 | 45.31 | 29.81 | 0.896 | 1.050 | 0.284 | nan | nan | nan | nan | nan | −0.600 | −0.111 | nan |
| 2018 | SARIMAX | 34.10 | 30.76 | 0.574 | 2.286 | 0.056 | 49.50 | 37.79 | 1.171 | 1.382 | 0.240 | nan | nan | nan | nan | nan | −3.244 | −0.398 | nan |
| 2018 | Ensemble_unweighted | 23.66 | 19.82 | 0.374 | 1.464 | 0.176 | 46.04 | 32.11 | 0.977 | 1.148 | 0.276 | nan | nan | nan | nan | nan | −1.048 | −0.160 | nan |
| 2018 | Ensemble_MASEweighted | 23.54 | 19.69 | 0.372 | 1.455 | 0.178 | 46.00 | 32.03 | 0.974 | 1.144 | 0.277 | nan | nan | nan | nan | nan | −1.029 | −0.157 | nan |
| 2018 | TFT | 18.81 | 13.32 | 0.250 | 0.978 | 0.045 | 49.57 | 29.70 | 0.872 | 1.010 | 0.166 | nan | nan | nan | nan | nan | −0.325 | −0.183 | nan |
| 2018 | TabPFN | 18.25 | 12.96 | 0.243 | 0.969 | 0.098 | 49.69 | 29.06 | 0.844 | 0.974 | 0.081 | nan | nan | nan | nan | nan | −0.240 | −0.201 | nan |
| 2019 | RF | nan | nan | nan | nan | nan | 48.57 | 29.90 | 0.784 | 0.785 | 0.473 | 71.21 | 45.02 | 0.913 | 0.840 | 0.499 | nan | 0.007 | 0.078 |
| 2019 | XGB | nan | nan | nan | nan | nan | 48.21 | 28.60 | 0.745 | 0.746 | 0.439 | 70.19 | 44.25 | 0.900 | 0.830 | 0.521 | nan | 0.075 | 0.101 |
| 2019 | LightGBM | nan | nan | nan | nan | nan | 47.80 | 27.67 | 0.715 | 0.716 | 0.468 | 71.34 | 44.84 | 0.919 | 0.843 | 0.511 | nan | 0.100 | 0.003 |
| 2019 | SARIMAX | nan | nan | nan | nan | nan | 47.93 | 31.36 | 0.836 | 0.837 | 0.448 | 69.13 | 41.55 | 0.814 | 0.753 | 0.490 | nan | 0.114 | 0.186 |
| 2019 | Ensemble_unweighted | nan | nan | nan | nan | nan | 47.05 | 27.70 | 0.710 | 0.711 | 0.465 | 69.61 | 42.74 | 0.861 | 0.793 | 0.520 | nan | 0.136 | 0.134 |
| 2019 | Ensemble_MASEweighted | nan | nan | nan | nan | nan | 47.07 | 27.71 | 0.710 | 0.711 | 0.465 | 69.64 | 42.78 | 0.863 | 0.794 | 0.520 | nan | 0.134 | 0.132 |
| 2019 | TFT | nan | nan | nan | nan | nan | 53.84 | 31.45 | 0.823 | 0.824 | 0.183 | 80.69 | 48.20 | 0.939 | 0.868 | 0.197 | nan | −0.068 | −0.102 |
| 2019 | TabPFN | nan | nan | nan | nan | nan | 55.86 | 32.09 | 0.809 | 0.811 | 0.505 | 91.60 | 56.29 | 1.087 | 1.000 | nan | nan | −0.090 | −0.460 |
| 2020 | RF | nan | nan | nan | nan | nan | 61.52 | 34.47 | 1.029 | 0.929 | 0.337 | 58.46 | 36.64 | 0.950 | 1.132 | 0.354 | nan | −0.070 | −1.147 |
| 2020 | XGB | nan | nan | nan | nan | nan | 63.10 | 34.87 | 1.021 | 0.917 | 0.251 | 58.14 | 35.90 | 0.938 | 1.134 | 0.340 | nan | −0.073 | −1.378 |
| 2020 | LightGBM | nan | nan | nan | nan | nan | 62.96 | 35.10 | 1.031 | 0.927 | 0.272 | 58.51 | 36.90 | 0.960 | 1.149 | 0.339 | nan | −0.068 | −1.229 |
| 2020 | SARIMAX | nan | nan | nan | nan | nan | 62.67 | 35.21 | 1.000 | 0.887 | 0.318 | 58.56 | 35.01 | 0.895 | 1.034 | 0.283 | nan | 0.060 | −0.921 |
| 2020 | Ensemble_unweighted | nan | nan | nan | nan | nan | 62.14 | 34.15 | 0.990 | 0.886 | 0.309 | 58.03 | 35.44 | 0.915 | 1.087 | 0.337 | nan | 0.000 | −1.076 |
| 2020 | Ensemble_MASEweighted | nan | nan | nan | nan | nan | 62.16 | 34.16 | 0.991 | 0.887 | 0.308 | 58.03 | 35.46 | 0.915 | 1.089 | 0.338 | nan | −0.002 | −1.082 |
| 2020 | TFT | nan | nan | nan | nan | nan | 67.50 | 35.41 | 0.998 | 0.886 | 0.184 | 61.31 | 37.28 | 0.982 | 1.071 | 0.294 | nan | −0.079 | −0.418 |
| 2020 | TabPFN | nan | nan | nan | nan | nan | 64.30 | 31.20 | 0.886 | 0.786 | 0.394 | 62.23 | 35.55 | 0.906 | 1.003 | 0.192 | nan | 0.034 | −0.510 |
| 2021 | RF | nan | nan | nan | nan | nan | 61.90 | 44.24 | 1.195 | 1.060 | 0.411 | 63.74 | 40.88 | 0.749 | 0.837 | 0.341 | nan | −0.141 | 0.008 |
| 2021 | XGB | nan | nan | nan | nan | nan | 60.78 | 43.89 | 1.145 | 1.019 | 0.413 | 63.78 | 41.31 | 0.740 | 0.791 | 0.318 | nan | −0.058 | 0.043 |
| 2021 | LightGBM | nan | nan | nan | nan | nan | 59.97 | 43.11 | 1.128 | 1.007 | 0.434 | 61.96 | 40.24 | 0.722 | 0.784 | 0.317 | nan | −0.043 | 0.073 |
| 2021 | SARIMAX | nan | nan | nan | nan | nan | 61.75 | 42.02 | 1.069 | 0.949 | 0.386 | 74.21 | 49.08 | 0.923 | 1.111 | 0.317 | nan | −0.036 | −0.742 |
| 2021 | Ensemble_unweighted | nan | nan | nan | nan | nan | 60.45 | 42.98 | 1.122 | 0.998 | 0.424 | 63.64 | 39.23 | 0.689 | 0.714 | 0.324 | nan | −0.034 | 0.075 |
| 2021 | Ensemble_MASEweighted | nan | nan | nan | nan | nan | 60.43 | 43.00 | 1.122 | 0.999 | 0.424 | 63.58 | 39.22 | 0.689 | 0.715 | 0.324 | nan | −0.034 | 0.077 |
| 2021 | TFT | nan | nan | nan | nan | nan | 77.57 | 49.54 | 1.512 | 1.329 | 0.308 | 75.73 | 52.45 | 0.967 | 1.127 | 0.298 | nan | −1.967 | −0.747 |
| 2021 | TabPFN | nan | nan | nan | nan | nan | 63.49 | 36.78 | 0.891 | 0.791 | 0.425 | 73.29 | 45.48 | 0.776 | 0.765 | 0.440 | nan | 0.058 | −0.133 |
| 2022 | RF | nan | nan | nan | nan | nan | 39.06 | 28.94 | 1.090 | 1.159 | 0.512 | 39.86 | 30.44 | 1.160 | 1.189 | 0.362 | nan | 0.105 | −0.331 |
| 2022 | XGB | nan | nan | nan | nan | nan | 38.53 | 27.87 | 1.038 | 1.102 | 0.491 | 37.79 | 28.69 | 1.099 | 1.126 | 0.388 | nan | 0.142 | −0.197 |
| 2022 | LightGBM | nan | nan | nan | nan | nan | 40.81 | 30.56 | 1.122 | 1.190 | 0.462 | 42.33 | 32.62 | 1.212 | 1.241 | 0.343 | nan | 0.054 | −0.404 |
| 2022 | SARIMAX | nan | nan | nan | nan | nan | 40.09 | 29.75 | 1.112 | 1.181 | 0.504 | 38.18 | 27.67 | 1.044 | 1.069 | 0.307 | nan | 0.063 | −0.143 |
| 2022 | Ensemble_unweighted | nan | nan | nan | nan | nan | 39.15 | 28.95 | 1.077 | 1.144 | 0.506 | 38.30 | 28.60 | 1.066 | 1.091 | 0.355 | nan | 0.116 | −0.144 |
| 2022 | Ensemble_MASEweighted | nan | nan | nan | nan | nan | 39.15 | 28.95 | 1.077 | 1.144 | 0.506 | 38.33 | 28.64 | 1.068 | 1.093 | 0.355 | nan | 0.116 | −0.148 |
| 2022 | TFT | nan | nan | nan | nan | nan | 40.44 | 29.35 | 1.112 | 1.182 | 0.542 | 52.84 | 40.46 | 1.524 | 1.561 | 0.344 | nan | −0.011 | −1.315 |
| 2022 | TabPFN | nan | nan | nan | nan | nan | 37.60 | 23.16 | 0.880 | 0.936 | 0.549 | 36.39 | 25.79 | 0.932 | 0.953 | 0.412 | nan | 0.171 | 0.046 |

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

## Secondary metric: scored against gap-filled target (exploratory)

**Caveat, stated before any numbers below.** Every result in this section scores forecasts against
`y_gapfilled` instead of `y_observed` — an explicit, bounded departure from `DECISIONS.md` D-36/D-37's
standing convention ("train on gap-filled, evaluate on observed") for this one secondary,
exploratory check, not a redefinition of that convention. There is a real circularity risk:
`y_gapfilled` seeds `history_init` (the pre-anchor AR memory every rollout builds forward from) and is
itself the output of a pooled RandomForest gap-filler (RFm) trained on met/soil features that
substantially overlap RF/XGB/LightGBM's own forecast features. **Agreement with `y_gapfilled` can
therefore partly reflect "the forecaster resembles the gap-filler," not real skill against reality.**
Read every number below directionally (does Tower 2 stop being unusable, does model ranking hold),
not as a validated accuracy claim on par with the observed-target tables above.

**Motivation and coverage unlocked.** Real `y_observed` is sparse — Tower 2 especially (816 real
data-points summed across every model/anchor/bin vs. 14,600 possible; Tower 4 10,560; Tower 9 7,200).
`y_gapfilled` is dense (14,600/14,600 at every tower — no gaps at all across the 5-anchor × 365-day
rollout window), so scoring against it removes the coverage problem entirely, at the cost of scoring
against an imperfect, partly-imputed target instead of reality:

| Tower | Real-observed *n* (summed) | Gap-filled *n* (summed) | Real-data coverage |
|---|---|---|---|
| T2 | 816 | 14,600 | 5.6% |
| T4 | 10,560 | 14,600 | 72.3% |
| T9 | 7,200 | 14,600 | 49.3% |

Average per-bin `real_frac` (fraction of each bin's days that were also real-observed, the direct
caveat-strength indicator) tracks the same pattern: T2 ≈ 9.3%, T4 ≈ 60.4%, T9 ≈ 40.1% — Tower 2's
gap-filled-target numbers below rest on the least real data of the three, by a wide margin, and
should be trusted least even though they are no longer literally `NaN`.

**All-tower pooled (gap-filled target) vs. the observed-target headline:**

| Model | RMSE (gapfilled) | RMSE (observed) | MASE (gapfilled) | MASE (observed) | Correlation (gapfilled) | Correlation (observed) | R² (gapfilled) | R² (observed) |
|---|---|---|---|---|---|---|---|---|
| RF | 25.85 | 52.23 | 0.800 | 0.968 | 0.529 | 0.375 | −0.593 | −0.241 |
| XGB | 25.72 | 51.57 | 0.761 | 0.922 | 0.483 | 0.368 | −0.502 | −0.184 |
| LightGBM | 26.11 | 52.08 | 0.774 | 0.941 | 0.496 | 0.368 | −0.480 | −0.206 |
| SARIMAX | 29.21 | 53.79 | 0.943 | 0.976 | 0.463 | 0.343 | −1.004 | −0.360 |
| **Ensemble_unweighted** | **25.39** | 51.57 | **0.751** | 0.918 | 0.523 | 0.375 | **−0.189** | **−0.165** |
| Ensemble_MASEweighted | 25.38 | 51.57 | 0.750 | 0.918 | 0.522 | 0.375 | −0.195 | −0.165 |
| TFT | 37.53 | 59.23 | 1.195 | 1.047 | 0.248 | 0.260 | −2.850 | −0.565 |
| TabPFN | 35.19 | 56.12 | 0.949 | 0.855 | 0.212 | 0.358 | −0.689 | −0.122 |
| DLinear | 390.93†† | 70.89 | 12.70†† | 1.637 | 0.153 | 0.241 | −6576.70†† | −5.057 |
| LSTM | 41.84 | 63.22 | 1.331 | 1.151 | 0.168 | 0.212 | −3.769 | −1.357 |

Source: `results/b10_b13_rerun_table_vs_gapfilled_all_towers.csv` +
`results/b10_b13_dl_extension_table_all_towers_vs_gapfilled.csv` (gapfilled) vs.
`results/b10_b13_rerun_table_all_towers.csv` + the DLinear/LSTM section below (observed).

††**DLinear's gap-filled-target R²/RMSE/MASE are dominated by a single catastrophic outlier, not a
representative number — read this as a diagnostic finding, not a headline statistic.** The 2018
anchor's DLinear draw (the known non-deterministic "cold start" case identified in the
Model-roster-extension section below) diverged this run to physically implausible predictions —
MAE up to ~7,545 nmol m⁻² s⁻¹ at Tower 9, when the entire CH4 series only spans roughly −1,559 to
+6,161 — a genuine autoregressive blow-up, not a subtle miss. Scoring that against the low-variance
`y_gapfilled` target (rather than the noisier `y_observed`, which absorbed the same draw far less
dramatically — see the −25.93 Tower-4/2018 cell above) amplifies it further, since R² divides by a
much smaller `SS_tot`. **Excluding the 2018 anchor** (2019–2022 only, 4 anchors): DLinear
R²=−7.035, RMSE=44.31, MASE=1.807, Correlation=0.192 — still clearly the worst model, consistent
with every other DLinear finding in this document, but not a number reflecting one anchor's
outright divergence. This is itself a further illustration of DLinear's instability (D-53/D-54): its
failure mode isn't just "worse fit," it can be a genuine numerical blow-up depending on which random
weight draw a given run happens to land on.

### Per-tower breakdown (same columns as the all-tower table above)

Same aggregation as the all-tower table (per-anchor n-weighted mean across the 6 lead-time bins,
then simple mean across the 5 anchors), computed **within each tower separately** rather than
pooling all 3 towers together first. Source: `results/b10_b13_rerun_table_per_tower_gf_vs_observed.csv`.

**A caveat before the numbers**: these per-tower values do **not** simply average into the all-tower
headline above — the two use a different aggregation order (per-tower-then-anchor-mean here vs.
per-anchor-across-all-towers-then-mean for the headline), and R² is non-linear enough that the two
orders give genuinely different numbers. This matters most for DLinear: its catastrophic 2018-anchor
divergence (discussed above) gets pooled across all 3 towers *within that one anchor* for the
headline table, then averaged with 4 normal anchors — concentrating the outlier into one extreme
per-anchor value before diluting it 5-fold. Here, each tower's own 2018 value is averaged with its
own other 4 anchors independently, which is why DLinear's worst per-tower gapfilled R² (Tower 4,
−10.33) looks nowhere near as extreme as the pooled −6576.7 above — same underlying data, different
(and here, less outlier-dominated) aggregation path. Also note: **TFT's numbers here reflect yet
another independent random draw** (this project's already-documented non-determinism, D-62) — very
close to, but not bit-identical with, the −2.850/−0.565 cited in the all-tower table above (current
draw: −2.82 gapfilled all-tower-pooled if recomputed today) — every other model's numbers are
deterministic and unaffected.

**Tower 2:**

| Model | RMSE (gapfilled) | RMSE (observed) | MASE (gapfilled) | MASE (observed) | Correlation (gapfilled) | Correlation (observed) | R² (gapfilled) | R² (observed) |
|---|---|---|---|---|---|---|---|---|
| RF | 14.50 | 22.13 | 0.689 | 0.346 | 0.459 | 0.087 | −0.108 | −0.889 |
| XGB | 14.77 | 20.82 | 0.638 | 0.312 | 0.372 | 0.214 | −0.153 | −0.618 |
| LightGBM | 14.64 | 20.67 | 0.653 | 0.310 | 0.420 | 0.182 | −0.147 | −0.600 |
| SARIMAX | 17.69 | 34.10 | 0.799 | 0.574 | 0.293 | 0.056 | −1.432 | −3.244 |
| **Ensemble_unweighted** | **14.61** | 23.66 | **0.646** | 0.374 | 0.433 | 0.176 | **−0.185** | **−1.048** |
| Ensemble_MASEweighted | 14.60 | 23.54 | 0.645 | 0.372 | 0.433 | 0.178 | −0.179 | −1.029 |
| TFT | 27.28 | 29.53 | 1.280 | 0.400 | 0.095 | 0.111 | −4.661 | −2.312 |
| TabPFN | 21.55 | 18.25 | 0.958 | 0.243 | −0.151 | 0.098 | −1.115 | −0.240 |
| DLinear | 37.40 | 36.53 | 1.911 | 0.588 | −0.023 | 0.009 | −8.597 | −5.421 |
| LSTM | 27.21 | 33.05 | 1.275 | 0.508 | 0.017 | 0.319 | −4.624 | −3.791 |

**Tower 4:**

| Model | RMSE (gapfilled) | RMSE (observed) | MASE (gapfilled) | MASE (observed) | Correlation (gapfilled) | Correlation (observed) | R² (gapfilled) | R² (observed) |
|---|---|---|---|---|---|---|---|---|
| RF | 32.61 | 51.62 | 0.916 | 1.026 | 0.528 | 0.403 | −1.490 | −0.067 |
| XGB | 32.47 | 51.30 | 0.862 | 0.970 | 0.500 | 0.373 | −1.201 | 0.003 |
| LightGBM | 32.51 | 51.44 | 0.865 | 0.980 | 0.506 | 0.385 | −1.171 | −0.014 |
| SARIMAX | 33.94 | 52.46 | 0.890 | 1.040 | 0.524 | 0.380 | −0.590 | −0.039 |
| **Ensemble_unweighted** | **31.83** | 51.04 | **0.827** | 0.977 | 0.535 | 0.397 | **−0.465** | **0.012** |
| Ensemble_MASEweighted | 31.83 | 51.03 | 0.827 | 0.977 | 0.534 | 0.397 | −0.484 | 0.011 |
| TFT | 37.78 | 54.48 | 1.052 | 1.014 | 0.398 | 0.315 | −1.972 | −0.228 |
| TabPFN | 38.43 | 54.26 | 0.842 | 0.864 | 0.498 | 0.391 | −0.334 | −0.006 |
| DLinear | 51.83 | 65.71 | 1.891 | 1.626 | 0.256 | 0.237 | −10.332 | −2.105 |
| LSTM | 44.95 | 59.83 | 1.140 | 1.106 | 0.300 | 0.228 | −1.311 | −0.439 |

**Tower 9:**

| Model | RMSE (gapfilled) | RMSE (observed) | MASE (gapfilled) | MASE (observed) | Correlation (gapfilled) | Correlation (observed) | R² (gapfilled) | R² (observed) |
|---|---|---|---|---|---|---|---|---|
| RF | 30.44 | 58.41 | 0.793 | 0.944 | 0.601 | 0.389 | −0.180 | −0.348 |
| XGB | 29.91 | 57.56 | 0.782 | 0.920 | 0.576 | 0.392 | −0.152 | −0.358 |
| LightGBM | 31.16 | 58.62 | 0.804 | 0.954 | 0.564 | 0.378 | −0.120 | −0.389 |
| SARIMAX | 36.02 | 60.12 | 1.141 | 0.920 | 0.572 | 0.350 | −0.991 | −0.406 |
| **Ensemble_unweighted** | **29.74** | 57.49 | **0.781** | 0.884 | 0.601 | 0.385 | **0.081** | **−0.253** |
| Ensemble_MASEweighted | 29.72 | 57.49 | 0.779 | 0.885 | 0.600 | 0.385 | 0.079 | −0.255 |
| TFT | 46.96 | 64.87 | 1.230 | 0.974 | 0.255 | 0.304 | −1.819 | −0.209 |
| TabPFN | 45.60 | 65.98 | 1.047 | 0.926 | 0.415 | 0.349 | −0.617 | −0.265 |
| DLinear | 52.31 | 66.59 | 1.753 | 1.251 | 0.215 | 0.287 | −4.753 | −1.073 |
| LSTM | 53.37 | 74.02 | 1.577 | 1.289 | 0.188 | 0.179 | −5.372 | −2.229 |

**Tower 9 is the only tower where the ensemble's gap-filled R² is positive** (+0.081/+0.079) — the
single best cell across every model/tower/target combination in this whole document. This is
consistent with, not contradicting, this project's own earlier finding (D-65, all-tower observed
table) that Tower 9's models track their own persistence baseline comparatively well even though
their absolute error scale is larger. Every tower shows the same broad pattern already established
at the pooled level: Ensemble_unweighted at or near the top of both target columns, SARIMAX/TFT/
DLinear/LSTM clustered at the bottom — full coverage confirms this holds tower-by-tower, not just on
average.

**Finding 1 — RMSE/MAE/WAPE/MASE all improve, but R² gets *worse*, and this is mechanistic, not a
contradiction.** `y_gapfilled` is smoother than `y_observed` (RFm predictions damp real spike noise),
so absolute errors shrink — but R² normalizes by the target's own variance, and a smoother target has
much lower variance, so the same (now-smaller) residual is a *larger* fraction of a *smaller* total
variance. R² penalizes exactly the flattening that makes the other four metrics look better. This is
a genuinely informative, if counter-intuitive, result: it shows concretely why R² and MASE/RMSE can
disagree about "is this better," and reinforces that no single metric should be read in isolation
(the same lesson D-65's own Correlation/RMSE split already surfaced for TabPFN).

**Finding 2 — model ranking mostly holds, with one real exception: TabPFN.** By R², the observed-
target ranking is TabPFN > Ensemble > XGB > LightGBM > RF > SARIMAX > TFT (worst). Under the
gap-filled target, TabPFN falls from **1st to 6th** — Ensemble_unweighted becomes the clear best,
LightGBM/XGB/RF reorder only slightly among themselves, and SARIMAX/TFT remain the two worst in both.
**Ensemble_unweighted staying at or near the top in both metrics reinforces the standing production
recommendation** (B-10's ensemble). TabPFN's drop is the one finding that should give real pause: its
observed-target R² advantage may be partly an artifact of scoring against the sparsest, most
spike-dominated subset of days (where TabPFN's zero-shot, non-autoregressive design might have an
edge) rather than a robust advantage across the fuller distribution `y_gapfilled` approximates.

**Finding 3 — Tower 2 stops being unusable, but does not stop looking like a genuinely harder tower.**
Every Tower 2 cell in the tower×year breakdown below is now a real number (previously ~95% `NaN`).
Tower 2's R² is negative in most cells, similar in magnitude to Tower 4/9's own negative cells in the
same years — not catastrophically worse, and occasionally *better* (2019/2020: several models post
small positive R² at T2, e.g. LightGBM +0.053 in 2019, Ensemble +0.147 in 2020). Read together with
Finding 1's caveat (T2's real-data coverage is only ~9%, far thinner than T4/T9's), this is best
described as "Tower 2 is no longer numerically blank, and what it now shows is broadly consistent
with — not contradicting — its already-documented data-scarce, harder-to-model status" rather than "the
coverage problem is solved."

### Tower × year × model breakdown (gap-filled target)

Same structure as the observed-target breakdown above (tower as parent column, year as parent row,
model nested beneath); flattened to `T{tower}_{metric}` for markdown, true nested `MultiIndex` in
`results/b10_b13_rerun_table_vs_gapfilled_by_tower_year.csv`.

| Year | Model | T2_RMSE | T2_MAE | T2_MASE | T2_WAPE | T2_Correlation | T4_RMSE | T4_MAE | T4_MASE | T4_WAPE | T4_Correlation | T9_RMSE | T9_MAE | T9_MASE | T9_WAPE | T9_Correlation | T2_R2 | T4_R2 | T9_R2 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2018 | Ensemble_MASEweighted | 12.01 | 9.16 | 0.209 | 1.025 | 0.381 | 29.01 | 21.36 | 0.846 | 0.742 | 0.513 | 27.56 | 21.04 | 1.094 | 0.650 | 0.628 | −0.783 | 0.024 | 0.028 |
| 2018 | Ensemble_unweighted | 12.07 | 9.23 | 0.211 | 1.035 | 0.382 | 29.02 | 21.41 | 0.848 | 0.745 | 0.513 | 27.67 | 21.23 | 1.108 | 0.658 | 0.629 | −0.810 | 0.022 | 0.011 |
| 2018 | LightGBM | 10.76 | 7.57 | 0.178 | 0.800 | 0.368 | 29.12 | 20.29 | 0.801 | 0.698 | 0.483 | 28.01 | 19.50 | 0.826 | 0.512 | 0.567 | −0.277 | 0.038 | 0.232 |
| 2018 | RF | 10.17 | 7.49 | 0.176 | 0.771 | 0.392 | 30.21 | 22.21 | 0.884 | 0.783 | 0.478 | 25.87 | 17.62 | 0.813 | 0.497 | 0.641 | −0.069 | −0.073 | 0.305 |
| 2018 | SARIMAX | 20.33 | 17.66 | 0.390 | 2.133 | 0.087 | 31.63 | 25.89 | 1.036 | 0.919 | 0.503 | 44.60 | 40.35 | 2.414 | 1.389 | 0.557 | −5.706 | −0.254 | −3.310 |
| 2018 | TFT | 20.97 | 16.24 | 0.360 | 2.127 | 0.081 | 45.33 | 33.80 | 1.261 | 1.110 | 0.201 | 54.00 | 39.32 | 1.900 | 1.110 | 0.055 | −8.956 | −1.461 | −4.246 |
| 2018 | TabPFN | 13.93 | 9.65 | 0.236 | 0.858 | −0.228 | 37.36 | 26.11 | 1.002 | 0.847 | 0.303 | 56.84 | 43.92 | 1.578 | 1.000 | nan | −0.521 | −0.468 | −1.627 |
| 2018 | XGB | 11.11 | 7.86 | 0.183 | 0.846 | 0.313 | 29.38 | 20.62 | 0.796 | 0.685 | 0.459 | 25.91 | 17.35 | 0.818 | 0.495 | 0.579 | −0.392 | 0.072 | 0.253 |
| 2019 | Ensemble_MASEweighted | 20.13 | 14.13 | 0.874 | 0.657 | 0.432 | 26.46 | 17.31 | 0.599 | 0.598 | 0.539 | 34.64 | 22.22 | 0.612 | 0.493 | 0.733 | −0.012 | −0.393 | 0.417 |
| 2019 | Ensemble_unweighted | 20.14 | 14.14 | 0.876 | 0.658 | 0.433 | 26.47 | 17.34 | 0.599 | 0.598 | 0.539 | 34.63 | 22.19 | 0.611 | 0.492 | 0.733 | −0.016 | −0.377 | 0.419 |
| 2019 | LightGBM | 19.90 | 13.84 | 0.850 | 0.638 | 0.411 | 26.73 | 17.06 | 0.621 | 0.620 | 0.538 | 36.75 | 24.51 | 0.707 | 0.558 | 0.704 | 0.053 | −0.824 | 0.216 |
| 2019 | RF | 19.69 | 13.94 | 0.872 | 0.651 | 0.454 | 27.28 | 18.32 | 0.694 | 0.692 | 0.527 | 35.63 | 23.51 | 0.640 | 0.518 | 0.722 | 0.010 | −1.354 | 0.405 |
| 2019 | SARIMAX | 22.18 | 16.29 | 1.049 | 0.795 | 0.312 | 31.28 | 24.31 | 0.738 | 0.739 | 0.560 | 38.17 | 24.50 | 0.667 | 0.544 | 0.731 | −0.594 | −0.070 | 0.335 |
| 2019 | TFT | 32.42 | 23.53 | 1.431 | 1.063 | −0.102 | 38.89 | 28.17 | 1.010 | 1.008 | 0.323 | 61.94 | 44.23 | 1.244 | 1.002 | 0.125 | −1.534 | −3.060 | −0.906 |
| 2019 | TabPFN | 28.66 | 20.05 | 1.236 | 0.907 | −0.189 | 40.05 | 28.45 | 0.808 | 0.808 | 0.522 | 66.09 | 45.12 | 1.237 | 1.000 | nan | −1.030 | −0.659 | −1.001 |
| 2019 | XGB | 20.64 | 14.29 | 0.875 | 0.659 | 0.337 | 27.11 | 17.62 | 0.648 | 0.647 | 0.517 | 35.24 | 23.32 | 0.652 | 0.523 | 0.708 | −0.014 | −1.045 | 0.354 |
| 2020 | Ensemble_MASEweighted | 14.94 | 10.72 | 0.806 | 0.628 | 0.420 | 38.91 | 22.37 | 0.869 | 0.623 | 0.502 | 36.36 | 23.77 | 0.728 | 0.594 | 0.644 | 0.147 | 0.243 | 0.339 |
| 2020 | Ensemble_unweighted | 14.95 | 10.75 | 0.808 | 0.629 | 0.421 | 38.88 | 22.33 | 0.867 | 0.622 | 0.504 | 36.37 | 23.76 | 0.727 | 0.593 | 0.644 | 0.147 | 0.245 | 0.339 |
| 2020 | LightGBM | 15.06 | 10.48 | 0.789 | 0.612 | 0.407 | 40.06 | 23.73 | 0.953 | 0.678 | 0.449 | 36.30 | 24.54 | 0.751 | 0.612 | 0.643 | 0.125 | 0.173 | 0.343 |
| 2020 | RF | 15.53 | 11.20 | 0.841 | 0.658 | 0.427 | 37.92 | 22.46 | 0.908 | 0.647 | 0.517 | 37.82 | 25.38 | 0.773 | 0.630 | 0.619 | 0.095 | 0.237 | 0.283 |
| 2020 | SARIMAX | 16.36 | 13.01 | 0.982 | 0.763 | 0.438 | 39.66 | 23.12 | 0.856 | 0.623 | 0.505 | 37.32 | 24.07 | 0.752 | 0.615 | 0.604 | −0.021 | 0.248 | 0.273 |
| 2020 | TFT | 23.22 | 16.45 | 1.281 | 0.906 | 0.137 | 43.22 | 25.92 | 1.054 | 0.743 | 0.412 | 47.25 | 30.47 | 1.002 | 0.805 | 0.256 | −1.203 | −0.026 | −0.336 |
| 2020 | TabPFN | 22.72 | 15.25 | 1.186 | 0.855 | −0.109 | 44.11 | 22.95 | 0.910 | 0.641 | 0.460 | 47.73 | 30.62 | 0.917 | 0.742 | 0.334 | −0.985 | −0.007 | −0.134 |
| 2020 | XGB | 15.13 | 10.25 | 0.770 | 0.601 | 0.367 | 40.25 | 23.49 | 0.936 | 0.666 | 0.436 | 36.43 | 24.11 | 0.743 | 0.605 | 0.630 | 0.119 | 0.172 | 0.327 |
| 2021 | Ensemble_MASEweighted | 13.01 | 9.82 | 0.873 | 0.598 | 0.219 | 42.05 | 31.30 | 0.952 | 0.731 | 0.542 | 26.62 | 17.70 | 0.553 | 0.487 | 0.545 | −0.173 | 0.135 | 0.173 |
| 2021 | Ensemble_unweighted | 13.01 | 9.82 | 0.873 | 0.598 | 0.217 | 42.06 | 31.28 | 0.951 | 0.730 | 0.542 | 26.63 | 17.71 | 0.552 | 0.487 | 0.547 | −0.175 | 0.137 | 0.175 |
| 2021 | LightGBM | 13.93 | 10.55 | 0.926 | 0.631 | 0.238 | 42.09 | 31.51 | 0.963 | 0.742 | 0.534 | 27.72 | 18.92 | 0.615 | 0.574 | 0.471 | −0.388 | 0.094 | −0.122 |
| 2021 | RF | 12.98 | 10.68 | 0.971 | 0.659 | 0.360 | 43.89 | 32.70 | 1.070 | 0.802 | 0.523 | 27.43 | 18.77 | 0.609 | 0.592 | 0.551 | −0.156 | −0.088 | −0.168 |
| 2021 | SARIMAX | 15.46 | 11.58 | 1.069 | 0.724 | 0.061 | 43.23 | 30.36 | 0.867 | 0.678 | 0.521 | 36.00 | 27.67 | 0.947 | 0.920 | 0.586 | −0.783 | 0.172 | −1.337 |
| 2021 | TFT | 23.37 | 17.11 | 1.551 | 0.964 | 0.227 | 48.77 | 32.47 | 1.123 | 0.797 | 0.410 | 43.09 | 31.33 | 1.235 | 1.178 | 0.431 | −2.626 | −0.967 | −4.599 |
| 2021 | TabPFN | 21.59 | 17.21 | 1.415 | 0.887 | −0.119 | 47.15 | 28.09 | 0.791 | 0.612 | 0.515 | 34.92 | 24.86 | 0.769 | 0.636 | 0.510 | −1.733 | 0.091 | −0.275 |
| 2021 | XGB | 13.01 | 9.70 | 0.839 | 0.574 | 0.215 | 42.72 | 32.17 | 0.991 | 0.756 | 0.528 | 27.70 | 19.09 | 0.611 | 0.557 | 0.492 | −0.111 | 0.046 | −0.041 |
| 2022 | Ensemble_MASEweighted | 12.89 | 8.91 | 0.462 | 0.561 | 0.712 | 22.72 | 16.99 | 0.871 | 1.223 | 0.576 | 23.43 | 17.99 | 0.911 | 0.951 | 0.451 | −0.076 | −2.428 | −0.561 |
| 2022 | Ensemble_unweighted | 12.87 | 8.88 | 0.460 | 0.559 | 0.712 | 22.70 | 16.98 | 0.868 | 1.216 | 0.575 | 23.38 | 17.93 | 0.905 | 0.945 | 0.450 | −0.070 | −2.351 | −0.537 |
| 2022 | LightGBM | 13.56 | 9.96 | 0.524 | 0.636 | 0.674 | 24.57 | 18.64 | 0.984 | 1.435 | 0.527 | 27.04 | 21.93 | 1.119 | 1.167 | 0.434 | −0.250 | −5.336 | −1.271 |
| 2022 | RF | 14.16 | 11.01 | 0.588 | 0.720 | 0.664 | 23.75 | 18.09 | 1.026 | 1.526 | 0.595 | 25.45 | 20.37 | 1.130 | 1.195 | 0.470 | −0.420 | −6.169 | −1.724 |
| 2022 | SARIMAX | 14.11 | 10.38 | 0.507 | 0.624 | 0.566 | 23.88 | 17.94 | 0.953 | 1.332 | 0.530 | 23.99 | 17.84 | 0.923 | 0.995 | 0.382 | −0.054 | −3.047 | −0.919 |
| 2022 | TFT | 17.42 | 12.77 | 0.690 | 0.832 | 0.252 | 23.12 | 17.13 | 1.032 | 1.501 | 0.597 | 39.90 | 30.74 | 1.748 | 1.805 | 0.320 | −1.253 | −5.106 | −6.470 |
| 2022 | TabPFN | 20.86 | 14.86 | 0.715 | 0.893 | −0.108 | 23.50 | 14.75 | 0.697 | 0.935 | 0.689 | 22.40 | 15.95 | 0.734 | 0.759 | 0.401 | −1.303 | −0.627 | −0.049 |
| 2022 | XGB | 13.95 | 9.94 | 0.521 | 0.629 | 0.626 | 22.91 | 16.85 | 0.940 | 1.385 | 0.559 | 24.28 | 18.93 | 1.087 | 1.142 | 0.471 | −0.366 | −5.248 | −1.653 |

## Model-roster extension: DLinear + LSTM (closing the `b10_chains` figure gap)

DLinear and LSTM were tested in B-09 (D-53: DLinear mean R²=−4.75, the worst/most unstable model
in the whole B-09→B-15 sequence, excluded from B-10's ensemble on that basis) and had their rollout
chains extended to all 3 towers × 5 anchors for **visualization only**
(`results/figures/b10_chains/T*_anchor*_{DLinear,LSTM}.png`, B-13's Part A) — but that chain-plot
extension never saved a `bin_metrics()` summary, so there was never a full 3-tower × 5-anchor
evaluation table for these two models with the D-65 metric set (RMSE/WAPE/Correlation). This
section closes that gap, reusing the exact recipe read directly from the committed
`B09_recursive_rollout.ipynb` (Section 4) — **not** B-10's H=1-native-retrain variant (D-54: that
retrain helped LSTM but hurt DLinear further, and isn't what the `b10_chains` figures used): track B
(`L=28, H=14`), pooled training (Towers 2+4+9, fit once per anchor, **no validation carve-out** —
deliberately different from TFT's regularized recipe, since B-09's original DLinear/LSTM training
never used one), rolled out per tower via `rr.dl_rollout` (already generic, zero new rollout code
needed), `y_true` from `fdl.tower_series(...)["Y"]` (the same daily-resampling source TFT already
uses, not `forecast_daily_v2.csv`'s `y_observed` directly).

**Verification, and a genuinely new, sharper finding about this project's own DL non-determinism.**
Tower 4 was checked against the only prior multi-anchor DLinear/LSTM record
(`results/b09_multi_anchor_summary.csv`) across all 5 anchors:

- **LSTM reproduces bit-for-bit exactly, in every bin, in all 5 anchors** — zero exceptions.
- **DLinear matches bit-for-bit exactly for 4 of 5 anchors (2019–2022)** — only the very **first**
  anchor processed in this run (2018) differs, and by a lot in one bin (91-180: −75.19 vs. the
  published −3.57).

This precisely **sharpens the existing D-62 addendum** (previously scoped as "TFT's initial weights
are never seeded before construction"). The real mechanism: `fdl.train_model()` calls
`torch.manual_seed(seed)` **after** `fdl.build_model()` has already constructed and randomly
initialized the model — so whichever model is built **first in a whole process, before any prior
`torch.manual_seed()` call anywhere in that process**, gets non-deterministic initial weights; every
model built afterward (a different model type, or the same model type on a later anchor) lands on
fully deterministic initialization, because some earlier `train_model()` call already fixed the
global PyTorch RNG. In this script DLinear is built first each anchor, but only the **very first**
anchor of the whole run (2018) is actually "first in the process" in the relevant sense — by anchor
2019 the RNG is already seed-derived from four prior `train_model()` calls (DLinear+LSTM ×
2018+2019's own DLinear), so every later anchor's DLinear (and LSTM, always built second) become
exactly reproducible. This is the likely explanation for TFT's own documented non-determinism too
(D-62) — TFT is typically the only/first torch model built in its script — and is a more precise,
actionable characterization than "TFT is unseeded": **seeding once at the very top of a script,
before building any model, would likely make every model in it exactly reproducible.** Not applied
here (would risk silently changing this project's own already-published TFT/DLinear/LSTM numbers
without being asked) — noted as a real, easy future fix.

**All-tower pooled (DLinear + LSTM only, same aggregation convention as the primary tables above):**

| Model | RMSE | MAE | MASE | WAPE | Correlation | R² |
|---|---|---|---|---|---|---|
| DLinear | 70.89 | 51.72 | 1.637 | 1.905 | 0.241 | −5.057 |
| LSTM | 63.22 | 41.06 | 1.151 | 1.268 | 0.212 | −1.357 |

**Tower-4-only (direct comparison to B-09's original headline):**

| Model | RMSE | MAE | MASE | WAPE | Correlation | R² |
|---|---|---|---|---|---|---|
| DLinear | 74.22 | 53.68 | 1.866 | 2.036 | 0.243 | −6.154 |
| LSTM | 59.83 | 37.27 | 1.106 | 1.105 | 0.228 | −0.439 |

Both are drastically worse than every one of the 8 models in the primary tables above (best all-tower
R² there is Ensemble_unweighted at −0.165) — **directly confirming, not contradicting, D-53/D-54's
already-established finding** that DLinear/LSTM are the least robust models in this whole sequence
and were correctly excluded from B-10's production ensemble. Full 3-tower coverage now shows this
holds at Tower 2/9 too, not just Tower 4.

**Note on the merged-table question.** These two models are deliberately reported in their own
standalone tables above, not merged into the primary 8-model "All-tower summary"/"Tower-4-only
table" sections — merging would require re-deriving TFT's numbers from the currently-saved
`results/b10_b13_rerun_summary.csv`, whose TFT row has (independently, while this DLinear/LSTM work
was underway) drifted to a **third** random draw since the primary tables above were built
(Tower-4-only TFT R² is currently −0.232 in that file vs. −0.568 cited above) — yet another
manifestation of TFT's already-documented non-determinism, but here surfacing as a staleness
mismatch between sibling artifacts rather than a single-run caveat. Flagged plainly rather than
silently picking whichever TFT draw is convenient; not resolved in this pass.

### Tower × year × model breakdown (DLinear + LSTM)

| Year | Model | T2_RMSE | T2_MAE | T2_MASE | T2_WAPE | T2_Correlation | T4_RMSE | T4_MAE | T4_MASE | T4_WAPE | T4_Correlation | T9_RMSE | T9_MAE | T9_MASE | T9_WAPE | T9_Correlation | T2_R2 | T4_R2 | T9_R2 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2018 | DLinear | 40.58 | 33.74 | 0.646 | 2.623 | 0.026 | 119.84 | 86.20 | 3.330 | 4.468 | −0.050 | nan | nan | nan | nan | nan | −6.762 | −25.928 | nan |
| 2018 | LSTM | 33.05 | 26.46 | 0.508 | 2.054 | 0.319 | 54.39 | 33.47 | 1.026 | 1.225 | 0.079 | nan | nan | nan | nan | nan | −3.791 | −0.678 | nan |
| 2019 | DLinear | nan | nan | nan | nan | nan | 63.57 | 45.01 | 1.381 | 1.383 | 0.364 | 83.37 | 58.84 | 1.186 | 1.093 | 0.282 | nan | −1.222 | −0.351 |
| 2019 | LSTM | nan | nan | nan | nan | nan | 56.10 | 33.15 | 0.842 | 0.843 | 0.326 | 86.21 | 52.44 | 1.006 | 0.926 | 0.134 | nan | −0.119 | −0.250 |
| 2020 | DLinear | nan | nan | nan | nan | nan | 72.66 | 49.14 | 1.544 | 1.392 | 0.138 | 61.82 | 43.85 | 1.195 | 1.365 | 0.277 | nan | −1.139 | −1.342 |
| 2020 | LSTM | nan | nan | nan | nan | nan | 70.13 | 40.93 | 1.231 | 1.085 | 0.175 | 75.78 | 53.74 | 1.634 | 1.925 | 0.220 | nan | −0.582 | −6.314 |
| 2021 | DLinear | nan | nan | nan | nan | nan | 63.90 | 48.20 | 1.548 | 1.312 | 0.460 | 71.78 | 49.58 | 0.991 | 1.248 | 0.291 | nan | −1.820 | −1.223 |
| 2021 | LSTM | nan | nan | nan | nan | nan | 67.10 | 43.00 | 1.135 | 0.998 | 0.292 | 77.83 | 50.57 | 0.882 | 0.904 | 0.152 | nan | −0.349 | −0.368 |
| 2022 | DLinear | nan | nan | nan | nan | nan | 51.15 | 39.85 | 1.529 | 1.627 | 0.301 | 49.40 | 41.08 | 1.633 | 1.674 | 0.299 | nan | −0.660 | −1.378 |
| 2022 | LSTM | nan | nan | nan | nan | nan | 51.41 | 35.79 | 1.294 | 1.371 | 0.267 | 56.25 | 41.55 | 1.634 | 1.675 | 0.212 | nan | −0.464 | −1.986 |

Tower 2 is only reachable in the 2018 anchor (same data-scarcity pattern as every other model in
this document). The 2018/Tower-4/DLinear cell (R²=−25.93) is the one confirmed non-deterministic
outlier discussed above — read as "this specific run's draw for a known-fragile anchor/model
combination," not a re-validated number.

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
- `results/b10_b13_rerun_summary_vs_gapfilled.csv` (secondary metric, 720 rows, +`real_frac` column)
- `results/b10_b13_rerun_table_vs_gapfilled_all_towers.csv` (secondary metric, all-tower pooled)
- `results/b10_b13_rerun_table_vs_gapfilled_by_tower_year.csv` (secondary metric, tower × year
  breakdown, true nested `MultiIndex`)
- `notebooks/05_benchmarking/b10_b13_dl_extension.py` (new, committed — DLinear/LSTM model-roster
  extension, reconstructed from `B09_recursive_rollout.ipynb`)
- `results/b10_b13_dl_extension_summary.csv` (180 rows: 2 models × 3 towers × 5 anchors × 6 bins)
- `results/b10_b13_dl_extension_table_all_towers.csv`, `_table_tower4.csv`,
  `_table_by_tower_year.csv` (DLinear/LSTM's own standalone aggregated tables, observed target)
- `results/b10_b13_dl_extension_summary_vs_gapfilled.csv` (DLinear/LSTM, secondary metric,
  180 rows, +`real_frac`), `results/b10_b13_dl_extension_table_all_towers_vs_gapfilled.csv`
  (all-tower pooled, used in the "Secondary metric" section's table above)

**Raw daily prediction chains (new, on request)** — the figures in `results/figures/b10_chains/`
were generated by an ad-hoc, uncommitted script that never saved its underlying per-day predictions
(only the binned metrics survived, in the CSVs above). `b10_b13_rerun_multi_anchor.py` and
`b10_b13_dl_extension.py` were extended to also save the raw chains they already compute in memory:
- `results/b10_b13_rerun_chains.csv` (5,475 rows: 3 towers × 5 anchors × 365 days; columns:
  `date, tower, anchor_year, y_true, y_true_tft, y_gapfilled, persistence`, + one column per model
  for RF/XGB/LightGBM/SARIMAX/Ensemble_unweighted/Ensemble_MASEweighted/TFT/TabPFN)
- `results/b10_b13_dl_extension_chains.csv` (5,475 rows, same shape, DLinear/LSTM)
- `results/b10_b13_full_chains.csv` — **the consolidated export**: both files merged on
  `date`/`tower`/`anchor_year`, all 10 models as columns, zero missing predictions anywhere.
  Recomputing `bin_metrics()` directly from this file's raw values was spot-checked to reproduce
  the already-published summary rows exactly (Tower 4/2021/RF, all 6 bins, bit-for-bit). Also
  visually verified against the archived `results/figures/b10_chains/` PNGs by re-plotting from
  this CSV (RF/LSTM/DLinear at Tower 4/anchor 2021 all matched pixel-for-pixel; DLinear at the 2018
  anchor did not, as expected from its documented non-determinism).
- `results/b10_b13_rerun_table_per_tower_gf_vs_observed.csv` (per-tower breakdown, all 10 models,
  gap-filled vs. observed side by side — used in the "Per-tower breakdown" section above)

No `benchmarks.csv` rows (a metrics backfill + verification exercise on already-logged results,
same precedent as D-44b's own backfill).
