# `03c_gap_filling_revisited` — Summary

**Notebooks:** `temp_gap_filing_exploration.ipynb` (pristine, self-contained reproduction, zero
`src/` imports) and `temp_gap_filing_exploration copy.ipynb` (all extended exploration below).
**Scope:** all 3 towers (T2, T4, T9) throughout, full-period gap-CV (D-35/D-49 methodology).

This document consolidates every metric produced across both notebooks into one reference. It
does not replace the notebooks' own inline markdown (which carries the full methodology/rationale
for each decision) — it is the fast-lookup index.

---

## 1. Standing champion

Self-contained rebuild of the established gap-filling pipeline (partial-pooled, external-sourced
RFm, full-period gap-CV) found and fixed a real bug: `mdc_gapfill()`'s interpolation cutoff was a
flat 2 hours for every driver — too short for low-diurnal-structure variables (soil moisture, soil
temperature, TA, VPD, WS), which have no strong daily cycle for the mean-diurnal-course fallback to
lean on. Extending the cutoff to 288h for those 5 variables only, validated end-to-end, moved every
tower.

| Tower | Prior (`BEST_RESULTS.md`) | Current champion | Δ |
|---|---|---|---|
| T2 | 0.574 | **0.576** | +0.002 |
| T4 | 0.402 | **0.404** | +0.002 |
| T9 | 0.418 | **0.426** | +0.008 |

**This is the standing recommendation. Every result below was tested against it and none beat it
outright** (two models edge it at Tower 4 only — see §2).

---

## 2. Additional models tested (full 3-tower)

| Model | T2 | T4 | T9 | Verdict |
|---|---|---|---|---|
| **RFm_pool (champion)** | **0.576** | 0.404 | **0.426** | standing |
| LightGBM | 0.522 | **0.410** | 0.422 | edges champion at T4 only (+0.006) |
| XGBoost | 0.551 | 0.349 | 0.369 | loses everywhere |
| TabPFN | 0.459 | 0.401 | 0.402 | loses everywhere; ~3.7h for a 60-fold sweep |
| TabICL | 0.558 | **0.423** | 0.364 | edges champion at T4 only (+0.019); ~2 min for the same sweep |
| SAITS | 0.358 | 0.293 | 0.285 | loses everywhere; F-11's own best config, substantially improved on the corrected features (was 0.192/0.225/0.110) but still short |
| BI-LSTM | 0.237 | 0.155 | 0.146 | loses everywhere, weakest of all six; custom self-supervised windowed bidirectional LSTM |

Source: `_data/model_comparison.csv`.

## 3. Lag/lead feature expansion (full 3-tower)

| Experiment | T2 | T4 | T9 | Verdict |
|---|---|---|---|---|
| Soil-lag bidirectional | 0.561 | 0.410 | 0.415 | reproduces F-12's null result almost exactly on the corrected features |
| Soil-lag lead-only | 0.564 | 0.412 | 0.411 | same |
| Target (FCH4) lag/lead | 0.495 | 0.329 | 0.353 | new — never tested before; clear regression, larger than any other variant tested |

Source: `_data/soil_lag_results.csv`, `_data/target_laglead_results.csv`.

---

## 4. Uncertainty quantification — Area of Applicability (dissimilarity index)

Meyer & Pebesma-style dissimilarity index (scaled nearest-neighbour distance to training data,
Tukey IQR-fence threshold), replicated inline from `src/models/scenario_hybrid.py`. A
trustworthiness diagnostic, not a prediction interval — purely geometric, never looks at the
target.

- Threshold: **1.950**. Flagged: **32.9%** of 18,001 genuinely held-out validation points.
- Correlation with absolute error: Pearson **+0.110** pooled (T2 +0.123, T4 +0.101, T9 +0.159),
  Spearman +0.145 pooled — weak but real, consistent direction at all 3 towers.
- Flagged vs. unflagged mean |error|: **59.53 vs. 37.87** pooled (+57%); T2 42.03 vs. 26.92, T4
  60.33 vs. 39.41, T9 69.45 vs. 38.94.
- Applied to the full production series (104,355 rows): % of gap-filled hours flagged — T2 48%,
  T4 30%, T9 37%.

Source: `_data/fch4_gapfilled_with_uq.csv`.

## 5. Prediction intervals — hourly (QRF + TabICL native quantiles)

QRF: hand-rolled, reads quantiles from the champion's own already-fitted leaves (no retraining).
TabICL: native `output_type="quantiles"`. Scored on 178,560 held-out points (89,280 per model,
all 5 scenarios).

| Model | Tower | Coverage (target 0.90) | Mean width | corr(width, \|err\|) |
|---|---|---|---|---|
| QRF | T2 | 0.941 | 162.7 | +0.644 |
| QRF | T4 | 0.919 | 197.2 | +0.493 |
| QRF | T9 | 0.898 | 213.7 | +0.504 |
| TabICL | T2 | 0.918 | 141.7 | +0.590 |
| TabICL | T4 | 0.901 | 184.3 | +0.509 |
| TabICL | T9 | 0.893 | 220.8 | +0.482 |

Both land close to the 90% target at every tower, and both show a genuinely informative
width-vs-error relationship (0.48–0.64) — a solid, above-average result given FCH4's spike-dominated
irreducible noise.

Source: `_data/prediction_intervals_summary.csv`, `_data/prediction_intervals_detail.csv`.
Figures: `_figures/fanchart_T{2,4,9}.png`, `_figures/calibration_reliability.png`.

## 6. Applied to the real gap-filled series — the blind spot

Same QRF/TabICL applied to every one of 104,355 real hours (real + gap-filled). Tests whether
interval width tracks `gap_run_length_h` (length of each gap-filled point's own real blackout) —
something the synthetic validation in §5 cannot test, since synthetic gaps can only be placed
where the target (and, per this project's block-structured-missingness finding, IMP-01, the
covariates too) were genuinely observed.

| Tower | QRF corr | TabICL corr |
|---|---|---|
| T2 | -0.060 | +0.029 |
| T4 | -0.135 | -0.072 |
| T9 | -0.136 | +0.100 |
| **Pooled** | **-0.093** | **+0.077** |

**QRF's interval gets narrower, not wider, for longer real gaps, at every tower.** Root cause
(diagnosed, not just observed): covariates are themselves smoothly interpolated during long real
blackouts, so they look artificially "typical" to the model — most overconfident exactly where it
should be least. **Practical implication:** a narrow interval on a long real gap cannot be read as
"trust this value" — `gap_run_length_h` itself is a more honest reliability signal than the
interval width.

Source: `_data/fch4_gapfilled_with_intervals.csv`.
Figures: `_figures/production_interval_T{2,4,9}.png` (raw band only, superseded visually by §7's
figures which show raw + calibrated).

## 7. Fix attempt 1 — gap-length-stratified conformal calibration

Split-conformal (CQR): margins derived from a genuinely held-out half of the validation set, per
`(model, tower, scenario)` bucket, applied to production intervals via each point's own
`gap_run_length_h` mapped to the nearest bucket.

| Model | Raw coverage | Calibrated coverage |
|---|---|---|
| QRF | 0.919 | **0.899** |
| TabICL | 0.902 | **0.898** |

Aggregate coverage lands almost exactly on 90% — genuinely validated, not circular (proper
train/test split for the calibration itself). **But the width-vs-gap-length correlation barely
moved** (QRF pooled stayed ≈ -0.09 to -0.13). A 4-bucket step function is too coarse to reshape a
continuous relationship — sharpness stayed broken even though calibration succeeded.

Source: `_data/conformal_calibration_summary.csv`, `_data/fch4_gapfilled_with_intervals_calibrated.csv`.
Figures: `_figures/production_interval_T{2,4,9}.png` (final version, raw + calibrated bands shown together).

## 8. Fix attempt 2 — structural fix (`dist_to_real_obs` as a model feature)

A leakage-safe feature (hours to nearest real observation, either direction; masks the target at
the current fold's own held-out points *first*, so a neighbour's true value can never leak through)
added directly to the champion's own architecture.

| Check | T2 | T4 | T9 | Pooled |
|---|---|---|---|---|
| Point R² vs. champion (Δ) | +0.000 | -0.000 | -0.000 | — |
| corr(dist, width) — before | -0.060 | -0.135 | -0.136 | -0.093 |
| corr(dist, width) — after | +0.025 | +0.003 | -0.018 | **-0.000** |

No accuracy cost at all. The negative bias is gone (no longer confidently narrows on long real
gaps), but it landed at noise level, not genuine positive calibration. Most likely cause: QRF only
widens an interval where the leaves it splits on show materially different target spread in
training data — FCH4's baseline-dominated distribution may look similarly tight regardless of
distance from real data, even when the RF is otherwise using that feature correctly for point
prediction.

Source: `_data/gapdist_feature_r2_check.csv`, `_data/gapdist_feature_interval_check.csv`.

## 9. Consolidated UQ export

One row per (Datetime, tower), merging §4's DI/AoA flag with §7's raw + calibrated QRF/TabICL
intervals, plus each gap-filled hour's own `gap_run_length_h` mapped to a human-readable
`scenario_assigned` label (`vs`/`s`/`m`/`l`).

- **104,355 rows × 19 columns.** 68,769 gap-filled hours, all with a scenario assigned.
- Scenario distribution among gap-filled hours:

| Tower | vs (~1h) | s (~4h) | m (~32h) | l (~288h) |
|---|---|---|---|---|
| T2 | 1,802 | 2,674 | 2,031 | 3,892 |
| T4 | 7,110 | 10,621 | 9,119 | 8,454 |
| T9 | 4,088 | 5,998 | 4,722 | 8,258 |

At T2 and T9, `l` (the longest, most UQ-caveated bucket) is the single largest category.

Source: `_data/fch4_uq_consolidated.csv`.

---

## 10. Daily-resolution prediction intervals

Deliberately **not** a propagation of the hourly `q05`/`q95` bands upward (would require an
unproven assumption about how hourly errors correlate within a day). Instead: aggregate the
champion's point predictions to daily mean (matching §11's own "primary" designation), then
calibrate a band directly on **genuine daily-resolution residuals from held-out data** via
split-conformal — mirrors this project's own forecasting-track pattern (`conformal_margins_by_bin`,
U-02), reimplemented locally. No new RF fits — reuses `HELD_OUT_PAIRS_L` (§11) directly.

**Note on the model**: the point estimate here is the same champion RF used everywhere in this
notebook (via `run_rf_capture`/`rf_prod`), not QRF or TabICL. The interval is a single, fixed,
symmetric margin per tower (`[daily_pred - margin, daily_pred + margin]`) — a structurally simpler
construction than the hourly per-point QRF/TabICL bands, and does not vary day-to-day.

| Tower | Conformal margin | Conformal PICP (target 0.90) | Naive normal-approx PICP | Cal/test days |
|---|---|---|---|---|
| T2 | ±52.36 | 0.866 | 0.866 (identical) | 112 / 112 |
| T4 | ±80.76 | 0.923 | 0.935 | 430 / 430 |
| T9 | ±62.34 | 0.861 | 0.910 | 245 / 245 |

All three towers land within normal sampling noise of the 90% target (±~0.05 given these sample
sizes). **The naive normal-approximation band did as well as, or better than, the distribution-free
conformal one at every tower** — daily averaging appears to smooth FCH4's hourly spike-skew enough
that the extra sophistication of conformal calibration isn't clearly earning its keep at this
resolution (unlike hourly, where it mattered more).

**A scope diagnostic (checking whether "m"/32h should also be in scope alongside "l"/288h) came
back uninformative** — both scenarios showed 0.0% fully-masked calendar days at every tower. Cause:
the test required 100% real-observed hours in a day, an almost-impossible bar given FCH4's own
~12–45% baseline sparsity, regardless of synthetic gap length. Doesn't undermine the "l"-only scope
choice, which rests on §11's own independent precedent (`groupby(index.normalize()).mean()` doesn't
require full-day coverage) — just a broken litmus test, reported rather than hidden.

**Downstream-use finding**: this daily interval was originally built with downstream daily-level
forecasting in mind, but on reflection is likely moot for that purpose — this project's forecasting
track builds its own UQ from its own rollout residuals binned by lead-time (U-02 pattern), and
would not consume this notebook's interval regardless of how it's constructed. Retained here as a
standalone daily *view* of the gap-filling result's own uncertainty (paralleling why the hourly
bands matter for the hourly report views), not as a forecasting input.

Source: `_data/daily_conformal_calibration_summary.csv`, `_data/fch4_daily_gapfilled_with_intervals.csv`.
Figures: `_figures/daily_calibration_reliability.png`, `_figures/daily_production_intervals.png`.

## 11. Daily R² — genuinely held-out, both scopes

**Important methodology note, caught via user cross-check:** do **not** compute R² from
`FCH4_GAPFILLED`/`fch4_gapfilled.csv` or any of its UQ-augmented derivatives (§4, §6, §7, §9, §10's
production files). Those are built from `rf_prod`, fit on *all* real data with nothing withheld —
comparing `y_observed` vs. `y_gapfilled` on real-observed rows is an in-sample comparison (the model
saw those exact rows during training) and produces inflated, meaningless accuracy numbers (a
same-data spot-check in Excel returned R²=0.879 at Tower 9 — pure leakage, not genuine skill). The
only valid source for accuracy figures is the genuinely held-out gap-CV mechanism below.

**§11a — "l" (288h) scenario only** (`HELD_OUT_PAIRS_L`, hourly pairs aggregated to daily via
`groupby(index.normalize()).mean()`):

| Tower | Held-out hours → days | R² hourly | R² daily | Δ |
|---|---|---|---|---|
| T2 | 2,565 → 224 | 0.533 | 0.207 | -0.325 |
| T4 | 9,754 → 860 | 0.268 | 0.321 | +0.053 |
| T9 | 5,682 → 490 | 0.281 | 0.504 | +0.223 |

**§11b — full scope, all 5 scenarios, median** (directly comparable to the champion's own hourly
headline — same aggregation method, extended from "l"-only to every scenario, reusing the
champion's own cached RF fits):

| Tower | Hourly headline (all 5 scenarios) | Daily median (all 5 scenarios) | Δ |
|---|---|---|---|
| T2 | 0.576 | **0.698** | +0.122 |
| T4 | 0.404 | **0.504** | +0.100 |
| T9 | 0.426 | **0.485** | +0.059 |

**Daily beats hourly at every tower once the full scenario scope is used** — confirms the general
"aggregation cancels noise" intuition, which the "l"-only comparison alone had obscured (T2's
"l"-scenario daily result is a real outlier specific to that hardest/most data-scarce case, not
representative of daily performance across scenarios generally — T2 in fact shows the *strongest*
daily uplift of all 3 towers on the other 4 scenarios, up to 0.805).

Full per-scenario daily R²:

| Tower | vs | s | m | l | m1 |
|---|---|---|---|---|---|
| T2 | 0.698 | 0.805 | 0.788 | 0.207 | 0.697 |
| T4 | 0.504 | 0.571 | 0.359 | 0.321 | 0.532 |
| T9 | 0.524 | 0.336 | 0.377 | 0.504 | 0.485 |

Source: `_data/daily_r2_full_scope.csv` (§11b); `HELD_OUT_PAIRS_L`/`DAILY_R2` are in-notebook only
(§11a, not separately exported — see notebook section 13).

**§11c — sensitivity to daily coverage.** §11b's daily R² is unweighted: a day represented by a
single held-out hour counts identically to one represented by 24. Checked against the standard
FLUXNET/AmeriFlux completeness convention (daily aggregate valid only with ≥50% of hours present)
and a softer alternative (`r2_score`'s own `sample_weight`, weighting each day by its real-hour
count) — both reusing the same cached RF fits as §11b, no new fitting.

| Tower | Unweighted (§11b) | ≥50% coverage only | Hour-count weighted |
|---|---|---|---|
| T2 | 0.698 | 0.638 | 0.721 |
| T4 | 0.504 | 0.446 | 0.528 |
| T9 | 0.485 | **0.586** | 0.498 |

**Not a uniform effect** — at T2/T4 the ≥50% threshold *lowers* R² (weighting nudges it back up
slightly); at T9 the threshold *raises* it to 0.586, its best of the three. No clean "rigor always
makes it worse/better" story; it depends on which specific days are dense vs. sparse at each tower.
Days surviving the ≥50% (≥12h) threshold per scenario confirm `vs`/`s` lose the large majority of
their days but not all of them (occasional coincidental overlap of multiple short blocks on one
calendar day): T2 vs=12/457, s=29/431; T4 vs=50/1826, s=121/1707; T9 vs=22/1023, s=72/967 (`m`/`l`/
`m1` retain far more — e.g. T4 `l`=374/860).

Source: `_data/daily_r2_coverage_sensitivity.csv` (§11c, additive — `daily_r2_full_scope.csv`/§11b
untouched).

---

## Overall verdict

- **Accuracy**: RFm_pool remains the standing gap-filling recommendation at every tower —
  T2 **0.576**, T4 **0.404**, T9 **0.426** (hourly); **0.698 / 0.504 / 0.485** (daily, full scope,
  genuinely held-out). Nothing tested this session beats it decisively; TabICL and LightGBM are
  narrow, Tower-4-only exceptions.
- **Uncertainty**: two usable layers exist — the AoA flag (§4, weak-but-real) and the hourly
  QRF/TabICL prediction intervals (§5, well-calibrated and genuinely informative in validation).
  Applied to real data, both under-react to long real blackouts (§6); conformal calibration fixed
  aggregate coverage but not per-point sharpness (§7); adding gap-distance as a feature removed the
  harmful negative bias without achieving positive calibration (§8). **Read a narrow interval on a
  long real gap with appropriate skepticism** — this is the honest state of the art here, not a
  solved problem.
- **Daily resolution**: a genuinely held-out daily R² now exists and beats the hourly number at
  every tower (§11b: 0.698/0.504/0.485 vs. 0.576/0.404/0.426). That number is unweighted, though —
  §11c shows it's somewhat sensitive to how sparse-day coverage is handled (±0.06–0.10 depending on
  tower and method), so treat 0.698/0.504/0.485 as the headline but not the only defensible daily
  number. A daily prediction interval also exists (§10, simple fixed-margin construction) but was
  found likely moot for its original intended use (feeding downstream forecasting) — retained as a
  standalone reporting view only.

## File manifest

**`_data/`**: `model_comparison.csv` (§2) · `soil_lag_results.csv`, `target_laglead_results.csv`
(§3) · `fch4_gapfilled_with_uq.csv` (§4) · `prediction_intervals_{summary,detail}.csv` (§5) ·
`fch4_gapfilled_with_intervals.csv` (§6) · `conformal_calibration_summary.csv`,
`fch4_gapfilled_with_intervals_calibrated.csv` (§7) · `gapdist_feature_{r2,interval}_check.csv`
(§8) · `fch4_uq_consolidated.csv` (§9) · `daily_conformal_calibration_summary.csv`,
`fch4_daily_gapfilled_with_intervals.csv` (§10) · `daily_r2_full_scope.csv` (§11b) ·
`daily_r2_coverage_sensitivity.csv` (§11c).

**`_figures/`**: `fanchart_T{2,4,9}.png`, `calibration_reliability.png` (§5) ·
`production_interval_T{2,4,9}.png` (§6, final version reflects §7's raw+calibrated overlay) ·
`daily_calibration_reliability.png`, `daily_production_intervals.png` (§10).

**Project-level docs updated**: `DECISIONS.md` (D-77, D-78), `BEST_RESULTS.md` §1,
`CONTEXT.md` current-status — cover the champion fix and §2/§3 model/lag-lead work only; the UQ
arc (§4–§11, sections 17/21–27 of the notebook) is not yet reflected there as of this document.
