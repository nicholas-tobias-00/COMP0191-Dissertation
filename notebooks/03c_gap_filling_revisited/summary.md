# `03c_gap_filling_revisited` — Summary

**Notebooks:** `temp_gap_filing_exploration.ipynb` (pristine, self-contained reproduction, zero
`src/` imports) and `temp_gap_filing_exploration copy.ipynb` (all extended exploration in §1-§11
below). §12 covers a separate, third notebook, `temp_gap_filling_pipeline.ipynb` — a tidied/
condensed rebuild (EDA → feature imputation → feature engineering → gap-filling → UQ) kept as an
ongoing base for further experimentation; its own results are independent of §1-§11's.
**Scope:** all 3 towers (T2, T4, T9) throughout, full-period gap-CV (D-35/D-49 methodology).

This document consolidates every metric produced across all three notebooks into one reference. It
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

---

## 12. `temp_gap_filling_pipeline.ipynb` — full comparison: MDS floor through every rework + TabICL

Separate, condensed notebook (its own §0-11 internal structure: EDA → feature imputation →
feature engineering → gap-filling → UQ), built as a tidier base for iterative experimentation.
Its own champion reproduces §1's architecture (partial-pooled RFm, full-period gap-CV) and lands
at the same numbers: **T2 0.576, T4 0.404, T9 0.426**. This section lines up *every* experiment run
in this notebook against two fixed reference points — **MDS** (the 2-driver literature baseline,
the floor) and the **RFm champion** (the ceiling reached so far) — including two more
model-agnostic statistical baselines, **Mean** and **MICE**, alongside RF/TabICL — then reruns the
feature-set reworks through **TabICL** instead of RF (§12.2).

### 12.1 Master comparison — every RF experiment vs. the MDS floor and the champion

R²/RMSE/MAE/nMAE all reported as T2 / T4 / T9 (RMSE/MAE in the target's native units,
nmol m⁻² s⁻¹; lower is better for RMSE/MAE/nMAE, unlike R²). "Δ vs MDS" and "Δ vs champion" are
both **R²** deltas (positive = better than that reference), reported the same way. Rows are
ordered by increasing model/feature sophistication, not by performance, so the table reads as a
build-up from nothing to everything tried.

**nMAE = MAE / std(observed FCH4)**, one fixed constant per tower computed from all real,
QC'd, in-domain target values (T2 std=140.9, T4 std=128.5, T9 std=146.0 nmol m⁻² s⁻¹) — the same
denominator is applied to every experiment below, so nMAE columns are directly comparable across
rows. Chosen over range-based normalization specifically because FCH4's range (~3200-3350) is set
by a single extreme spike per tower, which would compress every experiment's nMAE into a
near-meaningless ~0.01-0.02 band; std is far less distorted by 1-2 outlier points (same
spike-sensitivity reasoning behind `CLAUDE.md`'s MASE-first guidance for the forecasting track).

**MICE implementation note (bug caught and fixed before results were trusted)**: pooling stacks 3
towers that all share the same underlying hourly `Datetime` index. A first attempt used
label-based `.loc[gt_ts]` to read the imputed target back out at the held-out timestamps — on the
concatenated matrix this matched *every* tower's row at that timestamp, not just the tower under
evaluation (`ValueError: Found input variables with inconsistent numbers of samples` surfaced this
immediately in a smoke test, before any real compute was spent). Fixed by tracking each held-out
point's absolute *row position* in the concatenated matrix instead of relying on its (non-unique)
timestamp label.

| Experiment | Features (+3 dummies) | R² | RMSE | MAE | nMAE | Δ R² vs MDS | Δ R² vs champion |
|---|---|---|---|---|---|---|---|
| **Mean** (training-set constant — no model, no features, the trivial floor) | 0 | -0.003 / -0.001 / -0.001 | 139.5 / 125.6 / 138.0 | 46.8 / 50.7 / 61.5 | 0.332 / 0.395 / 0.421 | +0.164 / +0.206 / +0.336 | -0.579 / -0.405 / -0.427 |
| **MDS** (2-driver, SW_IN+TA — literature baseline, the floor) | -- | -0.167 / -0.207 / -0.337 | 158.8 / 138.5 / 155.3 | 51.4 / 56.3 / 69.3 | 0.365 / 0.438 / 0.475 | +0.000 / +0.000 / +0.000 | -0.743 / -0.611 / -0.763 |
| RFm met-only (raw met/micromet drivers only — no AUX, no livestock, no lags, no mgmt, no gpp/reco) | 11 | 0.052 / 0.036 / 0.059 | 135.8 / 119.5 / 133.1 | 43.1 / 50.4 / 59.6 | 0.306 / 0.392 / 0.408 | +0.219 / +0.243 / +0.396 | -0.524 / -0.368 / -0.367 |
| MICE (`sklearn.IterativeImputer`, default `BayesianRidge`, champion's own `FEATURES`) | 30 | 0.081 / 0.118 / 0.107 | 132.6 / 114.6 / 130.4 | 59.6 / 55.9 / 61.7 | 0.423 / 0.435 / 0.423 | +0.248 / +0.325 / +0.444 | -0.495 / -0.286 / -0.319 |
| **RFm champion** (`lsu_dens`, `FEATURES` — the ceiling) | 30 | 0.576 / 0.404 / 0.426 | 75.0 / 100.3 / 107.1 | 31.2 / 42.8 / 49.3 | 0.221 / 0.333 / 0.338 | +0.743 / +0.611 / +0.763 | +0.000 / +0.000 / +0.000 |
| 4.3.a speciesDens (cattle/sheep/lamb split, replaces `lsu_dens`) | 32 | 0.593 / 0.404 / 0.428 | 74.5 / 99.5 / 106.6 | 31.6 / 42.8 / 49.1 | 0.224 / 0.333 / 0.336 | +0.760 / +0.611 / +0.765 | +0.017 / +0.000 / +0.002 |
| D1 lagmemSoil (derived point-lag + causal rolling TS/SWC) | 30 | 0.576 / 0.404 / 0.428 | 75.4 / 100.2 / 107.3 | 31.1 / 42.8 / 49.0 | 0.221 / 0.333 / 0.336 | +0.743 / +0.611 / +0.765 | +0.000 / +0.000 / +0.002 |
| D2 livestockMem (species-split + per-species rolling mean + grazing recency) | 36 | 0.601 / 0.395 / 0.425 | 74.9 / 99.8 / 107.0 | 31.4 / 42.6 / 49.5 | 0.223 / 0.331 / 0.339 | +0.768 / +0.602 / +0.762 | +0.025 / -0.009 / -0.001 |
| D3 neighborNoAug (nearest-observation target features) | 38 | 0.554 / 0.267 / 0.308 | 91.8 / 104.9 / 113.7 | 37.8 / 45.2 / 52.9 | 0.268 / 0.352 / 0.362 | +0.721 / +0.474 / +0.645 | -0.022 / -0.137 / -0.118 |
| D3 neighborAug (+training-time neighbour-feature augmentation) | 38 | 0.537 / 0.382 / 0.386 | 82.3 / 99.8 / 110.0 | 32.5 / 44.8 / 51.2 | 0.231 / 0.349 / 0.351 | +0.704 / +0.589 / +0.723 | -0.039 / -0.022 / -0.040 |
| D4 laglead_full, original (bidirectional lag+lead at 10 hours [1,2,3,6,24,48,168,336,504,672] for TS, SWC, species-split rolling LSU, and target) | 141 | 0.510 / 0.318 / 0.292 | *not retained¹* | *not retained¹* | *not retained¹* | +0.677 / +0.525 / +0.629 | -0.066 / -0.086 / -0.134 |
| D4 laglead_full, **revised additive** (original D4 + every column unique to D1/D2/D3 that D4 lacked: D1's pruned point-lags/rolls, D2's raw species density + plain roll + grazing recency, D3's nearest-observation `NEIGHBOR_COLS`) | 164 | 0.543 / 0.262 / 0.293 | 94.1 / 102.5 / 114.6 | 37.9 / 42.9 / 51.6 | 0.269 / 0.334 / 0.353 | +0.710 / +0.469 / +0.630 | -0.033 / -0.142 / -0.133 |

¹ The original (141-feature) D4's RMSE/MAE were computed and briefly displayed in-notebook, but
that cell was overwritten in place by the revised-D4 rework (per this notebook's usual "revise in
place, don't fork a new section" pattern for this specific request) before being captured to a
saved file — only its R² survived (captured to chat/this document at the time). Not recoverable
without rerunning the original 141-feature version specifically.

**Reading the table as a build-up**: **Mean** (fill every held-out point with that fold's own
training-set constant — literally no model) lands at R²≈0 at every tower, exactly as the
definition predicts — the honest zero-information floor. **MDS actually sits *below* that** at
every tower (negative R²) — the literature 2-driver baseline is measurably *worse than doing
nothing*. Switching to RF with the same class of information (11 raw met/micromet drivers, zero
engineering) already climbs past both floors to a small positive R² (0.04-0.06) and a large
RMSE/MAE improvement over MDS — confirming most of MDS's weakness is algorithm choice, not driver
choice. **MICE** (`IterativeImputer`, default `BayesianRidge`) — given the *exact same 30 features*
as the champion — reaches R²=0.08-0.12, ahead of met-only RF (fewer features but a nonlinear
model) yet far behind the champion (identical features, RF instead of linear chained regression).
Comparing MICE-vs-champion (same features, different model) against met-only-vs-champion (same
model, fewer features) separates the two axes cleanly: neither the richer feature set nor RF's
nonlinearity alone gets you to the champion's 0.4-0.6 — you need both together. The jump from
met-only RF to the champion (+0.37 to +0.52 R², depending on tower) is entirely attributable to
the champion's engineered features (livestock density, soil lags, management recency, GPP/Reco,
diurnal/seasonal encoding) — this is the first time this notebook has quantified how much of the
champion's skill is "raw weather" versus "everything else." None of the six further reworks beats
the champion outright.
4.3.a and D2 come closest (small T2 gains, real T4 cost); D3 and D4 — both touching the target
itself — regress hardest, still comfortably ahead of met-only RF but clearly behind the champion.
Root cause diagnosed for D4: `RF_PARAMS` never caps `max_features`, so every split considers all
candidate features (undermining RF's tree-decorrelation mechanism), compounded by heavy
multicollinearity between adjacent-hour lag columns and near-in-time target lags going NaN
(mean-imputed) inside exactly the larger held-out gaps where they'd matter most. **The additive
revision (164 features) directly tested whether this was fixable by "just add more relevant
signal" — it wasn't**: T2 improved slightly (0.510→0.543) but T4 got meaningfully worse
(0.318→0.262), net evidence that the dilution problem scales with column count regardless of how
individually well-motivated each addition is, rather than being specific to D4's original wide
hour grid.

### 12.2 Same five feature sets, rerun through TabICL instead of RF

RF rows here repeat §12.1's numbers for direct side-by-side comparison; MDS/met-only aren't rerun
through TabICL (they're floor references, not feature-set reworks) — see §12.1 for that context.

R²/RMSE/MAE/nMAE all reported as T2 / T4 / T9. **Bold** marks whichever of RF/TabICL wins that
specific metric at that specific tower (higher for R², lower for RMSE/MAE/nMAE) — bolding is
per-cell, not per-row, since the metrics don't always agree on a winner. nMAE uses the same
per-tower std-normalization as §12.1 (T2 std=140.9, T4 std=128.5, T9 std=146.0), so it always
tracks MAE's win/loss pattern exactly (same divisor per tower) — included for direct comparability
with §12.1's rows, not because it can flip a verdict MAE already gave. RMSE/MAE/nMAE not retained
for the original 141-feature D4 (see §12.1 footnote ¹) — its RF/TabICL rows are R²-only.

| Experiment | Model | R² | RMSE | MAE | nMAE | Verdict |
|---|---|---|---|---|---|---|
| D1 lagmemSoil | RF | 0.576 / 0.404 / 0.428 | 75.4 / 100.2 / 107.3 | 31.1 / 42.8 / 49.0 | 0.221 / 0.333 / 0.336 | -- |
| D1 lagmemSoil | TabICL | 0.561 / **0.414** / 0.372 | **75.3** / 102.9 / 108.4 | **30.3** / **42.4** / 50.5 | **0.215** / **0.330** / 0.346 | edges RF's R² at T4 only; mixed on RMSE/MAE/nMAE |
| D2 livestockMem | RF | 0.601 / 0.395 / 0.425 | 74.9 / 99.8 / 107.0 | 31.4 / 42.6 / 49.5 | 0.223 / 0.331 / 0.339 | -- |
| D2 livestockMem | TabICL | 0.567 / **0.411** / 0.377 | **73.7** / 102.6 / 109.0 | **30.9** / **42.3** / 51.2 | **0.219** / **0.329** / 0.351 | edges RF's R² at T4 only; mixed on RMSE/MAE/nMAE |
| D3 neighborNoAug | RF | 0.554 / 0.267 / 0.308 | 91.8 / 104.9 / 113.7 | 37.8 / 45.2 / 52.9 | 0.268 / 0.352 / 0.362 | -- |
| D3 neighborNoAug | TabICL | **0.570** / **0.323** / **0.350** | **90.1** / **101.6** / 114.1 | **37.1** / **41.7** / 55.1 | **0.263** / **0.324** / 0.377 | beats RF's R² at all 3 towers; RMSE/MAE/nMAE agree at T2/T4 but reverse at T9 |
| D3 neighborAug | RF | 0.537 / 0.382 / 0.386 | 82.3 / 99.8 / 110.0 | 32.5 / 44.8 / 51.2 | 0.231 / 0.349 / 0.351 | -- |
| D3 neighborAug | TabICL | **0.542** / **0.393** / 0.375 | 83.9 / 100.3 / 110.7 | **31.9** / **43.9** / 51.8 | **0.226** / **0.342** / 0.355 | edges RF's R² at T2/T4; RMSE/nMAE slightly worse everywhere despite that |
| D4 laglead_full, original (141 feat) | RF | 0.510 / 0.318 / 0.292 | *n/a* | *n/a* | *n/a* | -- |
| D4 laglead_full, original (141 feat) | TabICL | **0.514** / 0.259 / 0.285 | *n/a* | *n/a* | *n/a* | mixed, TabICL edges T2 R² only |
| D4 laglead_full, revised additive (164 feat) | RF | 0.543 / 0.262 / 0.293 | 94.1 / 102.5 / 114.6 | 37.9 / 42.9 / 51.6 | 0.269 / 0.334 / 0.353 | -- |
| D4 laglead_full, revised additive (164 feat) | TabICL | 0.405 / 0.228 / 0.239 | 103.9 / 104.4 / 120.3 | 38.8 / **42.0** / 53.1 | 0.275 / **0.327** / 0.364 | loses to RF on R²/RMSE at all 3 towers; only T4 MAE/nMAE is marginally better |

RMSE/MAE and R² don't always agree on which model "wins" at a given tower (e.g. D3 neighborNoAug's
T9: TabICL has the better R² but the worse RMSE; D4 revised's T4: TabICL has the worse R² but a
marginally better MAE) — a reminder that R² alone can be misleading when FCH4's spike-dominated
error distribution is at play (see `CLAUDE.md`'s MASE-first guidance for the forecasting track;
this gap-filling notebook doesn't compute MASE, but the same spike-sensitivity caveat applies here).

Neither model rescues a feature set that doesn't help RF — the one partial exception is
D3-neighborNoAug, where TabICL modestly outperforms RF on the identical (weaker) feature set at
every tower, though both still trail the champion. TabICL's runtime is far more stable across
feature-set size than RF's: at the original D4's 141 features, TabICL took ~70-85s/tower vs. RF's
~30-37 min/tower (cold cache) — TabICL's in-context cost is dominated by its fixed 10,000-row cap,
not column count, while RF's uncapped `max_features` makes its split search scale with feature
count. That said, going from 141→164 features (+`NEIGHBOR_COLS`) cost TabICL its one T2 edge over
RF — column growth eventually hurts TabICL too, just via attention dilution rather than split-search
dilution, and by less than it hurts RF in absolute terms.

### 12.3 Feature-set audit + saved training data

Every experiment's exact `X_train` column list is printed directly in-notebook from the live
feature-list variables (not hand-transcribed), and the corresponding pooled, real-data-only
training frame (target + that experiment's features + tower + Datetime, no synthetic gaps
withheld — the same construction as the notebook's own production fit) is saved per feature set.
`laglead_full_FEAT_LAGLEAD_FULL.csv` reflects the **revised** (164-feature, additive) D4 — it was
updated in place, not renamed, so it no longer matches the original 141-feature D4 row in §12.1/12.2.

For D4 specifically, a second, richer export exists: `laglead_full_v2_train_test_missing_combined.csv`
(104,355 rows × 168 cols, all 3 towers, revised 164-feature set) labels every domain row with this
project's own standing temporal split (`CLAUDE.md`: train=2018-2021, test=2022-2023,
held_out_2024=2024 — empty, no tower's domain reaches 2024) plus `missing` for genuine field gaps
(the actual gap-filling deliverable) and `other` for the small pre-2018 (2017) sliver. Split into
one file per category under `_data_gf_training/laglead_full_v2_split/`: `train.csv` (21,998 rows),
`test.csv` (12,949), `to_gapfill.csv` (68,769), `other_pre2018.csv` (639).

**`_data/`** (this notebook): `reference_baseline_met_only.csv` (§12.1 Mean + MDS + RFm met-only +
MICE + champion floor comparison), `lagmem_results_4_3_b.csv` (§12.1 champion/4.3.a/D1/D2/D3
consolidated), `tabicl_vs_rf_results.csv` (§12.2 D1/D2/D3), `laglead_full_results_d4.csv` (§12.1
D4, revised), `tabicl_vs_rf_d4.csv` (§12.2 D4, revised).

**`_data_gf_training/`**: `champion_FEATURES.csv`, `speciesDens_FEAT_SPECIES_DENS.csv`,
`lagmemSoil_FEAT_LAGMEM_SOIL.csv`, `livestockMem_FEAT_LIVESTOCK_MEM.csv`,
`neighbor_FEAT_NEIGHBOR.csv`, `laglead_full_FEAT_LAGLEAD_FULL.csv` (revised) — one pooled training
frame per feature set above — plus `laglead_full_v2_train_test_missing_combined.csv` and the
`laglead_full_v2_split/` subfolder described above.

### 12.4 Raw predictions for every experiment (not just the champion)

Every experiment above (§12.1/12.2) previously kept only the *aggregated* median-of-medians
R²/RMSE/MAE/MBE per (tower, scenario) — the raw `(timestamp, actual, predicted)` pairs behind each
fold were computed, fed into `mets()`, and discarded. `_data/all_raw_predictions.csv` captures them
for real: **1,517,760 rows × 8 columns** (`experiment`, `model`, `tower`, `scenario`, `rep`,
`Datetime`, `y_actual`, `y_pred`), 113.1 MB, covering all 3 towers × 5 scenarios × 2 reps for every
one of **19** experiment/model combinations — Mean, MDS, RFm_solo, RFm_pool champion, RFm met-only,
MICE, 4.3.a speciesDens, D1/D2/D3(×2 arms)/D4, each RF and (where applicable) TabICL. (Mean/MICE
were added after the file's initial build — same "extend the raw-predictions capture as new
experiments are added, not a separate later ask" convention as everything else in this section.)

**Verification**: since `insert_calendar_gaps` uses a fixed `seed=0` by default (every call site in
this notebook relies on that), the held-out fold structure is fully deterministic, so RF fits still
hit `_model_cache/` and reproduce byte-identical results to the original runs. Recomputing R² from
this exported file alone — correctly, per (tower, scenario, rep) then median across reps then
median across scenarios, exactly matching `mets()`/`med_metrics()`'s own methodology — reproduces
the established headline numbers exactly for 15 of the 17 combinations (e.g. champion: T2/T4/T9 =
0.576/0.404/0.426, an exact match).

**Known caveat**: the two **D3 neighborAug** rows (RF and TabICL) show small T4/T9 deviations from
the numbers quoted in §12.1/12.2 (e.g. RF: T4/T9 = 0.359/0.385 here vs. 0.382/0.386 established).
Root cause: `augmented_mask`'s seed increments continuously across all 5 scenarios within
`run_rf_neighbor`'s single original call, but restarts at a fixed base each time the "capture"
harness is invoked separately per scenario — a genuinely different (still valid, still
leakage-safe) random training-augmentation draw, not a data-integrity problem. Every other
combination matches exactly.

**A related fix, applied retroactively**: `run_rf_capture` (used by §9/§10 for the champion's own
`held_out_pairs_l_scenario.csv`/`held_out_pairs_all_scenarios.csv`) had this same missing-`rep`-tag
gap — duplicate/overlapping timestamps between the 2 independent reps (confirmed empirically: 372
of 2565 rows for Tower 2's "l" scenario alone) meant a naive pooled R² across both reps together
silently conflated two different fits, most visibly at Tower 2 (its shorter `DOMAIN` window leaves
less room for `insert_calendar_gaps`'s 2 independent draws to avoid overlapping). Patched to tag
every row with its `rep` — both champion files now carry this column too.

**Not yet reflected in project-level docs** (`DECISIONS.md`/`BEST_RESULTS.md`/`CONTEXT.md`) — same
caveat as §1-§11's closing note above.

---

## 13. `temp_mds.ipynb` — literature-correct MDS reconstruction + an R² metric-definition discovery

A fourth, separate self-contained notebook (edits confined to it; `temp_gap_filling_pipeline.ipynb`
read-only reference throughout). Built to audit the project's MDS implementation against the actual
Reichstein (2005)/REddyProc algorithm — the primary paper isn't in `documents/Literature/`, so
verification used direct inspection of REddyProc's own R source (`EddyGapfilling.R`) and MPI-BGC's
official documentation instead.

### 13.1 The algorithm audit — three confirmed bugs, not one

The current production `mds_fill_batch` deviates from the real MDS algorithm in three ways: (1) it
applies an hour-of-day ±1h restriction to *every* window/case, when the real algorithm only
restricts by hour in the final meteo-free fallback (Case 3, mean diurnal course) — the
meteorological look-up cases (1: SW+TA+VPD, 2: SW-only) should use ALL records in the day-window;
(2) it is missing the intermediate Case 2 (SW-only look-up) entirely, jumping straight from
"all 3 drivers required" to a crude fallback; (3) that fallback is a single fixed ±7-day box with no
meteorological constraint, not the real algorithm's expanding mean-diurnal-course window (0/1/2 days,
then step-7 to ±210 days). Confirmed mechanistically, not just asserted: instrumented diagnostic
variants (with an exhaustive predictions-match assertion, not a single spot-check) showed candidate
pools were chronically tiny (median 2-4 real observations) under the old implementation, across
*every* scenario and tower — and grew 2-4x once the hour restriction was removed.

### 13.2 OLD vs NEW implementation — headline R² (sklearn convention, this project's standing metric)

| Tower | OLD (current production) | NEW (literature-correct) | Δ |
|---|---|---|---|
| T2 | -0.214 | **-0.023** | +0.191 |
| T4 | -0.302 | **-0.113** | +0.189 |
| T9 | -0.293 | **-0.073** | +0.220 |

R² improved in **every single (tower, scenario) cell**, no exceptions. Tower 2's long-gap ("l",
288h) scenario improved (-2.457 → -1.947) but remains by far the worst cell in either version —
consistent with, though not fully explained by, Tower 2's short ~21-month domain (see §13.3).

### 13.3 Site-identity discovery: Tower 2/4/9 ARE Zhu et al. (2023a)'s named challenging-ecosystem sites

Cross-referencing `Zhu_2023a_...pdf` against `NWFP_UG_Design_Develop.pdf` (farmlet map, land-
management-change timeline) confirms — not "similar ecosystems" but the same physical fields:

| This project | NWFP farmlet | Zhu et al. 2023a site |
|---|---|---|
| Tower 2 (Catchment 2, "Great Field") | Red | ROTH_HS |
| Tower 4 (Catchment 4, "Burrows") | Green | ROTH_PP |
| Tower 9 (Catchment 9, "Dairy South") | Blue | ROTH_HSC |

Confirmed via three independent lines of evidence: the farmlet map's colour-coding; Tower 2's
2018-with-cattle/2019-without-cattle regime shift being literally the "Red farmlet transitioned to
an arable system... cattle permanently housed... sheep production only on green and blue systems"
event the design guide describes for April 2019 (not an independently-occurring parallel case); and
this project's own `Catchment 4 After 2013/08/13` column suffix quoting the design guide's §5.1
almost verbatim. This project's `SCENARIOS = {vs, s, m, l, m1}` gap-CV taxonomy also directly mirrors
Zhu et al.'s own Part B gap-scenario naming — the comparison against their Fig. 6d/Table 7d figures
is a same-site, same-methodology comparison, not an analogous one.

### 13.4 The R² metric-definition discovery

Zhu et al. state their R² explicitly (p.6): *"...the coefficient of determination (R²) and slope of
the ordinary least squares regression..."* — R² and slope together, both from an OLS fit of
(measured, filled) pairs. That R² is mathematically the **squared Pearson correlation coefficient**,
bounded in [0, 1] by construction. This project's `mets()` instead uses
`sklearn.metrics.r2_score(y, p)` — `1 - SS_res/SS_tot`, **unbounded below**, punishing any systematic
bias/scale mismatch hard. These are two different statistics that happen to share a name.

Recomputed on the exact same predictions with both definitions:

| Tower | OLD / sklearn | OLD / OLS (Zhu-style) | NEW / sklearn | NEW / OLS (Zhu-style) |
|---|---|---|---|---|
| T2 | -0.214 | 0.015 | -0.023 | **0.072** |
| T4 | -0.302 | 0.039 | -0.113 | **0.038** |
| T9 | -0.293 | 0.048 | -0.073 | **0.054** |

Under Zhu et al.'s own metric, **both the OLD and NEW implementations already land close to their
published ~0.03-0.05 figure** for these exact sites — strong independent validation that this
reconstruction reproduces the literature number on the literal same sites once measured the same
way. The algorithmic fix's benefit under this metric is real but uneven across towers: meaningful
for T2 (0.015→0.072), essentially flat for T4 (0.039→0.038), small for T9 (0.048→0.054) — a
different, more modest picture than the sklearn-R² comparison alone suggests.

### 13.5 Both R² definitions, extended to every model this project has evaluated

Using the existing `_data/all_raw_predictions.csv` (read-only; produced by
`temp_gap_filling_pipeline.ipynb` §12.4 — no retraining needed). **Important methodology note**:
`experiment` alone is not a unique key — D1-D4 and `speciesDens` each have both an RF and a TabICL
row for the same held-out points, so `model` must be included in any groupby over this file or two
different models' predictions get silently pooled together (caught during this check, fixed before
these numbers were used for anything).

| Experiment | Model | R²_sklearn (T2 / T4 / T9) | R²_OLS, Zhu-style (T2 / T4 / T9) | Mean Δ (OLS − sklearn) |
|---|---|---|---|---|
| Mean (trivial baseline) | — | -0.003 / -0.001 / -0.001 | 0.000 / 0.000 / 0.000 | +0.001 |
| **MDS (old, current production)** | — | -0.214 / -0.302 / -0.293 | 0.015 / 0.039 / 0.048 | **+0.304** |
| **MDS (new, literature-correct — §13.2 fix)** | — | -0.023 / -0.113 / -0.073 | 0.072 / 0.038 / 0.054 | **+0.124** |
| MICE | — | 0.081 / 0.118 / 0.107 | 0.104 / 0.128 / 0.110 | +0.012 |
| RFm met-only | RF | 0.052 / 0.036 / 0.059 | 0.054 / 0.042 / 0.065 | +0.004 |
| RFm_solo | RF | 0.398 / 0.382 / 0.356 | 0.445 / 0.393 / 0.358 | +0.020 |
| **RFm champion (RFm_pool)** | RF | **0.576 / 0.404 / 0.426** | 0.601 / 0.409 / 0.430 | +0.012 |
| speciesDens | RF | 0.593 / 0.404 / 0.428 | 0.615 / 0.408 / 0.431 | +0.010 |
| D1 lagmemSoil | RF | 0.576 / 0.404 / 0.428 | 0.604 / 0.409 / 0.435 | +0.014 |
| D1 lagmemSoil | TabICL | 0.561 / 0.414 / 0.372 | 0.596 / 0.417 / 0.377 | +0.014 |
| D2 livestockMem | RF | 0.601 / 0.395 / 0.425 | 0.622 / 0.399 / 0.428 | +0.009 |
| D2 livestockMem | TabICL | 0.567 / 0.411 / 0.377 | 0.593 / 0.420 / 0.379 | +0.012 |
| D3 neighborAug | RF | 0.537 / 0.359 / 0.385 | 0.562 / 0.372 / 0.403 | +0.019 |
| D3 neighborAug | TabICL | 0.542 / 0.390 / 0.347 | 0.582 / 0.401 / 0.355 | +0.020 |
| D3 neighborNoAug | RF | 0.554 / 0.267 / 0.308 | 0.562 / 0.303 / 0.353 | +0.030 |
| D3 neighborNoAug | TabICL | 0.570 / 0.323 / 0.350 | 0.607 / 0.375 / 0.368 | +0.036 |
| D4 laglead_full (revised) | RF | 0.543 / 0.262 / 0.293 | 0.558 / 0.304 / 0.351 | +0.038 |
| D4 laglead_full (revised) | TabICL | 0.405 / 0.228 / 0.239 | 0.444 / 0.288 / 0.298 | **+0.053** |

**The R²-definition sensitivity is essentially an MDS-specific phenomenon, not a general property
of this project's evaluation.** Every RF- and TabICL-based model (the champion included — a
difference of ~0.02-0.03, not ~0.3), MICE, and the trivial Mean baseline all show small, consistent
gaps between the two definitions (+0.001 to +0.053). **MDS (old) is the outlier by roughly an order
of magnitude** (+0.304); even the literature-correct MDS fix (+0.124) is still 2-6x any other
model's gap. Mechanistic reason: RF/TabICL/MICE are all fit to directly minimize error against the
target, so their held-out predictions are already close to an OLS-calibrated fit — the two
definitions can't diverge much for them by construction. MDS's "mean of similar-condition historical
observations" rule is never calibrated against the specific held-out fold at all — it can be
genuinely *correlated* with truth (nonzero Pearson r, real physical analogs carry real information)
while being systematically *miscalibrated* in scale/bias for any given fold (poor out-of-sample R²).

**Practical implication**: this project's standing metric convention (sklearn R², used throughout
`temp_gap_filling_pipeline.ipynb` and `BEST_RESULTS.md`) is **not** distorting the RFm-vs-MDS
"improvement over MDS" framing (D-34/D-35) — RFm's own number barely moves under the alternative
definition, so the ~0.6-1.0 R²-unit gap this project has repeatedly reported is real under either
metric. The definition choice matters specifically when comparing this project's *own* MDS numbers
against *external* literature figures computed the other way — exactly the comparison this notebook
exists to make.

### 13.6 Curiosity extension (explicitly "not real MDS"): feature-richer window-matching

Two exploratory variants, built on top of the literature-correct MDS engine, isolating each change
to Case 1 only (Case 2/3 held identical throughout for a clean comparison):

| Variant | T2 | T4 | T9 | Verdict |
|---|---|---|---|---|
| driver3 (literature-correct, §13.2) | -0.023 | -0.113 | -0.073 | baseline for this comparison |
| driver_m (+8 more met variables: PPFD, RN, WS, USTAR, SHF, precip, soil T, soil moisture) | -0.161 | -0.211 | -0.287 | worse at all 3 towers |
| driver_full (driver_m + livestock stocking density + grazing exact-match + management recency) | -0.165 | -0.227 | -0.189 | worse than driver3 everywhere; mixed vs. driver_m |

**driver_m makes things worse everywhere** — a curse-of-dimensionality effect: requiring
simultaneous tolerance-matching across 11 variables (even soft-matched, skipping missing ones)
shrinks candidate pools further (e.g. Tower 2 "l": median 4→3 candidates) despite the extra
information, because window-matching (unlike RF) has no mechanism to weight informative vs.
uninformative dimensions.

**driver_full is tower-dependent** — helps Tower 9 (+0.098 vs. driver_m; Tower 9 is grazed 61.4% of
hours, a balanced split) but leaves Tower 2 flat/worse (+0.004 vs. driver_m, still worse than
driver3) because Tower 2 is grazed only 11.9% of hours (median LSU/ha = 0) — an exact-match
constraint on a rare state shrinks an already-scarce candidate pool faster than the signal can
compensate. **This is evidence against "MDS can't see livestock" being a sufficient, standalone
explanation for Tower 2's residual long-gap collapse specifically** (the DECISIONS.md D-34
hypothesis) — giving a match-based method the exact signal D-34 flags as missing doesn't recover
the loss at Tower 2, even though it demonstrably helps at the more balanced Tower 9. Sheer data
scarcity/imbalance in Tower 2's short domain looks like the more fundamental bottleneck.

### 13.7 Port-back status

Not yet ported into `temp_gap_filling_pipeline.ipynb` — `temp_mds.ipynb`'s §8 has exact
instructions (copy `mds_fill_batch_new` in as `mds_fill_batch`; add a `fill_fn`/`data` default-arg
generalization to `run_mds`/`run_mds_capture`, backward-compatible with existing call sites). The
driver_m/driver_full experiments (§13.6) are explicitly exploratory, not literature MDS, and net
negative for the audit's actual goal — not recommended for port-back.

**Not yet reflected in project-level docs** (`DECISIONS.md`/`BEST_RESULTS.md`/`CONTEXT.md`) — same
caveat as §1-§12's closing notes above. If ported, `BEST_RESULTS.md`'s "improvement over MDS" figure
and DECISIONS.md's D-34 framing would both need updating to cite the corrected MDS baseline and this
section's counter-evidence on the livestock-blind hypothesis.

Source: computed live in `temp_mds.ipynb` (§1-11; no new saved artifacts) plus
`_data/all_raw_predictions.csv` (§13.5, existing file, read-only).
