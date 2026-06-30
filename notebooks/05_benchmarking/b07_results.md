# B-07 — spike-classifier diagnostics, recency features, early-warning threshold analysis (D-44)

**Notebook:** `B07_spike_diagnostics.ipynb`. **Results:** `b07_summary.csv`; `B07` rows in `benchmarks.csv`
(108 rows: RF/XGB/Hurdle-RF/Hurdle-XGB x recency features, both tracks, all horizons, towers 4/9 + Tower-2
folds); `results/figures/b07_pr_curve.png`.

Follow-up to B-06 (two-stage hurdle, D-43), whose root cause was diagnosed as the spike classifier's **low
precision** (0.25-0.42 daily) rather than its recall (0.58-0.93, already high). Three steps: (1) diagnose what
the classifier's false positives/negatives actually look like, (2) add leak-free recency/clustering features
and retest the full B-06 harness regardless of the diagnostic outcome (per agreed scope), (3) reframe the
classifier's good recall as a standalone early-warning signal via a precision-recall threshold analysis.

## 1  Diagnostic — false positives are context-indistinguishable from true positives

At representative combos (daily h=1/h=14, hourly h=1/h=24, Towers 4/9, both algos; 55,928 diagnostic rows):

| daily, mean by outcome | precip | days_since_grazing | is_growing |
|---|---|---|---|
| TP (true spike, caught) | 0.60 | 34.0 | 0.96 |
| **FP (false alarm)** | 0.59 | 33.9 | 0.92 |
| FN (missed spike) | 0.60 | 38.4 | 0.70 |
| TN (correct quiet) | 0.64 | 16.8 | 0.48 |

False positives are **statistically indistinguishable from true positives** on precipitation, grazing recency,
and growing-season flag — both sit deep in the growing season with similar grazing recency. The contrast that
exists is TP/FP vs TN (growing-season flag 0.92-0.96 vs 0.48; days-since-grazing ~34 vs ~17) — the classifier
has correctly learned "growing season + recent-ish grazing = elevated risk," but that same signal applies to
roughly twice as many quiet days as spike days, so it can't discriminate further with the available features.

False-positive rate is sharply seasonal: **0% Nov-Mar, 21-38% Jun-Aug** (daily track) — entirely consistent
with the growing-season-driven signal above, not a separate artefact.

**Implication:** this is exactly the signal recency/clustering features were hypothesised to add — the
diagnostic doesn't contradict building them, it just clarifies that the missing ingredient is event-level
timing within the growing season, not season identification (already captured by `fx_is_growing`).

## 2  Recency features — leak-free, verified causal

`ar_days_since_spike`, `ar_spike_count_<w>`, `ar_rolling_max_<w>` (daily w=7/28, hourly w=24/168), computed from
each tower's gap-filled CH4 series against the same frozen q90 threshold as B-06. Causality spot-check on
Tower 4 daily: `ar_days_since_spike` only fails the monotonic-or-reset check on the 4 origin rows before the
very first spike in the 2017 warm-up period (still-9999 plateau) — expected, not a leak. No other violations.

## 3  Retest — marginal, inconsistent; does not flip the B-06 verdict

### Plain RF/XGB (non-hurdle): daily R² changes are within noise

| daily R², RF | B-03 | B-07 +recency | daily R², XGB | B-03 | B-07 +recency |
|---|---|---|---|---|---|
| T4 h=1 | 0.357 | 0.359 (+0.002) | T4 h=1 | 0.362 | 0.383 (+0.021) |
| T4 h=14 | 0.270 | 0.282 (+0.012) | T4 h=14 | 0.259 | 0.269 (+0.010) |
| T9 h=1 | 0.388 | 0.375 (-0.013) | T9 h=1 | 0.324 | 0.316 (-0.008) |
| T9 h=14 | 0.342 | 0.340 (-0.002) | T9 h=14 | 0.275 | 0.276 (+0.001) |

No consistent direction — half the cells improve, half degrade, all by <0.02. Hourly h=1 shows a slightly
larger positive bump for Tower 4 (RF +0.026, XGB +0.034) but Tower 9 moves the other way at the same horizon
(RF -0.014, XGB -0.016) — tower-inconsistent, not a generalizable signal.

### Hurdle architecture + recency: precision improves modestly, but R² stays mixed

Daily classifier precision rose from B-06's 0.25-0.42 to **0.26-0.44** with recency features — a real but small
improvement, consistent with the diagnostic's finding that growing-season/grazing-recency context (which the
new features correlate with via `ar_rolling_max`/`ar_spike_count`) was already close to fully exploited.

| Hurdle-RF daily R² | B-06 | B-07 +recency | Hurdle-XGB daily R² | B-06 | B-07 +recency |
|---|---|---|---|---|---|
| T4 h=1 | 0.253 | 0.258 (+0.005) | T4 h=1 | 0.179 | 0.131 (-0.048) |
| T4 h=14 | 0.154 | 0.141 (-0.013) | T4 h=14 | -0.078 | -0.035 (+0.043) |
| T9 h=1 | 0.130 | 0.145 (+0.015) | T9 h=1 | -0.280 | -0.353 (-0.073) |
| T9 h=14 | 0.059 | 0.067 (+0.008) | T9 h=14 | -0.431 | -0.348 (+0.083) |

Short-horizon Hurdle-RF nudges up slightly at both towers; Hurdle-XGB moves in both directions by tower and
horizon with no discernible pattern. **None of these reach B-03's single-regressor numbers** — the hurdle
architecture (with or without recency features) remains below the production config at every daily
tower/horizon.

## 4  Early-warning threshold analysis — the one genuinely useful artefact from this round

Reframing the daily classifier as a "flag elevated-emission-risk day" signal at a recall>=0.8 operating point
(deliberately favouring recall, since false alarms are cheaper than missed events for farm management):

| horizon | algo | tower | threshold | recall | precision |
|---|---|---|---|---|---|
| 1d | RF | 4 | 0.459 | 0.806 | 0.425 |
| 1d | RF | 9 | 0.513 | 0.814 | 0.357 |
| 14d | RF | 4 | 0.498 | 0.806 | 0.391 |
| 14d | RF | 9 | 0.397 | 0.814 | 0.280 |
| 1d | XGB | 4 | 0.675 | 0.806 | 0.432 |
| 14d | XGB | 4 | 0.686 | 0.806 | 0.391 |

At this operating point, **RF and XGB catch ~81% of true elevated-emission days while flagging roughly 1 in
2.5-3.5 days as a false alarm** (precision 0.28-0.43) — a usable trade for a farm-management screening tool
("watch this period more closely"), even though the same classifier is not a good enough regression-blend
component to beat B-03. PR curves: `results/figures/b07_pr_curve.png`.

## Verdict

**Negative for the regression task, same conclusion as B-06** — recency/clustering features and the hurdle
architecture, separately or combined, do not close the daily forecasting gap; movement is within noise and
inconsistent across towers/horizons. The diagnostic explains why: false positives are context-indistinguishable
from true positives on the features available (precip, grazing recency, growing season), so no feature
engineering within this information set can meaningfully separate them — the ~q90 spike events appear to be
driven by factors not represented in the current `ar_`/`fx_` feature set (e.g. instantaneous wind/turbulence
conditions at sub-daily resolution, not captured by daily aggregates).

**Positive, narrow finding:** the classifier's recall, while not precise enough for the regression blend, is
good enough to support a standalone early-warning framing (catch ~80% of elevated-emission days at precision
~0.3-0.43) — a decision-support artefact independent of the forecasting R² ceiling.

## Recommendation
- **Production forecaster unchanged**: B-03 enriched trees on the daily track remain the config of record.
- **Stop here on the spike-architecture lever** — B-05 (transform), B-06 (hurdle), and B-07 (diagnostics +
  recency features) have now all been tried and documented negative/marginal for daily forecasting R². Revisit
  only with a materially different feature source (e.g. sub-daily turbulence/wind data, if it becomes
  available), not further tuning of the existing feature set.
- **Early-warning framing is usable as-is** if a decision-support tool is wanted later (Phase 07 candidate),
  decoupled from the regression benchmark track.
- Move to the next agreed item: **B-08 driver-realism sensitivity** (renumbered from B-07, D-45), then Phase 07
  scenario analysis.

## Files / scope
Additive only — B03/B04/B05/B06 artifacts, `forecast_features_v2.csv`, `forecast_daily_v2.csv` untouched.
Recency features computed in-notebook, not written back to the v2 matrices. New `early_warning_op_point` rows
in `b07_summary.csv` (not appended to `benchmarks.csv`, which keeps only the regression-comparable rows).

*Source: `B07_spike_diagnostics.ipynb`, `b07_summary.csv`, `benchmarks.csv` (B07 rows),
`results/figures/b07_pr_curve.png`. Decision D-44.*
