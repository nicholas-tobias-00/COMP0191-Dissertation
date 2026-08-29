# U-06 — Conformalized Quantile Regression (CQR): fixing the spike-coverage failure U-04/U-05 both had

**Scripts:** `u06_cqr_recalibration.py`, `u06_cqr_comparison_plots.py`.
**Data:** `results/u06_u04_cqr_summary.csv`, `u06_u05_cqr_summary.csv`, `u06_spike_coverage_U04.csv`,
`u06_spike_coverage_U05.csv`. **No new model calls anywhere** — pure recalibration of U-04's and
U-05's already-saved chains (`u04_chains.csv`, `u05_chains.csv`).

## Context

User observation, checked directly rather than taken on faith: "a lot of spikes are still beyond
the interval" in U-04/U-05's fancharts. Confirmed and quantified against U-04's chains: overall
PICP≈0.89 looked fine, but **75% of the top-10%-magnitude (spike) days fell entirely outside the
interval, vs. 3.3% for the bottom 90%** — split-conformal's flat, symmetric per-bin margin
(`median ± constant`) guarantees only *average* coverage, and was achieving its ~90% target almost
entirely by nailing the easy majority of low-flux days while systematically failing the rare
high-magnitude ones. Not a new problem — U-01 (D-40) already flagged "even wide intervals miss the
biggest spikes," and B-05/B-06's point-forecast-level fixes (arcsinh transform, hurdle model) both
came back negative — but not yet addressed at the UQ-calibration level specifically.

**A second check, before building anything**: does the model's own *raw* (uncalibrated) quantile
spread already track spike magnitude better than the median does? **Yes, checked directly**: raw
`q95-q05` spread widens 1.3–1.8× on spike days vs. normal days, and on spike days the raw q95 alone
already sits close to (TabPFN: 182.7 vs. actual 193.1) or exceeds (TabICLv2: ~360 vs. 193.1) the
actual spike value — while the median (~35) massively undershoots. This directly motivated CQR:
conformalize the model's own already-adaptive raw quantiles, not a constant margin around a point
estimate that structurally can't track spikes.

## Method

**Conformalized Quantile Regression (Romano et al. 2019).** Nonconformity score =
`max(q05 - y_true, y_true - q95)` instead of split-conformal's `|y_true - median|`. Calibrated
interval = `[q05 - margin, q95 + margin]` (asymmetric-capable) instead of `[median - margin,
median + margin]`. **`rr.conformal_margins_by_bin()` is reused completely unchanged** — it was
already generic over whatever nonconformity-score array it's given; CQR only changes what's
computed and how the margin is applied, not the calibration function itself. Same leave-one-anchor-
out structure, same lead-time bins, same 3-tower coverage as U-02/U-04/U-05.

**A real bug caught and fixed before reporting**: the first aggregation pass showed T2 as `0.0000`
PICP instead of the expected `NaN`. Checked directly against the per-bin rows (correctly NaN) —
the bug was in the new script's own `wavg()` aggregator, missing the same all-NaN guard
`u02_multi_anchor_tower.py`'s own `wavg()` already carries (pandas silently sums an all-NaN column
to `0.0`, not `NaN`). Fixed by replicating that exact guard.

## Result: spike coverage roughly triples, at an honest cost

| Model / data | Spike coverage (old→CQR) | Normal-day coverage (old→CQR) | Spike interval width (old→CQR) |
|---|---|---|---|
| TabICLv2 (U-04, BASE+species) | 24.3% → **79.7%** | 96.7% → 83.7% | 183 → 394 |
| TabPFN (U-04, BASE+species) | 24.3% → **57.2%** | 96.7% → 88.4% | 177 → 248 |
| TabICLv2 (U-05, FX_A_SPECIES) | 22.1% → **79.3%** | 96.1% → 82.5% | 185 → 405 |

**TabICLv2 benefits substantially more than TabPFN** (~80% vs. ~57% spike coverage) — consistent
with the earlier check: TabICLv2's raw q95 already *exceeds* actual spike values on average, while
TabPFN's sits just short of them, so TabICLv2's raw quantiles were a better foundation for CQR to
build on. Normal-day coverage drops from ~96–97% to ~83–88% — still comfortably above the 80%
practical floor — and spike intervals roughly double in width. This is the honest price of actually
covering the events that matter: a flat, narrow interval can only look good on average by ignoring
them, which is exactly what the old symmetric margin was doing.

Aggregate PICP (all bins pooled, not spike-specific) stays in a similar ~0.76–0.90 range to
U-04/U-05's original numbers — CQR does not change the *headline* PICP number much, because that
number was never the problem; it changes *where* the coverage comes from, which the headline number
alone could never show.

## Figures

`u06_cqr_comparison_plots.py` — side-by-side before/after fancharts (old symmetric band, top panel;
new CQR band, bottom panel), same chain, same day-by-day predictions — only the interval
construction differs. Selected at each (tower, model)'s single spikiest anchor (most real
observed days ≥ that tower's own 90th percentile), not arbitrarily chosen. 6 figures:
`results/figures/u06_cqr/T{tower}_anchor{year}_{model}_{U04,U05}.png`.

## Practical implications

1. **CQR should replace the symmetric-margin approach as this project's standing UQ method**,
   given the magnitude of the spike-coverage fix and the fact it costs nothing extra to compute
   (same calibration function, same inputs already available from any model with native quantile
   output).
2. **Not yet applied to S-05's actual scenario trajectories** — U-05's Step 4 (`u05_{axis}_
   with_uq.csv`) used a %-of-mean margin derived from the *old* symmetric calibration. Re-running
   Step 4 with CQR would need S-05's scenario points to have raw daily q05/q95 saved, which the
   current daily-chains-subset scripts (`s05_daily_chains_subset.py`, `s05_practices_daily_chains_
   subset.py`) don't currently request (`tabicl_forecast()` called without `quantiles=`) — a small,
   cheap extension (add the parameter, matching U-04/U-05's own call pattern) if wanted next, not
   yet built.
3. **This reframes what "the UQ gap is closed" means from U-04/U-05** — those closed the "no
   interval exists at all" gap; this closes the "the interval exists but silently fails on the
   days that matter most" gap, which the PICP headline number alone never revealed.

## Files

- `notebooks/06_interpretability_uq/u06_cqr_recalibration.py`, `u06_cqr_comparison_plots.py`.
- `results/u06_u04_cqr_summary.csv` (94 rows), `u06_u05_cqr_summary.csv` (47 rows).
- `results/u06_spike_coverage_U04.csv`, `u06_spike_coverage_U05.csv`.
- `results/figures/u06_cqr/` (6 comparison figures).

No `benchmarks.csv` rows (UQ output, same exclusion precedent as U-01 through U-05).
