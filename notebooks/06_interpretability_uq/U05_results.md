# U-05 — Scenario-analysis UQ ("Option B" of the UQ plan; Option A = U-04)

**Scripts:** `u05_scenario_uq.py` (Steps 1–4), `u05_fanchart_plots.py` (calibration-set figures),
`u05_trajectory_with_uq_plots.py` (applied-to-S-05 figure).
**Data:** `results/u05_chains.csv`, `u05_summary.csv`, `u05_aoa_residual_correlation.csv`,
`u05_aoa_residual_correlation_by_tower.csv`, `u05_{livestock,grazing,fertilizer}_with_uq.csv`,
`u05_trajectory_with_uq_summary.csv`.

## Context

U-04 closed the UQ gap for the current forecasting champion (TabPFN+species, TabICLv2) — but its
calibration is on `forecast_daily_v3.csv`'s `BASE+species` config (52 columns), not S-05's own
`FX_A_SPECIES` (13 columns, S-03's Variant A + species). Different feature space means a
genuinely different model with different error characteristics — S-03's whole ablation exists to
show that removing/degrading features measurably changes accuracy. U-04's calibration is not
reusable for S-05; this repeats its *method* on the correct architecture, then extends it to
answer the actual scenario-analysis question: how should the resulting interval widen for scenario
points S-05 already flags as out-of-envelope?

## Method

**Step 1–2 (calibration set)**: TabICLv2 zero-shot, `FX_A_SPECIES` config, 5 real historical
anchors (2018–2022) × 3 towers, quantiles (0.05, 0.5, 0.95), leave-one-anchor-out
`rr.conformal_margins_by_bin()` — identical method to U-02/U-04, `evaluate_stage()` imported
unmodified a third time. **Full run: 9 seconds** (zero-shot, no retraining).

**A real bug caught during smoke-testing, not shipped**: the first version computed each tower's
AOA nearest-neighbour training set from the *unrestricted* full historical record. Since U-05
tests on real historical anchors (unlike S-05's own AOA check, which is safe because its scenario
dates are genuinely future and never overlap real data), a test point could be a literal row
already inside its own unrestricted training set — distance-to-self = 0 for every point, caught
directly (`aoa_dist` was uniformly 0.0 in the smoke test). Fixed: AOA training set restricted to
pre-anchor-only data, recomputed fresh per anchor, matching this project's standing no-leakage
convention.

**Step 3 (the actual design question — resolved empirically, not assumed)**: does |residual|
correlate with AOA-flagged status, in real historical data? **Result: the raw linear correlation
is weak (pooled Pearson r = 0.146) but the categorical split is real and substantial** —
out-of-AOA residuals are ~48% larger than in-AOA, pooled (46.03 vs. 31.14, n=529 vs. n=1,793).
Per-tower: T4 (+49%: 29.19→43.42, n=1,005/315) and T9 (+39%: 36.54→50.64, n=690/210) both show the
same direction cleanly. T2 shows a *reversed* small difference (13.12→9.22) but on n=4 out-of-AOA
points — not trusted, consistent with T2's already-established data-scarcity limitations.

**Resolution**: weak linear correlation rules out a smooth continuous widening function (Level 2
from the original plan) — there isn't a clean enough relationship to fit one defensibly. The real,
substantial categorical difference rules out ignoring AOA entirely. **Landed on a two-tier margin**
(a middle ground between the plan's Level 1 and Level 2): in-AOA points get the base calibrated
margin, out-of-AOA points get a wider one, interpolated continuously by each point's own
`aoa_flagged_pct` (0–100%) rather than a hard cutoff — smoother than a strict binary split, without
overclaiming a continuous distance-response relationship the data doesn't support.

**Step 4 (apply to S-05's existing output — no new model calls)**: joins the two-tier margin onto
`s05_trajectory_realizations_2050.csv` (livestock, 229,500 rows), `s05_practices_grazing.csv`
(25,500), `s05_practices_fertilizer.csv` (25,500) — pure post-processing against each row's own
already-saved `aoa_flagged_pct`. **T2 is forced to NaN throughout**, gated on Step 2's own
`conformal_mpiw` (not just whether AOA-stratified residuals exist) — T2 already fails proper
leave-one-anchor-out calibration (only 1 anchor has real ground truth), and its out-of-AOA sample
(n=4) is too small to trust regardless; falling back to a cruder standard for T2 here would quietly
undermine the same finding U-02/U-04 already established.

## Headline results

**Calibration converges to ~0.88–0.89 PICP at T4/T9** (T4: 0.8907, T9: 0.8821), matching U-02/U-04's
own pattern. T2 remains uncalibratable (NaN), same pre-existing, now three-times-confirmed
limitation.

**The calibrated interval is genuinely very wide** — worth stating plainly, not softened:

| Tower | In-AOA margin (% of mean) | Out-of-AOA margin (% of mean) |
|---|---|---|
| T4 | 93.9% | 139.6% |
| T9 | 100.4% | 139.2% |
| T2 | — (no valid calibration) | — |

A calibrated 90% interval on a scenario prediction is roughly **±94–140% of the point estimate**.
This is consistent with, not contradicting, U-01's original finding ("intervals are wide — ~150–260
nmol around fluxes whose typical magnitude is tens of nmol... CH4 at a grazed pasture carries large
aleatoric uncertainty") — the spike-dominated distribution this project has documented repeatedly
(D-44b onward) makes any honest interval this wide. A tighter number would be miscalibrated, not
better.

## Figures

- `u05_fanchart_plots.py` — same convention as U-02/U-04's fancharts, TabICLv2 on `FX_A_SPECIES`,
  real historical anchors, **AOA-flagged days marked directly on the chain** (red tick marks along
  the bottom) so the two-tier margin's trigger is visible on the same plot as the interval itself.
  3 towers × 5 anchors = **15 figures**, `results/figures/u05_fancharts/T{tower}_anchor{year}_
  TabICLv2.png`.
- `u05_trajectory_with_uq_plots.py` — **the actual deliverable**: S-05's livestock-baseline
  trajectory (2050 horizon) with the two-tier calibrated interval overlaid, kept **visibly separate**
  from S-05's own realization-spread band (a different uncertainty source — weather/GCM draw
  variability, not predictive/model uncertainty) rather than merged into one number, per this
  project's own pooled-vs-isolated realization-spread lesson (D-85). AOA-flagged-% plotted on a
  secondary axis for direct visual correlation with where the interval widens.
  `results/figures/u05_fancharts/s05_trajectory_with_uq_all_towers.png`.

## Practical implications

1. **Scenario-analysis UQ is no longer missing** — S-05's livestock/grazing/fertilizer outputs now
   carry a calibrated interval (`u05_{axis}_with_uq.csv`'s `uq_lo`/`uq_hi` columns), not just a
   point estimate.
2. **The interval is honest about being wide, not falsely precise** — a naive reuse of a tighter
   interval (e.g., forcing U-04's `BASE+species` calibration onto S-05's narrower architecture)
   would have understated uncertainty; the two-tier AOA structure means it's also honest about
   being *wider* specifically where the scenario extrapolates furthest.
3. **T2 is consistently excluded from calibrated UQ across U-02/U-04/U-05** — a structural,
   repeatedly-confirmed data-scarcity limitation, not something any feature or method change fixes.
4. **The weak-correlation/strong-categorical-difference finding (Step 3) is itself a small,
   genuine methodological result** — worth remembering if this pattern is ever revisited: AOA
   distance as a *continuous* predictor of error is not well-supported here, but AOA as a
   *categorical* flag is.

## Caveats

- Interval width is a **%-of-mean conversion** applied to S-05's *annual-mean* outputs — the
  underlying calibration margins were measured on *daily* points (matching U-02/U-04's own
  resolution). Stated explicitly as an approximation, not silently assumed equivalent.
- The two-tier margin's out-of-AOA figure rests on real but limited historical out-of-AOA samples
  (T4: n=315, T9: n=210) — real historical anchors are naturally much less often out-of-envelope
  (16% flagged, checked directly) than S-05's genuinely-future scenario points (60–88% flagged) —
  so the out-of-AOA calibration set, while real, is thinner than the in-AOA one.
- Same standing caveat as everywhere else in this project's UQ work: this calibrates against real
  historical residuals, not against ground truth for a genuine future scenario (none exists) — a
  diagnostic best estimate, not a certified guarantee for 2050.

## Files

- `notebooks/06_interpretability_uq/u05_scenario_uq.py`, `u05_fanchart_plots.py`,
  `u05_trajectory_with_uq_plots.py` (all committed, smoke-tested before the full runs).
- `results/u05_chains.csv` (5,475 rows), `u05_summary.csv` (47 rows).
- `results/u05_aoa_residual_correlation.csv` (pooled), `u05_aoa_residual_correlation_by_tower.csv`.
- `results/u05_livestock_with_uq.csv` (229,500 rows), `u05_grazing_with_uq.csv` (25,500),
  `u05_fertilizer_with_uq.csv` (25,500).
- `results/u05_trajectory_with_uq_summary.csv` (85 rows).
- `results/figures/u05_fancharts/` (15 calibration fancharts + 1 applied-trajectory figure).

No `benchmarks.csv` rows (UQ output, same exclusion precedent as U-01/U-02/U-03/U-04).
