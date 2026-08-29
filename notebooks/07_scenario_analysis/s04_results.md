# S-04 results: transient 2025-2050 trajectory, realization-level spread, SSP2-4.5 vs SSP5-8.5

## Context

S-01's recommended next steps (1) extend to SSP5-8.5, and (2) move from a single ensemble-mean
climatology snapshot to realization-level spread, as an additional uncertainty signal. Both were
built and fully run on 2026-07-15/16 (`s04_trajectory_2050.py`, `s04_daily_top3_2050.py`) but never
analyzed or written up — `BEST_RESULTS.md`/`CONTEXT.md` still described S-01 as current and listed
these as outstanding. This document is that analysis, run against the already-computed output with
no new model fitting (`notebooks/07_scenario_analysis/s04_analysis.py`).

## Method (as built 2026-07-15/16, not modified here)

**Two models run in parallel, in the same style as S-01/U-03:**
- **Primary (the real answer): S-01's frozen level-residual hybrid**, loaded from its persisted
  `joblib` artifacts (no retraining) — a single vectorized `.predict()` call per scenario point, run
  at full realization scale.
- **B-10 diagnostic benchmark**: a genuine day-by-day tree-rollout (RF/XGB/LightGBM) + SARIMAX
  `get_forecast()`, explicitly labeled a *diagnostic*, not a competing candidate for the real answer
  (U-03/S-03 already established it's the least stable choice under this kind of extrapolation
  stress). Run on a 10-realization subset, stratified 2-per-GCM across all 5 GCMs (`N_PER_GCM_B10=2`,
  a tracked, deadline-driven scope cut from an originally-approved 4-per-GCM/20-realization plan —
  legitimate follow-up if time permits, not a limitation of the method).

**Scope:** all 3 towers, **SSP2-4.5 and SSP5-8.5**, **all 5 GCMs × 100 realizations each** for the
primary hybrid (500 realization-years/SSP), **26 annual points (2025–2050)** built from real
transient CMIP6 daily weather (not an ensemble-mean climatological composite), 3 livestock
multipliers (1×/2×/3×, day-of-year climatology basis, same construction as S-01). AR history is
freshly climatology-seeded every year, never carried forward across years (user-confirmed) — there
is no real recent AR history for a genuinely blind 2025–2050 future, so this avoids compounding a
fabricated seed.

**Data produced (this session verified complete, not partial):**
`s04_trajectory_realizations.csv` (234,000 rows, primary hybrid, both SSPs complete),
`s04_trajectory_realizations_b10benchmark.csv` (28,080 rows, both SSPs complete),
`s04_aoa_by_year.csv` (4,680 rows, AOA on the same 10-realization subset), `s04_daily_top3_2050.csv`
(5.1M rows, full daily chains for each tower's top-3 B-10 models — 180 chain figures already exist
in `results/figures/s04_chains/`, not reprocessed here; annual-resolution files are sufficient for
the trajectory-level questions below).

## Results

### 1. Realization-level spread is real but modest — climate/weather is not where the uncertainty lives

Pooling all 26 years, the p10–p90 band across realizations (500/SSP for the primary hybrid) is
**~1.3–4.8% of the mean**, and — genuinely non-obvious — **the band *narrows* in relative terms as
the livestock multiplier increases** (e.g. Tower 4: 3.2% of mean at 1×, down to 1.3% at 3×). The
absolute band width barely moves with multiplier (Tower 4: 0.82 → 0.86 nmol, 1×→3×) while the mean
triples — because weather-driven residual variance is roughly constant, but the level-residual
hybrid's trend response to livestock scales up around it. **Practical read: realization spread is a
real, quantifiable uncertainty source (worth reporting) but it is small relative to the livestock
scenario question — consistent with S-01's Finding 5 ("the climate axis is not the extrapolation
risk"), now confirmed at full realization scale rather than a single ensemble-mean point.**
Full table: `results/s04_realization_spread.csv`. Figure: `s04_trajectory_bands.png`.

### 2. SSP2-4.5 vs SSP5-8.5: real, and the gap widens toward 2050 — but stays small in absolute terms

At baseline (1×) livestock, SSP5-8.5 predicts higher FCH4 than SSP2-4.5 at every tower, and **the
gap roughly doubles from the early window (2025–2029) to the late window (2046–2050)**:

| Tower | SSP585 vs SSP245, 2025–2029 | SSP585 vs SSP245, 2046–2050 |
|---|---|---|
| T2 | +0.34% | +0.74% |
| T4 | +0.26% | +0.64% |
| T9 | +0.24% | +0.34% |

This is the expected direction (SSP5-8.5's higher warming trajectory diverges from SSP2-4.5 further
into the century) but the magnitude is small — under 1% of the mean even by 2050 — reinforcing
Finding 1's point that the SSP/climate choice matters far less to this scenario's headline numbers
than the livestock multiplier does. Full table: `results/s04_ssp_divergence.csv`. Figure:
`s04_ssp_divergence.png`.

### 3. AOA-flagged extrapolation risk does NOT grow toward 2050 — but is materially higher than S-01's single-snapshot number, at every multiplier including 1×

No long-term upward trend in AOA-flagged-% is visible across 2025–2050 at any tower/SSP/multiplier
(`s04_aoa_trend.png`) — reinforces Finding 3's point again: climate drift alone doesn't push the
scenario further out-of-envelope over the horizon tested.

**But the absolute level is a real, unexpected departure from S-01's reported numbers, and worth
flagging directly rather than glossing over.** S-01 (single ensemble-mean climatological snapshot,
2041–2060) found **0% flagged at 1×/2× for every tower, only 5.5–6.0% at 3× (T4/T9 only), and 0% at
every multiplier for T2.** Here, using real transient annual weather instead of a smoothed
ensemble-mean composite, **even the 1× baseline is flagged 9–15% of days at every tower**, rising to
17–20% at 3× (T4/T9) and staying near 8–10% at all multipliers for T2:

| Tower | 1× (pooled both SSPs, all years) | 3× |
|---|---|---|
| T2 | ~9.2% | ~9.9% |
| T4 | ~12.4% | ~17.3% |
| T9 | ~14.9% | ~18.4% |

**Interpretation:** a single real transient weather-year is naturally further from the smoothed
2018–2023 training climatology in the full 42-dimensional AOA feature space than a 20-year,
100-realization ensemble-mean day is — S-01's Finding 7 already flagged that the AOA check "can be
diluted by many in-range dimensions," and this is the concrete confirmation: smoothing (S-01's
ensemble-mean construction) suppresses the flagged rate relative to genuine day-to-day weather
variability (S-04's construction), independent of any livestock scaling at all. **This is exactly
the kind of signal the realization-level extension was built to surface — it changes the read of
"is 2050 out-of-distribution" materially: under a transient/realization view, a non-trivial minority
of days are already flagged even under an unchanged, present-day livestock baseline, not just under
an aggressive 3× scenario.** Full table: `results/s04_aoa_trend.csv`,
`results/s04_aoa_early_vs_late.csv`. Figure: `s04_aoa_trend.png`.

### 4. Hybrid-vs-trees divergence (S-01 vs U-03's central finding) holds, and grows, across the full 26-year × both-SSP trajectory

Matched on the shared 10-realization stratified subset (fair comparison, same scenario inputs for
both models), pooled across both SSPs and all 26 years:

| Tower | Primary hybrid, 1×→3× | B-10 diagnostic ensemble, 1×→3× |
|---|---|---|
| T2 | **+38.6%** | +20.4% |
| T4 | **+156.4%** | +76.6% |
| T9 | **+120.3%** | +62.0% |

> **Bias-corrected figures (D-100, 2026-08-17):** the primary hybrid's own baseline-reconstruction
> gap (S-01 Finding 1) means the table above is relative to the model's own imperfect 1× baseline,
> not the real historical mean. Applying the same per-tower bias offset used at S-01 (same frozen
> model, same bias, no re-derivation needed), paired within each (tower, ssp, gcm, realization,
> year) group: **T2 +48.1 to +48.5%** (was +38.6%), **T4 +152.4 to +153.2%** (was +156.4%,
> materially unchanged), **T9 +133.1 to +133.7%** (was +120.3%) — pooled across both SSPs. Same
> direction and rough magnitude of correction as S-01's own single-snapshot version. See
> `results/s04_trajectory_summary_bias_corrected.csv`,
> `notebooks/07_scenario_analysis/d100_bias_correction_s01_s04.py`.

This reproduces S-01's headline finding (the level–residual hybrid measurably fixes the tree-only
plateau U-03 found) — not just at a single 2041–2060 snapshot, but consistently across 26 years and
both emission scenarios. The B-10 diagnostic ensemble's own response here (+20–77%) sits between
U-03's original "trees alone" figure (+21–23%, no SARIMAX) and its "both ensembles" figure (+49–50%,
25%-weighted SARIMAX) — expected, since this diagnostic benchmark's ensemble is the same 4-model mix
(RF+XGB+LightGBM+SARIMAX) and SARIMAX (U-03's single most extreme extrapolator, +150% mean) pulls the
blend up from the trees' own plateau. **At baseline (1×), the ranking even flips at T4/T9** — the
diagnostic tree/SARIMAX ensemble actually predicts *higher* FCH4 than the hybrid at 1× (T4: 27.85 vs
25.66, T9: 35.48 vs 34.55) — only the *rate of change* under livestock stress is where the hybrid's
advantage shows up, not the absolute baseline level. Full tables:
`results/s04_hybrid_vs_benchmark_summary.csv`, `results/s04_hybrid_vs_benchmark_response.csv`.
Figure: `s04_hybrid_vs_benchmark.png`.

## Findings summary

1. **Realization-level spread is a real, quantifiable, but small uncertainty source** (1–5% of the
   mean) — confirms rather than overturns S-01's "climate axis isn't the extrapolation risk" framing.
2. **SSP2-4.5 vs SSP5-8.5 divergence is real and grows toward 2050 as expected, but stays under 1%
   of the mean even by 2050** — the emission-scenario choice is a minor lever on this dissertation's
   headline scenario numbers compared to the livestock multiplier.
3. **AOA-flagged extrapolation risk does not trend upward across the 2025–2050 horizon** — no
   evidence climate drift alone pushes the scenario further out-of-envelope over time.
4. **New, non-obvious finding: the transient/realization-level AOA check flags materially more days
   than S-01's smoothed ensemble-mean snapshot did, at every multiplier including the unchanged 1×
   baseline** (9–15% here vs 0% in S-01 at 1×). The construction method (smoothed ensemble-mean vs.
   real transient weather) measurably changes how "out-of-distribution" a scenario looks, independent
   of the livestock question — an important caveat for how any AOA-flagged number from this project
   should be read and compared across S-01 vs S-04.
5. **S-01's central finding (hybrid fixes the tree-only extrapolation plateau) holds and is
   reinforced across the full 26-year × both-SSP trajectory**, not just a single climatological
   snapshot — hybrid response to 3× livestock is ~2× the diagnostic tree/SARIMAX ensemble's own
   response at every tower.

## Explicit caveats (carried forward from S-01, still apply)

All of S-01's original caveats still apply unchanged (parametric not mechanistic trend, 9/11 drivers
historical-day-resampled, naive livestock multiplier not a self-consistent mechanistic timeline,
U-02/U-03 conformal intervals not attached). Two S-04-specific additions:

1. **The B-10 diagnostic benchmark's realization coverage (10, stratified) is narrower than the
   primary hybrid's (500, full)** — a tracked, deadline-driven scope cut (originally approved at 20).
   Re-running at higher N_PER_GCM_B10 is legitimate follow-up work, not required for the findings
   above (the hybrid-vs-benchmark divergence direction is unlikely to reverse with more realizations
   given how consistent it already is across 10).
2. **AOA is only computed on the same 10-realization stratified subset**, not the full 500 — the
   levels reported in Finding 3 above should be read as a representative sample, not an exhaustive
   count.

## Remaining, not attempted here (per S-01's own queued list)

1. A self-consistent mechanistic livestock-scenario construction (grazing-recency/days-since-grazing
   features still not adjusted jointly with density in S-04 — the multiplier remains a naive
   column-scale, same as S-01).
2. SPACSYS process-model route for the trend/level component, if time permits before the 1 Sept
   deadline.
3. (Optional follow-up, not blocking) re-run the B-10 diagnostic benchmark at higher realization
   coverage than the current 10-realization scope cut.

## Files

- `notebooks/07_scenario_analysis/s04_trajectory_2050.py`, `s04_daily_top3_2050.py`,
  `s04_chain_plots.py` (the original S-04 build, 2026-07-15/16, unmodified here)
- `notebooks/07_scenario_analysis/s04_analysis.py` (this analysis, new)
- `results/s04_trajectory_realizations.csv`, `s04_trajectory_realizations_b10benchmark.csv`,
  `s04_aoa_by_year.csv`, `s04_daily_top3_2050.csv` (source data, unmodified)
- `results/s04_trajectory_summary.csv`, `s04_ssp_divergence.csv`, `s04_realization_spread.csv`,
  `s04_aoa_trend.csv`, `s04_aoa_early_vs_late.csv`, `s04_hybrid_vs_benchmark_raw.csv`,
  `s04_hybrid_vs_benchmark_summary.csv`, `s04_hybrid_vs_benchmark_response.csv` (new, this analysis)
- `results/figures/s04_summary/s04_trajectory_bands.png`, `s04_ssp_divergence.png`,
  `s04_aoa_trend.png`, `s04_hybrid_vs_benchmark.png` (new, this analysis)
- `results/figures/s04_chains/` (180 daily-chain figures, from the original build, unchanged)

No `benchmarks.csv` rows (scenario-simulation output, not a point-forecast/interval-calibration
benchmark — same exclusion precedent as S-01/U-01/U-02/U-03).
