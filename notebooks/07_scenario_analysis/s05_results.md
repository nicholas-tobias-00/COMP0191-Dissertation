# S-05 results: TabICLv2 + Variant A (driver removal) + species-disaggregated livestock, 10-year
# transient CMIP6 trajectory

## Update: baseline-reconstruction bias found and corrected — a large one, unlike S-01/S-04's (D-100)

**Context:** part of a broader task applying S-01's own already-documented delta-method bias
correction (`s01_results.md` Finding 1: the 1× baseline doesn't exactly reconstruct the real
historical mean) across all of Phase 07. S-05 had never had this specific check run — it's a
structurally different model (TabICLv2, zero-shot, not S-01's fitted Ridge+tree hybrid), so S-01's
own small, already-accepted gap (9–20%) was explicitly not assumed to transfer.

**It doesn't transfer — the gap here is far larger.** S-05's `baseline_1x1x1x` scenario (real
climatology-resampled drivers, no livestock perturbation, `FX_A_SPECIES` — S-05's current
architecture, matching S-03's Variant A + species split) underpredicts the real historical mean by
**40–80% at every tower**, checked against both `y_gapfilled` (S-01/S-04's own convention) and
`y_observed` (this project's authoritative target) for consistency: T2 −69.9%/−81.1%, T4
−52.4%/−51.6%, T9 −40.8%/−39.8%. Always an underprediction, never an overprediction — a
systematically one-directional bias, not noise.

**Applying the same delta-method correction (anchored to `y_gapfilled`, for direct comparability
with S-01/S-04) roughly halves several of this project's most-cited headline percentages:**

| Result | Raw | Corrected |
|---|---|---|
| Cattle-3× alone, T4 (main species table) | +213.9% | **+101.8%** |
| Cattle-3× alone, T9 | +187.3% | **+110.4%** |
| `own_max` all-species (D-97 redesigned ladder), T4 | +128.8% | **+61.2%** |
| `own_max` all-species, T9 | +142.8% | **+84.2%** |
| `own_max` all-species, T2 | +57.9% | **+16.8%** |

**Read this with real caution, more than S-01/S-04's correction.** Delta-method rests on trusting
the model's SHAPE of response even when its LEVEL is off — a defensible assumption for a 9–20% gap,
much less so for 40–80%. The cattle-dominance finding's *direction* is completely unaffected (cattle
still dominates far beyond its LSU-weight share under either raw or corrected numbers) — what's
genuinely uncertain now is the exact magnitude, which should be read as somewhere between the raw
and corrected figures, not settled at either one pending further investigation (not attempted here,
out of scope for this task — see D-100 for the explicit boundary).

**Consequence, stated directly: prior citations of the raw magnitude (e.g. "+214.5%" cattle-3× at
T4, `BEST_RESULTS.md`, D-85, and the report's Chapter 7) are now flagged as needing a follow-up
pass** — not silently left inconsistent, but not corrected everywhere in this same update either,
since that magnitude has been the basis for several downstream figures/tables that would each need
their own review. Full detail, methodology, and the S-01/S-04 companion correction:
`notebooks/07_scenario_analysis/d100_bias_check_s05.py`, `d100_bias_correction_s01_s04.py`,
`results/s05_trajectory_realizations_2050_bias_corrected.csv`,
`results/s05_practices_livestock_v2_bias_corrected.csv`, DECISIONS.md D-100.

## Update: livestock scenario ladder redesigned to absolute, plausibility-anchored levels (D-97)

**Supervisor feedback, 2026-08-13**: is 3x livestock plausible given real catchment area? Checked
directly, quantitatively, before redesigning anything.

**The concern was correct.** The original 1x/2x/3x multiplier scales the smoothed day-of-year
climatology curve (not raw data). Comparing the result against each catchment's own real history:

| Tower | Climatology baseline peak (LSU/ha) | OLD 3x | Real historical **instantaneous** max ever recorded |
|---|---|---|---|
| T2 | 0.71 | 2.13 | 4.51 |
| T4 | 2.48 | **7.44** | 4.99 |
| T9 | 2.58 | **7.74** | 5.65 |

At T4/T9, 3x asked the model to sustain a full-year-smoothed density 30-50% ABOVE the single
highest instantaneous day this catchment has ever recorded — denser than the farm has ever been,
held all year. T2's 3x, conversely, was still below its own historical spike — the same flat
multiplier was simultaneously too extreme for T4/T9 and too mild for T2.

**Redesign: absolute, externally-anchored levels replace the flat multiplier.**
- `half` (0.5x baseline), `baseline` (1x, unmodified) — unchanged mechanism, no plausibility
  question when scaling down or leaving as-is.
- `lit_ceil` — uniform 3.0 LSU/ha across towers, grounded in UK grassland stocking-density
  literature: typical conventional stocking is 1.5-2.5 GLU/ha; >3 GLU/ha is described as reachable
  only under "very best growing conditions + near-optimal N fertiliser"; NVZ manure-N regulation
  caps at 170 kg N/ha/yr (up to 250 under grassland derogation), roughly ~2 GLU/ha by common
  convention; an AHDB rotational-grazing case study achieved 2.4 LSU/ha as a real high-end example.
- `own_max` — each catchment's own real historical instantaneous peak (T2=4.51, T4=4.99,
  T9=5.65 LSU/ha) — "as dense as this specific catchment has itself actually been, at least once."

Multipliers solved by bisection (not assumed) so each level's resulting fx_lsu_dens climatology
curve peak hits its named target exactly — e.g. T4's `lit_ceil` needs only 1.21x (baseline is
already close to 3.0 LSU/ha), while T2's `own_max` needs 6.33x (baseline is far below its own
historical spike). **Two families, mirroring the original design**: `all_species` (cattle/sheep/
lamb scaled together) and `cattle_alone` (only cattle scaled, sheep/lamb at real baseline) — D-84's
cattle-dominance finding remains the interesting axis, just re-anchored. 7 combos total (down from
27), full 3-tower x 2-SSP x 5-GCM x 10-realization coverage (2,100 calls, ~1.9h).

**Result: monotonic, sensible response at every level, cattle-dominance confirmed again.**

| Tower | Baseline | half (both families ≈) | lit_ceil (both families ≈) | own_max, all_species | own_max, cattle_alone |
|---|---|---|---|---|---|
| T2 | 5.83 | −14 to −16% | +18 to +27% | +56.2% | +37.9% |
| T4 | 14.27 | −26% | +20% | **+128.9%** | **+132.4%** |
| T9 | 21.70 | −24 to −32% | +13% | **+142.3%** | **+147.5%** |

`all_species` and `cattle_alone` land within a few points of each other at every level (T4/T9
own_max: 128.9% vs. 132.4%, 142.3% vs. 147.5%) — reconfirms cattle drives almost the entire
livestock response regardless of whether sheep/lamb scale alongside it, the same finding D-84
established under the old multiplier, now holding under the new one too. No change to the
project's standing conclusions — this redesign fixes *how intense* the upper scenario level is,
not *which* species drives the response.

### Files

- `src/features/build_transient_scenario_drivers_livestock_v2.py` — new module, solved
  multipliers + level/family definitions (imports from, does not modify,
  `build_transient_scenario_drivers_species.py`).
- `notebooks/07_scenario_analysis/s05_livestock_v2_trajectory.py` — full sweep (additive; the
  original 27-combo `s05_trajectory_realizations_2050.csv`/`s05_daily_chains_2050.parquet` are
  untouched).
- `notebooks/07_scenario_analysis/s05_livestock_v2_daily_chains_subset.py` +
  `s05_livestock_v2_daily_chains_plots.py` — representative-subset daily chains (3 towers x
  ACCESS-ESM1-5/1 x both SSPs x 7 combos, 42 calls) + 12 figures (`results/figures/s05_summary/
  s05_livestock_v2_{all_species,cattle_alone}_daily_*.png`).
- `results/s05_practices_livestock_v2.csv` (59,500 rows), `s05_livestock_v2_summary.csv`
  (% change vs. baseline table), `s05_livestock_v2_daily_chains_subset.csv` (434,350 rows).

## Update: U-06/U-07's CQR calibrations attached to actual scenario trajectories (D-92)

Standing gap closed: U-06 (CQR) and U-07 (LSU-density-stratified CQR) had only ever been fitted
and validated on U-04/U-05's *historical calibration* chains — never applied to S-05's own
scenario output, because the main S-05 sweeps (`s05_trajectory_2050.py`,
`s05_practices_trajectory.py`) only ever requested point predictions (`pred`), not quantiles.
Scenario analysis (Objective 5) and UQ (Objective 4) existed side by side but never actually
touched each other in the output until now.

**Method, two new scripts, no new calibration fitting** (U-06/U-07's margins are reused exactly as
already fitted — this is pure inference + post-processing):
1. `s05_uq_daily_chains_subset.py` — reruns the SAME small representative subset already used for
   the sanity-check daily-chain figures (3 towers × 2 SSPs × 3 levels/combos = 18 calls/axis, one
   representative GCM/realization ACCESS-ESM1-5/1, 2050 horizon), but requests
   `quantiles=(0.05, 0.5, 0.95)` from `tabicl_forecast()` instead of a point prediction only.
   **Cost, measured not assumed**: 4.3s/call, identical to the point-only runs — TabICL always
   computes an internal quantile grid regardless of the `quantiles=` argument (per its own
   docstring), so requesting q05/q95 explicitly is free. Full 54-call run (livestock + grazing +
   fertilizer): **~2.5 minutes**, 0 failed calls.
2. `s05_uq_cqr_apply.py` — attaches U-06's flat CQR margin and U-07's LSU-tier CQR margin
   (`u06_u05_cqr_summary.csv`, `u07_u05_lsu_cqr_summary.csv` — the FX_A_SPECIES-architecture
   calibrations, i.e. U-05's, not U-04's BASE+species ones, per D-89's finding that feature space
   determines error characteristics) to every scenario day via a (tower, lead-time-bin[, LSU-tier])
   lookup, pooling the margin across U-05's 5 anchor years (S-05's scenario points don't have their
   own anchor year to key on). LSU tiers use edges from U-05's full pooled calibration set (no
   leave-one-out needed — S-05's scenario points are genuinely future, same reasoning as S-05's own
   `precompute_aoa()` already uses).

**Two explicit, stated extrapolation assumptions** (neither hidden):
- **Lead times beyond 365 days hold the widest calibrated bin's margin flat.** U-02's original bins
  only go to 271-365 days (sized for the 1-year forecasting evaluation window); S-05's horizon runs
  to 2050 (thousands of days out). No calibration evidence exists beyond ~1 year, so the last bin's
  margin is held flat rather than extrapolated by a trend — likely an **underestimate** of true
  uncertainty at year 20+ (error should plausibly grow with lead time, not plateau), so read
  far-horizon CQR bands as a floor, not a ceiling.
- **Grazing/fertilizer axes reuse the livestock-architecture (FX_A_SPECIES) margins** even though
  their own models add extra covariates (`GRAZING_COLS`/`FERT_COLS`) — same approximation U-05's
  own Step 4 already made when applying its margins across all three axes uniformly.

**Result: works cleanly, >99% coverage at T4/T9** (T2 0% — same pre-established degeneracy as
every other UQ step this project has run into, not new). The only gap: one thin (lead-bin ×
LSU-tier) cell at Tower 4 (days 1-7, mid-density tier) has zero calibration samples across all 5
U-05 anchors — a genuine data-sparsity limit inherited from U-07's own summary, surfaced as `NaN`
rather than filled in. Sanity checks passed directly (not assumed): no interval inversions
(`lo > hi`) anywhere, U-06/U-07 bands both narrower than the model's own raw `[q05,q95]` on average
at Tower 4 (raw MPIW 572.6 vs. U-06 514.6 vs. U-07 523.4) — CQR correctly tightens an
overconfident-in-the-wrong-direction raw quantile spread here, consistent with U-06's original
historical-data finding.

Figures: `results/figures/s05_uq_cqr/s05_uq_cqr_{livestock,grazing,fertilizer}_{ssp245,ssp585}.png`
(6 total, both SSPs, one representative most-extreme level per axis) — raw `[q05,q95]` vs. U-06
flat CQR vs. U-07 LSU-stratified CQR bands directly on the 2050 scenario trajectory. **All 3 towers
shown in every figure, including Tower 2** — its panel plots the raw model quantiles (TabICL
forecasts fine for T2) with an explicit annotation that no calibrated CQR band exists (same
pre-established degeneracy as U-04/U-05's own Step 2), rather than being silently omitted the way
the first draft of this figure did.

Data: `results/s05_{livestock,grazing,fertilizer}_with_cqr.csv` (186,150 rows each — the 18-call
subset's full daily chains × axis, all towers/SSPs/levels, all three interval variants attached).

## Update: extended to 2050 + full daily chains (supersedes the 10-year results below)

User follow-up: (1) run the same 8,100-call grid to **2050** (matching S-04's own endpoint) instead
of a fixed 10 years -- T4/T9 now run 2024-2050 (27 years, 9,855 days/call), T2 2020-2050 (31 years,
11,315 days/call); (2) save **full daily chains for every call**, not just the annual mean, now
that compute (not I/O) is the bottleneck anyway. Both requested together and run in one pass
(`s05_trajectory_2050.py`) rather than two separate sweeps.

**Cost, measured before committing (not guessed)**: a single 27-year call timed at 4.07s (vs.
~1.2-1.3s for the original 10-year call) -- horizon extension, not the daily save, dominates the
new cost. Full grid estimated ~9h; **actual runtime 5.44h**, 0 failed calls. Daily output:
83,767,500 rows, written incrementally via `pyarrow.parquet.ParquetWriter` (never held in memory
at once) -- **1.25 GB as Parquet** (would be ~6GB as CSV). Reproducibility spot-checked directly:
the same (tower, GCM, realization, SSP, year, combo) point scored 9.981 in the original 10-year run
and 9.974 in this rerun -- 0.07% apart, consistent with ordinary GPU inference variance, not drift.

**Every finding from the 10-year version replicates at the extended horizon, several more clearly
than before**:

| Finding | 10-year value | 2050 value |
|---|---|---|
| Cattle response, T4 3x alone | +205.6% | **+214.5%** |
| Cattle response, T9 3x alone | +195.6% | **+186.4%** |
| Joint-vs-additive synergy, T4 | −0.2% | −0.6% |
| Joint-vs-additive synergy, T9 | +8.8% | +9.1% |
| Realization spread (isolated), range | 2.4-6.6% | 2.5-7.6% |
| AOA flagged %, start vs. end of horizon | ~flat (62-68%) | **still flat** (62.3-68.9% across the full 27-31 years) |

**One genuine improvement from the longer horizon**: SSP2-4.5 vs SSP5-8.5 divergence now visibly
**grows** from the early to the late window (e.g. T4: +0.09% -> +0.77%; T9: +0.09% -> −0.75%,
direction-inconsistent at T9 specifically, worth noting rather than smoothing over) — the 10-year
window was too short to show this pattern clearly; at the 2050 horizon it matches S-04's own
"divergence grows toward end of century" finding much more directly. Full table:
`results/s05_ssp_divergence_2050.csv`.

**AOA's stability over time is now confirmed far more strongly**: flat within ~1 percentage point
across the *entire* 27-31-year horizon at every tower, not just the first 10 years -- the "AOA
doesn't grow with horizon length" finding is now tested over close to 3x the original span and
holds just as cleanly.

**Daily-resolution follow-up (separate, smaller run, not the full grid)**: saved full daily chains
for an 18-call representative subset (before the 2050 extension) to sanity-check the seasonal
shape -- see "Daily-resolution sanity check" below, unchanged by the 2050 extension (that check
used the same underlying model/features, just a shorter window at the time). The full-grid Parquet
file now saved alongside the 2050 run supersedes that subset for anyone wanting the complete daily
picture at every scenario point, not just the 18 illustrative ones.

All tables/figures below with a `_2050` suffix are the current, canonical version. The originals
(no suffix) are kept on disk, not deleted, and referenced below for anyone tracing how the results
changed from the 10-year to the 2050-horizon run -- this is a genuine methodology extension, not a
correction, so both remain part of the record.

**Files (2050 horizon, additive to the original file list below)**:
`notebooks/07_scenario_analysis/s05_trajectory_2050.py`, `s05_analysis_2050.py`,
`results/s05_trajectory_realizations_2050.csv` (229,500 rows),
`results/s05_daily_chains_2050.parquet` (83,767,500 rows, 1.25GB),
`results/s05_trajectory_summary_2050.csv`, `s05_realization_spread_pooled_2050.csv`,
`s05_realization_spread_isolated_2050.csv`, `s05_ssp_divergence_2050.csv`, `s05_aoa_trend_2050.csv`,
`s05_species_marginal_response_2050.csv`, `s05_joint_vs_additive_2050.csv`,
`results/figures/s05_summary/s05_trajectory_bands_2050.png`, `s05_aoa_trend_2050.png`,
`s05_species_response_2050.png`.

---

## Second update: farming-practice scenarios (grazing timing, fertilizer schedule)

User follow-up, same session: extend beyond the livestock-density axis to two management-practice
levers -- grazing timing and fertilizer application schedule -- run as **two separate experiments**
(not stacked onto the 27-combo livestock grid), each holding livestock at baseline (1x/1x/1x real
climatology) and sweeping only its own 3 scenario levels, at the current canonical 2050 horizon.

**Why these two, and why separately**: F-01/F-04/F-05's repeated "redundant on the rich base"
finding for management features on real historical data set a prior expectation that both should
show a smaller effect than livestock density's cattle result -- but the two axes were expected to
differ from each other: grazing timing is directly tied to livestock presence (the #1 driver
throughout this project), fertilizer's stronger mechanistic link is to N2O, not CH4 (this project
doesn't model N2O). Both expectations were stated *before* running, not fitted after the fact.

**Grazing construction**: the real day-of-year species-density climatology (already built for
`FX_A_SPECIES`) is phase-shifted at the season edges -- for the first half of the year, sampled at
`doy+shift_days` (pulling a later, more in-season value earlier); for the second half, at
`doy-shift_days` -- simulating earlier turnout / later housing without needing per-tower/species
boundary-detection logic. `fx_grazing_active`/`fx_days_since_grazing` are re-derived from the same
shifted presence pattern using `days_since_grazing()`, the exact function the real columns are
built from (`build_forecasting_matrix_v2.py`), not reimplemented. Levels: historical (no shift),
+2 weeks, +4 weeks each end.

**Fertilizer construction**: real per-tower fertN event history (`Field_Event_Data_Format_1.csv`)
summarized into a "typical year" template (event count, DOY range, mean rate) rather than
replaying one arbitrary specific year -- checked directly: T4 ~8.25 events/yr (DOY 82-234, mean
127 kg/ha), T9 ~4/yr (DOY 87-204, mean 90 kg/ha), T2 ~5/yr (DOY 55-268, mean 137 kg/ha). Synthetic
future events (evenly spaced within the template's DOY range, same schedule repeating every
scenario year) are appended after each tower's real pre-anchor history and run through
`recency_series()` -- the exact `exp(-days_since_event/14)` decay function
`fx_mgmt_fertN_recency`/`_rate` are built from (`build_management_features.py`), not reimplemented.
Levels: historical (template as-is), +50% rate (same schedule, 1.5x kg/ha), +50% frequency (1.5x
event count, same rate).

**Feature sets**: `FX_A_SPECIES` (13 cols) + `fx_grazing_active`/`fx_days_since_grazing` (15 total)
for the grazing run; `FX_A_SPECIES` + `fx_mgmt_fertN_recency`/`fx_mgmt_fertN_rate` (15 total) for
fertilizer -- AOA threshold recomputed per-tower in each 15-dim space (not reusing the main
experiment's 13-dim precompute, the wrong feature space for this one).

**Scope**: 3 towers x 2 SSPs x 5 GCMs x 10 realizations x 3 levels = 900 calls per axis, both
smoke-tested (1 tower, 1 SSP, 5 GCMs, 1 realization each, all 3 levels) before the full run.
**Actual runtime: ~51 minutes per axis** (~1.7h combined), 0 skipped/failed calls, 0 NaN.

### Result: the two expectations set before running were both confirmed, cleanly

**Grazing timing shows a real, substantial, monotonic effect at every tower**:

| Tower | Historical | +2 weeks | +4 weeks | % change (max) |
|---|---|---|---|---|
| T2 | 7.43 | 7.55 | 7.74 | +4.1% |
| T4 | 14.81 | 16.09 | 17.61 | **+18.9%** |
| T9 | 30.67 | 33.12 | 35.93 | **+17.2%** |

Consistent direction, consistent monotonicity, at every tower including T2 (muted in magnitude as
usual, but the sign and monotonicity still hold) -- extending the grazing season genuinely moves
the model's prediction, not a noise-level effect.

**Fertilizer schedule shows a small effect, inconsistent in DIRECTION across towers**:

| Tower | Historical | +50% frequency | +50% rate |
|---|---|---|---|
| T2 | 5.03 | 4.82 (−4.1%) | 4.99 (−0.8%) |
| T4 | 13.38 | 13.55 (+1.2%) | 13.50 (+0.9%) |
| T9 | 17.19 | 16.81 (−2.2%) | 17.20 (+0.04%) |

T2/T9 show *increased* fertilizer frequency *decreasing* predicted FCH4; T4 shows the opposite;
every magnitude is under 5%. Read as genuinely weak/noise-level, not a directional finding either
way -- consistent with the prior set before running (F-01/F-04/F-05's "redundant on the rich base"
finding, now shown to extend from real-data feature importance to scenario response too).

**AOA side-finding**: grazing's AOA-flagged-% is both higher in absolute level (76-88% vs.
fertilizer's 59-68%) and grows monotonically with the shift level (e.g. T4: 76.0%→82.0%→88.0%) --
extending the grazing season genuinely pushes the scenario further from the real training
distribution, a sensible, expected pattern (unlike fertilizer, where AOA doesn't move cleanly with
level). Consistent with this project's now-repeated (S-04→S-05→here) finding that AOA's level
tracks feature-space and scenario-construction specifics, not a fixed property of "how far into the
future" a scenario reaches.

### Practical implications

1. **Grazing-season length is a genuine, previously-untested management lever** worth taking
   seriously alongside livestock density -- a farm-management "what if we extended the grazing
   season" question has a real, substantial, monotonic answer in this model, not a null result.
2. **Fertilizer schedule (rate or frequency) does not show a robust, directional effect** on
   predicted CH4 at any tower -- consistent with, and now extends, F-01/F-04/F-05's standing
   finding that fertilizer/management features are largely redundant once livestock + FCO2 +
   pooling are already in the model. This is itself a legitimate, useful null result for the
   dissertation, not a failed experiment.
3. **No change to any standing recommendation** -- both are diagnostic scenario extensions, not
   production-config changes.

### Correction (D-97, 2026-08-13): fertilizer units were wrong -- rate meant kg product/ha, not kg N/ha

**Supervisor question, checked directly**: "what are the units of measurement? what's the
difference between rate and frequency?" Answering it precisely surfaced a real data issue.
`Application_rate_per_ha` (the raw column the old template's `mean_rate` was built from) is **kg
of product per ha, not kg of nitrogen per ha** -- and the `fertN` channel it was pulled from
(`classify()`, `build_management_features.py`, used project-wide) tags **any** inorganic
fertiliser event, including pure P/K/S/Mg products with **0% nitrogen content** (31% of all
"fertN"-tagged events site-wide: e.g. Triple Superphosphate, Muriate of Potash). The old template
mixed these in as if they were nitrogen applications.

**Fix, scenario-scope only** (confirmed with the user before proceeding; `build_management_features.py`
itself, used by ~15 prior experiments, is deliberately left untouched given the Sept 1 deadline):
recomputed the template using true kg N/ha (`Application_rate_per_ha x N-content-% / 100`, parsed
from `Application_Info`, e.g. "34.5% N"), restricted to events with N% > 0. **Units, stated
plainly**: **rate** = kg true nitrogen per hectare, per application event. **frequency** =
applications per year. These are the two independently-scalable axes the `+50% rate`/`+50%
frequency` scenario levels each isolate -- rate answers "what if each application carried more
nitrogen," frequency answers "what if nitrogen were applied more often at the same per-event
amount."

**Corrected template** (kg true N/ha, true-N events/yr only -- was: kg product/ha, all events):

| Tower | OLD events/yr (all products) | NEW events/yr (true N only) | OLD mean rate (kg product/ha) | NEW mean rate (kg true N/ha) |
|---|---|---|---|---|
| T2 | 5 | 3.3 | 137 | 51.9 |
| T4 | 8 | 7.4 | 127 | 43.6 |
| T9 | 4 | 1.55 | 90 | 26.1 |

T9's frequency drops the most (4 -> 1.55/yr) -- most of its "fertiliser" events turn out not to be
nitrogen applications at all.

**Rerun result: the headline finding is unchanged, now on correct units.**

| Tower | Historical (kg CH4-equivalent flux, unchanged metric) | +50% frequency | +50% rate |
|---|---|---|---|
| T2 | 5.27 | 5.05 (−4.1%) | 5.24 (−0.5%) |
| T4 | 13.05 | 13.19 (+1.1%) | 13.16 (+0.8%) |
| T9 | 18.28 | 18.28 (**0.0%**) | 18.25 (−0.1%) |

Still small (<5%), still sign-inconsistent across towers (T2 negative, T4 positive, T9 essentially
exactly flat) -- the original "redundant on the rich base" conclusion holds under the corrected
units; only the exact per-tower numbers and the units label changed. Original (pre-fix) files kept
for audit trail as `*_PRE_D97_UNITS_FIX.csv`, not deleted.

### Files

- `src/features/build_transient_scenario_drivers_practices.py` -- grazing-shift and
  fertilizer-schedule driver construction (reuses `days_since_grazing()`/`recency_series()`
  unchanged from their real-data construction, not reimplemented). Fertilizer template corrected
  to true kg N/ha, D-97.
- `notebooks/07_scenario_analysis/s05_practices_trajectory.py` -- sweep script (both axes,
  smoke-tested before the full run).
- `notebooks/07_scenario_analysis/s05_practices_analysis.py` -- this analysis.
- `results/s05_practices_grazing.csv` (unaffected by D-97), `s05_practices_fertilizer.csv`
  (25,500 rows each, raw; corrected units, D-97), `s05_fertilizer_corrected_summary.csv`.
  Pre-correction originals kept as `s05_practices_fertilizer_PRE_D97_UNITS_FIX.csv` and siblings.
- `results/s05_practices_grazing_summary.csv`, `_aoa.csv`; `s05_practices_fertilizer_summary.csv`,
  `_aoa.csv` (derived tables).
- `results/figures/s05_summary/s05_practices_grazing.png`, `s05_practices_fertilizer.png`.
- `notebooks/07_scenario_analysis/s05_practices_daily_chains_subset.py` -- follow-up: saves full
  daily chains (not just annual_mean) for a representative 9-call subset per axis (3 towers x
  ACCESS-ESM1-5/1/ssp245 x each axis's 3 levels), mirroring `s05_daily_chains_subset.py`'s
  livestock-axis pattern.
- `notebooks/07_scenario_analysis/s05_practices_daily_chains_plots.py` -- same 3-view figure style
  as the livestock daily-chain set (full horizon / single-year zoom / monthly-smoothed).
- `results/s05_practices_grazing_daily_chains_subset.csv`, `s05_practices_fertilizer_daily_chains_subset.csv`
  (93,075 rows each).
- `results/figures/s05_summary/s05_practices_{grazing,fertilizer}_daily_{full_horizon,zoom2035,monthly_smoothed}_{ssp245,ssp585}.png`
  (12 figures) -- grazing's level separation visibly grows over the horizon (matching its real
  effect); fertilizer's levels stay close together throughout (matching its muted/inconsistent
  effect), in both SSPs. Livestock's matching set:
  `s05_livestock_daily_{full_horizon,zoom2035,monthly_smoothed}_{ssp245,ssp585}.png` (6 figures,
  `s05_livestock_daily_chains_plots.py`, renamed from an earlier `s05_daily_chains_2050_plots.py`
  for naming consistency across all three scenario families). **Naming convention, applied
  uniformly across all 18 figures**: `s05_{axis}_daily_{view}_{ssp}.png`, `axis` in
  {livestock, practices_grazing, practices_fertilizer}, `view` in {full_horizon, zoom2035,
  monthly_smoothed}, `ssp` always explicit (no unlabeled default-SSP file). SSP5-8.5 uses the same
  representative GCM/realization (ACCESS-ESM1-5/1) as SSP2-4.5 — S-05's own isolated-realization-
  spread finding (2.4-7.6% of the mean) means one GCM/realization is a representative draw, but SSP
  is a real, separate axis worth showing both sides of, not assumed similar.

### Addendum: regulatory-grounded `reg_cap` fertiliser level (D-105, 2026-08-19)

Added a 4th fertiliser level, `reg_cap`, alongside `historical`/`plus50pct_rate`/`plus50pct_freq`:
rate scaled (frequency held at historical) so the true, correctly area-weighted typical-year N
loading hits exactly the UK NVZ N-max for grassland (300 kg N/ha/yr, gov.uk) -- **T2 x1.926, T4
x2.062, T9 x10.897** (T9's real fertiliser use is so low it needs ~11x to reach the same absolute
ceiling T2/T4 reach at ~2x). Added to both S-05 (real drivers) and S-06 (bias-corrected drivers),
all 3 towers, full coverage. **Result: negligible effect even at the regulatory ceiling** -- T2
-0.9%/-0.7% (S-05/S-06), T4 +1.5%/+1.7%, T9 +0.1%/+0.1% -- consistent with and reinforcing the
"fertiliser schedule is not a meaningful CH4 lever" finding above, now anchored at a real regulatory
limit rather than an arbitrary +50%. See D-105 for the full derivation, including a real
double-counting bug the grounding check surfaced (in chat-only regulatory-comparison arithmetic,
never in production/scenario code -- no rerun of the existing 3 levels was needed).

---

## Context (original, 10-year version -- background/method unchanged by the 2050 extension)

Live-discussion follow-up to S-03: since TabICLv2 is a one-shot (not recursive-compounding)
foundation model, its 365-day S-03 horizon can be extended arbitrarily far without the
compounding-error risk a tree/SARIMAX rollout would carry. S-03's Variant A (driver removal) keeps
only the 10 columns a CMIP6 scenario can actually supply (TA/SWIN/PRECIP/DOY/season/livestock) —
almost exactly what `data/Simulated Climate Data/` (the Semenov et al. CMIP6/LARS-WG dataset,
2020–2090, 5 GCMs × 3 SSPs × 100 realizations, already used by S-04) provides directly. This
combines three prior pieces of work rather than building a new pipeline: S-03's Variant A feature
set, F-10/D-67's species-disaggregated livestock density (the same family behind the standing
`TabPFN+species` champion), and S-04's transient scenario machinery (real CMIP6 weather,
realization-level sampling, AOA extrapolation flagging).

**Livestock multiplier — user-confirmed Option B**: independent per-species multipliers (cattle,
sheep, lamb each scaled separately, 1×/2×/3× per species = 27 combos), not S-01/S-04's single
shared scalar. `fx_lsu_dens` is not independently resampled — it is rebuilt as the exact
LSU-weighted sum of the (scaled) species densities (`1.0×cattle + 0.1×sheep + 0.05×lamb`),
preserving F-10's own construction identity under any combination of multipliers.

## Method

- **Model**: TabICLv2 only (`rr.tabicl_forecast()`), zero-shot, per-tower — same architecture S-03
  already validated, no new HPO. `hist_target` = real `y_observed` (never `y_gapfilled`, same
  rationale as everywhere else this function is used — avoids gap-filler optimism).
- **Feature set (`FX_A_SPECIES`, 13 columns)**: S-03's own Variant-A `FX_A` (`fx_TA_mean/min/max`,
  `fx_SWIN_mean`, `fx_PRECIP_sum`, `fx_DOY_sin/cos`, `fx_is_growing`, `fx_is_winter`,
  `fx_lsu_dens`) + F-10's 3 species columns (`fx_cattle_dens`, `fx_sheep_dens`, `fx_lamb_dens`) —
  exactly the "BASE+species" config that is this project's standing forecasting champion, applied
  to Variant A's already-narrower BASE.
- **Anchor**: each tower's own last real `y_observed` date (T4/T9: 2023-12-29; T2: 2019-05-31,
  its usual data-scarce anchor) — matches S-01/S-04's convention of projecting forward from the
  end of all real data, not a mid-history anchor (there is no real future to leak from for a
  genuinely blind trajectory).
- **Horizon**: 10 years post-anchor (3,650 nominal days, Jan-1-based annual frames matching S-04's
  own LARS-WG-calendar convention), one vectorized `tabicl_forecast()` call per (tower, SSP, GCM,
  realization, multiplier-combo) — **not** a recursive day-by-day rollout.
- **Scope** (cost measured empirically before committing — a single 10-year call takes ~3.8s cold,
  ~1.2-1.3s warm): 3 towers × 2 SSPs (ssp245/ssp585, matching S-04) × 5 GCMs × **10 realizations/
  GCM** (stratified, S-04's own precedent for cutting this exact axis) × **27 species-multiplier
  combos** = 8,100 calls. Actual runtime: **2.54 hours**, 0 skipped/failed calls, 0 NaN in output.
- **AOA**: nearest-neighbour dissimilarity (Meyer & Pebesma 2021-style, same convention as
  `scenario_hybrid.dissimilarity_index()`), computed in `FX_A_SPECIES`'s own 13-dimensional space,
  training threshold precomputed **once per tower** (not per scenario call — the naive
  per-call approach would have recomputed an O(N_train²) distance matrix 8,100 times).
- **Data**: `results/s05_trajectory_realizations.csv` (81,000 rows = 8,100 calls × 10 years).

## Results

### 1. Species response is highly asymmetric — cattle dominates, far beyond its own LSU-weight share

Holding the other two species at 1×, scaling **cattle alone**:

| Tower | Cattle 2× | Cattle 3× | Sheep 3× | Lamb 3× |
|---|---|---|---|---|
| T2 | +1.3% | +2.3% | +1.7% | +10.0% |
| T4 | **+120.4%** | **+205.6%** | +5.4% | +10.0% |
| T9 | **+90.8%** | **+195.6%** | +22.0% | +13.6% |

At T4/T9, tripling cattle density alone roughly **triples the predicted FCH₄ mean** — a far larger
effect than tripling sheep or lamb (both stay under 25% even at 3×). This is *more* cattle-dominant
than the raw LSU-weight composition alone would predict: cattle contributes 74–88% of baseline
`fx_lsu_dens` at these towers (checked directly, real historical means), but its share of the
*response* is even higher than that — the model isn't just tracking the LSU-weighted aggregate, it
has genuinely learned cattle-specific signal beyond what the single `fx_lsu_dens` feature alone
could carry (the entire point of F-10's species-disaggregation, now shown to hold at CMIP6-scenario
scale, not just the real-historical-anchor scale F-10's own signal check used).

**Tower 2 is muted across every species** (max +2.3% even at cattle 3×) — consistent with this
project's repeated finding that T2 sits in a genuinely different, lower-density regime (D-18
lineage), not a new limitation introduced here.

**Update (S05-T2, D-95, 2026-08-10): does pooling T2's context with T4/T9's real (livestock-rich)
history rescue this muted response? No — exactly 0.0 percentage points of difference, for both
TabICLv2 and TabPFN.** Direct test of the hypothesis that T2's zero-shot solo call simply has no
historical livestock→CH4 covariation to learn from (T2's real `fx_lsu_dens` never exceeds ~0.71,
vs. T4/T9's own 1× baseline of ~5) and that pooling — the exact mechanism that rescued Tower 9 in
gap-filling, F-02/F-03 — might let it borrow T4/T9's learned cattle sensitivity instead. Pooled
context (T2+T4+T9 real history, `item_id`-tagged) vs. solo, same 3 combos × 2 SSPs:

| Model | Solo (cattle 3×) | Pooled (cattle 3×) | Solo (all-species 3×) | Pooled (all-species 3×) |
|---|---|---|---|---|
| TabICLv2 | +1.8% | +1.8% | +4.2% | +4.2% |
| TabPFN | +11.0–11.2% | +11.0–11.2% | +17.2–17.5% | +17.2–17.5% |

Not just small — an exact match to the decimal, both models, every combo/SSP. Mechanistic read:
this style of pooling (`item_id`-tagged rows in one batched in-context call) shares context, not
fitted parameters — unlike Track A's RF/XGB pooling, where every tree split is genuinely informed
by all towers' data together, a zero-shot forecaster's output for one series still appears to be
driven almost entirely by that series' own history in the query; other towers being present in the
same batch doesn't move Tower 2's forecast at all. **T2's muted response should be read as a
genuine model-extrapolation limit, not a fixable data-availability gap** — the "maybe pooling fixes
it" question is now closed empirically. Secondary finding: TabPFN's own solo response at T2
(+11–17.5%) is meaningfully larger than TabICLv2's (+1.8–4.2%), independent of pooling. Files:
`notebooks/07_scenario_analysis/s05_t2_pooled_test.py`,
`results/s05_t2_{pooled_trajectory,tabpfn_solo_trajectory,pooled_vs_solo_compare}.csv`.

Full table: `results/s05_species_marginal_response.csv`. Figure: `s05_species_response.png`.

### 2. Joint (all-species) scaling is close to additive — one exception worth flagging

| Tower | Baseline | Actual (3×/3×/3×) | Additive prediction | Synergy |
|---|---|---|---|---|
| T2 | 5.52 | 5.78 | 6.29 | −8.1% |
| T4 | 13.21 | 42.32 | 42.41 | **−0.2%** |
| T9 | 20.35 | 73.34 | 67.42 | **+8.8%** |

T4's response to scaling all three species together is **almost exactly** the sum of each species'
individual marginal effect (additive prediction 42.41 vs. actual 42.32) — a clean, reassuring
result: the model isn't producing surprising interaction effects at the best-covered tower. T9
shows a real +8.8% super-additive effect (joint scaling costs more than the sum of parts) — worth
flagging as a genuine, if modest, interaction, not noise (T9's individual deltas are large enough
that an 8.8% gap is unlikely to be pure sampling noise). T2's −8.1% is not read as a real effect —
its absolute deltas are tiny (0.1–0.5 nmol), so a percentage synergy figure there is dominated by
noise, not signal.

Full table: `results/s05_joint_vs_additive.csv`.

### 3. Realization/GCM spread is small — but this required a genuine correction to isolate correctly

**First pass (pooling year+realization+GCM together, matching S-04 Finding 1's own pooling
convention) gave a startling 32–69% band-width-as-%-of-mean** — an order of magnitude larger than
S-04's 1–5% finding. Investigated directly rather than reported at face value: a single fixed
(GCM, realization, SSP) already shows a 9.98–15.46 range across its own 10 years (T4) — the pooled
number was dominated by **genuine year-to-year weather variability within a decade**, not by which
of the 50 weather sequences (5 GCMs × 10 realizations) was drawn. Per-GCM means (pooled across
realizations/years) sit within 13.11–13.29 of each other at T4 — essentially identical.

**Isolating the realization/GCM axis alone (fixed year, varying only GCM+realization) gives a much
smaller 2.4–6.6% band** — consistent with S-04's own finding that weather-draw uncertainty is a
real but small effect. **This is a genuine, TabICLv2-specific finding, not a contradiction of
S-04**: S-04's hybrid has an explicit smooth Ridge trend component that inherently damps
year-to-year weather noise; TabICLv2, with no such smoothing backbone, is directly and much more
strongly sensitive to which specific year's weather sequence it's asked to predict from — a real
architectural difference between the two models' scenario behavior, not an error in either.

Full tables: `results/s05_realization_spread_pooled.csv` (the S-04-comparable-but-conflated
number), `results/s05_realization_spread_isolated.csv` (the corrected, apples-to-apples number).

### 4. AOA extrapolation risk is high in absolute level but flat over the 10-year horizon

Baseline (1×/1×/1×) AOA-flagged-% sits at **62–68% across all 3 towers**, essentially unchanged
from year 1 to year 10 post-anchor (moves by ~1 percentage point at most) — no evidence the
extrapolation risk *grows* as the trajectory extends further out, consistent with S-04's own
Finding 3 (climate drift alone doesn't push the scenario further out-of-envelope over time).

**The absolute level (62–68%) is much higher than S-04's own 9–15% baseline finding** — expected,
not concerning: S-04's own Finding 3 already flagged that the AOA check "can be diluted by many
in-range dimensions." S-04's feature space is much broader (~40+ dimensions, including AR history
and dummies); S-05's is deliberately narrow (13 `FX_A_SPECIES` columns, Variant A's whole point).
Fewer, more load-bearing dimensions dilute less — this is the concrete confirmation of S-04's own
hypothesis about dilution, not a new or contradictory finding.

Full table: `results/s05_aoa_trend.csv`. Figure: `s05_aoa_trend.png`.

### 5. SSP2-4.5 vs SSP5-8.5 divergence: small and inconsistent in direction at this horizon

| Tower | Early (yr 1-5) | Late (yr 6-10) |
|---|---|---|
| T2 | +0.76% | +0.64% |
| T4 | +0.09% | +0.34% |
| T9 | +0.09% | −0.01% |

Under 1% of the mean throughout, same order of magnitude as S-04's finding — the SSP choice is a
minor lever relative to the livestock question, consistent everywhere this project has looked at
it. (10 years post-anchor is a much shorter horizon than S-04's 2025–2050 window, so the "grows
toward the end of century" pattern S-04 found isn't expected to show up clearly here, and doesn't.)

Full table: `results/s05_ssp_divergence.csv`.

### 6. Daily-resolution sanity check (follow-up): the seasonal pattern is physically sensible

`s05_trajectory_10yr.py` discards each call's 3,650 daily predictions after averaging them into
`annual_mean` — fine for the questions above, but leaves no way to check whether the within-year
seasonal shape looks sensible. Re-ran a small, representative subset (3 towers × 2 SSPs × one
GCM/realization × the 3 multiplier combos already central to this write-up = 18 calls, ~24s) this
time keeping the full daily chain. **T4 baseline shows a clear summer peak (~18 nmol m⁻² s⁻¹ in
June) and winter trough (~3 nmol in December)** — physically consistent with real methane flux
seasonality (warmer-season microbial activity, grazing/manure timing), no negative predictions
anywhere, range (1.1–83.0 nmol) consistent with this project's own historical FCH₄ scale. The
cattle-driven separation between combos (Result 1) is visible growing across the 10-year horizon
in the daily chain, not just in the annual-mean summary.

Full data: `results/s05_daily_chains_subset.csv` (65,700 rows). Figure:
`s05_daily_chains_subset.png`. Not run for the full 8,100-call grid (would be ~29.6M rows, the
same problem S-04 avoided the same way — a small representative subset, not the full realization
grid).

## Practical implications

1. **F-10's species-split feature genuinely earns its place in a scenario context, not just on
   real historical anchors.** Cattle density is the dominant lever by a wide margin at T4/T9 —
   more so than its own LSU-weight share would predict — while sheep/lamb contribute comparatively
   little even at 3×. A digital-shadow interface built on the aggregate `fx_lsu_dens` alone would
   miss this asymmetry entirely.
2. **TabICLv2 is a viable long-horizon scenario forecaster**, and its one-shot (non-recursive)
   architecture is what makes a 10-year, 8,100-combination sweep tractable in ~2.5 hours — a
   genuinely different cost profile from S-04's B-10 tree/SARIMAX diagnostic, which needed a
   realization-count cut for a much shorter list of scenario points.
3. **Realization/GCM-choice uncertainty is small once correctly isolated (2-7%)**, echoing S-04 —
   but TabICLv2's year-to-year sensitivity (no smoothing trend component) is a real, separate
   source of variation worth reporting alongside it, not conflating into one number.
4. **AOA's absolute flagged-% depends heavily on feature-space breadth**, now confirmed a second
   time (S-04 → S-05) under two different feature spaces — any AOA number from this project should
   be read with its feature-space dimensionality stated alongside it, not compared raw across
   experiments.

## Explicit caveats

- **Scope cut, tracked**: 10 realizations/GCM (of 100 available), both SSPs, full 27-combo grid —
  a user-confirmed, deliberate choice given the 1 Sept deadline (full 100-realization scope would
  be ~85 hours for the full 27-combo grid). Re-running at higher realization count is legitimate
  follow-up, not required for the findings above (the isolated realization-spread check already
  shows this axis contributes little).
- **No ground-truth scoring** — like S-04, this is a genuinely blind trajectory (T4/T9's real
  record ends Dec 2023; T2's ends May 2019), so nothing here is a validated backtest. Read
  alongside S-02/S-04's own AOA findings, not as a replacement for them.
- **ssp126 unused** — available in `data/Simulated Climate Data/` (3 SSPs × 5 GCMs × 100
  realizations exist) but not run here, matching S-04's own SSP choice (245/585) for direct
  comparability. Legitimate follow-up if a third emissions pathway is wanted.
- **Single-model** — TabICLv2 only, no diagnostic cross-check against another model the way S-04
  ran the B-10 ensemble in parallel. Adding TabPFN (also one-shot, same cost profile) alongside
  would be a natural, cheap follow-up.
- **Livestock multiplier remains a naive uniform per-species scale** (not a self-consistent
  mechanistic management timeline) — same standing caveat S-01/S-04 already carry, now applied
  independently per species rather than once to the aggregate.

## Files

- `src/features/build_transient_scenario_drivers_species.py` — species-disaggregated driver-frame
  builder (sibling to `build_transient_scenario_drivers.py`, imports from it, modifies nothing).
- `notebooks/07_scenario_analysis/s05_trajectory_10yr.py` — main sweep script (committed,
  smoke-tested before the full run).
- `notebooks/07_scenario_analysis/s05_analysis.py` — this analysis (read-only against the sweep
  output, no new model fitting).
- `notebooks/07_scenario_analysis/s05_daily_chains_subset.py` — follow-up: saves full daily chains
  (not just annual_mean) for a small 18-call representative subset.
- `results/s05_trajectory_realizations.csv` (81,000 rows, raw sweep output).
- `results/s05_daily_chains_subset.csv` (65,700 rows, daily-resolution follow-up).
- `results/s05_trajectory_summary.csv`, `s05_realization_spread_pooled.csv`,
  `s05_realization_spread_isolated.csv`, `s05_ssp_divergence.csv`, `s05_aoa_trend.csv`,
  `s05_species_marginal_response.csv`, `s05_joint_vs_additive.csv` (derived tables).
- `results/figures/s05_summary/s05_trajectory_bands.png`, `s05_aoa_trend.png`,
  `s05_species_response.png`.

No `benchmarks.csv` rows (scenario-simulation output, not a point-forecast/interval-calibration
benchmark — same exclusion precedent as S-01/S-04/U-01/U-02/U-03).
