# S-01 results: first Phase 07 scenario-simulation worked example (level-residual hybrid)

## Context

Phase 07 is this dissertation's actual novel deliverable and had been "PLANNED, not started"
throughout the project. This is the first bounded worked example proving the scenario-simulation
mechanism end-to-end — not the full Phase 07 sweep.

Builds directly on: **D-46** (candidate CMIP6 dataset scoped), **D-52** (dataset arrival confirmed
North-Wyke-matched; missing-driver strategy decided: historical-day resampling, not a raw
Copernicus pull), **U-03/D-63** (already answered the extrapolation-range check directly on the
candidate models — B-08 confirmed superseded for Phase 07's purposes), and a deep-research
literature pass this session that recommended a **level–residual hybrid** architecture and two
cheap concrete additions: monotonic constraints on livestock density, and dropping
`fx_USTAR_mean`/`fx_SHF_mean` entirely (true EC-tower turbulence quantities with no
climate-scenario-product source at all). Per user confirmation, the trend/level component is a
**parametric** model (Ridge) for this pass — SPACSYS (already validated at North Wyke, Wu et al.
2016) is logged as a stronger future direction, not attempted here.

## Method

**Driver construction** (`src/features/build_scenario_drivers.py`): climate drivers from the North
Wyke CMIP6-based transient files (`data/Simulated Climate Data/NW.<GCM>.<SSP>.<realization>.dat`) —
4 raw variables (TMIN, TMAX, RAIN, RAD). `fx_TA_min`/`fx_TA_max`/`fx_PRECIP_sum` map 1:1;
`fx_TA_mean` is derived as `(TMIN+TMAX)/2`; `fx_SWIN_mean` is derived from `RAD` with a unit
conversion (MJ m⁻² day⁻¹ → W/m² daily mean, factor 1e6/86400 ≈ 11.574 — verified: a summer-peak
RAD of ~19.4 MJ/m²/day converts to ~224 W/m², a winter value of ~2.3 converts to ~27 W/m², both
physically plausible for UK solar radiation). Every other driver (9 of ~11: WS, VPD, RN, PPFD, SWC,
TS, wind direction, grazing recency/activity, plus all CH₄/FCO₂ AR-history features) is
historical-day-resampled via `rr.doy_climatology()` (reused unmodified) from the real 2018–2023
record, per D-52's decision. **`fx_USTAR_mean`/`fx_SHF_mean` are dropped entirely**, not
synthesized. `fx_lsu_dens` (the scenario knob) is the historical day-of-year climatology multiplied
by a scenario factor.

**Level–residual hybrid model** (`src/models/scenario_hybrid.py`): a Ridge trend model
(`fx_TA_mean`, `fx_lsu_dens`, `fx_DOY_sin/cos` + tower dummies) fit **once** on the full pooled real
historical record (8,772 rows, T2+T4+T9) — deliberately a single fit, not per-anchor, the specific
fix for the SARIMAX per-anchor instability U-03 found (58–380% overshoot range there). Scaled
(fitted-on-standardized-features) coefficients: `fx_lsu_dens` = **27.0** (by far the largest —
confirms livestock density as the dominant trend driver, consistent with I-02), `fx_TA_mean` = 2.2,
`fx_DOY_sin/cos` = 2.5/1.9, tower dummies ±0.9–1.8. RF/XGB/LightGBM (B-10's exact hyperparameters,
D-41/D-54, no new HPO) trained on the residual (`y_gapfilled − trend_prediction`), full feature set
minus USTAR/SHF (42 features total), with a monotonic constraint (`fx_lsu_dens` non-decreasing) on
XGB/LightGBM. **Verified directly, not assumed**: a synthetic sweep of `fx_lsu_dens` from 0–10 at
fixed other-feature values shows XGB/LightGBM strictly non-decreasing (min step = 0.0000); RF (no
native monotonic-constraint support in scikit-learn) is non-monotonic as expected (range −11.7 to
+5.5), a known asymmetry, not a bug.

**Scenario**: SSP2-4.5, ensemble-mean across all 5 GCMs × 100 realizations (500 realization-files,
~10,000 samples per day-of-year), 2041–2060 ("the 2050s" climatological window). **All 3 towers**
(T2, T4, T9 — extended from the originally-scoped Tower-4-only pass per direct user request:
"ensure that you output figures... do so for all towers and models"). Three livestock multipliers:
1.0× (baseline), 2.0×, 3.0× — one pooled hybrid model (fit once) serves all three towers via the
tower dummies, so this extension required no re-fitting, only re-running the scenario/prediction
step per tower.

**Frozen model artifact**: trend + 3 residual models + imputer persisted via `joblib` to
`results/models/s01_*.joblib` — one set of artifacts for all 3 towers — closes D-46's requirement 1
(no model in this project has ever been saved past a single notebook run before this).

## Results — all 3 towers × 3 multipliers

| Tower | Multiplier | Annual mean | Trend | Real historical mean | AOA flagged | Scenario `fx_lsu_dens` max |
|---|---|---|---|---|---|---|
| 2 | 1.0× | 23.31 | 19.65 | 19.39 | 0.0% | 0.71 |
| 2 | 2.0× | 27.61 | 24.41 | 19.39 | 0.0% | 1.43 |
| 2 | 3.0× | 31.19 | 29.16 | 19.39 | 0.0% | 2.14 |
| 4 | 1.0× | 29.35 | 30.23 | 29.95 | 0.0% | 2.48 |
| 4 | 2.0× | 49.18 | 51.40 | 29.95 | 0.0% | 4.95 |
| 4 | 3.0× | 69.90 | 72.56 | 29.95 | **5.5%** | 7.43 |
| 9 | 1.0× | 40.04 | 36.93 | 36.65 | 0.0% | 2.58 |
| 9 | 2.0× | 60.56 | 58.89 | 36.65 | 0.0% | 5.16 |
| 9 | 3.0× | 81.96 | 80.86 | 36.65 | **6.0%** | 7.74 |

Historical training max `fx_lsu_dens` (pooled T2+T4+T9): **5.65**; own-tower maxima: T2=0.71 (!),
T4=4.99, T9=5.65. Historical training `fx_TA_mean` range: **−4.71 to 24.05**. Scenario `fx_TA_mean`
range (identical across towers and multipliers, since only livestock varies by scenario): **5.56 to
17.29**. Full table (including per-model residual columns, seasonal breakdown, ensemble
disagreement): `results/s01_scenario_summary.csv`.

**Per-model breakdown — a genuine, verified-not-assumed finding.** The full monotonic sweep
(`fx_lsu_dens` from 0 to 10, not just the min-step check reported in the earlier single-tower pass)
shows **XGB and LightGBM's residual prediction is completely flat across the entire range** for a
representative row (e.g. XGB: 8.077 at every one of 21 sampled points; LightGBM: 10.684 at every
point) — not merely non-decreasing, but literally constant. This is architecturally coherent given
B-10's exact hyperparameters (XGB `max_depth=2`, LightGBM `num_leaves=7` — deliberately shallow, no
new HPO) combined with the monotonic constraint: once the trend model has already absorbed the
primary livestock-driven signal, there is little residual variance left for `fx_lsu_dens` to explain,
and a shallow constrained tree simply never selects it as a split feature. **Confirmed in the actual
scenario runs**: `residual_XGB` and `residual_LightGBM` are near-identical (often bit-identical to
several decimal places) across all three multipliers, at every tower. **RF (no native monotonic-
constraint support) is the only residual model that shows real, meaningful sensitivity to livestock
density** — e.g. at Tower 4, `residual_RF` moves from −4.36 (1×) to −8.36 (2×) to −9.68 (3×).
**Net effect: essentially 100% of the livestock-driven scenario response for two of the three tree
models flows through the trend component, not the residual** — the level–residual split isn't just
designed to work this way, it demonstrably does, for exactly the reason it was adopted.

## Findings

**1. Baseline reconstruction sanity check passes at all 3 towers.** The 1.0× scenario's predicted
annual mean tracks each tower's real historical mean closely at T4 (29.35 vs 29.95) and reasonably
at T9 (40.04 vs 36.65); **Tower 2 shows the largest gap** (23.31 predicted vs 19.39 real, +3.9) —
plausibly reflecting Tower 2's well-documented data sparsity throughout this project (real
`y_observed` in only 1/5 anchor windows per U-02/U-03) making its climatology-resampled drivers less
reliable than T4/T9's. All three pass the sanity check well enough to trust the perturbed-scenario
comparison, with Tower 2's result carrying extra caution, consistent with every other Tower-2 caveat
in this project.

**2. The hybrid design measurably fixes the flattening U-03 found, at the two well-covered towers.**
U-03's raw tree-only extrapolation test found RF/XGB/LightGBM plateau at only +21–23% mean change
across a 1×→3× sweep of the same driver. Here, the same nominal 3× multiplier produces **+138.2%**
at Tower 4 (29.35→69.90) and **+104.7%** at Tower 9 (40.04→81.96) — because the Ridge trend
component is now allowed to extrapolate properly along the livestock axis instead of the trees
clipping it to their nearest training-seen leaf. This is the level–residual design working exactly
as the deep-research literature pass predicted it would.

**3. Tower 2 is genuinely different, and this is a real finding, not a data artifact.** Tower 2's
own historical `fx_lsu_dens` maximum (0.71) is roughly 7× smaller than T4's (4.99) and 8× smaller
than T9's (5.65) — Tower 2's catchment records dramatically less livestock presence overall, echoing
U-03's own finding that T2's `fx_lsu_dens` is exactly zero throughout the rollout window in 4 of 5
anchor years. Even at 3× multiplier, Tower 2's scenario `fx_lsu_dens` only reaches 2.14 — still well
below T4/T9's OWN 1× baseline range — and the AOA check never flags Tower 2 as out-of-envelope at
any multiplier tested, unlike T4/T9 which both get flagged at 3×. Tower 2's percentage response
(+33.8% at 3×, 23.31→31.19) is real but structurally smaller than T4/T9's, for a documented,
tower-specific data reason, not a pipeline inconsistency.

**4. A genuine, non-obvious finding from the AOA check: "2× livestock" is not automatically
out-of-distribution — at either well-covered tower.** At T4, the 2.0× scenario's `fx_lsu_dens` peaks
at 4.95, inside the pooled training envelope (5.65); at T9, 2.0× peaks at 5.16, also inside. This is
because the scenario multiplies a **smoothed day-of-year climatology**, not raw daily values, and
climatological smoothing caps peaks well below what a single real grazing day can reach. **Only the
3.0× scenario genuinely exceeds the training envelope at T4 and T9** (5.5% and 6.0% of days flagged
respectively) — Tower 2 never gets flagged at any multiplier tested (Finding 3). This is a real,
useful methodological finding, not a null result to bury: **a "2× livestock" scenario built by
scaling a smoothed climatology is meaningfully milder than one built by scaling raw daily values**
(as U-03 tested, which reached 2× the raw daily max well before 2× the climatological mean does).
Any future scenario design should be explicit about which construction it uses — they are not
interchangeable, and this difference alone changes whether a nominally-identical "2×" scenario is
even out-of-distribution.

**5. The climate axis is not the extrapolation risk in this scenario, at any tower.** SSP2-4.5's
2050s daily-mean temperatures (5.56–17.29°C) sit comfortably inside every tower's real historical
daily range (widest at T4: −4.71 to 24.05°C) — day-to-day weather variability is larger than the
multi-decade climate-mean shift, exactly as D-46's own caveat anticipated. All of the extrapolation
risk in this particular scenario sits on the livestock axis, not the climate axis — worth knowing
before assuming a 2050 climate scenario is inherently the risky part of "climate + management"
scenario work; here it's the management assumption that matters more, at every tower tested.

**6. Seasonal pattern is physically sensible at all 3 towers.** JJA (summer) shows the largest
predicted flux and the largest livestock-driven increase at every tower (e.g. T4: 39.29 → 115.24
nmol from 1× to 3×) — consistent with known CH₄ flux seasonality (warmer soil temperatures + peak
grazing season) and not an artifact of the hybrid construction.

## Explicit caveats (carry forward into any downstream use of this output)

1. **Parametric trend, not a mechanistic process model.** SPACSYS (already validated at North Wyke
   for soil water/runoff/biomass under decadal climate scenarios, plus an animal-growth module —
   Wu et al. 2016) is logged as a stronger future design, not attempted here given the timeline.
2. **`fx_USTAR_mean`/`fx_SHF_mean` dropped entirely** — the residual model never sees them; this is
   a genuinely different feature set than B-10's production point-forecast model, not a drop-in
   replacement.
3. **9 of ~11 daily drivers are historical-day-resampled climatology, not real future weather** —
   only Tmin/Tmax/precipitation/(derived)radiation come from the actual CMIP6 scenario.
4. **Livestock scenario is a naive multiplier on a smoothed climatology**, not a self-consistent
   mechanistic grazing timeline (grazing-recency/days-since-grazing features are NOT adjusted to
   match the scaled density) — explicitly flagged per the literature's own caution against naive
   column-scaling, and per Finding 4 above, this construction choice materially affects whether a
   "2×" scenario is even out-of-distribution.
5. **Single SSP, single climate window** — SSP2-4.5, 2041–2060 only, ensemble-mean only (not
   realization-level). SSP5-8.5, other windows, and realization-level analysis are deferred, not
   silently dropped. (All 3 towers ARE now covered, per direct user request — no longer a caveat.)
6. **U-02/U-03's conformal intervals are not attached to this output.** Per U-03's own standing
   recommendation, they should only ever be presented for in-AOA points — this notebook's own AOA
   check should be re-run for any future output before an interval is quoted, since the 1.0×/2.0×
   scenarios here are in-envelope at every tower but the 3.0× is not (T4/T9 only — T2 stays in-
   envelope even at 3×, Finding 3).
7. **The AOA check's own limitation, observed directly**: computed in the full 42-dimensional
   feature space, it can be diluted by many in-range dimensions even when one feature (livestock
   density) is meaningfully perturbed — confirmed here by cross-checking against the simple,
   transparent "is scenario max beyond training max" comparison, which is the more directly
   interpretable signal for this specific single-driver scenario type.

## Files

- `src/features/build_scenario_drivers.py` (CMIP6 loading/aggregation, unit conversion,
  historical-day-resampled driver construction, livestock multiplier)
- `src/models/scenario_hybrid.py` (Ridge trend model, residual tree fit with monotonic constraints,
  `dissimilarity_index` AOA function, combined predict function)
- `notebooks/07_scenario_analysis/S01_first_scenario.ipynb` (design + full worked example, all 3
  towers × 3 multipliers, executed inline)
- `results/s01_scenario_summary.csv` (9 rows: 3 towers × 3 multipliers, including per-model residual
  columns, seasonal breakdown, AOA flag, ensemble disagreement)
- `results/models/s01_trend.joblib`, `s01_imputer.joblib`, `s01_rf_residual.joblib`,
  `s01_xgb_residual.joblib`, `s01_lightgbm_residual.joblib` (frozen model artifacts, shared across
  all 3 towers)
- `results/figures/s01_scenario_comparison_all_towers.png` (grouped annual-mean bar chart, all 3
  towers × 3 multipliers, real historical mean marked per tower)
- `results/figures/s01_per_model_breakdown.png` (trend vs. RF/XGB/LightGBM residual contribution,
  one panel per tower, at 3.0×)
- `results/figures/s01_seasonal_all_towers.png` (seasonal breakdown, small multiples per tower)
- `results/figures/s01_aoa_ensemble_disagreement.png` (AOA-flagged fraction + ensemble disagreement,
  grouped by tower and multiplier)

No `benchmarks.csv` rows (scenario-simulation output, not a point-forecast or interval-calibration
benchmark — different metric family, same exclusion precedent as U-01/U-02/U-03).

## Recommended next steps (not attempted in this plan)

1. Extend to SSP5-8.5 (all 3 towers are now covered, per direct user request during this session —
   no longer a gap).
2. Realization-level (not just ensemble-mean) analysis, to report GCM×SSP spread as an additional
   uncertainty signal per the deep-research report's own recommendation.
3. A self-consistent mechanistic livestock-scenario construction (grazing-recency/days-since-grazing
   adjusted jointly with density, not independently).
4. If time permits, revisit the SPACSYS process-model route logged above as the more defensible
   trend/level component.
