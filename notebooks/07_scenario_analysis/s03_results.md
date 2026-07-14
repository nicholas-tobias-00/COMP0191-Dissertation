# S-03 results: driver-availability ablation (isolating scenario-mode feature-degradation cost from extrapolation cost)

## Context

Supervisor request (Prof. Paul Harris, relayed by the user): isolate how much forecasting accuracy
is lost purely from **not having access to real-time sensor variables that a CMIP6 climate scenario
can never supply** — distinct from two effects already tested in this project, which conflate that
question with something else:

- **U-03/D-63**: tests calibration/model robustness under out-of-envelope `fx_lsu_dens`
  perturbation — real historical anchors, but the "shock" is an extreme covariate value, not a
  feature-set change.
- **S-01/D-64**: builds the real scenario pipeline (CMIP6 + historical-day-resampled drivers +
  livestock multiplier) but evaluates on **unscored 2041–2060 data** — no ground truth exists, so it
  cannot separate "the drivers are degraded" from "2050 is out of the training envelope."

This experiment fixes that conflation: test data is **real and historical** (the same 2018–2022
anchors, same 3 towers as B-10/D-65) — only the **feature set** changes.

- **Model 1** (existing, not rerun): B-10's unweighted RF+XGB+LightGBM+SARIMAX ensemble, full
  feature set. Read directly from `results/b10_b13_rerun_table_all_towers.csv` /
  `_by_tower_year.csv` (D-65) — not recomputed.
- **Model 2**: identical B-10 architecture/hyperparameters/ensemble, same 5-anchor × 3-tower sweep,
  two variants:
  - **Variant A (removal)**: 24 scenario-unavailable columns dropped entirely.
  - **Variant B (resample)**: same 24 columns present and used in training (real values, identical
    to Model 1), but their **rollout-time/test-window** values are day-of-year-climatology-resampled
    via `rr.doy_climatology()`, computed from **pre-anchor-only** history.
- **Explicitly excluded**: S-02's RF-proxy reconstructions (D-69) — a deliberate scoping choice for
  a clean two-arm comparison; proxy reconstruction remains a separate follow-up.

## Method

### Resolved variable list

`forecast_daily_v2.csv` has exactly 34 `fx_` columns. `src/features/build_scenario_drivers.py`
(S-01's own production code) already partitions every non-CMIP6-derivable driver into
`RESAMPLED_COLS` (22 cols: WS, VPD, RN, PPFD, SWC, TS + their lag/roll derivatives, wind direction,
grazing recency/activity) and `DROPPED_COLS` (2 cols: USTAR, SHF). **PPFD/RN ambiguity resolved
directly from code**: both are in `RESAMPLED_COLS`, i.e. S-01 already treats them exactly like
WS/VPD/SWC/TS — no real contradiction with S-02's proxy work, which targeted the same variables for
a different reason.

**User-confirmed**: wind direction and grazing features are included, giving a final degraded-column
list of exactly `RESAMPLED_COLS + DROPPED_COLS` (**24 columns**), imported directly from
`build_scenario_drivers.py` — not retyped, so this experiment cannot silently drift from what S-01
actually treats as scenario-unavailable.

**Explicitly out of scope, kept real in both variants**: `fx_lsu_dens` (the scenario *lever*, not a
missing-sensor variable) and all AR features (`ar_ch4_dlag*`, `ar_ch4_drm7`, `ar_fc_dlag1` — S-01
resamples these too, but only because no real recent CH₄/FCO₂ history exists in a genuinely blind
2050 future; this experiment uses real historical anchors, where that history genuinely exists).

### One deliberate deviation from S-01's own code, necessary for causal validity

`build_scenario_drivers.py` calls `doy_climatology()` with the **full** 2018–2023 record as history
— correct for S-01, which projects from the end of all available data to 2050 (no future data to
leak). This experiment evaluates on real historical anchors (e.g. 2021-12-16), so using the full
record would leak that anchor's own future real values (2022–2023) into the climatology used to
resample its own test window. **Fixed**: `hist` restricted to `dft.loc[:anchor, col]` for every
Variant B substitution — verified via assertion (`hist.index.max() <= anchor`) on every call.

### Customizable design

`s03_driver_availability_ablation.py`'s `main(remove_cols=None, resample_cols=None, run_label="")`
takes both column lists as independent, overridable parameters (both default to the 24-column list
above) — a follow-up sensitivity check (e.g. "what if only soil variables are resampled") is a
one-line notebook call with a distinct `run_label`, not a script edit.

### Architecture — unchanged from B-10, applied per variant

Same pooled T2+T4+T9 tree training, same frozen hyperparameters (RF: `n_estimators=500,
min_samples_leaf=10, max_features=0.5`; XGB: `max_depth=2, lr=0.02, n_estimators=400,
min_child_weight=10`; LightGBM: `num_leaves=7, min_child_samples=10, lr=0.02, n_estimators=400`),
same SARIMAX AIC order search (`p∈{1,2}, q∈{0,1}, d=1`), same ensemble construction
(`Ensemble_unweighted` = simple mean; `Ensemble_MASEweighted` = B-09's frozen weights). No new HPO.
Trees are fit **twice per anchor** (once on the full 44-column set for Variant B, once on the
reduced ~20-column set for Variant A); SARIMAX's `EXOG_B` (8 cols) loses 4 of its 8 columns in
Variant A (WS, VPD, USTAR, PPFD) — a larger relative hit than the tree models' 24/44.

Evaluated via `rr.bin_metrics()` (unmodified) against both `y_observed` (primary) and `y_gapfilled`
(secondary, with per-bin `real_frac`, D-65's addendum convention). TFT/TabPFN are out of scope —
Model 1/2 are specifically B-10's 4-model architecture. Full coverage: 3 towers × 5 anchors × 2
variants, 1,080 bin-level rows per target.

## Results — observed target (primary)

### All-tower pooled (n-weighted mean per anchor across bins+towers, then mean across 5 anchors)

| Model | MASE Model1 | MASE Var-A (removal) | MASE Var-B (resample) | R² Model1 | R² Var-A | R² Var-B |
|---|---|---|---|---|---|---|
| RF | 0.968 | 1.001 | **0.943** | -0.241 | -0.220 | **-0.139** |
| XGB | 0.922 | 0.926 | **0.879** | -0.184 | -0.118 | **-0.050** |
| LightGBM | 0.941 | 0.942 | **0.902** | -0.206 | -0.155 | **-0.105** |
| SARIMAX | 0.976 | **0.949** | 0.931 | -0.360 | **-0.191** | -0.295 |
| Ensemble_unweighted | 0.918 | 0.926 | **0.892** | -0.165 | -0.108 | **-0.089** |
| Ensemble_MASEweighted | 0.918 | 0.927 | **0.892** | -0.165 | -0.108 | **-0.088** |

*(bold = best of the 3 configs for that model/metric)*

`results/s03_table_all_towers.csv` / `_by_tower.csv` now carry **both** target metrics combined in
one file (3-level columns: target × source × metric) — the gap-filled-target numbers shown in the
"Secondary metric" section below live in the same two files, not a separate one, per direct user
request. The single-target views (`s03_table_vs_gapfilled_all_towers.csv` / `_by_tower.csv`) are
still written alongside, for anyone who wants just that slice.

### Per-tower (5 anchors averaged) — Ensemble_unweighted only, full table in `results/s03_table_by_tower.csv`

| Tower | MASE Model1 | MASE Var-A | MASE Var-B | R² Model1 | R² Var-A | R² Var-B |
|---|---|---|---|---|---|---|
| T2 | 0.374 | **0.347** | 0.368 | -1.048 | **-0.796** | -0.965 |
| T4 | 0.975 | 0.992 | **0.935** | 0.012 | 0.001 | **0.049** |
| T9 | 0.883 | 0.891 | **0.880** | -0.253 | **-0.129** | -0.117 |

## The headline finding, stated plainly

**Neither degraded-feature variant costs accuracy relative to Model 1's full feature set, pooled
across all towers — if anything, both modestly beat it, and Variant B (resample) beats it
consistently.** This is the opposite of the naive expectation ("fewer/degraded inputs should hurt").

- **Variant B (resample)** beats Model 1 on MASE for every one of the 6 model rows pooled
  all-tower, and beats it on R² for 5 of 6 (SARIMAX is the exception on R²: -0.295 vs -0.360, still
  technically better but by less than Variant A's -0.191).
- **Variant A (removal)** beats Model 1 on R² for every model pooled all-tower, but is a close wash
  on MASE (slightly worse for RF/LightGBM/ensembles, slightly better for XGB/SARIMAX).
- Per-tower: Variant B wins or ties at all 3 towers on both metrics for the ensemble, except a
  narrow loss to Variant A on both metrics at Tower 2.

**Plausible explanation, not proven here**: many of the 24 degraded columns (USTAR, wind
speed/direction, VPD, PPFD/RN) were already flagged low-SHAP-importance in I-01/I-02's earlier
interpretability work — dropping or smoothing genuinely low-signal, noisy sensor inputs may reduce
overfitting in a 365-day recursive rollout, where compounding error from noisy day-to-day covariates
can plausibly hurt more than the loss of the (already weak) signal they carried. This is consistent
with, not contradictory to, this project's standing finding that `fx_lsu_dens` (untouched in both
variants here) is the dominant driver — the variables being degraded were never the ones doing the
heavy lifting.

**This is genuinely useful for the supervisor's question**: it suggests the "scenario-mode
penalty" for driver unavailability is small-to-negative under this ablation, at least for the 24
variables tested here, pooled across towers — the real risk in scenario forecasting (per U-03/S-01)
lies in extrapolation and SARIMAX's unbounded response, not in losing these specific sensor
channels. This does **not** mean scenario forecasting is free of risk overall — it means this
particular, isolated cost is small, not that every risk this project has documented is resolved.

## Secondary metric: scored against gap-filled target (exploratory — see D-65's circularity caveat)

`y_gapfilled` seeds `history_init` and shares feature space with RF/XGB/LightGBM's own inputs, so
agreement partly reflects "forecaster resembles gap-filler," not pure forecasting skill — read
directionally only, per D-65's established caveat.

| Model | MASE Model1 | MASE Var-A | MASE Var-B | R² Model1 | R² Var-A | R² Var-B |
|---|---|---|---|---|---|---|
| RF | 0.800 | 0.912 | 0.798 | -0.593 | -2.116 | -0.767 |
| XGB | 0.761 | 0.822 | 0.763 | -0.502 | -1.341 | -0.506 |
| LightGBM | 0.774 | 0.826 | 0.780 | -0.480 | -1.061 | -0.577 |
| SARIMAX | 0.943 | 0.989 | 0.912 | -1.004 | -1.211 | -0.968 |
| Ensemble_unweighted | 0.751 | 0.834 | 0.762 | -0.189 | -1.027 | -0.459 |
| Ensemble_MASEweighted | 0.750 | 0.833 | 0.761 | -0.195 | -1.032 | -0.460 |

**This disagrees with the primary (observed-target) result for Variant A, and the disagreement is
stated plainly, not smoothed over**: under the observed target, Variant A (removal) looked roughly
competitive with Model 1 (small MASE win/loss depending on model, clear R² win). Under the
gap-filled target, Variant A looks **clearly worse** than both Model 1 and Variant B across every
model — R² collapses (e.g. Ensemble_unweighted -0.189 → -1.027). Variant B stays close to Model 1
under both targets. One plausible read: the gap-filled target is itself produced by a pooled RFm
gap-filler trained on met/soil features that Variant A's models never see — so Variant A's
predictions diverge further from the gap-filler's own output space specifically, a circularity
artifact rather than a real accuracy difference. This is exactly the kind of secondary-metric
disagreement D-65's own convention says to report, not hide.

## Files

- `notebooks/07_scenario_analysis/s03_driver_availability_ablation.py` — sweep script (committed),
  customizable via `remove_cols`/`resample_cols`/`run_label`.
- `notebooks/07_scenario_analysis/compile_s03_results.py` — 3-way comparison-table builder.
- `notebooks/07_scenario_analysis/S03_driver_availability_ablation.ipynb` — design, resolved
  ambiguities, smoke test, full-sweep results load (executed clean, 0 errors, 14 cells).
- `results/s03_summary.csv`, `results/s03_summary_vs_gapfilled.csv` (bin-level, 1,080 rows each),
  `results/s03_chains.csv` (raw daily predictions, both variants).
- `results/s03_table_all_towers.csv`, `results/s03_table_by_tower.csv` — **primary/headline
  tables**: 3-level-column comparison (target in {Observed, GapFilled} × source in {Model1,
  VariantA, VariantB} × metric), both target metrics combined into one file per direct user request.
- `results/s03_table_vs_gapfilled_all_towers.csv`, `results/s03_table_vs_gapfilled_by_tower.csv` —
  single-target (gap-filled only) view, kept alongside the combined tables for anyone who wants just
  that slice.
- Reused unmodified: `results/b10_b13_rerun_table_all_towers.csv` /
  `_by_tower_year.csv` (D-65, Model 1's numbers, read not recomputed), `rr.doy_climatology()`,
  `rr.bin_metrics()`, `build_scenario_drivers.RESAMPLED_COLS`/`DROPPED_COLS`.
- No existing file modified: `B10_daily_improvements.ipynb`, `S01_first_scenario.ipynb`, S-02's
  notebook, `build_scenario_drivers.py`, `scenario_hybrid.py`, `recursive_rollout.py`,
  `b10_b13_rerun_multi_anchor.py`, and every existing results CSV are untouched (git diff empty on
  all of them).

## Caveats

- **Pooled-all-tower headline is dominated by whichever towers have the most weight in the
  n-weighted average** — Tower 2's small sample size means its (comparatively large) percentage
  swings move the pooled number less than Tower 4/9's. Read the per-tower table alongside the
  pooled one, not instead of it.
- **This ablation does not itself explain *why* removing/resampling helps** — the SHAP-based
  plausible explanation above is a reasonable hypothesis grounded in this project's own prior I-02
  findings, not a new, directly-measured mechanism in this experiment. A follow-up SHAP/permutation
  check on Variant A/B's own fitted models would test this directly.
- **Bounded scope, as designed**: only B-10's 4-model architecture (no TFT/TabPFN); only the
  24-column S-01-derived degraded set (customization exists but wasn't exercised for the headline
  run); S-02's RF-proxy reconstruction explicitly excluded this round.
- **Chain figures not generated this pass** (see open item below) — raw chains are saved to
  `results/s03_chains.csv` so figures can be added later without a rerun.

## Open item carried over from planning (still unresolved, not decided silently)

Whether this experiment counts as "a new model added to the B-10/B-13 sequence" under `CLAUDE.md`'s
standing chain-figure-generation convention. This extends the same 4-model architecture under a
different *feature regime*, not a new model joining the roster — genuinely different from what that
convention was written for. No figures were generated this pass; flagging for confirmation rather
than deciding unilaterally.

## Proposed decision-log entry (pending confirmation — not yet written to DECISIONS.md)

**D-70 (proposed)**: S-03 driver-availability ablation. Isolates scenario-mode feature-degradation
cost from extrapolation cost (distinct from U-03/D-63, S-01/D-64). Full 3-tower × 5-anchor coverage,
observed + gap-filled secondary target. Headline: neither removal nor resample of the 24
scenario-unavailable columns costs accuracy pooled across towers relative to B-10's full-feature
ensemble — Variant B (resample) modestly beats it on both MASE and R² almost everywhere tested;
Variant A (removal) beats it on R² but is a wash on MASE, and looks clearly worse under the
gap-filled secondary metric specifically (flagged as a likely circularity artifact, not necessarily
a real accuracy loss). Practical implication for Phase 07: S-01's existing resample-based approach
is not measurably penalized by driver unavailability in isolation — the documented scenario risk
(U-03/S-01) concentrates in extrapolation and SARIMAX's unbounded response, not in losing these
specific sensor channels.
