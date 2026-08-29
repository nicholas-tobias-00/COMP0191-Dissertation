# S-03 results: driver-availability ablation (isolating scenario-mode feature-degradation cost from extrapolation cost)

## At a glance — condensed summary, all 11 models (observed target, all-tower pooled)

Model 1 = B-10's full-feature config for each model (read from its own existing table, not
rerun). Variant A = the 24 scenario-unavailable columns dropped entirely. Variant B = same columns
real in training, climatology-resampled at rollout time only. **MASE is this project's primary
metric** (CLAUDE.md convention); R² kept alongside per the project's standing table format.
Bold = best of the 3 configs for that model/metric.

| Model | MASE M1 | MASE A (removal) | MASE B (resample) | R² M1 | R² A | R² B | Verdict vs. Model 1 |
|---|---|---|---|---|---|---|---|
| RF | 0.968 | 1.001 | **0.943** | -0.241 | -0.220 | **-0.139** | B beats M1 on both; A worse on MASE |
| XGB | 0.922 | 0.926 | **0.879** | -0.184 | -0.118 | **-0.050** | B beats M1 on both |
| LightGBM | 0.941 | 0.942 | **0.902** | -0.206 | -0.155 | **-0.105** | B beats M1 on both |
| SARIMAX | 0.976 | **0.949** | 0.931 | -0.360 | **-0.191** | -0.295 | Both beat M1 on both metrics (A more so) |
| Ensemble_unweighted | 0.918 | 0.926 | **0.892** | -0.165 | -0.108 | **-0.089** | B beats M1 on both |
| Ensemble_MASEweighted | 0.918 | 0.927 | **0.892** | -0.165 | -0.108 | **-0.088** | B beats M1 on both |
| TFT | 0.972 | **0.971** | 0.976 | -0.363 | **-0.276** | -0.492 | **A beats M1; B is *worse* than M1 on both — the one reversal** |
| TabPFN | 0.855 | **0.840** | 0.857 | -0.122 | **-0.091** | -0.138 | A beats M1 on both; B slightly worse on both |
| DLinear | 1.460 | **1.176** | 1.245 | -2.068 | **-0.573** | -0.972 | Both beat M1 dramatically (A more so) |
| LSTM | 1.151 | 1.156 | **1.062** | -1.357 | -0.824 | **-0.638** | B beats M1 on both; A ~flat on MASE, better R² |
| TabICLv2 | 0.930 | 0.936 | **0.916** | -0.330 | -0.334 | **-0.277** | B beats M1 on both; A ~flat/slightly worse |

**Reading this table**: on MASE, **Variant B (resample) beats or ties Model 1 for 9 of 11 models** —
only TFT is genuinely worse (0.976 vs 0.972) and TabPFN is a near-wash (0.857 vs 0.855). **Variant A
(removal) is far more mixed**: a clear win for SARIMAX/TabPFN/DLinear, a wash or slightly-worse
result for RF/LightGBM/both ensembles/LSTM/TabICLv2, and genuinely ambiguous for TFT/XGB (ties or
tiny losses). **TFT is the one model where the headline direction reverses on R²** — Variant B makes
it measurably *worse* than Model 1 (-0.363 → -0.492), the opposite of every other model tested.
**Bottom line for Phase 07**: the "driver-availability cost is small, even beneficial" finding holds
for the production-recommended ensemble and most of the roster, but does not transfer uniformly —
it does not hold for TFT.

Full per-tower breakdown: `results/s03_table_by_tower.csv`. Gap-filled secondary metric (same
circularity caveat throughout): `results/s03_table_all_towers.csv`'s `GapFilled` column block, or
`s03_table_vs_gapfilled_all_towers.csv` for a single-target view.

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

## Resolved: chain figures generated, D-70 logged (superseding the two notes below)

The two sections immediately below (an "open item" about chain figures and a "proposed" decision-log
entry) were the pre-finalization draft state of this document — both were resolved shortly after:
chain figures were generated and merged into `results/b10_b13_full_chains.csv`/`b10_b13_chain_plots.py`
per direct user request, and D-70 was written to `DECISIONS.md`. Left in place below as the original
record; see `DECISIONS.md`'s D-70 entry for what was actually committed.

## Open item carried over from planning (historical — resolved, see above)

Whether this experiment counts as "a new model added to the B-10/B-13 sequence" under `CLAUDE.md`'s
standing chain-figure-generation convention. This extends the same 4-model architecture under a
different *feature regime*, not a new model joining the roster — genuinely different from what that
convention was written for. No figures were generated this pass; flagging for confirmation rather
than deciding unilaterally.

## Proposed decision-log entry (historical — this is what actually became D-70)

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

---

## Addendum: model-roster extension to the full 11-model B-10/B-13/B-16 roster (D-70 follow-up)

**This was a real scope gap, not a deliberate exclusion, and it was missed until the user asked
directly** ("does S-03 have TabPFN and TabICL??") whether the ablation covered the foundation/DL
models B-13/B-16 later added to the project's recursive-rollout roster. It did not — the original
design explicitly scoped Model 1/2 to "B-10's 4-model architecture" (RF/XGB/LightGBM/SARIMAX +
2 ensembles), and that line was true when written but stopped being the *complete* picture once
TFT/TabPFN (B-13) and DLinear/LSTM/TabICLv2 (B-09 extension + D-66) joined the standing 11-model
roster used everywhere else (I-02, U-02, U-03, `b10_b13_full_chains.csv`). This addendum closes that
gap: same two variants (A_removal, B_resample), same 2018–2022 anchors, same 3 towers, run against
all 5 remaining models.

### Method (new script: `s03_model_roster_extension.py`)

- **TabPFN, TabICLv2**: zero-shot, per-tower/per-anchor, daily track — mirrors
  `b16_foundation_models_v3.py`'s integration exactly (`mode="local"`, never pooled). Variant A/B
  covariate construction reuses `climatology_substitute()`/`DEFAULT_DEGRADED_COLS` from the original
  S-03 script unchanged.
- **TFT, DLinear, LSTM**: pooled per anchor (T2+T4+T9), hourly Track B, rolled out separately per
  tower — mirrors `b16_dl_models_v3.py`/`b10_b13_dl_extension.py` exactly (TFT: D-45/B-13a recipe,
  weight_decay=1e-3/90-day val carve-out/patience=5; DLinear/LSTM: plain B-09 recipe, no val split).
  No new HPO anywhere.
- **Hourly column mapping (a real design decision, not an oversight)**: of the 24 daily degraded
  columns, 12 have a direct hourly analogue in `forecast_features_v2.csv` (verified against
  `build_forecasting_matrix_v2.py`'s source — literally the same underlying hourly series the daily
  columns are `.resample("D").mean()` of): WS/VPD/RN/PPFD/SWC/TS means, wd_sin/cos, grazing_active→
  `fx_graze`, days_since_grazing, USTAR, SHF→`fx_shf3`. The remaining 12 (the SWC/TS daily
  lag-7/14/21/28 + rolling-7/14 ladder) have **no hourly equivalent at all** — a daily-resolution
  memory feature simply doesn't exist at hourly resolution — and are out of scope for the hourly DL
  track, stated explicitly rather than silently dropped. Variant B's climatology-resampling patches
  both the encoder's recent-history window and the decoder's future window for post-anchor dates
  (the DL rollout's encoder window slides into post-anchor dates as the chain progresses, unlike the
  tree/SARIMAX variant B's single combined future-frame).
- Deliberately reuses S-03's own v2 data (`forecast_daily_v2.csv`/`forecast_features_v2.csv`), not
  B-16's v3/F-10-enriched data — keeps this experiment isolated to the driver-availability question.
- Smoke-tested (1 anchor, all 3 towers, all 5 models) before the full 5-anchor sweep, per the
  project's standing full-coverage-by-default convention.

### Results — observed target, all-tower pooled (same aggregation as the table above)

| Model | MASE Model1 | MASE Var-A | MASE Var-B | R² Model1 | R² Var-A | R² Var-B |
|---|---|---|---|---|---|---|
| TFT | 0.972 | 0.971 | 0.976 | -0.363 | **-0.276** | -0.492 |
| TabPFN | 0.855 | **0.840** | 0.857 | -0.122 | **-0.091** | -0.138 |
| DLinear | 1.460 | **1.176** | 1.245 | -2.068 | **-0.573** | -0.972 |
| LSTM | 1.151 | 1.156 | **1.062** | -1.357 | -0.824 | **-0.638** |
| TabICLv2 | 0.930 | 0.936 | **0.916** | -0.330 | -0.334 | **-0.277** |

*(bold = best of the 3 configs for that model/metric; the 6 original models' rows are unchanged from
the table above and now live in the same `results/s03_table_all_towers.csv`.)*

### The refined headline, stated plainly (this changes the original finding's generality, not its direction)

**The full 11-model roster does NOT uniformly confirm "both variants beat Model 1" — it refines it.**
On MASE (this project's primary metric): **Variant B (resample) beats or ties Model 1 for 9 of the
11 models** (only TFT is genuinely worse, 0.976 vs 0.972, and TabPFN is a near-wash). **Variant A
(removal) is much more mixed across the extended roster** — it clearly helps SARIMAX/TabPFN/DLinear,
but is flat-to-worse for RF/XGB/LightGBM/both ensembles/TFT/LSTM/TabICLv2.

**TFT is the one genuine reversal worth flagging on its own**: R² gets *measurably worse* under
Variant B specifically (-0.363 → -0.492), the opposite direction from every tree/ensemble/foundation
model tested. A plausible mechanism (not directly measured here): TFT's attention mechanism may be
more sensitive to the *smoothness* discontinuity a climatology-substitution introduces at the
rollout boundary than the tree models are (trees split on thresholds; a smoothly-resampled future
series is a genuinely different distribution shape at the point features switch from real to
resampled) — flagged as a hypothesis for a future interpretability follow-up, not a proven mechanism.

DLinear/LSTM (the two weakest models in the whole B-09-B-16 sequence, per their own established
negative-R² baseline) both still improve under degradation, same direction as the original 6-model
finding — consistent with the "low-signal noisy inputs cost more via rollout-compounding than they
contribute in real skill" explanation already proposed for the tree models.

**Practical implication, updated**: the original "driver-availability cost is small, even
beneficial" finding **holds for the production-recommended Ensemble (still B-10's own 4-model
architecture) and for most of the extended roster**, but does **not** hold uniformly for every
architecture — TFT specifically shows the opposite pattern under resampling. Anyone relying on TFT
(not the standing recommendation, but present in the roster) for scenario work should not assume
this ablation's reassurance transfers to it.

### Gap-filled secondary metric (same circularity caveat as the primary table)

Same pattern as the original 6-model finding, extended: TFT/DLinear/LSTM (whose training target is
`y_observed`, not `y_gapfilled`) score worse under gap-filled scoring with no consistent
variant-ordering signal; TabPFN/TabICLv2 (also `y_observed`-trained) likewise show no clean
improvement pattern under this secondary metric — full numbers in `results/s03_table_all_towers.csv`
(`GapFilled` column block) and `s03_table_vs_gapfilled_all_towers.csv`.

### Files added/modified this addendum

- `notebooks/07_scenario_analysis/s03_model_roster_extension.py` — new script (committed), the 5
  new models' sweep, smoke-testable via `python s03_model_roster_extension.py smoke`.
- `notebooks/05_benchmarking/b10_b13_chain_plots.py` — fixed a latent bug: `plot_chain()`'s
  `y_true_col` selection checked `model in ("TFT","DLinear","LSTM")` verbatim, which silently missed
  every variant-suffixed column (e.g. `TFT_S03_A_removal`) added by this addendum, always falling
  back to the wrong ground-truth series for those columns. Fixed to strip the `_S03_*` suffix before
  the check. `MODEL_COLORS` extended with 10 new entries (muted tint per variant, matching the
  existing convention).
- `notebooks/07_scenario_analysis/compile_s03_results.py` — `MODEL_ORDER` extended to all 11 models;
  Model 1's numbers for the 5 extension models are now built by concatenating the raw per-bin
  summary files (`b10_b13_dl_extension_summary*.csv`, `b10_b13_tabicl_extension_summary*.csv`) with
  the original `b10_b13_rerun_summary*.csv` and reusing `build_pooled`/`build_by_tower` — verified to
  reproduce the previously-published `b10_b13_rerun_table_all_towers.csv` bit-for-bit before
  adopting this approach, rather than reading each family's own differently-shaped pre-built table.
- `results/s03_summary.csv`, `s03_summary_vs_gapfilled.csv` — extended in place (1,080 → 1,980 rows
  each); `results/s03_chains.csv` — extended in place with 5 new model columns + `y_true_tft` (row
  count unchanged, 10,950); `results/b10_b13_full_chains.csv` — extended in place with 10 new
  variant-suffixed columns (row count unchanged, 5,475); 165 new chain figures in
  `results/figures/b10_chains/` (5 models × 2 variants × 15 tower/anchor combos + the original 12
  tree/ensemble columns' own figures, all regenerated in the same pass — 495 total figures now).
- `results/s03_table_all_towers.csv`, `s03_table_by_tower.csv`,
  `s03_table_vs_gapfilled_all_towers.csv`, `s03_table_vs_gapfilled_by_tower.csv` — regenerated with
  the full 11-model roster (previously overwritten these files still exist, now superseded).

---

## Second addendum (D-83, 2026-08-08): climatology MASE + TabICL-sourced data, full roster wired
## into this notebook

Three changes: (1) MASE baseline switched from chain-persistence to day-of-year climatology,
matching D-80's project-wide convention change (2026-08-02, postdates this experiment's original
run by three weeks); (2) Variant A/B rerun on `forecast_daily_v2_tabicl.csv` (D-79's TabICL-sourced
gap-filling — schema-identical to the original file, confirmed by direct diff: only
`y_gapfilled`/`ar_ch4_*` differ, every `fx_` driver and `y_observed` are byte-identical); (3) the
full 11-model roster (the addendum above) wired directly into `S03_driver_availability_ablation.ipynb`
for the first time — it had only ever run as a standalone script.

**A design tension resolved along the way.** Model 1 was originally read from an existing,
never-rerun table (D-65, RF-sourced, persistence-scored). Swapping only Variant A/B to
TabICL-sourced data while leaving Model 1 untouched would reintroduce exactly the data-source/
feature-availability conflation this experiment exists to avoid. Fixed by recomputing Model 1 too,
wherever a TabICL-sourced daily file makes that possible — via the *same* code path as the real
ablation (`remove_cols=[]`/`resample_cols=[]`/`degraded_cols=[]` collapses both variants to an
identical, fully real, undegraded feature set), so Model 1 and Variant A/B stay guaranteed
comparable. TFT/DLinear/LSTM have no TabICL-sourced hourly data anywhere in this project
(`build_forecasting_matrix_v2_tabicl.py`'s own docstring: "hourly does not depend on the
gap-filled CH4 series at all") — these 3 stay RF-sourced throughout (Model 1 *and* Variant A/B),
an unavoidable limit, not a gap introduced here; their existing chains were rescored with the new
climatology baseline rather than retrained for a metric-only change.

New committed script: `s03_climatology_tabicl_update.py` (~21 min, both halves smoke-tested
first). Output: `results/s03_table_all_towers_climatology_tabicl.csv` /
`s03_table_by_tower_climatology_tabicl.csv`.

### Result 1: the original finding replicates, qualitatively unchanged

For every one of the 11 models, Variant B (resample) still beats or is within noise of Model 1 on
MASE, and the removal/resample ordering matches the original persistence-scored, RF-sourced run
(TFT remains the one genuine reversal; TabICLv2 remains a close wash). Driver degradation itself
still does not cost material accuracy under either the climatology-baseline switch or the TabICL
data swap.

### Result 2 (unplanned, and the more consequential one): TabICL is a bad *training target* for the tree/SARIMAX/ensemble family

D-80 found TabICL-sourced *context* makes TabPFN/TabICLv2 forecasting modestly worse (MASE +0.05
to +0.10). Here, RF/XGB/LightGBM/SARIMAX/the 2 ensembles — which train directly ON `y_gapfilled`
as their fitting target, not just condition on it — show a much larger absolute MASE increase
(roughly 1.3–2.5x) at **all 3 towers**, not just the sparse ones (Tower 4, the best-covered tower,
shows the single largest jump: climatology-MASE 2.487 vs. an apples-to-apples ~0.96–1.06 range
checked directly on RF-sourced data this session).

**Full table (all-tower pooled, climatology MASE, TabICL-sourced where possible) — MASE first
per this project's standing convention, R² rightmost. Both target choices shown side by side:
`Observed` is this project's primary, authoritative evaluation convention (train on gap-filled,
evaluate on observed, D-36/D-37); `GapFilled` is a secondary/exploratory metric only — several of
these models (RF/XGB/LightGBM/SARIMAX/both ensembles) train directly ON `y_gapfilled`, so scoring
against that same series is partially circular and inflates their apparent accuracy rather than
testing real generalization (D-65's standing caveat). Read `Observed` as the number that matters;
`GapFilled` for context only.**

| Model | Model1 MASE (obs) | VarA MASE (obs) | VarB MASE (obs) | Model1 MASE (gf) | VarA MASE (gf) | VarB MASE (gf) | Model1 R² (obs) | VarA R² (obs) | VarB R² (obs) |
|---|---|---|---|---|---|---|---|---|---|
| RF | 2.270 | 2.037 | 1.814 | 1.990 | 1.779 | 1.564 | −7.051 | −4.530 | −3.400 |
| XGB | 1.297 | 1.259 | 1.159 | 1.139 | 1.136 | 1.109 | −1.611 | −1.632 | −1.322 |
| LightGBM | 1.593 | 1.376 | 1.316 | 1.274 | 1.149 | 1.166 | −2.682 | −2.327 | −1.933 |
| SARIMAX | 1.108 | 1.056 | 1.046 | 1.038 | 0.976 | 0.991 | −1.416 | −1.153 | −1.164 |
| Ensemble_unweighted | 1.454 | 1.346 | 1.250 | 1.133 | 1.111 | 1.054 | −2.104 | −1.728 | −1.394 |
| Ensemble_MASEweighted | 1.455 | 1.346 | 1.250 | 1.133 | 1.111 | 1.055 | −2.101 | −1.733 | −1.398 |
| TFT | 0.841 | 0.822 | 0.840 | 0.878 | 0.766 | 0.790 | −0.363 | −0.276 | −0.492 |
| TabPFN | 0.733 | 0.725 | 0.742 | 1.017 | 1.005 | 1.006 | −0.122 | −0.091 | −0.138 |
| DLinear | 1.265 | 1.003 | 1.055 | 1.308 | 0.937 | 0.998 | −2.068 | −0.573 | −0.972 |
| LSTM | 0.956 | 0.958 | 0.906 | 1.018 | 1.016 | 0.861 | −1.357 | −0.824 | −0.638 |
| TabICLv2 | 0.782 | 0.803 | 0.783 | 1.046 | 1.019 | 1.031 | −0.330 | −0.334 | −0.277 |

(`gf` = GapFilled-target MASE. GapFilled R² also computed, omitted here for width — see
`results/s03_table_all_towers_climatology_tabicl.csv` for the full Observed+GapFilled × MASE+R²
matrix. Note the direction reverses for RF/XGB/LightGBM/SARIMAX/both ensembles — `gf` MASE is
*lower* than `obs` MASE for all six, the circularity effect D-65 flagged, biggest for RF (2.270→
1.990) since RF's own predictions sit closest to what it was fit to reproduce. TabPFN/TabICLv2/TFT/
DLinear/LSTM show the opposite or a much smaller gap, since their targets are less/not directly
`y_gapfilled` — TabPFN/TabICLv2's context is always real `y_observed`, so scoring them against
`GapFilled` instead is scoring them against a series they were never fit to predict, and `gf` MASE
is correspondingly *worse*, not better, for those two specifically.)

**Per-tower breakdown is Observed-target only** — the per-tower CSV
(`results/s03_table_by_tower_climatology_tabicl.csv`) does not carry a GapFilled block, only the
all-tower-pooled table above does.

**Per-tower breakdown (Observed target, MASE):**

| Tower | Model | Model1 | VarA (removal) | VarB (resample) |
|---|---|---|---|---|
| T2 | RF | 1.770 | 1.562 | 1.263 |
| T2 | XGB | 0.629 | 0.598 | 0.587 |
| T2 | LightGBM | 1.368 | 1.427 | 1.319 |
| T2 | SARIMAX | 1.357 | 0.988 | 1.351 |
| T2 | Ensemble_unweighted | 1.247 | 1.103 | 1.098 |
| T2 | Ensemble_MASEweighted | 1.238 | 1.099 | 1.091 |
| T2 | TFT | 0.801 | 0.522 | 0.790 |
| T2 | TabPFN | 0.444 | 0.444 | 0.483 |
| T2 | DLinear | 1.125 | 0.652 | 0.668 |
| T2 | LSTM | 0.952 | 0.572 | 0.750 |
| T2 | TabICLv2 | 0.419 | 0.417 | 0.471 |
| T4 | RF | 2.487 | 2.223 | 1.887 |
| T4 | XGB | 1.335 | 1.255 | 1.135 |
| T4 | LightGBM | 1.597 | 1.330 | 1.272 |
| T4 | SARIMAX | 0.956 | 0.922 | 0.899 |
| T4 | Ensemble_unweighted | 1.515 | 1.369 | 1.235 |
| T4 | Ensemble_MASEweighted | 1.515 | 1.368 | 1.234 |
| T4 | TFT | 0.874 | 0.883 | 0.895 |
| T4 | TabPFN | 0.760 | 0.762 | 0.766 |
| T4 | DLinear | 1.364 | 1.081 | 1.156 |
| T4 | LSTM | 0.947 | 1.016 | 0.943 |
| T4 | TabICLv2 | 0.829 | 0.847 | 0.823 |
| T9 | RF | 1.792 | 1.579 | 1.568 |
| T9 | XGB | 1.243 | 1.333 | 1.273 |
| T9 | LightGBM | 1.563 | 1.380 | 1.337 |
| T9 | SARIMAX | 1.427 | 1.385 | 1.317 |
| T9 | Ensemble_unweighted | 1.259 | 1.263 | 1.225 |
| T9 | Ensemble_MASEweighted | 1.263 | 1.264 | 1.227 |
| T9 | TFT | 0.761 | 0.743 | 0.703 |
| T9 | TabPFN | 0.709 | 0.675 | 0.721 |
| T9 | DLinear | 0.995 | 0.846 | 0.914 |
| T9 | LSTM | 1.003 | 0.907 | 0.858 |
| T9 | TabICLv2 | 0.720 | 0.752 | 0.725 |

Tower 4 (best-covered, most reliable) shows the tree/SARIMAX/ensemble degradation most starkly —
RF alone jumps from 2.487 (Model1) to still 1.887 even after Variant B's resample mitigation, far
above every foundation/DL model at the same tower. Tower 2's RF/LightGBM numbers are noisier
(shorter real-data domain) but the same qualitative pattern holds.

**Isolating the two changes (climatology-vs-persistence, RF-sourced-vs-TabICL-sourced), so the
1.3–2.5x jump above isn't misread as an ensemble/tree degradation.** This table's numbers should
NOT be compared directly against B16's own real headline (`Ensemble_unweighted` = **0.809** MASE,
climatology-scored, `BASE+ALL`/F-10's v3 feature set, `results/b09_b16_climatology_mase_full_table.csv`)
— that number is unaffected by anything in this addendum and remains the production-relevant
result, but it runs on a *different feature set* (v2 here vs. v3/`BASE+ALL` there), so it isn't a
valid same-experiment isolation point for what changed in *this* table. Derived instead, purely
arithmetically (no new model calls — climatology depends only on real `y_observed`, confirmed
byte-identical between the RF-sourced and TabICL-sourced daily files, so its own MAE can be backed
out of the already-computed TabICL table and reapplied to the RF-sourced file's own saved MAE,
mirroring D-80's own rescoring technique) — holding S03's own v2 feature set fixed on both sides:

| | Persistence-scored | Climatology-scored |
|---|---|---|
| **RF-sourced** | 0.936 (VarA) / 0.900 (VarB) | **0.736 (VarA) / 0.717 (VarB)** |
| **TabICL-sourced** | *(not computed — out of scope)* | 1.346 (VarA) / 1.250 (VarB) |

(Ensemble_unweighted, all-tower pooled; same pattern holds for RF/XGB/LightGBM/SARIMAX/
Ensemble_MASEweighted — see `results/s03_rf_sourced_climatology_isolation.csv`.) Switching metric
convention alone (top-left → top-right, RF-sourced held fixed) genuinely *improves* MASE, as
D-80's own finding predicts. RF-sourced+climatology (0.717–0.736) lands in the same healthy
ballpark as B16's own 0.809 — different feature set, same order of magnitude, both good — **the
ensemble mechanism itself is fine on S03's feature set too, under normal (RF-sourced) conditions.**
The real effect is entirely the TabICL-sourced swap (climatology held fixed, right column):
MASE nearly doubles (0.717–0.736 → 1.250–1.346). The 1.3–2.5x figures reported above for the full
Model1/VarA/VarB table are this same TabICL-sourced-data effect, confirmed isolated and not an
artifact of the metric-convention switch or a contradiction of any standing production number.

**Mechanism, confirmed directly, not inferred**: TabICL's gap-filled CH4 series sits at a
substantially different mean level than RF's at every tower:

| Tower | y_observed (real) | y_gapfilled (RF-sourced) | y_gapfilled (TabICL-sourced) |
|---|---|---|---|
| T2 | 30.8 | 19.4 | 54.2 |
| T4 | 29.5 | 30.0 | 33.7 |
| T9 | 36.1 | 36.7 | 72.2 |

Training a regression target directly on a differently-calibrated series propagates that
miscalibration into every prediction — a much larger effect than merely using it as historical
*context* for a zero-shot foundation model. TabPFN/TabICLv2 are far less affected because their
context is always real `y_observed`, never `y_gapfilled` (`y_gapfilled` only touches their `fx_`
driver columns and the secondary gap-filled-target metric here); TFT/DLinear/LSTM are entirely
unaffected since no TabICL hourly data exists to swap to.

### Practical implication

Independently confirms and extends D-80's own conclusion ("the standing forecasting champion is
unchanged — D-79's better gap-filling is a real, useful result for gap-filling itself, but does
not transfer to forecasting"). It now also does not transfer to the tree/SARIMAX/ensemble family,
and the failure mode there is considerably more severe than the foundation-model-context case D-80
already flagged. **RF-sourced gap-filling remains the right choice for every model family this
project forecasts with** — no change to the standing recommendation, but a sharper, independently
derived reason why for the tree/SARIMAX/ensemble family specifically.

### Files added/modified this addendum

- `notebooks/07_scenario_analysis/s03_driver_availability_ablation.py` — added `climatology_baseline()`
  and a `daily_csv` parameter; MASE/RMSSE now score against climatology, not persistence.
- `notebooks/07_scenario_analysis/s03_model_roster_extension.py` — same climatology swap for
  TabPFN/TabICLv2; `run_foundation_models()` gained `daily_csv`/`degraded_cols` parameters.
- `notebooks/07_scenario_analysis/s03_climatology_tabicl_update.py` — new orchestrator script (6
  steps: tree/SARIMAX/ensemble + TabPFN/TabICLv2 reruns on TabICL data for both Variant A/B and a
  Model-1-equivalent, DL chain rescoring, final table assembly).
- `S03_driver_availability_ablation.ipynb` — new cells documenting this update, loading the
  combined tables, a direct y_gapfilled mean-level sanity check, and an updated verdict.
  Re-executed clean end to end via `nbconvert`.
- `results/s03_foundation_tabicl_summary*.csv`, `s03_foundation_model1_tabicl_summary*.csv`,
  `s03_dl_climatology_summary*.csv`, `s03_summary_tabicl.csv`, `s03_summary_model1_tabicl.csv`
  (+ `_vs_gapfilled` siblings) — new raw per-bin outputs.
- `results/s03_table_all_towers_climatology_tabicl.csv`, `s03_table_by_tower_climatology_tabicl.csv`
  — new combined 11-model headline tables (kept alongside, not overwriting, the original
  persistence-scored/RF-sourced tables — this is a genuine methodology version, not a bug fix).
- No existing file overwritten: `results/s03_table_all_towers.csv` and siblings (the original,
  persistence-scored/RF-sourced tables) are untouched.
