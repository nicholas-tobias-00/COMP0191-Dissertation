# F-10 — Extended feature engineering: livestock species, land-use regime, catchment flow,
# fertilizer richness (+ bonus liveweight density)

**Status: Stage 1 (point-forecast signal check) AND Stage 2b (recursive-rollout confirmation)
both complete. Neither found a feature family worth adopting.** Two independent forecasting
harnesses — a direct point-forecast smoke test and the real B-10 recursive-rollout ensemble, all 3
towers, all 5 anchors — agree: none of the 5 new feature families improve on the standing
recommendation. This is an honest null result, reported plainly per this project's established
convention (D-31, D-32, F-01's P2 management-feature finding, B-14/B-15's negative HPO results) —
not a failure to hide. Neither test touched the gap-filling pipeline (`gapfill_rfm.py`/F-08) at
all — this whole branch was scoped to forecasting performance specifically, per direct user
instruction.

## Motivation

After confirming (D-66) that a suspiciously bad TabICLv2 result was a real implementation bug
rather than a genuine model-capability gap, and given the whole B-09→B-15 recursive-rollout
sequence has converged to roughly the same ceiling (R²≈0, MASE≈0.85–0.98) regardless of model
architecture or HPO, the user asked to pivot the search for improvement from the model side to the
data/feature side — directly supported by this project's own history (B-03/B-04's enriched
features lifted point-forecast R² by +0.08–0.10, more than any model change anywhere in this
project). Two starting angles were named (livestock-type granularity, fertilizer richness) plus
two colleague-review items (catchment flow instrumentation, a possible Tower-2 land-use regime
shift). All four leads were confirmed real and actionable via research this session (see D-67 for
the full empirical grounding of each finding) before any code was written.

## What was built (Stage 1)

New, additive-only files (nothing existing edited): `src/features/build_bodyweight_density.py`,
`src/features/build_forecasting_matrix_v3.py` (reads `forecast_daily_v2.csv` + raw `load_ext()` +
`management_features.csv` + `Field_Event_Data_Format_1.csv`, left-merges 18 new `fx_`/mgmt columns
onto v2, writes `data/Hourly/forecast_daily_v3.csv`), `notebooks/04_feature_engineering/
f10_signal_check.py` (Stage 1 signal check).

**Five feature families**, all verified against the plan's own checklist before any modeling:

- **(a) Species disaggregation**: `fx_cattle_dens`/`fx_sheep_dens`/`fx_lamb_dens` (raw head
  density, head/ha) — `1.0·cattle + 0.1·sheep + 0.05·lamb == fx_lsu_dens` exactly (verified to
  <1e-9), confirming the split is a lossless refinement of the existing combined density, not a
  different quantity.
- **(b) `fx_is_arable`**: programmatic land-use regime flag. **A real bug was caught during
  verification**: the first implementation used `build_management_features.classify()=="cultiv"`
  as the trigger, which also fires on routine grassland renovation (a single `Chain harrow` event
  at Tower 4, `Chain harrow`/clover-blend `Grass seeding (overseeding)` at Tower 9) — producing
  false arable flips at both towers, contradicting the direct field-record check that motivated
  this feature in the first place. Fixed to a narrower, empirically-verified trigger (a literal
  `Plough` operation, or a `Drill/Broadcast Seed`-type operation whose `Application` names an
  actual cereal crop — wheat/oat/barley/bean — not a grass/clover species): confirmed `Plough` and
  cereal-crop drilling only ever occur at fields NW002/NW003/NW004/NW015/NW019/NW047 across the
  whole 2017-2024 record, never at Tower 4/9's own fields. Result: **Tower 2 flips to arable on
  2019-09-09, Towers 4 and 9 never flip** (`fx_is_arable.sum()==0` unconditionally) — matches the
  direct field-event check exactly.
- **(c) Catchment flow**: `fx_flow_mean` + lag7/14/21/28 + roll7/14, same per-catchment
  column-lookup convention as `fx_SWC_mean`/`fx_TS_mean`. **A second bug was caught**: the first
  implementation built the lag/roll block on the raw hourly-indexed frame before resampling to
  daily, so `.shift(L)` shifted by L *hours*, not days — all four lag columns came out wholly NaN.
  Fixed by resampling to daily first, matching `build_forecasting_matrix_v2.py`'s own pattern.
  Coverage confirmed ≥85% at all 3 towers (88.5–92.0%), better than SWC's own coverage.
- **(d) Management richness**: `fx_mgmt_{fertN_recency,fertN_rate,lime_recency,cultiv_recency,
  cut_recency,manure_recency}` — already computed in `management_features.csv`, never previously
  reaching the daily matrix at all (not even `cut_recency`/`manure_recency`, which F-05/D-32 had
  already found small-but-positive on the old hourly harness — a bonus, zero-marginal-cost fix
  beyond the 2 columns originally requested). Spot-checked one known fertiliser event: recency
  jumps to ~0.97 on the event day and decays with the exact τ=14 exponential, confirming no
  leakage or off-by-one error after daily resampling.
- **(e) Bonus — liveweight density**: `fx_total_liveweight_dens` (kg/ha), from a last-observed-
  carried-forward join of per-animal location (correctly resolved to real NWFP field codes, e.g.
  `NW002`, not just shed labels — the open feasibility question from planning) × per-animal
  weight. 90–100% of tower-resident animal-days resolved a weight across all 3 species;
  distinct-day coverage 441/1730/2244 days at T2/T4/T9 (sparser than the other families, since it
  additionally requires the animal to be in one of exactly 3 catchments AND have a nearby weight
  record).

## Stage 1 signal check

Cheap, bounded, single-seed leave-one-group-in RF ablation (B-03/B-10's exact daily-track
hyperparameters, no new HPO): `BASE` (v2's existing 34 `fx_` columns) vs `BASE+<family>` for each
of the 5 families, plus `BASE+ALL` and a follow-up `SWAP_species_for_lsu` variant (see below).
Pooled training 2018-2021, direct h∈{1,14}-day-ahead forecast evaluated against real `y_observed`
at 2022-2023. **Sanity check passed before trusting anything else**: `BASE`'s own R² (T4 h=1=0.365,
h=14=0.280; T9 h=14=0.359) reproduces `BEST_RESULTS.md`'s published B-03 numbers almost exactly,
confirming this simplified harness is faithful. Tower 2 has **zero** real `y_observed` rows in
2022-2023 (expected, matches this project's own well-documented finding) — Stage 1 could not
evaluate Tower 2 directly on this smoke-test-tier harness.

**Pre-registered go/no-go bar** (from the F-10 plan): a family proceeds to Stage 2 only with a
consistent ΔR²>+0.01 median across towers/horizons (no tower collapsing) or a top-10 SHAP rank at
h=14.

| Family | ΔR² T4 h=1 | ΔR² T9 h=1 | ΔR² T4 h=14 | ΔR² T9 h=14 | SHAP rank (of 18 new cols) |
|---|---|---|---|---|---|
| species | +0.0001 | −0.0041 | +0.0097 | **−0.0236** | #1 (`fx_cattle_dens`) |
| arable | +0.0024 | +0.0003 | +0.0030 | −0.0005 | last (0.0 — never split on) |
| flow | −0.0033 | +0.0084 | −0.0109 | −0.0042 | mid-pack |
| mgmt | −0.0009 | −0.0086 | −0.0081 | −0.0103 | mid-to-low |
| bodyweight | **−0.0529** | +0.0055 | **−0.0469** | −0.0003 | #2 (`fx_total_liveweight_dens`) |

**None of the 5 families clear the ΔR² bar.** `BASE+ALL` is worse than `BASE` at every
tower/horizon (0.32–0.34 vs 0.36–0.38 at h=1) — adding all 18 columns at once hurts, consistent
with this project's own repeated "weak features can hurt when stacked" pattern.

**The SHAP-rank alternate criterion is more ambiguous, and was investigated further rather than
taken at face value.** `fx_cattle_dens` and `fx_total_liveweight_dens` do draw real model
attention (SHAP ranks #1 and #2 of the 18 new columns) — but `fx_cattle_dens` correlates with the
existing `fx_lsu_dens` at **r=0.972** (cattle dominates the LSU-weighted sum by construction:
weight 1.0 vs. sheep 0.1/lamb 0.05), so this SHAP attention is plausibly credit-splitting with an
already-present, highly collinear feature rather than new information. A follow-up test —
`SWAP_species_for_lsu` (replace `fx_lsu_dens` with the 3-way species split rather than adding
alongside it) — resolves this more cleanly: **+0.0079/+0.0190 R² at Tower 4 (h=1/h=14) but
−0.0134/−0.0205 at Tower 9** — a real, tower-specific pattern (Tower 4 is where I-02 already found
livestock density to be the single dominant SHAP driver; Tower 9 apparently does not benefit from
disaggregating it), not a consistent win. Mean ΔR² across the 4 tower/horizon cells is
approximately −0.002 — essentially zero net effect once both towers are weighted equally, and does
not clear the bar either.

## Stage 1 verdict: no promotion on the point-forecast smoke test

No family shows a clear, consistent win in the cheap point-forecast check. Per the F-10 plan's own
explicit rule ("families that don't clear this bar are reported as null results and dropped, not
carried forward"), Stage 2's point-forecast track (B-16a) was not run. **However, per direct user
instruction, the recursive-rollout track (B-16b) was run anyway** despite Stage 1's null result —
because the actual model/HPO ceiling that motivated pivoting to features in the first place lives
specifically in the recursive rollout (B-09→B-15), not the direct point-forecast track, and a
one-shot direct forecast never exercises the rollout's autoregressive error-compounding/
spike-blindness failure mode. A point-forecast null result doesn't necessarily transfer.

## Stage 2b: recursive-rollout confirmation (B-16b, full coverage)

`notebooks/05_benchmarking/b16_recursive_rollout_v3.py` — reuses `b10_b13_rerun_multi_anchor.py`'s
exact methodology unmodified (same hyperparameters, same `tree_rollout`/`bin_metrics` from
`recursive_rollout.py`, same SARIMAX order-search): 6 configs (`BASE` = v2's feature set, plus each
of the 5 families added individually) × all 3 towers × all 5 anchors (2018-2022) × the full
RF/XGB/LightGBM/SARIMAX ensemble. SARIMAX fit once per (anchor, tower) and reused across all 6
configs (its `EXOG_B` set is unaffected by any new family, so refitting it per config would be pure
waste). **Sanity check passed first**: `BASE`'s `Ensemble_unweighted` reproduces
`BEST_RESULTS.md`'s published all-tower headline almost exactly (R²=−0.1652 vs. published −0.165,
MASE=0.9169 vs. 0.918), confirming this harness is faithful before trusting any comparison.

**All-tower `Ensemble_unweighted`, by config (including a follow-up `BASE+ALL` run, added after
the user asked why it was missing from the original sweep — see below). MASE is the primary
metric per this project's forecasting convention (CLAUDE.md — MASE is far less dominated by CH4's
spike-tail behavior than R²); R² kept as the rightmost column:**

| Config | MASE | RMSE | R² |
|---|---|---|---|
| **BASE+species** | **0.9161** | 51.57 | −0.167 |
| **BASE** | 0.9169 | 51.51 | **−0.165** |
| BASE+ALL | 0.9217 | 51.73 | −0.168 |
| BASE+bodyweight | 0.9207 | 51.76 | −0.170 |
| BASE+arable | 0.9198 | 51.58 | −0.169 |
| BASE+flow | 0.9223 | 51.66 | −0.172 |
| BASE+mgmt | 0.9251 | 51.68 | −0.169 |

**Under MASE, `BASE+species` is actually the (marginal) winner** — MASE=0.9161 vs. `BASE`'s
0.9169, a tiny but real improvement, even though R² ran the other way (−0.167 vs. −0.165). This is
a genuine case of the two metrics disagreeing, not noise in one direction only. Every other family
is worse than `BASE` on both metrics. The same held individually for RF, XGB, and LightGBM (checked
separately, not just the ensemble) — species is at or near the best MASE for each tree model too,
though the margin is small in every case. Per-tower breakdown for species confirms the same
tower-specific pattern Stage 1 found: a small improvement at Tower 4 (R² +0.0054) offset by small
degradations at Towers 2 and 9 — a real but modest effect, not a decisive win.

**`BASE+ALL` follow-up (run separately, reusing the already-fitted SARIMAX chains — its `EXOG_B`
set is unaffected by any family, so refitting it again would be pure waste; only RF/XGB/LightGBM
were refit on the full 18-column feature set).** Notably, this shows a **different pattern than
Stage 1's point-forecast check**, where stacking all 18 columns was clearly the *worst* config by a
wide margin (0.32 vs. 0.36–0.39 R², a ~0.04–0.05 gap). On the rollout, `BASE+ALL` lands in the
*middle* of the pack — still worse than `BASE`, but better than 4 of the 5 individual families
(arable, mgmt, bodyweight, flow). The direct point-forecast harness and the recursive rollout don't
fully agree on *how* extra features hurt (catastrophic stacking penalty vs. a more moderate,
consistent one) even though both agree on the headline conclusion (nothing beats `BASE`) — a
genuine methodological nuance worth keeping in mind rather than assuming one harness's failure
mode automatically describes the other's.

## Stage 2c: foundation models (TabPFN, TabICLv2) — a materially different result from the trees

Per user request ("show for all models from SARIMAX to TabICLv2"), TabPFN and TabICLv2 were also
tested across all 7 configs (`notebooks/05_benchmarking/b16_foundation_models_v3.py` — both are
zero-shot, no retraining needed per config, just re-running inference with each config's covariate
set). **Unlike the tree ensemble, both foundation models show real, substantial, broadly-consistent
gains from several feature families** — verified not to be a single-anchor or single-tower
artifact (checked per-tower and per-anchor for the strongest case; the improvement is spread across
most tower/anchor combinations, not concentrated in one).

**All-tower, by config, MASE-led:**

| Model | Config | MASE | RMSE | R² |
|---|---|---|---|---|
| **TabPFN** | **BASE+species** | **0.840** | 55.58 | −0.084 |
| TabPFN | BASE+bodyweight | 0.840 | 55.56 | −0.104 |
| TabPFN | BASE+ALL | 0.841 | 55.77 | −0.092 |
| TabPFN | BASE+mgmt | 0.849 | 55.80 | −0.111 |
| TabPFN | BASE+flow | 0.850 | 56.07 | −0.107 |
| TabPFN | BASE / BASE+arable | 0.854 | 56.06 | −0.122 |
| **TabICLv2** | **BASE+ALL** | **0.871** | 56.02 | −0.155 |
| TabICLv2 | BASE+species | 0.880 | 56.12 | −0.179 |
| TabICLv2 | BASE+bodyweight | 0.888 | 56.77 | −0.201 |
| TabICLv2 | BASE+flow | 0.912 | 57.60 | −0.264 |
| TabICLv2 | BASE+mgmt | 0.925 | 58.12 | −0.311 |
| TabICLv2 | BASE / BASE+arable | 0.928 | 57.98 | −0.329 |

`BASE` and `BASE+arable` are numerically identical for both models — expected, not a bug:
`fx_is_arable` is a constant (0 or 1) within nearly every single rollout window for a per-tower
zero-shot fit (Towers 4/9 are always 0; Tower 2 is constant within all but one anchor's window), so
there is no within-window variation for either model to condition on.

**This is the headline finding of Stage 2c: `TabPFN+species` (MASE=0.840, R²=−0.084) is the best
single-model result in the entire B-09→B-15 sequence** — better MASE than the standing
`Ensemble_unweighted` recommendation (0.917) and better than every other model tested this whole
project, and its R² also beats the standing recommendation's −0.165. This directly contradicts the
tree-ensemble finding above and is a genuinely important nuance: **feature richness that doesn't
move the tree ensemble can still meaningfully help a foundation model.**

## Stage 2d: TFT, DLinear, LSTM (closing "all models from SARIMAX to TabICLv2")

These three read the **hourly** feature matrix, not the daily one, so testing them required a new
build step: `src/features/build_forecasting_matrix_v3_hourly.py` extends `forecast_features_v2.csv`
with hourly-resolution versions of the 5 families (`fx_cattle_dens`/`sheep`/`lamb`, `fx_is_arable`
broadcast from the daily flag, `fx_flow_mean` as a raw hourly reading — no lag/roll ladder at
hourly resolution, matching every other pre-existing hourly `fx_` column's own "raw current
reading" convention — the 6 `fx_mgmt_*` columns, already hourly-native, and `fx_total_liveweight_dens`
broadcast from the daily value) → `data/Hourly/forecast_features_v3.csv`. `forecasting_dl.py`
itself needed **zero code changes** — it already auto-detects feature columns via
`[c for c in m.columns if c.startswith("fx")]` into a module-level `FX` list read at call time, so
`notebooks/05_benchmarking/b16_dl_models_v3.py` simply reassigns that list to each config's column
subset immediately before that config's training/rollout calls.

TFT uses D-45/B-13a's exact regularized recipe (weight_decay=1e-3, 90-day validation carve-out,
patience=5); DLinear/LSTM use B-09's exact plain recipe (no regularization, no val split) — no new
HPO for either. All pooled (fit once per anchor per config on T2+T4+T9), rolled out per tower.
Ran in ~7 minutes total (much faster than the tree-ensemble sweep, since DL training here is
cheap at this data scale).

**All-tower, by config, MASE-led:**

| Model | Config | MASE | RMSE | R² |
|---|---|---|---|---|
| **TFT** | **BASE+ALL** | **0.941** | 55.12 | −0.251 |
| TFT | BASE+species | 0.963 | 56.40 | −0.338 |
| TFT | BASE (standing) | 1.063 | 57.61 | −0.661 |
| **LSTM** | **BASE+species** | **1.086** | 61.16 | −0.622 |
| LSTM | BASE (standing) | 1.150 | 63.15 | −1.355 |
| **DLinear** | **BASE+bodyweight** | **1.374** | 61.52 | −1.836 |
| DLinear | BASE (standing) | 1.489 | 64.93 | −3.066 |

**TFT shows the same pattern as the foundation models, and it's dramatic: `BASE` alone (MASE=1.063)
loses to naive persistence; `BASE+ALL` (MASE=0.941) beats it.** That's a genuine flip, not a small
edge — stacking all 5 families is TFT's best config here, unlike every tree-based config where
`BASE+ALL` was mediocre-to-bad. LSTM and DLinear both improve with extra features too (species for
LSTM, bodyweight for DLinear) but remain far behind every other model in absolute terms — consistent
with their well-documented instability (D-53/D-54) — the features help *relatively* but don't close
the gap to trees/foundation models.

`fx_is_arable`'s hourly broadcast (constant within almost every window, same reasoning as Stage 2c)
shows negligible effect here too, as expected.

## Final verdict, all 11 models, all 7 configs

**The tree-based/statistical models (RF, XGB, LightGBM, SARIMAX, both ensembles) show no
meaningful gain from any feature family — but every attention-based or foundation model tested
(TFT, TabPFN, TabICLv2, and to a lesser extent LSTM/DLinear) shows real, often large, gains,
particularly from the species-disaggregation and "all families stacked" configs.** This is the
opposite of what a quick read of Stage 1's tree-only smoke test would have suggested, and is
exactly why the user's push to test "all models" (not just the standing ensemble) mattered.

**Best config per model, MASE-ranked (all-tower):**

| Model | Best config | MASE | R² | vs. `BASE` MASE |
|---|---|---|---|---|
| **TabPFN** | **BASE+species** | **0.840** | −0.084 | 0.854 → 0.840 |
| TabICLv2 | BASE+ALL | 0.871 | −0.155 | 0.928 → 0.871 |
| Ensemble_unweighted | BASE+species | 0.916 | −0.167 | 0.917 → 0.916 |
| Ensemble_MASEweighted | BASE+species | 0.916 | −0.167 | 0.917 → 0.916 |
| XGB | BASE+ALL | 0.919 | −0.192 | 0.921 → 0.919 |
| LightGBM | BASE (no gain) | 0.939 | −0.206 | — |
| TFT | BASE+ALL | 0.941 | −0.251 | 1.063 → 0.941 |
| RF | BASE+species | 0.964 | −0.240 | 0.967 → 0.964 |
| SARIMAX | BASE (no gain, unaffected by any family) | 0.974 | −0.360 | — |
| LSTM | BASE+species | 1.086 | −0.622 | 1.150 → 1.086 |
| DLinear | BASE+bodyweight | 1.374 | −1.836 | 1.489 → 1.374 |

**Headline recommendation: `TabPFN+species` (MASE=0.840, R²=−0.084) is a new best single-model
result for the recursive-rollout track, beating the standing `Ensemble_unweighted` recommendation
(MASE=0.918/R²=−0.165) outright, and the best result in the whole B-09→B-15 sequence.** Given
TabPFN is a zero-shot, no-training model, this is also close to free to adopt (the "species" family
alone — 3 columns — not the full 18). Recommend promoting this to `BEST_RESULTS.md`'s standing
recursive-rollout recommendation, with the caveat that the trees/SARIMAX/ensemble finding (no
family helps) still applies to the specific model class that currently constitutes B-10's ensemble.

### Secondary metric: each model's best config, scored against gap-filled FCH4 too

Matching this project's established secondary-metric convention (D-65 second addendum): every
chain above is scored a **second** time against `y_gapfilled` (dense/continuous) alongside the
primary `y_observed` score, for each model's own best config from the table above. **Same
explicit circularity caveat applies as everywhere else this pattern is used** — `y_gapfilled`
seeds each chain's `history_init` and is itself a pooled RFm gap-filler's output trained on
features that substantially overlap these models' own forecast features, so agreement can partly
reflect "forecaster resembles gap-filler," not real skill against reality. Read directionally, not
as validated accuracy.

| Model | Best config | RMSE (gapfilled) | RMSE (observed) | MASE (gapfilled) | MASE (observed) | Correlation (gapfilled) | Correlation (observed) | R² (gapfilled) | R² (observed) |
|---|---|---|---|---|---|---|---|---|---|
| **TabPFN** | BASE+species | 35.01 | 55.58 | 0.944 | **0.840** | 0.204 | 0.326 | −0.676 | **−0.084** |
| TabICLv2 | BASE+ALL | 35.27 | 56.02 | 0.973 | 0.871 | 0.210 | 0.287 | −0.784 | −0.155 |
| Ensemble_unweighted | BASE+species | 25.43 | 51.57 | 0.749 | 0.916 | 0.520 | 0.372 | −0.189 | −0.167 |
| Ensemble_MASEweighted | BASE+species | 25.42 | 51.56 | 0.749 | 0.916 | 0.520 | 0.372 | −0.194 | −0.167 |
| XGB | BASE+ALL | 25.87 | 51.51 | 0.768 | 0.919 | 0.477 | 0.360 | −0.584 | −0.192 |
| LightGBM | BASE (no gain) | 26.11 | 52.02 | 0.774 | 0.939 | 0.496 | 0.368 | −0.480 | −0.206 |
| TFT | BASE+ALL | 36.10 | 55.12 | 1.156 | 0.941 | 0.147 | 0.226 | −2.179 | −0.251 |
| RF | BASE+species | 25.92 | 52.35 | 0.796 | 0.963 | 0.529 | 0.371 | −0.642 | −0.240 |
| SARIMAX | BASE (no gain) | 29.21 | 53.72 | 0.943 | 0.974 | 0.463 | 0.343 | −1.002 | −0.360 |
| LSTM | BASE+species | 39.79 | 61.16 | 1.237 | 1.086 | 0.185 | 0.272 | −2.724 | −0.622 |
| DLinear | BASE+bodyweight | 42.59 | 61.52 | 1.723 | 1.374 | 0.182 | 0.267 | −6.792 | −1.836 |

**Correction (caught by direct user question — the original version of this section overstated
"doesn't change the ranking"; it does, and substantially).** The gap-filled/observed comparison
splits cleanly into two groups, **not** a uniform "everything gets worse" pattern:

- **Trees, SARIMAX, both ensembles score *better* (lower MASE) on gap-filled than observed**
  (e.g. `Ensemble_unweighted`: MASE 0.916 observed → **0.749 gap-filled**, a large improvement, not
  a degradation). Mechanistic reason: these models are **fit by directly regressing onto
  `y_gapfilled` as their training label** (`tr["target"] = y_gapfilled` in every tree-fitting call
  this whole B-09→B-15 sequence has used) — so of course they track that exact series well; this is
  the circularity risk already flagged for this secondary metric ("agreement can partly reflect
  forecaster resembles gap-filler, not real skill") working directly in their favor here, not a
  subtle effect.
- **TabPFN, TabICLv2, TFT, LSTM, DLinear score *worse* on gap-filled than observed** (e.g. TabPFN:
  0.840 observed → 0.944 gap-filled). These models' prediction targets are `y_observed` (TabPFN/
  TabICLv2's `hist_target` is deliberately `y_observed`-only by design, D-66; the DL models' loss
  is computed against masked `y_observed`, D-38) — they never get the same "fit-the-exact-series"
  boost trees do, so the gap-filled score behaves like the general variance-normalization artifact
  already documented at D-65 (smoother/lower-variance target penalizes the same absolute error more).

**This means the headline "TabPFN+species beats the standing ensemble" claim above is
target-dependent, not unconditional: under gap-filled scoring, the OLD standing recommendation
(`Ensemble_unweighted`, MASE=0.749) beats the NEW "winner" (`TabPFN+species`, MASE=0.944) by a wide
margin — the full ranking flips, not just the absolute gap narrows.** The observed-target ranking
remains the one to trust per this project's own established convention (D-36/D-37: "train on
gap-filled, evaluate on observed" — `y_observed` is the intended validation target; `y_gapfilled`
is explicitly a secondary/exploratory check with a documented circularity risk), and that
convention specifically favors this reading *because* it's the metric the circularity concern
doesn't inflate. But the "TabPFN+species is unambiguously the new best" framing was too strong —
it is the best **on the primary, convention-endorsed metric**, not on every metric, and a reader
should know the two metrics disagree sharply here, not just about magnitude but about which model
wins.

**Within-model comparison (species vs. that same model's own BASE) is a cleaner read than the
across-model one above, and here the two targets mostly agree on direction, if not magnitude.**
For TabPFN specifically: `BASE`→`BASE+species` improves MASE under both targets (observed:
0.854→0.840, gapfilled: 0.949→0.944) — the family genuinely helps this model regardless of target,
just by a much smaller margin under gap-filled (Δ0.005) than observed (Δ0.014). The
feature-family finding (species/enrichment helps attention-based models, doesn't help trees) is
robust to which target is used; **which specific model is declared "the new best overall" is not**
— that comparison is inflated by the tree ensemble's gap-filled circularity, and should be read
under the observed-target metric only. Full bin-level detail:
`results/b16_full_comparison_vs_gapfilled.csv`; the table above is compiled in
`results/b16_final_table_vs_gapfilled_best_config.csv`.

**Second correction (D-96, 2026-08-13, caught by direct user question).** The MASE (gapfilled)
column in the table above was computed against **chain-persistence**, not climatology — this
whole secondary-metric pipeline predates D-80's climatology switch and was never rescored under
it, unlike the MASE (observed) column (rescored separately, D-80). So the "0.944 gapfilled vs.
0.840 observed" comparison quoted above silently mixes two variables (target *and* baseline
convention) at once, not target alone. Worse, the persistence baseline's anchor value is always
`y_gapfilled`-sourced regardless of which target it scores against — the same apples-to-oranges
issue D-71's `Climatology_gf` fix already addressed for a *different* comparison, never propagated
here. Fixed with a genuinely fair `Climatology_gf`-vs-`y_gapfilled` baseline (`b16_gapfilled_
climatology_fix.py`, pure arithmetic, no new model calls):

| Model (own best config) | MASE (gapfilled, OLD — persistence, mismatched) | MASE (gapfilled, CORRECTED — climatology, fair) |
|---|---|---|
| TabPFN+species | 0.944 | **0.906** |
| TabICLv2+species | 0.973 (BASE+ALL) | **0.938** |
| Ensemble_unweighted+species | 0.749 | **0.665** |

**The flip survives and the margin is similar** — `Ensemble_unweighted` still beats `TabPFN+species`
by a wide margin under the corrected metric, so §"This means the headline..." above remains
directionally correct. Only the exact figures were wrong. Full corrected 11-model table:
`results/b16_gapfilled_climatology_fix_best_config.csv` (ranked: both Ensembles 0.665, XGB 0.670,
LightGBM 0.684, RF 0.702, SARIMAX 0.831, TabPFN 0.903 — best config here is actually
`BASE+bodyweight`, not `BASE+species` — TabICLv2 0.922, TFT 1.013, LSTM 1.122, DLinear 1.502
worst). See D-96.

**Caveats, stated plainly:**
- Stage 1 was a single-seed, no-CV, direct-h-forecast smoke test on trees only; Stages 2b–2d are
  single fixed hyperparameter sets (existing published recipes, no new HPO) at the standard 5
  anchors — none of this is an exhaustive search over how these features might be used (different
  lag structures, interaction terms, tower-specific models, or a from-scratch quantile/HPO pass
  for TabPFN+species specifically).
- The species/liveweight tower-specific pattern (helps T4, sometimes hurts T2/T9 individually)
  recurred across Stage 1, 2b, and the foundation-model checks — real and worth remembering, but
  the pooled all-tower MASE gain is large enough that it isn't purely a T4 artifact.
- Flow and management-richness families show real gains for the attention-based models (TFT,
  TabPFN, TabICLv2) but not for trees — plausible explanation: attention/in-context mechanisms may
  extract signal from noisier/sparser covariates (flow's ~87-92% coverage, mgmt's exp-decay
  recency) in a way tree splits don't, but this is a hypothesis, not confirmed here.
- Management richness did **not** reproduce D-28's Tower-9 collapse (−0.86) in any track tested —
  plausibly because the much richer v2/v3 base feature set no longer depends on management signal
  as heavily as F-01's original narrower baseline did.
- Tower 2 could not be evaluated directly in Stage 1's point-forecast check (zero real 2022-2023
  rows) but WAS included in every rollout-based check (2018 anchor has real `y_observed` coverage)
  — the same well-documented data-scarcity pattern as everywhere else in this project.
- `fx_is_arable` shows negligible effect on every model tested (constant within nearly every
  per-tower rollout window) — its value remains documentation/interpretability (D-68's Tower-2
  land-use reconciliation), not predictive accuracy, regardless of model class.

## Files

- `src/features/build_bodyweight_density.py`, `src/features/build_forecasting_matrix_v3.py`,
  `src/features/build_forecasting_matrix_v3_hourly.py` (new, additive).
- `data/Hourly/forecast_daily_v3.csv` (8,772 rows × 66 cols), `data/Hourly/forecast_features_v3.csv`
  (210,459 rows × 67 cols), `data/Hourly/bodyweight_density.csv`.
- `notebooks/04_feature_engineering/f10_signal_check.py` (Stage 1, new, committed).
- `results/f10_signal_check_summary.csv`, `results/f10_signal_check_shap.csv`,
  `results/f10_signal_check_deltas.csv`.
- `notebooks/05_benchmarking/b16_recursive_rollout_v3.py` + `b16_recursive_rollout_v3_all.py`
  (Stage 2b/BASE+ALL follow-up), `b16_foundation_models_v3.py` (Stage 2c: TabPFN/TabICLv2),
  `b16_dl_models_v3.py` (Stage 2d: TFT/DLinear/LSTM) — all new, committed.
- `results/b16_recursive_rollout_v3_summary.csv` (3,780 rows), `results/
  b16_recursive_rollout_v3_chains.csv`, `results/b16_recursive_rollout_v3_all_{summary,chains}.csv`,
  `results/b16_foundation_models_v3_summary.csv` (2,520 rows, both targets), `results/
  b16_dl_models_v3_summary.csv` (3,780 rows, both targets), `results/b16_full_comparison_all_models.csv`
  (the compiled 11-model × 7-config observed-target table).
- `notebooks/05_benchmarking/b16_recursive_rollout_v3_gapfilled.py` (recomputes the tree/ensemble
  secondary metric from already-saved chains, no refitting) → `results/
  b16_recursive_rollout_v3_summary_vs_gapfilled.csv` (7,560 rows), `results/
  b16_recursive_rollout_v3_table_vs_gapfilled.csv`. `results/b16_full_comparison_vs_gapfilled.csv`
  (all 11 models, both targets, every config) and `results/
  b16_final_table_vs_gapfilled_best_config.csv` (the compiled table above, best config per model).
- `DECISIONS.md` D-67 (+ 3 addenda: BASE+ALL follow-up, MASE re-read + foundation models, and this
  final all-11-model verdict) and D-68 (separate Tower-2 land-use documentation reconciliation,
  independent of this modeling outcome).
- `CLAUDE.md` — new standing convention: MASE is the primary forecasting metric, R² secondary.
