# F-12 Results: bidirectional (lead) soil lags for pooled RFm — tested, not adopted

**Notebook:** `F12_bidirectional_soil_lags_RFm.ipynb`
**Scope:** all 3 towers (T2, T4, T9), pooled, all 5 gap-length scenarios (CLAUDE.md full-coverage
default), 3-arm ablation.
**Executed:** 2026-07-20

---

## 1  Why this experiment

RFm's champion config (D-35/D-49) only ever uses **backward-only** soil lag features
(`swc_l{lag}`/`ts_l{lag}` via `.shift(lag)`, `LAG_HOURS=[168,336,504,672]`) — despite the fact
that (a) the upstream met/soil driver gap-filling (`reddyproc_pipeline.py`) already uses a
bidirectional, centered expanding ±7/14/28/60-day window, and (b) F-11's SAITS alternative is
inherently bidirectional (unmasked self-attention over 336h windows). Nobody had tested whether
giving RFm itself forward-looking (future) soil context — legitimately available in a
*gap-filling* task, since sensors either side of a masked FCH4 gap are genuinely observed, unlike
in true forecasting — would improve its R². This closes that gap.

## 2  Methodology

A controlled 3-arm ablation on **only** the swc/ts lag block, everything else held fixed (partial
pooling T2+T4+T9 + tower dummies, EXT sourcing, full-period gap-CV, identical met/fc/AUX/
lsu_dens/graze/mgmt/gpp/reco channels):

- **Arm A (baseline)** — current backward-only lags, rerun fresh in this notebook (not just cited
  from `BEST_RESULTS.md`) for a clean same-run comparison point, given prior reruns of this exact
  protocol have shown small cross-run drift (F-08 vs F-09a, a ~0.01–0.014 gap never reconciled).
- **Arm B (bidir)** — Arm A's features **plus** new forward lags `swc_f{lag}`/`ts_f{lag}` via
  `.shift(-lag)`, same `LAG_HOURS`. Tests: does adding future context help at all?
- **Arm C (leadonly)** — forward lags **replacing** the backward ones, same feature *count* as
  Arm A. Tests: is it *direction* that matters, or just having more temporal context of any kind?

Reused F-08's exact `insert_calendar_gaps`/`dom_mask`/`mets`/`med_metrics` harness verbatim, and
`gapfill_rfm.py`'s `load_ext`/`cfg`/`ts_col_for`/`feat_list`/`frame`/`fit`/`TOWERS`/`LAG_HOURS`/
`DUM` via import (never edited — it's imported live by the production precompute). Arms B/C are
additive clones (`frame_bidir`, `frame_leadonly`) of `gapfill_rfm.frame`, changing only the lag
block.

**Deliberate deviation from F-08/F-09a, stated explicitly:** `N_REPS=2` (not F-08's 5). A first
attempt at the full `N_REPS=5` protocol (225 fits, ~3.45h extrapolated from the smoke test) was
killed by the environment after ~2h20m with **zero progress saved**, since `nbconvert --inplace`
only writes the notebook file once the entire run finishes. Rebuilt with `N_REPS=2` (90 fits, ~83
min) and a cell-by-cell execution driver (`nbclient`) that checkpoints the notebook to disk after
every cell — this second attempt completed cleanly in 5,261s (~87.7 min). Coverage (3 arms × 3
towers × 5 scenarios) is unchanged; only the rep count is reduced, matching F-09a's own precedent
for cutting reps under time pressure while keeping full tower/scenario coverage.

**Data-leakage checks (see notebook for full detail):**
1. **Target leakage** — asserted programmatically that no arm's feature list references the
   target or any FCH4-derived column; `swc_f`/`ts_f` are built strictly from the external
   per-catchment soil moisture/temperature series.
2. **Held-out/training overlap** — a permanent runtime assertion inside `run_rf` (every one of the
   90 fits, not a one-off check) confirms no held-out timestamp remains in that fit's training
   partition before the pooled concat. All 90 fits passed.
3. **Scope caveat (not leakage, documented):** the forward lag features use soil-sensor readings
   from *after* the reconstructed timestamp — legitimate for gap-filling an already-recorded
   historical archive (those readings genuinely exist by the time gap-filling runs, same
   justification as `reddyproc_pipeline.py`'s own bidirectional windows), but **not** legitimate
   in a live forecasting deployment. This is a gap-filling-only finding, not transferable to the
   forecasting pipeline without re-deriving as backward-only.

## 3  Feasibility (Phase 1 smoke test) — GO

Tower 4, scenario `m` (32h), 1 rep, all 3 arms: 163.7s total, R² in a narrow 0.371–0.383 band
(single-rep point, not conclusive on its own), lead-column NaN fraction only 0.96% (the tail
`max(LAG_HOURS)`=672h of each series — symmetric to how the existing backward lags NaN the *head*
672h). ~55s/fit mean.

## 4  Results — full run, 3 arms × 3 towers × 5 scenarios × 2 reps (90 fits)

**Overall median R² by arm:**

| Tower | baseline | bidir | leadonly | Δ(bidir−baseline) | Δ(leadonly−baseline) | Δ(bidir−leadonly) |
|---|---|---|---|---|---|---|
| T2 | 0.574 | 0.555 | 0.557 | −0.019 | −0.017 | −0.002 |
| T4 | 0.402 | 0.410 | 0.410 | +0.008 | +0.008 | 0.000 |
| T9 | 0.418 | 0.408 | 0.407 | −0.010 | −0.011 | +0.001 |

**vs. recorded RFm champion (0.574 / 0.402 / 0.418):**

| Tower | baseline beats champion | bidir beats champion | leadonly beats champion |
|---|---|---|---|
| T2 | — (reproduces exactly) | No | No |
| T4 | — (reproduces exactly) | Yes (+0.008) | Yes (+0.008) |
| T9 | — (reproduces exactly) | No | No |

Arm A reproduces the recorded champion numbers **exactly** at all 3 towers (0.574/0.402/0.418) —
confirms `frame_bidir`/`frame_leadonly`'s reuse of `frame_baseline` introduced no regression to
the baseline path itself.

Full per-scenario table: `results/f12_summary.csv`. 45 rows tagged `F-12` in
`results/benchmarks.csv`.

## 5  Feature-importance diagnostic

Single clean refit per arm on its full pooled training set (`native_importance_tree`,
`src/interpretability/importance.py`), summed over the lead (`swc_f*`/`ts_f*`) vs. lag
(`swc_l*`/`ts_l*`) column groups:

| Arm | lead-column importance mass | lag-column importance mass |
|---|---|---|
| baseline | 0.000 (n/a) | 0.107 |
| bidir | 0.099 | 0.081 |
| leadonly | 0.123 | 0.000 (n/a) |

Top-5 features are identical across all 3 arms (`fc`, `lsu_dens`, `PPFD_1_1_1`, `RN_1_1_1`,
`SWIN_1_1_1`) — the lag/lead block never dominates the model regardless of arm. The lead columns
are **not** ignored by the RF (importance mass is real and non-trivial in both Arm B and Arm C,
ruling out a "fake win via mean-imputation noise" explanation for any positive delta) — if
anything, leads-only (Arm C, 0.123) pick up slightly *more* importance mass than backward-only
(Arm A, 0.107) or the split in Arm B (0.099+0.081=0.180, more total context but each half
individually lower than its single-direction counterpart). So the soil-lag block as a whole
matters to the model in every arm; direction of that block does not.

## 6  Interpretation

The RF genuinely uses whichever lag direction it's given (§5), but this does not translate into a
better held-out gap-filling R² on balance: **one tower (T4) shows a small, tied +0.008 gain from
either bidir or leadonly, while T2 and T9 both regress by a similar or larger magnitude
(−0.010 to −0.019).** A ±0.008–0.019 R² swing is within the range of cross-run noise already
documented for this exact protocol (the F-08/F-09a discrepancy was ~0.01–0.014 at T4/T9 alone),
and reduced reps here (`N_REPS=2` vs. F-08's 5) widen that noise band further. There is no tower
where a lead-containing arm wins by a margin large enough to be confidently attributed to real
signal rather than rep-to-rep variance.

Physically, this is a plausible outcome: soil moisture/temperature vary smoothly and are already
well captured by contemporaneous values plus backward context (autocorrelation in soil state
is strong in both directions, so forward lags carry largely redundant, not novel, information
relative to backward lags at the same offsets) — consistent with leads-only (Arm C) performing
essentially identically to bidir (Arm B), not better despite "seeing" both a different direction
and, in Arm B's case, more total features.

## 7  Recommendation

**Not adopted.** Neither Arm B (bidir) nor Arm C (leadonly) beats the RFm champion (D-35/D-49) on
balance across towers — a marginal, noise-level gain at T4 alongside regressions at T2/T9 is not
a basis for promotion. No change to `BEST_RESULTS.md` §1. Logged as a legitimate,
well-diagnosed null result (mirroring F-09b/F-11's handling elsewhere in this project): the
pipeline runs correctly (Arm A reproduces the champion exactly, confirming no implementation
bug), leakage checks pass on every fit, and the feature-importance diagnostic explains *why* the
result is null (real but redundant signal) rather than leaving it a mystery.

**If revisited:** the ~83-minute, 90-fit run (`N_REPS=2`) leaves room to rerun at F-08's full
`N_REPS=5` now that the checkpointed execution pattern (cell-by-cell `nbclient` driver) avoids the
zero-progress-on-kill risk that blocked the first attempt — worth doing before fully closing this
question, since T4's +0.008 delta is the one result close enough to the noise floor that more
reps could plausibly resolve it either way. Not pursued in this pass — the reduced-rep result was
already unambiguous at T2/T9 (regressions clearly larger than at T4) and the marginal-improvement
case at T4 alone does not justify tripling the compute budget for the champion determination.

**Sources:** `notebooks/04_feature_engineering/F12_bidirectional_soil_lags_RFm.ipynb`,
`results/f12_summary.csv`, 45 rows tagged `F-12` in `results/benchmarks.csv`.
