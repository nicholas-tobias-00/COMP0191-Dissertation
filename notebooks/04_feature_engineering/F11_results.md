# F-11 Results: SAITS gap-filling for FCH4 — tested, not adopted

**Notebook:** `F11_SAITS_Implementation.ipynb`
**Scope:** all 3 towers (T2, T4, T9), pooled, all 5 gap-length scenarios (CLAUDE.md full-coverage default).
**Executed:** 2026-07-19

---

## 1  Why this experiment

SAITS (Self-Attention-based Imputation for Time Series, Du et al. 2023, via `pypots`) jointly
imputes a multivariate window using diagonally-masked self-attention across time and features at
once — conceptually a stronger fit for gap-filling than RFm's point-wise regression-on-lag-features,
and worth a real try against this project's recorded best (partial-pooled, external-sourced RFm
under full-period gap-CV, D-35/D-49):

| Tower | RFm champion R² |
|---|---|
| T2 | 0.574 |
| T4 | 0.402 |
| T9 | 0.418 |

## 2  Environment note

`pypots` was not installed; installing it surfaced a real blocker — `pypots.imputation.__init__`
eagerly imports every bundled model (including `TimeLLM`, which needs `transformers` →
`torchvision`), and the environment's `torchvision` (0.21.0+cu124) was ABI-mismatched against
`torch` (2.11.0+cu128), crashing the import entirely (not just SAITS's own dependency chain, which
by itself needs neither `pytorch-lightning` nor `transformers`). Fixed by reinstalling
`torchvision` from the matching cu128 index (`0.26.0+cu128`) — resolved cleanly, no `torch`
version change, and fixed the project's previously-broken `pytorch-lightning` import as a bonus.

## 3  Methodology (deliberate deviations from RFm's evaluation, stated explicitly)

Reused the *exact same* `insert_calendar_gaps` held-out-timestamp generator F08 used (same
`SCENARIOS`/`MASK_FRAC`/`DOMAIN`/seed), so every point SAITS is scored on is identical to what RFm
was scored on. But RFm retrains a fresh model per `(tower, scenario, rep)` (75 cheap fits); SAITS is
a deep model, so instead:

- One **pooled SAITS model** (T2+T4+T9 + one-hot tower channels, matching RFm's own pooled config),
  trained **once**, on the **union** of all 25 `(scenario, rep)` held-out sets per tower excluded
  from training — guarantees zero leakage of any scored point, across every scenario at once.
- Predictions sliced back out per `(tower, scenario, rep)` and scored with the same `mets`/
  `med_metrics` RFm used, into the same `(tower, scenario) → R2/RMSE/MAE/MBE` table shape as
  `results/f09a_summary.csv`.
- **Net effect, if anything, is harder for SAITS**: its one training set excludes the union of all
  25 held-out sets, whereas RFm's per-scenario retrain still had the other 4 scenarios' held-out
  points available as valid training data for any given run.
- Feature set: RFm's EXT-variant channels (met, gap-filled FCO2, cyclical time, livestock density,
  grazing flag, management recency, GPP/Reco, tower dummies) **minus** the hand-engineered
  SWC/TS lag columns (`swc_l*`/`ts_l*`) — dropped as redundant once SAITS sees raw temporal context
  via windowing (336h windows, 24h stride). Per-channel `StandardScaler`, fit once on the pooled,
  domain-restricted, union-masked data; inverted on the target channel before scoring.
- Model: `pypots.imputation.SAITS`, `n_layers=2, d_model=128, n_heads=4, d_ffn=256, dropout=0.1,
  epochs=100, patience=10` — one configuration, no HPO sweep (first-pass simplicity).

## 4  Feasibility (Phase 1 smoke test) — GO

Tower 4 solo, scenario `m` (32h), 1 rep, 20-epoch cap: **35.9s total wall time**, R²=0.027,
RMSE/MAE in the same physical range as RFm's own numbers (confirms scaling/inverse-transform
correctness). Loss was still falling at the epoch cap — a real "needs more training," not a
red flag — so the full run proceeded with the fuller 100-epoch/patience-10 budget.

## 5  Results — full pooled run, all 3 towers × 5 scenarios

**Full pooled fit: 300.5s (early-stopped at epoch 97, best epoch 87). Total Phase 2 wall time: 317s.**
Cheap — the constraint on this experiment was never compute.

| Tower | scenario | R² | RMSE | MAE | MBE |
|---|---|---|---|---|---|
| T2 | vs | 0.034 | 157.88 | 38.87 | -15.23 |
| T2 | s | 0.027 | 151.37 | 38.93 | -15.66 |
| T2 | m | 0.043 | 139.93 | 37.77 | -15.71 |
| T2 | l | 0.017 | 153.24 | 43.23 | -18.16 |
| T2 | m1 | 0.020 | 94.53 | 33.62 | -12.72 |
| T4 | vs | -0.003 | 122.45 | 46.29 | -15.08 |
| T4 | s | -0.003 | 131.44 | 48.42 | -16.77 |
| T4 | m | 0.001 | 131.97 | 48.67 | -16.12 |
| T4 | l | -0.008 | 118.48 | 45.56 | -14.41 |
| T4 | m1 | -0.003 | 119.45 | 44.14 | -16.22 |
| T9 | vs | -0.015 | 139.57 | 55.45 | -28.55 |
| T9 | s | -0.019 | 147.18 | 58.38 | -30.63 |
| T9 | m | -0.015 | 146.25 | 55.11 | -28.91 |
| T9 | l | -0.037 | 146.65 | 56.59 | -35.14 |
| T9 | m1 | -0.024 | 161.48 | 61.71 | -34.55 |

**Headline (median R² across 5 scenarios, matching the champion's own convention):**

| Tower | SAITS median R² | RFm champion R² | delta |
|---|---|---|---|
| T2 | 0.027 | 0.574 | -0.547 |
| T4 | -0.003 | 0.402 | -0.405 |
| T9 | -0.019 | 0.418 | -0.437 |

**SAITS loses to the RFm champion at every tower, by a wide margin.** Result reproduced closely on
a full independent rerun (T2 0.031, T4 0.000, T9 -0.016) — not run-to-run noise.

## 6  Interpretation — why it underperforms

The consistent, large **negative MBE across every single row** (-9 to -35 nmol m⁻² s⁻¹) is the key
diagnostic: SAITS is systematically under-predicting, not just noisily wrong. Three compounding,
non-exclusive causes:

1. **FCH4 is far sparser than SAITS's design target.** SAITS assumes a mostly-dense multivariate
   series with occasional gaps — self-attention reconstruction needs real neighboring target
   observations to anchor on. FCH4 itself is only ~25–45% valid *before* any masking (T4 44.6%,
   T9 25.6%, T2 12.1%). Layering the union-mask on top (excluding ~35% of T4's domain from
   training, see §3) leaves the target channel very sparse while every covariate channel stays
   fully dense — an asymmetry that likely pushes the model toward a "typical value" fallback
   rather than genuine per-timestep reconstruction.
2. **FCH4 is heavily right-skewed/spike-dominated** — this project's own recurring "MASE<1
   alongside near-zero/negative R² = spike-tail signature" (first named D-44b, recurring through
   B-05/B-06/B-09 onward). A model trained under a symmetric loss on a skewed target regresses
   toward the low/typical value rather than reconstructing rare high-flux events, consistent with
   the systematic under-prediction observed here.
3. **RFm has structural advantages not replicated here**: 500 trees can carve explicit high-flux
   decision regions directly from `lsu_dens`/season, plus RFm's hand-engineered SWC/TS lags
   (up to 672h) — deliberately dropped for SAITS on the theory that windowing replaces them. A
   2-layer/128-dim attention encoder trained for 100 epochs on ~3,900 pooled windows may simply
   not recover equivalent structure unassisted, at this scale and without HPO.

## 7  Recommendation

**Not adopted.** RFm (D-35/D-49) remains the standing gap-filling recommendation — no change to
`BEST_RESULTS.md` §1. Logged here as a legitimate tested-and-rejected alternative (mirroring how
B-03a/F-09b are handled elsewhere in this project), not a technical failure: the pipeline runs
correctly (confirmed via the smoke test's RMSE/MAE unit sanity-check and a full independent rerun
reproducing closely), it's cheap (~5 minutes for the full 3-tower run), and the negative result is
diagnosable (§6), not a mystery.

**If revisited:** the most promising single change would be addressing the target-channel sparsity
directly — e.g. per-scenario retraining (closer to RFm's own protocol, more expensive but avoids
the union-mask's extra sparsity), restricting windows to only those with adequate observed-target
density, or a training-loss reweighting toward the flux tail (matching this project's established
"spike-tail" finding elsewhere). Not pursued in this pass — first-look feasibility was the ask, and
the answer is "technically yes, easily; competitively, not without further work."

## 8  Follow-up: testing the diagnosed failure modes (same session, direct user request)

User asked to "test all of this out" against the three causes diagnosed in §6. Five levers tested,
staged cheapest/most-diagnostic first, each stage's continuation decided by its own result rather
than committing to a full sweep upfront. All runs seeded (`torch.manual_seed`) — §5's original
result had drifted run-to-run from unseeded init (e.g. T2 ~0.03→~0.002 across two identical runs).

**EXP_B — per-scenario retraining (cause 1, sparsity):** retrained per scenario (5 fits, only that
scenario's 5 reps excluded) instead of one model trained on the union of all 25 held-out sets.
**Confirmed directionally but small**: flips T4 and T9 from negative to positive R² (+0.02 to
+0.05), T2 roughly flat. Sparsity was a real contributing cause, not the dominant one.

**EXP_C — solo vs. pooled structural probe (scenario `m` only):** genuinely noise-level. First run:
solo wins by mean (0.031 vs. 0.028). Independent full rerun: **flips to pooled winning** (0.028 vs.
0.027) — a ~0.001 margin either way. Not a real structural effect at this experiment's scale;
locked in **solo** for the remaining stages regardless (a defensible, not "winning," choice).

**EXP_D — spike-weighted loss (cause 2, FCH4's spike-dominated skew):** custom `SpikeWeightedMAE`
(upweights points by `1 + |target|` in standardized units) in place of SAITS's default MAE.
**The single largest lever tested in this notebook** — reproduced in both full reruns regardless
of solo/pooled structure: roughly **5–6x'd R²** on its own (e.g. solo run: T2 0.018→0.076, T4
0.029→0.181, T9 0.040→0.119; pooled run: T2 0.028→0.143, T4 0.033→0.108, T9 0.022→0.092).
Confirms cause (2) — not sparsity, not pooling — was the dominant failure mode.

**EXP_E — bigger model (cause 3, no HPO):** `n_layers=3, d_model=256` (up from 2/128) on top of the
winning loss. A further consistent, real gain in both reruns (+0.04 to +0.11 depending on tower).

**Final confirmation — solo + SpikeWeightedMAE + n_layers=3/d_model=256, all 5 scenarios (not just
the `m` probe):**

| Tower | Final median R² | Original naive baseline | RFm champion | Gap now (was) |
|---|---|---|---|---|
| T2 | 0.192 | 0.020–0.028 | 0.574 | -0.38 (was -0.55) |
| T4 | 0.225 | -0.03 to 0.033 | 0.402 | -0.18 (was -0.43) |
| T9 | 0.110 | -0.02 to 0.040 | 0.418 | -0.31 (was -0.44) |

**Still not adopted — RFm remains the standing recommendation (D-35/D-49), no `BEST_RESULTS.md`
change** — SAITS loses at every tower even after this pass. But the gap narrowed substantially
(T4 more than halved), a qualitatively different picture from §5's naive baseline. **Not uniform**:
T2's `l` (288h) scenario actually went *negative* (R²=-0.073, MBE=+36.3 — the one case in this
whole follow-up with a large *positive* bias, opposite the systematic under-prediction seen
everywhere else), reported here rather than smoothed over by the median.

**If revisited further:** prioritize the loss objective (a proper quantile/focal-style loss, not
just this hand-rolled weighting) and a real architecture search — that's where the confirmed gains
came from. Sparsity fixes and the solo/pooled structural choice are lower priority; both showed
real-but-small or noise-level effects by comparison. 15 rows tagged `F-11-phase4` in
`results/benchmarks.csv`; `results/f11_phase4_final_summary.csv`.

## 9  Appendix: observed-vs-gap-filled chain figures (for all downstream gap-filling work)

Prompted by inspecting §5's numbers directly — a general-purpose diagnostic, **not specific to
SAITS or this experiment**, added to the end of `F11_SAITS_Implementation.ipynb` (kept in-notebook
per direct user instruction, rather than a standalone script). Reads straight from the production
precompute `data/Hourly/fch4_gapfilled.csv` (D-36, the current RFm champion's output) — no
retraining needed, so it stays valid regardless of which model is later adopted.

**24 figures** (`results/figures/gapfill_chains/T{2,4,9}_{2017..2024}.png`, one per tower × full
calendar year, 2025's 25-hour stub excluded), styled after `b10_b13_chain_plots.py`'s
committed-and-rerunnable convention: **solid black = real observed FCH4, dotted black = gap-filled
(model-estimated) FCH4**, split from the same `FCH4_gapfilled [Tower N]` column by
`FCH4_observed_mask [Tower N]`. Each title reports that tower/year's % observed, giving an
at-a-glance read of how much of any given panel is real vs. estimated — e.g. Tower 4, 2019 is only
35% observed, so most of that year's spike structure visible in the chain is the gap-filler's
estimate, not measurement. Useful going forward for spot-checking *any* new gap-filling model's
output at a glance, the same way `b10_chains` does for forecasting.

**Redesigned three times after direct user feedback**, each pass converging closer to what
actually reads at this data density:
1. Original ask (solid-black-observed vs. dotted-black-gap-filled) at compact
   `figsize=(13,5)`/`dpi=100` — illegible at ~8,760-hourly-points/year density.
2. Widened to `figsize=(28,8)`/`dpi=150` with 8-day-interval rotated date ticks and light
   gridlines — still illegible; two overlapping same-color line styles visually collapse into the
   same black blur once many short segments alternate at hourly resolution, regardless of figure
   size. Tried a one-line-plus-`axvspan`-shading variant at this stage too (single continuous
   black line for the value, light gray shading behind gap-filled stretches) — legible, but not
   what the user was actually asking to see reproduced.
3. **Final design, prompted by a reference chart the user supplied**: resample to a **daily sum**
   first (`.resample("1D").sum(min_count=1)`, split into `Sum of y_observed`/`Sum of y_gapfilled`
   by `FCH4_observed_mask`, each with `NaN` gaps where that category doesn't apply for the day).
   Dropping density to ~365 points/year is what actually makes solid-vs-dotted legible — the
   original ask was achievable, the missing piece was resolution, not styling.

One implementation bug caught and fixed along the way: an in-place cell edit had dropped the
loop/print code that actually *calls* the plotting function (the cell ran with zero errors and
zero output, silently producing no new figures) — caught by checking file timestamps after
execution rather than trusting a clean nbconvert exit code alone.

**Sources:** `notebooks/04_feature_engineering/F11_SAITS_Implementation.ipynb` (main experiment +
Phase 4 follow-up + Appendix chain-figure section), `results/f11_summary.csv`,
`results/f11_phase4_final_summary.csv`, 15 rows tagged `F-11` + 15 rows tagged `F-11-phase4` in
`results/benchmarks.csv`, `results/figures/gapfill_chains/` (24 figures).
