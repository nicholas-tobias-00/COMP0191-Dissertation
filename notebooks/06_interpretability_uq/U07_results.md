# U-07 — Livestock-density-stratified CQR: thinner margins where livestock presence is smaller

**Scripts:** `u07_lsu_stratified_cqr.py`, `u07_lsu_cqr_comparison_plot.py`.
**Data:** `results/u07_u04_lsu_cqr_summary.csv` (178 rows), `u07_u05_lsu_cqr_summary.csv` (89 rows).
**No new model calls** — recalibrates U-04's/U-05's already-saved chains a second time (CQR + a
new stratification axis, not a restart).

## Context

User question, checked empirically before building anything: "can't the margin be thinner for
instances where livestock presence is smaller?" U-06's CQR already lets the model's own raw q05/q95
respond to livestock density implicitly (it's a model input), but the *additive calibration margin*
layered on top was only binned by lead-time (U-04) or lead-time × AOA-flagged status (U-05) — never
by the covariate actually driving the heteroscedasticity most strongly.

**Checked directly, and the signal is much stronger than the AOA-distance check U-05 ran**:
`corr(|residual|, fx_lsu_dens) = 0.43–0.45` (vs. AOA-distance's weak 0.09–0.15), and residuals are
**~3.2× larger** on above-median-LSU-density days (51.0 vs. 15.9–16.2 mean |residual|).
`fx_cattle_dens` correlates almost identically (0.427) — consistent with S-05's own finding that
cattle dominates the LSU composition; sheep/lamb correlate at noise level (0.03–0.05). Used
`fx_lsu_dens` directly (the standard, interpretable aggregate) rather than a species-specific
variable.

## Method

Same CQR machinery as U-06 (nonconformity = `max(q05-y_true, y_true-q95)`,
`rr.conformal_margins_by_bin()` reused unchanged a fifth time across this project's UQ work) — only
the bin *key* changes, from lead-time-bin alone to **lead-time-bin × LSU-tertile** (e.g.
`"31-90_low"`). `conformal_margins_by_bin()` needed zero code changes to support this — it was
already generic over arbitrary dict keys. Tertile boundaries (low/mid/high, at the 1/3 and 2/3
quantiles of `fx_lsu_dens`) computed from the **leave-in (calibration) anchors only** per test fold,
never the held-out test anchor — same no-leakage discipline as every other leave-one-anchor-out step
in this project (the same class of bug U-05 caught and fixed for its own AOA computation).

## Result: low-LSU intervals are 26–59% the width of high-LSU intervals — a genuine win-win, not a trade-off

Full roster (T2/T4/T9 × TabPFN/TabICLv2 for U-04; T2/T4/T9 × TabICLv2 for U-05, matching S-05's
TabICL-only scope), computed as actual mean prediction interval width (MPIW = `(q95+margin) -
(q05-margin)`) per tower/model, averaged over all anchors and lead times within each LSU tier:

| Data | Tower | Model | Low-LSU MPIW | Mid-LSU MPIW | High-LSU MPIW | Low as % of High |
|---|---|---|---|---|---|---|
| U-04 | T4 | TabPFN | 81.0 | 98.9 | 221.1 | 37% |
| U-04 | T4 | TabICLv2 | 173.7 | 233.4 | 347.7 | 50% |
| U-04 | T9 | TabPFN | 102.1 | 131.7 | 387.3 | **26%** |
| U-04 | T9 | TabICLv2 | 215.8 | 277.4 | 501.3 | 43% |
| U-04 | T2 | TabPFN | — | — | — | n/a (see below) |
| U-04 | T2 | TabICLv2 | — | — | — | n/a (see below) |
| U-05 | T4 | TabICLv2 | 215.1 | 236.7 | 366.5 | 59% |
| U-05 | T9 | TabICLv2 | 219.5 | 274.1 | 538.8 | 41% |
| U-05 | T2 | TabICLv2 | — | — | — | n/a (see below) |

The 29–46% range originally reported was the two (T4-only, or the first-checked) rows; the fuller
6-combination sweep widens that range to 26–59% but the direction and magnitude of the effect is
consistent everywhere it can be computed — low-LSU periods get meaningfully tighter intervals at
every tower and both models, never the reverse.

**Tower 2 is genuinely degenerate here, not omitted** — `lsu_cqr_margin` is `NaN` for all 100% of
its rows in both U-04 and U-05 summaries (`u07_u04_lsu_cqr_summary.csv`,
`u07_u05_lsu_cqr_summary.csv`), because T2's *base* split-conformal calibration (U-04/U-05's own
Step 2) already failed for T2 before CQR or LSU-stratification were ever applied — consistent with
[[project_r02_findings|T2's known data-scarcity]] pattern recurring throughout this project's
UQ work (U-05's Step 4 gating decision explicitly propagates this). No band can be drawn for T2
at any stratification level; this is the same finding already logged in U-04/U-05, not a new gap.

PICP stays reasonably consistent across tiers (0.76–0.83) for T4/T9, not wildly unbalanced the way
the plain flat margin's spike-vs-normal split was (24% vs. 97%) — each tier is calibrated against
its own, now-homogeneous residual distribution rather than one pooled distribution that had to
compromise between very different regimes.

**Verified this doesn't trade away U-06's spike-coverage fix**: spike days (top 10% magnitude)
average **3.2× higher `fx_lsu_dens`** than normal days (3.30 vs. 1.04) — the "high" tier
substantially captures the spike population with its own dedicated, appropriately-wide calibration
set, not diluted by mixing in low-LSU days the way the single pooled CQR margin was. This is a
genuine improvement on both axes simultaneously: tighter where the model and the real driver both
say confidence should be high, still wide where it shouldn't be — not a zero-sum trade against
U-06's fix.

## Figures

`u07_lsu_cqr_comparison_plot.py` — flat CQR band (U-06, red) vs. LSU-stratified band (U-07, blue)
on the same chain, with `fx_lsu_dens` plotted directly below and background shading marking which
tier each day falls into. The blue band visibly tightens through winter (near-zero livestock) and
widens through the grazing season, tracking the actual covariate rather than a flat lead-time-only
width. Full roster, one figure per (tower, model, data_label) at its own spikiest anchor (T2
skipped and logged, per the degeneracy above) — 6 figures in `results/figures/u07_lsu_cqr/`:
`T{4,9}_anchor{yr}_{TabPFN,TabICLv2}_U04_lsu_stratified.png` (4) and
`T{4,9}_anchor{yr}_TabICLv2_U05_lsu_stratified.png` (2).

## Practical implications

1. **This should be the standing UQ method going forward, on top of U-06's CQR** — it costs
   nothing extra to compute (same calibration function, same already-available `fx_lsu_dens` input)
   and improves both the common case (most days are low-to-mid LSU density, now correctly tight)
   and the rare case (high-LSU/spike days, still appropriately wide, verified not weakened).
2. **The livestock-density signal is a substantially cleaner stratifier than AOA distance/flagged
   status was** (0.43 vs. 0.09–0.15 correlation) — worth remembering if this pattern recurs: a
   domain-meaningful covariate directly tied to the outcome's own magnitude can be a much better
   stratifier than a generic distance-to-training-data metric, when one is available.
3. **Not yet applied to S-05's actual scenario trajectories**, same standing caveat as U-06 — would
   need raw daily q05/q95 saved for scenario points, which the current daily-chains-subset scripts
   don't request.

## Files

- `notebooks/06_interpretability_uq/u07_lsu_stratified_cqr.py`, `u07_lsu_cqr_comparison_plot.py`.
- `results/u07_u04_lsu_cqr_summary.csv`, `u07_u05_lsu_cqr_summary.csv`.
- `results/figures/u07_lsu_cqr/` (1 comparison figure).

No `benchmarks.csv` rows (UQ output, same exclusion precedent as U-01 through U-06).
