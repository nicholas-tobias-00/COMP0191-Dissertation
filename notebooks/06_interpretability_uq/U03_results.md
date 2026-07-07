# U-03 results: does U-02's conformal calibration hold up under distribution shift?

## Context

U-02 (D-62) built full leave-one-anchor-out conformal calibration across all 8 recursive-rollout
models, 3 towers, 5 anchors — converging every model to ~0.88–0.90 PICP regardless of raw
coverage. That calibration is a **vanilla split-conformal, symmetric additive margin**
(`median ± margin`, computed per lead-time bin). Split-conformal's coverage guarantee is only
valid if the point being predicted is *exchangeable* with the calibration data — an assumption
`U02_results.md`/`DECISIONS.md` D-62 never discussed. This became directly relevant once the
project's next phase shifted from "long-horizon forecasting" to **scenario simulation** (2x
livestock density, CMIP6 2050 climate scenarios) for the Phase-07 digital-shadow work — a genuine
scenario is, by construction, *not* exchangeable with 2018–2022's historical calibration data.

U-03 tests this directly, in two honestly distinct parts. **There is no real observed FCH4 for a
hypothetical 2x-livestock or 2050-climate world**, so "coverage under a genuine future scenario"
cannot be empirically validated — only diagnosed:

- **Part A** (real ground truth, cheap re-analysis): does U-02's already-measured conformal PICP
  correlate with how different a real historical anchor's driver distribution was from the other
  4 anchors used to calibrate it?
- **Part B** (no ground truth — a sensitivity/diagnostic check, not a coverage claim): as the
  headline scenario knob (`fx_lsu_dens`, livestock density) is pushed synthetically beyond its
  training-seen range, does each model's point prediction keep responding, or does it plateau?

Both parts reuse U-02's existing quantile-rollout and conformal-calibration machinery
(`src/models/recursive_rollout.py`: `tree_rollout_quantile`, `sarimax_quantile`,
`dl_rollout_quantile`, `conformal_margins_by_bin`, `lead_time_bin`) completely unmodified — no
changes to any shared module, `u02_multi_anchor_tower.py`, or `u02_fanchart_plots.py`.

## Part A: natural-shift stress test

**Method.** For each of the 5 anchors × 3 towers, compute a shift score: mean |z-score| of 4
scenario-relevant drivers (`fx_lsu_dens`, `fx_TA_mean`, `fx_PRECIP_sum`, `fx_SWIN_mean`) — each
anchor's real 365-day rollout-window mean, z-scored against the pooled mean/std of the **other 4
anchors'** equivalent windows for that tower (the exact reference set U-02's own leave-one-
anchor-out scheme used to calibrate that anchor). Joined against U-02's already-computed,
n-weighted mean conformal PICP per anchor/tower/model.

**Result — per-anchor-tower table (mean conformal PICP across the 8 models):**

| Anchor | Tower | Shift score | Conformal PICP | Raw PICP (mean across models) |
|---|---|---|---|---|
| 2018 | 4 | 0.588 | 0.917 | 0.675 |
| 2019 | 4 | 0.836 | 0.871 | 0.707 |
| 2020 | 4 | **1.991** | 0.895 | 0.678 |
| 2021 | 4 | 1.876 | 0.838 | 0.628 |
| 2022 | 4 | 1.624 | 0.952 | 0.639 |
| 2019 | 9 | 1.281 | 0.838 | 0.580 |
| 2020 | 9 | 1.934 | 0.896 | 0.655 |
| 2021 | 9 | 1.187 | 0.814 | 0.724 |
| 2022 | 9 | **1.554** | **0.982** | 0.719 |
| 2018 | 2 | 0.544 | NaN | 0.581 |

**Headline finding: no clear evidence, within the range of historical shift actually observed
(2018–2022), that conformal calibration degrades under distribution shift.**
Correlation(shift_score, conformal_picp): **Tower 4 = −0.166** (n=5, essentially no relationship);
**Tower 9 = +0.562** (n=4, and in the *opposite* direction from the exchangeability-violation
concern — the most-shifted Tower-9 anchors, 2020 and 2022, calibrated *better* than average, not
worse). With only 4–5 anchors per tower, neither correlation should be treated as statistically
robust on its own — this is not strong evidence of robustness, only an *absence of the feared
degradation signal* in the data that exists. **Tower 2 has zero usable data points** (all
conformal columns NaN across all 5 anchors), the same data-scarcity limitation already documented
in D-62 — confirmed here rather than newly discovered.

The single most-shifted (anchor, tower) pair among T4/T9 is **anchor 2020, Tower 4** (shift score
1.991, driven almost entirely by that year's anomalous weather — `z_fx_TA_mean = −6.09`, a large
cold anomaly relative to the other 4 anchors). Its conformal PICP (0.895) is close to the T4
average (0.895 across all 5 anchors) — i.e. even the year that was most different from its own
calibration set showed no visible degradation in realized coverage. Highlighted fan charts for
this anchor/tower/all-8-models are saved to `results/figures/u03_fancharts/T4_anchor2020_*.png`
(same visual convention as U-02's fan charts: solid calibrated band, hatched raw fallback).

**Caveat, stated plainly:** the shift magnitudes tested here (max shift_score ≈ 2.0, essentially
one anomalous weather year) are far smaller than a genuine 2x-livestock or 2050-climate scenario
would produce. This result clears a much smaller bar — "does calibration wobble across ordinary
year-to-year variation" — and **cannot be extrapolated to claim that conformal calibration would
remain valid under a real future scenario shift**. It is reassuring evidence that the method is
not fragile to *modest* shift, nothing more.

## Part B: synthetic extrapolation diagnostic

**Note (added after this section was first written, then updated a second time): the
single-anchor/single-tower, 5-model result below was superseded twice** — first by a 5-anchor ×
2-tower sweep (still only 5 of 8 models, still missing Tower 2), and then, after the user directly
questioned that narrower scope too ("I don't think U03 has been completed for all models, towers,
and years?" / "tower 2 should also be included... always include tower 2"), by the genuinely
complete **8-model × 3-tower × 5-anchor** sweep in "Part B: full coverage" below. **The qualitative
direction below holds up (trees plateau, SARIMAX doesn't); the specific magnitudes and the
TFT-sits-in-between claim do not** — see the full-coverage section for the corrected, complete
numbers. Kept here for the historical record of what the pilot run found and why the expansion
mattered, twice. **One additional wrinkle, itself informative**: re-executing this pilot later (via
the notebook) produced a *different* TFT trajectory than the one quoted below (declining rather
than rising) — a direct, unplanned demonstration of TFT's already-documented unseeded-initialization
non-determinism (D-62), not a new bug. Treat any single TFT number in this pilot section as
illustrative of that instability, not as a trustworthy point estimate — the full-coverage section's
TFT numbers (averaged over 15 anchor/tower fits) are the ones to actually rely on.

**Method (pilot, single anchor/tower).** Tower 4, anchor 2021-12-16. RF/XGB(×3 quantile)/LightGBM(×3 quantile)/SARIMAX/
TFTQuantile refit exactly as `u02_multi_anchor_tower.py`'s fit stage does — the fitted models are
never retrained on perturbed data. `fx_lsu_dens` is multiplied by a factor swept over
**{1.0 (real), 1.5, 2.0, 2.5, 3.0}**, applied only to the post-anchor rollout window (pre-anchor
training history is always real). Tower 4's pre-anchor (training) maximum `fx_lsu_dens` is 4.99;
the real rollout window's own maximum is 3.87, so the training-range boundary is crossed at
**multiplier ≈ 1.29×** — the sweep spans comfortably from within-range to well beyond it (window
max reaches 11.6 at 3.0×). TabPFN and both ensembles are explicitly excluded (their behavior under
a directly-perturbed covariate at an unseen magnitude isn't well-characterized and isn't the
flattening question being asked) — deferred, not silently dropped.

**Result — mean predicted median FCH4 (nmol m⁻² s⁻¹), averaged across the 6 lead-time bins:**

| Model | 1.0× | 1.5× | 2.0× | 2.5× | 3.0× | % change (1.0→3.0×) |
|---|---|---|---|---|---|---|
| RF | 34.73 | 35.85 | 36.75 | 36.97 | 37.60 | **+8.3%** |
| XGB | 28.68 | 29.55 | 29.82 | 30.41 | 30.83 | **+7.5%** |
| LightGBM | 28.50 | 29.05 | 29.39 | 29.87 | 30.26 | **+6.2%** |
| TFT | 19.75 | 21.73 | 23.32 | 24.62 | 25.64 | **+29.9%** |
| SARIMAX | 34.61 | 45.18 | 55.74 | 66.31 | 76.88 | **+122.1%** |

**Headline finding: a clean, structural split by model family.** RF/XGB/LightGBM show a strongly
**decelerating, near-flat** response — mean prediction rises only 6–8% across a 3× range of the
input, even though the input itself is scaled 3× (200% increase). This is exactly the
tree-extrapolation-ceiling signature the literature review predicted: RF/XGB/LightGBM split on
leaf boundaries fit to the 2018–2021 training range and cannot extrapolate a trend past it — once
`fx_lsu_dens` exceeds ~5 (training max), additional increases mostly fall into the same terminal
leaves and stop changing the prediction.

**SARIMAX shows the opposite pattern**: a near-perfectly **linear** response (+122%, increments of
≈15.8 nmol per 0.5× step — consistent with a fixed linear exogenous-regression coefficient that
does not saturate). This is *not* necessarily more correct than the trees' flattening: by 3.0× the
mean prediction (76.9) is more than double Tower 4's real historical mean (33.5, per CONTEXT.md),
extrapolated from a coefficient estimated only on the 2018–2021 range with no structural guarantee
it remains valid 3× beyond it. Unbounded linear extrapolation risks *overstating* the scenario
effect exactly as much as the trees risk *understating* it.

**TFT sits in between** (+29.9%) — a real, non-trivial response that also decelerates, but far
less severely than the trees. The mechanism is plausibly different (no literal leaf-boundary
clipping; more likely a training-range-driven saturation in the learned nonlinear function), but
the net effect — a response that weakens as the input moves further from what the model has seen
— is qualitatively the same caution as the tree models', just less extreme.

**Explicit caveat, stated in every output:** the "nominal calibrated width" values recorded in
`results/u03_extrapolation_stress_test.csv` (`median ± margin`, using U-02's real, frozen,
anchor-2021 conformal margins) are a **mechanical application, not a validated interval** — the
whole point of this test is that the exchangeability assumption underlying that margin is
deliberately violated once the input is scenario-perturbed. Nothing in this section should be read
as a claim about the interval's actual coverage under these synthetic inputs.

Response-curve plot (median prediction vs. multiplier, one line per model, training-range boundary
marked): `results/figures/u03_fancharts/T4_anchor2021_lsu_perturbation_response.png`.

## Part B: full coverage — all 8 models × all 3 towers × 5 anchors

**Why this was needed, twice.** The pilot used one anchor and one tower. A first expansion (5
anchors × Towers 4/9, still 5 of 8 models) fixed the anchor-count problem but repeated the same
mistake at smaller scale — it still silently dropped TabPFN, both ensembles, and Tower 2. The user
caught both gaps directly ("I don't think U03 has been completed for all models, towers, and
years?", then "tower 2 should also be included... always include tower 2"). This section is the
result of closing both gaps at once: **all 8 U-02/B-10/B-13 models, all 3 towers, all 5 anchors —
the same scope U-02 itself used**, not a narrower diagnostic subset.

**Method.** Same fit/rollout logic as the pilot (RF/XGB/LightGBM/TFT pooled-fit per anchor,
SARIMAX/TabPFN fit per anchor/tower, never pooled). Two additions: **TabPFN** via the same
`rr.tabpfn_forecast(..., quantiles=...)` call U-02 uses, with `fx_lsu_dens` perturbed in its
`future_covariates` frame before the call (local-mode inference, `TABPFN_TOKEN` already
provisioned from B-13/U-02). **Both ensembles** constructed post-hoc from the already-perturbed
RF/XGB/LightGBM/SARIMAX outputs at each multiplier — an unweighted mean and B-09's frozen
MASE-weighted mean, identical to U-02's own ensemble definition, at zero extra model-fitting cost.
3,600 rows total (5 anchors × 3 towers × 5 multipliers × 6 bins × 8 models), zero TabPFN skips.

**A real, non-bug finding surfaced immediately by including Tower 2: the livestock-extrapolation
diagnostic is structurally degenerate there for 4 of the 5 anchors.** Tower 2's `fx_lsu_dens` is
**exactly 0.0 for the entire 365-day rollout window** in anchors 2019, 2020, 2021, and 2022 (0/365
nonzero days each) — multiplying zero by any factor is still zero, so there is nothing for the
perturbation to act on. Anchor 2018 has a small real signal (20/365 nonzero days, window max 3.38,
well below the ~4.5 training-max other anchors show) and produces small but genuine responses (XGB
+11.0%, RF +10.5%, SARIMAX +13.0%, TabPFN only +0.75%, TFT only +2.4%) — even at this much smaller
scale, SARIMAX still comes out relatively highest and TFT/TabPFN relatively flattest, consistent
with the T4/T9 pattern below. **Tower 2 is reported in the data (`u03_extrapolation_stress_test_multi.csv`) but excluded from the headline comparison table and the robustness plot** — averaging
in four rows that are 0% by construction, for a reason that has nothing to do with model behavior,
would misleadingly drag every model's summary toward zero.

**Result — % change in mean predicted median FCH4 (1.0× → 3.0× `fx_lsu_dens`), Towers 4+9 only
(10 anchor × tower combinations; Tower 2 excluded for the reason above):**

| Model | mean | std | min | max |
|---|---|---|---|---|
| LightGBM | 21.1% | 11.7 | 5.2% | 34.8% |
| XGB | 22.3% | 11.7 | 6.0% | 39.1% |
| RF | 23.4% | 10.2 | 8.3% | 36.6% |
| TFT | 26.1% | 15.0 | 11.0% | 59.2% |
| TabPFN | 30.2% | 33.9 | **−4.9%** | 90.1% |
| Ensemble_unweighted | 50.2% | 11.0 | 38.8% | 73.3% |
| Ensemble_MASEweighted | 49.2% | 10.8 | 37.7% | 71.6% |
| SARIMAX | **150.3%** | 88.6 | 58.6% | 379.8% |

**New headline finding — the production-recommended ensemble is not immune to this problem.**
B-10's standing recommendation (the unweighted/MASE-weighted 4-model ensemble) is 75%
tree-weighted, so a naive expectation would be that it inherits the trees' flat, well-behaved
response. It doesn't: **the ensembles show +49–50% mean overshoot, more than double the pure
tree-mean (~23%)**, because SARIMAX's 25% weight pulls the blended prediction up substantially even
though it's only one of four members. This is a genuinely new, actionable finding for Phase 07 —
the model this project has repeatedly recommended for production carries real extrapolation risk
from its SARIMAX component, not just a diluted, safely-averaged-out trace of it.

**TabPFN is the least predictable model under this perturbation** — widest range of any model
(−4.9% to +90.1%) and the only one that sometimes *decreases* under more livestock. A zero-shot
foundation model with no explicit prior on this driver's direction of effect behaving erratically
under a synthetic covariate shift is itself a notable, reportable result, not noise to explain away.

**What holds up robustly across every one of the 10 T4/T9 cases, confirmed pairwise against all 7
other models, not just the tree/TFT subset checked previously:**
- **SARIMAX is the maximum of all 8 models in 10/10 cases** — the one fully robust ordering claim,
  now even more robust than before (it beats the ensembles and TabPFN too, not just the trees).
- Trees, TFT, and TabPFN are best described as a broadly comparable "muted response" cluster (5–34%
  typical range, with TabPFN's tail reaching further both up and down); the ensembles form a
  distinct, elevated middle tier (38–73%); SARIMAX is the outlier on its own (59–380%).

Per-anchor-tower response-curve plots (15 total, all 3 towers × 5 anchors, same visual convention
throughout — Tower 2's plots make the fx_lsu_dens=0 degeneracy directly visible as flat lines for
2019–2022): `results/figures/u03_fancharts/T{2,4,9}_anchor{2018..2022}_lsu_perturbation_response.png`.
Robustness summary plot (%-change scatter, one column per model, Towers 4+9 only, 10 points/model):
`results/figures/u03_fancharts/pct_change_summary_all_anchors_towers.png`.

## Discussion: the exchangeability assumption

Split-conformal prediction's coverage guarantee (Lei et al. 2018 / Romano et al. 2019, the
methodology `conformal_margins_by_bin` already implements) holds *if and only if* the calibration
set and the test point are exchangeable — informally, drawn from a distribution where reordering
doesn't change anything. U-02's leave-one-anchor-out design satisfies this reasonably well for
its own stated purpose (5 real historical years, all from the same underlying 2018–2023 regime,
tested against each other) — Part A's finding is consistent with that design being sound *for
that purpose*.

It does **not** extend to genuine scenario extrapolation, and Part B demonstrates concretely why:
once `fx_lsu_dens` (or any driver) is pushed via scenario construction rather than sampled from
the real data-generating process, the resulting input is not exchangeable with U-02's calibration
set by construction — and the two model families respond in opposite, both-untrustworthy ways
(flattening vs. unbounded linear growth). A calibration margin computed on real 2018–2022 residuals
carries no guarantee once applied to either failure mode.

## Recommendation for Phase 07

1. **Do not apply U-02's conformal margins to genuine scenario predictions as if they were
   validated intervals.** They remain the right tool for held-out historical-regime evaluation
   (which is all U-02 ever claimed), not for scenario-conditional simulation.
2. **The extrapolation-ceiling problem is confirmed as a real, structural issue for this
   project's specific models and data, robust across both well-covered towers and five different
   anchor years** — not a single-anchor fluke and not merely a theoretical concern from the
   literature. This directly motivates the detrend-and-residual or hybrid process+ML approach
   already flagged in this session's literature discussion and sent as a deep-research prompt.
3. **SARIMAX is the highest-risk model for a naive scenario extrapolation** — it never plateaus,
   beats every other model (including both ensembles and TabPFN) in 10/10 cases, and its
   extrapolation magnitude is highly anchor-dependent (59–380% across the 10 usable T4/T9 cases),
   meaning even a "which anchor's coefficients do we use" choice materially changes a 2050
   projection's scale.
4. **The B-10 ensemble (this project's standing production recommendation) is not a safe default
   for scenario extrapolation as-is** — despite being 75% tree-weighted, it shows +49–50% mean
   overshoot, more than double the pure tree-mean, because SARIMAX's 25% weight is not diluted away.
   Any Phase-07 scenario pipeline that reuses B-10's ensemble unmodified inherits this risk; a
   scenario-specific reweighting (or dropping SARIMAX for scenario runs specifically, keeping it for
   historical-regime forecasting where it remains competitive) is worth considering.
5. Any future scenario-uncertainty treatment will need either (a) a model class/architecture that
   extrapolates more defensibly than the extremes seen here, or (b) an explicit, separately
   justified widening of the interval for out-of-range scenario inputs (e.g. inflating the margin
   as a function of distance from the training envelope) — not a bare reuse of U-02's historical
   margins.
6. **Tower 2's livestock density is effectively absent from the data in 4 of 5 recent years** —
   worth knowing independent of this specific diagnostic: any Phase-07 scenario involving Tower 2's
   catchment needs a different baseline assumption for "current" livestock levels, since the
   2019–2022 window this project's models were fit against shows essentially zero recorded grazing
   there.

## Files

- `notebooks/06_interpretability_uq/u03_shift_calibration_analysis.py` (Part A)
- `notebooks/06_interpretability_uq/u03_extrapolation_stress_test.py` (Part B pilot — single
  anchor/tower, 5 models; superseded, kept as the notebook's inline worked example. Its saved
  output file/figure were later overwritten by the full-coverage script's Tower-4/anchor-2021 slice)
- `notebooks/06_interpretability_uq/u03_extrapolation_stress_test_multi.py` (Part B full coverage —
  all 8 models × all 3 towers × 5 anchors, the basis for the "Part B: full coverage" findings above)
- `notebooks/06_interpretability_uq/u03_response_curve_plot.py`,
  `u03_response_curve_plot_multi.py` (Part B visualizations, pilot and full-coverage respectively)
- `notebooks/06_interpretability_uq/U03_uncertainty_shift_robustness.ipynb` (design + full run of
  Part A + the Part B pilot, executed inline; the full-coverage sweep runs as a standalone script
  per this project's established "notebook = bounded example, script = full sweep" convention,
  given its ~35-minute runtime)
- `results/u03_shift_calibration_summary.csv`, `results/u03_shift_vs_picp_per_anchor_tower.csv`
  (Part A)
- `results/u03_extrapolation_stress_test.csv` (Part B pilot, superseded), `results/
  u03_extrapolation_stress_test_multi.csv` (Part B full coverage, 3,600 rows), `results/
  u03_pct_change_summary.csv` (per anchor/tower/model %-change, all 3 towers, the robustness table)
- `results/figures/u03_fancharts/` — 8 highlighted-anchor fan charts (Part A, anchor 2020/Tower 4)
  + 15 per-anchor-tower response-curve plots (all 3 towers × 5 anchors, Part B full coverage) + 1
  robustness summary scatter plot (T4/T9 only)

No `benchmarks.csv` rows (diagnostic/robustness analysis, not a point-forecast or interval-
calibration benchmark — same exclusion precedent as U-01/U-02).
