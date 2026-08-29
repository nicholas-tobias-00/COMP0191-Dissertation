# Chapter 6: Scenario Projection — Finalized Outline

<!-- _Working reference, not chapter prose. Title deliberately excludes
"Interpretability" — same relationship as UQ inside Chapter 5: it motivates the
chapter's central lever (livestock density), it isn't a co-equal topic._ -->

<!-- _**All figures/numbers in this file are illustrative only** — captured to show
the shape of a finding, not to be trusted as final. Will be independently
validated before anything goes into the actual report._ -->

## Opening framing (before any section content)

  scenario projection is a structurally different from every prior chapter.
  Everything through Chapter 5 is checkable against real observations. This
  chapter isn't — there is no ground truth, ever, regardless of what
  gets built.

## 6.1 Outline of what Scenario Projection does

## 6.2 Interpretability and Feature Importance

- Methodology: permutation importance implemented on TabICLv2's output.
- Function in the chapter: motivates why the scenario architecture is built
  around livestock density specifically, not an unmotivated design choice.
- **The old SHAP-based lead-time-growth figure (7.6→38.4 across lead time) is
  disregarded, not reused here.** Different method, different model, different
  chapter's work. Any lead-time or magnitude findings for this section come
  fresh from the TabICLv2 permutation-importance run, not that number.

## 6.3 Simulated Climate Dataset

- Semenov et al. 2025, CMIP6-based, SSP scenarios for the North Wyke location.
- Data description.

<!-- ## 6.4 Extrapolation Problem — Scene-Setting (deprecated, brief)

- **S01/S04 (Ridge trend + RF/XGB/LightGBM residual, monotonic constraint) —
  not the headline, deprecated but sound**, kept only as brief scene-setting
  for why extrapolation is a known hard problem here.
- **Explicit caveat required if kept:** the model-*behavior* finding (trees
  plateau under distribution shift; a structural trend/residual split
  extrapolates instead) is likely robust regardless of magnitude. The specific
  percentages (+138.2%/+104.7%/+33.8%, S01; +38.6%/+156.4%/+120.3% vs.
  +20.4%/+76.6%/+62.0%, S04) were generated under the **now-superseded naive
  1×/2×/3× multiplier scheme** — cite the finding, not the numbers, or state
  plainly that the numbers predate the livestock-ladder redesign. -->

## 6.4 Data Pre-processing and Model Adjustment

### 6.4.1 Standardisation of New Data with NWFP dataset
- Delta/anomaly bias-correction method (Hay et al. 2000) — CMIP6 series
  corrected against NWFP's own historical baseline (TA_min, TA_max, etc.).
- **Finding:** CMIP6's historical values don't fully match NWFP's
  own observed climate.

### 6.4.2 Generation of the S06 Model (headline)
- TabICLv2, `TabICLForecaster`, one-shot in-context prediction — no
  trend/residual split, no fitting step on this dataset.
- 13-feature `FX_A_SPECIES` set (fx_TA_mean/min/max, fx_SWIN_mean,
  fx_PRECIP_sum, fx_DOY_sin/cos, fx_is_growing, fx_is_winter, fx_lsu_dens,
  fx_cattle/sheep/lamb_dens) vs. the deprecated hybrid's ~42.
- **S05 gets a passing mention only** — "S06 adds the historical
  bias-correction step (6.5.1) on top of this" — not its own section. S06 is
  the version presented as the chapter's result throughout.
- Note: TabICLv2 is NOT the dissertation's forecasting champion (that's still
  TabPFN+species/bodyweight, Ch5) — it was selected for this simpler
  architecture's specific requirements, not because it wins on MASE.

### 6.4.3 Result of the simplified model
- Modest, quantifiable cost of simplification: pooled MASE 0.740516 (full
  TabICLv2 BASE+species) → 0.758792 (less-informed FX_A_SPECIES), +0.018275
  (≈2.47% worse). Small, reportable, not catastrophic.

## 6.5 Inference with Simulated Climate Dataset

### 6.5.1 Base Case

### 6.5.2 Induced Perturbations
- Livestock ladder: Baseline (1×), Half (0.5×), Literature ceiling (3.0
  LSU/ha), Own historical max — **confirmed running on the redesigned,
  externally-anchored construction**, not the deprecated naive multiplier.
- Worth confirming (not yet verified from the chart alone, only from stated
  values): at T2, literature ceiling should exceed own historical max (2.13 <
  3.0); at T4/T9, own historical max should exceed literature ceiling (4.99/
  5.65 > 3.0) — ties directly to Ch2's "intensive rotational-grazing platform"
  framing if confirmed against raw values rather than read off plot colours.
- Fertiliser/other perturbations as applicable.
<!-- 
---

## Limitations (for this chapter specifically, feeds into Ch7/Ch8)

- **TabICLv2's extrapolation behavior has not been formally stress-tested to
  U-03's standard** — confirmed independently by two separate audits. Not
  included in U-03's model roster; S05's out-of-range livestock exposure was a
  partial behavioural probe only (no ground truth, confounded variables, not
  run under U-03's controlled protocol). TabICLv2 belongs to a model family
  (TabPFN-family, in-context learning on synthetic priors) with a documented
  structural extrapolation failure mode in the project's own cited literature
  (Hoo et al. 2025). This is now a limitation of the **headline result**, not
  a secondary footnote — S06 carries the dissertation's primary scenario
  finding.
- **No existing diagnostic can detect a smooth-but-wrong extrapolation.**
  Structural, not a tooling gap: historical PICP measures anchor calibration,
  not scenario-shifted coverage; scenario PICP can't be computed at all (no
  future y_true); MPIW can stay narrow while confidently wrong; the >365-day
  conformal margin is held flat by construction (explicitly understates
  long-horizon uncertainty); livestock-tier widening reflects historical
  heteroscedasticity, not out-of-support validity; GCM/realization spread
  measures climate-input sensitivity, not model-form error.
- **CMIP6-vs-NWFP historical mismatch** (6.5.1) — a real, separate limitation
  on top of the extrapolation-testing gap, independent of model choice.

---

## Repo fixes surfaced during this discussion (not outline items — actual
## corrections needed in committed dissertation text, separate from this file)

- **AOA is still present in S05 code and the current `report/7 Scenario
  Analysis.tex`**, despite the "cut everywhere" decision made in this
  conversation. The decision hasn't been applied yet — only agreed here.
- **The current Ch7 text's ">99% coverage" claim is wrong, not just
  optimistic.** `s05_uq_cqr_apply.py:138` computes proportion of rows with a
  non-null interval, not actual PICP. Needs correcting regardless of any other
  decision in this chapter.
- **Stale R² figure:** Ch1/Ch5's stated TabPFN+species R² (−0.084) doesn't
  match the current saved table (−0.038213). Confirmed via direct audit —
  needs fixing in `report/1 Introduction.tex:102` and `report/5
  Forecasting.tex:180`. (Not present in this project's own compiled outline
  files — only in the actual `.tex` source.) -->
