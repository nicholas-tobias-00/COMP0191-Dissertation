# BEST_RESULTS.md

**Purpose:** the fast-reference index of the current best-validated result per project phase.
`CONTEXT.md` remains the narrative/status log and `DECISIONS.md` remains the full decision
history — this file is a scannable pointer into both, not a replacement for either. **Any new
experiment should be compared against the numbers here before being reported as an improvement.**

**Style rule (keep this file honest to its own purpose):** tables and one-line caveats only, no
multi-paragraph prose. If a section starts growing narrative, cut it back and push the detail to
`DECISIONS.md` via the cited decision ID instead.

**How to update:** part of the standard end-of-session checklist (`CLAUDE.md`) — whenever a new
experiment beats a phase's recorded best, update that phase's section, update the quick-reference
row, and bump "last verified."

---

## Quick reference

| Phase | Best config | Headline metric | Decision ID | Last verified |
|---|---|---|---|---|
| Gap-filling | External-sourced pooled RFm, full-period gap-CV | R² T2=0.574, T4=0.402, T9=0.418 | D-35, D-49 (F-09a) | 2026-07-02 |
| Forecasting — point/direct | B-03 enriched RF/XGB, daily track | R² T4 h1=0.365 h14=0.280; T9 h14=0.359 | D-41, D-49 | 2026-07-02 |
| Forecasting — recursive rollout | B-10 ensemble (best ensemble) | R²=0.012, MASE=0.975 | D-54 | 2026-07-06 |
| Interpretability | I-02 | `fx_lsu_dens` dominant, importance grows with lead time | D-61 | 2026-07-06 |
| Uncertainty quantification | U-02 (calibration) + U-03 (its limits) | Calibrated PICP ≈0.89; SARIMAX extrapolation outlier +150%/+380% max | D-62, D-63 | 2026-07-07 |
| Scenario analysis (Phase 07) | Not started | — | — (queued: D-47 slot / B-08) | 2026-07-07 |

---

## 1. Gap-filling

**Best config:** partial-pooled (T2+T4+T9 + tower-indicator dummies) external-sourced (Site MET)
RFm + stocking-density (LSU/ha) livestock feature + lags + pruned management + gap-filled met/GPP
(REddyProc-style) + per-catchment external soil temperature, evaluated via **full-period gap-CV**
(not a year-split — F-07's fix). Methodology established at F-07/F-08 (D-34/D-35).

**Current best-validated R² (post D-48 USTAR/VPD-plausibility fix, re-verified at F-09a/D-49,
2026-07-02):**

| Tower | R² |
|---|---|
| T2 | 0.574 |
| T4 | 0.402 |
| T9 | 0.418 |

All three towers beat MDS by roughly 0.6–1.0 R² units.

**Known discrepancy (stated, not resolved):** F-08's own headline write-up gives slightly
different numbers for the same config (T4 0.362, T9 0.350) than F-09a's "before" comparison figures
for that same config (T4 0.376, T9 0.364) — a ~0.01–0.014 gap likely from an averaging-methodology
difference between the two write-ups, never reconciled in the source docs. Use the F-09a numbers
above regardless — both readings are superseded by them.

**Confirmed null result:** F-09b (D-51) tested outlier-correction techniques (hard truncation vs.
winsorization vs. Hampel filter) for a WS/TA contamination issue at Tower 2. Truncation is the most
complete fix among the three, but downstream gap-filling R² was statistically indistinguishable
across every config tested, including uncorrected — **not adopted**, and not urgent regardless
since production's external-sourced pipeline never had this contamination to begin with.

**Sources:** `notebooks/04_feature_engineering/F08_external_sensors_RFm.ipynb`, `F08_results.md`,
`results/f08_summary.csv`, `results/f09a_summary.csv`.

---

## 2. Forecasting — point/direct

**Best config:** B-03 — enriched-feature RF/XGB on the daily track (`forecast_daily_v2.csv`),
productionised `NWFP_T9_Dataset_Structure.md` features + Round-1 HPO (D-41).

**Current numbers (post D-48 fix, re-verified D-49, 2026-07-02):**

| Tower | Horizon | R² |
|---|---|---|
| T4 (RF) | h=1 | 0.365 |
| T4 (RF) | h=14 | 0.280 |
| T9 (RF) | h=14 | 0.359 |

Original D-41 headline (pre-D-48 fix, historical reference only): daily best R² T4 0.362, T9 0.388.

**Competitive alternative, not adopted as production:** B-03a (SARIMAX), post-D-48-fix — mean R²
(T4/T9) 0.416 at h=1 declining to 0.284 at h=14, MASE<1 from h=3 onward.

**Sources:** `notebooks/05_benchmarking/B03_enriched_ML.ipynb`, `b03_b04_results.md`.

---

## 3. Forecasting — recursive rollout

Three distinct "bests" by different criteria — recursive-rollout R² is inherently much lower than
point-forecast R² (a genuinely harder task: 365-day autoregressive forecasting with no real recent
AR data), so these numbers are not directly comparable to Section 2 above.

| Criterion | Winner | Metric | Decision |
|---|---|---|---|
| Best ensemble | B-10 (RF+XGB+LightGBM+SARIMAX, unweighted) | R²=0.012, MASE=0.975 | D-54 |
| Best single model, MASE | TabPFN (B-13, zero-shot, zero HPO) | MASE=0.862, R²=−0.006 | D-57 |
| Best single model, R² | LightGBM, rollout-tuned (B-15) | R²=0.017 | D-59 |

B-15's tuned LightGBM (`num_leaves=7, min_child_samples=20, learning_rate=0.05`) does **not**
transfer cross-tower (T9 R²=−0.388 reusing T4's config) — a single-tower result, flagged as such.

**Standing production recommendation: B-10's ensemble** — but see Section 5, U-03: this ensemble
is **not immune** to extrapolation risk under scenario-style inputs, inherited from its SARIMAX
component.

**D-47 note:** this decision-ID slot is reserved for B-08 (driver-realism sensitivity), which has
not yet run — not a numbering gap/error.

**Sources:** `notebooks/05_benchmarking/B10_daily_improvements.ipynb`, `b10_results.md`,
`B13_tft_tabpfn.ipynb`, `b13_results.md`, `B15_rollout_tuning.ipynb`, `b15_results.md`.

---

## 4. Interpretability

**Current:** I-02 (D-61). I-01 (D-39) targeted a different, earlier, superseded harness and is not
used as precedent.

**Headline finding:** `fx_lsu_dens` (livestock density) confirmed as the dominant driver by every
method that can see it (RF/XGB native importance, RF/XGB/LightGBM SHAP, SARIMAX coefficients at
all 3 towers, TabPFN permutation importance at T4/T9). Its SHAP importance **grows** with lead
time, not shrinks — mean |SHAP| at Tower 4: 7.6 (bin 1–7 days) → 38.4 (bin 181–270 days).

**Sources:** `notebooks/06_interpretability_uq/I02_feature_importance_rollout.ipynb`,
`I02_results.md`.

---

## 5. Uncertainty quantification

**Current:** U-02 (D-62, builds the calibration) + U-03 (D-63, stress-tests it). U-01 (D-40)
superseded, different harness.

**U-02 — calibration works, in-distribution:** leave-one-anchor-out conformal calibration
converges every model to **~0.88–0.90 PICP** at T4/T9, regardless of raw coverage. Raw PICP: trees
(RF/XGB/LightGBM) 0.35–0.50 (badly overconfident); SARIMAX/TabPFN/TFT 0.72–0.92 (already
reasonable raw). Calibrated Ensemble/RF are sharpest (lowest pinball). **Tower 2 cannot support
calibration at all** (no usable margins — real `y_observed` in only 1/5 anchor windows).

**U-03 — calibration's real limits, out-of-distribution (final, full 8-model × 3-tower × 5-anchor
coverage):**
- Part A (real historical shift): no evidence conformal PICP degrades within the shift range
  actually observed 2018–2022 — reassuring but bounded; this is far smaller than a genuine future
  scenario shift, so it does not certify calibration survives real scenario extrapolation.
- Part B (synthetic `fx_lsu_dens` 1.0×→3.0× stress test, Towers 4/9, 10 usable cases; Tower 2
  structurally degenerate — `fx_lsu_dens`=0.0 throughout the rollout window in 4/5 anchors):

| Model group | Mean % change (1.0×→3.0×) |
|---|---|
| RF/XGB/LightGBM (trees) | +21–23% (plateau — tree-extrapolation ceiling) |
| TFT | +26% |
| TabPFN | +30% (range −4.9% to +90.1% — least predictable, sometimes inverts) |
| **Both ensembles** | **+49–50%** (production B-10 ensemble NOT immune) |
| **SARIMAX** | **+150% mean, up to +380%** (max of all 8 models in 10/10 cases) |

**Standing recommendation:** do not reuse U-02's conformal margins as validated intervals for
genuine scenario predictions; do not reuse B-10's ensemble unmodified for scenario extrapolation
without addressing its SARIMAX-inherited risk (reweight, or drop SARIMAX for scenario runs
specifically while keeping it for historical-regime forecasting).

**Sources:** `notebooks/06_interpretability_uq/U02_uncertainty_rollout.ipynb`, `U02_results.md`,
`U03_uncertainty_shift_robustness.ipynb`, `U03_results.md`.

---

## 6. Scenario analysis (Phase 07)

**Status: not started** (`07_scenario_analysis/` is PLANNED in the repo layout).

**Queued next:** (1) B-08 — driver-realism sensitivity (D-47 slot, no longer blocked per D-60); (2)
Phase 07 scenario analysis itself, informed by the CMIP6 candidate dataset (Semenov et al. 2025),
B-10's ensemble as the AR-history strategy (**with the U-03 caveat that its SARIMAX component
carries real extrapolation risk**), and I-02/U-02/U-03's importance/uncertainty findings.

**Discussion, not yet a logged decision:** neither B-03 (Section 2) nor B-10 (Section 3) is
directly reusable for long-horizon scenario work as-is — B-03 needs real AR-history that won't
exist for 2050; B-10's literal 365-day recursive-rollout mechanism has no validated precedent past
one year. Current leaning: reuse B-10's tree ensemble (SARIMAX dropped or reweighted) plus
`doy_climatology()` (`src/models/recursive_rollout.py`) as a fixed AR baseline, evaluated as
scenario-conditional snapshot queries rather than a continuous multi-decade rollout — to be
formalized as a real decision once B-08/Phase-07 actually starts.
