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
| Gap-filling | External-sourced pooled RFm, full-period gap-CV | R² T2=0.576, T4=0.404, T9=0.426 | D-35, D-49, D-77 | 2026-07-23 |
| Forecasting — point/direct | B-03 enriched RF/XGB, daily track | R² T4 h1=0.365 h14=0.280; T9 h14=0.359 | D-41, D-49 | 2026-07-02 |
| Forecasting — recursive rollout | **TabPFN+species (F-10)** — best single model overall | All-tower: MASE=0.840, R²=−0.084 | D-67 | 2026-07-10 |
| Interpretability | I-02 | `fx_lsu_dens` dominant, importance grows with lead time | D-61 | 2026-07-06 |
| Uncertainty quantification | U-02 (calibration) + U-03 (its limits) | Calibrated PICP ≈0.89; SARIMAX extrapolation outlier +150%/+380% max | D-62, D-63 | 2026-07-07 |
| Scenario analysis (Phase 07) | S-01 level-residual hybrid (proof-of-mechanism) | 3x livestock: +138%/+105% at T4/T9 (vs. trees-alone +21-23%, U-03) | D-64 | 2026-07-07 |

---

## 1. Gap-filling

**Best config:** partial-pooled (T2+T4+T9 + tower-indicator dummies) external-sourced (Site MET)
RFm + stocking-density (LSU/ha) livestock feature + lags + pruned management + gap-filled met/GPP
(REddyProc-style) + per-catchment external soil temperature, evaluated via **full-period gap-CV**
(not a year-split — F-07's fix). Methodology established at F-07/F-08 (D-34/D-35).

**Current best-validated R² (post D-77 `mdc_gapfill` extended-interpolation fix, 2026-07-23):**

| Tower | R² |
|---|---|
| T2 | 0.576 |
| T4 | 0.404 |
| T9 | 0.426 |

All three towers beat MDS by roughly 0.6–1.0 R² units.

**Prior number (post D-48 fix, F-09a/D-49, superseded by D-77 above):** T2=0.574, T4=0.402,
T9=0.418. D-77 rebuilt the gap-filling pipeline as a fully self-contained notebook
(`03c_gap_filling_revisited/`, zero `src/` imports) and found `mdc_gapfill()`'s flat 2h
interpolation cutoff was too short for low-diurnal-structure drivers (soil moisture/temperature,
TA, VPD, WS) — extending it to 288h for those 5 variables only lifted R² at every tower.

**Extensively re-tested against this D-77 base (D-78, 2026-07-23/24) — nothing beat it outright:**
an Area-of-Applicability UQ layer (validated, weak-but-real error correlation, additive to
production); six additional models (LightGBM/XGBoost/TabPFN/TabICL/SAITS/BI-LSTM) — LightGBM and
TabICL edge the champion at Tower 4 only (+0.006/+0.019), everything else loses everywhere; soil-lag
bidirectional/lead-only re-expansion (reproduces F-12's null result on the corrected features); and
target (FCH4) lag/lead features (new, never tested before — a clear regression). See D-78.

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

**Confirmed null result:** F-11 (D-74) tested SAITS (self-attention imputation) as a replacement —
loses to RFm by a wide margin at every tower, **not adopted**. F-12 (D-75) tested adding
forward-looking ("lead") soil lags to RFm's feature set (backward-only vs. bidirectional vs.
leads-only) — a marginal, noise-level gain at T4 only, offset by regressions at T2/T9; **not
adopted**, no change to the numbers below.

**Sources:** `notebooks/04_feature_engineering/F08_external_sensors_RFm.ipynb`, `F08_results.md`,
`results/f08_summary.csv`, `results/f09a_summary.csv`, `F12_results.md`,
`results/f12_summary.csv`, `notebooks/03c_gap_filling_revisited/temp_gap_filing_exploration.ipynb`
(D-77 fix), `temp_gap_filing_exploration copy.ipynb` (D-78 extended exploration),
`notebooks/03c_gap_filling_revisited/_data/model_comparison.csv`,
`notebooks/03c_gap_filling_revisited/_data/soil_lag_results.csv`,
`notebooks/03c_gap_filling_revisited/_data/target_laglead_results.csv`.

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

**Standing best overall, on the observed-target metric specifically (2026-07-10, F-10/D-67):
`TabPFN+species`** — TabPFN (zero-shot, zero training) fed the daily feature matrix plus
`fx_cattle_dens`/`fx_sheep_dens`/`fx_lamb_dens` (livestock disaggregated by species instead of the
single combined `fx_lsu_dens`), all 3 towers, all 5 anchors. **MASE=0.840, R²=−0.084 (observed
target)** — beats every other model/config tested in this entire project on this metric, at
near-zero adoption cost (no retraining, only 3 extra input columns). MASE is this project's
primary forecasting metric (CLAUDE.md convention, added 2026-07-10 — CH4's spike-tail behavior
repeatedly destabilizes R²).

**Important caveat, not a footnote: this ranking is target-dependent, and the choice of target
matters here more than usual.** Scored against `y_gapfilled` instead, the OLD standing
recommendation (`Ensemble_unweighted`, MASE=**0.749** gap-filled) beats this NEW "winner"
(`TabPFN+species`, MASE=**0.944** gap-filled) by a wide margin — the ranking fully flips, not just
narrows. Reason: trees/SARIMAX/ensembles are fit by directly regressing onto `y_gapfilled` as
their training label, so they mechanically track it well (the exact circularity risk already
flagged for this secondary metric, here working in their favor); TabPFN's target is `y_observed`
only, so it gets no equivalent boost. **The observed-target ranking above is still the one to
trust** — it's this project's established primary evaluation convention (D-36/D-37: "train on
gap-filled, evaluate on observed" — `y_observed` is the intended validation target specifically
because it isn't inflated by that circularity) — but "TabPFN+species is the new best" should be
read as "best on the metric this project treats as authoritative," not as an unconditional
statement true under every scoring choice. Full detail:
`notebooks/04_feature_engineering/F10_results.md`, `results/
b16_final_table_vs_gapfilled_best_config.csv`.

**Historical "bests" by different criteria, kept for continuity** — recursive-rollout R² is
inherently much lower than point-forecast R² (a genuinely harder task: 365-day autoregressive
forecasting with no real recent AR data), so these numbers are not directly comparable to Section 2
above; note F-10's finding above now supersedes both rows below on the "best single model" question:

| Criterion | Winner | Metric | Decision |
|---|---|---|---|
| Best ensemble (pre-F-10 model roster, unenriched features) | B-10 (RF+XGB+LightGBM+SARIMAX, unweighted) | R²=0.012, MASE=0.975 (T4-only) / R²=−0.165, MASE=0.918 (all-tower) | D-54, D-65 |
| Best single model, MASE (pre-F-10, unenriched features) | TabPFN (B-13, zero-shot, zero HPO) | MASE=0.862, R²=−0.006 (T4-only) / MASE=0.855, R²=−0.122 (all-tower) | D-57, D-65 |
| Best single model, R² (single-tower, does not transfer) | LightGBM, rollout-tuned (B-15) | R²=0.017 | D-59 |

B-15's tuned LightGBM (`num_leaves=7, min_child_samples=20, learning_rate=0.05`) does **not**
transfer cross-tower (T9 R²=−0.388 reusing T4's config) — a single-tower result, flagged as such.

**F-10 finding, all 11 models tested (D-67): feature enrichment splits sharply by model class.**
Trees/SARIMAX (RF, XGB, LightGBM, SARIMAX, both B-10 ensembles) show no meaningful gain from any of
5 new feature families (livestock-species split, a Tower-2 land-use `fx_is_arable` flag, catchment
flow, fertilizer/management recency, liveweight density) — flat to mildly negative throughout.
**Every attention-based or foundation model (TFT, TabPFN, TabICLv2, and to a lesser extent
LSTM/DLinear) shows real, often large, gains** — TFT's `BASE` alone loses to naive persistence
(MASE=1.063) while `BASE+ALL` beats it (MASE=0.941); TabICLv2's best config (`BASE+ALL`,
MASE=0.871) is a large improvement over its own `BASE` (0.928). `fx_is_arable` shows negligible
effect on every model (it's constant within nearly every per-tower rollout window) — its value is
documentation/interpretability (D-68's Tower-2 land-use reconciliation), not predictive accuracy.

**Full metric set, all 3 towers (D-65 + addendum)** — `bin_metrics()` originally tracked only
R²/MAE/MASE; RMSE, WAPE, and Correlation (Pearson r) were added, and B-10+B-13 rerun for **all 3
towers** (T4 reproduces existing published numbers bit-for-bit; T2/T9 have no prior CSV to check
against but reuse the same verified fit/rollout code). Headline is now the **all-tower pooled**
table (per-anchor n-weighted mean across bins and towers, then mean across anchors) — the
Tower-4-only table is kept below for continuity with earlier citations:

**All-tower (headline):**

| Model | R² | RMSE | MAE | MASE | WAPE | Correlation |
|---|---|---|---|---|---|---|
| RF | −0.241 | 52.23 | 34.84 | 0.968 | 1.050 | 0.375 |
| XGB | −0.184 | 51.57 | 33.81 | 0.922 | 0.991 | 0.368 |
| LightGBM | −0.206 | 52.08 | 34.32 | 0.941 | 1.012 | 0.368 |
| SARIMAX | −0.360 | 53.79 | 36.06 | 0.976 | 1.105 | 0.343 |
| **Ensemble_unweighted** | **−0.165** | 51.57 | 33.75 | **0.918** | 0.998 | **0.375** |
| Ensemble_MASEweighted | −0.165 | 51.57 | 33.74 | 0.918 | 0.998 | 0.375 |
| TFT (one unseeded draw) | −0.363 | 56.59 | 35.62 | 0.972 | 1.045 | 0.292 |
| TabPFN | −0.122 | 56.12 | 33.14 | 0.855 | 0.899 | 0.358 |

**Tower-4-only (original scope, kept for continuity):**

| Model | R² | RMSE | MAE | MASE | WAPE | Correlation |
|---|---|---|---|---|---|---|
| RF | −0.067 | 51.54 | 34.08 | 1.024 | 1.028 | 0.402 |
| XGB | 0.003 | 51.23 | 33.08 | 0.968 | 0.964 | 0.372 |
| LightGBM | −0.014 | 51.37 | 33.25 | 0.978 | 0.978 | 0.384 |
| SARIMAX | −0.039 | 52.39 | 35.23 | 1.038 | 1.047 | 0.379 |
| **Ensemble_unweighted** | **0.012** | **50.96** | 33.18 | 0.975 | 0.977 | 0.396 |
| Ensemble_MASEweighted | 0.011 | 50.96 | 33.17 | 0.975 | 0.977 | 0.396 |
| TFT (one unseeded draw) | −0.228 | 54.48 | 33.43 | 1.014 | 1.020 | 0.315 |
| TabPFN | −0.006 | 54.19 | 30.46 | **0.862** | 0.860 | 0.391 |

**Every model's R² drops once T9/T2 are included** (Ensemble 0.012→−0.165) while MASE holds or
improves slightly (0.975→0.918) — Tower 9 is a consistently harder tower across this whole project
and Tower 2 is largely degenerate outside 2018; model *ranking* is unchanged (Ensemble/TabPFN best
on MASE, SARIMAX/TFT worst throughout), so this is a scope-of-evaluation effect, not a different
conclusion about which model wins. **New finding: TabPFN's best-in-sequence MASE does not extend to
RMSE** — its RMSE is second-worst in both tables, clearly worse than every tree model. RMSE squares
errors and is dominated by TabPFN's worst individual misses, unlike the linear MAE/MASE. TabPFN's
real strength is consistency relative to persistence, not small worst-case errors. **Correlation is
weak-to-moderate everywhere (0.26–0.40)**, including the standing ensemble recommendation — no model
tracks the true pattern strongly by this measure. Does not change the standing recommendation
(Ensemble_unweighted remains best on R² and RMSE in both tables). A tower×year×model breakdown table
is also available. Full detail: `notebooks/05_benchmarking/b10_b13_metrics_rerun.md`.

**Secondary, exploratory metric (not a headline number):** the same rerun also scores every chain
against `y_gapfilled` instead of `y_observed` (explicit, bounded departure from D-36/D-37's
"train on gap-filled, evaluate on observed" convention, real circularity risk — `y_gapfilled` seeds
the AR history and shares features with the forecasters). Unlocks full Tower 2 coverage (816→14,600
of 14,600) but R² gets *worse* while RMSE/MASE improve (a variance-normalization artifact of scoring
against a smoother target, not a contradiction); Ensemble_unweighted stays near the top either way,
but TabPFN drops from best-R² (observed) to 6th of 8 (gap-filled) — the one real ranking
disagreement. See "Secondary metric" section, `notebooks/05_benchmarking/b10_b13_metrics_rerun.md`.

**Model-roster extension (DLinear/LSTM):** closes the gap between the `b10_chains` figures (which
existed for these two models across all 3 towers) and their evaluation metrics (which didn't).
Confirms D-53/D-54's finding at full coverage — both are drastically worse than every model above
(all-tower R²: DLinear −5.06, LSTM −1.36) — correctly excluded from B-10's ensemble. Also produced a
sharper, generalized version of D-62's TFT non-determinism finding: LSTM reproduces bit-for-bit
exactly every time; DLinear only differs on the very first anchor processed in a run — traced to
`torch.manual_seed()` being called *after* model construction, so only the first torch model built
in a process (before any prior seed call) gets non-deterministic weights. See "Model-roster
extension" section, `notebooks/05_benchmarking/b10_b13_metrics_rerun.md`. DLinear/LSTM were also
added to the gap-filled-target secondary-metric table — DLinear's pooled R² there (−6576.7) is
dominated by a genuine numerical divergence in the 2018 anchor's non-deterministic draw (MAE up to
~7,545 nmol m⁻² s⁻¹, physically implausible), amplified by scoring against the low-variance
gap-filled target; excluding that one anchor gives R²=−7.035, still worst but interpretable.

**TabICLv2 (D-66, corrected 2026-07-10):** new tabular foundation model (ICML 2026, "heavily
inspired by TabPFN-TS"), added as a new sibling script (`b10_b13_tabicl_extension.py`), mirroring
TabPFN's per-tower/per-anchor/never-pooled integration. **A real point-estimate bug was found and
fixed** after the user was skeptical of an initial result that looked implausibly worse than
TabPFN's: `tabicl_forecast()` was extracting a **mean**-based point column (`TabICLForecaster`'s
default) rather than the median, badly biased high on this heavy-tailed, spike-dominated flux
distribution. Fixed to use the median (`0.5`) quantile column instead. **Corrected all-tower
R²=−0.329 (observed), −0.886 (gap-filled)** — now beats SARIMAX (−0.360) and TFT (−0.363), and its
MASE (0.928) is 4th-best of all 10 models, comfortably below 1.0. Still behind TabPFN (−0.122) and
the standing recommendation (Ensemble_unweighted, −0.165), so the standing recommendation is
unchanged — but TabICLv2 is a genuinely competitive mid-pack model, not a near-bottom one. Full
3-tower × 5-anchor sweep completes in **~10 seconds total**, still far cheaper than every other
model here — the best accuracy-per-compute-second result in the sequence. Shares TabPFN's exact
Tower-9/2019 degenerate-forecast limitation (zero real `y_observed` in pre-anchor history → flat
~0.0 prediction, unaffected by the point-estimate fix) — a known, already-accepted data-scarcity
issue, not new. See "Model-roster extension: TabICLv2" section,
`notebooks/05_benchmarking/b10_b13_metrics_rerun.md`.

**Standing production recommendation: B-10's ensemble** — but see Section 5, U-03: this ensemble
is **not immune** to extrapolation risk under scenario-style inputs, inherited from its SARIMAX
component.

**D-47 note:** this decision-ID slot is reserved for B-08 (driver-realism sensitivity), which has
not yet run — not a numbering gap/error.

**Sources:** `notebooks/05_benchmarking/B10_daily_improvements.ipynb`, `b10_results.md`,
`B13_tft_tabpfn.ipynb`, `b13_results.md`, `B15_rollout_tuning.ipynb`, `b15_results.md`,
`b10_b13_rerun_multi_anchor.py`, `b10_b13_metrics_rerun.md`,
`results/b10_b13_rerun_table_all_towers.csv`, `results/b10_b13_rerun_table_by_tower_year.csv`.

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

**Current: S-01 (D-64)** — first bounded worked example, proving the scenario-simulation mechanism
end-to-end. **B-08 confirmed superseded for Phase 07's purposes by U-03** (its extrapolation-check
finding already answers what B-08 would have; B-08 remains available separately for the
point-forecast track).

**Architecture: level-residual hybrid.** A Ridge trend model (`fx_TA_mean`, `fx_lsu_dens`,
`fx_DOY_sin/cos` + tower dummies), fit **once** on the full pooled real record (not per-anchor —
the fix for U-03's SARIMAX-instability finding), carries the climate+livestock extrapolation.
RF/XGB/LightGBM (B-10's exact hyperparameters, no new HPO; monotonic constraint on `fx_lsu_dens`
for XGB/LightGBM) correct only the residual. `fx_USTAR_mean`/`fx_SHF_mean` dropped entirely (no
climate-scenario-product source at all). Climate drivers from the North Wyke CMIP6 files
(`data/Simulated Climate Data/`, D-52) — 4 of ~11 variables direct/derived (TA min/max/mean,
precip, radiation with a verified MJ/m²/day → W/m² unit conversion); the other ~9 historical-day-
resampled via `rr.doy_climatology()` (D-52's decision).

**Scenario tested:** SSP2-4.5, ensemble-mean (5 GCMs × 100 realizations), 2041–2060 ("the 2050s"),
**all 3 towers**, livestock multipliers {1×, 2×, 3×} on the day-of-year climatology of
`fx_lsu_dens`.

**Result — the hybrid measurably fixes U-03's flattening finding:**

| Tower | 3× annual-mean % change | AOA flagged at 3× | Own-tower `fx_lsu_dens` max (training) |
|---|---|---|---|
| T2 | +33.8% | 0.0% (never flagged, any multiplier) | 0.71 |
| T4 | **+138.2%** | 5.5% | 4.99 |
| T9 | **+104.7%** | 6.0% | 5.65 |

(U-03's trees-alone comparison at the same 3× sweep: only +21–23%.)

**Two genuine, verified findings:**
1. A "2×" scenario built by scaling a *smoothed climatology* is meaningfully milder than one built
   by scaling raw daily values (U-03's method) — only 3× exceeds the training envelope at T4/T9;
   Tower 2 never does, consistent with U-03's own finding that T2's livestock signal is near-absent.
2. A full monotonic sweep shows XGB/LightGBM's residual correction is **completely flat** with
   respect to livestock density (shallow B-10 hyperparameters + monotonic constraint + trend
   already absorbing the primary signal) — for 2 of 3 tree models, ~100% of the scenario response
   flows through the trend, not the residual. Only RF shows real residual sensitivity.

**Explicitly a proof-of-mechanism, not a final output.** Caveats: parametric (not mechanistic —
SPACSYS logged as future work) trend; 9/11 drivers historical-day-resampled, not real future
weather; naive livestock multiplier (not a self-consistent management timeline); U-02/U-03's
conformal intervals NOT attached (only ever valid for in-AOA points per U-03).

**Queued next:** extend S-01 — SSP5-8.5, realization-level (not just ensemble-mean) spread, a
self-consistent mechanistic livestock-scenario construction, and (if time permits) SPACSYS
(already validated at North Wyke, Wu et al. 2016) for the trend/level component.

**Related diagnostic, not a production-config change (D-70)**: S-03 isolated the cost of
scenario-mode driver unavailability from extrapolation, on real historical anchors — found the
cost is small-to-negative pooled across towers (neither dropping nor climatology-resampling 24
scenario-unavailable columns beat Model 1's full feature set; resample modestly beats it). Supports
the standing recommendation that scenario risk concentrates in extrapolation/SARIMAX, not driver
loss. **Extended (D-70 addendum, 2026-07-15) to the full 11-model roster** (originally B-10's
4-model architecture only, a real scope gap fixed after direct user challenge) — the small/beneficial
finding **holds for the production ensemble and most models, but not uniformly**: TFT's R² gets
measurably worse under the resample variant (-0.363→-0.492), the one model where this diagnostic's
reassurance does not transfer. Still not a production-config change (B-10's ensemble / TabPFN+species
remain the standing recommendations). See D-70 addendum, `notebooks/07_scenario_analysis/s03_results.md`.

**Methodology check, not a production-config change (D-71, 2026-07-15)**: user asked whether
chain-persistence (MASE's denominator throughout this project, D-37) is a valid baseline for a
seasonal series, given Hyndman & Koehler's own convention recommends seasonal-naive scaling
instead. Extended B-09's `doy_climatology()` baseline (previously single tower/anchor only) to full
3-tower × 5-anchor coverage and rescored all 11 B-10/B-13 models' MASE against it. **Result reverses
the motivating hypothesis**: pooled, climatology is the *weaker* baseline (own MAE 43.79 vs.
persistence's 37.50 against real `y_true`), likely because FCH₄'s spike-dominated record makes a
±7-day day-of-year average over sparse real history a noisy estimate, not a stable seasonal curve.
Reinforces keeping persistence as the primary MASE denominator going forward — not merely for
cross-table consistency, but because the available seasonal alternative isn't empirically more
reliable here. Climatology-scaled MASE kept as a secondary comparison column only. See
`results/b10_b13_climatology_mase_table_all_towers.csv`/`_by_tower.csv`.
**Follow-up (same day): fairness fix.** Original comparison used real `y_observed` for climatology
vs. `y_gapfilled` for persistence's anchor value — not apples-to-apples. A gap-filled-basis
climatology variant (`Climatology_gf`) narrows the gap (pooled MAE 40.74 vs. persistence's 37.50,
vs. the original 43.79) and reverses at Tower 2 (climatology-gf wins there). Conclusion unchanged
pooled, but tower-dependent. See `results/b10_b13_climatology_gf_mase_table_all_towers.csv`.

**Sources:** `notebooks/07_scenario_analysis/S01_first_scenario.ipynb`, `s01_results.md`,
`src/features/build_scenario_drivers.py`, `src/models/scenario_hybrid.py`,
`results/s01_scenario_summary.csv`, `results/figures/s01_*.png` (4 figures),
`results/models/s01_*.joblib` (frozen artifacts).
