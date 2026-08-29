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
| Gap-filling | External-sourced pooled RFm, full-period gap-CV (production); TabICL-solo edges it at T2/T4 (benchmark-only, D-79) | R² T2=0.576, T4=0.404, T9=0.426 | D-35, D-49, D-77, D-79 | 2026-08-02 |
| Forecasting — point/direct | B-03 enriched RF/XGB, daily track | R² T4 h1=0.365 h14=0.280; T9 h14=0.359 | D-41, D-49 | 2026-07-02 |
| Forecasting — 365-day fixed-origin/rollout | **B18 equal mean of three direct TabPFN forecasts** — exploratory numerical best; p95-corrected TabPFN is the best single/gated forecast | All-tower observed target: MASE=**0.6908**, R²=0.192 (single/gated: MASE=0.6924, R²=0.205) | D-106 | 2026-08-19 |
| Interpretability | I-02 (production); **I-03b (D-108) recalibrates for the actual B18 champion** | `fx_lsu_dens` dominant, importance grows with lead time; I-03b: still #1 (1.746, up from 1.1456), Tower 2 livestock-blind 5th confirmation | D-61, D-108 | 2026-08-20 |
| Uncertainty quantification | U-02+U-03 + U-04 (champion recalib.) + U-05 (scenario UQ) + U-06 (CQR spike fix) + U-07 (LSU-stratified CQR); **U-08/U-05b-07b (D-108) recalibrate for B18** | U-06/07: spike coverage ~24%→~80%, low-LSU 26-59% of high-LSU width; U-08/U05b-07b: same pattern replicates under B18 (U05b spike coverage → 90.1%) | D-62, D-63, D-88–D-92, D-108 | 2026-08-20 |
| Scenario analysis (Phase 07) | S-06 (bias-corrected drivers) is the report-facing architecture; **S-06b (D-108) replicates it on the B18-derived `Direct_TabICLv2` solo+trend architecture** | Every S-06 headline finding replicates under S-06b: T9 lit_ceil still < baseline (-2.6% vs old -2.3%), grazing +4wk T4/T9 +19.8%/+19.9% (vs S-06 old +18.5%/+18.6%), fertilizer still null, cattle dominance reconfirmed | D-70, D-84–D-86, D-100, D-104, D-108 | 2026-08-20 |

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

All three towers beat MDS by roughly 0.5–0.6 R² units under the literature-correct MDS baseline
(D-79; was ~0.6–1.0 under the old, buggy MDS implementation — see below).

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

**A separate, parallel notebook (`temp_gap_filling_pipeline.ipynb`, D-79, 2026-08-02) reproduces
this same champion (0.576/0.404/0.426) and found the first result to genuinely beat it at more
than one tower:**

| Model | T2 | T4 | T9 | Note |
|---|---|---|---|---|
| **RFm pooled (production-adopted)** | 0.576 | 0.404 | **0.426** | standing production config — full UQ/production-fill tooling built around it |
| **TabICL-solo, champion FEATURES** | **0.676** | **0.428** | 0.423 | beats RFm at T2 (+0.100) and T4 (+0.024); within noise at T9 (-0.003) — **benchmark-best, not yet production-adopted** (no UQ/production-fill tooling exists for this config yet; solo-not-pooled is TabICL-specific, since its fixed 10,000-row context cap makes pooling actively harmful, unlike RF) |
| HyperImpute (AutoML per-column imputer, same features as MICE) | 0.509 | 0.336 | 0.354 | far ahead of MICE (0.081/0.118/0.107) on identical inputs, still behind both RF and TabICL |

Also tested (D-79), all additive on top of RFm/TabICL, none adopted: dimensionality-reduction
feature *selection* (clean, stable, but downstream KNN features are a wash), TICA-components/
model-uncertainty fed back as features (uncertainty is actively harmful, esp. for TabICL),
TabICL feature-drop/TICA-swap (flat), and TabICL row-cap bagging (small real gain at Tower 4 only,
+0.012). See D-79, `notebooks/03c_gap_filling_revisited/summary.md` §12-18.

**Confirmed null result, D-99 (2026-08-17): confidence-gated self-training second pass for
TabICL** (user's own idea — promote points TabICL predicts within a tight interval width into the
training pool as pseudo-labels, refit, re-predict the rest). Full-set R² change vs. plain
TabICL-solo: T2 -0.006 to **-0.017** (worsens as the promoted band widens), T4 +0.002 to +0.005
then reverses to -0.005 at the widest band, T9 noise. **The pre-registered risk was confirmed, not
just theorized**: promoted (narrow-interval) points skew non-spike, so the enriched pool measurably
hurts spike-stratum accuracy at T2 and T9 (T9 down to -0.053 at the widest band). **Not adopted.**
See D-99, `notebooks/03c_gap_filling_revisited/summary.md` §19.

**Metric-completeness fill, D-103 (2026-08-19): R2_OLS (scipy `linregress`/Zhu et al. 2023a
convention, bounded [0,1]) recalculated for D-78's five non-TabPFN challenger models** (LightGBM/
XGBoost/TabICL/SAITS/BI-LSTM), extending §13.5's own RF/TabICL/MDS/MICE R2_OLS comparison to this
table for the first time — no raw prediction pairs existed for these models before, so this
required a genuine (additive-only) refit, not a free recompute. **Ranking unchanged**: every
model's R2_OLS sits above its sklearn R² (same direction as every other model in this project), but
LightGBM/TabICL remain the only two that edge the champion, both only at Tower 4 (R2_OLS 0.418/0.425
vs. champion 0.404); SAITS/BI-LSTM stay behind everywhere under either metric. LightGBM/XGBoost/
TabICL/BI-LSTM all verified bit-exact against the original sklearn R² numbers; SAITS showed a small
(~0.02 R²) mismatch from neural-net training non-determinism (NumPy global-RNG-dependent validation
masking), not a bug. **No champion change.** See D-103,
`notebooks/03c_gap_filling_revisited/_data/d100_ols_recalc_summary.csv`.

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

## 3. Forecasting — 365-day fixed-origin / recursive rollout

**Current long-horizon numerical best (B18/D-106, 2026-08-19; all 3 towers × 5 anchors):**

| Config | Status | MASE | MAE | RMSE | R² |
|---|---|---:|---:|---:|---:|
| **Equal mean: p95-corrected + 1,095-day raw + 1,460-day tower-robust direct TabPFN** | Exploratory numerical champion | **0.6908** | **29.821** | 60.019 | 0.192 |
| p95 TabPFN + 25% predicted spike excess | Best single/gated B18 forecast | 0.6924 | 29.857 | **59.530** | **0.205** |
| Direct pooled TabPFN, 1,095-day history | Best recency-only variant | 0.6930 | 29.976 | 60.323 | 0.184 |
| Direct pooled TabPFN, all history (B17) | Direct-architecture control | 0.6958 | 30.073 | 60.708 | 0.173 |
| B16-style TabPFN-TS v2, `BASE+ALL` | Same-protocol B16 comparator | 0.7123 | 30.669 | 61.704 | 0.146 |
| B16 genuine-species TabPFN-TS v3 | Prior `BASE+species` comparator | 0.7154 | 30.890 | 62.228 | 0.131 |

| B16→B18 attribution | MASE change | Relative change |
|---|---:|---:|
| TabPFN-TS wrapper → direct pooled TabPFN (B16→B17) | −0.0165 | −2.31% |
| Direct all-history → final B18 ensemble (B17→B18) | −0.0050 | −0.72% |
| **Total: B16-style v2 `BASE+ALL` → final B18** | **−0.0215** | **−3.01%** |

One-line caveat: the fixed equal-weight ensemble is the lowest observed score, but its extra gain is
not independently stable under block-wise weight/model selection; retain 0.6924 as the clearest
single/gated result. Spike magnitude remains the limiting regime (pre-anchor p95 MASE=3.285 vs.
0.524 non-spike). See D-106 and `report/Outlines/B18_forecasting_experiment_results.md`.

**Integration status:** B18 updates the forecasting benchmark only. I-03/U-04 and Phase-07 scenario
work still use the prior B16-style champion architecture and are not silently re-labelled as B18.

**MASE baseline changed 2026-08-02 (D-80): climatology, not persistence — see CLAUDE.md.** The B18
table above and prior-B16 headline below use climatology. Most older tables retain
`chain_persistence()` (D-37's original choice) and were not retroactively rewritten.
D-80's full climatology recompute of the 55 then-existing configurations left their internal
ranking unchanged (`results/b09_b16_climatology_mase_full_table.csv`). Within that historical B16
comparison, `TabPFN+bodyweight` effectively tied `TabPFN+species` (0.715 vs. 0.715), ahead of the
next cluster (`TabPFN_gf` variants, 0.716–0.722).

**Prior B16-style champion, on the observed-target metric specifically (2026-07-10, F-10/D-67;
rescored under climatology 2026-08-02, D-80; superseded numerically by D-106): `TabPFN+species`** — TabPFN (zero-shot, zero
training) fed the daily feature matrix plus `fx_cattle_dens`/`fx_sheep_dens`/`fx_lamb_dens`
(livestock disaggregated by species instead of the single combined `fx_lsu_dens`), all 3 towers,
all 5 anchors. **MASE≈0.715 (climatology-scored), R²=−0.084 (observed target)** — beats every
other model/config tested before B17/B18 on this metric, at near-zero adoption cost (no
retraining, only 3 extra input columns). MASE is this project's primary forecasting metric
(CLAUDE.md convention, added 2026-07-10 — CH4's spike-tail behavior repeatedly destabilizes R²).

**Follow-up tested and found negative (D-93, 2026-08-10): TICA embeddings + static AR-lag features
on top of `BASE+species`, TabICLv2 only.** Both individually land within noise of the baseline
(TICA: 0.7348 vs. 0.7353; static AR: 0.7358 vs. 0.7353) — TICA replicates D-79's own gap-filling
"wash" finding in a different task; static AR features add little since TabICLv2 already sees raw
`y_observed` history natively. **Combining both is clearly worse** (0.7603, +0.025), consistent
across all 3 towers — matches this project's recurring "stacking too many feature families hurts"
pattern (D-67's `BASE+ALL` finding). `BASE+species` remains unbeaten within that B16/TabICL
sub-experiment. Full detail:
`notebooks/05_benchmarking/B16_tica_static_ar_features.ipynb`.

**Follow-up tested (D-94, 2026-08-10): does pooling (Track A's F-02/F-03 recipe) help TabPFN/
TabICLv2 on forecasting specifically, not just gap-filling?** Never tested before with the
champion's own architecture — the only prior pooled-vs-solo precedent (D-79) used a different API
(sklearn-style `TabICLRegressor`, not the TS-native `TabICLForecaster`) on a different task
(gap-filling). **Splits by model**: TabICLv2 pooled ≈ solo (0.7355 vs. 0.7353, noise) — direction
consistent with D-79's "TabICL prefers solo" finding but far smaller in magnitude. **TabPFN pooled
beats solo at all 3 towers** (0.7138 vs. 0.7166 overall; T4's R² crosses from slightly negative to
slightly positive) — small but real and consistent, and never tested by D-79 at all (TabPFN wasn't
part of that comparison). At the time this was not large enough to force an immediate champion
switch; D-106 subsequently validates direct pooled regression as the main B16→B18 gain. Full detail:
`notebooks/05_benchmarking/B16_pooled_vs_solo.ipynb`.

**Important caveat, not a footnote: this ranking is target-dependent, and the choice of target
matters here more than usual.** Scored against `y_gapfilled` instead, the OLD standing
recommendation (`Ensemble_unweighted`) beats this NEW "winner" (`TabPFN+species`) by a wide
margin — the ranking fully flips, not just narrows. Reason: trees/SARIMAX/ensembles are fit by
directly regressing onto `y_gapfilled` as their training label, so they mechanically track it well
(the exact circularity risk already flagged for this secondary metric, here working in their
favor); TabPFN's target is `y_observed` only, so it gets no equivalent boost. **The observed-target
ranking above is still the one to trust** — it's this project's established primary evaluation
convention (D-36/D-37: "train on gap-filled, evaluate on observed" — `y_observed` is the intended
validation target specifically because it isn't inflated by that circularity) — but "TabPFN+species
is the new best" should be read as "best on the metric this project treats as authoritative," not
as an unconditional statement true under every scoring choice.

**Correction (D-96, 2026-08-13):** the gap-filled-target numbers originally quoted here
(`Ensemble_unweighted`=0.749, `TabPFN+species`=0.944) were computed against chain-persistence, not
climatology — that pipeline was never rescored under D-80's climatology switch, so it was silently
mixing two variables (target *and* baseline convention) instead of isolating target alone. Caught
via direct user question, fixed with a genuinely fair `Climatology_gf` baseline (climatology built
from `y_gapfilled` history, scored against `y_gapfilled` truth — D-71's "same-series" principle,
never previously applied to this comparison). **Corrected gap-filled-target MASE: `TabPFN+species`
0.906, `Ensemble_unweighted+species` 0.665** — the flip survives and the margin is similar, so the
conclusion above is unchanged, but the exact figures were wrong. Full 11-model corrected table
tops out at `Ensemble_MASEweighted`/`Ensemble_unweighted` (0.665), bottoms at `DLinear` (1.502).
See D-96. Full detail: `notebooks/04_feature_engineering/F10_results.md`,
`notebooks/05_benchmarking/b16_gapfilled_climatology_fix.py`,
`results/b16_gapfilled_climatology_fix_best_config.csv` (corrected), `results/
b16_final_table_vs_gapfilled_best_config.csv` (superseded, kept for history).

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

**Current:** I-02 (D-61) + **I-03 (D-102, 2026-08-18, recalibrates the champion)**. I-01 (D-39)
targeted a different, earlier, superseded harness and is not used as precedent.

**I-02 headline finding:** `fx_lsu_dens` (livestock density) confirmed as the dominant driver by
every method that can see it (RF/XGB native importance, RF/XGB/LightGBM SHAP, SARIMAX coefficients
at all 3 towers, TabPFN permutation importance at T4/T9). Its SHAP importance **grows** with lead
time, not shrinks — mean |SHAP| at Tower 4: 7.6 (bin 1–7 days) → 38.4 (bin 181–270 days).

**I-03 — closes the gap I-02 left behind.** I-02 predates TabICLv2 (D-66) and F-10's species
features (D-67) by days — its results were computed on the old 8-model roster and old (BASE-only)
feature set, never on the model this project actually recommends (TabPFN+species, the standing
champion). I-03 recalibrates TabPFN's interpretability (champion-focused scope, mirroring U-04's
precedent — TabICLv2 not covered, flagged as a follow-up) on `forecast_daily_v3.csv`'s
`BASE+species` config, same permutation-importance method as I-02's own TabPFN treatment,
~22-minute runtime (zero-shot, no retraining). **Result: `fx_lsu_dens` dominance holds on the
actual champion** (mean importance 1.1456, #1 of 52 features), reconfirming I-02. **New finding
I-02 could not make**: `fx_cattle_dens` is a clear #2 (0.8043) while `fx_sheep_dens`/`fx_lamb_dens`
rank near the bottom (0.0143/0.0320) — F-10's species-disaggregation gain is concentrated entirely
in the cattle component, independently corroborating S-05's scenario-projection cattle-dominance
finding via a completely different method. **Tower 2 shows zero livestock features in its top 10**
(top drivers are all TA/TS/SWIN) — a fourth independent confirmation of T2's livestock-blindness
(after U-03, S-01, S05-T2/D-95). Full detail: `notebooks/06_interpretability_uq/I03_results.md`.

**Sources:** `notebooks/06_interpretability_uq/I02_feature_importance_rollout.ipynb`,
`I02_results.md`, `i03_champion_interpretability.py`, `I03_results.md`.

---

## 5. Uncertainty quantification

**Current:** U-02 (D-62, builds the calibration) + U-03 (D-63, stress-tests it) + **U-04 (D-88,
2026-08-10, recalibrates the champion)**. U-01 (D-40) superseded, different harness.

**U-04 — closes the gap U-02 left behind.** U-02 predates TabICLv2 (D-66) and F-10's species
features (D-67) by days — its "TabPFN" interval is calibrated for a feature configuration this
project no longer recommends, and TabICLv2 has never had UQ at all. U-04 recalibrates TabPFN and
TabICLv2 (champion-focused scope, user-confirmed — the other 6 U-02 models' feature config never
changed) on `forecast_daily_v3.csv`'s `BASE+species` config, same leave-one-anchor-out conformal
method, **25-second runtime** (zero-shot, no retraining). **Result: calibration converges to
~0.89–0.90 PICP at T4/T9**, matching U-02's own headline finding on the new feature config.
**Species enrichment improved point accuracy without materially changing calibration** — TabPFN
conformal MPIW/pinball are essentially unchanged old-vs-new (T4: 148.44→149.52 MPIW, 10.56→10.56
pinball; T9: 190.31→188.93, 13.00→12.86) — mechanistically sensible (margins track residual
distribution, and the species-config MASE gain was modest, not dramatic) but not assumed in
advance. T2 still cannot support calibration, confirmed to persist independent of the feature-set
change (real coverage ends May 2019 regardless of which columns are used). Foundation for
scenario-analysis UQ (queued next, same machinery, AOA-stratified). Full detail:
`notebooks/06_interpretability_uq/U04_results.md`. Fancharts added same day (30 figures,
`results/figures/u04_fancharts/`) — U-04 had initially shipped without them despite U-01/U-02/U-03
all having them.

**U-05 — scenario-analysis UQ, closes the "scenario predictions have no interval" gap (D-89,
2026-08-10).** U-04's calibration isn't reusable for S-05 (different, narrower feature space —
`FX_A_SPECIES`, 13 cols, vs. U-04's `BASE+species`, 52 cols — is a genuinely different model).
Repeats U-04's method (TabICLv2 zero-shot, leave-one-anchor-out conformal, 5 real anchors × 3
towers) on S-05's own architecture — **9-second runtime**. **Design question resolved
empirically**: does prediction error correlate with AOA-flagged status? Weak raw correlation
(r=0.146) but a real, substantial categorical gap (out-of-AOA residuals ~48% larger, pooled) —
resolved with a **two-tier margin interpolated by each point's own `aoa_flagged_pct`**, not a
smooth continuous function (the correlation doesn't support one) and not a flat ignore-AOA margin
either. Applied to S-05's existing livestock/grazing/fertilizer outputs with **zero new model
calls** (pure post-processing join against already-saved AOA data). **Calibration converges to
~0.88–0.89 PICP at T4/T9**, matching U-02/U-04 a third time; T2 stays uncalibratable (third
confirmation of the same structural limitation). **Interval is genuinely wide**: ±94–100% of the
mean in-AOA, ±139–140% out-of-AOA — consistent with U-01's original "large aleatoric uncertainty"
finding, not a contradiction. Full detail: `notebooks/06_interpretability_uq/U05_results.md`.

**U-06 — CQR fixes the spike-coverage failure U-04/U-05's own fancharts revealed visually (D-90,
2026-08-10).** User observation, checked directly: "a lot of spikes are still beyond the
interval." Confirmed and quantified: overall PICP≈0.89 looked fine, but **75% of top-10%-
magnitude days fell entirely outside the interval, vs. 3.3% for the bottom 90%** — split-conformal's
flat symmetric margin only guarantees *average* coverage. A pre-build check found the model's own
raw q95 already sits close to (TabPFN) or exceeds (TabICLv2) actual spike values, while the median
massively undershoots — motivating **Conformalized Quantile Regression** (nonconformity =
`max(q05-y_true, y_true-q95)`, interval = `[q05-margin, q95+margin]`) instead of a flat margin
around the median. `conformal_margins_by_bin()` reused completely unchanged — **no new model
calls**, pure recalibration of already-saved chains. **Result: spike coverage roughly triples**
(TabICLv2: 24.3%→**79.7%** U-04, 22.1%→**79.3%** U-05; TabPFN: 24.3%→**57.2%**), at the honest
cost of normal-day coverage dropping from ~96–97% to ~83–88% (still comfortably above 80%) and
spike intervals roughly doubling. TabICLv2 benefits more than TabPFN, consistent with its raw q95
already exceeding actual spikes. **CQR should replace the symmetric-margin approach as this
project's standing UQ method going forward.** Now applied to S-05's actual scenario trajectories —
see D-92 below. Full detail: `notebooks/06_interpretability_uq/U06_results.md`.

**U-07 — livestock-density-stratified CQR: thinner margins where livestock presence is smaller
(D-91, 2026-08-10).** Direct user question on U-06's output, checked empirically before building:
"can't the margin be thinner where livestock presence is smaller?" **Signal much stronger than
U-05's AOA-distance check**: `corr(|residual|, fx_lsu_dens)=0.43–0.45` (vs. AOA's weak 0.09–0.15),
residuals ~3.2× larger on above-median-LSU days. `fx_cattle_dens` correlates almost identically
(0.427), consistent with S-05's cattle-dominance finding. Same CQR machinery as U-06 — only the
bin key changes to lead-time × LSU-tertile; `conformal_margins_by_bin()` needed zero code changes
(5th reuse across U-02/U-04–U-07). **Result: low-LSU intervals are 29–46% the width of high-LSU
intervals** (TabPFN: 84.3 vs. 293.7 nmol; TabICLv2: 177–206 vs. 419–447) — **a genuine win-win,
not a trade-off**: verified directly that spike days (3.2× higher `fx_lsu_dens`) still get their
own dedicated, appropriately-wide calibration in the "high" tier, not diluted by low-LSU days the
way the single pooled CQR margin was. **Should be the standing UQ method going forward, layered on
U-06's CQR.** Full-roster figures (T4/T9 × both champion models, T2 explicitly logged as
degenerate) widen the low-as-%-of-high range to **26–59%** — same direction/magnitude everywhere.
Full detail: `notebooks/06_interpretability_uq/U07_results.md`.

**S-05 + UQ — U-06/U-07's CQR calibrations attached to S-05's ACTUAL scenario trajectories (D-92,
2026-08-10), closing the last standing gap between scenario analysis and UQ.** Two new scripts,
zero new calibration fitting: reruns S-05's existing 18-call/axis representative subset (54 calls,
2050 horizon) requesting `quantiles=(0.05,0.5,0.95)` instead of a point prediction — confirmed free
(4.3s/call, same as point-only; TabICL always computes an internal quantile grid regardless), full
run **~2.5 min**, 0 failures. Attaches U-05's own FX_A_SPECIES-architecture margins via a (tower,
lead-bin[, LSU-tier]) lookup pooled across U-05's 5 anchor years. Two explicit extrapolation
assumptions: lead times beyond 365 days hold the widest calibrated bin's margin flat (likely
**understates** true uncertainty at year 20+ of the 2050 horizon — read far-horizon bands as a
floor, not a ceiling); grazing/fertilizer axes reuse the livestock-architecture margins despite
extra covariates (same approximation U-05's own Step 4 already made). **Result: works cleanly,
>99% coverage at T4/T9** (T2 0%, pre-established degeneracy, not new); one genuine thin spot (T4
days-1-7 × mid-LSU-tier, zero calibration samples, surfaced as NaN). Verified directly: zero
interval inversions, CQR correctly tightens the model's own raw quantile spread on average at T4
(raw MPIW 572.6 vs. U-06 514.6 vs. U-07 523.4). **This closes the U-04→U-07 UQ arc's last open
caveat — Objectives 4 (UQ) and 5 (scenario analysis) now genuinely connect in the output.** Full
detail: `notebooks/07_scenario_analysis/s05_results.md` ("Update: U-06/U-07's CQR calibrations
attached...").

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

**Current: S-04 (D-82, 2026-08-06 analysis of a 2026-07-15/16 build)** — extends S-01's
level-residual hybrid from a single SSP2-4.5/ensemble-mean/2041-2060-snapshot worked example to a
real transient trajectory: **both SSP2-4.5 and SSP5-8.5, full realization scale (500/SSP for the
primary hybrid), annual 2025-2050** (not a single climatological composite), plus a B-10-ensemble
diagnostic benchmark run in parallel on a stratified 10-realization subset. Uses S-01's exact frozen
model artifacts — no retraining. **Headline: S-01's central finding (the hybrid fixes the tree-only
extrapolation plateau U-03 found) holds and is reinforced across the full 26-year x both-SSP
trajectory** — 1x->3x livestock response: hybrid +38.6%/+156.4%/+120.3% (T2/T4/T9) vs. the
diagnostic tree/SARIMAX ensemble's own +20.4%/+76.6%/+62.0% (matched, same-inputs comparison).

**New finding beyond S-01: realization-level/transient construction reveals materially more AOA
extrapolation risk than S-01's smoothed ensemble-mean snapshot did, even at the unchanged 1x
livestock baseline** (9-15% of days flagged here vs. 0% in S-01 at 1x/2x everywhere) — the
scenario-construction method (smoothed composite vs. real transient weather) measurably changes how
out-of-distribution a scenario looks, independent of the livestock question. SSP2-4.5 vs SSP5-8.5
divergence is real, widens toward 2050 as expected, but stays under 1% of the mean throughout —a
minor lever next to the livestock multiplier. Realization-level spread itself is small (1-5% of the
mean, pooled) and *narrows* in relative terms as livestock stress increases. **Bias-corrected
(D-100, 2026-08-17):** anchoring to the real historical mean (same per-tower offset as S-01, same
frozen model) shifts the headline 1x-to-3x figures to **T2 +48.1-48.5%** (was +38.6%), **T4
+152.4-153.2%** (was +156.4%), **T9 +133.1-133.7%** (was +120.3%) — a modest correction, consistent
with S-01's own already-small, already-accepted baseline gap. Full detail:
`notebooks/07_scenario_analysis/s04_results.md`.

**Diagnostic extension: S-05 (D-84, 2026-08-08)** — different model, different question: TabICLv2
(one-shot, not a recursive rollout) + S-03's Variant A feature set (10 CMIP6-derivable columns) +
F-10's species-disaggregated livestock density, run as a 10-year transient trajectory with
**independent per-species multipliers** (cattle/sheep/lamb scaled separately, 27 combos, not a
single shared scalar) — 8,100 calls, 2.54h. **Headline: cattle dominates the FCH4 response far
beyond its own LSU-weight share** (tripling cattle alone ~triples predicted FCH4 at T4/T9;
sheep/lamb stay under 25% even at 3×) — the species-split family (behind the standing
`TabPFN+species` champion, D-67) earns its place in a scenario context too, not just on real
historical anchors. A first-pass "realization spread" metric (pooling year+realization+GCM, S-04's
own convention) gave a misleading 32–69% of the mean; isolating realization/GCM alone (fixed year)
gives 2.4–6.6%, consistent with S-04 — the gap is genuine year-to-year weather sensitivity specific
to TabICLv2's lack of a smoothing trend component, caught and corrected before being reported as a
headline number. AOA flagged-% is high (62–68%, vs. S-04's 9–15%) but flat over the horizon —
second confirmation that AOA's absolute level tracks feature-space breadth, not a new risk. No
change to any standing recommendation. Full detail: `notebooks/07_scenario_analysis/s05_results.md`.

**S-05 extended to 2050 + full daily chains (D-85, 2026-08-08, same session)** — same 8,100-call
grid, horizon extended from a fixed 10 years to 2050 (matching S-04's endpoint), plus full daily
chains saved for every scenario point (83.8M rows, 1.25GB Parquet), not just the annual mean.
Actual runtime 5.44h (estimated ~9h). **Every finding above replicates, several more clearly**:
cattle dominance holds/strengthens (T4 3×-alone +205.6%→+214.5%); realization/GCM spread stays in
the same small isolated range (2.4–6.6%→2.5–7.6%); AOA flatness now confirmed across the full
27–31-year horizon. **New pattern the 10-year window was too short to show**: SSP2-4.5 vs
SSP5-8.5 divergence now visibly grows from early to late window, matching S-04's own "widens
toward end of century" finding. No change to any standing recommendation. **Bias-corrected
(D-100, 2026-08-17): S-05's own baseline-reconstruction check (never run before D-100) found a
40-80% underprediction of the real historical mean at every tower — far larger than S-01/S-04's**
**9-20% gap. Anchoring to real history via the same delta-method roughly HALVES this headline: T4
3×-alone +214.5%→+101.8%, T9 +187.3%→+110.4%.** The dominance finding's direction is unaffected;
the exact magnitude is now genuinely uncertain between raw and corrected pending further
investigation. See D-100, `s05_results.md`.

**S-05 extended to farming-practice scenarios (D-86, 2026-08-08, same session)** — grazing timing
and fertilizer schedule, two separate baseline-livestock experiments (900 calls each, ~51 min/axis).
Priors stated before building anything, both confirmed: **grazing timing shows a real, monotonic
effect at every tower** (T4: 14.81→17.61 nmol, +18.9% at +4 weeks; T9: +17.2%) — tied directly to
livestock presence, the #1 driver throughout this project. **Fertilizer schedule shows a small,
sign-inconsistent effect across towers** (T2/T9 negative, T4 positive, all under 5%) — extends
F-01/F-04/F-05's "redundant on the rich base" finding from real-data feature importance to
scenario response, the first time that finding has been tested this way. Both DERIVED features
(grazing presence pattern; fertilizer event-decay) reconstructed by reusing their real-data
construction functions unchanged, not reimplemented. No change to any standing recommendation —
grazing-season length joins livestock density as a genuine management lever worth reporting;
fertilizer's null result is itself a legitimate, now three-times-confirmed finding — **including at
a regulatory-grounded ceiling**: D-105 added a `reg_cap` level (rate scaled so the true, area-
weighted N loading hits exactly the UK NVZ N-max, 300 kg N/ha/yr, gov.uk), run on both S-05 and S-06,
and the effect stays negligible even there (T2 -0.9%/-0.7%, T4 +1.5%/+1.7%, T9 +0.1%/+0.1%,
S-05/S-06) — this rules out "the +50% levels just weren't extreme enough" as an explanation for the
null result.

**Architecture reference: S-01 (D-64)** — first bounded worked example, proving the
scenario-simulation mechanism end-to-end; S-04 reuses its exact frozen hybrid, unmodified. **B-08
confirmed superseded for Phase 07's purposes by U-03** (its extrapolation-check finding already
answers what B-08 would have; B-08 remains available separately for the point-forecast track).

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

**Bias-corrected (D-100, 2026-08-17):** anchored to the real historical mean instead of the
model's own imperfectly-reconstructed 1× baseline (Finding 1, `s01_results.md`) — **T2 +40.7%**,
**T4 +135.4%**, **T9 +114.4%**. A modest, expected correction (single digits to low teens of
percentage points); the standing conclusion (hybrid measurably fixes the tree-only plateau) is
unaffected. See D-100, `results/s01_scenario_summary_bias_corrected.csv`.

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

**Queued-next update (D-82): SSP5-8.5 and realization-level spread are now DONE — see S-04 above.**
Still outstanding: a self-consistent mechanistic livestock-scenario construction, and (if time
permits) SPACSYS (already validated at North Wyke, Wu et al. 2016) for the trend/level component.

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

**Second addendum (D-83, 2026-08-08)**: rescored with climatology MASE (D-80) and reran on
TabICL-sourced gap-filled data (D-79) — the original finding replicates unchanged. **Unplanned,
more consequential finding**: TabICL-sourced `y_gapfilled` is a substantially worse *training
target* for the tree/SARIMAX/ensemble family specifically (~1.3-2.5x higher MASE at every tower,
traced directly to TabICL's gap-filled series sitting at a materially different mean level than
RF's) — a much larger effect than D-80's earlier finding that TabICL context only mildly hurts
TabPFN/TabICLv2. Still not a production-config change — reinforces, independently, that RF-sourced
gap-filling remains the right choice project-wide. See D-83, `s03_results.md`'s second addendum.

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

**S05-T2 (D-95, 2026-08-10): does pooling rescue Tower 2's muted livestock-scenario response? No —
exactly 0.0 percentage points of difference, both TabICLv2 and TabPFN.** T2's cattle-3× response is
only +1.8-2.3% (vs. T4/T9's +186-215%) because T2's real historical `fx_lsu_dens` never exceeds
~0.71 — the zero-shot model has no historical livestock→CH4 covariation to learn from at that
tower. Tested whether pooling T2's context with T4/T9's real livestock-rich history (the exact
mechanism that rescued Tower 9 in gap-filling, F-02/F-03) transfers a learned cattle sensitivity
in. **Result: an exact decimal match between pooled and solo for both models, every combo/SSP** —
not a small effect, zero measurable effect. Mechanistic read: this style of pooling shares context
rows within one batched call, not fitted parameters the way Track A's trees are jointly fit — a
zero-shot forecaster's output for one series stays driven almost entirely by that series' own
history regardless of what else is in the same batch. **T2's muted livestock-scenario response
should be read as a genuine model-extrapolation limit, not a fixable data-availability gap.** Full
detail: `notebooks/07_scenario_analysis/s05_t2_pooled_test.py`, `s05_results.md`.

**S-06b (D-108, 2026-08-20): S-06's core grid replicated on the B18-derived architecture.** Closes
the gap D-106 (B18) explicitly left open ("does not silently replace the model inside... Phase-07
scenario pipelines"). Gated by a 3-round real-anchor validation (S-03b/c/d): B18's own winning
feature set can't run in scenario mode (needs real future antecedent/flow data), and the naive
pooled-across-towers adaptation isn't faithful to S-05/S-06's per-tower-anchor design — the
locked-in config is **solo per-tower `Direct_TabICLv2` regression on `FX_A_SPECIES` +
`b17_days_since_2010`** (+2.79% MASE vs. the old `tabicl_forecast()` TS-wrapper, extrapolation
safety to 2050 confirmed directly via a sanity check, not assumed). Runs ~2.2h for the full 3-axis
grid (vs. S-05/S-06's original ~6-9h) since the new architecture fits once per tower and reuses
that fit across every scenario combo, instead of refitting per call.

**Every S-06 headline finding replicates cleanly**: T9's real stocking density still exceeds the
UK regulatory ceiling (lit_ceil -2.6% vs. baseline, was -2.3%); T4's lit_ceil margin stays
thin-positive (+0.8-0.9%, both architectures, near-exact match); grazing +4wk still a real lever
(T4 +19.8%, T9 +19.9%, vs. S-06 old +18.5%/+18.6%); fertilizer still a null result (<8%,
sign-inconsistent, vs. old <5%); cattle dominance reconfirmed (own_max cattle-alone ≈ all-species
at T4/T9, T9 cattle-alone edges ahead). A real bug (D-104's lit_ceil correction wasn't initially
carried over into the new pipeline) was caught by a direct user question and fixed via the same
scoped rerun-and-merge pattern as the original fix — see D-108 for the full account.

**Not yet done**: `reg_cap` fertiliser level for S-06b (D-105's addendum, deferred); U-05b Step 4
(attach calibration to S-06b's actual outputs, mirroring D-92); `report/TODO.tex`'s Ch4-6 text
still describes the old architecture. Full detail: `notebooks/07_scenario_analysis/S03b_results.md`,
`notebooks/06_interpretability_uq/{I03b,U08,U06b_U07b}_results.md`, D-108.

**Sources:** `notebooks/07_scenario_analysis/S01_first_scenario.ipynb`, `s01_results.md`,
`src/features/build_scenario_drivers.py`, `src/models/scenario_hybrid.py`,
`results/s01_scenario_summary.csv`, `results/figures/s01_*.png` (4 figures),
`results/models/s01_*.joblib` (frozen artifacts).
