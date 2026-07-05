# F-09 — root-cause fix: unfiltered USTAR/VPD outliers corrupting met-driver gap-filling (D-48)

**Scripts:** `src/data/reddyproc_pipeline.py`, `src/data/build_sms_met_dataset.py` (fix); `src/data/build_fch4_gapfilled.py`,
`src/features/build_forecasting_matrix.py`, `src/features/build_forecasting_matrix_v2.py` (regenerated downstream).
**Trigger:** user-observed anomaly — a visually obvious, sustained spike in Tower 2's gap-filled FCH4 during a
period with zero real observations, also present at Towers 4/9.

## 1  The observation

Tower 2's `y_gapfilled` jumps from ~2–20 nmol (Jan–May 2019, some real data) to **325–413 nmol** for June–
December 2019, with `observed_mask == 0` for 100% of that stretch (Tower 2's analyser had relocated to Tower 9,
D-15/D-34). The same signature — implausible magnitude inflation during a fully-unobserved stretch — was
confirmed at **Tower 9** (pre-Feb-2020, before installation) and **Tower 4** (2024–2025, the known
not-yet-downloaded gap): every tower shows elevated gap-filled values whenever it has an extended run with no
real anchor point nearby.

## 2  Root-cause investigation

Ruled out (checked empirically before accepting): a cascading error from FCO2/GPP/Reco reconstruction (all
unremarkable/plausibly-scaled during the spike window); a vague "unfamiliar feature combination" explanation
(tested directly — real training rows across all 3 towers with a similar FC/livestock/season combination to
Tower 2's July-2019 profile show **low** targets, 1.4–16.2 nmol, not high, ruling out simple donor-tower
pattern-borrowing).

**Confirmed mechanism, via SHAP TreeExplainer on the actual pooled RFm gap-filler**: for a representative
Tower-2 July-2019 row, **`USTAR_0_0_1` alone contributes +244 nmol and `WS_0_0_1` +107 nmol** of a total +380
nmol lift from the model's base value (31.7) to its prediction (411.9) — over 92% of the entire spike from two
features. Tracing further:

- **`USTAR_0_0_1 [Tower 2]` has zero real observations in July 2019** — the value feeding the model is 100%
  gap-filled.
- **Tower 2's raw USTAR column was never quality/plausibility-filtered anywhere in the pipeline** (unlike FCH4
  `[-500,3000]`, D-13, and FCO2 `[-100,100]`, D-25). It contains 722 historical readings above 5 m/s, up to a
  maximum of **1039.9 m/s** — physically impossible for friction velocity (sane values are well under 1–2
  m/s). Raw mean = 2.450, median = 0.233 — a ~10x mean/median gap diagnostic of severe outlier contamination.
- **`reddyproc_pipeline.py`'s `mdc_gapfill()` uses the arithmetic mean as its last-resort fallback** (for hours
  with no nearby real data to interpolate from or build an hourly climatology around — exactly the case for an
  extended blackout). Confirmed: the actual `USTAR_0_0_1 [Tower 2]__f` feature is **flat at 2.468 for July 2019,
  identical to the mean across the entire series** — the gap-filler has fallen through to the contaminated
  global-mean fallback, producing a non-seasonal, non-physical constant.
- **The CH4 RF model itself is not wrong to weight USTAR heavily** (it's the pipeline's #1 feature at 29%
  importance) — real turbulence-driven CH4 release is a known EC-flux phenomenon. The bug is that a fabricated
  ~2.5 m/s constant falls in a range the model associates (correctly, from rare *real* high-turbulence events
  in training) with elevated flux — it has no way to distinguish a genuine turbulence event from an
  outlier-contaminated fallback constant.

**Broader audit** (all met columns, all 3 towers): only **USTAR** (1.4–1.7% of readings > 3 m/s, max up to
1408.8) and **VPD** (1.8–12.6% of readings > 15 kPa, worst at Tower 4, max 36–41 kPa — physically implausible,
since VPD is bounded by saturation vapour pressure and rarely exceeds ~8 kPa) show this contamination
signature. PPFD/WS/SHF were checked and are clean (their mean/median gaps reflect normal diurnal skew, not
outlier contamination).

## 3  The fix

Two changes in `reddyproc_pipeline.py` (reused by `build_sms_met_dataset.py`, so both the EC and external
SMS/MET variants get the fix from one place):

1. **`plausibility_filter()`** — new function, applied before gap-filling, mirroring the existing FCH4/FCO2
   pattern: `USTAR_0_0_1` bounded to `[0, 3]` m/s, `VPD_0_0_1` to `[0, 15]` kPa; values outside are set to NaN
   and handled by the normal interpolation/MDC/fallback chain like any other gap.
2. **`mdc_gapfill()`'s last-resort fallback changed from mean to median** (`hourly_mean`→`hourly_med`,
   `out.mean()`→`out.median()`) — a general robustness improvement: this tier only fires for long blackouts
   with zero nearby real data, exactly where a residual contaminated mean does the most damage, and the median
   is robust to outliers by construction. The rolling-window MDC step itself (mean, matching the literature
   convention for "mean diurnal course") was left unchanged — it's far less vulnerable given it always has a
   local window of mostly-real data to average over, unlike the global fallback.

## 4  Validation — the fix resolves the identified spikes

| Tower | Before fix | After fix | Real baseline for comparison |
|---|---|---|---|
| 2, Jun–Dec 2019 | 325–413 nmol | **2.9–27.7 nmol** | Jan–May 2019 real data: 2–20 nmol |
| 4, 2024 | mean 227.6 | **mean 26.6** | 2017–2023 real-data years: 16–80 |
| 9, pre-2020 | 10–207 nmol (noisy) | 10–160 nmol (still elevated some months, smaller effect) | — |

Tower 2 and Tower 4 are cleanly resolved — gap-filled values now sit squarely within each tower's own plausible
historical range. Tower 9 shows a smaller improvement (its case was less extreme to start with, and it may have
other, undiagnosed contributing factors — not fully investigated further given the fix's clear correctness
elsewhere).

## 5  Scope discovery — this is not a narrow boundary effect

Regenerating the full downstream chain (`build_fch4_gapfilled.py` → `build_forecasting_matrix.py` →
`build_forecasting_matrix_v2.py`) and comparing AR-feature means across the **entire 2018–2023 evaluated
window** (not just the isolated blackout periods) revealed the impact is systematic, not localised:

| Tower | AR-feature mean, 2018–2023 (before) | AR-feature mean, 2018–2023 (after) | Real observed CH4 mean |
|---|---|---|---|
| 2 | 71.9–72.2 | **20.2** | 31.9–33.1 |
| 4 | 54.6 | **31.1** | 29.2–29.8 |
| 9 | 79.2–79.4 | **36.6–36.7** | 36.0–36.3 |

**The corrected AR-feature means now sit close to each tower's real observed CH4 mean** (Tower 9: 36.6 vs. 36.0
real; Tower 4: 31.1 vs. 29.2–29.8 real) — exactly what a correctly-functioning autoregressive feature should
look like. Before the fix, every tower's AR feature was inflated 1.5–2× above its own real distribution. This
is independent corroborating evidence the fix is correct, beyond resolving the originally-observed spike.

**Consequence: AR/CH4-history — one of the most important predictors in the whole forecasting phase (I-01's
SHAP analysis) — was systematically biased throughout the entire 2018–2023 training/test window for all three
towers, for every one of B01 through B08.** The R²/MASE/etc. numbers currently in `benchmarks.csv` for the
whole forecasting phase are stale relative to the corrected data. Since the fix mainly *reduces* a previously
inflated bias (bringing AR features closer to the true CH4 distribution), the most likely direction of change
on re-run is neutral-to-positive for skill (a less-biased predictor should be more, not less, informative) —
but this is a plausibility argument, not a measured result; the actual re-run has not been done yet (staged
separately per user request).

## Recommendation
- **Fix adopted** — `reddyproc_pipeline.py`/`build_sms_met_dataset.py` now filter USTAR/VPD and use a robust
  median fallback; `reddyproc_processed.csv`, `reddyproc_processed_SMS_MET.csv`, `consolidated_hourly_SMS_MET.csv`,
  `fch4_gapfilled.csv`, `forecast_features.csv`, `forecast_features_v2.csv`, `forecast_daily_v2.csv` have all
  been regenerated with the fix already (data files are current; only the benchmark notebooks/results are stale).
- **B01 through B08 all need re-running** against the corrected data before their reported numbers can be
  trusted — staged as a separate follow-up (not done in this session, per user's explicit choice).
- **B-08 (driver-realism, queued) should wait for this re-run** rather than building on stale AR features.
- Tower 9's smaller, incompletely-resolved residual improvement is noted but not further investigated — worth
  a follow-up if Tower 9-specific results look off after the re-run.

## Files / scope
Not additive-only, by necessity — this is a correctness fix to shared upstream data-generation scripts, so it
regenerates `reddyproc_processed.csv`, `reddyproc_processed_SMS_MET.csv`, `consolidated_hourly_SMS_MET.csv`,
`fch4_gapfilled.csv`, `forecast_features.csv`, `forecast_features_v2.csv`, `forecast_daily_v2.csv` in place.
`benchmarks.csv` and all B01-B08 notebook outputs are **not yet regenerated** and should be treated as stale
until the flagged re-run happens.

*Source: ad-hoc diagnostic investigation (SHAP attribution, before/after comparison), documented here rather
than as a notebook given its nature as a pipeline bug-fix rather than a benchmarking experiment. Decision D-48.*

---

## Addendum (2026-07-02, D-49) — F-09a: standalone gap-filling re-verification

The forecasting-downstream re-run (B-03/B-03a/B-03b, see `b03a_b03b_results.md`) confirmed the fix's effect on
*forecasting* skill, but not on the gap-filling task itself (recovering real masked FCH4 observations under
full-period gap-CV, F-08's original question, D-35). The original `F08_external_sensors_RFm.ipynb` could not be
re-run directly for a clean before/after comparison — it timed out at both 1800s and 3600s even pre-fix (its
full EC×EXT × solo×pool × 3-tower × 5-scenario × 5-rep grid runs 300+ individual 500-tree RF fits). **Per
explicit instruction, the original notebook, `results/f08_summary.csv`, and F-08's rows in `benchmarks.csv`
were left untouched** (confirmed via `git diff` — no changes) to preserve the historical pre-fix D-35 baseline.

Instead, a separate, reduced-scope script (`f09a_gapfill_check.py`, ad-hoc/scratchpad — not committed to the
repo, same precedent as the earlier FCO2-reconstruction check) reused F-08's exact gap-CV methodology verbatim
(constants, `insert_calendar_gaps`, `frame`, `fit`, `dom_mask`, `mets`/`med_metrics`) but scoped to EXT variant
only (what the production forecasting pipeline actually uses, D-35/D-37) and `RFm_pool` only (F-08's own
recommended config), 2 reps instead of 5 — completing in ~20 minutes.

**Result: real gap-filling accuracy improved, not just downstream AR-feature statistics.** Overall median R²
across all 5 gap-scenarios (EXT, RFm_pool):

| Tower | Pre-fix (F-08, D-35) | Post-fix (F-09a) |
|---|---|---|
| T2 | 0.490 | **0.574** |
| T4 | 0.376 | **0.402** |
| T9 | 0.364 | **0.418** |

All three towers improved. This is independent corroborating evidence — beyond D-48's AR-feature-mean
plausibility check — that the plausibility-filter fix genuinely improves the gap-filler's ability to recover
real FCH4 values, not merely a redistribution of bias. Saved to `results/f09a_summary.csv` (per-tower,
per-scenario detail); does not modify `results/f08_summary.csv` or its `benchmarks.csv` rows.

*Cross-ref: D-48 (the fix), D-49 (this re-run + the B-03/B-03a/B-03b forecasting-side results).*
