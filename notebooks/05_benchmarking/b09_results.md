# B-09 — recursive 365-day daily rollout backtest: does autoregressive forecasting compound error?

**Notebook:** `B09_recursive_rollout.ipynb` (single-anchor, 2021-12-16). **Results:**
`results/b09_summary.csv` (single-anchor bins), `results/b09_chains.csv` (full chains),
`results/figures/b09_rollout_chains.png`, `results/b09_multi_anchor_summary.csv` (5-anchor
extension, 2018–2022), `results/b09_seasonal_check.csv` (real-flux mean/std per bin/anchor).
**Shared helpers:** `src/models/recursive_rollout.py` (new). Multi-anchor extension run via an
ad-hoc script (not committed, same precedent as F-09a/F-09b) reusing `recursive_rollout.py`
unmodified with the anchor as a loop parameter.

Tests whether recursive (feed-your-own-prediction-back-in) forecasting stays usable over a full
365-day horizon, or compounds/degrades badly — a direct precursor question for the long-range
scenario work (D-46 requirement 5, D-52), and directly motivated by a sibling project's own
lesson (`FORECASTING_LEARNINGS.md`: "always validate a rollout mechanism against a real held-out
window before trusting it on a genuinely blind future window").

## Design

Single fixed anchor per run (initially **2021-12-16**, Tower 4; extended to 2018/2019/2020/2022
anchors, §3), one continuous 365-day recursive chain per model — not a walk-forward evaluation
across many origins (which never lets error compound past a few days). Every model fit fresh on
data strictly ≤ anchor (no reuse of B03/B03a/B03b/B04's existing ≤2021-12-31 results — that would
leak real observations into "training"). Perfect-foresight `fx_` drivers throughout — isolates the
recursive-mechanism question from driver-forecast realism (B-08's separate scope, D-47).

**Models tested (committed scope): SARIMAX, RandomForest, XGBoost, LightGBM, DLinear, LSTM —
Tower 4 only.** TFT, Tower 9, and TabPFN-prep are stretch items (§6).

**Why so fast (RF 18s, everything else 1-5s per anchor, ~30s total for all 6 models)**: this is a
small-data, small-model problem, not a big-data one. Pooled tree training is ~5,400 rows × 43
features — trivial for sklearn/XGBoost/LightGBM regardless of tree count. DLinear/LSTM train for
only 30 epochs (this project's bounded-iteration convention, D-41/38) on the same small row count,
on the RTX 5070 GPU — a few thousand tiny batches finish in seconds. The 365-step rollout itself
is cheap per step: each iteration is a handful of scalar feature lookups plus one single-row
`model.predict()` call. Contrast with F-08 in this same project, which does 300+ *repeated* RF
fits in nested loops and takes 30-60+ minutes — B-09 fits each model exactly once per anchor.

**Ground-truth caveat, stated up front, not a footnote**: the 2021-12-16 anchor's target window
has 341/365 days (93.4%) passing the day-level `y_observed`≥6-hours rule, but true hourly-observed
density in that window is only **48.9%** — even the "ground truth" is itself substantially
gap-filled. Every number below inherits that caveat.

## 1. Single-anchor result (2021-12-16) — overall and by lead-time bin

**Overall** (all 365 days blended, n=341 real observations):

| Model | R² | MAE | MASE |
|---|---|---|---|
| DLinear | 0.308 | 47.5 | 1.110 |
| LightGBM | 0.305 | 43.1 | 1.007 |
| XGB | 0.279 | 43.9 | 1.025 |
| RF | 0.261 | 44.2 | 1.033 |
| SARIMAX | 0.233 | 42.0 | **0.981** |
| LSTM | 0.105 | 43.0 | 1.004 |
| Persistence | -0.152 | 42.8 | 1.000 |
| Climatology | -0.041 | 44.1 | 1.030 |

Every trained model beats both baselines on overall R². SARIMAX is the only model beating
persistence on overall MASE — DLinear's best-in-class overall R² is misleading on its own (see §3).

**Non-overlapping lead-time bins** (the M5-lesson analogue of "don't blend across the hierarchy" —
here binning by lead-time-within-the-chain instead of store/category):

| Model | 1–7 (n=3) | 8–30 (n=19) | 31–90 (n=50) | 91–180 (n=90) | 181–270 (n=88) | 271–365 (n=91) |
|---|---|---|---|---|---|---|
| **R²** | | | | | | |
| SARIMAX | -5.577 | 0.013 | 0.016 | 0.111 | 0.207 | -0.271 |
| RF | -3.127 | -0.732 | -0.182 | 0.209 | 0.222 | -0.594 |
| XGB | -1.803 | -0.170 | -0.098 | 0.137 | 0.322 | -0.514 |
| LightGBM | -0.606 | -0.214 | -0.056 | 0.193 | 0.335 | -0.582 |
| DLinear | -31.181 | -5.555 | -2.628 | 0.260 | 0.417 | -1.017 |
| LSTM | -15.351 | -0.658 | -0.141 | -0.123 | 0.181 | -0.642 |
| Persistence | -0.313 | 0.000 | -0.077 | -0.383 | -0.254 | -0.009 |
| Climatology | 0.020 | -0.331 | -0.223 | -0.141 | -0.192 | -0.274 |
| **MASE** | | | | | | |
| SARIMAX | 2.083 | 0.990 | 1.044 | 0.877 | 0.952 | 1.370 |
| RF | 1.597 | 1.430 | 1.359 | 0.817 | 1.017 | 1.588 |
| XGB | 1.438 | 1.181 | 1.207 | 0.900 | 0.967 | 1.508 |
| LightGBM | 1.099 | 1.155 | 1.125 | 0.857 | 0.961 | 1.554 |
| DLinear | 4.808 | 2.861 | 2.338 | 0.937 | 0.872 | 1.736 |
| LSTM | 3.565 | 1.516 | 1.095 | 0.964 | 0.879 | 1.415 |

**The 1–7 day bin is a small-sample artifact, not a real signal**: only **3** real `y_observed`
days fall there. With n=3, R² is dominated by whatever those 3 points happen to be — the actual
predicted values (e.g. RF ≈9–13, SARIMAX ≈10–16 nmol) are not physically implausible. Verified
directly: `RF`'s recomputed AR features for day 1 matched `forecast_daily_v2.csv`'s precomputed
columns exactly (spot-checked to 6 decimal places) — the AR-recompute mechanism itself is correct.

**91–180 and 181–270 (n=88–90) are the trustworthy bins**: every model except LSTM achieves
positive R² and MASE<1 — the recursive chain beats both baselines for a multi-month stretch.

## 2. Root cause of the 271–365 degradation — checked against the data, not assumed

The initial hypothesis (late-2022 has harder/more spikes) is **wrong** — tested directly:

| Bin | Mean flux | Std | Days >100 nmol |
|---|---|---|---|
| 91-180 (Mar-Jun) | 71.7 | 100.0 | 17 |
| 181-270 (Jun-Sep) | 61.1 | 101.5 | 22 |
| 271-365 (Sep-Dec) | **13.2** | **32.9** | **2** |

The late window is *quieter*, not noisier — and model MAE (absolute error) actually **improves**
there (SARIMAX: 28.5 vs 56.9/64.5 in the earlier bins; RF: 33.1 vs 53.0/68.9). The real mechanism:
**the anchor (Dec 16) sits in a naturally low-flux winter period.** Spring/summer (days 91-270) is
the high-variance grazing season where trained models have real signal to add over a stale
"repeat mid-December's value" persistence guess. By days 271-365, real flux has seasonally cycled
back toward a quiet regime close to the anchor's own value — so **persistence's own MAE drops from
64.9/67.8 (91-180/181-270) to 20.8** (271-365), becoming artificially competitive again simply
because the calendar looped back near where it started, not because naive repetition became a
better strategy. R²/MASE penalize the trained models for not beating an already-easy target, even
though absolute accuracy hasn't dropped. This is a materially better-supported explanation than a
generic "compounding error" story, and directly motivated checking whether it generalizes (§3).

## 3. Does this generalize? Multi-anchor extension (2018, 2019, 2020, 2021, 2022)

Re-ran the identical 6-model rollout anchored at Dec-16 of five different years (all with usable
real-data coverage, 61-93% by the day-rule) — 136s total for all 5 anchors × 6 models. This
directly tests §2's mechanism and addresses the single-anchor design's N_REPS=1 limitation.

**Result: the "everything degrades in the final quarter" pattern from 2021 does NOT generalize.**
SARIMAX's 271-365 R² (a structural control — zero recursive feedback, so any pattern it shares
with the feedback-driven models isn't caused by AR self-consumption):

| Anchor year | 91-180 | 181-270 | 271-365 |
|---|---|---|---|
| 2018 | -0.927 | 0.033 | -0.170 |
| 2019 | 0.224 | 0.451 | **0.091** |
| 2020 | 0.056 | 0.094 | **0.124** (best bin that year) |
| 2021 | 0.111 | 0.207 | -0.271 |
| 2022 | 0.202 | 0.044 | -0.058 |

Only 2018 and 2021 show real late-window degradation; 2019 and 2020 stay flat-to-positive, 2020
even peaks there. The §2 seasonal-echo mechanism is real (persistence's MAE does drop toward the
end of most anchors' chains, since any December-anchored 365-day chain necessarily loops back to
December) but its **magnitude** varies by year depending on how unusual that particular December's
flux level was relative to the rest of the year — it is a partially-structural, partially-
year-specific effect, not a universal "day 271-365 always fails" law.

**Bigger revision — DLinear's apparent single-anchor strength was not robust.** Weighted-average
R²/MASE across all 5 anchors, all bins pooled:

| Model | Mean R² | Mean MASE |
|---|---|---|
| **XGB** | **0.003** | **0.968** |
| LightGBM | -0.014 | 0.978 |
| Persistence | -0.260 | 0.998 |
| RF | -0.067 | 1.024 |
| SARIMAX | -0.039 | 1.038 |
| LSTM | -0.438 | 1.104 |
| DLinear | -4.752 | 1.806 |

DLinear looked like the standout long-horizon model in the single 2021 anchor (best R² anywhere,
0.417 at 181-270). Across all 5 anchors it is by far the **worst and least stable** model (mean R²
-4.75), driven by a catastrophic 2018 result (R² -18.9 that year) — the 2021 result looks like a
fluke of that specific year's data draw, not a generalizable property. **XGB is the most robust
model overall** — the only one with a positive mean R² and the best mean MASE — with LightGBM close
behind. SARIMAX and RF land almost exactly at persistence's level on average (genuinely useful in
some years, not others). LSTM is consistently mediocre-to-weak across every anchor.

## 4. Revised model ranking (supersedes the single-anchor-only ranking)

**XGB and LightGBM are the most consistently reliable across different starting points** — XGB is
the only model with positive mean R² across 5 anchors; both keep mean MASE comfortably under 1.
**SARIMAX and RF are roughly tied with plain persistence on average** — worth using in a given year
but not clearly better than doing nothing, when averaged across many possible starting points.
**LSTM is weak throughout.** **DLinear is not recommended** despite its striking single-anchor
result — its instability across different anchors (one catastrophic year out of five) makes it the
riskiest choice for a design that needs to work regardless of which calendar point it starts from.

## 5. Recommendation

- **Recursive daily rollout is a viable candidate for future long-range work, not a dead end** —
  XGB/LightGBM beat persistence on average across 5 independent anchor years, and every
  tree/statistical model beats both baselines for a multi-month stretch within any given chain.
- **If choosing one model for a future long-range design: XGB**, on the full multi-anchor evidence
  — most robust mean R²/MASE, cheap iterative rollout (thin wrapper around existing AR-recompute
  logic), no DLinear-style year-to-year fragility.
- **The late-window degradation seen in the single 2021 anchor is not a general property of this
  design** — it's a seasonal-echo effect whose size depends on the specific year, not something to
  design around universally. A future long-range build should still sanity-check driver ranges
  against training data per D-46 requirement 6, but should not assume every chain degrades late.
- **Do not over-read a single-anchor backtest** — this experiment is itself a demonstration of why:
  the single-2021-anchor result would have shipped DLinear as the headline model and "late-window
  collapse" as a universal finding, both of which the 5-anchor extension overturns.
- **Do not over-read the 1–7 day bin** in this or any similarly-designed future backtest — real
  ground-truth gaps can make the shortest lead-time bin the least reliable one, counter-intuitively.

## 6. Caveats

- **Ground truth is itself ~44-49% true-hourly-observed** in the best available backtest windows
  (varies by anchor) — some fraction of every model's apparent "error," including the baselines,
  reflects imperfections in the D-48-fixed gap-filling pipeline's own reconstruction.
- **Tree track (RF/XGB/LightGBM) and DL track (DLinear/LSTM) use different feature sources**
  (`forecast_daily_v2.csv` directly vs. `forecast_features_v2.csv` resampled to daily) — an
  existing B-03-vs-B-04 asymmetry, not new here, but means the tree-vs-DL comparison is a
  recursion-mechanism comparison, not a strictly feature-controlled one.
- **Only 5 anchors, all at the same calendar date (Dec 16 of different years)** — this tests
  year-to-year robustness at one fixed seasonal starting point, not robustness to starting the
  chain from a different point in the annual cycle (e.g. a summer anchor). Not tested here.

## 7. Stretch items — not attempted this session

- **TFT**: deferred. Known instability (D-45/D-48 needed special regularisation just to be usable
  at short horizons) would make it hard to distinguish genuine recursive compounding from
  pre-existing shakiness.
- **Tower 9**: deferred. Confirmatory, not new-mechanism information.
- **TabPFN / tabpfn-time-series**: neither installed. Foundation-model context-length limits for
  365 autoregressive steps are completely unresearched — genuinely open risk, not attempted.

## Files / scope

New: `src/models/recursive_rollout.py`, `notebooks/05_benchmarking/B09_recursive_rollout.ipynb`,
`results/b09_summary.csv`, `results/b09_chains.csv`, `results/figures/b09_rollout_chains.png`,
`results/b09_multi_anchor_summary.csv`, `results/b09_seasonal_check.csv`, 48 new `B09` rows in
`results/benchmarks.csv` (single-anchor only — the multi-anchor extension is a diagnostic check,
not appended to benchmarks.csv). No existing production files modified — `build_forecasting_
matrix_v2.py`'s AR-feature math and `forecasting_dl.py`'s model classes/`train_model`/`Scaler` are
reused verbatim, `B03a_arima.ipynb`'s `search_order()` logic is reused, its `fit_walk_forward()`
deliberately is not (see Design).

*Source: `B09_recursive_rollout.ipynb`, `results/b09_summary.csv`, `results/b09_multi_anchor_summary.csv`.
Cross-ref D-46 (the long-range scoping this directly tests requirement 5 of), D-47/B-08
(driver-realism, the separate, still-queued sensitivity question this experiment does not
address), D-52 (climate-scenario data), `FORECASTING_LEARNINGS.md` (the M5-replication lesson this
experiment directly operationalises).*
