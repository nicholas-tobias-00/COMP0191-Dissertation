# B-10 — recursive-rollout improvements: does anything fix the spike-blindness?

**Notebook:** `B10_daily_improvements.ipynb` (single-anchor smoke test, 2021-12-16). **Results:**
`results/b10_ar_blend_summary.csv`, `results/b10_ensemble_summary.csv`, `results/b10_h1_retrain_summary.csv`
(single-anchor bins); `results/b10_ar_blend_multi_anchor.csv`, `results/b10_ensemble_multi_anchor.csv`,
`results/b10_h1_retrain_multi_anchor.csv` (5-anchor extension, 2018–2022 — the headline verdicts below).
**Extended (backward-compatibly):** `src/models/recursive_rollout.py`'s `tree_rollout` gained optional
`alpha`/`clim_series` params. Multi-anchor extension run via an ad-hoc script (not committed, same
precedent as B-09/F-09a/F-09b), reusing `recursive_rollout.py`/`forecasting_dl.py` unmodified with the
anchor as a loop parameter.

B-09 (D-53) established that recursive daily rollout does not catastrophically compound error over a
365-day horizon, but R² stayed poor for most models (multi-anchor mean: XGB best at 0.003) even though
MASE mostly beat persistence — the classic "MASE<1 alongside near-zero/negative R² = spike-tail
signature" (D-44b). The user asked for concrete improvement ideas; three were tested here, each reusing
B-09's own machinery:

1. **Blended AR** — blend each tree model's recursive memory with day-of-year climatology
   (`alpha*pred + (1-alpha)*climatology` fed back into history). alpha ∈ {1.0, 0.5, 0.0}.
2. **Ensemble** — mean (unweighted and MASE-weighted, weights frozen from B-09's own multi-anchor
   MASE) of RF+XGB+LightGBM+SARIMAX.
3. **H=1 DL retrain** — retrain DLinear/LSTM natively for single-step-ahead instead of reusing the
   H=14-trained model's first output.

**Design, unchanged from B-09**: single anchor(s), Tower 4, real perfect-foresight `fx_` drivers
throughout, lead-time-binned evaluation against real `y_observed`. Per B-09's own headline lesson
("don't trust a single anchor"), every verdict below is drawn from the 5-anchor (2018–2022) sweep, not
the single-anchor smoke test alone.

## 0. Backward-compatibility check

Before trusting `alpha=0.5`/`alpha=0.0`, confirmed `alpha=1.0, clim_series` given reproduces B-09's
original chain **bit-for-bit**: max abs diff = 0.0000000000 for all three tree models against
`results/b09_chains.csv`. The new optional parameters do not alter existing callers' behaviour.

## 1. Part 1 — blended AR: fails, cleanly and monotonically

Multi-anchor mean R² (n-weighted across bins, averaged across 5 anchors):

| Model | alpha=0.0 | alpha=0.5 | alpha=1.0 (= B-09) |
|---|---|---|---|
| XGB | -0.114 | -0.050 | **0.003** |
| LightGBM | -0.114 | -0.067 | **-0.014** |
| RF | -0.106 | -0.081 | **-0.067** |

Pure recursion (alpha=1.0) beats every blend for all three models, monotonically. Blending in
climatology only dilutes signal the recursive chain already captures — **this idea does not improve
on B-09**, and generalizes from the single-anchor smoke test (which showed the identical ordering).

## 2. Part 2 — ensemble: a modest, genuine win

Multi-anchor mean R²/MASE:

| Model | Mean R² | Mean MASE |
|---|---|---|
| **Ensemble_unweighted** | **0.012** | 0.975 |
| Ensemble_MASEweighted | 0.011 | 0.975 |
| XGB (best individual) | 0.003 | **0.968** |
| LightGBM | -0.014 | 0.978 |
| SARIMAX | -0.039 | 1.038 |
| RF | -0.067 | 1.024 |

Both ensemble variants **beat the best individual model (XGB) on R²** (0.012/0.011 vs 0.003) —
the best mean R² of any configuration tested in B-09 or B-10. MASE ticks up marginally (0.975 vs
0.968, still comfortably <1). Unweighted and MASE-weighted ensembles perform almost identically
(the frozen B-09 MASE weights are tightly clustered — 0.24–0.26 each — so weighting barely
differs from a plain mean). **This is the one idea that clearly helps, though the gain is small
in absolute terms.**

## 3. Part 3 — H=1 DL retrain: mixed, model-dependent

Multi-anchor mean R²/MASE, H=1 retrain vs. B-09's original H=14-truncated chains:

| Model | H=1 Mean R² | H=1 Mean MASE | H=14-orig Mean R² (B-09) | H=14-orig Mean MASE (B-09) |
|---|---|---|---|---|
| LSTM | **-0.364** | **1.073** | -0.438 | 1.104 |
| DLinear | -1.729 | 1.542 | **-1.460** | **1.580** |

**LSTM-H1 is a genuine, consistent improvement** on both R² and MASE — still negative R² overall,
but clearly better than reusing the H=14 model's first output. **DLinear-H1 is worse on R²** despite
a marginally better MASE — DLinear's instability (already flagged in B-09 as anchor-dependent and
not robust, mean R² -4.752 across anchors) persists under H=1 retraining. Retraining for the native
horizon does not rescue DLinear specifically; it helps LSTM specifically. Not a uniform DL fix.

**Single-anchor smoke test showed the opposite ranking for DLinear** (H1 dramatically fixed the
1–7/8–30 bins at that one anchor) — another instance of DLinear's single-anchor behaviour not
generalizing, consistent with B-09's own finding about this same model.

## 4. Revised picture

Of the three ideas: **ensemble is a small real win**, **blended AR is a clean failure**, and
**H=1 retrain helps LSTM but not DLinear**. None of these closes the R² gap to a genuinely good
result — the best mean R² across all of B-09+B-10 is still only 0.012 (ensemble). The spike-blindness
identified in B-05/B-06/B-07/B-09 is not fixed by any of these three ideas; at best it is nudged.
This matches the honest-reporting norm established in D-42/43/44/53 — report the modest gain plainly
rather than oversell it.

## 5. Recommendation

- **If deploying a single recursive daily rollout, use the unweighted ensemble of
  RF+XGB+LightGBM+SARIMAX** — the best mean R² found across B-09 and B-10, cheap (four already-fit
  models, a mean), and doesn't require the frozen-weight bookkeeping the MASE-weighted variant needs
  for near-identical performance.
- **Do not use blended AR** — no configuration tested beats pure recursion.
- **LSTM benefits from H=1 retraining if LSTM is used at all** — but LSTM remains the weaker DL
  model overall (per B-09), so this is a second-order recommendation.
- **Do not add H=1 retraining as a blanket DL fix** — it is model-specific (helps LSTM, hurts
  DLinear), so it is not a substitute for choosing tree/statistical models as B-09 already recommends.

## 6. Caveats

- Same ground-truth caveat as B-09: target-window real-day coverage is 61–93% by the day-rule
  depending on anchor year, with true hourly-observed density lower still.
- Ensemble weights (MASE-weighted variant) are frozen from B-09's own multi-anchor MASE values, not
  re-derived from B-10's evaluation window — avoids circularity, but means the weighting is coarse
  (all four models cluster within 0.24–0.26) and essentially indistinguishable from an unweighted mean.
- Only 5 anchors, all at the same calendar date (Dec 16 of different years) — same limitation as B-09,
  not re-tested here.

## Files / scope

New: `notebooks/05_benchmarking/B10_daily_improvements.ipynb`, `results/b10_ar_blend_summary.csv`,
`results/b10_ensemble_summary.csv`, `results/b10_h1_retrain_summary.csv` (single-anchor),
`results/b10_ar_blend_multi_anchor.csv`, `results/b10_ensemble_multi_anchor.csv`,
`results/b10_h1_retrain_multi_anchor.csv` (5-anchor), 102 new `B10` rows in `results/benchmarks.csv`
(single-anchor smoke-test rows only — the multi-anchor extension is a diagnostic check, not appended,
matching B-09's own precedent). Extended: `src/models/recursive_rollout.py` (`tree_rollout` gained
optional, default-preserving `alpha`/`clim_series` params — verified bit-for-bit backward compatible).
No existing production files modified.

*Source: `B10_daily_improvements.ipynb`, `results/b10_*_summary.csv`, `results/b10_*_multi_anchor.csv`.
Cross-ref D-53/B-09 (the baseline this improves on), D-46 (the long-range scoping this precursor work
supports), `FORECASTING_LEARNINGS.md` (the improvement ideas' M5-replication lineage).*
