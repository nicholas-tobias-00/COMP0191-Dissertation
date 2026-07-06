# U-02 -- Quantile-ML + Conformal Uncertainty for the Recursive-Rollout Models

**Objective:** Quantify B-10/B-13's recursive-rollout point forecasts' uncertainty, both via a
quantile-regression-style interval (adapts to known heteroscedasticity) and a conformal-calibrated
version (guarantees the interval's coverage is actually correct against held-out data). Fresh
methodology -- not based on the old `U01_uncertainty.ipynb`; `pinball`/`picp`/`mpiw` freshly
written in `src/evaluation/metrics.py`. All 8 models, all 3 towers, full 5-anchor sweep with
leave-one-anchor-out conformal calibration per lead-time bin.

## How to read these numbers

**Check PICP first, then compare MPIW/pinball only among models that pass.** PICP (coverage) close
to the 0.90 target means the interval is honest; a PICP far below target (e.g. 0.35) means the
interval is overconfident and *cannot* be trusted regardless of how narrow or how good its pinball
loss looks. Only once PICP is reasonable does a lower MPIW (narrower/sharper) or lower pinball
(the combined accuracy+calibration score) become a meaningful "better" signal.

## Key Findings

### 1. Conformal calibration works -- consistently, across every model type, at T4/T9

| Model | T4 raw PICP | T4 conformal PICP | T9 raw PICP | T9 conformal PICP |
|---|---|---|---|---|
| RF | 0.379 | **0.882** | 0.379 | **0.892** |
| XGB | 0.498 | **0.889** | 0.467 | **0.895** |
| LightGBM | 0.502 | **0.894** | 0.471 | **0.893** |
| SARIMAX | 0.835 | **0.889** | 0.923 | **0.895** |
| TFT | 0.830 | **0.893** | 0.851 | **0.894** |
| TabPFN | 0.869 | **0.896** | 0.718 | **0.887** |
| Ensemble_unweighted | 0.697 | **0.889** | 0.784 | **0.893** |
| Ensemble_MASEweighted | 0.694 | **0.891** | 0.780 | **0.893** |

*(TFT's raw column was originally "n/a" -- it launched without a native quantile mechanism,
conformal-wrapping its point forecast only. A follow-up round (below) added `TFTQuantile`, giving
it a real raw interval like every other model; these are the final numbers.)*

Every single model, regardless of its raw coverage (ranging from a badly-overconfident 0.38 for RF
to an already-decent 0.92 for SARIMAX at T9), converges to **0.88-0.90 PICP after calibration** --
almost exactly the 0.90 target. This is the headline result: leave-one-anchor-out conformal
calibration, applied per lead-time bin, reliably fixes miscalibration regardless of what produced
the original interval. This directly answers this session's own methodological question (conformal
vs quantile-ML) with real numbers: **quantile regression alone is not reliably calibrated (RF/XGB/
LightGBM's raw PICP of 0.38-0.50 proves this concretely), but conformal calibration recovers
honest coverage every time.**

### 2. Raw (pre-calibration) coverage reveals a genuine model-type split

**Tree models (RF/XGB/LightGBM) are consistently overconfident** -- raw PICP 0.35-0.50 across all
towers, meaning their native "90% interval" only actually covers 35-50% of real outcomes. This
mirrors U-01's own historical finding (LSTM-pinball under-covering) but for a different mechanism
(quantile-regression-forest / native quantile-objective trees here, not pinball-loss LSTM) --
the same underlying lesson holds: a model trained to predict a quantile does not automatically
produce a calibrated one.

**SARIMAX, TabPFN, and (once given a native quantile head) TFT are all already reasonably
calibrated even without conformal adjustment** (SARIMAX: 0.83-0.92 raw PICP; TabPFN: 0.72-0.87;
TFT: 0.76-0.85, including 0.76 at data-scarce Tower 2). This is a genuinely useful distinction for
future model choice: SARIMAX's Gaussian predictive distribution, TabPFN's in-context quantile
estimation, and TFT's pinball-loss-trained quantile head (once it has one) all produce honest
uncertainty "for free," while tree-based quantile methods (RF/XGB/LightGBM, all three trained the
same way here) need calibration layered on top to be trustworthy regardless of architecture family
sophistication -- the split is about the *training objective* (quantile/pinball loss vs the
tree-specific quantile mechanisms used here), not about which model is more complex.

### 3. After calibration, sharpness (pinball) separates the models -- Ensemble and RF are sharpest

| Model | T4 conformal pinball | T9 conformal pinball |
|---|---|---|
| RF | **10.09** | 11.18 |
| Ensemble_unweighted | **10.08** | **11.18** |
| Ensemble_MASEweighted | **10.08** | **11.19** |
| XGB | 10.23 | 11.49 |
| LightGBM | 10.18 | 11.46 |
| SARIMAX | 10.67 | 11.58 |
| TFT | 11.21 | 11.91 |
| TabPFN | 10.56 | 13.00 |

At both T4 and T9, the **ensembles and RF have the lowest (best) calibrated pinball loss** --
interesting given RF was the *worst*-calibrated model raw (PICP 0.38) -- once corrected, its
underlying point accuracy carries through to a sharp, honest interval. **TabPFN, despite good raw
coverage, has the worst calibrated pinball at Tower 9** (13.00) -- its intervals are wide (raw MPIW
125.8, conformal MPIW 190.3) without a matching accuracy payoff at that tower, consistent with
B-13's own finding that TabPFN's strength is bulk-error robustness (MASE), not necessarily
interval sharpness. **TFT, now that it has a real quantile head, has the highest (worst) calibrated
pinball at Tower 4** (11.21) despite genuinely good raw coverage (0.83) -- giving it a native
quantile mechanism closed the coverage gap but didn't make it sharp; consistent with this whole
project's standing finding that TFT is a solid but not class-leading model (D-45, D-57): honest
uncertainty, not the most accurate one.

### 4. Tower 2: conformal calibration is not possible -- reported honestly, not hidden

All conformal columns are NaN for Tower 2. This is not a bug: leave-one-anchor-out calibration
needs real residuals from the *other* anchors to build a margin, and Tower 2 has real
`y_observed` coverage in only 1 of 5 anchor windows (2018) -- for any anchor chosen as the test
anchor, the calibration pool (the other 4 anchors) contains at most one anchor with any real T2
data, and often zero. Only RAW metrics are meaningful for Tower 2, and even those should be read
with the same low-confidence caveat this project has applied to Tower 2 throughout (B-13, B-15).
For what it's worth, Tower 2's raw numbers show the same tree-overconfident /
non-tree-already-reasonable pattern as T4/T9 (SARIMAX raw PICP 0.89, TabPFN 0.81, TFT 0.76, trees
0.24-0.35) -- suggestive that the pattern is general, but not confirmed given the data scarcity.

### 5. Ensemble_unweighted vs Ensemble_MASEweighted: genuinely different now, but barely

An implementation bug (caught before finalizing these numbers) initially made the two ensemble
variants identical -- fixed so `Ensemble_MASEweighted` actually uses B-09's frozen per-model MASE
weights (~0.24-0.26 each, close to the unweighted 0.25) rather than accidentally reusing the
unweighted combination. The fix changes the numbers slightly (e.g. T4 raw PICP: 0.6965 unweighted
vs 0.6935 MASE-weighted) but the two remain close throughout, since B-09's frozen weights were
already nearly equal -- no meaningful UQ preference between the two ensemble variants.

## Visualizations

`results/figures/u02_fancharts/T{tower}_anchor{year}_{model}.png` (120 plots): actual/gap-filled/
predicted-median + shaded 90% interval. **Solid band = conformal-calibrated; hatched/dashed band =
raw (uncalibrated) fallback**, applied **per day, not per chain** -- a single chain can be
genuinely mixed (calibration available for some lead-time bins but not others, e.g. when the
specific test anchor lacks enough real `y_observed` in an early bin even though later bins and
other anchors have plenty). An earlier version of this script decided calibrated-vs-raw once for
the whole chain, which silently produced blank gaps wherever calibration was missing even though a
perfectly good raw interval existed for those exact days -- caught via user inspection of the
actual figures and fixed to fall back day-by-day. **A true blank now only appears where neither a
calibrated nor a raw interval exists at all** -- this happens for TFT specifically, which has no
native quantile mechanism to fall back to (by design, see below), so when its calibration is
missing for a bin (e.g. Tower 9's 2021 anchor in the last ~90 days, where that specific anchor has
zero real observations to verify against), there's genuinely nothing to show.

Visual check confirms the calibrated band widens during the high-variance spike season (Apr-Sep)
and narrows during quiet months -- genuine heteroscedasticity-awareness, not a flat uninformative
margin -- while still (correctly, for a 90% interval) missing some of the very largest spikes.

**Update:** TFT's "no raw fallback at all" limitation described above has since been closed (see
the TFTQuantile addendum below) -- TFT now has a real raw interval like every other model, so the
one class of genuine total-blank gap this experiment produced no longer occurs for TFT. The
specific case that originally exposed it (Tower 9, anchor 2021, previously blank for the last ~90
days) now shows the hatched raw-interval fallback there instead, confirmed by direct comparison of
the regenerated figure.

## Per-model quantile mechanism (recap)

- **RF**: quantile-regression-forest trick on the already-fitted point model (no retraining).
- **XGB/LightGBM**: 3 separately-fit quantile-objective models per anchor (q=0.05/0.5/0.95), same
  hyperparameters as B-10's point models otherwise (no new HPO).
- **SARIMAX**: `get_forecast().conf_int()` -- quantiles essentially for free.
- **TFT**: originally launched with the quantile head deferred (conformal-wrapping its point
  forecast only) -- a follow-up round added a real `TFTQuantile` class (pinball-loss-trained,
  same architecture as point TFT with the output head widened and non-crossing enforced via sort)
  plus a new `dl_rollout_quantile` rollout function, giving TFT the same native raw interval every
  other model has. See the addendum below.
- **TabPFN**: native quantile output via `tabpfn-time-series`'s own `quantiles=` parameter (a real
  bug was caught and fixed here: the library returns quantile columns keyed by the float value
  itself, e.g. `0.05`, not its string representation `'0.05'` -- a naive string-based lookup
  silently produced 100% NaN raw intervals before the fix).
- **Ensembles**: median = weighted mean of the 4 constituents' medians (same definition as B-10's
  point-forecast ensemble); raw interval = weighted mean of constituent q0.05/q0.95 bounds.
  Conformal-calibrated on top, same as everything else.

## Bugs found and fixed during this experiment (stated plainly)

1. **TabPFN quantile column-matching bug**: `str(q)` was compared against `preds.columns`, but the
   library returns float-keyed columns -- fixed in `recursive_rollout.py`'s `tabpfn_forecast()`.
2. **Ensemble_MASEweighted duplicate-of-unweighted bug**: both ensemble variants computed an
   unweighted (1/4 each) combination -- fixed to use B-09's frozen MASE weights for the MASE-weighted
   variant, matching B-10's own ensemble definition.
3. **Aggregate all-NaN-column display bug**: pandas' `.sum()` on an all-NaN column silently returns
   0.0, not NaN -- caused TFT/TabPFN's (at the time all-NaN) raw metrics to misleadingly print as a
   confident "0.0000" rather than "no data" -- fixed with an explicit all-NaN guard before aggregating.
4. **Fan-chart whole-chain fallback bug** (caught via direct user inspection of the generated
   figures, not automated testing): `u02_fanchart_plots.py` originally decided calibrated-vs-raw
   once per chain (`if margin.notna().any()`) rather than per day, so any chain with a *mix* of
   calibrated and uncalibrated bins (common -- see Tower 9/2022 above) silently showed a blank gap
   for the uncalibrated portion even when a perfectly good raw interval existed for those exact
   days. Fixed to fall back day-by-day; re-verified against the exact case that exposed it
   (Tower 4, anchor 2022, SARIMAX) and against a genuine no-fallback case (TFT) to confirm the fix
   doesn't paper over real gaps, only spurious ones.

## Addendum -- giving TFT a native quantile head (TFTQuantile)

**Motivation:** direct user inspection of the fan charts (not automated testing) surfaced that TFT
was the one model with a genuine blank-gap failure mode (no raw fallback when calibration was
unavailable for a bin). Rather than leave this as a permanent scope limitation, a follow-up round
added a real quantile mechanism for TFT, closing the gap.

**Design (backward-compatible by construction):** a new `TFTQuantile` class in
`forecasting_dl.py`, architecturally identical to the existing point-forecast `TFT` class (same
VSN/GRN/attention body, which is agnostic to output head width) with exactly the two edits the
codebase's own `LSTMQuantile`-vs-`LSTMSeq2Seq` precedent already established: the output head
widened from `nn.Linear(d_model, 1)` to `nn.Linear(d_model, nq)`, and the final `.squeeze(-1)`
dropped so the model returns `(B, H, Q)` instead of `(B, H)`. **Not wired into `build_model`** --
constructed directly by its one caller, mirroring `LSTMQuantile`'s own precedent (also never routed
through the dispatcher). `pinball_loss`/`train_quantile`/`predict_quantile` were already generic
over any 3-arg `(enc, dec, static)` model emitting `(B,H,Q)` (confirmed by reading their
implementations before making any change) -- reused verbatim, with one addition:
`train_quantile` gained the same optional `weight_decay`/`val_data`/`patience` early-stopping
parameters `train_model` already has (D-45), since TFT is known to overfit the bounded 30-epoch
budget without them; defaults preserve the original `LSTMQuantile` caller's (U-01's) exact
behavior. A new `dl_rollout_quantile` function mirrors `tree_rollout_quantile`'s already-verified
contract (feed back only the median into a single coherent AR history; record all quantiles per
day; enforce non-crossing via sort).

**Verified unaffected:** `git diff` on both modified files shows the existing `TFT` class,
`build_model`, `LSTMQuantile`, `pinball_loss`, `predict_quantile`, and `dl_rollout` completely
untouched (confirmed line-by-line, not just diff-stat). `B03b_tft.ipynb`, `B13_tft_tabpfn.ipynb`,
and `i02_multi_anchor_tower.py` (the three other TFT callers in the repo, confirmed exhaustively --
`B10_daily_improvements.ipynb` does not use TFT at all) still call the unmodified point `TFT` class
and are unaffected. A numeric before/after comparison of I-02's TFT VSN weights showed different
values between runs -- traced to a **pre-existing** characteristic of this codebase (TFT's initial
weights are never seeded; `torch.manual_seed` is only called inside `train_model`/`train_quantile`,
after model construction, so fresh Python process runs start from different random initial
weights), not a regression from this change.

**Result:** TFT now has real raw PICP/MPIW/pinball (Tables above, updated in place). Its raw
coverage (0.83 at T4, 0.85 at T9, 0.76 at T2) puts it in the same "already reasonably calibrated"
group as SARIMAX/TabPFN, not the overconfident tree-model group -- interesting given the tree
models here also use a from-scratch quantile mechanism (quantile-forest/quantile-objective) while
TFT's is pinball-loss-trained, suggesting the training objective (not architecture complexity)
drives calibration quality. Its calibrated pinball is the *worst* of any model at Tower 4 (11.21)
despite the good coverage -- a real, honestly-reported finding, not smoothed over: giving TFT a
quantile head fixed its calibration gap but didn't make it competitively sharp, consistent with
this project's standing view of TFT (D-45, D-57) as a solid-but-not-class-leading model.

## Files

- `results/u02_chains.csv` -- per-day median + raw q05/q95 for every model/tower/anchor (1825 rows/model/tower)
- `results/u02_summary.csv` -- raw vs conformal-calibrated PICP/MPIW/pinball per model/tower/bin (376 rows)
- `src/evaluation/metrics.py` -- `pinball`/`picp`/`mpiw` (new)
- `src/models/recursive_rollout.py` -- `RFQuantileAdapter`, `MultiModelQuantileAdapter`,
  `tree_rollout_quantile`, `sarimax_quantile`, `conformal_margins_by_bin`, `lead_time_bin`,
  `dl_rollout_quantile` (new); `tabpfn_forecast` extended with an optional `quantiles` parameter
- `src/models/forecasting_dl.py` -- `TFTQuantile` class (new); `train_quantile` extended with
  optional `weight_decay`/`val_data`/`patience` early-stopping parameters
- `notebooks/06_interpretability_uq/U02_uncertainty_rollout.ipynb` -- design + worked example
- `notebooks/06_interpretability_uq/u02_multi_anchor_tower.py` -- full sweep script (TFT block
  updated to use `TFTQuantile`/`train_quantile`/`dl_rollout_quantile`)
- `notebooks/06_interpretability_uq/u02_fanchart_plots.py` -- visualization script

## Recommendation

**For production uncertainty estimates: use conformal-calibrated intervals, not raw quantile
output, for any tree-based model** -- raw tree quantiles are demonstrably overconfident (PICP as
low as 0.35-0.50). **The calibrated Ensemble or RF give the sharpest (lowest-pinball) intervals at
both T4 and T9** once calibrated, making either a reasonable choice for production uncertainty
bands, consistent with B-10's ensemble already being the point-forecast recommendation. **SARIMAX
and TabPFN are useful when a calibration step isn't available or convenient**, since their raw
uncertainty is already reasonably honest. **Tower 2 cannot currently support calibrated
uncertainty estimates at all** -- this is a data-availability limit, not a methodology failure, and
should be treated the same way this project treats every other Tower-2 caveat: an open question
pending more held-out data.

## Cross-Reference

- **D-40 (U-01)**: the original, unrelated UQ work (different harness) -- not used as precedent, left untouched
- **D-53/D-57**: B-09/B-13, source of the models calibrated here
- **D-58/D-59**: B-14/B-15's tuning findings -- this experiment's UQ layer sits on top of those same models unchanged
- **I-02**: this experiment's companion (feature importance for the same models)
