# I-02 -- Feature Importance (Native / SHAP / LIME) for the Recursive-Rollout Models

**Objective:** Determine what actually drives each B-10/B-13 recursive-rollout model's predictions,
using three complementary importance families across all 8 models (RF, XGB, LightGBM, SARIMAX,
Ensemble_unweighted, Ensemble_MASEweighted, TFT, TabPFN), all 3 towers (T2/T4/T9), and the full
5-anchor (2018-2022) sweep. Fresh methodology -- not based on the old `I01_feature_importance.ipynb`
(a different, unrelated harness).

## Key Findings

### 1. Livestock density (`fx_lsu_dens`) is the dominant driver, confirmed by every method that can see it

Native importance, global SHAP, and instance-level SHAP/LIME all independently rank `fx_lsu_dens`
at or near the top, across RF, XGB, LightGBM, and SARIMAX:

| Model | Method | #1 feature | #2 feature |
|---|---|---|---|
| RF | native | fx_lsu_dens (0.362) | ar_ch4_dlag1 (0.212) |
| XGB | native | fx_lsu_dens (0.269) | ar_ch4_dlag1 (0.118) |
| LightGBM | native | ar_ch4_dlag1 (259.8) | fx_lsu_dens (229.2) |
| SARIMAX | native (coef) | fx_USTAR_mean | fx_lsu_dens |
| RF | global SHAP | fx_lsu_dens (13.43) | ar_ch4_dlag1 (7.61) |
| XGB | global SHAP | fx_lsu_dens (13.62) | ar_ch4_dlag1 (8.20) |
| LightGBM | global SHAP | fx_lsu_dens (15.01) | ar_ch4_dlag1 (8.24) |

This directly reconfirms this project's own central thesis (first established at the
gap-filling/feature-engineering stage, F-01/F-02, D-27/D-30: livestock density is the #1 CH4
driver) -- now independently re-derived from the recursive-rollout models themselves, via three
different importance methods that were never used on this data before.

**A genuinely new finding: `fx_lsu_dens`'s SHAP importance *grows* at longer lead times, it doesn't
shrink.** Per-bin instance SHAP for RF at Tower 4 (mean |SHAP|, averaged across representative
days):

| Bin | fx_lsu_dens | ar_ch4_dlag1 |
|---|---|---|
| 1-7 | 7.58 | 4.34 |
| 8-30 | 9.04 | 3.80 |
| 31-90 | 9.06 | 3.23 |
| 91-180 | 24.94 | 7.92 |
| 181-270 | **38.43** | 11.96 |
| 271-365 | 21.32 | 9.87 |

Both grow with lead time, but `fx_lsu_dens`'s lead grows faster and never gets overtaken. The
likely mechanism: at short lead times, the AR features (`ar_ch4_dlag1` etc.) still carry real
recent history; as the rollout progresses and AR features become dominated by the model's own
compounding predictions (which drift toward a smoothed mean, per this project's well-documented
spike-blindness), livestock density -- an independent, perfect-foresight exogenous signal that
never degrades -- becomes relatively *more* informative, not less. This is a plausible, if not
fully proven, explanation for why B-15's tuning found the ensemble (heavier on trees, lighter on
the exogenous-driven SARIMAX) underperforms a model that leans harder on `fx_lsu_dens`.

**SHAP and LIME agree at the instance level, not just in aggregate.** Spot-check (RF, Tower 4,
anchor 2021, bin 31-90): SHAP top-3 = `fx_lsu_dens` (-8.88), `fx_wd_sin` (1.62), `ar_ch4_drm7`
(-1.60); LIME top-3 = `fx_lsu_dens` (+7.95), `ar_ch4_drm7` (3.10), `ar_ch4_dlag2` (1.65). Both
methods -- one global/game-theoretic, one local/surrogate-based -- independently agree on the
dominant feature and largely overlap on the runner-up, despite being fundamentally different
techniques. (Sign differs between the two by convention/baseline, not a contradiction.)

### 2. Tower comparison: SARIMAX's driver ranking is stable across towers; TabPFN's is not

**SARIMAX (fit per tower, genuinely comparable):** `fx_USTAR_mean` and `fx_lsu_dens` are the top 2
coefficients at **all three towers** (T2, T4, T9) -- a stable, cross-tower finding, reassuring given
SARIMAX's simpler linear structure.

| Tower | #1 | #2 |
|---|---|---|
| T2 | fx_USTAR_mean (75.1) | fx_lsu_dens (16.7) |
| T4 | fx_USTAR_mean (49.8) | fx_lsu_dens (21.7) |
| T9 | fx_USTAR_mean (51.4) | fx_lsu_dens (21.8) |

**TabPFN (permutation importance) tells a different, more concerning story:**

| Tower | #1 | #2 | Real data coverage |
|---|---|---|---|
| T4 | **fx_lsu_dens** (1.196) | fx_grazing_active (0.722) | 5/5 anchors usable |
| T9 | **fx_lsu_dens** (0.759) | fx_grazing_active (0.578) | 4/5 anchors usable |
| T2 | fx_TA_max (0.159) | fx_TS_lag28 (0.092) -- **fx_lsu_dens absent from top 4** | 1/5 anchors usable |

At Tower 4 and Tower 9, TabPFN's permutation importance lands on the same livestock-density story
as every other model. **At Tower 2, it doesn't** -- livestock density doesn't even appear in the
top 4; temperature/soil-lag features dominate instead. Given TabPFN's context uses real
`y_observed` (not gap-filled, by design, D-53/B-13's own convention), and Tower 2 has real data in
only 1 of 5 anchor windows, this is most plausibly explained by the same data-scarcity problem
already documented for Tower 2 throughout this project (B-13's addendum, B-15's addendum) -- with
almost no real signal to learn from, TabPFN's permutation importance likely reflects noise/context
degeneracy rather than a genuine ecosystem difference at Tower 2. **Not read as "Tower 2 has a
different driver structure"** -- read as "Tower 2 doesn't have enough real data for this method to
say anything reliable," consistent with this project's standing caution about that tower.

**TFT's VSN weights are reported as a genuine scope limitation, not a finding.** The encoder/decoder
channels are indexed generically (`enc_0`...`enc_N`, `dec_0`...`dec_N`) rather than mapped back to
real feature names, because `forecasting_dl.py`'s window-building doesn't preserve an accessible
column-name mapping at the point VSN weights are read out. The channel-level weights differ
substantially by tower (e.g. `dec_24` ranks highly at T4/T9 but not T2's top-4), but without a
name mapping this can't be turned into an actionable "TFT thinks X matters more at tower Y"
statement -- flagged as unfinished work, not silently glossed over.

### 3. Ensemble importance (additive combination) mirrors its constituents, as designed

Both `Ensemble_unweighted` and `Ensemble_MASEweighted`'s combined importance is dominated by
`fx_lsu_dens` and `ar_ch4_dlag1`, inheriting the same signal from RF/XGB/LightGBM (weighted 0.25
each, or by B-09's frozen MASE weights respectively) plus a smaller SARIMAX coefficient
contribution. This is expected by construction (SHAP additivity guarantees it, given the same
weights used for the point-forecast ensemble itself) -- not an independent finding, but a
consistency check that passed.

## Scope limitations (stated explicitly)

- **SARIMAX, TFT, TabPFN are excluded from KernelSHAP/LIME** (only get native importance). SARIMAX:
  already has an exact closed-form linear-effect view via its own coefficients -- a KernelExplainer
  pass would be both redundant and computationally intractable at this scale (re-running a 365-step
  `get_forecast` per perturbation sample). TFT/TabPFN: architecturally mismatched with the row-wise
  tabular explainer framework (TFT needs a full L-day encoder window per prediction; TabPFN is a
  one-shot whole-horizon forecast from a whole dataframe, not a per-row predictor).
- **Ensemble LIME uses the tree-weighted portion only**, with SARIMAX's contribution held as a fixed
  per-day offset rather than re-perturbed -- a stated approximation, not the full 4-member black box.
- **Instance-level SHAP/LIME are bounded to 6 representative days per anchor per tower** (one per
  lead-time bin, picked as the largest-|y_observed| real day in that bin) -- not all 365 days.
- **TFT's VSN channel names are unmapped** (see above) -- a genuine gap, not a hidden one.
- **Tower 2's results throughout are low-confidence** given 1/5 usable anchors -- consistent with
  every prior finding about this tower (B-13, B-15).

## Files

- `results/i02_native_importance.csv` -- native importance, every model/tower/anchor (3450 rows)
- `results/i02_shap_summary.csv` -- global SHAP, RF/XGB/LightGBM, pooled per anchor (660 rows)
- `results/i02_shap_instances.csv` -- per-instance SHAP, 5 models x 6 bins x 3 towers x 5 anchors (19800 rows)
- `results/i02_lime_instances.csv` -- per-instance LIME, same coverage (19800 rows)
- `src/interpretability/importance.py` -- shared native/SHAP/LIME dispatch module (new)
- `notebooks/06_interpretability_uq/I02_feature_importance_rollout.ipynb` -- design + worked example
- `notebooks/06_interpretability_uq/i02_multi_anchor_tower.py` -- full sweep script

## Cross-Reference

- **D-27/D-30**: original livestock-density-is-#1-driver finding (gap-filling/feature-engineering phase), reconfirmed here for the recursive-rollout models specifically
- **D-53/D-57**: B-09/B-13, source of the recursive-rollout models explained here
- **D-59 addendum**: B-15's Tower 2/9 cross-tower finding (tuned hyperparameters don't transfer to T9) -- this experiment's lead-time-growing-importance finding is a plausible partial explanation for why
- **D-4x (I-01, D-39)**: the original, unrelated feature-importance work -- explicitly not used as precedent for this experiment, left untouched
