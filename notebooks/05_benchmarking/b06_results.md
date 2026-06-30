# B-06 — spike-aware two-stage hurdle model (occurrence x magnitude, D-43)

**Notebook:** `B06_spike_hurdle.ipynb`. **Results:** `b06_summary.csv`; `B06` rows in `benchmarks.csv`
(new `precision`/`recall`/`f1` columns, mirrors how FC-03/D-40 added `picp`/`mpiw`/`pinball`).

Tests the structural alternative flagged after B-05 (arcsinh transform, D-42, negative): instead of squashing
spikes in the target, split the architecture into (1) a classifier predicting `P(spike|x)` (tower-specific
q90 of `y_observed`, frozen on 2018-2021 train years), (2) two magnitude regressors fit only on non-spike /
only on spike training rows, combined via the soft blend `P(spike)*spike_model + (1-P(spike))*base_model`.
Same data, partial pooling (T2+T4+T9 + dummies, D-30), CV, and B-03's already-tuned RF/XGB hyperparameters
(no new HPO this round, D-41 bounded-iteration norm). Both tracks per the session's scope decision.

## Headline — daily track is a clear loss; hourly Tower 4 is a genuine (small) win

### Daily track (Track B, the production forecaster): worse at every tower/horizon/algo

| daily R² | RF (B-03) | Hurdle-RF | XGB (B-03) | Hurdle-XGB |
|---|---|---|---|---|
| Tower 4, h=1 | 0.357 | **0.253** | 0.362 | **0.179** |
| Tower 4, h=14 | 0.270 | **0.154** | 0.259 | **-0.078** |
| Tower 9, h=1 | 0.388 | **0.130** | 0.324 | **-0.280** |
| Tower 9, h=14 | 0.342 | **0.059** | 0.275 | **-0.431** |

Mean R² delta vs B-03 (towers 4/9): **RF −0.12 (T4) / −0.27 (T9); XGB −0.27 (T4) / −0.65 (T9)** — Hurdle-XGB
even goes negative at T9. Tower 2 is closer to flat (RF +0.01 to +0.07, XGB negative), consistent with its
larger pooled training set softening the effect but not reversing it.

### Hourly track (Track A): Tower 4 wins at every horizon; Tower 9 is a wash (RF) to a loss (XGB)

| hourly R² | RF (B-03) | Hurdle-RF | XGB (B-03) | Hurdle-XGB |
|---|---|---|---|---|
| Tower 4, h=1 | 0.136 | **0.142** | 0.119 | **0.140** |
| Tower 4, h=24 | 0.049 | **0.110** | 0.030 | **0.078** |
| Tower 9, h=1 | 0.159 | 0.136 | 0.132 | 0.036 |
| Tower 9, h=24 | 0.087 | 0.091 | 0.064 | 0.009 |

Mean R² delta (towers 4/9): **Tower 4: RF +0.041, XGB +0.024** (positive at every one of the 5 horizons for
both algos). **Tower 9: RF +0.009 (essentially flat), XGB −0.043** (net negative).

## Mechanism — why: low classifier precision inflates non-spike error more than it shrinks spike error

The conditional RMSE breakdown (spike-day vs non-spike-day) shows the trade clearly. Hurdle **does** what it's
designed to do on the spike side — e.g. daily Tower 9 XGB: spike RMSE 153.5→106.8 (h=14), a **−46.7 nmol**
improvement. But non-spike RMSE moves the *other* way, by more: 31.7→74.9, a **+43.2 nmol** degradation. Since
~88-90% of test rows are non-spike, this is a net loss in aggregate R² — the same failure shape as B-05
(trading spike accuracy for bulk accuracy, or here the reverse), just reached by a different mechanism.

The root cause is the classifier's **precision**, not its recall. Daily-track precision is **0.25–0.42**
(i.e. 60-75% of "predicted spike" test rows are false positives) while recall is high (0.58–0.93) — `q90`
spikes are genuinely hard to anticipate from `ar_`/`fx_` features alone, so the classifier over-triggers to
catch them. Every false-positive blend pulls a non-spike test point's prediction toward the spike-only
regressor's (much larger-magnitude) output, inflating non-spike RMSE broadly. Daily Tower 9's spike-only
regressor is also fit on only ~45-100 pre-pooling training rows (the small-sample caveat flagged in the plan)
— a noisier regressor whose errors get blended into 10x as many non-spike predictions.

Hourly Tower 4 escapes this because (a) precision is somewhat better at short horizons (0.40–0.56 vs daily's
0.25–0.42) and (b) the spike-only regressor has ~1,200 pre-pooling training rows, an order of magnitude more
than daily Tower 9 — enough to be a genuinely useful specialist rather than a noisy one. Tower 9 hourly sits
in between (precision 0.30–0.49) and lands close to neutral for RF, negative for XGB.

## Verdict

**Negative for the production daily forecaster** — B-03's single regressor stays the production config; the
hurdle architecture does not clear the bar B-05 also failed to clear, via the inverse mechanism (non-spike
collateral damage instead of spike compression). **Mixed-to-positive for hourly Tower 4 only** — too narrow
and too marginal (+0.02 to +0.04 mean R², still well below the daily numbers in absolute terms) to justify a
second production path. Combined with B-05, this closes out both levers flagged after B-03/B-04 (transform,
hurdle) without finding a route past daily R² ≈ 0.36–0.39.

The classifier's low precision is the limiting factor, not the magnitude regressors themselves — a
hard-classify-then-route variant would not fix this (it would make the non-spike collateral damage worse, not
better, since soft blending already provides partial protection against false positives). A higher spike
threshold (q95) was ruled out at the planning stage on sample-size grounds and remains untested.

## Recommendation
- **Production forecaster unchanged**: B-03 enriched trees on the daily track remain the config of record.
- **Stop here on the spike lever** — both the target-transform (B-05) and architecture-split (B-06) attacks
  have now been tried and documented negative for daily forecasting. Revisit only if a materially better
  spike *classifier* becomes available (more predictive features, not more tuning — D-41 already exhausted
  HPO on this feature set).
- Move to the next agreed item: B-07 driver-realism sensitivity, then Phase 07 scenario analysis.

## Files / scope
Additive only — B03/B04/B05 artifacts, `forecast_features_v2.csv`, `forecast_daily_v2.csv` untouched. New
`precision`/`recall`/`f1` columns added to `benchmarks.csv` (NaN for non-Hurdle rows).

*Source: `B06_spike_hurdle.ipynb`, `b06_summary.csv`, `benchmarks.csv` (B06 rows). Decision D-43.*
