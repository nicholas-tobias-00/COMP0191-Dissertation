# B-11 — monthly-resolution rollout + downscaling back to daily

**Notebook:** `B11_monthly_rollout.ipynb` (single-anchor smoke test, 2021-11-01 monthly /
2021-12-16 daily). **Results:** `results/b11_monthly_summary.csv`, `results/b11_downscaled_summary.csv`
(single-anchor); `results/b11_monthly_multi_anchor.csv`, `results/b11_downscaled_multi_anchor.csv`
(5-anchor extension, 2018–2022 — the headline verdicts below). **New:**
`src/features/build_forecasting_matrix_monthly.py` (emits `data/Hourly/forecast_monthly_v2.csv`).
**Extended:** `src/models/recursive_rollout.py` gained `ar_features_for_month`, `monthly_rollout`,
`moy_climatology`, `downscale_monthly_to_daily`. Multi-anchor extension run via an ad-hoc script
(not committed, same precedent as B-09/B-10), reusing each anchor year's own B-09 daily chain
(`results/b09_chains_anchor{yr}.csv`) as that year's downscaling shape template.

B-09/B-10 (D-53/D-54) found recursive daily rollout's R² stays modest because models miss CH4's
large spikes. M5's own hierarchy findings show coarser aggregates score better — a single missed
spike-day matters far less to a monthly mean than to that one day's value. This tests whether
that holds here, and critically, **whether the improvement survives being downscaled back to a
daily series** directly comparable to B-09/B-10 (the actual deliverable, since the long-range work
this precursor supports needs day-level granularity eventually).

**Model roster: SARIMAX + RF/XGB/LightGBM only.** DLinear/LSTM scoped out — only ~90 monthly rows
per tower, too thin for a from-scratch DL window regime, a flagged stretch item not silently
dropped (same precedent as B-09 §7).

**Anchor alignment**: the daily anchor (2021-12-16) has no clean monthly equivalent, so the
monthly anchor is the last *fully complete* month before it (2021-11-01) — using November avoids
leaking December 2021's post-anchor days into "training". 13 months are forecast (Dec 2021 –
Dec 2022), spanning the same range as B-09/B-10's 365-day daily window.

## 0. Downscaling exactness check

Before trusting any downscaled number, confirmed `downscale_monthly_to_daily`'s recentering is
exact by construction: the downscaled daily series' own monthly mean equals the input monthly
prediction, for every month, every model — max abs diff = 0.0000000000.

## 1. Monthly-native evaluation: a real, substantial improvement

Multi-anchor mean R²/MASE (evaluated directly against monthly-aggregated real `y_observed`):

| Model | Mean R² (monthly) | Mean MASE (monthly) | Mean R² (B-09 daily) | Mean MASE (B-09 daily) |
|---|---|---|---|---|
| LightGBM | **0.156** | **0.724** | -0.014 | 0.978 |
| XGB | 0.147 | 0.722 | 0.003 | 0.968 |
| RF | 0.050 | 0.813 | -0.067 | 1.024 |
| SARIMAX | -0.120 | 0.888 | -0.039 | 1.038 |

Every tree model's monthly-native R² is dramatically better than its daily-native equivalent
(LightGBM: -0.014 → 0.156; XGB: 0.003 → 0.147) and MASE improves substantially too (comfortably
<0.9 for all three trees). **This directly confirms the M5-hierarchy prediction**: coarser
temporal aggregation genuinely dampens the spike-miss error that dominates the daily-resolution
R². SARIMAX is the one exception — its monthly R² (-0.120) is actually worse than its daily R²
(-0.039), the only model where this doesn't hold.

## 2. Downscaled-to-daily: the improvement does NOT survive

Multi-anchor mean R²/MASE, after downscaling back to a daily series and evaluating on the
identical `bin_metrics` framework as B-09/B-10:

| Model | Downscaled Mean R² | Downscaled Mean MASE | B-09 original Mean R² | B-09 original Mean MASE |
|---|---|---|---|---|
| XGB | -0.000 | 0.980 | 0.003 | 0.968 |
| LightGBM | -0.016 | 0.989 | -0.014 | 0.978 |
| RF | -0.064 | 1.023 | -0.067 | 1.024 |
| SARIMAX | -0.244 | 1.104 | -0.039 | 1.038 |

**Essentially unchanged from B-09's own daily-native numbers** for RF/XGB/LightGBM (differences in
the third decimal place), and **worse for SARIMAX** (-0.244 vs -0.039). The dramatic monthly-native
gain from §1 evaporates almost entirely once downscaled back to daily.

By lead-time bin (mean R² across 5 anchors, downscaled vs. B-09 original):

| Model | Bin | Downscaled R² | B-09 original R² |
|---|---|---|---|
| XGB/LightGBM/RF | 271-365 | 0.05–0.06 | -0.51 to -0.59 |
| XGB/LightGBM/RF | 181-270 | 0.16–0.18 | 0.22–0.34 |
| XGB/LightGBM/RF | 8-30 | -0.85 to -1.18 | -0.17 to -0.73 |
| XGB/LightGBM/RF | 1-7 | -1.5 to -2.8 | -0.6 to -3.1 |

**The late-window bin (271-365) genuinely improves** — a real, consistent win across all four
models, several flipping from clearly negative to slightly positive. **181-270 is roughly a wash**
(slightly worse for trees, similar for SARIMAX). **The short/mid bins (1-7, 8-30) get *worse* on
average across 5 anchors** — the opposite of what §1's monthly-native result would suggest.

## 3. Why the gain doesn't transfer — mechanism, not a bug

`downscale_monthly_to_daily` reuses the daily template's own within-month **shape** unchanged and
only recenters its **monthly mean** to match the monthly model's independent prediction
(`daily_synth[d] = daily_template[d] - mean(daily_template over month) + monthly_pred[month]`).
This means:

- Any within-month spike-miss error is **inherited unchanged** from the daily template — the
  monthly model never sees individual days, so it cannot fix a daily model's specific spike misses.
- The only thing downscaling can change is the **coarse month-to-month bias**. Where B-09's daily
  models already capture the seasonal trend reasonably (they train on AR + seasonal `fx_`
  features), this correction adds little — and where the monthly model's own seasonal fit
  disagrees with the daily model's, it can actively subtract signal (visible in the 8-30/1-7 bins'
  regression).
- The late-window improvement is the one place this mechanism clearly helps: B-09's daily models
  are weakest there (see D-53's addendum — the seasonal-echo effect), so an independently-derived
  monthly bias correction has real room to help, and does.

This is exactly the caveat flagged in advance in the B-11 design: **"this makes B-11's evaluation
not fully independent of the daily models' own shape errors"** — confirmed empirically, not just
a theoretical concern.

## 4. Recommendation

- **Monthly resolution is a genuinely better way to evaluate/report long-range CH4 forecasts** —
  if a future long-range design can operate at monthly granularity natively (e.g. a digital-shadow
  scenario that only needs monthly aggregates), use it: the R²/MASE improvement is real and large.
- **Do not expect a monthly model to fix a daily model's spike-blindness via downscaling** — this
  hybrid-calibration method, as designed, cannot transfer the monthly gain to genuine daily-level
  accuracy. If daily-level output is required, B-10's ensemble (D-54) remains the best available
  daily-resolution result (mean R²=0.012).
- **The late-window bin is the one place downscaling helps** — worth combining with B-09's daily
  chain specifically for that horizon if a hybrid daily/monthly reporting scheme is ever built.
- A genuinely better downscaling method (e.g. distributing the monthly total via a learned daily
  profile rather than reusing an existing daily model's own errors) is future work, not attempted
  here — out of scope for this bounded experiment.

## 5. Caveats

- Same ground-truth caveat as B-09/B-10: target-window real-day coverage varies by anchor year.
- **Downscaling is not independent of the daily template's own errors** (see §3) — this is the
  central, load-bearing caveat of this entire experiment, not a minor footnote.
- Monthly anchor (last complete month before the daily anchor) introduces a small edge mismatch
  at the very start/end of the 13-month forecast window vs. the exact 365-day daily window.
- Only 5 anchors, same calendar date each year (Dec-16-equivalent) — same limitation as B-09/B-10.
- SARIMAX is the one model where monthly-native R² is *worse* than daily-native — not explained
  further here (open question, not investigated given bounded scope).

## Files / scope

New: `src/features/build_forecasting_matrix_monthly.py`, `data/Hourly/forecast_monthly_v2.csv`,
`notebooks/05_benchmarking/B11_monthly_rollout.ipynb`, `results/b11_monthly_summary.csv`,
`results/b11_downscaled_summary.csv` (single-anchor), `results/b11_monthly_multi_anchor.csv`,
`results/b11_downscaled_multi_anchor.csv` (5-anchor), 24 new `B11` rows in `results/benchmarks.csv`
(single-anchor smoke-test rows only, matching B-09/B-10's precedent). Extended:
`src/models/recursive_rollout.py` (`ar_features_for_month`, `monthly_rollout`, `moy_climatology`,
`downscale_monthly_to_daily` — all new functions, existing daily functions untouched). No existing
production files modified.

*Source: `B11_monthly_rollout.ipynb`, `results/b11_*_summary.csv`, `results/b11_*_multi_anchor.csv`.
Cross-ref D-53/B-09 (source of the daily downscaling templates), D-54/B-10 (the best daily-native
result this does not beat), `FORECASTING_LEARNINGS.md` (the M5-hierarchy lesson this operationalises).*
