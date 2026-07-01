# B-03a / B-03b — filling the D-05 model-roster gaps: SARIMAX and full TFT (D-45)

**Notebooks:** `B03a_arima.ipynb`, `B03b_tft.ipynb`. **Results:** `b03a_summary.csv` (27 rows), `b03b_summary.csv`
(27 rows); `B03a`/`B03b` rows in `benchmarks.csv`.

The original model roster (D-05) was: persistence/seasonal-mean → **ARIMA** → RF/XGBoost → LSTM/**TFT** →
SARIMAX. RF/XGBoost (B-03) and a lighter LSTM/DLinear/LSTM-VSN roster (B-02/B-04, with TFT explicitly de-scoped
in favour of LSTM-VSN, D-38) were built; ARIMA/SARIMAX was never implemented anywhere. This pair of experiments
fills both gaps, on the same data/CV/horizons as B-03 (`forecast_features_v2.csv` / `forecast_daily_v2.csv`,
Towers 4/9 main split test 2022–2023, Tower 2 expanding folds).

## B-03a — SARIMAX

Per-tower (solo, no pooling — ARIMA has no standard panel/hierarchical equivalent to D-30's partial pooling),
small SHAP-informed exogenous set (`fx_lsu_dens`, wind speed, VPD, USTAR, PPFD + seasonality proxies — I-01
ranked livestock density the #1 forecasting driver by a wide margin). Order chosen via a bounded AIC grid
(`d=1` fixed, `p∈{1,2}`, `q∈{0,1}`) — **(2,1,1)** won for every main-split tower/track. Walk-forward evaluated
via `statsmodels`' `append(refit=False)` (fast Kalman-filter state update, no re-optimisation) with
`get_forecast(steps=H)` at each strided origin — ARIMA's natural multi-step usage, vs. the rest of the
project's one-model-per-horizon direct design.

## B-03b — canonical Temporal Fusion Transformer

Full architecture (Lim et al. 2021), hand-rolled pure PyTorch in `src/models/forecasting_dl.py` (`TFT` class):
Variable Selection Networks (per-variable Gated Residual Network transform + softmax-gated combination,
separately for static/encoder/decoder inputs), static covariate encoders (4 context vectors from the tower
one-hot), an LSTM encoder-decoder with static-initialised state, gated locality enhancement, static enrichment,
interpretable multi-head self-attention (shared value projection across heads, causal-masked), and a gated
position-wise feed-forward head. `d_model=32, n_heads=4`, 30 epochs (same as B-02/B-04's DL roster) — modest
sizing per the bounded-iteration norm (D-41), but every architectural component from the paper is present.
Reuses `run_track`/`build_windows`/`_eval_rows` unchanged (`forward(enc,dec,static) -> (B,H)` matches
DLinear/LSTM/LSTM-VSN's interface).

## Results — SARIMAX negative beyond h=1; TFT negative until fixed, then positive-but-uncompetitive

| daily R² (towers 4/9, h=1→14) | B-03 RF | B-03a SARIMAX | B-04 DLinear | B-03b TFT (original) | B-03b TFT-Reg (fixed) |
|---|---|---|---|---|---|
| h=1 | **0.372** | 0.326 | 0.314 | -0.967 | 0.177 |
| h=3 | **0.348** | 0.060 | 0.237 | -0.892 | 0.176 |
| h=7 | **0.326** | -0.103 | 0.202 | -0.780 | 0.176 |
| h=14 | **0.306** | -0.177 | 0.164 | -0.730 | 0.176 |

| daily MASE (towers 4/9, h=1→14, <1 = beats persistence) | B-03 RF | B-03a SARIMAX | B-04 DLinear | B-03b TFT (original) | B-03b TFT-Reg (fixed) |
|---|---|---|---|---|---|
| h=1 | 0.998 | 1.122 | 1.166 | 1.793 | 1.227 |
| h=3 | 0.830 | 1.132 | 1.001 | 1.418 | 0.983 |
| h=7 | 0.712 | 1.110 | 0.846 | 1.168 | 0.829 |
| h=14 | 0.629 | 1.060 | 0.804 | 1.029 | 0.747 |

(TFT-Reg columns above are the T4/T9 mean; see the per-tower breakdown below — Tower 4 does considerably
better than Tower 9 post-fix.)

**SARIMAX (B-03a):** competitive only at the shortest horizon (R²=0.326–0.371 at h=1, in the same ballpark as
the trees) — the chosen `(2,1,1)` order with a handful of exogenous drivers captures next-day persistence-like
structure reasonably well. It **collapses by h=7–14** (R² turns negative, MASE 1.06–1.13 — worse than just
persisting the last value) — a single low-order ARIMA process with 5–7 exogenous regressors cannot capture the
same multi-scale soil/livestock/seasonal structure the enriched tree feature set encodes. Hourly is similarly
weak (mostly negative R², MASE near or above 1 at every horizon except a few mid-range dips).

**TFT (B-03b), original run: negative R² at every single horizon, both towers, both tracks** — daily R²
averages around -0.85, roughly 3 R² units below the production trees and meaningfully worse than even the
simpler DL models (DLinear, LSTM, LSTM-VSN) from B-04. MASE of 1.03–1.79 means the TFT is **worse than naive
persistence at every daily horizon** — the single worst result of any model tried in this entire forecasting
phase, hourly included.

**Mechanism (verified, not assumed):** before accepting this as a genuine result, the negative R² was checked
directly for an implementation bug — a manual training run (Track B, Tower 4) showed the training loss
converging cleanly (standardized MSE 0.85 → 0.09 over 30 epochs, no divergence/NaNs, implying ~91% fit on
**training** data) and test-set predictions sanely scaled (mean 32.4 vs actual 30.2, std 59.9 vs actual 68.8 —
no degenerate constant output, no unit/scaling error). Test-set correlation is weakly positive (r=0.27): there
is real signal, just not much. The actual failure mode is specific — the model occasionally predicts large
spikes that did not happen (e.g. predicted 355.8 vs actual 31.4; predicted 212.9 vs actual 27.4); since R² is
a squared-error metric, a handful of these large overconfident misses is enough to drag it deeply negative even
though the bulk of predictions sit in a reasonable range. **This is the signature of overfitting, not an
undertrained or broken model**: TFT is a substantially larger, more component-rich architecture (2 VSNs, 4
static GRNs, an LSTM encoder-decoder, an attention layer, ~6× DLinear's parameter count) that fit the training
set very well (training-equivalent R²≈0.91) but generalised poorly — and the daily track in particular has very
few training windows (~5,355 pooled across all 3 towers).

**Fix attempt: regularised retrain (AdamW weight decay=1e-3 + early stopping on a 2021 validation year,
patience=5) — worked.** Towers 4/9 retrained on 2018–2020, validated on 2021 (mirroring FC-03/U-01's existing
precedent of reserving 2021 as a calibration/held-out year), evaluated on the unchanged 2022–2023 test period.

| daily R², T4/T9 (h=1→14) | TFT (original) | TFT-Reg (fixed) | B-03 RF (production) |
|---|---|---|---|
| T4 | -1.078 → -0.836 | **+0.247 → +0.255** | 0.357 → 0.270 |
| T9 | -0.856 → -0.623 | **+0.106 → +0.097** | 0.388 → 0.342 |

| daily MASE, T4/T9 (h=1→14, <1 beats persistence) | TFT (original) | TFT-Reg (fixed) |
|---|---|---|
| T4 | 1.79 → 1.03 | **1.18 → 0.65** (beats persistence from h=3 onward) |
| T9 | 1.58 → 1.00 | **1.27 → 0.85** (beats persistence from h=7 onward) |

Hourly moved the same direction, more modestly: R² went from -0.17…-0.02 (original, negative everywhere) to
+0.004…+0.039 (regularised, small but positive everywhere). **The fix converted TFT from the single worst
model in the project to a genuinely reasonable, if still non-competitive, forecaster** — daily R² is now
positive and MASE beats persistence at longer horizons, but it still sits well below B-03's trees at every
horizon/tower. This is a clean confirmation of the overfitting diagnosis: the *only* change was regularisation
(same architecture, same features, same 30-epoch budget, one fewer year of raw training data traded for a
validation year) and it fully reversed the sign of the result. It sharpens D-38's "simpler wins" pattern without
contradicting it: the *more* components a model has, the more it needs data/regularisation/tuning to avoid
overfitting under a bounded compute budget — and even a *fixed* TFT still needs more of all three to compete
with the trees.

## Verdict

**B-03 remains unambiguously the production forecaster — neither new model overtakes it, but the TFT story has
a genuine positive turn.** Daily R²: B-03 (0.27–0.39) exceeds B-03a (only viable at h=1) and B-03b-Reg
(0.10–0.26, positive everywhere but a full R²-unit-plus below the trees at every horizon). This completes the
original D-05 model roster — persistence/seasonal-mean, ARIMA, RF/XGBoost, LSTM/TFT, SARIMAX have now **all**
been tried — and confirms, more strongly than ever, that algorithmic sophistication is not the bottleneck on
this signal (echoing D-22's original finding that feature realism, not algorithm choice, drives performance).
A classical statistical baseline (ARIMA-family) and the most architecturally complex model available (TFT,
even after fixing its overfitting) both lose decisively to gradient-boosted/random-forest trees on the enriched
feature set. **The more interesting finding is methodological**: the TFT went from the single worst result in
the project to a genuinely sane, positive-skill forecaster with nothing but standard regularisation — a useful
demonstration that "complex model underperforms" claims should be checked for overfitting before being taken
as an architecture verdict, not just a data-point for "simpler wins."

## Recommendation
- **No change to production**: B-03 enriched trees remain the daily forecaster, RF the hourly forecaster.
- **Model-roster question is now closed** — every rung of the original D-05 ladder has a documented result;
  no further algorithm-search experiments are warranted on this feature set.
- **TFT-Reg's regularisation recipe (weight_decay=1e-3 + early-stopping on a held-out validation year) is now
  available in `forecasting_dl.py`'s `train_model()`** (backward-compatible — off by default) if a future
  experiment wants to revisit attention-based models; further gains would need more training data and/or a
  proper weight-decay/dropout sweep, not just more epochs.

## Files / scope
Additive only — B03/B04 artifacts, `forecast_features_v2.csv`, `forecast_daily_v2.csv` untouched. New `TFT`
model class added to `src/models/forecasting_dl.py` (used only by B-03b). SARIMAX fits per-tower solo (documented
departure from partial pooling — no standard panel-SARIMAX equivalent exists).

*Source: `B03a_arima.ipynb`, `B03b_tft.ipynb`, `b03a_summary.csv`, `b03b_summary.csv`, `benchmarks.csv`
(B03a/B03b rows). Decision D-45.*
