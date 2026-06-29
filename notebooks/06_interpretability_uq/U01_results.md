# U-01 — Forecast uncertainty quantification (FC-03, D-40)

**Notebook:** `U01_uncertainty.ipynb` · **Models:** `src/models/forecasting_dl.py` (LSTMQuantile + pinball) · **Results:** `results/fc03_uq_summary.csv`; 54 `FC-03` rows in `benchmarks.csv`; figures `results/figures/fc03_*`.

Calibrated 90% prediction intervals for the production forecasters, three paradigms — **split-conformal** (RF hourly, DLinear daily; calibrated 2021, Mondrian per horizon), **quantile XGBoost** (`reg:quantileerror`), **LSTM-pinball** — on Towers 4/9. Metrics: **PICP@90%** (coverage; target 0.90), **MPIW** (width), **pinball loss**.

## Headline — calibrated but *wide*; the spikes are irreducibly uncertain

| method | mean PICP | mean MPIW | mean pinball |
|---|---|---|---|
| **conformal (RF hourly)** | **0.87** | 240 | — |
| **conformal (DLinear daily)** | **0.88** | 153 | — |
| quantile XGB | 0.84 | 153 | **13.0** |
| LSTM-pinball | 0.77 | 155 | 15.5 |

- **Conformal is the most reliable** — coverage sits near nominal (RF 0.88–0.89 hourly; DLinear 0.84–0.88 daily), as its split-conformal guarantee implies; the price is the **widest** intervals (RF ~227–261 nmol).
- **Quantile-XGB is the best calibration–sharpness trade-off** — slightly lower coverage (~0.84) but the **sharpest** intervals (hourly 166–180) and the **best pinball loss** (13.0). The decision-useful default.
- **LSTM-pinball under-covers** (0.73–0.82 hourly, down to **0.62 at 14-day**) and has the worst pinball — consistent with the DL models trailing in B-02. **Not the UQ method to trust.**

## What the intervals say (the honest UQ story)

1. **Intervals are wide** — ~150–260 nmol (hourly) / ~100–160 (daily) around fluxes whose typical magnitude is tens of nmol. CH₄ at a grazed pasture carries **large aleatoric uncertainty**.
2. **Even wide intervals miss the biggest spikes.** The fan chart (`fc03_fanchart_T4_h24.png`) shows the 90% band covering the baseline, but the **extreme episodic spikes (≈600–1520 nmol) fall outside it** — the direct cause of the slight under-coverage.
3. **Coverage degrades with horizon for the quantile models** (LSTM daily 0.62 @14 d) as the upper tail gets harder; **conformal stays stable** (per-horizon recalibration).
4. This is the **quantified motivation for spike-aware modelling**: the uncertainty is concentrated in *when a high-emission event occurs*, which a mean/quantile regressor on daily-resolution livestock cannot resolve.

## Recommendation
- **Report conformal intervals** for guaranteed-coverage statements; **use quantile-XGB** when sharper, decision-useful bands are needed. Drop LSTM-pinball (under-covers).
- For the **digital shadow (07)**: ship forecasts **with conformal 90% bands**; the band *width itself* is a useful signal of spike risk.

## Figures
- `fc03_coverage_sharpness.png` — PICP & MPIW vs horizon, all methods (T4 hourly).
- `fc03_fanchart_T4_h24.png` — RF 24h forecast + 90% conformal band over a 2022 window (spikes burst the band).

## Caveats / next
- Conformal assumes exchangeability; the 2021→2022/23 temporal shift explains coverage sitting just under 0.90.
- Untuned quantile models; CRPS (full predictive distribution) deferred — pinball over 3 quantiles is the proxy.
- **Next: spike-aware modelling** (two-stage hurdle: occurrence → magnitude) — the UQ here shows the spike tail is where the uncertainty (and the opportunity) lives.

*Source: `U01_uncertainty.ipynb`, `results/fc03_uq_summary.csv`, `benchmarks.csv` (FC-03). Decision D-40.*
