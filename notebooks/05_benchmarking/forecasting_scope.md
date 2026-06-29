# Forecasting phase (05_benchmarking) — scope

The project's **novel contribution**: multi-step EC CH₄ flux *forecasting* at NWFP, benchmarked across model
families, with uncertainty quantification, feeding the digital-shadow scenario engine (07). Scoped after the
gap-filling + feature-engineering phases (R-01…R-03, 03b, F-01…F-08). Decision: **D-36**.

## Forecasting ≠ gap-filling (the key shift)

Gap-filling interpolated using data *before and after* each gap and leaned on **concurrent FCO₂** — the decisive
lever (D-22/D-26). Forecasting is **past-only and leak-free**, which removes that lever: FCO₂, GPP, Reco are EC
fluxes, **unknown at forecast time** → usable only as **lagged** features, never concurrent. Forecasting therefore
relies on **autoregressive lagged CH₄**, **future met** (weather forecast / scenario), and **planned
livestock/management**. Expect **materially lower R²** than the 0.35–0.49 gap-filling numbers — frame results around
**skill vs baseline**, not absolute R² (supervisor steer: improvement over baseline for an open system).

## Locked design decisions (user, this session)

1. **Two task tracks** (`Both`):
   - **Track A — hourly nowcast:** forecast hourly FCH₄ at horizons h ∈ {1, 6, 12, 24, 48} h (direct multi-horizon).
   - **Track B — daily-mean forecast:** forecast daily-mean FCH₄ at horizons d ∈ {1, 3, 7, 14} days.
2. **Driver-conditional** — future exogenous drivers are *supplied* (future met = weather-forecast/scenario; livestock
   & management = planned). Initially use **observed met as a "perfect-forecast" proxy** (optimistic upper bound;
   a degraded-driver sensitivity test is a later add). This directly serves the digital shadow (07).
3. **Train on gap-filled, evaluate on observed** — train/AR-features use the F-06/F-08 best gap-filled continuous
   CH₄ series (regular sampling for sequence models); **metrics computed only on genuinely observed timestamps**
   (Track B: only days with ≥ N observed hours, N TBD ~ 6).

## Inherited (already decided — do not re-litigate)

- **Model roster (D-05):** persistence/seasonal-mean baselines → RF/XGBoost → LSTM/TFT → SARIMAX; no architecture
  assumed superior; include a strong simple-DL baseline (DLinear/N-HiTS) per the Zeng 2023 caveat.
- **Temporal CV only (D-04):** train 2018–2021, test 2022–2023; rolling-origin backtest within test.
- **Partial-pooled multi-tower (D-30):** one global model + tower-indicator dummies.
- **Feature base (F-06/F-08):** livestock density + grazing flag, SWC/TS lags, pruned management, gap-filled met,
  **external per-catchment soil temperature (D-35)**, cyclical time. FCO₂/GPP/Reco **lagged-only** here.

## Data reality (verified this session)

- **Tower 4 & 9 are the forecasting test targets.** Test-period valid CH₄: T4 2022 76% / 2023 51%; T9 2022 44% /
  2023 61%. 
- **Tower 2 cannot be a forecasting test target** — no valid CH₄ after Jun 2019. It can only be a **pooled training
  donor** (2018–2019). 
- **Held-out 2024 is still empty** — the index now runs to 2025-01-02, but 2024 FCH₄ = 0% valid for all towers.
  The final-benchmark held-out window is unavailable until 2024 EC fluxes are published/downloaded (open item).

## Target & feature construction

1. **Continuous gap-filled CH₄ series** (per tower, training input + AR features): train the F-06/F-08 best RFm
   gap-filler on *all* observed CH₄ and fill every gap → persist `data/Hourly/fch4_gapfilled.csv`. (New precompute.)
2. **Forecasting feature matrix** at each origin t for horizon h:
   - AR: lagged gap-filled CH₄ {1,2,3,6,12,24,48,168 h} + rolling means (Track B: daily lags {1,2,3,7,14 d}).
   - Future exog (driver-conditional, at target t+h): gap-filled met + external soil temp + soil moisture + precip.
   - Planned: livestock density, grazing flag, management recency at t+h.
   - Lagged-only fluxes: FCO₂/GPP/Reco at t and earlier (never t+h).
   - Calendar: hour/doy cyclical; tower dummies (pooled).

## Evaluation

- **Metrics on observed targets only:** RMSE, MAE, R², MBE, per horizon (skill-decay curves).
- **Skill score** = % RMSE reduction vs persistence and vs seasonal-diurnal climatology (the improvement-over-baseline
  framing, analogous to improvement-over-MDS).
- Rolling-origin backtest across 2022–2023; per-tower + pooled.

## Uncertainty quantification (project aim)

- **Conformal prediction** (model-agnostic, wraps any point model) for calibrated intervals + **quantile outputs**
  from TFT / gradient-boosting quantile loss. Report interval coverage + sharpness. Start conformal (cheap, rigorous).

## Digital-shadow tie-in (07, later)

The driver-conditional forecaster + UQ *is* the scenario engine: feed counterfactual management/stocking/weather
→ CH₄ response distribution. Build in 07; here just keep the driver-conditional interface clean.

## Sequencing (deadline 1 Sept 2026)

1. ✅ **Precompute** — `build_fch4_gapfilled.py`, `build_forecasting_matrix.py` (Stage 0).
2. ✅ **Baselines + ML benchmark** — FC-01 (`B01`, RF/XGB + persistence/climatology). RF beats persistence; skill-vs-baseline is the metric.
3. ✅ **DL** — FC-02 (`B02`, hand-rolled DLinear/LSTM/LSTM-VSN, pure PyTorch, GPU; D-38). *Finding: complexity doesn't pay off — RF (hourly) / DLinear (daily) win; LSTM only at Tower 2.* (Full TFT/N-HiTS de-scoped → VSN supplies native importance.)
4. ✅ **Feature importance** — I-01 (`06/I01`, permutation+SHAP+VSN, D-39): importance shifts with horizon; livestock density #1.
5. ✅ **UQ layer** — FC-03 (`06/U01`, D-40): conformal + quantile-XGB + LSTM-pinball. *Calibrated but wide; conformal most reliable, quantile-XGB sharpest; intervals miss the biggest spikes → motivates spike-aware modelling.*
6. ⏭ **Spike-aware modelling** (two-stage hurdle) — the agreed next experiment (plan file).
7. ⏭ **Write-up**; revisit held-out 2024 if data lands.
8. ⏭ **Scenario demo** (07, digital shadow).

## Open items / risks

- Held-out **2024 CH₄ unavailable** (blocks the final held-out benchmark; 2022–2023 test still solid).
- **Perfect-forecast met proxy** is optimistic; add a degraded-driver sensitivity later.
- **FCO₂ lever lost** (concurrent → lagged-only) ⇒ low forecasting R²; lead with skill-vs-baseline.
- **Tower 2 not a test target** ⇒ forecasting headline is Towers 4 & 9.
