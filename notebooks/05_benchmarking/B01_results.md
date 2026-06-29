# B-01 — Forecasting baselines + ML benchmark (FC-01, D-36/D-37)

**Notebook:** `B01_baselines_and_ML.ipynb` · **Precompute:** `src/data/build_fch4_gapfilled.py`, `src/features/build_forecasting_matrix.py`, `src/models/gapfill_rfm.py` · **Results:** `results/fc01_summary.csv`; 108 rows tagged `FC-01` in `results/benchmarks.csv`.

The first forecasting benchmark (the project's novel contribution). **Driver-conditional, direct multi-horizon, partial-pooled**, External SMS/MET data (D-35). Two tracks — A hourly {1,6,12,24,48} h, B daily-mean {1,3,7,14} d. Baselines (persistence, climatology) + RF + XGBoost. Trained on the gap-filled CH₄ series; **evaluated on observed timestamps only**. Towers 4/9 test 2022–2023; Tower 2 via expanding-window rolling-origin within 2017–2019 (donor = Tower 4).

## Headline — RF skill vs persistence (% RMSE reduction)

| | h1 | h6 | h12 | h24 | h48 | | d1 | d3 | d7 | d14 |
|---|---|---|---|---|---|---|---|---|---|---|
| **T4** | +0.11 | +0.20 | +0.19 | +0.17 | +0.14 | | −0.15 | +0.06 | +0.19 | **+0.37** |
| **T9** | +0.09 | +0.22 | +0.25 | +0.19 | +0.23 | | +0.05 | +0.20 | +0.34 | +0.32 |
| **T2** | +0.08 | +0.15 | +0.23 | +0.20 | +0.24 | | +0.04 | +0.09 | +0.17 | +0.24 |

RF R² (Towers 4/9): hourly 0.02–0.15; **daily 0.15–0.30** (aggregation smooths the noise). Tower 2 R² ≈ 0 / unstable (as predicted — near-zero-variance regime); RMSE/skill are its valid metrics.

## What the benchmark shows

1. **ML genuinely forecasts — it beats persistence at almost every horizon.** Hourly: RF cuts RMSE 8–25% vs persistence across all towers. The gain *grows* from h1 to h6–h12 (persistence decays fast once you leave the last value behind) then is broadly flat to 48 h.
2. **Forecasting R² is low but positive** (hourly ~0.03–0.15) — exactly as scoped: the concurrent-FCO₂ lever is gone (leak-free), this is an open system, and absolute R² is *not* the metric. **Skill-vs-baseline is the story** (the supervisor framing, analogous to improvement-over-MDS).
3. **Daily aggregation helps the model.** Track B RF R² reaches 0.15–0.30; at the 2-week horizon RF cuts RMSE by ~32–37% vs persistence at T4/T9 — the budget/scenario-relevant horizon looks the strongest.
4. **The honest caveat: 1-day-ahead daily persistence is hard to beat.** At d1, daily CH₄ is so autocorrelated that yesterday's mean ≈ today's; RF *loses* to persistence at T4 (skill −0.15). The model's value appears at **≥3-day** horizons.
5. **Climatology is a strong baseline.** Persistence collapses at longer horizons (negative skill-vs-climatology), but the hour×month / month climatology is competitive; RF's edge *over climatology* is modest (+0.02–0.15). Reporting both baselines keeps the claim honest — RF's win over persistence is large, its win over a seasonal-diurnal climatology is real but smaller.
6. **RF > XGBoost** here (XGB shows larger positive MBE and occasionally loses to climatology) — untuned XGB; revisit with tuning in a later stage.
7. **Tower 2 is forecastable** under rolling-origin: +0.08…+0.24 skill over persistence across horizons, stable RMSE (~46–48 daily) — the flip from "donor-only" to a genuine (if R²-caveated) test target succeeded.

## Methodology notes / honesty

- **Driver-conditional with a perfect-forecast met proxy** (observed met as future input) — an **optimistic upper bound**; real NWP-forecast error would lower skill. A degraded-driver sensitivity is a planned later add.
- **Leak-free:** FCO₂/GPP/Reco enter only as lagged (≤ t) features; future exog at t+h is met/soil/livestock/management/calendar only.
- **Train on gap-filled, eval on observed**; the gap-filler is globally trained, so AR features carry minor optimism (documented preprocessing simplification).
- **Tower 2 R² is degenerate** on its near-zero 2019 variance — RMSE/MAE/skill are the reported metrics.
- **Held-out 2024 unavailable** (2024 CH₄ = 0% valid) — final held-out benchmark deferred.

## Next (Stage 2+)

- **Deep learning** (LSTM, **TFT** with quantiles, DLinear/N-HiTS) — same harness; the test of whether sequence models beat RF here (Zeng-2023 caveat, D-05).
- **UQ** (conformal intervals + TFT quantiles).
- **XGB tuning**; **degraded-driver (real-forecast) sensitivity**; revisit held-out 2024 if the data lands.

*Source: `B01_baselines_and_ML.ipynb`, `results/fc01_summary.csv`, `results/benchmarks.csv` (FC-01). Decisions D-36, D-37.*
