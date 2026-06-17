# F-04 Results: Adding R-03 hydrological/thermal lags to the density + partial-pooling models (RFm)

**Notebook:** `F04_lags_partial_pooling_RFm.ipynb`
**Executed:** 2026-06-16
**Question:** R-03 found `RF_lag` (RF + **SWC/TS lagged 1–4 weeks**) was the *best* model at **Tower 9** — but those lags were never carried into F-01→F-03 (which built on R-02-CO2's lag-free `driver_m` + FCO₂). Does re-adding them help, now that the base also has FCO₂ + stocking density + partial pooling?

Lags (Kim 2020 / D-23): SWC and TS each at 168/336/504/672 h (8 columns). Four variants, all density (D-29), evaluated on Towers 2/4/9; same training rows ± lags; R-02 gap harness.

---

## 1  The lag effect (Δ vs no-lag, overall median R²)

| Tower | solo Δ | **partial-pool Δ** |
|---|---|---|
| Tower 2 | +0.011 | **+0.116** |
| Tower 4 | +0.018 | +0.003 |
| Tower 9 | −0.010 | −0.004 |

## 2  Median R² by tower × variant (short gaps, vs)

| Tower | solo | solo+lags | partial | **partial+lags** |
|---|---|---|---|---|
| Tower 2 | −0.712 | −0.694 | −0.245 | **−0.074** |
| Tower 4 | 0.248 | 0.241 | 0.238 | 0.244 |
| Tower 9 | −0.054 | −0.080 | 0.293 | 0.291 |

(Overall-median: Tower 2 partial −0.179 → **−0.062**; Tower 4 0.067 → 0.070; Tower 9 0.251 → 0.247.)

---

## 3  Interpretation — the R-03 lag advantage does **not** transfer to Tower 9

1. **Lags do NOT help Tower 9 here** (Δ ≈ −0.00 to −0.01) — the opposite of R-03, where `RF_lag` was Tower 9's best model. The reason: in R-03 the base was poorer (lag-free met + LE/H/FC), so SWC/TS lags added genuinely new hydrological-memory information. In F-04 the base already contains **gap-filled FCO₂ + stocking density + cross-tower pooling**, which evidently *already encode* the slow soil-moisture/temperature memory the lags were proxying. Once that richer base exists, the lags are largely **redundant** for Tower 9.
2. **Lags help Tower 2 the most** (partial Δ **+0.116**; −0.179 → −0.062 overall) — the tower with the *weakest* base signal (Red farmlet, arable conversion, broken split) gains most from the extra temporal memory. Still negative (D-15 split), but the best Tower 2 result yet.
3. **Lags help Tower 4 marginally**, mostly at **medium/long gaps** (l: 0.028 → 0.052; m: 0.236 → 0.244) — where short-term drivers are less informative and slow memory matters more.

**General lesson:** feature value is **context-dependent**. A feature that was decisive on a weak base (R-03 `RF_lag` at Tower 9) can be redundant once a strong base (FCO₂ + density + pooling) is in place. Tower 9's lever is the pooling+density+FCO₂ combination, *not* the lags.

---

## 4  Best configuration & recommendation

- **Tower 9:** partial pooling + density (no lags needed) — R² ≈ +0.29. Lags neutral.
- **Tower 4:** partial pooling + density + lags — small gain at medium/long gaps.
- **Tower 2:** partial pooling + density + lags — best Tower 2 yet (still negative; needs split redesign).

**Net:** keep the SWC/TS lags in the standard feature set — they are cheap, never materially hurt, and help Tower 2 and Tower 4 (longer gaps) — but recognise they are **not** the Tower 9 lever. The recommended global configuration into forecasting is **partial pooling + stocking density + SWC/TS lags**.

300 rows tagged `F-04` in `results/benchmarks.csv` (total 2365): `model = RFm_solo / RFm_solo_lags / RFm_partial / RFm_partial_lags`.
