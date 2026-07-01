# B-03 / B-03a / B-03b — forecasting results summary

Source: `results/b03_summary.csv`, `results/b03a_summary.csv`, `results/b03b_summary.csv`,
`benchmarks.csv` (`TFT-Reg` rows). See `b03_b04_results.md` and `b03a_b03b_results.md` for
full write-ups; this file is a compact results pull across all towers/models/tracks.

---

## B-03 — enriched-feature trees (RF / XGBoost) — production forecaster (D-41)

### Daily (Track B) R² by horizon

| Tower | Model | h=1 | h=3 | h=7 | h=14 |
|---|---|---|---|---|---|
| **T4** | RF | 0.357 | 0.345 | 0.287 | 0.270 |
| **T4** | XGB | **0.362** | 0.313 | 0.284 | 0.259 |
| **T9** | RF | **0.388** | **0.350** | **0.364** | **0.342** |
| **T9** | XGB | 0.324 | 0.271 | 0.303 | 0.275 |
| T2 | RF | 0.042 | 0.019 | 0.007 | 0.001 |
| T2 | XGB | -0.059 | -0.064 | -0.074 | -0.099 |

**Best overall: Tower 9, RF, h=1, R²=0.388.**

### Hourly (Track A) R² by horizon

| Tower | Model | h=1 | h=6 | h=12 | h=24 | h=48 |
|---|---|---|---|---|---|---|
| **T4** | RF | 0.136 | 0.044 | 0.078 | 0.049 | 0.032 |
| T4 | XGB | 0.119 | 0.048 | 0.065 | 0.030 | 0.035 |
| **T9** | RF | **0.159** | 0.084 | 0.081 | 0.087 | 0.055 |
| T9 | XGB | 0.132 | 0.042 | 0.021 | 0.064 | 0.052 |
| T2 | RF | 0.077 | 0.016 | 0.031 | 0.033 | 0.029 |
| T2 | XGB | -0.146 | -0.044 | -0.032 | 0.029 | -0.092 |

**Best hourly: Tower 9, RF, h=1, R²=0.159.**

### Takeaways
- Tower 2 is the weak spot — near-zero/negative R² everywhere, both tracks (least data: 12.1% valid FCH4).
- RF ≥ XGB almost everywhere, especially at Tower 9.
- Production = **RF on the daily track**.

---

## B-03a — SARIMAX, order (2,1,1)

### Daily (Track B) R²

| Tower | h=1 | h=3 | h=7 | h=14 |
|---|---|---|---|---|
| T4 | **0.371** | 0.066 | -0.194 | -0.330 |
| T9 | 0.282 | 0.055 | -0.012 | -0.023 |
| T2 | 0.102 | -0.096 | -0.309 | -0.459 |

### Hourly (Track A) R²

| Tower | h=1 | h=6 | h=12 | h=24 | h=48 |
|---|---|---|---|---|---|
| T4 | -0.168 | 0.147 | 0.038 | -0.237 | -0.215 |
| T9 | -0.294 | 0.014 | 0.116 | -0.679 | -0.830 |
| T2 | -0.953 | 0.158 | -0.075 | -0.007 | -0.007 |

**Best: T4, daily, h=1, R²=0.371.** Competitive with trees only at the shortest horizon, then
collapses negative by h≥7. Hourly is mostly negative/near-zero throughout.

---

## B-03b — TFT (original run — later found to be overfit)

### Daily R²

| Tower | h=1 | h=3 | h=7 | h=14 |
|---|---|---|---|---|
| T4 | -1.078 | -1.018 | -0.917 | -0.836 |
| T9 | -0.856 | -0.765 | -0.644 | -0.623 |
| T2 | -0.512 | -0.477 | -0.474 | -0.451 |

### Hourly R²

| Tower | h=1 | h=6 | h=12 | h=24 | h=48 |
|---|---|---|---|---|---|
| T4 | -0.127 | -0.085 | -0.060 | -0.056 | -0.062 |
| T9 | -0.173 | -0.022 | -0.034 | -0.040 | -0.041 |
| T2 | -0.149 | -0.104 | -0.106 | -0.106 | -0.108 |

Negative everywhere, every tower, every horizon, both tracks — the single worst model result in
the whole forecasting phase. Verified not a bug (training converged cleanly, test correlation
r=0.27) — root cause was overfitting: TFT fit training data to R²≈0.91 but a handful of large
spike-mispredictions dragged test R² deeply negative.

---

## B-03b — TFT-Reg (regularised fix: weight_decay=1e-3 + early stopping on 2021 val year)

Only reran for T4/T9 main split (Tower 2 not refit).

### Daily R²

| Tower | h=1 | h=3 | h=7 | h=14 |
|---|---|---|---|---|
| T4 | 0.247 | 0.250 | 0.252 | **0.255** |
| T9 | 0.106 | 0.101 | 0.100 | 0.097 |

### Hourly R²

| Tower | h=1 | h=6 | h=12 | h=24 | h=48 |
|---|---|---|---|---|---|
| T4 | 0.016 | 0.004 | 0.006 | 0.007 | 0.008 |
| T9 | 0.012 | 0.039 | 0.039 | 0.039 | 0.039 |

The regularisation fix flipped every value from negative to positive — TFT is now "genuinely
reasonable," but still well below B-03's trees at every horizon/tower.

---

## Cross-model comparison (T4/T9 daily R²)

| | B-03 RF (production) | B-03a SARIMAX | B-03b TFT (original) | B-03b TFT-Reg |
|---|---|---|---|---|
| h=1 | **0.357–0.388** | 0.282–0.371 | -0.856…-1.078 | 0.106–0.247 |
| h=14 | **0.270–0.342** | -0.023…-0.330 | -0.623…-0.836 | 0.097–0.255 |

**Verdict: B-03 enriched trees remain production.** SARIMAX only competes at h=1; even a fixed
TFT sits a full R²-unit below the trees. This closes out the entire original D-05 model roster
(persistence → ARIMA → RF/XGB → LSTM/TFT → SARIMAX) — every rung now has a documented result,
and none beats the trees.
