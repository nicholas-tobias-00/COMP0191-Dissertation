# B-02 — Deep-learning forecasters (FC-02, D-38)

**Notebook:** `B02_deep_learning.ipynb` · **Models:** `src/models/forecasting_dl.py` · **Results:** `results/fc02_summary.csv`; 81 rows `FC-02` in `benchmarks.csv`.

Hand-rolled pure-PyTorch, native sequence-to-sequence, driver-conditional, partial-pooled — **DLinear** (Zeng-2023 decomposition-linear), **LSTM** seq2seq, **LSTM+VSN**. Same data (External SMS/MET), CV and observed eval points as FC-01. Trained on the **RTX 5070 GPU** (torch 2.11+cu128). Leak-free: encoder sees CH₄+drivers+lagged-flux history; decoder sees only known-future drivers.

## Headline — model complexity does **not** pay off here

| | hourly (Track A) | daily (Track B) |
|---|---|---|
| **Best model** | **RF / XGB** (trees) | **DLinear** (simple linear) ≈ RF |
| LSTM / LSTM+VSN | usually *worse* than RF (neg. R²) | worse, esp. short daily |
| Exception | **Tower 2: LSTM beats RF** | — |

This is the **Zeng-2023 finding, confirmed** for open-system grassland CH₄: the heavier recurrent models rarely beat a Random Forest (hourly) or a single linear layer (daily). Complexity helps only where there's strong autoregressive structure (Tower 2).

## Track A (hourly) — skill vs persistence (RF from FC-01 for reference)

| Tower 4 | h1 | h6 | h12 | h24 | h48 |
|---|---|---|---|---|---|
| RF | **0.108** | **0.203** | **0.187** | **0.172** | **0.136** |
| LSTM+VSN | −0.10 | 0.164 | 0.012 | 0.053 | 0.069 |
| DLinear | −0.04 | 0.057 | −0.11 | 0.004 | 0.039 |

- **RF/XGB clearly win at Towers 4 & 9** (skill +0.11–0.25). The DL models trail and have **negative R²** on T4/T9 (e.g. DLinear R² −0.43…−0.48; LSTM similar) — i.e. they beat *persistence* on RMSE at mid horizons but are still worse than predicting the test mean, whereas RF reaches **positive R²** (0.02–0.15). So RF is unambiguously the better hourly forecaster.
- **Tower 2 flips:** **LSTM skill = 0.375 at h1** (vs RF 0.080) and stays ahead at h24/h48 (0.25/0.28 vs 0.20/0.24). The sequence model captures Tower 2's strong livestock-on/off autoregressive regime that the tabular RF can't.

## Track B (daily) — skill vs persistence

| | d1 | d3 | d7 | d14 |
|---|---|---|---|---|
| **T4 DLinear** | −0.09 | 0.040 | 0.190 | **0.330** |
| T4 RF | −0.15 | 0.055 | 0.193 | 0.372 |
| **T9 DLinear** | **0.063** | **0.235** | 0.337 | 0.311 |
| T9 RF | 0.049 | 0.196 | 0.338 | 0.321 |

- **DLinear is competitive with — and at Tower 9 beats — RF** on daily forecasting (T9 d3: 0.235 vs 0.196). The simplest model (one decomposition + linear layer) is the right tool at the daily/budget scale.
- **LSTM/LSTM+VSN are the worst on daily** (negative skill at d1–d3): the recurrent capacity overfits the short (~2,900-day), data-limited daily series.

## Why (interpretation)

1. **Data is limited and the signal is episodic/noisy** — exactly the regime where high-capacity sequence models don't earn their keep (cf. Zeng et al. 2023; the D-05 caveat). Trees + linear models with the engineered driver features are hard to beat.
2. **Negative DL R² with positive persistence-skill** is the tell: persistence is a weak baseline at long horizons, so "beats persistence" ≠ "beats the mean." RF clears both bars; the DL models often only clear the first.
3. **Tower 2 is the one place complexity helps** — its near-binary livestock regime is strongly autoregressive, which the LSTM exploits (best T2 short-horizon forecaster in the project).
4. **GPU made it cheap** — each model trains in seconds–minutes on the RTX 5070, so the negative result is well-powered, not under-trained (30 epochs; loss converged).

## Recommendation
- **Carry RF (hourly) and DLinear (daily) as the production forecasters**; keep the LSTM only for Tower 2 / as a documented complexity baseline.
- The dissertation contribution here is the **honest benchmark**: for this open system, **simpler models win** — a clean, citable result, not a failure to make DL work.

## Caveats / next
- Untuned DL (fixed hidden=64, 30 epochs, no early-stopping/HPO) — light tuning *might* close some gap, but the daily-DLinear and hourly-RF dominance is unlikely to reverse.
- Perfect-forecast met proxy (optimistic, as in FC-01); FCO₂/GPP lagged-only.
- **Next:** I-01 feature importance (permutation across all models + SHAP for trees + VSN-native), then UQ.

*Source: `B02_deep_learning.ipynb`, `results/fc02_summary.csv`, `benchmarks.csv` (FC-02). Decision D-38.*
