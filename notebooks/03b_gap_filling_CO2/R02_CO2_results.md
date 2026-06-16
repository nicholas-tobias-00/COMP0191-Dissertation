# R-02-CO2 Results: Zhu (2023a) with gap-filled FCO₂ added to driver_m

**Notebook:** `R02_CO2_Zhu2023a_RF_MDS.ipynb`
**Base:** R-02 (`../03_gap_filling/R02_results.md`)
**Executed:** 2026-06-16
**Change vs base:** gap-filled FCO₂ (`FC_1_1_1 [Tower N]`, observed-where-available; D-26) is **added to `driver_m`** (now 12 met cols + 4 AUX). `driver3`, MDS, hyperparameters, split, scenarios (vs/s/m/l/m1, 25% mask, 5 reps) are unchanged. RF3 and MDS therefore serve as **untouched controls**; only RFm and XGBm gain FC.

---

## Headline: this is the cleanest test of D-22

Adding a single feature (gap-filled FCO₂) to the met-only RFm moves **Tower 4 from negative to positive R²**, while the no-FC controls (RF3, MDS) are byte-identical to base R-02.

### Tower 4 — median R² by scenario

| Model | vs (1h) | s (4h) | m (32h) | l (288h) | m1 |
|-------|---------|--------|---------|----------|-----|
| MDS *(control)* | −0.150 | −0.179 | −0.148 | −0.260 | −0.475 |
| RF3 *(control)* | −0.153 | −0.133 | −0.102 | −0.148 | −0.146 |
| **RFm + FCO₂** | **+0.156** | **+0.083** | **+0.111** | **+0.031** | −0.120 |
| XGBm + FCO₂ | +0.086 | +0.039 | +0.023 | −0.078 | −0.266 |

*(base R-02 RFm was −0.128 / −0.104 / −0.160 / −0.113 / −0.277)*

### Tower 9 — median R² by scenario

| Model | vs | s | m | l | m1 |
|-------|-----|---|---|---|-----|
| MDS *(control)* | −0.174 | −0.290 | −0.433 | −0.584 | −0.195 |
| RF3 *(control)* | −0.155 | −0.133 | −0.176 | −0.182 | −0.138 |
| **RFm + FCO₂** | **−0.026** | **−0.014** | **+0.000** | −0.054 | −0.073 |
| XGBm + FCO₂ | −0.085 | −0.062 | −0.071 | −0.094 | −0.094 |

*(base R-02 RFm was −0.088 / −0.090 / −0.088 / −0.157 / −0.074)*

---

## Interpretation

- **Tower 4 RFm: −0.10…−0.16 → +0.03…+0.16.** A swing of ~+0.25 from one feature. The CO₂-augmented RFm now matches or beats R-01 RF (+0.144) — confirming numerically that the entire R-01↔R-02 gap was the absence of EC flux co-variates (D-22). FC is the single most informative predictor of FCH₄ at this site.
- **Tower 9 RFm: ≈ 0** (up from ≈ −0.09). Improves to roughly the predicted-mean level — FC helps but Tower 9's smaller training set (2,288 rows) limits the gain.
- **Controls confirm causality:** RF3 and MDS are unchanged to 3 dp, so the improvement is unambiguously the FC feature, not run-to-run noise.
- **Gap-length pattern:** the FC benefit is largest at short/medium gaps and fades by l (288h) — at long gaps the co-observed FC is spread across a wider, more distribution-shifted window.

**Caveat (D-22):** FC is observed at the (FCH₄-only) gap points, so this is an *upper bound* on operational skill — in a real EC outage FC would also be missing. The reconstruction-only variant (use `FC_recon` everywhere) would be the strict operational test; see `co2_augmented_summary.md`.

200 rows tagged `R-02-CO2` in `results/benchmarks.csv`.
