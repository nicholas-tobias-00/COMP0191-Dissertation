# R-03-CO2 Results: Kim (2020) with gap-filled FCO₂

**Notebook:** `R03_CO2_Kim2020_RF_ANN_SVM_MDS_PCA.ipynb`
**Base:** R-03 (`../03_gap_filling/R03_results.md`)
**Executed:** 2026-06-16
**Change vs base:** `FC_1_1_1 [Tower N]` in `feat_all` (and hence `feat_lag`) is replaced by the **gap-filled FCO₂** series (observed-where-available; D-26). Lag features, PCA, models, split, scenarios (short/medium/long/xlong, 10% mask, 5 reps) unchanged. MDS and SVM are effectively controls (MDS uses only SW+TA; SVM barely moves).

---

## Results — median R² by scenario

### Tower 4

| Model | short | medium | long | xlong | (base R-03 best) |
|-------|-------|--------|------|-------|------------------|
| MDS | −0.164 | −0.085 | −0.193 | −0.404 | unchanged |
| RF | −0.002 | +0.063 | −0.055 | −0.055 | was +0.136 / −0.038 / −0.089 / −0.058 |
| RF_lag | +0.032 | +0.042 | −0.034 | −0.000 | was +0.135 / −0.115 / −0.125 / −0.088 |
| RF_PCA7 | +0.105 | +0.053 | +0.040 | +0.038 | was +0.086 / +0.005 / +0.003 / +0.047 |
| SVM | −0.010 | −0.010 | −0.016 | −0.025 | ≈ unchanged |
| **ANN** | **+0.163** | **+0.168** | **+0.159** | **+0.118** | was +0.097 / +0.091 / +0.077 / +0.057 |

### Tower 9

| Model | short | medium | long | xlong |
|-------|-------|--------|------|-------|
| MDS | −0.104 | −0.333 | −0.194 | −0.379 |
| RF | +0.121 | +0.086 | −0.042 | −0.097 |
| RF_lag | +0.103 | +0.111 | −0.013 | −0.014 |
| RF_PCA7 | +0.072 | −0.011 | +0.060 | +0.000 |
| SVM | −0.022 | −0.000 | −0.016 | +0.000 |
| ANN | +0.028 | +0.010 | −0.015 | −0.634 |

---

## Interpretation

- **ANN is the big winner at Tower 4: ~+0.06–0.10 → +0.12–0.17, best model at every gap length.** The MLP exploits the cleaned, complete FCO₂ feature better than the trees, and (unlike RF) keeps near-zero bias. This is the strongest single-model result across the whole gap-filling programme.
- **RF / RF_lag drop at short gaps** (RF short +0.136 → −0.002), the same effect seen in R-01-CO2: QC'ing FC removes the raw-FC co-artifact signal the trees had leaned on. RF_PCA7 is steadier (slightly up) because PCA already de-weighted noisy inputs.
- **Tower 9 is mixed/flat.** RF and RF_lag stay near base; ANN does *not* benefit (and still collapses at xlong, −0.634 — the ≈1-gap small-sample artefact). FC carries less marginal information at Tower 9, consistent with R-01-CO2 and R-02-CO2.
- **Kim findings under CO₂ augmentation:** RF_PCA7 ≥ RF_lag still holds at Tower 4 (PCA-degrades-ML still *not* confirmed); lag features still help Tower 9 more than Tower 4.

**Caveat (D-22):** observed FC at FCH₄-gap points → optimistic upper bound. See `co2_augmented_summary.md`.

240 rows tagged `R-03-CO2` in `results/benchmarks.csv`.
