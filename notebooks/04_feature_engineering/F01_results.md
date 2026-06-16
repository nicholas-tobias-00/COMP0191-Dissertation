# F-01 Results: Feature-Engineering Ablation (P1–P6) — RFm

**Notebook:** `F01_feature_ablation_RFm.ipynb`
**Executed:** 2026-06-16
**Base config:** 03b R-02-CO2 RFm = `driver_m` (11 met + 4 AUX) + gap-filled FCO₂.
**Harness:** R-02 calendar gaps (vs/s/m/l/m1, MASK_FRAC 0.25, 5 reps); RFm only; Towers 4 & 9.
**Design:** cumulative ablation with shared training rows + shared eval gaps (only feature columns change; sparse features mean-imputed).
**Features:** P1 livestock (own-catchment cattle/sheep/lamb, LSU, grazing flag, 24h/7d lags), P2 management (site + tower-area exp-decay time-since), P3 wind (sin/cos WD), P4 antecedent moisture (precip 7/30d, SWC 7d mean), P5 per-catchment soil-temp@15cm + productivity proxy, P6 soil/water chemistry.

---

## 1  Validation

BASE reproduces 03b R-02-CO2 RFm **exactly** (Tower 4 vs-gap R² = +0.156) — confirms the harness and that only the new features drive any change.

---

## 2  Tower 4 — median R² by feature set × scenario

| feature set | vs (1h) | s (4h) | m (32h) | l (288h) | m1 |
|---|---|---|---|---|---|
| BASE | 0.156 | 0.084 | 0.111 | 0.031 | −0.121 |
| **+P1 livestock** | **0.256** | **0.159** | **0.186** | 0.014 | −0.072 |
| +P2 mgmt | 0.209 | 0.124 | 0.159 | 0.005 | −0.124 |
| +P3 wind | 0.224 | 0.143 | 0.170 | 0.017 | −0.072 |
| +P4 moisture | 0.228 | 0.152 | 0.177 | 0.026 | −0.064 |
| +P5 soiltemp/prod | 0.216 | 0.143 | 0.164 | 0.019 | −0.057 |
| +P6 chem | 0.217 | 0.140 | 0.162 | 0.015 | −0.060 |

> **Mapping correction (D-28):** management features use the complete field→catchment table from `NWFP_UG_Design_Develop.pdf` Appendix D. Tower 4 = Catchment 4 = {NW005, NW006} (an earlier draft over-broadly used the whole Green farmlet, 495 → 124 events); Tower 9 = Catchment 9 = {NW013, NW039} (already correct). Conclusions are unchanged — P2 management remains the weakest/over-fitting group.

**Δ vs BASE from P1 livestock alone: +0.100 (vs), +0.075 (s), +0.075 (m).** Livestock is by far the largest single contribution. Groups added after P1 give flat-to-slightly-negative returns (P2 management even removes some of P1's gain — see §4).

## 3  Tower 9 — median R² by feature set × scenario

| feature set | vs | s | m | l | m1 |
|---|---|---|---|---|---|
| BASE | −0.026 | −0.014 | 0.001 | −0.054 | −0.073 |
| +P1 livestock | −0.074 | −0.069 | −0.016 | −0.127 | −0.024 |
| +P2 mgmt | −0.757 | −0.860 | −0.775 | −1.287 | −0.855 |
| +P3 … +P6 | ≈ −0.7 to −0.9 (P2 damage persists) | | | | |

Tower 9 gains from **nothing**; P1 marginally hurts, and **P2 management collapses it** (see §4).

## 4  SHAP — Tower 4 full model (the decisive evidence)

Top features by mean|SHAP|:

| Rank | Feature | mean\|SHAP\| |
|---|---|---|
| 1 | **`_lsu` (livestock units)** | **28.2** |
| 2 | `FC_1_1_1` (gap-filled FCO₂) | 14.7 |
| 3 | `TS_1_1_1 [Tower 9]` (soil temp) | 9.2 |
| 4 | `cattle_Catchment 4` | 5.4 |
| 5 | `PPFD` | 4.7 |
| 6 | `Soil Temperature @ 15cm [Cat 4]` | 4.5 |
| 7–9 | `_wd_cos`, `_prod_proxy`, `_wd_sin` | 4.2 / 3.8 / 3.7 |

**Livestock units is the #1 predictor — roughly 2× the next feature (FCO₂) and 3× soil temperature.** Two of the top four features are animal-derived. This is the quantitative confirmation of Felber et al. (2015): the animal-in-footprint signal dominates managed-pasture EC CH₄.

---

## 5  Interpretation

1. **Hypothesis confirmed (Tower 4).** Livestock density is the dominant driver and the biggest single R² lift (+0.10 at short gaps). The drivers review and Felber (2015) are validated on NWFP data.
2. **Diminishing returns beyond livestock.** P3 wind gives a small recovery; P4–P6 are flat. The animal signal carries most of the recoverable information; the remaining variance is the irreducible spike stochasticity + metric sensitivity.
3. **Management features (P2) overfit — the key cautionary finding.** Twelve management columns (site-level + tower-area exp-decay) *reduce* R² at Tower 4 and **collapse Tower 9** (−0.86). Cause: small training sets (T9 = 2,288 rows) + management-timing distribution shift between 2018–21 and 2022–23 (the Red-farmlet arable conversion changed farm-wide campaign timing). The model latches onto training-period management calendars that invert at test. Because the ablation is cumulative, P2's damage persists into P3–P6.
4. **Tower 9 is feature-fragile.** With ~⅓ of Tower 4's training data, even P1 marginally hurts. Data volume — not feature availability — is Tower 9's binding constraint.

## 6  Best configuration & recommendations

- **Tower 4:** BASE + P1 livestock (+ optional P3 wind) → short-gap R² ≈ **+0.26** (best in the whole programme). Keep livestock; drop P2 as implemented.
- **Tower 9:** BASE (no additions help); needs regularization / more data, not more features.
- **Next steps:**
  1. **Prune management features** — keep only 2–3 tower-specific recency channels (e.g. cut, manure); drop site-level + `fertN_rate`. Re-test.
  2. **Non-cumulative (leave-one-group-in) ablation** — so P2's overfit doesn't mask P3–P6.
  3. **Regularize for Tower 9** — fewer features, shallower trees, or pool towers.
  4. Carry **livestock features forward** into the forecasting phase — lagged livestock is a legitimate, high-value predictor.

350 rows tagged `F-01` (with `feature_set` column) in `results/benchmarks.csv` (total 1290).
