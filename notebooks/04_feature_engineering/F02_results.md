# F-02 Results: Stocking Density + Pruned Management + Leave-One-Group-In (RFm)

**Notebook:** `F02_stocking_density_pooled_RFm.ipynb`
**Executed:** 2026-06-16
**Builds on F-01.** Three changes: (1) **stocking-density** features (LSU/ha + per-species/ha) from NWFP fenced areas (Design_Develop Appendix D, D-29); (2) **pruned management** (tower-specific cut + manure recency only — F-01's 12-col set overfit); (3) **leave-one-group-in** ablation (BASE vs BASE+each *single* group) for clean per-group attribution, plus a **pooled T2+T4+T9** model to test density where it can actually matter.

---

## 1  Why a pooled model is needed to test density

Catchments 4 and 9 are **both 7.75 ha** (Appendix D), so `density = LSU/7.75` is a constant rescale of LSU. A Random Forest is invariant to monotonic rescaling of a single feature → for a **single-tower** model (or a T4+T9 pool, equal areas) count and density are equivalent. Density only carries information when catchments of **different** area share one model — so the pooled experiment adds **Tower 2's 2018 grassland data** (Catchment 2 = **6.65 ha**), making the count→flux relationship area-dependent and the density normalisation meaningful.

*(The single-tower "count vs density" groups differ slightly only because they contain different columns — count includes the grazing flag + lags, density does not — not because of rescaling.)*

---

## 2  Headline: pooling + stocking density rescues Tower 9

Pooled RFm (train T2 2018 + T4 + T9 2018–21 = 12,558 rows; test 2022–23), median R²:

**POOLED count**
| Tower | vs | s | m | l | m1 |
|---|---|---|---|---|---|
| Tower 4 | 0.239 | 0.160 | 0.212 | −0.014 | −0.107 |
| Tower 9 | 0.091 | 0.015 | 0.179 | −0.107 | 0.227 |

**POOLED density (LSU/ha)**
| Tower | vs | s | m | l | m1 |
|---|---|---|---|---|---|
| Tower 4 | 0.239 | 0.151 | **0.236** | **0.026** | −0.110 |
| **Tower 9** | **0.287** | **0.217** | **0.294** | **0.207** | **0.250** |

- **Tower 9 jumps to R² ≈ +0.21…+0.29 across all gap lengths** — the **best Tower 9 result in the entire project** (its solo BASE was ≈ 0; F-01 best was barely positive at short gaps only).
- Density beats count decisively at Tower 9 (e.g. vs +0.287 vs +0.091; l +0.207 vs −0.107) and modestly helps Tower 4 (m +0.236 vs +0.212).
- **Both pooling and density contribute:** pooling alone lifts Tower 9 above its solo BASE; density on top makes the cross-catchment livestock signal coherent, which the data-poor tower benefits from most.

This is the concrete payoff of the stocking-density feature: it makes a 20-head herd on Tower 2's 6.65 ha (3.0 LSU/ha) properly distinct from 20 head on Tower 9's 7.75 ha (2.6 LSU/ha), so pooled training transfers cleanly to the data-poor tower.

---

## 3  Leave-one-group-in (per tower) — clean attribution

### Tower 4 — median R² (Δ vs BASE in brackets), short gaps shown
| group | vs | Δvs | note |
|---|---|---|---|
| BASE | 0.156 | — | = 03b R-02-CO2 |
| **livestock_count** | **0.256** | **+0.100** | top single group (as F-01) |
| livestock_density | 0.247 | +0.091 | ≈ count (different cols) |
| mgmt_pruned | 0.178 | +0.022 | **mild help — overfit fixed** |
| wind | 0.165 | +0.009 | small |
| moisture | 0.164 | +0.008 | small |
| soiltemp_prod | 0.141 | −0.015 | slightly hurts |
| chem | 0.154 | −0.002 | ~neutral |
| ALL | 0.262 | +0.106 | best |

### Tower 9 — median R² (Δ vs BASE)
| group | vs | Δvs | note |
|---|---|---|---|
| BASE | −0.026 | — | |
| livestock_count | −0.074 | −0.048 | hurts solo (small data) |
| livestock_density | −0.055 | −0.029 | |
| **mgmt_pruned** | −0.012 | **+0.014** | **helps (was −0.86 in F-01!)** |
| wind | −0.019 | +0.007 | |
| moisture | −0.070 | −0.044 | hurts |
| soiltemp_prod | −0.021 | +0.005 | small help |
| chem | −0.055 | −0.029 | hurts |
| **ALL** | **0.035** | **+0.061** | positive — combined help |

**The pruned-management fix is decisive:** F-01's cumulative 12-column management group collapsed Tower 9 to **−0.86**; the pruned 2-column tower-specific version (cut + manure recency) **helps** (+0.01…+0.04). Confirms the F-01 diagnosis (overfitting from too many sparse/site-level management columns).

---

## 4  Interpretation

1. **Stocking density works — in the multi-area pooled setting.** It lifts Tower 9 to ≈ +0.29 (best in project) by making cross-catchment livestock comparable; the equal-area T4/T9 single-tower case is provably inert (as expected).
2. **Pooling is a viable fix for data-poor towers.** Tower 9 (2,288 solo rows) generalises far better when trained in a pool of 12,558 rows — especially with density normalisation.
3. **Pruned management is the right call.** Two tower-specific recency channels help; F-01's 12-column set overfit. Leave-one-group-in attribution avoids the cumulative-masking artefact.
4. **Livestock remains the dominant per-tower driver** (Tower 4 livestock_count Δ+0.10, top single group; consistent with F-01 SHAP).

## 5  Recommendations / next steps

> **Follow-up (F-03, `F03_results.md`):** the pooled approach was refined into **partial pooling** (pooled + tower-indicator dummies). Partial pooling ≥ full pooling at every tower — keeps the Tower 9 rescue (≈ +0.29), helps the most-different Tower 2 most, and protects data-rich Tower 4. Adopt partial pooling + density (D-30) as the multi-tower default.

- **Adopt the pooled + density configuration** as the Tower 9 (and general data-poor) modelling approach; carry into forecasting.
- Keep **pruned tower-specific management**; drop site-level + fertN_rate + lime/cultiv channels.
- Consider adding a **tower/catchment-area feature** or per-catchment effects to the pool; test pooling with more towers/catchments (different areas) to further exploit density.
- Validate the Tower-9 pooled-density gain on the held-out 2024 window once downloaded.

550 rows tagged `F-02` in `results/benchmarks.csv` (total 1840): per-tower leave-one-group-in (`model=RFm`) + pooled count/density (`model=RFm_pooled`).
