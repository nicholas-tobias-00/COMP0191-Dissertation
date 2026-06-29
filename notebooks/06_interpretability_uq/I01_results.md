# I-01 — Forecasting feature importance (D-39)

**Notebook:** `I01_feature_importance.ipynb` · **Outputs:** `results/fc_importance_summary.csv`, `fc_importance_shap_rf.csv`, `fc_importance_vsn.csv`; figures `results/figures/fc_importance_*.png`.

Cross-model importance over the forecasters (Track A hourly, Tower 4 main split): **permutation** (grouped by family, per horizon) for RF + LSTM, **SHAP** (RF), **VSN-native** (LSTM+VSN). Families: `ch4_ar`, `flux_lag` (FCO₂/GPP/Reco, lagged-only), `met`, `planned` (livestock+management), `calendar`, `tower`.

## Headline — importance *shifts with horizon*, exactly as hypothesised

Permutation importance (ΔRMSE when a family is shuffled), **RF**:

| family | h=1 | h=24 | h=48 |
|---|---|---|---|
| **ch4_ar** | **12.8** | 3.9 | 1.8 |
| **planned** (livestock+mgmt) | 2.6 | **5.4** | **8.8** |
| met | 2.0 | 2.7 | 3.5 |
| flux_lag | 2.1 | −0.8 | 0.2 |
| calendar | 0.1 | 0.7 | 0.8 |

- **At short horizons the recent CH₄ history dominates** (ch4_ar ΔRMSE 12.8 at h=1) — persistence-like memory.
- **As the horizon grows, the autoregressive signal decays and the *planned* drivers (livestock + management) take over** (8.8 at h=48 — the largest single family). Met and seasonality also rise. This is the central forecasting-interpretability result: **what you can plan (stocking, cuts) matters more the further ahead you forecast.**

## SHAP (RF, h=24) — livestock density is #1, again

| feature | mean\|SHAP\| |
|---|---|
| **fx_lsu_dens** (livestock density) | **31.8** |
| ar_ch4_lag1 | 9.1 |
| ar_ch4_lag24 | 7.3 |
| fx_mgmt_manure | 4.9 |
| fx_WS, fx_VPD, ar_ch4_lag48, ar_fc_t (FCO₂) | 1.8–4.1 |

**Livestock density is the single most important forecasting feature** — echoing the gap-filling finding (F-01: `_lsu` top SHAP) and confirming the whole project's thesis carries into forecasting: at a grazed pasture, **the animals are the dominant CH₄ signal**. Recent CH₄ autoregression + management + FCO₂ follow.

## RF vs LSTM — strikingly different feature usage (why DL underperforms)

The LSTM permutation tells a different story: `ch4_ar` is huge at h=1 (ΔRMSE 53) but **collapses to ≈0 by h=24**, after which the LSTM leans almost entirely on **future met** (ΔRMSE ~38–40) and calendar. RF instead keeps using AR-history *and* the planned/livestock family across horizons. So the LSTM effectively **abandons the autoregressive memory** at longer horizons and over-relies on the (perfect-forecast) met inputs — a plausible reason it trails RF (B-02): it isn't combining memory + drivers as effectively as the tree.

## VSN-native (LSTM+VSN gate)
Top gated variables: **CH₄** (0.092, ~2× the 1/23 uniform), VPD, **FCO₂** (`ar_fc_t`), WS, SHF, Reco, calendar — corroborating CH₄-memory + a few met/flux channels, though the soft gate is more diffuse than permutation/SHAP.

## Figures
- `fc_importance_permutation.png` — ΔRMSE by family vs horizon, RF vs LSTM (the headline).
- `fc_importance_shap_rf.png` — SHAP top-15 (RF, h=24).
- `fc_importance_vsn.png` — VSN gate weights.

## Takeaways
1. **Forecasting horizon changes what matters:** AR-history short-term → planned-livestock/management + met + seasonality long-term.
2. **Livestock density is the dominant single feature** (SHAP), consistent across gap-filling and forecasting — the project's through-line.
3. **RF and LSTM use features very differently**, which helps explain the tree models' edge (B-02): RF blends memory + planned drivers; the LSTM drops memory and over-trusts the met inputs.
4. Method agreement: permutation and SHAP agree on livestock + CH₄-memory; VSN broadly corroborates. Permutation is the clean cross-model comparator.

*Source: `I01_feature_importance.ipynb`, `results/fc_importance_*.csv`. Decision D-39. Reuses the F-01 SHAP pattern.*
