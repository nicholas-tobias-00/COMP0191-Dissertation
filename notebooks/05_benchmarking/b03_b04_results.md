# B-03 / B-04 — enriched-feature forecasting reruns (D-41)

**Notebooks:** `B03_enriched_ML.ipynb` (= FC-01 trees), `B04_enriched_DL.ipynb` (= FC-02 DL).
**Builder:** `src/features/build_forecasting_matrix_v2.py` → `data/Hourly/forecast_features_v2.csv` (hourly,
+7 `fx_`) + `forecast_daily_v2.csv` (daily, 34 `fx_` + 7 `ar_`). **Results:** `b03_summary.csv`,
`b04_summary.csv`; `B03`/`B04` rows in `benchmarks.csv`.

Productionises the `NWFP_T9_Dataset_Structure.md` feature engineering across **all towers** (D-35 external
soil): **wind-direction circular encoding** (`fx_wd_sin/cos`), **TA min/max**, **multi-week external-soil
lags/rolling** (daily), **`days_since_grazing`**, expanded calendar (`is_growing/is_winter`), **3-sensor SHF
mean**, and **lagged-only FCO₂** in the daily track. Same CV/eval as FC-01/FC-02 (T4/9 train≤2021/test 2022–23;
T2 expanding folds; train gap-filled / eval observed; leak-free).

## Headline — the enriched features lift the **tree** forecasters; the DL is unmoved

### B-03 (trees) vs FC-01 — daily track (the meaningful gain)

| best daily R² | FC-01 | **B-03 (v2 + HPO)** | Δ |
|---|---|---|---|
| Tower 4 | 0.263 | **0.362** | **+0.099** |
| Tower 9 | 0.304 | **0.388** | **+0.084** |

- **Mean daily ΔR² (T4/9): RF +0.118, XGB +0.166**; ΔRMSE −4.7 / −6.2 nmol. Improvement at *every* daily horizon.
- **Daily h=1 R²:** T4 0.357 (RF) / 0.362 (XGB); T9 0.388 (RF) / 0.324 (XGB) — up from FC-01's 0.26 / 0.30.
- **Skill vs persistence (RF, daily)** grows with horizon: T4 −0.07→**+0.42** (h1→h14), T9 +0.11→**+0.37**.
  (Day-ahead persistence is still hard to beat at T4, as in FC-01.)
- **Two rounds:** Round 0 = features (RF +0.071, XGB +0.091 mean daily ΔR²); **Round 1 = HPO** on the daily
  trees (RF leaf10/max-features0.5; XGB depth2/lr0.02/400-trees) added a further ~+0.05 → the numbers above.
  Hourly kept the B01-validated settings.

### B-03 — hourly track
Marginal: mean ΔR² **+0.016 (RF) / +0.036 (XGB)**; best h=1 R² ≈ 0.14–0.16. Wind direction + daytime help a
little, but hourly CH₄ stays dominated by turbulence/noise (as the literature predicts).

### B-04 (DL) vs FC-02 — no benefit
- **DLinear (the production DL) is flat:** best daily R² T4 0.333→**0.337**, T9 0.326→**0.292**.
- LSTM improved off a low base (mean daily ΔR² +0.119) but remains negative at short horizons.
- **Why:** the seq2seq encoder's 28-day lookback already *sees* the soil/TA history that the new daily lag
  features encode — so adding them as channels is redundant for the DL. Confirms the D-38 "simpler wins" story
  and the plan's hypothesis.

## Verdict against the R² ~ 0.5 target
Best **daily forecasting R² now ≈ 0.36–0.39** (enriched trees), a real **+0.08–0.10 over FC-01** — but still
**short of the 0.5 stretch**, exactly as flagged: leak-free forecasting is harder than our gap-filling ceiling
(0.36–0.49) and the literature floor is <0.1 (Zhu 2023a). Reaching ~0.5 would need levers the user deferred
this round (target transform for the spikes, coarser/cumulative evaluation, or the two-stage hurdle model).
The honest gain is **better skill + the best-achievable R²**, met.

## Recommendation
- **Production forecaster = enriched trees on the daily track** (RF/XGB on `forecast_daily_v2.csv`); RF for
  hourly; DLinear remains the daily DL baseline (no change). Wind direction, `days_since_grazing`, and the
  external-soil daily lags are the features worth keeping.
- **Stop here** for this round (features + HPO exhausted, per the agreed bounded plan). If we later want to
  chase ≥0.4–0.5: spike-aware hurdle model + target transform are the next levers.

## Addendum — B-05 arcsinh target transform (D-42, NEGATIVE)
Tested whether an `arcsinh` target transform (signed-safe spike compression) lifts daily R²
(`B05_asinh_ML.ipynb`). **It does not.** Naive `sinh` back-transform was badly biased (R² 0.13–0.21,
MBE ≈ −20); with **Duan smearing** it recovers to daily best **T4 0.337 / T9 0.347 — still below B-03's
0.362 / 0.388**. A scale sweep only approaches identity from below as the transform weakens. R² is scored in
original units where spikes dominate the variance, so squashing them trades the spike accuracy R² rewards for
bulk accuracy — a net loss. **B-03 stays the production config; the spike problem needs a structural attack
(hurdle model), not a transform.** Kept in benchmarks as a flagged negative result.

## Files / scope
Additive only — B01/B02, `forecast_features.csv`, and FC-01/FC-02 artifacts untouched; the sole shared-code
change is the backward-compatible `forecasting_dl.load_matrix(path=None)`.

*Source: `B03_enriched_ML.ipynb`, `B04_enriched_DL.ipynb`, `b03_summary.csv`, `b04_summary.csv`,
`benchmarks.csv` (B03/B04). Decision D-41.*
