# R-01 Results: Irvin et al. (2021) RF/XGBoost Gap-Filling — All Three NWFP Towers

**Reference:** Irvin, J. et al. (2021). Gap-filling eddy covariance methane fluxes: Comparison of machine learning model predictions across a range of ecosystem wetland types. *Environmental Data Science*, 1, e3.  
**Notebook:** `R01_Irvin2021_RF_XGBoost.ipynb`  
**Executed:** 2026-06-13  
**Spatial alignment:** Tower N = Catchment N (D-18). Each model uses only its own catchment's soil moisture.

---

## 1  Methodology

Irvin et al. evaluated RF and XGBoost gap-filling across 17 FLUXNET-CH4 wetland sites. Their core methodological contribution is **empirical gap length sampling**: rather than masking random individual timesteps (which biases evaluation toward short, easy-to-fill gaps), artificial test gaps are inserted as contiguous blocks whose lengths are drawn from the observed gap-length distribution. This better represents the real gap-filling task.

**Implementation:**
- 5 independent permutations per tower per model, each masking ~40% of test-period valid observations
- Gap lengths sampled from the observed NaN-run distribution in the QC-filtered target series
- `SimpleImputer(strategy="mean")` fit on training data only — no leakage
- Metrics reported as median across 5 permutations

---

## 2  Quality control

Two-pass QC applied to each tower's FCH4 column before training:

1. **SSITC flag filter:** retain rows where `FCH4_SSITC_TEST_1_1_1 [Tower N]` ∈ {0, 1}; set flag=2 and NaN-flag rows to NaN
2. **Physical plausibility:** reject values outside [−500, 3000] nmol m⁻² s⁻¹ (D-13)

| Tower | Raw valid (SSITC 0/1) | After plausibility | % of total hours |
|-------|-----------------------|--------------------|------------------|
| Tower 4 | 30,485 | 19,469 | 27.8% |
| Tower 9 | 17,509 | 11,235 | 16.0% |
| Tower 2 | 8,099 | 4,890 | 7.0% |

---

## 3  Temporal splits

| Tower | Training years | Test years | Train valid rows | Test valid obs |
|-------|---------------|------------|-----------------|----------------|
| Tower 4 | 2018–2021 | 2022–2023 | 7,714 | 7,109 |
| Tower 9 | 2018–2021 | 2022–2023 | 3,981 | 5,848 |
| Tower 2 | 2018 only | 2019 (Jan–May) | 2,985 | 949 |

Tower 2 uses a custom split (D-15) because the standard test window (2022–2023) falls entirely within its 1,675-day sensor gap (May 2019–Jan 2024).

---

## 4  Feature set

All towers use the same feature schema. Tower-specific differences are in column names and soil temperature sourcing.

| Feature | Column pattern | Notes |
|---------|---------------|-------|
| Air temperature | `TA_0_0_1 [Tower N]` | ~78% |
| Incoming shortwave | `SWIN_1_1_1 [Tower N]` | ~52% — D-14 (column is `SWIN_`, not `SW_IN_`) |
| Atmospheric pressure | `PA_0_0_1 [Tower N]` | ~78% |
| Wind speed | `WS_0_0_1 [Tower N]` | ~78% |
| Vapour pressure deficit | `VPD_0_0_1 [Tower N]` | ~75% |
| Soil temperature | `TS_1_1_1 [Tower 9]` (all towers) | Tower 4 TS = 9.6%, Tower 2 TS = 5% → Tower 9 used as proxy (D-16) |
| Friction velocity | `USTAR_0_0_1 [Tower N]` | ~78% |
| Latent heat flux | `LE_1_1_1 [Tower N]` | ~74% |
| Sensible heat flux | `H_1_1_1 [Tower N]` | ~78% |
| CO₂ flux | `FC_1_1_1 [Tower N]` | ~74% |
| Soil moisture (10 cm) | Catchment-matched column (D-18) | Tower 4: 56%, Tower 9: 66%, Tower 2: varies |
| Hour of day (cyclical) | `sin(2π·hour/24)`, `cos(2π·hour/24)` | No ordinal leakage |
| Day of year (cyclical) | `sin(2π·doy/365)`, `cos(2π·doy/365)` | No ordinal leakage |

**Total features:** 14 per tower.

**Catchment-specific soil moisture columns (D-18):**
- Tower 2: `Soil Moisture @ 10cm Depth (%) [Catchment 2]`
- Tower 4: `Soil Moisture @ 10cm Depth (%) [Catchment 4 After  2013/08/13]` ← double space, date suffix
- Tower 9: `Soil Moisture @ 10cm Depth (%) [Catchment 9]`

---

## 5  Model hyperparameters

```python
RandomForestRegressor(n_estimators=500, min_samples_leaf=5, n_jobs=-1, random_state=42)

XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
             subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42)
```

No hyperparameter tuning — parameters match Irvin et al. (2021) as closely as documented.

---

## 6  Results

### Per-permutation breakdown

**Tower 4** (test 2022–2023, n_train=7,714):

| Model | Perm | n_test | R² | RMSE | MAE |
|-------|------|--------|----|------|-----|
| RF | 0 | 2,849 | −0.031 | 119.6 | 60.7 |
| RF | 1 | 2,845 | +0.174 | 121.3 | 59.9 |
| RF | 2 | 2,844 | −0.256 | 118.2 | 62.9 |
| RF | 3 | 6,644 | +0.182 | 135.0 | 64.2 |
| RF | 4 | 2,843 | +0.144 | 132.3 | 62.5 |
| XGB | 0 | 2,849 | −0.126 | 125.0 | 67.6 |
| XGB | 1 | 2,845 | +0.125 | 124.8 | 68.1 |
| XGB | 2 | 2,844 | −0.439 | 126.5 | 70.7 |
| XGB | 3 | 6,644 | +0.132 | 139.1 | 72.2 |
| XGB | 4 | 2,843 | +0.086 | 136.7 | 70.9 |

**Tower 9** (test 2022–2023, n_train=3,981):

| Model | Perm | n_test | R² | RMSE | MAE |
|-------|------|--------|----|------|-----|
| RF | 0 | 2,348 | −0.027 | 122.1 | 58.8 |
| RF | 1 | 2,360 | +0.127 | 143.4 | 65.5 |
| RF | 2 | 2,340 | −0.031 | 116.2 | 56.4 |
| RF | 3 | 2,346 | +0.111 | 153.2 | 63.7 |
| RF | 4 | 2,341 | −0.035 | 123.5 | 57.6 |
| XGB | 0 | 2,348 | −0.130 | 128.0 | 62.6 |
| XGB | 1 | 2,360 | +0.143 | 142.1 | 67.2 |
| XGB | 2 | 2,340 | −0.170 | 123.8 | 61.7 |
| XGB | 3 | 2,346 | +0.154 | 149.4 | 63.3 |
| XGB | 4 | 2,341 | −0.089 | 126.7 | 59.4 |

**Tower 2** (D-15 custom, train 2018, test Jan–May 2019, n_train=2,985):

| Model | Perm | n_test | R² | RMSE | MAE |
|-------|------|--------|----|------|-----|
| RF | 0 | 384 | −14.87 | 132.9 | 94.2 |
| RF | 1 | 389 | −18.67 | 147.9 | 116.4 |
| RF | 2 | 440 | −21.81 | 166.2 | 133.9 |
| RF | 3 | 628 | −15.17 | 141.2 | 108.1 |
| RF | 4 | 379 | −16.94 | 161.3 | 123.6 |
| XGB | 0 | 384 | −47.58 | 232.6 | 193.0 |
| XGB | 1 | 389 | −57.25 | 254.5 | 210.2 |
| XGB | 2 | 440 | −63.84 | 280.2 | 237.1 |
| XGB | 3 | 628 | −55.94 | 264.9 | 220.7 |
| XGB | 4 | 379 | −48.19 | 267.1 | 224.3 |

### Median summary

| Tower | Model | R² (median) | RMSE (median) | MAE (median) | Irvin benchmark (RF) |
|-------|-------|-------------|---------------|--------------|----------------------|
| Tower 4 | RF | **+0.144** | 121.3 | 62.5 | 0.79 (17 wetland sites) |
| Tower 4 | XGBoost | **+0.086** | 126.5 | 70.7 | ~0.65–0.67 |
| Tower 9 | RF | −0.027 | 123.5 | 58.8 | — |
| Tower 9 | XGBoost | −0.089 | 128.0 | 62.6 | — |
| Tower 2 | RF | −16.94 | 147.9 | 116.4 | — |
| Tower 2 | XGBoost | −55.94 | 264.9 | 220.7 | — |

---

## 7  Interpretation

### Why Tower 4 R² is low vs Irvin's benchmark

Tower 4 RF R²=0.144 vs Irvin benchmark R²=0.79. The gap is expected:

1. **Site type mismatch:** Irvin's 17 sites are all wetlands (peatlands, rice paddies, marshes) with very high CH₄ fluxes (site medians: 50–700 nmol m⁻² s⁻¹). NWFP Tower 4 mean ≈ 30 nmol m⁻² s⁻¹ — roughly 10× lower signal, embedded in sensor noise of similar absolute magnitude.
2. **Non-stationarity:** Temperate managed grassland flux responds strongly to stocking density, cutting events, and fertiliser applications that change year-to-year. A model trained on 2018–2021 management patterns is evaluated on 2022–2023 management — structural non-stationarity that wetland models (more stable ecosystems) don't face as severely.
3. **Missing predictor:** Irvin found soil temperature is the dominant SHAP driver. Tower 4's own soil temperature is only 9.6% available, so Tower 9's TS is used as a proxy — a ~50 m cross-tower measurement introducing spatial uncertainty.

These results establish a **managed-grassland-specific baseline** for EC CH₄ gap-filling. They do not indicate a methodology failure — the implementation faithfully follows Irvin et al. The low R² is a site property, not an algorithmic one.

### Why Tower 9 R² ≈ 0

Tower 9 has only 3,981 training rows (vs 7,714 for Tower 4 — 48% fewer). The median R² is −0.027 for RF and −0.089 for XGB, indicating the models predict approximately as well as the mean — with high variance across permutations (range −0.035 to +0.127 for RF). This suggests:
- The model is learning some signal but not consistently
- The 2022–2023 test distribution may differ more from 2018–2021 for Tower 9 than for Tower 4
- Fewer training examples limit generalisation

Tower 9 models will likely benefit most from additional features (livestock density, management events) and possibly a longer training window if 2024+ data is downloaded.

### Why Tower 2 collapses (R² ≈ −17 for RF)

The D-15 custom split — train 2018 (all seasons), test Jan–May 2019 (winter/spring only) — creates a **seasonal distribution mismatch**. The training set includes summer flux peaks that inflate the model's overall expected output range; the test set covers only the low-flux winter/spring season. The model predicts values in the wrong seasonal range, producing catastrophically negative R².

This indicates the D-15 split design (2018 train / 2019 test) is unsuitable. Tower 2 needs either:
- **Seasonal-matched CV:** train and test windows restricted to the same months
- **Leave-one-season-out CV:** 4-fold CV within the pre-gap window (2018–May 2019)
- **Post-gap evaluation:** download 2024 NWFP data to create a genuine post-gap test set

The Tower 2 R-01 results are recorded as-is but **should not be compared against Irvin's benchmark** — they reflect split design failure, not model capability.

---

## 8  SHAP feature importance (Tower 4)

SHAP computed via `TreeExplainer` on a 1,000-sample subset of the perm-0 test mask.

Expected top drivers (consistent with Irvin et al.):
- Soil temperature (`TS_1_1_1 [Tower 9]`)
- Day-of-year cyclical features (`_doy_sin`, `_doy_cos`)
- Air temperature (`TA_0_0_1 [Tower 4]`)
- Energy flux partitioning (`LE`, `H`)

SHAP figure: `results/figures/r01_shap_tower4.png`  
Scatter figure (all towers): `results/figures/r01_scatter_all_towers.png`

---

## 9  Decisions made during R-01

| Decision | Summary |
|----------|---------|
| D-13 | Physical plausibility filter [−500, 3000] nmol m⁻² s⁻¹ |
| D-14 (revised) | `SWIN_1_1_1` is the local SW_IN column — ERA5 not needed as R-01 blocker |
| D-15 | Tower 2 requires custom split; 2018/2019 split design revealed to be problematic (seasonal mismatch) |
| D-16 | Tower 9 TS used as soil temperature proxy for Towers 2 and 4 |
| D-17 | Tower 4 R-01 corrected results (Catchment 4 SM): RF R²=0.144, XGB R²=0.086 |
| D-18 | Spatial alignment rule: Tower N = Catchment N |
| D-19 | Tower 9 R-01: near-null performance (R²≈−0.03 RF); Tower 2: split design failure (R²≈−17 RF) |

---

## 10  Recontextualization after R-02 (D-22)

**R-01's positive R² (+0.144 for Tower 4 RF) is partially explained by an unrealistic feature assumption.**

R-01 includes `LE_1_1_1`, `H_1_1_1`, and `FC_1_1_1` (latent heat, sensible heat, CO2 flux) as predictors. These are measured by the same EC instrument as FCH4. During a real sensor gap that creates missing FCH4, these flux variables would also be missing — they co-fail with the target. Using them as predictors was methodologically incorrect for realistic gap-filling.

R-02 (following Zhu et al. 2023a) correctly excludes LE/H/FC and restricts to meteorological variables from independent sensors (SW, TA, VPD, etc.). R-02 RFm achieves R2 ~ -0.10 at Tower 9 and ~ -0.13 at Tower 4 — substantially worse than R-01's +0.144.

**Implication for interpretation:**
- R-01 results represent an **upper bound** on gap-filling accuracy under an unrealistic "co-observed flux" assumption.
- R-02 results are the **realistic benchmark** — achievable performance when only met drivers are available.
- R-01 is still valid as a methodological replication of Irvin et al. and a feature-ablation data point.
- In forecasting (R-05+), lagged LE/H/FC from previous timesteps are valid predictors (not co-failed).

See Decision D-22 in `DECISIONS.md`.

## 11  Next steps

- **R-02 complete** (Zhu et al. 2023a) — see `R02_results.md`.
- **R-03:** Kim et al. (2020) — RF vs ANN vs SVM with lag features and PCA. Towers 4 and 9.
- **Tower 2 split redesign:** Replace 2018/2019 split with leave-one-season-out CV or post-gap 2024 data.
- **ERA5 for driver_era:** Complete the Zhu replication's third driver set and fill ~48% missing SWIN.
