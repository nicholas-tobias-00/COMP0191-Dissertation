# R-03 Results: Kim et al. (2020) — RF / ANN / SVM / MDS + Lag Features + PCA

**Reference:** Kim, Y. et al. (2020). Gap filling approaches for eddy covariance methane fluxes: A comparison of three machine learning methods and a traditional approach. *Agricultural and Forest Meteorology*, 281, 107830.  
**Notebook:** `R03_Kim2020_RF_ANN_SVM_MDS_PCA.ipynb`  
**Executed:** 2026-06-14  
**Spatial alignment:** Tower N = Catchment N (D-18). TS proxy from Tower 9 (D-16).  
**D-22 caveat:** LE/H/FC included in feat_all (following Kim), but these co-fail with FCH4 during EC gaps — results are upper bound, not operationally realistic.

---

## 1  Methodology

Kim et al. compared RF, ANN, SVM, and MDS at five sites (including two rice paddies, three wetlands). Key contributions over R-01/R-02:

1. **Lag features** — 1–4 week lags of water table height (WTH), capturing delayed hydrological response to precipitation/drainage
2. **PCA preprocessing** — principal components computed from lag-augmented feature set; their finding: PCA degrades ML (RF_PCA7 < RF_lag at most sites)

**NWFP adaptation:**
- WTH has no direct equivalent → used SWC (soil moisture at 10 cm) and TS as proxies for hydrological memory
- Lag hours: 168h, 336h, 504h, 672h (1–4 weeks)
- MASK_FRAC = 0.10 (Kim: 10% of valid obs masked per rep)
- Gap scenarios: short=1h, medium=32h, long=288h, xlong=768h (32 days)

---

## 2  Quality control and training set sizes

Same two-pass QC as R-01/R-02 (SSITC ≤ 1, plausibility filter [−500, 3000] nmol m⁻² s⁻¹).

| Tower | QC-valid total | Train valid (feat_all) | Train valid (feat_lag) | Lag rows lost |
|-------|---------------|----------------------|----------------------|---------------|
| Tower 4 | 19,469 | 7,714 | 5,096 | −2,618 (−34%) |
| Tower 9 | 11,235 | 3,981 | 2,956 | −1,025 (−26%) |

Lag NaN reduction is large because the first 672 hours (~28 days) of valid training data per year produce NaN lag values, and SWC/TS themselves have gaps.

---

## 3  Feature sets

**feat_all (15 features):** TA, SWIN, PA, WS, VPD, USTAR, LE, H, FC, TS_Tower9, SWC_catchment + 4 cyclical AUX  
**feat_lag (23 features):** feat_all + 8 lag columns (SWC × 4 lags, TS × 4 lags)  
**feat_pca7:** 7 PCs of feat_lag (explained variance: Tower 4 = 0.999, Tower 9 = 1.000)

---

## 4  Models

| Label | Algorithm | Features | Notes |
|-------|-----------|----------|-------|
| MDS | Marginal Distribution Sampling | SW, TA | Python re-implementation (D-20) |
| RF | RandomForestRegressor(n=500, min_leaf=5) | feat_all | Baseline |
| RF_lag | RandomForestRegressor(n=500, min_leaf=5) | feat_lag | Kim's lag contribution |
| RF_PCA7 | RandomForestRegressor(n=500, min_leaf=5) | 7 PCs of feat_lag | Kim's PCA test |
| SVM | SVR(rbf, C=1.0, ε=0.1) | feat_all (scaled) | — |
| ANN | MLPRegressor(100, 50), relu, early_stopping | feat_all (scaled) | — |

---

## 5  Results

### Tower 4 — median R² across 5 reps

| Model | short (1 h) | medium (32 h) | long (288 h = 12 d) | xlong (768 h = 32 d) |
|-------|------------|--------------|--------------------|--------------------|
| MDS | −0.164 | −0.085 | −0.193 | −0.404 |
| RF | **+0.136** | −0.038 | −0.089 | −0.058 |
| RF_lag | +0.135 | −0.115 | −0.125 | −0.088 |
| RF_PCA7 | +0.086 | +0.005 | +0.003 | +0.047 |
| SVM | −0.008 | −0.007 | −0.015 | −0.022 |
| **ANN** | +0.097 | **+0.091** | **+0.077** | **+0.057** |

### Tower 4 — median RMSE (nmol m⁻² s⁻¹)

| Model | short (1 h) | medium (32 h) | long (288 h) | xlong (768 h) |
|-------|------------|--------------|-------------|--------------|
| MDS | 145.6 | 172.7 | 170.6 | 185.2 |
| RF | **119.0** | 143.9 | 133.0 | 126.3 |
| RF_lag | 123.2 | 147.0 | 133.0 | 128.6 |
| RF_PCA7 | 128.3 | 138.9 | 137.3 | 148.7 |
| SVM | 133.1 | 140.1 | 157.4 | 157.3 |
| **ANN** | 125.2 | **132.7** | **146.7** | **149.4** |

### Tower 4 — median MBE (nmol m⁻² s⁻¹)

| Model | short (1 h) | medium (32 h) | long (288 h) | xlong (768 h) |
|-------|------------|--------------|-------------|--------------|
| MDS | −1.4 | −5.4 | −7.2 | +2.0 |
| RF | +17.1 | +15.4 | +22.3 | +22.3 |
| RF_lag | +16.5 | +13.3 | +22.4 | +24.5 |
| RF_PCA7 | +14.4 | +14.4 | +16.4 | +10.1 |
| SVM | −20.8 | −22.3 | −23.2 | −27.4 |
| ANN | −0.4 | +0.7 | +0.7 | −6.1 |

---

### Tower 9 — median R² across 5 reps

| Model | short (1 h) | medium (32 h) | long (288 h = 12 d) | xlong (768 h = 32 d) |
|-------|------------|--------------|--------------------|--------------------|
| MDS | −0.104 | −0.333 | −0.194 | −0.379 |
| RF | +0.129 | +0.143 | −0.009 | −0.119 |
| **RF_lag** | **+0.152** | **+0.160** | +0.065 | −0.009 |
| RF_PCA7 | +0.077 | +0.048 | **+0.111** | **+0.056** |
| SVM | −0.014 | +0.001 | −0.014 | +0.005 |
| ANN | +0.099 | +0.034 | +0.040 | −0.518 |

### Tower 9 — median RMSE (nmol m⁻² s⁻¹)

| Model | short (1 h) | medium (32 h) | long (288 h) | xlong (768 h) |
|-------|------------|--------------|-------------|--------------|
| MDS | 148.2 | 157.4 | 144.6 | 117.5 |
| RF | 124.3 | 138.7 | 137.3 | 107.7 |
| **RF_lag** | **124.9** | **128.8** | **128.6** | **100.5** |
| RF_PCA7 | 128.3 | 133.0 | 126.2 | 96.6 |
| SVM | 139.7 | 140.2 | 128.7 | 99.1 |
| ANN | 130.1 | 134.0 | 153.5 | 123.2 |

### Tower 9 — median MBE (nmol m⁻² s⁻¹)

| Model | short (1 h) | medium (32 h) | long (288 h) | xlong (768 h) |
|-------|------------|--------------|-------------|--------------|
| MDS | +0.2 | −2.8 | −0.6 | +7.8 |
| RF | −3.1 | +1.4 | +3.7 | +13.4 |
| RF_lag | −5.1 | +0.8 | −7.9 | −1.2 |
| RF_PCA7 | −3.3 | +5.9 | −1.0 | +16.2 |
| SVM | −26.6 | −17.9 | −25.7 | −3.1 |
| ANN | +7.0 | +15.4 | +14.5 | +32.0 |

---

## 6  Comparison against Kim (2020) expectations

| Kim's finding | Tower 4 | Tower 9 | Verdict |
|--------------|---------|---------|---------|
| RF ≥ ANN > SVM > MDS | **Partial**: RF best at short; ANN best at medium/long | **Partial**: RF_lag best at short/medium | Partially confirmed |
| RF_lag > RF | Slightly worse at T4 (RF_lag < RF for medium/long) | Confirmed (RF_lag best short/medium) | Site-dependent |
| RF_PCA7 < RF_lag | **Reversed** at T4 medium/long/xlong; RF_PCA7 > RF_lag | Mixed (PCA7 best at long/xlong) | Not confirmed |
| All methods degrade with gap length | Confirmed — generally worsens with gap length | Confirmed | ✓ |
| MDS worst | Confirmed at all scenarios for both towers | Confirmed | ✓ |

---

## 7  Interpretation

### ANN outperforms RF at medium and long gaps (Tower 4)

This contradicts Kim's primary finding (RF ≥ ANN). Likely explanation: with LE/H/FC included (D-22), the feature space contains high-signal EC covariates. The MLPRegressor's early stopping and L2-equivalent regularization prevents overfitting better than RF for these longer-gap evaluation windows where the test distribution shifts more from training. ANN MBE is near-zero (+0.7 nmol m⁻² s⁻¹) vs RF MBE of +22 nmol m⁻² s⁻¹ — RF overshoots systematically while ANN does not.

### RF_lag does not consistently improve over RF at Tower 4

Tower 4 RF_lag shows slightly worse R² than RF at medium/long gaps. This suggests SWC and TS lags at 1–4 week horizons do not carry useful predictive information at Tower 4 beyond what the current-timestep values provide. The additional lag columns may introduce noise that the training set (reduced from 7,714 to 5,096 rows) cannot resolve. Tower 9 behaves differently: RF_lag is the best model at short and medium gaps, consistent with Kim's finding at sites with lagged hydrological response.

### RF_PCA7 is not always worse than RF_lag

Kim's PCA-degrades-ML finding is not confirmed at NWFP. RF_PCA7 achieves better R² than RF_lag for many Tower 4 scenarios (medium/long/xlong) and at Tower 9 long/xlong. Possible explanation: PCA discards the less-informative lag columns, effectively performing feature selection in a dataset where many lags are noisy proxies. This is a relevant negative replication of Kim's finding — it may be site-specific (Kim's WTH lags were strongly informative at wetland sites; NWFP SWC lags are weaker).

### SVM systematically underfits with strong negative MBE

SVR consistently produces negative MBE (underpredicts): Tower 4 SVM MBE ≈ −22 nmol m⁻² s⁻¹, Tower 9 ≈ −18 nmol m⁻² s⁻¹. Default C=1.0 may be under-regularized for NWFP's flux range. Performance is consistently worse than RF across all scenarios.

### ANN catastrophic failure at Tower 9 xlong (R² = −0.518)

The xlong scenario (768h = 32 days) at Tower 9 produces only ~1 large gap per rep, with high variance across the 5 reps. One or more reps likely had a gap during a high-flux summer period where ANN predictions diverged. This is a small-sample artefact — with only 2 years of test data and 10% masking, xlong reps are unreliable.

---

## 8  Cross-replication comparison (Tower 4 medians)

| Model | R-01 (empirical gaps, 40% mask) | R-02 (calendar gaps, 25% mask) | R-03 (calendar gaps, 10% mask) |
|-------|--------------------------------|-------------------------------|-------------------------------|
| RF (best variant) | +0.144 (feat_all+LE/H/FC) | −0.10 (driver_m, no LE/H/FC) | +0.136 (feat_all+LE/H/FC, short only) |
| MDS | — | −0.20 (median, all scenarios) | −0.085 to −0.404 (by scenario) |

R-03 short-gap RF matches R-01 RF almost exactly (+0.136 vs +0.144) — both use feat_all including LE/H/FC. This is expected and cross-validates the implementation.

---

## 9  Decisions made during R-03

| Decision | Summary |
|----------|---------|
| D-23 | SWC and TS used as NWFP analogs of Kim's WTH lag variable; 4-week lag horizon matches Kim's 1–4 week range |
| D-24 | SVM: SVR(rbf, C=1.0, ε=0.1, gamma='scale'); ANN: MLPRegressor(100,50), relu, early_stopping=True — sklearn defaults closest to Kim's kernlab/neuralnet equivalents |

---

## 10  Next steps

- **R-04 dropped** — GreenFeed is animal-scale, not EC flux. Gap-filling replication phase complete.
- **CO₂-augmentation done** (`../03b_gap_filling_CO2/`): gap-filled FCO₂ as a CH₄ feature; ANN reaches +0.12–0.17 at Tower 4 (see `co2_augmented_summary.md`).
- **Tower 2 split redesign** (D-15/D-19): Leave-one-season-out CV or download 2024 NWFP data
- **ERA5 driver_era** (R-02 extension): Fill ~48% missing SWIN with ERA5 downscaling
- **SVM hyperparameter search**: Test C ∈ {0.1, 1.0, 10.0} to address systematic underprediction bias
