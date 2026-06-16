# Gap-Filling Replications — Process Flowcharts & Feature Dictionary

Companion to `gap_filling_summary.md`. Part 1 gives the process flow for each of the three replications; Part 2 lists every feature column used, with a brief description and a flag for whether it was engineered/custom for that specific technique.

All column names follow the `consolidated_hourly.csv` schema. `[Tower N]` / `[Catchment N]` are tower-specific; exact per-tower strings are given where they differ.

---

# Part 1 — Process flowcharts

All three share the same front-end (load → per-tower QC → features → temporal split 2018–2021 / 2022–2023); they diverge at the model/evaluation stage.

## R-01 — Irvin et al. (2021): RF / XGBoost, empirical gap sampling

```
┌─────────────────────────────────────────────────────────┐
│ Load consolidated_hourly.csv (70,153 h × 449 cols)       │
└───────────────────────────┬─────────────────────────────┘
                            ↓   (loop per tower: 4, 9, 2)
┌─────────────────────────────────────────────────────────┐
│ Two-pass QC on FCH4: SSITC∈{0,1} → plausibility[-500,3000]│
└───────────────────────────┬─────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│ Build feat_cols (15): 9 met INCL. LE/H/FC + TS(T9) +      │
│ catchment SWC + 4 cyclical AUX                            │
└───────────────────────────┬─────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│ Measure EMPIRICAL gap-length distribution of the series  │
│ (gap_run_lengths → lengths of real NaN runs)             │
└───────────────────────────┬─────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│ TRAIN (2018–2021): dropna → fit SimpleImputer(mean)      │
│ → fit RF & XGBoost                                        │
└───────────────────────────┬─────────────────────────────┘
                            ↓   (loop ×5 permutations)
┌─────────────────────────────────────────────────────────┐
│ Mask ~40% of TEST(2022–23) valid obs using contiguous    │
│ blocks SAMPLED FROM the empirical gap-length distribution│
│   → impute features → RF.predict, XGB.predict            │
│   → R² / RMSE / MAE vs hidden truth                      │
└───────────────────────────┬─────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│ MEDIAN over 5 perms → 1 number per tower per model       │
│ + SHAP (T4), scatter, time-series → benchmarks.csv       │
└─────────────────────────────────────────────────────────┘
```

## R-02 — Zhu et al. (2023a): MDS vs RF₃ / RFₘ / XGBₘ, fixed gap scenarios

```
┌─────────────────────────────────────────────────────────┐
│ Load → per tower (4, 9) → Two-pass QC → cyclical AUX     │
└───────────────────────────┬─────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│ Build TWO driver sets (NO LE/H/FC — D-22):               │
│   driver₃ (7): SW, TA, VPD + AUX                         │
│   driver_m (15): + PPFD, USTAR, WS, RN, P, TS, SWC, SHF  │
└───────────────────────────┬─────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│ TRAIN (2018–2021): dropna+impute per set                 │
│ → fit RF₃ (driver₃),  RFₘ (driver_m),  XGBₘ (driver_m)   │
└───────────────────────────┬─────────────────────────────┘
                            ↓   (loop: 5 scenarios × 5 reps)
┌─────────────────────────────────────────────────────────┐
│ Scenario gap = {1, 4, 32, 288, mixed} h                  │
│ insert_calendar_gaps: place FIXED-LENGTH windows until   │
│ ~25% of test valid obs masked                            │
│   ├─ save truth → NULL the gap in df                     │
│   ├─ MDS: search ±7/14/28/91 d for same-hour,            │
│   │        similar TA(±2.5°C)/SW(±50) analog → mean      │
│   ├─ RF₃ / RFₘ / XGBₘ: impute + predict at gap points    │
│   ├─ R² / RMSE / MBE / MAE                               │
│   └─ RESTORE truth                                       │
└───────────────────────────┬─────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│ MEDIAN per scenario per model → benchmarks.csv + plots   │
└─────────────────────────────────────────────────────────┘
```

## R-03 — Kim et al. (2020): + lag features, PCA, SVM, ANN

```
┌─────────────────────────────────────────────────────────┐
│ Load → per tower (4, 9) → Two-pass QC → cyclical AUX     │
└───────────────────────────┬─────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│ add_lag_features: SWC & TS lagged 168/336/504/672 h      │
│ (8 cols — analog of Kim's water-table lags, D-23)        │
└───────────────────────────┬─────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│ Build feat_all (15, INCL. LE/H/FC) and                   │
│        feat_lag (23 = feat_all + 8 lags)                 │
└───────────────────────────┬─────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│ TRAIN (2018–2021): impute(feat_all), impute(feat_lag)    │
│   → StandardScaler on feat_all   (for SVM, ANN)          │
│   → PCA(7) on feat_lag           (for RF_PCA7)           │
│ Fit: RF(feat_all) · RF_lag(feat_lag) · RF_PCA7(7 PCs)    │
│      · SVM(scaled) · ANN(scaled)                         │
└───────────────────────────┬─────────────────────────────┘
                            ↓   (loop: 4 scenarios × 5 reps)
┌─────────────────────────────────────────────────────────┐
│ Scenario gap = {1, 32, 288, 768} h                       │
│ insert_calendar_gaps: ~10% of test valid masked          │
│   ├─ save truth → NULL the gap                           │
│   ├─ MDS (reused from R-02)                              │
│   ├─ RF / RF_lag / RF_PCA7 / SVM / ANN → predict         │
│   │     (each with its own impute/scale/PCA transform)   │
│   ├─ R² / RMSE / MBE / MAE                               │
│   └─ RESTORE truth                                       │
└───────────────────────────┬─────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│ MEDIAN per scenario per model → benchmarks.csv + plots   │
└─────────────────────────────────────────────────────────┘
```

### What changes across the three

| Stage | R-01 | R-02 | R-03 |
|---|---|---|---|
| Features | 15 (incl. LE/H/FC) | met-only (7 / 15) | 15 + 8 lags + PCA |
| Co-failed fluxes | ✅ used | ❌ excluded (D-22) | ✅ used |
| Gap construction | empirical lengths, 40% | fixed scenarios, 25% | fixed scenarios, 10% |
| Models | RF, XGB | MDS, RF₃, RFₘ, XGBₘ | MDS, RF, RF_lag, RF_PCA7, SVM, ANN |
| Reporting | 1 number/tower | per gap-length scenario | per gap-length scenario |

---

# Part 2 — Feature dictionary

**Target (all replications):** `FCH4_1_1_1 [Tower N]` — eddy-covariance CH₄ flux, nmol m⁻² s⁻¹.

"**Source**" = `raw` sensor channel vs `engineered` (derived in-notebook). "**Custom for technique**" flags features introduced specifically to replicate a given paper's method.

## R-01 — `feat_cols` (15 features)

| # | Column | Description | Source | Custom for technique |
|---|--------|-------------|--------|----------------------|
| 1 | `TA_0_0_1 [Tower N]` | Air temperature (°C) | raw | — |
| 2 | `SWIN_1_1_1 [Tower N]` | Incoming shortwave radiation (W m⁻²) | raw | — |
| 3 | `PA_0_0_1 [Tower N]` | Atmospheric pressure (kPa) | raw | — |
| 4 | `WS_0_0_1 [Tower N]` | Wind speed (m s⁻¹) | raw | — |
| 5 | `VPD_0_0_1 [Tower N]` | Vapour pressure deficit (hPa) | raw | — |
| 6 | `USTAR_0_0_1 [Tower N]` | Friction velocity (m s⁻¹) | raw | — |
| 7 | `LE_1_1_1 [Tower N]` | Latent heat flux (W m⁻²) — **co-failed EC** (D-22) | raw | — |
| 8 | `H_1_1_1 [Tower N]` | Sensible heat flux (W m⁻²) — **co-failed EC** (D-22) | raw | — |
| 9 | `FC_1_1_1 [Tower N]` | CO₂ flux (µmol m⁻² s⁻¹) — **co-failed EC** (D-22) | raw | — |
| 10 | `TS_1_1_1 [Tower 9]` | Soil temperature (°C) — Tower 9 proxy for all towers (D-16) | raw | — |
| 11 | catchment SWC (see note) | Soil moisture @ 10 cm (%), catchment-matched (D-18) | raw | — |
| 12 | `_hour_sin` | sin(2π·hour/24) — cyclical hour-of-day | engineered | shared |
| 13 | `_hour_cos` | cos(2π·hour/24) — cyclical hour-of-day | engineered | shared |
| 14 | `_doy_sin` | sin(2π·doy/365) — cyclical day-of-year | engineered | shared |
| 15 | `_doy_cos` | cos(2π·doy/365) — cyclical day-of-year | engineered | shared |

**Catchment SWC column (D-18):**
- Tower 4 → `Soil Moisture @ 10cm Depth (%) [Catchment 4 After  2013/08/13]` (note the double space)
- Tower 9 → `Soil Moisture @ 10cm Depth (%) [Catchment 9]`
- Tower 2 → `Soil Moisture @ 10cm Depth (%) [Catchment 2]`

## R-02 — two driver sets (Zhu et al. Table 2). **No LE/H/FC (D-22).**

### `driver3` (7 features)
| # | Column | Description | Source | Custom for technique |
|---|--------|-------------|--------|----------------------|
| 1 | `SWIN_1_1_1 [Tower N]` | Incoming shortwave radiation | raw | — |
| 2 | `TA_0_0_1 [Tower N]` | Air temperature | raw | — |
| 3 | `VPD_0_0_1 [Tower N]` | Vapour pressure deficit | raw | — |
| 4–7 | `_hour_sin/_cos`, `_doy_sin/_cos` | Cyclical time (AUX) | engineered | shared |

### `driver_m` (15 features) = driver₃ met + 8 more
| # | Column | Description | Source | Custom for technique |
|---|--------|-------------|--------|----------------------|
| 1 | `SWIN_1_1_1 [Tower N]` | Incoming shortwave radiation | raw | — |
| 2 | `TA_0_0_1 [Tower N]` | Air temperature | raw | — |
| 3 | `VPD_0_0_1 [Tower N]` | Vapour pressure deficit | raw | — |
| 4 | `PPFD_1_1_1 [Tower N]` | Photosynthetic photon flux density | raw | **new in R-02** (vs R-01) |
| 5 | `USTAR_0_0_1 [Tower N]` | Friction velocity | raw | — |
| 6 | `WS_0_0_1 [Tower N]` | Wind speed | raw | — |
| 7 | `RN_1_1_1 [Tower N]` | Net radiation (NETRAD) | raw | **new in R-02** |
| 8 | Precipitation (see note) | Precipitation (mm), catchment-matched | raw | **new in R-02** |
| 9 | `TS_1_1_1 [Tower 9]` | Soil temperature — Tower 9 proxy (D-16) | raw | — |
| 10 | catchment SWC (see R-01 note) | Soil moisture @ 10 cm, catchment-matched (D-18) | raw | — |
| 11 | `SHF_1_1_1 [Tower N]` | Soil heat flux | raw | **new in R-02** |
| 12–15 | `_hour_sin/_cos`, `_doy_sin/_cos` | Cyclical time (AUX) | engineered | shared |

**Precipitation column (D-18):** Tower 4 → `Precipitation (mm) [Catchment 4 After  2013/08/13]`; Tower 9 → `Precipitation (mm) [Catchment 9]`.

> **MDS uses only two matching variables** — `sw_col = SWIN_1_1_1 [Tower N]` and `ta_col = TA_0_0_1 [Tower N]` — to find analog observations (same hour ±1; similar SW ±50 W m⁻², TA ±2.5 °C). It does **not** consume the full driver vector. The same applies to MDS in R-03.

## R-03 — three representations (Kim et al.)

### `feat_all` (15 features)
Identical to R-01 `feat_cols` (same 9 met incl. LE/H/FC + TS proxy + catchment SWC + 4 AUX). See the R-01 table above. Used by RF; scaled copy used by SVM and ANN.

### `feat_lag` (23 features) = `feat_all` + 8 lag columns
The lag block is **custom-engineered specifically for the Kim replication** (D-23) — the NWFP analog of Kim's lagged water-table height, capturing delayed hydrological/thermal memory of CH₄.

| # | Column | Description | Source | Custom for technique |
|---|--------|-------------|--------|----------------------|
| 16 | `…[Catchment N]__lag168h` | Soil moisture, 1-week lag | engineered | **R-03 lag feature (D-23)** |
| 17 | `…[Catchment N]__lag336h` | Soil moisture, 2-week lag | engineered | **R-03 lag feature (D-23)** |
| 18 | `…[Catchment N]__lag504h` | Soil moisture, 3-week lag | engineered | **R-03 lag feature (D-23)** |
| 19 | `…[Catchment N]__lag672h` | Soil moisture, 4-week lag | engineered | **R-03 lag feature (D-23)** |
| 20 | `TS_1_1_1 [Tower 9]__lag168h` | Soil temperature, 1-week lag | engineered | **R-03 lag feature (D-23)** |
| 21 | `TS_1_1_1 [Tower 9]__lag336h` | Soil temperature, 2-week lag | engineered | **R-03 lag feature (D-23)** |
| 22 | `TS_1_1_1 [Tower 9]__lag504h` | Soil temperature, 3-week lag | engineered | **R-03 lag feature (D-23)** |
| 23 | `TS_1_1_1 [Tower 9]__lag672h` | Soil temperature, 4-week lag | engineered | **R-03 lag feature (D-23)** |

### `feat_pca7` (7 features)
7 principal components of the imputed `feat_lag` matrix (`PCA(n_components=7)`, explained variance > 0.999). **Custom transformation for the Kim replication** — used only by RF_PCA7 to test Kim's "PCA degrades ML" claim. Not human-interpretable columns.

### Per-model feature usage in R-03
| Model | Input representation | Pre-transform |
|-------|----------------------|---------------|
| MDS | SWIN + TA matching only | — |
| RF | `feat_all` (15) | mean impute |
| RF_lag | `feat_lag` (23) | mean impute |
| RF_PCA7 | `feat_pca7` (7 PCs) | mean impute → PCA |
| SVM | `feat_all` (15) | mean impute → StandardScaler |
| ANN | `feat_all` (15) | mean impute → StandardScaler |

---

## Summary: what is custom vs raw

- **Shared engineered features (all three reps):** 4 cyclical time features (`_hour_sin/_cos`, `_doy_sin/_cos`) — standard practice, not paper-specific.
- **R-01:** no paper-specific custom features; uses raw sensor channels + cyclical AUX.
- **R-02:** introduces PPFD, NETRAD (RN), precipitation, and soil heat flux as **new raw drivers** vs R-01 (following Zhu Table 2), and **removes** LE/H/FC (D-22). No engineered features beyond AUX.
- **R-03:** the only replication with **bespoke engineered features** — the 8 SWC/TS **lag columns** (D-23, the water-table-lag analog) and the **7 PCA components**, both created specifically to replicate Kim's methodology.

---

## CO₂-augmented variant (`../03b_gap_filling_CO2/`, D-26)

The 03b experiment adds one engineered input on top of the above:

| Feature | Description | Source | Custom for technique |
|---|---|---|---|
| `FC_1_1_1 [Tower N]` (gap-filled) | CO₂ flux, **observed-where-available** then RFm-reconstructed from met drivers; QC'd per D-25 | engineered (two-stage) | **03b CO₂-augmentation (D-26)** |

Process delta: one step is prepended to the front-end — *load `data/Hourly/fco2_gapfilled.csv` → set `FC` to the gap-filled FCO₂* (R-02 also appends `FC` to `driver_m`). Everything downstream is unchanged. Full flowchart/feature delta and results in `../03b_gap_filling_CO2/co2_augmented_summary.md`.

*Source: `R01_Irvin2021_RF_XGBoost.ipynb`, `R02_Zhu2023a_RF_MDS.ipynb`, `R03_Kim2020_RF_ANN_SVM_MDS_PCA.ipynb` (TOWER_CONFIGS + feature-construction cells); 03b variants + `src/data/fco2_gapfill.py`.*
