# Gap-Filling Replications — Synthesis (R-01 / R-02 / R-03)

Consolidated comparison of the three completed EC CH₄ gap-filling replications at the North Wyke Farm Platform (NWFP). This document is the dissertation-ready synthesis of `R01_results.md`, `R02_results.md`, and `R03_results.md`.

**Scope note:** R-04 (Partridge et al. 2024, NWFP GreenFeed Gradient Boosting) has been **dropped** — it predicts animal-scale breath-sample CH₄, not ecosystem-scale EC flux, and is therefore not a valid gap-filling comparison. The gap-filling replication phase is complete at R-01 → R-03.

All three replications share one source table, one quality-control pipeline, and one temporal split, so differences in results are attributable to **method** and **feature set**, not data handling.

| Common element | Value |
|---|---|
| Source | `data/Hourly/consolidated_hourly.csv` — 70,153 hourly rows × 449 cols, 2017-01-01 → 2025-01-02 |
| Target | `FCH4_1_1_1 [Tower N]` (nmol m⁻² s⁻¹) |
| QC pass 1 | Retain SSITC flag ∈ {0, 1}; reject 2 / NaN-flag (`FCH4_SSITC_TEST_1_1_1 [Tower N]`) |
| QC pass 2 | Physical plausibility filter [−500, 3000] nmol m⁻² s⁻¹ (D-13) |
| Towers | Modelled independently (D-11). R-01: Towers 2/4/9. R-02 & R-03: Towers 4/9 (Tower 2 deferred, D-15/D-19) |
| Imputation | `SimpleImputer(strategy="mean")` fit on training data only — no leakage |
| Metric engine | `sklearn.metrics.r2_score`, `mean_squared_error`, `mean_absolute_error` |

QC-valid observation counts (post both passes): Tower 4 = 19,469; Tower 9 = 11,235; Tower 2 = 4,890.

---

## 1  Overview

| Rep | Paper | Methods | Methodological contribution | Mask | Gap structure |
|-----|-------|---------|-----------------------------|------|---------------|
| **R-01** | Irvin et al. (2021) | RF, XGBoost | Empirical gap-length sampling (blocks drawn from observed NaN-run distribution) | 40% | Contiguous blocks, lengths sampled from real gap distribution |
| **R-02** | Zhu et al. (2023a) | MDS, RF3, RFm, XGBm | MDS traditional baseline; meteorology-only drivers (excludes co-failed EC fluxes, D-22) | 25% | Fixed calendar gaps: vs/s/m/l/m1 = 1 / 4 / 32 / 288 / mixed h |
| **R-03** | Kim et al. (2020) | MDS, RF, RF_lag, RF_PCA7, SVM, ANN | Lag features (hydrological memory), PCA preprocessing, SVM, ANN | 10% | Fixed calendar gaps: short/medium/long/xlong = 1 / 32 / 288 / 768 h |

**Reading the progression:** R-01 establishes the tree-based baseline with the full biophysical feature set. R-02 introduces the FLUXNET-standard MDS baseline and — critically — **removes the EC flux co-variates** (LE/H/FC) that are not available during a real instrument gap. R-03 re-introduces them (following Kim) while adding lag features, dimensionality reduction, and two new model families.

---

## 2  Datasets and columns

All feature sets append four cyclical time features ("AUX"): `_hour_sin`, `_hour_cos`, `_doy_sin`, `_doy_cos` (no ordinal leakage). Spatial-alignment rules apply throughout: **catchment-matched soil moisture** (D-18) and **Tower 9 soil temperature as a cross-tower proxy** (D-16, because Tower 4 TS ≈ 9.6% and Tower 2 TS ≈ 5% available).

### R-01 — `feat_cols` (15 features)

| Group | Columns |
|-------|---------|
| Meteorology / EC (9) | `TA_0_0_1`, `SWIN_1_1_1`, `PA_0_0_1`, `WS_0_0_1`, `VPD_0_0_1`, `USTAR_0_0_1`, **`LE_1_1_1`**, **`H_1_1_1`**, **`FC_1_1_1`** `[Tower N]` |
| Soil temperature (1) | `TS_1_1_1 [Tower 9]` (proxy, D-16) |
| Soil moisture (1) | Catchment-matched SWC (D-18) |
| AUX (4) | hour_sin, hour_cos, doy_sin, doy_cos |

> **Includes LE / H / FC** — latent heat, sensible heat, CO₂ flux. These are measured by the same EC instrument as FCH₄.

### R-02 — two driver sets (Zhu et al. Table 2)

| Set | Size | Columns |
|-----|------|---------|
| `driver3` | 7 | `SWIN_1_1_1`, `TA_0_0_1`, `VPD_0_0_1` `[Tower N]` + 4 AUX |
| `driver_m` | 15 | driver3 met + `PPFD_1_1_1`, `USTAR_0_0_1`, `WS_0_0_1`, `RN_1_1_1`, Precipitation (catchment), `TS_1_1_1 [Tower 9]`, catchment SWC, `SHF_1_1_1` + 4 AUX |

> **Excludes LE / H / FC and PA** (D-21, D-22). `driver_m` adds PPFD, NETRAD (RN), precipitation, and soil heat flux versus R-01, but deliberately drops the three EC flux co-variates.

### R-03 — three feature representations (Kim et al.)

| Set | Size | Columns |
|-----|------|---------|
| `feat_all` | 15 | **Identical to R-01 `feat_cols`** — including LE/H/FC |
| `feat_lag` | 23 | `feat_all` + 8 lag columns: SWC and TS each lagged 168, 336, 504, 672 h (1–4 weeks, D-23) |
| `feat_pca7` | 7 PCs | 7 principal components of `feat_lag` (explained variance > 0.999), used by RF_PCA7 |

> SVM (SVR) and ANN (MLPRegressor) operate on `feat_all` after `StandardScaler`. Lag variables (SWC, TS) are the NWFP analogue of Kim's lagged water-table height — proxies for delayed hydrological/thermal memory (D-23).

**Feature-count correction:** the current `R03_results.md` states `feat_all` = 18 and `feat_lag` = 26. Those are miscounts; the authoritative notebook configuration gives **15 and 23** (9 met + TS + SWC + 4 AUX = 15; + 8 lags = 23). The counts in this document match the notebook.

### Feature realism at a glance

| | LE / H / FC included? | Headline R² behaviour |
|---|---|---|
| R-01 (`feat_cols`) | ✅ Yes | Tower 4 RF positive (+0.144) |
| R-02 (`driver_m`) | ❌ No | All methods negative |
| R-03 (`feat_all` / lag) | ✅ Yes | Tower 4/9 recover to ~+0.13–0.16 (short/medium) |

This contrast is the spine of the root-cause analysis (Section 5).

---

## 3  How R² is evaluated

### Common definition

All three replications compute R² with `sklearn.metrics.r2_score(y_true, y_pred)`:

```
R² = 1 − SS_res / SS_tot ,   SS_tot = Σ (y_true − mean(y_true))²
```

The crucial property is that **`SS_tot` is the variance about the mean of the evaluation subset** (the masked observations being scored), *not* the training mean. A model is therefore penalised relative to "always predict the test-subset mean." This makes R² highly sensitive to how the evaluation subset is constructed — central to Section 5.

### Papers' own metric vs our implementation

| Paper | Reported R² | Our implementation |
|-------|-------------|--------------------|
| Irvin et al. (2021) | R² ≈ 0.79 (RF) across 17 FLUXNET-CH4 wetland sites | sklearn r2_score |
| Zhu et al. (2023a) | Predicts FCH₄ R² < 0.10 at managed pastures | sklearn r2_score |
| Kim et al. (2020) | Per-scenario R² of predicted vs observed | sklearn r2_score |

We standardise on `sklearn.r2_score` across all three so the numbers are directly comparable, even where the original papers used slightly different formulations or site-aggregated scores.

### Per-replication evaluation protocol

| | R-01 | R-02 | R-03 |
|---|------|------|------|
| Mask fraction | 40% of test-valid obs | 25% per scenario | 10% per scenario |
| Gap construction | `insert_artificial_gaps` — contiguous blocks, lengths sampled from the **empirical** NaN-run distribution | `insert_calendar_gaps` — **fixed-length** non-overlapping windows | `insert_calendar_gaps` — **fixed-length** non-overlapping windows |
| Repetitions | 5 permutations | 5 reps × 5 scenarios | 5 reps × 4 scenarios |
| Scored subset (MDS) | n/a | filled timestamps only (fill-rate = 1.0) | filled timestamps only (fill-rate = 1.0) |
| Scored subset (ML) | whole masked set | all eval timestamps in gap windows | all eval timestamps in gap windows |
| Aggregation | median across 5 perms → **one number per tower** | median across 5 reps → **per scenario** | median across 5 reps → **per scenario** |

MDS is scored only where it produces a fill (always 100% here, given ≥4 years of background data within the ±91-day search window). ML models predict at every evaluation timestamp.

### Per-tower temporal splits and training-set sizes

| Tower | Split | R-01 n_train | R-02 n_train (driver3 / driver_m) | R-03 n_train (feat_all / feat_lag) |
|-------|-------|-------------|-----------------------------------|------------------------------------|
| Tower 4 | Standard: train 2018–2021 / test 2022–2023 | 7,714 | 10,862 / 7,285 | 7,714 / 5,096 |
| Tower 9 | Standard: train 2018–2021 / test 2022–2023 | 3,981 | 4,048 / 2,288 | 3,981 / 2,956 |
| Tower 2 | D-15 custom: train 2018 / test Jan–May 2019 | 2,985 | — (deferred) | — (deferred) |

Training-set size varies by feature set because `dropna` requires all selected columns to be simultaneously present: `driver_m` is smaller than `driver3` (extra columns), and `feat_lag` is smaller than `feat_all` (the first ~672 h of each block produce NaN lags). Tower 9 has ~48% fewer training rows than Tower 4 throughout. No random splits are used anywhere (D-04).

---

## 4  Metrics summary

All values are **median R²** (sklearn). Positive and best-in-group values in **bold**.

### R-01 (one number per tower — median across 5 permutations)

| Tower | RF | XGBoost | Note |
|-------|----|---------|------|
| Tower 4 | **+0.144** | +0.086 | Apples-to-apples vs Irvin benchmark (0.79) |
| Tower 9 | −0.027 | −0.089 | Near-null; high permutation variance |
| Tower 2 | −16.9 | −55.9 | D-15 split failure (seasonal mismatch) — not comparable |

### R-02 (median R² by gap scenario; met-only drivers, no LE/H/FC)

**Tower 4**
| Method | vs (1h) | s (4h) | m (32h) | l (288h) | m1 (mixed) |
|--------|---------|--------|---------|----------|-----------|
| MDS | −0.150 | −0.179 | −0.148 | −0.260 | −0.475 |
| RF3 | −0.153 | −0.133 | **−0.102** | −0.149 | −0.146 |
| RFm | **−0.128** | **−0.104** | −0.160 | **−0.113** | −0.277 |
| XGBm | −0.325 | −0.270 | −0.380 | −0.302 | −0.614 |

**Tower 9**
| Method | vs (1h) | s (4h) | m (32h) | l (288h) | m1 (mixed) |
|--------|---------|--------|---------|----------|-----------|
| MDS | −0.174 | −0.290 | −0.433 | −0.584 | −0.195 |
| RF3 | −0.155 | −0.133 | −0.177 | −0.182 | −0.138 |
| RFm | **−0.089** | **−0.090** | **−0.088** | **−0.157** | **−0.075** |
| XGBm | −0.132 | −0.115 | −0.126 | −0.181 | −0.132 |

Confirmed paper findings: all methods R² < 0.10; **RF > MDS for long (288 h) gaps** (Tower 9: RF3 −0.182 vs MDS −0.584); multi-driver RFm ≥ 3-driver RF3 at Tower 9; XGBm worst at Tower 4.

### R-03 (median R² by gap scenario; feat_all includes LE/H/FC)

**Tower 4**
| Model | short (1h) | medium (32h) | long (288h) | xlong (768h) |
|-------|-----------|-------------|------------|-------------|
| MDS | −0.164 | −0.085 | −0.193 | −0.404 |
| RF | **+0.136** | −0.038 | −0.089 | −0.058 |
| RF_lag | +0.135 | −0.115 | −0.125 | −0.088 |
| RF_PCA7 | +0.086 | +0.005 | +0.003 | +0.047 |
| SVM | −0.008 | −0.007 | −0.015 | −0.022 |
| ANN | +0.097 | **+0.091** | **+0.077** | **+0.057** |

**Tower 9**
| Model | short (1h) | medium (32h) | long (288h) | xlong (768h) |
|-------|-----------|-------------|------------|-------------|
| MDS | −0.104 | −0.333 | −0.194 | −0.379 |
| RF | +0.129 | +0.143 | −0.009 | −0.119 |
| RF_lag | **+0.152** | **+0.160** | +0.065 | −0.009 |
| RF_PCA7 | +0.077 | +0.048 | **+0.111** | **+0.056** |
| SVM | −0.014 | +0.001 | −0.014 | +0.005 |
| ANN | +0.099 | +0.034 | +0.040 | −0.518 |

R-03 notable deviations from Kim: **ANN > RF at medium/long gaps (Tower 4)**; **lag features help Tower 9 but not Tower 4**; **PCA-degrades-ML NOT confirmed** (RF_PCA7 often matches/beats RF_lag); SVM systematically underpredicts. The ANN xlong −0.518 is a small-sample artefact (~1 gap per rep).

### The cross-walk that ties the three together

> R-01 Tower 4 RF (**+0.144**, feat_cols *with* LE/H/FC) ≈ R-03 Tower 4 RF short-gap (**+0.136**, feat_all *with* LE/H/FC). R-02 RFm (**≈ −0.10 to −0.16**, driver_m *without* LE/H/FC) is consistently negative.

The implementations cross-validate each other: when the feature set and gap regime align, the numbers align. The only structural difference that moves R² from positive to negative is the **presence or absence of the EC flux co-variates** — which leads directly to the root cause.

---

## 5  Root cause for poor metrics

Ranked by explanatory weight.

### 1. Feature realism — the dominant factor (D-22)

LE, H, and FC are measured by the **same EC instrument** as FCH₄. During a real sensor gap that removes FCH₄, these flux co-variates are also missing — they *co-fail* with the target. Including them (R-01, R-03 `feat_all`) lifts Tower 4 RF to ~+0.14; excluding them (R-02 `driver_m`) drives every method negative. The implication is stark: the **operationally realistic ceiling for met-only gap-filling at NWFP is near-zero-to-negative R²**. R-01/R-03 positive numbers are an *upper bound* under an unrealistic "co-observed flux" assumption, not an achievable operational score. (In *forecasting*, lagged LE/H/FC from prior timesteps are legitimate predictors — they are not co-failed.)

### 2. Low signal-to-noise of managed-grassland CH₄

NWFP FCH₄ has mean ≈ 30 and std ≈ 120–170 nmol m⁻² s⁻¹ — the flux is near zero embedded in noise of comparable absolute magnitude. Irvin's 17 wetland sites carry roughly **10× higher signal** (site medians 50–700 nmol m⁻² s⁻¹), which is why their RF reaches R² ≈ 0.79. This is a site property, not an algorithmic failure; the methodology faithfully replicates each paper.

### 3. Non-stationarity between train and test windows

Models train on 2018–2021 and are scored on 2022–2023. Managed-grassland CH₄ responds to stocking density, cutting, and fertiliser events that change year-to-year. A static four-year training window cannot represent the 2022–2023 management regime, so the model's predicted mean/variance is misaligned with the test subset — exactly what R² penalises. Wetland ecosystems (Irvin) are far more temporally stable.

### 4. R² baseline sensitivity to gap structure

Because `SS_tot` references the **evaluation subset's own mean**, the score is highly sensitive to gap geometry:
- **Long fixed gaps** (R-02 `l`, R-03 `long`/`xlong`) isolate a contiguous block whose local mean can sit far from anything a cross-period model predicts → large negative R².
- **Tiny-sample reps** (R-03 `xlong` ≈ one 32-day gap per rep) make the statistic volatile → the ANN −0.518 outlier at Tower 9.
- **Short, interleaved gaps** (R-01 empirical, R-02/R-03 `short`) sit among observed context with a representative local mean → less negative or positive.

This is why the *same model* swings from positive at short gaps to negative at long gaps, and why R-01's 40%-empirical regime reads more favourably than R-02/R-03's isolated long calendar gaps.

### 5. Soil-temperature proxy displacement (D-16)

Soil temperature is consistently a top SHAP driver (Irvin), but Tower 4's own TS sensor is only 9.6% available (Tower 2 ≈ 5%), so **Tower 9 TS is substituted for all towers**. The single most informative predictor is therefore spatially displaced by ~tens of metres, injecting cross-site error precisely where the model most needs signal.

### 6. Per-tower data scarcity and a broken split

Tower 9 trains on ~48% fewer rows than Tower 4, limiting generalisation (and producing high permutation variance, e.g. R-01 RF range −0.035 to +0.127). Tower 2's D-15 split is *structurally* broken — training on all-season 2018 but testing on winter/spring 2019 only creates a seasonal mismatch that yields R² ≈ −17 (RF) / −56 (XGB); those numbers reflect split design, not model capability. In R-03, lag-feature attrition removes a further 26–34% of training rows.

### 7. Hourly aggregation

All modelling uses 1-hour resampled data (D-12), which coarsens the diurnal CH₄ signal relative to the native 30-minute EC resolution — a minor but real loss of within-day structure.

### Forward pointer

Across all three replications, **algorithm choice is clearly not the bottleneck** — MDS, RF, XGBoost, SVM, and ANN cluster within a narrow band, and the one variable that decisively moves R² (LE/H/FC) is unavailable operationally. The highest-value next lever is therefore **management-event features** (livestock density, cutting, fertiliser applications) to attack the non-stationarity in factor 3, rather than further model tuning.

---

## Follow-up experiment

The **CO₂-augmentation experiment** (`../03b_gap_filling_CO2/co2_augmented_summary.md`) directly tests the D-22 root cause: it gap-fills FCO₂ from met drivers and feeds it back as a CH₄ feature. Result — adding FCO₂ to the met-only RFm moves Tower 4 from negative to positive R² (controls unchanged), confirming FC is the decisive predictor.

---

## Source documents

- `notebooks/03_gap_filling/R01_results.md` — Irvin replication, all three towers
- `notebooks/03_gap_filling/R02_results.md` — Zhu replication, MDS vs RF gap-length analysis
- `notebooks/03_gap_filling/R03_results.md` — Kim replication, lag/PCA/SVM/ANN
- `results/benchmarks.csv` — append-only ledger; base replications = 470 rows (30 R-01, 200 R-02, 240 R-03). Total 940 incl. the CO₂-augmented runs (R-0X-CO2).
- `DECISIONS.md` — D-13, D-15, D-16, D-18, D-22, D-23, D-24 referenced above
