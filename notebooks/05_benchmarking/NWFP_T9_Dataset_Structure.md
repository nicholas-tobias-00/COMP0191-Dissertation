# NWFP Tower 9 — Modelling Dataset Structure & Feature Engineering Guide

**Project:** AI for Agriculture: Towards Digital Twins for Methane Emissions Forecasting  
**Site:** North Wyke Farm Platform (NWFP), Devon, UK — Tower 9 (Green Farmlet)  
**Primary target variable:** Daily methane flux `FCH4_daily` (nmol m⁻² s⁻¹)  
**Primary temporal resolution:** Daily aggregates (from 30-min EddyPro output)  
**Approximate dataset size:** ~2,500–2,800 rows pre-QC; ~1,800–2,200 usable post-QC (2018–present)  
**Optional extension:** Sub-daily (hourly) forecasting — see [Appendix A](#appendix-a-hourly-resolution-extension)

---

## 1. Canonical Dataset Structure

> One row = one calendar day. All models are trained from this single table.  
> Model-specific representations (sequences for LSTM/TFT, flat rows for tree models) are derived views, not separate datasets.

---

## 2. Target Variable

| Column | Source | Units | Notes |
|---|---|---|---|
| `FCH4_daily` | EddyPro → gap-filled at 30-min → daily mean | nmol m⁻² s⁻¹ | Gap-fill using RFR **before** daily aggregation; wind speed must be included in gap-fill model |

**Key calibration note:** Zhu et al. (2023a) establishes R² < 0.1 as a realistic baseline for daily CH₄ at managed UK pasture. Low per-day R² does not invalidate the pipeline — cumulative annual sums remain estimable even under high daily noise, provided bias is symmetric.

---

## 3. Feature Groups

### 3.1 Group 1 — Contemporaneous Meteorological Features

Daily aggregates of in-situ met sensors from the Tower 9 EddyPro / met station outputs.

| Column | Aggregation | Source variable | Physical role |
|---|---|---|---|
| `WS_mean` | mean | `WS_0_0_1` | **Dominant driver** — turbulent transport of CH₄ from source to sensor |
| `USTAR_mean` | mean | `USTAR_0_0_1` | Friction velocity; turbulence intensity; correlated with WS but physically distinct |
| `TA_mean` | mean | `TA_1_1_1` | Air temperature; thermal forcing of microbial activity |
| `TA_min` | min | `TA_1_1_1` | Diurnal minimum; relevant for overnight emission suppression |
| `TA_max` | max | `TA_1_1_1` | Diurnal maximum; captures daytime thermal peak |
| `VPD_mean` | mean | `VPD_0_0_1` | Vapour pressure deficit; proxy for atmospheric demand and stomatal state |
| `SWIN_mean` | mean | `SWIN_1_1_1` | Incoming shortwave radiation; photosynthetic driver |
| `RN_mean` | mean | net radiation (if available) | Full radiation budget; more integrative than SWIN alone |
| `PPFD_mean` | mean | `PPFD_0_0_1` | Photosynthetically active radiation; plant productivity proxy |
| `SWC_mean` | mean | average of `SWC_1_1_1`, `SWC_2_1_1`, `SWC_3_1_1` | Soil water content; controls WFPS and methanogenesis/methanotrophy balance |
| `SHF_mean` | mean | average of `SHF_1_1_1`, `SHF_2_1_1`, `SHF_3_1_1` | Soil heat flux; subsurface thermal state |
| `TS_mean` | mean | average of `TS_1_1_1`, `TS_2_1_1`, `TS_3_1_1` | Soil temperature; most important driver of microbial CH₄ production in peatlands and wet grasslands |
| `WD_sin` | circular encoding | `WD_0_0_1` | `sin(WD × π/180)` — **never use raw degrees** |
| `WD_cos` | circular encoding | `WD_0_0_1` | `cos(WD × π/180)` — pair with `WD_sin` to avoid 0/360 discontinuity |
| `PRECIP_daily` | sum | rain gauge (met station) | Daily precipitation total; if available, strong soil moisture driver |

> **Aggregation notes:**
> - For SWC and soil temperature, average across available sensors (up to 3 per tower). If a sensor has missing data on a given day, average the remainder rather than propagating NaN.
> - Wind direction **must** be circular-encoded before use in any ML model. A raw value of 359° and 1° are physically adjacent but numerically distant — this introduces a large artificial error.

---

### 3.2 Group 2 — Lagged Features (System Memory)

CH₄ emissions are driven by antecedent soil conditions, not just instantaneous state. Kim et al. (2020) showed lagged variables improve CH₄ prediction substantially more than CO₂. Feigenwinter et al. (2023) confirmed lagged precipitation and WFPS as the most important predictors for managed grassland CH₄.

**Discrete lags** — point values from N days prior:

| Column | Lag | Rationale |
|---|---|---|
| `SWC_lag7` | 7 days | Short-term soil moisture memory; rapid drainage/wetting response |
| `SWC_lag14` | 14 days | Medium-term antecedent wetness |
| `SWC_lag21` | 21 days | 3-week cumulative moisture state |
| `SWC_lag28` | 28 days | Monthly antecedent condition; upper bound of your specified 168–672 hr range |
| `TS_lag7` | 7 days | Soil thermal memory |
| `TS_lag14` | 14 days | |
| `TS_lag21` | 21 days | |
| `TS_lag28` | 28 days | |

**Rolling means** — smoother alternative, less sensitive to single-day outliers (recommended for tree models):

| Column | Window | Description |
|---|---|---|
| `SWC_roll7` | 7-day trailing mean | Weekly soil moisture average |
| `SWC_roll14` | 14-day trailing mean | Biweekly soil moisture average |
| `TS_roll7` | 7-day trailing mean | Weekly soil temperature average |
| `TS_roll14` | 14-day trailing mean | Biweekly soil temperature average |

> **Implementation note:** When computing lags across data gaps (missing days), use `pandas.DataFrame.shift(N)` on the date-indexed series. Do not fill gaps before lagging — this would introduce information leakage. Rows where a lag value falls within a data gap should have the lag set to `NaN` and be excluded from training, or imputed using a separate strategy (e.g. forward-fill for short gaps ≤ 3 days only).

---

### 3.3 Group 3 — Calendar / Temporal Encodings

Zero-cost features that give every model access to seasonal structure.

| Column | Formula | Type | Notes |
|---|---|---|---|
| `DOY_sin` | `sin(2π × DOY / 365)` | float | **Use this, not raw DOY** — eliminates Dec/Jan discontinuity |
| `DOY_cos` | `cos(2π × DOY / 365)` | float | Pair with `DOY_sin` to fully encode position in annual cycle |
| `month` | integer 1–12 | int | Useful as a categorical split variable in tree models |
| `year` | integer (e.g. 2018) | int | Inter-annual drift / trend proxy; treat as ordinal |
| `week_of_year` | integer 1–52 | int | Finer seasonal resolution than month |
| `is_growing_season` | binary 0/1 | int | 1 if April–September (approx.); encodes phenological state |
| `is_winter` | binary 0/1 | int | 1 if December–February; low-emission baseline period |

> **Important:** Do **not** use raw `DOY` (day-of-year as an integer) as a model feature. It creates an artificial discontinuity at day 365 → 1 that all models will learn as a spurious feature boundary.

---

### 3.4 Group 4 — Management and Biological Covariates

Sparse, discrete, but high-leverage. Ignoring management events guarantees under-prediction of emission peaks (Barczyk et al. 2024; Sharma et al.).

| Column | Type | Source | Notes |
|---|---|---|---|
| `livestock_density` | continuous (LSU/ha or head/ha) | Farm management records | Interpolate linearly between known stocking dates for the Green farmlet; do not forward-fill across known herd removal events |
| `grazing_active` | binary 0/1 | Farm management records | 1 if livestock confirmed present on the Green farmlet that day |
| `days_since_grazing_start` | integer | Derived from management records | Resets to 0 on each new grazing rotation entry; useful for capturing emission build-up dynamics |
| `FCO2_gapfilled` | continuous | EddyPro gap-filled CO₂ flux | Acts as ecosystem productivity state proxy; ecosystem carbon uptake modulates CH₄ substrate availability (Cardenas et al. 2022 — same infrastructure) |

> **If management records are unavailable or incomplete:** Include `livestock_density` as NaN and use the indicator column `grazing_active` as a best-effort binary flag. The scenario analysis module depends on management features being present — excluding them from the model means they cannot be varied in scenarios.

---

### 3.5 Group 5 — QC and Data Quality Flags

**Do not feed these as model input features.** Use them for row filtering, sample weighting, and stratified train/test splitting.

| Column | Source | Use |
|---|---|---|
| `FCH4_qc_flag` | SSITCE QC output (0 = high, 1 = medium, 2 = low) | Filter: train on 0 and 1 only; optionally weight by quality |
| `data_availability` | Fraction of 30-min periods in the day with valid raw observations | Sample weight: down-weight days with < 50% coverage |
| `gap_filled_fraction` | Fraction of the day reconstructed by gap-fill model | Diagnostic: flag days where > 75% is gap-filled as low-confidence targets |
| `tower_id` | Constant = 9 | Future-proof if multi-tower analysis is added |

---

## 4. Model-Specific Feature Representations

> All models consume the same canonical table but require different structural representations.

---

### 4.1 Seasonal Mean & ARIMA (Statistical Baselines)

**Purpose:** Establish the performance floor. These are the benchmarks every ML model must beat to justify complexity.

**Features used:** Target series only (`FCH4_daily`) + calendar structure.

- **Seasonal mean baseline:** Compute mean FCH₄ by `month` (or `week_of_year`) from the training set. Predict test days using the corresponding monthly/weekly mean. This is your absolute minimum benchmark.
- **ARIMA / SARIMA:** Operates on the univariate `FCH4_daily` series. Seasonal ARIMA (SARIMA) with period P=365 (daily) or P=52 (weekly) captures the annual cycle. Use `auto_arima` (pmdarima) to select orders. No exogenous features required for the baseline; optionally extend to ARIMAX with `WS_mean` as an exogenous regressor.

**Key decision:** If ARIMA with `WS_mean` as an exogenous variable performs comparably to RF, this is an important finding that supports the Zeng et al. (2023) caution against assuming architectural complexity is warranted.

---

### 4.2 Tree-Based Models (RF, XGBoost, GBR, Stacking/Voting Regressors)

**Purpose:** Core modelling tier. Each row is treated as independent — temporal structure is injected via lag features, not model architecture.

**Feature table (flat, one row per day):**

```
Target:
  FCH4_daily

Contemporaneous met (Group 1):
  WS_mean, USTAR_mean, TA_mean, TA_min, TA_max, VPD_mean,
  SWIN_mean, RN_mean, PPFD_mean, SWC_mean, SHF_mean, TS_mean,
  WD_sin, WD_cos, PRECIP_daily

Lagged features (Group 2):
  SWC_lag7, SWC_lag14, SWC_lag21, SWC_lag28,
  TS_lag7, TS_lag14, TS_lag21, TS_lag28,
  SWC_roll7, SWC_roll14, TS_roll7, TS_roll14

Calendar encodings (Group 3):
  DOY_sin, DOY_cos, month, year, week_of_year,
  is_growing_season, is_winter

Management / biological (Group 4):
  livestock_density, grazing_active, days_since_grazing_start,
  FCO2_gapfilled
```

**Notes:**
- Tree models are robust to correlated features — no need to remove `TS_mean` because it correlates with `TA_mean`. Let SHAP reveal which is actually used.
- `month` and `year` can be left as integers; tree splits on ordinal integers are valid.
- **Stacking / Voting Regressors** use this same feature table for the base models. The meta-learner (in stacking) trains on out-of-fold predictions from base models — use time-respecting cross-validation folds (forward-chaining, not shuffle-split) to generate them.
- **Do not use `VotingClassifier` or `StackingClassifier`** — this is a regression task. Use `sklearn.ensemble.VotingRegressor` and `StackingRegressor`.

---

### 4.3 LSTM (Deep Learning — Sequential)

**Purpose:** Test whether temporal architecture (sequential inductive bias) improves on tree models for daily CH₄.

**Representation:** 3D tensor `(samples, timesteps, features)` via a sliding window.

**Window configuration:**

| Parameter | Recommended value | Notes |
|---|---|---|
| Lookback window `T` | 14–28 days | Start with 14; tune as a hyperparameter |
| Forecast horizon | 1 day (next-day prediction) | Single-step; extend to multi-step if time allows |
| Stride | 1 day | Slide by one day per sample |

**Features within each window timestep:**

```
Per timestep (T rows of features):
  FCH4_lag (autoregressive — previous FCH4_daily values)
  WS_mean, USTAR_mean, TA_mean, TA_min, TA_max, VPD_mean,
  SWIN_mean, PPFD_mean, SWC_mean, SHF_mean, TS_mean,
  WD_sin, WD_cos, PRECIP_daily,
  DOY_sin, DOY_cos,
  livestock_density, grazing_active, FCO2_gapfilled
```

> **Note on lags within LSTM:** You do not need to pre-compute `SWC_lag7` etc. as separate columns — the LSTM window inherently provides the model access to past SWC values via the lookback. Pre-computed lags are redundant in the sequence input and may confuse the attention mechanisms. Keep the feature set to contemporaneous values within the window.

**Handling data gaps in sequences:** Any window that spans a known data gap (missing date in the sequence) should be excluded from training. Do not bridge gaps — a gap breaks the temporal continuity assumption that LSTM relies on.

**Architecture starting point:**
```python
# Example PyTorch structure
LSTM(input_size=N_features, hidden_size=64, num_layers=2,
     dropout=0.2, batch_first=True)
Linear(64, 1)
```
Tune `hidden_size` (32, 64, 128), `num_layers` (1, 2), and `dropout` (0.1–0.3).

---

### 4.4 Temporal Fusion Transformer — TFT (Deep Learning — Stretch Goal)

**Purpose:** Test multi-input attention architecture with explicit temporal variable selection. TFT's structural advantage is its three-way feature slot classification — it attends differently to known-future, observed-past, and static inputs.

**Feature slot classification (mandatory for TFT):**

| TFT slot | Features | Rationale |
|---|---|---|
| **Static covariates** (time-invariant context) | `tower_id` (= 9), farmlet identifier | Constant across the series; conditions the network's initial state |
| **Known future inputs** (available at forecast time) | `DOY_sin`, `DOY_cos`, `month`, `week_of_year`, `is_growing_season`, `is_winter` | Calendar features are always known in advance — you always know what day it will be |
| **Observed past inputs** (historical only) | `WS_mean`, `USTAR_mean`, `TA_mean`, `TA_min`, `TA_max`, `VPD_mean`, `SWIN_mean`, `RN_mean`, `PPFD_mean`, `SWC_mean`, `SHF_mean`, `TS_mean`, `WD_sin`, `WD_cos`, `PRECIP_daily`, `livestock_density`, `grazing_active`, `FCO2_gapfilled`, `FCH4_daily` (autoregressive) | Met and management features are only observable historically; the model learns to weight them accordingly |

> **Why this classification matters:** In a forecasting scenario, TFT can attend to future calendar features (e.g. "it will be mid-summer in 7 days") while appropriately treating past met observations as the history it is conditioning on. This is architecturally impossible in a vanilla LSTM.

**Recommended library:** PyTorch Forecasting (`TemporalFusionTransformer` class). It handles the three-slot input structure natively via `TimeSeriesDataSet`.

**Architecture starting point:**
```python
TemporalFusionTransformer(
    hidden_size=32,
    attention_head_size=4,
    dropout=0.1,
    hidden_continuous_size=16,
    output_size=7,          # 7 quantiles for probabilistic output
    loss=QuantileLoss()
)
```

---

## 5. Train / Test Split Strategy

> **Never use shuffle-split on a time series.** Always split temporally.

| Split | Period | Notes |
|---|---|---|
| Training set | 2018–2021 | ~4 years; covers multiple seasonal cycles |
| Validation set | 2022 | Used for hyperparameter tuning and early stopping |
| Test set | 2023 (and beyond if available) | Held out completely until final evaluation |

**Cross-validation:** Use time-series-aware cross-validation (forward-chaining / expanding window) for hyperparameter tuning within the training set. Do not use `KFold` — it leaks future data into training folds.

For stacking regressors specifically, generate out-of-fold predictions using `TimeSeriesSplit` from sklearn.

---

## 6. Feature Engineering Implementation Checklist

```
[ ] 1. Load EddyPro CSVs for Tower 9; parse timestamps; set DatetimeIndex
[ ] 2. Gap-fill FCH4 at 30-min resolution using RFR (include WS in gap-fill features)
[ ] 3. Aggregate to daily: mean FCH4 → FCH4_daily; record gap_filled_fraction
[ ] 4. Aggregate met variables to daily means/min/max (Group 1)
[ ] 5. Circular-encode WD: WD_sin, WD_cos (drop raw WD_mean)
[ ] 6. Compute discrete lags: SWC_lag{7,14,21,28}, TS_lag{7,14,21,28}
[ ] 7. Compute rolling means: SWC_roll{7,14}, TS_roll{7,14}
[ ] 8. Compute calendar features: DOY_sin, DOY_cos, month, year, week_of_year, flags
[ ] 9. Join management records: livestock_density, grazing_active, days_since_grazing_start
[ ] 10. Join gap-filled FCO2 as a feature column
[ ] 11. Attach QC flags: FCH4_qc_flag, data_availability (do not use as model features)
[ ] 12. Filter rows: remove FCH4_qc_flag == 2; flag gap_filled_fraction > 0.75
[ ] 13. Verify no future leakage: all lag features computed strictly from past rows
[ ] 14. Confirm no date gaps in index; document any multi-day gaps for LSTM sequence handling
[ ] 15. Temporal split: assign train/val/test labels by year
[ ] 16. Scale continuous features: StandardScaler or MinMaxScaler fit on train set only;
         apply same scaler to val and test (never fit on full dataset)
```

---

## 7. Feature Summary Table (Full Reference)

| # | Column | Group | Type | Model tiers | Notes |
|---|---|---|---|---|---|
| T | `FCH4_daily` | Target | float | All | Gap-filled daily mean; primary prediction target |
| 1 | `WS_mean` | Met | float | All | Dominant driver |
| 2 | `USTAR_mean` | Met | float | All | Turbulence proxy |
| 3 | `TA_mean` | Met | float | All | Mean air temp |
| 4 | `TA_min` | Met | float | Tree, LSTM, TFT | Diurnal minimum |
| 5 | `TA_max` | Met | float | Tree, LSTM, TFT | Diurnal maximum |
| 6 | `VPD_mean` | Met | float | All | Atmospheric demand |
| 7 | `SWIN_mean` | Met | float | All | Incoming shortwave |
| 8 | `RN_mean` | Met | float | Tree, LSTM, TFT | Net radiation |
| 9 | `PPFD_mean` | Met | float | All | PAR proxy |
| 10 | `SWC_mean` | Met | float | All | Soil water content |
| 11 | `SHF_mean` | Met | float | All | Soil heat flux |
| 12 | `TS_mean` | Met | float | All | Soil temperature |
| 13 | `WD_sin` | Met | float | All | Circular wind direction |
| 14 | `WD_cos` | Met | float | All | Circular wind direction |
| 15 | `PRECIP_daily` | Met | float | All | Daily precipitation |
| 16 | `SWC_lag7` | Lag | float | Tree | 7-day SWC lag |
| 17 | `SWC_lag14` | Lag | float | Tree | 14-day SWC lag |
| 18 | `SWC_lag21` | Lag | float | Tree | 21-day SWC lag |
| 19 | `SWC_lag28` | Lag | float | Tree | 28-day SWC lag |
| 20 | `TS_lag7` | Lag | float | Tree | 7-day soil T lag |
| 21 | `TS_lag14` | Lag | float | Tree | 14-day soil T lag |
| 22 | `TS_lag21` | Lag | float | Tree | 21-day soil T lag |
| 23 | `TS_lag28` | Lag | float | Tree | 28-day soil T lag |
| 24 | `SWC_roll7` | Rolling | float | Tree | 7-day rolling mean |
| 25 | `SWC_roll14` | Rolling | float | Tree | 14-day rolling mean |
| 26 | `TS_roll7` | Rolling | float | Tree | 7-day rolling mean |
| 27 | `TS_roll14` | Rolling | float | Tree | 14-day rolling mean |
| 28 | `DOY_sin` | Calendar | float | All | Seasonal encoding |
| 29 | `DOY_cos` | Calendar | float | All | Seasonal encoding |
| 30 | `month` | Calendar | int | Tree, ARIMA | Ordinal month |
| 31 | `year` | Calendar | int | All | Inter-annual trend |
| 32 | `week_of_year` | Calendar | int | Tree, TFT | Finer seasonality |
| 33 | `is_growing_season` | Calendar | binary | Tree, LSTM, TFT | Apr–Sep flag |
| 34 | `is_winter` | Calendar | binary | Tree, LSTM, TFT | Dec–Feb flag |
| 35 | `livestock_density` | Management | float | Tree, LSTM, TFT | Head/ha; interpolated |
| 36 | `grazing_active` | Management | binary | All | Herd presence flag |
| 37 | `days_since_grazing_start` | Management | int | Tree, LSTM, TFT | Rotation counter |
| 38 | `FCO2_gapfilled` | Ecosystem | float | All | CO₂ flux proxy |
| — | `FCH4_qc_flag` | QC | int | Filter only | Not a model feature |
| — | `data_availability` | QC | float | Weight only | Not a model feature |
| — | `gap_filled_fraction` | QC | float | Diagnostic only | Not a model feature |

---

## 8. Key Literature References for Feature Decisions

| Decision | Source |
|---|---|
| Lagged SWC and soil T improve CH₄ prediction substantially more than CO₂ | Kim et al. (2020), *Global Change Biology* |
| Lagged WFPS and precipitation are most important features for managed grassland CH₄ | Feigenwinter et al. (2023), *Agric. For. Meteorol.* |
| Wind speed as dominant EC flux driver at NWFP | Zhu et al. (2023a); EDA results (Tower 9) |
| Daily R² < 0.1 is the expected baseline for CH₄ at UK managed pasture | Zhu et al. (2023a) |
| Management events must be included to avoid peak under-prediction | Barczyk et al. (2024); Sharma et al. |
| Soil temperature as most important predictor across FLUXNET-CH4 wetland sites | Irvin et al. (2021), *Agric. For. Meteorol.* |
| SHAP over WFPS dominates CH₄ prediction in Dutch peatlands | Buzacott et al. (2024), *Global Change Biology* |
| TFT known-future / observed-past / static slot classification | Lim et al. (2021), *Int. J. Forecasting* |
| Simple linear baselines can outperform transformers on LTSF benchmarks | Zeng et al. (2023), *AAAI* |
| Hourly/sub-daily resolution dominated by diurnal dynamics; lower SNR than daily | Zhu et al. (2023a); FLUXNET-CH4 gap-filling literature |

---

## Appendix A: Hourly Resolution Extension

> **Status: Optional bounded extension.** Daily is the primary modelling resolution and the one where all benchmarks, SHAP, UQ, and scenario analysis are conducted. Hourly forecasting is a secondary experiment, gated on completion of the full daily pipeline. Scope strictly: one model (RFR), one comparison metric (daily-aggregated RMSE vs primary daily model). No separate SHAP or UQ analysis at hourly level.

---

### A.1 Why Hourly Is a Harder Problem

Hourly (or 30-min) FCH₄ is a materially different modelling challenge, not simply a resolution change:

| Property | Daily | Hourly |
|---|---|---|
| Signal-to-noise ratio | Higher — diurnal variation averaged out | Lower — dominated by turbulence, artefacts, diurnal swings |
| Dataset size | ~1,800–2,200 rows | ~43,000–53,000 rows (but strongly autocorrelated) |
| Dominant drivers | WS (turbulent transport averaged) | USTAR, PPFD, TA (diurnal cycling); WS still important |
| QC / gap fraction | Manageable pre-aggregation | Gap-filling and forecasting interact — circular dependency risk |
| ARIMA feasibility | SARIMA tractable (P=365 or 52) | Intractable at hourly (P=8,760); STL decomposition required first |
| LSTM window size | 14–28 timesteps (14–28 days) | 336–672 timesteps for equivalent 14–28 day lookback |
| Literature precedent | Standard for forecasting | Standard only for gap-filling, not forecasting |

**Key implication:** What looks like "noise" at hourly resolution is partly real signal (diurnal biological patterns) and partly genuine instrument noise. Your models cannot distinguish these without explicit diurnal structure features.

---

### A.2 Additional Features Required at Hourly Resolution

The canonical daily feature table (Sections 3.1–3.4) applies in full. The following features must be **added** for hourly modelling:

#### Diurnal temporal encodings (mandatory)

| Column | Formula | Notes |
|---|---|---|
| `hour_sin` | `sin(2π × hour / 24)` | **Never use raw hour integer** — same circular encoding rationale as WD and DOY |
| `hour_cos` | `cos(2π × hour / 24)` | Pair with `hour_sin` to fully encode position in diurnal cycle |
| `hour` | integer 0–23 | For tree models as a categorical split variable (supplement to circular encoding) |
| `is_daytime` | binary 0/1 | 1 if SWIN > 5 W m⁻² (radiation-based definition); cleaner than fixed hours |
| `is_nighttime` | binary 0/1 | Complement of `is_daytime`; overnight CH₄ accumulation is physically distinct |

#### Higher-frequency met aggregations

At hourly resolution, use the raw 30-min values (or hourly means of two 30-min periods) rather than daily aggregates. The `TA_min`/`TA_max` columns become unnecessary — the diurnal range is captured directly through the hourly sequence.

#### Autoregressive lags re-specified in hours

| Column | Lag | Equivalent daily lag |
|---|---|---|
| `FCH4_lag1h` | 1 hour | — |
| `FCH4_lag3h` | 3 hours | — |
| `FCH4_lag6h` | 6 hours | — |
| `FCH4_lag24h` | 24 hours | 1-day lag |
| `SWC_lag168h` | 168 hours | 7-day lag |
| `SWC_lag336h` | 336 hours | 14-day lag |
| `SWC_lag504h` | 504 hours | 21-day lag |
| `SWC_lag672h` | 672 hours | 28-day lag |
| `TS_lag168h` | 168 hours | 7-day lag |
| `TS_lag336h` | 336 hours | 14-day lag |
| `TS_lag504h` | 504 hours | 21-day lag |
| `TS_lag672h` | 672 hours | 28-day lag |

> These match exactly the 168–672 hour lag range already specified for the project — they are simply expressed in hours rather than days.

---

### A.3 Model-Specific Changes at Hourly Resolution

#### ARIMA / statistical baselines

Standard SARIMA is computationally intractable at hourly resolution with a seasonal period of 8,760. Use instead:

1. **STL decomposition** (Seasonal-Trend decomposition using LOESS) to extract the diurnal and annual seasonal components separately.
2. **ARIMA on the STL residual** — the residual is approximately stationary and tractable.
3. Alternatively, **Prophet** (Meta's time series library) handles multiple seasonality periods natively and is a practical substitute for SARIMA at sub-daily resolution.

#### Tree models (RFR — recommended for hourly extension)

Same flat-row structure as the daily pipeline. Add `hour_sin`, `hour_cos`, `hour`, `is_daytime` to the feature table. Hourly lags replace daily lags. This is the recommended model for the hourly extension because it requires no architectural changes, trains quickly, and the output can be directly aggregated to daily for comparison.

```python
# Hourly feature table additions (append to daily feature set)
hourly_extra_features = [
    'hour_sin', 'hour_cos', 'hour', 'is_daytime', 'is_nighttime',
    'FCH4_lag1h', 'FCH4_lag3h', 'FCH4_lag6h', 'FCH4_lag24h',
    'SWC_lag168h', 'SWC_lag336h', 'SWC_lag504h', 'SWC_lag672h',
    'TS_lag168h', 'TS_lag336h', 'TS_lag504h', 'TS_lag672h',
]
```

#### LSTM at hourly resolution

Architecturally identical to the daily LSTM but with a much larger window:

| Parameter | Daily setting | Hourly equivalent |
|---|---|---|
| Lookback window `T` | 14–28 days | 336–672 timesteps |
| Batch size | 32–64 | 16–32 (memory pressure) |
| Training time | Low | High — GPU strongly recommended |
| Gap handling | Exclude windows spanning gaps | Same rule; gaps are more frequent at 30-min |

> **Practical warning:** A 672-timestep LSTM window with 20+ features requires substantial GPU memory. If training locally (CPU), restrict the window to 168 timesteps (7 days) and accept the reduced lookback. This is another reason to treat hourly LSTM as out-of-scope unless GPU resources are available.

#### TFT at hourly resolution

The three-slot classification (Section 4.4) is unchanged conceptually. Add `hour_sin`, `hour_cos`, `is_daytime` to the **known future inputs** slot — you always know what time of day it will be at any forecast horizon.

---

### A.4 Evaluation Strategy for the Hourly Extension

The purpose of the hourly extension is a single focused comparison: **does hourly modelling add value when evaluated at daily resolution?**

**Primary comparison metric:**

1. Train the daily RFR on the daily canonical dataset.
2. Train the hourly RFR on the hourly dataset.
3. Aggregate hourly RFR predictions to daily means.
4. Compare daily-aggregated hourly predictions vs daily model predictions using identical test period metrics (RMSE, MAE, R²).

If the daily-aggregated hourly model does not outperform the daily model, conclude that sub-daily resolution adds no forecasting value at the daily budget scale. This is a clean, publishable finding consistent with the Zhu et al. (2023a) gap-filling literature.

**Do not evaluate the hourly model on hourly metrics alone** — hourly R² will be lower than daily R² by construction (more noise) and is not comparable to the daily benchmarks in the literature.

---

### A.5 Scope and Timeline Gate

| Gate condition | Action |
|---|---|
| Daily pipeline complete (all models trained, SHAP done, UQ done) | Proceed to hourly extension |
| Daily pipeline incomplete or behind schedule | Skip hourly extension; note as future work in dissertation |
| Hourly extension exceeds 1 week of active work | Stop; write up partial results or omit |

The hourly extension must not delay dissertation writing, Streamlit development, or the scenario analysis module. Log this gate decision in `DECISIONS.md`.

---

*Last updated: June 2026 | Dissertation: MSc AI for Sustainable Development, UCL | Supervisor: Prof. Phil Harris, Rothamsted Research*
