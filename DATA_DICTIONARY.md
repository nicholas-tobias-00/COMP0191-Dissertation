# DATA_DICTIONARY.md

**Purpose:** a single reference for every input feature used by (1) the gap-filling model (F-08's
RFm) and (2) the forecasting pipeline (B-09→B-15's recursive rollout), with exact derivations,
units, and production-set membership. Every fact below is traced to a specific source file —
verified by direct code reading, not recalled from memory. Two genuine documentation
inconsistencies already present elsewhere in this repo are flagged explicitly, not silently
resolved (matching this project's own "state caveats plainly" convention).

**How to use this**: if you're wiring a new experiment and need to know what a column means, where
it comes from, or whether it's actually used in production — check here first before re-deriving it
from the notebooks. If a new feature is added anywhere in `src/features/` or `src/models/gapfill_rfm.py`,
add it here too.

---

## Part 0 — Raw source files (`data/Consolidated/` → `data/Compiled/`)

Every column in Parts 1–2 ultimately traces back to one of these `data/Consolidated/*.csv` files
(one file per year per type; `01_data_compilation/Compile Datasets.ipynb` merges each group into a
single multi-year file in `data/Compiled/`, 1:1 named).

| Consolidated file(s) | → Compiled | Variables it produces |
|---|---|---|
| `greenhouse_YYYY-01-01_YYYY+1-01-01.csv` (2017–2024) | `greenhouse.csv` | All EC tower flux/met columns — anything with `[Tower N]`: `FCH4_1_1_1`, `FCH4_SSITC_TEST_1_1_1`, `FC_1_1_1` (CO₂ flux), `SWIN_1_1_1`, `TA_0_0_1`, `VPD_0_0_1`, `PPFD_1_1_1`, `RN_1_1_1`, `WS_0_0_1`, `USTAR_0_0_1`, `SHF_1/2/3_1_1`, `WD_0_0_1`, `TS_1_1_1` (Tower 9 cross-tower fallback, D-16) |
| `measurements_YYYY-01-01_YYYY+1-01-01.csv` (2017–2025) | `measurements.csv` | Anything with `[Catchment N]`: `Precipitation (mm)`, `Soil Moisture @ 10cm Depth (%)`, `Soil Temperature @ 15cm Depth (oC)`, plus flow data (unused downstream) |
| `Animal_location_counts_Cattle Basic Data_*.csv` | `Animal_location_counts_Cattle_Basic_Data.csv` | `cattle_Catchment N` daily head counts → `lsu_dens`, `graze`/`fx_grazing_active` (§1.4) |
| `Animal_location_counts_Breeding Sheep Basic Data_*.csv` | `Animal_location_counts_Breeding_Sheep_Basic_Data.csv` | `sheep_Catchment N` → same |
| `Animal_location_counts_Lamb Basic Data_*.csv` | `Animal_location_counts_Lamb_Basic_Data.csv` | `lamb_Catchment N` → same |
| `Field Event Data_Format 1_*.csv` | `Field_Event_Data_Format_1.csv` | `mgmt_cut`/`mgmt_manure` recency (+22 unused pruned mgmt columns — fertN, lime, cultiv, other scopes; §1.6) |

**Derived, not a direct raw source:** `GPP`/`Reco` are computed by `reddyproc_pipeline.py` from
`FC`+`TA` (Lloyd-Taylor/Reichstein partitioning) — not a Consolidated column themselves. The
external-sourced (SMS/MET) `SWIN`/`TA`/`WS` variant (F-08's EC-vs-external swap, D-35) comes from a
different portal source entirely via `build_sms_met_dataset.py`, not from `Consolidated/greenhouse`.

**Compiled but never used downstream** (present in `data/Consolidated/`, touched only by the
one-off compilation notebook, absent from every `src/` script): `Feed Type Data_Format 1_*.csv`;
`Field Survey Data_Format 1_{Botanical,Grain,Herbage,Silage Cut,Soil Chemistry & Physics} Survey_*.csv`;
`Cattle/Lamb/Breeding Sheep {Basic Data, Location Data, Sales Data, Weight Data} Format 1_*.csv` and
`Cattle/Breeding Sheep Condition Score Data_*.csv`. **Easy confusion to avoid**: these per-animal
`*_Basic_Data_Format_1` files are a *different* file family from `Animal_location_counts_*_Basic_Data`
— only the latter (head counts per catchment) feeds `lsu_dens`/`graze`; the former are compiled into
`livestock_weight_long.csv`/`livestock_condition_score_long.csv` and go nowhere near the model.

---

## Part 1 — Gap-filling model (F-08 RFm)

**Production config:** partial-pooled (T2+T4+T9 + tower dummies), external-sourced (SMS/MET) RandomForest
gap-filler — `src/models/gapfill_rfm.py` (shared module used by both `notebooks/04_feature_engineering/
F08_external_sensors_RFm.ipynb` and `src/data/build_fch4_gapfilled.py`).

### 1.1 Target

| Column | Derivation | Unit | Notes |
|---|---|---|---|
| `target` (→ `FCH4_1_1_1 [Tower t]`) | SSITC gate: rows where `FCH4_SSITC_TEST_1_1_1` not in `{0,1}` → NaN; then plausibility filter `[-500, 3000]` → NaN (D-13) | nmol CH₄ m⁻² s⁻¹ | Not a feature — the value being gap-filled |

### 1.2 Hyperparameters & training logic

```python
imp = SimpleImputer(strategy="mean")
rf  = RandomForestRegressor(n_estimators=500, min_samples_leaf=5, n_jobs=-1, random_state=42)
```
- **Training set is pooled across all 3 towers, always** (`pooled=True` in every production call) —
  identical training rows regardless of which tower is being filled; only the tower-dummy value of
  the row being *predicted* differs.
- **Backfill pattern**: `filled = target.where(obs, rf.predict(...))` — real value where observed,
  RF prediction everywhere else. This is `y_gapfilled` (dense, no gaps).
- Feature imputation: mean-imputation on the feature matrix only (not the target), fit on training
  rows, applied to all rows at predict time.

### 1.3 Feature table (30 generic + 3 tower dummies = 33 total, always pooled)

| Column (model name) | Underlying raw source | Derivation | Unit | Source file |
|---|---|---|---|---|
| `SWIN_1_1_1` | EXT: `Solar Radiation (W/m2) [Site]`; EC: `SWIN_1_1_1 [Tower N]` | gap-filled (`mdc_gapfill`, `__f` suffix) | W m⁻² | `build_sms_met_dataset.py`, `reddyproc_pipeline.py` |
| `TA_0_0_1` | EXT: `Air Temperature (oC) [Site]`; EC: `TA_0_0_1 [Tower N]` | gap-filled `__f` | °C | same |
| `VPD_0_0_1` | `VPD_0_0_1 [Tower N]` (EC only, no external twin) | gap-filled `__f`; plausibility-filtered `[0,15]` pre-fill (D-48) | **⚠ unit conflict** — see §1.5 | `reddyproc_pipeline.py` |
| `PPFD_1_1_1` | `PPFD_1_1_1 [Tower N]` (EC only) | gap-filled `__f` | not documented anywhere in-repo — see §1.5 | `reddyproc_pipeline.py` |
| `RN_1_1_1` | `RN_1_1_1 [Tower N]` (EC only, net radiation) | gap-filled `__f` | W m⁻² (inferred by analogy, not explicitly labeled) | `reddyproc_pipeline.py` |
| `WS_0_0_1` | EXT: `Wind Speed (km/h) [Site]` ÷3.6; EC: `WS_0_0_1 [Tower N]` | gap-filled `__f` | m s⁻¹ | `build_sms_met_dataset.py` |
| `USTAR_0_0_1` | `USTAR_0_0_1 [Tower N]` (EC only, friction velocity) | gap-filled `__f`; plausibility `[0,3]` (D-48) | m s⁻¹ | `reddyproc_pipeline.py` |
| `SHF_1_1_1` | `SHF_1_1_1 [Tower N]` (EC only, soil heat flux) | gap-filled `__f` | W m⁻² (inferred, not explicitly labeled) | `reddyproc_pipeline.py` |
| `Precipitation (mm)` | `Precipitation (mm) [Catchment N]` (already external, D-18) | gap-filled `__f` | mm/hour | `reddyproc_pipeline.py` |
| `Soil Temperature @ 15cm Depth (oC)` | EXT: per-catchment; EC: cross-tower proxy `TS_1_1_1 [Tower 9]` (D-16) | gap-filled `__f` | °C | `build_sms_met_dataset.py`, `reddyproc_pipeline.py` |
| `Soil Moisture @ 10cm Depth (%)` | `[Catchment N]` (already external, D-18) | gap-filled `__f` | % (volumetric) | `reddyproc_pipeline.py` |
| `fc` | `FC_gapfilled [Tower N]` | observed FC (SSITC∈{0,1}, plausibility `[-100,100]`) where available, else RFm reconstruction from met-only drivers (D-25/D-26) | µmol CO₂ m⁻² s⁻¹ (D-25) | `fco2_gapfill.py`, wired via `gapfill_rfm.load_ext()` |
| `_hs`, `_hc` | `Datetime.hour` | `sin/cos(2π·hour/24)` | dimensionless | `gapfill_rfm.py::frame()` |
| `_ds`, `_dc` | `Datetime.dayofyear` | `sin/cos(2π·doy/365)` | dimensionless | `gapfill_rfm.py::frame()` |
| `lsu_dens` | `cattle_/sheep_/lamb_Catchment N` (`livestock_hourly.csv`) | `(cattle·1.0 + sheep·0.1 + lamb·0.05) / AREA_ha`; `AREA={2:6.65,4:7.75,9:7.75}` ha | **LSU ha⁻¹** | `gapfill_rfm.py::frame()` — see §1.4 |
| `graze` | same livestock columns | `1.0` if total head count > 0 in own catchment, else `0.0` | binary | `gapfill_rfm.py::frame()` |
| `swc_l168/336/504/672` | own tower's Soil Moisture `__f` | `.shift(lag)` at 168/336/504/672 hours (1–4 weeks, D-23) | % (lagged) | `gapfill_rfm.py::frame()` |
| `ts_l168/336/504/672` | own tower's Soil Temperature `__f` | `.shift(lag)` at 168/336/504/672 hours | °C (lagged) | `gapfill_rfm.py::frame()` |
| `mgmt_cut` | `mgmt_t{t}_cut_recency` | `exp(-days_since_last_cut / 21)` | dimensionless [0,1] | `build_management_features.py` (pruned set, see §1.6) |
| `mgmt_manure` | `mgmt_t{t}_manure_recency` | `exp(-days_since_last_manure / 30)` | dimensionless [0,1] | `build_management_features.py` (pruned set) |
| `gpp` | `GPP [Tower N]` | Reichstein/Lloyd-Taylor nighttime partitioning of `fc`, `=max(Reco−NEE,0)`, forced 0 at night | µmol CO₂ m⁻² s⁻¹ (basis of FC) | `reddyproc_pipeline.py` |
| `reco` | `Reco [Tower N]` | Lloyd-Taylor(T) with global E0 + 7-day block Rref refit | µmol CO₂ m⁻² s⁻¹ | `reddyproc_pipeline.py` |
| `is_t2`, `is_t4`, `is_t9` | tower identity | one-hot, 1.0 for own tower else 0.0 | binary | `gapfill_rfm.py::frame()` (only when `pooled=True`, i.e. always in production) |

**Not model features** (present in source data but explicitly excluded): `RH`, `WD` (wind direction) —
docstring states plainly "not model features."

### 1.4 Livestock density (`lsu_dens`) — the project's #1 driver, shared by both pipelines

```python
LSU = {"cattle": 1.0, "sheep": 0.1, "lamb": 0.05}
lsu_dens = sum(head_count[species] * weight for species, weight in LSU.items()) / AREA_ha
AREA = {2: 6.65, 4: 7.75, 9: 7.75}   # hectares, per NWFP_UG_Design_Develop.pdf Appendix D
```
Source columns: `cattle_Catchment N`, `sheep_Catchment N`, `lamb_Catchment N` (or
`Catchment 4 After  2013/08/13` for tower 4) in `data/Hourly/livestock_hourly.csv`.

### 1.5 Known documentation inconsistencies (flagged, not resolved)

- **VPD unit conflict**: `notebooks/03_gap_filling/gap_filling_flowcharts_and_features.md` states
  `VPD_0_0_1` is in **hPa**; `src/data/reddyproc_pipeline.py` and `F09_results.md` both treat/bound
  it in **kPa** (the D-48 plausibility filter `[0,15]` is stated as kPa, and "VPD rarely exceeds
  ~8 kPa" reasoning is used). The kPa treatment is what's actually implemented in the code path
  feeding the gap-filler — but the inconsistency between the two docs was never reconciled.
- **PPFD/SHF units not documented anywhere in-repo** — standard eddy-covariance convention would be
  µmol photons m⁻² s⁻¹ for PPFD and W m⁻² for SHF, but neither is explicitly stated in any code
  comment or notebook markdown found. Don't assume — verify against the raw NWFP data dictionary if
  precision matters for a new use case.
- **`mgmt_*_fertN_rate` has no standardized unit** — the raw `Application_rate_per_ha` column's
  `Units` field varies per event/product in the source data. Moot for F-08 (this feature is pruned
  from production anyway, see §1.6).

### 1.6 Pruned management features — why only 2 of 24 columns are used

`build_management_features.py` produces 24 `mgmt_*` columns (5 channels × 4 scopes ×
recency, + fertN rate × 4 scopes): channels `{fertN, manure, cut, lime, cultiv}` at scopes
`{site, t2, t4, t9}`.

**Only `mgmt_t{t}_cut_recency` and `mgmt_t{t}_manure_recency`** (own-tower-catchment scope) are fed
to the production RFm — the other 22 columns exist in `data/Hourly/management_features.csv` but are
unused.

**Why**: F-01 tested the full 12-column set (at the time) and found it caused mild R² loss at Tower 4
and a **collapse to R²=−0.86 at Tower 9** — small training sets combined with a management-timing
distribution shift (the Red-farmlet arable conversion changed farm-wide campaign timing) caused
overfitting. F-02 (D-29) pruned to just the tower-specific cut+manure pair, fixing Tower 9. F-05
(D-32) re-confirmed this pruned pair gives a small, non-harmful bump on the richer F-04 feature base
(+0.005 to +0.013 R² across towers) — kept because cheap and never harmful, not because it's a big
driver.

### 1.7 u* filtering — computed, not applied to CH4

`reddyproc_pipeline.py`'s binned-plateau u* threshold method (a documented simplification of the
bootstrapped Moving-Point-Test, Papale 2006) produces `ustar_filtered [Tower N]` (a 0/1 flag,
nighttime + low-turbulence). Example thresholds: T2=0.043, T4=0.106, T9=0.121 m/s. **This flag is
never applied to CH4** anywhere in the pipeline — "u*-filtering of CH4 is debated (ebullition); flag
only," per the code's own comment. It is not in the RFm's `feat_list()`.

---

## Part 2 — Forecasting pipeline (B-09→B-15 recursive rollout)

**Two different feature sources depending on model family — this is a real, load-bearing
distinction, not a stylistic one:**

| Model family | Feature source | Resolution | fx_ column count |
|---|---|---|---|
| RF, XGB, LightGBM, SARIMAX (tree/statistical) | `data/Hourly/forecast_daily_v2.csv` | Daily | 34 |
| DLinear, LSTM, LSTM-VSN, TFT (all DL models) | `data/Hourly/forecast_features_v2.csv` | Hourly (resampled to daily internally for Track B) | 26 |

The DL models **never load `forecast_daily_v2.csv`** — `forecasting_dl.py::tower_series()` builds
its own daily series by resampling the *hourly* matrix. This means the richer daily-native features
(`fx_SWC_lag*`, `fx_TA_min/max`, `fx_DOY_sin/cos` at daily granularity, etc.) are **not** seen by any
DL model — they only get the hourly feature set's daily mean. This is a real asymmetry (already
noted informally in this project as "an existing B-03-vs-B-04 asymmetry") — worth knowing before
assuming "the models all see the same inputs."

### 2.1 Targets (`forecast_daily_v2.csv`)

| Column | Derivation | Role |
|---|---|---|
| `y_observed` | `raw FCH4.resample("D").mean().where(hourly_count >= 6)` — daily mean of QC'd raw flux, only where ≥6 hourly observations exist that day | Evaluation target (D-36/D-37: never trained on) |
| `y_gapfilled` | `FCH4_gapfilled [Tower t].resample("D").mean()` (dense, from Part 1's RFm) | Training target + AR-history seed |

### 2.2 AR (recursive/autoregressive) features — `forecast_daily_v2.csv`, both model families

| Column | Derivation | Recursion behavior |
|---|---|---|
| `ar_ch4_dlag1/2/3/7/14` | `y_gapfilled.shift(L)` for L∈{1,2,3,7,14} days | Recomputed every rollout day from the growing real+predicted chain |
| `ar_ch4_drm7` | `y_gapfilled.shift(1).rolling(7, min_periods=1).mean()` | Recomputed every rollout day |
| `ar_fc_dlag1` | `fc.resample("D").mean().shift(1)` — lagged-only daily FCO2 | **Not** recursion-dependent (real historical FCO2, never a model's own FCH4 prediction) |

`src/models/recursive_rollout.py::ar_features_for_day()` mirrors this exact shift/rolling math — the
only feature-construction logic hardcoded inside the shared rollout module (plus its monthly
analogue, `AR_LAGS_MONTHLY=[1,2,3]`, for the B-11 monthly experiment). Every other feature
(`fx_*`, dummies) is supplied by the calling script, not computed by `recursive_rollout.py` itself.

### 2.3 Exogenous (`fx_`) features — daily, `forecast_daily_v2.csv` (34 columns, tree/SARIMAX track)

Fed as **real historical values throughout the rollout** (perfect-foresight assumption) — only the
AR columns above are recursive.

| Column | Derivation | Unit |
|---|---|---|
| `fx_WS_mean` | `WS_0_0_1.resample("D").mean()` | m s⁻¹ |
| `fx_USTAR_mean` | `USTAR_0_0_1.resample("D").mean()` | m s⁻¹ |
| `fx_TA_mean` / `fx_TA_min` / `fx_TA_max` | `TA_0_0_1.resample("D").mean()/min()/max()` | °C |
| `fx_VPD_mean` | `VPD_0_0_1.resample("D").mean()` | ⚠ see §1.5 unit conflict |
| `fx_SWIN_mean` | `SWIN_1_1_1.resample("D").mean()` | W m⁻² |
| `fx_RN_mean` | `RN_1_1_1.resample("D").mean()` | W m⁻² (inferred) |
| `fx_PPFD_mean` | `PPFD_1_1_1.resample("D").mean()` | not documented in-repo |
| `fx_SWC_mean` | per-catchment Soil Moisture, `.resample("D").mean()` (external, D-35) | % |
| `fx_TS_mean` | per-catchment Soil Temperature, `.resample("D").mean()` (external, D-35) | °C |
| `fx_SHF_mean` | mean of `SHF_1_1_1/2/3` (3-sensor mean — distinct from the gap-filler's single-sensor `SHF_1_1_1`), `.resample("D").mean()` | W m⁻² (inferred) |
| `fx_PRECIP_sum` | `Precipitation (mm).resample("D").sum()` | mm/day |
| `fx_wd_sin` / `fx_wd_cos` | `sin/cos(deg2rad(WD_0_0_1)).resample("D").mean()` — circular daily mean | dimensionless |
| `fx_SWC_lag7/14/21/28` | `fx_SWC_mean.shift(L)` | % (lagged) |
| `fx_TS_lag7/14/21/28` | `fx_TS_mean.shift(L)` | °C (lagged) |
| `fx_SWC_roll7/14` | `fx_SWC_mean.rolling(W, min_periods=1).mean()` | % (rolling) |
| `fx_TS_roll7/14` | `fx_TS_mean.rolling(W, min_periods=1).mean()` | °C (rolling) |
| `fx_DOY_sin` / `fx_DOY_cos` | `sin/cos(2π·dayofyear/365)` | dimensionless |
| `fx_is_growing` | `month ∈ {4,5,6,7,8,9}` | binary |
| `fx_is_winter` | `month ∈ {12,1,2}` | binary |
| `fx_lsu_dens` | see §1.4 (same formula, same source) | LSU ha⁻¹ |
| `fx_grazing_active` | `(daily max head-count > 0)` | binary |
| `fx_days_since_grazing` | counter resets to 0 at each grazing-spell onset, else increments | days |

Plus `is_t2`/`is_t4`/`is_t9` (pooling dummies, same as gap-filling).

### 2.4 Exogenous (`fx_`) features — hourly, `forecast_features_v2.csv` (26 columns, DL track)

- **19 base columns** (from `build_forecasting_matrix.py`, reused by v2 unchanged): 11 met/soil
  (`fx_SWIN_1_1_1`, `fx_TA_0_0_1`, `fx_VPD_0_0_1`, `fx_PPFD_1_1_1`, `fx_RN_1_1_1`, `fx_WS_0_0_1`,
  `fx_USTAR_0_0_1`, `fx_SHF_1_1_1`, `fx_Precipitation (mm)`, `fx_Soil Temperature @ 15cm Depth (oC)`,
  `fx_Soil Moisture @ 10cm Depth (%)`), 4 planned/management (`fx_lsu_dens`, `fx_graze`,
  `fx_mgmt_cut`, `fx_mgmt_manure`), 4 calendar (`fx_hs`, `fx_hc`, `fx_ds`, `fx_dc` — hour-of-day and
  day-of-year sin/cos, **at hourly resolution**, distinct from the daily table's `fx_DOY_sin/cos`).
- **7 new columns added by v2** (`build_forecasting_matrix_v2.py::hourly_new()`): `fx_wd_sin`,
  `fx_wd_cos` (raw hourly, not resampled), `fx_is_daytime` (`SWIN_1_1_1 > 5`), `fx_shf3` (3-sensor
  mean), `fx_is_growing`, `fx_is_winter`, `fx_days_since_grazing` (computed daily, broadcast to hourly
  rows — only changes once per day).
- Also present: `FLUX_T = ["ar_fc_t", "ar_gpp_t", "ar_reco_t"]` — realized-flux-at-origin features,
  **encoder-only** (never in the decoder/future input, since they wouldn't be knowable in advance).

**DL encoder tensor** = CH4 channel (1) + `fx_*` (26) + `FLUX_T` (3) = **30 channels**.
**DL decoder tensor** = `fx_*` (26) = **26 channels**.

### 2.5 SARIMAX's narrower exogenous subset (`EXOG_B`)

```python
EXOG_B = ["fx_lsu_dens", "fx_WS_mean", "fx_VPD_mean", "fx_USTAR_mean", "fx_PPFD_mean",
          "fx_DOY_sin", "fx_DOY_cos", "fx_is_growing"]
```
8 columns, not the full 34 — a deliberately narrow, SHAP-informed subset (from I-01's
`results/fc_importance_shap_rf.csv`, which ranks `fx_lsu_dens` as the #1 forecasting driver by a
wide margin, followed by wind speed/VPD/USTAR/PPFD), chosen to keep SARIMAX's exogenous-regressor
state space small enough for tractable MLE fitting at this row count. Two seasonality proxies
(day-of-year sin/cos) plus `fx_is_growing` round out the set. **AR lags are deliberately excluded**
from `EXOG_B` — SARIMAX's own `(p,d,q)` terms already model that autocorrelation structure;
including a lagged target as an exogenous regressor would be redundant. `EXOG_B` is defined
identically in `b10_b13_rerun_multi_anchor.py` and `B03a_arima.ipynb` — not inside
`recursive_rollout.py` itself.

### 2.6 Train/test split & pooling (recap — see the fuller conversational explanation for more detail)

- Train = everything `<= anchor` (a fixed Dec-16 date per year, 2018–2022); predict = `anchor+1`
  through `anchor+365` (365-day recursive rollout, no real ground truth available).
- **Pooled across all 3 towers, fit once per anchor**: RF, XGB, LightGBM, TFT, DLinear, LSTM
  (rolled out separately per tower afterward, reusing the one fitted model).
- **Never pooled, fit fresh per tower per anchor**: SARIMAX, TabPFN.
- Training target = `y_gapfilled` throughout; evaluation = `y_observed` (with a secondary,
  explicitly-caveated `y_gapfilled`-target metric added later, D-65 addendum — see
  `notebooks/05_benchmarking/b10_b13_metrics_rerun.md`).

---

## Cross-cutting notes

- **`fx_lsu_dens` (livestock density) is the single most important feature in both pipelines** —
  confirmed independently by F-01's gap-filling SHAP ranking and I-01's forecasting SHAP ranking.
  Same formula, same source data, in both Part 1 and Part 2.
- **The VPD unit conflict (§1.5) affects both pipelines** — `fx_VPD_mean` (forecasting) is
  downstream of the same `VPD_0_0_1` column the gap-filler uses, so the hPa-vs-kPa documentation
  inconsistency propagates to both.
- **The DL-vs-tree feature-source asymmetry (top of Part 2) is the single most consequential
  "hidden" fact in this dictionary** — anyone building a new DL-track experiment should not assume
  it sees the same daily-native features (soil lags, TA min/max, etc.) that the tree/SARIMAX track
  does; it only sees the hourly file's daily-resampled mean.
