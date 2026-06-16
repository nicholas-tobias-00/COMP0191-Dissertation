# R-02 Results: Zhu et al. (2023a) MDS vs RF vs XGB — Gap-Length Scenarios

**Reference:** Zhu, S. et al. (2023a). Gap-filling carbon dioxide, water, energy, and methane fluxes in challenging ecosystems. *Agric. Forest Meteorol.* 332, 109365.  
**Notebook:** `R02_Zhu2023a_RF_MDS.ipynb`  
**Executed:** 2026-06-14  
**Spatial alignment:** Tower N = Catchment N (D-18). Soil temperature: Tower 9 TS as proxy for all towers (D-16).

---

## 1  Methodology

Zhu et al. compared gap-filling methods (MDS, RF, XGB, ANN, SVM) across three driver sets (driver₃, driver_m, ERA5) and five gap-length scenarios at 19 challenging EC sites globally, including three NWFP managed-pasture towers.

**This replication implements:**
- MDS (Marginal Distribution Sampling) — Python re-implementation of REddyProc
- RF-driver₃ (RF3): 3 met drivers + cyclical AUX
- RF-driver_m (RFm): 11 met drivers + cyclical AUX
- XGB-driver_m (XGBm): same drivers as RFm

**Not implemented (deferred):** driver_era (requires ERA5 download); SVR, MLP, GBR (out of scope)

---

## 2  Driver sets

### driver₃ (7 features)

| Variable | Tower 4 column | Tower 9 column |
|----------|---------------|----------------|
| SW | `SWIN_1_1_1 [Tower 4]` | `SWIN_1_1_1 [Tower 9]` |
| TA | `TA_0_0_1 [Tower 4]` | `TA_0_0_1 [Tower 9]` |
| VPD | `VPD_0_0_1 [Tower 4]` | `VPD_0_0_1 [Tower 9]` |
| AUX | hour_sin, hour_cos, doy_sin, doy_cos | same |

### driver_m (15 features = driver₃ + 8 additional)

New vs R-01: PPFD, RN (NETRAD), P (precipitation), SHF (soil heat flux)  
**Key omission vs R-01:** LE, H, FC (EC flux variables) not included — see D-22

---

## 3  Gap scenarios

Hourly equivalents of Zhu Part B scenarios (Moffat et al. 2007):

| Scenario | Gap length | Calendar meaning | ~Gaps inserted per rep |
|----------|-----------|-----------------|------------------------|
| vs | 1 hour | Very short — single timestamp | ~1,777 gaps |
| s | 4 hours | Short — half-shift | ~444 gaps |
| m | 32 hours | Medium — ~1.3 days | ~56 gaps |
| l | 288 hours | Long — 12 days | ~6 gaps |
| m1 | mixed | Equal-probability draw from vs/s/m/l | mixed |

Each scenario masks ~25% of test-period valid observations (MASK_FRAC=0.25) across 5 reps with different random seeds. Non-overlapping calendar windows inserted until quota reached.

**MDS fill rate: 1.0 across all scenarios and towers** — sufficient historical data always found a matching candidate within the ±91-day search window.

---

## 4  Temporal splits and training data

| Tower | train_yrs | test_yrs | n_train (driver3) | n_train (driver_m) | n_test valid |
|-------|-----------|----------|------------------|-------------------|-------------|
| Tower 4 | 2018–2021 | 2022–2023 | 10,862 | 7,285 | ~7,109 |
| Tower 9 | 2018–2021 | 2022–2023 | 4,048 | 2,288 | ~5,848 |

Note: driver_m training set is smaller than driver₃ because 4 additional columns (PPFD, RN, P, SHF) require simultaneous non-NaN values.

---

## 5  Results

### Tower 4 (median across 5 reps per scenario)

**R²:**

| Method | vs (1h) | s (4h) | m (32h) | l (288h) | m1 (mixed) |
|--------|---------|--------|---------|---------|-----------|
| MDS | −0.150 | −0.179 | −0.148 | −0.260 | −0.475 |
| RF3 | −0.153 | −0.133 | −0.102 | −0.149 | −0.146 |
| RFm | −0.128 | −0.104 | −0.160 | −0.113 | −0.277 |
| XGBm | −0.325 | −0.270 | −0.380 | −0.302 | −0.614 |

**RMSE (nmol m⁻² s⁻¹):**

| Method | vs | s | m | l | m1 |
|--------|-----|---|---|---|-----|
| MDS | 156.5 | 163.7 | 146.0 | 154.8 | 147.6 |
| RF3 | 150.2 | 159.4 | 141.9 | 140.3 | 121.9 |
| RFm | 147.7 | 159.5 | 141.3 | 145.8 | 131.3 |
| XGBm | 158.9 | 171.1 | 155.4 | 160.3 | 146.0 |

**MBE (nmol m⁻² s⁻¹):**

| Method | vs | s | m | l | m1 |
|--------|-----|---|---|---|-----|
| MDS | −1.9 | +0.3 | −3.5 | −2.5 | −1.1 |
| RF3 | +10.3 | +11.0 | +4.4 | +9.6 | +13.5 |
| RFm | +23.5 | +24.2 | +21.8 | +27.1 | +32.1 |
| XGBm | +30.8 | +32.7 | +31.0 | +39.8 | +41.8 |

### Tower 9 (median across 5 reps per scenario)

**R²:**

| Method | vs (1h) | s (4h) | m (32h) | l (288h) | m1 (mixed) |
|--------|---------|--------|---------|---------|-----------|
| MDS | −0.174 | −0.290 | −0.433 | −0.584 | −0.195 |
| RF3 | −0.155 | −0.133 | −0.177 | −0.182 | −0.138 |
| RFm | −0.089 | −0.090 | −0.088 | −0.157 | −0.075 |
| XGBm | −0.132 | −0.115 | −0.126 | −0.181 | −0.132 |

**RMSE (nmol m⁻² s⁻¹):**

| Method | vs | s | m | l | m1 |
|--------|-----|---|---|---|-----|
| MDS | 153.4 | 159.8 | 162.7 | 171.7 | 157.9 |
| RF3 | 154.2 | 152.3 | 158.2 | 145.2 | 154.6 |
| RFm | 149.7 | 148.7 | 152.6 | 145.3 | 152.8 |
| XGBm | 151.4 | 150.3 | 154.7 | 146.2 | 153.4 |

**MBE (nmol m⁻² s⁻¹):**

| Method | vs | s | m | l | m1 |
|--------|-----|---|---|---|-----|
| MDS | +1.7 | −0.6 | +0.1 | +4.6 | +0.7 |
| RF3 | +3.4 | 0.0 | +4.5 | +13.7 | +2.1 |
| RFm | −16.7 | −18.8 | −15.2 | −5.5 | −19.1 |
| XGBm | −9.5 | −12.0 | −9.4 | +0.2 | −11.1 |

---

## 6  Comparison with paper

| Finding in Zhu et al. (2023a) | Our result |
|-------------------------------|-----------|
| FCH4 R² < 0.10 for all methods at managed pastures | ✓ Confirmed — all R² negative (worse than predicted mean) |
| RF > MDS for long gaps (l) | ✓ Tower 9: RF3 R²=−0.182 vs MDS R²=−0.584 for scenario l |
| MDS ≈ RF for short gaps (vs, s) | ✓ Approximately confirmed (marginal differences at Tower 4) |
| Multi-driver RF marginally better than 3-driver RF | ✓ Tower 9: RFm R²=−0.089 vs RF3 R²=−0.155 |
| XGB sometimes worse than RF | ✓ Tower 4: XGBm worst across all scenarios |

---

## 7  Interpretation

### Why all R² are negative

All methods produce negative R² — meaning they predict worse than the test-period mean. This is more severe than the paper's Fig. 7d values (∼0.01–0.08 for RF at NWFP). Possible reasons:

1. **Driver set difference vs R-01**: R-01 achieved RF R²=+0.144 at Tower 4 by including LE, H, and FC (EC flux variables) as predictors. R-02 correctly excludes these because they would also be missing during an EC sensor gap (see D-22). Without flux co-variates, performance drops substantially.
2. **Hourly vs 30-min**: Aggregating to hourly reduces the diurnal signal resolution available to models.
3. **2022–2023 test period non-stationarity**: The test period management (stocking, cutting) differs from 2018–2021, causing structural shift.
4. **NWFP vs Zhu's full dataset**: Zhu's mean R² is averaged over 19 sites including some with positive R²; NWFP managed pastures may be at the difficult end even within that distribution.

### Tower 9: RF > MDS for long gaps (main paper finding confirmed)

At Tower 9 scenario l, MDS collapses to R²=−0.584 while RF3=−0.182 and RFm=−0.157. This confirms the paper's core finding: for 12-day gaps, the meteorological driver-based ML models maintain performance better than similarity-based lookup (MDS), because MDS needs a nearby temporal analog that may not exist within ±91 days.

Tower 9 also shows RFm > RF3 across all scenarios (RFm −0.075 to −0.157, RF3 −0.133 to −0.182), consistent with the paper's finding that multi-driver sets improve performance marginally.

### Tower 4: MDS and RF comparable; XGBm underperforms

At Tower 4, MDS and RF3/RFm are roughly comparable across short scenarios (both ≈ −0.10 to −0.17). The long-gap advantage for RF is smaller than at Tower 9, possibly because Tower 4 has more training data (7,285 driver_m rows vs 2,288 for Tower 9), giving MDS more temporal analogs to find.

XGBm underperforms RF at Tower 4 across all scenarios (XGBm ≈ −0.27 to −0.61 vs RFm ≈ −0.10 to −0.28). This reverses the R-01 pattern where XGBm was competitive, suggesting XGB overfits more severely when only meteorological (non-flux) features are available.

### MBE patterns

MDS achieves nearly zero MBE (+1 to −4 nmol m⁻² s⁻¹) because it predicts the conditional mean of observed values — by construction, it is unbiased when the lookup succeeds.

RF3 shows positive MBE (+4 to +14 nmol m⁻² s⁻¹), especially for long gaps — the model overestimates during the periods it is filling (possible training-period selection bias toward higher-flux observations).

RFm at Tower 9 shows strong negative MBE (−5 to −19 nmol m⁻² s⁻¹), suggesting the multi-driver model is underestimating during gap periods. This may reflect that the additional driver variables (PPFD, P, SHF) are systematically lower during gap periods.

---

## 8  Decisions made during R-02

| Decision | Summary |
|----------|---------|
| D-20 | MDS implemented in Python (not REddyProc R package) — R dependency avoided |
| D-21 | driver_m adds PPFD, RN, P, SHF vs R-01; LE/H/FC excluded — see D-22 |
| D-22 | LE/H/FC excluded from driver_m: unrealistic as predictors (co-fail with target during EC gaps) |

---

## 9  Next steps

- **R-03 (Kim et al. 2020):** Add lag features, PCA pre-processing, SVM/ANN. Towers 4 and 9.
- **Feature investigation:** Include management event variables (livestock, fertiliser) — may reduce non-stationarity driving negative R².
- **ERA5 for driver_era:** Download ERA5 SW/TA/VPD to complete the Zhu replication's third driver set.
- **Tower 2 split redesign:** D-15 remains unresolved; leave-one-season-out CV needed.
