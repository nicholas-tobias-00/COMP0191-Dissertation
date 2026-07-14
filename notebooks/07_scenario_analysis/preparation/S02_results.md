# S-02 (D-69) — Driver-reconstruction feasibility: proxy models for CMIP6's missing scenario variables

**Status: preparation/feasibility pass, complete. Mixed result — real signal found for 3 of 6
variables, but a serious extrapolation caveat means nothing here is ready to wire into production.**
Nothing in `src/features/build_scenario_drivers.py` or `notebooks/07_scenario_analysis/
S01_first_scenario.ipynb` was touched. This document exists to be checked against, not to
implicitly recommend adoption.

## Motivation

Phase 07's climate-scenario pipeline (D-52/D-64, S-01) has a known, already-documented gap: the
CMIP6-derived climate dataset (`data/Simulated Climate Data/`) provides only 4 daily variables —
`Tmin, Tmax, Rain, RAD` (solar radiation), confirmed by reading the raw `.dat` files directly (no
header row, 6 whitespace-delimited columns `YEAR JDAY MIN MAX RAIN RAD`). Every other driver the
forecasting feature matrix needs (`fx_WS_mean`, `fx_VPD_mean`, `fx_PPFD_mean`, `fx_RN_mean`,
`fx_SWC_mean`, `fx_TS_mean`) is currently filled via historical-day-climatology-resampling
(`rr.doy_climatology()`, D-52's decision) — which samples a real historical day regardless of how
extreme or different the actual simulated future day's *available* drivers are. `fx_USTAR_mean`/
`fx_SHF_mean` are dropped entirely (D-64 — a physical/data-availability argument, not a statistical
one) and stayed out of scope here.

The user proposed a genuinely new alternative, confirmed via research to have never been considered
in D-52/D-64 (which only ever weighed a raw Copernicus CMIP6 pull vs. the climatology resampling
that was actually adopted): train small proxy models predicting the 6 missing variables **from**
the 4 available ones, using real historical NWFP data to fit/validate, then applying the trained
relationship to the actual simulated future driver trajectory. This is more scenario-responsive in
principle than climatology, since it would actually respond to the specific simulated TA/precip/RAD
values for a given future day rather than ignoring them entirely.

**Pre-registered feasibility argument** (given to the user before any code was written): D-50's
correlation matrix predicted an uneven picture — strong for soil temperature (`TA`-`TS` r=0.742),
moderate for PPFD/RN (r=0.48–0.56 with TA/RAD), weak for wind speed/VPD (r=−0.11 to 0.35, both
already in I-01's lowest SHAP-importance tier). **The user chose to attempt all 6 variables anyway**
(not a narrower pilot), for a complete, honest picture rather than pre-judging outcomes from linear
correlation alone.

## Method

**Architecture**: `src/data/fco2_gapfill.py`'s exact precedent (D-26) — a `RandomForestRegressor`
reconstructing one variable from others, calendar-based train/test split — adapted from
hourly/single-target to daily/6-target, pooled across all 3 towers with tower dummies (D-30's
partial-pooling default), rather than 18 separate single-tower models.

- **Predictors** (15 columns, shared across all 6 targets for simplicity): `fx_TA_mean`,
  `fx_TA_min`, `fx_TA_max`, `fx_PRECIP_sum`, `fx_SWIN_mean` (the exact 5 columns
  `build_scenario_drivers.py` already derives from CMIP6, including the same
  `RAD_MJ_TO_WM2 = 1e6/86400.0` unit conversion) + calendar (`fx_DOY_sin/cos`, `fx_is_growing`,
  `fx_is_winter`) + 7/14/28-day rolling `fx_PRECIP_sum` (antecedent-precipitation features, added
  specifically because D-50 found soil moisture is driven more by precipitation *history* than
  same-day precipitation) + tower dummies (`is_t2/is_t4/is_t9`).
- **Targets** (6 variables): `fx_WS_mean`, `fx_VPD_mean`, `fx_PPFD_mean`, `fx_RN_mean`,
  `fx_TS_mean`, `fx_SWC_mean` — already present as clean daily aggregates in
  `data/Hourly/forecast_daily_v2.csv` (external/EC-cleaned met layer, D-35). **Coverage: 100% at
  all 3 towers** (these are the already-gap-filled REddyProc-style daily aggregates, D-33/D-35 —
  not raw hourly EC readings, which are considerably sparser; worth remembering this experiment
  reconstructs an already-once-imputed series in gap periods, not pure raw sensor data).
- **Model**: `RandomForestRegressor(n_estimators=500, min_samples_leaf=5, n_jobs=-1,
  random_state=42)` + `SimpleImputer(strategy="mean")` — `fco2_gapfill.py`'s exact hyperparameters,
  reused verbatim, no new HPO.
- **Split**: train 2018–2021, test 2022–2023 (D-04 — no random splits, calendar-based only).
- **Baseline (the critical, otherwise-missing comparison)**: the real production
  `rr.doy_climatology()` function (not reimplemented), fit on each tower's own training-year
  history, predicting the same held-out test dates — exactly how `build_scenario_drivers.py`
  already generates these variables. This answers "does a trained proxy actually beat what's
  already being done," not just "does the proxy have positive R²."
- **Extrapolation check**: `scenario_hybrid.dissimilarity_index()` (already built for S-01, reused
  unmodified) applied to the real CMIP6 future driver trajectory — SSP2-4.5, 2041–2060
  ensemble-mean across 5 GCMs × 100 realizations (500 files, same convention S-01 already
  established), via `build_scenario_drivers.load_cmip6_climatology()`.

## Results

### Per-tower detail

| Variable | Tower | n (test) | RF R² | RF RMSE | Climatology R² | Climatology RMSE | Winner |
|---|---|---|---|---|---|---|---|
| `fx_WS_mean` | 2 | 730 | 0.407 | 1.180 | 0.038 | 1.504 | RF |
| `fx_WS_mean` | 4 | 730 | 0.415 | 1.173 | 0.038 | 1.504 | RF |
| `fx_WS_mean` | 9 | 730 | 0.268 | 1.312 | 0.038 | 1.504 | RF |
| `fx_VPD_mean` | 2 | 730 | 0.163 | 3.389 | 0.105 | 3.505 | RF |
| `fx_VPD_mean` | 4 | 730 | 0.084 | 2.751 | −0.099 | 3.012 | RF |
| `fx_VPD_mean` | 9 | 730 | −0.248 | 1.876 | −0.950 | 2.345 | RF |
| `fx_PPFD_mean` | 2 | 730 | 0.037 | 107.096 | −0.034 | 110.991 | RF |
| `fx_PPFD_mean` | 4 | 730 | 0.875 | 50.481 | 0.545 | 96.204 | RF |
| `fx_PPFD_mean` | 9 | 730 | 0.609 | 80.652 | 0.360 | 103.206 | RF |
| `fx_RN_mean` | 2 | 730 | 0.039 | 37.634 | 0.081 | 36.804 | Climatology |
| `fx_RN_mean` | 4 | 730 | 0.812 | 22.186 | 0.616 | 31.691 | RF |
| `fx_RN_mean` | 9 | 730 | 0.496 | 32.056 | 0.365 | 35.991 | RF |
| `fx_TS_mean` | 2 | 730 | 0.695 | 1.724 | 0.569 | 2.051 | RF |
| `fx_TS_mean` | 4 | 730 | −2.836 | 3.002 | −2.285 | 2.778 | Climatology |
| `fx_TS_mean` | 9 | 730 | −1.630 | 3.610 | −1.414 | 3.459 | Climatology |
| `fx_SWC_mean` | 2 | 730 | 0.442 | 6.267 | 0.242 | 7.306 | RF |
| `fx_SWC_mean` | 4 | 730 | −2.111 | 6.287 | −1.136 | 5.210 | Climatology |
| `fx_SWC_mean` | 9 | 730 | −0.317 | 3.883 | −0.428 | 4.043 | RF |

### All-tower pooled verdict (n-weighted)

| Variable | RF R² | RF RMSE | Climatology R² | Climatology RMSE | Winner |
|---|---|---|---|---|---|
| **`fx_PPFD_mean`** | **0.507** | 79.410 | 0.290 | 103.467 | **RF** |
| **`fx_RN_mean`** | **0.449** | 30.625 | 0.354 | 34.829 | **RF** |
| **`fx_WS_mean`** | **0.363** | 1.222 | 0.038 | 1.504 | **RF** (largest relative win) |
| `fx_VPD_mean` | −0.000 | 2.672 | −0.315 | 2.954 | RF (both weak) |
| `fx_SWC_mean` | −0.662 | 5.479 | −0.441 | 5.520 | Climatology |
| `fx_TS_mean` | −1.257 | 2.779 | −1.043 | 2.763 | Climatology |

Note Tower 2 is the one exception where `fx_TS_mean`/`fx_SWC_mean` post positive R² for both
methods — consistent with this project's repeated finding that Tower 2 behaves differently from
Towers 4/9 (here, favorably; usually the reverse), not a data-quality flag on its own.

## Findings

**1. Wind speed is the strongest relative RF win, despite being the correlation-evidence's weakest
candidate.** D-50's linear correlation predicted `fx_WS_mean` (r=−0.11 to 0.31 with anything
available) would be the hardest variable to reconstruct. Empirically it shows the largest *relative*
improvement over climatology (R²: 0.038→0.363, roughly a 10x jump in explained variance). This
confirms the reasoning stated before running anything: plain Pearson correlation misses
nonlinear/interaction structure a tree model can exploit — pre-judging feasibility from linear r
alone would have wrongly written this variable off.

**2. Soil temperature and soil moisture fail for BOTH methods, despite `TS` having the *strongest*
linear correlation of any candidate (r=0.742).** This is the opposite of the pre-registered
expectation. Root-caused with a quick, honest follow-up check (not pre-planned, done live after
seeing the anomaly): **test-period (2022–2023) variance is roughly half of training-period
(2018–2021) variance for these two variables at Tower 4** (`fx_TS_mean` std 3.56→1.53;
`fx_SWC_mean` std 6.96→3.57). R² divides by test-set variance, so this kind of train/test
distributional shift is punishing for *any* predictor, including an unbiased one — this is a
methodological artifact specific to soil variables in this particular train/test split, not
evidence the underlying TA–TS relationship is weak (which D-50's r=0.742 already argues against).

**3. Extrapolation check: 100% of 2041–2060 SSP2-4.5 scenario days are flagged outside the real
historical training envelope, at all 3 towers.**

| Tower | % of scenario days outside training envelope |
|---|---|
| 2 | 100.0 |
| 4 | 100.0 |
| 9 | 100.0 |

This is a stronger, more serious result than "some extrapolation risk" — it means every proxy model
built here, including the genuine winners (PPFD/RN/WS), would be applied entirely outside its
validated range if used for real future scenario projection as currently constructed. Plausibly
partly an artifact of the CMIP6 ensemble-mean's own construction (averaged across 500
GCM×realization files, so it has less day-to-day texture/variance than any single real year — a
smoothed trajectory can look "different" from real historical daily data almost by construction,
independent of whether the climate itself has genuinely shifted) rather than purely a climate-shift
signal — but this remains a real, serious caveat regardless of how much of the 100% is attributable
to each cause.

## Verdict

**PPFD, RN, and WS show genuine, validated within-envelope skill over the current climatology
baseline** and are legitimate candidates for a follow-up integration decision — but only after
addressing the 100%-extrapolation finding (e.g. re-testing against individual GCM/realization
trajectories, which retain real day-to-day texture, rather than the smoothed ensemble mean, to see
whether that specifically closes the gap).

**VPD shows no real skill either way** (both methods near-zero/negative R², RF only "wins" because
climatology is actively worse here).

**TS and soil moisture are not good candidates for this specific approach as built** — climatology
wins on pooled R² — though the variance-shift root cause suggests the underlying idea isn't
necessarily disproven for soil variables, just under-tested by this particular train/test split.

**Bottom line: real, non-trivial statistical signal was found for half the candidate variables, but
the practical question — can any of this actually be used for genuine future scenario
projection — remains open given the 100%-extrapolation result.** Nothing here should be treated as
ready to adopt.

## Caveats, stated plainly

- Targets are already-gap-filled daily aggregates (REddyProc-style, D-33/D-35), not raw sensor
  readings — coverage was 100% at every tower/variable, which is a property of the upstream
  gap-filling pipeline, not evidence these are pristine direct observations throughout.
- Single train/test split (2018–2021 / 2022–2023), no cross-validation — the TS/SWC variance-shift
  finding is specific to this split; a different split could show a different pattern.
- No new HPO anywhere — `fco2_gapfill.py`'s hyperparameters reused verbatim; a variable showing
  weak-but-not-hopeless RF skill (e.g. VPD) was not given a tuning pass.
- The AOA/extrapolation check's exact quantitative severity (100% vs. some lower number) is
  entangled with the ensemble-mean-smoothing artifact described in Finding 3 — the qualitative
  conclusion (real, serious extrapolation risk) is robust; the precise 100% figure should not be
  over-interpreted as a calibrated probability.
- This experiment does not revisit `fx_USTAR_mean`/`fx_SHF_mean`'s exclusion (D-64) — that decision
  was made on physical/data-availability grounds, not statistical ones, and remains unchanged.

## Files

- `notebooks/07_scenario_analysis/preparation/S02_driver_reconstruction_feasibility.ipynb` (new,
  executed, all outputs saved inline).
- `results/s02_driver_reconstruction_summary.csv` (per-tower detail, 18 rows).
- `results/s02_driver_reconstruction_pooled.csv` (all-tower pooled verdict, 6 rows).
- `DECISIONS.md` D-69 (full write-up), `CONTEXT.md` status bullet.
- **Reused, unmodified**: `src/data/fco2_gapfill.py` (methodology template only), `src/features/
  build_scenario_drivers.py` (`load_towers()`, `load_cmip6_climatology()`, `RAD_MJ_TO_WM2`),
  `src/models/recursive_rollout.py` (`doy_climatology()`), `src/models/scenario_hybrid.py`
  (`dissimilarity_index()`).
- **Not touched**: `src/features/build_scenario_drivers.py` (no edits, read-only import),
  `notebooks/07_scenario_analysis/S01_first_scenario.ipynb`, any existing Phase 07/06/05 artifact.
