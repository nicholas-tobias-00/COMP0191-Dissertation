"""S-01: builds a scenario-conditional daily driver frame for the Phase-07 digital-shadow work.

Climate drivers come from the North Wyke ("NW") CMIP6-based transient scenario files
(`data/Simulated Climate Data/NW.<GCM>.<SSP>.<realization>.dat`, confirmed site-matched to NWFP and
scoped in DECISIONS.md D-46/D-52) -- 4 raw variables only (TMIN, TMAX, RAIN, RAD). Every other
driver this project's forecasting models need is either derived from those 4, derived from the
calendar, or -- per D-52's decision -- historical-day-resampled ("climatology-by-analogue") from
the real 2018-2023 record via `recursive_rollout.doy_climatology` (reused, not reimplemented).
`fx_USTAR_mean`/`fx_SHF_mean` are dropped entirely (deep-research recommendation, S-01 plan): true
EC-tower turbulence quantities with no climate-scenario-product source at all.

Output granularity: ONE representative 365-day "scenario year" (nominal nonleap-year dates, e.g.
2050-01-01.. for bookkeeping only) whose per-day driver values are the climatological mean over the
requested {SSP, GCM set, calendar-year window} -- e.g. "the 2050s under SSP2-4.5" -- not a real
day-by-day weather trajectory. This matches the literature's own convention (coarse/seasonal
resolution, not a precise daily future trajectory) confirmed in this session's deep-research pass.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

ROOT = r"c:\Users\Nicholas\Documents\COMP0191 MSc Artificial Intelligence for Sustainable Development Project"
sys.path.insert(0, ROOT + r"\src")
import models.recursive_rollout as rr  # noqa: E402

CMIP6_DIR = rf"{ROOT}\data\Simulated Climate Data"
HOURLY = rf"{ROOT}\data\Hourly"

# The 4 raw CMIP6 variables -> derived fx_ columns (direct or with a simple transform)
RAD_MJ_TO_WM2 = 1e6 / 86400.0  # MJ/m^2/day (daily integrated total) -> W/m^2 (daily mean)

# Historical-day-resampled columns (D-52's decision) -- everything else the model needs that isn't
# derivable from CMIP6 or the calendar. USTAR/SHF are deliberately absent from this list (dropped).
RESAMPLED_COLS = [
    "fx_WS_mean", "fx_VPD_mean", "fx_RN_mean", "fx_PPFD_mean",
    "fx_SWC_mean", "fx_TS_mean", "fx_wd_sin", "fx_wd_cos",
    "fx_SWC_lag7", "fx_TS_lag7", "fx_SWC_lag14", "fx_TS_lag14",
    "fx_SWC_lag21", "fx_TS_lag21", "fx_SWC_lag28", "fx_TS_lag28",
    "fx_SWC_roll7", "fx_TS_roll7", "fx_SWC_roll14", "fx_TS_roll14",
    "fx_grazing_active", "fx_days_since_grazing",
]
AR_RESAMPLED_COLS = ["ar_ch4_dlag1", "ar_ch4_dlag2", "ar_ch4_dlag3",
                      "ar_ch4_dlag7", "ar_ch4_dlag14", "ar_ch4_drm7", "ar_fc_dlag1"]

DROPPED_COLS = ["fx_USTAR_mean", "fx_SHF_mean"]

GCMS = ["ACCESS-ESM1-5", "CNRM-CM6-1", "HadGEM3-GC31-LL", "MPI-ESM1-2-LR", "MRI-ESM2-0"]
N_REALIZATIONS = 100


def load_cmip6_climatology(ssp, year_start, year_end, gcms=GCMS, n_realizations=N_REALIZATIONS):
    """Load every {gcm, realization} .dat file for `ssp`, restrict to [year_start, year_end], and
    return a day-of-year (1-365) climatology: mean MIN/MAX/RAIN/RAD across every (GCM x realization
    x year) sample for that day-of-year. This is the "ensemble-mean, decadal-window" aggregation
    (e.g. "the 2050s under SSP2-4.5") the S-01 plan and the deep-research pass both specify --
    matches the literature's delta-method/coarse-window convention rather than a single noisy
    calendar-year trajectory."""
    frames = []
    for gcm in gcms:
        for r in range(1, n_realizations + 1):
            path = os.path.join(CMIP6_DIR, f"NW.{gcm}.{ssp}.{r}.dat")
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path, sep=r"\s+", names=["YEAR", "JDAY", "MIN", "MAX", "RAIN", "RAD"],
                              header=None, engine="python")
            df = df[(df["YEAR"] >= year_start) & (df["YEAR"] <= year_end)]
            frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No CMIP6 files found for ssp={ssp} in {CMIP6_DIR}")
    all_df = pd.concat(frames, ignore_index=True)
    n_samples_per_doy = all_df.groupby("JDAY").size()
    clim = all_df.groupby("JDAY")[["MIN", "MAX", "RAIN", "RAD"]].mean()
    print(f"[build_scenario_drivers] ssp={ssp} years={year_start}-{year_end}: "
          f"{len(frames)} realization-files loaded, "
          f"{n_samples_per_doy.mean():.0f} samples/day-of-year on average")
    return clim  # indexed by JDAY 1..365


def build_scenario_fx_frame(tower, T, cmip6_clim, lsu_multiplier=1.0, scenario_year=2050):
    """Builds the full daily fx_/ar_ feature frame for one scenario: a 365-day representative year
    (nominal dates in `scenario_year`, a common/nonleap year, for bookkeeping only) whose driver
    values come from `cmip6_clim` (CMIP6-derived) + `rr.doy_climatology` (historical-day-resampled,
    real 2018-2023 record for `tower`) + calendar formulas, exactly matching what
    `build_forecasting_matrix_v2.py`'s `daily_table()` computes for the historical/real case.

    T: dict {tower: DataFrame} as built by every other B-09-U-03 script (indexed by Datetime,
    columns include y_gapfilled + every fx_/ar_ column).
    lsu_multiplier: scenario livestock-density multiplier (1.0 = baseline, 2.0 = "2x livestock").
    Returns a DataFrame indexed by the 365 nominal dates, with every column the residual/trend
    models need (fx_ + ar_ + is_t2/is_t4/is_t9), EXCLUDING fx_USTAR_mean/fx_SHF_mean by design.
    """
    target_dates = pd.date_range(f"{scenario_year}-01-01", periods=365, freq="D")
    doy = target_dates.dayofyear.values

    out = pd.DataFrame(index=target_dates)
    out["fx_TA_min"] = cmip6_clim.loc[doy, "MIN"].values
    out["fx_TA_max"] = cmip6_clim.loc[doy, "MAX"].values
    out["fx_TA_mean"] = (out["fx_TA_min"] + out["fx_TA_max"]) / 2.0
    out["fx_PRECIP_sum"] = cmip6_clim.loc[doy, "RAIN"].values
    out["fx_SWIN_mean"] = cmip6_clim.loc[doy, "RAD"].values * RAD_MJ_TO_WM2

    out["fx_DOY_sin"] = np.sin(2 * np.pi * doy / 365.0)
    out["fx_DOY_cos"] = np.cos(2 * np.pi * doy / 365.0)
    month = target_dates.month
    out["fx_is_growing"] = np.isin(month, [4, 5, 6, 7, 8, 9]).astype(float)
    out["fx_is_winter"] = np.isin(month, [12, 1, 2]).astype(float)

    dft = T[tower]
    for col in RESAMPLED_COLS:
        hist = dft[col].dropna()
        out[col] = rr.doy_climatology(hist, target_dates, window=7)

    hist_lsu = dft["fx_lsu_dens"].dropna()
    out["fx_lsu_dens"] = rr.doy_climatology(hist_lsu, target_dates, window=7) * lsu_multiplier

    for col in AR_RESAMPLED_COLS:
        hist = dft[col].dropna()
        out[col] = rr.doy_climatology(hist, target_dates, window=7)

    out["is_t2"] = 1.0 if tower == 2 else 0.0
    out["is_t4"] = 1.0 if tower == 4 else 0.0
    out["is_t9"] = 1.0 if tower == 9 else 0.0

    return out


def load_towers():
    """Standard T dict, matching every other B-09-U-03 script's loading convention."""
    dv = pd.read_csv(f"{HOURLY}/forecast_daily_v2.csv", low_memory=False)
    dv["Datetime"] = pd.to_datetime(dv["Datetime"], format="mixed")
    T = {t: dv[dv.tower == t].set_index("Datetime").sort_index() for t in [2, 4, 9]}
    return T
