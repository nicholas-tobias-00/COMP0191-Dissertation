"""S-04: transient (real year-specific, not ensemble-mean-climatology) scenario driver frame for
the 2025-2050 trajectory work. Sibling to build_scenario_drivers.py (imports from it, does not
modify it) -- S-01's own build_scenario_fx_frame() only accepts a pre-aggregated day-of-year
climatology (load_cmip6_climatology()'s output, which discards the YEAR dimension via
.groupby("JDAY").mean()); this module adds the missing year-by-year path.

Split into two functions deliberately, for a real performance reason: doy_climatology() (the
expensive part of frame construction -- a Python loop over 365 target dates, called ~28 times per
frame for RESAMPLED_COLS/AR_RESAMPLED_COLS/lsu_dens/optionally USTAR+SHF) depends only on
(tower, year, include_ustar_shf) -- NOT on {SSP, GCM, realization, livestock multiplier}. At S-04's
scale (up to 500 realizations x 2 SSPs x 3 towers x 26 years x 3 multipliers for the primary hybrid
model), recomputing that climatology on every one of those combinations would be a large, wholly
avoidable cost. `build_climatology_base()` computes and caches the (tower, year)-only part ONCE;
`overlay_transient_drivers()` is a cheap vectorized overlay of the (SSP, GCM, realization,
multiplier)-dependent part on top of that cached base.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = r"c:\Users\Nicholas\Documents\COMP0191 MSc Artificial Intelligence for Sustainable Development Project"
sys.path.insert(0, ROOT + r"\src")
sys.path.insert(0, ROOT + r"\src\features")
import models.recursive_rollout as rr  # noqa: E402
from build_scenario_drivers import (  # noqa: E402
    CMIP6_DIR, RAD_MJ_TO_WM2, RESAMPLED_COLS, AR_RESAMPLED_COLS, DROPPED_COLS, GCMS,
)


def load_transient_years(gcm, ssp, realization, years):
    """Reads one NW.<gcm>.<ssp>.<realization>.dat file ONCE, returns {year: 365-row df} for every
    requested year -- avoids re-parsing the same file once per year (26x redundant I/O at S-04's
    annual scale otherwise). Raw real values, no aggregation across years/GCMs/realizations,
    the direct opposite of load_cmip6_climatology()'s groupby-collapse."""
    path = os.path.join(CMIP6_DIR, f"NW.{gcm}.{ssp}.{realization}.dat")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path, sep=r"\s+", names=["YEAR", "JDAY", "MIN", "MAX", "RAIN", "RAD"],
                      header=None, engine="python")
    out = {}
    for year in years:
        yr_df = df[df["YEAR"] == year].sort_values("JDAY").reset_index(drop=True)
        if len(yr_df) != 365:
            raise ValueError(f"{path}: expected 365 rows for year={year}, got {len(yr_df)}")
        out[year] = yr_df
    return out


def build_climatology_base(tower, T, year, include_ustar_shf=False):
    """Everything in the scenario driver frame that does NOT depend on {SSP, GCM, realization,
    livestock multiplier} -- cache the return value per (tower, year, include_ustar_shf) and reuse
    it across every (SSP, GCM, realization, multiplier) combination for that (tower, year).

    Every non-CMIP6 driver uses rr.doy_climatology() over the FULL real historical record as
    `hist` -- exactly S-01's own convention, explicitly NOT S-03's pre-anchor-only restriction
    (there is no leakage risk for a genuinely blind future the way there was for S-03's real
    historical anchors, which evaluated on real historical dates with a real future to leak from).

    fx_lsu_dens is returned UNMULTIPLIED as `fx_lsu_dens_base` -- overlay_transient_drivers()
    applies the scenario's livestock multiplier at overlay time so this cached base serves every
    multiplier without recomputation.

    Note on date indexing: nominal dates use pd.date_range(f"{year}-01-01", periods=365) -- the
    raw CMIP6 files use LARS-WG's fixed 365-day-per-year calendar (no leap days), so for a real
    leap year this nominal index runs one day "behind" the true calendar by December (e.g. ending
    Dec 30, not Dec 31) -- a cosmetic labeling artifact inherited from the source data's own
    convention, not a data error, consistent with how S-01 already treats scenario dates as
    nominal/bookkeeping only.
    """
    target_dates = pd.date_range(f"{year}-01-01", periods=365, freq="D")
    doy = target_dates.dayofyear.values

    out = pd.DataFrame(index=target_dates)
    out["fx_DOY_sin"] = np.sin(2 * np.pi * doy / 365.0)
    out["fx_DOY_cos"] = np.cos(2 * np.pi * doy / 365.0)
    month = target_dates.month
    out["fx_is_growing"] = np.isin(month, [4, 5, 6, 7, 8, 9]).astype(float)
    out["fx_is_winter"] = np.isin(month, [12, 1, 2]).astype(float)

    dft = T[tower]
    for col in RESAMPLED_COLS:
        hist = dft[col].dropna()
        out[col] = rr.doy_climatology(hist, target_dates, window=7)

    if include_ustar_shf:
        for col in DROPPED_COLS:  # fx_USTAR_mean, fx_SHF_mean
            hist = dft[col].dropna()
            out[col] = rr.doy_climatology(hist, target_dates, window=7)

    hist_lsu = dft["fx_lsu_dens"].dropna()
    out["fx_lsu_dens_base"] = rr.doy_climatology(hist_lsu, target_dates, window=7)

    for col in AR_RESAMPLED_COLS:
        hist = dft[col].dropna()
        out[col] = rr.doy_climatology(hist, target_dates, window=7)

    out["is_t2"] = 1.0 if tower == 2 else 0.0
    out["is_t4"] = 1.0 if tower == 4 else 0.0
    out["is_t9"] = 1.0 if tower == 9 else 0.0
    return out


def overlay_transient_drivers(clim_base, transient_year_df, lsu_multiplier):
    """Cheap, vectorized per-(SSP, GCM, realization, multiplier) overlay on a cached
    build_climatology_base() frame: plugs in real year-specific MIN/MAX/RAIN/RAD (derived
    fx_TA_min/max/mean, fx_PRECIP_sum, fx_SWIN_mean) and applies the livestock multiplier to the
    cached fx_lsu_dens_base. Returns a new DataFrame -- clim_base itself is never mutated, so the
    same cached base is safely reused across every scenario built from it."""
    out = clim_base.copy()
    out["fx_TA_min"] = transient_year_df["MIN"].values
    out["fx_TA_max"] = transient_year_df["MAX"].values
    out["fx_TA_mean"] = (out["fx_TA_min"] + out["fx_TA_max"]) / 2.0
    out["fx_PRECIP_sum"] = transient_year_df["RAIN"].values
    out["fx_SWIN_mean"] = transient_year_df["RAD"].values * RAD_MJ_TO_WM2
    out["fx_lsu_dens"] = out["fx_lsu_dens_base"] * lsu_multiplier
    return out.drop(columns=["fx_lsu_dens_base"])


def stratified_realizations(n_per_gcm, gcms=GCMS, seed=42):
    """n_per_gcm realizations drawn per GCM (stratified across all 5 GCMs, not from a single GCM)
    -- e.g. n_per_gcm=4 with the default 5 GCMs gives 20 total (gcm, realization) pairs. Realization
    numbers 1..n_per_gcm are used directly (not randomly sampled) -- the raw .dat files carry no
    metadata distinguishing "realization 1" as more/less representative than "realization 57", so a
    fixed low-numbered prefix is as valid a sample as a random draw, and is simpler to reproduce
    exactly (seed kept as a parameter for future use if random sampling is ever preferred instead)."""
    return [(gcm, r) for gcm in gcms for r in range(1, n_per_gcm + 1)]
