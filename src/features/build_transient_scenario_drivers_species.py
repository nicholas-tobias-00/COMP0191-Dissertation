"""S-05: species-disaggregated transient scenario driver frame, for TabICLv2 + S-03's Variant A
(driver-removal) feature set extended to F-10/D-67's species split. Sibling to
build_transient_scenario_drivers.py (imports from it and from it its own upstream
build_scenario_drivers.py, modifies neither) -- reuses load_transient_years()/
stratified_realizations()/CMIP6_DIR/GCMS unchanged, but builds a much narrower driver frame than
S-04's full-feature climatology base, since Variant A DROPS the 24 scenario-unavailable columns
entirely rather than resampling them.

FX_A_SPECIES (13 columns) = S-03's own Variant-A FX_A (10 cols: TA_mean/min/max, SWIN_mean,
PRECIP_sum, DOY_sin/cos, is_growing, is_winter, lsu_dens) + F-10/D-67's 3 species-density columns
(cattle_dens, sheep_dens, lamb_dens) -- exactly the "BASE+species" config that is this project's
standing forecasting champion (TabPFN+species, D-67/D-80), just restricted to Variant A's already-
narrower BASE.

Livestock multiplier structure (Option B, user-confirmed): INDEPENDENT per-species multipliers
(cattle/sheep/lamb each scaled separately), not a single shared scalar. fx_lsu_dens is NOT
independently resampled/scaled -- it is rebuilt as the exact LSU-weighted sum of the (scaled)
species densities (1.0*cattle + 0.1*sheep + 0.05*lamb), preserving F-10's own construction
identity under any combination of per-species multipliers.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = r"c:\Users\Nicholas\Documents\COMP0191 MSc Artificial Intelligence for Sustainable Development Project"
sys.path.insert(0, ROOT + r"\src")
sys.path.insert(0, ROOT + r"\src\features")
import models.recursive_rollout as rr  # noqa: E402
from build_transient_scenario_drivers import (  # noqa: E402
    CMIP6_DIR, RAD_MJ_TO_WM2, GCMS, load_transient_years, stratified_realizations,
)

FX_A_SPECIES = ["fx_TA_mean", "fx_TA_min", "fx_TA_max", "fx_SWIN_mean", "fx_PRECIP_sum",
                "fx_DOY_sin", "fx_DOY_cos", "fx_is_growing", "fx_is_winter",
                "fx_lsu_dens", "fx_cattle_dens", "fx_sheep_dens", "fx_lamb_dens"]
SPECIES = ["cattle", "sheep", "lamb"]
LSU_WEIGHTS = {"cattle": 1.0, "sheep": 0.1, "lamb": 0.05}


def build_climatology_base_species(tower, T, year):
    """Everything in FX_A_SPECIES that does NOT depend on {SSP, GCM, realization, per-species
    multipliers} -- cache per (tower, year), reused across every (SSP, GCM, realization,
    multiplier-combo) for that (tower, year), same performance rationale as S-04's own
    build_climatology_base(). Species/lsu climatology uses the FULL real historical record (no
    leakage risk for a genuinely blind future -- same convention as S-04, explicitly NOT S-03's
    pre-anchor-only restriction, which exists only because S-03 evaluates on real historical dates
    with a real future to leak from)."""
    target_dates = pd.date_range(f"{year}-01-01", periods=365, freq="D")
    doy = target_dates.dayofyear.values

    out = pd.DataFrame(index=target_dates)
    out["fx_DOY_sin"] = np.sin(2 * np.pi * doy / 365.0)
    out["fx_DOY_cos"] = np.cos(2 * np.pi * doy / 365.0)
    month = target_dates.month
    out["fx_is_growing"] = np.isin(month, [4, 5, 6, 7, 8, 9]).astype(float)
    out["fx_is_winter"] = np.isin(month, [12, 1, 2]).astype(float)

    dft = T[tower]
    for s in SPECIES:
        hist = dft[f"fx_{s}_dens"].dropna()
        out[f"fx_{s}_dens_base"] = rr.doy_climatology(hist, target_dates, window=7)

    return out


def overlay_transient_species(clim_base, transient_year_df, mult_cattle, mult_sheep, mult_lamb):
    """Cheap vectorized per-(SSP, GCM, realization, multiplier-combo) overlay on a cached
    build_climatology_base_species() frame: plugs in real year-specific MIN/MAX/RAIN/RAD, applies
    INDEPENDENT per-species multipliers to the cached *_dens_base columns, and rebuilds fx_lsu_dens
    as the exact LSU-weighted sum of the now-scaled species densities. clim_base itself is never
    mutated, so the same cached base is safely reused across every multiplier combo built from it."""
    out = clim_base.copy()
    out["fx_TA_min"] = transient_year_df["MIN"].values
    out["fx_TA_max"] = transient_year_df["MAX"].values
    out["fx_TA_mean"] = (out["fx_TA_min"] + out["fx_TA_max"]) / 2.0
    out["fx_PRECIP_sum"] = transient_year_df["RAIN"].values
    out["fx_SWIN_mean"] = transient_year_df["RAD"].values * RAD_MJ_TO_WM2

    mult = {"cattle": mult_cattle, "sheep": mult_sheep, "lamb": mult_lamb}
    for s in SPECIES:
        out[f"fx_{s}_dens"] = out[f"fx_{s}_dens_base"] * mult[s]
    out["fx_lsu_dens"] = sum(LSU_WEIGHTS[s] * out[f"fx_{s}_dens"] for s in SPECIES)

    return out.drop(columns=[f"fx_{s}_dens_base" for s in SPECIES])
