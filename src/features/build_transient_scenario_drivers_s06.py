"""S-06: bias-corrected drop-in replacement for `build_transient_scenario_drivers.load_transient_years()`
-- identical logic and signature, reading from `data/Simulated Climate Data Bias Corrected/`
(built by `build_bias_corrected_cmip6.py`) instead of the raw `data/Simulated Climate Data/`.
Every other constant (GCMS, RAD_MJ_TO_WM2, stratified_realizations) is re-exported unchanged from
the original module -- only the file-reading function differs, and only in which directory it
reads. Fully additive: does not modify build_transient_scenario_drivers.py.
"""
import os
import sys

import pandas as pd

ROOT = r"c:\Users\Nicholas\Documents\COMP0191 MSc Artificial Intelligence for Sustainable Development Project"
sys.path.insert(0, ROOT + r"\src")
sys.path.insert(0, ROOT + r"\src\features")
from build_transient_scenario_drivers import GCMS, RAD_MJ_TO_WM2, stratified_realizations  # noqa: F401,E402

CMIP6_DIR_S06 = rf"{ROOT}\data\Simulated Climate Data Bias Corrected"


def load_transient_years(gcm, ssp, realization, years):
    """Bias-corrected equivalent of build_transient_scenario_drivers.load_transient_years() --
    same parsing logic, reads the corrected .dat files instead of the raw ones."""
    path = os.path.join(CMIP6_DIR_S06, f"NW.{gcm}.{ssp}.{realization}.dat")
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
