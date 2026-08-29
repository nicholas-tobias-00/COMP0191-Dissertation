"""S-06: bias-corrects the raw CMIP6/LARS-WG driver files against NWFP's own real local
climatology, per-GCM, before any scenario construction happens. Motivated by a direct comparison
(same overlapping 2020-2024 calendar years, ruling out a baseline-period-mismatch explanation)
that found simulated precipitation running ~4x wetter than what NWFP's gauge actually measures,
and simulated temperature running 2-3.5C cooler than real -- both large enough, and both confirmed
as properties of the driver data rather than a real-data-gap artifact, to warrant correcting before
reuse rather than just flagging as a limitation.

Convention (standard climate-impact-modelling practice, not improvised):
  - Temperature (TA_min, TA_max): ADDITIVE correction (interval-scale variable -- multiplying
    temperature doesn't mean anything physically). corrected = simulated + (real_mean - sim_mean).
  - Precipitation, radiation: MULTIPLICATIVE/ratio correction (non-negative, ratio-scale, skewed
    for precip -- an additive shift risks negative rainfall on dry days). corrected = simulated *
    (real_mean / sim_mean).

Correction factors are computed PER-GCM (pooling realizations 1-10 and a 2020-2030 near-term
window, deliberately early rather than the full 2020-2050 horizon, so the correction targets each
GCM's own near-term bias rather than being contaminated by the genuine future warming signal it's
supposed to preserve), against NWFP's own full real historical record (Tower 4, best data coverage
of the 3 towers; all 3 towers share one site-level CMIP6 file, so one reference tower is used
rather than introducing a spurious per-tower climate distinction real inter-tower distance at this
single farm doesn't plausibly support).

Writes bias-corrected `.dat` files, IDENTICAL FORMAT to the originals, to a new directory --
`data/Simulated Climate Data Bias Corrected/` -- so every downstream S-05 script can be reused via
a one-line `CMIP6_DIR` swap rather than rewritten. Fully additive: originals untouched.

Run from project root:  python src/features/build_bias_corrected_cmip6.py
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = r"c:\Users\Nicholas\Documents\COMP0191 MSc Artificial Intelligence for Sustainable Development Project"
sys.path.insert(0, ROOT + r"\src")
sys.path.insert(0, ROOT + r"\src\features")
from build_transient_scenario_drivers import load_transient_years, GCMS, stratified_realizations, RAD_MJ_TO_WM2

CMIP6_DIR_RAW = rf"{ROOT}\data\Simulated Climate Data"
CMIP6_DIR_CORRECTED = rf"{ROOT}\data\Simulated Climate Data Bias Corrected"
HOURLY = rf"{ROOT}\data\Hourly"

SSPS = ["ssp245", "ssp585"]
REF_YEARS = list(range(2020, 2031))  # near-term window, avoids conflating correction with warming signal


def real_reference():
    """NWFP's own real local climatology (Tower 4, full historical record) -- the target every
    GCM's near-term simulated climate is corrected to match."""
    d = pd.read_csv(rf"{HOURLY}\consolidated_hourly.csv",
                     usecols=["Datetime", "TA_0_0_1 [Tower 4]", "SWIN_1_1_1 [Tower 4]",
                              "Precipitation (mm) [Catchment 4 After  2013/08/13]"], low_memory=False)
    d["Datetime"] = pd.to_datetime(d["Datetime"], format="mixed")
    d = d.set_index("Datetime")
    precip_col = "Precipitation (mm) [Catchment 4 After  2013/08/13]"

    real_ta_min = d["TA_0_0_1 [Tower 4]"].resample("D").min().mean()
    real_ta_max = d["TA_0_0_1 [Tower 4]"].resample("D").max().mean()
    real_swin = d["SWIN_1_1_1 [Tower 4]"].resample("D").mean().mean()
    valid_hrs = d[precip_col].resample("D").apply(lambda x: x.notna().sum())
    precip_mean24 = d[precip_col].resample("D").mean() * 24
    real_precip = precip_mean24[valid_hrs >= 12].mean()

    return {"TA_min": real_ta_min, "TA_max": real_ta_max, "SWIN": real_swin, "PRECIP": real_precip}


def compute_gcm_factors(real_ref):
    """Per-GCM correction factors, pooling realizations 1-10 and REF_YEARS, ssp245 only (near-term
    bias shouldn't meaningfully differ by SSP this early in the trajectory -- both SSPs share the
    same correction factor, applied identically below)."""
    rows = []
    for gcm in GCMS:
        frames = []
        for real in range(1, 11):
            try:
                ty = load_transient_years(gcm, "ssp245", real, REF_YEARS)
                frames.append(pd.concat(ty.values()))
            except FileNotFoundError:
                continue
        sim = pd.concat(frames)
        sim_ta_min, sim_ta_max = sim["MIN"].mean(), sim["MAX"].mean()
        sim_swin, sim_rain = (sim["RAD"] * RAD_MJ_TO_WM2).mean(), sim["RAIN"].mean()
        rows.append({
            "gcm": gcm,
            "offset_TA_min": real_ref["TA_min"] - sim_ta_min,
            "offset_TA_max": real_ref["TA_max"] - sim_ta_max,
            "ratio_SWIN": real_ref["SWIN"] / sim_swin,
            "ratio_RAIN": real_ref["PRECIP"] / sim_rain,
        })
    return pd.DataFrame(rows).set_index("gcm")


def correct_and_write_file(gcm, ssp, realization, factors):
    src_path = os.path.join(CMIP6_DIR_RAW, f"NW.{gcm}.{ssp}.{realization}.dat")
    df = pd.read_csv(src_path, sep=r"\s+", names=["YEAR", "JDAY", "MIN", "MAX", "RAIN", "RAD"],
                      header=None, engine="python")
    f = factors.loc[gcm]
    df["MIN"] = df["MIN"] + f["offset_TA_min"]
    df["MAX"] = df["MAX"] + f["offset_TA_max"]
    df["RAD"] = (df["RAD"] * f["ratio_SWIN"]).clip(lower=0)     # radiation can't be negative
    df["RAIN"] = (df["RAIN"] * f["ratio_RAIN"]).clip(lower=0)   # rainfall can't be negative

    out_path = os.path.join(CMIP6_DIR_CORRECTED, f"NW.{gcm}.{ssp}.{realization}.dat")
    df.to_csv(out_path, sep="\t", header=False, index=False)


def main():
    os.makedirs(CMIP6_DIR_CORRECTED, exist_ok=True)
    real_ref = real_reference()
    print(f"[OK] Real reference (Tower 4): {real_ref}")

    factors = compute_gcm_factors(real_ref)
    factors.to_csv(rf"{ROOT}\results\s06_bias_correction_factors.csv")
    print("[OK] Per-GCM correction factors:")
    print(factors.round(3).to_string())

    n_written = 0
    for gcm in GCMS:
        for ssp in SSPS:
            for real in range(1, 11):
                src = os.path.join(CMIP6_DIR_RAW, f"NW.{gcm}.{ssp}.{real}.dat")
                if not os.path.exists(src):
                    print(f"  [skip] missing source: {src}")
                    continue
                correct_and_write_file(gcm, ssp, real, factors)
                n_written += 1
    print(f"[OK] Wrote {n_written} bias-corrected .dat files to {CMIP6_DIR_CORRECTED}")


if __name__ == "__main__":
    main()
