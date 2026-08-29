"""S-06-only fix: rerun just the `lit_ceil` combos (all_species, cattle_alone) with the corrected
2.5 LSU/ha target (UK Countryside Stewardship Annex 8, non-SDA land -- replaces the original 3.0
synthesized estimate), for both the full annual sweep and the daily-chains subset. S-05's original
lit_ceil (target=3.0) is untouched. half/baseline/own_max are unaffected, not rerun.

Merges the corrected rows into the existing S-06 output files (replacing the old 3.0-based
lit_ceil rows), then regenerates only the affected figures.

Run from project root:  python notebooks/07_scenario_analysis/s06_lit_ceil_fix.py
"""
import sys

import pandas as pd

ROOT = r"c:\Users\Nicholas\Documents\COMP0191 MSc Artificial Intelligence for Sustainable Development Project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\src")
sys.path.insert(0, ROOT + r"\src\features")
sys.path.insert(0, ROOT + r"\notebooks\07_scenario_analysis")

import s05_practices_trajectory as spt
import s05_livestock_v2_trajectory as s05lv2
from build_transient_scenario_drivers_s06 import load_transient_years as load_transient_years_s06
from build_transient_scenario_drivers_livestock_v2_s06 import multiplier_for_s06
from build_transient_scenario_drivers_species import FX_A_SPECIES
from s05_trajectory_2050 import load_towers, tower_anchor, END_YEAR
import models.recursive_rollout as rr

RESULTS = rf"{ROOT}\results"
LIT_CEIL_COMBOS = ["lit_ceil__all_species", "lit_ceil__cattle_alone"]

# Patch: S-06 driver, corrected lit_ceil multiplier.
spt.load_transient_years = load_transient_years_s06
s05lv2.multiplier_for = multiplier_for_s06


def rerun_annual_sweep():
    corrected = spt.run_axis(FX_A_SPECIES, LIT_CEIL_COMBOS, s05lv2.build_livestock_frame,
                              n_per_gcm=10, run_label="s06_lit_ceil_fix")
    path = f"{RESULTS}/s05_practices_s06_livestock_v2.csv"
    existing = pd.read_csv(path)
    kept = existing[~existing.level.isin(LIT_CEIL_COMBOS)]
    merged = pd.concat([kept, corrected], ignore_index=True)
    merged.to_csv(path, index=False)
    print(f"[OK] Merged corrected lit_ceil rows into {path} ({len(merged)} rows total)")


def rerun_daily_subset():
    T = load_towers()
    GCM, REAL = "ACCESS-ESM1-5", 1
    SSPS = ["ssp245", "ssp585"]
    all_rows = []
    for tower in [2, 4, 9]:
        dft = T[tower]
        anchor = tower_anchor(T, tower)
        hist_target = dft.loc[:anchor, "y_observed"]
        hist_cov = dft.loc[:anchor, FX_A_SPECIES]
        years = list(range(anchor.year + 1, END_YEAR + 1))
        for ssp in SSPS:
            tyears = load_transient_years_s06(GCM, ssp, REAL, years)
            for level in LIT_CEIL_COMBOS:
                frame = s05lv2.build_livestock_frame(tower, T, dft, anchor, years, level, tyears)[FX_A_SPECIES]
                chain = rr.tabicl_forecast(hist_target, hist_cov, frame)
                cdf = chain.to_frame("pred").reset_index().rename(columns={"index": "timestamp"})
                cdf["tower"] = tower
                cdf["ssp"] = ssp
                cdf["level"] = level
                all_rows.append(cdf)
                print(f"  T{tower} {ssp} {level}: done")
    corrected = pd.concat(all_rows, ignore_index=True)

    path = f"{RESULTS}/s06_livestock_v2_daily_chains_subset.csv"
    existing = pd.read_csv(path)
    kept = existing[~existing.level.isin(LIT_CEIL_COMBOS)]
    merged = pd.concat([kept, corrected], ignore_index=True)
    merged.to_csv(path, index=False)
    print(f"[OK] Merged corrected lit_ceil daily chains into {path} ({len(merged)} rows total)")


if __name__ == "__main__":
    rerun_annual_sweep()
    rerun_daily_subset()
