"""S-05 livestock ladder redesign: saves FULL DAILY chains (not just annual_mean) for a small,
representative subset -- same rationale/convention as `s05_practices_daily_chains_subset.py`
(one representative GCM/realization, both SSPs, all towers, all levels of the axis).

Subset: 3 towers x 1 representative GCM/realization (ACCESS-ESM1-5/1) x BOTH SSPs x 7 combos
(half/lit_ceil/own_max x all_species/cattle_alone, + baseline) = 42 calls, 2050 horizon.

Run from project root:  python notebooks/07_scenario_analysis/s05_livestock_v2_daily_chains_subset.py
"""
import sys
import time
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

ROOT = r"c:\Users\Nicholas\Documents\COMP0191 MSc Artificial Intelligence for Sustainable Development Project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\src")
sys.path.insert(0, ROOT + r"\src\features")
sys.path.insert(0, ROOT + r"\notebooks\07_scenario_analysis")

import models.recursive_rollout as rr
from build_transient_scenario_drivers_species import FX_A_SPECIES, load_transient_years
from build_transient_scenario_drivers_livestock_v2 import COMBOS
from s05_trajectory_2050 import load_towers, tower_anchor, END_YEAR
from s05_livestock_v2_trajectory import build_livestock_frame

RESULTS = rf"{ROOT}\results"
TOWERS = [2, 4, 9]
SSPS = ["ssp245", "ssp585"]
GCM, REAL = "ACCESS-ESM1-5", 1


def run():
    T = load_towers()
    levels = list(COMBOS)
    all_rows = []
    t0 = time.time()

    for tower in TOWERS:
        dft = T[tower]
        anchor = tower_anchor(T, tower)
        hist_target = dft.loc[:anchor, "y_observed"]
        hist_cov = dft.loc[:anchor, FX_A_SPECIES]
        years = list(range(anchor.year + 1, END_YEAR + 1))

        for ssp in SSPS:
            tyears = load_transient_years(GCM, ssp, REAL, years)

            for level in levels:
                frame = build_livestock_frame(tower, T, dft, anchor, years, level, tyears)[FX_A_SPECIES]
                chain = rr.tabicl_forecast(hist_target, hist_cov, frame)
                cdf = chain.to_frame("pred").reset_index().rename(columns={"index": "timestamp"})
                cdf["tower"] = tower
                cdf["ssp"] = ssp
                cdf["level"] = level
                all_rows.append(cdf)
                print(f"  T{tower} {ssp} {level}: done ({time.time()-t0:.0f}s elapsed)")

    out = pd.concat(all_rows, ignore_index=True)
    out.to_csv(f"{RESULTS}/s05_livestock_v2_daily_chains_subset.csv", index=False)
    print(f"[OK] Saved s05_livestock_v2_daily_chains_subset.csv ({len(out)} rows)")
    return out


if __name__ == "__main__":
    run()
