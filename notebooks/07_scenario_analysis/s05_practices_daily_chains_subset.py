"""S-05 practices follow-up: saves FULL DAILY chains (not just annual_mean, which is all
`s05_practices_trajectory.py` kept) for a small, representative subset of the grazing and
fertilizer axes -- same rationale as `s05_daily_chains_subset.py` did for the original livestock
sweep: cheap to rerun (one representative GCM/realization/SSP, not the full 900-call grid), and
lets the within-year seasonal shape be sanity-checked / plotted per scenario level.

Subset per axis: 3 towers x 1 representative GCM/realization (ACCESS-ESM1-5/1, matching every
other daily-chain subset this session) x BOTH SSPs (245/585 -- realization/GCM choice barely moves
the annual-mean result per S-05's own isolated-realization-spread finding, 2.4-7.6%, so one GCM/
realization is a representative draw, but SSP is a real, separate axis worth showing both sides
of) x all 3 levels of that axis = 18 calls/axis, 2050 horizon (T4/T9: 27yr, T2: 31yr).

Run from project root:  python notebooks/07_scenario_analysis/s05_practices_daily_chains_subset.py
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
from build_transient_scenario_drivers_practices import GRAZING_SHIFT_LEVELS, FERT_LEVELS
from s05_trajectory_2050 import load_towers, tower_anchor, END_YEAR
from s05_practices_trajectory import (
    GRAZING_COLS, FERT_COLS, build_grazing_frame, build_fertilizer_frame,
)

RESULTS = rf"{ROOT}\results"
TOWERS = [2, 4, 9]
SSPS = ["ssp245", "ssp585"]
GCM, REAL = "ACCESS-ESM1-5", 1


def run(axis, levels, feat_cols, build_frame_fn):
    T = load_towers()
    all_rows = []
    t0 = time.time()

    for tower in TOWERS:
        dft = T[tower]
        anchor = tower_anchor(T, tower)
        hist_target = dft.loc[:anchor, "y_observed"]
        hist_cov = dft.loc[:anchor, feat_cols]
        years = list(range(anchor.year + 1, END_YEAR + 1))

        for ssp in SSPS:
            tyears = load_transient_years(GCM, ssp, REAL, years)

            for level in levels:
                frame = build_frame_fn(tower, T, dft, anchor, years, level, tyears)[feat_cols]
                chain = rr.tabicl_forecast(hist_target, hist_cov, frame)
                cdf = chain.to_frame("pred").reset_index().rename(columns={"index": "timestamp"})
                cdf["tower"] = tower
                cdf["ssp"] = ssp
                cdf["level"] = level
                all_rows.append(cdf)
                print(f"  T{tower} {ssp} {level}: done ({time.time()-t0:.0f}s elapsed)")

    out = pd.concat(all_rows, ignore_index=True)
    out.to_csv(f"{RESULTS}/s05_practices_{axis}_daily_chains_subset.csv", index=False)
    print(f"[OK] Saved s05_practices_{axis}_daily_chains_subset.csv ({len(out)} rows)")
    return out


def main():
    run("grazing", list(GRAZING_SHIFT_LEVELS), FX_A_SPECIES + GRAZING_COLS, build_grazing_frame)
    run("fertilizer", list(FERT_LEVELS), FX_A_SPECIES + FERT_COLS, build_fertilizer_frame)


if __name__ == "__main__":
    main()
