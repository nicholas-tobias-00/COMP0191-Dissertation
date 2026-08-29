"""S-05 follow-up: saves FULL DAILY chains (not just the annual_mean the main sweep keeps) for a
small, representative subset -- s05_trajectory_10yr.py discards the 3,650 daily predictions per
call after averaging them into annual_mean, which is fine for the realization/AOA/species-response
questions but leaves no way to sanity-check whether the within-year seasonal pattern looks
physically sensible. Saving daily chains for the FULL 8,100-call grid would be ~29.6M rows (the
same problem S-04 hit and avoided by only saving daily chains for a small subset, not its full
realization grid) -- this mirrors that precedent.

Subset (18 calls, not re-running the main sweep): 3 towers x 2 SSPs (one representative GCM/
realization each: ACCESS-ESM1-5/1, matching the timing-test calls earlier this session) x 3
multiplier combos (baseline 1x/1x/1x, cattle-alone 3x/1x/1x, all-3x 3x/3x/3x -- the 3 combos
already central to s05_results.md's write-up).

Run from project root:  python notebooks/07_scenario_analysis/s05_daily_chains_subset.py
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

import models.recursive_rollout as rr
from build_transient_scenario_drivers_species import (
    FX_A_SPECIES, build_climatology_base_species, overlay_transient_species, load_transient_years,
)
from s05_trajectory_10yr import load_towers, tower_anchor, N_YEARS

RESULTS = rf"{ROOT}\results"

TOWERS = [2, 4, 9]
SSPS = ["ssp245", "ssp585"]
GCM, REAL = "ACCESS-ESM1-5", 1
COMBOS = {
    "baseline_1x1x1x": (1.0, 1.0, 1.0),
    "cattle3x_alone": (3.0, 1.0, 1.0),
    "all_3x3x3x": (3.0, 3.0, 3.0),
}


def main():
    T = load_towers()
    all_rows = []
    t0 = time.time()

    for tower in TOWERS:
        dft = T[tower]
        anchor = tower_anchor(T, tower)
        hist_target = dft.loc[:anchor, "y_observed"]
        hist_cov = dft.loc[:anchor, FX_A_SPECIES]
        years = list(range(anchor.year + 1, anchor.year + 1 + N_YEARS))

        clim_cache = {yr: build_climatology_base_species(tower, T, yr) for yr in years}

        for ssp in SSPS:
            tyears = load_transient_years(GCM, ssp, REAL, years)

            for combo_name, (cattle, sheep, lamb) in COMBOS.items():
                year_frames = [overlay_transient_species(clim_cache[yr], tyears[yr], cattle, sheep, lamb)
                               for yr in years]
                frame = pd.concat(year_frames)[FX_A_SPECIES]

                chain = rr.tabicl_forecast(hist_target, hist_cov, frame)
                cdf = chain.to_frame("pred").reset_index().rename(columns={"index": "date"})
                cdf["tower"] = tower
                cdf["ssp"] = ssp
                cdf["gcm"] = GCM
                cdf["realization"] = REAL
                cdf["combo"] = combo_name
                cdf["mult_cattle"] = cattle
                cdf["mult_sheep"] = sheep
                cdf["mult_lamb"] = lamb
                all_rows.append(cdf)

                print(f"  T{tower} {ssp} {combo_name}: done ({time.time()-t0:.0f}s elapsed)")

    out = pd.concat(all_rows, ignore_index=True)
    out.to_csv(f"{RESULTS}/s05_daily_chains_subset.csv", index=False)
    print(f"\n[OK] Saved s05_daily_chains_subset.csv ({len(out)} rows), total {time.time()-t0:.0f}s")
    return out


if __name__ == "__main__":
    main()
