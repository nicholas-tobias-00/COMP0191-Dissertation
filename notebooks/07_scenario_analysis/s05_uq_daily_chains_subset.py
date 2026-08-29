"""S-05 + UQ: re-runs the SAME small representative daily-chain subset already used for the
livestock/grazing/fertilizer sanity-check figures (`s05_daily_chains_subset.py`,
`s05_practices_daily_chains_subset.py` -- 3 towers x 2 SSPs x 3 levels/combos = 18 calls/axis, one
representative GCM/realization, 2050 horizon), but requesting `quantiles=(0.05, 0.5, 0.95)` from
`tabicl_forecast()` instead of point predictions only. Per `tabicl_forecast()`'s own docstring,
TabICL always computes an internal quantile grid regardless of the `quantiles=` argument, so this
costs no more than the point-only runs already done -- confirmed by timing below, not assumed.

This is the one missing ingredient for connecting S-05 (scenario analysis) to U-06/U-07 (CQR /
LSU-stratified CQR) -- those calibrations exist and were validated on U-04/U-05's historical
chains, but S-05's own sweep never saved q05/q95, so there was nothing to attach a calibrated
interval to. `s05_uq_cqr_apply.py` (companion script) attaches U-06/U-07's already-fitted margins
to this output; no new calibration fitting happens here, purely new quantile inference calls.

Run from project root:  python notebooks/07_scenario_analysis/s05_uq_daily_chains_subset.py
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
from build_transient_scenario_drivers_species import (
    FX_A_SPECIES, build_climatology_base_species, overlay_transient_species, load_transient_years,
)
from build_transient_scenario_drivers_practices import GRAZING_SHIFT_LEVELS, FERT_LEVELS
from s05_trajectory_2050 import load_towers, tower_anchor, END_YEAR
from s05_practices_trajectory import GRAZING_COLS, FERT_COLS, build_grazing_frame, build_fertilizer_frame

RESULTS = rf"{ROOT}\results"
TOWERS = [2, 4, 9]
SSPS = ["ssp245", "ssp585"]
GCM, REAL = "ACCESS-ESM1-5", 1
QUANTILES = (0.05, 0.5, 0.95)

LIVESTOCK_COMBOS = {
    "baseline_1x1x1x": (1.0, 1.0, 1.0),
    "cattle3x_alone": (3.0, 1.0, 1.0),
    "all_3x3x3x": (3.0, 3.0, 3.0),
}


def qframe(chain_df, extra_cols):
    """chain_df: tabicl_forecast()'s quantile-mode output (index=date, cols=median,0.05,0.5,0.95 --
    0.5 is a redundant duplicate of 'median', dropped here)."""
    out = chain_df.drop(columns=[0.5]).reset_index().rename(
        columns={"index": "timestamp", "median": "pred", 0.05: "q05", 0.95: "q95"})
    for k, v in extra_cols.items():
        out[k] = v
    return out


def run_livestock():
    T = load_towers()
    all_rows = []
    t0 = time.time()
    for tower in TOWERS:
        dft = T[tower]
        anchor = tower_anchor(T, tower)
        hist_target = dft.loc[:anchor, "y_observed"]
        hist_cov = dft.loc[:anchor, FX_A_SPECIES]
        years = list(range(anchor.year + 1, END_YEAR + 1))
        clim_cache = {yr: build_climatology_base_species(tower, T, yr) for yr in years}

        for ssp in SSPS:
            tyears = load_transient_years(GCM, ssp, REAL, years)
            for combo_name, (cattle, sheep, lamb) in LIVESTOCK_COMBOS.items():
                year_frames = [overlay_transient_species(clim_cache[yr], tyears[yr], cattle, sheep, lamb)
                               for yr in years]
                frame = pd.concat(year_frames)[FX_A_SPECIES]
                chain = rr.tabicl_forecast(hist_target, hist_cov, frame, quantiles=list(QUANTILES))
                chain = chain.join(frame[["fx_lsu_dens"]])
                cdf = qframe(chain, {"tower": tower, "ssp": ssp, "gcm": GCM, "realization": REAL,
                                      "combo": combo_name, "mult_cattle": cattle, "mult_sheep": sheep,
                                      "mult_lamb": lamb})
                all_rows.append(cdf)
                print(f"  [livestock] T{tower} {ssp} {combo_name}: done ({time.time()-t0:.0f}s elapsed)")

    out = pd.concat(all_rows, ignore_index=True)
    out.to_csv(f"{RESULTS}/s05_livestock_daily_chains_subset_uq.csv", index=False)
    print(f"[OK] Saved s05_livestock_daily_chains_subset_uq.csv ({len(out)} rows), {time.time()-t0:.0f}s")
    return out


def run_practices(axis, levels, feat_cols, build_frame_fn):
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
                chain = rr.tabicl_forecast(hist_target, hist_cov, frame, quantiles=list(QUANTILES))
                chain = chain.join(frame[["fx_lsu_dens"]])
                cdf = qframe(chain, {"tower": tower, "ssp": ssp, "level": level})
                all_rows.append(cdf)
                print(f"  [{axis}] T{tower} {ssp} {level}: done ({time.time()-t0:.0f}s elapsed)")

    out = pd.concat(all_rows, ignore_index=True)
    out.to_csv(f"{RESULTS}/s05_practices_{axis}_daily_chains_subset_uq.csv", index=False)
    print(f"[OK] Saved s05_practices_{axis}_daily_chains_subset_uq.csv ({len(out)} rows), {time.time()-t0:.0f}s")
    return out


def main():
    run_livestock()
    run_practices("grazing", list(GRAZING_SHIFT_LEVELS), FX_A_SPECIES + GRAZING_COLS, build_grazing_frame)
    run_practices("fertilizer", list(FERT_LEVELS), FX_A_SPECIES + FERT_COLS, build_fertilizer_frame)


if __name__ == "__main__":
    main()
