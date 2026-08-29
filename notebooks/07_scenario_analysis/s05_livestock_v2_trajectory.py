"""S-05 livestock ladder redesign, run (supervisor feedback, 2026-08-13): replaces the original
27-combo 1x/2x/3x independent-per-species sweep (D-84/D-85) with a 7-combo absolute-anchored
ladder (`build_transient_scenario_drivers_livestock_v2.py`) -- half/baseline/lit_ceil/own_max, x
2 families (all_species, cattle_alone), baseline shared across families = 7 distinct combos.

Reuses `s05_practices_trajectory.py`'s `run_axis()` unchanged (generic over feat_cols/levels/
build_frame_fn -- no new loop logic needed) and TabICLv2 as S-05's own standing model. Same
n_per_gcm=10 convention as the practices follow-up (D-86) for full 3-tower x 2-SSP coverage
(CLAUDE.md's full-coverage-by-default rule) -- 3 towers x 2 SSPs x 5 GCMs x 10 realizations x 7
combos = 2,100 calls, ~2.3h at the measured ~4s/call rate.

Run from project root:  python notebooks/07_scenario_analysis/s05_livestock_v2_trajectory.py
Smoke test: python notebooks/07_scenario_analysis/s05_livestock_v2_trajectory.py smoke
"""
import sys

ROOT = r"c:\Users\Nicholas\Documents\COMP0191 MSc Artificial Intelligence for Sustainable Development Project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\src")
sys.path.insert(0, ROOT + r"\src\features")
sys.path.insert(0, ROOT + r"\notebooks\07_scenario_analysis")

from build_transient_scenario_drivers_species import FX_A_SPECIES, overlay_transient_species
from build_transient_scenario_drivers_livestock_v2 import COMBOS, multiplier_for
from s05_practices_trajectory import run_axis
import s05_practices_trajectory as spt


def build_livestock_frame(tower, T, dft, anchor, years, combo, tyears):
    from build_transient_scenario_drivers_species import build_climatology_base_species
    mult_cattle, mult_sheep, mult_lamb = multiplier_for(tower, combo)
    frames = []
    for yr in years:
        clim = build_climatology_base_species(tower, T, yr)
        frames.append(overlay_transient_species(clim, tyears[yr], mult_cattle, mult_sheep, mult_lamb))
    import pandas as pd
    return pd.concat(frames)


def main(n_per_gcm=10):
    return run_axis(FX_A_SPECIES, list(COMBOS), build_livestock_frame,
                     n_per_gcm=n_per_gcm, run_label="livestock_v2")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        spt.TOWERS = [4]
        spt.SSPS = ["ssp245"]
        run_axis(FX_A_SPECIES, list(COMBOS), build_livestock_frame, n_per_gcm=1,
                 run_label="livestock_v2_smoketest")
    else:
        main()
