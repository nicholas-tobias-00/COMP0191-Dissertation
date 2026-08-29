"""S-06: reruns every S-05 process (livestock ladder D-97, grazing timing, fertilizer schedule,
daily-chains subsets) against the bias-corrected CMIP6 drivers (build_bias_corrected_cmip6.py) --
same real-world levels/combos/coverage as S-05, only the underlying weather driver source differs.

Mechanism: `s05_practices_trajectory.py`'s `run_axis()` and the daily-chains-subset scripts all
resolve `load_transient_years` as a plain module-global name at CALL time (from
`from ... import load_transient_years`) -- so monkey-patching that name on the already-imported
module object, before calling its functions, correctly redirects every downstream call to the
bias-corrected loader, with zero duplication of run_axis()/build_livestock_frame()/etc. Same
technique already used and validated earlier this session (D-98's TOWERS/SSPS override).

Fully additive: S-05's own scripts/outputs are completely untouched; every output here is saved
under a new `s06_` prefix.

Run from project root:
  Smoke test (1 tower, 1 SSP, 1 realization, all axes):  python s06_master_runner.py smoke
  Full sweep (matches S-05's exact scope):                python s06_master_runner.py
"""
import sys

ROOT = r"c:\Users\Nicholas\Documents\COMP0191 MSc Artificial Intelligence for Sustainable Development Project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\src")
sys.path.insert(0, ROOT + r"\src\features")
sys.path.insert(0, ROOT + r"\notebooks\07_scenario_analysis")

import s05_practices_trajectory as spt
from build_transient_scenario_drivers_s06 import load_transient_years as load_transient_years_s06
from build_transient_scenario_drivers_species import FX_A_SPECIES
from build_transient_scenario_drivers_livestock_v2 import COMBOS
import s05_livestock_v2_trajectory as s05lv2

# --- the one patch point that redirects every axis to bias-corrected drivers ---
spt.load_transient_years = load_transient_years_s06


def run_livestock_v2(n_per_gcm):
    return spt.run_axis(FX_A_SPECIES, list(COMBOS), s05lv2.build_livestock_frame,
                         n_per_gcm=n_per_gcm, run_label="s06_livestock_v2")


def run_grazing_and_fertilizer(n_per_gcm):
    from build_transient_scenario_drivers_practices import GRAZING_SHIFT_LEVELS, FERT_LEVELS
    grazing = spt.run_axis(FX_A_SPECIES + spt.GRAZING_COLS, list(GRAZING_SHIFT_LEVELS),
                            spt.build_grazing_frame, n_per_gcm=n_per_gcm, run_label="s06_grazing")
    fert = spt.run_axis(FX_A_SPECIES + spt.FERT_COLS, list(FERT_LEVELS),
                         spt.build_fertilizer_frame, n_per_gcm=n_per_gcm, run_label="s06_fertilizer")
    return grazing, fert


def main():
    print(f"[S-06] load_transient_years patched: {spt.load_transient_years.__module__}")
    run_livestock_v2(n_per_gcm=10)
    run_grazing_and_fertilizer(n_per_gcm=10)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        spt.TOWERS = [4]
        spt.SSPS = ["ssp245"]
        print(f"[S-06 SMOKE] load_transient_years patched: {spt.load_transient_years.__module__}")
        run_livestock_v2(n_per_gcm=1)
        run_grazing_and_fertilizer(n_per_gcm=1)
    else:
        main()
