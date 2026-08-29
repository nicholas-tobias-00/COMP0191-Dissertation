"""S-06b: Phase 6 of the additive B18-integration plan (2026-08-20) -- reruns S-06's full core
scenario grid (livestock ladder D-97, grazing timing, fertilizer schedule) with the locked-in
B18-derived architecture (`Direct_TabICLv2_solo_trend`, S-03b/c/d-validated) in place of
`tabicl_forecast()` (the TS-wrapper S-05/S-06 always used). Same real-world levels/combos/coverage
as S-06, same bias-corrected CMIP6 driver source -- only the model-calling mechanism differs.

Mechanism: `s06b_direct_regression_engine.run_axis_b18()` is a drop-in replacement for
`s05_practices_trajectory.run_axis()` (same signature/behaviour, same
`load_transient_years`-monkey-patch redirection point via `spt.load_transient_years`, same
`build_livestock_frame`/`build_grazing_frame`/`build_fertilizer_frame` reused unchanged) but calls
the new architecture and writes `s06b_practices_*.csv` (not `s05_practices_*.csv`/
`s06_practices_*.csv`) -- fully additive, S-05's and S-06's own outputs are untouched.

NOT included here (scoped out, matching D-105's own precedent of a separate single-GCM/realization
addendum rather than blocking the main grid): the `reg_cap` fertiliser level. Follow-up, not a gap
in this run's own scope.

Run from project root:
  Smoke test (1 tower, 1 SSP, 1 realization, all axes):  python s06b_master_runner.py smoke
  Full sweep (matches S-06's exact scope):                python s06b_master_runner.py
"""
import sys

ROOT = r"c:\Users\Nicholas\Documents\COMP0191 MSc Artificial Intelligence for Sustainable Development Project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\src")
sys.path.insert(0, ROOT + r"\src\features")
sys.path.insert(0, ROOT + r"\notebooks\07_scenario_analysis")

import s05_practices_trajectory as spt
import s06b_direct_regression_engine as eng
from build_transient_scenario_drivers_s06 import load_transient_years as load_transient_years_s06
from build_transient_scenario_drivers_species import FX_A_SPECIES
from build_transient_scenario_drivers_livestock_v2 import COMBOS
import s05_livestock_v2_trajectory as s05lv2

# --- the one patch point that redirects every axis to bias-corrected drivers ---
spt.load_transient_years = load_transient_years_s06


def run_livestock_v2(n_per_gcm):
    return eng.run_axis_b18(FX_A_SPECIES, list(COMBOS), s05lv2.build_livestock_frame,
                             n_per_gcm=n_per_gcm, run_label="s06b_livestock_v2")


def run_grazing_and_fertilizer(n_per_gcm):
    from build_transient_scenario_drivers_practices import GRAZING_SHIFT_LEVELS, FERT_LEVELS
    grazing = eng.run_axis_b18(FX_A_SPECIES + spt.GRAZING_COLS, list(GRAZING_SHIFT_LEVELS),
                                spt.build_grazing_frame, n_per_gcm=n_per_gcm, run_label="s06b_grazing")
    fert = eng.run_axis_b18(FX_A_SPECIES + spt.FERT_COLS, list(FERT_LEVELS),
                             spt.build_fertilizer_frame, n_per_gcm=n_per_gcm, run_label="s06b_fertilizer")
    return grazing, fert


def main():
    print(f"[S-06b] load_transient_years patched: {spt.load_transient_years.__module__}")
    run_livestock_v2(n_per_gcm=10)
    run_grazing_and_fertilizer(n_per_gcm=10)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        eng.TOWERS = [4]
        eng.SSPS = ["ssp245"]
        print(f"[S-06b SMOKE] load_transient_years patched: {spt.load_transient_years.__module__}")
        run_livestock_v2(n_per_gcm=1)
        run_grazing_and_fertilizer(n_per_gcm=1)
    else:
        main()
