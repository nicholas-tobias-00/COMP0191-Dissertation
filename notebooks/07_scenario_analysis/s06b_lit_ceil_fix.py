"""S-06b lit_ceil fix: the main `s06b_master_runner.py` grid reuses `s05_livestock_v2_trajectory
.build_livestock_frame` unchanged, which imports `multiplier_for` directly from
`build_transient_scenario_drivers_livestock_v2` (D-97's original, uncorrected 3.0 LSU/ha target) --
it never had the D-104 correction (2.5 LSU/ha, UK Countryside Stewardship Annex 8) applied, unlike
`s06_lit_ceil_fix.py`'s own targeted patch for the OLD (TS-wrapper) architecture's output. Caught by
direct user question ("do the scenarios follow gov.uk regulations?") prompting a direct code check,
not assumed correct.

Reruns just the 2 `lit_ceil` combos (all_species, cattle_alone) with the corrected multiplier via
the SAME monkey-patch technique as the original fix, using the new `run_axis_b18` engine, and
merges into `s06b_practices_s06b_livestock_v2.csv` in place -- `half`/`baseline`/`own_max` rows
(5 of 7 levels, unaffected by this correction) are untouched.

Run from project root, AFTER s06b_master_runner.py's livestock_v2 axis has completed:
  python notebooks/07_scenario_analysis/s06b_lit_ceil_fix.py
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
import s06b_direct_regression_engine as eng
from build_transient_scenario_drivers_s06 import load_transient_years as load_transient_years_s06
from build_transient_scenario_drivers_livestock_v2_s06 import multiplier_for_s06
from build_transient_scenario_drivers_species import FX_A_SPECIES

RESULTS = rf"{ROOT}\results"
LIT_CEIL_COMBOS = ["lit_ceil__all_species", "lit_ceil__cattle_alone"]

# Same patch points as the original s06_lit_ceil_fix.py, now applied to the b18 engine.
spt.load_transient_years = load_transient_years_s06
s05lv2.multiplier_for = multiplier_for_s06


def main():
    print(f"[S-06b lit_ceil fix] load_transient_years: {spt.load_transient_years.__module__}, "
          f"multiplier_for: {s05lv2.multiplier_for.__module__}.{s05lv2.multiplier_for.__name__}")
    corrected = eng.run_axis_b18(FX_A_SPECIES, LIT_CEIL_COMBOS, s05lv2.build_livestock_frame,
                                  n_per_gcm=10, run_label="s06b_lit_ceil_fix")
    path = f"{RESULTS}/s06b_practices_s06b_livestock_v2.csv"
    existing = pd.read_csv(path)
    before_n = len(existing)
    kept = existing[~existing.level.isin(LIT_CEIL_COMBOS)]
    merged = pd.concat([kept, corrected], ignore_index=True)
    merged.to_csv(path, index=False)
    print(f"[OK] Merged corrected lit_ceil rows into {path}: {before_n} -> {len(merged)} rows "
          f"(should match if merge is clean)")


if __name__ == "__main__":
    main()
