"""S-06b: daily-chains subset for the livestock ladder, B18-derived architecture -- identical
representative subset as `s06_livestock_v2_daily_chains_subset.py` (3 towers x ACCESS-ESM1-5/1 x
both SSPs x 7 combos, 42 calls) but predicts from the locked-in solo per-tower `Direct_TabICLv2` +
trend regression (S-03b/c/d-validated) instead of `tabicl_forecast()`. Model fit ONCE per tower
(reused across both SSPs x 7 levels for that tower), same efficiency principle as
`run_axis_b18()`.

Run from project root:  python notebooks/07_scenario_analysis/s06b_livestock_v2_daily_chains_subset.py
"""
import sys
import time
import warnings

import pandas as pd
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

ROOT = r"c:\Users\Nicholas\Documents\COMP0191 MSc Artificial Intelligence for Sustainable Development Project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\src")
sys.path.insert(0, ROOT + r"\src\features")
sys.path.insert(0, ROOT + r"\notebooks\07_scenario_analysis")

from build_transient_scenario_drivers_species import FX_A_SPECIES
from build_transient_scenario_drivers_s06 import load_transient_years as load_transient_years_s06
from build_transient_scenario_drivers_livestock_v2 import COMBOS
from build_transient_scenario_drivers_livestock_v2_s06 import multiplier_for_s06
from s05_trajectory_2050 import load_towers, tower_anchor, END_YEAR
import s05_livestock_v2_trajectory as s05lv2
import s06b_direct_regression_engine as eng

# S-06-only lit_ceil correction (D-104, gov.uk Annex 8, 2.5 LSU/ha) -- same patch as
# s06b_lit_ceil_fix.py, applied here too since this subset covers all 7 levels including lit_ceil.
s05lv2.multiplier_for = multiplier_for_s06

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
        hist = dft.loc[:anchor]
        hist = hist.loc[hist["y_observed"].notna()]
        hist = eng.add_trend(hist)
        years = list(range(anchor.year + 1, END_YEAR + 1))
        model_features = FX_A_SPECIES + [eng.TREND_COL]

        imputer = SimpleImputer(strategy="mean")
        x_train = imputer.fit_transform(hist[model_features])
        model = eng.make_model()
        model.fit(x_train, hist["y_observed"].to_numpy())
        print(f"=== Tower {tower}: anchor={anchor.date()}, n_train={len(hist)}, fit done ===")

        for ssp in SSPS:
            tyears = load_transient_years_s06(GCM, ssp, REAL, years)

            for level in levels:
                frame_full = s05lv2.build_livestock_frame(tower, T, dft, anchor, years, level, tyears)
                frame_full = eng.add_trend(frame_full)
                x_frame = imputer.transform(frame_full[model_features])
                pred = model.predict(x_frame, output_type="median")
                cdf = pd.DataFrame({"timestamp": frame_full.index, "pred": pred})
                cdf["tower"] = tower
                cdf["ssp"] = ssp
                cdf["level"] = level
                all_rows.append(cdf)
                print(f"  T{tower} {ssp} {level}: done ({time.time()-t0:.0f}s elapsed)")

    out = pd.concat(all_rows, ignore_index=True)
    out.to_csv(f"{RESULTS}/s06b_livestock_v2_daily_chains_subset.csv", index=False)
    print(f"[OK] Saved s06b_livestock_v2_daily_chains_subset.csv ({len(out)} rows)")
    return out


if __name__ == "__main__":
    run()
