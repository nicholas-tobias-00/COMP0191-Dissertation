"""Additive fertiliser scenario level: `reg_cap` -- rate scaled so the TRUE area-weighted typical-
year N loading hits exactly the UK NVZ N-max for grassland (300 kg N/ha/yr, gov.uk). Added
alongside the existing `historical`/`plus50pct_rate`/`plus50pct_freq` levels, for BOTH S-05 (real
drivers) and S-06 (bias-corrected drivers) -- full coverage per project convention, all 3 towers.
Existing levels/files untouched; this only appends new `reg_cap` rows.

See src/features/build_transient_scenario_drivers_fertN_regcap.py for the multiplier derivation
and why it needs to be tower-specific (unlike every other FERT_LEVELS entry).

Run from project root:  python notebooks/07_scenario_analysis/s05_s06_fertN_regcap_fix.py
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

import s05_practices_trajectory as spt
import build_transient_scenario_drivers_practices as p
from build_transient_scenario_drivers_fertN_regcap import synthetic_fertN_events_regcap, REG_CAP_RATE_MULT
from build_transient_scenario_drivers_species import FX_A_SPECIES, load_transient_years as load_transient_years_s05
from build_transient_scenario_drivers_s06 import load_transient_years as load_transient_years_s06
from s05_trajectory_2050 import load_towers, tower_anchor, END_YEAR
import models.recursive_rollout as rr

RESULTS = rf"{ROOT}\results"
TOWERS = [2, 4, 9]
SSPS = ["ssp245", "ssp585"]
LEVELS = ["reg_cap"]
FEAT_COLS = FX_A_SPECIES + spt.FERT_COLS
GCM, REAL = "ACCESS-ESM1-5", 1

# Patch: reg_cap-aware synthetic event generator (delegates to the original for every other level).
p.synthetic_fertN_events = synthetic_fertN_events_regcap

print(f"[OK] REG_CAP_RATE_MULT: {REG_CAP_RATE_MULT}")


def rerun_annual_sweep(run_label, corrected_drivers, merge_path):
    spt.load_transient_years = load_transient_years_s06 if corrected_drivers else load_transient_years_s05
    corrected = spt.run_axis(FEAT_COLS, LEVELS, spt.build_fertilizer_frame, n_per_gcm=10, run_label=run_label)
    existing = pd.read_csv(merge_path)
    kept = existing[~existing.level.isin(LEVELS)]
    merged = pd.concat([kept, corrected], ignore_index=True)
    merged.to_csv(merge_path, index=False)
    print(f"[OK] Merged reg_cap rows into {merge_path} ({len(merged)} rows total)")


def rerun_daily_subset(corrected_drivers, out_path):
    loader = load_transient_years_s06 if corrected_drivers else load_transient_years_s05
    T = load_towers()
    all_rows = []
    t0 = time.time()
    for tower in TOWERS:
        dft = T[tower]
        anchor = tower_anchor(T, tower)
        hist_target = dft.loc[:anchor, "y_observed"]
        hist_cov = dft.loc[:anchor, FEAT_COLS]
        years = list(range(anchor.year + 1, END_YEAR + 1))
        for ssp in SSPS:
            tyears = loader(GCM, ssp, REAL, years)
            for level in LEVELS:
                frame = spt.build_fertilizer_frame(tower, T, dft, anchor, years, level, tyears)[FEAT_COLS]
                chain = rr.tabicl_forecast(hist_target, hist_cov, frame)
                cdf = chain.to_frame("pred").reset_index().rename(columns={"index": "timestamp"})
                cdf["tower"] = tower
                cdf["ssp"] = ssp
                cdf["level"] = level
                all_rows.append(cdf)
                print(f"  T{tower} {ssp} {level}: done ({time.time()-t0:.0f}s elapsed)")
    corrected = pd.concat(all_rows, ignore_index=True)

    existing = pd.read_csv(out_path)
    kept = existing[~existing.level.isin(LEVELS)]
    merged = pd.concat([kept, corrected], ignore_index=True)
    merged.to_csv(out_path, index=False)
    print(f"[OK] Merged reg_cap daily chains into {out_path} ({len(merged)} rows total)")


if __name__ == "__main__":
    print("=== S-05 annual sweep (real drivers) ===")
    rerun_annual_sweep("fertilizer_regcap_fix", corrected_drivers=False,
                        merge_path=f"{RESULTS}/s05_practices_fertilizer.csv")

    print("=== S-06 annual sweep (bias-corrected drivers) ===")
    rerun_annual_sweep("s06_fertilizer_regcap_fix", corrected_drivers=True,
                        merge_path=f"{RESULTS}/s05_practices_s06_fertilizer.csv")

    print("=== S-05 daily chains subset (real drivers) ===")
    rerun_daily_subset(corrected_drivers=False,
                        out_path=f"{RESULTS}/s05_practices_fertilizer_daily_chains_subset.csv")

    print("=== S-06 daily chains subset (bias-corrected drivers) ===")
    rerun_daily_subset(corrected_drivers=True,
                        out_path=f"{RESULTS}/s06_practices_fertilizer_daily_chains_subset.csv")
