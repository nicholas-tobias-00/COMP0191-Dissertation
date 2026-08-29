"""S-06b annual CH4 generation: reuses `s05_annual_ch4_generation.py`'s `add_ch4_generation()`/
`plot_axis()`/`summary_table()` UNCHANGED (same flux->mass conversion, same figure format), pointed
at S-06b's own annual-sweep outputs (`s06b_practices_s06b_livestock_v2.csv`/`s06b_grazing.csv`/
`s06b_fertilizer.csv`, B18-derived architecture) instead of S-05's.

Run from project root:  python notebooks/07_scenario_analysis/s06b_annual_ch4_generation.py
"""
import os
from pathlib import Path
import shutil
import sys

import pandas as pd

ROOT = r"c:\Users\Nicholas\Documents\COMP0191 MSc Artificial Intelligence for Sustainable Development Project"
sys.path.insert(0, ROOT + r"\notebooks\07_scenario_analysis")

import s05_annual_ch4_generation as base

RESULTS = rf"{ROOT}\results"
FIG_DIR = rf"{RESULTS}\figures\s06b_annual_ch4"
os.makedirs(FIG_DIR, exist_ok=True)
base.FIG_DIR = FIG_DIR
REPORT_FIG_DIR = Path(ROOT) / "report" / "Figures"
REPORT_FIG_DIR.mkdir(parents=True, exist_ok=True)
GCM = "ACCESS-ESM1-5"
REALIZATION = 1
SSP_LABELS = {"ssp245": "SSP2-4.5", "ssp585": "SSP5-8.5"}
MODEL_LABEL = "TabICLv2"


def main():
    sources = {
        "livestock_v2": (f"{RESULTS}/s06b_practices_s06b_livestock_v2.csv", "level"),
        "grazing": (f"{RESULTS}/s06b_practices_s06b_grazing.csv", "level"),
        "fertilizer": (f"{RESULTS}/s06b_practices_s06b_fertilizer.csv", "level"),
    }
    for name, (path, level_col) in sources.items():
        if not os.path.exists(path):
            print(f"[SKIP] {name}: {path} not found yet")
            continue
        df = base.add_ch4_generation(pd.read_csv(path))
        out_path = f"{RESULTS}/s06b_annual_ch4_{name}.csv"
        df.to_csv(out_path, index=False)
        print(f"[OK] Saved {out_path} ({len(df)} rows)")

        tab = base.summary_table(df, level_col)
        tab.to_csv(f"{RESULTS}/s06b_annual_ch4_{name}_summary.csv")
        print(tab.to_string())
        print()

        levels = sorted(df[level_col].unique())
        representative = df[(df["gcm"] == GCM) & (df["realization"] == REALIZATION)]
        variation_label = {
            "livestock_v2": "livestock",
            "grazing": "grazing timing",
            "fertilizer": "fertiliser management",
        }[name]
        for ssp, ssp_label in SSP_LABELS.items():
            fname = f"s06b_annual_ch4_{name}_{ssp}.png"
            base.plot_axis(
                representative,
                level_col,
                levels,
                {l: l for l in levels},
                f"Annual CH4 generation - variations inserted to {variation_label} "
                f"({ssp_label}, Model {MODEL_LABEL}, GCM {GCM})",
                fname,
                ssp=ssp,
            )
            report_stem = "annual_ch4_livestock" if name == "livestock_v2" else f"annual_ch4_{name}"
            shutil.copyfile(Path(FIG_DIR) / fname, REPORT_FIG_DIR / f"ch6_{report_stem}_{ssp}.png")
            if ssp == "ssp245":
                shutil.copyfile(Path(FIG_DIR) / fname, REPORT_FIG_DIR / f"ch6_{report_stem}.png")


if __name__ == "__main__":
    main()
