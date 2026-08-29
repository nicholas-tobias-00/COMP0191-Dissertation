"""S-06 continuity check: does TabICLv2's real-historical-anchor forecast (validated against
actual observations) join up smoothly with its own long-term baseline scenario projection, or is
there a visible jump at the transition -- the direct question the D-100/S-06 baseline-reconstruction
bias work raised but never actually visualized.

Same architecture on both sides (TabICLv2 + FX_A_SPECIES) for a fair continuity check -- using a
different model for either side would confound "does this model's behaviour jump" with "do two
different models disagree."

Two segments, per tower:
  1. HISTORICAL VALIDATED: anchor=2022-12-16 (fixed across towers, matching this project's own
     5-anchor convention), REAL historical drivers (not CMIP6/scenario), 365-day forecast checked
     directly against real y_observed in that window. Tower 2 has no real data in this window (its
     own real record ends 2019-05-31) -- shown anyway, explicitly labeled as unvalidatable, rather
     than silently dropped (this project's standing Tower-2 convention).
  2. FUTURE PROJECTION: starting from each tower's own TRUE last-real-data anchor (T2=2019-05-31,
     T4/T9=2023-12-29), baseline level (no livestock perturbation), BOTH SSPs, bias-corrected (S-06)
     drivers, one representative GCM/realization (ACCESS-ESM1-5/1, this project's standard choice
     for daily-chain figures) through 2050.

Run from project root:  python notebooks/07_scenario_analysis/s06_continuity_check.py
"""
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"c:\Users\Nicholas\Documents\COMP0191 MSc Artificial Intelligence for Sustainable Development Project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\src")
sys.path.insert(0, ROOT + r"\src\features")
sys.path.insert(0, ROOT + r"\notebooks\07_scenario_analysis")

import models.recursive_rollout as rr
from build_transient_scenario_drivers_species import (
    FX_A_SPECIES, build_climatology_base_species, overlay_transient_species,
)
from build_transient_scenario_drivers_s06 import load_transient_years as load_transient_years_s06
from s05_trajectory_2050 import load_towers, tower_anchor, END_YEAR

RESULTS = rf"{ROOT}\results"
FIG_DIR = rf"{RESULTS}\figures\s06_continuity"
import os
os.makedirs(FIG_DIR, exist_ok=True)

TOWERS = [2, 4, 9]
SSPS = ["ssp245", "ssp585"]
GCM, REAL = "ACCESS-ESM1-5", 1
HIST_ANCHOR = pd.Timestamp("2022-12-16")


def historical_validated_chain(tower, T):
    """Real-driver forecast from HIST_ANCHOR, checked against real y_observed where available."""
    dft = T[tower]
    hist_target = dft.loc[:HIST_ANCHOR, "y_observed"]
    hist_cov = dft.loc[:HIST_ANCHOR, FX_A_SPECIES]
    target_dates = pd.date_range(HIST_ANCHOR + pd.Timedelta(days=1), periods=365, freq="D")
    future_cov = dft.loc[target_dates, FX_A_SPECIES]  # REAL future drivers (perfect foresight), not scenario
    chain = rr.tabicl_forecast(hist_target, hist_cov, future_cov)
    real_y = dft["y_observed"].reindex(target_dates)
    n_valid = real_y.notna().sum()
    return chain, real_y, n_valid


def future_projection_chain(tower, T, ssp):
    anchor = tower_anchor(T, tower)
    dft = T[tower]
    hist_target = dft.loc[:anchor, "y_observed"]
    hist_cov = dft.loc[:anchor, FX_A_SPECIES]
    years = list(range(anchor.year + 1, END_YEAR + 1))
    tyears = load_transient_years_s06(GCM, ssp, REAL, years)

    frames = []
    for yr in years:
        clim = build_climatology_base_species(tower, T, yr)
        frames.append(overlay_transient_species(clim, tyears[yr], 1.0, 1.0, 1.0))  # baseline
    frame = pd.concat(frames)[FX_A_SPECIES]
    chain = rr.tabicl_forecast(hist_target, hist_cov, frame)
    return chain, anchor


def main():
    T = load_towers()
    results = {}
    for tower in TOWERS:
        print(f"Tower {tower}: building historical-validated segment (anchor {HIST_ANCHOR.date()})...")
        hist_chain, real_y, n_valid = historical_validated_chain(tower, T)
        print(f"  {n_valid}/365 real y_observed days available to validate against")

        future_chains = {}
        for ssp in SSPS:
            print(f"Tower {tower}: building future projection ({ssp})...")
            chain, anchor = future_projection_chain(tower, T, ssp)
            future_chains[ssp] = chain
        results[tower] = {"hist_chain": hist_chain, "real_y": real_y, "n_valid": n_valid,
                           "future_chains": future_chains, "future_anchor": anchor}

    # Save raw data
    rows = []
    for tower, r in results.items():
        for d, v in r["hist_chain"].items():
            rows.append({"tower": tower, "date": d, "series": "historical_validated_forecast", "value": v})
        for d, v in r["real_y"].dropna().items():
            rows.append({"tower": tower, "date": d, "series": "real_observed", "value": v})
        for ssp, chain in r["future_chains"].items():
            for d, v in chain.items():
                rows.append({"tower": tower, "date": d, "series": f"future_{ssp}", "value": v})
    out = pd.DataFrame(rows)
    out.to_csv(f"{RESULTS}/s06_continuity_check.csv", index=False)
    print(f"[OK] Saved s06_continuity_check.csv ({len(out)} rows)")

    # Plot: 3 panels, one per tower
    fig, axes = plt.subplots(3, 1, figsize=(16, 13), sharex=False)
    for ax, tower in zip(axes, TOWERS):
        r = results[tower]
        ax.plot(r["hist_chain"].index, r["hist_chain"].values, color="tab:green", linewidth=1.0,
                 label=f"Historical validated forecast (anchor {HIST_ANCHOR.date()})")
        if r["n_valid"] > 0:
            ax.plot(r["real_y"].dropna().index, r["real_y"].dropna().values, "o", color="black",
                     markersize=2, alpha=0.5, label=f"Real observed ({r['n_valid']}/365 days)")
        else:
            ax.text(0.02, 0.95, "NO real y_observed available in this window (T2 data gap)",
                    transform=ax.transAxes, fontsize=9, color="red", va="top")
        for ssp, color in zip(SSPS, ["tab:blue", "tab:red"]):
            chain = r["future_chains"][ssp]
            ax.plot(chain.index, chain.values, color=color, linewidth=0.4, alpha=0.8,
                     label=f"Future projection, baseline, {ssp}")
        ax.axvline(r["future_anchor"], color="grey", linestyle="--", linewidth=1,
                   label=f"Future-segment anchor ({r['future_anchor'].date()})")
        ax.set_title(f"Tower {tower}")
        ax.set_ylabel("FCH4 (nmol m-2 s-1)")
        ax.legend(fontsize=7, loc="upper left", ncol=2)
    fig.suptitle("S-06 continuity check: historical validated forecast (2022 anchor) vs. "
                 "long-term baseline projection to 2050 (bias-corrected drivers, both SSPs)")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/s06_continuity_check.png", dpi=120)
    plt.close(fig)
    print(f"[OK] Saved {FIG_DIR}/s06_continuity_check.png")


if __name__ == "__main__":
    main()
