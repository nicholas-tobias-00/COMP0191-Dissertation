"""S-05 + UQ figures: one representative chain per axis (livestock/grazing/fertilizer) x tower x
SSP, showing raw TabICL q05/q95 vs. U-06 flat-CQR vs. U-07 LSU-stratified-CQR bands on the actual
scenario trajectory out to 2050. One representative level per axis (livestock: all_3x3x3x, the
most extreme/widest-interval case; grazing/fertilizer: their own most extreme level) -- not all
combos, this is a UQ-shape illustration, not a new sweep. Both SSPs plotted (data already covers
both -- see `s05_uq_daily_chains_subset.py`), filename always carries the SSP suffix, matching the
naming convention the rest of S-05's daily-chain figures already use.

**Tower 2 is shown, not omitted** -- it has valid raw model quantiles (TabICL forecasts fine for
T2), just no CQR margin (T2's base conformal calibration was already degenerate from U-04/U-05's
Step 2 onward -- the same pre-established finding this project keeps re-confirming, not new here).
Its panel shows the raw [q05,q95] band and median with an explicit annotation that no calibrated
band exists, rather than a blank "skipped" placeholder or being left out of the figure entirely.

Run from project root:  python notebooks/07_scenario_analysis/s05_uq_cqr_plots.py
"""
import os
import sys

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"c:\Users\Nicholas\Documents\COMP0191 MSc Artificial Intelligence for Sustainable Development Project"
RESULTS = rf"{ROOT}\results"
FIG_DIR = rf"{RESULTS}\figures\s05_uq_cqr"
os.makedirs(FIG_DIR, exist_ok=True)

TOWERS = [2, 4, 9]
SSPS = ["ssp245", "ssp585"]

AXES = {
    "livestock": {"file": "s05_livestock_with_cqr.csv", "filter_col": "combo", "filter_val": "all_3x3x3x",
                  "label": "Livestock (all species 3x)"},
    "grazing": {"file": "s05_grazing_with_cqr.csv", "filter_col": "level", "filter_val": "plus4wk",
                "label": "Grazing (+4wk shift)"},
    "fertilizer": {"file": "s05_fertilizer_with_cqr.csv", "filter_col": "level", "filter_val": "plus50pct_freq",
                   "label": "Fertilizer (+50% frequency)"},
}


def plot_axis(axis, cfg, ssp):
    path = f"{RESULTS}/{cfg['file']}"
    if not os.path.exists(path):
        print(f"  [SKIP] {path} not found")
        return
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df[df["ssp"] == ssp]
    if cfg["filter_col"] in df.columns:
        df = df[df[cfg["filter_col"]] == cfg["filter_val"]]

    n_calibrated = 0
    fig, axes = plt.subplots(len(TOWERS), 1, figsize=(15, 4 * len(TOWERS)), sharex=False)
    for ax, tower in zip(axes, TOWERS):
        sub = df[df.tower == tower].sort_values("timestamp")
        if sub.empty:
            ax.text(0.5, 0.5, f"Tower {tower}: no data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"Tower {tower}")
            continue

        has_cqr = sub["u07_lo"].notna().any()
        ax.plot(sub["timestamp"], sub["pred"], "-", color="tab:green", linewidth=0.6, label="TabICLv2 median")
        ax.fill_between(sub["timestamp"], sub["q05"], sub["q95"], color="gray", alpha=0.15,
                         label="Raw TabICL [q05,q95]")
        if has_cqr:
            n_calibrated += 1
            ax.fill_between(sub["timestamp"], sub["u06_lo"], sub["u06_hi"], color="tab:red", alpha=0.12,
                             label="U-06: flat CQR")
            ax.fill_between(sub["timestamp"], sub["u07_lo"], sub["u07_hi"], color="tab:blue", alpha=0.18,
                             label="U-07: LSU-stratified CQR")
        else:
            ax.text(0.5, 0.92, "No calibrated CQR band -- base conformal calibration degenerate for this tower "
                                "(pre-established, U-04/U-05); raw model quantiles shown above",
                     ha="center", va="top", transform=ax.transAxes, fontsize=8, color="firebrick")
        ax.set_title(f"Tower {tower}")
        ax.set_ylabel("FCH4 (nmol m-2 s-1)")
        ax.legend(fontsize=8, loc="upper left")

    fig.suptitle(f"S-05 scenario trajectory to 2050 with UQ: {cfg['label']}, {ssp}")
    fig.tight_layout()
    fname = f"s05_uq_cqr_{axis}_{ssp}.png"
    fig.savefig(f"{FIG_DIR}/{fname}", dpi=110)
    plt.close(fig)
    print(f"[OK] Saved {fname} ({n_calibrated}/{len(TOWERS)} towers with calibrated CQR bands)")


def main():
    for axis, cfg in AXES.items():
        for ssp in SSPS:
            plot_axis(axis, cfg, ssp)


if __name__ == "__main__":
    main()
