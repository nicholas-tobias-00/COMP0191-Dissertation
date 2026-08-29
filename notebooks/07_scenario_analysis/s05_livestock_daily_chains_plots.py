"""S-05 livestock axis: daily-chain figures from the full `s05_daily_chains_2050.parquet` output
(the livestock-multiplier sweep's complete 83.8M-row daily grid, all 8,100 calls). Naming/style
matches the grazing/fertilizer practice-axis figures (`s05_practices_daily_chains_plots.py`) --
`s05_livestock_daily_{full_horizon,zoom<year>,monthly_smoothed}.png` -- so all three scenario
families (livestock, grazing, fertilizer) use one consistent naming convention. Uses pyarrow
predicate pushdown (`pq.read_table(..., filters=...)`) to pull one representative (GCM,
realization, SSP) slice across all 3 towers/27 combos without loading the full file -- ~0.6s per
slice, confirmed directly. (Predates the grazing/fertilizer axes and their own subset-generation
script, since livestock's full grid already has every daily prediction saved -- no separate
`_daily_chains_subset.csv` needed here the way `s05_practices_daily_chains_subset.py` needed one.)

Three figures per representative slice:
  1. Full horizon (27-31 years of daily predictions) -- direct analogue of the 10-year subset plot
     (`s05_daily_chains_subset.py`, superseded by the 2050 extension but kept on disk).
  2. Single-year zoom (2035, mid-horizon) -- the full-horizon plot is visually dense at this scale;
     this isolates one year's seasonal shape clearly.
  3. Monthly-mean smoothed -- separates the seasonal cycle from any longer-term trend across the
     decades (does the model show drift under climate change, independent of the seasonal cycle?).

Run from project root:  python notebooks/07_scenario_analysis/s05_livestock_daily_chains_plots.py
"""
import sys

import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"c:\Users\Nicholas\Documents\COMP0191 MSc Artificial Intelligence for Sustainable Development Project"
RESULTS = rf"{ROOT}\results"
FIG_DIR = rf"{RESULTS}\figures\s05_summary"

TOWERS = [2, 4, 9]
COMBOS = {(1.0, 1.0, 1.0): "Baseline (1x/1x/1x)", (3.0, 1.0, 1.0): "Cattle 3x alone",
          (3.0, 3.0, 3.0): "All species 3x"}
COLORS = {(1.0, 1.0, 1.0): "tab:blue", (3.0, 1.0, 1.0): "tab:orange", (3.0, 3.0, 3.0): "tab:red"}
ZOOM_YEAR = 2035


def load_slice(gcm, realization, ssp):
    tbl = pq.read_table(f"{RESULTS}/s05_daily_chains_2050.parquet",
                         filters=[("gcm", "=", gcm), ("realization", "=", realization), ("ssp", "=", ssp)])
    df = tbl.to_pandas()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def plot_full_horizon(df, gcm, realization, ssp):
    fig, axes = plt.subplots(3, 1, figsize=(16, 11), sharex=False)
    for ax, tower in zip(axes, TOWERS):
        sub = df[df.tower == tower]
        for combo, label in COMBOS.items():
            c, s, l = combo
            ss = sub[(sub.mult_cattle == c) & (sub.mult_sheep == s) & (sub.mult_lamb == l)].sort_values("timestamp")
            ax.plot(ss["timestamp"], ss["pred"], color=COLORS[combo], label=label, linewidth=0.4, alpha=0.8)
        ax.set_title(f"Tower {tower} -- daily FCH4 predictions to 2050 ({ssp}, {gcm}/realization {realization})")
        ax.set_ylabel("Predicted FCH4 (nmol m-2 s-1)")
        ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/s05_livestock_daily_full_horizon_{ssp}.png", dpi=120)
    plt.close(fig)
    print(f"[OK] figure: s05_livestock_daily_full_horizon_{ssp}.png")


def plot_zoom_year(df, gcm, realization, ssp, year=ZOOM_YEAR):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=False)
    for ax, tower in zip(axes, TOWERS):
        sub = df[(df.tower == tower) & (df.timestamp.dt.year == year)]
        for combo, label in COMBOS.items():
            c, s, l = combo
            ss = sub[(sub.mult_cattle == c) & (sub.mult_sheep == s) & (sub.mult_lamb == l)].sort_values("timestamp")
            ax.plot(ss["timestamp"], ss["pred"], color=COLORS[combo], label=label, linewidth=1.2)
        ax.set_title(f"Tower {tower} -- {year} (zoomed)")
        ax.set_ylabel("Predicted FCH4 (nmol m-2 s-1)")
        ax.legend(fontsize=8)
    fig.suptitle(f"S-05 (2050 horizon): single-year zoom, {ssp}, {gcm}/realization {realization}")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/s05_livestock_daily_zoom{year}_{ssp}.png", dpi=120)
    plt.close(fig)
    print(f"[OK] figure: s05_livestock_daily_zoom{year}_{ssp}.png")


def plot_monthly_smoothed(df, gcm, realization, ssp):
    df = df.copy()
    df["ym"] = df["timestamp"].dt.to_period("M")
    fig, axes = plt.subplots(3, 1, figsize=(16, 11), sharex=False)
    for ax, tower in zip(axes, TOWERS):
        sub = df[df.tower == tower]
        for combo, label in COMBOS.items():
            c, s, l = combo
            ss = sub[(sub.mult_cattle == c) & (sub.mult_sheep == s) & (sub.mult_lamb == l)]
            monthly = ss.groupby("ym")["pred"].mean()
            ax.plot(monthly.index.to_timestamp(), monthly.values, color=COLORS[combo], label=label, linewidth=1.0)
        ax.set_title(f"Tower {tower} -- monthly-mean FCH4 to 2050 ({ssp}, {gcm}/realization {realization})")
        ax.set_ylabel("Monthly mean FCH4 (nmol m-2 s-1)")
        ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/s05_livestock_daily_monthly_smoothed_{ssp}.png", dpi=120)
    plt.close(fig)
    print(f"[OK] figure: s05_livestock_daily_monthly_smoothed_{ssp}.png")


def main(gcm="ACCESS-ESM1-5", realization=1, ssp="ssp245"):
    df = load_slice(gcm, realization, ssp)
    print(f"[OK] loaded slice: {len(df):,} rows ({gcm}/{realization}/{ssp}, all towers/combos)")
    plot_full_horizon(df, gcm, realization, ssp)
    plot_zoom_year(df, gcm, realization, ssp)
    plot_monthly_smoothed(df, gcm, realization, ssp)


if __name__ == "__main__":
    for _ssp in ["ssp245", "ssp585"]:
        main(ssp=_ssp)
