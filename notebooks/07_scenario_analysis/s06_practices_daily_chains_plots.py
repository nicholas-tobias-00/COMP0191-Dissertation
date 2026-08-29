"""S-06 grazing/fertilizer: daily-chain figures, bias-corrected drivers -- identical format to
`s05_practices_daily_chains_plots.py`, reading S-06's own daily-chains subset.

Run from project root:  python notebooks/07_scenario_analysis/s06_practices_daily_chains_plots.py
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"c:\Users\Nicholas\Documents\COMP0191 MSc Artificial Intelligence for Sustainable Development Project"
RESULTS = rf"{ROOT}\results"
FIG_DIR = rf"{RESULTS}\figures\s06_summary"

TOWERS = [2, 4, 9]
SSPS = ["ssp245", "ssp585"]
ZOOM_YEAR = 2035

AXES = {
    "grazing": {
        "levels": {"historical": "Historical (no shift)", "plus2wk": "+2 weeks", "plus4wk": "+4 weeks"},
        "colors": {"historical": "tab:blue", "plus2wk": "tab:orange", "plus4wk": "tab:red"},
    },
    "fertilizer": {
        "levels": {"historical": "Historical", "plus50pct_rate": "+50% rate", "plus50pct_freq": "+50% frequency",
                   "reg_cap": "Regulatory cap (300 kg N/ha/yr, NVZ N-max)"},
        "colors": {"historical": "tab:blue", "plus50pct_rate": "tab:orange", "plus50pct_freq": "tab:red",
                   "reg_cap": "tab:purple"},
    },
}


def load(axis):
    df = pd.read_csv(f"{RESULTS}/s06_practices_{axis}_daily_chains_subset.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
    return df


def plot_full_horizon(df, axis, ssp, levels, colors):
    fig, axes = plt.subplots(3, 1, figsize=(16, 11), sharex=False)
    for ax, tower in zip(axes, TOWERS):
        sub = df[(df.tower == tower) & (df.ssp == ssp)]
        for level, label in levels.items():
            ss = sub[sub.level == level].sort_values("timestamp")
            ax.plot(ss["timestamp"], ss["pred"], color=colors[level], label=label, linewidth=0.4, alpha=0.8)
        ax.set_title(f"Tower {tower} -- daily FCH4, {axis} scenario, bias-corrected, to 2050 "
                     f"({ssp}, ACCESS-ESM1-5/realization 1)")
        ax.set_ylabel("Predicted FCH4 (nmol m-2 s-1)")
        ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/s06_practices_{axis}_daily_full_horizon_{ssp}.png", dpi=120)
    plt.close(fig)
    print(f"[OK] figure: s06_practices_{axis}_daily_full_horizon_{ssp}.png")


def plot_zoom_year(df, axis, ssp, levels, colors, year=ZOOM_YEAR):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=False)
    for ax, tower in zip(axes, TOWERS):
        sub = df[(df.tower == tower) & (df.ssp == ssp) & (df.timestamp.dt.year == year)]
        for level, label in levels.items():
            ss = sub[sub.level == level].sort_values("timestamp")
            ax.plot(ss["timestamp"], ss["pred"], color=colors[level], label=label, linewidth=1.2)
        ax.set_title(f"Tower {tower} -- {year} (zoomed)")
        ax.set_ylabel("Predicted FCH4 (nmol m-2 s-1)")
        ax.legend(fontsize=8)
    fig.suptitle(f"S-06 practices ({axis}), bias-corrected: single-year zoom, {ssp}, ACCESS-ESM1-5/realization 1")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/s06_practices_{axis}_daily_zoom{year}_{ssp}.png", dpi=120)
    plt.close(fig)
    print(f"[OK] figure: s06_practices_{axis}_daily_zoom{year}_{ssp}.png")


def plot_monthly_smoothed(df, axis, ssp, levels, colors):
    df = df.copy()
    df["ym"] = df["timestamp"].dt.to_period("M")
    fig, axes = plt.subplots(3, 1, figsize=(16, 11), sharex=False)
    for ax, tower in zip(axes, TOWERS):
        sub = df[(df.tower == tower) & (df.ssp == ssp)]
        for level, label in levels.items():
            ss = sub[sub.level == level]
            monthly = ss.groupby("ym")["pred"].mean()
            ax.plot(monthly.index.to_timestamp(), monthly.values, color=colors[level], label=label, linewidth=1.0)
        ax.set_title(f"Tower {tower} -- monthly-mean FCH4, {axis} scenario, bias-corrected, to 2050 "
                     f"({ssp}, ACCESS-ESM1-5/realization 1)")
        ax.set_ylabel("Monthly mean FCH4 (nmol m-2 s-1)")
        ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/s06_practices_{axis}_daily_monthly_smoothed_{ssp}.png", dpi=120)
    plt.close(fig)
    print(f"[OK] figure: s06_practices_{axis}_daily_monthly_smoothed_{ssp}.png")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    for axis, cfg in AXES.items():
        df = load(axis)
        print(f"[OK] loaded {axis}: {len(df):,} rows")
        for ssp in SSPS:
            plot_full_horizon(df, axis, ssp, cfg["levels"], cfg["colors"])
            plot_zoom_year(df, axis, ssp, cfg["levels"], cfg["colors"])
            plot_monthly_smoothed(df, axis, ssp, cfg["levels"], cfg["colors"])


if __name__ == "__main__":
    main()
