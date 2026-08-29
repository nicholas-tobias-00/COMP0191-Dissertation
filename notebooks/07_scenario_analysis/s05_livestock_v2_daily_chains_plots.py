"""S-05 livestock ladder redesign: daily-chain figures, matching this project's standard 3-view
style (full horizon, single-year zoom, monthly-smoothed) used for the grazing/fertilizer axes.
Two figures per view (all_species family, cattle_alone family) since 7 combos on one panel would
be unreadable -- baseline is shown on both for reference.

Run from project root:  python notebooks/07_scenario_analysis/s05_livestock_v2_daily_chains_plots.py
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"c:\Users\Nicholas\Documents\COMP0191 MSc Artificial Intelligence for Sustainable Development Project"
RESULTS = rf"{ROOT}\results"
FIG_DIR = rf"{RESULTS}\figures\s05_summary"

TOWERS = [2, 4, 9]
SSPS = ["ssp245", "ssp585"]
ZOOM_YEAR = 2035

FAMILIES = {
    "all_species": {
        "levels": {"baseline": "Baseline (1x)", "half__all_species": "Half (0.5x)",
                   "lit_ceil__all_species": "Literature ceiling (3.0 LSU/ha)",
                   "own_max__all_species": "Own historical max"},
        "colors": {"baseline": "tab:blue", "half__all_species": "tab:green",
                   "lit_ceil__all_species": "tab:orange", "own_max__all_species": "tab:red"},
    },
    "cattle_alone": {
        "levels": {"baseline": "Baseline (1x)", "half__cattle_alone": "Half (0.5x)",
                   "lit_ceil__cattle_alone": "Literature ceiling (3.0 LSU/ha)",
                   "own_max__cattle_alone": "Own historical max"},
        "colors": {"baseline": "tab:blue", "half__cattle_alone": "tab:green",
                   "lit_ceil__cattle_alone": "tab:orange", "own_max__cattle_alone": "tab:red"},
    },
}


def load():
    df = pd.read_csv(f"{RESULTS}/s05_livestock_v2_daily_chains_subset.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def plot_full_horizon(df, family, ssp, levels, colors):
    fig, axes = plt.subplots(3, 1, figsize=(16, 11), sharex=False)
    for ax, tower in zip(axes, TOWERS):
        sub = df[(df.tower == tower) & (df.ssp == ssp)]
        for level, label in levels.items():
            ss = sub[sub.level == level].sort_values("timestamp")
            ax.plot(ss["timestamp"], ss["pred"], color=colors[level], label=label, linewidth=0.4, alpha=0.8)
        ax.set_title(f"Tower {tower} -- daily FCH4, livestock ladder ({family}), to 2050 "
                     f"({ssp}, ACCESS-ESM1-5/realization 1)")
        ax.set_ylabel("Predicted FCH4 (nmol m-2 s-1)")
        ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/s05_livestock_v2_{family}_daily_full_horizon_{ssp}.png", dpi=120)
    plt.close(fig)
    print(f"[OK] figure: s05_livestock_v2_{family}_daily_full_horizon_{ssp}.png")


def plot_zoom_year(df, family, ssp, levels, colors, year=ZOOM_YEAR):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=False)
    for ax, tower in zip(axes, TOWERS):
        sub = df[(df.tower == tower) & (df.ssp == ssp) & (df.timestamp.dt.year == year)]
        for level, label in levels.items():
            ss = sub[sub.level == level].sort_values("timestamp")
            ax.plot(ss["timestamp"], ss["pred"], color=colors[level], label=label, linewidth=1.2)
        ax.set_title(f"Tower {tower} -- {year} (zoomed)")
        ax.set_ylabel("Predicted FCH4 (nmol m-2 s-1)")
        ax.legend(fontsize=8)
    fig.suptitle(f"S-05 livestock ladder ({family}): single-year zoom, {ssp}, ACCESS-ESM1-5/realization 1")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/s05_livestock_v2_{family}_daily_zoom{year}_{ssp}.png", dpi=120)
    plt.close(fig)
    print(f"[OK] figure: s05_livestock_v2_{family}_daily_zoom{year}_{ssp}.png")


def plot_monthly_smoothed(df, family, ssp, levels, colors):
    df = df.copy()
    df["ym"] = df["timestamp"].dt.to_period("M")
    fig, axes = plt.subplots(3, 1, figsize=(16, 11), sharex=False)
    for ax, tower in zip(axes, TOWERS):
        sub = df[(df.tower == tower) & (df.ssp == ssp)]
        for level, label in levels.items():
            ss = sub[sub.level == level]
            monthly = ss.groupby("ym")["pred"].mean()
            ax.plot(monthly.index.to_timestamp(), monthly.values, color=colors[level], label=label, linewidth=1.0)
        ax.set_title(f"Tower {tower} -- monthly-mean FCH4, livestock ladder ({family}), to 2050 "
                     f"({ssp}, ACCESS-ESM1-5/realization 1)")
        ax.set_ylabel("Monthly mean FCH4 (nmol m-2 s-1)")
        ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/s05_livestock_v2_{family}_daily_monthly_smoothed_{ssp}.png", dpi=120)
    plt.close(fig)
    print(f"[OK] figure: s05_livestock_v2_{family}_daily_monthly_smoothed_{ssp}.png")


def main():
    df = load()
    print(f"[OK] loaded: {len(df):,} rows")
    for family, cfg in FAMILIES.items():
        for ssp in SSPS:
            plot_full_horizon(df, family, ssp, cfg["levels"], cfg["colors"])
            plot_zoom_year(df, family, ssp, cfg["levels"], cfg["colors"])
            plot_monthly_smoothed(df, family, ssp, cfg["levels"], cfg["colors"])


if __name__ == "__main__":
    main()
