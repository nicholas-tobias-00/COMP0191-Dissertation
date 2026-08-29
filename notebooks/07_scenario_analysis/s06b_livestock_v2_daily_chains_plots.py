"""S-06b livestock ladder: daily-chain figures, B18-derived architecture -- identical format to
`s06_livestock_v2_daily_chains_plots.py` (full horizon, single-year zoom, monthly-smoothed x
all_species/cattle_alone x both SSPs), reading S-06b's own daily-chains subset. Label corrected to
the actual D-104 ceiling (2.5 LSU/ha, UK Countryside Stewardship Annex 8) -- S-06's own original
plot script's label text was never updated after that correction; fixed here from the start.

Run from project root:  python notebooks/07_scenario_analysis/s06b_livestock_v2_daily_chains_plots.py
"""
import os
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"c:\Users\Nicholas\Documents\COMP0191 MSc Artificial Intelligence for Sustainable Development Project"
RESULTS = rf"{ROOT}\results"
FIG_DIR = rf"{RESULTS}\figures\s06b_summary"
REPORT_FIG_DIR = Path(ROOT) / "report" / "Figures"

TOWERS = [2, 4, 9]
SSPS = ["ssp245", "ssp585"]
SSP_LABELS = {"ssp245": "SSP2-4.5", "ssp585": "SSP5-8.5"}
MODEL_LABEL = "TabICLv2"
GCM_LABEL = "ACCESS-ESM1-5"
ZOOM_YEAR = 2035

FAMILIES = {
    "all_species": {
        "variation_label": "livestock",
        "levels": {"baseline": "Baseline (1x)", "half__all_species": "Half (0.5x)",
                   "lit_ceil__all_species": "Regulation Ceiling (2.5 LSU/ha)",
                   "own_max__all_species": "Own historical max"},
        "colors": {"baseline": "tab:blue", "half__all_species": "tab:green",
                   "lit_ceil__all_species": "tab:orange", "own_max__all_species": "tab:red"},
    },
    "cattle_alone": {
        "variation_label": "cattle",
        "levels": {"baseline": "Baseline (1x)", "half__cattle_alone": "Half (0.5x)",
                   "lit_ceil__cattle_alone": "Regulation Ceiling (2.5 LSU/ha)",
                   "own_max__cattle_alone": "Own historical max"},
        "colors": {"baseline": "tab:blue", "half__cattle_alone": "tab:green",
                   "lit_ceil__cattle_alone": "tab:orange", "own_max__cattle_alone": "tab:red"},
    },
}


def load():
    df = pd.read_csv(f"{RESULTS}/s06b_livestock_v2_daily_chains_subset.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
    return df


def plot_full_horizon(df, family, ssp, levels, colors, variation_label):
    fig, axes = plt.subplots(3, 1, figsize=(16, 11), sharex=False)
    for ax, tower in zip(axes, TOWERS):
        sub = df[(df.tower == tower) & (df.ssp == ssp)]
        for level, label in levels.items():
            ss = sub[sub.level == level].sort_values("timestamp")
            ax.plot(ss["timestamp"], ss["pred"], color=colors[level], label=label, linewidth=0.4, alpha=0.8)
        ax.set_title(
            f"Tower {tower} - daily FCH4 - variations inserted to {variation_label} "
            f"({SSP_LABELS[ssp]}, Model {MODEL_LABEL}, GCM {GCM_LABEL})"
        )
        ax.set_ylabel("Predicted FCH4 (nmol m-2 s-1)")
        ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/s06b_livestock_v2_{family}_daily_full_horizon_{ssp}.png", dpi=120)
    plt.close(fig)
    print(f"[OK] figure: s06b_livestock_v2_{family}_daily_full_horizon_{ssp}.png")


def plot_zoom_year(df, family, ssp, levels, colors, variation_label, year=ZOOM_YEAR):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=False)
    for ax, tower in zip(axes, TOWERS):
        sub = df[(df.tower == tower) & (df.ssp == ssp) & (df.timestamp.dt.year == year)]
        for level, label in levels.items():
            ss = sub[sub.level == level].sort_values("timestamp")
            ax.plot(ss["timestamp"], ss["pred"], color=colors[level], label=label, linewidth=1.2)
        ax.set_title(f"Tower {tower} - {year} - variations inserted to {variation_label}")
        ax.set_ylabel("Predicted FCH4 (nmol m-2 s-1)")
        ax.legend(fontsize=8)
    fig.suptitle(f"{SSP_LABELS[ssp]}, Model {MODEL_LABEL}, GCM {GCM_LABEL}")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/s06b_livestock_v2_{family}_daily_zoom{year}_{ssp}.png", dpi=120)
    plt.close(fig)
    print(f"[OK] figure: s06b_livestock_v2_{family}_daily_zoom{year}_{ssp}.png")


def plot_monthly_smoothed(df, family, ssp, levels, colors, variation_label):
    df = df.copy()
    df["ym"] = df["timestamp"].dt.to_period("M")
    fig, axes = plt.subplots(3, 1, figsize=(16, 11), sharex=False)
    for ax, tower in zip(axes, TOWERS):
        sub = df[(df.tower == tower) & (df.ssp == ssp)]
        for level, label in levels.items():
            ss = sub[sub.level == level]
            monthly = ss.groupby("ym")["pred"].mean()
            ax.plot(monthly.index.to_timestamp(), monthly.values, color=colors[level], label=label, linewidth=1.0)
        ax.set_title(
            f"Tower {tower} - monthly mean FCH4 - variations inserted to {variation_label} "
            f"({SSP_LABELS[ssp]}, Model {MODEL_LABEL}, GCM {GCM_LABEL})"
        )
        ax.set_ylabel("Monthly mean FCH4 (nmol m-2 s-1)")
        ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    output = f"{FIG_DIR}/s06b_livestock_v2_{family}_daily_monthly_smoothed_{ssp}.png"
    fig.savefig(output, dpi=120)
    if family == "all_species":
        explicit_name = f"ch6_livestock_ladder_trajectory_{ssp}.png"
        fig.savefig(REPORT_FIG_DIR / explicit_name, dpi=120)
        if ssp == "ssp245":
            fig.savefig(REPORT_FIG_DIR / "ch6_livestock_ladder_trajectory.png", dpi=120)
    plt.close(fig)
    print(f"[OK] figure: s06b_livestock_v2_{family}_daily_monthly_smoothed_{ssp}.png")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    REPORT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load()
    print(f"[OK] loaded: {len(df):,} rows")
    for family, cfg in FAMILIES.items():
        for ssp in SSPS:
            plot_full_horizon(df, family, ssp, cfg["levels"], cfg["colors"], cfg["variation_label"])
            plot_zoom_year(df, family, ssp, cfg["levels"], cfg["colors"], cfg["variation_label"])
            plot_monthly_smoothed(df, family, ssp, cfg["levels"], cfg["colors"], cfg["variation_label"])


if __name__ == "__main__":
    main()
