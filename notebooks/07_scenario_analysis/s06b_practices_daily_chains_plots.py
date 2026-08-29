"""S-06b grazing/fertilizer: daily-chain figures, B18-derived architecture -- identical format to
`s06_practices_daily_chains_plots.py`, reading S-06b's own daily-chains subset. Fertilizer levels
here are the 3-level base set only (`reg_cap` deferred, matching this Phase's own explicit scoping
decision, see s06b_master_runner.py's docstring) -- add back if/when reg_cap is rerun for S-06b.

Run from project root:  python notebooks/07_scenario_analysis/s06b_practices_daily_chains_plots.py
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

AXES = {
    "grazing": {
        "variation_label": "grazing timing",
        "levels": {"historical": "Historical (no shift)", "plus2wk": "+2 weeks", "plus4wk": "+4 weeks"},
        "colors": {"historical": "tab:blue", "plus2wk": "tab:orange", "plus4wk": "tab:red"},
    },
    "fertilizer": {
        "variation_label": "fertiliser management",
        "levels": {"historical": "Historical", "plus50pct_rate": "+50% rate", "plus50pct_freq": "+50% frequency"},
        "colors": {"historical": "tab:blue", "plus50pct_rate": "tab:orange", "plus50pct_freq": "tab:red"},
    },
}


def load(axis):
    df = pd.read_csv(f"{RESULTS}/s06b_practices_{axis}_daily_chains_subset.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
    return df


def plot_full_horizon(df, axis, ssp, levels, colors, variation_label):
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
    fig.savefig(f"{FIG_DIR}/s06b_practices_{axis}_daily_full_horizon_{ssp}.png", dpi=120)
    plt.close(fig)
    print(f"[OK] figure: s06b_practices_{axis}_daily_full_horizon_{ssp}.png")


def plot_zoom_year(df, axis, ssp, levels, colors, variation_label, year=ZOOM_YEAR):
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
    fig.savefig(f"{FIG_DIR}/s06b_practices_{axis}_daily_zoom{year}_{ssp}.png", dpi=120)
    plt.close(fig)
    print(f"[OK] figure: s06b_practices_{axis}_daily_zoom{year}_{ssp}.png")


def plot_monthly_smoothed(df, axis, ssp, levels, colors, variation_label):
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
    output = f"{FIG_DIR}/s06b_practices_{axis}_daily_monthly_smoothed_{ssp}.png"
    fig.savefig(output, dpi=120)
    report_stem = "grazing_trajectory" if axis == "grazing" else "fertilizer_trajectory"
    fig.savefig(REPORT_FIG_DIR / f"ch6_{report_stem}_{ssp}.png", dpi=120)
    if ssp == "ssp245":
        fig.savefig(REPORT_FIG_DIR / f"ch6_{report_stem}.png", dpi=120)
    plt.close(fig)
    print(f"[OK] figure: s06b_practices_{axis}_daily_monthly_smoothed_{ssp}.png")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    REPORT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    for axis, cfg in AXES.items():
        df = load(axis)
        print(f"[OK] loaded {axis}: {len(df):,} rows")
        for ssp in SSPS:
            plot_full_horizon(df, axis, ssp, cfg["levels"], cfg["colors"], cfg["variation_label"])
            plot_zoom_year(df, axis, ssp, cfg["levels"], cfg["colors"], cfg["variation_label"])
            plot_monthly_smoothed(df, axis, ssp, cfg["levels"], cfg["colors"], cfg["variation_label"])


if __name__ == "__main__":
    main()
