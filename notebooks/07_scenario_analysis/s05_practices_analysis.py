"""S-05 practices analysis: summarizes `s05_practices_trajectory.py`'s two axes (grazing timing,
fertilizer schedule), each run at baseline (1x/1x/1x) livestock, 2050 horizon, 900 calls/axis.
Read-only, no new model fitting.

Produces per axis: pct-change-vs-historical by tower (pooled across SSP/GCM/realization/year),
AOA-flagged-% by level (does extending grazing season or changing fertilizer schedule push the
scenario further out of the training envelope?), and a bar-chart figure.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

ROOT = r"c:\Users\Nicholas\Documents\COMP0191 MSc Artificial Intelligence for Sustainable Development Project"
RESULTS = rf"{ROOT}\results"
FIG_DIR = rf"{RESULTS}\figures\s05_summary"
os.makedirs(FIG_DIR, exist_ok=True)

TOWERS = [2, 4, 9]
TOWER_COLORS = {2: "tab:blue", 4: "tab:orange", 9: "tab:green"}
GRAZING_LEVELS = ["historical", "plus2wk", "plus4wk"]
FERT_LEVELS = ["historical", "plus50pct_freq", "plus50pct_rate"]


def summarize(axis, levels):
    df = pd.read_csv(f"{RESULTS}/s05_practices_{axis}.csv")
    print(f"[OK] s05_practices_{axis}.csv: {len(df):,} rows")

    g = df.groupby(["tower", "level"])["annual_mean"].mean().unstack("level")[levels]
    g["pct_vs_historical_max"] = (g[levels[-1]] / g["historical"] - 1) * 100
    g.to_csv(f"{RESULTS}/s05_practices_{axis}_summary.csv")
    print(g.round(3).to_string())

    aoa = df.groupby(["tower", "level"])["aoa_flagged_pct"].mean().unstack("level")[levels]
    aoa.to_csv(f"{RESULTS}/s05_practices_{axis}_aoa.csv")
    print(f"\nAOA-flagged %% by level:\n{aoa.round(1).to_string()}")

    return g, aoa


def plot_axis(g, axis, levels, ylabel_suffix):
    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(levels))
    width = 0.25
    for i, t in enumerate(TOWERS):
        vals = g.loc[t, levels].values
        ax.bar(x + (i - 1) * width, vals, width, label=f"Tower {t}", color=TOWER_COLORS[t])
    ax.set_xticks(x)
    ax.set_xticklabels(levels)
    ax.set_ylabel(f"Predicted annual mean FCH4 (nmol m-2 s-1){ylabel_suffix}")
    ax.set_title(f"S-05 practices: {axis} scenario response (baseline livestock, 2050 horizon)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/s05_practices_{axis}.png", dpi=120)
    plt.close(fig)
    print(f"[OK] figure: s05_practices_{axis}.png")


def main():
    print("=== Grazing timing ===")
    g_summary, g_aoa = summarize("grazing", GRAZING_LEVELS)
    plot_axis(g_summary, "grazing", GRAZING_LEVELS, "")

    print("\n=== Fertilizer schedule ===")
    f_summary, f_aoa = summarize("fertilizer", FERT_LEVELS)
    plot_axis(f_summary, "fertilizer", FERT_LEVELS, "")

    print("\n[DONE] S-05 practices analysis complete.")


if __name__ == "__main__":
    main()
