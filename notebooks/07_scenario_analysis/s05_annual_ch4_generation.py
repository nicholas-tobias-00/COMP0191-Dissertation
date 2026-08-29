"""Annual methane generation, for visualization (supervisor feedback, 2026-08-13): converts S-05's
`annual_mean` column (mean daily FCH4 flux, nmol CH4 m-2 s-1 -- this project's native unit
throughout) into an annual mass estimate, the unit non-specialist readers actually interpret.

Conversion (standard EC flux unit algebra, derived and verified this session):
  kg CH4 ha-1 yr-1 = flux_nmol_m2_s * 1e-9 (mol) * 16.04 (g/mol, CH4 molar mass) * 1e-3 (kg/g)
                     * 10,000 (m2/ha) * 31,536,000 (s/yr, 365-day year -- matches this project's
                     own N_DAYS=365 rollout convention)
                   = flux_nmol_m2_s * 5.0584

`annual_mean` is already the mean of 365 equally-weighted daily chain values (recursive_rollout's
own construction), so this single multiplication is exact -- no need to re-derive from raw daily
chains. Total catchment mass (kg CH4/yr, not per-ha) uses each tower's real fenced area
(CATCHMENT_AREA_HA, build_management_features.py: T2=6.65, T4=7.75, T9=7.75 ha).

Run from project root:  python notebooks/07_scenario_analysis/s05_annual_ch4_generation.py
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"c:\Users\Nicholas\Documents\COMP0191 MSc Artificial Intelligence for Sustainable Development Project"
RESULTS = rf"{ROOT}\results"
FIG_DIR = rf"{RESULTS}\figures\s05_annual_ch4"
os.makedirs(FIG_DIR, exist_ok=True)

FLUX_TO_KG_HA_YR = 5.0584  # nmol CH4 m-2 s-1 -> kg CH4 ha-1 yr-1, see module docstring
AREA_HA = {2: 6.65, 4: 7.75, 9: 7.75}


def add_ch4_generation(df):
    """Adds annual_CH4_kg_ha_yr and annual_CH4_kg_total_yr columns in place; returns df."""
    df = df.copy()
    df["annual_CH4_kg_ha_yr"] = df["annual_mean"] * FLUX_TO_KG_HA_YR
    df["annual_CH4_kg_total_yr"] = df["annual_CH4_kg_ha_yr"] * df["tower"].map(AREA_HA)
    return df


def plot_axis(df, level_col, level_order, level_labels, title, fname, ssp="ssp245"):
    """One figure, 3 stacked panels (one per tower), annual CH4 generation (kg/ha/yr) vs. year,
    one line per level -- mirrors this project's standard scenario-trajectory figure format."""
    sub = df[df.ssp == ssp]
    towers = sorted(sub.tower.unique())
    fig, axes = plt.subplots(len(towers), 1, figsize=(12, 4 * len(towers)), sharex=True)
    if len(towers) == 1:
        axes = [axes]
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(level_order)))
    for ax, tower in zip(axes, towers):
        tsub = sub[sub.tower == tower]
        for lvl, color in zip(level_order, colors):
            lsub = tsub[tsub[level_col] == lvl].groupby("year")["annual_CH4_kg_ha_yr"].mean()
            if lsub.empty:
                continue
            ax.plot(lsub.index, lsub.values, color=color, linewidth=1.6,
                     label=level_labels.get(lvl, lvl), marker="o", markersize=3)
        ax.set_title(f"Tower {tower}")
        ax.set_ylabel("Annual CH4 generation\n(kg CH4 ha$^{-1}$ yr$^{-1}$)")
        ax.legend(fontsize=8, ncol=2, loc="upper left")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Year")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/{fname}", dpi=120)
    plt.close(fig)
    print(f"[OK] Saved {fname}")


def summary_table(df, level_col, group_cols=None):
    """Pooled (all years/GCMs/realizations/SSPs) mean annual CH4 generation per group -- the
    single-number headline table to sit alongside each scenario's existing % change finding."""
    group_cols = group_cols or ["tower", level_col]
    return df.groupby(group_cols)[["annual_mean", "annual_CH4_kg_ha_yr",
                                    "annual_CH4_kg_total_yr"]].mean().round(3)


def main():
    sources = {
        "livestock_v2": (f"{RESULTS}/s05_practices_livestock_v2.csv", "level"),
        "grazing": (f"{RESULTS}/s05_practices_grazing.csv", "level"),
        "fertilizer": (f"{RESULTS}/s05_practices_fertilizer.csv", "level"),
    }
    for name, (path, level_col) in sources.items():
        if not os.path.exists(path):
            print(f"[SKIP] {name}: {path} not found yet")
            continue
        df = add_ch4_generation(pd.read_csv(path))
        out_path = f"{RESULTS}/s05_annual_ch4_{name}.csv"
        df.to_csv(out_path, index=False)
        print(f"[OK] Saved {out_path} ({len(df)} rows)")

        tab = summary_table(df, level_col)
        tab.to_csv(f"{RESULTS}/s05_annual_ch4_{name}_summary.csv")
        print(tab.to_string())
        print()

        levels = sorted(df[level_col].unique())
        plot_axis(df, level_col, levels, {l: l for l in levels},
                  f"S-05 annual CH4 generation -- {name} axis (ssp245)",
                  f"s05_annual_ch4_{name}_ssp245.png")


if __name__ == "__main__":
    main()
