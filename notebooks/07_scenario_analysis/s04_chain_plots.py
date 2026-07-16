"""Generates daily chain figures (2025-2050) for S-04's top-3-per-tower daily sweep
(`results/s04_daily_top3_2050.csv`), following the same committed-script convention as
`notebooks/05_benchmarking/b10_b13_chain_plots.py` (CLAUDE.md: forecasting work should always
generate chain figures via a rerunnable script, not an ad-hoc one-off).

Unlike b10_chains (a real historical backtest against ground truth), S-04 is a blind future
projection -- there is no y_true/y_gapfilled to overlay. One figure per (tower, ssp, gcm,
realization, multiplier) combo, showing the full 26-year daily trajectory with that tower's 3
selected models (see s04_daily_top3_2050.py's TOP3 dict) overlaid on the same axes for direct
comparison. 3 towers x 2 SSPs x 5 GCMs x 2 realizations x 3 multipliers = 180 figures.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = r"c:\Users\Nicholas\Documents\COMP0191 MSc Artificial Intelligence for Sustainable Development Project"
RESULTS = rf"{ROOT}\results"
FIG_DIR = Path(RESULTS) / "figures" / "s04_chains"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Same hues as b10_b13_chain_plots.py's MODEL_COLORS for cross-figure consistency.
MODEL_COLORS = {
    "RF": "tab:green", "XGB": "tab:orange", "LightGBM": "tab:blue",
    "Ensemble_unweighted": "tab:purple", "Ensemble_MASEweighted": "tab:pink",
}


def plot_chain(sub, tower, ssp, gcm, realization, mult):
    piv = sub.pivot_table(index="date", columns="model", values="fch4")
    piv.index = pd.to_datetime(piv.index)
    piv = piv.sort_index()

    fig, ax = plt.subplots(figsize=(16, 5))
    for model in piv.columns:
        ax.plot(piv.index, piv[model], "-", color=MODEL_COLORS.get(model, "tab:gray"),
                 linewidth=0.8, label=model)
    ax.set_title(f"Tower {tower}, {ssp}, {gcm} r{realization}, {mult:g}x livestock "
                 f"(2025-2050, top-3 models)")
    ax.set_ylabel("FCH4 (nmol m-2 s-1)")
    ax.set_xlabel("Year")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    mult_tag = f"{mult:g}x"
    fig.savefig(FIG_DIR / f"T{tower}_{ssp}_{gcm}_r{realization}_{mult_tag}.png", dpi=100)
    plt.close(fig)


def main():
    df = pd.read_csv(f"{RESULTS}/s04_daily_top3_2050.csv")

    n_saved = 0
    groups = df.groupby(["tower", "ssp", "gcm", "realization", "multiplier"])
    n_groups = groups.ngroups
    print(f"[OK] {n_groups} (tower, ssp, gcm, realization, multiplier) combos to plot")

    for (tower, ssp, gcm, realization, mult), sub in groups:
        plot_chain(sub, tower, ssp, gcm, realization, mult)
        n_saved += 1
        if n_saved % 30 == 0:
            print(f"  {n_saved}/{n_groups} figures saved")

    print(f"[OK] Saved {n_saved} figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
