"""B-16 champion chain plots: actual/gap-filled/predicted, all 3 towers x 5 anchors, TabPFN+species
and TabICLv2+species (the standing forecasting champion and its closest competitor).

Mirrors b15_chain_plots.py's style (gray dotted = gap-filled, black solid = actual, colored solid =
predicted) extended to all 3 towers. Reads results/u04_chains.csv (raw per-day median predictions,
already saved by U-04 -- no new model calls) plus data/Hourly/forecast_daily_v3.csv (for the
gap-filled/actual reference series). Outputs to results/figures/b16_champion_chains/
T{tower}_anchor{year}_{model}.png.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = r"c:\Users\Nicholas\Documents\COMP0191 MSc Artificial Intelligence for Sustainable Development Project"
HOURLY = rf"{ROOT}\data\Hourly"
RESULTS = rf"{ROOT}\results"
FIG_DIR = Path(RESULTS) / "figures" / "b16_champion_chains"
FIG_DIR.mkdir(parents=True, exist_ok=True)

N_DAYS = 365
PRE_ANCHOR_DAYS = 30
ANCHOR_YEARS = [2018, 2019, 2020, 2021, 2022]
TOWERS = [2, 4, 9]

MODEL_COLORS = {
    "TabPFN": "tab:purple",
    "TabICLv2": "tab:blue",
}
DISPLAY_NAME = {
    "TabPFN": "TabPFN",
    "TabICLv2": "TabICL",
}


def plot_chain(dft, chains, yr, model, tower, anchor):
    target_dates = pd.date_range(anchor + pd.Timedelta(days=1), periods=N_DAYS, freq="D")
    window_start = anchor - pd.Timedelta(days=PRE_ANCHOR_DAYS)
    window = pd.date_range(window_start, target_dates[-1], freq="D")

    gapfilled = dft["y_gapfilled"].reindex(window)
    actual = dft["y_observed"].reindex(window)
    pred = chains.set_index("date")["median"].reindex([d.strftime("%Y-%m-%d") for d in target_dates])
    pred.index = target_dates

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(gapfilled.index, gapfilled.values, ":", color="gray", linewidth=1, label="Gap-filled FCH4")
    ax.plot(actual.index, actual.values, "-", color="black", linewidth=1, label="Actual FCH4 (observed)")
    disp = DISPLAY_NAME.get(model, model)
    ax.plot(pred.index, pred.values, "-", color=MODEL_COLORS.get(model, "tab:orange"), linewidth=1.5,
            label=f"{disp} (predicted)")
    ax.axvline(anchor, color="black", linestyle="--", linewidth=1, alpha=0.6,
               label="Forecast Start")

    ax.set_title(f"Tower {tower}, anchor {anchor.date()}, model={disp}")
    ax.set_ylabel("FCH4 (nmol m-2 s-1)")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"T{tower}_anchor{yr}_{disp}.png", dpi=100)
    plt.close(fig)


def main():
    dv = pd.read_csv(f"{HOURLY}/forecast_daily_v3.csv", low_memory=False)
    dv["Datetime"] = pd.to_datetime(dv["Datetime"], format="mixed")
    T = {t: dv[dv.tower == t].set_index("Datetime").sort_index() for t in TOWERS}

    chains_all = pd.read_csv(f"{RESULTS}/u04_chains.csv")
    chains_all["date"] = pd.to_datetime(chains_all["date"]).dt.strftime("%Y-%m-%d")

    n_saved = 0
    for yr in ANCHOR_YEARS:
        anchor = pd.Timestamp(f"{yr}-12-16")
        for tower in TOWERS:
            for model in MODEL_COLORS:
                chains = chains_all[(chains_all.anchor_year == yr) & (chains_all.eval_tower == tower)
                                     & (chains_all.model == model)]
                if chains.empty:
                    print(f"  [skip] T{tower} {yr} {model}: no data")
                    continue
                plot_chain(T[tower], chains, yr, model, tower, anchor)
                n_saved += 1

    print(f"[OK] Saved {n_saved} chain plots to {FIG_DIR}")


if __name__ == "__main__":
    main()
