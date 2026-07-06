"""B-15 chain plots: actual/gap-filled/predicted, Tower 4 only, 5 models x 5 anchors.

Mirrors results/figures/b10_chains/'s plot style (same title format, same line styles for
gap-filled/actual, predicted line in a per-model color) applied to B-15's rollout-tuned models.
Reads results/b15_chains.csv (per-day predicted values, produced by b15_multi_anchor.py) plus
data/Hourly/forecast_daily_v2.csv (for the gap-filled/actual reference series). Outputs to
results/figures/b15_chains/T4_anchor{year}_{model}.png.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = r"c:\Users\Nicholas\Documents\COMP0191 MSc Artificial Intelligence for Sustainable Development Project"
HOURLY = rf"{ROOT}\data\Hourly"
RESULTS = rf"{ROOT}\results"
FIG_DIR = Path(RESULTS) / "figures" / "b15_chains"
FIG_DIR.mkdir(parents=True, exist_ok=True)

N_DAYS = 365
PRE_ANCHOR_DAYS = 30
ANCHOR_YEARS = [2018, 2019, 2020, 2021, 2022]
TOWER_MAIN = 4

MODEL_COLORS = {
    "RF_tuned": "tab:green",
    "XGB_tuned": "tab:orange",
    "LightGBM_tuned": "tab:blue",
    "SARIMAX": "tab:red",
    "Ensemble_4model_tuned": "tab:purple",
}


def plot_chain(df4, chains, yr, model, anchor):
    target_dates = pd.date_range(anchor + pd.Timedelta(days=1), periods=N_DAYS, freq="D")
    window_start = anchor - pd.Timedelta(days=PRE_ANCHOR_DAYS)
    window = pd.date_range(window_start, target_dates[-1], freq="D")

    gapfilled = df4["y_gapfilled"].reindex(window)
    actual = df4["y_observed"].reindex(window)
    pred = chains.set_index("Datetime")[model].reindex(target_dates)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(gapfilled.index, gapfilled.values, ":", color="gray", linewidth=1, label="Gap-filled FCH4")
    ax.plot(actual.index, actual.values, "-", color="black", linewidth=1, label="Actual FCH4 (observed)")
    ax.plot(pred.index, pred.values, "-", color=MODEL_COLORS.get(model, "tab:purple"), linewidth=1.5,
            label=f"{model} (predicted)")

    ax.set_title(f"Tower {TOWER_MAIN}, anchor {anchor.date()}, model={model}")
    ax.set_ylabel("FCH4 (nmol m-2 s-1)")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"T{TOWER_MAIN}_anchor{yr}_{model}.png", dpi=100)
    plt.close(fig)


def main():
    dv = pd.read_csv(f"{HOURLY}/forecast_daily_v2.csv", low_memory=False)
    dv["Datetime"] = pd.to_datetime(dv["Datetime"], format="mixed")
    df4 = dv[dv.tower == TOWER_MAIN].set_index("Datetime").sort_index()

    chains_all = pd.read_csv(f"{RESULTS}/b15_chains.csv", parse_dates=["Datetime"])

    n_saved = 0
    for yr in ANCHOR_YEARS:
        anchor = pd.Timestamp(f"{yr}-12-16")
        chains = chains_all[chains_all.anchor_year == yr]
        for model in MODEL_COLORS:
            plot_chain(df4, chains, yr, model, anchor)
            n_saved += 1

    print(f"[OK] Saved {n_saved} chain plots to {FIG_DIR}")


if __name__ == "__main__":
    main()
