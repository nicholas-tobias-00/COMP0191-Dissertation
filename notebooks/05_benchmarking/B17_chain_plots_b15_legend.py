"""Additive B17 champion chain plots using B15's exact legend convention.

Writes a separate corrected figure set so the original B17 artifacts remain
untouched.  Legend entries follow B15: gap-filled, actual observed, and
``{model} (predicted)`` only.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "Hourly" / "forecast_daily_v3.csv"
CHAINS_PATH = ROOT / "results" / "b17_direct_tuning_chains.csv"
FIG_DIR = ROOT / "results" / "figures" / "b17_chains_b15_legend"
FIG_DIR.mkdir(parents=True, exist_ok=True)

TOWERS = [2, 4, 9]
ANCHOR_YEARS = [2018, 2019, 2020, 2021, 2022]
N_DAYS = 365
PRE_ANCHOR_DAYS = 30
MODEL = "TabPFN_v2"
CHAMPION_VARIANT = "pooled_time_seed137_raw_median"


def plot_chain(dft, chain, tower, year):
    anchor = pd.Timestamp(f"{year}-12-16")
    target_dates = pd.date_range(anchor + pd.Timedelta(days=1), periods=N_DAYS, freq="D")
    window = pd.date_range(
        anchor - pd.Timedelta(days=PRE_ANCHOR_DAYS), target_dates[-1], freq="D"
    )

    gapfilled = dft["y_gapfilled"].reindex(window)
    actual = dft["y_observed"].reindex(window)
    prediction = chain.set_index("date")["y_predict"].reindex(target_dates)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(
        gapfilled.index,
        gapfilled.values,
        ":",
        color="gray",
        linewidth=1,
        label="Gap-filled FCH4",
    )
    ax.plot(
        actual.index,
        actual.values,
        "-",
        color="black",
        linewidth=1,
        label="Actual FCH4 (observed)",
    )
    ax.plot(
        prediction.index,
        prediction.values,
        "-",
        color="tab:blue",
        linewidth=1.5,
        label=f"{MODEL} (predicted)",
    )

    ax.set_title(f"Tower {tower}, anchor {anchor.date()}, model={MODEL}")
    ax.set_ylabel("FCH4 (nmol m-2 s-1)")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"T{tower}_anchor{year}_{MODEL}.png", dpi=100)
    plt.close(fig)


def main():
    data = pd.read_csv(DATA_PATH, low_memory=False)
    data["Datetime"] = pd.to_datetime(data["Datetime"], format="mixed")
    tower_frames = {
        tower: data.loc[data["tower"].eq(tower)].set_index("Datetime").sort_index()
        for tower in TOWERS
    }

    chains = pd.read_csv(CHAINS_PATH, parse_dates=["date"])
    chains = chains.loc[
        chains["model"].eq("Direct_TabPFN_v2")
        & chains["config"].eq("BASE_ALL_52")
        & chains["variant"].eq(CHAMPION_VARIANT)
    ]

    saved = 0
    for tower in TOWERS:
        for year in ANCHOR_YEARS:
            chain = chains.loc[
                chains["tower"].eq(tower) & chains["anchor_year"].eq(year)
            ]
            if chain.empty:
                print(f"[skip] T{tower} anchor {year}: no champion chain")
                continue
            plot_chain(tower_frames[tower], chain, tower, year)
            saved += 1
    print(f"[OK] Saved {saved} B15-style B17 chain plots to {FIG_DIR}")


if __name__ == "__main__":
    main()
