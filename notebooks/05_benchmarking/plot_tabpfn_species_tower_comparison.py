"""Plot the saved TabPFN + BASE+species forecast against observed and gap-filled CH4.

The U04 chain contains the daily TabPFN median and observed target for five consecutive
365-day rollouts. Gap-filled values are joined from forecast_daily_v3.csv. The output is the
three-panel replacement figure intended for Forecasting Section 5.3.1.
"""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CHAINS = ROOT / "results" / "u04_chains.csv"
DAILY = ROOT / "data" / "Hourly" / "forecast_daily_v3.csv"
OUTPUT = ROOT / "report" / "Figures" / "ch5_tabpfn_species_tower_comparison.png"

TOWERS = (2, 4, 9)
PREDICTION_BLUE = "#0072B2"


def load_plot_data() -> pd.DataFrame:
    chains = pd.read_csv(CHAINS)
    chains = chains.loc[chains["model"].eq("TabPFN")].copy()
    chains["date"] = pd.to_datetime(chains["date"])
    chains = chains.rename(
        columns={"eval_tower": "tower", "median": "y_predict"}
    )

    daily = pd.read_csv(
        DAILY,
        usecols=["Datetime", "tower", "y_gapfilled"],
        low_memory=False,
    )
    daily["date"] = pd.to_datetime(daily["Datetime"], format="mixed")
    daily = daily.drop(columns="Datetime")

    out = chains.merge(daily, on=["tower", "date"], how="left", validate="one_to_one")
    if out["y_gapfilled"].isna().any():
        missing = int(out["y_gapfilled"].isna().sum())
        raise RuntimeError(f"Missing y_gapfilled values after merge: {missing}")
    return out.sort_values(["tower", "date"])


def main() -> None:
    data = load_plot_data()

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )

    fig, axes = plt.subplots(
        len(TOWERS),
        1,
        figsize=(11.2, 8.2),
        sharex=True,
        constrained_layout=False,
    )

    for ax, tower in zip(axes, TOWERS):
        sub = data.loc[data["tower"].eq(tower)].copy()

        # Plot the reconstruction first so genuine observations remain visually dominant.
        ax.plot(
            sub["date"],
            sub["y_gapfilled"],
            color="black",
            linestyle=":",
            linewidth=0.9,
            alpha=0.62,
            label=r"Gap-filled $y_{gapfilled}$",
            zorder=1,
        )
        ax.plot(
            sub["date"],
            sub["y_predict"],
            color=PREDICTION_BLUE,
            linewidth=1.25,
            alpha=0.95,
            label=r"Prediction $y_{predict}$",
            zorder=2,
        )
        ax.plot(
            sub["date"],
            sub["y_true"],
            color="black",
            linewidth=0.9,
            alpha=0.95,
            label=r"Observed $y_{true}$",
            zorder=3,
        )
        observed = sub["y_true"].notna()
        ax.scatter(
            sub.loc[observed, "date"],
            sub.loc[observed, "y_true"],
            color="black",
            s=5,
            linewidths=0,
            alpha=0.9,
            zorder=4,
        )

        # The five chains are consecutive but independently re-anchored each December.
        for year in (2019, 2020, 2021, 2022):
            ax.axvline(
                pd.Timestamp(f"{year}-12-17"),
                color="#B5B5B5",
                linestyle="--",
                linewidth=0.65,
                alpha=0.75,
                zorder=0,
            )

        n_observed = int(observed.sum())
        ax.set_title(f"Tower T{tower}  |  observed evaluation days: {n_observed}", loc="left")
        ax.set_ylabel(r"CH$_4$ flux")
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.65)
        ax.spines[["top", "right"]].set_visible(False)
        ax.margins(x=0)

    axes[-1].set_xlabel("Forecast date")
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    handles, labels = axes[0].get_legend_handles_labels()
    # Requested visual order: prediction, observed truth, gap-filled reconstruction.
    order = [1, 2, 0]
    fig.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.987),
        ncol=3,
        frameon=False,
    )
    fig.suptitle(
        "TabPFN + BASE+species: daily recursive forecasts by tower",
        fontsize=12,
        y=0.999,
    )
    fig.subplots_adjust(top=0.925, bottom=0.075, left=0.085, right=0.99, hspace=0.28)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()

