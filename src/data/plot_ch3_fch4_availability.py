"""Generate the Chapter 3 three-tower FCH4 availability overview.

The figure uses the common hourly EC analysis window (2017-01-01 to the
exclusive 2024-01-01 boundary). A usable target observation must have an
SSITC class of 0 or 1 and fall within the project's [-500, 3000] nmol m-2 s-1
plausibility range.
"""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "Hourly" / "consolidated_hourly.csv"
OUTPUT = ROOT / "report" / "Figures" / "ch3_fch4_three_tower_availability.png"

START = pd.Timestamp("2017-01-01")
END = pd.Timestamp("2024-01-01")
LOWER_BOUND = -500
UPPER_BOUND = 3000
DISPLAY_LOWER = -250
DISPLAY_UPPER = 1000
TOWERS = (2, 4, 9)
COLOURS = {2: "#C44E52", 4: "#55A868", 9: "#4C72B0"}


def load_data() -> pd.DataFrame:
    data = pd.read_csv(INPUT, low_memory=False)
    data["Datetime"] = pd.to_datetime(data["Datetime"], format="mixed")
    return data.loc[
        data["Datetime"].between(START, END, inclusive="left")
    ].set_index("Datetime")


def usable_target(
    data: pd.DataFrame, tower: int
) -> tuple[pd.Series, pd.Series, pd.Series]:
    target = pd.to_numeric(data[f"FCH4_1_1_1 [Tower {tower}]"], errors="coerce")
    quality = pd.to_numeric(
        data[f"FCH4_SSITC_TEST_1_1_1 [Tower {tower}]"], errors="coerce"
    )
    valid = (
        target.notna()
        & quality.isin([0, 1])
        & target.between(LOWER_BOUND, UPPER_BOUND, inclusive="both")
    )
    return target.where(valid), valid, target.notna()


def main() -> None:
    data = load_data()
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(10.5, 9.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1, 1, 0.9], "hspace": 0.20},
    )

    annual = {}
    for ax, tower in zip(axes[:3], TOWERS):
        target, valid, raw_available = usable_target(data, tower)
        daily = target.resample("D").agg(
            median="median",
            q05=lambda values: values.quantile(0.05),
            q95=lambda values: values.quantile(0.95),
        )
        raw_missing = 100 * (1 - raw_available.mean())
        annual[tower] = 100 * valid.groupby(valid.index.year).mean()

        colour = COLOURS[tower]
        ax.fill_between(
            daily.index,
            daily["q05"],
            daily["q95"],
            color=colour,
            alpha=0.22,
            linewidth=0,
            label="Daily 5th--95th percentile",
        )
        ax.plot(
            daily.index,
            daily["median"],
            color=colour,
            linewidth=0.85,
            label="Daily median",
        )
        ax.axhline(0, color="#555555", linewidth=0.55, alpha=0.65)
        ax.set_ylim(DISPLAY_LOWER, DISPLAY_UPPER)
        ax.set_title(
            f"Tower {tower}  |  {raw_missing:.1f}% of hourly raw target unavailable",
            loc="left",
            fontsize=10.5,
            fontweight="bold",
        )
        ax.grid(axis="y", alpha=0.22)

    axes[0].legend(loc="upper right", ncol=2, frameon=False, fontsize=8.5)
    fig.supylabel(r"FCH$_4$ flux (nmol m$^{-2}$ s$^{-1}$)", x=0.015, fontsize=10)

    annual_ax = axes[3]
    for tower in TOWERS:
        series = annual[tower]
        year_starts = pd.to_datetime(series.index.astype(str), format="%Y")
        annual_ax.plot(
            year_starts,
            series.values,
            color=COLOURS[tower],
            marker="o",
            linewidth=1.6,
            markersize=4.5,
            label=f"Tower {tower}",
        )
    annual_ax.axhline(50, color="#777777", linestyle="--", linewidth=0.8)
    annual_ax.set_ylim(0, 100)
    annual_ax.set_ylabel("QC-valid hours (%)")
    annual_ax.set_title("Annual completeness of the usable modelling target", loc="left", fontsize=10)
    annual_ax.grid(alpha=0.22)
    annual_ax.legend(loc="upper left", ncol=3, frameon=False, fontsize=8.5)
    annual_ax.set_xlabel("Year")
    annual_ax.xaxis.set_major_locator(mdates.YearLocator())
    annual_ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    annual_ax.set_xlim(START, END)

    fig.subplots_adjust(left=0.09, right=0.985, top=0.985, bottom=0.075)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
