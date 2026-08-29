"""Export report-ready daily figures from the latest TabICLv2 tower chains.

This is a plotting-only step: it reads the persisted six-month production
chain and performs no model fitting or prediction.
"""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
CHAIN_PATH = HERE / "_data" / "latest_tabicl_uq_6month_chains.csv"
OUTPUT_NAMES = {
    2: "ch4_tabiclv2_daily_T2_2019H1.png",
    4: "ch4_tabiclv2_daily_T4_2023H2.png",
    9: "ch4_tabiclv2_daily_T9_2023H2.png",
}


def plot_tower(chain: pd.DataFrame, tower_number: int) -> Path:
    tower = chain.loc[chain["tower"].eq(tower_number)].copy()
    if tower.empty:
        raise RuntimeError(f"The saved chain contains no Tower {tower_number} rows.")

    tower = tower.set_index("Datetime").sort_index()
    daily = pd.DataFrame(
        {
            "observed": tower["y_observed"].resample("1D").mean(),
            "completed": tower["y_gapfilled"].resample("1D").mean(),
            "observed_hours": tower["y_observed"].resample("1D").count(),
        }
    )
    daily.loc[daily["observed_hours"].eq(0), "observed"] = pd.NA

    coverage = 100.0 * tower["y_observed"].notna().mean()
    start = tower.index.min()
    end = tower.index.max()

    fig, ax = plt.subplots(figsize=(12.5, 4.8))
    ax.plot(
        daily.index,
        daily["completed"],
        color="#1976d2",
        linestyle=":",
        linewidth=1.8,
        label="TabICLv2-completed series (daily mean)",
        zorder=2,
    )
    ax.plot(
        daily.index,
        daily["observed"],
        color="black",
        linewidth=1.25,
        label="Observed hours (daily mean)",
        zorder=3,
    )

    ax.axhline(0, color="0.65", linewidth=0.7, zorder=1)
    ax.grid(axis="y", color="0.85", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlim(start.normalize(), end.normalize())
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_xlabel("Date")
    ax.set_ylabel(r"Daily mean FCH$_4$ (nmol m$^{-2}$ s$^{-1}$)")
    ax.set_title(
        f"Tower {tower_number}: observed and TabICLv2-completed FCH$_4$ "
        f"({coverage:.1f}% of hourly target observed)"
    )
    ax.legend(loc="upper left", frameon=True)

    fig.tight_layout()
    output_path = PROJECT_ROOT / "report" / "Figures" / OUTPUT_NAMES[tower_number]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    chain = pd.read_csv(CHAIN_PATH, parse_dates=["Datetime"])
    for tower_number in OUTPUT_NAMES:
        print(f"Saved {plot_tower(chain, tower_number)}")


if __name__ == "__main__":
    main()
