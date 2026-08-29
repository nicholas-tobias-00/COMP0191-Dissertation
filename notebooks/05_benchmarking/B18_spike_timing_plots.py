"""Plot when the spike observations used in the B18 evaluation occur.

The grey bands reproduce the exact population behind the published B18
``preanchor tower-specific p95`` spike metrics: an observed target, a valid
positive climatology-MAE denominator, and ``y_is_spike == 1``.  Thresholds
were estimated separately for each tower/anchor from pre-anchor observations
by ``B18_spike_models.py``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
SOURCE_PATH = RESULTS / "b18_spike_model_chains.csv"
BASELINE_PATH = RESULTS / "_today_climatology_baseline.csv"
PLOT_DATA_PATH = RESULTS / "b18_spike_timing_plot_data.csv"
MANIFEST_PATH = RESULTS / "b18_spike_timing_manifest.json"
FIGURE_DIR = RESULTS / "figures" / "b18_spike_timing"
REPORT_FIGURE_DIR = ROOT / "report" / "Figures"

EXPERIMENT_ID = "B18S04"
METHOD = "base_plus_fixed_excess_0.25"
TOWERS = (2, 4, 9)
BIN_LABELS = ("1-7", "8-30", "31-90", "91-180", "181-270", "271-365")
BIN_EDGES = (0, 7, 30, 90, 180, 270, 365)


def load_plot_data() -> pd.DataFrame:
    """Load the B18 p95 chain and identify spike days in the scored sample."""
    frame = pd.read_csv(SOURCE_PATH, parse_dates=["date"])
    frame = frame.loc[
        frame["experiment_id"].eq(EXPERIMENT_ID) & frame["method"].eq(METHOD)
    ].copy()
    frame = frame.drop_duplicates(["tower", "anchor_year", "date"])

    anchors = pd.to_datetime(frame["anchor_year"].astype(str) + "-12-16")
    frame["lead_day"] = (frame["date"] - anchors).dt.days
    frame["bin"] = pd.cut(
        frame["lead_day"], BIN_EDGES, labels=BIN_LABELS, include_lowest=True
    ).astype(str)

    baseline = pd.read_csv(BASELINE_PATH)
    frame = frame.merge(
        baseline[["tower", "anchor_year", "bin", "MAE_climatology"]],
        on=["tower", "anchor_year", "bin"],
        how="left",
        validate="many_to_one",
    )
    frame["is_evaluable"] = (
        frame["y_true"].notna()
        & frame["y_predict"].notna()
        & frame["MAE_climatology"].notna()
        & frame["MAE_climatology"].gt(0)
    )
    frame["is_spike_evaluation"] = frame["is_evaluable"] & frame["y_is_spike"].eq(1)

    if len(frame) != 5_475:
        raise ValueError(f"Expected 5,475 B18 chain rows, found {len(frame):,}")
    if set(frame["tower"].unique()) != set(TOWERS):
        raise ValueError(f"Unexpected tower set: {sorted(frame['tower'].unique())}")
    if int(frame["is_spike_evaluation"].sum()) != 130:
        raise ValueError(
            "Expected 130 spike observations from the published B18 evaluation; "
            f"found {int(frame['is_spike_evaluation'].sum())}"
        )

    keep = [
        "date",
        "tower",
        "anchor_year",
        "lead_day",
        "y_predict",
        "y_gapfilled",
        "y_true",
        "spike_threshold",
        "y_is_spike",
        "MAE_climatology",
        "is_evaluable",
        "is_spike_evaluation",
    ]
    return frame[keep].sort_values(["tower", "date"]).reset_index(drop=True)


def contiguous_spike_intervals(group: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Return inclusive runs of consecutive evaluated spike days."""
    dates = (
        group.loc[group["is_spike_evaluation"], "date"]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    if not dates:
        return []

    intervals: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    start = previous = pd.Timestamp(dates[0])
    for value in dates[1:]:
        current = pd.Timestamp(value)
        if current - previous > pd.Timedelta(days=1):
            intervals.append((start, previous))
            start = current
        previous = current
    intervals.append((start, previous))
    return intervals


def draw_tower(ax, group: pd.DataFrame, tower: int) -> None:
    group = group.sort_values("date")
    for start, end in contiguous_spike_intervals(group):
        ax.axvspan(
            start - pd.Timedelta(hours=12),
            end + pd.Timedelta(hours=12),
            color="0.75",
            alpha=0.55,
            linewidth=0,
            zorder=0,
        )

    ax.plot(
        group["date"],
        group["y_gapfilled"],
        color="black",
        linestyle=":",
        linewidth=1.0,
        alpha=0.72,
        zorder=2,
    )
    ax.plot(
        group["date"],
        group["y_predict"],
        color="black",
        linestyle="-",
        linewidth=1.15,
        zorder=3,
    )
    spike_count = int(group["is_spike_evaluation"].sum())
    ax.set_title(f"Tower {tower} ({spike_count} evaluated spike days)", loc="left")
    ax.set_ylabel(r"FCH$_4$ (nmol m$^{-2}$ s$^{-1}$)")
    ax.grid(axis="y", color="0.9", linewidth=0.6)
    ax.margins(x=0)


def legend_handles() -> list:
    return [
        Line2D([0], [0], color="black", linewidth=1.15, label="TabPFN prediction"),
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=":",
            linewidth=1.0,
            label="Gap-filled FCH4",
        ),
        Patch(
            facecolor="0.75",
            alpha=0.55,
            edgecolor="none",
            label="Identified Spike Day",
        ),
    ]


def save_combined(frame: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(len(TOWERS), 1, figsize=(15, 10), sharex=True)
    for ax, tower in zip(axes, TOWERS):
        draw_tower(ax, frame.loc[frame["tower"].eq(tower)], tower)
    axes[-1].set_xlabel("Forecast date")
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[0].legend(handles=legend_handles(), loc="upper right", frameon=True)
    fig.suptitle("Evaluated Spike Periods", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    path = FIGURE_DIR / "B18_p95_spike_timing_all_towers.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def save_individual(frame: pd.DataFrame, tower: int) -> Path:
    fig, ax = plt.subplots(figsize=(15, 4.8))
    draw_tower(ax, frame.loc[frame["tower"].eq(tower)], tower)
    ax.set_xlabel("Forecast date")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(handles=legend_handles(), loc="upper right", frameon=True)
    fig.tight_layout()
    path = FIGURE_DIR / f"B18_p95_spike_timing_T{tower}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_plot_data()
    frame.to_csv(PLOT_DATA_PATH, index=False)

    figure_paths = [save_combined(frame)]
    figure_paths.extend(save_individual(frame, tower) for tower in TOWERS)

    report_paths = []
    for source in figure_paths:
        report_name = f"ch5_{source.name}"
        destination = REPORT_FIGURE_DIR / report_name
        shutil.copyfile(source, destination)
        report_paths.append(destination)

    tower_counts = {
        f"T{tower}": int(
            frame.loc[frame["tower"].eq(tower), "is_spike_evaluation"].sum()
        )
        for tower in TOWERS
    }
    manifest = {
        "source": str(SOURCE_PATH.relative_to(ROOT)),
        "experiment_id": EXPERIMENT_ID,
        "method": METHOD,
        "forecast": "B18 TabPFN p95 + 0.25 excess",
        "spike_definition": "tower- and anchor-specific p95 estimated from pre-anchor observed targets",
        "shading_population": "y_is_spike == 1 among rows used by the published observed-target MASE evaluation",
        "spike_days_total": int(frame["is_spike_evaluation"].sum()),
        "spike_days_by_tower": tower_counts,
        "figures": [str(path.relative_to(ROOT)) for path in figure_paths],
        "report_copies": [str(path.relative_to(ROOT)) for path in report_paths],
        "plot_data": str(PLOT_DATA_PATH.relative_to(ROOT)),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Saved {len(figure_paths)} figures to {FIGURE_DIR}")
    print(f"Copied {len(report_paths)} report-ready figures to {REPORT_FIGURE_DIR}")
    print(f"Evaluated spike days: {manifest['spike_days_total']} ({tower_counts})")


if __name__ == "__main__":
    main()
