"""Re-export report-facing gap-filling UQ figures from persisted artifacts.

Called by temp_gap_filling_pipeline.ipynb. Performs no model fitting and does
not change benchmark data.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TOWERS = (2, 4, 9)
MODEL_COLORS = {"QRF_RFm": "#4C72B0", "TabICL_quantile": "#DD8452"}
SOURCE_NOTEBOOK = "temp_gap_filling_pipeline.ipynb"


def _save(fig, path: Path, dpi: int = 110) -> Path:
    fig.tight_layout()
    fig.savefig(
        path,
        dpi=dpi,
        metadata={
            "Software": SOURCE_NOTEBOOK,
            "Description": "Gap-filling UQ figure regenerated from persisted artifacts.",
        },
    )
    plt.close(fig)
    return path


def _hourly_calibration(data_dir: Path, fig_dir: Path) -> Path:
    detail = pd.read_csv(data_dir / "prediction_intervals_detail.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    coverage = detail.groupby(["tower", "model"])["covered"].mean().unstack("model")
    positions = np.arange(len(TOWERS))
    width = 0.35
    for index, model in enumerate(MODEL_COLORS):
        axes[0].bar(
            positions + (index - 0.5) * width,
            [coverage.loc[tower, model] for tower in TOWERS],
            width,
            label=model,
            color=MODEL_COLORS[model],
        )
    axes[0].axhline(0.90, color="crimson", linestyle="--", linewidth=1, label="90% nominal target")
    axes[0].set_xticks(positions, [f"Tower {tower}" for tower in TOWERS])
    axes[0].set_ylabel("Coverage")
    axes[0].set_title("Interval coverage")
    axes[0].legend(fontsize=8)

    for model, color in MODEL_COLORS.items():
        subset = detail.loc[detail["model"].eq(model)].copy()
        subset["width_decile"] = pd.qcut(subset["width"], 10, duplicates="drop")
        binned = subset.groupby("width_decile", observed=True).agg(
            mean_width=("width", "mean"),
            mean_abs_error=("abs_err", "mean"),
        )
        corr = subset["width"].corr(subset["abs_err"])
        axes[1].plot(
            binned["mean_width"],
            binned["mean_abs_error"],
            "o-",
            color=color,
            label=f"{model} (r={corr:+.2f})",
        )
    axes[1].set_xlabel("Mean interval width")
    axes[1].set_ylabel("Mean absolute error")
    axes[1].set_title("Does wider UQ indicate larger error?")
    axes[1].legend(fontsize=8)
    fig.suptitle("Hourly gap-filling interval calibration and reliability")
    return _save(fig, fig_dir / "calibration_reliability.png")


def _hourly_production(data_dir: Path, fig_dir: Path) -> list[Path]:
    data = pd.read_csv(
        data_dir / "fch4_gapfilled_with_intervals_calibrated.csv",
        parse_dates=["Datetime"],
    ).sort_values("Datetime")
    outputs = []
    panels = [
        ("QRF_RFm", "qrf_q05", "qrf_q95", "qrf_q05_cal", "qrf_q95_cal", "qrf_q50"),
        (
            "TabICL_quantile",
            "tabicl_q05",
            "tabicl_q95",
            "tabicl_q05_cal",
            "tabicl_q95_cal",
            "tabicl_q50",
        ),
    ]
    for tower in TOWERS:
        subset = data.loc[data["tower"].eq(tower)].set_index("Datetime")
        window = subset.loc[subset.index.max() - pd.Timedelta(days=180) :]
        fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
        for ax, (model, raw_lo, raw_hi, cal_lo, cal_hi, median) in zip(axes, panels):
            color = MODEL_COLORS[model]
            ax.fill_between(
                window.index,
                window[raw_lo],
                window[raw_hi],
                color=color,
                alpha=0.12,
                hatch="//",
                linestyle="--",
                edgecolor=color,
                label="90% raw interval",
            )
            ax.fill_between(
                window.index,
                window[cal_lo],
                window[cal_hi],
                color=color,
                alpha=0.25,
                label="90% gap-length-calibrated interval",
            )
            ax.plot(window.index, window[median], color=color, linewidth=1, label=f"{model} median")
            ax.plot(window.index, window["y_observed"], color="black", linewidth=1, label="Observed FCH4")
            ax.set_ylabel("FCH4 (nmol m-2 s-1)")
            ax.set_title(model)
            ax.legend(fontsize=7, loc="upper right")
        fig.suptitle(
            f"Tower {tower}: production gap-filled series with raw and calibrated UQ "
            f"({window.index.min().date()} to {window.index.max().date()})"
        )
        outputs.append(_save(fig, fig_dir / f"production_interval_T{tower}.png"))
    return outputs


def _hourly_tabicl_calibrated_examples(
    data_dir: Path,
    fig_dir: Path,
    report_dir: Path | None = None,
) -> list[Path]:
    """Three-month production views for the earlier conformally calibrated TabICL UQ model."""
    data = pd.read_csv(
        data_dir / "fch4_gapfilled_with_intervals_calibrated.csv",
        parse_dates=["Datetime"],
    ).sort_values("Datetime")
    outputs = []
    for tower in TOWERS:
        tower_data = data.loc[data["tower"].eq(tower)].copy()
        end = tower_data["Datetime"].max()
        window = tower_data.loc[tower_data["Datetime"].ge(end - pd.DateOffset(months=3))].copy()
        is_gap = window["y_observed"].isna().to_numpy()
        median = np.where(is_gap, window["tabicl_q50"], np.nan)
        lower = np.where(is_gap, window["tabicl_q05_cal"], np.nan)
        upper = np.where(is_gap, window["tabicl_q95_cal"], np.nan)

        fig, ax = plt.subplots(figsize=(15, 5.2))
        ax.fill_between(
            window["Datetime"],
            lower,
            upper,
            color="#357ABD",
            alpha=0.20,
            linewidth=0,
            label="TabICL conformal 90% interval",
        )
        ax.plot(
            window["Datetime"],
            median,
            color="#0066CC",
            linestyle=":",
            linewidth=1.0,
            label="TabICL gap-filled median",
        )
        ax.plot(
            window["Datetime"],
            window["y_observed"],
            color="black",
            linewidth=0.9,
            label="Observed FCH4",
        )
        coverage = 100 * window["y_observed"].notna().mean()
        ax.set_title(
            f"Tower {tower}: latest three months of hourly FCH4 "
            f"({coverage:.1f}% observed)"
        )
        ax.set_xlabel("Date")
        ax.set_ylabel(r"FCH4 (nmol m$^{-2}$ s$^{-1}$)")
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.grid(True, alpha=0.2)
        ax.legend(loc="upper right")

        output = fig_dir / f"calibrated_tabicl_uq_hourly_T{tower}_3month_example.png"
        outputs.append(_save(fig, output, dpi=180))
        if report_dir is not None:
            report_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(
                output,
                report_dir / f"ch4_calibrated_tabicl_uq_hourly_T{tower}_3month_example.png",
            )
    return outputs


def _daily_calibration(data_dir: Path, fig_dir: Path) -> Path:
    summary = pd.read_csv(data_dir / "daily_conformal_calibration_summary.csv")
    positions = np.arange(len(TOWERS))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    conformal = [
        summary.loc[summary["tower"].eq(tower), "PICP_test_conformal"].iloc[0]
        for tower in TOWERS
    ]
    naive = [
        summary.loc[summary["tower"].eq(tower), "PICP_test_naive_normal"].iloc[0]
        for tower in TOWERS
    ]
    ax.bar(positions - width / 2, conformal, width, label="Split-conformal", color="#4C72B0")
    ax.bar(positions + width / 2, naive, width, label="Normal approximation", color="#888888")
    ax.axhline(0.90, color="crimson", linestyle="--", linewidth=1, label="90% nominal target")
    ax.set_xticks(positions, [f"Tower {tower}" for tower in TOWERS])
    ax.set_ylabel("Held-out daily PICP")
    ax.set_title("Daily gap-filling interval calibration")
    ax.legend(fontsize=8)
    return _save(fig, fig_dir / "daily_calibration_reliability.png")


def _daily_production(data_dir: Path, fig_dir: Path) -> Path:
    data = pd.read_csv(
        data_dir / "fch4_daily_gapfilled_with_intervals.csv",
        parse_dates=["Datetime"],
    )
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=False)
    for ax, tower in zip(axes, TOWERS):
        subset = data.loc[data["tower"].eq(tower)].set_index("Datetime").sort_index()
        ax.fill_between(
            subset.index,
            subset["daily_q05_cal"],
            subset["daily_q95_cal"],
            color="#4C72B0",
            alpha=0.2,
            label="90% daily conformal interval",
        )
        ax.plot(
            subset.index,
            subset["y_gapfilled_daily_mean"],
            color="#4C72B0",
            linewidth=0.8,
            label="Gap-filled FCH4 (daily mean)",
        )
        ax.plot(
            subset.index,
            subset["y_observed_daily_mean"],
            color="black",
            linewidth=0.9,
            label="Observed FCH4 (daily mean)",
        )
        ax.set_title(f"Tower {tower}")
        ax.set_ylabel("FCH4 (nmol m-2 s-1)")
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("Daily gap-filled FCH4 with directly calibrated 90% prediction interval")
    return _save(fig, fig_dir / "daily_production_intervals.png")


def export_gapfilling_uq_figures(
    data_dir: str | Path = "_data",
    fig_dir: str | Path = "_figures",
    report_dir: str | Path | None = None,
) -> list[Path]:
    """Regenerate the six report-facing validation and production UQ figures."""
    data_dir = Path(data_dir)
    fig_dir = Path(fig_dir)
    report_dir = Path(report_dir) if report_dir is not None else None
    fig_dir.mkdir(parents=True, exist_ok=True)
    outputs = [_hourly_calibration(data_dir, fig_dir)]
    outputs.extend(_hourly_production(data_dir, fig_dir))
    outputs.extend(_hourly_tabicl_calibrated_examples(data_dir, fig_dir, report_dir))
    outputs.append(_daily_calibration(data_dir, fig_dir))
    outputs.append(_daily_production(data_dir, fig_dir))
    manifest = {
        "source_notebook": SOURCE_NOTEBOOK,
        "model_refit": False,
        "inputs": [
            "prediction_intervals_detail.csv",
            "fch4_gapfilled_with_intervals_calibrated.csv",
            "daily_conformal_calibration_summary.csv",
            "fch4_daily_gapfilled_with_intervals.csv",
        ],
        "figures": [path.name for path in outputs],
        "scope_note": (
            "Production RFm/QRF and earlier TabICL UQ configuration; these intervals are not "
            "attributed to the later TabICL-solo accuracy champion."
        ),
    }
    (fig_dir / "gapfilling_uq_figure_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return outputs


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    paths = export_gapfilling_uq_figures(
        data_dir=Path(__file__).with_name("_data"),
        fig_dir=Path(__file__).with_name("_figures"),
        report_dir=project_root / "report" / "Figures",
    )
    print("\n".join(f"Saved {path}" for path in paths))
