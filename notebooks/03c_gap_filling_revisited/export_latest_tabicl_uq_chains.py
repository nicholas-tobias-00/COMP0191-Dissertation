"""Fit the latest TabICL-solo gap-filler and export six-month hourly UQ chains.

This is the report-facing companion to section 19.2 of
temp_gap_filling_pipeline.ipynb. It deliberately refits the exact D5.5
TabICL-solo configuration because the older persisted TabICL interval artifact
belongs to a different model configuration.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


TOWERS = (2, 4, 9)
FEATURES = [
    "SWIN_1_1_1",
    "TA_0_0_1",
    "VPD_0_0_1",
    "PPFD_1_1_1",
    "RN_1_1_1",
    "WS_0_0_1",
    "USTAR_0_0_1",
    "SHF_1_1_1",
    "Precipitation (mm)",
    "Soil Temperature @ 15cm Depth (oC)",
    "Soil Moisture @ 10cm Depth (%)",
    "fc",
    "_hs",
    "_hc",
    "_ds",
    "_dc",
    "lsu_dens",
    "graze",
    "swc_l168",
    "swc_l336",
    "swc_l504",
    "swc_l672",
    "ts_l168",
    "ts_l336",
    "ts_l504",
    "ts_l672",
    "mgmt_cut",
    "mgmt_manure",
    "gpp",
    "reco",
]
ROW_CAP = 10_000
RANDOM_STATE = 42
QUANTILE_LEVELS = (0.05, 0.50, 0.95)
SOURCE_NOTEBOOK = "temp_gap_filling_pipeline.ipynb"


def _fit_and_predict(tower_data: pd.DataFrame, window: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    from tabicl import TabICLRegressor

    training = tower_data.loc[tower_data["target"].notna()]
    training = training.sample(n=min(ROW_CAP, len(training)), random_state=RANDOM_STATE)

    imputer = SimpleImputer(strategy="mean")
    x_train = imputer.fit_transform(training[FEATURES].to_numpy())
    x_window = imputer.transform(window[FEATURES].to_numpy())

    model = TabICLRegressor(random_state=RANDOM_STATE)
    model.fit(x_train, training["target"].to_numpy())
    predictions = model.predict(
        x_window,
        output_type=["mean", "quantiles"],
        alphas=list(QUANTILE_LEVELS),
    )
    mean = np.asarray(predictions["mean"], dtype=float).reshape(-1)
    quantiles = np.asarray(predictions["quantiles"], dtype=float)
    if quantiles.shape != (len(window), len(QUANTILE_LEVELS)):
        raise RuntimeError(f"Unexpected TabICL quantile shape: {quantiles.shape}")
    if not np.all((quantiles[:, 0] <= quantiles[:, 1]) & (quantiles[:, 1] <= quantiles[:, 2])):
        raise RuntimeError("TabICL returned crossed native quantiles")
    return mean, quantiles


def _plot_tower(
    chain: pd.DataFrame,
    output_path: Path,
    window_label: str = "latest six months",
) -> None:
    observed = chain["y_observed"].to_numpy(dtype=float)
    missing = chain["is_gap"].to_numpy(dtype=bool)
    gap_mean = np.where(missing, chain["y_predict_mean"], np.nan)
    gap_q05 = np.where(missing, chain["q05"], np.nan)
    gap_q95 = np.where(missing, chain["q95"], np.nan)

    fig, ax = plt.subplots(figsize=(15, 5.2))
    ax.fill_between(
        chain["Datetime"],
        gap_q05,
        gap_q95,
        color="#357ABD",
        alpha=0.20,
        linewidth=0,
        label="TabICL native 90% interval (uncalibrated)",
    )
    ax.plot(
        chain["Datetime"],
        gap_mean,
        color="#0066CC",
        linestyle=":",
        linewidth=1.0,
        label="TabICL gap-filled mean",
    )
    ax.plot(
        chain["Datetime"],
        observed,
        color="black",
        linewidth=0.9,
        label="Observed FCH4",
    )

    tower = int(chain["tower"].iloc[0])
    coverage = 100 * chain["y_observed"].notna().mean()
    ax.set_title(
        f"Tower {tower}: {window_label} of hourly FCH4 "
        f"({coverage:.1f}% observed)"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel(r"FCH4 (nmol m$^{-2}$ s$^{-1}$)")
    if window_label == "latest one month":
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    else:
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.grid(axis="y", alpha=0.18)
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    fig.autofmt_xdate(rotation=0, ha="center")
    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=180,
        metadata={
            "Software": SOURCE_NOTEBOOK,
            "Description": (
                "Hourly observed and latest TabICL-solo gap-filled FCH4 with raw native "
                "5th-95th percentile uncertainty over the latest six-month tower domain."
            ),
        },
    )
    plt.close(fig)


def export_one_month_from_saved_chains(
    data_dir: str | Path = "_data",
    fig_dir: str | Path = "_figures",
    report_dir: str | Path | None = None,
) -> list[Path]:
    """Export one-month views from the persisted, provenance-matched six-month chains."""
    data_dir = Path(data_dir)
    fig_dir = Path(fig_dir)
    report_dir = Path(report_dir) if report_dir is not None else None
    chain_path = data_dir / "latest_tabicl_uq_6month_chains.csv"
    chains = pd.read_csv(chain_path, parse_dates=["Datetime"])
    output_paths = []
    metadata = {}

    for tower in TOWERS:
        tower_chain = chains.loc[chains["tower"].eq(tower)].sort_values("Datetime")
        example_end = tower_chain["Datetime"].max()
        example_start = example_end - pd.DateOffset(months=1)
        example = tower_chain.loc[tower_chain["Datetime"].ge(example_start)].copy()
        output_path = fig_dir / f"latest_tabicl_uq_hourly_T{tower}_1month_example.png"
        _plot_tower(example, output_path, window_label="latest one month")
        output_paths.append(output_path)
        if report_dir is not None:
            report_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(
                output_path,
                report_dir / f"ch4_latest_tabicl_uq_hourly_T{tower}_1month_example.png",
            )
        metadata[str(tower)] = {
            "window_start": str(example["Datetime"].min()),
            "window_end": str(example["Datetime"].max()),
            "hourly_rows": int(len(example)),
            "observed_coverage": float(example["y_observed"].notna().mean()),
            "source": chain_path.name,
        }

    manifest_path = fig_dir / "latest_tabicl_uq_hourly_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["one_month_examples"] = metadata
        report_copies = manifest.setdefault("report_copies", [])
        for tower in TOWERS:
            name = f"ch4_latest_tabicl_uq_hourly_T{tower}_1month_example.png"
            if report_dir is not None and name not in report_copies:
                report_copies.append(name)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output_paths


def export_latest_tabicl_uq_chains(
    data_dir: str | Path = "_data",
    fig_dir: str | Path = "_figures",
    report_dir: str | Path | None = None,
) -> list[Path]:
    """Refit the latest TabICL-solo model and export one hourly chart per tower."""
    data_dir = Path(data_dir)
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    if report_dir is not None:
        report_dir = Path(report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)

    feature_frame = pd.read_csv(data_dir / "feature_frame.csv", parse_dates=["Datetime"])
    missing_columns = sorted(set(FEATURES + ["Datetime", "target", "tower"]) - set(feature_frame.columns))
    if missing_columns:
        raise KeyError(f"feature_frame.csv is missing required columns: {missing_columns}")

    chains = []
    output_paths = []
    tower_metadata = {}
    for tower in TOWERS:
        tower_data = feature_frame.loc[feature_frame["tower"].eq(tower)].sort_values("Datetime")
        window_end = tower_data["Datetime"].max()
        window_start = window_end - pd.DateOffset(months=6)
        window = tower_data.loc[tower_data["Datetime"].ge(window_start)].copy()
        training_rows = int(tower_data["target"].notna().sum())

        print(
            f"Tower {tower}: fitting latest TabICL-solo on "
            f"{min(ROW_CAP, training_rows):,}/{training_rows:,} observed rows; "
            f"predicting {len(window):,} hourly rows...",
            flush=True,
        )
        mean, quantiles = _fit_and_predict(tower_data, window)
        chain = pd.DataFrame(
            {
                "Datetime": window["Datetime"].to_numpy(),
                "tower": tower,
                "y_observed": window["target"].to_numpy(dtype=float),
                "is_gap": window["target"].isna().to_numpy(),
                "y_predict_mean": mean,
                "q05": quantiles[:, 0],
                "q50": quantiles[:, 1],
                "q95": quantiles[:, 2],
            }
        )
        chain["y_gapfilled"] = chain["y_observed"].fillna(chain["y_predict_mean"])
        chain["model"] = "TabICL-solo"
        chain["config"] = "FEATURES"
        chains.append(chain)

        output_path = fig_dir / f"latest_tabicl_uq_hourly_T{tower}.png"
        _plot_tower(chain, output_path)
        output_paths.append(output_path)
        if report_dir is not None:
            shutil.copy2(output_path, report_dir / f"ch4_latest_tabicl_uq_hourly_T{tower}.png")

        gap_width = chain.loc[chain["is_gap"], "q95"] - chain.loc[chain["is_gap"], "q05"]
        tower_metadata[str(tower)] = {
            "window_start": str(window["Datetime"].min()),
            "window_end": str(window_end),
            "hourly_rows": int(len(window)),
            "observed_rows": int(window["target"].notna().sum()),
            "gap_rows": int(window["target"].isna().sum()),
            "observed_coverage": float(window["target"].notna().mean()),
            "training_rows_available": training_rows,
            "training_rows_used": min(ROW_CAP, training_rows),
            "median_native_interval_width_at_gaps": float(gap_width.median()),
        }

    three_month_metadata = {}
    for tower, tower_chain in zip(TOWERS, chains):
        example_end = tower_chain["Datetime"].max()
        example_start = example_end - pd.DateOffset(months=3)
        example = tower_chain.loc[tower_chain["Datetime"].ge(example_start)].copy()
        example_path = fig_dir / f"latest_tabicl_uq_hourly_T{tower}_3month_example.png"
        _plot_tower(example, example_path, window_label="latest three months")
        output_paths.append(example_path)
        if report_dir is not None:
            shutil.copy2(
                example_path,
                report_dir / f"ch4_latest_tabicl_uq_hourly_T{tower}_3month_example.png",
            )
        three_month_metadata[str(tower)] = {
            "window_start": str(example["Datetime"].min()),
            "window_end": str(example["Datetime"].max()),
            "hourly_rows": int(len(example)),
            "observed_coverage": float(example["y_observed"].notna().mean()),
            "source": "latest_tabicl_uq_6month_chains.csv",
        }

    one_month_metadata = {}
    for tower, tower_chain in zip(TOWERS, chains):
        example_end = tower_chain["Datetime"].max()
        example_start = example_end - pd.DateOffset(months=1)
        example = tower_chain.loc[tower_chain["Datetime"].ge(example_start)].copy()
        example_path = fig_dir / f"latest_tabicl_uq_hourly_T{tower}_1month_example.png"
        _plot_tower(example, example_path, window_label="latest one month")
        output_paths.append(example_path)
        if report_dir is not None:
            shutil.copy2(
                example_path,
                report_dir / f"ch4_latest_tabicl_uq_hourly_T{tower}_1month_example.png",
            )
        one_month_metadata[str(tower)] = {
            "window_start": str(example["Datetime"].min()),
            "window_end": str(example["Datetime"].max()),
            "hourly_rows": int(len(example)),
            "observed_coverage": float(example["y_observed"].notna().mean()),
            "source": "latest_tabicl_uq_6month_chains.csv",
        }

    chain_path = data_dir / "latest_tabicl_uq_6month_chains.csv"
    pd.concat(chains, ignore_index=True).to_csv(chain_path, index=False)
    manifest = {
        "source_notebook": SOURCE_NOTEBOOK,
        "input": "feature_frame.csv",
        "raw_chain_output": chain_path.name,
        "model": "TabICLRegressor(random_state=42)",
        "configuration": "D5.5 latest TabICL-solo with champion FEATURES",
        "training": "per tower; all real observations; fixed random sample capped at 10,000 rows",
        "point_prediction": "TabICL native mean",
        "interval": "TabICL native 5th-95th percentiles; raw and uncalibrated",
        "window": "latest six calendar months available independently for each tower",
        "figures": [path.name for path in output_paths],
        "report_copies": (
            [f"ch4_latest_tabicl_uq_hourly_T{tower}.png" for tower in TOWERS]
            + [
                f"ch4_latest_tabicl_uq_hourly_T{tower}_3month_example.png"
                for tower in TOWERS
            ]
            + [
                f"ch4_latest_tabicl_uq_hourly_T{tower}_1month_example.png"
                for tower in TOWERS
            ]
            if report_dir is not None
            else []
        ),
        "three_month_examples": three_month_metadata,
        "one_month_examples": one_month_metadata,
        "towers": tower_metadata,
    }
    (fig_dir / "latest_tabicl_uq_hourly_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return output_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--one-month-from-saved",
        action="store_true",
        help="Create only the one-month views from the saved six-month raw chains.",
    )
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[2]
    kwargs = {
        "data_dir": Path(__file__).with_name("_data"),
        "fig_dir": Path(__file__).with_name("_figures"),
        "report_dir": project_root / "report" / "Figures",
    }
    paths = (
        export_one_month_from_saved_chains(**kwargs)
        if args.one_month_from_saved
        else export_latest_tabicl_uq_chains(**kwargs)
    )
    print("\n".join(f"Saved {path}" for path in paths))
