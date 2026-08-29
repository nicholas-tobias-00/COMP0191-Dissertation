"""Export report-ready FCH4 time-series diagnostics from QC-observed data only.

This intentionally does not use ``fch4_gapfilled.csv``.  That file is a useful
forecasting target/AR source, but extends across timestamps with no corresponding
EC observations (notably after the end of each tower's record).  It is therefore
not appropriate for descriptive ACF/PACF or STL figures in the report.

Outputs are additive:
  * results/figures/tsa_observed_diagnostics/
  * results/tsa_observed_diagnostics/observed_tsa_summary.csv
  * report/Figures/ch3_*_observed.png (staged replacements for Appendix A.5/A.6)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import pacf


TOWERS = (2, 4, 9)
COLORS = {2: "#4C72B0", 4: "#DD8452", 9: "#55A868"}
FCH4_LOW, FCH4_HIGH = -500.0, 3000.0
MIN_HOURS_PER_DAY = 6
MAX_HOURLY_LAG = 168


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def qc_observed_fch4(raw: pd.DataFrame, tower: int) -> pd.Series:
    """Apply the project's SSITC {0,1} and plausibility conventions."""
    value = raw[f"FCH4_1_1_1 [Tower {tower}]"] .copy()
    flag = raw[f"FCH4_SSITC_TEST_1_1_1 [Tower {tower}]"]
    value = value.where(flag.isin([0, 1]))
    return value.where(value.between(FCH4_LOW, FCH4_HIGH))


def longest_finite_segment(series: pd.Series) -> pd.Series:
    """Return the longest uninterrupted non-missing segment."""
    finite = series.notna()
    if not finite.any():
        return pd.Series(dtype=float)
    run_id = finite.ne(finite.shift(fill_value=False)).cumsum()
    segments = [part for _, part in series.loc[finite].groupby(run_id.loc[finite])]
    return max(segments, key=len)


def pairwise_acf(series: pd.Series, nlags: int) -> pd.DataFrame:
    """ACF at true hourly lags, using only observed pairs at each lag."""
    values = series.to_numpy(dtype=float)
    rows = []
    for lag in range(nlags + 1):
        left = values if lag == 0 else values[:-lag]
        right = values if lag == 0 else values[lag:]
        valid = np.isfinite(left) & np.isfinite(right)
        correlation = 1.0 if lag == 0 else np.nan
        if lag and valid.sum() >= 3:
            correlation = float(np.corrcoef(left[valid], right[valid])[0, 1])
        rows.append({"lag_hours": lag, "acf": correlation, "n_pairs": int(valid.sum())})
    return pd.DataFrame(rows).set_index("lag_hours")


def load_observed(root: Path) -> dict[int, pd.Series]:
    path = root / "data" / "Hourly" / "consolidated_hourly.csv"
    raw = pd.read_csv(path, low_memory=False)
    raw["Datetime"] = pd.to_datetime(raw["Datetime"], format="mixed")
    raw = raw.set_index("Datetime")
    return {tower: qc_observed_fch4(raw, tower) for tower in TOWERS}


def prepare_daily_segments(observed: dict[int, pd.Series]) -> dict[int, pd.Series]:
    """Create STL/PACF inputs without filling long target gaps.

    A daily mean needs at least six observed hours.  At most two *internal* missing
    days are linearly interpolated so that a short outage does not split an otherwise
    usable segment.  The longest resulting continuous segment is used per tower.
    """
    segments = {}
    for tower, hourly in observed.items():
        daily = hourly.resample("D").agg(["mean", "count"])
        adequate = daily["mean"].where(daily["count"] >= MIN_HOURS_PER_DAY)
        limited = adequate.interpolate(limit=2, limit_area="inside")
        segment = longest_finite_segment(limited)
        if len(segment) < 21:
            raise ValueError(f"Tower {tower} has fewer than 21 days for weekly STL.")
        segments[tower] = segment
    return segments


def plot_acf_pacf(
    observed: dict[int, pd.Series], segments: dict[int, pd.Series], output: Path
) -> tuple[dict[int, pd.DataFrame], dict[int, np.ndarray]]:
    fig, axes = plt.subplots(len(TOWERS), 2, figsize=(13, 10), constrained_layout=True)
    hourly_acfs, daily_pacfs = {}, {}

    for row, tower in enumerate(TOWERS):
        acf_frame = pairwise_acf(observed[tower], MAX_HOURLY_LAG)
        hourly_acfs[tower] = acf_frame
        ax = axes[row, 0]
        ax.vlines(acf_frame.index, 0, acf_frame["acf"], color=COLORS[tower], linewidth=0.8)
        ax.scatter(acf_frame.index, acf_frame["acf"], color=COLORS[tower], s=7)
        bound = 1.96 / np.sqrt(acf_frame.loc[1, "n_pairs"])
        ax.axhline(0, color="0.25", linewidth=0.7)
        ax.axhspan(-bound, bound, color="0.85", alpha=0.7, zorder=0)
        for lag in (24, 168):
            ax.axvline(lag, color="0.45", linestyle=":", linewidth=0.8)
        ax.set_xlim(0, MAX_HOURLY_LAG)
        ax.set_ylim(-0.2, 1.02)
        ax.set_title(f"Tower {tower}: hourly ACF")
        ax.set_ylabel("Correlation")
        if row == len(TOWERS) - 1:
            ax.set_xlabel("Lag (hours)")

        segment = segments[tower]
        nlags = min(30, len(segment) // 2 - 1)
        values = pacf(segment, nlags=nlags, method="ywm")
        daily_pacfs[tower] = values
        ax = axes[row, 1]
        lags = np.arange(len(values))
        bound = 1.96 / np.sqrt(len(segment))
        ax.vlines(lags, 0, values, color=COLORS[tower], linewidth=1.0)
        ax.scatter(lags, values, color=COLORS[tower], s=12)
        ax.axhline(0, color="0.25", linewidth=0.7)
        ax.axhspan(-bound, bound, color="0.85", alpha=0.7, zorder=0)
        ax.set_ylim(-0.4, 1.02)
        ax.set_title(f"Tower {tower}: daily PACF")
        ax.set_ylabel("Partial correlation")
        if row == len(TOWERS) - 1:
            ax.set_xlabel("Lag (days)")

    fig.suptitle("Observed FCH4 temporal dependence: hourly and daily", fontsize=13)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return hourly_acfs, daily_pacfs


def plot_stl(segments: dict[int, pd.Series], output: Path) -> dict[int, object]:
    fig, axes = plt.subplots(4, len(TOWERS), figsize=(13, 10), sharex="col", constrained_layout=True)
    results = {}
    for column, tower in enumerate(TOWERS):
        result = STL(segments[tower], period=7, robust=True).fit()
        results[tower] = result
        axes[0, column].plot(result.observed, color=COLORS[tower], linewidth=0.8)
        axes[1, column].plot(result.trend, color="black", linewidth=0.9)
        axes[2, column].plot(result.seasonal, color="#8172B2", linewidth=0.8)
        axes[3, column].plot(result.resid, color="0.35", linewidth=0.7)
        axes[0, column].set_title(
            f"Tower {tower}: {len(segments[tower])} days\n"
            f"{segments[tower].index.min().date()} to {segments[tower].index.max().date()}"
        )
        locator = mdates.MonthLocator(interval=2)
        axes[3, column].xaxis.set_major_locator(locator)
        axes[3, column].xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
        axes[3, column].set_xlabel("Date")

    for axis, label in zip(axes[:, 0], ("Observed", "Trend", "Weekly", "Residual")):
        axis.set_ylabel(label)
    fig.suptitle("STL decomposition of FCH4", fontsize=13)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return results


def write_summary(
    observed: dict[int, pd.Series],
    segments: dict[int, pd.Series],
    hourly_acfs: dict[int, pd.DataFrame],
    daily_pacfs: dict[int, np.ndarray],
    output: Path,
) -> pd.DataFrame:
    rows = []
    for tower in TOWERS:
        series = observed[tower]
        valid = series.dropna()
        row = {
            "tower": tower,
            "first_observed": valid.index.min(),
            "last_observed": valid.index.max(),
            "n_observed_hours": len(valid),
            "daily_segment_start": segments[tower].index.min(),
            "daily_segment_end": segments[tower].index.max(),
            "daily_segment_days": len(segments[tower]),
            "pacf_lag_1_day": daily_pacfs[tower][1] if len(daily_pacfs[tower]) > 1 else np.nan,
            "pacf_lag_7_days": daily_pacfs[tower][7] if len(daily_pacfs[tower]) > 7 else np.nan,
        }
        for lag in (1, 6, 24, 168):
            row[f"acf_lag_{lag}_hours"] = hourly_acfs[tower].loc[lag, "acf"]
            row[f"n_pairs_lag_{lag}_hours"] = hourly_acfs[tower].loc[lag, "n_pairs"]
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(output, index=False)
    return table


def main() -> pd.DataFrame:
    root = repository_root()
    figure_dir = root / "results" / "figures" / "tsa_observed_diagnostics"
    result_dir = root / "results" / "tsa_observed_diagnostics"
    report_dir = root / "report" / "Figures"
    for directory in (figure_dir, result_dir, report_dir):
        directory.mkdir(parents=True, exist_ok=True)

    observed = load_observed(root)
    segments = prepare_daily_segments(observed)
    acf_path = figure_dir / "tsa_acf_pacf_observed.png"
    stl_path = figure_dir / "tsa_stl_observed_daily.png"
    hourly_acfs, daily_pacfs = plot_acf_pacf(observed, segments, acf_path)
    plot_stl(segments, stl_path)
    summary = write_summary(
        observed, segments, hourly_acfs, daily_pacfs,
        result_dir / "observed_tsa_summary.csv",
    )

    # These are additive report-ready copies.  The .tex file remains untouched.
    (report_dir / "ch3_acf_pacf_observed.png").write_bytes(acf_path.read_bytes())
    (report_dir / "ch3_stl_decomposition_observed.png").write_bytes(stl_path.read_bytes())
    print("Wrote:")
    for path in (acf_path, stl_path, result_dir / "observed_tsa_summary.csv",
                 report_dir / "ch3_acf_pacf_observed.png",
                 report_dir / "ch3_stl_decomposition_observed.png"):
        print(f"  {path.relative_to(root)}")
    return summary


if __name__ == "__main__":
    print(main().round(3).to_string(index=False))
