"""Export a report-ready B18 uncertainty gallery for every tower and forecast origin.

The existing representative CQR figures obtain their margins from the evaluation-summary CSV.
That table deliberately omits a test lead-time bin when fewer than three observed targets are
available, which can leave an artificial hole in the plotted interval even though the held-out
calibration origins contain enough residuals to construct the interval.  This exporter computes
the leave-one-origin-out CQR margins directly from ``u08_chains.csv`` and therefore keeps interval
construction separate from whether the test origin can be scored.

Towers 4 and 9 use CQR intervals [q05 - margin, q95 + margin].  Tower 2 remains explicitly raw and
uncalibrated because its leave-one-origin-out calibration design is degenerate.  Existing report
figures and TeX references are not modified.

Run from the project root:
    python notebooks/06_interpretability_uq/u06c_report_cqr_gallery.py
"""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from models.recursive_rollout import conformal_margins_by_bin, lead_time_bin


DAILY_PATH = ROOT / "data" / "Hourly" / "forecast_daily_v3.csv"
CHAINS_PATH = ROOT / "results" / "u08_chains.csv"
OUTPUT_DIR = ROOT / "report" / "Figures" / "cqr_gallery"

TOWERS = (2, 4, 9)
ANCHOR_YEARS = (2018, 2019, 2020, 2021, 2022)
BINS = ((1, 7), (8, 30), (31, 90), (91, 180), (181, 270), (271, 365))
MODEL = "B18_TabPFN_champion"
ALPHA = 0.10
PRE_ANCHOR_DAYS = 30


def calibration_details(chains: pd.DataFrame, tower: int, holdout_year: int):
    """Return per-bin CQR margins and calibration counts from all other origins."""
    calibration = chains[
        (chains["model"] == MODEL)
        & (chains["eval_tower"] == tower)
        & (chains["anchor_year"] != holdout_year)
    ].copy()

    scores_by_bin = {f"{lo}-{hi}": [] for lo, hi in BINS}
    for calibration_year, rows in calibration.groupby("anchor_year"):
        anchor = pd.Timestamp(f"{int(calibration_year)}-12-16")
        labels = lead_time_bin(rows["date"].values, anchor, BINS)
        scores = np.maximum(
            rows["q05"].to_numpy() - rows["y_true"].to_numpy(),
            rows["y_true"].to_numpy() - rows["q95"].to_numpy(),
        )
        for label in scores_by_bin:
            values = scores[labels == label]
            scores_by_bin[label].extend(values[np.isfinite(values)].tolist())

    margins = conformal_margins_by_bin(scores_by_bin, alpha=ALPHA)
    counts = {label: len(values) for label, values in scores_by_bin.items()}
    return margins, counts


def plot_one(
    daily_tower: pd.DataFrame,
    chain: pd.DataFrame,
    margins: dict[str, float],
    tower: int,
    anchor_year: int,
):
    anchor = pd.Timestamp(f"{anchor_year}-12-16")
    chain = chain.sort_values("date").set_index("date")
    forecast_dates = chain.index
    display_dates = pd.date_range(
        anchor - pd.Timedelta(days=PRE_ANCHOR_DAYS), forecast_dates.max(), freq="D"
    )

    calibrated = tower != 2
    labels = lead_time_bin(forecast_dates.values, anchor, BINS)
    margin = pd.Series([margins.get(label, np.nan) for label in labels], index=forecast_dates)

    fig, ax = plt.subplots(figsize=(13, 5))
    gapfilled = daily_tower["y_gapfilled"].reindex(display_dates)
    observed = daily_tower["y_observed"].reindex(display_dates)
    ax.plot(gapfilled.index, gapfilled, ":", color="gray", linewidth=1, label="Gap-filled FCH4")
    ax.plot(observed.index, observed, "-", color="black", linewidth=1, label="Actual FCH4 (observed)")
    ax.plot(
        forecast_dates,
        chain["median"],
        "-",
        color="tab:green",
        linewidth=1.5,
        label="Prediction (median)",
    )

    if calibrated:
        lo = chain["q05"] - margin
        hi = chain["q95"] + margin
        # Very small calibration samples can yield a negative CQR margin large enough to invert
        # the bounds.  An empty interval is not useful in a report figure, so show the native raw
        # interval on those isolated days and retain CQR everywhere its bounds remain valid.
        calibrated_mask = margin.notna() & (lo <= hi)
        ax.fill_between(
            forecast_dates,
            lo,
            hi,
            where=calibrated_mask,
            color="tab:green",
            alpha=0.25,
            label="90% CQR-calibrated interval [q05-margin, q95+margin]",
        )

        raw_mask = ~calibrated_mask
        if raw_mask.any():
            ax.fill_between(
                forecast_dates,
                chain["q05"],
                chain["q95"],
                where=raw_mask,
                color="tab:green",
                alpha=0.12,
                hatch="//",
                linestyle="--",
                edgecolor="tab:green",
                label="90% raw interval where calibration is unavailable/degenerate",
            )
        title_prefix = "Prediction and Calibrated UQ (q05, q95)"
    else:
        ax.fill_between(
            forecast_dates,
            chain["q05"],
            chain["q95"],
            color="tab:green",
            alpha=0.15,
            hatch="//",
            linestyle="--",
            edgecolor="tab:green",
            label="90% raw interval [q05, q95] -- uncalibrated",
        )
        title_prefix = "Prediction and Raw UQ (q05, q95)"

    ax.set_title(
        f"{title_prefix} around gap-filled target - T{tower:02d} - {anchor_year}"
    )
    ax.set_ylabel("FCH4 (nmol m-2 s-1)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    output = OUTPUT_DIR / f"ch5_cqr_forecast_T{tower:02d}_{anchor_year}.png"
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    daily = pd.read_csv(DAILY_PATH, low_memory=False)
    daily["Datetime"] = pd.to_datetime(daily["Datetime"], format="mixed")
    daily_by_tower = {
        tower: daily[daily["tower"] == tower].set_index("Datetime").sort_index()
        for tower in TOWERS
    }

    chains = pd.read_csv(CHAINS_PATH, parse_dates=["date"])
    manifest_rows = []
    for tower in TOWERS:
        for anchor_year in ANCHOR_YEARS:
            chain = chains[
                (chains["model"] == MODEL)
                & (chains["eval_tower"] == tower)
                & (chains["anchor_year"] == anchor_year)
            ].copy()
            if chain.empty:
                print(f"[SKIP] T{tower:02d} {anchor_year}: no chain rows")
                continue

            margins, counts = calibration_details(chains, tower, anchor_year)
            output = plot_one(
                daily_by_tower[tower], chain, margins, tower, anchor_year
            )
            for lo, hi in BINS:
                label = f"{lo}-{hi}"
                manifest_rows.append(
                    {
                        "tower": tower,
                        "anchor_year": anchor_year,
                        "interval_type": "raw" if tower == 2 else "CQR",
                        "lead_bin": label,
                        "calibration_n": counts[label],
                        "cqr_margin": np.nan if tower == 2 else margins[label],
                        "figure": output.name,
                    }
                )
            print(f"[OK] {output.name}")

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(OUTPUT_DIR / "cqr_gallery_manifest.csv", index=False)
    print(f"[OK] Saved {manifest['figure'].nunique()} figures and manifest to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
