"""U-04 fan-chart plots: same visual convention as `u02_fanchart_plots.py` (actual/gap-filled/
predicted-median + shaded conformal-calibrated quantile band, raw [q05,q95] fallback hatched where
no calibration margin exists for that day's lead-time bin), applied to U-04's chains
(`results/u04_chains.csv`/`u04_summary.csv`, TabPFN+TabICLv2 on `forecast_daily_v3.csv`'s
BASE+species config) instead of U-02's. Every (tower, anchor, model) combination is plotted, same
as U-02 -- 3 towers x 5 anchors x 2 models = 30 figures.

Outputs to results/figures/u04_fancharts/T{tower}_anchor{year}_{model}.png.
"""
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"c:\Users\Nicholas\Documents\COMP0191 MSc Artificial Intelligence for Sustainable Development Project"
HOURLY = rf"{ROOT}\data\Hourly"
RESULTS = rf"{ROOT}\results"
FIG_DIR = Path(RESULTS) / "figures" / "u04_fancharts"
FIG_DIR.mkdir(parents=True, exist_ok=True)

PRE_ANCHOR_DAYS = 30
BINS = ((1, 7), (8, 30), (31, 90), (91, 180), (181, 270), (271, 365))
MODEL_COLORS = {"TabPFN": "tab:cyan", "TabICLv2": "tab:purple"}


def bin_label_for_date(d, anchor):
    lead = (d - anchor).days
    for lo, hi in BINS:
        if lo <= lead <= hi:
            return f"{lo}-{hi}"
    return None


def plot_fanchart(dft, chain_sub, margins_by_bin, tower, yr, model, anchor):
    target_dates = chain_sub["date"]
    window_start = anchor - pd.Timedelta(days=PRE_ANCHOR_DAYS)
    window = pd.date_range(window_start, target_dates.max(), freq="D")

    gapfilled = dft["y_gapfilled"].reindex(window)
    actual = dft["y_observed"].reindex(window)

    median = chain_sub.set_index("date")["median"].reindex(target_dates)
    labels = [bin_label_for_date(d, anchor) for d in target_dates]
    margin = pd.Series([margins_by_bin.get(lbl, float("nan")) for lbl in labels], index=target_dates)
    q05 = chain_sub.set_index("date")["q05"].reindex(target_dates)
    q95 = chain_sub.set_index("date")["q95"].reindex(target_dates)

    cal_lo = (median - margin).where(margin.notna())
    cal_hi = (median + margin).where(margin.notna())
    raw_lo = q05.where(margin.isna())
    raw_hi = q95.where(margin.isna())

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(gapfilled.index, gapfilled.values, ":", color="gray", linewidth=1, label="Gap-filled FCH4")
    ax.plot(actual.index, actual.values, "-", color="black", linewidth=1, label="Actual FCH4 (observed)")

    color = MODEL_COLORS.get(model, "tab:purple")
    ax.plot(median.index, median.values, "-", color=color, linewidth=1.5, label=f"{model} (median)")

    if cal_lo.notna().any():
        ax.fill_between(median.index, cal_lo.values, cal_hi.values, color=color, alpha=0.2,
                         label="90% conformal-calibrated interval")
    if raw_lo.notna().any():
        ax.fill_between(median.index, raw_lo.values, raw_hi.values, color=color, alpha=0.15,
                         hatch="//", linestyle="--", edgecolor=color,
                         label="90% raw (uncalibrated) interval")

    ax.set_title(f"Tower {tower}, anchor {anchor.date()}, model={model}, BASE+species (U-04 fan chart)")
    ax.set_ylabel("FCH4 (nmol m-2 s-1)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"T{tower}_anchor{yr}_{model}.png", dpi=100)
    plt.close(fig)


def main():
    dv = pd.read_csv(f"{HOURLY}/forecast_daily_v3.csv", low_memory=False)
    dv["Datetime"] = pd.to_datetime(dv["Datetime"], format="mixed")
    T = {t: dv[dv.tower == t].set_index("Datetime").sort_index() for t in [2, 4, 9]}

    chains = pd.read_csv(f"{RESULTS}/u04_chains.csv", parse_dates=["date"])
    summary = pd.read_csv(f"{RESULTS}/u04_summary.csv")

    n_saved = 0
    for (tower, yr, model), chain_sub in chains.groupby(["eval_tower", "anchor_year", "model"]):
        anchor = pd.Timestamp(f"{yr}-12-16")
        m_sub = summary[(summary.eval_tower == tower) & (summary.anchor_year == yr) & (summary.model == model)]
        margins_by_bin = dict(zip(m_sub["bin"], m_sub["conformal_margin"]))
        plot_fanchart(T[tower], chain_sub, margins_by_bin, tower, yr, model, anchor)
        n_saved += 1

    print(f"[OK] Saved {n_saved} fan-chart plots to {FIG_DIR}")


if __name__ == "__main__":
    main()
