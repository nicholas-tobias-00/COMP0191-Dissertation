"""U-05b fan-chart plots: same visual convention as `u05_fanchart_plots.py` (incl. the AOA-flagged-
day marker), applied to U-05b's calibration-set chains (`results/u05b_chains.csv`/
`u05b_summary.csv` -- the locked-in solo per-tower `Direct_TabICLv2` + trend architecture on
FX_A_SPECIES, real historical anchors). 3 towers x 5 anchors, minus 2 skipped T9 anchors
(insufficient pre-anchor history, per S-03b's own finding) = 13 figures.

Outputs to results/figures/u05b_fancharts/T{tower}_anchor{year}_Direct_TabICLv2_solo_trend.png.
"""
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"c:\Users\Nicholas\Documents\COMP0191 MSc Artificial Intelligence for Sustainable Development Project"
HOURLY = rf"{ROOT}\data\Hourly"
RESULTS = rf"{ROOT}\results"
FIG_DIR = Path(RESULTS) / "figures" / "u05b_fancharts"
FIG_DIR.mkdir(parents=True, exist_ok=True)

PRE_ANCHOR_DAYS = 30
BINS = ((1, 7), (8, 30), (31, 90), (91, 180), (181, 270), (271, 365))
MODEL = "Direct_TabICLv2_solo_trend"


def bin_label_for_date(d, anchor):
    lead = (d - anchor).days
    for lo, hi in BINS:
        if lo <= lead <= hi:
            return f"{lo}-{hi}"
    return None


def plot_fanchart(dft, chain_sub, margins_by_bin, tower, yr, anchor):
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
    aoa_flagged = chain_sub.set_index("date")["aoa_flagged"].reindex(target_dates)

    cal_lo = (median - margin).where(margin.notna())
    cal_hi = (median + margin).where(margin.notna())
    raw_lo = q05.where(margin.isna())
    raw_hi = q95.where(margin.isna())

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(gapfilled.index, gapfilled.values, ":", color="gray", linewidth=1, label="Gap-filled FCH4")
    ax.plot(actual.index, actual.values, "-", color="black", linewidth=1, label="Actual FCH4 (observed)")

    color = "tab:purple"
    ax.plot(median.index, median.values, "-", color=color, linewidth=1.5,
            label="Direct_TabICLv2_solo_trend (median), FX_A_SPECIES")

    if cal_lo.notna().any():
        ax.fill_between(median.index, cal_lo.values, cal_hi.values, color=color, alpha=0.2,
                         label="90% conformal-calibrated interval")
    if raw_lo.notna().any():
        ax.fill_between(median.index, raw_lo.values, raw_hi.values, color=color, alpha=0.15,
                         hatch="//", linestyle="--", edgecolor=color,
                         label="90% raw (uncalibrated) interval")

    flagged_dates = aoa_flagged[aoa_flagged == True].index
    if len(flagged_dates):
        ax.scatter(flagged_dates, [ax.get_ylim()[0]] * len(flagged_dates), marker="|", color="red",
                   s=20, label="AOA-flagged day", zorder=3)

    ax.set_title(f"Tower {tower}, anchor {anchor.date()}, {MODEL}, FX_A_SPECIES (U-05b fan chart)")
    ax.set_ylabel("FCH4 (nmol m-2 s-1)")
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"T{tower}_anchor{yr}_{MODEL}.png", dpi=100)
    plt.close(fig)


def main():
    dv = pd.read_csv(f"{HOURLY}/forecast_daily_v3.csv", low_memory=False)
    dv["Datetime"] = pd.to_datetime(dv["Datetime"], format="mixed")
    T = {t: dv[dv.tower == t].set_index("Datetime").sort_index() for t in [2, 4, 9]}

    chains = pd.read_csv(f"{RESULTS}/u05b_chains.csv", parse_dates=["date"])
    summary = pd.read_csv(f"{RESULTS}/u05b_summary.csv")

    n_saved = 0
    for (tower, yr), chain_sub in chains.groupby(["eval_tower", "anchor_year"]):
        anchor = pd.Timestamp(f"{yr}-12-16")
        m_sub = summary[(summary.eval_tower == tower) & (summary.anchor_year == yr) & (summary.model == MODEL)]
        margins_by_bin = dict(zip(m_sub["bin"], m_sub["conformal_margin"]))
        plot_fanchart(T[tower], chain_sub, margins_by_bin, tower, yr, anchor)
        n_saved += 1

    print(f"[OK] Saved {n_saved} fan-chart plots to {FIG_DIR}")


if __name__ == "__main__":
    main()
