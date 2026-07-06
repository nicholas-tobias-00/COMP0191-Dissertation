"""U-02 fan-chart plots: actual/gap-filled/predicted-median + shaded conformal-calibrated quantile
band, extending the existing chain-plot visual convention (results/figures/b10_chains,
results/figures/b15_chains) with an uncertainty band layered on top.

Reads results/u02_chains.csv (per-day median + raw q05/q95, produced by u02_multi_anchor_tower.py's
Stage A) and results/u02_summary.csv (per bin/model/tower/anchor conformal_margin, from Stage B) to
build the calibrated band: [median - margin, median + margin]. Falls back to the raw [q05, q95]
band (dashed edge, to visually distinguish "uncalibrated" from "calibrated") on a **per-day** basis
-- a chain can be genuinely mixed (e.g. calibration available for lead-time bins 91-180 onward but
not for 1-7/8-30, because the specific test anchor lacked >=3 real y_observed points in those early
bins even though other anchors' calibration data for those bins exists) -- an earlier version of
this script decided calibrated-vs-raw once for the WHOLE chain (`if margin.notna().any()`), which
silently produced blank gaps wherever calibration was missing even though a perfectly good raw
interval existed for those exact days; fixed to fall back day-by-day instead, so no interval is
ever hidden when one is actually available.
Outputs to results/figures/u02_fanchart_{tower}_{anchor}_{model}.png.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = r"c:\Users\Nicholas\Documents\COMP0191 MSc Artificial Intelligence for Sustainable Development Project"
HOURLY = rf"{ROOT}\data\Hourly"
RESULTS = rf"{ROOT}\results"
FIG_DIR = Path(RESULTS) / "figures" / "u02_fancharts"
FIG_DIR.mkdir(parents=True, exist_ok=True)

PRE_ANCHOR_DAYS = 30
BINS = ((1, 7), (8, 30), (31, 90), (91, 180), (181, 270), (271, 365))

MODEL_COLORS = {
    "RF": "tab:green", "XGB": "tab:orange", "LightGBM": "tab:blue", "SARIMAX": "tab:red",
    "TFT": "tab:brown", "TabPFN": "tab:cyan",
    "Ensemble_unweighted": "tab:purple", "Ensemble_MASEweighted": "tab:pink",
}


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

    # Per-day fallback: calibrated where a margin exists for that day's bin, raw where it doesn't
    # (but a raw quantile does), blank only where genuinely neither exists (e.g. TFT with no
    # calibration for that bin -- TFT has no raw mechanism to fall back to at all).
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

    ax.set_title(f"Tower {tower}, anchor {anchor.date()}, model={model} (U-02 fan chart)")
    ax.set_ylabel("FCH4 (nmol m-2 s-1)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"T{tower}_anchor{yr}_{model}.png", dpi=100)
    plt.close(fig)


def main():
    dv = pd.read_csv(f"{HOURLY}/forecast_daily_v2.csv", low_memory=False)
    dv["Datetime"] = pd.to_datetime(dv["Datetime"], format="mixed")
    T = {t: dv[dv.tower == t].set_index("Datetime").sort_index() for t in [2, 4, 9]}

    chains = pd.read_csv(f"{RESULTS}/u02_chains.csv", parse_dates=["date"])
    summary = pd.read_csv(f"{RESULTS}/u02_summary.csv")

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
