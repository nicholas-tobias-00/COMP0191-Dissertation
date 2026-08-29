"""Full-coverage, single-panel CQR figures for U-08 (B18 full champion: direct TabPFN + p95
spike-gate, BASE_ALL_52) -- all 3 towers x 5 anchors = 15 charts, one PNG per (tower, anchor),
CQR result only (no old-vs-symmetric comparison panel, unlike u06_cqr_comparison_plots.py's
representative-subset figures).

Tower 2 is left UNCALIBRATED by request: its `cqr_margin` is NaN throughout (base conformal
calibration is already degenerate for T2, established since U-02) -- rather than draw nothing,
this plots the model's raw [q05, q95] band directly (hatched, matching this project's established
raw-vs-calibrated visual convention) and labels it as such. Towers 4/9 use the CQR-calibrated band
([q05-margin, q95+margin], `results/u06b_u08_cqr_summary.csv`'s `cqr_margin`, per lead-time bin).

Run from project root:  python notebooks/06_interpretability_uq/u06_cqr_full_u08_plots.py
"""
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"c:\Users\Nicholas\Documents\COMP0191 MSc Artificial Intelligence for Sustainable Development Project"
HOURLY = rf"{ROOT}\data\Hourly"
RESULTS = rf"{ROOT}\results"
FIG_DIR = Path(RESULTS) / "figures" / "u06_cqr_u08_full"
FIG_DIR.mkdir(parents=True, exist_ok=True)

TOWERS = [2, 4, 9]
ANCHOR_YEARS = [2018, 2019, 2020, 2021, 2022]
BINS = ((1, 7), (8, 30), (31, 90), (91, 180), (181, 270), (271, 365))
MODEL = "B18_TabPFN_champion"
PRE_ANCHOR_DAYS = 30


def bin_label_for_date(d, anchor):
    lead = (d - anchor).days
    for lo, hi in BINS:
        if lo <= lead <= hi:
            return f"{lo}-{hi}"
    return None


def plot_one(dft, chain_sub, margins_by_bin, tower, yr, anchor, calibrated):
    target_dates = chain_sub["date"]
    window = pd.date_range(anchor - pd.Timedelta(days=PRE_ANCHOR_DAYS), target_dates.max(), freq="D")
    gapfilled = dft["y_gapfilled"].reindex(window)
    actual = dft["y_observed"].reindex(window)

    median = chain_sub.set_index("date")["median"].reindex(target_dates)
    q05 = chain_sub.set_index("date")["q05"].reindex(target_dates)
    q95 = chain_sub.set_index("date")["q95"].reindex(target_dates)

    if calibrated:
        labels = [bin_label_for_date(d, anchor) for d in target_dates]
        margin = pd.Series([margins_by_bin.get(l, float("nan")) for l in labels], index=target_dates)
        lo, hi = q05 - margin, q95 + margin
        band_label = "90% CQR-calibrated interval [q05-margin, q95+margin]"
        fill_kwargs = dict(color="tab:green", alpha=0.25)
    else:
        lo, hi = q05, q95
        band_label = "90% raw interval [q05, q95] -- UNCALIBRATED (T2 base calibration degenerate)"
        fill_kwargs = dict(color="tab:green", alpha=0.15, hatch="//", linestyle="--", edgecolor="tab:green")

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(gapfilled.index, gapfilled.values, ":", color="gray", linewidth=1, label="Gap-filled FCH4")
    ax.plot(actual.index, actual.values, "-", color="black", linewidth=1, label="Actual FCH4 (observed)")
    ax.plot(median.index, median.values, "-", color="tab:green", linewidth=1.5, label="Prediction (median)")
    ax.fill_between(median.index, lo.values, hi.values, label=band_label, **fill_kwargs)

    ax.set_title(f"Prediction and Calibrated UQ (q05, q95) around gap-filled target - T{tower:02d} - {yr}")
    ax.set_ylabel("FCH4 (nmol m-2 s-1)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fname = f"T{tower:02d}_anchor{yr}_CQR.png"
    fig.savefig(FIG_DIR / fname, dpi=110)
    plt.close(fig)
    return fname


def main():
    dv = pd.read_csv(f"{HOURLY}/forecast_daily_v3.csv", low_memory=False)
    dv["Datetime"] = pd.to_datetime(dv["Datetime"], format="mixed")
    T = {t: dv[dv.tower == t].set_index("Datetime").sort_index() for t in TOWERS}

    chains = pd.read_csv(f"{RESULTS}/u08_chains.csv", parse_dates=["date"])
    cqr_summary = pd.read_csv(f"{RESULTS}/u06b_u08_cqr_summary.csv")

    n_saved = 0
    for tower in TOWERS:
        for yr in ANCHOR_YEARS:
            chain_sub = chains[(chains.eval_tower == tower) & (chains.anchor_year == yr) & (chains.model == MODEL)]
            if chain_sub.empty:
                print(f"[SKIP] T{tower} {yr}: no chain rows")
                continue
            anchor = pd.Timestamp(f"{yr}-12-16")
            m_sub = cqr_summary[(cqr_summary.eval_tower == tower) & (cqr_summary.anchor_year == yr) & (cqr_summary.model == MODEL)]
            margins_by_bin = dict(zip(m_sub["bin"], m_sub["cqr_margin"]))
            calibrated = tower != 2  # T2 left uncalibrated by request
            fname = plot_one(T[tower], chain_sub, margins_by_bin, tower, yr, anchor, calibrated)
            print(f"[OK] {fname} ({'calibrated' if calibrated else 'UNCALIBRATED (T2)'})")
            n_saved += 1

    print(f"\n[OK] Saved {n_saved} figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
