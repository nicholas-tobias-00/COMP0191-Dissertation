"""U-06: before/after comparison fancharts -- OLD symmetric split-conformal band [median-margin,
median+margin] vs. NEW CQR band [q05-margin, q95+margin], on the same chain, side by side. A
representative subset (not all 45 combinations again): T4 and T9, both models, the anchors with
the most visible real spikes in-window, checked directly rather than picked arbitrarily.
"""
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"c:\Users\Nicholas\Documents\COMP0191 MSc Artificial Intelligence for Sustainable Development Project"
HOURLY = rf"{ROOT}\data\Hourly"
RESULTS = rf"{ROOT}\results"
FIG_DIR = Path(RESULTS) / "figures" / "u06_cqr"
FIG_DIR.mkdir(parents=True, exist_ok=True)

BINS = ((1, 7), (8, 30), (31, 90), (91, 180), (181, 270), (271, 365))


def bin_label_for_date(d, anchor):
    lead = (d - anchor).days
    for lo, hi in BINS:
        if lo <= lead <= hi:
            return f"{lo}-{hi}"
    return None


def plot_comparison(dft, chain_sub, old_margins, cqr_margins, tower, yr, model, anchor, data_label):
    target_dates = chain_sub["date"]
    window = pd.date_range(anchor - pd.Timedelta(days=30), target_dates.max(), freq="D")
    gapfilled = dft["y_gapfilled"].reindex(window)
    actual = dft["y_observed"].reindex(window)

    median = chain_sub.set_index("date")["median"].reindex(target_dates)
    q05 = chain_sub.set_index("date")["q05"].reindex(target_dates)
    q95 = chain_sub.set_index("date")["q95"].reindex(target_dates)
    labels = [bin_label_for_date(d, anchor) for d in target_dates]
    old_m = pd.Series([old_margins.get(l, float("nan")) for l in labels], index=target_dates)
    cqr_m = pd.Series([cqr_margins.get(l, float("nan")) for l in labels], index=target_dates)

    old_lo, old_hi = median - old_m, median + old_m
    cqr_lo, cqr_hi = q05 - cqr_m, q95 + cqr_m

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    for ax, lo, hi, title, color in [
        (axes[0], old_lo, old_hi, "OLD: split-conformal, symmetric around median", "tab:red"),
        (axes[1], cqr_lo, cqr_hi, "NEW: CQR, asymmetric around raw [q05,q95]", "tab:green"),
    ]:
        ax.plot(gapfilled.index, gapfilled.values, ":", color="gray", linewidth=1, label="Gap-filled")
        ax.plot(actual.index, actual.values, "-", color="black", linewidth=1, label="Actual (observed)")
        ax.plot(median.index, median.values, "-", color=color, linewidth=1.2, label=f"{model} median")
        ax.fill_between(median.index, lo.values, hi.values, color=color, alpha=0.2, label="90% interval")
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("FCH4 (nmol m-2 s-1)")
        ax.legend(fontsize=7, loc="upper right")

    fig.suptitle(f"U-06: Tower {tower}, anchor {anchor.date()}, {model} ({data_label}) -- "
                 f"old symmetric vs. new CQR interval")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"T{tower}_anchor{yr}_{model}_{data_label}.png", dpi=100)
    plt.close(fig)


def run(data_label, hourly_file, chains_file, old_summary_file, cqr_summary_file, towers, anchor_years):
    dv = pd.read_csv(f"{HOURLY}/{hourly_file}", low_memory=False)
    dv["Datetime"] = pd.to_datetime(dv["Datetime"], format="mixed")
    T = {t: dv[dv.tower == t].set_index("Datetime").sort_index() for t in towers}

    chains = pd.read_csv(f"{RESULTS}/{chains_file}", parse_dates=["date"])
    old_summary = pd.read_csv(f"{RESULTS}/{old_summary_file}")
    cqr_summary = pd.read_csv(f"{RESULTS}/{cqr_summary_file}")

    # pick anchors/towers with the most real-observed spikes in-window (top-decile threshold),
    # not arbitrary -- one figure per (tower, model) at its single spikiest anchor
    chains["is_spike"] = chains.groupby(["eval_tower"])["y_true"].transform(lambda s: s >= s.quantile(0.9))
    spikiness = chains.groupby(["eval_tower", "anchor_year", "model"])["is_spike"].sum().reset_index()

    n_saved = 0
    for tower in [4, 9]:
        for model in chains["model"].unique():
            sub = spikiness[(spikiness.eval_tower == tower) & (spikiness.model == model)]
            if sub.empty or sub["is_spike"].max() == 0:
                continue
            best_yr = sub.loc[sub["is_spike"].idxmax(), "anchor_year"]
            anchor = pd.Timestamp(f"{int(best_yr)}-12-16")

            chain_sub = chains[(chains.eval_tower == tower) & (chains.anchor_year == best_yr) & (chains.model == model)]
            om = old_summary[(old_summary.eval_tower == tower) & (old_summary.anchor_year == best_yr) & (old_summary.model == model)]
            cm = cqr_summary[(cqr_summary.eval_tower == tower) & (cqr_summary.anchor_year == best_yr) & (cqr_summary.model == model)]
            old_margins = dict(zip(om["bin"], om["conformal_margin"]))
            cqr_margins = dict(zip(cm["bin"], cm["cqr_margin"]))

            plot_comparison(T[tower], chain_sub, old_margins, cqr_margins, tower, int(best_yr), model, anchor, data_label)
            n_saved += 1

    print(f"[OK] Saved {n_saved} comparison figures for {data_label}")


def main():
    run("U04", "forecast_daily_v3.csv", "u04_chains.csv", "u04_summary.csv", "u06_u04_cqr_summary.csv",
        [2, 4, 9], [2018, 2019, 2020, 2021, 2022])
    run("U05", "forecast_daily_v3.csv", "u05_chains.csv", "u05_summary.csv", "u06_u05_cqr_summary.csv",
        [2, 4, 9], [2018, 2019, 2020, 2021, 2022])


if __name__ == "__main__":
    main()
