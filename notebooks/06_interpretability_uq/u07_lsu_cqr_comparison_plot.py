"""U-07: illustrates the livestock-density-stratified CQR band directly against fx_lsu_dens on the
same chain -- shows the interval visibly narrowing in winter (near-zero livestock) and widening
through the grazing season, tracking the covariate driving the heteroscedasticity, instead of the
flat lead-time-only band. Full roster: T2/T4/T9 x TabPFN+TabICLv2 (U-04) and T2/T4/T9 x TabICLv2
(U-05, matches S-05's TabICL-only scope) -- one figure per (tower, model, data_label) at its
spikiest anchor, same selection convention as u06_cqr_comparison_plots.py. T2 has zero valid
lsu_cqr_margin rows in both summaries (its base conformal calibration is already degenerate -- see
U-05's T2 finding) so no band can be drawn for it; this is reported explicitly rather than silently
skipped.
"""
import os
import sys

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"c:\Users\Nicholas\Documents\COMP0191 MSc Artificial Intelligence for Sustainable Development Project"
sys.path.insert(0, ROOT + r"\src")
import models.recursive_rollout as rr

HOURLY = rf"{ROOT}\data\Hourly"
RESULTS = rf"{ROOT}\results"
FIG_DIR = rf"{RESULTS}\figures\u07_lsu_cqr"
os.makedirs(FIG_DIR, exist_ok=True)

BINS = ((1, 7), (8, 30), (31, 90), (91, 180), (181, 270), (271, 365))
LSU_TIERS = ["low", "mid", "high"]


def lsu_tier(vals, edges):
    import numpy as np
    return np.where(vals <= edges[0], "low", np.where(vals <= edges[1], "mid", "high"))


def plot_one(dft, chains, lsu_summary, plain_summary, tower, model, data_label):
    sub = chains[(chains.eval_tower == tower) & (chains.model == model)]
    if sub.empty:
        return False

    lsu_m_all = lsu_summary[(lsu_summary.eval_tower == tower) & (lsu_summary.model == model)]
    if lsu_m_all["lsu_cqr_margin"].notna().sum() == 0:
        print(f"[SKIP] T{tower} {model} ({data_label}): no valid lsu_cqr_margin (base CQR degenerate for this tower)")
        return False

    spike_counts = sub.groupby("anchor_year")["y_true"].apply(lambda s: (s >= sub.y_true.quantile(0.9)).sum())
    yr = spike_counts.idxmax()
    anchor = pd.Timestamp(f"{yr}-12-16")
    chain_sub = sub[sub.anchor_year == yr].copy()
    chain_sub["lead_bin"] = rr.lead_time_bin(chain_sub["date"].values, anchor, BINS)
    chain_sub = chain_sub.merge(dft[["fx_lsu_dens"]].reset_index().rename(columns={"Datetime": "date"}), on="date", how="left")

    calib = sub[sub.anchor_year != yr].merge(dft[["fx_lsu_dens"]].reset_index().rename(columns={"Datetime": "date"}), on="date", how="left")
    edges = calib["fx_lsu_dens"].dropna().quantile([1 / 3, 2 / 3]).values
    chain_sub["lsu_tier"] = lsu_tier(chain_sub["fx_lsu_dens"].values, edges)

    lsu_m = lsu_m_all[lsu_m_all.anchor_year == yr]
    lsu_margins = {(r["bin"], r["lsu_tier"]): r["lsu_cqr_margin"] for _, r in lsu_m.iterrows()}
    plain_m = plain_summary[(plain_summary.eval_tower == tower) & (plain_summary.anchor_year == yr) & (plain_summary.model == model)]
    plain_margins = dict(zip(plain_m["bin"], plain_m["cqr_margin"]))

    chain_sub["lsu_margin"] = chain_sub.apply(lambda r: lsu_margins.get((r["lead_bin"], r["lsu_tier"]), float("nan")), axis=1)
    chain_sub["plain_margin"] = chain_sub["lead_bin"].map(plain_margins)
    chain_sub["lsu_lo"] = chain_sub["q05"] - chain_sub["lsu_margin"]
    chain_sub["lsu_hi"] = chain_sub["q95"] + chain_sub["lsu_margin"]
    chain_sub["plain_lo"] = chain_sub["q05"] - chain_sub["plain_margin"]
    chain_sub["plain_hi"] = chain_sub["q95"] + chain_sub["plain_margin"]

    window = pd.date_range(anchor - pd.Timedelta(days=30), chain_sub["date"].max(), freq="D")
    actual = dft["y_observed"].reindex(window)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    ax = axes[0]
    ax.plot(actual.index, actual.values, "-", color="black", linewidth=1, label="Actual (observed)")
    ax.plot(chain_sub["date"], chain_sub["median"], "-", color="tab:green", linewidth=1.2, label=f"{model} median")
    ax.fill_between(chain_sub["date"], chain_sub["plain_lo"], chain_sub["plain_hi"], color="tab:red", alpha=0.15,
                     label="U-06: flat CQR (lead-time only)")
    ax.fill_between(chain_sub["date"], chain_sub["lsu_lo"], chain_sub["lsu_hi"], color="tab:blue", alpha=0.25,
                     label="U-07: LSU-stratified CQR")
    ax.set_ylabel("FCH4 (nmol m-2 s-1)")
    ax.set_title(f"Tower {tower}, anchor {anchor.date()}, {model} ({data_label}): flat vs. LSU-density-stratified CQR")
    ax.legend(fontsize=8, loc="upper left")

    ax2 = axes[1]
    ax2.plot(chain_sub["date"], chain_sub["fx_lsu_dens"], "-", color="tab:brown", linewidth=1.2)
    ax2.set_ylabel("fx_lsu_dens")
    ax2.set_title("Livestock density driving the interval width above")
    for tier, color in zip(LSU_TIERS, ["tab:green", "tab:orange", "tab:red"]):
        mask = chain_sub["lsu_tier"] == tier
        ax2.fill_between(chain_sub["date"], 0, chain_sub["fx_lsu_dens"].max(), where=mask.values,
                          color=color, alpha=0.08)

    fig.tight_layout()
    fname = f"T{tower}_anchor{yr}_{model}_{data_label}_lsu_stratified.png"
    fig.savefig(f"{FIG_DIR}/{fname}", dpi=110)
    plt.close(fig)
    print(f"[OK] Saved {fname}")
    return True


def run(data_label, chains_file, lsu_summary_file, plain_summary_file, towers, models):
    dv = pd.read_csv(f"{HOURLY}/forecast_daily_v3.csv", low_memory=False)
    dv["Datetime"] = pd.to_datetime(dv["Datetime"], format="mixed")
    T = {t: dv[dv.tower == t].set_index("Datetime").sort_index() for t in towers}

    chains = pd.read_csv(f"{RESULTS}/{chains_file}", parse_dates=["date"])
    lsu_summary = pd.read_csv(f"{RESULTS}/{lsu_summary_file}")
    plain_summary = pd.read_csv(f"{RESULTS}/{plain_summary_file}")

    n_saved, n_skipped = 0, 0
    for tower in towers:
        for model in models:
            ok = plot_one(T[tower], chains, lsu_summary, plain_summary, tower, model, data_label)
            n_saved += int(ok)
            n_skipped += int(not ok)
    print(f"[OK] {data_label}: {n_saved} figures saved, {n_skipped} skipped (degenerate)")


def main():
    run("U04", "u04_chains.csv", "u07_u04_lsu_cqr_summary.csv", "u06_u04_cqr_summary.csv",
        towers=[2, 4, 9], models=["TabPFN", "TabICLv2"])
    run("U05", "u05_chains.csv", "u07_u05_lsu_cqr_summary.csv", "u06_u05_cqr_summary.csv",
        towers=[2, 4, 9], models=["TabICLv2"])


if __name__ == "__main__":
    main()
