"""U-05 Step 5: the actual deliverable -- S-05's livestock-baseline trajectory with the two-tier
calibrated UQ interval (Step 3/4) overlaid, kept VISIBLY SEPARATE from the realization-spread band
S-05's own `s05_livestock_daily_chains_plots.py`/`s05_analysis_2050.py` already show. These are two
different uncertainty sources -- realization spread is weather/GCM-draw variability (input
uncertainty), the conformal interval here is predictive/model uncertainty (calibrated against real
historical residuals) -- and this project's own pooled-vs-isolated realization-spread correction
(D-85) already established the cost of merging different uncertainty sources into one number
without labeling them. Not repeated here.

Reads `results/u05_livestock_with_uq.csv` (S-05's own annual_mean/aoa_flagged_pct, joined with
Step 4's uq_lo/uq_hi). Baseline (1x/1x/1x) only, pooled across GCM/realization/both SSPs -- same
grouping S-05's own trajectory-band figure uses.

Outputs to results/figures/u05_fancharts/s05_trajectory_with_uq_{tower}.png (one per tower) and a
combined 3-panel figure.
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"c:\Users\Nicholas\Documents\COMP0191 MSc Artificial Intelligence for Sustainable Development Project"
RESULTS = rf"{ROOT}\results"
FIG_DIR = rf"{RESULTS}\figures\u05_fancharts"

TOWERS = [2, 4, 9]
TOWER_COLORS = {2: "tab:blue", 4: "tab:orange", 9: "tab:green"}


def main():
    df = pd.read_csv(f"{RESULTS}/u05_livestock_with_uq.csv")
    base = df[(df.mult_cattle == 1) & (df.mult_sheep == 1) & (df.mult_lamb == 1)].copy()
    anchor_year = {2: 2019, 4: 2023, 9: 2023}
    base["year_offset"] = base.apply(lambda r: r["year"] - anchor_year[r["tower"]], axis=1)

    g = base.groupby(["tower", "year_offset"]).agg(
        mean=("annual_mean", "mean"),
        p10=("annual_mean", lambda s: s.quantile(0.10)),
        p90=("annual_mean", lambda s: s.quantile(0.90)),
        uq_lo=("uq_lo", "mean"),
        uq_hi=("uq_hi", "mean"),
        aoa_flagged_pct=("aoa_flagged_pct", "mean"),
    ).reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=False)
    for ax, tower in zip(axes, TOWERS):
        sub = g[g.tower == tower].sort_values("year_offset")
        color = TOWER_COLORS[tower]

        if sub["uq_lo"].notna().any():
            ax.fill_between(sub["year_offset"], sub["uq_lo"], sub["uq_hi"], color="red", alpha=0.12,
                             label="U-05 calibrated interval\n(predictive uncertainty, two-tier AOA)")
        ax.fill_between(sub["year_offset"], sub["p10"], sub["p90"], color=color, alpha=0.25,
                         label="Realization spread\n(weather/GCM draw, isolated)")
        ax.plot(sub["year_offset"], sub["mean"], "-", color=color, linewidth=1.8, label="Mean prediction")

        ax2 = ax.twinx()
        ax2.plot(sub["year_offset"], sub["aoa_flagged_pct"], ":", color="black", linewidth=1, alpha=0.6)
        ax2.set_ylabel("AOA-flagged %", fontsize=8, color="gray")
        ax2.tick_params(axis="y", labelsize=7, colors="gray")

        title = f"Tower {tower} (1x/1x/1x baseline)"
        if sub["uq_lo"].isna().all():
            title += "\n[no valid UQ -- T2 fails leave-one-anchor-out calibration]"
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Years post-anchor (to 2050)")
        ax.set_ylabel("Predicted annual mean FCH4 (nmol m-2 s-1)")
        ax.legend(fontsize=7, loc="upper left")

    fig.suptitle("U-05: S-05's livestock-baseline trajectory with two-tier calibrated UQ, "
                 "vs. realization spread (two DIFFERENT uncertainty sources, kept separate)")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/s05_trajectory_with_uq_all_towers.png", dpi=120)
    plt.close(fig)
    print(f"[OK] figure: s05_trajectory_with_uq_all_towers.png")

    g.to_csv(f"{RESULTS}/u05_trajectory_with_uq_summary.csv", index=False)
    print(f"[OK] u05_trajectory_with_uq_summary.csv ({len(g)} rows)")


if __name__ == "__main__":
    main()
