"""S-04 analysis: summarizes the already-computed transient 2025-2050 scenario trajectories
(`s04_trajectory_2050.py` / `s04_daily_top3_2050.py`, run 2026-07-15/16, never analyzed or written
up). Read-only against the existing S-04 output CSVs -- runs no models, fits nothing new.

Produces:
  1. Trajectory summary (mean + realization spread) per tower/ssp/multiplier/year -- primary hybrid.
  2. SSP2-4.5 vs SSP5-8.5 divergence at baseline (1x) livestock, early (2025-2029) vs late
     (2046-2050) window.
  3. Realization-level spread (p10/p90 band) as a fraction of the mean -- the actual new uncertainty
     signal S-04 adds over S-01's single ensemble-mean point estimate.
  4. AOA-flagged-% trend over the 26-year trajectory (does extrapolation risk grow as the scenario
     moves further from 2018-2023 training data, or does S-01's finding that it's livestock- not
     climate-driven hold at full transient scale?).
  5. Primary hybrid vs B-10 diagnostic-benchmark ensemble cross-check, on the shared stratified
     10-realization subset -- does the hybrid-vs-trees divergence found at a single time slice (S-01
     vs U-03) hold/grow across the full 26-year, both-SSP trajectory?

Figures: trajectory + spread band (per tower, both SSPs, 1x baseline), SSP245-vs-SSP585 divergence
bar chart, AOA-over-time line chart, hybrid-vs-benchmark divergence-by-multiplier chart.
"""
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

ROOT = r"c:\Users\Nicholas\Documents\COMP0191 MSc Artificial Intelligence for Sustainable Development Project"
sys.path.insert(0, ROOT + r"\src\features")
RESULTS = rf"{ROOT}\results"
FIG_DIR = rf"{RESULTS}\figures\s04_summary"

import os
os.makedirs(FIG_DIR, exist_ok=True)

from build_transient_scenario_drivers import stratified_realizations

TOWERS = [2, 4, 9]
SSPS = ["ssp245", "ssp585"]
MULTS = [1.0, 2.0, 3.0]
EARLY_YEARS = list(range(2025, 2030))   # 2025-2029
LATE_YEARS = list(range(2046, 2051))    # 2046-2050
N_PER_GCM_B10 = 2  # matches s04_trajectory_2050.py's tracked scope cut

TOWER_COLORS = {2: "tab:blue", 4: "tab:orange", 9: "tab:green"}


def load_data():
    prim = pd.read_csv(f"{RESULTS}/s04_trajectory_realizations.csv")
    bench = pd.read_csv(f"{RESULTS}/s04_trajectory_realizations_b10benchmark.csv")
    aoa = pd.read_csv(f"{RESULTS}/s04_aoa_by_year.csv")
    print(f"[OK] primary hybrid: {len(prim):,} rows; benchmark: {len(bench):,} rows; "
          f"AOA: {len(aoa):,} rows")
    return prim, bench, aoa


def trajectory_summary(prim):
    """Per (tower, ssp, multiplier, year): mean/std/p10/p90 annual_mean across all GCM x
    realization draws (up to 500/ssp)."""
    g = prim.groupby(["tower", "ssp", "multiplier", "year"])["annual_mean"]
    out = g.agg(
        mean="mean", std="std", n="count",
        p10=lambda s: s.quantile(0.10), p90=lambda s: s.quantile(0.90),
    ).reset_index()
    out.to_csv(f"{RESULTS}/s04_trajectory_summary.csv", index=False)
    print(f"[OK] s04_trajectory_summary.csv ({len(out)} rows)")
    return out


def ssp_divergence(traj):
    """At baseline (1x) livestock: early-window (2025-29) vs late-window (2046-50) mean, per
    tower/ssp, plus the SSP245-vs-SSP585 gap at each window."""
    sub = traj[traj["multiplier"] == 1.0].copy()
    sub["window"] = np.where(sub["year"].isin(EARLY_YEARS), "early_2025_29",
                       np.where(sub["year"].isin(LATE_YEARS), "late_2046_50", None))
    sub = sub.dropna(subset=["window"])
    win = sub.groupby(["tower", "ssp", "window"])["mean"].mean().reset_index()
    piv = win.pivot_table(index=["tower", "window"], columns="ssp", values="mean").reset_index()
    piv["ssp585_minus_ssp245"] = piv["ssp585"] - piv["ssp245"]
    piv["ssp585_pct_of_ssp245"] = (piv["ssp585"] / piv["ssp245"] - 1.0) * 100
    piv.to_csv(f"{RESULTS}/s04_ssp_divergence.csv", index=False)
    print(f"[OK] s04_ssp_divergence.csv\n{piv.to_string(index=False)}")
    return piv


def realization_spread(prim):
    """Realization-level spread (p10-p90 band as % of mean), pooled across all 26 years, per
    tower/ssp/multiplier -- the actual new uncertainty signal vs. S-01's single point estimate."""
    g = prim.groupby(["tower", "ssp", "multiplier"])["annual_mean"]
    out = g.agg(
        mean="mean", std="std",
        p10=lambda s: s.quantile(0.10), p90=lambda s: s.quantile(0.90),
        min="min", max="max", n="count",
    ).reset_index()
    out["band_width"] = out["p90"] - out["p10"]
    out["band_pct_of_mean"] = out["band_width"] / out["mean"] * 100
    out.to_csv(f"{RESULTS}/s04_realization_spread.csv", index=False)
    print(f"[OK] s04_realization_spread.csv\n{out.to_string(index=False)}")
    return out


def aoa_trend(aoa):
    """AOA-flagged-% trajectory over the 26 years, per tower/ssp/multiplier (stratified
    10-realization subset)."""
    g = aoa.groupby(["tower", "ssp", "multiplier", "year"])["aoa_flagged_pct"].mean().reset_index()
    g.to_csv(f"{RESULTS}/s04_aoa_trend.csv", index=False)

    # collapsed early-vs-late summary for the writeup
    g["window"] = np.where(g["year"].isin(EARLY_YEARS), "early_2025_29",
                     np.where(g["year"].isin(LATE_YEARS), "late_2046_50", None))
    summ = g.dropna(subset=["window"]).groupby(
        ["tower", "ssp", "multiplier", "window"])["aoa_flagged_pct"].mean().reset_index()
    summ.to_csv(f"{RESULTS}/s04_aoa_early_vs_late.csv", index=False)
    print(f"[OK] s04_aoa_trend.csv, s04_aoa_early_vs_late.csv")
    print(summ.pivot_table(index=["tower", "ssp", "multiplier"], columns="window",
                            values="aoa_flagged_pct").to_string())
    return g, summ


def hybrid_vs_benchmark(prim, bench):
    """Cross-check primary hybrid vs B-10 diagnostic-benchmark ensemble on the shared stratified
    subset (N_PER_GCM_B10=2 -> 10 (gcm, realization) pairs, both SSPs)."""
    strat = set(stratified_realizations(N_PER_GCM_B10))
    prim_sub = prim[prim.apply(lambda r: (r["gcm"], r["realization"]) in strat, axis=1)].copy()
    bench_ens = bench[bench["model"] == "Ensemble_unweighted"].copy()

    merged = prim_sub.merge(
        bench_ens, on=["ssp", "gcm", "realization", "tower", "year", "multiplier"],
        suffixes=("_hybrid", "_bench"),
    )
    merged["diff"] = merged["annual_mean_hybrid"] - merged["annual_mean_bench"]
    merged["pct_diff"] = merged["diff"] / merged["annual_mean_bench"].replace(0, np.nan) * 100
    merged.to_csv(f"{RESULTS}/s04_hybrid_vs_benchmark_raw.csv", index=False)

    summ = merged.groupby(["tower", "ssp", "multiplier"]).agg(
        hybrid_mean=("annual_mean_hybrid", "mean"),
        bench_mean=("annual_mean_bench", "mean"),
        mean_diff=("diff", "mean"),
        mean_pct_diff=("pct_diff", "mean"),
        n=("diff", "count"),
    ).reset_index()
    summ.to_csv(f"{RESULTS}/s04_hybrid_vs_benchmark_summary.csv", index=False)
    print(f"[OK] s04_hybrid_vs_benchmark_summary.csv ({len(merged)} matched rows)\n"
          f"{summ.to_string(index=False)}")

    # 1x -> 3x response comparison (S-01/U-03-style headline), pooled across years/SSPs/GCMs
    resp = merged.groupby(["tower", "multiplier"]).agg(
        hybrid_mean=("annual_mean_hybrid", "mean"), bench_mean=("annual_mean_bench", "mean"),
    ).reset_index()
    resp_piv = resp.pivot_table(index="tower", columns="multiplier",
                                 values=["hybrid_mean", "bench_mean"])
    print("\n[response 1x->3x, pooled all years/SSPs/GCMs in the shared subset]")
    print(resp_piv.to_string())
    for t in TOWERS:
        h1 = resp_piv.loc[t, ("hybrid_mean", 1.0)]
        h3 = resp_piv.loc[t, ("hybrid_mean", 3.0)]
        b1 = resp_piv.loc[t, ("bench_mean", 1.0)]
        b3 = resp_piv.loc[t, ("bench_mean", 3.0)]
        print(f"  T{t}: hybrid 1x->3x = {(h3/h1-1)*100:+.1f}%  |  benchmark-ensemble 1x->3x = "
              f"{(b3/b1-1)*100:+.1f}%")
    resp_piv.to_csv(f"{RESULTS}/s04_hybrid_vs_benchmark_response.csv")
    return merged, summ


def plot_trajectory_bands(traj):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
    for ax, tower in zip(axes, TOWERS):
        for ssp, ls in zip(SSPS, ["-", "--"]):
            sub = traj[(traj["tower"] == tower) & (traj["ssp"] == ssp) & (traj["multiplier"] == 1.0)]
            sub = sub.sort_values("year")
            ax.plot(sub["year"], sub["mean"], ls, color=TOWER_COLORS[tower],
                    label=f"{ssp} mean", linewidth=1.8)
            ax.fill_between(sub["year"], sub["p10"], sub["p90"],
                             color=TOWER_COLORS[tower], alpha=0.15 if ssp == "ssp245" else 0.08)
        ax.set_title(f"Tower {tower} (1x livestock, p10-p90 band)")
        ax.set_xlabel("Year")
        ax.set_ylabel("Predicted annual mean FCH4 (nmol m-2 s-1)")
        ax.legend(fontsize=8)
    fig.suptitle("S-04: 2025-2050 transient trajectory, primary hybrid, realization-level spread")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/s04_trajectory_bands.png", dpi=120)
    plt.close(fig)
    print("[OK] figure: s04_trajectory_bands.png")


def plot_ssp_divergence(div):
    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(TOWERS))
    width = 0.35
    early = div[div["window"] == "early_2025_29"].set_index("tower")
    late = div[div["window"] == "late_2046_50"].set_index("tower")
    ax.bar(x - width/2, [early.loc[t, "ssp585_pct_of_ssp245"] for t in TOWERS], width,
           label="2025-2029", color="tab:cyan")
    ax.bar(x + width/2, [late.loc[t, "ssp585_pct_of_ssp245"] for t in TOWERS], width,
           label="2046-2050", color="tab:red")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Tower {t}" for t in TOWERS])
    ax.set_ylabel("SSP5-8.5 vs SSP2-4.5, % difference in predicted mean FCH4 (1x livestock)")
    ax.set_title("S-04: does the SSP245/SSP585 gap widen from early to late window?")
    ax.legend()
    fig.tight_layout(rect=(0.02, 0, 1, 1))
    fig.savefig(f"{FIG_DIR}/s04_ssp_divergence.png", dpi=120)
    plt.close(fig)
    print("[OK] figure: s04_ssp_divergence.png")


def plot_aoa_trend(aoa_g):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, tower in zip(axes, TOWERS):
        for mult, color in zip(MULTS, ["tab:green", "tab:orange", "tab:red"]):
            for ssp, ls in zip(SSPS, ["-", "--"]):
                sub = aoa_g[(aoa_g["tower"] == tower) & (aoa_g["multiplier"] == mult) &
                            (aoa_g["ssp"] == ssp)].sort_values("year")
                ax.plot(sub["year"], sub["aoa_flagged_pct"], ls, color=color,
                        label=f"{mult:g}x {ssp}", linewidth=1.3)
        ax.set_title(f"Tower {tower}")
        ax.set_xlabel("Year")
        ax.set_ylabel("AOA-flagged %")
        ax.legend(fontsize=6, ncol=2)
    fig.suptitle("S-04: Area-of-Applicability flagged %, 2025-2050 (stratified 10-realization subset)")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/s04_aoa_trend.png", dpi=120)
    plt.close(fig)
    print("[OK] figure: s04_aoa_trend.png")


def plot_hybrid_vs_benchmark(merged):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
    for ax, tower in zip(axes, TOWERS):
        sub = merged[merged["tower"] == tower]
        g = sub.groupby("multiplier").agg(
            hybrid=("annual_mean_hybrid", "mean"), bench=("annual_mean_bench", "mean")).reset_index()
        ax.plot(g["multiplier"], g["hybrid"], "o-", label="Primary hybrid (S-01 design)",
                color="tab:purple")
        ax.plot(g["multiplier"], g["bench"], "s--", label="B-10 ensemble (diagnostic only)",
                color="tab:gray")
        ax.set_title(f"Tower {tower}")
        ax.set_xlabel("Livestock multiplier")
        ax.set_ylabel("Predicted annual mean FCH4")
        ax.set_xticks(MULTS)
        ax.legend(fontsize=8)
    fig.suptitle("S-04: hybrid vs. B-10 diagnostic-benchmark response to livestock scaling\n"
                 "(pooled across both SSPs, 2025-2050, 10-realization stratified subset)")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/s04_hybrid_vs_benchmark.png", dpi=120)
    plt.close(fig)
    print("[OK] figure: s04_hybrid_vs_benchmark.png")


def main():
    prim, bench, aoa = load_data()
    traj = trajectory_summary(prim)
    div = ssp_divergence(traj)
    spread = realization_spread(prim)
    aoa_g, aoa_summ = aoa_trend(aoa)
    merged, hb_summ = hybrid_vs_benchmark(prim, bench)

    plot_trajectory_bands(traj)
    plot_ssp_divergence(div)
    plot_aoa_trend(aoa_g)
    plot_hybrid_vs_benchmark(merged)

    print("\n[DONE] S-04 analysis complete.")


if __name__ == "__main__":
    main()
