"""U-03 Part B extended: per-(anchor,tower) response-curve plots (same visual convention as the
original single-anchor u03_response_curve_plot.py) PLUS the robustness check this extension exists
for -- does the tree-plateau / SARIMAX-unbounded-linear pattern found at the single (T4, 2021) case
hold consistently across all 5 anchors and both towers, or was it anchor/tower-specific (this
project's own repeated lesson, D-53/54/55/56: never trust a single-anchor read)?
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = r"c:\Users\Nicholas\Documents\COMP0191 MSc Artificial Intelligence for Sustainable Development Project"
RESULTS = rf"{ROOT}\results"
FIG_DIR = rf"{RESULTS}\figures\u03_fancharts"

MODEL_COLORS = {
    "RF": "tab:green", "XGB": "tab:orange", "LightGBM": "tab:blue", "SARIMAX": "tab:red", "TFT": "tab:brown",
    "TabPFN": "tab:cyan", "Ensemble_unweighted": "tab:purple", "Ensemble_MASEweighted": "tab:pink",
}
MODELS_ORDER = ["RF", "XGB", "LightGBM", "TFT", "TabPFN", "Ensemble_unweighted", "Ensemble_MASEweighted", "SARIMAX"]


def plot_one(df, tower, yr):
    sub = df[(df.eval_tower == tower) & (df.anchor_year == yr)]
    hist_max = sub["training_max_lsu_dens"].iloc[0]
    window_max_at_1x = sub[sub.multiplier == 1.0]["window_max_lsu_dens"].iloc[0]
    threshold_mult = hist_max / window_max_at_1x if window_max_at_1x > 0 else np.nan

    fig, ax = plt.subplots(figsize=(9, 6))
    for model, msub in sub.groupby("model"):
        agg = msub.groupby("multiplier")["mean_median_pred"].mean()
        ax.plot(agg.index, agg.values, "-o", color=MODEL_COLORS.get(model, "tab:purple"), label=model)

    if np.isfinite(threshold_mult):
        ax.axvline(threshold_mult, color="black", linestyle="--", linewidth=1,
                   label=f"training-range boundary ({threshold_mult:.2f}x,\ntraining max={hist_max:.2f})")
    ax.set_xlabel("fx_lsu_dens scenario multiplier (1.0 = real rollout-window values)")
    ax.set_ylabel("Mean predicted median FCH4 (nmol m-2 s-1), avg. across 6 lead-time bins")
    ax.set_title(f"U-03 Part B: livestock-density extrapolation response (Tower {tower}, anchor {yr}-12-16)\n"
                 "DIAGNOSTIC ONLY -- no ground truth exists for these scenario inputs")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/T{tower}_anchor{yr}_lsu_perturbation_response.png", dpi=100)
    plt.close(fig)


def main():
    df = pd.read_csv(f"{RESULTS}/u03_extrapolation_stress_test_multi.csv")

    n_saved = 0
    for tower in df.eval_tower.unique():
        for yr in df.anchor_year.unique():
            plot_one(df, int(tower), int(yr))
            n_saved += 1
    print(f"[OK] Saved {n_saved} per-anchor-tower response-curve plots to {FIG_DIR}")

    # ---- Robustness check: %-change (1.0x -> 3.0x) per model, per anchor, per tower ----
    pivot = df.groupby(["eval_tower", "anchor_year", "model", "multiplier"])["mean_median_pred"].mean().unstack("multiplier")
    pct = ((pivot[3.0] - pivot[1.0]) / pivot[1.0] * 100).rename("pct_change_1_to_3x").reset_index()
    pct.to_csv(f"{RESULTS}/u03_pct_change_summary.csv", index=False)
    print(f"\n[OK] Saved u03_pct_change_summary.csv ({len(pct)} rows)")

    print("\nPct change (1.0x -> 3.0x) by model, across all anchors/towers:")
    print(pct.groupby("model")["pct_change_1_to_3x"].agg(["mean", "std", "min", "max"]).round(1).to_string())

    # Box/strip plot: one column of points per model. T2 excluded from this plot (its livestock
    # perturbation is degenerate -- fx_lsu_dens is exactly 0 throughout the rollout window in 4/5
    # anchors, so its %-change values are mostly 0 by construction, not a genuine model response;
    # mixing them in would silently drag every model's summary toward 0 for the wrong reason).
    fig, ax = plt.subplots(figsize=(9, 6))
    pct_t49 = pct[pct.eval_tower.isin([4, 9])]
    for i, model in enumerate(MODELS_ORDER):
        vals = pct_t49[pct_t49.model == model]["pct_change_1_to_3x"].values
        color = MODEL_COLORS.get(model, "tab:gray")
        jitter = np.random.default_rng(0).uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, color=color, s=50, alpha=0.8, zorder=3)
        ax.scatter([i], [vals.mean()], color=color, marker="D", s=140, edgecolor="black", zorder=4)
    ax.set_xticks(range(len(MODELS_ORDER)))
    ax.set_xticklabels(MODELS_ORDER, rotation=20, ha="right")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_ylabel("% change in mean predicted median FCH4 (1.0x -> 3.0x fx_lsu_dens)")
    ax.set_title("U-03 Part B robustness check: %-change across 5 anchors x Towers 4+9 (10 points/model)\n"
                 "Tower 2 excluded here -- livestock perturbation is degenerate there (fx_lsu_dens=0 in 4/5 anchors)\n"
                 "diamond = mean; DIAGNOSTIC ONLY, not a validated prediction")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/pct_change_summary_all_anchors_towers.png", dpi=100)
    plt.close(fig)
    print(f"\n[OK] Saved robustness summary plot to {FIG_DIR}")


if __name__ == "__main__":
    main()
