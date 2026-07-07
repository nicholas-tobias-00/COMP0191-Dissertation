"""U-03 Part B visualization: livestock-multiplier response-curve plot. NOT a fan chart in the
literal time-series sense (x-axis is the scenario multiplier, not the calendar date) -- placed in
results/figures/u03_fancharts/ per the approved plan, clearly labeled as a diagnostic/sensitivity
chart, not a validated interval.
"""
import matplotlib.pyplot as plt
import pandas as pd

ROOT = r"c:\Users\Nicholas\Documents\COMP0191 MSc Artificial Intelligence for Sustainable Development Project"
RESULTS = rf"{ROOT}\results"
FIG_DIR = rf"{RESULTS}\figures\u03_fancharts"

MODEL_COLORS = {
    "RF": "tab:green", "XGB": "tab:orange", "LightGBM": "tab:blue", "SARIMAX": "tab:red", "TFT": "tab:brown",
}


def main():
    df = pd.read_csv(f"{RESULTS}/u03_extrapolation_stress_test.csv")
    hist_max = df["training_max_lsu_dens"].iloc[0]
    window_max_at_1x = df[df.multiplier == 1.0]["window_max_lsu_dens"].iloc[0]
    threshold_mult = hist_max / window_max_at_1x

    fig, ax = plt.subplots(figsize=(9, 6))
    for model, sub in df.groupby("model"):
        agg = sub.groupby("multiplier")["mean_median_pred"].mean()
        ax.plot(agg.index, agg.values, "-o", color=MODEL_COLORS.get(model, "tab:purple"), label=model)

    ax.axvline(threshold_mult, color="black", linestyle="--", linewidth=1,
                label=f"training-range boundary (mult={threshold_mult:.2f}x,\n"
                      f"beyond which window max fx_lsu_dens > {hist_max:.2f} training max)")
    ax.set_xlabel("fx_lsu_dens scenario multiplier (1.0 = real 2022 rollout-window values)")
    ax.set_ylabel("Mean predicted median FCH4 (nmol m-2 s-1), avg. across 6 lead-time bins")
    ax.set_title("U-03 Part B: livestock-density extrapolation response (Tower 4, anchor 2021-12-16)\n"
                  "DIAGNOSTIC ONLY -- no ground truth exists for these scenario inputs, not a validated prediction")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/T4_anchor2021_lsu_perturbation_response.png", dpi=100)
    plt.close(fig)
    print(f"[OK] Saved response-curve plot to {FIG_DIR}")

    # Also a %-change table for the results doc
    summary = df.groupby(["model", "multiplier"])["mean_median_pred"].mean().unstack("multiplier")
    pct = (summary[3.0] - summary[1.0]) / summary[1.0] * 100
    print("\nMean predicted median by multiplier:")
    print(summary.round(2).to_string())
    print("\nPct change (1.0x -> 3.0x):")
    print(pct.round(1).to_string())
    print(f"\nTraining-range boundary at multiplier={threshold_mult:.2f}x")


if __name__ == "__main__":
    main()
