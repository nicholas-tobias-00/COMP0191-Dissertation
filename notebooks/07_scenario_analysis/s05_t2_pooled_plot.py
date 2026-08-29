"""S05-T2 figure: solo vs. pooled annual-mean trajectories for Tower 2, both models, all 3 combos,
one representative SSP (ssp245) -- the direct visual of D-95's "exactly 0.0pp difference" finding.
Solo (dashed) and pooled (solid) should visually overlap completely if the numeric finding holds.

Run from project root:  python notebooks/07_scenario_analysis/s05_t2_pooled_plot.py
"""
import os

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"c:\Users\Nicholas\Documents\COMP0191 MSc Artificial Intelligence for Sustainable Development Project"
RESULTS = rf"{ROOT}\results"
FIG_DIR = rf"{RESULTS}\figures\s05_t2_pooled"
os.makedirs(FIG_DIR, exist_ok=True)

SSP = "ssp245"
COMBOS = ["baseline_1x1x1x", "cattle3x_alone", "all_3x3x3x"]
COLORS = {"baseline_1x1x1x": "tab:blue", "cattle3x_alone": "tab:orange", "all_3x3x3x": "tab:red"}
LABELS = {"baseline_1x1x1x": "Baseline (1x/1x/1x)", "cattle3x_alone": "Cattle 3x alone",
          "all_3x3x3x": "All species 3x"}


def load_data():
    pooled = pd.read_csv(f"{RESULTS}/s05_t2_pooled_trajectory.csv")
    pooled = pooled[pooled.ssp == SSP]

    icl_solo = pd.read_csv(f"{RESULTS}/s05_trajectory_realizations_2050.csv")
    icl_solo = icl_solo[(icl_solo.tower == 2) & (icl_solo.gcm == "ACCESS-ESM1-5") &
                         (icl_solo.realization == 1) & (icl_solo.ssp == SSP)]
    combo_map = {(1, 1, 1): "baseline_1x1x1x", (3, 1, 1): "cattle3x_alone", (3, 3, 3): "all_3x3x3x"}
    icl_solo["combo"] = icl_solo.apply(
        lambda r: combo_map.get((int(r.mult_cattle), int(r.mult_sheep), int(r.mult_lamb))), axis=1)
    icl_solo = icl_solo.dropna(subset=["combo"])
    icl_solo = icl_solo.rename(columns={"annual_mean": "value"})
    icl_solo["model"] = "TabICLv2"
    icl_solo["variant"] = "solo"

    pfn_solo = pd.read_csv(f"{RESULTS}/s05_t2_tabpfn_solo_trajectory.csv")
    pfn_solo = pfn_solo[pfn_solo.ssp == SSP].rename(columns={"annual_mean_solo": "value"})
    pfn_solo["variant"] = "solo"

    pooled = pooled.rename(columns={"annual_mean_pooled": "value"})
    pooled["variant"] = "pooled"

    return pd.concat([icl_solo[["model", "combo", "year", "value", "variant"]],
                       pfn_solo[["model", "combo", "year", "value", "variant"]],
                       pooled[["model", "combo", "year", "value", "variant"]]], ignore_index=True)


def main():
    df = load_data()
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    for ax, model in zip(axes, ["TabICLv2", "TabPFN"]):
        sub = df[df.model == model]
        for combo in COMBOS:
            csub = sub[sub.combo == combo].sort_values("year")
            solo = csub[csub.variant == "solo"]
            pooled = csub[csub.variant == "pooled"]
            ax.plot(solo["year"], solo["value"], "--", color=COLORS[combo], linewidth=2.5,
                     alpha=0.9, label=f"{LABELS[combo]} (solo)")
            ax.plot(pooled["year"], pooled["value"], "-", color=COLORS[combo], linewidth=1.2,
                     alpha=0.9, label=f"{LABELS[combo]} (pooled)")
        ax.set_title(f"Tower 2, {model} -- solo (dashed) vs. pooled (solid), {SSP}")
        ax.set_ylabel("Annual mean FCH4 (nmol m-2 s-1)")
        ax.legend(fontsize=7, ncol=2, loc="upper left")
    axes[1].set_xlabel("Year")
    fig.suptitle("S05-T2 (D-95): solo and pooled trajectories are visually identical -- "
                 "pooling does not change Tower 2's livestock-scenario response")
    fig.tight_layout()
    fname = f"s05_t2_pooled_vs_solo_{SSP}.png"
    fig.savefig(f"{FIG_DIR}/{fname}", dpi=120)
    plt.close(fig)
    print(f"[OK] Saved {fname}")


if __name__ == "__main__":
    main()
