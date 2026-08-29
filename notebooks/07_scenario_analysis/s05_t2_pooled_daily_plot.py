"""S05-T2 figure, daily-resolution format matching `s05_livestock_daily_chains_plots.py`'s
full-horizon figures exactly (thin lines, stacked panels, same color/title convention) -- 4 panels
(TabICLv2-solo, TabICLv2-pooled, TabPFN-solo, TabPFN-pooled), one figure per SSP, full 2020/2024-2050
horizon. TabICLv2 solo pulled directly from the existing `s05_daily_chains_2050.parquet` (already
on disk, no new calls needed); TabPFN solo/pooled and TabICLv2 pooled from
`s05_t2_pooled_daily_chains.csv`/`s05_t2_tabpfn_solo_daily_chains.csv` (D-95's rerun, daily chains
preserved this time).

Run from project root:  python notebooks/07_scenario_analysis/s05_t2_pooled_daily_plot.py
"""
import os

import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"c:\Users\Nicholas\Documents\COMP0191 MSc Artificial Intelligence for Sustainable Development Project"
RESULTS = rf"{ROOT}\results"
FIG_DIR = rf"{RESULTS}\figures\s05_t2_pooled"
os.makedirs(FIG_DIR, exist_ok=True)

GCM, REAL = "ACCESS-ESM1-5", 1
COMBOS = {(1.0, 1.0, 1.0): "Baseline (1x/1x/1x)", (3.0, 1.0, 1.0): "Cattle 3x alone",
          (3.0, 3.0, 3.0): "All species 3x"}
COMBO_NAMES = {(1.0, 1.0, 1.0): "baseline_1x1x1x", (3.0, 1.0, 1.0): "cattle3x_alone",
               (3.0, 3.0, 3.0): "all_3x3x3x"}
COLORS = {(1.0, 1.0, 1.0): "tab:blue", (3.0, 1.0, 1.0): "tab:orange", (3.0, 3.0, 3.0): "tab:red"}


def load_icl_solo(ssp):
    tbl = pq.read_table(f"{RESULTS}/s05_daily_chains_2050.parquet",
                         filters=[("tower", "=", 2), ("gcm", "=", GCM), ("realization", "=", REAL),
                                  ("ssp", "=", ssp)])
    df = tbl.to_pandas()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def load_pooled_and_pfn_solo(ssp):
    pooled = pd.read_csv(f"{RESULTS}/s05_t2_pooled_daily_chains.csv", parse_dates=["timestamp"])
    pooled = pooled[pooled.ssp == ssp]
    pfn_solo = pd.read_csv(f"{RESULTS}/s05_t2_tabpfn_solo_daily_chains.csv", parse_dates=["timestamp"])
    pfn_solo = pfn_solo[pfn_solo.ssp == ssp]
    return pooled, pfn_solo


def plot_panel(ax, dates_by_combo, title):
    for combo, label in COMBOS.items():
        d = dates_by_combo.get(combo)
        if d is None or d.empty:
            continue
        ax.plot(d["timestamp"], d["pred"], color=COLORS[combo], label=label, linewidth=0.4, alpha=0.8)
    ax.set_title(title)
    ax.set_ylabel("Predicted FCH4 (nmol m-2 s-1)")
    ax.legend(fontsize=8, loc="upper left")


def main(ssp):
    icl_solo = load_icl_solo(ssp)
    pooled, pfn_solo = load_pooled_and_pfn_solo(ssp)

    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=False)

    icl_by_combo = {c: icl_solo[(icl_solo.mult_cattle == c[0]) & (icl_solo.mult_sheep == c[1]) &
                                 (icl_solo.mult_lamb == c[2])].sort_values("timestamp") for c in COMBOS}
    plot_panel(axes[0], icl_by_combo, f"Tower 2, TabICLv2 (solo) -- daily FCH4 to 2050 ({ssp}, {GCM}/realization {REAL})")

    icl_pooled = pooled[pooled.model == "TabICLv2"]
    icl_pooled_by_combo = {c: icl_pooled[icl_pooled.combo == COMBO_NAMES[c]].rename(
        columns={"pred_pooled": "pred"}).sort_values("timestamp") for c in COMBOS}
    plot_panel(axes[1], icl_pooled_by_combo, f"Tower 2, TabICLv2 (pooled w/ T4+T9) -- daily FCH4 to 2050 ({ssp}, {GCM}/realization {REAL})")

    pfn_solo_by_combo = {c: pfn_solo[pfn_solo.combo == COMBO_NAMES[c]].rename(
        columns={"pred_solo": "pred"}).sort_values("timestamp") for c in COMBOS}
    plot_panel(axes[2], pfn_solo_by_combo, f"Tower 2, TabPFN (solo) -- daily FCH4 to 2050 ({ssp}, {GCM}/realization {REAL})")

    pfn_pooled = pooled[pooled.model == "TabPFN"]
    pfn_pooled_by_combo = {c: pfn_pooled[pfn_pooled.combo == COMBO_NAMES[c]].rename(
        columns={"pred_pooled": "pred"}).sort_values("timestamp") for c in COMBOS}
    plot_panel(axes[3], pfn_pooled_by_combo, f"Tower 2, TabPFN (pooled w/ T4+T9) -- daily FCH4 to 2050 ({ssp}, {GCM}/realization {REAL})")

    fig.suptitle(f"S05-T2 (D-95): daily-resolution solo vs. pooled, Tower 2, {ssp} -- "
                 f"pooling makes no visible difference for either model")
    fig.tight_layout()
    fname = f"s05_t2_pooled_daily_full_horizon_{ssp}.png"
    fig.savefig(f"{FIG_DIR}/{fname}", dpi=120)
    plt.close(fig)
    print(f"[OK] Saved {fname}")


if __name__ == "__main__":
    for _ssp in ["ssp245", "ssp585"]:
        main(_ssp)
