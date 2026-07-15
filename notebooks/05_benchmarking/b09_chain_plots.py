"""Chain figures for the D-71 climatology-baseline check: actual/gap-filled/predicted, PLUS both
MASE baselines (chain-persistence and day-of-year climatology) overlaid, for every (model, tower,
anchor_year) combination in the full 11-model B-10/B-13 roster.

Separate directory from `results/figures/b10_chains/` (which doesn't plot either baseline) rather
than modifying those existing figures in place -- this is a different, baseline-focused view built
specifically to make D-71's finding (climatology's own line visibly tracks real data worse than the
flat persistence line, at Towers 4/9 particularly) inspectable per chain, not a replacement for the
existing model-comparison figure set.

Scope: the 11 core B-10/B-13 models only (S-03's variant-suffixed columns excluded -- out of scope
for this baseline check, which is about B-10/B-13's own MASE convention, not S-03's ablation).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = r"c:\Users\Nicholas\Documents\COMP0191 MSc Artificial Intelligence for Sustainable Development Project"
RESULTS = rf"{ROOT}\results"
FIG_DIR = Path(RESULTS) / "figures" / "b09_chains"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["RF", "XGB", "LightGBM", "SARIMAX", "Ensemble_unweighted", "Ensemble_MASEweighted",
          "TFT", "TabPFN", "DLinear", "LSTM", "TabICLv2"]
MODEL_COLORS = {
    "RF": "tab:green", "XGB": "tab:orange", "LightGBM": "tab:blue", "SARIMAX": "tab:red",
    "Ensemble_unweighted": "tab:purple", "Ensemble_MASEweighted": "tab:pink",
    "TFT": "tab:brown", "TabPFN": "tab:cyan",
    "DLinear": "tab:olive", "LSTM": "orchid", "TabICLv2": "teal",
}
DL_MODELS = {"TFT", "DLinear", "LSTM"}


def plot_chain(sub, tower, yr, model):
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(sub.date, sub.y_gapfilled, ":", color="lightgray", linewidth=1, label="Gap-filled FCH4")
    y_true_col = "y_true_tft" if model in DL_MODELS else "y_true"
    ax.plot(sub.date, sub[y_true_col], "-", color="black", linewidth=1, label="Actual FCH4 (observed)")
    ax.plot(sub.date, sub["persistence"], "--", color="gray", linewidth=1.3, label="Persistence baseline")
    ax.plot(sub.date, sub["Climatology"], ":", color="dimgray", linewidth=1.3, label="Climatology baseline (y_observed basis)")
    ax.plot(sub.date, sub["Climatology_gf"], "-.", color="slategray", linewidth=1.3, label="Climatology baseline (y_gapfilled basis)")
    color = MODEL_COLORS.get(model, "tab:gray")
    ax.plot(sub.date, sub[model], "-", color=color, linewidth=1.2, label=f"{model} (predicted)")
    ax.set_title(f"Tower {tower}, anchor {yr}-12-16, model={model}")
    ax.set_ylabel("FCH4 (nmol m-2 s-1)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"T{tower}_anchor{yr}_{model}.png", dpi=100)
    plt.close(fig)


def main():
    chains = pd.read_csv(f"{RESULTS}/b10_b13_full_chains.csv", parse_dates=["date"])

    n_saved, n_skipped = 0, 0
    for (tower, yr), sub in chains.groupby(["tower", "anchor_year"]):
        sub = sub.sort_values("date")
        for model in MODELS:
            if model not in sub.columns or not sub[model].notna().any():
                n_skipped += 1
                continue
            plot_chain(sub, tower, yr, model)
            n_saved += 1

    print(f"[OK] Saved {n_saved} figures to {FIG_DIR} ({n_skipped} skipped, model not present/all-NaN)")


if __name__ == "__main__":
    main()
