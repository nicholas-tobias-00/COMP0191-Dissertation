"""B-15 chain plots for Tower 2 and Tower 9, extending b15_chain_plots.py's Tower-4-only coverage.

Three sets of plots, all following b10_chains'/b15_chains' style (dotted gray gap-filled, black
actual observed, colored predicted line):
- T2_anchor{yr}_{model}.png -- Tower 2 evaluated with T4-tuned hyperparameters (the only option;
  T2 has real y_observed coverage at only 1/5 anchors, 2018, too scarce for its own independent
  tuning search -- see b15_cross_tower_eval.py's docstring).
- T9_anchor{yr}_{model}.png -- Tower 9 evaluated with Tower-9-tuned hyperparameters (this tower's
  own best-available config, from b15_t9_rollout_grid_search.py/b15_t9_multi_anchor.py).
- T9_anchor{yr}_{model}_T4tuned.png -- Tower 9 evaluated with T4-tuned hyperparameters instead, for
  a direct visual comparison against the T9-tuned version above (motivated by the cross-tower
  finding that T4-tuned LightGBM, T4's best single model, is T9's *worst*).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = r"c:\Users\Nicholas\Documents\COMP0191 MSc Artificial Intelligence for Sustainable Development Project"
HOURLY = rf"{ROOT}\data\Hourly"
RESULTS = rf"{ROOT}\results"
FIG_DIR = Path(RESULTS) / "figures" / "b15_chains"
FIG_DIR.mkdir(parents=True, exist_ok=True)

N_DAYS = 365
PRE_ANCHOR_DAYS = 30
ANCHOR_YEARS = [2018, 2019, 2020, 2021, 2022]

MODEL_COLORS = {
    "RF_tuned": "tab:green",
    "XGB_tuned": "tab:orange",
    "LightGBM_tuned": "tab:blue",
    "SARIMAX": "tab:red",
    "Ensemble_4model_tuned": "tab:purple",
}


def plot_chain(dft, chains, yr, model, anchor, tower, out_name, title_suffix=""):
    target_dates = pd.date_range(anchor + pd.Timedelta(days=1), periods=N_DAYS, freq="D")
    window = pd.date_range(anchor - pd.Timedelta(days=PRE_ANCHOR_DAYS), target_dates[-1], freq="D")

    gapfilled = dft["y_gapfilled"].reindex(window)
    actual = dft["y_observed"].reindex(window)
    pred = chains.set_index("Datetime")[model].reindex(target_dates)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(gapfilled.index, gapfilled.values, ":", color="gray", linewidth=1, label="Gap-filled FCH4")
    ax.plot(actual.index, actual.values, "-", color="black", linewidth=1, label="Actual FCH4 (observed)")
    ax.plot(pred.index, pred.values, "-", color=MODEL_COLORS.get(model, "tab:purple"), linewidth=1.5,
            label=f"{model} (predicted)")

    ax.set_title(f"Tower {tower}, anchor {anchor.date()}, model={model}{title_suffix}")
    ax.set_ylabel("FCH4 (nmol m-2 s-1)")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / out_name, dpi=100)
    plt.close(fig)


def main():
    dv = pd.read_csv(f"{HOURLY}/forecast_daily_v2.csv", low_memory=False)
    dv["Datetime"] = pd.to_datetime(dv["Datetime"], format="mixed")
    T = {t: dv[dv.tower == t].set_index("Datetime").sort_index() for t in [2, 4, 9]}

    n_saved = 0

    # T2, T4-tuned (only option) -- from b15_cross_tower_chains.csv
    cross = pd.read_csv(f"{RESULTS}/b15_cross_tower_chains.csv", parse_dates=["Datetime"])
    for yr in ANCHOR_YEARS:
        anchor = pd.Timestamp(f"{yr}-12-16")
        chains = cross[(cross.anchor_year == yr) & (cross.eval_tower == 2)]
        for model in MODEL_COLORS:
            plot_chain(T[2], chains, yr, model, anchor, tower=2,
                       out_name=f"T2_anchor{yr}_{model}.png", title_suffix=" (T4-tuned)")
            n_saved += 1

    # T9, T9-tuned (this tower's own config) -- from b15_t9_chains.csv
    t9_tuned = pd.read_csv(f"{RESULTS}/b15_t9_chains.csv", parse_dates=["Datetime"])
    for yr in ANCHOR_YEARS:
        anchor = pd.Timestamp(f"{yr}-12-16")
        chains = t9_tuned[t9_tuned.anchor_year == yr]
        for model in MODEL_COLORS:
            plot_chain(T[9], chains, yr, model, anchor, tower=9,
                       out_name=f"T9_anchor{yr}_{model}.png")
            n_saved += 1

    # T9, T4-tuned (comparison) -- from b15_cross_tower_chains.csv
    for yr in ANCHOR_YEARS:
        anchor = pd.Timestamp(f"{yr}-12-16")
        chains = cross[(cross.anchor_year == yr) & (cross.eval_tower == 9)]
        for model in MODEL_COLORS:
            plot_chain(T[9], chains, yr, model, anchor, tower=9,
                       out_name=f"T9_anchor{yr}_{model}_T4tuned.png", title_suffix=" (T4-tuned, comparison)")
            n_saved += 1

    print(f"[OK] Saved {n_saved} chain plots to {FIG_DIR}")


if __name__ == "__main__":
    main()
