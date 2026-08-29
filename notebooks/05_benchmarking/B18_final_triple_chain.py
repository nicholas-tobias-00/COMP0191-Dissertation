"""Write and plot the additive B18 equal-weight three-forecast chain."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(NOTEBOOK_DIR))

import B18_blend_validation as blend
import B18_evaluate_and_plot as ev


RESULTS = ROOT / "results"
CHAIN_PATH = RESULTS / "b18_final_triple_blend_chains.csv"
SUMMARY_PATH = RESULTS / "b18_final_triple_blend_summary.csv"
TOWER_PATH = RESULTS / "b18_final_triple_blend_by_tower.csv"
MANIFEST_PATH = RESULTS / "b18_final_triple_blend_manifest.json"
FIG_DIR = RESULTS / "figures" / "b18_chains_final"


def main():
    wide = blend.load_wide()
    components = [blend.LEFT, blend.RIGHT, blend.THIRD]
    output = wide[["date", "lead_day", "tower", "anchor_year", "y_true", "y_gapfilled"]].copy()
    output["y_predict"] = wide[components].mean(axis=1)
    output["model"] = "TabPFN_v2"
    output["config"] = "B18_equal_mean_spike_recency_normalised"
    output["protocol"] = "observed_history_known_future_fx"
    order = ["date", "lead_day", "tower", "anchor_year", "model", "config", "protocol", "y_predict", "y_true", "y_gapfilled"]
    output[order].to_csv(CHAIN_PATH, index=False)

    scored = ev.add_scoring(output.rename(columns={"y_predict": "prediction"}))
    rows = []
    tower_rows = []
    for target, column in [("observed", "y_true"), ("gapfilled", "y_gapfilled")]:
        rows.append({"target": target, **ev.metrics(scored, target_col=column)})
        for tower, group in scored.groupby("tower"):
            tower_rows.append({"tower": tower, "target": target, **ev.metrics(group, target_col=column)})
    pd.DataFrame(rows).to_csv(SUMMARY_PATH, index=False)
    pd.DataFrame(tower_rows).to_csv(TOWER_PATH, index=False)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(ev.DATA_PATH, low_memory=False)
    data["Datetime"] = pd.to_datetime(data["Datetime"], format="mixed")
    frames = {tower: data.loc[data["tower"].eq(tower)].set_index("Datetime").sort_index() for tower in ev.TOWERS}
    for tower in ev.TOWERS:
        for year in ev.ANCHOR_YEARS:
            anchor = pd.Timestamp(f"{year}-12-16")
            dates = pd.date_range(anchor + pd.Timedelta(days=1), periods=ev.N_DAYS, freq="D")
            window = pd.date_range(anchor - pd.Timedelta(days=30), dates[-1], freq="D")
            prediction = output.loc[output["tower"].eq(tower) & output["anchor_year"].eq(year)].set_index("date")["y_predict"].reindex(dates)
            fig, ax = plt.subplots(figsize=(13, 5))
            ax.plot(window, frames[tower]["y_gapfilled"].reindex(window), ":", color="gray", linewidth=1, label="Gap-filled FCH4")
            ax.plot(window, frames[tower]["y_observed"].reindex(window), "-", color="black", linewidth=1, label="Actual FCH4 (observed)")
            ax.plot(dates, prediction, "-", color="tab:blue", linewidth=1.5, label="TabPFN_v2 (predicted)")
            ev.add_forecast_start_marker(ax, anchor)
            ax.set_title(f"Tower {tower}, anchor {anchor.date()}, model=TabPFN_v2")
            ax.set_ylabel("FCH4 (nmol m-2 s-1)")
            ax.legend(loc="upper right", fontsize=9)
            fig.tight_layout()
            fig.savefig(FIG_DIR / f"T{tower}_anchor{year}_TabPFN_v2.png", dpi=100)
            plt.close(fig)

    MANIFEST_PATH.write_text(
        json.dumps({"components": components, "weights": [1 / 3] * 3, "figures": 15, "legend": "B15 series convention + B16 Forecast Start marker"}, indent=2),
        encoding="utf-8",
    )
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Saved 15 final B18 figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
