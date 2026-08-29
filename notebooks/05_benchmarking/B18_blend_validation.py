"""B18 additive validation of conservative blends among leading forecasts.

Weights are tested as fixed values, leave-one-block-out estimates, and strictly
forward estimates.  This file reads B18 chains and writes new B18 artifacts only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(NOTEBOOK_DIR))

import B18_evaluate_and_plot as ev


RESULTS = ROOT / "results"
GRID_PATH = RESULTS / "b18_blend_grid.csv"
CHAINS_PATH = RESULTS / "b18_blend_validation_chains.csv"
SUMMARY_PATH = RESULTS / "b18_blend_validation_summary.csv"
FIXED_CHAIN_PATH = RESULTS / "b18_final_fixed_blend_chains.csv"
MANIFEST_PATH = RESULTS / "b18_blend_validation_manifest.json"
FIG_DIR = RESULTS / "figures" / "b18_chains_blend"

LEFT = ev.CHAMPION
RIGHT = ev.RECENCY
THIRD = "B18 TabPFN 1460-day tower-robust"
XGB_EVENT = "B18 XGB-gate p90 + 0.25 excess"
ZSCORE = "B18 TabPFN tower-zscore"
TOP = [LEFT, RIGHT, THIRD, XGB_EVENT, ZSCORE]


def load_wide():
    sources = {
        "direct": pd.read_csv(ev.DIRECT_PATH, parse_dates=["date"]),
        "spike": pd.read_csv(ev.SPIKE_PATH, parse_dates=["date"]),
        "screen": pd.read_csv(ev.B17_SCREEN_PATH, parse_dates=["date"]),
    }
    rows = pd.concat([ev.candidate_frame(spec, sources) for spec in ev.CANDIDATE_SPECS], ignore_index=True)
    return ev.build_wide(rows)


def best_pair_alpha(training, left=LEFT, right=RIGHT):
    grid = np.linspace(0, 1, 51)
    losses = []
    for alpha in grid:
        candidate = training.assign(_prediction=(1 - alpha) * training[left] + alpha * training[right])
        losses.append(ev.metrics(candidate, prediction_col="_prediction")["MASE"])
    return float(grid[int(np.nanargmin(losses))])


def best_triple_weights(training):
    best = (np.inf, 1 / 3, 1 / 3, 1 / 3)
    for i in range(11):
        for j in range(11 - i):
            w1, w2 = i / 10, j / 10
            w3 = 1 - w1 - w2
            prediction = w1 * training[LEFT] + w2 * training[RIGHT] + w3 * training[THIRD]
            score = ev.metrics(training.assign(_prediction=prediction), prediction_col="_prediction")["MASE"]
            if score < best[0]:
                best = (score, w1, w2, w3)
    return best[1:]


def append_method(parts, test, prediction, method, weights, validation):
    out = test.copy()
    out["prediction"] = prediction
    out["method"] = method
    out["weights"] = weights
    out["validation_scheme"] = validation
    parts.append(out)


def aggregate_grid(scored):
    rows = []
    for left_index, left in enumerate(TOP):
        for right in TOP[left_index + 1:]:
            for alpha in np.linspace(0, 1, 101):
                prediction = (1 - alpha) * scored[left] + alpha * scored[right]
                value = ev.metrics(scored.assign(_prediction=prediction), prediction_col="_prediction")
                rows.append({"left": left, "right": right, "right_weight": alpha, **value})
    result = pd.DataFrame(rows).sort_values("MASE")
    result.to_csv(GRID_PATH, index=False)
    return result


def validate(wide):
    scored = wide.loc[wide["y_true"].notna() & wide["MAE_climatology"].gt(0)].copy()
    parts = []
    append_method(parts, scored, 0.5 * scored[LEFT] + 0.5 * scored[RIGHT], "fixed_pair_equal", "0.50/0.50", "fixed")
    append_method(parts, scored, (scored[LEFT] + scored[RIGHT] + scored[THIRD]) / 3, "fixed_triple_equal", "1/3 each", "fixed")
    append_method(parts, scored, scored[[LEFT, RIGHT, THIRD]].median(axis=1), "fixed_triple_median", "median", "fixed")

    for tower, year in scored[["tower", "anchor_year"]].drop_duplicates().itertuples(index=False):
        mask = scored["tower"].eq(tower) & scored["anchor_year"].eq(year)
        test = scored.loc[mask].copy()
        other = scored.loc[~mask]

        alpha = best_pair_alpha(other)
        append_method(parts, test, (1 - alpha) * test[LEFT] + alpha * test[RIGHT], "lobo_pair_weight", f"{1-alpha:.2f}/{alpha:.2f}", "leave-one-tower-anchor-block-out")

        same_tower = other.loc[other["tower"].eq(tower)]
        alpha_tower = best_pair_alpha(same_tower) if not same_tower.empty else alpha
        append_method(parts, test, (1 - alpha_tower) * test[LEFT] + alpha_tower * test[RIGHT], "lobo_same_tower_pair_weight", f"{1-alpha_tower:.2f}/{alpha_tower:.2f}", "held-out block; other same-tower blocks")

        weights = best_triple_weights(other)
        triple = weights[0] * test[LEFT] + weights[1] * test[RIGHT] + weights[2] * test[THIRD]
        append_method(parts, test, triple, "lobo_triple_simplex", "/".join(f"{x:.1f}" for x in weights), "leave-one-tower-anchor-block-out; simplex grid")

        earlier = scored.loc[scored["anchor_year"].lt(year)]
        alpha_forward = best_pair_alpha(earlier) if not earlier.empty else 0.5
        append_method(parts, test, (1 - alpha_forward) * test[LEFT] + alpha_forward * test[RIGHT], "forward_pair_weight", f"{1-alpha_forward:.2f}/{alpha_forward:.2f}", "strictly earlier anchor years; equal fallback")

    combined = pd.concat(parts, ignore_index=True)
    keep = ["date", "lead_day", "bin", "tower", "anchor_year", "method", "weights", "validation_scheme", "prediction", "y_true", "y_gapfilled", "MAE_climatology"]
    combined[keep].to_csv(CHAINS_PATH, index=False)
    rows = []
    for method, group in combined.groupby("method", sort=False):
        for target, column in [("observed", "y_true"), ("gapfilled", "y_gapfilled")]:
            rows.append({"method": method, "target": target, **ev.metrics(group, target_col=column)})
    result = pd.DataFrame(rows).sort_values(["target", "MASE"])
    result.to_csv(SUMMARY_PATH, index=False)
    return result


def save_fixed_chain_and_figures(wide):
    full = wide[["date", "lead_day", "tower", "anchor_year", "y_true", "y_gapfilled", LEFT, RIGHT]].copy()
    full["y_predict"] = 0.5 * full[LEFT] + 0.5 * full[RIGHT]
    full["model"] = "TabPFN_v2"
    full["config"] = "B18_fixed_spike_recency_mean"
    full["protocol"] = "observed_history_known_future_fx"
    full[["date", "lead_day", "tower", "anchor_year", "model", "config", "protocol", "y_predict", "y_true", "y_gapfilled"]].to_csv(FIXED_CHAIN_PATH, index=False)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(ev.DATA_PATH, low_memory=False)
    data["Datetime"] = pd.to_datetime(data["Datetime"], format="mixed")
    frames = {tower: data.loc[data["tower"].eq(tower)].set_index("Datetime").sort_index() for tower in ev.TOWERS}
    for tower in ev.TOWERS:
        for year in ev.ANCHOR_YEARS:
            anchor = pd.Timestamp(f"{year}-12-16")
            dates = pd.date_range(anchor + pd.Timedelta(days=1), periods=ev.N_DAYS, freq="D")
            window = pd.date_range(anchor - pd.Timedelta(days=30), dates[-1], freq="D")
            prediction = full.loc[full["tower"].eq(tower) & full["anchor_year"].eq(year)].set_index("date")["y_predict"].reindex(dates)
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


def main():
    wide = load_wide()
    scored = wide.loc[wide["y_true"].notna() & wide["MAE_climatology"].gt(0)]
    grid = aggregate_grid(scored)
    summary = validate(wide)
    save_fixed_chain_and_figures(wide)
    MANIFEST_PATH.write_text(
        json.dumps(
            {
                "predeclared_pair": [LEFT, RIGHT],
                "fixed_primary_blend": "equal mean",
                "validation": ["fixed", "leave-one-block-out", "same-tower leave-one-block-out", "strict forward"],
                "aggregate_grid_is_exploratory": True,
                "chain_figures": 15,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\nBest exploratory aggregate blend weights")
    print(grid.head(15).to_string(index=False))
    print("\nBlend validation")
    print(summary.loc[summary["target"].eq("observed")].to_string(index=False))


if __name__ == "__main__":
    main()
