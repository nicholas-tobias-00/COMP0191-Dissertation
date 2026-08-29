"""Additive B18 TabICL-only and mixed TabPFN-TabICL ensembles.

Uses already-trained, complete B18 raw chains.  Reports exploratory aggregate
subsets separately from leave-one-block-out and strictly-forward validation.
No prior B18 artifact is modified.
"""

from __future__ import annotations

import itertools
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
DIRECT_PATH = RESULTS / "b18_direct_structure_chains.csv"
PFN_PATH = RESULTS / "b18_final_triple_blend_chains.csv"

COMPONENT_PATH = RESULTS / "b18_tabicl_ensemble_components.csv"
SUBSET_PATH = RESULTS / "b18_tabicl_subset_grid.csv"
MIXED_GRID_PATH = RESULTS / "b18_tabicl_mixed_grid.csv"
VALIDATION_CHAINS_PATH = RESULTS / "b18_tabicl_ensemble_validation_chains.csv"
SUMMARY_PATH = RESULTS / "b18_tabicl_ensemble_summary.csv"
TABICL_FINAL_PATH = RESULTS / "b18_tabicl_final_ensemble_chains.csv"
MIXED_FINAL_PATH = RESULTS / "b18_tabicl_mixed_final_chains.csv"
MANIFEST_PATH = RESULTS / "b18_tabicl_ensemble_manifest.json"

TABICL_FIG_DIR = RESULTS / "figures" / "b18_chains_tabicl_ensemble"
MIXED_FIG_DIR = RESULTS / "figures" / "b18_chains_mixed_ensemble"

COMPONENTS = {
    "B18D31": "ICL_all_raw",
    "B18D32": "ICL_all_tower_robust",
    "B18D34": "ICL_antecedent_raw",
    "B18D35": "ICL_antecedent_tower_robust",
    "B18D36": "ICL_1460_raw",
    "B18D37": "ICL_1460_tower_robust",
    "B18D38": "ICL_seasonal_tower_robust",
}
NON_SEASONAL = [name for experiment, name in COMPONENTS.items() if experiment != "B18D38"]
PREDECLARED_TOP3 = ["ICL_1460_tower_robust", "ICL_1460_raw", "ICL_all_tower_robust"]
PFN = "PFN_final_triple"


def load_components():
    direct = pd.read_csv(DIRECT_PATH, parse_dates=["date"])
    key = ["date", "lead_day", "tower", "anchor_year"]
    base = direct.loc[direct["experiment_id"].eq("B18D37"), key + ["y_true", "y_gapfilled"]].drop_duplicates(key)
    for experiment, alias in COMPONENTS.items():
        values = direct.loc[direct["experiment_id"].eq(experiment), key + ["y_predict"]].drop_duplicates(key).rename(columns={"y_predict": alias})
        if len(values) != 5475:
            raise ValueError(f"Incomplete component {experiment}: {len(values)} rows")
        base = base.merge(values, on=key, how="left", validate="one_to_one")
    pfn = pd.read_csv(PFN_PATH, parse_dates=["date"])[key + ["y_predict"]].rename(columns={"y_predict": PFN})
    base = base.merge(pfn, on=key, how="left", validate="one_to_one")
    base.to_csv(COMPONENT_PATH, index=False)
    return ev.add_scoring(base)


def subset_specs():
    specs = []
    for size in range(1, len(NON_SEASONAL) + 1):
        for columns in itertools.combinations(NON_SEASONAL, size):
            specs.append(tuple(columns))
    specs.append(tuple(COMPONENTS.values()))
    return specs


def subset_name(columns):
    return "+".join(columns)


def score_prediction(frame, prediction, target_col="y_true"):
    return ev.metrics(frame.assign(_prediction=prediction), prediction_col="_prediction", target_col=target_col)


def evaluate_subsets(full, specs):
    rows = []
    for columns in specs:
        prediction = full[list(columns)].mean(axis=1)
        for target, target_col in [("observed", "y_true"), ("gapfilled", "y_gapfilled")]:
            rows.append({"subset": subset_name(columns), "components": len(columns), "target": target, **score_prediction(full, prediction, target_col)})
    result = pd.DataFrame(rows).sort_values(["target", "MASE"])
    result.to_csv(SUBSET_PATH, index=False)
    return result


def best_alpha(training, icl_columns=PREDECLARED_TOP3, maximum=0.5):
    icl = training[list(icl_columns)].mean(axis=1)
    grid = np.linspace(0, maximum, int(maximum * 100) + 1)
    losses = [score_prediction(training, (1 - alpha) * training[PFN] + alpha * icl)["MASE"] for alpha in grid]
    return float(grid[int(np.nanargmin(losses))])


def best_subset(training, specs):
    scores = []
    for columns in specs:
        prediction = training[list(columns)].mean(axis=1)
        scores.append(score_prediction(training, prediction)["MASE"])
    return specs[int(np.nanargmin(scores))]


def mixed_grid(full, specs):
    rows = []
    for columns in specs:
        icl = full[list(columns)].mean(axis=1)
        for alpha in np.linspace(0, 0.5, 51):
            prediction = (1 - alpha) * full[PFN] + alpha * icl
            value = score_prediction(full, prediction)
            rows.append({"tabicl_subset": subset_name(columns), "tabicl_components": len(columns), "tabicl_weight": alpha, **value})
    result = pd.DataFrame(rows).sort_values("MASE")
    result.to_csv(MIXED_GRID_PATH, index=False)
    return result


def append_method(parts, test, prediction, method, detail, validation):
    out = test.copy()
    out["prediction"] = prediction
    out["method"] = method
    out["detail"] = detail
    out["validation_scheme"] = validation
    parts.append(out)


def validate(full, specs, aggregate_best_subset):
    scored = full.loc[full["y_true"].notna() & full["MAE_climatology"].gt(0)].copy()
    parts = []
    top3 = scored[PREDECLARED_TOP3].mean(axis=1)
    aggregate_best = scored[list(aggregate_best_subset)].mean(axis=1)
    append_method(parts, scored, top3, "tabicl_fixed_top3_equal", subset_name(PREDECLARED_TOP3), "fixed; top three standalone TabICL variants")
    append_method(parts, scored, aggregate_best, "tabicl_exploratory_best_subset", subset_name(aggregate_best_subset), "exploratory aggregate selection")
    for alpha in [0.10, 0.25, 0.50]:
        append_method(parts, scored, (1 - alpha) * scored[PFN] + alpha * top3, f"mixed_fixed_tabicl_{int(alpha * 100)}pct", f"PFN={1-alpha:.2f}; TabICL={alpha:.2f}", "fixed")

    for tower, year in scored[["tower", "anchor_year"]].drop_duplicates().itertuples(index=False):
        mask = scored["tower"].eq(tower) & scored["anchor_year"].eq(year)
        test = scored.loc[mask].copy()
        other = scored.loc[~mask]

        selected = best_subset(other, specs)
        append_method(parts, test, test[list(selected)].mean(axis=1), "tabicl_lobo_subset", subset_name(selected), "leave-one-tower-anchor-block-out")

        earlier = scored.loc[scored["anchor_year"].lt(year)]
        selected_forward = best_subset(earlier, specs) if not earlier.empty else tuple(PREDECLARED_TOP3)
        append_method(parts, test, test[list(selected_forward)].mean(axis=1), "tabicl_forward_subset", subset_name(selected_forward), "strictly earlier anchor years; fixed top-three fallback")

        alpha = best_alpha(other)
        test_icl = test[PREDECLARED_TOP3].mean(axis=1)
        append_method(parts, test, (1 - alpha) * test[PFN] + alpha * test_icl, "mixed_lobo_weight", f"TabICL={alpha:.2f}", "leave-one-tower-anchor-block-out")

        alpha_forward = best_alpha(earlier) if not earlier.empty else 0.0
        append_method(parts, test, (1 - alpha_forward) * test[PFN] + alpha_forward * test_icl, "mixed_forward_weight", f"TabICL={alpha_forward:.2f}", "strictly earlier anchor years; zero-TabICL fallback")

    combined = pd.concat(parts, ignore_index=True)
    keep = ["date", "lead_day", "bin", "tower", "anchor_year", "method", "detail", "validation_scheme", "prediction", "y_true", "y_gapfilled", "MAE_climatology"]
    combined[keep].to_csv(VALIDATION_CHAINS_PATH, index=False)
    rows = []
    for method, group in combined.groupby("method", sort=False):
        for target, target_col in [("observed", "y_true"), ("gapfilled", "y_gapfilled")]:
            rows.append({"method": method, "target": target, **ev.metrics(group, target_col=target_col)})
    result = pd.DataFrame(rows).sort_values(["target", "MASE"])
    result.to_csv(SUMMARY_PATH, index=False)
    return result


def save_chain(full, prediction, path, model, config):
    out = full[["date", "lead_day", "tower", "anchor_year", "y_true", "y_gapfilled"]].copy()
    out["model"] = model
    out["config"] = config
    out["protocol"] = "observed_history_known_future_fx"
    out["y_predict"] = prediction
    order = ["date", "lead_day", "tower", "anchor_year", "model", "config", "protocol", "y_predict", "y_true", "y_gapfilled"]
    out[order].to_csv(path, index=False)
    return out


def plot_chain_set(chains, fig_dir, model_label):
    fig_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(ev.DATA_PATH, low_memory=False)
    data["Datetime"] = pd.to_datetime(data["Datetime"], format="mixed")
    frames = {tower: data.loc[data["tower"].eq(tower)].set_index("Datetime").sort_index() for tower in ev.TOWERS}
    for tower in ev.TOWERS:
        for year in ev.ANCHOR_YEARS:
            anchor = pd.Timestamp(f"{year}-12-16")
            dates = pd.date_range(anchor + pd.Timedelta(days=1), periods=ev.N_DAYS, freq="D")
            window = pd.date_range(anchor - pd.Timedelta(days=30), dates[-1], freq="D")
            prediction = chains.loc[chains["tower"].eq(tower) & chains["anchor_year"].eq(year)].set_index("date")["y_predict"].reindex(dates)
            fig, ax = plt.subplots(figsize=(13, 5))
            ax.plot(window, frames[tower]["y_gapfilled"].reindex(window), ":", color="gray", linewidth=1, label="Gap-filled FCH4")
            ax.plot(window, frames[tower]["y_observed"].reindex(window), "-", color="black", linewidth=1, label="Actual FCH4 (observed)")
            ax.plot(dates, prediction, "-", color="tab:blue", linewidth=1.5, label=f"{model_label} (predicted)")
            ev.add_forecast_start_marker(ax, anchor)
            ax.set_title(f"Tower {tower}, anchor {anchor.date()}, model={model_label}")
            ax.set_ylabel("FCH4 (nmol m-2 s-1)")
            ax.legend(loc="upper right", fontsize=9)
            fig.tight_layout()
            safe = model_label.replace(" + ", "_").replace(" ", "_")
            fig.savefig(fig_dir / f"T{tower}_anchor{year}_{safe}.png", dpi=100)
            plt.close(fig)


def main():
    full = load_components()
    specs = subset_specs()
    subset_results = evaluate_subsets(full, specs)
    observed_subsets = subset_results.loc[subset_results["target"].eq("observed")]
    best_subset_text = observed_subsets.iloc[0]["subset"]
    best_subset_columns = tuple(best_subset_text.split("+"))
    mixed_results = mixed_grid(full, specs)
    validation = validate(full, specs, best_subset_columns)

    tabicl_prediction = full[list(best_subset_columns)].mean(axis=1)
    tabicl_chain = save_chain(full, tabicl_prediction, TABICL_FINAL_PATH, "TabICLv2", "B18_best_equal_subset")

    best_mixed = mixed_results.iloc[0]
    mixed_columns = best_mixed["tabicl_subset"].split("+")
    alpha = float(best_mixed["tabicl_weight"])
    mixed_prediction = (1 - alpha) * full[PFN] + alpha * full[mixed_columns].mean(axis=1)
    mixed_chain = save_chain(full, mixed_prediction, MIXED_FINAL_PATH, "TabPFN_v2 + TabICLv2", "B18_exploratory_mixed")

    plot_chain_set(tabicl_chain, TABICL_FIG_DIR, "TabICLv2")
    plot_chain_set(mixed_chain, MIXED_FIG_DIR, "TabPFN_v2 + TabICLv2")

    MANIFEST_PATH.write_text(
        json.dumps(
            {
                "complete_tabicl_components": COMPONENTS,
                "predeclared_top3": PREDECLARED_TOP3,
                "exploratory_best_tabicl_subset": list(best_subset_columns),
                "exploratory_best_mixed_tabicl_subset": mixed_columns,
                "exploratory_best_mixed_tabicl_weight": alpha,
                "validation": ["fixed", "leave-one-block-out", "strictly forward"],
                "figures": {"tabicl_only": 15, "mixed": 15},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\nBest TabICL-only equal subsets")
    print(observed_subsets.head(15).to_string(index=False))
    print("\nBest exploratory mixed ensembles")
    print(mixed_results.head(15).to_string(index=False))
    print("\nHeld-out and fixed validation")
    print(validation.loc[validation["target"].eq("observed")].to_string(index=False))


if __name__ == "__main__":
    main()
