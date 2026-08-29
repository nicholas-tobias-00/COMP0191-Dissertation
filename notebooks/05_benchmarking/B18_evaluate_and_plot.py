"""B18 additive evaluation, tower adaptation diagnostics, uncertainty, and figures.

Reads existing B17 controls and the new B18 raw chains.  It writes only new
``b18_*`` evaluation tables and figures; no prior result or report file is
modified.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
DATA_PATH = ROOT / "data" / "Hourly" / "forecast_daily_v3.csv"
BASELINE_PATH = RESULTS / "_today_climatology_baseline.csv"
DIRECT_PATH = RESULTS / "b18_direct_structure_chains.csv"
SPIKE_PATH = RESULTS / "b18_spike_model_chains.csv"
B17_SCREEN_PATH = RESULTS / "b17_foundation_screen_chains.csv"

REGISTRY_PATH = RESULTS / "b18_candidate_registry.csv"
CROSSFIT_CHAINS_PATH = RESULTS / "b18_tower_adaptive_chains.csv"
CROSSFIT_SUMMARY_PATH = RESULTS / "b18_tower_adaptive_summary.csv"
TOWER_PATH = RESULTS / "b18_champion_metrics_by_tower.csv"
BLOCK_PATH = RESULTS / "b18_champion_metrics_by_block.csv"
HORIZON_PATH = RESULTS / "b18_champion_metrics_by_horizon.csv"
SPIKE_METRICS_PATH = RESULTS / "b18_champion_spike_metrics.csv"
BOOTSTRAP_PATH = RESULTS / "b18_champion_block_bootstrap.csv"
PAIRWISE_PATH = RESULTS / "b18_pairwise_block_deltas.csv"
MANIFEST_PATH = RESULTS / "b18_evaluation_manifest.json"

CHAIN_FIG_DIR = RESULTS / "figures" / "b18_chains"
SUMMARY_FIG_DIR = RESULTS / "figures" / "b18_summary"

TOWERS = [2, 4, 9]
ANCHOR_YEARS = [2018, 2019, 2020, 2021, 2022]
N_DAYS = 365
BIN_LABELS = ["1-7", "8-30", "31-90", "91-180", "181-270", "271-365"]
BIN_EDGES = [0, 7, 30, 90, 180, 270, 365]

CHAMPION = "B18 TabPFN p95 + 0.25 excess"
RECENCY = "B18 TabPFN 1095-day"
B17_CONTROL = "B17 TabPFN direct all-history"
TABICL = "B18 TabICL 1460-day tower-robust"


def add_forecast_start_marker(ax, anchor):
    """Add the established B16-style forecast-origin marker."""
    ax.axvline(
        anchor,
        color="black",
        linestyle="--",
        linewidth=1,
        alpha=0.6,
        label="Forecast Start",
    )


CANDIDATE_SPECS = [
    {"alias": CHAMPION, "source": "spike", "experiment_id": "B18S04", "method": "base_plus_fixed_excess_0.25", "role": "best aggregate B18"},
    {"alias": "B18 XGB-gate p90 + 0.25 excess", "source": "spike", "experiment_id": "B18S07", "method": "base_plus_fixed_excess_0.25", "role": "alternative event classifier"},
    {"alias": "B18 TabPFN p90 + 0.25 excess", "source": "spike", "experiment_id": "B18S01", "method": "base_plus_fixed_excess_0.25", "role": "p90 event correction"},
    {"alias": "B18 TabPFN p95 soft mixture", "source": "spike", "experiment_id": "B18S04", "method": "soft_probability_mix", "role": "probability-mixture control"},
    {"alias": RECENCY, "source": "direct", "experiment_id": "B18D21", "role": "best recency window"},
    {"alias": "B18 TabPFN 1460-day raw", "source": "direct", "experiment_id": "B18D23", "role": "recency control"},
    {"alias": "B18 TabPFN 1460-day tower-robust", "source": "direct", "experiment_id": "B18D24", "role": "normalised recency candidate"},
    {"alias": "B18 TabPFN tower-zscore", "source": "direct", "experiment_id": "B18D11", "role": "normalisation candidate"},
    {"alias": "B18 TabPFN genuine species tower-robust", "source": "direct", "experiment_id": "B18D13", "role": "species ablation"},
    {"alias": "B18 TabPFN no management", "source": "direct", "experiment_id": "B18D08", "role": "feature ablation"},
    {"alias": "B18 TabPFN antecedent", "source": "direct", "experiment_id": "B18D15", "role": "engineered antecedent features"},
    {"alias": TABICL, "source": "direct", "experiment_id": "B18D37", "role": "best complete B18 TabICL"},
    {"alias": B17_CONTROL, "source": "direct", "experiment_id": "B18D03", "role": "B17 champion reproduction"},
    {"alias": "B16-style TabPFN-TS v2 BASE+ALL", "source": "screen", "model": "TabPFN_v2", "config": "BASE_ALL_52", "role": "B16 checkpoint control"},
    {"alias": "B16 genuine-species TabPFN-TS v3", "source": "screen", "model": "TabPFN_v3", "config": "BASE_species_37", "role": "B16 species control"},
]

# Predeclared mechanism-diverse shortlist for cross-fitted selection.  The
# retrospective aggregate ranking is not used to change this list.
SELECTION_SHORTLIST = [
    CHAMPION,
    "B18 XGB-gate p90 + 0.25 excess",
    RECENCY,
    "B18 TabPFN 1460-day tower-robust",
    "B18 TabPFN tower-zscore",
    "B18 TabPFN genuine species tower-robust",
    TABICL,
    B17_CONTROL,
    "B16-style TabPFN-TS v2 BASE+ALL",
    "B16 genuine-species TabPFN-TS v3",
]


def candidate_frame(spec, sources):
    source = sources[spec["source"]]
    if spec["source"] == "spike":
        mask = source["experiment_id"].eq(spec["experiment_id"]) & source["method"].eq(spec["method"])
    elif spec["source"] == "direct":
        mask = source["experiment_id"].eq(spec["experiment_id"])
    else:
        mask = source["model"].eq(spec["model"]) & source["config"].eq(spec["config"])
    columns = ["date", "tower", "anchor_year", "y_true", "y_gapfilled", "y_predict"]
    selected = source.loc[mask, columns].copy().rename(columns={"y_predict": "prediction"})
    selected = selected.drop_duplicates(["tower", "anchor_year", "date"])
    if selected.empty:
        raise ValueError(f"No rows for {spec['alias']}")
    selected["candidate"] = spec["alias"]
    return selected


def add_scoring(frame):
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"])
    anchors = pd.to_datetime(out["anchor_year"].astype(str) + "-12-16")
    out["lead_day"] = (out["date"] - anchors).dt.days
    out["bin"] = pd.cut(out["lead_day"], BIN_EDGES, labels=BIN_LABELS, include_lowest=True).astype(str)
    baseline = pd.read_csv(BASELINE_PATH)
    return out.merge(
        baseline[["tower", "anchor_year", "bin", "MAE_climatology"]],
        on=["tower", "anchor_year", "bin"],
        how="left",
    )


def metrics(group, prediction_col="prediction", target_col="y_true"):
    good = group[[target_col, prediction_col, "MAE_climatology"]].notna().all(axis=1)
    good &= group["MAE_climatology"].gt(0)
    g = group.loc[good]
    if g.empty:
        return {"n": 0, "MASE": np.nan, "MAE": np.nan, "RMSE": np.nan, "R2": np.nan, "bias": np.nan}
    y = g[target_col].to_numpy(float)
    p = g[prediction_col].to_numpy(float)
    err = y - p
    total = float(np.sum((y - y.mean()) ** 2))
    return {
        "n": len(g),
        "MASE": float(np.mean(np.abs(err) / g["MAE_climatology"].to_numpy(float))),
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "R2": float(1 - np.sum(err**2) / total) if total > 0 else np.nan,
        "bias": float(np.mean(p - y)),
    }


def build_registry(rows):
    scored = add_scoring(rows)
    records = []
    for spec in CANDIDATE_SPECS:
        group = scored.loc[scored["candidate"].eq(spec["alias"])]
        for target, target_col in [("observed", "y_true"), ("gapfilled", "y_gapfilled")]:
            records.append({**spec, "target": target, **metrics(group, target_col=target_col)})
    registry = pd.DataFrame(records).sort_values(["target", "MASE"])
    registry.to_csv(REGISTRY_PATH, index=False)
    return registry, scored


def build_wide(rows):
    key = ["tower", "anchor_year", "date"]
    base = rows.loc[rows["candidate"].eq(CHAMPION), key + ["y_true", "y_gapfilled"]].drop_duplicates(key)
    for candidate, group in rows.groupby("candidate", sort=False):
        values = group[key + ["prediction"]].drop_duplicates(key).rename(columns={"prediction": candidate})
        base = base.merge(values, on=key, how="left")
    return add_scoring(base)


def candidate_score(frame, candidate):
    return metrics(frame, prediction_col=candidate)["MASE"]


def select_best(training, candidates=SELECTION_SHORTLIST):
    scores = {name: candidate_score(training, name) for name in candidates if name in training and training[name].notna().any()}
    scores = {name: score for name, score in scores.items() if np.isfinite(score)}
    return min(scores, key=scores.get) if scores else B17_CONTROL


def method_rows(test, prediction, method, selected, validation):
    out = test.copy()
    out["prediction"] = prediction
    out["method"] = method
    out["selected_candidate"] = selected
    out["validation_scheme"] = validation
    return out


def tower_adaptive(wide):
    scored = wide.loc[wide["y_true"].notna() & wide["MAE_climatology"].gt(0)].copy()
    methods = [
        method_rows(scored, scored[CHAMPION], "raw_b18_champion", CHAMPION, "none; aggregate-selected raw model"),
        method_rows(scored, scored[RECENCY], "raw_recency_champion", RECENCY, "none; fixed recency model"),
        method_rows(
            scored,
            0.5 * scored[CHAMPION] + 0.5 * scored[RECENCY],
            "fixed_spike_recency_mean",
            f"equal mean: {CHAMPION} + {RECENCY}",
            "fixed equal weights",
        ),
    ]

    # Explicitly optimistic diagnostic: one model selected per tower on all that
    # tower's observations.  It is reported separately from held-out results.
    for tower, test in scored.groupby("tower", sort=False):
        selected = select_best(test)
        methods.append(method_rows(test, test[selected], "retrospective_tower_oracle", selected, "optimistic diagnostic; same-tower evaluation data used for selection"))

    blocks = scored[["tower", "anchor_year"]].drop_duplicates()
    for tower, year in blocks.itertuples(index=False):
        test_mask = scored["tower"].eq(tower) & scored["anchor_year"].eq(year)
        test = scored.loc[test_mask].copy()
        other = scored.loc[~test_mask]

        selected = select_best(other)
        methods.append(method_rows(test, test[selected], "lobo_global_selection", selected, "leave-one-tower-anchor-block-out"))

        same_tower = other.loc[other["tower"].eq(tower)]
        selected_tower = select_best(same_tower) if not same_tower.empty else selected
        methods.append(method_rows(test, test[selected_tower], "lobo_same_tower_selection", selected_tower, "held-out block; other anchor blocks at same tower"))

        earlier = scored.loc[scored["anchor_year"].lt(year)]
        selected_forward = select_best(earlier) if not earlier.empty else B17_CONTROL
        methods.append(method_rows(test, test[selected_forward], "forward_global_selection", selected_forward, "strictly earlier anchor years; B17 control fallback"))

        earlier_tower = earlier.loc[earlier["tower"].eq(tower)]
        selected_forward_tower = select_best(earlier_tower) if not earlier_tower.empty else selected_forward
        methods.append(method_rows(test, test[selected_forward_tower], "forward_same_tower_selection", selected_forward_tower, "strictly earlier same-tower blocks; global-forward fallback"))

    combined = pd.concat(methods, ignore_index=True)
    keep = ["date", "lead_day", "bin", "tower", "anchor_year", "method", "validation_scheme", "selected_candidate", "prediction", "y_true", "y_gapfilled", "MAE_climatology"]
    combined[keep].to_csv(CROSSFIT_CHAINS_PATH, index=False)
    summary = []
    for method, group in combined.groupby("method", sort=False):
        for target, target_col in [("observed", "y_true"), ("gapfilled", "y_gapfilled")]:
            summary.append({"method": method, "target": target, **metrics(group, target_col=target_col)})
    result = pd.DataFrame(summary).sort_values(["target", "MASE"])
    result.to_csv(CROSSFIT_SUMMARY_PATH, index=False)
    return combined, result


def champion_breakdowns(candidate_scored):
    champion = candidate_scored.loc[candidate_scored["candidate"].eq(CHAMPION)].copy()
    rows = []
    for tower, group in champion.groupby("tower"):
        for target, col in [("observed", "y_true"), ("gapfilled", "y_gapfilled")]:
            rows.append({"tower": tower, "target": target, **metrics(group, target_col=col)})
    pd.DataFrame(rows).to_csv(TOWER_PATH, index=False)

    rows = []
    for (tower, year), group in champion.groupby(["tower", "anchor_year"]):
        for target, col in [("observed", "y_true"), ("gapfilled", "y_gapfilled")]:
            rows.append({"tower": tower, "anchor_year": year, "target": target, **metrics(group, target_col=col)})
    pd.DataFrame(rows).to_csv(BLOCK_PATH, index=False)

    rows = []
    for horizon, group in champion.groupby("bin", sort=False):
        for target, col in [("observed", "y_true"), ("gapfilled", "y_gapfilled")]:
            rows.append({"bin": horizon, "target": target, **metrics(group, target_col=col)})
    pd.DataFrame(rows).to_csv(HORIZON_PATH, index=False)

    # Use the event model's own pre-anchor, tower-specific p95 thresholds, plus
    # global retrospective p90/p95 definitions for comparison with B17.
    raw = pd.read_csv(SPIKE_PATH, parse_dates=["date"])
    raw = raw.loc[raw["experiment_id"].eq("B18S04") & raw["method"].eq("base_plus_fixed_excess_0.25")]
    raw = add_scoring(raw.rename(columns={"y_predict": "prediction"}))
    spike_rows = []
    for label, subset in [("spike", raw.loc[raw["y_is_spike"].eq(1)]), ("non_spike", raw.loc[raw["y_is_spike"].eq(0)])]:
        spike_rows.append({"definition": "preanchor tower-specific p95", "threshold": np.nan, "class": label, **metrics(subset)})
    observed = champion.loc[champion["y_true"].notna()].copy()
    for percentile in [90, 95]:
        threshold = float(observed["y_true"].quantile(percentile / 100))
        for label, subset in [("spike", observed.loc[observed["y_true"].ge(threshold)]), ("non_spike", observed.loc[observed["y_true"].lt(threshold)])]:
            spike_rows.append({"definition": f"pooled retrospective p{percentile}", "threshold": threshold, "class": label, **metrics(subset)})
    pd.DataFrame(spike_rows).to_csv(SPIKE_METRICS_PATH, index=False)
    return champion


def block_bootstrap(wide):
    scored = wide.loc[wide["y_true"].notna() & wide["MAE_climatology"].gt(0)].copy()
    comparators = [RECENCY, B17_CONTROL, TABICL, "B16-style TabPFN-TS v2 BASE+ALL", "B16 genuine-species TabPFN-TS v3"]
    keys = list(scored[["tower", "anchor_year"]].drop_duplicates().itertuples(index=False, name=None))
    rng = np.random.default_rng(1801)
    bootstrap_rows = []
    pairwise_rows = []
    for comparator in comparators:
        values = []
        for tower, year in keys:
            block = scored.loc[scored["tower"].eq(tower) & scored["anchor_year"].eq(year)]
            good = block[[CHAMPION, comparator, "y_true", "MAE_climatology"]].notna().all(axis=1)
            g = block.loc[good]
            new = np.abs(g["y_true"] - g[CHAMPION]) / g["MAE_climatology"]
            old = np.abs(g["y_true"] - g[comparator]) / g["MAE_climatology"]
            values.append((len(g), float(new.sum()), float(old.sum())))
            pairwise_rows.append({"tower": tower, "anchor_year": year, "comparator": comparator, "champion_MASE": float(new.mean()), "comparator_MASE": float(old.mean()), "delta_MASE": float(new.mean() - old.mean())})
        values = np.asarray(values, dtype=float)
        draws = np.empty(10_000)
        for index in range(len(draws)):
            sample = values[rng.integers(0, len(values), size=len(values))]
            draws[index] = sample[:, 1].sum() / sample[:, 0].sum() - sample[:, 2].sum() / sample[:, 0].sum()
        observed_delta = values[:, 1].sum() / values[:, 0].sum() - values[:, 2].sum() / values[:, 0].sum()
        bootstrap_rows.append({
            "comparator": comparator,
            "champion": CHAMPION,
            "observed_delta_MASE": observed_delta,
            "ci_2.5": float(np.quantile(draws, 0.025)),
            "ci_97.5": float(np.quantile(draws, 0.975)),
            "probability_champion_better": float(np.mean(draws < 0)),
            "resamples": len(draws),
            "blocks": len(keys),
            "champion_block_wins": int(sum(row["delta_MASE"] < 0 for row in pairwise_rows if row["comparator"] == comparator)),
        })
    result = pd.DataFrame(bootstrap_rows)
    result.to_csv(BOOTSTRAP_PATH, index=False)
    pd.DataFrame(pairwise_rows).to_csv(PAIRWISE_PATH, index=False)
    return result


def plot_chains(champion):
    CHAIN_FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA_PATH, low_memory=False)
    data["Datetime"] = pd.to_datetime(data["Datetime"], format="mixed")
    frames = {tower: data.loc[data["tower"].eq(tower)].set_index("Datetime").sort_index() for tower in TOWERS}
    for tower in TOWERS:
        for year in ANCHOR_YEARS:
            anchor = pd.Timestamp(f"{year}-12-16")
            dates = pd.date_range(anchor + pd.Timedelta(days=1), periods=N_DAYS, freq="D")
            window = pd.date_range(anchor - pd.Timedelta(days=30), dates[-1], freq="D")
            prediction = champion.loc[champion["tower"].eq(tower) & champion["anchor_year"].eq(year)].set_index("date")["prediction"].reindex(dates)
            fig, ax = plt.subplots(figsize=(13, 5))
            ax.plot(window, frames[tower]["y_gapfilled"].reindex(window), ":", color="gray", linewidth=1, label="Gap-filled FCH4")
            ax.plot(window, frames[tower]["y_observed"].reindex(window), "-", color="black", linewidth=1, label="Actual FCH4 (observed)")
            ax.plot(dates, prediction, "-", color="tab:blue", linewidth=1.5, label="TabPFN_v2 (predicted)")
            add_forecast_start_marker(ax, anchor)
            ax.set_title(f"Tower {tower}, anchor {anchor.date()}, model=TabPFN_v2")
            ax.set_ylabel("FCH4 (nmol m-2 s-1)")
            ax.legend(loc="upper right", fontsize=9)
            fig.tight_layout()
            fig.savefig(CHAIN_FIG_DIR / f"T{tower}_anchor{year}_TabPFN_v2.png", dpi=100)
            plt.close(fig)


def plot_summaries(registry, champion):
    SUMMARY_FIG_DIR.mkdir(parents=True, exist_ok=True)
    observed = registry.loc[registry["target"].eq("observed")].sort_values("MASE")
    fig, ax = plt.subplots(figsize=(11, 7))
    colors = ["tab:blue" if name == CHAMPION else "0.6" for name in observed["alias"]]
    ax.barh(observed["alias"], observed["MASE"], color=colors)
    ax.axvline(1, color="black", linestyle="--", linewidth=1)
    ax.invert_yaxis()
    ax.set_xlabel("Climatology-scaled MASE (lower is better)")
    ax.set_title("B18 forecasting candidate ranking")
    fig.tight_layout()
    fig.savefig(SUMMARY_FIG_DIR / "b18_candidate_mase_ranking.png", dpi=140)
    plt.close(fig)

    selected = [CHAMPION, RECENCY, B17_CONTROL, TABICL]
    rows = []
    for (candidate, tower), group in registry_source.loc[registry_source["candidate"].isin(selected)].groupby(["candidate", "tower"]):
        rows.append({"candidate": candidate, "tower": tower, "MASE": metrics(group)["MASE"]})
    pivot = pd.DataFrame(rows).pivot(index="tower", columns="candidate", values="MASE")
    fig, ax = plt.subplots(figsize=(11, 5.5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("MASE")
    ax.set_xlabel("Tower")
    ax.set_title("B18 performance by tower")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(SUMMARY_FIG_DIR / "b18_mase_by_tower.png", dpi=140)
    plt.close(fig)

    rows = []
    for (tower, year), group in champion.groupby(["tower", "anchor_year"]):
        if group["y_true"].notna().any():
            rows.append({"block": f"T{tower}-{year}", "MASE": metrics(group)["MASE"]})
    blocks = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(blocks["block"], blocks["MASE"], color="tab:blue")
    ax.axhline(1, color="black", linestyle="--", linewidth=1)
    ax.set_ylabel("MASE")
    ax.set_title("B18 champion MASE by evaluated tower-anchor block")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(SUMMARY_FIG_DIR / "b18_champion_mase_by_block.png", dpi=140)
    plt.close(fig)


def main():
    sources = {
        "direct": pd.read_csv(DIRECT_PATH, parse_dates=["date"]),
        "spike": pd.read_csv(SPIKE_PATH, parse_dates=["date"]),
        "screen": pd.read_csv(B17_SCREEN_PATH, parse_dates=["date"]),
    }
    rows = pd.concat([candidate_frame(spec, sources) for spec in CANDIDATE_SPECS], ignore_index=True)
    registry, scored = build_registry(rows)
    global registry_source
    registry_source = scored
    wide = build_wide(rows)
    _, adaptive_summary = tower_adaptive(wide)
    champion = champion_breakdowns(scored)
    bootstrap = block_bootstrap(wide)
    plot_chains(champion)
    plot_summaries(registry, champion)
    MANIFEST_PATH.write_text(
        json.dumps(
            {
                "champion": CHAMPION,
                "candidate_count": len(CANDIDATE_SPECS),
                "selection_shortlist": SELECTION_SHORTLIST,
                "primary_metric": "observed-target point-equivalent climatology MASE",
                "evaluated_blocks": 9,
                "raw_chain_figures": 15,
                "bootstrap": "10,000 resamples of tower-anchor blocks",
                "caution": "aggregate candidate selection and retrospective tower oracle are exploratory; held-out LOBO and forward schemes are the adaptation checks",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\nObserved-target candidate ranking")
    print(registry.loc[registry["target"].eq("observed"), ["alias", "n", "MASE", "MAE", "RMSE", "R2", "bias"]].to_string(index=False))
    print("\nTower-adaptive diagnostics")
    print(adaptive_summary.loc[adaptive_summary["target"].eq("observed")].to_string(index=False))
    print("\nBlock bootstrap")
    print(bootstrap.to_string(index=False))
    print(f"\nSaved 15 B15-style chain figures to {CHAIN_FIG_DIR}")


if __name__ == "__main__":
    main()
