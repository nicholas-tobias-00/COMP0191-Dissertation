"""B17 additive final evaluation, cross-fitting diagnostics, and figures.

Reads completed B17 raw chains (plus the existing B16 XGB chain as an ensemble
candidate) and writes only new ``b17_*`` tables and figures.  The selected raw
champion is never altered by this script.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
DATA_PATH = ROOT / "data" / "Hourly" / "forecast_daily_v3.csv"
BASELINE_PATH = RESULTS / "_today_climatology_baseline.csv"

SCREEN_PATH = RESULTS / "b17_foundation_screen_chains.csv"
CONTEXT_PATH = RESULTS / "b17_context_target_chains.csv"
DIRECT_PATH = RESULTS / "b17_direct_recursive_chains.csv"
TUNING_PATH = RESULTS / "b17_direct_tuning_chains.csv"
B16_TREE_PATH = RESULTS / "b16_recursive_rollout_v3_all_chains.csv"

REGISTRY_PATH = RESULTS / "b17_candidate_registry.csv"
ENSEMBLE_CHAINS_PATH = RESULTS / "b17_crossfit_ensemble_chains.csv"
ENSEMBLE_SUMMARY_PATH = RESULTS / "b17_crossfit_ensemble_summary.csv"
CHAMPION_TOWER_PATH = RESULTS / "b17_champion_metrics_by_tower.csv"
CHAMPION_BLOCK_PATH = RESULTS / "b17_champion_metrics_by_block.csv"
CHAMPION_HORIZON_PATH = RESULTS / "b17_champion_metrics_by_horizon.csv"
CHAMPION_SPIKE_PATH = RESULTS / "b17_champion_spike_metrics.csv"
BOOTSTRAP_PATH = RESULTS / "b17_champion_block_bootstrap.csv"
MANIFEST_PATH = RESULTS / "b17_evaluation_manifest.json"

CHAIN_FIG_DIR = RESULTS / "figures" / "b17_chains"
SUMMARY_FIG_DIR = RESULTS / "figures" / "b17_summary"

TOWERS = [2, 4, 9]
ANCHOR_YEARS = [2018, 2019, 2020, 2021, 2022]
N_DAYS = 365
BIN_LABELS = ["1-7", "8-30", "31-90", "91-180", "181-270", "271-365"]
BIN_EDGES = [0, 7, 30, 90, 180, 270, 365]
CHAMPION = "Direct TabPFN v2, pooled 52 fx + tower/time, seed 137"


CANDIDATE_SPECS = [
    {
        "alias": CHAMPION,
        "source": "tuning",
        "model": "Direct_TabPFN_v2",
        "config": "BASE_ALL_52",
        "variant": "pooled_time_seed137_raw_median",
        "selection_role": "selected raw B17 champion",
    },
    {
        "alias": "Direct TabPFN v2, seed 42",
        "source": "direct",
        "model": "Direct_TabPFN_v2",
        "config": "BASE_ALL_52",
        "variant": "pooled_time1_futureFCO20_direct_median",
        "selection_role": "direct control",
    },
    {
        "alias": "Direct TabPFN v2, signed-log target",
        "source": "tuning",
        "model": "Direct_TabPFN_v2",
        "config": "BASE_ALL_52",
        "variant": "pooled_time_signed_log1p_signed_log1p_median",
        "selection_role": "robust-target candidate",
    },
    {
        "alias": "Direct TabPFN v2, asinh target",
        "source": "tuning",
        "model": "Direct_TabPFN_v2",
        "config": "BASE_ALL_52",
        "variant": "pooled_time_asinh_scale10_asinh_scale10_median",
        "selection_role": "robust-target candidate",
    },
    {
        "alias": "Direct TabPFN v2, average-before-softmax",
        "source": "tuning",
        "model": "Direct_TabPFN_v2",
        "config": "BASE_ALL_52",
        "variant": "pooled_time_avg_before_softmax_raw_median",
        "selection_role": "inference candidate",
    },
    {
        "alias": "Direct TabPFN v2, solo towers",
        "source": "direct",
        "model": "Direct_TabPFN_v2",
        "config": "BASE_ALL_52",
        "variant": "solo_time0_futureFCO20_direct_median",
        "selection_role": "pooling control",
    },
    {
        "alias": "TabPFN-TS v2, BASE+ALL",
        "source": "screen",
        "model": "TabPFN_v2",
        "config": "BASE_ALL_52",
        "variant": None,
        "selection_role": "B16-style checkpoint control",
    },
    {
        "alias": "TabPFN-TS v3, BASE+species",
        "source": "screen",
        "model": "TabPFN_v3",
        "config": "BASE_species_37",
        "variant": None,
        "selection_role": "genuine 37-feature B16 control",
    },
    {
        "alias": "TabPFN-TS v3, bodyweight, 1460-day context",
        "source": "context",
        "model": "TabPFN_v3",
        "config": "BASE_bodyweight_35",
        "variant": "window_days_1460",
        "selection_role": "context candidate",
    },
    {
        "alias": "Direct TabICL v2, asinh target",
        "source": "tuning",
        "model": "Direct_TabICLv2",
        "config": "BASE_ALL_52",
        "variant": "pooled_time_asinh_scale10_asinh_scale10_median",
        "selection_role": "best B17 TabICL candidate",
    },
    {
        "alias": "TabICL-TS v2, 730-day context",
        "source": "context",
        "model": "TabICLv2",
        "config": "BASE_species_bodyweight_38",
        "variant": "window_days_730",
        "selection_role": "TabICL context candidate",
    },
    {
        "alias": "B16 XGB BASE+ALL",
        "source": "b16_tree",
        "model": "XGB",
        "config": "BASE+ALL",
        "variant": "BASE+ALL__XGB",
        "selection_role": "non-foundation diversity candidate",
    },
]


def load_sources():
    sources = {
        "screen": pd.read_csv(SCREEN_PATH, parse_dates=["date"]),
        "context": pd.read_csv(CONTEXT_PATH, parse_dates=["date"]),
        "direct": pd.read_csv(DIRECT_PATH, parse_dates=["date"]),
        "tuning": pd.read_csv(TUNING_PATH, parse_dates=["date"]),
        "b16_tree": pd.read_csv(B16_TREE_PATH, parse_dates=["date"]),
    }
    return sources


def candidate_frame(spec, sources):
    source = sources[spec["source"]]
    if spec["source"] == "b16_tree":
        selected = source.loc[source["config"].eq(spec["config"])].copy()
        selected = selected[
            ["date", "tower", "anchor_year", "y_true", "y_gapfilled", spec["variant"]]
        ].rename(columns={spec["variant"]: "prediction"})
    else:
        mask = source["model"].eq(spec["model"]) & source["config"].eq(spec["config"])
        if spec["variant"] is not None:
            mask &= source["variant"].eq(spec["variant"])
        selected = source.loc[
            mask, ["date", "tower", "anchor_year", "y_true", "y_gapfilled", "y_predict"]
        ].rename(columns={"y_predict": "prediction"})
    selected = selected.drop_duplicates(["tower", "anchor_year", "date"])
    if selected.empty:
        raise ValueError(f"No rows for candidate: {spec}")
    selected["candidate"] = spec["alias"]
    return selected


def add_scoring_columns(frame):
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"])
    anchors = pd.to_datetime(out["anchor_year"].astype(str) + "-12-16")
    out["lead_day"] = (out["date"] - anchors).dt.days
    out["bin"] = pd.cut(
        out["lead_day"], BIN_EDGES, labels=BIN_LABELS, include_lowest=True
    ).astype(str)
    baseline = pd.read_csv(BASELINE_PATH)
    return out.merge(
        baseline[["tower", "anchor_year", "bin", "MAE_climatology"]],
        on=["tower", "anchor_year", "bin"],
        how="left",
    )


def point_mase(group, prediction_col="prediction", target_col="y_true"):
    good = group[[target_col, prediction_col, "MAE_climatology"]].notna().all(axis=1)
    good &= group["MAE_climatology"].gt(0)
    g = group.loc[good]
    if g.empty:
        return np.nan
    return float(np.mean(np.abs(g[target_col] - g[prediction_col]) / g["MAE_climatology"]))


def standard_metrics(group, prediction_col="prediction", target_col="y_true"):
    good = group[[target_col, prediction_col, "MAE_climatology"]].notna().all(axis=1)
    good &= group["MAE_climatology"].gt(0)
    g = group.loc[good]
    if g.empty:
        return {"n": 0, "MASE": np.nan, "MAE": np.nan, "RMSE": np.nan, "R2": np.nan, "bias": np.nan}
    y = g[target_col].to_numpy()
    p = g[prediction_col].to_numpy()
    error = y - p
    denominator = float(np.sum((y - y.mean()) ** 2))
    return {
        "n": len(g),
        "MASE": point_mase(g, prediction_col, target_col),
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "R2": float(1 - np.sum(error**2) / denominator) if denominator > 0 else np.nan,
        "bias": float(np.mean(p - y)),
    }


def build_wide(candidate_rows):
    key = ["tower", "anchor_year", "date"]
    champion = candidate_rows.loc[candidate_rows["candidate"].eq(CHAMPION)].copy()
    wide = champion[key + ["y_true", "y_gapfilled"]].drop_duplicates(key)
    for candidate, group in candidate_rows.groupby("candidate", sort=False):
        values = group[key + ["prediction"]].drop_duplicates(key).rename(
            columns={"prediction": candidate}
        )
        wide = wide.merge(values, on=key, how="left")
    return add_scoring_columns(wide)


def evaluate_registry(candidate_rows):
    scored = add_scoring_columns(candidate_rows)
    records = []
    for spec in CANDIDATE_SPECS:
        group = scored.loc[scored["candidate"].eq(spec["alias"])]
        for target, target_col in [("observed", "y_true"), ("gapfilled", "y_gapfilled")]:
            records.append(
                {
                    **spec,
                    "target": target,
                    **standard_metrics(group, target_col=target_col),
                }
            )
    registry = pd.DataFrame(records).sort_values(["target", "MASE"])
    registry.to_csv(REGISTRY_PATH, index=False)
    return registry, scored


def best_candidate(training, candidates):
    scores = {
        candidate: point_mase(training, prediction_col=candidate)
        for candidate in candidates
        if training[candidate].notna().any()
    }
    return min(scores, key=scores.get)


def best_pair_alpha(training, left, right, maximum=1.0):
    usable = training[["y_true", "MAE_climatology", left, right]].notna().all(axis=1)
    usable &= training["MAE_climatology"].gt(0)
    tr = training.loc[usable]
    grid = np.linspace(0, maximum, 101)
    losses = [
        point_mase(
            tr.assign(_prediction=(1 - alpha) * tr[left] + alpha * tr[right]),
            prediction_col="_prediction",
        )
        for alpha in grid
    ]
    return float(grid[int(np.nanargmin(losses))])


def crossfit_predictions(wide):
    scored = wide.loc[
        wide["y_true"].notna() & wide["MAE_climatology"].gt(0)
    ].copy()
    candidate_names = [spec["alias"] for spec in CANDIDATE_SPECS]
    selection_candidates = [name for name in candidate_names if name != "B16 XGB BASE+ALL"]
    methods = []

    raw = scored.copy()
    raw["prediction"] = raw[CHAMPION]
    raw["method"] = "raw_champion"
    raw["selected_candidate"] = CHAMPION
    raw["blend_alpha"] = 0.0
    raw["validation_scheme"] = "none; raw fixed-origin model"
    methods.append(raw)

    bag = scored.copy()
    bag_cols = [
        CHAMPION,
        "Direct TabPFN v2, seed 42",
        "Direct TabPFN v2, signed-log target",
        "Direct TabPFN v2, asinh target",
        "Direct TabPFN v2, average-before-softmax",
    ]
    bag["prediction"] = bag[bag_cols].mean(axis=1)
    bag["method"] = "fixed_direct_pfn_robust_mean"
    bag["selected_candidate"] = "equal mean of five predeclared direct TabPFN variants"
    bag["blend_alpha"] = np.nan
    bag["validation_scheme"] = "fixed equal weights"
    methods.append(bag)

    for tower, year in scored[["tower", "anchor_year"]].drop_duplicates().itertuples(index=False):
        test_mask = scored["tower"].eq(tower) & scored["anchor_year"].eq(year)
        train = scored.loc[~test_mask]
        test = scored.loc[test_mask].copy()

        selected = best_candidate(train, selection_candidates)
        selection = test.copy()
        selection["prediction"] = selection[selected]
        selection["method"] = "lobo_model_selection"
        selection["selected_candidate"] = selected
        selection["blend_alpha"] = np.nan
        selection["validation_scheme"] = "leave-one-tower-anchor-block-out"
        methods.append(selection)

        same_tower_train = train.loc[train["tower"].eq(tower)]
        if same_tower_train["anchor_year"].nunique() >= 1:
            selected_tower = best_candidate(same_tower_train, selection_candidates)
        else:
            selected_tower = selected
        tower_selection = test.copy()
        tower_selection["prediction"] = tower_selection[selected_tower]
        tower_selection["method"] = "lobo_same_tower_model_selection"
        tower_selection["selected_candidate"] = selected_tower
        tower_selection["blend_alpha"] = np.nan
        tower_selection["validation_scheme"] = "other anchor blocks at same tower; global fallback"
        methods.append(tower_selection)

        alpha = best_pair_alpha(train, CHAMPION, "B16 XGB BASE+ALL", maximum=0.5)
        blend = test.copy()
        blend["prediction"] = (1 - alpha) * blend[CHAMPION] + alpha * blend["B16 XGB BASE+ALL"]
        blend["method"] = "lobo_champion_xgb_blend"
        blend["selected_candidate"] = "champion + B16 XGB"
        blend["blend_alpha"] = alpha
        blend["validation_scheme"] = "leave-one-tower-anchor-block-out"
        methods.append(blend)

        earlier = scored.loc[scored["anchor_year"].lt(year)]
        forward = test.copy()
        if earlier.empty:
            forward_selected = CHAMPION
        else:
            forward_selected = best_candidate(earlier, selection_candidates)
        forward["prediction"] = forward[forward_selected]
        forward["method"] = "forward_year_model_selection"
        forward["selected_candidate"] = forward_selected
        forward["blend_alpha"] = np.nan
        forward["validation_scheme"] = "strictly earlier anchor years; champion fallback for 2018"
        methods.append(forward)

    combined = pd.concat(methods, ignore_index=True)
    keep = [
        "date",
        "lead_day",
        "bin",
        "tower",
        "anchor_year",
        "method",
        "validation_scheme",
        "selected_candidate",
        "blend_alpha",
        "prediction",
        "y_true",
        "y_gapfilled",
        "MAE_climatology",
    ]
    combined = combined[keep]
    combined.to_csv(ENSEMBLE_CHAINS_PATH, index=False)

    summary_rows = []
    for method, group in combined.groupby("method", sort=False):
        for target, target_col in [("observed", "y_true"), ("gapfilled", "y_gapfilled")]:
            summary_rows.append(
                {"method": method, "target": target, **standard_metrics(group, target_col=target_col)}
            )
    summary = pd.DataFrame(summary_rows).sort_values(["target", "MASE"])
    summary.to_csv(ENSEMBLE_SUMMARY_PATH, index=False)
    return combined, summary


def champion_breakdowns(champion_scored):
    champion = champion_scored.loc[champion_scored["candidate"].eq(CHAMPION)].copy()
    tower_rows = []
    for tower, group in champion.groupby("tower"):
        for target, target_col in [("observed", "y_true"), ("gapfilled", "y_gapfilled")]:
            tower_rows.append({"tower": tower, "target": target, **standard_metrics(group, target_col=target_col)})
    pd.DataFrame(tower_rows).to_csv(CHAMPION_TOWER_PATH, index=False)

    block_rows = []
    for (tower, year), group in champion.groupby(["tower", "anchor_year"]):
        for target, target_col in [("observed", "y_true"), ("gapfilled", "y_gapfilled")]:
            block_rows.append(
                {"tower": tower, "anchor_year": year, "target": target, **standard_metrics(group, target_col=target_col)}
            )
    pd.DataFrame(block_rows).to_csv(CHAMPION_BLOCK_PATH, index=False)

    horizon_rows = []
    for horizon, group in champion.groupby("bin", sort=False):
        for target, target_col in [("observed", "y_true"), ("gapfilled", "y_gapfilled")]:
            horizon_rows.append({"bin": horizon, "target": target, **standard_metrics(group, target_col=target_col)})
    pd.DataFrame(horizon_rows).to_csv(CHAMPION_HORIZON_PATH, index=False)

    observed = champion.loc[champion["y_true"].notna()].copy()
    spike_rows = []
    for percentile in [90, 95]:
        threshold = float(np.nanpercentile(observed["y_true"], percentile))
        for label, subset in [
            ("spike", observed.loc[observed["y_true"].ge(threshold)]),
            ("non_spike", observed.loc[observed["y_true"].lt(threshold)]),
        ]:
            spike_rows.append(
                {
                    "percentile": percentile,
                    "threshold": threshold,
                    "class": label,
                    **standard_metrics(subset),
                }
            )
    pd.DataFrame(spike_rows).to_csv(CHAMPION_SPIKE_PATH, index=False)
    return champion


def block_bootstrap(wide):
    scored = wide.loc[wide["y_true"].notna() & wide["MAE_climatology"].gt(0)].copy()
    comparisons = [
        ("TabPFN-TS v2, BASE+ALL", "B16-style best checkpoint"),
        ("TabPFN-TS v3, BASE+species", "genuine B16 BASE+species"),
        ("Direct TabPFN v2, seed 42", "new-direct default seed"),
    ]
    block_keys = list(scored[["tower", "anchor_year"]].drop_duplicates().itertuples(index=False, name=None))
    rng = np.random.default_rng(1701)
    rows = []
    for comparator, label in comparisons:
        block_values = []
        for tower, year in block_keys:
            group = scored.loc[scored["tower"].eq(tower) & scored["anchor_year"].eq(year)]
            good = group[[CHAMPION, comparator, "y_true", "MAE_climatology"]].notna().all(axis=1)
            g = group.loc[good]
            champion_loss = np.abs(g["y_true"] - g[CHAMPION]) / g["MAE_climatology"]
            comparator_loss = np.abs(g["y_true"] - g[comparator]) / g["MAE_climatology"]
            block_values.append((len(g), float(champion_loss.sum()), float(comparator_loss.sum())))
        values = np.asarray(block_values)
        draws = []
        for _ in range(10_000):
            sampled = values[rng.integers(0, len(values), size=len(values))]
            new_score = sampled[:, 1].sum() / sampled[:, 0].sum()
            old_score = sampled[:, 2].sum() / sampled[:, 0].sum()
            draws.append(new_score - old_score)
        draws = np.asarray(draws)
        observed_delta = values[:, 1].sum() / values[:, 0].sum() - values[:, 2].sum() / values[:, 0].sum()
        rows.append(
            {
                "comparison": label,
                "comparator": comparator,
                "champion": CHAMPION,
                "observed_delta_MASE": observed_delta,
                "ci_2.5": float(np.quantile(draws, 0.025)),
                "ci_97.5": float(np.quantile(draws, 0.975)),
                "probability_champion_better": float(np.mean(draws < 0)),
                "resamples": 10_000,
                "blocks": len(block_keys),
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(BOOTSTRAP_PATH, index=False)
    return result


def plot_chains(champion):
    CHAIN_FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA_PATH, low_memory=False)
    data["Datetime"] = pd.to_datetime(data["Datetime"], format="mixed")
    tower_frames = {
        tower: data.loc[data["tower"].eq(tower)].set_index("Datetime").sort_index()
        for tower in TOWERS
    }
    for tower in TOWERS:
        dft = tower_frames[tower]
        for year in ANCHOR_YEARS:
            anchor = pd.Timestamp(f"{year}-12-16")
            dates = pd.date_range(anchor + pd.Timedelta(days=1), periods=N_DAYS, freq="D")
            window = pd.date_range(anchor - pd.Timedelta(days=30), dates[-1], freq="D")
            pred = (
                champion.loc[
                    champion["tower"].eq(tower) & champion["anchor_year"].eq(year)
                ]
                .set_index("date")["prediction"]
                .reindex(dates)
            )
            fig, ax = plt.subplots(figsize=(13, 5))
            ax.plot(
                window,
                dft["y_gapfilled"].reindex(window),
                color="black",
                linestyle=":",
                linewidth=1.0,
                alpha=0.75,
                label="Gap-filled FCH4",
            )
            ax.plot(
                window,
                dft["y_observed"].reindex(window),
                color="black",
                linestyle="-",
                linewidth=1.0,
                label="Observed FCH4",
            )
            ax.plot(dates, pred, color="tab:blue", linewidth=1.5, label="B17 prediction")
            ax.axvline(anchor, color="0.35", linestyle="--", linewidth=1.0, label="Forecast start")
            ax.set_title(f"B17 direct TabPFN v2 — Tower {tower}, anchor {anchor.date()}")
            ax.set_ylabel("FCH4 (nmol m$^{-2}$ s$^{-1}$)")
            ax.legend(loc="upper right", fontsize=9)
            fig.tight_layout()
            fig.savefig(
                CHAIN_FIG_DIR / f"T{tower}_anchor{year}_B17_Direct_TabPFN_v2.png",
                dpi=120,
            )
            plt.close(fig)


def plot_summaries(registry, champion):
    SUMMARY_FIG_DIR.mkdir(parents=True, exist_ok=True)
    observed = registry.loc[registry["target"].eq("observed")].sort_values("MASE").head(12)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["tab:blue" if alias == CHAMPION else "0.55" for alias in observed["alias"]]
    ax.barh(observed["alias"], observed["MASE"], color=colors)
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1)
    ax.invert_yaxis()
    ax.set_xlabel("Climatology-scaled MASE (lower is better)")
    ax.set_title("B17 foundation-model candidate ranking")
    fig.tight_layout()
    fig.savefig(SUMMARY_FIG_DIR / "b17_candidate_mase_ranking.png", dpi=140)
    plt.close(fig)

    tower_aliases = [
        CHAMPION,
        "TabPFN-TS v2, BASE+ALL",
        "TabPFN-TS v3, BASE+species",
        "Direct TabICL v2, asinh target",
    ]
    rows = []
    for alias in tower_aliases:
        subset = champion if alias == CHAMPION else None
        if subset is None:
            continue
        for tower, group in subset.groupby("tower"):
            rows.append({"candidate": alias, "tower": tower, "MASE": point_mase(group)})
    all_sources = load_sources()
    candidate_rows = pd.concat(
        [candidate_frame(spec, all_sources) for spec in CANDIDATE_SPECS if spec["alias"] in tower_aliases],
        ignore_index=True,
    )
    candidate_rows = add_scoring_columns(candidate_rows)
    rows = []
    for (alias, tower), group in candidate_rows.groupby(["candidate", "tower"]):
        rows.append({"candidate": alias, "tower": tower, "MASE": point_mase(group)})
    table = pd.DataFrame(rows)
    pivot = table.pivot(index="tower", columns="candidate", values="MASE")
    fig, ax = plt.subplots(figsize=(11, 5.5))
    pivot.plot(kind="bar", ax=ax, color=["tab:blue", "tab:orange", "0.45", "0.7"])
    ax.set_ylabel("MASE")
    ax.set_xlabel("Tower")
    ax.set_title("B17 performance by tower")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(SUMMARY_FIG_DIR / "b17_mase_by_tower.png", dpi=140)
    plt.close(fig)

    blocks = []
    for (tower, year), group in champion.groupby(["tower", "anchor_year"]):
        blocks.append({"block": f"T{tower}-{year}", "MASE": point_mase(group)})
    block_frame = pd.DataFrame(blocks)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(block_frame["block"], block_frame["MASE"], color="tab:blue")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_ylabel("MASE")
    ax.set_title("B17 champion MASE by evaluated tower–anchor block")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(SUMMARY_FIG_DIR / "b17_champion_mase_by_block.png", dpi=140)
    plt.close(fig)


def main():
    sources = load_sources()
    frames = [candidate_frame(spec, sources) for spec in CANDIDATE_SPECS]
    candidate_rows = pd.concat(frames, ignore_index=True)
    registry, candidate_scored = evaluate_registry(candidate_rows)
    wide = build_wide(candidate_rows)
    crossfit_chains, crossfit_summary = crossfit_predictions(wide)
    champion = champion_breakdowns(candidate_scored)
    bootstrap = block_bootstrap(wide)
    plot_chains(champion)
    plot_summaries(registry, champion)
    MANIFEST_PATH.write_text(
        json.dumps(
            {
                "champion": CHAMPION,
                "candidate_count": len(CANDIDATE_SPECS),
                "chain_figures": 15,
                "primary_metric": "observed-target point-equivalent bin-weighted climatology MASE",
                "crossfit_note": "cross-fitted ensembles are diagnostics and do not replace the raw champion unless they improve held-out performance",
                "bootstrap": "10,000 resamples of the nine tower-anchor blocks",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\nCandidate ranking")
    print(registry.loc[registry["target"].eq("observed")][["alias", "n", "MASE", "MAE", "RMSE", "R2", "bias"]].to_string(index=False))
    print("\nCross-fit diagnostics")
    print(crossfit_summary.loc[crossfit_summary["target"].eq("observed")].to_string(index=False))
    print("\nBlock-bootstrap comparisons")
    print(bootstrap.to_string(index=False))
    print(f"\nSaved 15 chain figures to {CHAIN_FIG_DIR}")


if __name__ == "__main__":
    main()
