"""B18 additive direct-foundation structure experiments.

Tests tower-normalised pooling, direct-model feature ablation, recency windows,
recency replication, seasonal experts, and leakage-safe antecedent driver
features for TabPFN v2 and TabICL v2.  Writes only ``b18_*`` artifacts.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(NOTEBOOK_DIR))

import models.recursive_rollout as rr
import B17_foundation_screen as b17s
import B17_direct_and_recursive_foundations as b17d

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass


RESULTS = ROOT / "results"
CHAINS_PATH = RESULTS / "b18_direct_structure_chains.csv"
ERRORS_PATH = RESULTS / "b18_direct_structure_errors.csv"
BIN_PATH = RESULTS / "b18_direct_structure_bin_metrics.csv"
SUMMARY_PATH = RESULTS / "b18_direct_structure_summary.csv"
MANIFEST_PATH = RESULTS / "b18_direct_structure_manifest.json"
BASELINE_PATH = RESULTS / "_today_climatology_baseline.csv"

TOWERS = [2, 4, 9]
ANCHOR_YEARS = [2018, 2019, 2020, 2021, 2022]
N_DAYS = 365
BIN_LABELS = b17s.BIN_LABELS
BIN_EDGES = b17s.BIN_EDGES
STATIC_FEATURES = b17d.TOWER_DUMMIES + b17d.TIME_FEATURES


def add_antecedent_features(frame):
    parts = []
    for tower, group in frame.sort_values(["tower", "Datetime"]).groupby("tower", sort=False):
        g = group.copy()
        precip_prev = g["fx_PRECIP_sum"].shift(1)
        for days in [3, 7, 14, 30]:
            g[f"b18_precip_prev_sum{days}"] = precip_prev.rolling(days, min_periods=1).sum()

        for prefix, source in [
            ("swc", "fx_SWC_mean"),
            ("ts", "fx_TS_mean"),
            ("ta", "fx_TA_mean"),
        ]:
            previous = g[source].shift(1)
            for days in [3, 7, 14, 30]:
                g[f"b18_{prefix}_prev_mean{days}"] = previous.rolling(days, min_periods=1).mean()
            for days in [7, 30]:
                g[f"b18_{prefix}_prev_std{days}"] = previous.rolling(days, min_periods=2).std()
            g[f"b18_{prefix}_delta1"] = previous - g[source].shift(2)
            g[f"b18_{prefix}_delta7"] = previous - g[source].shift(8)

        for prefix, source in [
            ("vpd", "fx_VPD_mean"),
            ("ppfd", "fx_PPFD_mean"),
            ("flow", "fx_flow_mean"),
        ]:
            previous = g[source].shift(1)
            for days in [7, 14]:
                g[f"b18_{prefix}_prev_mean{days}"] = previous.rolling(days, min_periods=1).mean()

        g["b18_air_freeze"] = g["fx_TA_min"].le(0).astype(float)
        g["b18_soil_cold"] = g["fx_TS_mean"].le(5).astype(float)
        g["b18_thaw_transition"] = (
            g["fx_TS_mean"].gt(5) & g["fx_TS_mean"].shift(1).le(5)
        ).astype(float)
        g["b18_precip_event"] = g["fx_PRECIP_sum"].gt(2).astype(float)
        g["b18_wet_warm"] = g["fx_SWC_mean"] * g["fx_TS_mean"]
        g["b18_rain_wet"] = g["fx_PRECIP_sum"] * g["fx_SWC_mean"]
        g["b18_grazing_wet"] = g["fx_grazing_active"] * g["fx_SWC_mean"]
        g["b18_grazing_warm"] = g["fx_grazing_active"] * g["fx_TS_mean"]
        parts.append(g)
    return pd.concat(parts, ignore_index=True).sort_values(["tower", "Datetime"])


def build_feature_sets(frame):
    base = b17s.build_configs(frame)
    engineered = [c for c in frame.columns if c.startswith("b18_")]
    all52 = base["BASE_ALL_52"]
    feature_sets = {
        "BASE_34": base["BASE_34"],
        "BASE_species_37": base["BASE_species_37"],
        "BASE_ALL_52": all52,
        "BASE_cattle_35": base["BASE_cattle_35"],
        "BASE_bodyweight_35": base["BASE_bodyweight_35"],
        "BASE_species_bodyweight_38": base["BASE_species_bodyweight_38"],
        "LEAN_species_13": base["LEAN_species_13"],
        "ALL_minus_management_46": [c for c in all52 if c not in b17s.FAMILIES["mgmt"]],
        "ALL_minus_flow_45": [c for c in all52 if c not in b17s.FAMILIES["flow"]],
        "ALL_antecedent": list(dict.fromkeys(all52 + engineered)),
        "BASE_antecedent": list(dict.fromkeys(base["BASE_34"] + engineered)),
    }
    for name, columns in feature_sets.items():
        missing = sorted(set(columns) - set(frame.columns))
        assert not missing, (name, missing)
        assert len(columns) == len(set(columns)), name
    return feature_sets, engineered


def experiment_specs():
    specs = []
    ablations = [
        "BASE_34",
        "BASE_species_37",
        "BASE_ALL_52",
        "BASE_cattle_35",
        "BASE_bodyweight_35",
        "BASE_species_bodyweight_38",
        "LEAN_species_13",
        "ALL_minus_management_46",
        "ALL_minus_flow_45",
    ]
    for feature_set in ablations:
        specs.append(
            {
                "model": "Direct_TabPFN_v2",
                "feature_set": feature_set,
                "target_norm": "raw",
                "training": "all_history",
            }
        )
    for feature_set, norm in [
        ("BASE_ALL_52", "tower_robust"),
        ("BASE_ALL_52", "tower_zscore"),
        ("BASE_ALL_52", "tower_month_robust"),
        ("BASE_species_37", "tower_robust"),
        ("BASE_34", "tower_robust"),
        ("ALL_antecedent", "raw"),
        ("ALL_antecedent", "tower_robust"),
        ("ALL_antecedent", "tower_month_robust"),
        ("BASE_antecedent", "tower_robust"),
    ]:
        specs.append(
            {
                "model": "Direct_TabPFN_v2",
                "feature_set": feature_set,
                "target_norm": norm,
                "training": "all_history",
            }
        )
    for days in [730, 1095, 1460, 1825]:
        for norm in ["raw", "tower_robust"]:
            specs.append(
                {
                    "model": "Direct_TabPFN_v2",
                    "feature_set": "BASE_ALL_52",
                    "target_norm": norm,
                    "training": f"window_{days}",
                }
            )
    for norm in ["raw", "tower_robust"]:
        specs.append(
            {
                "model": "Direct_TabPFN_v2",
                "feature_set": "BASE_ALL_52",
                "target_norm": norm,
                "training": "recency_replicated",
            }
        )
        specs.append(
            {
                "model": "Direct_TabPFN_v2",
                "feature_set": "BASE_ALL_52",
                "target_norm": norm,
                "training": "seasonal_experts",
            }
        )

    for feature_set, norm, training in [
        ("BASE_ALL_52", "raw", "all_history"),
        ("BASE_ALL_52", "tower_robust", "all_history"),
        ("BASE_ALL_52", "tower_month_robust", "all_history"),
        ("ALL_antecedent", "raw", "all_history"),
        ("ALL_antecedent", "tower_robust", "all_history"),
        ("BASE_ALL_52", "raw", "window_1460"),
        ("BASE_ALL_52", "tower_robust", "window_1460"),
        ("BASE_ALL_52", "tower_robust", "seasonal_experts"),
    ]:
        specs.append(
            {
                "model": "Direct_TabICLv2",
                "feature_set": feature_set,
                "target_norm": norm,
                "training": training,
            }
        )
    for index, spec in enumerate(specs):
        spec["experiment_id"] = f"B18D{index + 1:02d}"
    return specs


def append_csv(frame, path):
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def completed_keys():
    if not CHAINS_PATH.exists():
        return set()
    old = pd.read_csv(CHAINS_PATH, usecols=["anchor_year", "experiment_id"])
    return set(map(tuple, old.drop_duplicates().itertuples(index=False, name=None)))


def training_rows(frame, anchor, scheme):
    train = frame.loc[frame["Datetime"].le(anchor) & frame["y_observed"].notna()].copy()
    if scheme.startswith("window_"):
        days = int(scheme.rsplit("_", 1)[1])
        train = train.loc[train["Datetime"].ge(anchor - pd.Timedelta(days=days - 1))]
    if scheme == "recency_replicated":
        age = (anchor - train["Datetime"]).dt.days
        repeats = np.select(
            [age.le(365), age.le(730), age.le(1460)], [4, 3, 2], default=1
        ).astype(int)
        train = train.loc[train.index.repeat(repeats)].copy()
    if len(train) > 10_000:
        train = train.sample(10_000, random_state=42).sort_values("Datetime")
    return train


def season(series):
    month = pd.to_datetime(series).dt.month
    return pd.Series(
        np.select(
            [month.isin([12, 1, 2]), month.isin([3, 4, 5]), month.isin([6, 7, 8])],
            ["DJF", "MAM", "JJA"],
            default="SON",
        ),
        index=series.index,
    )


def fit_target_transform(train, mode):
    work = train[["tower", "Datetime", "y_observed"]].copy()
    work["month"] = work["Datetime"].dt.month
    by_tower = work.groupby("tower")["y_observed"]
    tower_median = by_tower.median().to_dict()
    tower_mean = by_tower.mean().to_dict()
    tower_std = by_tower.std().replace(0, 1).fillna(1).to_dict()
    quantiles = by_tower.quantile([0.25, 0.75]).unstack()
    tower_iqr = (quantiles[0.75] - quantiles[0.25]).replace(0, 1).fillna(1).to_dict()
    month_median = work.groupby(["tower", "month"])["y_observed"].median().to_dict()

    def center_scale(rows):
        if mode == "raw":
            center = np.zeros(len(rows))
            scale = np.ones(len(rows))
        elif mode == "tower_zscore":
            center = rows["tower"].map(tower_mean).to_numpy(float)
            scale = rows["tower"].map(tower_std).to_numpy(float)
        elif mode == "tower_robust":
            center = rows["tower"].map(tower_median).to_numpy(float)
            scale = rows["tower"].map(tower_iqr).to_numpy(float)
        elif mode == "tower_month_robust":
            keys = zip(rows["tower"], rows["Datetime"].dt.month)
            center = np.asarray(
                [month_median.get(key, tower_median[key[0]]) for key in keys], dtype=float
            )
            scale = rows["tower"].map(tower_iqr).to_numpy(float)
        else:
            raise ValueError(mode)
        return center, scale

    return center_scale


def make_model(model_name):
    if model_name == "Direct_TabPFN_v2":
        from tabpfn import TabPFNRegressor

        return TabPFNRegressor(
            model_path=rr.tabpfn_v2_model_config()["model_path"],
            n_estimators=8,
            random_state=137,
        )
    if model_name == "Direct_TabICLv2":
        from tabicl import TabICLRegressor

        return TabICLRegressor(n_estimators=8, random_state=42)
    raise ValueError(model_name)


def fit_predict_one(model_name, train, future, feature_cols, target_transform):
    imputer = SimpleImputer(strategy="mean")
    x_train = imputer.fit_transform(train[feature_cols])
    x_future = imputer.transform(future[feature_cols])
    center_train, scale_train = target_transform(train)
    y_train = (train["y_observed"].to_numpy() - center_train) / scale_train
    model = make_model(model_name)
    model.fit(x_train, y_train)
    prediction_norm = model.predict(x_future, output_type="median")
    center_future, scale_future = target_transform(future)
    return center_future + scale_future * np.asarray(prediction_norm)


def fit_predict(spec, train, future, feature_cols, target_transform):
    if spec["training"] != "seasonal_experts":
        return fit_predict_one(spec["model"], train, future, feature_cols, target_transform)

    train = train.copy()
    future = future.copy()
    train["b18_season"] = season(train["Datetime"])
    future["b18_season"] = season(future["Datetime"])
    prediction = pd.Series(index=future.index, dtype=float)
    for label in ["DJF", "MAM", "JJA", "SON"]:
        train_season = train.loc[train["b18_season"].eq(label)]
        future_season = future.loc[future["b18_season"].eq(label)]
        if future_season.empty:
            continue
        prediction.loc[future_season.index] = fit_predict_one(
            spec["model"], train_season, future_season, feature_cols, target_transform
        )
    return prediction.reindex(future.index).to_numpy()


def raw_rows(frame, future, spec, prediction, n_features):
    output = future[["Datetime", "tower", "y_observed", "y_gapfilled"]].copy()
    output = output.rename(
        columns={"Datetime": "date", "y_observed": "y_true"}
    )
    output["anchor_year"] = int(future["b18_anchor_year"].iloc[0])
    anchor = pd.Timestamp(f"{output['anchor_year'].iloc[0]}-12-16")
    output["lead_day"] = (output["date"] - anchor).dt.days
    output["model"] = spec["model"]
    output["experiment_id"] = spec["experiment_id"]
    output["feature_set"] = spec["feature_set"]
    output["target_norm"] = spec["target_norm"]
    output["training"] = spec["training"]
    output["protocol"] = "observed_history_known_future_fx"
    output["n_features"] = n_features
    output["y_predict"] = prediction
    return output[
        [
            "date",
            "lead_day",
            "tower",
            "anchor_year",
            "model",
            "experiment_id",
            "feature_set",
            "target_norm",
            "training",
            "protocol",
            "n_features",
            "y_predict",
            "y_true",
            "y_gapfilled",
        ]
    ]


def run(frame, feature_sets, specs):
    completed = completed_keys()
    for year in ANCHOR_YEARS:
        anchor = pd.Timestamp(f"{year}-12-16")
        dates = pd.date_range(anchor + pd.Timedelta(days=1), periods=N_DAYS, freq="D")
        future = frame.loc[frame["Datetime"].isin(dates)].copy()
        future["b18_anchor_year"] = year

        for spec in specs:
            key = (year, spec["experiment_id"])
            if key in completed:
                print(f"[resume-skip] {year} {spec['experiment_id']}")
                continue
            started = time.time()
            try:
                train = training_rows(frame, anchor, spec["training"])
                target_transform = fit_target_transform(train, spec["target_norm"])
                features = list(dict.fromkeys(feature_sets[spec["feature_set"]] + STATIC_FEATURES))
                prediction = fit_predict(spec, train, future, features, target_transform)
                append_csv(raw_rows(frame, future, spec, prediction, len(features)), CHAINS_PATH)
                print(
                    f"[{year}] {spec['experiment_id']} {spec['model']} {spec['feature_set']} "
                    f"norm={spec['target_norm']} training={spec['training']} n={len(train)} "
                    f"{time.time() - started:.1f}s",
                    flush=True,
                )
            except Exception as exc:
                append_csv(
                    pd.DataFrame(
                        [
                            {
                                "anchor_year": year,
                                **spec,
                                "error_type": type(exc).__name__,
                                "error": str(exc)[:1000],
                                "traceback": traceback.format_exc()[-4000:],
                            }
                        ]
                    ),
                    ERRORS_PATH,
                )
                print(f"[ERROR] {year} {spec['experiment_id']}: {exc}", flush=True)


def score():
    chains = pd.read_csv(CHAINS_PATH, parse_dates=["date"])
    chains["bin"] = pd.cut(
        chains["lead_day"], BIN_EDGES, labels=BIN_LABELS, include_lowest=True
    ).astype(str)
    baseline = pd.read_csv(BASELINE_PATH)
    chains = chains.merge(
        baseline[["tower", "anchor_year", "bin", "MAE_climatology"]],
        on=["tower", "anchor_year", "bin"],
        how="left",
    )
    keys = [
        "tower",
        "anchor_year",
        "model",
        "experiment_id",
        "feature_set",
        "target_norm",
        "training",
        "protocol",
        "n_features",
        "bin",
    ]
    rows = []
    for key, group in chains.groupby(keys, sort=False, observed=True):
        denom = group["MAE_climatology"].dropna()
        if denom.empty or denom.iloc[0] <= 0:
            continue
        denominator = float(denom.iloc[0])
        info = dict(zip(keys, key))
        for target, target_col in [("observed", "y_true"), ("gapfilled", "y_gapfilled")]:
            good = group[[target_col, "y_predict"]].notna().all(axis=1)
            g = group.loc[good]
            if g.empty:
                continue
            y = g[target_col].to_numpy()
            p = g["y_predict"].to_numpy()
            error = y - p
            total = float(np.sum((y - y.mean()) ** 2))
            mae = float(np.mean(np.abs(error)))
            rows.append(
                {
                    **info,
                    "target": target,
                    "n": len(g),
                    "MAE": mae,
                    "RMSE": float(np.sqrt(np.mean(error**2))),
                    "R2": float(1 - np.sum(error**2) / total) if total > 0 else np.nan,
                    "bias": float(np.mean(p - y)),
                    "MAE_climatology": denominator,
                    "MASE": mae / denominator,
                }
            )
    bins = pd.DataFrame(rows)
    bins.to_csv(BIN_PATH, index=False)

    skeys = [
        "model",
        "experiment_id",
        "feature_set",
        "target_norm",
        "training",
        "protocol",
        "n_features",
        "target",
    ]
    summaries = []
    for key, group in bins.groupby(skeys, sort=False):
        ok = group["MASE"].notna() & group["n"].gt(0)
        r2ok = group["R2"].notna() & group["n"].gt(0)
        summaries.append(
            {
                **dict(zip(skeys, key)),
                "n": int(group.loc[ok, "n"].sum()),
                "n_blocks": int(group[["tower", "anchor_year"]].drop_duplicates().shape[0]),
                "MASE": float(np.average(group.loc[ok, "MASE"], weights=group.loc[ok, "n"])),
                "MAE": float(np.average(group.loc[ok, "MAE"], weights=group.loc[ok, "n"])),
                "RMSE_bin_weighted": float(
                    np.average(group.loc[ok, "RMSE"], weights=group.loc[ok, "n"])
                ),
                "R2_bin_weighted": float(
                    np.average(group.loc[r2ok, "R2"], weights=group.loc[r2ok, "n"])
                )
                if r2ok.any()
                else np.nan,
                "bias": float(np.average(group.loc[ok, "bias"], weights=group.loc[ok, "n"])),
            }
        )
    summary = pd.DataFrame(summaries).sort_values(["target", "MASE"])
    summary.to_csv(SUMMARY_PATH, index=False)
    return summary


def main():
    frame = pd.read_csv(b17s.DATA_PATH, low_memory=False)
    frame["Datetime"] = pd.to_datetime(frame["Datetime"], format="mixed")
    frame = b17d.add_b17_features(frame)
    frame = add_antecedent_features(frame)
    feature_sets, engineered = build_feature_sets(frame)
    specs = experiment_specs()
    MANIFEST_PATH.write_text(
        json.dumps(
            {
                "experiment": "B18 direct foundation structure",
                "protocol": "observed history and known future fx; no future methane input",
                "engineered_features": engineered,
                "feature_sets": {name: columns for name, columns in feature_sets.items()},
                "experiments": specs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    started = time.time()
    run(frame, feature_sets, specs)
    summary = score()
    print(f"\nCompleted in {(time.time() - started) / 60:.1f} min")
    print(summary.loc[summary["target"].eq("observed")].head(35).to_string(index=False))


if __name__ == "__main__":
    main()
