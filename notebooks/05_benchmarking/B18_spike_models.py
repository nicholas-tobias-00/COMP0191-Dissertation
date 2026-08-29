"""B18 additive calibrated two-stage spike forecasting experiments.

Each anchor defines spikes from pre-anchor observed FCH4 only.  A classifier
predicts event probability and separate foundation regressors predict ordinary
and spike magnitudes.  Gate thresholds and excess-correction strength are chosen
from a trailing pre-anchor calibration year when enough data exist.  No future
FCH4 is used as an input or for calibration.
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
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(NOTEBOOK_DIR))

import models.recursive_rollout as rr
import B17_foundation_screen as b17s
import B17_direct_and_recursive_foundations as b17d
import B18_direct_structure as b18d

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass


RESULTS = ROOT / "results"
CHAINS_PATH = RESULTS / "b18_spike_model_chains.csv"
ERRORS_PATH = RESULTS / "b18_spike_model_errors.csv"
BIN_PATH = RESULTS / "b18_spike_model_bin_metrics.csv"
SUMMARY_PATH = RESULTS / "b18_spike_model_summary.csv"
CLASSIFICATION_PATH = RESULTS / "b18_spike_classification_metrics.csv"
MANIFEST_PATH = RESULTS / "b18_spike_model_manifest.json"
BASELINE_PATH = RESULTS / "_today_climatology_baseline.csv"

TOWERS = [2, 4, 9]
ANCHOR_YEARS = [2018, 2019, 2020, 2021, 2022]
N_DAYS = 365
BIN_LABELS = b17s.BIN_LABELS
BIN_EDGES = b17s.BIN_EDGES
STATIC_FEATURES = b17d.TOWER_DUMMIES + b17d.TIME_FEATURES
METHODS = [
    "base_regressor",
    "hard_gate_0.5",
    "hard_gate_calibrated",
    "soft_probability_mix",
    "base_plus_fixed_excess_0.25",
    "base_plus_calibrated_excess",
]


def experiment_specs():
    specs = [
        ("TabPFN", "BASE_ALL_52", 90, False, "raw"),
        ("TabPFN", "BASE_ALL_52", 90, True, "raw"),
        ("TabPFN", "ALL_antecedent", 90, False, "raw"),
        ("TabPFN", "BASE_ALL_52", 95, False, "raw"),
        ("TabPFN", "BASE_ALL_52", 90, False, "asinh10"),
        ("TabICL", "BASE_ALL_52", 90, False, "raw"),
        ("XGBGate_TabPFNReg", "BASE_ALL_52", 90, False, "raw"),
        ("XGBGate_TabPFNReg", "ALL_antecedent", 90, False, "raw"),
    ]
    return [
        {
            "experiment_id": f"B18S{index + 1:02d}",
            "foundation": foundation,
            "feature_set": feature_set,
            "spike_percentile": percentile,
            "balance_probabilities": balance,
            "magnitude_transform": transform,
        }
        for index, (foundation, feature_set, percentile, balance, transform) in enumerate(specs)
    ]


def append_csv(frame, path):
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def completed_keys():
    if not CHAINS_PATH.exists():
        return set()
    old = pd.read_csv(CHAINS_PATH, usecols=["anchor_year", "experiment_id", "method"])
    return set(map(tuple, old.drop_duplicates().itertuples(index=False, name=None)))


def threshold_map(train, percentile):
    global_threshold = float(np.nanpercentile(train["y_observed"], percentile))
    tower_thresholds = (
        train.groupby("tower")["y_observed"].quantile(percentile / 100.0).to_dict()
    )
    return tower_thresholds, global_threshold


def event_labels(rows, tower_thresholds, global_threshold):
    thresholds = rows["tower"].map(tower_thresholds).fillna(global_threshold).to_numpy(float)
    labels = rows["y_observed"].to_numpy() >= thresholds
    return labels.astype(int), thresholds


def transform_y(values, transform):
    values = np.asarray(values, dtype=float)
    if transform == "raw":
        return values
    if transform == "asinh10":
        return np.arcsinh(values / 10.0)
    raise ValueError(transform)


def inverse_y(values, transform):
    values = np.asarray(values, dtype=float)
    if transform == "raw":
        return values
    if transform == "asinh10":
        return 10.0 * np.sinh(values)
    raise ValueError(transform)


def make_regressor(foundation):
    if foundation in {"TabPFN", "XGBGate_TabPFNReg"}:
        from tabpfn import TabPFNRegressor

        return TabPFNRegressor(
            model_path=rr.tabpfn_v2_model_config()["model_path"],
            n_estimators=8,
            random_state=137,
        )
    if foundation == "TabICL":
        from tabicl import TabICLRegressor

        return TabICLRegressor(n_estimators=8, random_state=42)
    raise ValueError(foundation)


def make_classifier(spec):
    if spec["foundation"] == "TabPFN":
        from tabpfn import TabPFNClassifier
        from tabpfn.model_loading import prepend_cache_path

        return TabPFNClassifier(
            model_path=prepend_cache_path("tabpfn-v2-classifier.ckpt"),
            n_estimators=8,
            random_state=137,
            balance_probabilities=spec["balance_probabilities"],
        )
    if spec["foundation"] == "TabICL":
        from tabicl import TabICLClassifier

        return TabICLClassifier(n_estimators=8, random_state=42)
    if spec["foundation"] == "XGBGate_TabPFNReg":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=300,
            max_depth=2,
            learning_rate=0.03,
            min_child_weight=8,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
    raise ValueError(spec["foundation"])


def median_predict(model, x):
    return np.asarray(model.predict(x, output_type="median"))


def fit_bundle(train, predict_rows, features, spec):
    tower_thresholds, global_threshold = threshold_map(train, spec["spike_percentile"])
    labels, _ = event_labels(train, tower_thresholds, global_threshold)
    if labels.min() == labels.max():
        raise ValueError("Spike labels contain only one class")

    imputer = SimpleImputer(strategy="mean")
    x_train = imputer.fit_transform(train[features])
    x_predict = imputer.transform(predict_rows[features])

    classifier = make_classifier(spec)
    classifier.fit(x_train, labels)
    probability = np.asarray(classifier.predict_proba(x_predict))[:, 1]

    base_model = make_regressor(spec["foundation"])
    base_model.fit(x_train, transform_y(train["y_observed"], spec["magnitude_transform"]))
    base_prediction = inverse_y(
        median_predict(base_model, x_predict), spec["magnitude_transform"]
    )

    normal = train.loc[labels == 0]
    spike = train.loc[labels == 1]
    if len(spike) < 10:
        raise ValueError(f"Only {len(spike)} spike rows")
    normal_model = make_regressor(spec["foundation"])
    spike_model = make_regressor(spec["foundation"])
    normal_model.fit(
        imputer.transform(normal[features]),
        transform_y(normal["y_observed"], spec["magnitude_transform"]),
    )
    spike_model.fit(
        imputer.transform(spike[features]),
        transform_y(spike["y_observed"], spec["magnitude_transform"]),
    )
    normal_prediction = inverse_y(
        median_predict(normal_model, x_predict), spec["magnitude_transform"]
    )
    spike_prediction = inverse_y(
        median_predict(spike_model, x_predict), spec["magnitude_transform"]
    )
    predict_thresholds = (
        predict_rows["tower"].map(tower_thresholds).fillna(global_threshold).to_numpy(float)
    )
    return {
        "probability": probability,
        "base": base_prediction,
        "normal": normal_prediction,
        "spike": spike_prediction,
        "threshold": predict_thresholds,
        "tower_thresholds": tower_thresholds,
        "global_threshold": global_threshold,
    }


def calibration_parameters(train, anchor, features, spec):
    calibration_start = anchor - pd.Timedelta(days=365)
    subtrain = train.loc[train["Datetime"].lt(calibration_start)]
    calibration = train.loc[train["Datetime"].ge(calibration_start)]
    if len(subtrain) < 200 or len(calibration) < 40:
        return 0.5, 0.0, len(subtrain), len(calibration)
    try:
        bundle = fit_bundle(subtrain, calibration, features, spec)
    except Exception:
        return 0.5, 0.0, len(subtrain), len(calibration)

    y = calibration["y_observed"].to_numpy()
    gate_grid = np.linspace(0.1, 0.9, 17)
    gate_losses = []
    for gate in gate_grid:
        prediction = np.where(
            bundle["probability"] >= gate, bundle["spike"], bundle["normal"]
        )
        gate_losses.append(np.mean(np.abs(y - prediction)))
    best_gate = float(gate_grid[int(np.argmin(gate_losses))])

    excess = np.maximum(bundle["spike"] - bundle["normal"], 0)
    lambda_grid = np.linspace(0, 1, 21)
    lambda_losses = [
        np.mean(np.abs(y - (bundle["base"] + weight * bundle["probability"] * excess)))
        for weight in lambda_grid
    ]
    best_lambda = float(lambda_grid[int(np.argmin(lambda_losses))])
    return best_gate, best_lambda, len(subtrain), len(calibration)


def predictions_from_bundle(bundle, gate, correction_weight):
    probability = bundle["probability"]
    excess = np.maximum(bundle["spike"] - bundle["normal"], 0)
    return {
        "base_regressor": bundle["base"],
        "hard_gate_0.5": np.where(probability >= 0.5, bundle["spike"], bundle["normal"]),
        "hard_gate_calibrated": np.where(
            probability >= gate, bundle["spike"], bundle["normal"]
        ),
        "soft_probability_mix": (1 - probability) * bundle["normal"]
        + probability * bundle["spike"],
        "base_plus_fixed_excess_0.25": bundle["base"] + 0.25 * probability * excess,
        "base_plus_calibrated_excess": bundle["base"]
        + correction_weight * probability * excess,
    }


def run(frame, feature_sets, specs):
    completed = completed_keys()
    for year in ANCHOR_YEARS:
        anchor = pd.Timestamp(f"{year}-12-16")
        dates = pd.date_range(anchor + pd.Timedelta(days=1), periods=N_DAYS, freq="D")
        train = frame.loc[frame["Datetime"].le(anchor) & frame["y_observed"].notna()].copy()
        future = frame.loc[frame["Datetime"].isin(dates)].copy()

        for spec in specs:
            if all((year, spec["experiment_id"], method) in completed for method in METHODS):
                continue
            started = time.time()
            features = list(
                dict.fromkeys(feature_sets[spec["feature_set"]] + STATIC_FEATURES)
            )
            try:
                gate, correction_weight, subtrain_n, calibration_n = calibration_parameters(
                    train, anchor, features, spec
                )
                bundle = fit_bundle(train, future, features, spec)
                predictions = predictions_from_bundle(bundle, gate, correction_weight)
                for method, prediction in predictions.items():
                    if (year, spec["experiment_id"], method) in completed:
                        continue
                    output = future[
                        ["Datetime", "tower", "y_observed", "y_gapfilled"]
                    ].copy()
                    output = output.rename(
                        columns={"Datetime": "date", "y_observed": "y_true"}
                    )
                    output["lead_day"] = (output["date"] - anchor).dt.days
                    output["anchor_year"] = year
                    output["experiment_id"] = spec["experiment_id"]
                    output["foundation"] = spec["foundation"]
                    output["feature_set"] = spec["feature_set"]
                    output["spike_percentile"] = spec["spike_percentile"]
                    output["balance_probabilities"] = spec["balance_probabilities"]
                    output["magnitude_transform"] = spec["magnitude_transform"]
                    output["method"] = method
                    output["protocol"] = "preanchor_calibrated_spike_model_known_future_fx"
                    output["n_features"] = len(features)
                    output["gate_threshold"] = gate
                    output["correction_weight"] = correction_weight
                    output["calibration_train_n"] = subtrain_n
                    output["calibration_n"] = calibration_n
                    output["p_spike"] = bundle["probability"]
                    output["spike_threshold"] = bundle["threshold"]
                    output["base_prediction"] = bundle["base"]
                    output["normal_prediction"] = bundle["normal"]
                    output["spike_prediction"] = bundle["spike"]
                    output["y_predict"] = prediction
                    output["y_is_spike"] = np.where(
                        output["y_true"].notna(),
                        (output["y_true"] >= output["spike_threshold"]).astype(float),
                        np.nan,
                    )
                    append_csv(output, CHAINS_PATH)
                print(
                    f"[{year}] {spec['experiment_id']} {spec['foundation']} {spec['feature_set']} "
                    f"p{spec['spike_percentile']} gate={gate:.2f} lambda={correction_weight:.2f} "
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
        "experiment_id",
        "foundation",
        "feature_set",
        "spike_percentile",
        "balance_probabilities",
        "magnitude_transform",
        "method",
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
        "experiment_id",
        "foundation",
        "feature_set",
        "spike_percentile",
        "balance_probabilities",
        "magnitude_transform",
        "method",
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

    class_rows = []
    probability_rows = chains.drop_duplicates(
        ["date", "tower", "anchor_year", "experiment_id"]
    )
    for experiment_id, group in probability_rows.groupby("experiment_id"):
        good = group[["y_is_spike", "p_spike"]].notna().all(axis=1)
        g = group.loc[good]
        labels = g["y_is_spike"].astype(int).to_numpy()
        probability = g["p_spike"].to_numpy()
        if len(np.unique(labels)) < 2:
            continue
        gate = float(np.nanmedian(g["gate_threshold"]))
        hard = probability >= gate
        class_rows.append(
            {
                "experiment_id": experiment_id,
                "n": len(g),
                "prevalence": float(labels.mean()),
                "AUROC": float(roc_auc_score(labels, probability)),
                "average_precision": float(average_precision_score(labels, probability)),
                "Brier": float(brier_score_loss(labels, probability)),
                "median_calibrated_gate": gate,
                "precision_at_gate": float(labels[hard].mean()) if hard.any() else np.nan,
                "recall_at_gate": float(hard[labels == 1].mean()) if labels.any() else np.nan,
                "predicted_event_rate": float(hard.mean()),
            }
        )
    pd.DataFrame(class_rows).sort_values("average_precision", ascending=False).to_csv(
        CLASSIFICATION_PATH, index=False
    )
    return summary


def main():
    frame = pd.read_csv(b17s.DATA_PATH, low_memory=False)
    frame["Datetime"] = pd.to_datetime(frame["Datetime"], format="mixed")
    frame = b17d.add_b17_features(frame)
    frame = b18d.add_antecedent_features(frame)
    feature_sets, engineered = b18d.build_feature_sets(frame)
    specs = experiment_specs()
    MANIFEST_PATH.write_text(
        json.dumps(
            {
                "experiment": "B18 calibrated two-stage spike models",
                "protocol": "spike definitions and calibration use pre-anchor observed targets only",
                "methods": METHODS,
                "experiments": specs,
                "engineered_features": engineered,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    started = time.time()
    run(frame, feature_sets, specs)
    summary = score()
    print(f"\nCompleted in {(time.time() - started) / 60:.1f} min")
    print(summary.loc[summary["target"].eq("observed")].head(40).to_string(index=False))


if __name__ == "__main__":
    main()
