"""B17 additive tuning of the new direct pooled foundation champion.

Tests ensemble size, temperature, ensemble aggregation, random seed, and robust
target transforms for direct pooled TabPFN v2; also increases TabICL's ensemble
size.  All fits use only observed rows dated on or before each anchor.
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
CHAINS_PATH = RESULTS / "b17_direct_tuning_chains.csv"
ERRORS_PATH = RESULTS / "b17_direct_tuning_errors.csv"
BIN_PATH = RESULTS / "b17_direct_tuning_bin_metrics.csv"
SUMMARY_PATH = RESULTS / "b17_direct_tuning_summary.csv"
MANIFEST_PATH = RESULTS / "b17_direct_tuning_manifest.json"
BASELINE_PATH = RESULTS / "_today_climatology_baseline.csv"

TOWERS = b17s.TOWERS
ANCHOR_YEARS = b17s.ANCHOR_YEARS
N_DAYS = b17s.N_DAYS
BIN_LABELS = b17s.BIN_LABELS
BIN_EDGES = b17s.BIN_EDGES

PFN_SETTINGS = [
    ("n16", {"n_estimators": 16, "auto_scale_n_estimators": False, "random_state": 42}),
    ("n32", {"n_estimators": 32, "auto_scale_n_estimators": False, "random_state": 42}),
    ("temp07", {"n_estimators": 8, "softmax_temperature": 0.7, "random_state": 42}),
    ("temp11", {"n_estimators": 8, "softmax_temperature": 1.1, "random_state": 42}),
    ("avg_before_softmax", {"n_estimators": 8, "average_before_softmax": True, "random_state": 42}),
    ("seed0", {"n_estimators": 8, "random_state": 0}),
    ("seed137", {"n_estimators": 8, "random_state": 137}),
    (
        "n16_temp07",
        {
            "n_estimators": 16,
            "auto_scale_n_estimators": False,
            "softmax_temperature": 0.7,
            "random_state": 42,
        },
    ),
]

ICL_SETTINGS = [
    ("n16", {"n_estimators": 16, "random_state": 42}),
    ("n32", {"n_estimators": 32, "random_state": 42}),
    ("seed0", {"n_estimators": 8, "random_state": 0}),
    ("seed137", {"n_estimators": 8, "random_state": 137}),
]

TRANSFORMS = ["asinh_scale10", "signed_log1p"]


def append_csv(frame, path):
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def completed_keys():
    if not CHAINS_PATH.exists():
        return set()
    old = pd.read_csv(CHAINS_PATH, usecols=["anchor_year", "model", "variant"])
    return set(map(tuple, old.drop_duplicates().itertuples(index=False, name=None)))


def transform_target(values, transform):
    values = np.asarray(values, dtype=float)
    if transform == "raw":
        return values
    if transform == "asinh_scale10":
        return np.arcsinh(values / 10.0)
    if transform == "signed_log1p":
        return np.sign(values) * np.log1p(np.abs(values))
    raise ValueError(transform)


def inverse_target(values, transform):
    values = np.asarray(values, dtype=float)
    if transform == "raw":
        return values
    if transform == "asinh_scale10":
        return 10.0 * np.sinh(values)
    if transform == "signed_log1p":
        return np.sign(values) * np.expm1(np.abs(values))
    raise ValueError(transform)


def make_model(model_name, kwargs):
    if model_name == "Direct_TabPFN_v2":
        from tabpfn import TabPFNRegressor

        return TabPFNRegressor(
            model_path=rr.tabpfn_v2_model_config()["model_path"], **kwargs
        )
    if model_name == "Direct_TabICLv2":
        from tabicl import TabICLRegressor

        return TabICLRegressor(**kwargs)
    raise ValueError(model_name)


def predict_median(model_name, model, x):
    if model_name == "Direct_TabPFN_v2":
        return model.predict(x, output_type="median")
    return model.predict(x, output_type="median")


def experiments():
    rows = []
    for name, kwargs in PFN_SETTINGS:
        rows.append(("Direct_TabPFN_v2", name, kwargs, "raw"))
    for name, kwargs in ICL_SETTINGS:
        rows.append(("Direct_TabICLv2", name, kwargs, "raw"))
    for transform in TRANSFORMS:
        rows.append(
            (
                "Direct_TabPFN_v2",
                transform,
                {"n_estimators": 8, "random_state": 42},
                transform,
            )
        )
        rows.append(
            (
                "Direct_TabICLv2",
                transform,
                {"n_estimators": 8, "random_state": 42},
                transform,
            )
        )
    return rows


def run(frame, configs):
    completed = completed_keys()
    feature_cols = configs["BASE_ALL_52"] + b17d.TOWER_DUMMIES + b17d.TIME_FEATURES
    for year in ANCHOR_YEARS:
        anchor = pd.Timestamp(f"{year}-12-16")
        dates = pd.date_range(anchor + pd.Timedelta(days=1), periods=N_DAYS, freq="D")
        train = frame.loc[
            frame["Datetime"].le(anchor) & frame["y_observed"].notna()
        ].copy()
        future = frame.loc[frame["Datetime"].isin(dates)].copy()
        imputer = SimpleImputer(strategy="mean")
        x_train = imputer.fit_transform(train[feature_cols])
        x_future = imputer.transform(future[feature_cols])

        for model_name, setting, kwargs, transform in experiments():
            variant = f"pooled_time_{setting}_{transform}_median"
            key = (year, model_name, variant)
            if key in completed:
                continue
            started = time.time()
            try:
                model = make_model(model_name, kwargs)
                model.fit(x_train, transform_target(train["y_observed"], transform))
                prediction = inverse_target(
                    predict_median(model_name, model, x_future), transform
                )
                predicted = future[["Datetime", "tower"]].copy()
                predicted["prediction"] = prediction
                for tower in TOWERS:
                    dft = frame.loc[frame["tower"].eq(tower)].set_index("Datetime")
                    tower_prediction = (
                        predicted.loc[predicted["tower"].eq(tower)]
                        .set_index("Datetime")["prediction"]
                        .reindex(dates)
                    )
                    append_csv(
                        b17d.raw_rows(
                            dft,
                            dates,
                            tower,
                            year,
                            model_name,
                            "BASE_ALL_52",
                            variant,
                            "observed_history_known_future_fx",
                            tower_prediction,
                            len(feature_cols),
                        ),
                        CHAINS_PATH,
                    )
                print(
                    f"[tune] {year} {model_name} {variant}: {time.time() - started:.1f}s",
                    flush=True,
                )
            except Exception as exc:
                append_csv(
                    pd.DataFrame(
                        [
                            {
                                "anchor_year": year,
                                "model": model_name,
                                "variant": variant,
                                "error_type": type(exc).__name__,
                                "error": str(exc)[:1000],
                                "traceback": traceback.format_exc()[-4000:],
                            }
                        ]
                    ),
                    ERRORS_PATH,
                )
                print(f"[tune] ERROR {year} {model_name} {variant}: {exc}")


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
    keys = ["tower", "anchor_year", "model", "config", "variant", "protocol", "n_features", "bin"]
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
    skeys = ["model", "config", "variant", "protocol", "n_features", "target"]
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
                "RMSE_bin_weighted": float(np.average(group.loc[ok, "RMSE"], weights=group.loc[ok, "n"])),
                "R2_bin_weighted": float(np.average(group.loc[r2ok, "R2"], weights=group.loc[r2ok, "n"]))
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
    configs = b17s.build_configs(frame)
    MANIFEST_PATH.write_text(
        json.dumps(
            {
                "experiment": "B17 direct foundation tuning",
                "TabPFN_settings": PFN_SETTINGS,
                "TabICL_settings": ICL_SETTINGS,
                "target_transforms": TRANSFORMS,
                "protocol": "observed history, known future fx, pooled towers",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    started = time.time()
    run(frame, configs)
    summary = score()
    print(f"\nCompleted in {(time.time() - started) / 60:.1f} min")
    print(summary.loc[summary["target"].eq("observed")].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
