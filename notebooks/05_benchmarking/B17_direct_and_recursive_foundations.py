"""B17 additive direct-regressor and recursive-chunk foundation experiments.

This stage tests estimators that are materially different from B16's one-shot
foundation pipeline: direct tabular foundation regression, self-fed forecast
chunks, and pre-anchor-only target densification.  Strict and sensitivity
protocols are labelled separately in every raw row.
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

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass


RESULTS = ROOT / "results"
CHAINS_PATH = RESULTS / "b17_direct_recursive_chains.csv"
ERRORS_PATH = RESULTS / "b17_direct_recursive_errors.csv"
BIN_PATH = RESULTS / "b17_direct_recursive_bin_metrics.csv"
SUMMARY_PATH = RESULTS / "b17_direct_recursive_summary.csv"
MANIFEST_PATH = RESULTS / "b17_direct_recursive_manifest.json"
BASELINE_PATH = RESULTS / "_today_climatology_baseline.csv"

TOWERS = b17s.TOWERS
ANCHOR_YEARS = b17s.ANCHOR_YEARS
N_DAYS = b17s.N_DAYS
BIN_LABELS = b17s.BIN_LABELS
BIN_EDGES = b17s.BIN_EDGES
TOWER_DUMMIES = ["b17_is_t2", "b17_is_t4", "b17_is_t9"]
TIME_FEATURES = ["b17_year", "b17_days_since_2010"]

DIRECT_EXPERIMENTS = [
    # model, config, pooling, add_time, include_realized_future_fco2
    ("Direct_TabPFN_v2", "BASE_ALL_52", "pooled", False, False),
    ("Direct_TabPFN_v2", "BASE_ALL_52", "pooled", True, False),
    ("Direct_TabPFN_v3TS", "BASE_ALL_52", "pooled", False, False),
    ("Direct_TabPFN_v3TS", "BASE_ALL_52", "pooled", True, False),
    ("Direct_TabPFN_v2", "BASE_species_bodyweight_38", "pooled", True, False),
    ("Direct_TabICLv2", "BASE_ALL_52", "pooled", False, False),
    ("Direct_TabICLv2", "BASE_ALL_52", "pooled", True, False),
    ("Direct_TabPFN_v2", "BASE_ALL_52", "solo", False, False),
    ("Direct_TabICLv2", "BASE_ALL_52", "solo", False, False),
    # Conditional upper bounds, never mixed into the strict fixed-origin ranking.
    ("Direct_TabPFN_v2", "BASE_ALL_52", "pooled", True, True),
    ("Direct_TabICLv2", "BASE_ALL_52", "pooled", True, True),
]

CHUNK_EXPERIMENTS = [
    ("TabPFN_v2", "BASE_ALL_52", 30),
    ("TabPFN_v2", "BASE_ALL_52", 90),
    ("TabPFN_v2", "BASE_ALL_52", 180),
    ("TabPFN_v3", "BASE_bodyweight_35", 90),
    ("TabPFN_v3", "BASE_species_37", 90),
    ("TabICLv2", "BASE_species_bodyweight_38", 90),
]

DENSE_CONTEXT_EXPERIMENTS = [
    ("TabPFN_v2", "BASE_ALL_52", "linear_interpolate"),
    ("TabPFN_v2", "BASE_ALL_52", "doy_climatology_fill"),
    ("TabPFN_v3", "BASE_bodyweight_35", "linear_interpolate"),
    ("TabPFN_v3", "BASE_bodyweight_35", "doy_climatology_fill"),
    ("TabICLv2", "BASE_species_bodyweight_38", "linear_interpolate"),
    ("TabICLv2", "BASE_species_bodyweight_38", "doy_climatology_fill"),
]


def append_csv(frame, path):
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def completed_keys():
    if not CHAINS_PATH.exists():
        return set()
    cols = ["tower", "anchor_year", "model", "config", "variant", "protocol"]
    old = pd.read_csv(CHAINS_PATH, usecols=cols)
    return set(map(tuple, old.drop_duplicates().itertuples(index=False, name=None)))


def add_b17_features(frame):
    out = frame.copy()
    for tower in TOWERS:
        out[f"b17_is_t{tower}"] = out["tower"].eq(tower).astype(float)
    out["b17_year"] = out["Datetime"].dt.year.astype(float)
    out["b17_days_since_2010"] = (
        out["Datetime"] - pd.Timestamp("2010-01-01")
    ).dt.days.astype(float)
    return out


def raw_rows(
    dft,
    dates,
    tower,
    year,
    model,
    config,
    variant,
    protocol,
    prediction,
    n_features,
):
    anchor = pd.Timestamp(f"{year}-12-16")
    observed_hist = dft.loc[:anchor, "y_observed"].dropna()
    climatology = rr.doy_climatology(observed_hist, dates)
    return pd.DataFrame(
        {
            "date": dates,
            "lead_day": np.arange(1, N_DAYS + 1),
            "tower": tower,
            "anchor_year": year,
            "model": model,
            "config": config,
            "variant": variant,
            "protocol": protocol,
            "n_features": n_features,
            "y_predict": np.asarray(prediction),
            "y_true": dft["y_observed"].reindex(dates).to_numpy(),
            "y_gapfilled": dft["y_gapfilled"].reindex(dates).to_numpy(),
            "y_climatology": climatology,
        }
    )


def direct_model(model_name):
    if model_name.startswith("Direct_TabPFN"):
        from tabpfn import TabPFNRegressor

        if model_name == "Direct_TabPFN_v2":
            model_path = rr.tabpfn_v2_model_config()["model_path"]
        else:
            from tabpfn_time_series.pipeline import resolve_default_ckpt

            model_path = resolve_default_ckpt({})["model_path"]
        return TabPFNRegressor(model_path=model_path, random_state=42, n_estimators=8)
    if model_name == "Direct_TabICLv2":
        from tabicl import TabICLRegressor

        return TabICLRegressor(random_state=42, n_estimators=8)
    raise ValueError(model_name)


def direct_predict(model_name, model, matrix):
    if model_name.startswith("Direct_TabPFN"):
        result = model.predict(matrix, output_type="main")
        predictions = {
            "direct_mean": result["mean"],
            "direct_median": result["median"],
            "direct_mode": result["mode"],
        }
        for quantile, values in zip(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], result["quantiles"]
        ):
            if quantile in (0.4, 0.5, 0.6):
                predictions[f"direct_quantile_{quantile:.1f}"] = values
        return predictions
    result = model.predict(matrix, output_type=["mean", "median"])
    return {"direct_mean": result["mean"], "direct_median": result["median"]}


def direct_variant_root(pooling, add_time, fco2):
    return f"{pooling}_time{int(add_time)}_futureFCO2{int(fco2)}"


def run_direct(frame, configs):
    completed = completed_keys()
    for year in ANCHOR_YEARS:
        anchor = pd.Timestamp(f"{year}-12-16")
        dates = pd.date_range(anchor + pd.Timedelta(days=1), periods=N_DAYS, freq="D")
        for model_name, config, pooling, add_time, fco2 in DIRECT_EXPERIMENTS:
            root = direct_variant_root(pooling, add_time, fco2)
            protocol = (
                "conditional_known_future_fx_plus_realized_fco2"
                if fco2
                else "observed_history_known_future_fx"
            )
            output_names = ["direct_mean", "direct_median"]
            if model_name.startswith("Direct_TabPFN"):
                output_names += [
                    "direct_mode",
                    "direct_quantile_0.4",
                    "direct_quantile_0.5",
                    "direct_quantile_0.6",
                ]

            feature_cols = list(configs[config])
            if pooling:
                feature_cols += TOWER_DUMMIES
            if add_time:
                feature_cols += TIME_FEATURES
            if fco2:
                feature_cols += ["ar_fc_dlag1"]
            feature_cols = list(dict.fromkeys(feature_cols))

            target_towers = TOWERS if pooling == "pooled" else TOWERS
            if all(
                (tower, year, model_name, config, f"{root}_{name}", protocol) in completed
                for tower in target_towers
                for name in output_names
            ):
                continue

            if pooling == "pooled":
                groups = [("ALL", TOWERS)]
            else:
                groups = [(str(tower), [tower]) for tower in TOWERS]

            for group_name, group_towers in groups:
                started = time.time()
                try:
                    train = frame.loc[
                        frame["tower"].isin(group_towers)
                        & frame["Datetime"].le(anchor)
                        & frame["y_observed"].notna()
                    ].copy()
                    future = frame.loc[
                        frame["tower"].isin(group_towers) & frame["Datetime"].isin(dates)
                    ].copy()
                    if train.empty or future.empty:
                        continue
                    if len(train) > 10_000:
                        train = train.sample(10_000, random_state=42).sort_values("Datetime")
                    imputer = SimpleImputer(strategy="mean")
                    x_train = imputer.fit_transform(train[feature_cols])
                    x_future = imputer.transform(future[feature_cols])
                    model = direct_model(model_name)
                    model.fit(x_train, train["y_observed"].to_numpy())
                    outputs = direct_predict(model_name, model, x_future)

                    for output_name, prediction in outputs.items():
                        variant = f"{root}_{output_name}"
                        future_with_prediction = future[["Datetime", "tower"]].copy()
                        future_with_prediction["prediction"] = prediction
                        for tower in group_towers:
                            key = (tower, year, model_name, config, variant, protocol)
                            if key in completed:
                                continue
                            dft = frame.loc[frame["tower"].eq(tower)].set_index("Datetime")
                            tower_prediction = (
                                future_with_prediction.loc[
                                    future_with_prediction["tower"].eq(tower)
                                ]
                                .set_index("Datetime")["prediction"]
                                .reindex(dates)
                            )
                            append_csv(
                                raw_rows(
                                    dft,
                                    dates,
                                    tower,
                                    year,
                                    model_name,
                                    config,
                                    variant,
                                    protocol,
                                    tower_prediction,
                                    len(feature_cols),
                                ),
                                CHAINS_PATH,
                            )
                    print(
                        f"[direct] {year} {model_name} {config} {root} group={group_name} "
                        f"train_n={len(train)}: {time.time() - started:.1f}s",
                        flush=True,
                    )
                except Exception as exc:
                    append_csv(
                        pd.DataFrame(
                            [
                                {
                                    "tower": group_name,
                                    "anchor_year": year,
                                    "model": model_name,
                                    "config": config,
                                    "variant": root,
                                    "protocol": protocol,
                                    "error_type": type(exc).__name__,
                                    "error": str(exc)[:1000],
                                    "traceback": traceback.format_exc()[-4000:],
                                }
                            ]
                        ),
                        ERRORS_PATH,
                    )
                    print(f"[direct] ERROR {year} {model_name} {root} {group_name}: {exc}")


def ts_forecast(model, hist_target, hist_cov, future_cov):
    if model == "TabPFN_v2":
        return rr.tabpfn_forecast(
            hist_target,
            hist_cov,
            future_cov,
            mode="local",
            tabpfn_model_config=rr.tabpfn_v2_model_config(),
        )
    if model == "TabPFN_v3":
        return rr.tabpfn_forecast(hist_target, hist_cov, future_cov, mode="local")
    if model == "TabICLv2":
        return rr.tabicl_forecast(hist_target, hist_cov, future_cov)
    raise ValueError(model)


def chunked_forecast(model, hist, dft, dates, features, chunk_days):
    target = hist["y_observed"].copy()
    covariates = hist[features].copy()
    pieces = []
    for start in range(0, len(dates), chunk_days):
        chunk = dates[start : start + chunk_days]
        future_cov = dft.loc[chunk, features]
        prediction = ts_forecast(model, target, covariates, future_cov).reindex(chunk)
        pieces.append(prediction)
        target = pd.concat([target, prediction])
        covariates = pd.concat([covariates, future_cov])
    return pd.concat(pieces).reindex(dates)


def dense_target(hist, method):
    observed = hist["y_observed"].copy()
    if method == "linear_interpolate":
        return observed.interpolate(method="time", limit_direction="both")
    if method == "doy_climatology_fill":
        values = rr.doy_climatology(observed.dropna(), hist.index)
        return observed.fillna(pd.Series(values, index=hist.index))
    raise ValueError(method)


def run_ts_variants(frame, configs):
    completed = completed_keys()
    towers = {
        tower: frame.loc[frame["tower"].eq(tower)].set_index("Datetime").sort_index()
        for tower in TOWERS
    }
    blocks = b17s.available_blocks(frame)
    for tower, year, _ in blocks:
        anchor = pd.Timestamp(f"{year}-12-16")
        dates = pd.date_range(anchor + pd.Timedelta(days=1), periods=N_DAYS, freq="D")
        dft = towers[tower]
        hist = dft.loc[:anchor]
        protocol = "observed_history_known_future_fx"

        for model, config, chunk_days in CHUNK_EXPERIMENTS:
            variant = f"self_fed_chunks_{chunk_days}d"
            key = (tower, year, model, config, variant, protocol)
            if key in completed:
                continue
            started = time.time()
            try:
                prediction = chunked_forecast(
                    model, hist, dft, dates, configs[config], chunk_days
                )
                append_csv(
                    raw_rows(
                        dft,
                        dates,
                        tower,
                        year,
                        model,
                        config,
                        variant,
                        protocol,
                        prediction,
                        len(configs[config]),
                    ),
                    CHAINS_PATH,
                )
                print(
                    f"[chunks] T{tower} {year} {model} {config} {chunk_days}d: "
                    f"{time.time() - started:.1f}s",
                    flush=True,
                )
            except Exception as exc:
                append_csv(
                    pd.DataFrame(
                        [
                            {
                                "tower": tower,
                                "anchor_year": year,
                                "model": model,
                                "config": config,
                                "variant": variant,
                                "protocol": protocol,
                                "error_type": type(exc).__name__,
                                "error": str(exc)[:1000],
                                "traceback": traceback.format_exc()[-4000:],
                            }
                        ]
                    ),
                    ERRORS_PATH,
                )

        for model, config, method in DENSE_CONTEXT_EXPERIMENTS:
            variant = f"preanchor_target_{method}"
            key = (tower, year, model, config, variant, protocol)
            if key in completed:
                continue
            started = time.time()
            try:
                prediction = ts_forecast(
                    model,
                    dense_target(hist, method),
                    hist[configs[config]],
                    dft.loc[dates, configs[config]],
                ).reindex(dates)
                append_csv(
                    raw_rows(
                        dft,
                        dates,
                        tower,
                        year,
                        model,
                        config,
                        variant,
                        protocol,
                        prediction,
                        len(configs[config]),
                    ),
                    CHAINS_PATH,
                )
                print(
                    f"[dense] T{tower} {year} {model} {config} {method}: "
                    f"{time.time() - started:.1f}s",
                    flush=True,
                )
            except Exception as exc:
                append_csv(
                    pd.DataFrame(
                        [
                            {
                                "tower": tower,
                                "anchor_year": year,
                                "model": model,
                                "config": config,
                                "variant": variant,
                                "protocol": protocol,
                                "error_type": type(exc).__name__,
                                "error": str(exc)[:1000],
                                "traceback": traceback.format_exc()[-4000:],
                            }
                        ]
                    ),
                    ERRORS_PATH,
                )


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
        "config",
        "variant",
        "protocol",
        "n_features",
        "bin",
    ]
    rows = []
    for key, group in chains.groupby(keys, sort=False, observed=True):
        denom_values = group["MAE_climatology"].dropna()
        if denom_values.empty or denom_values.iloc[0] <= 0:
            continue
        denominator = float(denom_values.iloc[0])
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
    summary = pd.DataFrame(summaries).sort_values(["protocol", "target", "MASE"])
    summary.to_csv(SUMMARY_PATH, index=False)
    return summary


def main():
    frame = pd.read_csv(b17s.DATA_PATH, low_memory=False)
    frame["Datetime"] = pd.to_datetime(frame["Datetime"], format="mixed")
    frame = add_b17_features(frame)
    configs = b17s.build_configs(frame)
    MANIFEST_PATH.write_text(
        json.dumps(
            {
                "experiment": "B17 direct and recursive foundation models",
                "direct_experiments": DIRECT_EXPERIMENTS,
                "chunk_experiments": CHUNK_EXPERIMENTS,
                "dense_context_experiments": DENSE_CONTEXT_EXPERIMENTS,
                "strict_protocol": "observed_history_known_future_fx",
                "conditional_protocol": "known future fx plus realized FCO2; upper bound only",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    started = time.time()
    run_direct(frame, configs)
    run_ts_variants(frame, configs)
    summary = score()
    strict = summary.loc[
        summary["target"].eq("observed")
        & summary["protocol"].eq("observed_history_known_future_fx")
    ]
    conditional = summary.loc[
        summary["target"].eq("observed")
        & summary["protocol"].eq("conditional_known_future_fx_plus_realized_fco2")
    ]
    print(f"\nCompleted in {(time.time() - started) / 60:.1f} min")
    print("\nStrict ranking")
    print(strict.head(40).to_string(index=False))
    print("\nConditional realized-FCO2 upper-bound ranking")
    print(conditional.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
