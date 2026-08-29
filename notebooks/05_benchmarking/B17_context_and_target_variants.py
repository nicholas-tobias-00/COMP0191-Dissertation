"""B17 additive context, target, quantile, and pooling experiments.

Consumes the immutable B17 screen definitions but writes only new
``b17_context_*`` artifacts.  Every scored forecast is fixed-origin with observed
pre-anchor target history and known future ``fx_*`` drivers.  No post-anchor flux
or gap-filled target is used as a predictor.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd


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
CHAINS_PATH = RESULTS / "b17_context_target_chains.csv"
ERRORS_PATH = RESULTS / "b17_context_target_errors.csv"
BIN_PATH = RESULTS / "b17_context_target_bin_metrics.csv"
SUMMARY_PATH = RESULTS / "b17_context_target_summary.csv"
MANIFEST_PATH = RESULTS / "b17_context_target_manifest.json"
BASELINE_PATH = RESULTS / "_today_climatology_baseline.csv"

TOWERS = b17s.TOWERS
ANCHOR_YEARS = b17s.ANCHOR_YEARS
N_DAYS = b17s.N_DAYS
BIN_LABELS = b17s.BIN_LABELS
BIN_EDGES = b17s.BIN_EDGES


EXPERIMENTS = [
    # model, feature config, mechanism, parameter
    ("TabPFN_v2", "BASE_ALL_52", "window_days", 730),
    ("TabPFN_v2", "BASE_ALL_52", "window_days", 1095),
    ("TabPFN_v2", "BASE_ALL_52", "window_days", 1460),
    ("TabPFN_v3", "BASE_bodyweight_35", "window_days", 730),
    ("TabPFN_v3", "BASE_bodyweight_35", "window_days", 1095),
    ("TabPFN_v3", "BASE_bodyweight_35", "window_days", 1460),
    ("TabPFN_v3", "BASE_species_37", "window_days", 730),
    ("TabPFN_v3", "BASE_species_37", "window_days", 1095),
    ("TabPFN_v3", "BASE_species_37", "window_days", 1460),
    ("TabICLv2", "BASE_species_bodyweight_38", "window_days", 730),
    ("TabICLv2", "BASE_species_bodyweight_38", "window_days", 1095),
    ("TabICLv2", "BASE_species_bodyweight_38", "window_days", 1460),
    ("TabPFN_v2", "BASE_ALL_52", "context_mean_impute", None),
    ("TabPFN_v3", "BASE_bodyweight_35", "context_mean_impute", None),
    ("TabPFN_v3", "BASE_species_37", "context_mean_impute", None),
    ("TabICLv2", "BASE_species_bodyweight_38", "context_mean_impute", None),
    ("TabPFN_v2", "BASE_ALL_52", "climatology_residual", None),
    ("TabPFN_v3", "BASE_bodyweight_35", "climatology_residual", None),
    ("TabPFN_v3", "BASE_species_37", "climatology_residual", None),
    ("TabICLv2", "BASE_species_bodyweight_38", "climatology_residual", None),
    ("TabPFN_v2", "BASE_ALL_52", "quantiles", None),
    ("TabPFN_v3", "BASE_bodyweight_35", "quantiles", None),
    ("TabPFN_v3", "BASE_species_37", "quantiles", None),
    ("TabICLv2", "BASE_species_bodyweight_38", "quantiles", None),
]

POOLED_EXPERIMENTS = [
    ("TabPFN_v2", "BASE_ALL_52"),
    ("TabPFN_v3", "BASE_bodyweight_35"),
    ("TabPFN_v3", "BASE_species_37"),
    ("TabICLv2", "BASE_species_bodyweight_38"),
]


def append_csv(frame, path):
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def completed_keys():
    if not CHAINS_PATH.exists():
        return set()
    columns = ["tower", "anchor_year", "model", "config", "variant"]
    old = pd.read_csv(CHAINS_PATH, usecols=columns)
    return set(map(tuple, old.drop_duplicates().itertuples(index=False, name=None)))


def base_forecast(model, hist_target, hist_cov, future_cov, quantiles=None):
    if model == "TabPFN_v3":
        return rr.tabpfn_forecast(
            hist_target, hist_cov, future_cov, mode="local", quantiles=quantiles
        )
    if model == "TabPFN_v2":
        return rr.tabpfn_forecast(
            hist_target,
            hist_cov,
            future_cov,
            mode="local",
            quantiles=quantiles,
            tabpfn_model_config=rr.tabpfn_v2_model_config(),
        )
    if model == "TabICLv2":
        return rr.tabicl_forecast(hist_target, hist_cov, future_cov, quantiles=quantiles)
    raise ValueError(model)


def mean_impute(hist_cov, future_cov):
    means = hist_cov.mean(numeric_only=True)
    return hist_cov.fillna(means), future_cov.fillna(means)


def make_rows(dft, dates, tower, year, model, config, variant, prediction, n_features):
    hist_target = dft.loc[: pd.Timestamp(f"{year}-12-16"), "y_observed"]
    climatology = rr.doy_climatology(hist_target.dropna(), dates)
    return pd.DataFrame(
        {
            "date": dates,
            "lead_day": np.arange(1, N_DAYS + 1),
            "tower": tower,
            "anchor_year": year,
            "model": model,
            "config": config,
            "variant": variant,
            "protocol": "observed_history_known_future_fx",
            "n_features": n_features,
            "y_predict": prediction.reindex(dates).to_numpy(),
            "y_true": dft["y_observed"].reindex(dates).to_numpy(),
            "y_gapfilled": dft["y_gapfilled"].reindex(dates).to_numpy(),
            "y_climatology": climatology,
        }
    )


def run_solo(frame, configs):
    towers = {
        tower: frame.loc[frame["tower"].eq(tower)].set_index("Datetime").sort_index()
        for tower in TOWERS
    }
    completed = completed_keys()
    blocks = b17s.available_blocks(frame)
    total = len(blocks) * len(EXPERIMENTS)
    counter = 0

    for tower, year, _ in blocks:
        anchor = pd.Timestamp(f"{year}-12-16")
        dates = pd.date_range(anchor + pd.Timedelta(days=1), periods=N_DAYS, freq="D")
        dft = towers[tower]
        full_hist = dft.loc[:anchor]

        for model, config, mechanism, parameter in EXPERIMENTS:
            counter += 1
            features = configs[config]
            variant_root = mechanism if parameter is None else f"{mechanism}_{parameter}"
            expected_variants = (
                [f"quantile_{q:.1f}" for q in (0.4, 0.5, 0.6, 0.7)]
                if mechanism == "quantiles"
                else [variant_root]
            )
            if all((tower, year, model, config, v) in completed for v in expected_variants):
                print(f"[{counter}/{total}] resume-skip T{tower} {year} {model} {variant_root}")
                continue

            started = time.time()
            try:
                hist = full_hist
                if mechanism == "window_days":
                    first_date = anchor - pd.Timedelta(days=int(parameter) - 1)
                    hist = full_hist.loc[first_date:]
                hist_target = hist["y_observed"].copy()
                hist_cov = hist[features].copy()
                future_cov = dft.loc[dates, features].copy()

                if mechanism == "context_mean_impute":
                    hist_cov, future_cov = mean_impute(hist_cov, future_cov)

                if mechanism == "climatology_residual":
                    full_obs = full_hist["y_observed"].dropna()
                    hist_clim = rr.doy_climatology(full_obs, hist.index)
                    future_clim = rr.doy_climatology(full_obs, dates)
                    hist_target = hist_target - hist_clim
                    residual = base_forecast(model, hist_target, hist_cov, future_cov)
                    predictions = {variant_root: residual + future_clim}
                elif mechanism == "quantiles":
                    forecast = base_forecast(
                        model, hist_target, hist_cov, future_cov, quantiles=[0.4, 0.5, 0.6, 0.7]
                    )
                    predictions = {
                        f"quantile_{q:.1f}": pd.Series(forecast[q].to_numpy(), index=forecast.index)
                        for q in (0.4, 0.5, 0.6, 0.7)
                    }
                else:
                    predictions = {
                        variant_root: base_forecast(model, hist_target, hist_cov, future_cov)
                    }

                for variant, prediction in predictions.items():
                    key = (tower, year, model, config, variant)
                    if key in completed:
                        continue
                    append_csv(
                        make_rows(
                            dft,
                            dates,
                            tower,
                            year,
                            model,
                            config,
                            variant,
                            prediction,
                            len(features),
                        ),
                        CHAINS_PATH,
                    )
                print(
                    f"[{counter}/{total}] T{tower} {year} {model} {config} {variant_root}: "
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
                                "variant": variant_root,
                                "error_type": type(exc).__name__,
                                "error": str(exc)[:1000],
                                "traceback": traceback.format_exc()[-4000:],
                            }
                        ]
                    ),
                    ERRORS_PATH,
                )
                print(f"[{counter}/{total}] ERROR T{tower} {year} {model} {variant_root}: {exc}")


def pooled_frames(towers, anchor, dates, features):
    context_parts = []
    future_parts = []
    for tower in TOWERS:
        hist = towers[tower].loc[:anchor]
        context = hist[features].copy()
        context["timestamp"] = context.index
        context["target"] = hist["y_observed"].to_numpy()
        context["item_id"] = tower
        context_parts.append(context.reset_index(drop=True))

        future = towers[tower].loc[dates, features].copy()
        future["timestamp"] = future.index
        future["item_id"] = tower
        future_parts.append(future.reset_index(drop=True))

    context = pd.concat(context_parts, ignore_index=True)
    future = pd.concat(future_parts, ignore_index=True)
    means = context[features].mean()
    context[features] = context[features].fillna(means)
    future[features] = future[features].fillna(means)
    return context, future


def pooled_forecast(model, context, future):
    if model.startswith("TabPFN"):
        import tabpfn_time_series as tts

        kwargs = {"tabpfn_mode": tts.TabPFNMode.LOCAL}
        if model == "TabPFN_v2":
            kwargs["tabpfn_model_config"] = rr.tabpfn_v2_model_config()
        forecaster = tts.TabPFNTSPipeline(**kwargs)
    else:
        from tabicl import TabICLForecaster

        forecaster = TabICLForecaster()
    predictions = forecaster.predict_df(context, future_df=future, quantiles=[0.5]).reset_index()
    predictions["timestamp"] = pd.to_datetime(predictions["timestamp"])
    return {
        tower: predictions.loc[predictions["item_id"].eq(tower)].set_index("timestamp")[0.5]
        for tower in TOWERS
    }


def run_pooled(frame, configs):
    towers = {
        tower: frame.loc[frame["tower"].eq(tower)].set_index("Datetime").sort_index()
        for tower in TOWERS
    }
    completed = completed_keys()
    for year in ANCHOR_YEARS:
        anchor = pd.Timestamp(f"{year}-12-16")
        dates = pd.date_range(anchor + pd.Timedelta(days=1), periods=N_DAYS, freq="D")
        for model, config in POOLED_EXPERIMENTS:
            variant = "pooled_towers_mean_impute"
            if all((tower, year, model, config, variant) in completed for tower in TOWERS):
                continue
            started = time.time()
            try:
                context, future = pooled_frames(towers, anchor, dates, configs[config])
                predictions = pooled_forecast(model, context, future)
                for tower in TOWERS:
                    key = (tower, year, model, config, variant)
                    if key in completed:
                        continue
                    append_csv(
                        make_rows(
                            towers[tower],
                            dates,
                            tower,
                            year,
                            model,
                            config,
                            variant,
                            predictions[tower],
                            len(configs[config]),
                        ),
                        CHAINS_PATH,
                    )
                print(f"[pooled] {year} {model} {config}: {time.time() - started:.1f}s", flush=True)
            except Exception as exc:
                append_csv(
                    pd.DataFrame(
                        [
                            {
                                "tower": "ALL",
                                "anchor_year": year,
                                "model": model,
                                "config": config,
                                "variant": variant,
                                "error_type": type(exc).__name__,
                                "error": str(exc)[:1000],
                                "traceback": traceback.format_exc()[-4000:],
                            }
                        ]
                    ),
                    ERRORS_PATH,
                )
                print(f"[pooled] ERROR {year} {model} {config}: {exc}")


def metrics_for(group, target_col):
    good = group[[target_col, "y_predict"]].notna().all(axis=1)
    g = group.loc[good]
    if g.empty:
        return None
    y = g[target_col].to_numpy()
    p = g["y_predict"].to_numpy()
    error = y - p
    denom = float(np.sum((y - y.mean()) ** 2))
    return {
        "n": len(g),
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "R2": float(1 - np.sum(error**2) / denom) if denom > 0 else np.nan,
        "bias": float(np.mean(p - y)),
    }


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
        info = dict(zip(keys, key))
        denominator = group["MAE_climatology"].dropna()
        if denominator.empty or denominator.iloc[0] <= 0:
            continue
        for target, target_col in [("observed", "y_true"), ("gapfilled", "y_gapfilled")]:
            values = metrics_for(group, target_col)
            if values is None:
                continue
            values["MAE_climatology"] = float(denominator.iloc[0])
            values["MASE"] = values["MAE"] / values["MAE_climatology"]
            rows.append({**info, "target": target, **values})
    bins = pd.DataFrame(rows)
    bins.to_csv(BIN_PATH, index=False)

    summary_rows = []
    skeys = ["model", "config", "variant", "protocol", "n_features", "target"]
    for key, group in bins.groupby(skeys, sort=False):
        ok = group["MASE"].notna() & group["n"].gt(0)
        r2ok = group["R2"].notna() & group["n"].gt(0)
        summary_rows.append(
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
    summary = pd.DataFrame(summary_rows).sort_values(["target", "MASE"])
    summary.to_csv(SUMMARY_PATH, index=False)
    return summary


def main():
    frame = pd.read_csv(b17s.DATA_PATH, low_memory=False)
    frame["Datetime"] = pd.to_datetime(frame["Datetime"], format="mixed")
    configs = b17s.build_configs(frame)
    MANIFEST_PATH.write_text(
        json.dumps(
            {
                "experiment": "B17 context and target variants",
                "solo_experiments": EXPERIMENTS,
                "pooled_experiments": POOLED_EXPERIMENTS,
                "primary_metric": "observed-target, lead-bin-weighted frozen-climatology MASE",
                "protocol": "observed pre-anchor target history + known post-anchor fx drivers",
                "no_future_target_or_flux_input": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    started = time.time()
    run_solo(frame, configs)
    run_pooled(frame, configs)
    summary = score()
    print(f"\nCompleted in {(time.time() - started) / 60:.1f} min")
    print(summary.loc[summary["target"].eq("observed")].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
