"""B17 additive foundation-model feature screen.

This script never writes outside the ``b17_*`` namespace.  It reruns the genuine
34-column BASE, genuine 37-column BASE+species, and explicitly named alternatives
on the observed-target protocol used by B16.  Raw daily chains are checkpointed
after every successful model call so a long local run can be resumed safely.

Run from the project root:

    python notebooks/05_benchmarking/B17_foundation_screen.py

The primary score is the project's bin-weighted climatology MASE on observed
FCH4.  Future ``fx_*`` drivers retain B16's known-driver convention.  No future
FCH4, gap-filled target, or post-anchor autoregressive flux is supplied as input.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import models.recursive_rollout as rr

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass


DATA_PATH = ROOT / "data" / "Hourly" / "forecast_daily_v3.csv"
RESULTS = ROOT / "results"
CHAINS_PATH = RESULTS / "b17_foundation_screen_chains.csv"
ERRORS_PATH = RESULTS / "b17_foundation_screen_errors.csv"
BIN_PATH = RESULTS / "b17_foundation_screen_bin_metrics.csv"
SUMMARY_PATH = RESULTS / "b17_foundation_screen_summary.csv"
MANIFEST_PATH = RESULTS / "b17_foundation_screen_manifest.json"

TOWERS = [2, 4, 9]
ANCHOR_YEARS = [2018, 2019, 2020, 2021, 2022]
N_DAYS = 365

FAMILIES = {
    "species": ["fx_cattle_dens", "fx_sheep_dens", "fx_lamb_dens"],
    "arable": ["fx_is_arable"],
    "flow": [
        "fx_flow_mean",
        "fx_flow_lag7",
        "fx_flow_lag14",
        "fx_flow_lag21",
        "fx_flow_lag28",
        "fx_flow_roll7",
        "fx_flow_roll14",
    ],
    "mgmt": [
        "fx_mgmt_fertN_recency",
        "fx_mgmt_fertN_rate",
        "fx_mgmt_lime_recency",
        "fx_mgmt_cultiv_recency",
        "fx_mgmt_cut_recency",
        "fx_mgmt_manure_recency",
    ],
    "bodyweight": ["fx_total_liveweight_dens"],
}
ALL_NEW = sorted({col for columns in FAMILIES.values() for col in columns})

# Exact S03 scenario-unavailable list, copied here deliberately so B17 is
# self-contained and cannot trigger imports with scenario-building side effects.
DEGRADED_SENSOR_COLS = [
    "fx_WS_mean",
    "fx_VPD_mean",
    "fx_RN_mean",
    "fx_PPFD_mean",
    "fx_SWC_mean",
    "fx_TS_mean",
    "fx_wd_sin",
    "fx_wd_cos",
    "fx_SWC_lag7",
    "fx_TS_lag7",
    "fx_SWC_lag14",
    "fx_TS_lag14",
    "fx_SWC_lag21",
    "fx_TS_lag21",
    "fx_SWC_lag28",
    "fx_TS_lag28",
    "fx_SWC_roll7",
    "fx_TS_roll7",
    "fx_SWC_roll14",
    "fx_TS_roll14",
    "fx_grazing_active",
    "fx_days_since_grazing",
    "fx_USTAR_mean",
    "fx_SHF_mean",
]

CORE_METEO = [
    "fx_TA_mean",
    "fx_TA_min",
    "fx_TA_max",
    "fx_SWIN_mean",
    "fx_PRECIP_sum",
    "fx_DOY_sin",
    "fx_DOY_cos",
    "fx_is_growing",
    "fx_is_winter",
    "fx_lsu_dens",
]

BIN_LABELS = ["1-7", "8-30", "31-90", "91-180", "181-270", "271-365"]
BIN_EDGES = [0, 7, 30, 90, 180, 270, 365]


def _unique(columns):
    return list(dict.fromkeys(columns))


def build_configs(frame):
    fx_all = [c for c in frame.columns if c.startswith("fx")]
    base = [c for c in fx_all if c not in ALL_NEW]
    species = FAMILIES["species"]

    assert len(fx_all) == 52, f"Expected 52 total fx columns, found {len(fx_all)}"
    assert len(ALL_NEW) == 18, f"Expected 18 F10 additions, found {len(ALL_NEW)}"
    assert len(base) == 34, f"Expected genuine BASE=34, found {len(base)}"

    configs = {
        "BASE_34": base,
        "BASE_species_37": _unique(base + species),
        "BASE_ALL_52": _unique(base + ALL_NEW),
        "BASE_cattle_35": _unique(base + ["fx_cattle_dens"]),
        "BASE_bodyweight_35": _unique(base + FAMILIES["bodyweight"]),
        "BASE_species_bodyweight_38": _unique(base + species + FAMILIES["bodyweight"]),
        "BASE_without_lsu_species_36": _unique(
            [c for c in base if c != "fx_lsu_dens"] + species
        ),
        "LEAN_species_13": _unique(
            [c for c in base if c not in DEGRADED_SENSOR_COLS] + species
        ),
        "CORE_species_13": _unique(CORE_METEO + species),
    }

    expected = {
        "BASE_34": 34,
        "BASE_species_37": 37,
        "BASE_ALL_52": 52,
        "BASE_cattle_35": 35,
        "BASE_bodyweight_35": 35,
        "BASE_species_bodyweight_38": 38,
        "BASE_without_lsu_species_36": 36,
        "LEAN_species_13": 13,
        "CORE_species_13": 13,
    }
    for name, cols in configs.items():
        assert len(cols) == expected[name], (name, len(cols), expected[name])
        assert len(cols) == len(set(cols)), f"Duplicate feature in {name}"
        missing = sorted(set(cols) - set(frame.columns))
        assert not missing, f"Missing columns in {name}: {missing}"
    return configs


def available_blocks(frame):
    blocks = []
    for year in ANCHOR_YEARS:
        anchor = pd.Timestamp(f"{year}-12-16")
        dates = pd.date_range(anchor + pd.Timedelta(days=1), periods=N_DAYS, freq="D")
        for tower in TOWERS:
            dft = frame.loc[frame["tower"].eq(tower)].set_index("Datetime")
            n_observed = int(dft["y_observed"].reindex(dates).notna().sum())
            if n_observed:
                blocks.append((tower, year, n_observed))
    return blocks


def read_completed():
    if not CHAINS_PATH.exists():
        return set()
    old = pd.read_csv(CHAINS_PATH, usecols=["tower", "anchor_year", "model", "config"])
    return set(map(tuple, old.drop_duplicates().itertuples(index=False, name=None)))


def append_csv(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def run_model(model, hist_target, hist_covariates, future_covariates):
    if model == "TabPFN_v3":
        return rr.tabpfn_forecast(hist_target, hist_covariates, future_covariates, mode="local")
    if model == "TabPFN_v2":
        return rr.tabpfn_forecast(
            hist_target,
            hist_covariates,
            future_covariates,
            mode="local",
            tabpfn_model_config=rr.tabpfn_v2_model_config(),
        )
    if model == "TabICLv2":
        return rr.tabicl_forecast(hist_target, hist_covariates, future_covariates)
    raise ValueError(model)


def run_screen(frame, configs, models):
    towers = {
        tower: frame.loc[frame["tower"].eq(tower)].set_index("Datetime").sort_index()
        for tower in TOWERS
    }
    completed = read_completed()
    blocks = available_blocks(frame)
    total = len(blocks) * sum(
        len(configs) if model != "TabPFN_v2" else 3 for model in models
    )
    done = 0
    started = time.time()

    v2_configs = {"BASE_34", "BASE_species_37", "BASE_ALL_52"}
    for tower, year, _ in blocks:
        anchor = pd.Timestamp(f"{year}-12-16")
        dates = pd.date_range(anchor + pd.Timedelta(days=1), periods=N_DAYS, freq="D")
        dft = towers[tower]
        hist = dft.loc[:anchor]
        hist_target = hist["y_observed"]
        y_true = dft["y_observed"].reindex(dates)
        y_gapfilled = dft["y_gapfilled"].reindex(dates)
        y_climatology = rr.doy_climatology(hist_target.dropna(), dates)

        for model in models:
            for config, features in configs.items():
                if model == "TabPFN_v2" and config not in v2_configs:
                    continue
                key = (tower, year, model, config)
                done += 1
                if key in completed:
                    print(f"[{done}/{total}] resume-skip T{tower} {year} {model} {config}", flush=True)
                    continue
                call_started = time.time()
                try:
                    prediction = run_model(
                        model,
                        hist_target,
                        hist[features],
                        dft.loc[dates, features],
                    ).reindex(dates)
                    rows = pd.DataFrame(
                        {
                            "date": dates,
                            "lead_day": np.arange(1, N_DAYS + 1),
                            "tower": tower,
                            "anchor_year": year,
                            "model": model,
                            "config": config,
                            "protocol": "observed_history_known_future_fx",
                            "n_features": len(features),
                            "y_predict": prediction.to_numpy(),
                            "y_true": y_true.to_numpy(),
                            "y_gapfilled": y_gapfilled.to_numpy(),
                            "y_climatology": np.asarray(y_climatology),
                        }
                    )
                    append_csv(rows, CHAINS_PATH)
                    elapsed = time.time() - call_started
                    print(
                        f"[{done}/{total}] T{tower} {year} {model} {config}: {elapsed:.1f}s",
                        flush=True,
                    )
                except Exception as exc:
                    error = pd.DataFrame(
                        [
                            {
                                "tower": tower,
                                "anchor_year": year,
                                "model": model,
                                "config": config,
                                "error_type": type(exc).__name__,
                                "error": str(exc)[:1000],
                                "traceback": traceback.format_exc()[-4000:],
                            }
                        ]
                    )
                    append_csv(error, ERRORS_PATH)
                    print(f"[{done}/{total}] ERROR {key}: {exc}", flush=True)

    print(f"Screen completed in {(time.time() - started) / 60:.1f} min", flush=True)


def score_target(group, target_col):
    valid = group[[target_col, "y_predict", "y_climatology"]].notna().all(axis=1)
    g = group.loc[valid]
    if g.empty:
        return None
    residual = g[target_col].to_numpy() - g["y_predict"].to_numpy()
    clim_residual = g[target_col].to_numpy() - g["y_climatology"].to_numpy()
    y = g[target_col].to_numpy()
    pred = g["y_predict"].to_numpy()
    mae = float(np.mean(np.abs(residual)))
    clim_mae = float(np.mean(np.abs(clim_residual)))
    rmse = float(np.sqrt(np.mean(residual**2)))
    denom = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1.0 - np.sum(residual**2) / denom) if denom > 0 else np.nan
    return {
        "n": len(g),
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAE_climatology": clim_mae,
        "MASE": mae / clim_mae if clim_mae > 0 else np.nan,
        "bias": float(np.mean(pred - y)),
    }


def make_metrics():
    chains = pd.read_csv(CHAINS_PATH, parse_dates=["date"])
    chains["bin"] = pd.cut(
        chains["lead_day"], bins=BIN_EDGES, labels=BIN_LABELS, include_lowest=True
    ).astype(str)

    keys = ["tower", "anchor_year", "model", "config", "protocol", "n_features", "bin"]
    records = []
    for key, group in chains.groupby(keys, sort=False, observed=True):
        base = dict(zip(keys, key))
        for target, target_col in [("observed", "y_true"), ("gapfilled", "y_gapfilled")]:
            values = score_target(group, target_col)
            if values is not None:
                records.append({**base, "target": target, **values})
    bins = pd.DataFrame(records)
    bins.to_csv(BIN_PATH, index=False)

    summary_records = []
    summary_keys = ["model", "config", "protocol", "n_features", "target"]
    for key, group in bins.groupby(summary_keys, sort=False):
        finite_mase = group["MASE"].notna() & group["n"].gt(0)
        finite_r2 = group["R2"].notna() & group["n"].gt(0)
        n = int(group.loc[finite_mase, "n"].sum())
        summary_records.append(
            {
                **dict(zip(summary_keys, key)),
                "n": n,
                "n_blocks": int(group[["tower", "anchor_year"]].drop_duplicates().shape[0]),
                "MASE": float(
                    np.average(group.loc[finite_mase, "MASE"], weights=group.loc[finite_mase, "n"])
                ),
                "MAE": float(
                    np.average(group.loc[finite_mase, "MAE"], weights=group.loc[finite_mase, "n"])
                ),
                "RMSE_bin_weighted": float(
                    np.average(group.loc[finite_mase, "RMSE"], weights=group.loc[finite_mase, "n"])
                ),
                "R2_bin_weighted": float(
                    np.average(group.loc[finite_r2, "R2"], weights=group.loc[finite_r2, "n"])
                )
                if finite_r2.any()
                else np.nan,
                "bias": float(
                    np.average(group.loc[finite_mase, "bias"], weights=group.loc[finite_mase, "n"])
                ),
            }
        )
    summary = pd.DataFrame(summary_records).sort_values(["target", "MASE"])
    summary.to_csv(SUMMARY_PATH, index=False)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["TabPFN_v3", "TabPFN_v2", "TabICLv2"],
        default=["TabPFN_v3", "TabICLv2", "TabPFN_v2"],
    )
    args = parser.parse_args()

    frame = pd.read_csv(DATA_PATH, low_memory=False)
    frame["Datetime"] = pd.to_datetime(frame["Datetime"], format="mixed")
    configs = build_configs(frame)
    blocks = available_blocks(frame)
    manifest = {
        "experiment": "B17 foundation-model feature screen",
        "data_path": str(DATA_PATH.relative_to(ROOT)),
        "models": args.models,
        "configs": {name: columns for name, columns in configs.items()},
        "feature_counts": {name: len(columns) for name, columns in configs.items()},
        "scored_blocks": [
            {"tower": tower, "anchor_year": year, "n_observed": n}
            for tower, year, n in blocks
        ],
        "primary_metric": "observed-target, lead-bin-weighted climatology MASE",
        "protocol": "observed pre-anchor target history + known post-anchor fx drivers",
        "forbidden_inputs": ["post-anchor y_observed", "post-anchor y_gapfilled", "post-anchor ar_fc_dlag1"],
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    run_screen(frame, configs, args.models)
    summary = make_metrics()
    print("\nObserved-target ranking")
    print(summary.loc[summary["target"].eq("observed")].to_string(index=False))
    print(f"\nSaved raw chains: {CHAINS_PATH}")
    print(f"Saved bin metrics: {BIN_PATH}")
    print(f"Saved summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
