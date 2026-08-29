"""B18 additive completion of tower-month target normalisation.

The main B18 sweep intentionally preserves six failed early-anchor attempts where
Tower 9 had no observed pre-anchor target.  This separate experiment uses pooled
pre-anchor robust statistics only for an unseen tower, and otherwise uses the
tower/month -> tower -> pooled fallback hierarchy.  Existing B18 files are never
modified.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(NOTEBOOK_DIR))

import B17_foundation_screen as b17s
import B17_direct_and_recursive_foundations as b17d
import B18_direct_structure as core


RESULTS = ROOT / "results"
CHAINS_PATH = RESULTS / "b18_monthly_fallback_chains.csv"
SUMMARY_PATH = RESULTS / "b18_monthly_fallback_summary.csv"
ERRORS_PATH = RESULTS / "b18_monthly_fallback_errors.csv"
MANIFEST_PATH = RESULTS / "b18_monthly_fallback_manifest.json"
BASELINE_PATH = RESULTS / "_today_climatology_baseline.csv"

SPECS = [
    {
        "experiment_id": "B18M01",
        "model": "Direct_TabPFN_v2",
        "feature_set": "BASE_ALL_52",
        "target_norm": "tower_month_robust_safe",
        "training": "all_history",
    },
    {
        "experiment_id": "B18M02",
        "model": "Direct_TabPFN_v2",
        "feature_set": "ALL_antecedent",
        "target_norm": "tower_month_robust_safe",
        "training": "all_history",
    },
    {
        "experiment_id": "B18M03",
        "model": "Direct_TabICLv2",
        "feature_set": "BASE_ALL_52",
        "target_norm": "tower_month_robust_safe",
        "training": "all_history",
    },
]


def append_csv(frame, path):
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def safe_monthly_transform(train):
    work = train[["tower", "Datetime", "y_observed"]].copy()
    work["month"] = work["Datetime"].dt.month
    pooled_center = float(work["y_observed"].median())
    pooled_iqr = float(work["y_observed"].quantile(0.75) - work["y_observed"].quantile(0.25))
    if not np.isfinite(pooled_iqr) or pooled_iqr <= 0:
        pooled_iqr = 1.0

    by_tower = work.groupby("tower")["y_observed"]
    tower_center = by_tower.median().to_dict()
    tower_q = by_tower.quantile([0.25, 0.75]).unstack()
    tower_scale = (tower_q[0.75] - tower_q[0.25]).replace(0, np.nan).to_dict()
    month_center = work.groupby(["tower", "month"])["y_observed"].median().to_dict()

    def center_scale(rows):
        centers = []
        scales = []
        for tower, month in zip(rows["tower"], rows["Datetime"].dt.month):
            centers.append(month_center.get((tower, month), tower_center.get(tower, pooled_center)))
            scale = tower_scale.get(tower, pooled_iqr)
            scales.append(scale if np.isfinite(scale) and scale > 0 else pooled_iqr)
        return np.asarray(centers, dtype=float), np.asarray(scales, dtype=float)

    return center_scale


def completed_keys():
    if not CHAINS_PATH.exists():
        return set()
    old = pd.read_csv(CHAINS_PATH, usecols=["anchor_year", "experiment_id"])
    return set(map(tuple, old.drop_duplicates().itertuples(index=False, name=None)))


def run(frame, feature_sets):
    completed = completed_keys()
    for year in core.ANCHOR_YEARS:
        anchor = pd.Timestamp(f"{year}-12-16")
        dates = pd.date_range(anchor + pd.Timedelta(days=1), periods=core.N_DAYS, freq="D")
        future = frame.loc[frame["Datetime"].isin(dates)].copy()
        future["b18_anchor_year"] = year
        train = core.training_rows(frame, anchor, "all_history")
        transform = safe_monthly_transform(train)

        for spec in SPECS:
            key = (year, spec["experiment_id"])
            if key in completed:
                print(f"[resume-skip] {year} {spec['experiment_id']}")
                continue
            started = time.time()
            try:
                features = list(dict.fromkeys(feature_sets[spec["feature_set"]] + core.STATIC_FEATURES))
                prediction = core.fit_predict_one(
                    spec["model"], train, future, features, transform
                )
                rows = core.raw_rows(frame, future, spec, prediction, len(features))
                rows["protocol"] = "observed_history_known_future_fx_safe_global_fallback"
                append_csv(rows, CHAINS_PATH)
                print(
                    f"[{year}] {spec['experiment_id']} {spec['model']} "
                    f"{spec['feature_set']} n={len(train)} {time.time() - started:.1f}s",
                    flush=True,
                )
            except Exception as exc:
                append_csv(
                    pd.DataFrame(
                        [{
                            "anchor_year": year,
                            **spec,
                            "error_type": type(exc).__name__,
                            "error": str(exc)[:1000],
                            "traceback": traceback.format_exc()[-4000:],
                        }]
                    ),
                    ERRORS_PATH,
                )
                print(f"[ERROR] {year} {spec['experiment_id']}: {exc}", flush=True)


def score():
    chains = pd.read_csv(CHAINS_PATH)
    chains["bin"] = pd.cut(
        chains["lead_day"], core.BIN_EDGES, labels=core.BIN_LABELS, include_lowest=True
    ).astype(str)
    baseline = pd.read_csv(BASELINE_PATH)
    chains = chains.merge(
        baseline[["tower", "anchor_year", "bin", "MAE_climatology"]],
        on=["tower", "anchor_year", "bin"],
        how="left",
    )
    rows = []
    group_cols = ["experiment_id", "model", "feature_set", "target_norm", "training", "protocol", "n_features"]
    for key, group in chains.groupby(group_cols, sort=False):
        info = dict(zip(group_cols, key))
        for target, target_col in [("observed", "y_true"), ("gapfilled", "y_gapfilled")]:
            good = group[[target_col, "y_predict", "MAE_climatology"]].notna().all(axis=1)
            good &= group["MAE_climatology"].gt(0)
            g = group.loc[good]
            y = g[target_col].to_numpy(float)
            p = g["y_predict"].to_numpy(float)
            err = y - p
            total = float(np.sum((y - y.mean()) ** 2))
            rows.append(
                {
                    **info,
                    "target": target,
                    "n": len(g),
                    "n_blocks": int(g[["tower", "anchor_year"]].drop_duplicates().shape[0]),
                    "MASE": float(np.mean(np.abs(err) / g["MAE_climatology"].to_numpy(float))),
                    "MAE": float(np.mean(np.abs(err))),
                    "RMSE": float(np.sqrt(np.mean(err**2))),
                    "R2": float(1 - np.sum(err**2) / total) if total > 0 else np.nan,
                    "bias": float(np.mean(p - y)),
                }
            )
    result = pd.DataFrame(rows).sort_values(["target", "MASE"])
    result.to_csv(SUMMARY_PATH, index=False)
    return result


def main():
    frame = pd.read_csv(b17s.DATA_PATH, low_memory=False)
    frame["Datetime"] = pd.to_datetime(frame["Datetime"], format="mixed")
    frame = b17d.add_b17_features(frame)
    frame = core.add_antecedent_features(frame)
    feature_sets, _ = core.build_feature_sets(frame)
    MANIFEST_PATH.write_text(
        json.dumps(
            {
                "experiment": "B18 tower-month normalisation safe fallback completion",
                "protocol": "observed history and known future fx; no future methane input",
                "fallback": "tower-month median -> tower median -> pooled pre-anchor median; tower IQR -> pooled pre-anchor IQR",
                "experiments": SPECS,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    started = time.time()
    run(frame, feature_sets)
    result = score()
    print(f"\nCompleted in {(time.time() - started) / 60:.1f} min")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
