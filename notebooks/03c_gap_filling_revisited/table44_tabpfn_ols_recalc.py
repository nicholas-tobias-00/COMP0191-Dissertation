"""Recover the missing per-scenario OLS R2 values for Table 4.4's TabPFN row.

The original D-78 comparison persisted aggregate sklearn metrics but not raw
TabPFN predictions. This runner reproduces only that TabPFN arm, checkpoints
each fold, saves raw predictions, and verifies the original sklearn R2 values
before the OLS results are used. It does not edit the report.
"""

from __future__ import annotations

import gc
import json
import runpy
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from scipy.stats import linregress
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from tabpfn import TabPFNRegressor


ROW_CAP = 10_000
RANDOM_STATE = 42
EXPECTED_SKLEARN = {
    2: {"vs": 0.459, "s": 0.704, "m": 0.684, "l": 0.092, "m1": 0.456},
    4: {"vs": 0.455, "s": 0.401, "m": 0.456, "l": 0.214, "m1": 0.346},
    9: {"vs": 0.443, "s": 0.402, "m": 0.543, "l": 0.190, "m1": 0.350},
}


def score(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if np.var(actual) > 0 and np.var(predicted) > 0:
        slope, _, correlation, _, _ = linregress(actual, predicted)
        r2_ols = correlation**2
    else:
        slope, r2_ols = np.nan, np.nan
    error = predicted - actual
    return {
        "R2": float(r2_score(actual, predicted)),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "MAE": float(np.mean(np.abs(error))),
        "MBE": float(np.mean(error)),
        "R2_OLS": float(r2_ols),
        "OLS_slope": float(slope),
    }


def main() -> None:
    started = time.time()
    here = Path(__file__).resolve().parent
    setup = runpy.run_path(str(here / "_d100_ols_runner_setup.py"))
    data_dir = Path(setup["DATA_DIR"])
    checkpoint_dir = data_dir / "table44_tabpfn_ols_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    towers = setup["TOWERS"]
    scenarios = setup["SCENARIOS"]
    features = setup["FEATURES"] + setup["DUM"]
    frame = setup["frame"]
    d_all = setup["d_all"]
    dom_mask = setup["dom_mask"]
    insert_calendar_gaps = setup["insert_calendar_gaps"]
    frames = {tower: frame(tower, True, d_all) for tower in towers}

    raw_parts: list[pd.DataFrame] = []
    summary_rows: list[dict[str, float | int | str]] = []
    for tower in towers:
        target_frame = frames[tower]
        domain = dom_mask(target_frame.index, tower)
        for scenario, gap_hours in scenarios.items():
            fold_scores = []
            for fold, gap_times in enumerate(
                insert_calendar_gaps(target_frame, "target", domain, gap_hours)
            ):
                if len(gap_times) < 5:
                    continue
                checkpoint = checkpoint_dir / f"TabPFN_T{tower}_{scenario}_rep{fold}.csv"
                if checkpoint.exists():
                    raw = pd.read_csv(checkpoint, parse_dates=["Datetime"])
                    print(f"checkpoint: T{tower} {scenario} rep={fold}", flush=True)
                else:
                    base = target_frame.loc[domain & target_frame["target"].notna().to_numpy()]
                    training = base.drop(index=gap_times, errors="ignore")
                    others = []
                    for other_tower in towers:
                        if other_tower == tower:
                            continue
                        other = frames[other_tower]
                        other_domain = dom_mask(other.index, other_tower)
                        others.append(
                            other.loc[other_domain & other["target"].notna().to_numpy()]
                        )
                    training = pd.concat([training] + others, ignore_index=True)
                    training = training.sample(
                        n=min(ROW_CAP, len(training)), random_state=RANDOM_STATE
                    )
                    imputer = SimpleImputer(strategy="mean")
                    x_train = imputer.fit_transform(training[features].to_numpy())
                    x_test = imputer.transform(target_frame.loc[gap_times, features].to_numpy())
                    model = TabPFNRegressor(random_state=RANDOM_STATE)
                    model.fit(x_train, training["target"].to_numpy())
                    predicted = model.predict(x_test)
                    raw = pd.DataFrame(
                        {
                            "Datetime": gap_times,
                            "actual": target_frame.loc[gap_times, "target"].to_numpy(),
                            "pred": predicted,
                            "rep": fold,
                            "scenario": scenario,
                            "tower": tower,
                            "model": "TabPFN",
                        }
                    )
                    raw.to_csv(checkpoint, index=False)
                    del model, x_train, x_test
                    gc.collect()
                    try:
                        import torch

                        torch.cuda.empty_cache()
                    except (ImportError, RuntimeError):
                        pass
                    print(
                        f"computed: T{tower} {scenario} rep={fold} "
                        f"[{time.time() - started:.0f}s]",
                        flush=True,
                    )
                fold_scores.append(score(raw["actual"], raw["pred"]))
                raw_parts.append(raw)

            scenario_score = {
                metric: float(np.nanmedian([row[metric] for row in fold_scores]))
                for metric in ["R2", "RMSE", "MAE", "MBE", "R2_OLS", "OLS_slope"]
            }
            expected = EXPECTED_SKLEARN[tower][scenario]
            difference = scenario_score["R2"] - expected
            verification = "OK" if abs(difference) < 0.002 else "MISMATCH"
            summary_rows.append(
                {
                    "model": "TabPFN",
                    "tower": tower,
                    "scenario": scenario,
                    **scenario_score,
                    "expected_R2_sklearn": expected,
                    "R2_difference": difference,
                    "verification": verification,
                }
            )
            print(
                f"T{tower} {scenario}: sklearn={scenario_score['R2']:.3f} "
                f"(expected {expected:.3f}, {verification}); "
                f"OLS={scenario_score['R2_OLS']:.3f}",
                flush=True,
            )

    summary = pd.DataFrame(summary_rows)
    raw_predictions = pd.concat(raw_parts, ignore_index=True)
    summary_path = data_dir / "table44_tabpfn_ols_summary.csv"
    raw_path = data_dir / "table44_tabpfn_ols_raw_predictions.csv"
    manifest_path = data_dir / "table44_ols_recalc_manifest.json"
    summary.to_csv(summary_path, index=False)
    raw_predictions.to_csv(raw_path, index=False)
    manifest_path.write_text(
        json.dumps(
            {
                "purpose": "OLS R2 replacement values for Table 4.4 TabPFN row",
                "report_edited": False,
                "model": "TabPFNRegressor(random_state=42)",
                "row_cap": ROW_CAP,
                "features": "FEATURES + tower dummies; pooled, matching D-78",
                "fold_protocol": "original insert_calendar_gaps; 5 scenarios x 2 reps x 3 towers",
                "verification_rule": "rerun sklearn R2 within 0.002 of model_comparison.csv",
                "all_verification_passed": bool(summary["verification"].eq("OK").all()),
                "summary": summary_path.name,
                "raw_predictions": raw_path.name,
                "elapsed_seconds": time.time() - started,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved {summary_path}", flush=True)
    print(f"Saved {raw_path}", flush=True)
    print(summary.to_string(index=False), flush=True)
    if not summary["verification"].eq("OK").all():
        raise RuntimeError("TabPFN rerun did not reproduce every original sklearn R2 value")


if __name__ == "__main__":
    main()
