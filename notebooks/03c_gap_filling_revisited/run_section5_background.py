"""Run Section 5 independently of Jupyter and report durable status/log files."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import traceback

import pandas as pd

from section5_gap_filling import Section5Benchmark, Section5Config


HERE = Path(__file__).resolve().parent
SPLIT_DIR = HERE / "_data_gf_training" / "laglead_full_v2_split"
OUTPUT_DIR = HERE / "_data_focus"
CACHE_DIR = HERE / "_model_cache_focus"
STATUS_PATH = OUTPUT_DIR / "section5_background_status.json"
DOMAIN = {
    2: ("2017-10-01", "2019-06-30"),
    4: ("2017-10-01", "2023-12-31"),
    9: ("2020-02-01", "2023-12-31"),
}
SPLIT_FILES = {
    "train": "train.csv",
    "test": "test.csv",
    "to_gapfill": "to_gapfill.csv",
    "other_pre2018": "other_pre2018.csv",
}


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def write_status(payload: dict, status_path: Path = STATUS_PATH) -> None:
    status_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = status_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(status_path)


def load_prepared() -> pd.DataFrame:
    parts = []
    schemas = {}
    for split, filename in SPLIT_FILES.items():
        path = SPLIT_DIR / filename
        if not path.exists():
            raise FileNotFoundError(path)
        part = pd.read_csv(path, parse_dates=["Datetime"], low_memory=False)
        part["split"] = split
        schemas[split] = set(part.columns) - {"split"}
        parts.append(part)
        print(f"loaded {split:>13s}: {len(part):>7,} rows", flush=True)
    reference = schemas["train"]
    mismatch = {
        split: sorted(columns ^ reference)
        for split, columns in schemas.items()
        if columns != reference
    }
    if mismatch:
        raise ValueError(f"Prepared split schemas differ: {mismatch}")
    return (
        pd.concat(parts, ignore_index=True)
        .sort_values(["tower", "Datetime"])
        .reset_index(drop=True)
    )


def real_gap_table(prepared: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tower, (start, final_day) in DOMAIN.items():
        expected = pd.date_range(start, pd.Timestamp(final_day), freq="h", name="Datetime")
        target = (
            prepared.loc[prepared["tower"].eq(tower), ["Datetime", "target"]]
            .set_index("Datetime")
            .reindex(expected)["target"]
        )
        missing = target.isna()
        run_id = missing.ne(missing.shift(fill_value=False)).cumsum()
        for _, block in target.loc[missing].groupby(run_id.loc[missing]):
            rows.append(
                {
                    "tower": tower,
                    "start": block.index.min(),
                    "end": block.index.max(),
                    "n_hours": len(block),
                }
            )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["development", "record"],
        default="development",
        help="development=2 RF/MDS reps; record=20 RF/MDS reps",
    )
    parser.add_argument(
        "--tabicl",
        action="store_true",
        help="Also run TabICL (1 development rep or 5 record reps).",
    )
    parser.add_argument(
        "--rf-arms",
        nargs="+",
        default=None,
        help="Optional subset of feature-arm names.",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=None,
        help="Override the RF/MDS repetition count (default: 2 development / 20 record).",
    )
    parser.add_argument(
        "--skip-mds",
        action="store_true",
        help="Skip the MDS baseline (useful when only comparing RF feature arms).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output subdirectory name under this notebook's folder (default: _data_focus). "
        "Use a distinct name to avoid overwriting a prior run's saved CSVs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    is_record = args.mode == "record"
    output_dir = HERE / args.output_dir if args.output_dir else OUTPUT_DIR
    status_path = output_dir / "section5_background_status.json"
    reps = args.reps if args.reps is not None else (20 if is_record else 2)
    run_mds = not args.skip_mds
    status = {
        "status": "starting",
        "pid": os.getpid(),
        "started_at": now(),
        "mode": args.mode,
        "reps": reps,
        "run_mds": run_mds,
        "run_tabicl": bool(args.tabicl),
        "rf_arms": args.rf_arms or "all",
        "output_directory": str(output_dir),
    }
    write_status(status, status_path)
    print("=" * 78, flush=True)
    print(f"Section 5 background benchmark started at {status['started_at']}", flush=True)
    print(f"PID={status['pid']} mode={args.mode} TabICL={args.tabicl}", flush=True)

    try:
        prepared = load_prepared()
        gaps = real_gap_table(prepared)
        config = Section5Config(
            mask_fraction=0.25,
            mds_rf_repetitions=reps,
            tabicl_repetitions=5 if is_record else 1,
            run_tabicl=args.tabicl,
            rf_n_estimators=500,
            rf_min_samples_leaf=5,
            rf_max_features=1.0,
            rf_n_jobs=-1,
            use_model_cache=True,
            tabicl_row_cap=10_000,
            tabicl_feature_arms=("compact24",),
            target_neighbour_horizon=72,
            tica_lag_hours=24,
        )
        benchmark = Section5Benchmark(
            prepared=prepared,
            domain=DOMAIN,
            output_dir=output_dir,
            cache_dir=CACHE_DIR,
            real_gaps=gaps,
            config=config,
        )
        rf_arms = tuple(args.rf_arms or benchmark.feature_arms)
        status.update(
            {
                "status": "running",
                "configuration": {
                    "mask_fraction": config.mask_fraction,
                    "mds_rf_repetitions": config.mds_rf_repetitions,
                    "tabicl_repetitions": config.tabicl_repetitions,
                    "rf_n_estimators": config.rf_n_estimators,
                    "rf_arms": list(rf_arms),
                    "tabicl_feature_arms": (
                        list(config.tabicl_feature_arms) if args.tabicl else []
                    ),
                },
            }
        )
        write_status(status, status_path)
        print(f"RF arms ({len(rf_arms)}): {rf_arms}", flush=True)
        print(f"Repetitions: {reps}  run_mds={run_mds}", flush=True)
        print(f"Output dir: {output_dir}  (cache dir, reused/shared: {CACHE_DIR})", flush=True)
        print("Beginning shared-fold benchmark...", flush=True)

        raw = benchmark.run(
            run_mds=run_mds,
            rf_feature_arms=rf_arms,
            run_tabicl=args.tabicl,
            tabicl_feature_arms=config.tabicl_feature_arms,
        )
        metrics, summary, audit = benchmark.save_results(raw)
        if "champion30_reference" in rf_arms:
            champion_check = benchmark.champion_reference_check(summary)
            print("\nHistorical champion reconstruction check:", flush=True)
            print(champion_check.to_string(index=False), flush=True)

        status.update(
            {
                "status": "completed",
                "completed_at": now(),
                "raw_prediction_rows": len(raw),
                "metric_rows": len(metrics),
                "summary_rows": len(summary),
                "feature_audit_rows": len(audit),
                "outputs": [
                    str(output_dir / "section5_raw_predictions.csv"),
                    str(output_dir / "section5_metrics_by_rep.csv"),
                    str(output_dir / "section5_summary.csv"),
                    str(output_dir / "section5_feature_audit.csv"),
                ],
            }
        )
        write_status(status, status_path)
        print(f"\nCompleted successfully at {status['completed_at']}", flush=True)
        return 0
    except BaseException as exc:
        status.update(
            {
                "status": "failed",
                "failed_at": now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        write_status(status, status_path)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
