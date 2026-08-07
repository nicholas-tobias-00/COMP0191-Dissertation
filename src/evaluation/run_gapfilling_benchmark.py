"""Runs every production-worthy FCH4 gap-filling model through the shared gap-CV harness
(`gap_cv.py`) and reports MAE, nMAE, RMSE, R2 (sklearn), and R2_OLS (Zhu et al. 2023a's own
OLS/Pearson-r2 convention) for each -- the model roster this project actually validated as
worth having in production (D-79): MDS (literature-correct floor), Mean (trivial floor), MICE,
HyperImpute, RFm met-only, RFm champion (production-adopted), TabICL-solo (benchmark-best at
T2/T4). D5-D8's negative feedback-feature experiments (env-KNN, TICA/uncertainty-as-feature) are
deliberately not included here -- they were rejected, not adopted; see
`notebooks/03c_gap_filling_revisited/summary.md` S15-18 for that full record.

HyperImpute is genuinely slow (per-column AutoML search every chained-equations iteration) --
skip it with --models if you just want a quick check.

Usage:
    python src/evaluation/run_gapfilling_benchmark.py
    python src/evaluation/run_gapfilling_benchmark.py --models mean,mds,rfm,tabicl
    python src/evaluation/run_gapfilling_benchmark.py --towers 4
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
from gap_cv import headline  # noqa: E402
from gapfill_rfm import load_ext, TOWERS, evaluate_tower as rfm_evaluate  # noqa: E402
from gapfill_mds import evaluate_tower as mds_evaluate  # noqa: E402
from gapfill_baselines import (  # noqa: E402
    evaluate_tower_mean, evaluate_tower_mice, evaluate_tower_hyperimpute, evaluate_tower_met_only,
)
from gapfill_tabicl import evaluate_tower as tabicl_evaluate  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
BENCHMARKS_CSV = RESULTS_DIR / "benchmarks.csv"
OUT_CSV = RESULTS_DIR / "gapfilling_benchmark.csv"

MODEL_RUNNERS = {
    "mean": lambda t, d: evaluate_tower_mean(t, d),
    "mds": lambda t, d: mds_evaluate(t, d),
    "mice": lambda t, d: evaluate_tower_mice(t, pooled=True, d_all=d),
    "hyperimpute": lambda t, d: evaluate_tower_hyperimpute(t, pooled=True, d_all=d),
    "met_only": lambda t, d: evaluate_tower_met_only(t, d),
    "rfm": lambda t, d: rfm_evaluate(t, d),
    "tabicl": lambda t, d: tabicl_evaluate(t, d, n_bags=1),
}
MODEL_ORDER = ["mean", "mds", "mice", "hyperimpute", "met_only", "rfm", "tabicl"]


def append_to_benchmarks_ledger(rows_df, date_str):
    """Appends new rows to results/benchmarks.csv (append-only ledger, CLAUDE.md convention) --
    existing rows/columns are never modified; nMAE/R2_OLS are new columns, blank for old rows."""
    existing = pd.read_csv(BENCHMARKS_CSV)
    new_rows = pd.DataFrame({
        "replication": "D-79", "model": rows_df["model"], "tower": "Tower" + rows_df["tower"].astype(str),
        "R2": rows_df["R2"], "RMSE": rows_df["RMSE"], "MAE": rows_df["MAE"], "MBE": pd.NA,
        "date": date_str, "notes": "gap-filling model benchmark (D-79, src/evaluation/run_gapfilling_benchmark.py)",
        "nMAE": rows_df["nMAE"], "R2_OLS": rows_df["R2_OLS"],
    })
    combined = pd.concat([existing, new_rows], ignore_index=True, sort=False)
    combined.to_csv(BENCHMARKS_CSV, index=False)
    print(f"Appended {len(new_rows)} rows to {BENCHMARKS_CSV}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODEL_ORDER),
                     help=f"comma-separated subset of {MODEL_ORDER}")
    ap.add_argument("--towers", default=",".join(str(t) for t in TOWERS))
    ap.add_argument("--no-ledger", action="store_true", help="skip appending to results/benchmarks.csv")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",")]
    towers = [int(t.strip()) for t in args.towers.split(",")]
    unknown = set(models) - set(MODEL_RUNNERS)
    if unknown:
        raise ValueError(f"unknown model(s) {unknown}, expected subset of {MODEL_ORDER}")

    d_all = load_ext()
    print(f"Loaded EXT layer {d_all.shape}\n")

    rows = []
    for model in models:
        if model == "hyperimpute":
            print("[hyperimpute] this is genuinely slow (per-column AutoML search) -- be patient")
        for t in towers:
            h = headline(MODEL_RUNNERS[model](t, d_all))
            rows.append({"model": model, "tower": t, **h})
            print(f"[{model:11s}] Tower {t}: MAE={h['MAE']:7.2f}  nMAE={h['nMAE']:.3f}  "
                  f"RMSE={h['RMSE']:7.2f}  R2={h['R2']:6.3f}  R2_OLS={h['R2_OLS']:.3f}")

    results = pd.DataFrame(rows)
    results.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}  ({len(results)} rows)")

    print("\nHeadline R2 by model/tower:")
    print(results.pivot(index="model", columns="tower", values="R2").reindex(MODEL_ORDER).round(3))

    if not args.no_ledger:
        append_to_benchmarks_ledger(results, pd.Timestamp.now().strftime("%Y-%m-%d"))


if __name__ == "__main__":
    main()
