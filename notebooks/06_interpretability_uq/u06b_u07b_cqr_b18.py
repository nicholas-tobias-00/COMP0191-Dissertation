"""U-06b/U-07b: CQR spike-fix + LSU-stratified CQR for the B18-derived architecture -- Phase 5 of
the additive B18-integration plan (2026-08-20). Pure recalibration, ZERO new model calls: U-08's
and U-05b's chains (`u08_chains.csv`, `u05b_chains.csv`) already carry the exact schema
(anchor_year, eval_tower, model, date, q05, median, q95, y_true) that `evaluate_cqr()`/
`spike_coverage_check()` (U-06) and `evaluate_lsu_cqr()` (U-07) already expect -- both imported
UNCHANGED, an 8th/9th reuse of `rr.conformal_margins_by_bin()` across this project's UQ arc.

U-06's spike-coverage question was specifically about the champion's spike behaviour; U-08 IS the
B18 champion's own chains, so this closes that gap directly, not just by analogy.

Run from project root:  python notebooks/06_interpretability_uq/u06b_u07b_cqr_b18.py
"""
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(r"C:\Users\Nicholas\Documents\COMP0191 MSc Artificial Intelligence for Sustainable Development Project")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notebooks" / "06_interpretability_uq"))

import u06_cqr_recalibration as u06
import u07_lsu_stratified_cqr as u07

RESULTS = ROOT / "results"
JOBS = [
    ("U08", "u08_chains.csv", "u08_summary.csv", [2018, 2019, 2020, 2021, 2022], [2, 4, 9]),
    ("U05b", "u05b_chains.csv", "u05b_summary.csv", [2018, 2019, 2020, 2021, 2022], [2, 4, 9]),
]


def wavg(g, col):
    vals = g[col]
    if vals.isna().all():
        import numpy as np

        return np.nan
    return (vals * g["n"]).sum() / g["n"].sum() if g["n"].sum() > 0 else float("nan")


def run_u06b():
    for label, chains_file, summary_file, anchor_years, towers in JOBS:
        print("=" * 70)
        print(f"U-06b: CQR recalibration of {label}'s chains")
        print("=" * 70)
        chains = pd.read_csv(RESULTS / chains_file)
        old_summary = pd.read_csv(RESULTS / summary_file)

        cqr_summary = u06.evaluate_cqr(chains, anchor_years, towers)
        out_path = RESULTS / f"u06b_{label.lower()}_cqr_summary.csv"
        cqr_summary.to_csv(out_path, index=False)
        print(f"[OK] Saved {out_path.name} ({len(cqr_summary)} rows)")

        agg = cqr_summary.groupby(["model", "eval_tower"]).apply(
            lambda g: pd.Series(
                {"cqr_picp": wavg(g, "cqr_picp"), "cqr_mpiw": wavg(g, "cqr_mpiw"), "cqr_pinball": wavg(g, "cqr_pinball")}
            ),
            include_groups=False,
        ).reset_index()
        print(agg.round(4).to_string(index=False))

        old_summary_renamed = old_summary.rename(columns={"conformal_margin": "conformal_margin"})
        u06.spike_coverage_check(chains, cqr_summary, old_summary_renamed, anchor_years, towers, f"b18_{label.lower()}")


def run_u07b():
    lsu = u07.load_lsu()
    for label, chains_file, _, anchor_years, towers in JOBS:
        print("=" * 70)
        print(f"U-07b: LSU-density-stratified CQR recalibration of {label}'s chains")
        print("=" * 70)
        chains = pd.read_csv(RESULTS / chains_file)
        summary = u07.evaluate_lsu_cqr(chains, lsu, anchor_years, towers)
        out_path = RESULTS / f"u07b_{label.lower()}_lsu_cqr_summary.csv"
        summary.to_csv(out_path, index=False)
        print(f"[OK] Saved {out_path.name} ({len(summary)} rows)")

        agg = summary.groupby(["model", "lsu_tier"]).apply(
            lambda g: pd.Series(
                {"picp": wavg(g, "lsu_cqr_picp"), "mpiw": wavg(g, "lsu_cqr_mpiw"), "pinball": wavg(g, "lsu_cqr_pinball"), "n": g["n"].sum()}
            ),
            include_groups=False,
        ).reset_index()
        agg["lsu_tier"] = pd.Categorical(agg["lsu_tier"], categories=u07.LSU_TIERS, ordered=True)
        print(agg.sort_values(["model", "lsu_tier"]).round(3).to_string(index=False))
        print()


def main():
    run_u06b()
    run_u07b()


if __name__ == "__main__":
    main()
