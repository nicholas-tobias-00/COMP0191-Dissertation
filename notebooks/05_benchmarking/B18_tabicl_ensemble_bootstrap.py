"""Block-bootstrap uncertainty for the additive B18 TabICL ensembles."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import B18_evaluate_and_plot as ev


INPUT = ROOT / "results" / "b18_tabicl_ensemble_components.csv"
OUTPUT = ROOT / "results" / "b18_tabicl_ensemble_bootstrap.csv"


def main():
    data = pd.read_csv(INPUT, parse_dates=["date"])
    data["ICL_best_ensemble"] = data[["ICL_antecedent_tower_robust", "ICL_1460_raw", "ICL_1460_tower_robust"]].mean(axis=1)
    data["ICL_best_standalone"] = data["ICL_1460_tower_robust"]
    data["mixed_10pct"] = 0.9 * data["PFN_final_triple"] + 0.1 * data[["ICL_1460_tower_robust", "ICL_1460_raw", "ICL_all_tower_robust"]].mean(axis=1)
    scored = ev.add_scoring(data)
    scored = scored.loc[scored["y_true"].notna() & scored["MAE_climatology"].gt(0)]
    comparisons = [
        ("ICL_best_ensemble", "ICL_best_standalone", "TabICL ensemble minus standalone"),
        ("mixed_10pct", "PFN_final_triple", "10% TabICL mix minus TabPFN ensemble"),
    ]
    blocks = list(scored[["tower", "anchor_year"]].drop_duplicates().itertuples(index=False, name=None))
    rng = np.random.default_rng(1811)
    rows = []
    for candidate, comparator, label in comparisons:
        values = []
        for tower, year in blocks:
            group = scored.loc[scored["tower"].eq(tower) & scored["anchor_year"].eq(year)]
            candidate_loss = np.abs(group["y_true"] - group[candidate]) / group["MAE_climatology"]
            comparator_loss = np.abs(group["y_true"] - group[comparator]) / group["MAE_climatology"]
            values.append((len(group), candidate_loss.sum(), comparator_loss.sum(), candidate_loss.mean() < comparator_loss.mean()))
        values = np.asarray(values, dtype=float)
        draws = []
        for _ in range(10_000):
            sample = values[rng.integers(0, len(values), size=len(values))]
            draws.append(sample[:, 1].sum() / sample[:, 0].sum() - sample[:, 2].sum() / sample[:, 0].sum())
        draws = np.asarray(draws)
        delta = values[:, 1].sum() / values[:, 0].sum() - values[:, 2].sum() / values[:, 0].sum()
        rows.append({
            "comparison": label,
            "candidate": candidate,
            "comparator": comparator,
            "delta_MASE": delta,
            "ci_2.5": np.quantile(draws, 0.025),
            "ci_97.5": np.quantile(draws, 0.975),
            "probability_candidate_better": np.mean(draws < 0),
            "candidate_block_wins": int(values[:, 3].sum()),
            "blocks": len(blocks),
            "resamples": len(draws),
        })
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT, index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
