"""Shared gap-CV harness for FCH4 gap-filling evaluation (D-79, 03c_gap_filling_revisited).

Ported from `temp_gap_filling_pipeline.ipynb`'s own harness (`insert_calendar_gaps`/`SCENARIOS`/
`mets`/`med_metrics`) so every gap-filling model in `src/models/` can be scored the same way --
same deterministic (seed=0) synthetic-gap placement, same metric set (MAE, nMAE, RMSE, R2 both
sklearn's and Zhu et al. 2023a's OLS/Pearson-r2 convention).

Methodology: full-period gap-CV (D-34/D-49) -- synthetic gaps placed anywhere in a tower's own
DOMAIN window (not a train/test year split), since gap-filling is interpolation, not forecasting.
5 scenarios (very-short/short/medium/long/mixed gap lengths) x N_REPS independent draws;
median-across-reps-then-median-across-scenarios is this project's standing aggregation (never
pool raw reps -- CLAUDE.md).

This module provides primitives only (gap placement + scoring) -- each model in `src/models/`
implements its own `evaluate_tower(t)`-style function using these directly, mirroring the
notebook's own per-model `run_*` functions, rather than forcing every model through one generic
predict-function contract (pooling/feature needs differ too much across MDS/imputers/RF/TabICL
for that to stay simple).
"""
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import mae, nmae, rmse, r2, r2_ols  # noqa: E402  single source of truth

DOMAIN = {2: ("2017-10-01", "2019-06-30"), 4: ("2017-10-01", "2023-12-31"),
          9: ("2020-02-01", "2023-12-31")}
SCENARIOS = {"vs": 1, "s": 4, "m": 32, "l": 288, "m1": "mixed"}
MASK_FRAC = 0.25
N_REPS = 2   # matches F-09a's exact scope, not F-08's original 5


def dom_mask(idx, t):
    a, b = DOMAIN[t]
    return (idx >= pd.Timestamp(a)) & (idx <= pd.Timestamp(b))


def insert_calendar_gaps(df_qc, target, domain_mask, gap_hours, n_reps=N_REPS, seed=0):
    """Randomly place n_reps independent sets of non-overlapping calendar-shaped gaps within the
    domain, each covering ~MASK_FRAC of the domain's valid target points. seed=0 default makes
    this fully deterministic across processes/reruns -- every caller in this project relies on
    that for reproducibility."""
    dom_ts = df_qc.index[domain_mask]; valid = df_qc.loc[domain_mask, target].notna().values
    n = len(dom_ts); target_n = max(1, int(valid.sum() * MASK_FRAC)); rb = np.random.default_rng(seed)
    reps = []
    for _ in range(n_reps):
        rng = np.random.default_rng(int(rb.integers(0, 2**31))); occ = np.zeros(n, bool); m = 0
        for sp in rng.permutation(n):
            if m >= target_n:
                break
            gh = int(rng.choice([1, 4, 32, 288])) if gap_hours == "mixed" else gap_hours
            ep = min(int(sp) + gh, n)
            if occ[sp:ep].any():
                continue
            occ[sp:ep] = True; m += int(valid[sp:ep].sum())
        reps.append(dom_ts[occ & valid])
    return reps


def gapfilling_metrics(y, p, train_std=None):
    """The 4 headline gap-filling metrics for one set of (actual, predicted) pairs: MAE, nMAE
    (requires train_std -- the training target's own std, a fixed per-tower constant, NOT the
    test-set std), RMSE, plus both R2 conventions (sklearn's r2_score and Zhu et al.'s OLS/
    Pearson-r2)."""
    return dict(MAE=mae(y, p), nMAE=nmae(y, p, train_std) if train_std is not None else np.nan,
                RMSE=rmse(y, p), R2=r2(y, p), R2_OLS=r2_ols(y, p))


def median_metrics(rows):
    """rows: list of per-rep gapfilling_metrics() dicts. Median across reps -- this project's
    standing rule, never pool raw reps together."""
    if not rows:
        return {k: np.nan for k in ["MAE", "nMAE", "RMSE", "R2", "R2_OLS"]}
    keys = rows[0].keys()
    return {k: float(np.nanmedian([row[k] for row in rows])) for k in keys}


def headline(scenario_results):
    """Median-across-scenarios of each metric, given {scenario: median_metrics_dict} -- the
    project's standing headline aggregation (median-across-reps-then-median-across-scenarios)."""
    keys = next(iter(scenario_results.values())).keys()
    return {k: float(np.nanmedian([v[k] for v in scenario_results.values()])) for k in keys}
