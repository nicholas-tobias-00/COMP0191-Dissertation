"""Consolidated, correctly-baselined forecasting metrics outline across the full 11-model roster,
both targets (y_observed / y_gapfilled), all metrics (MASE, R2, RMSE, WAPE, Correlation).

MASE uses the D-80 climatology convention throughout, on the MATCHING target for each column
(CLIM for observed, Climatology_gf -- D-96's fix -- for gapfilled) -- the two conventions that were
previously silently mixed (D-96). RMSE/WAPE/Correlation/R2 are baseline-independent and pulled
directly as n-weighted means from the same already-saved per-bin raw files used throughout B-16.

"Best config" per model = lowest MASE_climatology on the OBSERVED target (D-36/D-37's authoritative
metric) -- this is what determines champion status; the gapfilled columns are then read off that
SAME config for a genuine same-config, both-targets comparison, not each column's own best config.

Run from project root:  python notebooks/05_benchmarking/b16_full_metrics_outline.py
"""
import sys

import numpy as np
import pandas as pd

ROOT = r"c:\Users\Nicholas\Documents\COMP0191 MSc Artificial Intelligence for Sustainable Development Project"
RESULTS = rf"{ROOT}\results"


def load_raw():
    sources = [
        pd.read_csv(f"{RESULTS}/b16_foundation_models_v3_summary.csv"),        # TabPFN, TabICLv2
        pd.read_csv(f"{RESULTS}/b16_recursive_rollout_v3_summary_vs_gapfilled.csv"),  # RF/XGB/LightGBM/SARIMAX/Ensembles
        pd.read_csv(f"{RESULTS}/b16_dl_models_v3_summary.csv"),                # TFT/DLinear/LSTM
    ]
    return pd.concat(sources, ignore_index=True)


def single_stage_wavg(df, value_cols, n_col="n"):
    """Matches D-80's own `mase_climatology()` convention exactly (temp_forecasting_pipeline.ipynb,
    cell 16 -- the function that produced the officially-cited 0.715 champion MASE): drop rows with
    a NaN in the value column, then take ONE n-weighted mean per (model, config) directly across all
    (tower, anchor_year, bin) rows -- no anchor-then-mean intermediate stage. Reused here (rather
    than a different two-stage convention that exists elsewhere in this project, e.g.
    b16_recursive_rollout_v3_gapfilled.py's wavg()/per_anchor/allt) specifically so this outline's
    MASE_observed column reproduces the already-published 0.715 headline bit-for-bit instead of
    introducing a THIRD aggregation convention and a new number to reconcile."""
    rows = []
    for (model, cfg), g in df.groupby(["model", "config"]):
        row = {"model": model, "config": cfg}
        for c in value_cols:
            gc = g.dropna(subset=[c, n_col])
            gc = gc[gc[n_col] > 0]
            row[c] = np.average(gc[c], weights=gc[n_col]) if len(gc) else np.nan
        row["n"] = int(g[n_col].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def wavg_metrics(df, target):
    sub = df[df.target == target]
    out = single_stage_wavg(sub, ["R2", "RMSE", "WAPE", "Correlation"])
    return out.rename(columns={c: f"{c}_{target}" for c in ["R2", "RMSE", "WAPE", "Correlation"]})


def mase_climatology(raw, clim, target, clim_col="MAE_climatology"):
    df = raw[raw.target == target].merge(clim, on=["tower", "anchor_year", "bin"], how="left")
    df = df.dropna(subset=["MAE", clim_col, "n"])
    df = df[(df.n > 0) & (df[clim_col] > 0)]
    df["MASE_val"] = df["MAE"] / df[clim_col]
    out = single_stage_wavg(df, ["MASE_val"])
    return out.rename(columns={"MASE_val": f"MASE_{target}"}).drop(columns="n")


def main():
    raw = load_raw()
    clim_obs = pd.read_csv(f"{RESULTS}/_today_climatology_baseline.csv")
    clim_gf = pd.read_csv(f"{RESULTS}/b16_climatology_gf_baseline_v3.csv")

    mase_obs = mase_climatology(raw, clim_obs, "observed")
    mase_gf = mase_climatology(raw, clim_gf, "gapfilled", clim_col="MAE_climatology_gf")
    metrics_obs = wavg_metrics(raw, "observed")
    metrics_gf = wavg_metrics(raw, "gapfilled")

    full = mase_obs.merge(metrics_obs, on=["model", "config"]).merge(
        mase_gf, on=["model", "config"], how="left").merge(
        metrics_gf.drop(columns="n"), on=["model", "config"], how="left")

    best = full.loc[full.groupby("model")["MASE_observed"].idxmin()].sort_values("MASE_observed")
    cols = ["model", "config", "MASE_observed", "R2_observed", "RMSE_observed", "WAPE_observed",
            "Correlation_observed", "MASE_gapfilled", "R2_gapfilled", "RMSE_gapfilled",
            "WAPE_gapfilled", "Correlation_gapfilled", "n"]
    best = best[cols]
    best.to_csv(f"{RESULTS}/b16_full_metrics_outline.csv", index=False)
    print(f"[OK] Saved b16_full_metrics_outline.csv ({len(best)} rows)")
    print(best.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
