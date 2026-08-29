"""Fix for a real methodological gap in F-10/D-67's "scored against y_gapfilled" secondary
metric (`b16_final_table_vs_gapfilled_best_config.csv`, `F10_results.md`): that table's MASE was
computed via `rr.bin_metrics(..., y_persist=persist)` where `persist = chain_persistence(anchor_val,
N_DAYS)` and `anchor_val = y_gapfilled` at the anchor date -- i.e. the OLD (D-37) chain-persistence
baseline, never rescored under D-80's climatology convention. So the two numbers this project has
been quoting side by side (0.715 observed-target climatology-scored headline vs. 0.944
gapfilled-target persistence-scored secondary) differ in BOTH target AND baseline convention at
once, not target alone -- caught via direct user question, 2026-08-13.

This script builds a genuinely fair comparison: a `Climatology_gf` baseline (day-of-year mean,
+/-7-day window, built from `y_gapfilled` history -- D-71's existing, already-approved recipe for
"baseline should be sourced from the same series as the target it's evaluating against", reused
unchanged from `b10_b13_climatology_gf_baseline.py`) scored against `y_gapfilled` truth, then
MASE_climatology_gf = MAE_model / MAE_climatology_gf, recomputed purely arithmetically from
already-saved per-bin MAE columns (no new model calls) across the full 11-model roster.

Run from project root:  python notebooks/05_benchmarking/b16_gapfilled_climatology_fix.py
"""
import sys

import numpy as np
import pandas as pd

ROOT = r"c:\Users\Nicholas\Documents\COMP0191 MSc Artificial Intelligence for Sustainable Development Project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\src")

import models.recursive_rollout as rr

HOURLY = rf"{ROOT}\data\Hourly"
RESULTS = rf"{ROOT}\results"

TOWERS = [2, 4, 9]
N_DAYS = 365
ANCHOR_YEARS = [2018, 2019, 2020, 2021, 2022]
BINS = ((1, 7), (8, 30), (31, 90), (91, 180), (181, 270), (271, 365))


def build_climatology_gf_baseline(T):
    """Per (tower, anchor_year, bin) MAE of a y_gapfilled-sourced day-of-year climatology,
    scored against y_gapfilled truth -- the fair denominator for the gapfilled-target secondary
    metric. Mirrors `build_climatology_baseline()` (temp_forecasting_pipeline.ipynb cell 15,
    D-80's observed-target baseline) exactly, except both the climatology basis AND the truth are
    y_gapfilled instead of y_observed -- so this is apples-to-apples on the target axis, unlike
    the old chain-persistence secondary metric."""
    rows = []
    for yr in ANCHOR_YEARS:
        anchor = pd.Timestamp(f"{yr}-12-16")
        target_dates = pd.date_range(anchor + pd.Timedelta(days=1), periods=N_DAYS, freq="D")
        for tower in TOWERS:
            dft = T[tower]
            hist_gf = dft.loc[:anchor - pd.Timedelta(days=1), "y_gapfilled"].dropna()
            y_true = dft["y_gapfilled"].reindex(target_dates).values
            clim = rr.doy_climatology(hist_gf, target_dates)
            bin_labels = rr.lead_time_bin(target_dates, anchor, bins=BINS) if hasattr(rr, "lead_time_bin") else None
            if bin_labels is None:
                lead = np.array([(d - anchor).days for d in target_dates])
                bin_labels = np.full(len(target_dates), None, dtype=object)
                for lo, hi in BINS:
                    m = (lead >= lo) & (lead <= hi)
                    bin_labels[m] = f"{lo}-{hi}"
            for lo, hi in BINS:
                lbl = f"{lo}-{hi}"
                m = bin_labels == lbl
                yt = y_true[m]; cl = np.asarray(clim)[m]
                valid = np.isfinite(yt) & np.isfinite(cl)
                n = int(valid.sum())
                mae_clim = float(np.mean(np.abs(yt[valid] - cl[valid]))) if n > 0 else np.nan
                rows.append({"tower": tower, "anchor_year": yr, "bin": lbl, "n_clim_gf": n,
                             "MAE_climatology_gf": mae_clim})
    return pd.DataFrame(rows)


def rescale(df, clim_gf, target_filter="gapfilled"):
    df = df[df.target == target_filter].copy()
    df = df.merge(clim_gf, on=["tower", "anchor_year", "bin"], how="left")
    df = df.dropna(subset=["MAE", "MAE_climatology_gf", "n"])
    df = df[(df.n > 0) & (df.MAE_climatology_gf > 0)]
    df["MASE_climatology_gf"] = df["MAE"] / df["MAE_climatology_gf"]
    rows = []
    for (model, cfg), g in df.groupby(["model", "config"]):
        rows.append({"model": model, "config": cfg,
                     "MASE_climatology_gf": np.average(g["MASE_climatology_gf"], weights=g["n"]),
                     "R2": np.average(g["R2"], weights=g["n"]),
                     "n_total": int(g["n"].sum())})
    return pd.DataFrame(rows)


def main():
    dv = pd.read_csv(f"{HOURLY}/forecast_daily_v3.csv", low_memory=False)
    dv["Datetime"] = pd.to_datetime(dv["Datetime"], format="mixed")
    T = {t: dv[dv.tower == t].set_index("Datetime").sort_index() for t in TOWERS}

    clim_gf = build_climatology_gf_baseline(T)
    clim_gf.to_csv(f"{RESULTS}/b16_climatology_gf_baseline_v3.csv", index=False)
    print(f"[OK] Climatology_gf baseline: {len(clim_gf)} rows, "
          f"{clim_gf.MAE_climatology_gf.notna().sum()} valid")

    sources = {
        "foundation": pd.read_csv(f"{RESULTS}/b16_foundation_models_v3_summary.csv"),      # TabPFN, TabICLv2
        "trees": pd.read_csv(f"{RESULTS}/b16_recursive_rollout_v3_summary_vs_gapfilled.csv"),  # RF/XGB/LightGBM/SARIMAX/Ensembles
        "dl": pd.read_csv(f"{RESULTS}/b16_dl_models_v3_summary.csv"),                        # TFT/DLinear/LSTM
    }

    rescaled_all = []
    for name, df in sources.items():
        r = rescale(df, clim_gf, target_filter="gapfilled")
        r["source"] = name
        rescaled_all.append(r)
        print(f"[OK] {name}: {len(r)} model/config rows rescaled")

    rescaled = pd.concat(rescaled_all, ignore_index=True)
    rescaled.to_csv(f"{RESULTS}/b16_gapfilled_climatology_fix_all_configs.csv", index=False)
    print(f"[OK] Saved b16_gapfilled_climatology_fix_all_configs.csv ({len(rescaled)} rows)")

    best = rescaled.loc[rescaled.groupby("model")["MASE_climatology_gf"].idxmin()].sort_values("MASE_climatology_gf")
    best = best[["model", "config", "MASE_climatology_gf", "R2", "n_total"]]
    best.to_csv(f"{RESULTS}/b16_gapfilled_climatology_fix_best_config.csv", index=False)
    print(f"[OK] Saved b16_gapfilled_climatology_fix_best_config.csv")
    print(best.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
