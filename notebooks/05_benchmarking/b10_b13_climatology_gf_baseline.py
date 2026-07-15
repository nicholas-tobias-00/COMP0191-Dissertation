"""Gap-filled-basis variant of D-71's climatology baseline, added after a direct fairness concern
(user-raised): `b10_b13_climatology_baseline.py`'s Climatology column was built from real
`y_observed` history only (B-09's original recipe), while `persistence`'s single anchor value comes
from `y_gapfilled` -- so the original comparison wasn't apples-to-apples. Persistence's one value
can itself be a gap-filler's (smoothed, always-available) output; climatology's entire basis was
sparse, spike-dominated real measurements. This recomputes climatology from `y_gapfilled` history
instead (dense, ~continuous, same series persistence draws its anchor value from) so the two
baselines are now built from the SAME underlying series -- isolating "flat vs. seasonal" from
"real vs. gap-filled data source".

Adds a second new column (`Climatology_gf`) to `b10_b13_full_chains.csv` alongside the existing
`Climatology` (real-y_observed-basis) column -- neither is removed, both are now available for
comparison. Reuses `b10_b13_climatology_baseline.py`'s aggregation helpers directly (no
duplication) since the recipe is identical except for the source series.
"""
import sys

import numpy as np
import pandas as pd

ROOT = r"c:\Users\Nicholas\Documents\COMP0191 MSc Artificial Intelligence for Sustainable Development Project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\src")
sys.path.insert(0, ROOT + r"\notebooks\05_benchmarking")

import models.recursive_rollout as rr
from b10_b13_climatology_baseline import (
    TOWERS, N_DAYS, ANCHOR_YEARS, MODEL_ORDER, DL_MODELS, METRICS,
    build_pooled, build_by_tower, load_persistence_raw,
)

HOURLY = rf"{ROOT}\data\Hourly"
RESULTS = rf"{ROOT}\results"


def build_climatology_gf_chains():
    """Same recipe as build_climatology_chains() (rr.doy_climatology, window=7, strictly
    pre-anchor history) but sourced from y_gapfilled instead of y_observed."""
    dv = pd.read_csv(f"{HOURLY}/forecast_daily_v2.csv", low_memory=False)
    dv["Datetime"] = pd.to_datetime(dv["Datetime"], format="mixed")
    T = {t: dv[dv.tower == t].set_index("Datetime").sort_index() for t in TOWERS}

    rows = []
    for tower in TOWERS:
        dft = T[tower]
        for yr in ANCHOR_YEARS:
            anchor = pd.Timestamp(f"{yr}-12-16")
            target_dates = pd.date_range(anchor + pd.Timedelta(days=1), periods=N_DAYS, freq="D")
            hist_gf = dft.loc[:anchor - pd.Timedelta(days=1), "y_gapfilled"].dropna()
            clim = rr.doy_climatology(hist_gf, target_dates, window=7)
            rows.append(pd.DataFrame({"date": target_dates.strftime("%Y-%m-%d"), "tower": tower,
                                       "anchor_year": yr, "Climatology_gf": clim}))
    out = pd.concat(rows, ignore_index=True)
    print(f"[OK] Built gap-filled-basis climatology chains: {len(out)} rows")
    return out


def merge_into_full_chains(clim_df):
    chains = pd.read_csv(f"{RESULTS}/b10_b13_full_chains.csv")
    if "Climatology_gf" in chains.columns:
        raise RuntimeError("b10_b13_full_chains.csv already has a Climatology_gf column")
    before = len(chains)
    chains = chains.merge(clim_df, on=["date", "tower", "anchor_year"], how="left")
    assert len(chains) == before, "merge changed row count -- key mismatch"
    n_nan = chains["Climatology_gf"].isna().sum()
    print(f"[INFO] {n_nan} rows have NaN Climatology_gf" if n_nan else "[INFO] no NaN Climatology_gf rows")
    chains.to_csv(f"{RESULTS}/b10_b13_full_chains.csv", index=False)
    print(f"[OK] Merged Climatology_gf column into b10_b13_full_chains.csv ({len(chains)} rows, "
          f"{len(chains.columns)} cols)")
    return chains


def recompute_mase_vs_climatology_gf(chains):
    all_rows = []
    for tower in TOWERS:
        for yr in ANCHOR_YEARS:
            sub = chains[(chains.tower == tower) & (chains.anchor_year == yr)].copy()
            sub["date"] = pd.to_datetime(sub["date"])
            sub = sub.sort_values("date")
            anchor = pd.Timestamp(f"{yr}-12-16")
            target_dates = pd.DatetimeIndex(sub["date"])
            clim_gf = sub["Climatology_gf"].values
            clim_arg = None if np.isnan(clim_gf).all() else clim_gf

            for model in MODEL_ORDER:
                if model not in sub.columns:
                    continue
                yp = sub[model].values
                if np.isfinite(yp).sum() == 0:
                    continue
                y_true = sub["y_true_tft"].values if model in DL_MODELS else sub["y_true"].values
                bm = rr.bin_metrics(y_true, yp, target_dates, anchor, y_persist=clim_arg)
                bm["model"] = model
                bm["anchor_year"] = yr
                bm["tower"] = tower
                all_rows.append(bm)

    out = pd.concat(all_rows, ignore_index=True)
    out.to_csv(f"{RESULTS}/b10_b13_climatology_gf_mase_summary.csv", index=False)
    print(f"[OK] Saved b10_b13_climatology_gf_mase_summary.csv ({len(out)} rows)")
    return out


def build_comparison_tables(clim_gf_summary):
    persist_raw = load_persistence_raw()
    clim_obs_raw = pd.read_csv(f"{RESULTS}/b10_b13_climatology_mase_summary.csv")

    persist_pooled = build_pooled(persist_raw)[["MASE"]].rename(columns={"MASE": "MASE_persistence"})
    clim_obs_pooled = build_pooled(clim_obs_raw)[["MASE"]].rename(columns={"MASE": "MASE_climatology_obs"})
    clim_gf_pooled = build_pooled(clim_gf_summary)[["MASE"]].rename(columns={"MASE": "MASE_climatology_gf"})
    r2_pooled = build_pooled(persist_raw)[["R2"]]
    cmp_all = pd.concat([persist_pooled, clim_obs_pooled, clim_gf_pooled, r2_pooled], axis=1)
    cmp_all.to_csv(f"{RESULTS}/b10_b13_climatology_gf_mase_table_all_towers.csv")
    print("[OK] Saved b10_b13_climatology_gf_mase_table_all_towers.csv")
    print(cmp_all)

    persist_bt = build_by_tower(persist_raw)[["MASE"]].rename(columns={"MASE": "MASE_persistence"})
    clim_obs_bt = build_by_tower(clim_obs_raw)[["MASE"]].rename(columns={"MASE": "MASE_climatology_obs"})
    clim_gf_bt = build_by_tower(clim_gf_summary)[["MASE"]].rename(columns={"MASE": "MASE_climatology_gf"})
    cmp_bt = pd.concat([persist_bt, clim_obs_bt, clim_gf_bt], axis=1)
    cmp_bt.to_csv(f"{RESULTS}/b10_b13_climatology_gf_mase_table_by_tower.csv")
    print("[OK] Saved b10_b13_climatology_gf_mase_table_by_tower.csv")

    return cmp_all, cmp_bt


def main():
    clim_df = build_climatology_gf_chains()
    chains = merge_into_full_chains(clim_df)
    clim_gf_summary = recompute_mase_vs_climatology_gf(chains)
    build_comparison_tables(clim_gf_summary)


if __name__ == "__main__":
    main()
