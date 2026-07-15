"""Extends B-09's day-of-year climatology baseline (originally computed for a single tower/anchor
in B09_recursive_rollout.ipynb) to full coverage -- all 3 towers x 5 anchors (2018-2022) -- and
uses it to recompute MASE for the full B-10/B-13 11-model roster, alongside the existing
flat-persistence-scaled MASE.

Motivation (user-raised, live discussion): chain-persistence (the anchor day's real value, held
flat for the full 365-day chain) is a legitimate but "easy" MASE baseline for a series with real
seasonality -- it ignores season entirely, so a model that merely tracks the seasonal cycle can
beat it without demonstrating much skill, especially at long lead times. Hyndman & Koehler's own
MASE paper recommends a SEASONAL-naive scaling for seasonal series. This project's closest
analogue is `rr.doy_climatology()` (day-of-year mean, +/-7-day window, computed from strictly
pre-anchor real y_observed history) -- already implemented and used once, informally, in B-09's
single-anchor smoke test (D-53), but never extended to full coverage or used to rescale B-10/B-13's
headline metrics.

This does NOT change the project's standing MASE convention (D-37: scale against out-of-sample
persistence) -- that stays the primary metric everywhere per CLAUDE.md. This adds
climatology-scaled MASE as a SECONDARY comparison column, computed from the SAME already-existing
chain predictions in `b10_b13_full_chains.csv` (no models refit -- R2/RMSE/MAE/WAPE/Correlation
are baseline-independent and unchanged; only MASE's denominator changes).

Model roster: the 11 non-S03 columns already in b10_b13_full_chains.csv (RF/XGB/LightGBM/SARIMAX/
2 ensembles/TFT/TabPFN/DLinear/LSTM/TabICLv2) -- S-03's degraded-driver variants are explicitly out
of scope for this pass (user asked for "B10 & B13", not S-03).
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
MODEL_ORDER = ["RF", "XGB", "LightGBM", "SARIMAX", "Ensemble_unweighted", "Ensemble_MASEweighted",
               "TFT", "TabPFN", "DLinear", "LSTM", "TabICLv2"]
DL_MODELS = {"TFT", "DLinear", "LSTM"}  # use y_true_tft, per the project's established asymmetry
METRICS = ["R2", "RMSE", "MAE", "MASE", "WAPE", "Correlation"]


def build_climatology_chains():
    """Reproduces B09_recursive_rollout.ipynb's exact climatology recipe (hist_obs = strictly
    pre-anchor real y_observed, rr.doy_climatology(window=7)) for every (tower, anchor_year)."""
    dv = pd.read_csv(f"{HOURLY}/forecast_daily_v2.csv", low_memory=False)
    dv["Datetime"] = pd.to_datetime(dv["Datetime"], format="mixed")
    T = {t: dv[dv.tower == t].set_index("Datetime").sort_index() for t in TOWERS}

    rows = []
    for tower in TOWERS:
        dft = T[tower]
        for yr in ANCHOR_YEARS:
            anchor = pd.Timestamp(f"{yr}-12-16")
            target_dates = pd.date_range(anchor + pd.Timedelta(days=1), periods=N_DAYS, freq="D")
            hist_obs = dft.loc[:anchor - pd.Timedelta(days=1), "y_observed"].dropna()
            clim = rr.doy_climatology(hist_obs, target_dates, window=7)
            rows.append(pd.DataFrame({"date": target_dates.strftime("%Y-%m-%d"), "tower": tower,
                                       "anchor_year": yr, "Climatology": clim}))
    out = pd.concat(rows, ignore_index=True)
    print(f"[OK] Built climatology chains: {len(out)} rows ({len(TOWERS)} towers x "
          f"{len(ANCHOR_YEARS)} anchors x {N_DAYS} days)")
    return out


def merge_into_full_chains(clim_df):
    chains = pd.read_csv(f"{RESULTS}/b10_b13_full_chains.csv")
    if "Climatology" in chains.columns:
        raise RuntimeError("b10_b13_full_chains.csv already has a Climatology column -- "
                            "restore the pre-merge backup before rerunning this script")
    before = len(chains)
    chains = chains.merge(clim_df, on=["date", "tower", "anchor_year"], how="left")
    assert len(chains) == before, "merge changed row count -- key mismatch"
    n_nan = chains["Climatology"].isna().sum()
    if n_nan:
        # Expected, real-data-driven: Tower 9 has ZERO real y_observed history before its 2018/2019
        # anchors (confirmed directly -- n_hist_obs=0 for both), so doy_climatology's global_mean
        # fallback is itself undefined (nanmean of an empty array). Not a bug -- same T9 data-
        # scarcity already documented throughout this project (D-18 lineage).
        combos = (chains[chains["Climatology"].isna()][["tower", "anchor_year"]]
                  .drop_duplicates().to_records(index=False).tolist())
        print(f"[INFO] {n_nan} rows have NaN Climatology (no pre-anchor real y_observed history): "
              f"(tower, anchor_year) = {combos}")
    chains.to_csv(f"{RESULTS}/b10_b13_full_chains.csv", index=False)
    print(f"[OK] Merged Climatology column into b10_b13_full_chains.csv ({len(chains)} rows, "
          f"{len(chains.columns)} cols)")
    return chains


def recompute_mase_vs_climatology(chains):
    """For every (tower, anchor_year, model), reruns rr.bin_metrics with y_persist=Climatology
    instead of the stored `persistence` column. R2/RMSE/MAE/WAPE/Correlation are recomputed too
    (cheap, and guarantees identical row/bin masking to the persistence-scaled tables) but are
    mathematically identical to the existing summary files -- only MASE differs."""
    all_rows = []
    for tower in TOWERS:
        for yr in ANCHOR_YEARS:
            sub = chains[(chains.tower == tower) & (chains.anchor_year == yr)].copy()
            sub["date"] = pd.to_datetime(sub["date"])
            sub = sub.sort_values("date")
            anchor = pd.Timestamp(f"{yr}-12-16")
            target_dates = pd.DatetimeIndex(sub["date"])
            clim = sub["Climatology"].values
            # T9 2018/2019 have zero pre-anchor real y_observed history (confirmed directly) --
            # doy_climatology's global_mean fallback is itself undefined there, so every bin's
            # MASE_climatology is correctly NaN, not a crash (sklearn's mean_absolute_error raises
            # rather than propagating NaN, unlike numpy) -- skip the y_persist arg in that case.
            clim_arg = None if np.isnan(clim).all() else clim

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
    out.to_csv(f"{RESULTS}/b10_b13_climatology_mase_summary.csv", index=False)
    print(f"[OK] Saved b10_b13_climatology_mase_summary.csv ({len(out)} rows)")
    return out


def wavg(g, col):
    d = g.dropna(subset=[col])
    return (d[col] * d["n"]).sum() / d["n"].sum() if d["n"].sum() > 0 else np.nan


def build_pooled(summary_df):
    per_anchor = (summary_df.groupby(["model", "anchor_year"])
                  .apply(lambda g: pd.Series({m: wavg(g, m) for m in METRICS}), include_groups=False)
                  .reset_index())
    return per_anchor.groupby("model")[METRICS].mean().round(3).reindex(MODEL_ORDER)


def build_by_tower(summary_df):
    per_anchor = (summary_df.groupby(["tower", "model", "anchor_year"])
                  .apply(lambda g: pd.Series({m: wavg(g, m) for m in METRICS}), include_groups=False)
                  .reset_index())
    out = per_anchor.groupby(["tower", "model"])[METRICS].mean().round(3)
    out = out.reindex(pd.MultiIndex.from_product([TOWERS, MODEL_ORDER], names=["tower", "model"]))
    return out


def load_persistence_raw():
    """Model 1's existing raw per-bin summaries -- the persistence-scaled MASE already on record."""
    core = pd.read_csv(f"{RESULTS}/b10_b13_rerun_summary.csv")
    dl = pd.read_csv(f"{RESULTS}/b10_b13_dl_extension_summary.csv")
    tabicl = pd.read_csv(f"{RESULTS}/b10_b13_tabicl_extension_summary.csv")
    return pd.concat([core, dl, tabicl], ignore_index=True)


def build_comparison_tables(clim_summary):
    persist_raw = load_persistence_raw()

    persist_pooled = build_pooled(persist_raw)[["MASE"]].rename(columns={"MASE": "MASE_persistence"})
    clim_pooled = build_pooled(clim_summary)[["MASE"]].rename(columns={"MASE": "MASE_climatology"})
    r2_pooled = build_pooled(persist_raw)[["R2"]].rename(columns={"R2": "R2"})
    cmp_all = pd.concat([persist_pooled, clim_pooled, r2_pooled], axis=1)
    cmp_all["verdict"] = np.where(
        cmp_all["MASE_climatology"] < cmp_all["MASE_persistence"],
        "better (lower) vs climatology", "worse (higher) vs climatology")
    cmp_all.to_csv(f"{RESULTS}/b10_b13_climatology_mase_table_all_towers.csv")
    print("[OK] Saved b10_b13_climatology_mase_table_all_towers.csv")
    print(cmp_all)

    persist_bt = build_by_tower(persist_raw)[["MASE"]].rename(columns={"MASE": "MASE_persistence"})
    clim_bt = build_by_tower(clim_summary)[["MASE"]].rename(columns={"MASE": "MASE_climatology"})
    cmp_bt = pd.concat([persist_bt, clim_bt], axis=1)
    cmp_bt.to_csv(f"{RESULTS}/b10_b13_climatology_mase_table_by_tower.csv")
    print("[OK] Saved b10_b13_climatology_mase_table_by_tower.csv")

    return cmp_all, cmp_bt


def main():
    clim_df = build_climatology_chains()
    chains = merge_into_full_chains(clim_df)
    clim_summary = recompute_mase_vs_climatology(chains)
    build_comparison_tables(clim_summary)


if __name__ == "__main__":
    main()
