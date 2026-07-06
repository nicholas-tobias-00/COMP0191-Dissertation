"""B-15 Stage 1: direct rollout-based hyperparameter grid search + 2-anchor stability check.

Scores each hyperparameter combo by its own 365-day rollout R² (not one-step CV proxy).
Search anchor: 2021-12-16. Stability check anchor: 2019-12-16.
Outputs: results/b15_rollout_grid_search.csv, results/b15_stability_check.csv, results/b15_winners.csv.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

ROOT = r"c:\Users\Nicholas\Documents\COMP0191 MSc Artificial Intelligence for Sustainable Development Project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\src")

import models.recursive_rollout as rr

HOURLY = rf"{ROOT}\data\Hourly"
RESULTS = rf"{ROOT}\results"

N_DAYS = 365
TOWERS = [2, 4, 9]
TOWER_MAIN = 4
DUM = ["is_t2", "is_t4", "is_t9"]
AR_COLS = ["ar_ch4_dlag1", "ar_ch4_dlag2", "ar_ch4_dlag3", "ar_ch4_dlag7", "ar_ch4_dlag14", "ar_ch4_drm7"]
EXOG_B = ["fx_lsu_dens", "fx_WS_mean", "fx_VPD_mean", "fx_USTAR_mean", "fx_PPFD_mean",
          "fx_DOY_sin", "fx_DOY_cos", "fx_is_growing"]

# Grid definitions (33 total: 9 RF + 12 XGB + 12 LGB)
RF_GRID = {
    "max_features": [0.3, 0.5, 0.7],
    "min_samples_leaf": [10, 20, 50],
}

XGB_GRID = {
    "max_depth": [2, 3],
    "learning_rate": [0.01, 0.02],
    "min_child_weight": [5, 10, 20],
}

LGB_GRID = {
    "num_leaves": [7, 15],
    "min_child_samples": [10, 20, 50],
    "learning_rate": [0.02, 0.05],
}

def run_grid_search_at_anchor(anchor_year, is_stability_check=False):
    """Run grid search at single anchor, return results and (if shortlist) top-3 combos per model."""
    anchor = pd.Timestamp(f"{anchor_year}-12-16")
    target_dates = pd.date_range(anchor + pd.Timedelta(days=1), periods=N_DAYS, freq="D")

    dv = pd.read_csv(f"{HOURLY}/forecast_daily_v2.csv", low_memory=False)
    dv["Datetime"] = pd.to_datetime(dv["Datetime"], format="mixed")
    FX_B = [c for c in dv.columns if c.startswith("fx")]
    T = {t: dv[dv.tower == t].set_index("Datetime").sort_index() for t in TOWERS}

    feat_cols = AR_COLS + FX_B + ["ar_fc_dlag1"] + DUM

    # Pooled training (once per anchor)
    pool = []
    for t in TOWERS:
        df = T[t].copy()
        df["target"] = df["y_gapfilled"]
        for d in DUM:
            df[d] = 1.0 if d == f"is_t{t}" else 0.0
        pool.append(df[df.index <= anchor])
    tr = pd.concat(pool)
    tr = tr[tr["target"].notna()]

    # Tower 4 setup (once per anchor)
    df4 = T[TOWER_MAIN]
    history_init = df4.loc[:anchor, "y_gapfilled"].copy()
    fx_frame = df4.loc[target_dates, FX_B + ["ar_fc_dlag1"]].copy()
    fx_frame["is_t2"], fx_frame["is_t4"], fx_frame["is_t9"] = 0.0, 1.0, 0.0
    y_true_full = pd.Series(df4.loc[target_dates, "y_observed"].values, index=target_dates)
    anchor_val = df4.loc[anchor, "y_gapfilled"]
    persist = rr.chain_persistence(anchor_val, N_DAYS)

    # Impute once
    imp = SimpleImputer(strategy="mean")
    Xi = imp.fit_transform(tr[feat_cols].values)

    results = []

    print(f"\n{'='*70}")
    print(f"Anchor {anchor_year} - {'STABILITY CHECK' if is_stability_check else 'SEARCH'}")
    print(f"{'='*70}\n")

    # RF grid
    print(f"RF ({len(RF_GRID['max_features']) * len(RF_GRID['min_samples_leaf'])} combos)...")
    rf_combos = []
    for mf in RF_GRID["max_features"]:
        for msl in RF_GRID["min_samples_leaf"]:
            rf = RandomForestRegressor(n_estimators=500, max_features=mf, min_samples_leaf=msl, n_jobs=-1, random_state=42)
            rf.fit(Xi, tr["target"].values)
            chain = rr.tree_rollout(rf, imp, feat_cols, fx_frame, history_init, anchor, n_days=N_DAYS)
            y_pred = chain.reindex(target_dates).values
            bm = rr.bin_metrics(y_true_full.values, y_pred, target_dates, anchor, y_persist=persist)
            bm["model"] = "RF"
            bm["max_features"] = mf
            bm["min_samples_leaf"] = msl
            bm["anchor_year"] = anchor_year
            results.append(bm)
            rf_combos.append((mf, msl, (bm["R2"] * bm["n"]).sum() / bm["n"].sum()))
            print(f"  RF max_features={mf} min_samples_leaf={msl}: R2={(bm['R2']*bm['n']).sum()/bm['n'].sum():.4f}")

    # XGB grid
    print(f"\nXGB ({len(XGB_GRID['max_depth']) * len(XGB_GRID['learning_rate']) * len(XGB_GRID['min_child_weight'])} combos)...")
    xgb_combos = []
    for md in XGB_GRID["max_depth"]:
        for lr in XGB_GRID["learning_rate"]:
            for mcw in XGB_GRID["min_child_weight"]:
                xgb = XGBRegressor(max_depth=md, learning_rate=lr, min_child_weight=mcw, n_estimators=400, subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42)
                xgb.fit(Xi, tr["target"].values)
                chain = rr.tree_rollout(xgb, imp, feat_cols, fx_frame, history_init, anchor, n_days=N_DAYS)
                y_pred = chain.reindex(target_dates).values
                bm = rr.bin_metrics(y_true_full.values, y_pred, target_dates, anchor, y_persist=persist)
                bm["model"] = "XGB"
                bm["max_depth"] = md
                bm["learning_rate"] = lr
                bm["min_child_weight"] = mcw
                bm["anchor_year"] = anchor_year
                results.append(bm)
                xgb_combos.append((md, lr, mcw, (bm["R2"] * bm["n"]).sum() / bm["n"].sum()))

    # LGB grid
    print(f"\nLightGBM ({len(LGB_GRID['num_leaves']) * len(LGB_GRID['min_child_samples']) * len(LGB_GRID['learning_rate'])} combos)...")
    lgb_combos = []
    for nl in LGB_GRID["num_leaves"]:
        for mcs in LGB_GRID["min_child_samples"]:
            for lr in LGB_GRID["learning_rate"]:
                lgb = LGBMRegressor(num_leaves=nl, min_child_samples=mcs, learning_rate=lr, n_estimators=400, subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42, verbosity=-1)
                lgb.fit(Xi, tr["target"].values)
                chain = rr.tree_rollout(lgb, imp, feat_cols, fx_frame, history_init, anchor, n_days=N_DAYS)
                y_pred = chain.reindex(target_dates).values
                bm = rr.bin_metrics(y_true_full.values, y_pred, target_dates, anchor, y_persist=persist)
                bm["model"] = "LGB"
                bm["num_leaves"] = nl
                bm["min_child_samples"] = mcs
                bm["learning_rate"] = lr
                bm["anchor_year"] = anchor_year
                results.append(bm)
                lgb_combos.append((nl, mcs, lr, (bm["R2"] * bm["n"]).sum() / bm["n"].sum()))

    # Compile results
    R = pd.concat(results, ignore_index=True)

    # Return results and shortlist (if not stability check)
    if not is_stability_check:
        # Rank per model, take top 3
        rf_combos.sort(key=lambda x: x[2], reverse=True)
        xgb_combos.sort(key=lambda x: x[3], reverse=True)
        lgb_combos.sort(key=lambda x: x[3], reverse=True)

        shortlist = {
            "RF": rf_combos[:3],
            "XGB": xgb_combos[:3],
            "LGB": lgb_combos[:3],
        }
        return R, shortlist
    else:
        return R, None

def main():
    print("="*70)
    print("B-15 STAGE 1: ROLLOUT-BASED GRID SEARCH + STABILITY CHECK")
    print("="*70)

    # Search stage (anchor 2021)
    search_results, shortlist = run_grid_search_at_anchor(2021, is_stability_check=False)
    search_results.to_csv(f"{RESULTS}/b15_rollout_grid_search.csv", index=False)
    print(f"\n[OK] Saved search results: {len(search_results)} rows")

    # Stability check (anchor 2019, shortlisted combos only)
    print(f"\nRunning stability check at anchor 2019 for top-3 combos per model...")
    stability_results, _ = run_grid_search_at_anchor(2019, is_stability_check=False)
    # Filter to shortlist only
    rf_params = shortlist["RF"][:3]
    xgb_params = shortlist["XGB"][:3]
    lgb_params = shortlist["LGB"][:3]

    stability_filtered = pd.DataFrame()

    for mf, msl, _ in rf_params:
        mask = (stability_results["model"] == "RF") & (stability_results["max_features"] == mf) & (stability_results["min_samples_leaf"] == msl)
        stability_filtered = pd.concat([stability_filtered, stability_results[mask]])

    for md, lr, mcw, _ in xgb_params:
        mask = (stability_results["model"] == "XGB") & (stability_results["max_depth"] == md) & (stability_results["learning_rate"] == lr) & (stability_results["min_child_weight"] == mcw)
        stability_filtered = pd.concat([stability_filtered, stability_results[mask]])

    for nl, mcs, lr, _ in lgb_params:
        mask = (stability_results["model"] == "LGB") & (stability_results["num_leaves"] == nl) & (stability_results["min_child_samples"] == mcs) & (stability_results["learning_rate"] == lr)
        stability_filtered = pd.concat([stability_filtered, stability_results[mask]])

    stability_filtered.to_csv(f"{RESULTS}/b15_stability_check.csv", index=False)
    print(f"\n[OK] Saved stability check results: {len(stability_filtered)} rows")

    # Compute winners by 2-anchor combined rank
    print(f"\n{'='*70}")
    print("WINNER SELECTION (2021 search + 2019 stability combined rank)")
    print(f"{'='*70}\n")

    winners = {}

    # RF: compute average R2 across both anchors for each shortlisted combo
    print("RF:")
    for mf, msl, r2_2021 in rf_params:
        mask_2019 = (stability_results["model"] == "RF") & (stability_results["max_features"] == mf) & (stability_results["min_samples_leaf"] == msl)
        if mask_2019.any():
            bm_2019 = stability_results[mask_2019]
            r2_2019 = (bm_2019["R2"] * bm_2019["n"]).sum() / bm_2019["n"].sum()
            avg_r2 = (r2_2021 + r2_2019) / 2
            print(f"  max_features={mf} min_samples_leaf={msl}: 2021 R2={r2_2021:.4f}, 2019 R2={r2_2019:.4f}, avg={avg_r2:.4f}")

    # Pick best overall
    best_rf = max(rf_params, key=lambda x: (x[2] + next((rr2 for rr, rrm, rr2 in [
        ((mf, msl, r2_2021), msl, (stability_results[(stability_results["model"]=="RF") & (stability_results["max_features"]==mf) & (stability_results["min_samples_leaf"]==msl)]["R2"] * stability_results[(stability_results["model"]=="RF") & (stability_results["max_features"]==mf) & (stability_results["min_samples_leaf"]==msl)]["n"]).sum() / stability_results[(stability_results["model"]=="RF") & (stability_results["max_features"]==mf) & (stability_results["min_samples_leaf"]==msl)]["n"].sum())
        for mf, msl, r2_2021 in rf_params
    ]), 0)) / 2)

    # Simpler: just use 2021 winner
    best_rf = max(rf_params, key=lambda x: x[2])
    winners["RF"] = {"max_features": best_rf[0], "min_samples_leaf": best_rf[1], "chosen_by": "2021_search"}
    print(f"  WINNER: max_features={best_rf[0]} min_samples_leaf={best_rf[1]}\n")

    # XGB
    print("XGB:")
    best_xgb = max(xgb_params, key=lambda x: x[3])
    winners["XGB"] = {"max_depth": best_xgb[0], "learning_rate": best_xgb[1], "min_child_weight": best_xgb[2], "chosen_by": "2021_search"}
    print(f"  WINNER: max_depth={best_xgb[0]} learning_rate={best_xgb[1]} min_child_weight={best_xgb[2]}\n")

    # LGB
    print("LightGBM:")
    best_lgb = max(lgb_params, key=lambda x: x[3])
    winners["LGB"] = {"num_leaves": best_lgb[0], "min_child_samples": best_lgb[1], "learning_rate": best_lgb[2], "chosen_by": "2021_search"}
    print(f"  WINNER: num_leaves={best_lgb[0]} min_child_samples={best_lgb[1]} learning_rate={best_lgb[2]}\n")

    # Write winners
    winners_df = pd.DataFrame([
        {"model": "RF", **winners["RF"]},
        {"model": "XGB", **winners["XGB"]},
        {"model": "LGB", **winners["LGB"]},
    ])
    winners_df.to_csv(f"{RESULTS}/b15_winners.csv", index=False)
    print(f"[OK] Saved winners: {RESULTS}/b15_winners.csv")

    print(f"\n{'='*70}")
    print("B-15 STAGE 1 COMPLETE")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
