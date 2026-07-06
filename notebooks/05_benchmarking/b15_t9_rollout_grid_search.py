"""B-15 Tower-9 tuning: does an independent rollout-based grid search scored on Tower 9 pick
different hyperparameters than the Tower-4 search (results/b15_winners.csv)?

Motivated by b15_cross_tower_eval.py's finding that T4-tuned hyperparameters generalize poorly to
T9 (e.g. LightGBM_tuned is T4's best single model but T9's *worst*). Tower 9 has usable real
y_observed coverage at 4/5 anchors (2019-2022, 49-74%; 2018 is 0%) -- good enough for its own
search+stability-check pair, unlike Tower 2 (usable at only 1/5 anchors, too data-scarce for an
independent search -- see b15_results_t2_t9.md for that caveat).

Same grid/design as b15_rollout_grid_search.py (33 combos, search anchor 2021, stability anchor
2019 -- both have decent T9 coverage, checked before running), same combined-rank winner selection
(mean of n-weighted R2 across both anchors -- built correctly from the start here, avoiding the
dead-code bug the original T4 script had before its Sonnet-review fix).
"""

import sys
import numpy as np
import pandas as pd
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
TOWER_MAIN = 9
DUM = ["is_t2", "is_t4", "is_t9"]
AR_COLS = ["ar_ch4_dlag1", "ar_ch4_dlag2", "ar_ch4_dlag3", "ar_ch4_dlag7", "ar_ch4_dlag14", "ar_ch4_drm7"]
EXOG_B = ["fx_lsu_dens", "fx_WS_mean", "fx_VPD_mean", "fx_USTAR_mean", "fx_PPFD_mean",
          "fx_DOY_sin", "fx_DOY_cos", "fx_is_growing"]

RF_GRID = {"max_features": [0.3, 0.5, 0.7], "min_samples_leaf": [10, 20, 50]}
XGB_GRID = {"max_depth": [2, 3], "learning_rate": [0.01, 0.02], "min_child_weight": [5, 10, 20]}
LGB_GRID = {"num_leaves": [7, 15], "min_child_samples": [10, 20, 50], "learning_rate": [0.02, 0.05]}


def run_grid_search_at_anchor(anchor_year):
    anchor = pd.Timestamp(f"{anchor_year}-12-16")
    target_dates = pd.date_range(anchor + pd.Timedelta(days=1), periods=N_DAYS, freq="D")

    dv = pd.read_csv(f"{HOURLY}/forecast_daily_v2.csv", low_memory=False)
    dv["Datetime"] = pd.to_datetime(dv["Datetime"], format="mixed")
    FX_B = [c for c in dv.columns if c.startswith("fx")]
    T = {t: dv[dv.tower == t].set_index("Datetime").sort_index() for t in TOWERS}

    feat_cols = AR_COLS + FX_B + ["ar_fc_dlag1"] + DUM

    pool = []
    for t in TOWERS:
        df = T[t].copy()
        df["target"] = df["y_gapfilled"]
        for d in DUM:
            df[d] = 1.0 if d == f"is_t{t}" else 0.0
        pool.append(df[df.index <= anchor])
    tr = pd.concat(pool)
    tr = tr[tr["target"].notna()]

    df9 = T[TOWER_MAIN]
    history_init = df9.loc[:anchor, "y_gapfilled"].copy()
    fx_frame = df9.loc[target_dates, FX_B + ["ar_fc_dlag1"]].copy()
    fx_frame["is_t2"], fx_frame["is_t4"], fx_frame["is_t9"] = 0.0, 0.0, 1.0
    y_true_full = pd.Series(df9.loc[target_dates, "y_observed"].values, index=target_dates)
    anchor_val = df9.loc[anchor, "y_gapfilled"]
    persist = rr.chain_persistence(anchor_val, N_DAYS)

    imp = SimpleImputer(strategy="mean")
    Xi = imp.fit_transform(tr[feat_cols].values)

    results = []
    print(f"\n{'='*70}\nAnchor {anchor_year} (Tower 9)\n{'='*70}\n")

    print(f"RF ({len(RF_GRID['max_features']) * len(RF_GRID['min_samples_leaf'])} combos)...")
    rf_combos = []
    for mf in RF_GRID["max_features"]:
        for msl in RF_GRID["min_samples_leaf"]:
            rf = RandomForestRegressor(n_estimators=500, max_features=mf, min_samples_leaf=msl, n_jobs=-1, random_state=42)
            rf.fit(Xi, tr["target"].values)
            chain = rr.tree_rollout(rf, imp, feat_cols, fx_frame, history_init, anchor, n_days=N_DAYS)
            y_pred = chain.reindex(target_dates).values
            bm = rr.bin_metrics(y_true_full.values, y_pred, target_dates, anchor, y_persist=persist)
            bm["model"], bm["max_features"], bm["min_samples_leaf"], bm["anchor_year"] = "RF", mf, msl, anchor_year
            results.append(bm)
            r2 = (bm["R2"] * bm["n"]).sum() / bm["n"].sum()
            rf_combos.append((mf, msl, r2))
            print(f"  RF max_features={mf} min_samples_leaf={msl}: R2={r2:.4f}")

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
                bm["model"], bm["max_depth"], bm["learning_rate"], bm["min_child_weight"], bm["anchor_year"] = "XGB", md, lr, mcw, anchor_year
                results.append(bm)
                xgb_combos.append((md, lr, mcw, (bm["R2"] * bm["n"]).sum() / bm["n"].sum()))

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
                bm["model"], bm["num_leaves"], bm["min_child_samples"], bm["learning_rate"], bm["anchor_year"] = "LGB", nl, mcs, lr, anchor_year
                results.append(bm)
                lgb_combos.append((nl, mcs, lr, (bm["R2"] * bm["n"]).sum() / bm["n"].sum()))

    R = pd.concat(results, ignore_index=True)
    rf_combos.sort(key=lambda x: x[2], reverse=True)
    xgb_combos.sort(key=lambda x: x[3], reverse=True)
    lgb_combos.sort(key=lambda x: x[3], reverse=True)
    shortlist = {"RF": rf_combos[:3], "XGB": xgb_combos[:3], "LGB": lgb_combos[:3]}
    return R, shortlist


def main():
    print("="*70)
    print("B-15 TOWER-9 STAGE 1: ROLLOUT-BASED GRID SEARCH + STABILITY CHECK")
    print("="*70)

    search_results, shortlist = run_grid_search_at_anchor(2021)
    search_results.to_csv(f"{RESULTS}/b15_t9_rollout_grid_search.csv", index=False)
    print(f"\n[OK] Saved search results: {len(search_results)} rows")

    print(f"\nRunning stability check at anchor 2019 (full grid, filtered to shortlist)...")
    stability_results, _ = run_grid_search_at_anchor(2019)

    rf_params, xgb_params, lgb_params = shortlist["RF"], shortlist["XGB"], shortlist["LGB"]
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
    stability_filtered.to_csv(f"{RESULTS}/b15_t9_stability_check.csv", index=False)
    print(f"[OK] Saved stability check results: {len(stability_filtered)} rows")

    print(f"\n{'='*70}\nWINNER SELECTION (2021 search + 2019 stability combined rank)\n{'='*70}\n")

    def wavg_r2(mask):
        sub = stability_results[mask]
        return (sub["R2"] * sub["n"]).sum() / sub["n"].sum() if sub["n"].sum() > 0 else np.nan

    winners = {}

    print("RF:")
    rf_scored = []
    for mf, msl, r2_2021 in rf_params:
        mask = (stability_results["model"] == "RF") & (stability_results["max_features"] == mf) & (stability_results["min_samples_leaf"] == msl)
        r2_2019 = wavg_r2(mask)
        avg = np.nanmean([r2_2021, r2_2019])
        rf_scored.append((mf, msl, avg))
        print(f"  max_features={mf} min_samples_leaf={msl}: 2021={r2_2021:.4f} 2019={r2_2019:.4f} avg={avg:.4f}")
    best_rf = max(rf_scored, key=lambda x: x[2])
    winners["RF"] = {"max_features": best_rf[0], "min_samples_leaf": best_rf[1], "chosen_by": "combined_2021_2019_T9"}
    print(f"  WINNER: max_features={best_rf[0]} min_samples_leaf={best_rf[1]}\n")

    print("XGB:")
    xgb_scored = []
    for md, lr, mcw, r2_2021 in xgb_params:
        mask = (stability_results["model"] == "XGB") & (stability_results["max_depth"] == md) & (stability_results["learning_rate"] == lr) & (stability_results["min_child_weight"] == mcw)
        r2_2019 = wavg_r2(mask)
        avg = np.nanmean([r2_2021, r2_2019])
        xgb_scored.append((md, lr, mcw, avg))
        print(f"  max_depth={md} lr={lr} min_child_weight={mcw}: 2021={r2_2021:.4f} 2019={r2_2019:.4f} avg={avg:.4f}")
    best_xgb = max(xgb_scored, key=lambda x: x[3])
    winners["XGB"] = {"max_depth": best_xgb[0], "learning_rate": best_xgb[1], "min_child_weight": best_xgb[2], "chosen_by": "combined_2021_2019_T9"}
    print(f"  WINNER: max_depth={best_xgb[0]} learning_rate={best_xgb[1]} min_child_weight={best_xgb[2]}\n")

    print("LightGBM:")
    lgb_scored = []
    for nl, mcs, lr, r2_2021 in lgb_params:
        mask = (stability_results["model"] == "LGB") & (stability_results["num_leaves"] == nl) & (stability_results["min_child_samples"] == mcs) & (stability_results["learning_rate"] == lr)
        r2_2019 = wavg_r2(mask)
        avg = np.nanmean([r2_2021, r2_2019])
        lgb_scored.append((nl, mcs, lr, avg))
        print(f"  num_leaves={nl} min_child_samples={mcs} lr={lr}: 2021={r2_2021:.4f} 2019={r2_2019:.4f} avg={avg:.4f}")
    best_lgb = max(lgb_scored, key=lambda x: x[3])
    winners["LGB"] = {"num_leaves": best_lgb[0], "min_child_samples": best_lgb[1], "learning_rate": best_lgb[2], "chosen_by": "combined_2021_2019_T9"}
    print(f"  WINNER: num_leaves={best_lgb[0]} min_child_samples={best_lgb[1]} learning_rate={best_lgb[2]}\n")

    winners_df = pd.DataFrame([
        {"model": "RF", **winners["RF"]},
        {"model": "XGB", **winners["XGB"]},
        {"model": "LGB", **winners["LGB"]},
    ])
    winners_df.to_csv(f"{RESULTS}/b15_t9_winners.csv", index=False)
    print(f"[OK] Saved winners: {RESULTS}/b15_t9_winners.csv")
    print(f"\n{'='*70}\nB-15 TOWER-9 STAGE 1 COMPLETE\n{'='*70}")


if __name__ == "__main__":
    main()
