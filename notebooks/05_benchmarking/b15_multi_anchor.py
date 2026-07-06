"""B-15 Stage 2: 5-anchor (2018-2022) validation of B-15's winning hyperparameters.

Reads winners from results/b15_winners.csv (produced by Stage 1), refits all three models
(RF/XGB/LightGBM) + SARIMAX (per-anchor AIC search) + 4-model ensemble (RF+XGB+LightGBM+SARIMAX,
fixing B-14's 3-model-only mistake) across all 5 anchors. Outputs: results/b15_tuned_rollout_summary.csv.
"""

import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = r"c:\Users\Nicholas\Documents\COMP0191 MSc Artificial Intelligence for Sustainable Development Project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\src")

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX

import models.recursive_rollout as rr

HOURLY = rf"{ROOT}\data\Hourly"
RESULTS = rf"{ROOT}\results"

N_DAYS = 365
TOWERS = [2, 4, 9]
ANCHOR_YEARS = [2018, 2019, 2020, 2021, 2022]
TOWER_MAIN = 4

DUM = ["is_t2", "is_t4", "is_t9"]
AR_COLS = ["ar_ch4_dlag1", "ar_ch4_dlag2", "ar_ch4_dlag3", "ar_ch4_dlag7", "ar_ch4_dlag14", "ar_ch4_drm7"]
EXOG_B = ["fx_lsu_dens", "fx_WS_mean", "fx_VPD_mean", "fx_USTAR_mean", "fx_PPFD_mean",
          "fx_DOY_sin", "fx_DOY_cos", "fx_is_growing"]

def main():
    # Load winners from Stage 1
    winners_df = pd.read_csv(f"{RESULTS}/b15_winners.csv")
    winners = {}
    for _, row in winners_df.iterrows():
        model = row["model"]
        params = {k: v for k, v in row.items() if pd.notna(v) and k not in ["model", "chosen_by"]}
        winners[model] = params

    print("="*70)
    print("B-15 STAGE 2: 5-ANCHOR VALIDATION")
    print("="*70)
    print(f"\nWinners loaded:")
    for model, params in winners.items():
        print(f"  {model}: {params}")

    # Load data once
    dv = pd.read_csv(f"{HOURLY}/forecast_daily_v2.csv", low_memory=False)
    dv["Datetime"] = pd.to_datetime(dv["Datetime"], format="mixed")
    FX_B = [c for c in dv.columns if c.startswith("fx")]
    T = {t: dv[dv.tower == t].set_index("Datetime").sort_index() for t in TOWERS}

    feat_cols = AR_COLS + FX_B + ["ar_fc_dlag1"] + DUM

    rows_summary = []
    rows_chains = []

    for yr in ANCHOR_YEARS:
        anchor = pd.Timestamp(f"{yr}-12-16")
        target_dates = pd.date_range(anchor + pd.Timedelta(days=1), periods=N_DAYS, freq="D")

        print(f"\n=== Anchor {yr} ===")
        t_anchor_start = time.time()

        # Pooled training
        pool = []
        for t in TOWERS:
            df = T[t].copy()
            df["target"] = df["y_gapfilled"]
            for d in DUM:
                df[d] = 1.0 if d == f"is_t{t}" else 0.0
            pool.append(df[df.index <= anchor])
        tr = pd.concat(pool)
        tr = tr[tr["target"].notna()]

        # Tower 4
        df4 = T[TOWER_MAIN]
        history_init = df4.loc[:anchor, "y_gapfilled"].copy()
        fx_frame = df4.loc[target_dates, FX_B + ["ar_fc_dlag1"]].copy()
        fx_frame["is_t2"], fx_frame["is_t4"], fx_frame["is_t9"] = 0.0, 1.0, 0.0

        y_true_full = pd.Series(df4.loc[target_dates, "y_observed"].values, index=target_dates)
        anchor_val = df4.loc[anchor, "y_gapfilled"]
        persist = rr.chain_persistence(anchor_val, N_DAYS)

        # Impute
        imp = SimpleImputer(strategy="mean")
        Xi = imp.fit_transform(tr[feat_cols].values)

        # RF
        print(f"  RF (tuned)...", end=" ", flush=True)
        rf_params = {k: (int(v) if k in ["min_samples_leaf"] else v) for k, v in winners["RF"].items()}
        rf = RandomForestRegressor(n_estimators=500, **rf_params, n_jobs=-1, random_state=42)
        rf.fit(Xi, tr["target"].values)
        chain_rf = rr.tree_rollout(rf, imp, feat_cols, fx_frame, history_init, anchor, n_days=N_DAYS)
        y_pred = chain_rf.reindex(target_dates).values
        bm_rf = rr.bin_metrics(y_true_full.values, y_pred, target_dates, anchor, y_persist=persist)
        bm_rf["model"] = "RF_tuned"
        bm_rf["anchor_year"] = yr
        rows_summary.append(bm_rf)
        mean_r2 = (bm_rf["R2"] * bm_rf["n"]).sum() / bm_rf["n"].sum()
        mean_mase = (bm_rf["MASE"] * bm_rf["n"]).sum() / bm_rf["n"].sum()
        print(f"R2={mean_r2:.3f}, MASE={mean_mase:.3f}")

        # XGB
        print(f"  XGB (tuned)...", end=" ", flush=True)
        xgb_params = {k: (int(v) if k in ["max_depth", "min_child_weight"] else v) for k, v in winners["XGB"].items()}
        xgb = XGBRegressor(n_estimators=400, subsample=0.8, colsample_bytree=0.8, **xgb_params, n_jobs=-1, random_state=42)
        xgb.fit(Xi, tr["target"].values)
        chain_xgb = rr.tree_rollout(xgb, imp, feat_cols, fx_frame, history_init, anchor, n_days=N_DAYS)
        y_pred = chain_xgb.reindex(target_dates).values
        bm_xgb = rr.bin_metrics(y_true_full.values, y_pred, target_dates, anchor, y_persist=persist)
        bm_xgb["model"] = "XGB_tuned"
        bm_xgb["anchor_year"] = yr
        rows_summary.append(bm_xgb)
        mean_r2 = (bm_xgb["R2"] * bm_xgb["n"]).sum() / bm_xgb["n"].sum()
        mean_mase = (bm_xgb["MASE"] * bm_xgb["n"]).sum() / bm_xgb["n"].sum()
        print(f"R2={mean_r2:.3f}, MASE={mean_mase:.3f}")

        # LightGBM
        print(f"  LightGBM (tuned)...", end=" ", flush=True)
        lgb_params = {k: (int(v) if k in ["num_leaves", "min_child_samples"] else v) for k, v in winners["LGB"].items()}
        lgb = LGBMRegressor(n_estimators=400, subsample=0.8, colsample_bytree=0.8, **lgb_params, n_jobs=-1, random_state=42, verbosity=-1)
        lgb.fit(Xi, tr["target"].values)
        chain_lgb = rr.tree_rollout(lgb, imp, feat_cols, fx_frame, history_init, anchor, n_days=N_DAYS)
        y_pred = chain_lgb.reindex(target_dates).values
        bm_lgb = rr.bin_metrics(y_true_full.values, y_pred, target_dates, anchor, y_persist=persist)
        bm_lgb["model"] = "LightGBM_tuned"
        bm_lgb["anchor_year"] = yr
        rows_summary.append(bm_lgb)
        mean_r2 = (bm_lgb["R2"] * bm_lgb["n"]).sum() / bm_lgb["n"].sum()
        mean_mase = (bm_lgb["MASE"] * bm_lgb["n"]).sum() / bm_lgb["n"].sum()
        print(f"R2={mean_r2:.3f}, MASE={mean_mase:.3f}")

        # SARIMAX
        print(f"  SARIMAX...", end=" ", flush=True)
        y = df4["y_gapfilled"].astype(float)
        X = df4[EXOG_B].astype(float).ffill().bfill()
        y_tr, X_tr = y.loc[:anchor], X.loc[:anchor]

        best = None
        for p in [1, 2, 3]:
            for q in [0, 1, 2]:
                try:
                    m = SARIMAX(y_tr, exog=X_tr, order=(p, 1, q), enforce_stationarity=False, enforce_invertibility=False)
                    res = m.fit(disp=False, maxiter=50)
                    if best is None or res.aic < best[0]:
                        best = (res.aic, (p, 1, q), res)
                except:
                    pass

        sarimax_order, sarimax_res = best[1], best[2]
        future_X = X.loc[target_dates]
        fc = sarimax_res.get_forecast(steps=N_DAYS, exog=future_X)
        chain_sarimax = pd.Series(fc.predicted_mean.values, index=target_dates)
        y_pred = chain_sarimax.reindex(target_dates).values
        bm_sarimax = rr.bin_metrics(y_true_full.values, y_pred, target_dates, anchor, y_persist=persist)
        bm_sarimax["model"] = "SARIMAX"
        bm_sarimax["anchor_year"] = yr
        rows_summary.append(bm_sarimax)
        mean_r2 = (bm_sarimax["R2"] * bm_sarimax["n"]).sum() / bm_sarimax["n"].sum()
        mean_mase = (bm_sarimax["MASE"] * bm_sarimax["n"]).sum() / bm_sarimax["n"].sum()
        print(f"order={sarimax_order}, R2={mean_r2:.3f}, MASE={mean_mase:.3f}")

        # 4-model ensemble (RF+XGB+LightGBM+SARIMAX)
        print(f"  Ensemble (4-model)...", end=" ", flush=True)
        ens_df = pd.DataFrame({"RF": chain_rf, "XGB": chain_xgb, "LightGBM": chain_lgb, "SARIMAX": chain_sarimax})
        ens_unweighted = ens_df.mean(axis=1)
        y_pred = ens_unweighted.reindex(target_dates).values
        bm_ens = rr.bin_metrics(y_true_full.values, y_pred, target_dates, anchor, y_persist=persist)
        bm_ens["model"] = "Ensemble_4model_tuned"
        bm_ens["anchor_year"] = yr
        rows_summary.append(bm_ens)
        mean_r2 = (bm_ens["R2"] * bm_ens["n"]).sum() / bm_ens["n"].sum()
        mean_mase = (bm_ens["MASE"] * bm_ens["n"]).sum() / bm_ens["n"].sum()
        print(f"R2={mean_r2:.3f}, MASE={mean_mase:.3f}")

        # Save per-day chains for plotting (results/figures/b15_chains/)
        chain_frame = pd.DataFrame({
            "RF_tuned": chain_rf.reindex(target_dates),
            "XGB_tuned": chain_xgb.reindex(target_dates),
            "LightGBM_tuned": chain_lgb.reindex(target_dates),
            "SARIMAX": chain_sarimax.reindex(target_dates),
            "Ensemble_4model_tuned": ens_unweighted.reindex(target_dates),
        })
        chain_frame = chain_frame.reset_index().rename(columns={"index": "Datetime"})
        chain_frame["anchor_year"] = yr
        rows_chains.append(chain_frame)

        print(f"  Anchor {yr} done ({time.time()-t_anchor_start:.0f}s)")

    # Aggregate results
    R = pd.concat(rows_summary, ignore_index=True)
    R.to_csv(f"{RESULTS}/b15_tuned_rollout_summary.csv", index=False)

    C = pd.concat(rows_chains, ignore_index=True)
    C.to_csv(f"{RESULTS}/b15_chains.csv", index=False)
    print(f"\n[OK] Saved chains: {RESULTS}/b15_chains.csv ({len(C)} rows)")

    def wavg(g, col):
        w = g["n"]
        return (g[col] * w).sum() / w.sum() if w.sum() > 0 else np.nan

    print("\n" + "="*70)
    print("=== FINAL RESULTS: B-15 Tuned Models (5-anchor mean) ===")
    print("="*70)

    agg = R.groupby("model").apply(
        lambda g: pd.Series({"R2_mean": wavg(g, "R2"), "MASE_mean": wavg(g, "MASE"), "n_anchors": g["anchor_year"].nunique()}),
        include_groups=False
    ).reset_index()
    agg = agg.sort_values("R2_mean", ascending=False)

    print("\nPer-model aggregate (5-anchor n-weighted mean):")
    print(agg.round(4).to_string(index=False))

    print("\n" + "="*70)
    print("Results saved: b15_tuned_rollout_summary.csv")
    print("="*70)

if __name__ == "__main__":
    main()
