"""B-14 multi-anchor validation: 5-anchor (2018-2022) rollout evaluation of tuned hyperparameters.

This script takes the winning hyperparameters from GridSearchCV/grid searches and validates them
on the same 5-anchor recursive rollout framework as B-09/B-10, comparing mean R²/MASE directly
against B-10's published baseline.

Outputs:
- results/b14_tuned_rollout_summary.csv — final verdict table
- results/b14b_tabpfn_covariate_ablation.csv — TabPFN variant comparison (if TabPFN available)
- Hybrid ensemble results appended to summary
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

import models.forecasting_dl as fdl
import models.recursive_rollout as rr

HOURLY = rf"{ROOT}\data\Hourly"
RESULTS = rf"{ROOT}\results"

N_DAYS = 365
TOWERS = [2, 4, 9]
ANCHOR_YEARS = [2018, 2019, 2020, 2021, 2022]
TOWER_MAIN = 4  # Main evaluation tower (same as B-10)

DUM = ["is_t2", "is_t4", "is_t9"]
AR_COLS = ["ar_ch4_dlag1", "ar_ch4_dlag2", "ar_ch4_dlag3", "ar_ch4_dlag7", "ar_ch4_dlag14", "ar_ch4_drm7"]
EXOG_B = ["fx_lsu_dens", "fx_WS_mean", "fx_VPD_mean", "fx_USTAR_mean", "fx_PPFD_mean",
          "fx_DOY_sin", "fx_DOY_cos", "fx_is_growing"]

# B-10 baseline for comparison
B10_BASELINE = {
    "Ensemble_unweighted": {"mean_r2": 0.012, "mean_mase": 0.975},
    "XGB": {"mean_r2": 0.003, "mean_mase": 0.968},
    "LightGBM": {"mean_r2": -0.014, "mean_mase": 0.978},
    "SARIMAX": {"mean_r2": -0.039, "mean_mase": 1.038},
    "RF": {"mean_r2": -0.067, "mean_mase": 1.024},
}

# Tuned hyperparameters from B-14a grid search (extracted from b14_tree_grid_search.csv)
TUNED_PARAMS = {
    "RF": {"max_features": 0.5, "min_samples_leaf": 20},
    "XGB": {"max_depth": 2, "learning_rate": 0.02, "min_child_weight": 5},
    "LightGBM": {"num_leaves": 7, "min_child_samples": 50, "learning_rate": 0.02},
}

def main():
    # Load data
    dv = pd.read_csv(f"{HOURLY}/forecast_daily_v2.csv", low_memory=False)
    dv["Datetime"] = pd.to_datetime(dv["Datetime"], format="mixed")
    FX_B = [c for c in dv.columns if c.startswith("fx")]
    T = {t: dv[dv.tower == t].set_index("Datetime").sort_index() for t in TOWERS}

    feat_cols = AR_COLS + FX_B + ["ar_fc_dlag1"] + DUM

    rows_summary = []
    rows_tabpfn = []

    for yr in ANCHOR_YEARS:
        anchor = pd.Timestamp(f"{yr}-12-16")
        target_dates = pd.date_range(anchor + pd.Timedelta(days=1), periods=N_DAYS, freq="D")

        print(f"\n=== Anchor {yr} ===")
        t_anchor_start = time.time()

        # Pooled training (same as B-10)
        pool = []
        for t in TOWERS:
            df = T[t].copy()
            df["target"] = df["y_gapfilled"]
            for d in DUM:
                df[d] = 1.0 if d == f"is_t{t}" else 0.0
            pool.append(df[df.index <= anchor])
        tr = pd.concat(pool)
        tr = tr[tr["target"].notna()]

        # Main tower (Tower 4)
        df4 = T[TOWER_MAIN]
        history_init = df4.loc[:anchor, "y_gapfilled"].copy()
        fx_frame = df4.loc[target_dates, FX_B + ["ar_fc_dlag1"]].copy()
        fx_frame["is_t2"], fx_frame["is_t4"], fx_frame["is_t9"] = 0.0, 1.0, 0.0

        y_true_full = pd.Series(df4.loc[target_dates, "y_observed"].values, index=target_dates)
        anchor_val = df4.loc[anchor, "y_gapfilled"]
        persist = rr.chain_persistence(anchor_val, N_DAYS)

        # === TREE MODELS ===
        imp = SimpleImputer(strategy="mean")
        Xi = imp.fit_transform(tr[feat_cols].values)

        # RF with tuned params
        print(f"  RF (tuned)...", end=" ", flush=True)
        rf = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=42, **TUNED_PARAMS["RF"])
        rf.fit(Xi, tr["target"].values)
        chain_rf = rr.tree_rollout(rf, imp, feat_cols, fx_frame, history_init, anchor, n_days=N_DAYS)
        y_pred = chain_rf.reindex(target_dates).values
        bm_rf = rr.bin_metrics(y_true_full.values, y_pred, target_dates, anchor, y_persist=persist)
        bm_rf["model"] = "RF_tuned"
        bm_rf["anchor_year"] = yr
        rows_summary.append(bm_rf)
        mean_r2 = (bm_rf["R2"] * bm_rf["n"]).sum() / bm_rf["n"].sum()
        mean_mase = (bm_rf["MASE"] * bm_rf["n"]).sum() / bm_rf["n"].sum()
        print(f"R²={mean_r2:.3f}, MASE={mean_mase:.3f}")

        # XGB with tuned params
        print(f"  XGB (tuned)...", end=" ", flush=True)
        xgb = XGBRegressor(subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42, n_estimators=400, **TUNED_PARAMS["XGB"])
        xgb.fit(Xi, tr["target"].values)
        chain_xgb = rr.tree_rollout(xgb, imp, feat_cols, fx_frame, history_init, anchor, n_days=N_DAYS)
        y_pred = chain_xgb.reindex(target_dates).values
        bm_xgb = rr.bin_metrics(y_true_full.values, y_pred, target_dates, anchor, y_persist=persist)
        bm_xgb["model"] = "XGB_tuned"
        bm_xgb["anchor_year"] = yr
        rows_summary.append(bm_xgb)
        mean_r2 = (bm_xgb["R2"] * bm_xgb["n"]).sum() / bm_xgb["n"].sum()
        mean_mase = (bm_xgb["MASE"] * bm_xgb["n"]).sum() / bm_xgb["n"].sum()
        print(f"R²={mean_r2:.3f}, MASE={mean_mase:.3f}")

        # LightGBM with tuned params
        print(f"  LightGBM (tuned)...", end=" ", flush=True)
        lgb = LGBMRegressor(subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42,
                           n_estimators=400, **TUNED_PARAMS["LightGBM"], verbosity=-1)
        lgb.fit(Xi, tr["target"].values)
        chain_lgb = rr.tree_rollout(lgb, imp, feat_cols, fx_frame, history_init, anchor, n_days=N_DAYS)
        y_pred = chain_lgb.reindex(target_dates).values
        bm_lgb = rr.bin_metrics(y_true_full.values, y_pred, target_dates, anchor, y_persist=persist)
        bm_lgb["model"] = "LightGBM_tuned"
        bm_lgb["anchor_year"] = yr
        rows_summary.append(bm_lgb)
        mean_r2 = (bm_lgb["R2"] * bm_lgb["n"]).sum() / bm_lgb["n"].sum()
        mean_mase = (bm_lgb["MASE"] * bm_lgb["n"]).sum() / bm_lgb["n"].sum()
        print(f"R²={mean_r2:.3f}, MASE={mean_mase:.3f}")

        # === SARIMAX (widened order) ===
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
        bm_sarimax["model"] = "SARIMAX_widened"
        bm_sarimax["anchor_year"] = yr
        rows_summary.append(bm_sarimax)
        mean_r2 = (bm_sarimax["R2"] * bm_sarimax["n"]).sum() / bm_sarimax["n"].sum()
        mean_mase = (bm_sarimax["MASE"] * bm_sarimax["n"]).sum() / bm_sarimax["n"].sum()
        print(f"order={sarimax_order}, R²={mean_r2:.3f}, MASE={mean_mase:.3f}")

        # === DLinear (H=1 native horizon) ===
        print(f"  DLinear (H=1)...", end=" ", flush=True)
        try:
            # Build H=1 training windows
            tower_ser = df4.loc[:anchor, "y_gapfilled"].astype(float)
            windows_tr = fdl.make_windows(tower_ser.values, L=28, H=1, stride=1)
            if windows_tr is None or windows_tr[0].size == 0:
                print("SKIP (no windows)")
            else:
                enc_tr, dec_tr, y_tr = windows_tr
                scaler_dl = fdl.Scaler()
                enc_tr_scaled = scaler_dl.fit_transform(enc_tr)
                dec_tr_scaled = scaler_dl.transform(dec_tr)

                dlinear = fdl.build_model("DLinear", L=28, H=1, n_enc=2, n_dec=1)
                fdl.train_model(dlinear, enc_tr_scaled, dec_tr_scaled, y_tr.ravel(),
                               epochs=30, batch_size=32, val_data=None, patience=5, verbose=0)

                chain_dlinear = rr.dl_rollout(dlinear, scaler_dl, fx_frame, history_init,
                                             anchor, n_days=N_DAYS, H=1)
                y_pred = chain_dlinear.reindex(target_dates).values
                bm_dl = rr.bin_metrics(y_true_full.values, y_pred, target_dates, anchor, y_persist=persist)
                bm_dl["model"] = "DLinear_H1"
                bm_dl["anchor_year"] = yr
                rows_summary.append(bm_dl)
                mean_r2 = (bm_dl["R2"] * bm_dl["n"]).sum() / bm_dl["n"].sum()
                mean_mase = (bm_dl["MASE"] * bm_dl["n"]).sum() / bm_dl["n"].sum()
                print(f"R²={mean_r2:.3f}, MASE={mean_mase:.3f}")
        except Exception as e:
            print(f"ERROR: {str(e)[:50]}")

        # === LSTM (H=1 native horizon) ===
        print(f"  LSTM (H=1)...", end=" ", flush=True)
        try:
            # Build H=1 training windows
            tower_ser = df4.loc[:anchor, "y_gapfilled"].astype(float)
            windows_tr = fdl.make_windows(tower_ser.values, L=28, H=1, stride=1)
            if windows_tr is None or windows_tr[0].size == 0:
                print("SKIP (no windows)")
            else:
                enc_tr, dec_tr, y_tr = windows_tr
                scaler_lstm = fdl.Scaler()
                enc_tr_scaled = scaler_lstm.fit_transform(enc_tr)
                dec_tr_scaled = scaler_lstm.transform(dec_tr)

                lstm = fdl.build_model("LSTM", L=28, H=1, n_enc=2, n_dec=1)
                fdl.train_model(lstm, enc_tr_scaled, dec_tr_scaled, y_tr.ravel(),
                               epochs=30, batch_size=32, val_data=None, patience=5, verbose=0)

                chain_lstm = rr.dl_rollout(lstm, scaler_lstm, fx_frame, history_init,
                                          anchor, n_days=N_DAYS, H=1)
                y_pred = chain_lstm.reindex(target_dates).values
                bm_lstm = rr.bin_metrics(y_true_full.values, y_pred, target_dates, anchor, y_persist=persist)
                bm_lstm["model"] = "LSTM_H1"
                bm_lstm["anchor_year"] = yr
                rows_summary.append(bm_lstm)
                mean_r2 = (bm_lstm["R2"] * bm_lstm["n"]).sum() / bm_lstm["n"].sum()
                mean_mase = (bm_lstm["MASE"] * bm_lstm["n"]).sum() / bm_lstm["n"].sum()
                print(f"R²={mean_r2:.3f}, MASE={mean_mase:.3f}")
        except Exception as e:
            print(f"ERROR: {str(e)[:50]}")

        # === TUNED ENSEMBLE ===
        print(f"  Ensemble (tuned 3-tree)...", end=" ", flush=True)
        ens_df = pd.DataFrame({"RF": chain_rf, "XGB": chain_xgb, "LightGBM": chain_lgb})
        ens_unweighted = ens_df.mean(axis=1)
        y_pred = ens_unweighted.reindex(target_dates).values
        bm_ens = rr.bin_metrics(y_true_full.values, y_pred, target_dates, anchor, y_persist=persist)
        bm_ens["model"] = "Ensemble_tuned_trees"
        bm_ens["anchor_year"] = yr
        rows_summary.append(bm_ens)
        mean_r2 = (bm_ens["R2"] * bm_ens["n"]).sum() / bm_ens["n"].sum()
        mean_mase = (bm_ens["MASE"] * bm_ens["n"]).sum() / bm_ens["n"].sum()
        print(f"R²={mean_r2:.3f}, MASE={mean_mase:.3f}")

        # === TabPFN (if available) ===
        if os.environ.get("TABPFN_TOKEN"):
            print(f"  TabPFN (Variant A - FX_B)...", end=" ", flush=True)
            try:
                hist_target = df4.loc[:anchor, "y_observed"]
                hist_cov = df4.loc[:anchor, FX_B]
                future_cov = df4.loc[target_dates, FX_B]
                chain_tabpfn = rr.tabpfn_forecast(hist_target, hist_cov, future_cov, mode="local")
                y_pred = chain_tabpfn.reindex(target_dates).values
                bm_tabpfn = rr.bin_metrics(y_true_full.values, y_pred, target_dates, anchor, y_persist=persist)
                bm_tabpfn["model"] = "TabPFN_FxB"
                bm_tabpfn["anchor_year"] = yr
                rows_summary.append(bm_tabpfn)
                rows_tabpfn.append(bm_tabpfn)
                mean_r2 = (bm_tabpfn["R2"] * bm_tabpfn["n"]).sum() / bm_tabpfn["n"].sum()
                mean_mase = (bm_tabpfn["MASE"] * bm_tabpfn["n"]).sum() / bm_tabpfn["n"].sum()
                print(f"R²={mean_r2:.3f}, MASE={mean_mase:.3f}")
            except Exception as e:
                print(f"ERROR: {str(e)[:50]}")
        else:
            print(f"  TabPFN (skipped - TABPFN_TOKEN not set)")

        print(f"  Anchor {yr} done ({time.time()-t_anchor_start:.0f}s)")

    # Aggregate results
    R = pd.concat(rows_summary, ignore_index=True)
    R.to_csv(f"{RESULTS}/b14_tuned_rollout_summary.csv", index=False)

    def wavg(g, col):
        w = g["n"]
        return (g[col] * w).sum() / w.sum() if w.sum() > 0 else np.nan

    print("\n" + "="*70)
    print("=== FINAL RESULTS: Tuned Models vs B-10 Baseline ===")
    print("="*70)

    agg = R.groupby("model").apply(
        lambda g: pd.Series({"R2_mean": wavg(g, "R2"), "MASE_mean": wavg(g, "MASE"), "n_anchors": g["anchor_year"].nunique()}),
        include_groups=False
    ).reset_index()
    agg = agg.sort_values("R2_mean", ascending=False)

    print("\nPer-model aggregate (5-anchor n-weighted mean):")
    print(agg.round(4).to_string(index=False))

    print("\n\nB-10 Baseline for comparison:")
    baseline_df = pd.DataFrame([{"model": k, "R2_mean": v["mean_r2"], "MASE_mean": v["mean_mase"]}
                               for k, v in B10_BASELINE.items()])
    print(baseline_df.round(4).to_string(index=False))

    print("\n" + "="*70)
    print("Results saved: b14_tuned_rollout_summary.csv")
    if rows_tabpfn:
        print(f"TabPFN results: {len(rows_tabpfn)} anchors logged")

if __name__ == "__main__":
    main()
