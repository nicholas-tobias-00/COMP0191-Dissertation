"""U-03 Part B: synthetic extrapolation diagnostic -- what happens to each model's rollout
prediction when the headline scenario knob (fx_lsu_dens, livestock density) is pushed beyond its
training-seen range?

NOT a coverage validation -- there is no real y_observed for a hypothetical 2x/3x-livestock world,
so nothing here is scored against ground truth. This is a sensitivity/diagnostic sweep only:
does each model's point prediction keep responding linearly-ish, or does it plateau (the
tree-extrapolation-ceiling signature -- RF/XGB/LightGBM split on leaf boundaries and cannot
extrapolate past the range of values seen in training)?

Design (matches the approved plan):
- Tower 4, anchor 2021-12-16 (this project's standing single-anchor smoke-test default).
- Refit RF/XGB(x3 quantile)/LightGBM(x3 quantile)/SARIMAX/TFTQuantile EXACTLY as
  u02_multi_anchor_tower.py's fit_stage does for this anchor (same pooled T2+T4+T9 tree training,
  same per-tower SARIMAX/TFT fit) -- only the ROLLOUT-TIME fx_lsu_dens input is perturbed; the
  fitted models themselves are never retrained on perturbed data.
- Multiplier sweep on fx_lsu_dens: {1.0 (real), 1.5, 2.0, 2.5, 3.0}, applied only to the post-anchor
  (rollout-window) days -- pre-anchor training history is always real/unperturbed.
- Calibration margins are READ from the existing results/u02_summary.csv (frozen, real-data-
  derived for anchor=2021/tower=4) and applied mechanically (median +/- margin) -- explicitly NOT
  a validated interval once the input is no longer exchangeable with what the margin was
  calibrated on.
- Explicitly excluded: TabPFN, both ensembles (not the flattening question being asked here).
"""
import sys
import time
import warnings

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
import models.forecasting_dl as fdl

HOURLY = rf"{ROOT}\data\Hourly"
RESULTS = rf"{ROOT}\results"

TOWER = 4
ANCHOR = pd.Timestamp("2021-12-16")
N_DAYS = 365
VAL_DAYS = 90
QUANTILES = (0.05, 0.5, 0.95)
MULTIPLIERS = [1.0, 1.5, 2.0, 2.5, 3.0]
BINS = ((1, 7), (8, 30), (31, 90), (91, 180), (181, 270), (271, 365))

DUM = ["is_t2", "is_t4", "is_t9"]
AR_COLS = ["ar_ch4_dlag1", "ar_ch4_dlag2", "ar_ch4_dlag3", "ar_ch4_dlag7", "ar_ch4_dlag14", "ar_ch4_drm7"]
EXOG_B = ["fx_lsu_dens", "fx_WS_mean", "fx_VPD_mean", "fx_USTAR_mean", "fx_PPFD_mean",
          "fx_DOY_sin", "fx_DOY_cos", "fx_is_growing"]
TOWERS = [2, 4, 9]


def main():
    dv = pd.read_csv(f"{HOURLY}/forecast_daily_v2.csv", low_memory=False)
    dv["Datetime"] = pd.to_datetime(dv["Datetime"], format="mixed")
    FX_B = [c for c in dv.columns if c.startswith("fx")]
    T = {t: dv[dv.tower == t].set_index("Datetime").sort_index() for t in TOWERS}
    feat_cols = AR_COLS + FX_B + ["ar_fc_dlag1"] + DUM

    target_dates = pd.date_range(ANCHOR + pd.Timedelta(days=1), periods=N_DAYS, freq="D")
    hist_max_lsu = float(T[TOWER].loc[:ANCHOR, "fx_lsu_dens"].max())
    print(f"Tower {TOWER} pre-anchor (<= {ANCHOR.date()}) training max fx_lsu_dens = {hist_max_lsu:.3f}")

    # ---- Fit stage: pooled T2+T4+T9 trees (identical to u02_multi_anchor_tower.py's fit_stage) ----
    t0 = time.time()
    pool = []
    for t in TOWERS:
        df = T[t].copy()
        df["target"] = df["y_gapfilled"]
        for d in DUM:
            df[d] = 1.0 if d == f"is_t{t}" else 0.0
        pool.append(df[df.index <= ANCHOR])
    tr = pd.concat(pool)
    tr = tr[tr["target"].notna()]

    imp_ = SimpleImputer(strategy="mean")
    Xi = imp_.fit_transform(tr[feat_cols].values)

    rf = RandomForestRegressor(n_estimators=500, max_features=0.5, min_samples_leaf=10, n_jobs=-1, random_state=42)
    rf.fit(Xi, tr["target"].values)
    rf_adapter = rr.RFQuantileAdapter(rf)

    xgb_models = {}
    for q in QUANTILES:
        m = XGBRegressor(n_estimators=400, max_depth=2, learning_rate=0.02, min_child_weight=10,
                          subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42,
                          objective="reg:quantileerror", quantile_alpha=q)
        m.fit(Xi, tr["target"].values)
        xgb_models[q] = m
    xgb_adapter = rr.MultiModelQuantileAdapter(xgb_models)

    lgb_models = {}
    for q in QUANTILES:
        m = LGBMRegressor(n_estimators=400, num_leaves=7, min_child_samples=10, learning_rate=0.02,
                           subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42, verbosity=-1,
                           objective="quantile", alpha=q)
        m.fit(Xi, tr["target"].values)
        lgb_models[q] = m
    lgb_adapter = rr.MultiModelQuantileAdapter(lgb_models)
    print(f"Pooled RF/XGB(x3)/LightGBM(x3) fit ({time.time()-t0:.0f}s)")

    dft = T[TOWER]
    history_init = dft.loc[:ANCHOR, "y_gapfilled"].copy()

    # ---- SARIMAX fit (real, unperturbed exog -- identical AIC order search) ----
    y = dft["y_gapfilled"].astype(float)
    X_full = dft[EXOG_B].astype(float).ffill().bfill()
    y_tr, X_tr = y.loc[:ANCHOR], X_full.loc[:ANCHOR]
    best = None
    for p in [1, 2, 3]:
        for q in [0, 1, 2]:
            try:
                m = SARIMAX(y_tr, exog=X_tr, order=(p, 1, q), enforce_stationarity=False, enforce_invertibility=False)
                res = m.fit(disp=False, maxiter=50)
                if best is None or res.aic < best[0]:
                    best = (res.aic, (p, 1, q), res)
            except Exception:
                pass
    sarimax_res = best[2]
    print(f"SARIMAX fit, order={best[1]}, AIC={best[0]:.1f}")

    # ---- TFTQuantile fit (identical to u02_multi_anchor_tower.py's fit_stage TFT block) ----
    fdl_m = fdl.load_matrix(f"{HOURLY}/forecast_features_v2.csv")
    device = fdl.get_device()
    tft_cfg = fdl.TRACKS["B"]
    cutoff = ANCHOR + pd.Timedelta(hours=23, minutes=59)
    val_start = ANCHOR - pd.Timedelta(days=VAL_DAYS)
    Wd = fdl.build_windows(fdl_m, "B")
    train_parts, val_parts = [], []
    for t in TOWERS:
        ttime = pd.DatetimeIndex(Wd[t]["ttime"][:, -1])
        train_parts.append(fdl._subset(Wd[t], ttime <= val_start))
        val_parts.append(fdl._subset(Wd[t], (ttime > val_start) & (ttime <= cutoff)))
    train_tft = fdl._cat(train_parts); val_tft = fdl._cat(val_parts)
    se_t, sd_t = fdl.Scaler().fit(train_tft["enc"]), fdl.Scaler().fit(train_tft["dec"])
    yv = train_tft["y"][np.isfinite(train_tft["y"])]
    mu_t, sdy_t = float(yv.mean()), float(yv.std() + 1e-6)
    train_tft["enc"], train_tft["dec"] = se_t.tf(train_tft["enc"]), sd_t.tf(train_tft["dec"])
    val_tft["enc"], val_tft["dec"] = se_t.tf(val_tft["enc"]), sd_t.tf(val_tft["dec"])
    n_enc, n_dec = train_tft["enc"].shape[-1], train_tft["dec"].shape[-1]

    tft_model = fdl.TFTQuantile(tft_cfg["L"], tft_cfg["H"], n_enc, n_dec, 3, nq=len(QUANTILES))
    fdl.train_quantile(tft_model, train_tft, device, quantiles=list(QUANTILES), epochs=30,
                        ch4_mu=mu_t, ch4_sd=sdy_t, seed=0,
                        weight_decay=1e-3, val_data=val_tft, patience=5)
    print("TFTQuantile fit")

    ser_t = fdl.tower_series(fdl_m, TOWER, "B")
    lsu_idx = fdl.FX.index("fx_lsu_dens")
    dates_full_tft = ser_t["idx"]
    anchor_idx_tft = dates_full_tft.get_loc(ANCHOR)
    history_init_tft = ser_t["ch4"][:anchor_idx_tft + 1]

    # ---- Load frozen, real-data-derived calibration margins for this exact anchor/tower ----
    u02_summary = pd.read_csv(f"{RESULTS}/u02_summary.csv")
    margins_row = u02_summary[(u02_summary.anchor_year == 2021) & (u02_summary.eval_tower == TOWER)]

    def get_margin(model_name, bin_label):
        r = margins_row[(margins_row.model == model_name) & (margins_row.bin == bin_label)]
        return float(r["conformal_margin"].iloc[0]) if len(r) and pd.notna(r["conformal_margin"].iloc[0]) else np.nan

    # ---- Multiplier sweep ----
    rows = []
    for mult in MULTIPLIERS:
        print(f"\n--- multiplier={mult} ---")

        fx_frame = dft.loc[target_dates, FX_B + ["ar_fc_dlag1"]].copy()
        fx_frame["fx_lsu_dens"] = fx_frame["fx_lsu_dens"] * mult
        fx_frame["is_t2"] = 0.0
        fx_frame["is_t4"] = 1.0
        fx_frame["is_t9"] = 0.0
        window_max_lsu = float(fx_frame["fx_lsu_dens"].max())

        df_rf = rr.tree_rollout_quantile(rf_adapter, imp_, feat_cols, fx_frame, history_init, ANCHOR, N_DAYS, QUANTILES)
        df_xgb = rr.tree_rollout_quantile(xgb_adapter, imp_, feat_cols, fx_frame, history_init, ANCHOR, N_DAYS, QUANTILES)
        df_lgb = rr.tree_rollout_quantile(lgb_adapter, imp_, feat_cols, fx_frame, history_init, ANCHOR, N_DAYS, QUANTILES)

        X_pert = X_full.copy()
        X_pert.loc[target_dates, "fx_lsu_dens"] = X_full.loc[target_dates, "fx_lsu_dens"] * mult
        df_sarimax = rr.sarimax_quantile(sarimax_res, target_dates, X_pert.loc[target_dates], alpha=0.10)

        enc_ex_pert = ser_t["enc_ex"].copy()
        dec_ex_pert = ser_t["dec_ex"].copy()
        post_anchor_mask = dates_full_tft > ANCHOR
        enc_ex_pert[post_anchor_mask, lsu_idx] *= mult
        dec_ex_pert[post_anchor_mask, lsu_idx] *= mult
        df_tft = rr.dl_rollout_quantile(tft_model, se_t, sd_t, mu_t, sdy_t, device, fdl.TOW[TOWER],
                                         enc_ex_pert, dec_ex_pert, dates_full_tft, history_init_tft, ANCHOR,
                                         quantiles=QUANTILES, L=tft_cfg["L"], H=tft_cfg["H"], n_days=N_DAYS)

        model_dfs = {
            "RF": df_rf.rename(columns={0.05: "q05", 0.95: "q95"}),
            "XGB": df_xgb.rename(columns={0.05: "q05", 0.95: "q95"}),
            "LightGBM": df_lgb.rename(columns={0.05: "q05", 0.95: "q95"}),
            "SARIMAX": df_sarimax.rename(columns={0.05: "q05", 0.95: "q95"}),
            "TFT": df_tft.rename(columns={0.05: "q05", 0.95: "q95"}),
        }

        for lo, hi in BINS:
            label = f"{lo}-{hi}"
            bin_dates = target_dates[(np.arange(1, N_DAYS + 1) >= lo) & (np.arange(1, N_DAYS + 1) <= hi)]
            for model_name, mdf in model_dfs.items():
                sub = mdf.loc[bin_dates]
                margin = get_margin(model_name, label)
                raw_width = float((sub["q95"] - sub["q05"]).mean())
                cal_width = 2 * margin if np.isfinite(margin) else np.nan
                rows.append({
                    "multiplier": mult, "model": model_name, "bin": label,
                    "mean_median_pred": float(sub["median"].mean()),
                    "mean_raw_width": raw_width,
                    "nominal_calibrated_width": cal_width,
                    "window_max_lsu_dens": window_max_lsu,
                    "training_max_lsu_dens": hist_max_lsu,
                })

        print(f"  window max fx_lsu_dens = {window_max_lsu:.2f} "
              f"(training max = {hist_max_lsu:.2f}, {'WITHIN' if window_max_lsu <= hist_max_lsu else 'BEYOND'} range)")
        for model_name, mdf in model_dfs.items():
            print(f"  {model_name:10s} mean median (full window) = {mdf['median'].mean():.2f}")

    out = pd.DataFrame(rows)
    out.to_csv(f"{RESULTS}/u03_extrapolation_stress_test.csv", index=False)
    print(f"\n[OK] Saved u03_extrapolation_stress_test.csv ({len(out)} rows)")
    print("\nNOTE: nominal_calibrated_width is a MECHANICAL application of a real-data-derived "
          "margin (from u02_summary.csv, anchor=2021) to a synthetically-perturbed, non-"
          "exchangeable scenario input -- NOT a validated coverage guarantee. See U03_results.md.")


if __name__ == "__main__":
    main()
