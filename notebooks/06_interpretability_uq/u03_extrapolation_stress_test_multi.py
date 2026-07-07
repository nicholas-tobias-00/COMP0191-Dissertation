"""U-03 Part B, full coverage: the synthetic fx_lsu_dens extrapolation diagnostic, generalized to
ALL 8 U-02/B-10/B-13 models x ALL 3 towers (T2 included) x the full 5-anchor (2018-2022) sweep --
matching U-02's own scope exactly (rather than the narrower 5-model/2-tower/1-anchor pilot this
project started with, which the user correctly flagged as an incomplete read).

Tower 2 is included per explicit user instruction ("always include tower 2") despite its known
severe data scarcity -- run it anyway, report honestly. Its calibration margins will be NaN
throughout (leave-one-anchor-out calibration needs real residuals from other anchors, and T2 has
real y_observed in only 1/5 anchor windows, exactly as documented in U02_results.md/D-62) -- this
means "nominal_calibrated_width" is NaN for every T2 row, but the raw extrapolation-response
diagnostic (does the point prediction plateau or not) is unaffected by that and computed normally.

Models are fit ONCE per anchor where pooling applies (RF/XGB/LightGBM/TFT: pooled T2+T4+T9,
exactly matching u02_multi_anchor_tower.py's fit_stage), never retrained on perturbed data -- only
the ROLLOUT-TIME fx_lsu_dens input is perturbed, per tower, per multiplier. SARIMAX and TabPFN are
fit/called per tower/anchor (never pooled, matching every other B-09-U-02 call for these two
models). Ensembles are constructed post-hoc from the already-perturbed RF/XGB/LightGBM/SARIMAX
outputs at each multiplier -- no extra model fitting, exactly matching U-02's own ensemble
definition. Frozen calibration margins are read from the existing results/u02_summary.csv for the
matching (anchor, tower, model, bin) -- never recomputed.
"""
import os
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

from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))

HOURLY = rf"{ROOT}\data\Hourly"
RESULTS = rf"{ROOT}\results"

TOWERS = [2, 4, 9]           # pooled training set -- all 3 always contribute to the tree/TFT fit
TOWERS_B = [2, 4, 9]         # rollout/evaluation towers -- ALL 3, per explicit user instruction
ANCHOR_YEARS = [2018, 2019, 2020, 2021, 2022]
N_DAYS = 365
VAL_DAYS = 90
QUANTILES = (0.05, 0.5, 0.95)
MULTIPLIERS = [1.0, 1.5, 2.0, 2.5, 3.0]
BINS = ((1, 7), (8, 30), (31, 90), (91, 180), (181, 270), (271, 365))
MODELS = ["RF", "XGB", "LightGBM", "SARIMAX", "TFT", "TabPFN", "Ensemble_unweighted", "Ensemble_MASEweighted"]

DUM = ["is_t2", "is_t4", "is_t9"]
AR_COLS = ["ar_ch4_dlag1", "ar_ch4_dlag2", "ar_ch4_dlag3", "ar_ch4_dlag7", "ar_ch4_dlag14", "ar_ch4_drm7"]
EXOG_B = ["fx_lsu_dens", "fx_WS_mean", "fx_VPD_mean", "fx_USTAR_mean", "fx_PPFD_mean",
          "fx_DOY_sin", "fx_DOY_cos", "fx_is_growing"]

# B-09's frozen MASE-derived weights (D-54) -- same constants U-02/B-10 use, not re-derived here.
_mase = {"RF": 1.024, "XGB": 0.968, "LightGBM": 0.978, "SARIMAX": 1.038}
_inv = {k: 1.0 / v for k, v in _mase.items()}
_s = sum(_inv.values())
ENSEMBLE_WEIGHTS_MASE = {k: v / _s for k, v in _inv.items()}
ENSEMBLE_WEIGHTS_UNWEIGHTED = {"RF": 0.25, "XGB": 0.25, "LightGBM": 0.25, "SARIMAX": 0.25}


def main():
    dv = pd.read_csv(f"{HOURLY}/forecast_daily_v2.csv", low_memory=False)
    dv["Datetime"] = pd.to_datetime(dv["Datetime"], format="mixed")
    FX_B = [c for c in dv.columns if c.startswith("fx")]
    T = {t: dv[dv.tower == t].set_index("Datetime").sort_index() for t in TOWERS}
    feat_cols = AR_COLS + FX_B + ["ar_fc_dlag1"] + DUM

    fdl_m = fdl.load_matrix(f"{HOURLY}/forecast_features_v2.csv")
    device = fdl.get_device()
    tft_cfg = fdl.TRACKS["B"]
    lsu_idx = fdl.FX.index("fx_lsu_dens")
    tabpfn_ok = bool(os.environ.get("TABPFN_TOKEN"))
    if not tabpfn_ok:
        print("WARNING: TABPFN_TOKEN not set -- TabPFN steps will be skipped this run.")

    u02_summary = pd.read_csv(f"{RESULTS}/u02_summary.csv")

    def get_margin(yr, tower, model_name, bin_label):
        r = u02_summary[(u02_summary.anchor_year == yr) & (u02_summary.eval_tower == tower) &
                         (u02_summary.model == model_name) & (u02_summary.bin == bin_label)]
        return float(r["conformal_margin"].iloc[0]) if len(r) and pd.notna(r["conformal_margin"].iloc[0]) else np.nan

    all_rows = []

    for yr in ANCHOR_YEARS:
        anchor = pd.Timestamp(f"{yr}-12-16")
        target_dates = pd.date_range(anchor + pd.Timedelta(days=1), periods=N_DAYS, freq="D")
        print(f"\n{'='*70}\nAnchor {yr}\n{'='*70}")
        t_anchor = time.time()

        # ---- Fit stage: pooled T2+T4+T9 trees (identical to u02_multi_anchor_tower.py) ----
        pool = []
        for t in TOWERS:
            df = T[t].copy()
            df["target"] = df["y_gapfilled"]
            for d in DUM:
                df[d] = 1.0 if d == f"is_t{t}" else 0.0
            pool.append(df[df.index <= anchor])
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
        print(f"  Pooled RF/XGB(x3)/LightGBM(x3) fit ({time.time()-t_anchor:.0f}s)")

        # ---- TFTQuantile fit (identical to u02_multi_anchor_tower.py's fit_stage TFT block) ----
        cutoff = anchor + pd.Timedelta(hours=23, minutes=59)
        val_start = anchor - pd.Timedelta(days=VAL_DAYS)
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
        print(f"  TFTQuantile fit ({time.time()-t_anchor:.0f}s total)")

        for tower in TOWERS_B:
            t_tower = time.time()
            dft = T[tower]
            history_init = dft.loc[:anchor, "y_gapfilled"].copy()
            hist_max_lsu = float(dft.loc[:anchor, "fx_lsu_dens"].max())

            # SARIMAX fit (real, unperturbed exog -- identical AIC order search)
            y = dft["y_gapfilled"].astype(float)
            X_full = dft[EXOG_B].astype(float).ffill().bfill()
            y_tr, X_tr = y.loc[:anchor], X_full.loc[:anchor]
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

            ser_t = fdl.tower_series(fdl_m, tower, "B")
            dates_full_tft = ser_t["idx"]
            anchor_idx_tft = dates_full_tft.get_loc(anchor)
            history_init_tft = ser_t["ch4"][:anchor_idx_tft + 1]

            # TabPFN context (real y_observed only, per-tower, never pooled -- matches every other
            # B-13/U-02 TabPFN call)
            hist_tabpfn = dft.loc[:anchor]
            hist_target_tabpfn = hist_tabpfn["y_observed"]
            hist_cov_tabpfn = hist_tabpfn[EXOG_B]

            for mult in MULTIPLIERS:
                fx_frame = dft.loc[target_dates, FX_B + ["ar_fc_dlag1"]].copy()
                fx_frame["fx_lsu_dens"] = fx_frame["fx_lsu_dens"] * mult
                fx_frame["is_t2"] = 1.0 if tower == 2 else 0.0
                fx_frame["is_t4"] = 1.0 if tower == 4 else 0.0
                fx_frame["is_t9"] = 1.0 if tower == 9 else 0.0
                window_max_lsu = float(fx_frame["fx_lsu_dens"].max())

                df_rf = rr.tree_rollout_quantile(rf_adapter, imp_, feat_cols, fx_frame, history_init, anchor, N_DAYS, QUANTILES)
                df_xgb = rr.tree_rollout_quantile(xgb_adapter, imp_, feat_cols, fx_frame, history_init, anchor, N_DAYS, QUANTILES)
                df_lgb = rr.tree_rollout_quantile(lgb_adapter, imp_, feat_cols, fx_frame, history_init, anchor, N_DAYS, QUANTILES)

                X_pert = X_full.copy()
                X_pert.loc[target_dates, "fx_lsu_dens"] = X_full.loc[target_dates, "fx_lsu_dens"] * mult
                df_sarimax = rr.sarimax_quantile(sarimax_res, target_dates, X_pert.loc[target_dates], alpha=0.10)
                lo_key, hi_key = round(0.10 / 2, 4), round(1 - 0.10 / 2, 4)

                enc_ex_pert = ser_t["enc_ex"].copy()
                dec_ex_pert = ser_t["dec_ex"].copy()
                post_anchor_mask = dates_full_tft > anchor
                enc_ex_pert[post_anchor_mask, lsu_idx] *= mult
                dec_ex_pert[post_anchor_mask, lsu_idx] *= mult
                df_tft = rr.dl_rollout_quantile(tft_model, se_t, sd_t, mu_t, sdy_t, device, fdl.TOW[tower],
                                                 enc_ex_pert, dec_ex_pert, dates_full_tft, history_init_tft, anchor,
                                                 quantiles=QUANTILES, L=tft_cfg["L"], H=tft_cfg["H"], n_days=N_DAYS)

                model_dfs = {
                    "RF": df_rf.rename(columns={0.05: "q05", 0.95: "q95"}),
                    "XGB": df_xgb.rename(columns={0.05: "q05", 0.95: "q95"}),
                    "LightGBM": df_lgb.rename(columns={0.05: "q05", 0.95: "q95"}),
                    "SARIMAX": df_sarimax.rename(columns={lo_key: "q05", hi_key: "q95"}),
                    "TFT": df_tft.rename(columns={0.05: "q05", 0.95: "q95"}),
                }

                # Ensembles: post-hoc weighted combination of the already-perturbed RF/XGB/
                # LightGBM/SARIMAX outputs -- no extra model fitting, matches U-02's own definition.
                constituents = {"RF": df_rf, "XGB": df_xgb, "LightGBM": df_lgb,
                                 "SARIMAX": df_sarimax.rename(columns={lo_key: 0.05, hi_key: 0.95})}
                for ens_name, weights in [("Ensemble_unweighted", ENSEMBLE_WEIGHTS_UNWEIGHTED),
                                           ("Ensemble_MASEweighted", ENSEMBLE_WEIGHTS_MASE)]:
                    ens_df = pd.DataFrame(index=target_dates)
                    ens_df["median"] = sum(weights[n] * constituents[n]["median"] for n in constituents)
                    ens_df["q05"] = sum(weights[n] * constituents[n][0.05] for n in constituents)
                    ens_df["q95"] = sum(weights[n] * constituents[n][0.95] for n in constituents)
                    model_dfs[ens_name] = ens_df

                # TabPFN: one-shot, per-tower, real (perturbed) future covariates.
                if tabpfn_ok:
                    try:
                        future_cov_tabpfn = dft.loc[target_dates, EXOG_B].copy()
                        future_cov_tabpfn["fx_lsu_dens"] = future_cov_tabpfn["fx_lsu_dens"] * mult
                        df_tabpfn = rr.tabpfn_forecast(hist_target_tabpfn, hist_cov_tabpfn, future_cov_tabpfn,
                                                        mode="local", quantiles=list(QUANTILES))
                        model_dfs["TabPFN"] = df_tabpfn.rename(columns={0.05: "q05", 0.95: "q95"})
                    except Exception as e:
                        print(f"    TabPFN SKIPPED (anchor={yr}, tower={tower}, mult={mult}): {str(e)[:80]}")

                for lo, hi in BINS:
                    label = f"{lo}-{hi}"
                    bin_dates = target_dates[(np.arange(1, N_DAYS + 1) >= lo) & (np.arange(1, N_DAYS + 1) <= hi)]
                    for model_name, mdf in model_dfs.items():
                        sub = mdf.loc[bin_dates]
                        margin = get_margin(yr, tower, model_name, label)
                        raw_width = float((sub["q95"] - sub["q05"]).mean())
                        cal_width = 2 * margin if np.isfinite(margin) else np.nan
                        all_rows.append({
                            "anchor_year": yr, "eval_tower": tower, "multiplier": mult,
                            "model": model_name, "bin": label,
                            "mean_median_pred": float(sub["median"].mean()),
                            "mean_raw_width": raw_width,
                            "nominal_calibrated_width": cal_width,
                            "window_max_lsu_dens": window_max_lsu,
                            "training_max_lsu_dens": hist_max_lsu,
                        })

            print(f"  Tower {tower}: training max fx_lsu_dens = {hist_max_lsu:.2f} ({time.time()-t_tower:.0f}s)")

        print(f"  Anchor {yr} total ({time.time()-t_anchor:.0f}s)")

    out = pd.DataFrame(all_rows)
    out.to_csv(f"{RESULTS}/u03_extrapolation_stress_test_multi.csv", index=False)
    print(f"\n[OK] Saved u03_extrapolation_stress_test_multi.csv ({len(out)} rows)")


if __name__ == "__main__":
    main()
