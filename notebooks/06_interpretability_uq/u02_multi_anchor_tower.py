"""U-02: quantile-ML + conformal uncertainty for B-10's ensemble models (RF/XGB/LightGBM/SARIMAX +
2 ensembles) and B-13's TFT/TabPFN, across all 3 towers (T2/T4/T9) and the full 5-anchor
(2018-2022) sweep. Fresh methodology -- NOT modeled on the old U-01 notebook (different harness,
explicitly not used as precedent per the user's instruction); pinball/PICP/MPIW are freshly written
in src/evaluation/metrics.py, not migrated from U-01's inline versions.

Per-model quantile mechanism (stated explicitly, matches the approved plan):
- RF: quantile-regression-forest trick on the ALREADY-FITTED point model (RFQuantileAdapter) -- no
  retraining.
- XGB/LightGBM: 3 separately-fit quantile-objective models per anchor (q=0.05/0.5/0.95), same
  hyperparameters as B-10's point models otherwise (no new HPO).
- SARIMAX: `get_forecast().conf_int()` -- quantiles essentially for free from the existing fit.
- TFT: quantile head deferred (real architecture change, out of scope) -- conformal-wraps its own
  point-forecast chain instead, exactly like the ensemble.
- TabPFN: native quantile output via tabpfn-time-series's own `quantiles=` parameter (confirmed via
  its signature -- genuine library support, no fallback needed).
- Ensembles: no single native quantile mechanism -- raw interval = mean of constituent (RF, XGB,
  LightGBM, SARIMAX) q0.05/q0.95 bounds; median = mean of constituent medians (matches B-10's own
  point-forecast ensemble definition exactly). Conformal-calibrated on top, same as everything else.

Conformal calibration: leave-one-anchor-out, per lead-time bin (reusing bin_metrics's 6 bins) --
for each anchor as the held-out test anchor, pool absolute residuals |y_true - median| from the
OTHER 4 anchors' rollouts, grouped by bin, and use conformal_margins_by_bin's standard split-
conformal finite-sample correction. RAW (pre-calibration) and CALIBRATED (post-calibration)
coverage/width/pinball are both reported, so the value calibration actually adds is visible.
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
from evaluation.metrics import pinball, picp, mpiw

from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))

HOURLY = rf"{ROOT}\data\Hourly"
RESULTS = rf"{ROOT}\results"

N_DAYS = 365
TOWERS = [2, 4, 9]
ANCHOR_YEARS = [2018, 2019, 2020, 2021, 2022]
VAL_DAYS = 90
QUANTILES = (0.05, 0.5, 0.95)
ALPHA = 0.10  # 90% target coverage, matches QUANTILES

DUM = ["is_t2", "is_t4", "is_t9"]
AR_COLS = ["ar_ch4_dlag1", "ar_ch4_dlag2", "ar_ch4_dlag3", "ar_ch4_dlag7", "ar_ch4_dlag14", "ar_ch4_drm7"]
EXOG_B = ["fx_lsu_dens", "fx_WS_mean", "fx_VPD_mean", "fx_USTAR_mean", "fx_PPFD_mean",
          "fx_DOY_sin", "fx_DOY_cos", "fx_is_growing"]

BINS = ((1, 7), (8, 30), (31, 90), (91, 180), (181, 270), (271, 365))
MODELS = ["RF", "XGB", "LightGBM", "SARIMAX", "TFT", "TabPFN", "Ensemble_unweighted", "Ensemble_MASEweighted"]

# B-09's frozen MASE-derived weights (D-54) -- same constants B-10 itself uses, not re-derived here.
_mase = {"RF": 1.024, "XGB": 0.968, "LightGBM": 0.978, "SARIMAX": 1.038}
_inv = {k: 1.0 / v for k, v in _mase.items()}
_s = sum(_inv.values())
ENSEMBLE_WEIGHTS_MASE = {k: v / _s for k, v in _inv.items()}


def fit_stage(dv, feat_cols, FX_B, T, fdl_m, device, tft_cfg, tabpfn_ok):
    """Stage A: fit + roll out every model at every anchor/tower, producing per-day [q05, median,
    q95] chains. Returns a long DataFrame: anchor_year, eval_tower, model, date, q05, median, q95,
    y_true."""
    rows = []

    for yr in ANCHOR_YEARS:
        anchor = pd.Timestamp(f"{yr}-12-16")
        target_dates = pd.date_range(anchor + pd.Timedelta(days=1), periods=N_DAYS, freq="D")
        print(f"\n{'='*70}\nAnchor {yr}\n{'='*70}")
        t_anchor = time.time()

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

        for tower in TOWERS:
            t_tower = time.time()
            dft = T[tower]
            history_init = dft.loc[:anchor, "y_gapfilled"].copy()
            fx_frame = dft.loc[target_dates, FX_B + ["ar_fc_dlag1"]].copy()
            fx_frame["is_t2"] = 1.0 if tower == 2 else 0.0
            fx_frame["is_t4"] = 1.0 if tower == 4 else 0.0
            fx_frame["is_t9"] = 1.0 if tower == 9 else 0.0
            y_true_full = pd.Series(dft.loc[target_dates, "y_observed"].values, index=target_dates)

            df_rf = rr.tree_rollout_quantile(rf_adapter, imp_, feat_cols, fx_frame, history_init, anchor, N_DAYS, QUANTILES)
            df_xgb = rr.tree_rollout_quantile(xgb_adapter, imp_, feat_cols, fx_frame, history_init, anchor, N_DAYS, QUANTILES)
            df_lgb = rr.tree_rollout_quantile(lgb_adapter, imp_, feat_cols, fx_frame, history_init, anchor, N_DAYS, QUANTILES)

            y = dft["y_gapfilled"].astype(float)
            X = dft[EXOG_B].astype(float).ffill().bfill()
            y_tr, X_tr = y.loc[:anchor], X.loc[:anchor]
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
            df_sarimax = rr.sarimax_quantile(sarimax_res, target_dates, X.loc[target_dates], alpha=ALPHA)
            lo_key, hi_key = round(ALPHA / 2, 4), round(1 - ALPHA / 2, 4)

            for name, df in [("RF", df_rf), ("XGB", df_xgb), ("LightGBM", df_lgb)]:
                for d in target_dates:
                    rows.append({"anchor_year": yr, "eval_tower": tower, "model": name, "date": d,
                                 "q05": df.loc[d, 0.05], "median": df.loc[d, "median"], "q95": df.loc[d, 0.95],
                                 "y_true": y_true_full.loc[d]})
            for d in target_dates:
                rows.append({"anchor_year": yr, "eval_tower": tower, "model": "SARIMAX", "date": d,
                             "q05": df_sarimax.loc[d, lo_key], "median": df_sarimax.loc[d, "median"],
                             "q95": df_sarimax.loc[d, hi_key], "y_true": y_true_full.loc[d]})

            # Ensembles: raw interval = combination of constituent bounds; median = combination of
            # constituent medians. Unweighted = plain mean (1/4 each). MASE-weighted = B-09's frozen
            # per-model weights (D-54) -- genuinely different from unweighted, not a duplicate.
            constituents = {"RF": df_rf, "XGB": df_xgb, "LightGBM": df_lgb, "SARIMAX": df_sarimax}
            uw = {"RF": 0.25, "XGB": 0.25, "LightGBM": 0.25, "SARIMAX": 0.25}
            for ens_name, weights in [("Ensemble_unweighted", uw), ("Ensemble_MASEweighted", ENSEMBLE_WEIGHTS_MASE)]:
                ens_median = sum(weights[n] * constituents[n]["median"] for n in constituents)
                ens_lo = sum(weights[n] * (constituents[n][lo_key] if n == "SARIMAX" else constituents[n][0.05]) for n in constituents)
                ens_hi = sum(weights[n] * (constituents[n][hi_key] if n == "SARIMAX" else constituents[n][0.95]) for n in constituents)
                for d in target_dates:
                    rows.append({"anchor_year": yr, "eval_tower": tower, "model": ens_name, "date": d,
                                 "q05": ens_lo.loc[d], "median": ens_median.loc[d], "q95": ens_hi.loc[d],
                                 "y_true": y_true_full.loc[d]})

            # TFT: native quantile head (TFTQuantile), same treatment as every other model --
            # replaces the original conformal-only wrap now that TFTQuantile/train_quantile's
            # early-stopping/dl_rollout_quantile exist (see DECISIONS.md D-62 addendum).
            try:
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

                ser_t = fdl.tower_series(fdl_m, tower, "B")
                dates_full_tft, enc_ex_full, dec_ex_full = ser_t["idx"], ser_t["enc_ex"], ser_t["dec_ex"]
                anchor_idx_tft = dates_full_tft.get_loc(anchor)
                history_init_tft = ser_t["ch4"][:anchor_idx_tft + 1]
                df_tft = rr.dl_rollout_quantile(tft_model, se_t, sd_t, mu_t, sdy_t, device, fdl.TOW[tower],
                                                 enc_ex_full, dec_ex_full, dates_full_tft, history_init_tft, anchor,
                                                 quantiles=QUANTILES, L=tft_cfg["L"], H=tft_cfg["H"], n_days=N_DAYS)
                for d in target_dates:
                    rows.append({"anchor_year": yr, "eval_tower": tower, "model": "TFT", "date": d,
                                 "q05": df_tft.loc[d, 0.05], "median": df_tft.loc[d, "median"],
                                 "q95": df_tft.loc[d, 0.95], "y_true": y_true_full.loc[d]})
                print(f"    TFT native quantile done")
            except Exception as e:
                print(f"    TFT SKIPPED (error: {str(e)[:80]})")

            # TabPFN: native quantile via quantiles=
            if tabpfn_ok:
                try:
                    hist = dft.loc[:anchor]
                    hist_target = hist["y_observed"]
                    hist_cov = hist[FX_B]
                    future_cov = dft.loc[target_dates, FX_B]
                    tabpfn_df = rr.tabpfn_forecast(hist_target, hist_cov, future_cov, mode="local", quantiles=list(QUANTILES))
                    for d in target_dates:
                        rows.append({"anchor_year": yr, "eval_tower": tower, "model": "TabPFN", "date": d,
                                     "q05": tabpfn_df.loc[d, 0.05], "median": tabpfn_df.loc[d, "median"],
                                     "q95": tabpfn_df.loc[d, 0.95], "y_true": y_true_full.loc[d]})
                    print(f"    TabPFN native quantile done")
                except Exception as e:
                    print(f"    TabPFN SKIPPED (error: {str(e)[:80]})")

            print(f"  Tower {tower} done ({time.time()-t_tower:.0f}s)")

        print(f"  Anchor {yr} total ({time.time()-t_anchor:.0f}s)")

    return pd.DataFrame(rows)


def evaluate_stage(chains):
    """Stage B: leave-one-anchor-out conformal calibration + RAW vs CALIBRATED evaluation, per
    model/tower/bin. Returns a summary DataFrame."""
    summary_rows = []
    chains["date"] = pd.to_datetime(chains["date"])

    for model in chains["model"].unique():
        for tower in TOWERS:
            sub_all = chains[(chains.model == model) & (chains.eval_tower == tower)]
            if sub_all.empty:
                continue

            for test_yr in ANCHOR_YEARS:
                anchor = pd.Timestamp(f"{test_yr}-12-16")
                test_sub = sub_all[sub_all.anchor_year == test_yr]
                if test_sub.empty:
                    continue
                test_bins = rr.lead_time_bin(test_sub["date"].values, anchor, BINS)

                # RAW metrics (native quantile / conformal-free), only if q05/q95 exist for this model
                has_native_q = test_sub["q05"].notna().any()

                # Calibration: pool residuals from the OTHER anchors, grouped by bin
                calib_sub = sub_all[sub_all.anchor_year != test_yr]
                calib_res_by_bin = {}
                for calib_yr in calib_sub["anchor_year"].unique():
                    c_anchor = pd.Timestamp(f"{calib_yr}-12-16")
                    c_rows = calib_sub[calib_sub.anchor_year == calib_yr]
                    c_bins = rr.lead_time_bin(c_rows["date"].values, c_anchor, BINS)
                    resid = np.abs(c_rows["y_true"].values - c_rows["median"].values)
                    for lo, hi in BINS:
                        label = f"{lo}-{hi}"
                        mask = (c_bins == label) & np.isfinite(resid)
                        calib_res_by_bin.setdefault(label, []).extend(resid[mask].tolist())
                margins = rr.conformal_margins_by_bin(calib_res_by_bin, alpha=ALPHA)

                for lo, hi in BINS:
                    label = f"{lo}-{hi}"
                    m = test_bins == label
                    if m.sum() < 3:
                        continue
                    yt = test_sub["y_true"].values[m]
                    med = test_sub["median"].values[m]
                    real_mask = np.isfinite(yt)
                    if real_mask.sum() < 3:
                        continue
                    yt_r, med_r = yt[real_mask], med[real_mask]

                    row = {"anchor_year": test_yr, "eval_tower": tower, "model": model, "bin": label,
                           "n": int(real_mask.sum())}

                    if has_native_q:
                        q05 = test_sub["q05"].values[m][real_mask]
                        q95 = test_sub["q95"].values[m][real_mask]
                        row["raw_picp"] = picp(yt_r, q05, q95)
                        row["raw_mpiw"] = mpiw(q05, q95)
                        row["raw_pinball"] = pinball(yt_r, {0.05: q05, 0.5: med_r, 0.95: q95}, QUANTILES)

                    margin = margins.get(label, np.nan)
                    cal_lo, cal_hi = med_r - margin, med_r + margin
                    row["conformal_picp"] = picp(yt_r, cal_lo, cal_hi)
                    row["conformal_mpiw"] = mpiw(cal_lo, cal_hi)
                    row["conformal_pinball"] = pinball(yt_r, {0.05: cal_lo, 0.5: med_r, 0.95: cal_hi}, QUANTILES)
                    row["conformal_margin"] = margin

                    summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def main():
    dv = pd.read_csv(f"{HOURLY}/forecast_daily_v2.csv", low_memory=False)
    dv["Datetime"] = pd.to_datetime(dv["Datetime"], format="mixed")
    FX_B = [c for c in dv.columns if c.startswith("fx")]
    T = {t: dv[dv.tower == t].set_index("Datetime").sort_index() for t in TOWERS}
    feat_cols = AR_COLS + FX_B + ["ar_fc_dlag1"] + DUM

    fdl_m = fdl.load_matrix(f"{HOURLY}/forecast_features_v2.csv")
    device = fdl.get_device()
    tft_cfg = fdl.TRACKS["B"]
    tabpfn_ok = bool(os.environ.get("TABPFN_TOKEN"))
    if not tabpfn_ok:
        print("WARNING: TABPFN_TOKEN not set -- TabPFN steps will be skipped this run.")

    print("="*70)
    print("U-02 STAGE A: FIT + ROLL OUT QUANTILE/POINT CHAINS")
    print("="*70)
    chains = fit_stage(dv, feat_cols, FX_B, T, fdl_m, device, tft_cfg, tabpfn_ok)
    chains.to_csv(f"{RESULTS}/u02_chains.csv", index=False)
    print(f"\n[OK] Saved u02_chains.csv ({len(chains)} rows)")

    print("\n" + "="*70)
    print("U-02 STAGE B: LEAVE-ONE-ANCHOR-OUT CONFORMAL CALIBRATION + EVALUATION")
    print("="*70)
    summary = evaluate_stage(chains)
    summary.to_csv(f"{RESULTS}/u02_summary.csv", index=False)
    print(f"\n[OK] Saved u02_summary.csv ({len(summary)} rows)")

    def wavg(g, col):
        # explicit all-NaN guard: pandas' .sum() on an all-NaN column silently returns 0.0 (sum of
        # nothing), not NaN -- without this check, a fully-missing column (e.g. TFT's raw_picp,
        # which never has native quantiles) would misleadingly report as a confident 0.0 rather
        # than "no data".
        vals = g[col]
        if vals.isna().all():
            return np.nan
        w = g["n"]
        return (vals * w).sum() / w.sum() if w.sum() > 0 else np.nan

    print("\nPer-model/tower aggregate (n-weighted mean across bins):")
    agg = summary.groupby(["model", "eval_tower"]).apply(
        lambda g: pd.Series({
            "raw_picp": wavg(g, "raw_picp") if "raw_picp" in g else np.nan,
            "raw_mpiw": wavg(g, "raw_mpiw") if "raw_mpiw" in g else np.nan,
            "raw_pinball": wavg(g, "raw_pinball") if "raw_pinball" in g else np.nan,
            "conformal_picp": wavg(g, "conformal_picp"),
            "conformal_mpiw": wavg(g, "conformal_mpiw"),
            "conformal_pinball": wavg(g, "conformal_pinball"),
        }), include_groups=False
    ).reset_index()
    print(agg.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
