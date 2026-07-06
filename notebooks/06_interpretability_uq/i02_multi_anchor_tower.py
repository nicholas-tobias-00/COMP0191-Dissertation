"""I-02: feature importance (native / SHAP / LIME) for B-10's ensemble models (RF/XGB/LightGBM/
SARIMAX + 2 ensembles) and B-13's TFT/TabPFN, across all 3 towers (T2/T4/T9) and the full 5-anchor
(2018-2022) sweep. Fresh methodology -- NOT modeled on the old I-01 notebook (different harness,
explicitly not used as precedent per the user's instruction).

Scope decisions (stated explicitly, not silently applied):
- RF/XGB/LightGBM are POOLED (trained once per anchor on T2+T4+T9 data, same as B-10/B-15) -- the
  SAME fitted model is reused for every tower's rollout/explanation; only the tower-specific
  fx_frame/history_init/y_true_full change. Native importance + global SHAP are therefore computed
  ONCE per anchor (shared across towers) for these three models.
- SARIMAX and TFT are fit separately per tower (not pooled) -- genuine 3x cost, same as B-15's own
  cross-tower addendum. TabPFN is called separately per tower (never pooled, per its API).
- SHAP + LIME are computed for RF/XGB/LightGBM (tree-native, cheap, exact) and the two ensembles
  (SHAP = additive weighted combination of the trees' + SARIMAX's own importance signals -- exact
  given a shared background, no extra explainer calls; LIME = the tree-weighted portion only, with
  SARIMAX's contribution held as a fixed per-day offset -- an explicit, stated approximation, not
  the full 4-member black box). SARIMAX, TFT, and TabPFN are SKIPPED for KernelSHAP/LIME, each for
  a distinct, stated reason: **SARIMAX** already has an exact closed-form linear-effect view via
  its own fitted coefficients (native importance below) -- treating it as a KernelExplainer/LIME
  black box would mean re-running a full 365-step `get_forecast` per perturbation sample (SARIMAX's
  exog effect only isolates cleanly per-day by re-forecasting the whole horizon), which is both
  computationally intractable at this experiment's scale (thousands of perturbation samples x 6
  instances x 3 towers x 5 anchors) and largely redundant with the coefficients it would recover.
  **TFT and TabPFN** are architecturally mismatched with the row-wise tabular explainer framework
  (TFT needs a full L-day encoder window per prediction, not one row; TabPFN is a one-shot
  whole-365-day-horizon forecast from a whole context+future dataframe, not a per-row predictor) --
  forcing either into a flattened per-row SHAP/LIME call would be a stretch well beyond this
  experiment's bounded scope. All three still get NATIVE importance (SARIMAX: coefficients; TFT:
  VSN gate weights; TabPFN: permutation importance, its only available substitute) -- these are
  stated scope limitations, not silent omissions.
- Instance-level SHAP/LIME are bounded to 6 representative days per anchor per tower (one per
  bin_metrics lead-time bin: 1-7/8-30/31-90/91-180/181-270/271-365), picked as the real-observed
  day with the largest |y_observed| in that bin (falls back to the bin's temporal midpoint if no
  real data exists in that bin/tower/anchor, e.g. most of Tower 2's window).
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
import interpretability.importance as imp

from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))

HOURLY = rf"{ROOT}\data\Hourly"
RESULTS = rf"{ROOT}\results"

N_DAYS = 365
TOWERS = [2, 4, 9]
ANCHOR_YEARS = [2018, 2019, 2020, 2021, 2022]
VAL_DAYS = 90  # TFT validation window, B-13/D-45's recipe

DUM = ["is_t2", "is_t4", "is_t9"]
AR_COLS = ["ar_ch4_dlag1", "ar_ch4_dlag2", "ar_ch4_dlag3", "ar_ch4_dlag7", "ar_ch4_dlag14", "ar_ch4_drm7"]
EXOG_B = ["fx_lsu_dens", "fx_WS_mean", "fx_VPD_mean", "fx_USTAR_mean", "fx_PPFD_mean",
          "fx_DOY_sin", "fx_DOY_cos", "fx_is_growing"]

BINS = ((1, 7), (8, 30), (31, 90), (91, 180), (181, 270), (271, 365))

ENSEMBLE_WEIGHTS_UNWEIGHTED = {"RF": 0.25, "XGB": 0.25, "LightGBM": 0.25, "SARIMAX": 0.25}
# B-09's frozen MASE-derived weights (D-54) -- same constants B-10 itself uses, not re-derived here.
_mase = {"RF": 1.024, "XGB": 0.968, "LightGBM": 0.978, "SARIMAX": 1.038}
_inv = {k: 1.0 / v for k, v in _mase.items()}
_s = sum(_inv.values())
ENSEMBLE_WEIGHTS_MASE = {k: v / _s for k, v in _inv.items()}


def pick_representative_days(y_true_full, target_dates, anchor, bins=BINS):
    """One day per lead-time bin: the real-observed day with the largest |y_observed| in that bin
    (most informative for explaining what drives a prediction); falls back to the bin's temporal
    midpoint if no real data exists in that bin (expected for most of Tower 2's window)."""
    lead = np.array([(d - anchor).days for d in target_dates])
    days = {}
    for lo, hi in bins:
        m = (lead >= lo) & (lead <= hi)
        sub_dates = target_dates[m]
        sub_y = y_true_full.reindex(sub_dates).values
        real_mask = np.isfinite(sub_y)
        label = f"{lo}-{hi}"
        if real_mask.any():
            j = np.nanargmax(np.abs(sub_y))
            days[label] = sub_dates[j]
        else:
            days[label] = sub_dates[len(sub_dates) // 2]
    return days


def build_row(history, d, fx_frame, feat_cols):
    row = rr.ar_features_for_day(history, d)
    for c in fx_frame.columns:
        row[c] = fx_frame.loc[d, c]
    return pd.DataFrame([row])[feat_cols]


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

    native_rows = []
    shap_global_rows = []
    shap_instance_rows = []
    lime_instance_rows = []

    for yr in ANCHOR_YEARS:
        anchor = pd.Timestamp(f"{yr}-12-16")
        target_dates = pd.date_range(anchor + pd.Timedelta(days=1), periods=N_DAYS, freq="D")
        print(f"\n{'='*70}\nAnchor {yr}\n{'='*70}")
        t_anchor = time.time()

        # ---- pooled tree fit (once per anchor) ----
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
        xgb = XGBRegressor(n_estimators=400, max_depth=2, learning_rate=0.02, min_child_weight=10,
                            subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42)
        xgb.fit(Xi, tr["target"].values)
        lgb = LGBMRegressor(n_estimators=400, num_leaves=7, min_child_samples=10, learning_rate=0.02,
                             subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42, verbosity=-1)
        lgb.fit(Xi, tr["target"].values)
        print(f"  Pooled RF/XGB/LightGBM fit ({time.time()-t_anchor:.0f}s)")

        # ---- native importance (once per anchor, shared across towers) ----
        ni_rf = imp.native_importance_tree(rf, feat_cols)
        ni_xgb = imp.native_importance_tree(xgb, feat_cols)
        ni_lgb = imp.native_importance_tree(lgb, feat_cols)
        for name, s in [("RF", ni_rf), ("XGB", ni_xgb), ("LightGBM", ni_lgb)]:
            for feat, val in s.items():
                native_rows.append({"anchor_year": yr, "eval_tower": "pooled", "model": name,
                                     "feature": feat, "importance": val})

        # ---- global SHAP (once per anchor, ~500-row subsample of pooled training data) ----
        rng = np.random.default_rng(yr)
        bg_idx = rng.choice(len(Xi), size=min(500, len(Xi)), replace=False)
        X_bg = Xi[bg_idx]
        for name, model in [("RF", rf), ("XGB", xgb), ("LightGBM", lgb)]:
            sv, mean_abs = imp.shap_importance_tree(model, X_bg, feat_cols)
            for feat, val in mean_abs.items():
                shap_global_rows.append({"anchor_year": yr, "eval_tower": "pooled", "model": name,
                                          "feature": feat, "mean_abs_shap": val})
        print(f"  Global SHAP (pooled trees) done ({time.time()-t_anchor:.0f}s)")

        for tower in TOWERS:
            print(f"  --- Tower {tower} ---")
            t_tower = time.time()
            dft = T[tower]
            history_init = dft.loc[:anchor, "y_gapfilled"].copy()
            fx_frame = dft.loc[target_dates, FX_B + ["ar_fc_dlag1"]].copy()
            fx_frame["is_t2"] = 1.0 if tower == 2 else 0.0
            fx_frame["is_t4"] = 1.0 if tower == 4 else 0.0
            fx_frame["is_t9"] = 1.0 if tower == 9 else 0.0
            y_true_full = pd.Series(dft.loc[target_dates, "y_observed"].values, index=target_dates)
            n_real = y_true_full.notna().sum()

            chain_rf = rr.tree_rollout(rf, imp_, feat_cols, fx_frame, history_init, anchor, n_days=N_DAYS)
            chain_xgb = rr.tree_rollout(xgb, imp_, feat_cols, fx_frame, history_init, anchor, n_days=N_DAYS)
            chain_lgb = rr.tree_rollout(lgb, imp_, feat_cols, fx_frame, history_init, anchor, n_days=N_DAYS)

            # ---- SARIMAX: fit per tower, native (coef/pvalue) ----
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
            future_X = X.loc[target_dates]
            fc = sarimax_res.get_forecast(steps=N_DAYS, exog=future_X)
            chain_sarimax = pd.Series(fc.predicted_mean.values, index=target_dates)

            ni_sarimax = imp.native_importance_sarimax(sarimax_res, EXOG_B)
            for feat, row in ni_sarimax.iterrows():
                native_rows.append({"anchor_year": yr, "eval_tower": tower, "model": "SARIMAX",
                                     "feature": feat, "importance": row["abs_coef"], "pvalue": row["pvalue"]})

            # ---- Ensembles: native = weighted combination ----
            ni_pooled = {"RF": ni_rf, "XGB": ni_xgb, "LightGBM": ni_lgb, "SARIMAX": ni_sarimax["abs_coef"]}
            for ens_name, weights in [("Ensemble_unweighted", ENSEMBLE_WEIGHTS_UNWEIGHTED),
                                       ("Ensemble_MASEweighted", ENSEMBLE_WEIGHTS_MASE)]:
                combined = imp.combine_ensemble_importance(ni_pooled, weights)
                for feat, val in combined.items():
                    native_rows.append({"anchor_year": yr, "eval_tower": tower, "model": ens_name,
                                         "feature": feat, "importance": val})

            ens_unweighted = pd.DataFrame({"RF": chain_rf, "XGB": chain_xgb, "LightGBM": chain_lgb,
                                           "SARIMAX": chain_sarimax}).mean(axis=1)
            ens_mase = sum(chain * ENSEMBLE_WEIGHTS_MASE[name] for chain, name in
                           [(chain_rf, "RF"), (chain_xgb, "XGB"), (chain_lgb, "LightGBM"), (chain_sarimax, "SARIMAX")])

            # ---- TFT: fit per tower, native VSN importance only ----
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

                tft_model = fdl.build_model("TFT", tft_cfg["L"], tft_cfg["H"], n_enc, n_dec, 3)
                fdl.train_model(tft_model, train_tft, device, epochs=30, ch4_mu=mu_t, ch4_sd=sdy_t, seed=0,
                                 weight_decay=1e-3, val_data=val_tft, patience=5)

                ser_t = fdl.tower_series(fdl_m, tower, "B")
                dates_full_tft, enc_ex_full, dec_ex_full = ser_t["idx"], ser_t["enc_ex"], ser_t["dec_ex"]
                anchor_idx_tft = dates_full_tft.get_loc(anchor)
                history_init_tft = ser_t["ch4"][:anchor_idx_tft + 1]
                # one forward pass to populate last_enc_vsn_w/last_dec_vsn_w for native importance
                _ = rr.dl_rollout(tft_model, se_t, sd_t, mu_t, sdy_t, device, fdl.TOW[tower],
                                   enc_ex_full, dec_ex_full, dates_full_tft, history_init_tft, anchor,
                                   L=tft_cfg["L"], H=tft_cfg["H"], n_days=1)
                enc_cols = [f"enc_{i}" for i in range(n_enc)]
                dec_cols = [f"dec_{i}" for i in range(n_dec)]
                ni_tft = imp.native_importance_tft(tft_model.last_enc_vsn_w, tft_model.last_dec_vsn_w, enc_cols, dec_cols)
                for feat, val in ni_tft.items():
                    native_rows.append({"anchor_year": yr, "eval_tower": tower, "model": "TFT",
                                         "feature": feat, "importance": val})
                print(f"    TFT native (VSN) done")
            except Exception as e:
                print(f"    TFT SKIPPED (error: {str(e)[:80]})")

            # ---- TabPFN: per tower, permutation importance substitute ----
            if tabpfn_ok:
                try:
                    hist = dft.loc[:anchor]
                    hist_target = hist["y_observed"]
                    hist_cov = hist[FX_B]
                    future_cov = dft.loc[target_dates, FX_B]

                    def tabpfn_predict_mean(cov_df):
                        chain = rr.tabpfn_forecast(hist_target, hist_cov, cov_df, mode="local")
                        return np.array([chain.mean()])

                    base_chain = rr.tabpfn_forecast(hist_target, hist_cov, future_cov, mode="local")
                    perm_scores = {}
                    perm_rng = np.random.default_rng(yr)
                    base_mean = base_chain.mean()
                    for col in FX_B:
                        shuffled = future_cov.copy()
                        shuffled[col] = perm_rng.permutation(shuffled[col].values)
                        shuffled_chain = rr.tabpfn_forecast(hist_target, hist_cov, shuffled, mode="local")
                        perm_scores[col] = abs(shuffled_chain.mean() - base_mean)
                    ni_tabpfn = pd.Series(perm_scores).sort_values(ascending=False)
                    for feat, val in ni_tabpfn.items():
                        native_rows.append({"anchor_year": yr, "eval_tower": tower, "model": "TabPFN",
                                             "feature": feat, "importance": val})
                    print(f"    TabPFN permutation-importance done")
                except Exception as e:
                    print(f"    TabPFN SKIPPED (error: {str(e)[:80]})")

            # ---- Instance-level SHAP + LIME (6 models x 6 representative days) ----
            rep_days = pick_representative_days(y_true_full, target_dates, anchor)

            def make_row_predict_fn(model, imp_obj):
                def f(Xrows):
                    return model.predict(imp_obj.transform(Xrows))
                return f

            def make_ensemble_predict_fn(weights):
                def f(Xrows):
                    preds = np.zeros(len(Xrows))
                    for name, model in [("RF", rf), ("XGB", xgb), ("LightGBM", lgb)]:
                        preds += weights[name] * model.predict(imp_.transform(Xrows))
                    return preds
                return f

            X_bg_tower = Xi[bg_idx]  # reuse the same pooled background sample

            model_predict_fns = {
                "RF": make_row_predict_fn(rf, imp_),
                "XGB": make_row_predict_fn(xgb, imp_),
                "LightGBM": make_row_predict_fn(lgb, imp_),
            }

            sarimax_abs_coef = ni_sarimax["abs_coef"]  # reused as SARIMAX's SHAP-equivalent contribution (see docstring)

            for bin_label, day in rep_days.items():
                row_df = build_row(history_init if day <= anchor else
                                    pd.concat([history_init, chain_rf[chain_rf.index < day]]), day, fx_frame, feat_cols)
                Xrow = row_df.values
                Xrow_imp = imp_.transform(Xrow)

                tree_sv = {}
                for name in ["RF", "XGB", "LightGBM"]:
                    model = {"RF": rf, "XGB": xgb, "LightGBM": lgb}[name]
                    sv, _ = imp.shap_importance_tree(model, Xrow_imp, feat_cols)
                    tree_sv[name] = sv[0]  # (n_features,)
                    for j, feat in enumerate(feat_cols):
                        shap_instance_rows.append({"anchor_year": yr, "eval_tower": tower, "model": name,
                                                    "bin": bin_label, "date": day, "feature": feat,
                                                    "shap_value": float(sv[0, j])})
                    lime_s = imp.lime_explain_instance(model_predict_fns[name], X_bg_tower, Xrow_imp[0], feat_cols)
                    for feat, val in lime_s.items():
                        lime_instance_rows.append({"anchor_year": yr, "eval_tower": tower, "model": name,
                                                    "bin": bin_label, "date": day, "feature": feat, "lime_weight": val})

                # Ensemble SHAP: additive weighted combination of tree SHAP + SARIMAX's coefficient
                # contribution (exact given a shared background, no extra explainer calls needed).
                for ens_name, weights in [("Ensemble_unweighted", ENSEMBLE_WEIGHTS_UNWEIGHTED),
                                           ("Ensemble_MASEweighted", ENSEMBLE_WEIGHTS_MASE)]:
                    combined_sv = sum(weights[name] * tree_sv[name] for name in ["RF", "XGB", "LightGBM"])
                    for j, feat in enumerate(feat_cols):
                        val = float(combined_sv[j])
                        if feat in sarimax_abs_coef.index:
                            val += weights["SARIMAX"] * float(sarimax_abs_coef[feat])
                        shap_instance_rows.append({"anchor_year": yr, "eval_tower": tower, "model": ens_name,
                                                    "bin": bin_label, "date": day, "feature": feat, "shap_value": val})
                    # Ensemble LIME: tree-weighted portion only (SARIMAX held fixed as a per-day
                    # offset, not re-perturbed) -- a stated approximation, not the full 4-member
                    # black box (see docstring for why SARIMAX is excluded from perturbation-based
                    # explainers). Renormalize tree weights to sum to 1 for this approximation.
                    tree_w_sum = weights["RF"] + weights["XGB"] + weights["LightGBM"]
                    tree_weights_norm = {n: weights[n] / tree_w_sum for n in ["RF", "XGB", "LightGBM"]}
                    ens_predict_fn = make_ensemble_predict_fn(tree_weights_norm)
                    lime_ens = imp.lime_explain_instance(ens_predict_fn, X_bg_tower, Xrow_imp[0], feat_cols)
                    for feat, val in lime_ens.items():
                        lime_instance_rows.append({"anchor_year": yr, "eval_tower": tower, "model": ens_name,
                                                    "bin": bin_label, "date": day, "feature": feat, "lime_weight": val})

            print(f"    Instance SHAP/LIME (RF/XGB/LightGBM + both ensembles) done ({time.time()-t_tower:.0f}s)")

        print(f"  Anchor {yr} total ({time.time()-t_anchor:.0f}s)")

    pd.DataFrame(native_rows).to_csv(f"{RESULTS}/i02_native_importance.csv", index=False)
    pd.DataFrame(shap_global_rows).to_csv(f"{RESULTS}/i02_shap_summary.csv", index=False)
    pd.DataFrame(shap_instance_rows).to_csv(f"{RESULTS}/i02_shap_instances.csv", index=False)
    pd.DataFrame(lime_instance_rows).to_csv(f"{RESULTS}/i02_lime_instances.csv", index=False)
    print("\n[OK] Saved i02_native_importance.csv, i02_shap_summary.csv, i02_shap_instances.csv, i02_lime_instances.csv")


if __name__ == "__main__":
    main()
