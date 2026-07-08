"""D-65: rerun B-10 (RF/XGB/LightGBM/SARIMAX + 2 ensembles) and B-13 (TFT/TabPFN) with the
extended `bin_metrics` (RMSE/WAPE/Correlation added to R2/MAE/MASE), ALL 3 TOWERS (T2/T4/T9),
5 anchors (2018-2022), combined into one table.

Extended from the original Tower-4-only scope per the project's own new standing convention
(CLAUDE.md "Full coverage by default") after the user asked for a tower x year breakdown and
pointed out the narrower scope should have been the default from the start.

Neither B-10's nor B-13's original multi-anchor script was ever committed (both were "ad-hoc,
not committed" per this project's own stated precedent, confirmed via `git log --all
--diff-filter=A` finding nothing). This reconstructs both from their documented methodology --
hyperparameters read directly from the committed single-anchor notebooks
(B10_daily_improvements.ipynb, B13_tft_tabpfn.ipynb) -- and is itself committed, closing that
reproducibility gap rather than repeating it (matching the newer B-14/B-15 precedent of
committing their multi-anchor scripts).

Two y_true sources are used deliberately, matching each original notebook's own convention:
- Trees/SARIMAX/ensembles/TabPFN: `forecast_daily_v2.csv`'s `y_observed` column directly
  (B-10's and B-13b's own convention).
- TFT: `fdl.tower_series`'s resampled `Y` (B-13a's own convention -- a different daily-
  resampling pipeline, count>=6 threshold). Using the wrong one for either model would silently
  change the evaluated rows and break reproduction.

Pooling structure (matches every other multi-tower B-09-U-03 script this session): RF/XGB/
LightGBM/TFT are fit ONCE per anchor on pooled T2+T4+T9 data, then rolled out separately per
tower (reusing the same fitted model, only the tower-specific fx_frame/history/static vector
changes). SARIMAX and TabPFN are never pooled -- fit fresh per anchor PER TOWER.

--- Addendum: secondary gap-filled-target metric (user's own idea, live discussion) ---
Alongside the primary y_observed-target bin_metrics() call, every chain is ALSO scored a second
time against y_gapfilled (dense/continuous, unlike sparse y_observed -- especially at Tower 2,
which is ~100% NaN under y_observed in every anchor except 2018). This is an EXPLICIT, bounded
departure from D-36/D-37's "train on gap-filled, evaluate on observed" convention for this one
secondary/exploratory check -- not a redefinition of that convention. Real circularity risk:
y_gapfilled seeds history_init (the pre-anchor AR memory every rollout builds forward from) AND
is itself a pooled RFm gap-filler's output trained on met/soil features that substantially
overlap RF/XGB/LightGBM's own forecast features -- so agreement can partly reflect "forecaster
resembles gap-filler," not real skill against reality. Every downstream table/section built from
this secondary metric must carry that caveat visibly, plus a per-bin real_frac (fraction of that
bin's days that were also real-observed) so the caveat is backed by numbers, not just prose.
bin_metrics() itself is untouched (already generic over y_true) -- this only adds a second call
per chain, reusing every already-fitted model/already-rolled-out chain, no new fits.
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

TOWERS = [2, 4, 9]
N_DAYS = 365
VAL_DAYS = 90
ANCHOR_YEARS = [2018, 2019, 2020, 2021, 2022]

DUM = ["is_t2", "is_t4", "is_t9"]
AR_COLS = ["ar_ch4_dlag1", "ar_ch4_dlag2", "ar_ch4_dlag3", "ar_ch4_dlag7", "ar_ch4_dlag14", "ar_ch4_drm7"]
EXOG_B = ["fx_lsu_dens", "fx_WS_mean", "fx_VPD_mean", "fx_USTAR_mean", "fx_PPFD_mean",
          "fx_DOY_sin", "fx_DOY_cos", "fx_is_growing"]

# B-09's frozen multi-anchor mean MASE -- NOT re-derived here, exactly matching B-10's original.
B09_MEAN_MASE = {"XGB": 0.968, "LightGBM": 0.978, "RF": 1.024, "SARIMAX": 1.038}


def fit_tree(algo, tr, feat_cols):
    """Byte-identical to B10_daily_improvements.ipynb's fit_tree()."""
    imp = SimpleImputer(strategy="mean")
    Xi = imp.fit_transform(tr[feat_cols].values)
    if algo == "RF":
        m = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=42,
                                   min_samples_leaf=10, max_features=0.5)
    elif algo == "XGB":
        m = XGBRegressor(subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42,
                          max_depth=2, learning_rate=0.02, n_estimators=400, min_child_weight=10)
    elif algo == "LightGBM":
        m = LGBMRegressor(subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42,
                           num_leaves=7, min_child_samples=10, learning_rate=0.02, n_estimators=400,
                           verbosity=-1)
    else:
        raise ValueError(algo)
    m.fit(Xi, tr["target"].values)
    return m, imp


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
        print("WARNING: TABPFN_TOKEN not set -- TabPFN will be skipped.")

    all_rows = []
    all_rows_gf = []  # secondary metric vs. y_gapfilled (addendum, see module docstring)
    all_chain_rows = []  # raw daily prediction chains, for export (never persisted before this addendum)

    for yr in ANCHOR_YEARS:
        anchor = pd.Timestamp(f"{yr}-12-16")
        target_dates = pd.date_range(anchor + pd.Timedelta(days=1), periods=N_DAYS, freq="D")
        print(f"\n{'='*70}\nAnchor {yr}\n{'='*70}")
        t_anchor = time.time()

        # ---- Pooled trees (RF/XGB/LightGBM), fit ONCE per anchor, shared across towers ----
        pool = []
        for t in TOWERS:
            df = T[t].copy()
            df["target"] = df["y_gapfilled"]
            for d in DUM:
                df[d] = 1.0 if d == f"is_t{t}" else 0.0
            pool.append(df[df.index <= anchor])
        tr = pd.concat(pool)
        tr = tr[tr["target"].notna()]

        tree_models = {}
        for algo in ["RF", "XGB", "LightGBM"]:
            tree_models[algo] = fit_tree(algo, tr, feat_cols)
        print(f"  Pooled trees fit ({time.time()-t_anchor:.0f}s)")

        # ---- TFT, fit ONCE per anchor on pooled data (B-13a's exact recipe) ----
        tft_model = None
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
            print(f"  TFT (pooled) fit ({time.time()-t_anchor:.0f}s total)")
        except Exception as e:
            print(f"  TFT FIT SKIPPED: {str(e)[:100]}")

        for tower in TOWERS:
            t_tower = time.time()
            dft = T[tower]
            history_init = dft.loc[:anchor, "y_gapfilled"].copy()
            fx_frame = dft.loc[target_dates, FX_B + ["ar_fc_dlag1"]].copy()
            fx_frame["is_t2"] = 1.0 if tower == 2 else 0.0
            fx_frame["is_t4"] = 1.0 if tower == 4 else 0.0
            fx_frame["is_t9"] = 1.0 if tower == 9 else 0.0

            y_true = dft["y_observed"].reindex(target_dates).values
            anchor_val = dft.loc[anchor, "y_gapfilled"]
            persist = rr.chain_persistence(anchor_val, N_DAYS)

            # ---- Secondary metric setup: y_gapfilled target + per-bin real-data coverage ----
            y_gf = dft["y_gapfilled"].reindex(target_dates).values
            bin_labels = rr.lead_time_bin(target_dates, anchor)
            real_frac_by_bin = {}
            for lo, hi in ((1, 7), (8, 30), (31, 90), (91, 180), (181, 270), (271, 365)):
                lbl = f"{lo}-{hi}"
                bm_mask = bin_labels == lbl
                real_frac_by_bin[lbl] = float(np.isfinite(y_true[bm_mask]).mean()) if bm_mask.sum() > 0 else np.nan

            # ---- Trees: reuse the already-fitted pooled models, roll out for this tower ----
            tree_chains = {}
            for algo, (model, imp) in tree_models.items():
                tree_chains[algo] = rr.tree_rollout(model, imp, feat_cols, fx_frame, history_init, anchor, n_days=N_DAYS)

            # ---- SARIMAX (B-10's original grid: p in [1,2], q in [0,1], d=1) -- fresh per tower ----
            y = dft["y_gapfilled"].astype(float)
            X = dft[EXOG_B].astype(float).ffill().bfill()
            y_tr, X_tr = y.loc[:anchor], X.loc[:anchor]
            best = None
            for p in [1, 2]:
                for q in [0, 1]:
                    try:
                        m = SARIMAX(y_tr, exog=X_tr, order=(p, 1, q), enforce_stationarity=False, enforce_invertibility=False)
                        res = m.fit(disp=False, maxiter=50)
                        if best is None or res.aic < best[0]:
                            best = (res.aic, (p, 1, q), res)
                    except Exception:
                        continue
            sarimax_res = best[2]
            future_X = X.loc[target_dates]
            fc = sarimax_res.get_forecast(steps=N_DAYS, exog=future_X)
            sarimax_chain = pd.Series(fc.predicted_mean.values, index=target_dates)

            # ---- Ensembles ----
            ens_df = pd.DataFrame({**tree_chains, "SARIMAX": sarimax_chain})
            ens_unweighted = ens_df.mean(axis=1)
            w = {k: 1.0 / v for k, v in B09_MEAN_MASE.items()}
            wsum = sum(w.values())
            w = {k: v / wsum for k, v in w.items()}
            ens_weighted = sum(ens_df[k] * w[k] for k in ens_df.columns)

            b10_chains = {**tree_chains, "SARIMAX": sarimax_chain,
                          "Ensemble_unweighted": ens_unweighted, "Ensemble_MASEweighted": ens_weighted}

            # ---- Raw daily chains, for export (never persisted before this addendum) ----
            chain_df = pd.DataFrame({name: c.reindex(target_dates) for name, c in b10_chains.items()})
            chain_df["y_true"] = y_true
            chain_df["y_gapfilled"] = y_gf
            chain_df["persistence"] = persist
            chain_df["tower"] = tower
            chain_df["anchor_year"] = yr
            chain_df.index.name = "date"

            for name, chain in b10_chains.items():
                yp = chain.reindex(target_dates).values
                bm = rr.bin_metrics(y_true, yp, target_dates, anchor, y_persist=persist)
                bm["model"] = name
                bm["anchor_year"] = yr
                bm["tower"] = tower
                all_rows.append(bm)

                bm_gf = rr.bin_metrics(y_gf, yp, target_dates, anchor, y_persist=persist)
                bm_gf["model"] = name
                bm_gf["anchor_year"] = yr
                bm_gf["tower"] = tower
                bm_gf["real_frac"] = bm_gf["bin"].map(real_frac_by_bin)
                all_rows_gf.append(bm_gf)

            # ---- TFT rollout: reuse the pooled-fitted model, roll out for this tower ----
            if tft_model is not None:
                try:
                    ser_t = fdl.tower_series(fdl_m, tower, "B")
                    dates_full_tft, enc_ex_full, dec_ex_full = ser_t["idx"], ser_t["enc_ex"], ser_t["dec_ex"]
                    anchor_idx_tft = dates_full_tft.get_loc(anchor)
                    history_init_tft = ser_t["ch4"][:anchor_idx_tft + 1]

                    tft_chain = rr.dl_rollout(tft_model, se_t, sd_t, mu_t, sdy_t, device, fdl.TOW[tower],
                                               enc_ex_full, dec_ex_full, dates_full_tft, history_init_tft, anchor,
                                               L=tft_cfg["L"], H=tft_cfg["H"], n_days=N_DAYS)

                    y_true_tft_full = pd.Series(ser_t["Y"], index=dates_full_tft)
                    y_true_tft = y_true_tft_full.reindex(target_dates).values
                    yp = tft_chain.reindex(target_dates).values
                    bm = rr.bin_metrics(y_true_tft, yp, target_dates, anchor, y_persist=persist)
                    bm["model"] = "TFT"
                    bm["anchor_year"] = yr
                    bm["tower"] = tower
                    all_rows.append(bm)

                    bm_gf = rr.bin_metrics(y_gf, yp, target_dates, anchor, y_persist=persist)
                    bm_gf["model"] = "TFT"
                    bm_gf["anchor_year"] = yr
                    bm_gf["tower"] = tower
                    bm_gf["real_frac"] = bm_gf["bin"].map(real_frac_by_bin)
                    all_rows_gf.append(bm_gf)

                    chain_df["TFT"] = tft_chain.reindex(target_dates).values
                    chain_df["y_true_tft"] = y_true_tft
                except Exception as e:
                    print(f"    Tower {tower} TFT rollout SKIPPED: {str(e)[:100]}")

            # ---- TabPFN (B-13b's exact recipe; same y_true source as trees/SARIMAX) -- fresh per tower ----
            if tabpfn_ok:
                try:
                    hist = dft.loc[:anchor]
                    hist_target = hist["y_observed"]
                    hist_covariates = hist[FX_B]
                    future_covariates = dft.loc[target_dates, FX_B]
                    tabpfn_chain = rr.tabpfn_forecast(hist_target, hist_covariates, future_covariates, mode="local")
                    yp = tabpfn_chain.reindex(target_dates).values
                    bm = rr.bin_metrics(y_true, yp, target_dates, anchor, y_persist=persist)
                    bm["model"] = "TabPFN"
                    bm["anchor_year"] = yr
                    bm["tower"] = tower
                    all_rows.append(bm)

                    bm_gf = rr.bin_metrics(y_gf, yp, target_dates, anchor, y_persist=persist)
                    bm_gf["model"] = "TabPFN"
                    bm_gf["anchor_year"] = yr
                    bm_gf["tower"] = tower
                    bm_gf["real_frac"] = bm_gf["bin"].map(real_frac_by_bin)
                    all_rows_gf.append(bm_gf)

                    chain_df["TabPFN"] = tabpfn_chain.reindex(target_dates).values
                except Exception as e:
                    print(f"    Tower {tower} TabPFN SKIPPED: {str(e)[:100]}")

            all_chain_rows.append(chain_df.reset_index())
            print(f"  Tower {tower} done ({time.time()-t_tower:.0f}s)")

        print(f"  Anchor {yr} total ({time.time()-t_anchor:.0f}s)")

    out = pd.concat(all_rows, ignore_index=True)
    out.to_csv(f"{RESULTS}/b10_b13_rerun_summary.csv", index=False)
    print(f"\n[OK] Saved b10_b13_rerun_summary.csv ({len(out)} rows)")

    # ---- Secondary metric outputs (addendum, see module docstring) ----
    out_gf = pd.concat(all_rows_gf, ignore_index=True)
    out_gf.to_csv(f"{RESULTS}/b10_b13_rerun_summary_vs_gapfilled.csv", index=False)
    print(f"[OK] Saved b10_b13_rerun_summary_vs_gapfilled.csv ({len(out_gf)} rows)")

    METRICS = ["R2", "RMSE", "MAE", "MASE", "WAPE", "Correlation"]
    MODEL_ORDER = ["RF", "XGB", "LightGBM", "SARIMAX", "Ensemble_unweighted",
                   "Ensemble_MASEweighted", "TFT", "TabPFN"]

    def wavg(g, col):
        d = g.dropna(subset=[col])
        return (d[col] * d["n"]).sum() / d["n"].sum() if d["n"].sum() > 0 else np.nan

    # All-tower pooled headline (per-anchor n-weighted mean across bins+towers, then mean across anchors)
    per_anchor_gf = (out_gf.groupby(["model", "anchor_year"])
                     .apply(lambda g: pd.Series({m: wavg(g, m) for m in METRICS}), include_groups=False)
                     .reset_index())
    all_towers_gf = per_anchor_gf.groupby("model")[METRICS].mean().round(3).reindex(MODEL_ORDER)
    all_towers_gf.to_csv(f"{RESULTS}/b10_b13_rerun_table_vs_gapfilled_all_towers.csv")
    print("[OK] Saved b10_b13_rerun_table_vs_gapfilled_all_towers.csv")

    # Tower x year x model breakdown (true MultiIndex: tower=parent column, anchor_year=parent row)
    per_year_gf = (out_gf.groupby(["tower", "model", "anchor_year"])
                   .apply(lambda g: pd.Series({m: wavg(g, m) for m in METRICS}), include_groups=False)
                   .reset_index())
    table_gf = per_year_gf.pivot_table(index=["anchor_year", "model"], columns="tower", values=METRICS)
    table_gf = table_gf.reorder_levels([1, 0], axis=1).sort_index(axis=1, level=0, sort_remaining=False)
    table_gf = table_gf.reindex(columns=TOWERS, level=0)
    table_gf = table_gf.reindex(columns=METRICS, level=1)
    table_gf = table_gf.sort_index(level=[0, 1])
    table_gf.to_csv(f"{RESULTS}/b10_b13_rerun_table_vs_gapfilled_by_tower_year.csv")
    print("[OK] Saved b10_b13_rerun_table_vs_gapfilled_by_tower_year.csv")

    # Coverage comparison: real-observed n vs gap-filled n, per tower (motivating hypothesis check)
    cov_obs = out.groupby("tower")["n"].sum()
    cov_gf = out_gf.groupby("tower")["n"].sum()
    print("\nCoverage (summed n across all models/anchors/bins) -- observed vs gap-filled target:")
    for t in TOWERS:
        print(f"  Tower {t}: observed n={cov_obs.get(t, 0)}, gap-filled n={cov_gf.get(t, 0)}")

    # ---- Raw daily prediction chains (never persisted before this addendum) ----
    chains = pd.concat(all_chain_rows, ignore_index=True)
    chains.to_csv(f"{RESULTS}/b10_b13_rerun_chains.csv", index=False)
    print(f"[OK] Saved b10_b13_rerun_chains.csv ({len(chains)} rows)")


if __name__ == "__main__":
    main()
