"""B-15 cross-tower evaluation: do T4-tuned hyperparameters (results/b15_winners.csv) generalize
to Tower 2 and Tower 9?

Reuses the exact same pooled-fit RF/XGB/LightGBM per anchor (training already pools T2+T4+T9 data
via one-hot dummies, so the fitted model objects are tower-agnostic) -- only the rollout target
(history_init/fx_frame/y_true_full/persist) and dummy indicator change per evaluation tower.
SARIMAX is fit separately per tower (not pooled). Real y_observed coverage in the target window is
severely uneven across towers (checked before running this): Tower 2 has usable real data at only
1/5 anchors (2018, 27.9%; 2019-2022 all 0%), Tower 9 at 4/5 (2019-2022, 49-74%; 2018 is 0%), Tower 4
at all 5 (61-93%). bin_metrics() already handles bins with <3 real points by returning NaN, so
anchors/towers with no real coverage simply contribute no signal -- not a crash, just an
uninformative row, and the final aggregate (pandas .mean(), skipna=True by default) correctly
ignores them.
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

HOURLY = rf"{ROOT}\data\Hourly"
RESULTS = rf"{ROOT}\results"

N_DAYS = 365
TOWERS = [2, 4, 9]
EVAL_TOWERS = [2, 4, 9]
ANCHOR_YEARS = [2018, 2019, 2020, 2021, 2022]

DUM = ["is_t2", "is_t4", "is_t9"]
AR_COLS = ["ar_ch4_dlag1", "ar_ch4_dlag2", "ar_ch4_dlag3", "ar_ch4_dlag7", "ar_ch4_dlag14", "ar_ch4_drm7"]
EXOG_B = ["fx_lsu_dens", "fx_WS_mean", "fx_VPD_mean", "fx_USTAR_mean", "fx_PPFD_mean",
          "fx_DOY_sin", "fx_DOY_cos", "fx_is_growing"]


def main():
    winners_df = pd.read_csv(f"{RESULTS}/b15_winners.csv")
    winners = {}
    for _, row in winners_df.iterrows():
        model = row["model"]
        params = {k: v for k, v in row.items() if pd.notna(v) and k not in ["model", "chosen_by"]}
        winners[model] = params

    print("="*70)
    print("B-15 CROSS-TOWER EVALUATION (T4-tuned winners on T2/T4/T9)")
    print("="*70)
    print(f"\nWinners loaded (from T4-only tuning): {winners}")

    dv = pd.read_csv(f"{HOURLY}/forecast_daily_v2.csv", low_memory=False)
    dv["Datetime"] = pd.to_datetime(dv["Datetime"], format="mixed")
    FX_B = [c for c in dv.columns if c.startswith("fx")]
    T = {t: dv[dv.tower == t].set_index("Datetime").sort_index() for t in TOWERS}

    feat_cols = AR_COLS + FX_B + ["ar_fc_dlag1"] + DUM
    rf_params = {k: (int(v) if k in ["min_samples_leaf"] else v) for k, v in winners["RF"].items()}
    xgb_params = {k: (int(v) if k in ["max_depth", "min_child_weight"] else v) for k, v in winners["XGB"].items()}
    lgb_params = {k: (int(v) if k in ["num_leaves", "min_child_samples"] else v) for k, v in winners["LGB"].items()}

    rows = []
    rows_chains = []

    for yr in ANCHOR_YEARS:
        anchor = pd.Timestamp(f"{yr}-12-16")
        target_dates = pd.date_range(anchor + pd.Timedelta(days=1), periods=N_DAYS, freq="D")
        print(f"\n=== Anchor {yr} ===")
        t0 = time.time()

        # Pooled training (once per anchor, same as b15_multi_anchor.py)
        pool = []
        for t in TOWERS:
            df = T[t].copy()
            df["target"] = df["y_gapfilled"]
            for d in DUM:
                df[d] = 1.0 if d == f"is_t{t}" else 0.0
            pool.append(df[df.index <= anchor])
        tr = pd.concat(pool)
        tr = tr[tr["target"].notna()]

        imp = SimpleImputer(strategy="mean")
        Xi = imp.fit_transform(tr[feat_cols].values)

        rf = RandomForestRegressor(n_estimators=500, **rf_params, n_jobs=-1, random_state=42)
        rf.fit(Xi, tr["target"].values)
        xgb = XGBRegressor(n_estimators=400, subsample=0.8, colsample_bytree=0.8, **xgb_params, n_jobs=-1, random_state=42)
        xgb.fit(Xi, tr["target"].values)
        lgb = LGBMRegressor(n_estimators=400, subsample=0.8, colsample_bytree=0.8, **lgb_params, n_jobs=-1, random_state=42, verbosity=-1)
        lgb.fit(Xi, tr["target"].values)

        for eval_t in EVAL_TOWERS:
            dft = T[eval_t]
            history_init = dft.loc[:anchor, "y_gapfilled"].copy()
            fx_frame = dft.loc[target_dates, FX_B + ["ar_fc_dlag1"]].copy()
            fx_frame["is_t2"] = 1.0 if eval_t == 2 else 0.0
            fx_frame["is_t4"] = 1.0 if eval_t == 4 else 0.0
            fx_frame["is_t9"] = 1.0 if eval_t == 9 else 0.0

            y_true_full = pd.Series(dft.loc[target_dates, "y_observed"].values, index=target_dates)
            n_real = y_true_full.notna().sum()
            anchor_val = dft.loc[anchor, "y_gapfilled"]
            persist = rr.chain_persistence(anchor_val, N_DAYS)

            print(f"  Tower {eval_t} ({n_real}/{N_DAYS} real obs)...", end=" ", flush=True)

            chain_rf = rr.tree_rollout(rf, imp, feat_cols, fx_frame, history_init, anchor, n_days=N_DAYS)
            chain_xgb = rr.tree_rollout(xgb, imp, feat_cols, fx_frame, history_init, anchor, n_days=N_DAYS)
            chain_lgb = rr.tree_rollout(lgb, imp, feat_cols, fx_frame, history_init, anchor, n_days=N_DAYS)

            # SARIMAX: fit separately per tower (not pooled)
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
            if best is not None:
                sarimax_res = best[2]
                future_X = X.loc[target_dates]
                fc = sarimax_res.get_forecast(steps=N_DAYS, exog=future_X)
                chain_sarimax = pd.Series(fc.predicted_mean.values, index=target_dates)
            else:
                chain_sarimax = pd.Series(np.nan, index=target_dates)

            ens_df = pd.DataFrame({"RF": chain_rf, "XGB": chain_xgb, "LightGBM": chain_lgb, "SARIMAX": chain_sarimax})
            ens_unweighted = ens_df.mean(axis=1)

            for model_name, chain in [("RF_tuned", chain_rf), ("XGB_tuned", chain_xgb), ("LightGBM_tuned", chain_lgb),
                                       ("SARIMAX", chain_sarimax), ("Ensemble_4model_tuned", ens_unweighted)]:
                y_pred = chain.reindex(target_dates).values
                bm = rr.bin_metrics(y_true_full.values, y_pred, target_dates, anchor, y_persist=persist)
                bm["model"] = model_name
                bm["anchor_year"] = yr
                bm["eval_tower"] = eval_t
                rows.append(bm)

            chain_frame = pd.DataFrame({
                "RF_tuned": chain_rf.reindex(target_dates),
                "XGB_tuned": chain_xgb.reindex(target_dates),
                "LightGBM_tuned": chain_lgb.reindex(target_dates),
                "SARIMAX": chain_sarimax.reindex(target_dates),
                "Ensemble_4model_tuned": ens_unweighted.reindex(target_dates),
            })
            chain_frame = chain_frame.reset_index().rename(columns={"index": "Datetime"})
            chain_frame["anchor_year"] = yr
            chain_frame["eval_tower"] = eval_t
            rows_chains.append(chain_frame)

            print("done")

        print(f"  Anchor {yr} total ({time.time()-t0:.0f}s)")

    R = pd.concat(rows, ignore_index=True)
    R.to_csv(f"{RESULTS}/b15_cross_tower_summary.csv", index=False)
    C = pd.concat(rows_chains, ignore_index=True)
    C.to_csv(f"{RESULTS}/b15_cross_tower_chains.csv", index=False)

    def wavg(g, col):
        w = g["n"]
        return (g[col] * w).sum() / w.sum() if w.sum() > 0 else np.nan

    print("\n" + "="*70)
    print("=== CROSS-TOWER RESULTS (T4-tuned hyperparameters, per-anchor-then-mean) ===")
    print("="*70)

    per_anchor = R.groupby(["eval_tower", "model", "anchor_year"]).apply(
        lambda g: pd.Series({"R2": wavg(g, "R2"), "MASE": wavg(g, "MASE"), "n_real": g["n"].sum()}),
        include_groups=False
    ).reset_index()
    agg = per_anchor.groupby(["eval_tower", "model"]).agg(
        R2_mean=("R2", "mean"), MASE_mean=("MASE", "mean"),
        n_anchors_usable=("R2", lambda s: s.notna().sum())
    ).reset_index()
    agg = agg.sort_values(["eval_tower", "R2_mean"], ascending=[True, False])
    print(agg.round(4).to_string(index=False))

    print(f"\n[OK] Saved: {RESULTS}/b15_cross_tower_summary.csv ({len(R)} rows)")


if __name__ == "__main__":
    main()
