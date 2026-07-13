"""F-10 (D-67) Stage 2b: does any of the 5 new feature families help the RECURSIVE-ROLLOUT
forecasting track specifically (B-10's ensemble), as opposed to the direct point-forecast track
that Stage 1's signal check used?

Motivation (user, 2026-07-10): Stage 1's point-forecast smoke test (f10_signal_check.py) found no
family cleared the go/no-go bar, but that harness only exercises a direct, non-autoregressive
h-step-ahead forecast. The actual problem this whole F-10 branch was meant to address -- the
B-09->B-15 recursive-rollout sequence's model/HPO ceiling -- lives specifically in the 365-day
autoregressive rollout (error compounding, spike-blindness), a genuinely different failure mode a
one-shot direct forecast never exercises. Per direct user instruction, this Stage 2b runs anyway
despite Stage 1's null result, since "the point of this experiment is forecasting performance
improvements" and the rollout is the track that actually matters.

Mirrors b10_b13_rerun_multi_anchor.py's exact methodology (same TOWERS/ANCHOR_YEARS/DUM/AR_COLS/
EXOG_B, same fit_tree hyperparameters, same SARIMAX order-search, same tree_rollout/bin_metrics
calls from recursive_rollout.py, unmodified) -- the ONLY thing that varies across the 6 runs below
is which fx_ columns the tree models see (feat_cols), reading from forecast_daily_v3.csv instead
of v2. SARIMAX is fit ONCE per (anchor, tower) and reused across all 6 configs (its EXOG_B set is
unchanged by any of the new families, so refitting it per config would be pure waste).

Run from project root:  python notebooks/05_benchmarking/b16_recursive_rollout_v3.py
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

HOURLY = rf"{ROOT}\data\Hourly"
RESULTS = rf"{ROOT}\results"

TOWERS = [2, 4, 9]
N_DAYS = 365
ANCHOR_YEARS = [2018, 2019, 2020, 2021, 2022]

DUM = ["is_t2", "is_t4", "is_t9"]
AR_COLS = ["ar_ch4_dlag1", "ar_ch4_dlag2", "ar_ch4_dlag3", "ar_ch4_dlag7", "ar_ch4_dlag14", "ar_ch4_drm7"]
EXOG_B = ["fx_lsu_dens", "fx_WS_mean", "fx_VPD_mean", "fx_USTAR_mean", "fx_PPFD_mean",
          "fx_DOY_sin", "fx_DOY_cos", "fx_is_growing"]

B09_MEAN_MASE = {"XGB": 0.968, "LightGBM": 0.978, "RF": 1.024, "SARIMAX": 1.038}

FAMILIES = {
    "species": ["fx_cattle_dens", "fx_sheep_dens", "fx_lamb_dens"],
    "arable": ["fx_is_arable"],
    "flow": ["fx_flow_mean", "fx_flow_lag7", "fx_flow_lag14", "fx_flow_lag21", "fx_flow_lag28",
             "fx_flow_roll7", "fx_flow_roll14"],
    "mgmt": ["fx_mgmt_fertN_recency", "fx_mgmt_fertN_rate", "fx_mgmt_lime_recency",
             "fx_mgmt_cultiv_recency", "fx_mgmt_cut_recency", "fx_mgmt_manure_recency"],
    "bodyweight": ["fx_total_liveweight_dens"],
}
ALL_NEW = sorted({c for cols in FAMILIES.values() for c in cols})


def fit_tree(algo, tr, feat_cols):
    """Byte-identical hyperparameters to b10_b13_rerun_multi_anchor.py's fit_tree()."""
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


def fit_sarimax(dft, anchor, target_dates):
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
    return pd.Series(fc.predicted_mean.values, index=target_dates)


def main():
    dv = pd.read_csv(f"{HOURLY}/forecast_daily_v3.csv", low_memory=False)
    dv["Datetime"] = pd.to_datetime(dv["Datetime"], format="mixed")
    fx_all = [c for c in dv.columns if c.startswith("fx")]
    BASE_FX = [c for c in fx_all if c not in ALL_NEW]
    T = {t: dv[dv.tower == t].set_index("Datetime").sort_index() for t in TOWERS}

    configs = {"BASE": BASE_FX}
    for fam, cols in FAMILIES.items():
        configs[f"BASE+{fam}"] = BASE_FX + cols
    print(f"Configs: {list(configs.keys())}")

    all_rows = []
    all_chain_rows = []
    sarimax_cache = {}

    for yr in ANCHOR_YEARS:
        anchor = pd.Timestamp(f"{yr}-12-16")
        target_dates = pd.date_range(anchor + pd.Timedelta(days=1), periods=N_DAYS, freq="D")
        print(f"\n{'='*70}\nAnchor {yr}\n{'='*70}")
        t_anchor = time.time()

        # ---- SARIMAX: fit ONCE per (anchor, tower), reused across all 6 configs ----
        for tower in TOWERS:
            sarimax_cache[(yr, tower)] = fit_sarimax(T[tower], anchor, target_dates)
        print(f"  SARIMAX fit for all 3 towers ({time.time()-t_anchor:.0f}s)")

        for cfg_name, fx_cols in configs.items():
            t_cfg = time.time()
            feat_cols = AR_COLS + fx_cols + ["ar_fc_dlag1"] + DUM

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

            for tower in TOWERS:
                dft = T[tower]
                history_init = dft.loc[:anchor, "y_gapfilled"].copy()
                fx_frame = dft.loc[target_dates, fx_cols + ["ar_fc_dlag1"]].copy()
                fx_frame["is_t2"] = 1.0 if tower == 2 else 0.0
                fx_frame["is_t4"] = 1.0 if tower == 4 else 0.0
                fx_frame["is_t9"] = 1.0 if tower == 9 else 0.0

                y_true = dft["y_observed"].reindex(target_dates).values
                anchor_val = dft.loc[anchor, "y_gapfilled"]
                persist = rr.chain_persistence(anchor_val, N_DAYS)
                y_gf = dft["y_gapfilled"].reindex(target_dates).values

                tree_chains = {}
                for algo, (model, imp) in tree_models.items():
                    tree_chains[algo] = rr.tree_rollout(model, imp, feat_cols, fx_frame, history_init, anchor, n_days=N_DAYS)

                sarimax_chain = sarimax_cache[(yr, tower)]

                ens_df = pd.DataFrame({**tree_chains, "SARIMAX": sarimax_chain})
                ens_unweighted = ens_df.mean(axis=1)
                w = {k: 1.0 / v for k, v in B09_MEAN_MASE.items()}
                wsum = sum(w.values())
                w = {k: v / wsum for k, v in w.items()}
                ens_weighted = sum(ens_df[k] * w[k] for k in ens_df.columns)

                chains = {**tree_chains, "SARIMAX": sarimax_chain,
                          "Ensemble_unweighted": ens_unweighted, "Ensemble_MASEweighted": ens_weighted}

                for name, chain in chains.items():
                    yp = chain.reindex(target_dates).values
                    bm = rr.bin_metrics(y_true, yp, target_dates, anchor, y_persist=persist)
                    bm["model"] = name
                    bm["config"] = cfg_name
                    bm["anchor_year"] = yr
                    bm["tower"] = tower
                    all_rows.append(bm)

                if cfg_name in ("BASE", "BASE+species", "BASE+arable", "BASE+flow", "BASE+mgmt", "BASE+bodyweight"):
                    chain_df = pd.DataFrame({f"{cfg_name}__{name}": c.reindex(target_dates) for name, c in chains.items()})
                    chain_df["y_true"] = y_true
                    chain_df["y_gapfilled"] = y_gf
                    chain_df["persistence"] = persist
                    chain_df["tower"] = tower
                    chain_df["anchor_year"] = yr
                    chain_df["config"] = cfg_name
                    chain_df.index.name = "date"
                    all_chain_rows.append(chain_df.reset_index())

            print(f"  {cfg_name} done ({time.time()-t_cfg:.0f}s)")

        print(f"  Anchor {yr} total ({time.time()-t_anchor:.0f}s)")

    out = pd.concat(all_rows, ignore_index=True)
    out.to_csv(f"{RESULTS}/b16_recursive_rollout_v3_summary.csv", index=False)
    print(f"\n[OK] Saved b16_recursive_rollout_v3_summary.csv ({len(out)} rows)")

    chains_out = pd.concat(all_chain_rows, ignore_index=True)
    chains_out.to_csv(f"{RESULTS}/b16_recursive_rollout_v3_chains.csv", index=False)
    print(f"[OK] Saved b16_recursive_rollout_v3_chains.csv ({len(chains_out)} rows)")


if __name__ == "__main__":
    main()
