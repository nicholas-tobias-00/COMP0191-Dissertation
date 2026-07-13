"""F-10 (D-67) Stage 2b follow-up: BASE+ALL (all 5 new feature families stacked at once) on the
recursive-rollout track -- the one config missing from b16_recursive_rollout_v3.py's original run
(an oversight, not a deliberate omission -- Stage 1's point-forecast check DID include BASE+ALL and
found it clearly worse than BASE; this checks whether that transfers to the rollout).

Reuses the already-fitted SARIMAX chains from results/b16_recursive_rollout_v3_chains.csv (its
EXOG_B set is unaffected by any new family, so it's identical across every config -- no need to
refit). Only RF/XGB/LightGBM are refit, on the full 18-new-column feature set, per anchor.

Run from project root:  python notebooks/05_benchmarking/b16_recursive_rollout_v3_all.py
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

import models.recursive_rollout as rr

HOURLY = rf"{ROOT}\data\Hourly"
RESULTS = rf"{ROOT}\results"

TOWERS = [2, 4, 9]
N_DAYS = 365
ANCHOR_YEARS = [2018, 2019, 2020, 2021, 2022]

DUM = ["is_t2", "is_t4", "is_t9"]
AR_COLS = ["ar_ch4_dlag1", "ar_ch4_dlag2", "ar_ch4_dlag3", "ar_ch4_dlag7", "ar_ch4_dlag14", "ar_ch4_drm7"]
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
    dv = pd.read_csv(f"{HOURLY}/forecast_daily_v3.csv", low_memory=False)
    dv["Datetime"] = pd.to_datetime(dv["Datetime"], format="mixed")
    fx_all = [c for c in dv.columns if c.startswith("fx")]
    BASE_FX = [c for c in fx_all if c not in ALL_NEW]
    T = {t: dv[dv.tower == t].set_index("Datetime").sort_index() for t in TOWERS}
    fx_cols = BASE_FX + ALL_NEW
    feat_cols = AR_COLS + fx_cols + ["ar_fc_dlag1"] + DUM
    cfg_name = "BASE+ALL"

    # ---- reuse already-fitted SARIMAX chains (identical across every config) ----
    # The chains CSV stacks all 6 prior configs as separate rows sharing the same
    # (tower, anchor_year, date) keys -- filter to rows where BASE__SARIMAX is populated
    # (i.e. the BASE config's own rows) before indexing, or the (tower, anchor_year, date)
    # key would have 6x duplicates (one per config) and reindex() would fail.
    prev_chains = pd.read_csv(f"{RESULTS}/b16_recursive_rollout_v3_chains.csv", parse_dates=["date"])
    prev_chains = prev_chains[prev_chains["BASE__SARIMAX"].notna()]
    sarimax_lookup = prev_chains.set_index(["tower", "anchor_year", "date"])["BASE__SARIMAX"]

    all_rows = []
    all_chain_rows = []

    for yr in ANCHOR_YEARS:
        anchor = pd.Timestamp(f"{yr}-12-16")
        target_dates = pd.date_range(anchor + pd.Timedelta(days=1), periods=N_DAYS, freq="D")
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

            sarimax_chain = sarimax_lookup.loc[tower, yr].reindex(target_dates)

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

            chain_df = pd.DataFrame({f"{cfg_name}__{name}": c.reindex(target_dates) for name, c in chains.items()})
            chain_df["y_true"] = y_true
            chain_df["y_gapfilled"] = y_gf
            chain_df["persistence"] = persist
            chain_df["tower"] = tower
            chain_df["anchor_year"] = yr
            chain_df["config"] = cfg_name
            chain_df.index.name = "date"
            all_chain_rows.append(chain_df.reset_index())

        print(f"  Anchor {yr} done ({time.time()-t_anchor:.0f}s)")

    out = pd.concat(all_rows, ignore_index=True)
    prev_summary = pd.read_csv(f"{RESULTS}/b16_recursive_rollout_v3_summary.csv")
    combined = pd.concat([prev_summary, out], ignore_index=True)
    combined.to_csv(f"{RESULTS}/b16_recursive_rollout_v3_summary.csv", index=False)
    print(f"\n[OK] Appended BASE+ALL to b16_recursive_rollout_v3_summary.csv (now {len(combined)} rows)")

    out.to_csv(f"{RESULTS}/b16_recursive_rollout_v3_all_summary.csv", index=False)
    chains_out = pd.concat(all_chain_rows, ignore_index=True)
    chains_out.to_csv(f"{RESULTS}/b16_recursive_rollout_v3_all_chains.csv", index=False)
    print("[OK] Saved b16_recursive_rollout_v3_all_summary.csv and _all_chains.csv")


if __name__ == "__main__":
    main()
