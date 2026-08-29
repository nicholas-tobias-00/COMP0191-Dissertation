"""D-79: same as `b16_foundation_models_v3_gf.py` (gap-filled-context TabPFN/TabICLv2, F-10
feature families), but sourced from the TabICL-solo gap-filled daily table
(`forecast_daily_v3_tabicl.csv`) instead of the RFm champion's. This is the test that actually
matters for whether the improved upstream gap-filling helps forecasting: here, `y_gapfilled` feeds
directly into the model as context, unlike the original (y_observed-context) comparison where
swapping gap-fill source could not possibly change the model's predictions (confirmed empirically:
identical MAE/R2 across both gap-fill sources there).

Combines two previously-separate lines of work that were never tested together: F-10/D-67's
7-config feature-family sweep (`b16_foundation_models_v3.py`) and D-72's finding that TabPFN/
TabICLv2 genuinely improve when given `y_gapfilled` (not `y_observed`) as their historical
context. D-72 only tested the gap-filled-context choice on the plain BASE feature set
(`forecast_daily_v2.csv`); this combines it with the F-10 feature families -- the only code change
from `b16_foundation_models_v3.py` is `hist_target = hist["y_gapfilled"]` instead of
`hist["y_observed"]` (mirrors `b10_b13_foundation_gf_extension.py`'s exact ablation).

Evaluation ground truth is untouched (real y_observed) -- only the historical context changes.

Run from project root:  python notebooks/05_benchmarking/b16_foundation_models_v3_tabicl_gf.py
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

import models.recursive_rollout as rr

from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))

HOURLY = rf"{ROOT}\data\Hourly"
RESULTS = rf"{ROOT}\results"

TOWERS = [2, 4, 9]
N_DAYS = 365
ANCHOR_YEARS = [2018, 2019, 2020, 2021, 2022]

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


def main():
    dv = pd.read_csv(f"{HOURLY}/forecast_daily_v3_tabicl.csv", low_memory=False)
    dv["Datetime"] = pd.to_datetime(dv["Datetime"], format="mixed")
    fx_all = [c for c in dv.columns if c.startswith("fx")]
    BASE_FX = [c for c in fx_all if c not in ALL_NEW]
    T = {t: dv[dv.tower == t].set_index("Datetime").sort_index() for t in TOWERS}

    configs = {"BASE": BASE_FX}
    for fam, cols in FAMILIES.items():
        configs[f"BASE+{fam}"] = BASE_FX + cols
    configs["BASE+ALL"] = BASE_FX + ALL_NEW

    tabpfn_ok = bool(os.environ.get("TABPFN_TOKEN"))
    if not tabpfn_ok:
        print("WARNING: TABPFN_TOKEN not set -- TabPFN will be skipped.")

    all_rows = []
    t0 = time.time()

    for yr in ANCHOR_YEARS:
        anchor = pd.Timestamp(f"{yr}-12-16")
        target_dates = pd.date_range(anchor + pd.Timedelta(days=1), periods=N_DAYS, freq="D")

        for tower in TOWERS:
            dft = T[tower]
            hist = dft.loc[:anchor]
            hist_target = hist["y_gapfilled"]   # <-- the only change vs. b16_foundation_models_v3.py
            y_true = dft["y_observed"].reindex(target_dates).values
            y_gf = dft["y_gapfilled"].reindex(target_dates).values
            anchor_val = dft.loc[anchor, "y_gapfilled"]
            persist = rr.chain_persistence(anchor_val, N_DAYS)

            for cfg_name, fx_cols in configs.items():
                hist_covariates = hist[fx_cols]
                future_covariates = dft.loc[target_dates, fx_cols]

                if tabpfn_ok:
                    try:
                        chain = rr.tabpfn_forecast(hist_target, hist_covariates, future_covariates, mode="local")
                        yp = chain.reindex(target_dates).values
                        bm = rr.bin_metrics(y_true, yp, target_dates, anchor, y_persist=persist)
                        bm["target"] = "observed"; bm["model"] = "TabPFN_gf"
                        bm["config"] = cfg_name
                        bm["anchor_year"] = yr
                        bm["tower"] = tower
                        all_rows.append(bm)
                        bm_gf = rr.bin_metrics(y_gf, yp, target_dates, anchor, y_persist=persist)
                        bm_gf["target"] = "gapfilled"; bm_gf["model"] = "TabPFN_gf"
                        bm_gf["config"] = cfg_name
                        bm_gf["anchor_year"] = yr
                        bm_gf["tower"] = tower
                        all_rows.append(bm_gf)
                    except Exception as e:
                        print(f"    T{tower} {yr} {cfg_name} TabPFN_gf SKIPPED: {str(e)[:150]}")

                try:
                    chain = rr.tabicl_forecast(hist_target, hist_covariates, future_covariates)
                    yp = chain.reindex(target_dates).values
                    bm = rr.bin_metrics(y_true, yp, target_dates, anchor, y_persist=persist)
                    bm["target"] = "observed"; bm["model"] = "TabICLv2_gf"
                    bm["config"] = cfg_name
                    bm["anchor_year"] = yr
                    bm["tower"] = tower
                    all_rows.append(bm)
                    bm_gf = rr.bin_metrics(y_gf, yp, target_dates, anchor, y_persist=persist)
                    bm_gf["target"] = "gapfilled"; bm_gf["model"] = "TabICLv2_gf"
                    bm_gf["config"] = cfg_name
                    bm_gf["anchor_year"] = yr
                    bm_gf["tower"] = tower
                    all_rows.append(bm_gf)
                except Exception as e:
                    print(f"    T{tower} {yr} {cfg_name} TabICLv2_gf SKIPPED: {str(e)[:150]}")

            print(f"  Tower {tower} anchor {yr} done ({time.time()-t0:.0f}s elapsed)")

    out = pd.concat(all_rows, ignore_index=True)
    out.to_csv(f"{RESULTS}/b16_foundation_models_v3_tabicl_gf_summary.csv", index=False)
    print(f"\n[OK] Saved b16_foundation_models_v3_tabicl_gf_summary.csv ({len(out)} rows), total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
