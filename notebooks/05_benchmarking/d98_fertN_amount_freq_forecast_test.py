"""D-98 additive test (Process 2/3): does adding corrected fertN_amount (true kg N/ha,
recency-weighted) + the new fertN_freq (trailing-365d true-N event count) to the standing
forecasting champion (TabPFN+species, F-10/D-67/D-80) help or hurt real-observed-target MASE?

BASE+mgmt (the existing fx_mgmt_fertN_recency/_rate family) already showed a small real gain
(D-98 background check: TabPFN BASE 0.724 -> BASE+mgmt 0.719 MASE_climatology) but lost the
"pick one family" competition to species/bodyweight. This tests the corrected amount+frequency
pair ON TOP OF the actual champion config (BASE+species), not as a standalone family swap.

Fully additive: new script, new config, does not touch b16_foundation_models_v3.py or
forecast_daily_v3.csv. Champion baseline for comparison: MASE=0.715 (climatology, observed target).

Run from project root:  python notebooks/05_benchmarking/d98_fertN_amount_freq_forecast_test.py
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

import models.recursive_rollout as rr

from dotenv import load_dotenv
import os
load_dotenv(os.path.join(ROOT, ".env"))

HOURLY = rf"{ROOT}\data\Hourly"
RESULTS = rf"{ROOT}\results"

TOWERS = [2, 4, 9]
N_DAYS = 365
ANCHOR_YEARS = [2018, 2019, 2020, 2021, 2022]
BINS = ((1, 7), (8, 30), (31, 90), (91, 180), (181, 270), (271, 365))

SPECIES_FX = ["fx_cattle_dens", "fx_sheep_dens", "fx_lamb_dens"]
NEW_FERTN_FX = ["fx_fertN_amount_v2", "fx_fertN_freq_v2"]


def load_fertN_daily():
    fq = pd.read_csv(f"{HOURLY}/fertN_amount_freq_features.csv", low_memory=False)
    fq["Datetime"] = pd.to_datetime(fq["Datetime"], format="mixed")
    fq = fq.set_index("Datetime")
    out = {}
    for t in TOWERS:
        d = fq[[f"fertN_amount_t{t}", f"fertN_freq_t{t}"]].resample("D").mean()
        d.columns = NEW_FERTN_FX
        out[t] = d
    return out


def main():
    dv = pd.read_csv(f"{HOURLY}/forecast_daily_v3.csv", low_memory=False)
    dv["Datetime"] = pd.to_datetime(dv["Datetime"], format="mixed")
    fx_all = [c for c in dv.columns if c.startswith("fx")]
    ALL_NEW_F10 = sorted({c for fam in [SPECIES_FX, ["fx_is_arable"],
                                          [c for c in fx_all if c.startswith("fx_flow")],
                                          [c for c in fx_all if c.startswith("fx_mgmt")],
                                          ["fx_total_liveweight_dens"]] for c in fam})
    BASE_FX = [c for c in fx_all if c not in ALL_NEW_F10]
    champion_cfg = BASE_FX + SPECIES_FX
    test_cfg = champion_cfg + NEW_FERTN_FX

    fertN_daily = load_fertN_daily()
    T = {}
    for t in TOWERS:
        dft = dv[dv.tower == t].set_index("Datetime").sort_index()
        dft = dft.join(fertN_daily[t], how="left")
        dft[NEW_FERTN_FX] = dft[NEW_FERTN_FX].fillna(0.0)
        T[t] = dft

    print(f"[OK] champion config: {len(champion_cfg)} cols; test config (+fertN amount/freq): "
          f"{len(test_cfg)} cols")

    all_rows = []
    t0 = time.time()
    for yr in ANCHOR_YEARS:
        anchor = pd.Timestamp(f"{yr}-12-16")
        target_dates = pd.date_range(anchor + pd.Timedelta(days=1), periods=N_DAYS, freq="D")
        for tower in TOWERS:
            dft = T[tower]
            hist = dft.loc[:anchor]
            hist_target = hist["y_observed"]
            y_true = dft["y_observed"].reindex(target_dates).values
            anchor_val = dft.loc[anchor, "y_gapfilled"]
            persist = rr.chain_persistence(anchor_val, N_DAYS)

            for cfg_name, cols in [("champion_BASE+species", champion_cfg),
                                    ("test_BASE+species+fertN_v2", test_cfg)]:
                hist_cov = hist[cols]
                future_cov = dft.loc[target_dates, cols]
                try:
                    chain = rr.tabpfn_forecast(hist_target, hist_cov, future_cov, mode="local")
                    yp = chain.reindex(target_dates).values
                    bm = rr.bin_metrics(y_true, yp, target_dates, anchor, y_persist=persist)
                    bm["config"] = cfg_name; bm["anchor_year"] = yr; bm["tower"] = tower
                    all_rows.append(bm)
                except Exception as e:
                    print(f"  T{tower} {yr} {cfg_name} SKIPPED: {str(e)[:150]}")
        print(f"  anchor {yr} done ({time.time()-t0:.0f}s elapsed)")

    out = pd.concat(all_rows, ignore_index=True)
    out.to_csv(f"{RESULTS}/d98_forecast_fertN_amount_freq_test.csv", index=False)
    print(f"[OK] Saved d98_forecast_fertN_amount_freq_test.csv ({len(out)} rows)")


if __name__ == "__main__":
    main()
