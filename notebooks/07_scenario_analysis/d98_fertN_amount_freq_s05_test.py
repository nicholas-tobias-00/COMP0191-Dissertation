"""D-98 additive test (Process 3/3): does adding corrected fertN_amount (true kg N/ha,
recency-weighted) + the new fertN_freq (trailing-365d true-N event count) to S-05's own
architecture (TabICLv2 + FX_A_SPECIES, S-03 Variant A + F-10 species split) help or hurt real
historical accuracy? Added as CONSTANT covariates (not a swept scenario lever like S-05's own
D-86/D-97 fertilizer axis) -- same "does this feature help the model" question the general
forecasting test (d98_fertN_amount_freq_forecast_test.py) asks, scoped to S-05's own narrower
Variant-A feature space instead of the general BASE_FX.

Evaluated on real historical anchors (S-03/S-05's own convention -- TabICLv2, all 3 towers, 5
anchors), not a future scenario sweep -- this answers "does the model get more accurate," a
different question from "what does the scenario projection show," which S-05's fertilizer axis
already answers separately.

Fully additive: new script, new config, does not touch S-05's production driver scripts.

Run from project root:  python notebooks/07_scenario_analysis/d98_fertN_amount_freq_s05_test.py
"""
import sys
import time
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

ROOT = r"c:\Users\Nicholas\Documents\COMP0191 MSc Artificial Intelligence for Sustainable Development Project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\src")
sys.path.insert(0, ROOT + r"\src\features")

import models.recursive_rollout as rr
from build_transient_scenario_drivers_species import FX_A_SPECIES

HOURLY = rf"{ROOT}\data\Hourly"
RESULTS = rf"{ROOT}\results"

TOWERS = [2, 4, 9]
N_DAYS = 365
ANCHOR_YEARS = [2018, 2019, 2020, 2021, 2022]
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

    fertN_daily = load_fertN_daily()
    T = {}
    for t in TOWERS:
        dft = dv[dv.tower == t].set_index("Datetime").sort_index()
        dft = dft.join(fertN_daily[t], how="left")
        dft[NEW_FERTN_FX] = dft[NEW_FERTN_FX].fillna(0.0)
        T[t] = dft

    champion_cfg = FX_A_SPECIES
    test_cfg = FX_A_SPECIES + NEW_FERTN_FX
    print(f"[OK] champion (FX_A_SPECIES): {len(champion_cfg)} cols; "
          f"test (+fertN amount/freq): {len(test_cfg)} cols")

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

            for cfg_name, cols in [("champion_FX_A_SPECIES", champion_cfg),
                                    ("test_FX_A_SPECIES+fertN_v2", test_cfg)]:
                hist_cov = hist[cols]
                future_cov = dft.loc[target_dates, cols]
                try:
                    chain = rr.tabicl_forecast(hist_target, hist_cov, future_cov)
                    yp = chain.reindex(target_dates).values
                    bm = rr.bin_metrics(y_true, yp, target_dates, anchor, y_persist=persist)
                    bm["config"] = cfg_name; bm["anchor_year"] = yr; bm["tower"] = tower
                    all_rows.append(bm)
                except Exception as e:
                    print(f"  T{tower} {yr} {cfg_name} SKIPPED: {str(e)[:150]}")
        print(f"  anchor {yr} done ({time.time()-t0:.0f}s elapsed)")

    out = pd.concat(all_rows, ignore_index=True)
    out.to_csv(f"{RESULTS}/d98_s05_fertN_amount_freq_test.csv", index=False)
    print(f"[OK] Saved d98_s05_fertN_amount_freq_test.csv ({len(out)} rows)")


if __name__ == "__main__":
    main()
