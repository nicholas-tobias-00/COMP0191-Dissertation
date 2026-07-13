"""D-66: adds TabICLv2 (`tabicl` package, `TabICLForecaster`) to the B-10/B-13 recursive-rollout
sequence -- a tabular foundation model released Feb 2026 (ICML 2026), the first version with
regression support. Its own docs describe it as "heavily inspired by TabPFN-TS", so this mirrors
`rr.tabpfn_forecast()`'s exact block structure from `b10_b13_rerun_multi_anchor.py` (per-tower,
per-anchor, never pooled -- the simple predict_df API has no static-covariate/pooling support,
same limitation as TabPFN) and `b10_b13_dl_extension.py`'s minimal sibling-script output pattern
(no edits to the parent script or any historical B-13 notebook).

y_true source: `y_observed` directly from `forecast_daily_v2.csv` (same convention as TabPFN, not
the DL-track's `fdl.tower_series(...)["Y"]` source -- TabICLv2 uses the daily fx_ table directly,
just like TabPFN and the tree models, not the hourly-resampled DL feature matrix).

Covariates: FX_B (all `fx`-prefixed daily columns, forecast_daily_v2.csv) -- same set TabPFN uses,
NOT the narrower 8-column EXOG_B SARIMAX uses.

API contract verified empirically (not assumed from docs) via a smoke test before writing this
script: TabICLForecaster.predict_df(context_df, future_df=...) takes covariates as plain extra
columns on both context_df/future_df, identical convention to tabpfn_forecast()'s own
context_df/future_df construction -- see `rr.tabicl_forecast()`'s docstring for the full contract
and the MultiIndex/string-timestamp/always-present-default-quantiles quirks found along the way.
Local-only inference confirmed (downloads a HuggingFace checkpoint once, cached thereafter, no
token/API key needed, unlike TabPFN's TABPFN_TOKEN).
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

HOURLY = rf"{ROOT}\data\Hourly"
RESULTS = rf"{ROOT}\results"

TOWERS = [2, 4, 9]
N_DAYS = 365
ANCHOR_YEARS = [2018, 2019, 2020, 2021, 2022]
BINS = ((1, 7), (8, 30), (31, 90), (91, 180), (181, 270), (271, 365))


def main():
    dv = pd.read_csv(f"{HOURLY}/forecast_daily_v2.csv", low_memory=False)
    dv["Datetime"] = pd.to_datetime(dv["Datetime"], format="mixed")
    FX_B = [c for c in dv.columns if c.startswith("fx")]
    T = {t: dv[dv.tower == t].set_index("Datetime").sort_index() for t in TOWERS}

    all_rows = []
    all_rows_gf = []
    all_chain_rows = []

    for yr in ANCHOR_YEARS:
        anchor = pd.Timestamp(f"{yr}-12-16")
        target_dates = pd.date_range(anchor + pd.Timedelta(days=1), periods=N_DAYS, freq="D")
        print(f"\n{'='*70}\nAnchor {yr}\n{'='*70}")
        t_anchor = time.time()

        for tower in TOWERS:
            t_tower = time.time()
            dft = T[tower]

            y_true = dft["y_observed"].reindex(target_dates).values
            anchor_val = dft.loc[anchor, "y_gapfilled"]
            persist = rr.chain_persistence(anchor_val, N_DAYS)

            # ---- Secondary metric setup: y_gapfilled target + per-bin real-data coverage
            # (mirrors b10_b13_rerun_multi_anchor.py's own addendum) ----
            y_gf = dft["y_gapfilled"].reindex(target_dates).values
            bin_labels = rr.lead_time_bin(target_dates, anchor)
            real_frac_by_bin = {}
            for lo, hi in BINS:
                lbl = f"{lo}-{hi}"
                bmask = bin_labels == lbl
                real_frac_by_bin[lbl] = float(np.isfinite(y_true[bmask]).mean()) if bmask.sum() > 0 else np.nan

            try:
                hist = dft.loc[:anchor]
                hist_target = hist["y_observed"]
                hist_covariates = hist[FX_B]
                future_covariates = dft.loc[target_dates, FX_B]
                chain = rr.tabicl_forecast(hist_target, hist_covariates, future_covariates)
                yp = chain.reindex(target_dates).values

                bm = rr.bin_metrics(y_true, yp, target_dates, anchor, y_persist=persist)
                bm["model"] = "TabICLv2"
                bm["anchor_year"] = yr
                bm["tower"] = tower
                all_rows.append(bm)

                bm_gf = rr.bin_metrics(y_gf, yp, target_dates, anchor, y_persist=persist)
                bm_gf["model"] = "TabICLv2"
                bm_gf["anchor_year"] = yr
                bm_gf["tower"] = tower
                bm_gf["real_frac"] = bm_gf["bin"].map(real_frac_by_bin)
                all_rows_gf.append(bm_gf)

                chain_df = pd.DataFrame({"TabICLv2": chain.reindex(target_dates)})
                chain_df["y_true"] = y_true
                chain_df["y_gapfilled"] = y_gf
                chain_df["persistence"] = persist
                chain_df["tower"] = tower
                chain_df["anchor_year"] = yr
                chain_df.index.name = "date"
                all_chain_rows.append(chain_df.reset_index())
            except Exception as e:
                print(f"    Tower {tower} anchor {yr} TabICLv2 SKIPPED: {str(e)[:200]}")

            print(f"  Tower {tower} done ({time.time()-t_tower:.0f}s)")

        print(f"  Anchor {yr} total ({time.time()-t_anchor:.0f}s)")

    out = pd.concat(all_rows, ignore_index=True)
    out.to_csv(f"{RESULTS}/b10_b13_tabicl_extension_summary.csv", index=False)
    print(f"\n[OK] Saved b10_b13_tabicl_extension_summary.csv ({len(out)} rows)")

    out_gf = pd.concat(all_rows_gf, ignore_index=True)
    out_gf.to_csv(f"{RESULTS}/b10_b13_tabicl_extension_summary_vs_gapfilled.csv", index=False)
    print(f"[OK] Saved b10_b13_tabicl_extension_summary_vs_gapfilled.csv ({len(out_gf)} rows)")

    chains = pd.concat(all_chain_rows, ignore_index=True)
    chains.to_csv(f"{RESULTS}/b10_b13_tabicl_extension_chains.csv", index=False)
    print(f"[OK] Saved b10_b13_tabicl_extension_chains.csv ({len(chains)} rows)")


if __name__ == "__main__":
    main()
