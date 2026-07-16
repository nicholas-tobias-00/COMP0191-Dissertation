"""D-7x: gap-filled-context ablation for TabPFN/TabICLv2 -- sibling to `b10_b13_dl_gf_extension.py`
and `b10_b13_tft_gf_extension.py`, same motivation. TabPFN/TabICLv2 are frozen in-context foundation
models with no gradient training -- there is no loss to swap, but there IS a direct analogue: the
historical `hist_target` context series they condition on. Both `rr.tabpfn_forecast()` and
`rr.tabicl_forecast()` docstrings (`recursive_rollout.py`) explicitly document a PRIOR, deliberate
decision to use real y_observed (not y_gapfilled) as this context, "avoiding the diffuse
globally-trained-gap-filler optimism flagged for every other model's training target." This script
empirically tests that prior decision by swapping `hist_target` to `y_gapfilled` (dense, no NaNs) --
per the user's explicit confirmation that context is the in-context analogue of "training data," not
a held-out test set, so it should follow the same convention as the DLinear/LSTM/TFT gf variants.

Evaluation ground truth is untouched (real y_observed, same convention as the original TabPFN/
TabICLv2 scripts). Outputs `TabPFN_gf`/`TabICLv2_gf` columns -- kept alongside (not replacing) the
existing `TabPFN`/`TabICLv2` columns.
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
BINS = ((1, 7), (8, 30), (31, 90), (91, 180), (181, 270), (271, 365))


def run(anchor_years, towers=TOWERS, tag=""):
    dv = pd.read_csv(f"{HOURLY}/forecast_daily_v2.csv", low_memory=False)
    dv["Datetime"] = pd.to_datetime(dv["Datetime"], format="mixed")
    FX_B = [c for c in dv.columns if c.startswith("fx")]
    T = {t: dv[dv.tower == t].set_index("Datetime").sort_index() for t in towers}

    tabpfn_ok = bool(os.environ.get("TABPFN_TOKEN"))
    if not tabpfn_ok:
        print("WARNING: TABPFN_TOKEN not set -- TabPFN_gf will be skipped.")

    all_rows, all_rows_gf, all_chain_rows = [], [], []

    for yr in anchor_years:
        anchor = pd.Timestamp(f"{yr}-12-16")
        target_dates = pd.date_range(anchor + pd.Timedelta(days=1), periods=N_DAYS, freq="D")
        print(f"\n{'='*70}\nAnchor {yr}\n{'='*70}")
        t_anchor = time.time()

        for tower in towers:
            t_tower = time.time()
            dft = T[tower]

            y_true = dft["y_observed"].reindex(target_dates).values  # real, unchanged
            anchor_val = dft.loc[anchor, "y_gapfilled"]
            persist = rr.chain_persistence(anchor_val, N_DAYS)

            y_gf = dft["y_gapfilled"].reindex(target_dates).values
            bin_labels = rr.lead_time_bin(target_dates, anchor)
            real_frac_by_bin = {}
            for lo, hi in BINS:
                lbl = f"{lo}-{hi}"
                bmask = bin_labels == lbl
                real_frac_by_bin[lbl] = float(np.isfinite(y_true[bmask]).mean()) if bmask.sum() > 0 else np.nan

            chain_df = pd.DataFrame(index=target_dates)
            chain_df.index.name = "date"

            hist = dft.loc[:anchor]
            hist_target_gf = hist["y_gapfilled"]  # dense context, the actual ablation
            hist_covariates = hist[FX_B]
            future_covariates = dft.loc[target_dates, FX_B]

            def score(name, chain):
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

                chain_df[name] = chain.reindex(target_dates)

            if tabpfn_ok:
                try:
                    chain = rr.tabpfn_forecast(hist_target_gf, hist_covariates, future_covariates, mode="local")
                    score("TabPFN_gf", chain)
                except Exception as e:
                    print(f"    Tower {tower} anchor {yr} TabPFN_gf SKIPPED: {str(e)[:200]}")

            try:
                chain = rr.tabicl_forecast(hist_target_gf, hist_covariates, future_covariates)
                score("TabICLv2_gf", chain)
            except Exception as e:
                print(f"    Tower {tower} anchor {yr} TabICLv2_gf SKIPPED: {str(e)[:200]}")

            chain_df["y_true"] = y_true
            chain_df["y_gapfilled"] = y_gf
            chain_df["persistence"] = persist
            chain_df["tower"] = tower
            chain_df["anchor_year"] = yr
            all_chain_rows.append(chain_df.reset_index())

            print(f"  Tower {tower} done ({time.time()-t_tower:.0f}s)")

        print(f"  Anchor {yr} total ({time.time()-t_anchor:.0f}s)")

    out = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    out_gf = pd.concat(all_rows_gf, ignore_index=True) if all_rows_gf else pd.DataFrame()
    chains = pd.concat(all_chain_rows, ignore_index=True)

    suffix = f"_{tag}" if tag else ""
    out.to_csv(f"{RESULTS}/b10_b13_foundation_gf_extension_summary{suffix}.csv", index=False)
    out_gf.to_csv(f"{RESULTS}/b10_b13_foundation_gf_extension_summary_vs_gapfilled{suffix}.csv", index=False)
    chains.to_csv(f"{RESULTS}/b10_b13_foundation_gf_extension_chains{suffix}.csv", index=False)
    print(f"\n[OK] Saved b10_b13_foundation_gf_extension_*{suffix}.csv "
          f"(summary={len(out)}, chains={len(chains)} rows)")
    return out, out_gf, chains


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true", help="Tower 4, anchor 2021 only")
    args = p.parse_args()

    if args.smoke:
        run(anchor_years=[2021], towers=[4], tag="smoke")
    else:
        run(anchor_years=[2018, 2019, 2020, 2021, 2022], towers=TOWERS)
