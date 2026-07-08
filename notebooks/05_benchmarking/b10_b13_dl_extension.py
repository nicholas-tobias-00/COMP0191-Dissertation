"""Closes a model-roster gap in the D-65 B-10+B-13 rerun: DLinear and LSTM were tested in B-09
(D-53, mean R2=-4.75 for DLinear -- the worst/most unstable model in the whole B-09-B15 sequence,
excluded from B-10's ensemble on that basis) and had their rollout chains extended to all 3 towers
x 5 anchors for visualization only (`results/figures/b10_chains/T*_anchor*_{DLinear,LSTM}.png`,
B-13's Part A) -- but that chain-plot extension never saved a bin_metrics summary, so there has
never been a full 3-tower x 5-anchor evaluation table for these two models with the D-65 metric
set (RMSE/WAPE/Correlation added to R2/MAE/MASE). This script closes that gap.

Exact recipe reconstructed from the committed `B09_recursive_rollout.ipynb` (Section 4, "DL models
(DLinear, LSTM)") -- NOT B-10's H=1-native-retrain variant (D-54: that retrain helped LSTM but hurt
DLinear further, and was never the model actually used for the b10_chains figures per the B-13 Part
A plan, which explicitly said "reuse forecasting_dl.build_model/train_model/Scaler +
recursive_rollout.dl_rollout exactly as B-09 already does (no new training logic)"):
- Track B (`fdl.TRACKS["B"]`: L=28, H=14).
- Pooled training (Towers 2+4+9 windows via `fdl.build_windows`/`_subset`/`_cat`, cutoff=anchor,
  NO validation carve-out -- this is deliberately NOT the same `train_tft`/`val_tft` split the
  b10_b13_rerun_multi_anchor.py script builds for TFT, which reserves the last 90 days for early
  stopping; DLinear/LSTM's original B-09 recipe uses the full pooled set <= anchor with no split).
- `fdl.train_model(model, train, device, epochs=30, ch4_mu=mu, ch4_sd=sdy, seed=0)` -- plain,
  unregularized (no weight_decay/val_data/patience, unlike TFT's D-45 recipe).
- Rollout via `rr.dl_rollout` (already generic over any 3-arg (enc,dec,static) model -- the same
  function this project's TFT block already reuses, zero new rollout code needed).
- y_true source: `fdl.tower_series(...)["Y"]` (matches TFT's own y_true convention -- a different
  daily-resampling pipeline than `forecast_daily_v2.csv`'s `y_observed`, count>=6h/day threshold --
  preserved exactly as B-09 used it, not the tree-track source).
- history_init: `fdl.tower_series(...)["ch4"]` sliced to <= anchor (matches TFT's history_init_tft).

Pooling structure: fit ONCE per anchor (pooled T2+T4+T9), rolled out separately per tower --
identical to how RF/XGB/LightGBM/TFT are already handled in b10_b13_rerun_multi_anchor.py.

Verification: Tower 4's R2/MAE/MASE are checked against the already-published
`results/b09_multi_anchor_summary.csv` (the only prior multi-anchor DLinear/LSTM record, itself
Tower-4-only) across all 5 anchors before trusting the Tower 2/9 extension.
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
import models.forecasting_dl as fdl

HOURLY = rf"{ROOT}\data\Hourly"
RESULTS = rf"{ROOT}\results"

TOWERS = [2, 4, 9]
N_DAYS = 365
ANCHOR_YEARS = [2018, 2019, 2020, 2021, 2022]
TRACK = "B"


def main():
    dv = pd.read_csv(f"{HOURLY}/forecast_daily_v2.csv", low_memory=False)
    dv["Datetime"] = pd.to_datetime(dv["Datetime"], format="mixed")
    T = {t: dv[dv.tower == t].set_index("Datetime").sort_index() for t in TOWERS}

    fdl_m = fdl.load_matrix(f"{HOURLY}/forecast_features_v2.csv")
    device = fdl.get_device()
    cfg = fdl.TRACKS[TRACK]

    all_rows = []
    all_rows_gf = []  # secondary metric vs. y_gapfilled (mirrors b10_b13_rerun_multi_anchor.py)
    all_chain_rows = []  # raw daily prediction chains, for export

    for yr in ANCHOR_YEARS:
        anchor = pd.Timestamp(f"{yr}-12-16")
        target_dates = pd.date_range(anchor + pd.Timedelta(days=1), periods=N_DAYS, freq="D")
        print(f"\n{'='*70}\nAnchor {yr}\n{'='*70}")
        t_anchor = time.time()

        # ---- Pooled DL training (T2+T4+T9), fit ONCE per anchor, no val split (B-09's recipe) ----
        cutoff = anchor + pd.Timedelta(hours=23, minutes=59)
        Wd = fdl.build_windows(fdl_m, TRACK)
        tr_parts = [fdl._subset(Wd[t], pd.DatetimeIndex(Wd[t]["ttime"][:, -1]) <= cutoff) for t in TOWERS]
        dl_train = fdl._cat(tr_parts)
        se, sd = fdl.Scaler().fit(dl_train["enc"]), fdl.Scaler().fit(dl_train["dec"])
        yv = dl_train["y"][np.isfinite(dl_train["y"])]
        mu, sdy = float(yv.mean()), float(yv.std() + 1e-6)
        dl_train["enc"], dl_train["dec"] = se.tf(dl_train["enc"]), sd.tf(dl_train["dec"])
        n_enc, n_dec = dl_train["enc"].shape[-1], dl_train["dec"].shape[-1]

        dl_models = {}
        for name in ["DLinear", "LSTM"]:
            model = fdl.build_model(name, cfg["L"], cfg["H"], n_enc, n_dec, 3)
            fdl.train_model(model, dl_train, device, epochs=30, ch4_mu=mu, ch4_sd=sdy, seed=0)
            dl_models[name] = model
        print(f"  Pooled DLinear/LSTM fit ({time.time()-t_anchor:.0f}s)")

        for tower in TOWERS:
            t_tower = time.time()
            dft = T[tower]
            ser_t = fdl.tower_series(fdl_m, tower, TRACK)
            dates_full, enc_ex_full, dec_ex_full = ser_t["idx"], ser_t["enc_ex"], ser_t["dec_ex"]
            anchor_idx = dates_full.get_loc(anchor)
            dl_history_init = ser_t["ch4"][:anchor_idx + 1]

            y_true_full = pd.Series(ser_t["Y"], index=dates_full)
            y_true = y_true_full.reindex(target_dates).values

            anchor_val = dft.loc[anchor, "y_gapfilled"]
            persist = rr.chain_persistence(anchor_val, N_DAYS)

            # ---- Secondary metric setup: y_gapfilled target + per-bin real-data coverage
            # (mirrors b10_b13_rerun_multi_anchor.py's own addendum -- same caveats apply here) ----
            y_gf = dft["y_gapfilled"].reindex(target_dates).values
            bin_labels = rr.lead_time_bin(target_dates, anchor)
            real_frac_by_bin = {}
            for lo, hi in ((1, 7), (8, 30), (31, 90), (91, 180), (181, 270), (271, 365)):
                lbl = f"{lo}-{hi}"
                bmask = bin_labels == lbl
                real_frac_by_bin[lbl] = float(np.isfinite(y_true[bmask]).mean()) if bmask.sum() > 0 else np.nan

            dl_out_chains = {}
            for name, model in dl_models.items():
                chain = rr.dl_rollout(model, se, sd, mu, sdy, device, fdl.TOW[tower],
                                       enc_ex_full, dec_ex_full, dates_full, dl_history_init, anchor,
                                       L=cfg["L"], H=cfg["H"], n_days=N_DAYS)
                dl_out_chains[name] = chain
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

            # ---- Raw daily chains, for export (never persisted before this addendum) ----
            chain_df = pd.DataFrame({name: c.reindex(target_dates) for name, c in dl_out_chains.items()})
            chain_df["y_true_dl"] = y_true
            chain_df["y_gapfilled"] = y_gf
            chain_df["persistence"] = persist
            chain_df["tower"] = tower
            chain_df["anchor_year"] = yr
            chain_df.index.name = "date"
            all_chain_rows.append(chain_df.reset_index())

            print(f"  Tower {tower} done ({time.time()-t_tower:.0f}s)")

        print(f"  Anchor {yr} total ({time.time()-t_anchor:.0f}s)")

    out = pd.concat(all_rows, ignore_index=True)
    out.to_csv(f"{RESULTS}/b10_b13_dl_extension_summary.csv", index=False)
    print(f"\n[OK] Saved b10_b13_dl_extension_summary.csv ({len(out)} rows)")

    out_gf = pd.concat(all_rows_gf, ignore_index=True)
    out_gf.to_csv(f"{RESULTS}/b10_b13_dl_extension_summary_vs_gapfilled.csv", index=False)
    print(f"[OK] Saved b10_b13_dl_extension_summary_vs_gapfilled.csv ({len(out_gf)} rows)")

    chains = pd.concat(all_chain_rows, ignore_index=True)
    chains.to_csv(f"{RESULTS}/b10_b13_dl_extension_chains.csv", index=False)
    print(f"[OK] Saved b10_b13_dl_extension_chains.csv ({len(chains)} rows)")


if __name__ == "__main__":
    main()
