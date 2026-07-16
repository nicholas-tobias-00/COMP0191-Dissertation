"""D-7x: gap-filled-target ablation for TFT -- sibling to `b10_b13_dl_gf_extension.py` (DLinear/
LSTM), same motivation (tests the D-36/D-37 tree/SARIMAX "train on y_gapfilled, evaluate on
y_observed" convention against the DL family). Identical recipe to `b10_b13_rerun_multi_anchor.py`'s
TFT block (pooled T2+T4+T9 fit per anchor, VAL_DAYS-reserved validation split, weight_decay=1e-3,
patience=5) -- the ONLY change is building BOTH `train_tft` and `val_tft` windows with
`y_source="gapfilled"` instead of the default `y_source="observed"`, per the user's explicit
confirmation that TFT's validation split should stay consistent with its training target (both are
part of "fitting," not final evaluation). Evaluation ground truth
(`fdl.tower_series(...)["Y"]` = real y_observed, the existing `y_true_tft` convention) is untouched.

Outputs a `TFT_gf` column -- kept alongside (not replacing) the existing `TFT` column from
`b10_b13_rerun_multi_anchor.py`.
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
import models.forecasting_dl as fdl

HOURLY = rf"{ROOT}\data\Hourly"
RESULTS = rf"{ROOT}\results"

TOWERS = [2, 4, 9]
N_DAYS = 365
VAL_DAYS = 90
TRACK = "B"


def run(anchor_years, towers=TOWERS, tag=""):
    dv = pd.read_csv(f"{HOURLY}/forecast_daily_v2.csv", low_memory=False)
    dv["Datetime"] = pd.to_datetime(dv["Datetime"], format="mixed")
    T = {t: dv[dv.tower == t].set_index("Datetime").sort_index() for t in towers}

    fdl_m = fdl.load_matrix(f"{HOURLY}/forecast_features_v2.csv")
    device = fdl.get_device()
    tft_cfg = fdl.TRACKS[TRACK]

    all_rows, all_rows_gf, all_chain_rows = [], [], []

    for yr in anchor_years:
        anchor = pd.Timestamp(f"{yr}-12-16")
        target_dates = pd.date_range(anchor + pd.Timedelta(days=1), periods=N_DAYS, freq="D")
        print(f"\n{'='*70}\nAnchor {yr}\n{'='*70}")
        t_anchor = time.time()

        # ---- TFT, fit ONCE per anchor on pooled data, gap-filled train+val target ----
        tft_model = None
        try:
            cutoff = anchor + pd.Timedelta(hours=23, minutes=59)
            val_start = anchor - pd.Timedelta(days=VAL_DAYS)
            Wd = {t: fdl.make_windows(fdl.tower_series(fdl_m, t, TRACK), tft_cfg["L"], tft_cfg["H"],
                                       tft_cfg["stride"], y_source="gapfilled") for t in TOWERS}
            train_parts, val_parts = [], []
            for t in TOWERS:
                ttime = pd.DatetimeIndex(Wd[t]["ttime"][:, -1])
                train_parts.append(fdl._subset(Wd[t], ttime <= val_start))
                val_parts.append(fdl._subset(Wd[t], (ttime > val_start) & (ttime <= cutoff)))
            train_tft = fdl._cat(train_parts)
            val_tft = fdl._cat(val_parts)
            se_t, sd_t = fdl.Scaler().fit(train_tft["enc"]), fdl.Scaler().fit(train_tft["dec"])
            yv = train_tft["y"][np.isfinite(train_tft["y"])]
            mu_t, sdy_t = float(yv.mean()), float(yv.std() + 1e-6)
            train_tft["enc"], train_tft["dec"] = se_t.tf(train_tft["enc"]), sd_t.tf(train_tft["dec"])
            val_tft["enc"], val_tft["dec"] = se_t.tf(val_tft["enc"]), sd_t.tf(val_tft["dec"])
            n_enc, n_dec = train_tft["enc"].shape[-1], train_tft["dec"].shape[-1]
            print(f"  Train/val target finite frac (gap-filled): "
                  f"{np.isfinite(train_tft['y']).mean():.3f} / {np.isfinite(val_tft['y']).mean():.3f}")

            tft_model = fdl.build_model("TFT", tft_cfg["L"], tft_cfg["H"], n_enc, n_dec, 3)
            fdl.train_model(tft_model, train_tft, device, epochs=30, ch4_mu=mu_t, ch4_sd=sdy_t, seed=0,
                             weight_decay=1e-3, val_data=val_tft, patience=5)
            print(f"  TFT (pooled) fit ({time.time()-t_anchor:.0f}s total)")
        except Exception as e:
            print(f"  TFT FIT SKIPPED: {str(e)[:200]}")

        for tower in towers:
            t_tower = time.time()
            dft = T[tower]

            y_true = dft["y_observed"].reindex(target_dates).values
            anchor_val = dft.loc[anchor, "y_gapfilled"]
            persist = rr.chain_persistence(anchor_val, N_DAYS)

            y_gf = dft["y_gapfilled"].reindex(target_dates).values
            bin_labels = rr.lead_time_bin(target_dates, anchor)
            real_frac_by_bin = {}
            for lo, hi in ((1, 7), (8, 30), (31, 90), (91, 180), (181, 270), (271, 365)):
                lbl = f"{lo}-{hi}"
                bmask = bin_labels == lbl
                real_frac_by_bin[lbl] = float(np.isfinite(y_true[bmask]).mean()) if bmask.sum() > 0 else np.nan

            chain_df = pd.DataFrame(index=target_dates)
            chain_df.index.name = "date"

            if tft_model is not None:
                try:
                    ser_t = fdl.tower_series(fdl_m, tower, TRACK)
                    dates_full, enc_ex_full, dec_ex_full = ser_t["idx"], ser_t["enc_ex"], ser_t["dec_ex"]
                    anchor_idx = dates_full.get_loc(anchor)
                    history_init_tft = ser_t["ch4"][:anchor_idx + 1]

                    tft_chain = rr.dl_rollout(tft_model, se_t, sd_t, mu_t, sdy_t, device, fdl.TOW[tower],
                                               enc_ex_full, dec_ex_full, dates_full, history_init_tft, anchor,
                                               L=tft_cfg["L"], H=tft_cfg["H"], n_days=N_DAYS)

                    y_true_tft_full = pd.Series(ser_t["Y"], index=dates_full)  # real y_observed, unchanged
                    y_true_tft = y_true_tft_full.reindex(target_dates).values
                    yp = tft_chain.reindex(target_dates).values

                    bm = rr.bin_metrics(y_true_tft, yp, target_dates, anchor, y_persist=persist)
                    bm["model"] = "TFT_gf"
                    bm["anchor_year"] = yr
                    bm["tower"] = tower
                    all_rows.append(bm)

                    bm_gf = rr.bin_metrics(y_gf, yp, target_dates, anchor, y_persist=persist)
                    bm_gf["model"] = "TFT_gf"
                    bm_gf["anchor_year"] = yr
                    bm_gf["tower"] = tower
                    bm_gf["real_frac"] = bm_gf["bin"].map(real_frac_by_bin)
                    all_rows_gf.append(bm_gf)

                    chain_df["TFT_gf"] = tft_chain.reindex(target_dates).values
                    chain_df["y_true_tft"] = y_true_tft
                except Exception as e:
                    print(f"    Tower {tower} TFT_gf rollout SKIPPED: {str(e)[:200]}")

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
    out.to_csv(f"{RESULTS}/b10_b13_tft_gf_extension_summary{suffix}.csv", index=False)
    out_gf.to_csv(f"{RESULTS}/b10_b13_tft_gf_extension_summary_vs_gapfilled{suffix}.csv", index=False)
    chains.to_csv(f"{RESULTS}/b10_b13_tft_gf_extension_chains{suffix}.csv", index=False)
    print(f"\n[OK] Saved b10_b13_tft_gf_extension_*{suffix}.csv "
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
