"""D-7x: gap-filled-target ablation for DLinear/LSTM -- tests whether extending the tree/SARIMAX
D-36/D-37 convention ("train on y_gapfilled, evaluate on y_observed") to the DL family helps, hurts,
or mostly just inflates apparent skill via gap-filler mimicry (the exact circularity risk already
named in `b10_b13_rerun_multi_anchor.py`'s own docstring and in `tabpfn_forecast()`/
`tabicl_forecast()`'s docstrings, `recursive_rollout.py`).

Identical recipe to `b10_b13_dl_extension.py` (pooled T2+T4+T9 fit per anchor, no val split, plain
`fdl.train_model`, `rr.dl_rollout` rollout, `fdl.tower_series(...)["Y"]` as evaluation ground truth)
-- the ONLY change is building windows with `y_source="gapfilled"` (dense target, near-100% finite)
instead of the default `y_source="observed"` (masked, ~45-55% finite). Encoder history (already
y_gapfilled via `ch4`) and evaluation ground truth are untouched -- only the training LOSS target
changes. Bypasses `fdl.build_windows`/`TRACKS` (calls `fdl.make_windows` directly per tower before
pooling), mirroring B-10's own H=1-retrain precedent of not touching the shared batch-window helper.

Outputs `DLinear_gf`/`LSTM_gf` columns -- kept alongside (not replacing) the existing
`DLinear`/`LSTM` columns from `b10_b13_dl_extension.py`.
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
TRACK = "B"


def run(anchor_years, towers=TOWERS, tag=""):
    dv = pd.read_csv(f"{HOURLY}/forecast_daily_v2.csv", low_memory=False)
    dv["Datetime"] = pd.to_datetime(dv["Datetime"], format="mixed")
    T = {t: dv[dv.tower == t].set_index("Datetime").sort_index() for t in towers}

    fdl_m = fdl.load_matrix(f"{HOURLY}/forecast_features_v2.csv")
    device = fdl.get_device()
    cfg = fdl.TRACKS[TRACK]

    all_rows, all_rows_gf, all_chain_rows = [], [], []

    for yr in anchor_years:
        anchor = pd.Timestamp(f"{yr}-12-16")
        target_dates = pd.date_range(anchor + pd.Timedelta(days=1), periods=N_DAYS, freq="D")
        print(f"\n{'='*70}\nAnchor {yr}\n{'='*70}")
        t_anchor = time.time()

        # ---- Pooled DL training (T2+T4+T9), fit ONCE per anchor, gap-filled target ----
        cutoff = anchor + pd.Timedelta(hours=23, minutes=59)
        Wd = {t: fdl.make_windows(fdl.tower_series(fdl_m, t, TRACK), cfg["L"], cfg["H"], cfg["stride"],
                                   y_source="gapfilled") for t in TOWERS}
        tr_parts = [fdl._subset(Wd[t], pd.DatetimeIndex(Wd[t]["ttime"][:, -1]) <= cutoff) for t in TOWERS]
        dl_train = fdl._cat(tr_parts)
        se, sd = fdl.Scaler().fit(dl_train["enc"]), fdl.Scaler().fit(dl_train["dec"])
        yv = dl_train["y"][np.isfinite(dl_train["y"])]
        mu, sdy = float(yv.mean()), float(yv.std() + 1e-6)
        dl_train["enc"], dl_train["dec"] = se.tf(dl_train["enc"]), sd.tf(dl_train["dec"])
        n_enc, n_dec = dl_train["enc"].shape[-1], dl_train["dec"].shape[-1]
        print(f"  Training target finite frac (gap-filled): {np.isfinite(dl_train['y']).mean():.3f}")

        dl_models = {}
        for name in ["DLinear", "LSTM"]:
            model = fdl.build_model(name, cfg["L"], cfg["H"], n_enc, n_dec, 3)
            fdl.train_model(model, dl_train, device, epochs=30, ch4_mu=mu, ch4_sd=sdy, seed=0)
            dl_models[name] = model
        print(f"  Pooled DLinear/LSTM fit ({time.time()-t_anchor:.0f}s)")

        for tower in towers:
            t_tower = time.time()
            dft = T[tower]
            ser_t = fdl.tower_series(fdl_m, tower, TRACK)
            dates_full, enc_ex_full, dec_ex_full = ser_t["idx"], ser_t["enc_ex"], ser_t["dec_ex"]
            anchor_idx = dates_full.get_loc(anchor)
            dl_history_init = ser_t["ch4"][:anchor_idx + 1]

            y_true_full = pd.Series(ser_t["Y"], index=dates_full)  # real y_observed, unchanged
            y_true = y_true_full.reindex(target_dates).values

            anchor_val = dft.loc[anchor, "y_gapfilled"]
            persist = rr.chain_persistence(anchor_val, N_DAYS)

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
                col = f"{name}_gf"
                dl_out_chains[col] = chain
                yp = chain.reindex(target_dates).values
                bm = rr.bin_metrics(y_true, yp, target_dates, anchor, y_persist=persist)
                bm["model"] = col
                bm["anchor_year"] = yr
                bm["tower"] = tower
                all_rows.append(bm)

                bm_gf = rr.bin_metrics(y_gf, yp, target_dates, anchor, y_persist=persist)
                bm_gf["model"] = col
                bm_gf["anchor_year"] = yr
                bm_gf["tower"] = tower
                bm_gf["real_frac"] = bm_gf["bin"].map(real_frac_by_bin)
                all_rows_gf.append(bm_gf)

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
    out_gf = pd.concat(all_rows_gf, ignore_index=True)
    chains = pd.concat(all_chain_rows, ignore_index=True)

    suffix = f"_{tag}" if tag else ""
    out.to_csv(f"{RESULTS}/b10_b13_dl_gf_extension_summary{suffix}.csv", index=False)
    out_gf.to_csv(f"{RESULTS}/b10_b13_dl_gf_extension_summary_vs_gapfilled{suffix}.csv", index=False)
    chains.to_csv(f"{RESULTS}/b10_b13_dl_gf_extension_chains{suffix}.csv", index=False)
    print(f"\n[OK] Saved b10_b13_dl_gf_extension_*{suffix}.csv "
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
