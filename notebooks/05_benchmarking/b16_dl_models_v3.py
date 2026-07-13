"""F-10 (D-67) Stage 2b extension: TFT + DLinear + LSTM across all 7 configs (BASE + 5 families +
BASE+ALL), all 3 towers, all 5 anchors -- closing the last part of "all models from SARIMAX to
TabICLv2" for the feature-family comparison.

Uses the new `data/Hourly/forecast_features_v3.csv` (built by
`build_forecasting_matrix_v3_hourly.py`) -- the DL models read from the hourly matrix via
`forecasting_dl.py`'s `load_matrix()`/`tower_series()`/`build_windows()`, which auto-detect
feature columns via `[c for c in m.columns if c.startswith("fx")]` into the MODULE-LEVEL `fdl.FX`
list. This script reassigns `fdl.FX` to each config's specific column subset immediately before
that config's training/rollout calls -- `forecasting_dl.py` itself is NOT edited; `FX` is read at
call time by `tower_series()`, so overriding it from here is sufficient and fully additive.

TFT: exact D-45/B-13a recipe (weight_decay=1e-3, 90-day pre-anchor validation carve-out, patience=5),
matching `b10_b13_rerun_multi_anchor.py`'s TFT block. DLinear/LSTM: exact B-09 recipe (plain,
unregularized, no val split), matching `b10_b13_dl_extension.py`. Both pooled (fit once per
anchor per config on T2+T4+T9), rolled out separately per tower via the unmodified `rr.dl_rollout`.

Run from project root:  python notebooks/05_benchmarking/b16_dl_models_v3.py
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
ANCHOR_YEARS = [2018, 2019, 2020, 2021, 2022]
TRACK = "B"

# Hourly-track new columns (no lag/roll ladder at hourly resolution, unlike the daily track --
# matches every other pre-existing hourly fx_ column's own "raw current reading" convention).
FAMILIES = {
    "species": ["fx_cattle_dens", "fx_sheep_dens", "fx_lamb_dens"],
    "arable": ["fx_is_arable"],
    "flow": ["fx_flow_mean"],
    "mgmt": ["fx_mgmt_fertN_recency", "fx_mgmt_fertN_rate", "fx_mgmt_lime_recency",
             "fx_mgmt_cultiv_recency", "fx_mgmt_cut_recency", "fx_mgmt_manure_recency"],
    "bodyweight": ["fx_total_liveweight_dens"],
}
ALL_NEW = sorted({c for cols in FAMILIES.values() for c in cols})


def main():
    fdl_m = fdl.load_matrix(f"{HOURLY}/forecast_features_v3.csv")
    device = fdl.get_device()
    tft_cfg = fdl.TRACKS[TRACK]
    FULL_FX = list(fdl.FX)
    BASE_FX = [c for c in FULL_FX if c not in ALL_NEW]
    print(f"BASE_FX hourly ({len(BASE_FX)}), ALL_NEW hourly ({len(ALL_NEW)}): {ALL_NEW}")

    dv = pd.read_csv(f"{HOURLY}/forecast_daily_v3.csv", low_memory=False)
    dv["Datetime"] = pd.to_datetime(dv["Datetime"], format="mixed")
    T = {t: dv[dv.tower == t].set_index("Datetime").sort_index() for t in TOWERS}

    configs = {"BASE": BASE_FX}
    for fam, cols in FAMILIES.items():
        configs[f"BASE+{fam}"] = BASE_FX + cols
    configs["BASE+ALL"] = BASE_FX + ALL_NEW

    all_rows = []
    t00 = time.time()

    for yr in ANCHOR_YEARS:
        anchor = pd.Timestamp(f"{yr}-12-16")
        target_dates = pd.date_range(anchor + pd.Timedelta(days=1), periods=N_DAYS, freq="D")
        print(f"\n{'='*70}\nAnchor {yr}\n{'='*70}")

        for cfg_name, fx_cols in configs.items():
            t_cfg = time.time()
            fdl.FX = fx_cols   # override the module-level feature list for this config only

            # ---- TFT: D-45/B-13a exact recipe, pooled, 90-day val carve-out ----
            tft_model = None
            try:
                cutoff = anchor + pd.Timedelta(hours=23, minutes=59)
                val_start = anchor - pd.Timedelta(days=VAL_DAYS)
                Wd = fdl.build_windows(fdl_m, TRACK)
                train_parts, val_parts = [], []
                for t in TOWERS:
                    ttime = pd.DatetimeIndex(Wd[t]["ttime"][:, -1])
                    train_parts.append(fdl._subset(Wd[t], ttime <= val_start))
                    val_parts.append(fdl._subset(Wd[t], (ttime > val_start) & (ttime <= cutoff)))
                train_tft = fdl._cat(train_parts); val_tft = fdl._cat(val_parts)
                se_t, sd_t = fdl.Scaler().fit(train_tft["enc"]), fdl.Scaler().fit(train_tft["dec"])
                yv = train_tft["y"][np.isfinite(train_tft["y"])]
                mu_t, sdy_t = float(yv.mean()), float(yv.std() + 1e-6)
                train_tft["enc"], train_tft["dec"] = se_t.tf(train_tft["enc"]), sd_t.tf(train_tft["dec"])
                val_tft["enc"], val_tft["dec"] = se_t.tf(val_tft["enc"]), sd_t.tf(val_tft["dec"])
                n_enc, n_dec = train_tft["enc"].shape[-1], train_tft["dec"].shape[-1]

                tft_model = fdl.build_model("TFT", tft_cfg["L"], tft_cfg["H"], n_enc, n_dec, 3)
                fdl.train_model(tft_model, train_tft, device, epochs=30, ch4_mu=mu_t, ch4_sd=sdy_t, seed=0,
                                 weight_decay=1e-3, val_data=val_tft, patience=5)
            except Exception as e:
                print(f"    {cfg_name} TFT FIT SKIPPED: {str(e)[:150]}")

            # ---- DLinear/LSTM: B-09 exact recipe, pooled, no val split ----
            dl_models = {}
            try:
                Wd2 = fdl.build_windows(fdl_m, TRACK)
                tr_parts = [fdl._subset(Wd2[t], pd.DatetimeIndex(Wd2[t]["ttime"][:, -1]) <= cutoff) for t in TOWERS]
                dl_train = fdl._cat(tr_parts)
                se, sd = fdl.Scaler().fit(dl_train["enc"]), fdl.Scaler().fit(dl_train["dec"])
                yv2 = dl_train["y"][np.isfinite(dl_train["y"])]
                mu, sdy = float(yv2.mean()), float(yv2.std() + 1e-6)
                dl_train["enc"], dl_train["dec"] = se.tf(dl_train["enc"]), sd.tf(dl_train["dec"])
                n_enc2, n_dec2 = dl_train["enc"].shape[-1], dl_train["dec"].shape[-1]
                for name in ["DLinear", "LSTM"]:
                    model = fdl.build_model(name, tft_cfg["L"], tft_cfg["H"], n_enc2, n_dec2, 3)
                    fdl.train_model(model, dl_train, device, epochs=30, ch4_mu=mu, ch4_sd=sdy, seed=0)
                    dl_models[name] = model
            except Exception as e:
                print(f"    {cfg_name} DLinear/LSTM FIT SKIPPED: {str(e)[:150]}")

            for tower in TOWERS:
                dft = T[tower]
                y_true = dft["y_observed"].reindex(target_dates).values
                y_gf = dft["y_gapfilled"].reindex(target_dates).values
                anchor_val = dft.loc[anchor, "y_gapfilled"]
                persist = rr.chain_persistence(anchor_val, N_DAYS)

                if tft_model is not None:
                    try:
                        ser_t = fdl.tower_series(fdl_m, tower, TRACK)
                        dates_full_tft, enc_ex_full, dec_ex_full = ser_t["idx"], ser_t["enc_ex"], ser_t["dec_ex"]
                        anchor_idx_tft = dates_full_tft.get_loc(anchor)
                        history_init_tft = ser_t["ch4"][:anchor_idx_tft + 1]
                        tft_chain = rr.dl_rollout(tft_model, se_t, sd_t, mu_t, sdy_t, device, fdl.TOW[tower],
                                                   enc_ex_full, dec_ex_full, dates_full_tft, history_init_tft, anchor,
                                                   L=tft_cfg["L"], H=tft_cfg["H"], n_days=N_DAYS)
                        y_true_tft_full = pd.Series(ser_t["Y"], index=dates_full_tft)
                        y_true_tft = y_true_tft_full.reindex(target_dates).values
                        yp = tft_chain.reindex(target_dates).values
                        bm = rr.bin_metrics(y_true_tft, yp, target_dates, anchor, y_persist=persist)
                        bm["target"] = "observed"; bm["model"] = "TFT"; bm["config"] = cfg_name; bm["anchor_year"] = yr; bm["tower"] = tower
                        all_rows.append(bm)
                        bm_gf = rr.bin_metrics(y_gf, yp, target_dates, anchor, y_persist=persist)
                        bm_gf["target"] = "gapfilled"; bm_gf["model"] = "TFT"; bm_gf["config"] = cfg_name; bm_gf["anchor_year"] = yr; bm_gf["tower"] = tower
                        all_rows.append(bm_gf)
                    except Exception as e:
                        print(f"    T{tower} {cfg_name} TFT rollout SKIPPED: {str(e)[:150]}")

                for name, model in dl_models.items():
                    try:
                        ser_t = fdl.tower_series(fdl_m, tower, TRACK)
                        dates_full, enc_ex_full, dec_ex_full = ser_t["idx"], ser_t["enc_ex"], ser_t["dec_ex"]
                        anchor_idx = dates_full.get_loc(anchor)
                        dl_history_init = ser_t["ch4"][:anchor_idx + 1]
                        chain = rr.dl_rollout(model, se, sd, mu, sdy, device, fdl.TOW[tower],
                                               enc_ex_full, dec_ex_full, dates_full, dl_history_init, anchor,
                                               L=tft_cfg["L"], H=tft_cfg["H"], n_days=N_DAYS)
                        y_true_full = pd.Series(ser_t["Y"], index=dates_full)
                        y_true_dl = y_true_full.reindex(target_dates).values
                        yp = chain.reindex(target_dates).values
                        bm = rr.bin_metrics(y_true_dl, yp, target_dates, anchor, y_persist=persist)
                        bm["target"] = "observed"; bm["model"] = name; bm["config"] = cfg_name; bm["anchor_year"] = yr; bm["tower"] = tower
                        all_rows.append(bm)
                        bm_gf = rr.bin_metrics(y_gf, yp, target_dates, anchor, y_persist=persist)
                        bm_gf["target"] = "gapfilled"; bm_gf["model"] = name; bm_gf["config"] = cfg_name; bm_gf["anchor_year"] = yr; bm_gf["tower"] = tower
                        all_rows.append(bm_gf)
                    except Exception as e:
                        print(f"    T{tower} {cfg_name} {name} rollout SKIPPED: {str(e)[:150]}")

            print(f"  {cfg_name} done ({time.time()-t_cfg:.0f}s, {time.time()-t00:.0f}s elapsed)")

    fdl.FX = FULL_FX  # restore, in case anything else in-process relies on the module global
    out = pd.concat(all_rows, ignore_index=True)
    out.to_csv(f"{RESULTS}/b16_dl_models_v3_summary.csv", index=False)
    print(f"\n[OK] Saved b16_dl_models_v3_summary.csv ({len(out)} rows), total {time.time()-t00:.0f}s")


if __name__ == "__main__":
    main()
