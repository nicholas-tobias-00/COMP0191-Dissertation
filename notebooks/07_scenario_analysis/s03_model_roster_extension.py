"""S-03 model-roster extension (D-70 follow-up): closes a real scope gap in the driver-availability
ablation. The original `s03_driver_availability_ablation.py` covered only B-10's 6-model
tree/SARIMAX/ensemble roster ("Model 1 = B-10's full-feature ensemble", by its own docstring) --
never TFT/TabPFN/DLinear/LSTM/TabICLv2, the 5 models B-13/B-16/D-66 later added to the project's full
recursive-rollout roster. Flagged as a mistake by the user after being asked directly whether S-03
covered TabPFN/TabICLv2 -- it did not. This script runs the SAME two variants (A_removal, B_resample)
against those 5 remaining models:

  - TabPFN, TabICLv2: zero-shot, per-tower/per-anchor, daily track -- mirrors
    `b16_foundation_models_v3.py`'s integration exactly (mode="local", never pooled).
  - TFT, DLinear, LSTM: pooled per anchor (T2+T4+T9), hourly Track B, rolled out separately per
    tower -- mirrors `b16_dl_models_v3.py` / `b10_b13_dl_extension.py` exactly (TFT: D-45/B-13a
    recipe with weight_decay=1e-3/90-day val carve-out/patience=5; DLinear/LSTM: plain B-09 recipe,
    no val split). No new HPO anywhere.

Deliberately reuses S-03's OWN v2 data (`forecast_daily_v2.csv` / `forecast_features_v2.csv`), NOT
B-16's v3/F-10-enriched data -- keeps this experiment isolated to the driver-availability question,
consistent with the original S-03 script's explicit choice.

Column-mapping note (hourly DL track): of S-03's 24 degraded daily columns
(`DEFAULT_DEGRADED_COLS`, imported unchanged from `s03_driver_availability_ablation.py`), 12 have a
direct hourly analogue in `forecast_features_v2.csv` -- verified against
`build_forecasting_matrix_v2.py`'s `daily_table()`/`hourly_new()` source, since the daily columns are
literally `.resample("D").mean()` of these same underlying hourly series:
    fx_WS_mean       -> fx_WS_0_0_1
    fx_VPD_mean      -> fx_VPD_0_0_1
    fx_RN_mean       -> fx_RN_1_1_1
    fx_PPFD_mean     -> fx_PPFD_1_1_1
    fx_SWC_mean      -> "fx_Soil Moisture @ 10cm Depth (%)"
    fx_TS_mean       -> "fx_Soil Temperature @ 15cm Depth (oC)"
    fx_wd_sin        -> fx_wd_sin        (same name)
    fx_wd_cos        -> fx_wd_cos        (same name)
    fx_grazing_active-> fx_graze         (closest analogue: hourly instantaneous flag vs. daily
                                          any-grazing-that-day flag -- not identical semantics, noted)
    fx_days_since_grazing -> fx_days_since_grazing (same name)
    fx_USTAR_mean    -> fx_USTAR_0_0_1   (DROPPED_COLS)
    fx_SHF_mean      -> fx_shf3          (the v2 3-sensor-mean hourly feature -- NOT fx_SHF_1_1_1, a
                                          pre-existing single-sensor v1 column outside S-03's scope)
The remaining 12 (fx_SWC_lag{7,14,21,28}/roll{7,14}, fx_TS_lag{7,14,21,28}/roll{7,14}) are a
daily-only lag/rolling ladder with NO hourly equivalent at all -- genuinely out of scope for the
hourly DL track (no daily-resolution memory feature exists at hourly resolution to remove or
resample), not silently dropped. Variant A (removal) for the DL track therefore drops only the 12
mapped columns; Variant B (resample) climatology-substitutes only those same 12, at both the
encoder's recent-history window and the decoder's future window for post-anchor dates (the DL
rollout's encoder window slides into post-anchor dates as the chain progresses, so both need the
same treatment for internal consistency -- unlike the tree/SARIMAX variant B, which only has one
combined future-frame concept).

Run from project root:  python notebooks/07_scenario_analysis/s03_model_roster_extension.py
Optional quick smoke test:  python notebooks/07_scenario_analysis/s03_model_roster_extension.py smoke
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
sys.path.insert(0, ROOT + r"\src\features")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))

import models.recursive_rollout as rr
import models.forecasting_dl as fdl
from s03_driver_availability_ablation import (
    DEFAULT_DEGRADED_COLS, climatology_substitute, TOWERS, N_DAYS, VARIANTS,
)

HOURLY = rf"{ROOT}\data\Hourly"
RESULTS = rf"{ROOT}\results"
TRACK = "B"
VAL_DAYS = 90
ANCHOR_YEARS_FULL = [2018, 2019, 2020, 2021, 2022]

HOURLY_DEGRADED_MAP = {
    "fx_WS_mean": "fx_WS_0_0_1",
    "fx_VPD_mean": "fx_VPD_0_0_1",
    "fx_RN_mean": "fx_RN_1_1_1",
    "fx_PPFD_mean": "fx_PPFD_1_1_1",
    "fx_SWC_mean": "fx_Soil Moisture @ 10cm Depth (%)",
    "fx_TS_mean": "fx_Soil Temperature @ 15cm Depth (oC)",
    "fx_wd_sin": "fx_wd_sin",
    "fx_wd_cos": "fx_wd_cos",
    "fx_grazing_active": "fx_graze",
    "fx_days_since_grazing": "fx_days_since_grazing",
    "fx_USTAR_mean": "fx_USTAR_0_0_1",
    "fx_SHF_mean": "fx_shf3",
}
HOURLY_DEGRADED_COLS = list(HOURLY_DEGRADED_MAP.values())
DAILY_ONLY_NO_HOURLY_ANALOGUE = [c for c in DEFAULT_DEGRADED_COLS if c not in HOURLY_DEGRADED_MAP]

BINS = ((1, 7), (8, 30), (31, 90), (91, 180), (181, 270), (271, 365))


def real_frac_map(y_true, bin_labels):
    out = {}
    for lo, hi in BINS:
        lbl = f"{lo}-{hi}"
        m = bin_labels == lbl
        out[lbl] = float(np.isfinite(y_true[m]).mean()) if m.sum() > 0 else np.nan
    return out


def bm_rows(y_true, y_gf, yp, target_dates, anchor, persist, model, variant, yr, tower, real_frac_by_bin):
    bm = rr.bin_metrics(y_true, yp, target_dates, anchor, y_persist=persist)
    bm["model"] = model; bm["anchor_year"] = yr; bm["tower"] = tower; bm["variant"] = variant
    bm_gf = rr.bin_metrics(y_gf, yp, target_dates, anchor, y_persist=persist)
    bm_gf["model"] = model; bm_gf["anchor_year"] = yr; bm_gf["tower"] = tower; bm_gf["variant"] = variant
    bm_gf["real_frac"] = bm_gf["bin"].map(real_frac_by_bin)
    return bm, bm_gf


# ============================================================ Part 1: TabPFN + TabICLv2 (daily)
def run_foundation_models(anchor_years):
    dv = pd.read_csv(f"{HOURLY}/forecast_daily_v2.csv", low_memory=False)
    dv["Datetime"] = pd.to_datetime(dv["Datetime"], format="mixed")
    FX_B = [c for c in dv.columns if c.startswith("fx")]
    assert len(FX_B) == 34, f"expected 34 fx_ columns, got {len(FX_B)}"
    FX_A = [c for c in FX_B if c not in DEFAULT_DEGRADED_COLS]
    T = {t: dv[dv.tower == t].set_index("Datetime").sort_index() for t in TOWERS}

    tabpfn_ok = bool(os.environ.get("TABPFN_TOKEN"))
    if not tabpfn_ok:
        print("WARNING: TABPFN_TOKEN not set -- TabPFN will be skipped.")

    all_rows, all_rows_gf, all_chain_rows = [], [], []
    t0 = time.time()

    for yr in anchor_years:
        anchor = pd.Timestamp(f"{yr}-12-16")
        target_dates = pd.date_range(anchor + pd.Timedelta(days=1), periods=N_DAYS, freq="D")

        for tower in TOWERS:
            dft = T[tower]
            hist = dft.loc[:anchor]
            hist_target = hist["y_observed"]
            y_true = dft["y_observed"].reindex(target_dates).values
            y_gf = dft["y_gapfilled"].reindex(target_dates).values
            anchor_val = dft.loc[anchor, "y_gapfilled"]
            persist = rr.chain_persistence(anchor_val, N_DAYS)
            bin_labels = rr.lead_time_bin(target_dates, anchor)
            real_frac_by_bin = real_frac_map(y_true, bin_labels)

            per_variant_chains = {}
            for variant in VARIANTS:
                if variant == "A_removal":
                    hist_cov = hist[FX_A]
                    fut_cov = dft.loc[target_dates, FX_A]
                else:
                    hist_cov = hist[FX_B]
                    fut_cov = dft.loc[target_dates, FX_B].copy()
                    clim = climatology_substitute(dft, DEFAULT_DEGRADED_COLS, anchor, target_dates)
                    fut_cov[DEFAULT_DEGRADED_COLS] = clim[DEFAULT_DEGRADED_COLS]

                chain_cols = {}
                if tabpfn_ok:
                    try:
                        chain = rr.tabpfn_forecast(hist_target, hist_cov, fut_cov, mode="local")
                        yp = chain.reindex(target_dates).values
                        bm, bm_gf = bm_rows(y_true, y_gf, yp, target_dates, anchor, persist,
                                             "TabPFN", variant, yr, tower, real_frac_by_bin)
                        all_rows.append(bm); all_rows_gf.append(bm_gf)
                        chain_cols["TabPFN"] = chain.reindex(target_dates)
                    except Exception as e:
                        print(f"    T{tower} {yr} {variant} TabPFN SKIPPED: {str(e)[:150]}")

                try:
                    chain = rr.tabicl_forecast(hist_target, hist_cov, fut_cov)
                    yp = chain.reindex(target_dates).values
                    bm, bm_gf = bm_rows(y_true, y_gf, yp, target_dates, anchor, persist,
                                         "TabICLv2", variant, yr, tower, real_frac_by_bin)
                    all_rows.append(bm); all_rows_gf.append(bm_gf)
                    chain_cols["TabICLv2"] = chain.reindex(target_dates)
                except Exception as e:
                    print(f"    T{tower} {yr} {variant} TabICLv2 SKIPPED: {str(e)[:150]}")

                if chain_cols:
                    per_variant_chains[variant] = chain_cols

            for variant, cols in per_variant_chains.items():
                cdf = pd.DataFrame(cols, index=target_dates)
                cdf["y_true"] = y_true
                cdf["y_gapfilled"] = y_gf
                cdf["persistence"] = persist
                cdf["tower"] = tower
                cdf["anchor_year"] = yr
                cdf["variant"] = variant
                cdf.index.name = "date"
                all_chain_rows.append(cdf.reset_index())

            print(f"  [foundation] T{tower} anchor {yr} done ({time.time()-t0:.0f}s elapsed)")

    return all_rows, all_rows_gf, all_chain_rows


# ================================================= Part 2: TFT + DLinear + LSTM (hourly Track B)
def run_dl_models(anchor_years):
    fdl_m = fdl.load_matrix(f"{HOURLY}/forecast_features_v2.csv")
    device = fdl.get_device()
    cfg = fdl.TRACKS[TRACK]
    FULL_FX = list(fdl.FX)
    FX_A = [c for c in FULL_FX if c not in HOURLY_DEGRADED_COLS]
    print(f"[S-03 ext] hourly FULL_FX={len(FULL_FX)}, FX_A (removal)={len(FX_A)}, "
          f"resample_cols_hourly={len(HOURLY_DEGRADED_COLS)}, "
          f"daily-only (no hourly analogue, excluded)={DAILY_ONLY_NO_HOURLY_ANALOGUE}")

    dv = pd.read_csv(f"{HOURLY}/forecast_daily_v2.csv", low_memory=False)
    dv["Datetime"] = pd.to_datetime(dv["Datetime"], format="mixed")
    T = {t: dv[dv.tower == t].set_index("Datetime").sort_index() for t in TOWERS}

    raw_h = pd.read_csv(f"{HOURLY}/forecast_features_v2.csv", low_memory=False)
    raw_h["Datetime"] = pd.to_datetime(raw_h["Datetime"], format="mixed")
    daily_src = {t: raw_h[raw_h.tower == t].set_index("Datetime").sort_index()[HOURLY_DEGRADED_COLS]
                 .resample("D").mean() for t in TOWERS}

    all_rows, all_rows_gf, all_chain_rows = [], [], []
    t00 = time.time()

    for yr in anchor_years:
        anchor = pd.Timestamp(f"{yr}-12-16")
        target_dates = pd.date_range(anchor + pd.Timedelta(days=1), periods=N_DAYS, freq="D")
        cutoff = anchor + pd.Timedelta(hours=23, minutes=59)
        print(f"\n{'='*70}\nAnchor {yr}\n{'='*70}")

        chain_frames = {v: {} for v in VARIANTS}

        for variant in VARIANTS:
            t_v = time.time()
            fdl.FX = FX_A if variant == "A_removal" else FULL_FX

            # ---- TFT: D-45/B-13a recipe, pooled, 90-day val carve-out ----
            tft_model = se_t = sd_t = mu_t = sdy_t = None
            try:
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
                tft_model = fdl.build_model("TFT", cfg["L"], cfg["H"], n_enc, n_dec, 3)
                fdl.train_model(tft_model, train_tft, device, epochs=30, ch4_mu=mu_t, ch4_sd=sdy_t, seed=0,
                                 weight_decay=1e-3, val_data=val_tft, patience=5)
            except Exception as e:
                print(f"    {variant} TFT FIT SKIPPED: {str(e)[:150]}")

            # ---- DLinear/LSTM: B-09 recipe, pooled, no val split ----
            dl_models = {}
            se = sd = mu = sdy = None
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
                    model = fdl.build_model(name, cfg["L"], cfg["H"], n_enc2, n_dec2, 3)
                    fdl.train_model(model, dl_train, device, epochs=30, ch4_mu=mu, ch4_sd=sdy, seed=0)
                    dl_models[name] = model
            except Exception as e:
                print(f"    {variant} DLinear/LSTM FIT SKIPPED: {str(e)[:150]}")

            print(f"  {variant} pooled fit done ({time.time()-t_v:.0f}s)")

            for tower in TOWERS:
                dft = T[tower]
                y_gf = dft["y_gapfilled"].reindex(target_dates).values
                anchor_val = dft.loc[anchor, "y_gapfilled"]
                persist = rr.chain_persistence(anchor_val, N_DAYS)

                ser_t = fdl.tower_series(fdl_m, tower, TRACK)
                dates_full = ser_t["idx"]
                enc_ex_full, dec_ex_full = ser_t["enc_ex"].copy(), ser_t["dec_ex"].copy()
                anchor_idx = dates_full.get_loc(anchor)
                history_init = ser_t["ch4"][:anchor_idx + 1]
                y_true_full = pd.Series(ser_t["Y"], index=dates_full)
                y_true = y_true_full.reindex(target_dates).values
                bin_labels = rr.lead_time_bin(target_dates, anchor)
                real_frac_by_bin = real_frac_map(y_true, bin_labels)

                if variant == "B_resample":
                    post_mask = dates_full > anchor
                    post_dates = dates_full[post_mask]
                    for col in HOURLY_DEGRADED_COLS:
                        cidx = FULL_FX.index(col)
                        src = daily_src[tower][col]
                        hist_pre = src.loc[:anchor].dropna()
                        clim_vals = rr.doy_climatology(hist_pre, post_dates, window=7)
                        enc_ex_full[post_mask, cidx] = clim_vals
                        dec_ex_full[post_mask, cidx] = clim_vals

                tower_chains = {}
                if tft_model is not None:
                    try:
                        chain = rr.dl_rollout(tft_model, se_t, sd_t, mu_t, sdy_t, device, fdl.TOW[tower],
                                               enc_ex_full, dec_ex_full, dates_full, history_init, anchor,
                                               L=cfg["L"], H=cfg["H"], n_days=N_DAYS)
                        yp = chain.reindex(target_dates).values
                        bm, bm_gf = bm_rows(y_true, y_gf, yp, target_dates, anchor, persist,
                                            "TFT", variant, yr, tower, real_frac_by_bin)
                        all_rows.append(bm); all_rows_gf.append(bm_gf)
                        tower_chains["TFT"] = chain.reindex(target_dates)
                    except Exception as e:
                        print(f"    T{tower} {variant} TFT rollout SKIPPED: {str(e)[:150]}")

                for name, model in dl_models.items():
                    try:
                        chain = rr.dl_rollout(model, se, sd, mu, sdy, device, fdl.TOW[tower],
                                               enc_ex_full, dec_ex_full, dates_full, history_init, anchor,
                                               L=cfg["L"], H=cfg["H"], n_days=N_DAYS)
                        yp = chain.reindex(target_dates).values
                        bm, bm_gf = bm_rows(y_true, y_gf, yp, target_dates, anchor, persist,
                                            name, variant, yr, tower, real_frac_by_bin)
                        all_rows.append(bm); all_rows_gf.append(bm_gf)
                        tower_chains[name] = chain.reindex(target_dates)
                    except Exception as e:
                        print(f"    T{tower} {variant} {name} rollout SKIPPED: {str(e)[:150]}")

                if tower_chains:
                    tower_chains["y_true_dl"] = pd.Series(y_true, index=target_dates)
                    tower_chains["y_gapfilled"] = pd.Series(y_gf, index=target_dates)
                    tower_chains["persistence"] = pd.Series(persist, index=target_dates)
                    chain_frames[variant][tower] = tower_chains

        for variant in VARIANTS:
            for tower in TOWERS:
                cols = chain_frames[variant].get(tower)
                if not cols:
                    continue
                cdf = pd.DataFrame(cols)
                cdf["tower"] = tower
                cdf["anchor_year"] = yr
                cdf["variant"] = variant
                cdf.index.name = "date"
                all_chain_rows.append(cdf.reset_index())

        print(f"  Anchor {yr} total ({time.time()-t00:.0f}s)")

    fdl.FX = FULL_FX
    return all_rows, all_rows_gf, all_chain_rows


def main(anchor_years=None, run_label=""):
    anchor_years = ANCHOR_YEARS_FULL if anchor_years is None else anchor_years
    suffix = f"_{run_label}" if run_label else ""

    rows1, rows1_gf, chains1 = run_foundation_models(anchor_years)
    rows2, rows2_gf, chains2 = run_dl_models(anchor_years)

    out = pd.concat([pd.concat(rows1, ignore_index=True) if rows1 else pd.DataFrame(),
                      pd.concat(rows2, ignore_index=True) if rows2 else pd.DataFrame()], ignore_index=True)
    out.to_csv(f"{RESULTS}/s03_model_roster_extension_summary{suffix}.csv", index=False)
    print(f"\n[OK] Saved s03_model_roster_extension_summary{suffix}.csv ({len(out)} rows)")

    out_gf = pd.concat([pd.concat(rows1_gf, ignore_index=True) if rows1_gf else pd.DataFrame(),
                         pd.concat(rows2_gf, ignore_index=True) if rows2_gf else pd.DataFrame()], ignore_index=True)
    out_gf.to_csv(f"{RESULTS}/s03_model_roster_extension_summary_vs_gapfilled{suffix}.csv", index=False)
    print(f"[OK] Saved s03_model_roster_extension_summary_vs_gapfilled{suffix}.csv ({len(out_gf)} rows)")

    chains1_df = pd.concat(chains1, ignore_index=True) if chains1 else pd.DataFrame()
    chains2_df = pd.concat(chains2, ignore_index=True) if chains2 else pd.DataFrame()
    chains1_df.to_csv(f"{RESULTS}/s03_model_roster_extension_chains_foundation{suffix}.csv", index=False)
    chains2_df.to_csv(f"{RESULTS}/s03_model_roster_extension_chains_dl{suffix}.csv", index=False)
    print(f"[OK] Saved chain files ({len(chains1_df)} + {len(chains2_df)} rows)")

    return out, out_gf, chains1_df, chains2_df


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        main(anchor_years=[2021], run_label="smoketest")
    else:
        main()
