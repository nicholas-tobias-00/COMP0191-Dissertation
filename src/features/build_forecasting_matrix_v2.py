"""Stage 0b-v2 (forecasting, D-41): enriched forecasting feature matrices (all towers).

Productionises the `NWFP_T9_Dataset_Structure.md` feature engineering across Towers 2/4/9,
on top of the existing `forecast_features.csv` base. Two artifacts:

  forecast_features_v2.csv  (hourly, long) = existing matrix + new future-exog fx_ columns that
        resample-by-mean cleanly: fx_wd_sin/cos (circular wind dir), fx_is_daytime, fx_shf3
        (3-sensor SHF mean), fx_is_growing, fx_is_winter, fx_days_since_grazing.
        -> consumed by B03 Track A and ALL of B04.

  forecast_daily_v2.csv     (daily, long) = guide daily table per tower with proper per-feature
        aggregation (TA min/max, precip sum, external soil daily lags/rolling, circular daily WD,
        days_since_grazing, calendar) + daily AR (ar_ch4_dlag* + ar_ch4_drm7 + ar_fc_dlag1) +
        targets (>=6 observed-hours rule) + tower dummies.
        -> consumed by B03 Track B only (DL's 28-day lookback already sees that history).

Soil source = EXTERNAL per-catchment (D-35). Reuses load_ext/frame/LSU/AREA from gapfill_rfm.py
and the gap-filled CH4 series from fch4_gapfilled.csv, exactly like build_forecasting_matrix.py.

Run from project root:  python src/features/build_forecasting_matrix_v2.py
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
from gapfill_rfm import load_ext, frame, TOWERS  # noqa: E402

HOURLY = Path(__file__).resolve().parents[2] / "data" / "Hourly"
SWC = "Soil Moisture @ 10cm Depth (%)"   # EXTERNAL per-catchment (D-35), gap-filled in frame()
TS = "Soil Temperature @ 15cm Depth (oC)"


def days_since_grazing(ga):
    """Counter resetting to 0 at each grazing-spell onset; 0 off-grazing. (ga = daily bool)."""
    spell = (ga != ga.shift()).cumsum()
    return ga.groupby(spell).cumcount().where(ga, 0).astype(float)


def shf3_series(t, d, idx):
    cols = [f"SHF_{i}_1_1 [Tower {t}]" for i in (1, 2, 3)]
    return d[cols].reindex(idx).mean(axis=1)


def hourly_new(t, d, g):
    """New hourly future-exog fx_ columns for tower t (indexed by Datetime)."""
    out = pd.DataFrame(index=g.index)
    wd = np.deg2rad(d[f"WD_0_0_1 [Tower {t}]"].reindex(g.index))
    out["fx_wd_sin"] = np.sin(wd)
    out["fx_wd_cos"] = np.cos(wd)
    out["fx_is_daytime"] = (g["SWIN_1_1_1"] > 5).astype(float)
    out["fx_shf3"] = shf3_series(t, d, g.index)
    mo = g.index.month
    out["fx_is_growing"] = mo.isin([4, 5, 6, 7, 8, 9]).astype(float)
    out["fx_is_winter"] = mo.isin([12, 1, 2]).astype(float)
    ga = g["graze"].resample("D").max() > 0
    dsg = days_since_grazing(ga)
    out["fx_days_since_grazing"] = dsg.reindex(g.index.normalize()).to_numpy()
    out["tower"] = t
    return out.reset_index().rename(columns={"index": "Datetime"})


def daily_table(t, d, g, gf):
    """Guide daily canonical table for tower t (fx_ aggregations + daily AR + dummies)."""
    obs_h = g["target"].resample("D").count()
    yo = g["target"].resample("D").mean().where(obs_h >= 6)             # eval target
    yg = gf[f"FCH4_gapfilled [Tower {t}]"].resample("D").mean()         # continuous training target

    dd = pd.DataFrame(index=yg.index)
    dd["tower"] = t
    dd["y_observed"] = yo
    dd["y_gapfilled"] = yg

    # met daily aggregations
    dd["fx_WS_mean"] = g["WS_0_0_1"].resample("D").mean()
    dd["fx_USTAR_mean"] = g["USTAR_0_0_1"].resample("D").mean()
    dd["fx_TA_mean"] = g["TA_0_0_1"].resample("D").mean()
    dd["fx_TA_min"] = g["TA_0_0_1"].resample("D").min()
    dd["fx_TA_max"] = g["TA_0_0_1"].resample("D").max()
    dd["fx_VPD_mean"] = g["VPD_0_0_1"].resample("D").mean()
    dd["fx_SWIN_mean"] = g["SWIN_1_1_1"].resample("D").mean()
    dd["fx_RN_mean"] = g["RN_1_1_1"].resample("D").mean()
    dd["fx_PPFD_mean"] = g["PPFD_1_1_1"].resample("D").mean()
    dd["fx_SWC_mean"] = g[SWC].resample("D").mean()                    # EXTERNAL (D-35)
    dd["fx_TS_mean"] = g[TS].resample("D").mean()                      # EXTERNAL (D-35)
    dd["fx_SHF_mean"] = shf3_series(t, d, g.index).resample("D").mean()
    dd["fx_PRECIP_sum"] = g["Precipitation (mm)"].resample("D").sum()
    wd = np.deg2rad(d[f"WD_0_0_1 [Tower {t}]"].reindex(g.index))
    dd["fx_wd_sin"] = np.sin(wd).resample("D").mean()                  # circular daily mean
    dd["fx_wd_cos"] = np.cos(wd).resample("D").mean()

    # external-soil daily lags + rolling
    for L in (7, 14, 21, 28):
        dd[f"fx_SWC_lag{L}"] = dd["fx_SWC_mean"].shift(L)
        dd[f"fx_TS_lag{L}"] = dd["fx_TS_mean"].shift(L)
    for W in (7, 14):
        dd[f"fx_SWC_roll{W}"] = dd["fx_SWC_mean"].rolling(W, min_periods=1).mean()
        dd[f"fx_TS_roll{W}"] = dd["fx_TS_mean"].rolling(W, min_periods=1).mean()

    # calendar
    idx = dd.index
    dd["fx_DOY_sin"] = np.sin(2 * np.pi * idx.dayofyear / 365)
    dd["fx_DOY_cos"] = np.cos(2 * np.pi * idx.dayofyear / 365)
    dd["fx_is_growing"] = idx.month.isin([4, 5, 6, 7, 8, 9]).astype(float)
    dd["fx_is_winter"] = idx.month.isin([12, 1, 2]).astype(float)

    # management / biological
    dd["fx_lsu_dens"] = g["lsu_dens"].resample("D").mean()
    ga = g["graze"].resample("D").max() > 0
    dd["fx_grazing_active"] = ga.astype(float)
    dd["fx_days_since_grazing"] = days_since_grazing(ga)

    # daily AR (origin-known): CH4 lags + 7d rolling; lagged-only FCO2 (leak-free enrichment)
    for L in (1, 2, 3, 7, 14):
        dd[f"ar_ch4_dlag{L}"] = yg.shift(L)
    dd["ar_ch4_drm7"] = yg.shift(1).rolling(7, min_periods=1).mean()
    dd["ar_fc_dlag1"] = g["fc"].resample("D").mean().shift(1)

    for tt in TOWERS:
        dd[f"is_t{tt}"] = 1.0 if tt == t else 0.0
    dd.index.name = "Datetime"
    return dd.reset_index()


def main():
    d = load_ext()
    gf = pd.read_csv(HOURLY / "fch4_gapfilled.csv", low_memory=False)
    gf["Datetime"] = pd.to_datetime(gf["Datetime"], format="mixed"); gf = gf.set_index("Datetime")
    mat = pd.read_csv(HOURLY / "forecast_features.csv", low_memory=False)
    mat["Datetime"] = pd.to_datetime(mat["Datetime"], format="mixed")
    print(f"Loaded EXT {d.shape}, gap-filled CH4 {gf.shape}, base matrix {mat.shape}")

    frames = {t: frame(t, pooled=True, d=d).sort_index() for t in TOWERS}

    # ---- hourly v2: base matrix + new fx_ columns ----
    new_h = pd.concat([hourly_new(t, d, frames[t]) for t in TOWERS], ignore_index=True)
    v2 = mat.merge(new_h, on=["Datetime", "tower"], how="left")
    new_cols = [c for c in new_h.columns if c.startswith("fx_")]
    nan_h = [c for c in new_cols if v2[c].isna().all()]
    v2.to_csv(HOURLY / "forecast_features_v2.csv", index=False)
    print(f"hourly v2 {v2.shape}: +{len(new_cols)} new fx_ {new_cols}")
    print("  all-NaN new cols:", nan_h if nan_h else "none")

    # ---- daily v2 ----
    daily = pd.concat([daily_table(t, d, frames[t], gf) for t in TOWERS], ignore_index=True)
    fx_b = [c for c in daily.columns if c.startswith("fx")]
    ar_b = [c for c in daily.columns if c.startswith("ar_")]
    nan_d = [c for c in fx_b + ar_b if daily[c].isna().all()]
    daily.to_csv(HOURLY / "forecast_daily_v2.csv", index=False)
    print(f"daily v2 {daily.shape}: {len(fx_b)} fx_ + {len(ar_b)} ar_ features")
    for t in TOWERS:
        sub = daily[daily.tower == t]
        print(f"  Tower {t}: {len(sub):>5} days, observed {int(sub.y_observed.notna().sum()):>4}")
    print("  all-NaN feature cols:", nan_d if nan_d else "none")


if __name__ == "__main__":
    main()
