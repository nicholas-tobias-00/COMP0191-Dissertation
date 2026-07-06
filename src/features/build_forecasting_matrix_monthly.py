"""Stage 0c (forecasting, D-55/B-11): monthly-resolution guide table, built from
forecast_daily_v2.csv.

Coarser-resolution companion to build_forecasting_matrix_v2.py's daily_table() -- dampens the
influence any single missed spike-day has on evaluation (the same mechanism M5's hierarchy
findings showed: coarser aggregates score better). Downscaled back to daily via
recursive_rollout.downscale_monthly_to_daily() for direct comparison against B-09/B-10 on the
same bin_metrics framework.

Aggregation rules per fx_ column (B-10/B-11 plan, D-54/D-55):
  - sum: fx_PRECIP_sum
  - mean: fx_WS_mean, fx_USTAR_mean, fx_TA_mean, fx_VPD_mean, fx_SWIN_mean, fx_RN_mean,
    fx_PPFD_mean, fx_SWC_mean, fx_TS_mean, fx_SHF_mean, fx_wd_sin, fx_wd_cos, fx_lsu_dens
  - min/max: fx_TA_min (monthly min of daily mins), fx_TA_max (monthly max of daily maxes)
  - fraction-of-month (mean of the 0/1 daily flag): fx_is_growing, fx_is_winter, fx_grazing_active
  - month-end value (not mean -- a mean of a monotonically-resetting counter is meaningless):
    fx_days_since_grazing
  - recomputed fresh from month-of-year, not resampled: fx_DOY_sin/cos
  - re-lagged/re-rolled at monthly windows on the newly-monthly fx_SWC_mean/fx_TS_mean (not
    resampled from the daily-lag columns, meaningless once the base frequency changes):
    fx_SWC_lag{1,2,3}/fx_TS_lag{1,2,3} (months), fx_SWC_roll{2,3}/fx_TS_roll{2,3} (months)
  - new ar_ch4_mlag{1,2,3}: monthly lags of monthly-resampled y_gapfilled

Emits data/Hourly/forecast_monthly_v2.csv (same location convention as forecast_daily_v2.csv).

Run from project root:  python src/features/build_forecasting_matrix_monthly.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

HOURLY = Path(__file__).resolve().parents[2] / "data" / "Hourly"

MEAN_COLS = ["fx_WS_mean", "fx_USTAR_mean", "fx_TA_mean", "fx_VPD_mean", "fx_SWIN_mean",
             "fx_RN_mean", "fx_PPFD_mean", "fx_SWC_mean", "fx_TS_mean", "fx_SHF_mean",
             "fx_wd_sin", "fx_wd_cos", "fx_lsu_dens"]
FRACTION_COLS = ["fx_is_growing", "fx_is_winter", "fx_grazing_active"]


def monthly_table(daily_df):
    """daily_df: one tower's slice of forecast_daily_v2.csv (has a Datetime column, daily rows)."""
    d = daily_df.set_index("Datetime").sort_index()
    t = d["tower"].iloc[0]

    idx = pd.PeriodIndex(d.index, freq="M").unique().sort_values().to_timestamp()
    mm = pd.DataFrame(index=idx)
    mm["tower"] = t

    mm["y_gapfilled"] = d["y_gapfilled"].resample("MS").mean()
    mm["y_observed"] = d["y_observed"].resample("MS").mean()

    mm["fx_PRECIP_sum"] = d["fx_PRECIP_sum"].resample("MS").sum()
    for c in MEAN_COLS:
        mm[c] = d[c].resample("MS").mean()
    mm["fx_TA_min"] = d["fx_TA_min"].resample("MS").min()
    mm["fx_TA_max"] = d["fx_TA_max"].resample("MS").max()
    for c in FRACTION_COLS:
        mm[c] = d[c].resample("MS").mean()
    mm["fx_days_since_grazing"] = d["fx_days_since_grazing"].resample("MS").last()

    mi = mm.index
    mm["fx_DOY_sin"] = np.sin(2 * np.pi * mi.month / 12)
    mm["fx_DOY_cos"] = np.cos(2 * np.pi * mi.month / 12)

    for L in (1, 2, 3):
        mm[f"fx_SWC_lag{L}"] = mm["fx_SWC_mean"].shift(L)
        mm[f"fx_TS_lag{L}"] = mm["fx_TS_mean"].shift(L)
    for W in (2, 3):
        mm[f"fx_SWC_roll{W}"] = mm["fx_SWC_mean"].rolling(W, min_periods=1).mean()
        mm[f"fx_TS_roll{W}"] = mm["fx_TS_mean"].rolling(W, min_periods=1).mean()

    for L in (1, 2, 3):
        mm[f"ar_ch4_mlag{L}"] = mm["y_gapfilled"].shift(L)

    for tt in [2, 4, 9]:
        mm[f"is_t{tt}"] = 1.0 if tt == t else 0.0
    mm.index.name = "Datetime"
    return mm.reset_index()


def main():
    daily = pd.read_csv(HOURLY / "forecast_daily_v2.csv", low_memory=False)
    daily["Datetime"] = pd.to_datetime(daily["Datetime"], format="mixed")
    monthly = pd.concat([monthly_table(daily[daily.tower == t].copy()) for t in [2, 4, 9]], ignore_index=True)
    monthly.to_csv(HOURLY / "forecast_monthly_v2.csv", index=False)
    print(f"monthly v2 {monthly.shape}")
    for t in [2, 4, 9]:
        sub = monthly[monthly.tower == t]
        print(f"  Tower {t}: {len(sub):>4} months, observed {int(sub.y_observed.notna().sum()):>3}")


if __name__ == "__main__":
    main()
