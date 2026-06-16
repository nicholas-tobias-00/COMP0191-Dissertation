"""Precompute gap-filled FCO2 (CO2 flux) per tower for the 03b CO2-augmentation experiment.

Reconstructs ``FC_1_1_1 [Tower N]`` from meteorological-only drivers (the R-02
``driver_m`` set) using a Random Forest (RFm), fit on the training window's
QC-passed FC. The reconstruction makes a CO2-flux estimate available even where
FC is missing, so a complete FCO2 series can be used as a CH4 gap-filling feature.

Output: ``data/Hourly/fco2_gapfilled.csv`` with, per tower:
  FC_obs_qc [Tower N]    - observed FC after QC (SSITC in {0,1} + plausibility)
  FC_recon [Tower N]     - RFm reconstruction from met drivers (all timestamps)
  FC_gapfilled [Tower N] - observed-where-available, else reconstruction (used as feature)

Decisions:
  D-25  FC plausibility filter [-100, 100] umol m-2 s-1 (after SSITC 0/1).
  D-26  CO2-augmented gap-filling: reconstruct FC from independent met drivers,
        then use the (observed-where-available) FCO2 as a CH4 gap-filling feature.
  Reuses R-02 driver_m (D-21) and spatial-alignment / proxy rules (D-16, D-18).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

HOURLY = Path(__file__).resolve().parents[2] / "data" / "Hourly"
FC_LOW, FC_HIGH = -100.0, 100.0        # umol m-2 s-1  (D-25)
AUX = ["_hour_sin", "_hour_cos", "_doy_sin", "_doy_cos"]

# driver_m (met-only) per tower — identical to R-02 (D-21); Tower 2 added here.
TOWER_CONFIGS = {
    2: {
        "fc"   : "FC_1_1_1 [Tower 2]",
        "ssitc": "FC_SSITC_TEST_1_1_1 [Tower 2]",
        "driver_m": [
            "SWIN_1_1_1 [Tower 2]", "TA_0_0_1 [Tower 2]", "VPD_0_0_1 [Tower 2]",
            "PPFD_1_1_1 [Tower 2]", "USTAR_0_0_1 [Tower 2]", "WS_0_0_1 [Tower 2]",
            "RN_1_1_1 [Tower 2]", "Precipitation (mm) [Catchment 2]",
            "TS_1_1_1 [Tower 9]", "Soil Moisture @ 10cm Depth (%) [Catchment 2]",
            "SHF_1_1_1 [Tower 2]",
        ],
        "train_yrs": [2018],                      # Tower 2: only 2018 usable (D-15)
        "test_yrs" : [2019],
    },
    4: {
        "fc"   : "FC_1_1_1 [Tower 4]",
        "ssitc": "FC_SSITC_TEST_1_1_1 [Tower 4]",
        "driver_m": [
            "SWIN_1_1_1 [Tower 4]", "TA_0_0_1 [Tower 4]", "VPD_0_0_1 [Tower 4]",
            "PPFD_1_1_1 [Tower 4]", "USTAR_0_0_1 [Tower 4]", "WS_0_0_1 [Tower 4]",
            "RN_1_1_1 [Tower 4]", "Precipitation (mm) [Catchment 4 After  2013/08/13]",
            "TS_1_1_1 [Tower 9]", "Soil Moisture @ 10cm Depth (%) [Catchment 4 After  2013/08/13]",
            "SHF_1_1_1 [Tower 4]",
        ],
        "train_yrs": list(range(2018, 2022)),
        "test_yrs" : list(range(2022, 2024)),
    },
    9: {
        "fc"   : "FC_1_1_1 [Tower 9]",
        "ssitc": "FC_SSITC_TEST_1_1_1 [Tower 9]",
        "driver_m": [
            "SWIN_1_1_1 [Tower 9]", "TA_0_0_1 [Tower 9]", "VPD_0_0_1 [Tower 9]",
            "PPFD_1_1_1 [Tower 9]", "USTAR_0_0_1 [Tower 9]", "WS_0_0_1 [Tower 9]",
            "RN_1_1_1 [Tower 9]", "Precipitation (mm) [Catchment 9]",
            "TS_1_1_1 [Tower 9]", "Soil Moisture @ 10cm Depth (%) [Catchment 9]",
            "SHF_1_1_1 [Tower 9]",
        ],
        "train_yrs": list(range(2018, 2022)),
        "test_yrs" : list(range(2022, 2024)),
    },
}


def add_cyclical(df):
    hour, doy = df.index.hour, df.index.dayofyear
    df["_hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["_hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["_doy_sin"]  = np.sin(2 * np.pi * doy / 365)
    df["_doy_cos"]  = np.cos(2 * np.pi * doy / 365)


def reconstruct_tower(df, tower, cfg):
    """Return (fc_obs_qc, fc_recon, fc_gapfilled) Series + a stats dict for one tower."""
    fc, ssitc = cfg["fc"], cfg["ssitc"]
    feats = cfg["driver_m"] + AUX

    # --- QC observed FC: SSITC in {0,1}, then plausibility [-100, 100] (D-25) ---
    fc_obs = df[fc].copy()
    raw_n = fc_obs.notna().sum()
    fc_obs[~df[ssitc].isin([0, 1])] = np.nan
    fc_obs[fc_obs.notna() & ~fc_obs.between(FC_LOW, FC_HIGH, inclusive="both")] = np.nan
    qc_n = fc_obs.notna().sum()

    # --- Train RFm on training-window observed FC ---
    train_mask = df.index.year.isin(cfg["train_yrs"]) & fc_obs.notna()
    df_train = df.loc[train_mask, feats].copy()
    y_train = fc_obs.loc[train_mask].values
    df_train = df_train.dropna(how="all")  # keep rows; impute residual NaN
    imputer = SimpleImputer(strategy="mean")
    X_train = imputer.fit_transform(df.loc[train_mask, feats].values)

    rf = RandomForestRegressor(n_estimators=500, min_samples_leaf=5,
                               n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)

    # --- Reconstruct FC at every timestamp ---
    X_all = imputer.transform(df[feats].values)
    fc_recon = pd.Series(rf.predict(X_all), index=df.index)

    # --- Observed-where-available, else reconstruction ---
    fc_gapfilled = fc_obs.copy()
    fc_gapfilled[fc_gapfilled.isna()] = fc_recon[fc_gapfilled.isna()]

    # --- Validation: reconstruction vs observed FC in test years ---
    test_mask = df.index.year.isin(cfg["test_yrs"]) & fc_obs.notna()
    if test_mask.sum() >= 20:
        yt = fc_obs.loc[test_mask].values
        yp = fc_recon.loc[test_mask].values
        val_r2  = r2_score(yt, yp)
        val_rmse = float(mean_squared_error(yt, yp) ** 0.5)
        val_n = int(test_mask.sum())
    else:
        val_r2, val_rmse, val_n = np.nan, np.nan, int(test_mask.sum())

    stats = {
        "tower": tower, "raw_valid": int(raw_n), "qc_valid": int(qc_n),
        "n_train": int(train_mask.sum()), "filled_by_recon": int(fc_obs.isna().sum()),
        "recon_test_r2": round(val_r2, 4) if not np.isnan(val_r2) else None,
        "recon_test_rmse": round(val_rmse, 3) if not np.isnan(val_rmse) else None,
        "recon_test_n": val_n,
    }
    return fc_obs, fc_recon, fc_gapfilled, stats


def main():
    df = pd.read_csv(HOURLY / "consolidated_hourly.csv", low_memory=False)
    df["Datetime"] = pd.to_datetime(df["Datetime"], format="mixed")
    df = df.set_index("Datetime")
    add_cyclical(df)
    print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} cols")

    out = pd.DataFrame(index=df.index)
    all_stats = []
    for tower, cfg in TOWER_CONFIGS.items():
        fc_obs, fc_recon, fc_gap, stats = reconstruct_tower(df, tower, cfg)
        out[f"FC_obs_qc [Tower {tower}]"]    = fc_obs
        out[f"FC_recon [Tower {tower}]"]     = fc_recon
        out[f"FC_gapfilled [Tower {tower}]"] = fc_gap
        all_stats.append(stats)
        print(f"Tower {tower}: raw={stats['raw_valid']:,} QC={stats['qc_valid']:,} "
              f"n_train={stats['n_train']:,} recon-filled={stats['filled_by_recon']:,} "
              f"| recon test R2={stats['recon_test_r2']} RMSE={stats['recon_test_rmse']} "
              f"(n={stats['recon_test_n']:,})")

    out.index.name = "Datetime"
    dest = HOURLY / "fco2_gapfilled.csv"
    out.to_csv(dest)
    print(f"\nWrote {dest}  ({out.shape[0]:,} rows x {out.shape[1]} cols)")
    return all_stats


if __name__ == "__main__":
    main()
