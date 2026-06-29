"""Stage 0b (forecasting, D-36): build the forecasting feature matrix.

Long-format per-tower hourly table for driver-conditional, direct multi-horizon
forecasting. Column groups (by prefix) tell the notebook how to align each to a
forecast origin t and horizon h:

  ar_*  : autoregressive / origin features — value KNOWN AT t (no shift).
          gap-filled CH4 lags + rolling stats; lagged-only fluxes (FCO2/GPP/Reco at t
          and earlier). FCO2/GPP/Reco are EC fluxes => never used at t+h (leak-free).
  fx_*  : future-exog — value AT TARGET TIME t+h (driver-conditional; the notebook
          shifts these by -h). Gap-filled met + soil temp/moisture + precip (External
          SMS/MET, D-35) + planned livestock density / grazing / management + calendar.
  y_gapfilled / y_observed / observed_mask : at time tau (notebook shifts -h for target).
  tower, is_t2/is_t4/is_t9 : identifiers / pooling dummies (origin, constant).

Reuses `frame()` + `load_ext()` from src/models/gapfill_rfm.py and the gap-filled CH4
series from data/Hourly/fch4_gapfilled.csv (build_fch4_gapfilled.py).

Output: data/Hourly/forecast_features.csv
Run from project root:  python src/features/build_forecasting_matrix.py
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
from gapfill_rfm import load_ext, frame, AUX, TOWERS  # noqa: E402

HOURLY = Path(__file__).resolve().parents[2] / "data" / "Hourly"

CH4_LAGS = [1, 2, 3, 6, 12, 24, 48, 168]          # hours
FLUX_LAGS = [24, 168]                              # lagged-only FCO2/GPP/Reco (hours)
# future-exog engineered names produced by frame() (met are stripped of "[Tower N]")
MET_NAMES = ["SWIN_1_1_1", "TA_0_0_1", "VPD_0_0_1", "PPFD_1_1_1", "RN_1_1_1",
             "WS_0_0_1", "USTAR_0_0_1", "SHF_1_1_1", "Precipitation (mm)",
             "Soil Temperature @ 15cm Depth (oC)", "Soil Moisture @ 10cm Depth (%)"]
FX_PLANNED = ["lsu_dens", "graze", "mgmt_cut", "mgmt_manure"]


def build_tower(t, d, gf):
    g = frame(t, pooled=True, d=d)
    idx = g.index
    y_obs = g["target"]                                   # QC'd observed CH4
    y_gap = gf[f"FCH4_gapfilled [Tower {t}]"].reindex(idx)
    out = pd.DataFrame(index=idx)
    out["tower"] = t
    out["y_gapfilled"] = y_gap
    out["y_observed"] = y_obs
    out["observed_mask"] = y_obs.notna().astype(int)

    # ar_ : autoregressive on the gap-filled series (known at origin t)
    for L in CH4_LAGS:
        out[f"ar_ch4_lag{L}"] = y_gap.shift(L)
    out["ar_ch4_rmean24"] = y_gap.shift(1).rolling(24, min_periods=1).mean()
    out["ar_ch4_rstd24"]  = y_gap.shift(1).rolling(24, min_periods=1).std()
    out["ar_ch4_rmean168"] = y_gap.shift(1).rolling(168, min_periods=1).mean()
    # lagged-only fluxes (origin t and earlier; never t+h)
    for src, nm in [("fc", "fc"), ("gpp", "gpp"), ("reco", "reco")]:
        out[f"ar_{nm}_t"] = g[src]
        for L in FLUX_LAGS:
            out[f"ar_{nm}_lag{L}"] = g[src].shift(L)

    # fx_ : future-exog (value at target time; notebook shifts by -h)
    for nm in MET_NAMES:
        out[f"fx_{nm}"] = g[nm]
    for nm in FX_PLANNED:
        out[f"fx_{nm}"] = g[nm]
    for a in AUX:                                        # calendar at target time
        out[f"fx{a}"] = g[a]

    # pooling dummies (origin, constant)
    for tt in TOWERS:
        out[f"is_t{tt}"] = 1.0 if tt == t else 0.0
    out["_y"] = idx.year
    return out


def main():
    d = load_ext()
    gf = pd.read_csv(HOURLY / "fch4_gapfilled.csv", low_memory=False)
    gf["Datetime"] = pd.to_datetime(gf["Datetime"], format="mixed"); gf = gf.set_index("Datetime")
    print(f"Loaded EXT {d.shape}, gap-filled CH4 {gf.shape}")

    parts = [build_tower(t, d, gf) for t in TOWERS]
    mat = pd.concat(parts).reset_index().rename(columns={"index": "Datetime", "Datetime": "Datetime"})
    mat = mat.rename(columns={mat.columns[0]: "Datetime"})

    fx = [c for c in mat.columns if c.startswith("fx")]
    ar = [c for c in mat.columns if c.startswith("ar_")]
    print(f"Matrix {mat.shape}: {len(ar)} ar_ (origin) + {len(fx)} fx_ (future-exog) features")
    for t in TOWERS:
        sub = mat[mat.tower == t]
        print(f"  Tower {t}: {len(sub):>6} rows, observed CH4 {int(sub.observed_mask.sum()):>6}")
    nan_cols = [c for c in fx + ar if mat[c].isna().all()]
    print("  all-NaN feature columns:", nan_cols if nan_cols else "none")

    dest = HOURLY / "forecast_features.csv"
    mat.to_csv(dest, index=False)
    print(f"\nWrote {dest}  ({mat.shape[0]:,} rows x {mat.shape[1]} cols)")


if __name__ == "__main__":
    main()
