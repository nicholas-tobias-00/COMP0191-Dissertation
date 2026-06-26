"""F-08: Build an external-sourced (SMS/MET) parallel dataset (D-35).

Swaps the overlapping EC-tower drivers to the external catchment / Site sensor
network, WITHOUT touching ``consolidated_hourly.csv`` or ``reddyproc_processed.csv``.

Background: the project sources almost every driver from the co-located EC tower,
switching to the external network only for soil moisture (D-18). The NWFP runs a
second, independent network — one central MET station (``[Site]``) plus per-catchment
SMS stations (``[Catchment N]``). Seven variables overlap. F-08 tests whether external
sourcing helps (D-35).

Swaps (per tower 2/4/9), written under the EC column names so downstream code is
unchanged:
  SWIN_1_1_1 [Tower t]  <- Solar Radiation (W/m2) [Site]
  TA_0_0_1   [Tower t]  <- Air Temperature (oC) [Site]
  WS_0_0_1   [Tower t]  <- Wind Speed (km/h) [Site] / 3.6   (km/h -> m/s)
Soil temperature is swapped to the per-catchment external sensor in the gap-fill
driver map (ts -> "Soil Temperature @ 15cm Depth (oC) [catstr]"), replacing the
Tower-9 EC proxy (D-16). Kept EC (no external twin): VPD, PPFD, RN, USTAR, SHF.
Already external: Precipitation, Soil Moisture. RH/WD are not model features.
VPD is EC-derived with no direct external twin -> left as EC (documented).

Reuses the REddyProc utilities from reddyproc_pipeline.py (no duplication).

Outputs (both 70,153 rows):
  data/Hourly/consolidated_hourly_SMS_MET.csv   raw, with the swapped drivers
  data/Hourly/reddyproc_processed_SMS_MET.csv   gap-filled __f columns + GPP/Reco

Run from the project root:
    python src/data/build_sms_met_dataset.py
"""
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reddyproc_pipeline import (  # noqa: E402  reuse, don't duplicate
    mdc_gapfill, qc_fc, ustar_threshold, partition_nee, C4, TOWERS,
)

HOURLY = Path(__file__).resolve().parents[2] / "data" / "Hourly"

# External (SMS/MET) source columns. Air temp / solar / wind are Site-level only
# (one central station); soil temp is per-catchment.
SITE_SW = "Solar Radiation (W/m2) [Site]"
SITE_TA = "Air Temperature (oC) [Site]"
SITE_WS = "Wind Speed (km/h) [Site]"


def ext_driver_map(t, catstr):
    """Driver map for the external dataset: sw/ta/ws read the EC names (which now
    hold the swapped Site values); ts -> per-catchment external soil temperature."""
    return {
        "sw":   f"SWIN_1_1_1 [Tower {t}]",            # holds Site solar after swap
        "ta":   f"TA_0_0_1 [Tower {t}]",              # holds Site air temp after swap
        "vpd":  f"VPD_0_0_1 [Tower {t}]",             # kept EC (no external twin)
        "ppfd": f"PPFD_1_1_1 [Tower {t}]",            # kept EC
        "rn":   f"RN_1_1_1 [Tower {t}]",              # kept EC
        "ws":   f"WS_0_0_1 [Tower {t}]",              # holds Site wind (m/s) after swap
        "ustar":f"USTAR_0_0_1 [Tower {t}]",           # kept EC
        "shf":  f"SHF_1_1_1 [Tower {t}]",             # kept EC
        "precip": f"Precipitation (mm) [{catstr}]",   # already external
        "swc":  f"Soil Moisture @ 10cm Depth (%) [{catstr}]",         # already external (D-18)
        "ts":   f"Soil Temperature @ 15cm Depth (oC) [{catstr}]",     # external per-catchment (was Tower-9 proxy)
    }


def build_swapped_consolidated(df):
    """Copy of consolidated with EC SWIN/TA/WS overwritten by Site external values."""
    out = df.copy()
    for t in TOWERS:
        for ec, site, conv in [
            (f"SWIN_1_1_1 [Tower {t}]", SITE_SW, 1.0),
            (f"TA_0_0_1 [Tower {t}]",   SITE_TA, 1.0),
            (f"WS_0_0_1 [Tower {t}]",   SITE_WS, 1.0 / 3.6),   # km/h -> m/s
        ]:
            src = pd.to_numeric(df[site], errors="coerce") * conv
            n_ec, n_ext = df[ec].notna().sum(), src.notna().sum()
            out[ec] = src
            print(f"  T{t} {ec.split(' [')[0]:12s} <- {site:28s}  cov {100*n_ec/len(df):.0f}% -> {100*n_ext/len(df):.0f}%")
    return out


def gapfill_and_partition(df_swapped):
    """Run the REddyProc met gap-fill + u* + GPP/Reco on the swapped drivers."""
    out = pd.DataFrame(index=df_swapped.index); out.index.name = "Datetime"
    for t in TOWERS:
        catstr = C4 if t == 4 else f"Catchment {t}"
        dm = ext_driver_map(t, catstr)
        print(f"\n=== Tower {t} (external) ===")
        for k, col in dm.items():
            if col not in df_swapped.columns:
                print(f"  [skip] {col} missing"); continue
            filled, n0, n1 = mdc_gapfill(df_swapped[col])
            out[f"{col}__f"] = filled.round(4)
            print(f"  metfill {k:6s}: {100*df_swapped[col].notna().mean():.0f}% -> {100*filled.notna().mean():.0f}%  [{col}]")
        fc = qc_fc(df_swapped, t)
        sw = out[f"{dm['sw']}__f"]; ta = out[f"{dm['ta']}__f"]; us = out[f"{dm['ustar']}__f"]
        thr = ustar_threshold(ta, us, fc, sw)
        out[f"ustar_filtered [Tower {t}]"] = ((sw < 20) & (us < thr)).astype(int)
        print(f"  u* threshold = {thr:.3f} m/s")
        gpp, reco, e0 = partition_nee(fc, ta, sw)
        out[f"GPP [Tower {t}]"] = gpp.round(4); out[f"Reco [Tower {t}]"] = reco.round(4)
        print(f"  partitioning E0={e0}; GPP coverage {100*gpp.notna().mean():.0f}%")
    return out


def main():
    df = pd.read_csv(HOURLY / "consolidated_hourly.csv", low_memory=False)
    df["Datetime"] = pd.to_datetime(df["Datetime"], format="mixed")
    df = df.set_index("Datetime")
    print(f"Loaded {df.shape}")

    print("\n[1/2] Swapping EC drivers -> external (SMS/MET) under EC column names")
    swapped = build_swapped_consolidated(df)
    dest1 = HOURLY / "consolidated_hourly_SMS_MET.csv"
    swapped.to_csv(dest1)
    print(f"  saved: {dest1}  ({swapped.shape[0]:,} rows x {swapped.shape[1]} cols)")

    print("\n[2/2] REddyProc gap-fill + partitioning on the swapped drivers")
    proc = gapfill_and_partition(swapped)
    dest2 = HOURLY / "reddyproc_processed_SMS_MET.csv"
    proc.to_csv(dest2)
    print(f"\n  saved: {dest2}  ({proc.shape[0]:,} rows x {proc.shape[1]} cols)")


if __name__ == "__main__":
    main()
