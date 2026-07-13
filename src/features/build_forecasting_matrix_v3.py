"""F-10 (D-67): extended daily forecasting features -- livestock species disaggregation,
land-use regime flag, catchment flow, fertilizer/management richness, and (bonus) estimated
liveweight density. Additive clone of build_forecasting_matrix_v2.py's daily track only -- reads
the already-built forecast_daily_v2.csv and left-merges new fx_ columns onto it; does not edit
build_forecasting_matrix_v2.py, gapfill_rfm.py, or build_management_features.py, and does not
touch the hourly track (forecast_features_v2.csv).

Families (see DECISIONS.md D-67 for the full empirical grounding of each):
  (a) fx_cattle_dens / fx_sheep_dens / fx_lamb_dens -- raw per-species head-density (head/ha),
      NOT LSU-weighted (F-01 precedent). Existing fx_lsu_dens (v2) is untouched; by construction
      1.0*fx_cattle_dens + 0.1*fx_sheep_dens + 0.05*fx_lamb_dens == fx_lsu_dens exactly.
  (b) fx_is_arable -- 1 on/after the earliest classify()=="cultiv" event in the tower's own
      catchment fields, else 0. Derived programmatically per tower, not hardcoded (Tower 2 is
      expected, not assumed, to flip in Sept 2019; Towers 4/9 are expected to never flip).
  (c) fx_flow_mean (+lag7/14/21/28, roll7/14) -- catchment-level flume discharge, same per-
      catchment column-lookup convention as the existing fx_SWC_mean/fx_TS_mean.
  (d) mgmt_t{t}_{fertN_recency,fertN_rate,lime_recency,cultiv_recency,cut_recency,manure_recency}
      -- already computed in management_features.csv, never previously merged into the daily
      matrix (a real, separate gap from the hourly hourly hourly-track mgmt_cut/mgmt_manure gap).
  (e) fx_total_liveweight_dens -- bonus family, from build_bodyweight_density.py's output
      (must be run first: `python src/features/build_bodyweight_density.py`).

Run from project root:  python src/features/build_forecasting_matrix_v3.py
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gapfill_rfm import load_ext, LSU, AREA, TOWERS, C4  # noqa: E402
from build_management_features import CATCHMENT_FIELDS, TOWER_CATCHMENT  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
HOURLY = ROOT / "data" / "Hourly"
EVENTS = ROOT / "data" / "Compiled" / "Field_Event_Data_Format_1.csv"

FLOW = "Flow (l/s)"


def cat_str(t):
    return C4 if t == 4 else f"Catchment {t}"


def species_density(t, d):
    """(a) raw per-species head-density, head/ha -- mirrors gapfill_rfm.frame()'s lsu_dens
    computation exactly except NOT LSU-weighted, and species kept separate."""
    cat = cat_str(t)
    out = pd.DataFrame(index=d.index)
    for s in LSU:
        col = f"{s}_{cat}"
        out[f"fx_{s}_dens_hourly"] = d[col].fillna(0) / AREA[t]
    daily = out.resample("D").mean()
    daily.columns = [f"fx_{s}_dens" for s in LSU]
    return daily


ARABLE_CROP_KW = r"wheat|oat|barley|bean"


def is_arable_flag(t, index):
    """(b) programmatic land-use regime flag: 1 on/after the earliest genuine arable-conversion
    signal in the tower's own catchment fields, else 0 for the whole record.

    Deliberately NARROWER than classify()=="cultiv": that channel also fires on routine
    grassland renovation (chain harrowing, rolling, clover/grass reseeding -- e.g. 'Grass
    seeding (overseeding)' with a clover blend, or a single 'Chain harrow' event), which are
    NOT evidence of an arable regime shift. Confirmed empirically (F-10 build) that using the
    broad cultiv channel produces false positives at Towers 4/9 (a lone Chain harrow event each)
    that contradict the direct field-record check (no Plough/cereal-drilling event ever occurs
    at NW005/NW006 or NW013/NW039 in the whole 2017-2024 record). The genuine signal used here:
    a literal 'Plough' operation, or a Drill/Broadcast-Seed-type operation whose Application
    names an actual cereal/arable crop (wheat/oat/barley/bean) rather than a grass/clover
    species -- confirmed empirically that 'Plough' and arable-crop-keyword drilling only ever
    occur at fields NW002/NW003/NW004/NW015/NW019/NW047 in this dataset, never at T4/T9's own
    fields, matching the direct manual check exactly."""
    fe = pd.read_csv(EVENTS, low_memory=False)
    fe["dt"] = pd.to_datetime(fe["Event_Date"], errors="coerce")
    fe["field"] = fe["Field"].astype(str).str.strip()
    op = fe["Field_Operation"].astype(str)
    is_plough = op == "Plough"
    is_drill = op.isin(["Drill Seed", "Broadcast Seed"]) | op.str.startswith("Drilling")
    is_arable_crop = fe["Application"].astype(str).str.contains(ARABLE_CROP_KW, case=False, na=False)
    fe["is_arable_event"] = is_plough | (is_drill & is_arable_crop)

    cat = TOWER_CATCHMENT[t]
    fields = CATCHMENT_FIELDS[cat]
    sub = fe[(fe["field"].isin(fields)) & fe["is_arable_event"]].dropna(subset=["dt"])
    if len(sub) == 0:
        return pd.Series(0.0, index=index), None
    flip_date = sub["dt"].min().normalize()
    flag = (index.normalize() >= flip_date).astype(float)
    return pd.Series(flag, index=index), flip_date


def flow_daily(t, d):
    """(c) catchment flow, mean + lags/rolls, mirroring fx_SWC_lag*/fx_SWC_roll* line-for-line.

    Builds `dd` on the already-daily-resampled index (not the raw hourly `d.index`) before
    shifting/rolling -- shifting on an hourly-indexed frame would shift by hours, not days."""
    cat = cat_str(t)
    col = f"{FLOW} [{cat}]"
    daily_mean = d[col].resample("D").mean()
    dd = pd.DataFrame(index=daily_mean.index)
    dd["fx_flow_mean"] = daily_mean
    for L in (7, 14, 21, 28):
        dd[f"fx_flow_lag{L}"] = dd["fx_flow_mean"].shift(L)
    for W in (7, 14):
        dd[f"fx_flow_roll{W}"] = dd["fx_flow_mean"].rolling(W, min_periods=1).mean()
    return dd


def management_richness(t, idx):
    """(d) merge already-computed-but-unused management_features.csv columns."""
    mf = pd.read_csv(HOURLY / "management_features.csv", low_memory=False)
    mf["Datetime"] = pd.to_datetime(mf["Datetime"], format="mixed")
    mf = mf.set_index("Datetime")
    cols = [f"mgmt_t{t}_{ch}" for ch in
            ("fertN_recency", "fertN_rate", "lime_recency", "cultiv_recency",
             "cut_recency", "manure_recency")]
    daily = mf[cols].resample("D").mean()
    daily.columns = [c.replace(f"mgmt_t{t}_", "fx_mgmt_") for c in cols]
    return daily.reindex(idx)


def bodyweight_density(t, idx):
    """(e) bonus: estimated total liveweight density, from build_bodyweight_density.py's output."""
    path = HOURLY / "bodyweight_density.csv"
    if not path.exists():
        print("  [skip] bodyweight_density.csv not found -- run build_bodyweight_density.py first")
        return pd.DataFrame({"fx_total_liveweight_dens": np.nan}, index=idx)
    bw = pd.read_csv(path, parse_dates=["Datetime"])
    sub = bw[bw.tower == t].set_index("Datetime")["fx_total_liveweight_dens"]
    return sub.reindex(idx).to_frame()


def main():
    v2 = pd.read_csv(HOURLY / "forecast_daily_v2.csv", low_memory=False)
    v2["Datetime"] = pd.to_datetime(v2["Datetime"], format="mixed")
    print(f"Loaded forecast_daily_v2.csv {v2.shape}")

    d = load_ext()
    print(f"Loaded EXT (raw hourly, for species/flow columns) {d.shape}")

    new_frames = []
    flip_dates = {}
    for t in TOWERS:
        idx = pd.DatetimeIndex(v2.loc[v2.tower == t, "Datetime"].unique()).sort_values()

        sp = species_density(t, d).reindex(idx)
        arable, flip = is_arable_flag(t, idx)
        flip_dates[t] = flip
        flow = flow_daily(t, d).reindex(idx)
        mgmt = management_richness(t, idx)
        bw = bodyweight_density(t, idx)

        block = pd.concat([sp, arable.rename("fx_is_arable"), flow, mgmt, bw], axis=1)
        block["tower"] = t
        block.index.name = "Datetime"
        new_frames.append(block.reset_index())

    new_cols_df = pd.concat(new_frames, ignore_index=True)
    v3 = v2.merge(new_cols_df, on=["Datetime", "tower"], how="left")

    assert len(v3) == len(v2), f"row count changed: {len(v2)} -> {len(v3)}"
    for c in v2.columns:
        pd.testing.assert_series_equal(v2[c], v3[c], check_names=False)
    print("Verified: row count unchanged, every pre-existing v2 column byte-identical post-merge.")

    new_cols = [c for c in new_cols_df.columns if c not in ("Datetime", "tower")]
    nan_cols = [c for c in new_cols if v3[c].isna().all()]
    print(f"New columns added: {new_cols}")
    print("All-NaN new columns:", nan_cols if nan_cols else "none")

    print("\nLand-use flip dates (fx_is_arable):")
    for t, fd in flip_dates.items():
        print(f"  Tower {t}: {fd if fd is not None else 'never flips (fx_is_arable=0 throughout)'}")

    print("\nPer-tower non-null coverage of new fx_ columns (%):")
    for t in TOWERS:
        sub = v3[v3.tower == t]
        cov = (sub[new_cols].notna().mean() * 100).round(1)
        print(f"  Tower {t}:\n{cov.to_string()}")

    dest = HOURLY / "forecast_daily_v3.csv"
    v3.to_csv(dest, index=False)
    print(f"\nWrote {dest} {v3.shape}")


if __name__ == "__main__":
    main()
