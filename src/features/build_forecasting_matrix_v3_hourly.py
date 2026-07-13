"""F-10 (D-67) Stage 2b extension: hourly-track v3 matrix, needed to test TFT/DLinear/LSTM with
the same 5 new feature families the daily track (`build_forecasting_matrix_v3.py`) already has.
Additive clone of `build_forecasting_matrix_v2.py`'s `hourly_new()` pattern -- reads
`forecast_features_v2.csv` (read-only) and left-merges new `fx_` columns onto it, writes
`data/Hourly/forecast_features_v3.csv`. `forecasting_dl.py` is NOT edited: it already
auto-detects feature columns via `[c for c in m.columns if c.startswith("fx")]`
(`load_matrix()`), so adding correctly-named `fx_` columns to the matrix is sufficient -- no
code change needed there.

Hourly convention differs from the daily track's lag/roll-heavy convention: v2's own hourly `fx_`
columns (fx_SWIN_1_1_1, fx_TA_0_0_1, etc.) are all RAW current-hour readings, no lags/rolls at
hourly resolution -- matched here for consistency:
  (a) species: raw hourly per-species head-density (head/ha), no daily resample.
  (b) fx_is_arable: the daily flag broadcast across every hour of that day (ffill).
  (c) flow: raw hourly Flow (l/s) reading, no lag/roll (matches the other hourly fx_ columns'
      own "point reading, not smoothed" convention).
  (d) mgmt: management_features.csv's columns are ALREADY hourly-native (exp-decay recency
      computed directly on the hourly timeline) -- just selected, not resampled.
  (e) bonus liveweight density: daily value broadcast across every hour of that day (the
      underlying location/weight records are daily-resolution at best, so hourly granularity
      beyond broadcast isn't meaningful).

Run from project root:  python src/features/build_forecasting_matrix_v3_hourly.py
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
from gapfill_rfm import load_ext, LSU, AREA, TOWERS, C4  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
HOURLY = ROOT / "data" / "Hourly"

FLOW = "Flow (l/s)"


def cat_str(t):
    return C4 if t == 4 else f"Catchment {t}"


def species_hourly(t, d):
    cat = cat_str(t)
    out = pd.DataFrame(index=d.index)
    for s in LSU:
        col = f"{s}_{cat}"
        out[f"fx_{s}_dens"] = d[col].fillna(0) / AREA[t]
    return out


def flow_hourly(t, d):
    cat = cat_str(t)
    col = f"{FLOW} [{cat}]"
    return pd.DataFrame({"fx_flow_mean": d[col]}, index=d.index)


def mgmt_hourly(t, idx):
    mf = pd.read_csv(HOURLY / "management_features.csv", low_memory=False)
    mf["Datetime"] = pd.to_datetime(mf["Datetime"], format="mixed")
    mf = mf.set_index("Datetime")
    cols = [f"mgmt_t{t}_{ch}" for ch in
            ("fertN_recency", "fertN_rate", "lime_recency", "cultiv_recency",
             "cut_recency", "manure_recency")]
    out = mf[cols].reindex(idx)
    out.columns = [c.replace(f"mgmt_t{t}_", "fx_mgmt_") for c in cols]
    return out


def arable_hourly(t, idx):
    """Broadcast the daily fx_is_arable flag (already built in forecast_daily_v3.csv) to hourly."""
    daily = pd.read_csv(HOURLY / "forecast_daily_v3.csv", low_memory=False)
    daily["Datetime"] = pd.to_datetime(daily["Datetime"], format="mixed")
    sub = daily[daily.tower == t].set_index("Datetime")["fx_is_arable"]
    return pd.DataFrame({"fx_is_arable": sub.reindex(idx.normalize()).to_numpy()}, index=idx)


def bodyweight_hourly(t, idx):
    path = HOURLY / "bodyweight_density.csv"
    bw = pd.read_csv(path, parse_dates=["Datetime"])
    sub = bw[bw.tower == t].set_index("Datetime")["fx_total_liveweight_dens"]
    daily_full = sub.reindex(pd.DatetimeIndex(idx.normalize().unique()))
    return pd.DataFrame({"fx_total_liveweight_dens": daily_full.reindex(idx.normalize()).to_numpy()}, index=idx)


def main():
    v2h = pd.read_csv(HOURLY / "forecast_features_v2.csv", low_memory=False)
    v2h["Datetime"] = pd.to_datetime(v2h["Datetime"], format="mixed")
    print(f"Loaded forecast_features_v2.csv {v2h.shape}")

    d = load_ext()
    print(f"Loaded EXT (raw hourly) {d.shape}")

    new_frames = []
    for t in TOWERS:
        idx = pd.DatetimeIndex(v2h.loc[v2h.tower == t, "Datetime"].unique()).sort_values()

        sp = species_hourly(t, d).reindex(idx)
        arable = arable_hourly(t, idx)
        flow = flow_hourly(t, d).reindex(idx)
        mgmt = mgmt_hourly(t, idx)
        bw = bodyweight_hourly(t, idx)

        block = pd.concat([sp, arable, flow, mgmt, bw], axis=1)
        block["tower"] = t
        block.index.name = "Datetime"
        new_frames.append(block.reset_index())

    new_cols_df = pd.concat(new_frames, ignore_index=True)
    v3h = v2h.merge(new_cols_df, on=["Datetime", "tower"], how="left")

    assert len(v3h) == len(v2h), f"row count changed: {len(v2h)} -> {len(v3h)}"
    for c in v2h.columns:
        pd.testing.assert_series_equal(v2h[c], v3h[c], check_names=False)
    print("Verified: row count unchanged, every pre-existing v2 column byte-identical post-merge.")

    new_cols = [c for c in new_cols_df.columns if c not in ("Datetime", "tower")]
    nan_cols = [c for c in new_cols if v3h[c].isna().all()]
    print(f"New columns added: {new_cols}")
    print("All-NaN new columns:", nan_cols if nan_cols else "none")

    print("\nPer-tower non-null coverage of new fx_ columns (%):")
    for t in TOWERS:
        sub = v3h[v3h.tower == t]
        cov = (sub[new_cols].notna().mean() * 100).round(1)
        print(f"  Tower {t}:\n{cov.to_string()}")

    dest = HOURLY / "forecast_features_v3.csv"
    v3h.to_csv(dest, index=False)
    print(f"\nWrote {dest} {v3h.shape}")


if __name__ == "__main__":
    main()
