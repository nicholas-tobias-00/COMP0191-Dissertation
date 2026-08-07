"""TabICL-sourced enriched daily forecasting features (D-79) -- same guide-feature enrichment as
`build_forecasting_matrix_v3.py` (species density, arable flag, catchment flow, management
richness, liveweight density -- none of which depend on which model produced the CH4 gap-fill),
reused unchanged, applied on top of `forecast_daily_v2_tabicl.csv` instead of
`forecast_daily_v2.csv`. This is the TabICL-sourced analogue of `forecast_daily_v3.csv`.

Run `build_forecasting_matrix_v2_tabicl.py` first if `forecast_daily_v2_tabicl.csv` doesn't exist
yet.

Output: data/Hourly/forecast_daily_v3_tabicl.csv
Run from project root:  python src/features/build_forecasting_matrix_v3_tabicl.py
"""
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gapfill_rfm import load_ext, TOWERS  # noqa: E402
from build_forecasting_matrix_v3 import (  # noqa: E402  reused unchanged
    species_density, is_arable_flag, flow_daily, management_richness, bodyweight_density,
)

HOURLY = Path(__file__).resolve().parents[2] / "data" / "Hourly"


def main():
    v2_path = HOURLY / "forecast_daily_v2_tabicl.csv"
    if not v2_path.exists():
        raise FileNotFoundError(f"{v2_path} not found -- run build_forecasting_matrix_v2_tabicl.py first")
    v2 = pd.read_csv(v2_path, low_memory=False)
    v2["Datetime"] = pd.to_datetime(v2["Datetime"], format="mixed")
    print(f"Loaded forecast_daily_v2_tabicl.csv {v2.shape}")

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
    print("Verified: row count unchanged, every pre-existing v2(tabicl) column byte-identical post-merge.")

    new_cols = [c for c in new_cols_df.columns if c not in ("Datetime", "tower")]
    nan_cols = [c for c in new_cols if v3[c].isna().all()]
    print(f"New columns added: {new_cols}")
    print("All-NaN new columns:", nan_cols if nan_cols else "none")

    print("\nLand-use flip dates (fx_is_arable):")
    for t, fd in flip_dates.items():
        print(f"  Tower {t}: {fd if fd is not None else 'never flips (fx_is_arable=0 throughout)'}")

    dest = HOURLY / "forecast_daily_v3_tabicl.csv"
    v3.to_csv(dest, index=False)
    print(f"\nWrote {dest} {v3.shape}")


if __name__ == "__main__":
    main()
