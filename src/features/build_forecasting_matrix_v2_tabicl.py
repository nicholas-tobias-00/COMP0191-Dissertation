"""TabICL-sourced daily forecasting features (D-79) -- same guide (fx_) feature engineering as
`build_forecasting_matrix_v2.py`'s `daily_table()`, reused unchanged (it already takes `gf` as a
parameter), but with `y_gapfilled`/`ar_ch4_*` derived from TabICL-solo's gap-filled series
(`fch4_gapfilled_tabicl.csv`) instead of the RFm champion's (`fch4_gapfilled.csv`). Lets
forecasting experiments compare a TabICL-sourced AR feature set against the existing RF-sourced
one without touching `build_forecasting_matrix_v2.py` or any of its outputs.

Only the daily track is reproduced here -- `forecast_features_v2.csv` (hourly) does not depend on
the gap-filled CH4 series at all (its new fx_ columns come straight from raw met/frame data), so
there is nothing TabICL-specific to add there.

Run `build_fch4_gapfilled_tabicl.py` first if `fch4_gapfilled_tabicl.csv` doesn't exist yet.

Output: data/Hourly/forecast_daily_v2_tabicl.csv
Run from project root:  python src/features/build_forecasting_matrix_v2_tabicl.py
"""
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gapfill_rfm import load_ext, frame, TOWERS  # noqa: E402
from build_forecasting_matrix_v2 import daily_table  # noqa: E402  reused unchanged, gf is a param

HOURLY = Path(__file__).resolve().parents[2] / "data" / "Hourly"


def main():
    d = load_ext()
    gf_path = HOURLY / "fch4_gapfilled_tabicl.csv"
    if not gf_path.exists():
        raise FileNotFoundError(f"{gf_path} not found -- run build_fch4_gapfilled_tabicl.py first")
    gf = pd.read_csv(gf_path, low_memory=False)
    gf["Datetime"] = pd.to_datetime(gf["Datetime"], format="mixed"); gf = gf.set_index("Datetime")
    print(f"Loaded EXT {d.shape}, TabICL gap-filled CH4 {gf.shape}")

    frames = {t: frame(t, pooled=True, d=d).sort_index() for t in TOWERS}
    daily = pd.concat([daily_table(t, d, frames[t], gf) for t in TOWERS], ignore_index=True)

    fx_b = [c for c in daily.columns if c.startswith("fx")]
    ar_b = [c for c in daily.columns if c.startswith("ar_")]
    nan_d = [c for c in fx_b + ar_b if daily[c].isna().all()]
    dest = HOURLY / "forecast_daily_v2_tabicl.csv"
    daily.to_csv(dest, index=False)
    print(f"daily v2 (TabICL) {daily.shape}: {len(fx_b)} fx_ + {len(ar_b)} ar_ features")
    for t in TOWERS:
        sub = daily[daily.tower == t]
        print(f"  Tower {t}: {len(sub):>5} days, observed {int(sub.y_observed.notna().sum()):>4}")
    print("  all-NaN feature cols:", nan_d if nan_d else "none")
    print(f"\nWrote {dest}")


if __name__ == "__main__":
    main()
