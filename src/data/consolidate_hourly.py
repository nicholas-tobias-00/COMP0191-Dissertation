"""
Consolidate all compiled NWFP datasets to a common 1-hour resolution.

Aggregation rules:
  - Sub-hourly (15-min measurements, 30-min greenhouse): resample → hourly mean.
  - Daily (livestock location counts): upsample → replicate value to each hour
    of that day using forward-fill with limit=23 so values never bleed across
    a day boundary into a day that is genuinely absent.
  - Non-numeric columns (quality strings, last-modified timestamps) are dropped.
  - NaN values are PRESERVED — this is not gap-filling.

Outputs written to data/Hourly/:
  greenhouse_hourly.csv    EC fluxes + met, 1h
  measurements_hourly.csv  Flow + soil moisture per catchment, 1h
  livestock_hourly.csv     Head counts per location per species, 1h
  consolidated_hourly.csv  All sources merged on a single 1h index

Run from the project root:
    python -m src.data.consolidate_hourly
    python src/data/consolidate_hourly.py
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd

COMPILED = Path("data/Compiled")
HOURLY   = Path("data/Hourly")


# ── helpers ──────────────────────────────────────────────────────────────────

def _to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce every column to numeric, dropping those that are entirely non-numeric
    (i.e. quality-flag strings like 'Acceptable'/'Not set' and ISO-8601 timestamp
    strings from 'Quality Last Modified' columns).
    """
    df = df.apply(pd.to_numeric, errors="coerce")
    return df.dropna(axis=1, how="all")


# ── loaders ──────────────────────────────────────────────────────────────────

def load_sub_hourly(path: Path, datetime_col: str = "Datetime") -> pd.DataFrame:
    """
    Load a sub-hourly CSV and return hourly means.

    Hours with at least one valid reading → mean of those readings.
    Hours with zero valid readings (all NaN in the window) → NaN.
    """
    print(f"  {path.name} ...", end=" ", flush=True)
    df = pd.read_csv(path, parse_dates=[datetime_col], low_memory=False)
    df = df.sort_values(datetime_col).set_index(datetime_col)
    df = _to_numeric(df)
    hourly = df.resample("1h").mean()
    print(f"{len(df):,} rows ->{len(hourly):,} hourly, {hourly.shape[1]} cols")
    return hourly


def load_daily(path: Path, date_col: str = "Date",
               col_prefix: str = "") -> pd.DataFrame:
    """
    Load a daily CSV (rows = dates, cols = location/value columns) and expand
    to hourly by replicating each day's value across its 24 hours.

    limit=23 in ffill: each midnight value propagates at most 23 steps forward,
    so a missing day (gap ≥ 24 h from the last non-NaN midnight) stays NaN.
    """
    print(f"  {path.name} ...", end=" ", flush=True)
    df = pd.read_csv(path)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col)
    df = _to_numeric(df)

    # Ensure one row per calendar day (deduplicate if needed)
    daily = df.resample("D").mean()
    # Upsample to 1 h, propagating midnight value for up to 23 h (within-day only)
    hourly = daily.resample("1h").ffill(limit=23)

    if col_prefix:
        hourly.columns = [f"{col_prefix}_{c}" for c in hourly.columns]
    hourly.index.name = "Datetime"

    print(f"{len(daily):,} daily ->{len(hourly):,} hourly, {hourly.shape[1]} cols")
    return hourly


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    HOURLY.mkdir(parents=True, exist_ok=True)

    sep = "=" * 64
    print(sep)
    print("NWFP Hourly Consolidation")
    print(sep)

    # 1. Greenhouse EC — 30-min -> 1 h
    print("\n[1/4] Greenhouse EC (30-min -> 1h)")
    gh = load_sub_hourly(COMPILED / "greenhouse.csv")
    gh.to_csv(HOURLY / "greenhouse_hourly.csv")
    print(f"       saved: {HOURLY / 'greenhouse_hourly.csv'}")

    # 2. Environmental measurements — 15-min -> 1 h
    print("\n[2/4] Environmental measurements (15-min -> 1h)")
    ms = load_sub_hourly(COMPILED / "measurements.csv")
    ms.to_csv(HOURLY / "measurements_hourly.csv")
    print(f"       saved: {HOURLY / 'measurements_hourly.csv'}")

    # 3. Livestock location counts — daily -> 1 h
    print("\n[3/4] Livestock counts (daily -> 1h)")
    cattle = load_daily(
        COMPILED / "Animal_location_counts_Cattle_Basic_Data.csv",
        col_prefix="cattle",
    )
    sheep = load_daily(
        COMPILED / "Animal_location_counts_Breeding_Sheep_Basic_Data.csv",
        col_prefix="sheep",
    )
    lambs = load_daily(
        COMPILED / "Animal_location_counts_Lamb_Basic_Data.csv",
        col_prefix="lamb",
    )
    livestock = cattle.join(sheep, how="outer").join(lambs, how="outer")
    livestock.to_csv(HOURLY / "livestock_hourly.csv")
    print(f"       saved: {HOURLY / 'livestock_hourly.csv'}")

    # 4. Merge all sources on common 1-h index (outer join keeps all timestamps)
    print("\n[4/4] Merging on common 1h index")
    consolidated = gh.join(ms, how="outer").join(livestock, how="outer")
    consolidated.index.name = "Datetime"
    consolidated.to_csv(HOURLY / "consolidated_hourly.csv")
    print(f"       saved: {HOURLY / 'consolidated_hourly.csv'}")

    print(f"\n{sep}")
    print("Summary")
    print(sep)
    for label, df in [
        ("greenhouse_hourly.csv   ", gh),
        ("measurements_hourly.csv ", ms),
        ("livestock_hourly.csv    ", livestock),
        ("consolidated_hourly.csv ", consolidated),
    ]:
        nan_pct = df.isna().mean().mean() * 100
        print(f"  {label}  {df.shape[0]:>7,} rows x {df.shape[1]:>4} cols  "
              f"  overall NaN: {nan_pct:.1f}%")
    print(f"\n  Output: {HOURLY.resolve()}")


if __name__ == "__main__":
    main()
