"""Summarise Chapter 3 environmental-driver coverage before and after filling.

The calculation follows the production external SMS/MET path used by the
gap-filling experiments. "Before" coverage is measured after the same
plausibility filter applied immediately before filling; "after" coverage is
read from the resulting ``__f`` columns. Both are restricted to the common
2017--2023 EC analysis window.
"""

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
HOURLY = ROOT / "data" / "Hourly"
OUTPUT = ROOT / "results" / "ch3_environmental_gapfill_availability.csv"
START = pd.Timestamp("2017-01-01")
END = pd.Timestamp("2024-01-01")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_sms_met_dataset import ext_driver_map  # noqa: E402
from reddyproc_pipeline import C4, TOWERS, plausibility_filter  # noqa: E402


LABELS = {
    "sw": "Shortwave radiation",
    "ta": "Air temperature",
    "vpd": "Vapour-pressure deficit",
    "ppfd": "Photosynthetic photon flux",
    "rn": "Net radiation",
    "ws": "Wind speed",
    "ustar": "Friction velocity",
    "shf": "Soil heat flux",
    "precip": "Precipitation",
    "swc": "Soil moisture",
    "ts": "Soil temperature",
}


def load_window(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, low_memory=False)
    data["Datetime"] = pd.to_datetime(data["Datetime"], format="mixed")
    data = data.set_index("Datetime")
    return data.loc[(data.index >= START) & (data.index < END)]


def main() -> None:
    source = load_window(HOURLY / "consolidated_hourly_SMS_MET.csv")
    filled = load_window(HOURLY / "reddyproc_processed_SMS_MET.csv").reindex(
        source.index
    )
    n_hours = len(source)
    rows = []

    for tower in TOWERS:
        catchment = C4 if tower == 4 else f"Catchment {tower}"
        for key, column in ext_driver_map(tower, catchment).items():
            raw = pd.to_numeric(source[column], errors="coerce")
            input_series = plausibility_filter(raw, column)
            output_series = pd.to_numeric(
                filled[f"{column}__f"], errors="coerce"
            )
            raw_n = int(raw.notna().sum())
            input_n = int(input_series.notna().sum())
            output_n = int(output_series.notna().sum())
            rows.append(
                {
                    "tower": f"T{tower}",
                    "driver": LABELS[key],
                    "source_column": column,
                    "window_hours": n_hours,
                    "raw_available_h": raw_n,
                    "raw_available_pct": 100 * raw_n / n_hours,
                    "input_after_plausibility_h": input_n,
                    "input_after_plausibility_pct": 100 * input_n / n_hours,
                    "filled_available_h": output_n,
                    "filled_available_pct": 100 * output_n / n_hours,
                    "hours_added_by_filling": output_n - input_n,
                }
            )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT, index=False)
    print(f"Wrote {OUTPUT} ({len(rows)} rows; {n_hours:,} hourly timestamps)")


if __name__ == "__main__":
    main()
