"""S-03d: disentangles WHY S-03b's pooled+trend config won by 4.4% while S-03c's solo+no-trend
config barely beat control (0.6%). Isolates the trend-feature contribution alone: solo per-tower
(same as S-03c), but WITH a `days_since_2010`-style trend feature added back (same as S-03b had).
If this recovers most of S-03b's margin, trend was the driver (and Phase 6 faces a real extrapolation-
risk decision, since 2050 is 30+ years beyond training range for that feature). If it doesn't, pooling
was the driver (and Phase 6 needs a different pooling strategy, e.g. shared min-anchor).

Run from project root:  python notebooks/07_scenario_analysis/s03d_solo_trend_check.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

ROOT = Path(r"C:\Users\Nicholas\Documents\COMP0191 MSc Artificial Intelligence for Sustainable Development Project")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "features"))
sys.path.insert(0, str(ROOT / "notebooks" / "05_benchmarking"))
sys.path.insert(0, str(ROOT / "notebooks" / "07_scenario_analysis"))

import B17_foundation_screen as b17s
import B17_direct_and_recursive_foundations as b17d
from build_transient_scenario_drivers_species import FX_A_SPECIES
import s03b_driver_availability_b18 as s03b

RESULTS = ROOT / "results"
CHAINS_PATH = RESULTS / "s03d_solo_trend_chains.csv"
SUMMARY_PATH = RESULTS / "s03d_solo_trend_summary.csv"

TOWERS = b17s.TOWERS
ANCHOR_YEARS = b17s.ANCHOR_YEARS
N_DAYS = b17s.N_DAYS
TREND_COL = "b17_days_since_2010"


def run_solo_trend(frame):
    from tabicl import TabICLRegressor

    features = FX_A_SPECIES + [TREND_COL]
    T = {t: frame.loc[frame["tower"].eq(t)].set_index("Datetime").sort_index() for t in TOWERS}
    rows = []
    for tower in TOWERS:
        dft = T[tower]
        for yr in ANCHOR_YEARS:
            anchor = pd.Timestamp(f"{yr}-12-16")
            dates = pd.date_range(anchor + pd.Timedelta(days=1), periods=N_DAYS, freq="D")
            train = dft.loc[:anchor].loc[dft.loc[:anchor, "y_observed"].notna()]
            future = dft.loc[dft.index.isin(dates)]
            if len(train) > 10_000:
                train = train.sample(10_000, random_state=42).sort_values("Datetime")
            if train.empty or future.empty:
                continue

            t0 = time.time()
            try:
                imputer = SimpleImputer(strategy="mean")
                x_train = imputer.fit_transform(train[features])
                model = TabICLRegressor(n_estimators=8, random_state=42)
                model.fit(x_train, train["y_observed"].to_numpy())
                x_future = imputer.transform(future[features])
                prediction = model.predict(x_future, output_type="median")
            except Exception as e:
                print(f"    T{tower} {yr} SKIPPED: {str(e)[:150]}")
                continue

            for d, p in zip(future.index, prediction):
                rows.append(
                    {
                        "candidate": "Direct_TabICLv2_solo_trend",
                        "anchor_year": yr,
                        "tower": tower,
                        "date": d,
                        "y_predict": p,
                        "y_true": future.loc[d, "y_observed"],
                        "y_gapfilled": future.loc[d, "y_gapfilled"],
                    }
                )
            print(f"    T{tower} {yr} done ({time.time() - t0:.0f}s)")
    return pd.DataFrame(rows)


def main():
    frame = pd.read_csv(b17s.DATA_PATH, low_memory=False)
    frame["Datetime"] = pd.to_datetime(frame["Datetime"], format="mixed")
    frame = b17d.add_b17_features(frame)
    print(f"[S-03d] trend feature range in training data: "
          f"{frame[TREND_COL].min():.0f} to {frame[TREND_COL].max():.0f} days since 2010")
    s03b.FULL_FRAME = frame

    print("=" * 70 + "\nCANDIDATE: Direct_TabICLv2_solo_trend (solo per-tower + trend, no pooling)\n" + "=" * 70)
    new_candidate = run_solo_trend(frame)

    print("=" * 70 + "\nCANDIDATE: control (current production tabicl_forecast)\n" + "=" * 70)
    control = s03b.run_control(frame)

    chains = pd.concat([new_candidate, control], ignore_index=True)
    chains.to_csv(CHAINS_PATH, index=False)
    print(f"\n[OK] Saved {CHAINS_PATH.name} ({len(chains)} rows)")

    summary, bins = s03b.score(chains)
    summary.to_csv(SUMMARY_PATH, index=False)
    print(f"\n[OK] Saved {SUMMARY_PATH.name}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
