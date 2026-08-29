"""S-03c: one more gate check before committing to the expensive Phase 6 production grid.

S-03b validated `Direct_TabICLv2_raw` POOLED across towers (tower-dummy static features) + a
time-trend static feature (`b17_days_since_2010`). Neither is appropriate for S-05/S-06's actual
production pipeline: (a) S-05/S-06 use PER-TOWER anchors (`tower_anchor()`, each tower's own last-
real-data date), so pooling with a single shared cutoff either wastes real data or leaks future data
depending on which tower's anchor is used -- solo per-tower fits are the clean, faithful choice
(also consistent with D-94's prior finding that TabICLv2 barely benefits from pooling anyway,
0.7355 vs 0.7353); (b) a "days since 2010" trend feature would be extrapolated to ~2050 in the real
scenario grid -- 30+ years beyond its ~12-year real training range, a genuine, never-tested
extrapolation risk, and the feature S-05/S-06's own `tabicl_forecast()` control never had either.

This checks the ACTUAL config Phase 6 will use: solo per-tower Direct_TabICLv2 regression, pure
`FX_A_SPECIES` (13 cols, no static features at all) -- against the same control and same 5-anchor
real backtest as S-03b, so the comparison stays apples-to-apples.

Run from project root:  python notebooks/07_scenario_analysis/s03c_solo_notrend_check.py
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

import models.recursive_rollout as rr
import B17_foundation_screen as b17s
import B17_direct_and_recursive_foundations as b17d
from build_transient_scenario_drivers_species import FX_A_SPECIES
import s03b_driver_availability_b18 as s03b

RESULTS = ROOT / "results"
CHAINS_PATH = RESULTS / "s03c_solo_notrend_chains.csv"
SUMMARY_PATH = RESULTS / "s03c_solo_notrend_summary.csv"

TOWERS = b17s.TOWERS
ANCHOR_YEARS = b17s.ANCHOR_YEARS
N_DAYS = b17s.N_DAYS


def run_solo(frame):
    """Per-tower solo fit, pure FX_A_SPECIES, no static features -- the exact Phase 6 mechanism."""
    from tabicl import TabICLRegressor

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
                x_train = imputer.fit_transform(train[FX_A_SPECIES])
                model = TabICLRegressor(n_estimators=8, random_state=42)
                model.fit(x_train, train["y_observed"].to_numpy())
                x_future = imputer.transform(future[FX_A_SPECIES])
                prediction = model.predict(x_future, output_type="median")
            except Exception as e:
                print(f"    T{tower} {yr} SKIPPED: {str(e)[:150]}")
                continue

            for d, p in zip(future.index, prediction):
                rows.append(
                    {
                        "candidate": "Direct_TabICLv2_solo_notrend",
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
    s03b.FULL_FRAME = frame

    print("=" * 70 + "\nCANDIDATE: Direct_TabICLv2_solo_notrend (the actual Phase 6 mechanism)\n" + "=" * 70)
    new_candidate = run_solo(frame)

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
