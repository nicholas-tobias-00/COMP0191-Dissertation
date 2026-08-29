"""S-05, extended to 2050 (user-requested follow-up to s05_trajectory_10yr.py): same 8,100-call
grid (3 towers x 2 SSPs x 5 GCMs x 10 realizations/GCM x 27 species-multiplier combos), but the
horizon now runs from each tower's anchor to 2050 (T4/T9: 2024-2050, 27 years/9,855 days; T2:
2020-2050, 31 years/11,315 days -- matches S-04's own 2050 endpoint) instead of a fixed 10 years.

Two real consequences of the horizon extension, both measured empirically before committing to
this scope (not assumed):
  - Every one of the 8,100 calls is now slower (a single 27-year call measured at 4.07s vs. the
    original 10-year call's ~1.2-1.3s warm) -- ~9h total, not ~2.5h. The horizon extension, not the
    daily-chain save below, is what dominates the new cost.
  - Also saves FULL DAILY chains for every call this time (not just annual_mean) -- since compute
    is already the bottleneck, the marginal cost of also persisting daily rows is negligible
    (I/O next to inference time), so both were requested together and done in one pass rather than
    two separate runs. At this scale (~83.8M rows across all 8,100 calls, T2's longer horizon
    included) CSV is impractical (~6GB, measured via extrapolation from the 10-year subset's own
    72.7 bytes/row) -- written incrementally to Parquet via pyarrow.parquet.ParquetWriter (7.5x
    smaller, measured directly on the same subset) so the full 83.8M rows are never held in memory
    at once, only one (tower, ssp, gcm, realization) batch (27 combos x ~9,855-11,315 days) at a time.

This SUPERSEDES s05_trajectory_10yr.py's results (not just extends them) -- the species-response/
joint-additive/realization-spread/AOA findings all get recomputed against this new horizon.
s05_trajectory_10yr.py itself is left unmodified/uncommitted-over for reference.

Run from project root:  python notebooks/07_scenario_analysis/s05_trajectory_2050.py
Smoke test: python notebooks/07_scenario_analysis/s05_trajectory_2050.py smoke
"""
import itertools
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = r"c:\Users\Nicholas\Documents\COMP0191 MSc Artificial Intelligence for Sustainable Development Project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\src")
sys.path.insert(0, ROOT + r"\src\features")

import models.recursive_rollout as rr
from build_transient_scenario_drivers_species import (
    FX_A_SPECIES, SPECIES, build_climatology_base_species, overlay_transient_species,
    load_transient_years, stratified_realizations,
)

HOURLY = rf"{ROOT}\data\Hourly"
RESULTS = rf"{ROOT}\results"

TOWERS = [2, 4, 9]
SSPS = ["ssp245", "ssp585"]
END_YEAR = 2050
MULT_LEVELS = [1.0, 2.0, 3.0]
MULT_COMBOS = list(itertools.product(MULT_LEVELS, repeat=3))  # (cattle, sheep, lamb), 27 combos

DAILY_SCHEMA = pa.schema([
    ("timestamp", pa.timestamp("ns")), ("pred", pa.float32()),
    ("tower", pa.int8()), ("ssp", pa.string()), ("gcm", pa.string()), ("realization", pa.int16()),
    ("mult_cattle", pa.float32()), ("mult_sheep", pa.float32()), ("mult_lamb", pa.float32()),
])


def load_towers():
    dv = pd.read_csv(f"{HOURLY}/forecast_daily_v3.csv", low_memory=False)
    dv["Datetime"] = pd.to_datetime(dv["Datetime"], format="mixed")
    return {t: dv[dv.tower == t].set_index("Datetime").sort_index() for t in TOWERS}


def tower_anchor(T, tower):
    dft = T[tower]
    return dft.loc[dft["y_observed"].notna()].index.max()


def precompute_aoa(T, tower):
    dft = T[tower]
    X_train = dft[FX_A_SPECIES].dropna().values
    scaler = StandardScaler().fit(X_train)
    Xtr = scaler.transform(X_train)
    d_train_matrix = cdist(Xtr, Xtr)
    np.fill_diagonal(d_train_matrix, np.inf)
    d_train_loo = d_train_matrix.min(axis=1)
    q1, q3 = np.percentile(d_train_loo, [25, 75])
    threshold = q3 + 1.5 * (q3 - q1)
    return scaler, Xtr, threshold


def aoa_flagged_frac(scaler, Xtr, threshold, X_scenario):
    Xsc = scaler.transform(X_scenario)
    d_scenario = cdist(Xsc, Xtr).min(axis=1)
    return float((d_scenario > threshold).mean())


def main(n_per_gcm=10, ssps=None, towers=None, run_label="", realizations=None, mult_combos=None,
         end_year=END_YEAR):
    ssps = SSPS if ssps is None else ssps
    towers = TOWERS if towers is None else towers
    mult_combos_local = MULT_COMBOS if mult_combos is None else mult_combos
    suffix = f"_{run_label}" if run_label else ""

    T = load_towers()
    realizations = stratified_realizations(n_per_gcm) if realizations is None else realizations
    n_total = len(towers) * len(ssps) * len(realizations) * len(mult_combos_local)
    print(f"[S-05/2050] {len(towers)} towers x {len(ssps)} SSPs x {len(realizations)} "
          f"(GCM,realization) pairs x {len(mult_combos_local)} multiplier combos = {n_total} calls")

    daily_path = f"{RESULTS}/s05_daily_chains_2050{suffix}.parquet"
    writer = pq.ParquetWriter(daily_path, DAILY_SCHEMA)

    all_rows = []
    t0 = time.time()
    n_done = 0

    try:
        for tower in towers:
            dft = T[tower]
            anchor = tower_anchor(T, tower)
            hist_target = dft.loc[:anchor, "y_observed"]
            hist_cov = dft.loc[:anchor, FX_A_SPECIES]
            years = list(range(anchor.year + 1, end_year + 1))
            print(f"\n=== Tower {tower}: anchor={anchor.date()}, years={years[0]}-{years[-1]} "
                  f"({len(years)} years, {len(years)*365} days) ===")

            scaler, Xtr, aoa_thresh = precompute_aoa(T, tower)
            clim_cache = {yr: build_climatology_base_species(tower, T, yr) for yr in years}

            for ssp in ssps:
                for gcm, real in realizations:
                    t_call0 = time.time()
                    try:
                        tyears = load_transient_years(gcm, ssp, real, years)
                    except FileNotFoundError as e:
                        print(f"  SKIPPED (file not found): {e}")
                        continue

                    for mc, cattle, sheep, lamb in [(mc, *mc) for mc in mult_combos_local]:
                        year_frames = [overlay_transient_species(clim_cache[yr], tyears[yr], cattle, sheep, lamb)
                                       for yr in years]
                        frame = pd.concat(year_frames)[FX_A_SPECIES]

                        chain = rr.tabicl_forecast(hist_target, hist_cov, frame)

                        # ---- daily rows -> Parquet (streamed, never held for the full grid) ----
                        daily_table = pa.table({
                            "timestamp": pa.array(chain.index.values, type=pa.timestamp("ns")),
                            "pred": pa.array(chain.values, type=pa.float32()),
                            "tower": pa.array([tower] * len(chain), type=pa.int8()),
                            "ssp": pa.array([ssp] * len(chain), type=pa.string()),
                            "gcm": pa.array([gcm] * len(chain), type=pa.string()),
                            "realization": pa.array([real] * len(chain), type=pa.int16()),
                            "mult_cattle": pa.array([cattle] * len(chain), type=pa.float32()),
                            "mult_sheep": pa.array([sheep] * len(chain), type=pa.float32()),
                            "mult_lamb": pa.array([lamb] * len(chain), type=pa.float32()),
                        }, schema=DAILY_SCHEMA)
                        writer.write_table(daily_table)

                        # ---- annual_mean + AOA -> summary CSV (same as s05_trajectory_10yr.py) ----
                        chain_df = chain.to_frame("pred")
                        chain_df["nominal_year"] = [d.year for d in frame.index]
                        for yr, g in chain_df.groupby("nominal_year"):
                            yr_frame = frame.loc[g.index]
                            aoa_pct = aoa_flagged_frac(scaler, Xtr, aoa_thresh, yr_frame.values) * 100
                            all_rows.append({
                                "tower": tower, "ssp": ssp, "gcm": gcm, "realization": real,
                                "mult_cattle": cattle, "mult_sheep": sheep, "mult_lamb": lamb,
                                "year": yr, "annual_mean": float(g["pred"].mean()),
                                "aoa_flagged_pct": aoa_pct,
                            })

                        n_done += 1

                    dt = time.time() - t_call0
                    elapsed = time.time() - t0
                    remaining = n_total - n_done
                    rate = n_done / elapsed if elapsed > 0 else 0
                    eta_h = (remaining / rate / 3600) if rate > 0 else float("nan")
                    print(f"  T{tower} {ssp} {gcm}/{real}: {len(mult_combos_local)} combos in {dt:.1f}s "
                          f"({n_done} done, elapsed {elapsed/3600:.2f}h, ETA {eta_h:.2f}h)")
    finally:
        writer.close()

    out = pd.DataFrame(all_rows)
    out.to_csv(f"{RESULTS}/s05_trajectory_realizations_2050{suffix}.csv", index=False)
    print(f"\n[OK] Saved s05_trajectory_realizations_2050{suffix}.csv ({len(out)} rows)")
    print(f"[OK] Saved s05_daily_chains_2050{suffix}.parquet")
    print(f"Total {time.time()-t0:.0f}s")
    return out


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        main(towers=[4], ssps=["ssp245"], realizations=[("ACCESS-ESM1-5", 1), ("ACCESS-ESM1-5", 2)],
             mult_combos=[(1.0, 1.0, 1.0), (2.0, 1.0, 1.0), (1.0, 2.0, 3.0)], run_label="smoketest")
    else:
        main(n_per_gcm=10)
