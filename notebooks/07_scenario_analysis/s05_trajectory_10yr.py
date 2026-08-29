"""S-05: TabICLv2 + S-03's Variant A (driver-removal) feature set, extended to F-10/D-67's
species-disaggregated livestock density, run as a 10-year transient CMIP6 trajectory -- the
combination of three prior pieces of work, not a new pipeline built from scratch:

  - S-03's Variant A: drops the 24 scenario-unavailable columns entirely, keeping only what a
    CMIP6 scenario can actually supply (TA/SWIN/PRECIP/DOY/season/livestock).
  - F-10/D-67's species split: fx_cattle_dens/fx_sheep_dens/fx_lamb_dens instead of (here: in
    addition to) the single aggregate fx_lsu_dens -- the same family that makes TabPFN+species
    this project's standing forecasting champion.
  - S-04's transient scenario machinery: real CMIP6 daily weather (not an ensemble-mean
    composite), realization-level sampling, both SSPs, AOA extrapolation flagging.

Livestock multiplier: INDEPENDENT per-species multipliers (Option B, user-confirmed) -- 27 combos
(3 species x {1x, 2x, 3x} each), not a single shared scalar. fx_lsu_dens is rebuilt as the exact
LSU-weighted sum under every combo (see build_transient_scenario_drivers_species.py).

TabICLv2 is NOT a recursive rollout (same architecture as tabpfn_forecast()/S-03's own usage) --
one vectorized forward pass per (tower, ssp, gcm, realization, multiplier-combo) over the full
10-year (3650-day) target window. Measured empirically before committing to the full sweep:
~3.8s/call. Anchored at each tower's own last real y_observed date (2023-12-29 for T4/T9,
2019-05-31 for T2 -- T2's usual data-scarce anchor, not a new limitation), NOT a mid-history
anchor like S-03 -- there is no real future to leak from for a genuinely blind 10-year-out
trajectory, matching S-01/S-04's own anchoring convention.

Scope (user-confirmed, cost measured before deciding): 3 towers x 2 SSPs (ssp245/ssp585, matching
S-04) x 5 GCMs x 10 realizations/GCM (stratified, S-04's own precedent for cutting an expensive
axis) x 27 multiplier combos = 8,100 tabicl_forecast() calls, ~8.5h estimated.

Run from project root:  python notebooks/07_scenario_analysis/s05_trajectory_10yr.py
Smoke test (1 tower, 1 SSP, 1 GCM, 2 realizations, 3 combos):
  python notebooks/07_scenario_analysis/s05_trajectory_10yr.py smoke
"""
import itertools
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
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
    load_transient_years, stratified_realizations, GCMS,
)

HOURLY = rf"{ROOT}\data\Hourly"
RESULTS = rf"{ROOT}\results"

TOWERS = [2, 4, 9]
SSPS = ["ssp245", "ssp585"]
N_YEARS = 10
MULT_LEVELS = [1.0, 2.0, 3.0]
MULT_COMBOS = list(itertools.product(MULT_LEVELS, repeat=3))  # (cattle, sheep, lamb), 27 combos


def load_towers():
    dv = pd.read_csv(f"{HOURLY}/forecast_daily_v3.csv", low_memory=False)
    dv["Datetime"] = pd.to_datetime(dv["Datetime"], format="mixed")
    T = {t: dv[dv.tower == t].set_index("Datetime").sort_index() for t in TOWERS}
    return T


def tower_anchor(T, tower):
    """Last real y_observed date -- matches S-01/S-04's own convention of projecting forward from
    the end of all available real data, not a mid-history anchor (there's no real future to leak
    from for a genuinely blind trajectory)."""
    dft = T[tower]
    return dft.loc[dft["y_observed"].notna()].index.max()


def precompute_aoa(T, tower):
    """Nearest-neighbour AOA threshold (Meyer & Pebesma 2021-style, same convention as
    scenario_hybrid.dissimilarity_index()), precomputed ONCE per tower from real historical
    FX_A_SPECIES rows -- NOT recomputed per scenario call (dissimilarity_index() itself redoes
    this O(N_train^2) step every call, a real cost at this sweep's scale; this factors it out)."""
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


def main(n_per_gcm=10, ssps=None, towers=None, run_label="", realizations=None, mult_combos=None):
    ssps = SSPS if ssps is None else ssps
    towers = TOWERS if towers is None else towers
    mult_combos_local = MULT_COMBOS if mult_combos is None else mult_combos
    suffix = f"_{run_label}" if run_label else ""

    T = load_towers()
    realizations = stratified_realizations(n_per_gcm) if realizations is None else realizations
    print(f"[S-05] {len(towers)} towers x {len(ssps)} SSPs x {len(realizations)} (GCM,realization) "
          f"pairs x {len(mult_combos_local)} multiplier combos = "
          f"{len(towers)*len(ssps)*len(realizations)*len(mult_combos_local)} calls")

    all_rows = []
    t0 = time.time()
    n_done = 0

    for tower in towers:
        dft = T[tower]
        anchor = tower_anchor(T, tower)
        hist_target = dft.loc[:anchor, "y_observed"]
        hist_cov = dft.loc[:anchor, FX_A_SPECIES]
        years = list(range(anchor.year + 1, anchor.year + 1 + N_YEARS))
        print(f"\n=== Tower {tower}: anchor={anchor.date()}, years={years[0]}-{years[-1]} ===")

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

                # Build the full N_YEARS-year frame ONCE per (tower, ssp, gcm, realization) for
                # each multiplier combo (climatology base + AOA are shared across combos; only the
                # per-species scaling + fx_lsu_dens rebuild differ).
                for mc, cattle, sheep, lamb in [(mc, *mc) for mc in mult_combos_local]:
                    year_frames = []
                    for yr in years:
                        f = overlay_transient_species(clim_cache[yr], tyears[yr], cattle, sheep, lamb)
                        year_frames.append(f)
                    frame = pd.concat(year_frames)[FX_A_SPECIES]

                    chain = rr.tabicl_forecast(hist_target, hist_cov, frame)
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
                remaining = (len(towers) * len(ssps) * len(realizations) * len(mult_combos_local) - n_done)
                rate = n_done / elapsed if elapsed > 0 else 0
                eta_h = (remaining / rate / 3600) if rate > 0 else float("nan")
                print(f"  T{tower} {ssp} {gcm}/{real}: 27 combos in {dt:.1f}s "
                      f"({n_done} done, elapsed {elapsed/3600:.2f}h, ETA {eta_h:.2f}h)")

    out = pd.DataFrame(all_rows)
    out.to_csv(f"{RESULTS}/s05_trajectory_realizations{suffix}.csv", index=False)
    print(f"\n[OK] Saved s05_trajectory_realizations{suffix}.csv ({len(out)} rows), "
          f"total {time.time()-t0:.0f}s")
    return out


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        # tiny smoke test: 1 tower, 1 SSP, 1 GCM, 2 realizations, 3 multiplier combos (not all 27)
        main(towers=[4], ssps=["ssp245"], realizations=[("ACCESS-ESM1-5", 1), ("ACCESS-ESM1-5", 2)],
             mult_combos=[(1.0, 1.0, 1.0), (2.0, 1.0, 1.0), (1.0, 2.0, 3.0)], run_label="smoketest")
    else:
        main(n_per_gcm=10)
