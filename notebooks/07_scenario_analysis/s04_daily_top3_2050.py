"""S-04 daily extension: persists full 365-day daily chains (2025-2050) for each tower's top-3
best-performing B-10 benchmark models, instead of collapsing to annual_mean like
s04_trajectory_2050.py's run_b10_benchmark_sweep(). Model selection is fixed here, not re-derived
-- ranking source is D-65's real-anchor MASE table (persistence baseline, primary), cross-checked
against D-71's climatology-baseline MASE (secondary): the top-3 SET is identical under both
baselines at all 3 towers, only the intra-tie order at Tower 2 differs (LightGBM/XGB tied at 0.580
under climatology).

Top-3 per tower (S-04's 6-model roster: RF/XGB/LightGBM/SARIMAX/Ensemble_unweighted/
Ensemble_MASEweighted; SARIMAX never makes top-3 anywhere, consistent with U-03's finding that it's
the least stable model under scenario extrapolation):
  Tower 2: LightGBM, XGB, RF
  Tower 4: XGB, Ensemble_unweighted, Ensemble_MASEweighted
  Tower 9: Ensemble_unweighted, Ensemble_MASEweighted, XGB

Reuses the EXACT same grid and model-fitting/rollout code as s04_trajectory_2050.py's
run_b10_benchmark_sweep() (10 realizations stratified across 5 GCMs, 2 SSPs, 3 towers, 26 years,
3 multipliers) for direct comparability with the already-published annual
s04_trajectory_realizations_b10benchmark.csv -- this is a genuine re-run of the rollout (the daily
chains were computed in memory there but discarded before writing), not a resume. b10_benchmark_
rollout() always computes all 6 models internally (ensembles need all 4 base models regardless of
which are saved) -- only the 3 selected columns per tower are written out here.

Checkpoints incrementally after each (ssp, gcm) block (N_PER_GCM_B10 realizations each) -- a crash
loses at most one block's work. `smoke` mode runs a tiny slice (1 ssp, 1 gcm, 1 realization, all 3
towers, 2 years, all multipliers) and cross-checks its daily-mean against the already-published
annual_mean for the same combo, per this project's smoke-test-before-full-sweep convention.
"""
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = r"c:\Users\Nicholas\Documents\COMP0191 MSc Artificial Intelligence for Sustainable Development Project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\src")
sys.path.insert(0, ROOT + r"\src\features")
sys.path.insert(0, ROOT + r"\src\models")
sys.path.insert(0, ROOT + r"\notebooks\07_scenario_analysis")

import build_scenario_drivers as bsd
import build_transient_scenario_drivers as btsd
from s04_trajectory_2050 import (
    TOWERS, SSPS, YEARS, MULTIPLIERS, N_PER_GCM_B10,
    fit_b10_benchmark, b10_benchmark_rollout, build_pool,
)

RESULTS = rf"{ROOT}\results"
OUT_PATH = f"{RESULTS}/s04_daily_top3_2050.csv"

TOP3 = {
    2: ["LightGBM", "XGB", "RF"],
    4: ["XGB", "Ensemble_unweighted", "Ensemble_MASEweighted"],
    9: ["Ensemble_unweighted", "Ensemble_MASEweighted", "XGB"],
}


def rows_for_combo(chains, ssp, gcm, realization, tower, year, mult):
    """Long-format daily rows for the 3 selected models at this (ssp, gcm, realization, tower,
    year, multiplier) combo."""
    rows = []
    for model_name in TOP3[tower]:
        series = chains[model_name]
        for date, value in series.items():
            rows.append({
                "ssp": ssp, "gcm": gcm, "realization": realization, "tower": tower,
                "year": year, "multiplier": mult, "model": model_name,
                "date": date.strftime("%Y-%m-%d"), "fch4": float(value),
            })
    return rows


def run_smoke():
    """Tiny slice: 1 ssp, 1 gcm, 1 realization, all 3 towers, 2 years, all multipliers. Verifies
    the daily series' own annual mean matches the already-published annual benchmark CSV for the
    same combo (same fitted models, same frame-building code -- should match closely)."""
    print("=== S-04 DAILY TOP-3 SMOKE TEST ===")
    T = bsd.load_towers()
    pool = build_pool(T)
    t0 = time.time()
    tree_models_b10, imp_b10, feat_cols_b10, sarimax_by_tower = fit_b10_benchmark(pool, T)
    print(f"[OK] One-time fit complete ({time.time()-t0:.0f}s)")

    published = pd.read_csv(f"{RESULTS}/s04_trajectory_realizations_b10benchmark.csv")

    ssp, gcm, realization = "ssp245", "ACCESS-ESM1-5", 1
    smoke_years = YEARS[:2]
    tyears = btsd.load_transient_years(gcm, ssp, realization, smoke_years)

    all_rows = []
    t0 = time.time()
    for tower in TOWERS:
        dft = T[tower]
        clim_base = None
        for year in smoke_years:
            clim_base = btsd.build_climatology_base(tower, T, year, include_ustar_shf=True)
            for mult in MULTIPLIERS:
                frame = btsd.overlay_transient_drivers(clim_base, tyears[year], mult)
                annual_means, chains = b10_benchmark_rollout(
                    tree_models_b10, imp_b10, feat_cols_b10, sarimax_by_tower[tower], frame, dft, year)
                rows = rows_for_combo(chains, ssp, gcm, realization, tower, year, mult)
                all_rows.extend(rows)

                for model_name in TOP3[tower]:
                    daily_mean = chains[model_name].mean()
                    ref = published[(published.ssp == ssp) & (published.gcm == gcm) &
                                     (published.realization == realization) & (published.tower == tower) &
                                     (published.year == year) & (published.multiplier == mult) &
                                     (published.model == model_name)]
                    ref_val = ref["annual_mean"].iloc[0] if len(ref) else float("nan")
                    match = "OK" if len(ref) and abs(daily_mean - ref_val) < 1e-6 else "MISMATCH"
                    print(f"  T{tower} {year} mult={mult} {model_name}: daily-series mean={daily_mean:.4f} "
                          f"vs published annual_mean={ref_val:.4f} [{match}]")
    elapsed = time.time() - t0
    n_combos = len(TOWERS) * len(smoke_years) * len(MULTIPLIERS)
    print(f"\n[OK] Smoke test: {n_combos} combos, {len(all_rows)} daily rows, {elapsed:.1f}s "
          f"({elapsed/n_combos:.2f}s/combo)")

    n_full = len(SSPS) * N_PER_GCM_B10 * 5 * len(TOWERS) * len(YEARS) * len(MULTIPLIERS)
    est_full_s = elapsed / n_combos * n_full
    n_rows_full = n_full * 3 * 365  # every tower's TOP3 has exactly 3 models
    print(f"[OK] Full sweep estimate: {n_full} combos -> ~{est_full_s/3600:.1f}h, "
          f"~{n_rows_full:,.0f} daily rows")

    pd.DataFrame(all_rows).to_csv(f"{RESULTS}/s04_daily_top3_smoke.csv", index=False)
    print(f"[OK] Smoke output written -> {RESULTS}/s04_daily_top3_smoke.csv")


def run_full(start_ssp_idx=0, start_gcm_idx=0, append=False):
    """Full sweep, checkpointed after each (ssp, gcm) block. start_ssp_idx/start_gcm_idx allow
    resuming after a crash without redoing already-flushed blocks; append=True when resuming."""
    print("=== S-04 DAILY TOP-3 FULL SWEEP ===")
    T = bsd.load_towers()
    pool = build_pool(T)
    t0 = time.time()
    tree_models_b10, imp_b10, feat_cols_b10, sarimax_by_tower = fit_b10_benchmark(pool, T)
    print(f"[OK] One-time fit complete ({time.time()-t0:.0f}s)")

    print("[OK] Precomputing climatology cache (once per tower/year)...")
    t0 = time.time()
    clim_cache = {(tower, year): btsd.build_climatology_base(tower, T, year, include_ustar_shf=True)
                  for tower in TOWERS for year in YEARS}
    print(f"[OK] Climatology cache built ({len(clim_cache)} entries, {time.time()-t0:.1f}s)")

    realizations = list(range(1, N_PER_GCM_B10 + 1))
    first_write = not append

    t_start = time.time()
    n_done = 0
    n_total = len(SSPS) * N_PER_GCM_B10 * 5 * len(TOWERS) * len(YEARS) * len(MULTIPLIERS)
    # account for combos already done in prior (ssp, gcm) blocks when resuming
    n_done_offset = (start_ssp_idx * 5 + start_gcm_idx) * N_PER_GCM_B10 * len(TOWERS) * len(YEARS) * len(MULTIPLIERS)
    n_done = n_done_offset

    for ssp_idx, ssp in enumerate(SSPS):
        if ssp_idx < start_ssp_idx:
            continue
        for gcm_idx, gcm in enumerate(btsd.GCMS):
            if ssp_idx == start_ssp_idx and gcm_idx < start_gcm_idx:
                continue
            block_rows = []
            for realization in realizations:
                tyears = btsd.load_transient_years(gcm, ssp, realization, YEARS)
                for tower in TOWERS:
                    dft = T[tower]
                    for year in YEARS:
                        clim_base = clim_cache[(tower, year)]
                        for mult in MULTIPLIERS:
                            frame = btsd.overlay_transient_drivers(clim_base, tyears[year], mult)
                            annual_means, chains = b10_benchmark_rollout(
                                tree_models_b10, imp_b10, feat_cols_b10, sarimax_by_tower[tower],
                                frame, dft, year)
                            block_rows.extend(rows_for_combo(chains, ssp, gcm, realization, tower, year, mult))
                            n_done += 1
                elapsed = time.time() - t_start
                rate = (n_done - n_done_offset) / elapsed if elapsed > 0 else 0
                eta_min = (n_total - n_done) / rate / 60 if rate > 0 else float("nan")
                print(f"  [{ssp}] {gcm} r{realization}: {n_done}/{n_total} combos done "
                      f"({elapsed/60:.1f} min elapsed this run, ETA {eta_min:.0f} min)")
            pd.DataFrame(block_rows).to_csv(OUT_PATH, mode="w" if first_write else "a",
                                             header=first_write, index=False)
            first_write = False
            print(f"[OK] Checkpoint saved after ssp={ssp} gcm={gcm} ({len(block_rows)} rows)")

    print(f"[OK] Daily top-3 sweep complete: {n_done} combos -> {OUT_PATH}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    if mode == "smoke":
        run_smoke()
    elif mode == "full":
        run_full()
    elif mode == "resume":
        # resume ssp_idx gcm_idx, e.g. `resume 0 3` to restart from ssp[0]'s 4th GCM
        ssp_idx = int(sys.argv[2])
        gcm_idx = int(sys.argv[3])
        run_full(start_ssp_idx=ssp_idx, start_gcm_idx=gcm_idx, append=True)
    else:
        raise ValueError(f"Unknown mode: {mode} (expected smoke/full/resume)")
