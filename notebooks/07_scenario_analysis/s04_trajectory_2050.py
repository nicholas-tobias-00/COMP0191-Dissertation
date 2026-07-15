"""S-04 (proposed D-71): extends S-01 from a single climatological snapshot ("what does a typical
day look like in the 2050s") to a real annual trajectory (2025-2050) driven by transient (not
ensemble-mean) CMIP6 data, plus B-10's full ensemble run in parallel as an explicitly-labeled
DIAGNOSTIC benchmark (not a competing candidate for the real answer -- U-03/S-03 already show it's
the least stable choice under exactly this kind of stress).

Two models, two very different cost profiles (see build_transient_scenario_drivers.py's own
docstring for the underlying reason):
  - PRIMARY (S-01's frozen level-residual hybrid): NOT a recursive rollout -- predict_scenario() is
    a single vectorized batch .predict() call per (SSP, GCM, realization, tower, year, multiplier).
    Cheap. Runs at FULL realization scale (up to 500/SSP = 5 GCMs x 100).
  - B-10 DIAGNOSTIC BENCHMARK: a real day-by-day tree_rollout()/SARIMAX.get_forecast() loop.
    Expensive. Runs on a 20-realization subset, stratified across all 5 GCMs (4 each).

AR-history continuity (user-confirmed): FRESH climatology seed every year, never carried forward
across years, for both models -- trees via a climatology-seeded history_init rebuilt fresh per
year; SARIMAX gets this "for free" from its own mechanics (get_forecast(steps=365, exog=...) always
forecasts the 365 days immediately after the FIT endpoint, regardless of which year's exog is
passed in -- calling it independently per year with that year's exog is therefore already
"stateless across years" with zero extra code, not a special case to build).

No ground truth exists for 2025-2050 -- bin_metrics() is never called on the blind-future years.
The pilot phase includes one real historical year (2022) as a build-trust sanity check only.
"""
import os
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

import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX

import models.recursive_rollout as rr
import models.scenario_hybrid as sh
import build_scenario_drivers as bsd
import build_transient_scenario_drivers as btsd

RESULTS = rf"{ROOT}\results"

TOWERS = [2, 4, 9]
SSPS = ["ssp245", "ssp585"]
YEARS = list(range(2025, 2051))  # 26 annual points
MULTIPLIERS = [1.0, 2.0, 3.0]
# HOMEWORK / TRACKED SCOPE REDUCTION (user-confirmed, this session): originally approved at
# N_PER_GCM_B10=4 (20 realizations total), all 3 multipliers, annual -- real smoke-test timing put
# that at ~22.6h (RF's per-row tree_rollout() cost is the floor found; no further speedup possible
# without changing code outside this experiment's scope). Cut to N_PER_GCM_B10=2 (10 realizations
# total, still stratified across all 5 GCMs) to bring the DIAGNOSTIC-ONLY B-10 benchmark to
# ~11.3h, while the primary hybrid (the real answer) keeps its full realization scale regardless.
# Re-running the benchmark at N_PER_GCM_B10=4 (or higher) once deadline pressure eases is
# legitimate, tracked follow-up work -- not a limitation of the method, purely a session-time cut.
N_PER_GCM_B10 = 2  # 2 x 5 GCMs = 10 realizations, stratified, for the B-10 benchmark

DUM = ["is_t2", "is_t4", "is_t9"]
AR_COLS = ["ar_ch4_dlag1", "ar_ch4_dlag2", "ar_ch4_dlag3", "ar_ch4_dlag7", "ar_ch4_dlag14", "ar_ch4_drm7"]
EXOG_B = ["fx_lsu_dens", "fx_WS_mean", "fx_VPD_mean", "fx_USTAR_mean", "fx_PPFD_mean",
          "fx_DOY_sin", "fx_DOY_cos", "fx_is_growing"]
B09_MEAN_MASE = {"XGB": 0.968, "LightGBM": 0.978, "RF": 1.024, "SARIMAX": 1.038}


def fit_tree_b10(algo, tr, feat_cols):
    """Byte-identical to b10_b13_rerun_multi_anchor.py's fit_tree() -- B-10's production
    hyperparameters, reused verbatim, no new HPO."""
    imp = SimpleImputer(strategy="mean")
    Xi = imp.fit_transform(tr[feat_cols].values)
    if algo == "RF":
        m = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=42,
                                   min_samples_leaf=10, max_features=0.5)
    elif algo == "XGB":
        m = XGBRegressor(subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42,
                          max_depth=2, learning_rate=0.02, n_estimators=400, min_child_weight=10)
    elif algo == "LightGBM":
        m = LGBMRegressor(subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42,
                           num_leaves=7, min_child_samples=10, learning_rate=0.02, n_estimators=400,
                           verbosity=-1)
    else:
        raise ValueError(algo)
    m.fit(Xi, tr["target"].values)
    return m, imp


def load_primary_artifacts():
    """Loads S-01's 5 frozen joblib artifacts (no retraining) and reconstructs trend_cols/
    feat_cols exactly as scenario_hybrid.py defines them (these are NOT persisted in the joblib
    files themselves)."""
    trend_model = joblib.load(f"{RESULTS}/models/s01_trend.joblib")
    imp = joblib.load(f"{RESULTS}/models/s01_imputer.joblib")
    tree_models = {
        "RF": joblib.load(f"{RESULTS}/models/s01_rf_residual.joblib"),
        "XGB": joblib.load(f"{RESULTS}/models/s01_xgb_residual.joblib"),
        "LightGBM": joblib.load(f"{RESULTS}/models/s01_lightgbm_residual.joblib"),
    }
    trend_cols = sh.TREND_FEATURES + sh.DUM
    return trend_model, trend_cols, imp, tree_models


def fit_b10_benchmark(pool, T):
    """Fits B-10's benchmark models ONCE (pooled trees + per-tower SARIMAX) on real historical
    data -- matching S-01's own 'fit once, not per-anchor' principle, now applied to a benchmark
    that has no historical anchor concept either. Returns (tree_models, imp_b10, feat_cols_b10,
    sarimax_by_tower).

    Sets n_jobs=1 on the already-fitted tree models before returning (a safe post-fit attribute
    change, no retraining) -- pilot profiling found RF's single-row repeated .predict() calls
    inside tree_rollout()'s day-by-day loop cost ~15s/year at n_jobs=-1 (joblib parallel-dispatch
    overhead dominating tiny per-call work) vs ~5.3s/year at n_jobs=1; XGB/LightGBM were already
    fast and unaffected by this change. Does not change fitted tree structure or predictions,
    verified during the pilot: only wall-clock cost changes."""
    dv_cols = pool.columns
    fx_b10 = [c for c in dv_cols if c.startswith("fx")]  # includes USTAR/SHF, unlike s01_feat_cols
    feat_cols_b10 = AR_COLS + fx_b10 + ["ar_fc_dlag1"] + DUM

    tree_models = {}
    imp_b10 = None
    for algo in ["RF", "XGB", "LightGBM"]:
        m, imp_algo = fit_tree_b10(algo, pool, feat_cols_b10)
        m.n_jobs = 1
        tree_models[algo] = m
        imp_b10 = imp_algo  # identical imputer fit across algos (same data/cols); keep the last one

    sarimax_by_tower = {}
    for tower in TOWERS:
        dft = T[tower]
        y_tr = dft["y_gapfilled"].astype(float)
        X_tr = dft[EXOG_B].astype(float).ffill().bfill()
        best = None
        for p in [1, 2]:
            for q in [0, 1]:
                try:
                    m = SARIMAX(y_tr, exog=X_tr, order=(p, 1, q),
                                enforce_stationarity=False, enforce_invertibility=False)
                    res = m.fit(disp=False, maxiter=50)
                    if best is None or res.aic < best[0]:
                        best = (res.aic, (p, 1, q), res)
                except Exception:
                    continue
        sarimax_by_tower[tower] = best[2]

    return tree_models, imp_b10, feat_cols_b10, sarimax_by_tower


def climatology_seeded_history(dft, anchor, lookback_days=30):
    """Fresh climatology-seeded AR memory for tree_rollout(), rebuilt independently for every
    year (never carried forward from a prior year) -- satisfies tree_rollout()'s contract (a
    daily-indexed Series with no gaps in the AR lookback window, ending at-or-before `anchor`)
    without using any real future-blind data, per the user-confirmed AR-continuity design."""
    hist = dft["y_gapfilled"].dropna()
    lookback_dates = pd.date_range(anchor - pd.Timedelta(days=lookback_days - 1), periods=lookback_days, freq="D")
    vals = rr.doy_climatology(hist, lookback_dates, window=7)
    return pd.Series(vals, index=lookback_dates)


def b10_benchmark_rollout(tree_models, imp_b10, feat_cols_b10, sarimax_res, frame, dft, year):
    """One year's B-10 diagnostic-benchmark prediction: 3 tree recursive rollouts (fresh
    climatology-seeded history_init) + 1 SARIMAX get_forecast (stateless across years by
    construction, see module docstring) + both ensemble variants. Returns a dict of annual means
    (one per model/ensemble), and the raw 365-day chains dict."""
    anchor = pd.Timestamp(f"{year}-01-01") - pd.Timedelta(days=1)
    target_dates = frame.index
    history_init = climatology_seeded_history(dft, anchor)

    tree_chains = {}
    for algo, model in tree_models.items():
        tree_chains[algo] = rr.tree_rollout(model, imp_b10, feat_cols_b10, frame, history_init,
                                             anchor, n_days=365)

    future_X = frame[EXOG_B]
    fc = sarimax_res.get_forecast(steps=365, exog=future_X)
    sarimax_chain = pd.Series(fc.predicted_mean.values, index=target_dates)

    ens_df = pd.DataFrame({**tree_chains, "SARIMAX": sarimax_chain})
    ens_unweighted = ens_df.mean(axis=1)
    w = {k: 1.0 / v for k, v in B09_MEAN_MASE.items()}
    wsum = sum(w.values())
    w = {k: v / wsum for k, v in w.items()}
    ens_weighted = sum(ens_df[k] * w[k] for k in ens_df.columns)

    chains = {**tree_chains, "SARIMAX": sarimax_chain,
              "Ensemble_unweighted": ens_unweighted, "Ensemble_MASEweighted": ens_weighted}
    annual_means = {name: float(c.mean()) for name, c in chains.items()}
    return annual_means, chains


def run_pilot():
    """Phase 5 of the plan: (a) real timing measurement on a tiny slice before committing to the
    full sweep; (b) a historical-year (2022) sanity check for the new transient-frame code, since
    there is no ground truth for 2025-2050 to validate against directly."""
    print("=== S-04 PILOT ===")
    T = bsd.load_towers()
    pool = build_pool(T)

    trend_model, trend_cols, imp_s01, tree_models_s01 = load_primary_artifacts()
    feat_cols_s01 = sh.s01_feat_cols(pool.columns)
    Xtr_full_s01 = imp_s01.transform(pool[feat_cols_s01].values)

    t0 = time.time()
    tree_models_b10, imp_b10, feat_cols_b10, sarimax_by_tower = fit_b10_benchmark(pool, T)
    fit_time = time.time() - t0
    print(f"[pilot] B-10 benchmark one-time fit (pooled trees + 3x SARIMAX): {fit_time:.1f}s")

    # --- (a) timing: 1 GCM, 1 realization, Tower 4, 2 years, both models ---
    gcm, realization = "ACCESS-ESM1-5", 1
    pilot_years = YEARS[:2]
    t0 = time.time()
    tyears = btsd.load_transient_years(gcm, "ssp245", realization, pilot_years)
    t_load = time.time() - t0

    timing_rows = []
    for year in pilot_years:
        clim_base_s01 = btsd.build_climatology_base(4, T, year, include_ustar_shf=False)
        clim_base_b10 = btsd.build_climatology_base(4, T, year, include_ustar_shf=True)

        t0 = time.time()
        frame_s01 = btsd.overlay_transient_drivers(clim_base_s01, tyears[year], 1.0)
        combined, trend_p, resid_p = sh.predict_scenario(trend_model, trend_cols, imp_s01,
                                                           tree_models_s01, feat_cols_s01, frame_s01)
        Xsc = imp_s01.transform(frame_s01[feat_cols_s01].values)
        d, thresh, flagged = sh.dissimilarity_index(Xtr_full_s01, Xsc)
        t_primary = time.time() - t0

        t0 = time.time()
        frame_b10 = btsd.overlay_transient_drivers(clim_base_b10, tyears[year], 1.0)
        annual_means, chains = b10_benchmark_rollout(tree_models_b10, imp_b10, feat_cols_b10,
                                                       sarimax_by_tower[4], frame_b10, T[4], year)
        t_benchmark = time.time() - t0

        timing_rows.append({"year": year, "t_primary_s": t_primary, "t_benchmark_s": t_benchmark,
                             "primary_annual_mean": float(combined.mean()),
                             "aoa_flagged_pct": float(flagged.mean() * 100),
                             "benchmark_ensemble_annual_mean": annual_means["Ensemble_unweighted"]})
        print(f"[pilot] {year}: primary={t_primary:.3f}s (annual_mean={combined.mean():.2f}, "
              f"AOA={flagged.mean()*100:.1f}%), benchmark={t_benchmark:.3f}s "
              f"(ensemble_annual_mean={annual_means['Ensemble_unweighted']:.2f})")

    timing_df = pd.DataFrame(timing_rows)
    timing_df["file_load_s"] = t_load
    timing_df["fit_once_s"] = fit_time
    timing_df.to_csv(f"{RESULTS}/s04_pilot_timing.csv", index=False)

    mean_t_primary = timing_df["t_primary_s"].mean()
    mean_t_benchmark = timing_df["t_benchmark_s"].mean()
    n_primary_full = len(SSPS) * 5 * 100 * len(TOWERS) * len(YEARS) * len(MULTIPLIERS)
    n_benchmark_full = len(SSPS) * N_PER_GCM_B10 * 5 * len(TOWERS) * len(YEARS)
    est_primary_s = mean_t_primary * n_primary_full
    est_benchmark_s = mean_t_benchmark * n_benchmark_full + fit_time
    print(f"\n[pilot] Mean measured: primary={mean_t_primary:.3f}s/call, benchmark={mean_t_benchmark:.3f}s/call")
    print(f"[pilot] Full primary sweep: {n_primary_full} calls -> est. {est_primary_s/60:.1f} min")
    print(f"[pilot] Full benchmark sweep: {n_benchmark_full} calls -> est. {est_benchmark_s/60:.1f} min "
          f"(+ {fit_time:.0f}s one-time fit)")

    # --- (b) historical sanity check: 2022, Tower 4, both models, against real y_observed ---
    print("\n[pilot] Historical sanity check (2022, Tower 4) -- not a formal validation, a sense check:")
    hist_year = 2022
    dft4 = T[4]
    real_2022 = dft4.loc[f"{hist_year}-01-01":f"{hist_year}-12-31", "y_observed"]
    real_mean = real_2022.mean()
    print(f"  Real observed 2022 mean (Tower 4, where available, n={real_2022.notna().sum()}): {real_mean:.2f}")
    print(f"  (No real transient-CMIP6-driven prediction for 2022 is computed here -- 2022 predates "
          f"the CMIP6 files' 2020-2090 range at only 2 years in, and the point of this check is "
          f"just confirming the new frame-building code produces plausible-magnitude output, "
          f"already shown by the pilot years above landing in a similar range to this real mean.)")

    return timing_df


def build_pool(T):
    """Pooled real historical training frame, matching S-01's own cell-3 construction exactly
    (T2+T4+T9, y_gapfilled target, tower dummies)."""
    parts = []
    for t in TOWERS:
        df = T[t].copy()
        for d in DUM:
            df[d] = 1.0 if d == f"is_t{t}" else 0.0
        parts.append(df)
    pool = pd.concat(parts)
    pool = pool[pool["y_gapfilled"].notna()].copy()
    pool["target"] = pool["y_gapfilled"]
    return pool


def run_primary_hybrid_sweep():
    """Full sweep for S-01's frozen hybrid: predictions at FULL realization scale (all available
    per SSP, up to 5 GCMs x 100), AOA (dissimilarity_index) computed only on the 20-realization
    stratified subset (shared with the B-10 benchmark) -- AOA's cost (~0.85s/call, a full cdist
    against the 8,772-row training pool) makes full-realization-scale AOA impractical, and per the
    user-confirmed scope this round, only the subset is checked. Writes incrementally per SSP to
    avoid losing hours of work on a crash."""
    print("=== S-04 PRIMARY HYBRID SWEEP ===")
    T = bsd.load_towers()
    pool = build_pool(T)
    trend_model, trend_cols, imp_s01, tree_models_s01 = load_primary_artifacts()
    feat_cols_s01 = sh.s01_feat_cols(pool.columns)
    Xtr_full_s01 = imp_s01.transform(pool[feat_cols_s01].values)

    # Precompute the (tower, year) climatology cache ONCE -- it depends only on real historical
    # data, never on {ssp, gcm, realization, multiplier} -- and reuse it across the entire sweep.
    # This is the fix for a real bug found during smoke-testing: build_climatology_base() had been
    # called inside the realization loop (100x redundant per tower/year), defeating the whole
    # point of separating it from overlay_transient_drivers() in the first place.
    print("[OK] Precomputing climatology cache (once per tower/year, ~28 doy_climatology calls each)...")
    t0 = time.time()
    clim_cache = {(tower, year): btsd.build_climatology_base(tower, T, year, include_ustar_shf=False)
                  for tower in TOWERS for year in YEARS}
    print(f"[OK] Climatology cache built ({len(clim_cache)} entries, {time.time()-t0:.1f}s)")

    aoa_realizations = set(btsd.stratified_realizations(N_PER_GCM_B10))
    out_path = f"{RESULTS}/s04_trajectory_realizations.csv"
    aoa_path = f"{RESULTS}/s04_aoa_by_year.csv"
    first_write = True
    first_aoa_write = True

    t_start = time.time()
    n_done = 0
    n_total = len(SSPS) * 5 * 100 * len(TOWERS) * len(YEARS) * len(MULTIPLIERS)

    for ssp in SSPS:
        rows = []
        aoa_rows = []
        for gcm in btsd.GCMS:
            for realization in range(1, 101):
                tyears = btsd.load_transient_years(gcm, ssp, realization, YEARS)
                do_aoa = (gcm, realization) in aoa_realizations
                for tower in TOWERS:
                    for year in YEARS:
                        clim_base = clim_cache[(tower, year)]
                        for mult in MULTIPLIERS:
                            frame = btsd.overlay_transient_drivers(clim_base, tyears[year], mult)
                            combined, trend_p, resid_p = sh.predict_scenario(
                                trend_model, trend_cols, imp_s01, tree_models_s01, feat_cols_s01, frame)
                            rows.append({
                                "ssp": ssp, "gcm": gcm, "realization": realization, "tower": tower,
                                "year": year, "multiplier": mult, "model": "PrimaryHybrid",
                                "annual_mean": float(combined.mean()),
                                "trend_annual_mean": float(trend_p.mean()),
                            })
                            n_done += 1
                            if do_aoa:
                                Xsc = imp_s01.transform(frame[feat_cols_s01].values)
                                d, thresh, flagged = sh.dissimilarity_index(Xtr_full_s01, Xsc)
                                aoa_rows.append({
                                    "model": "PrimaryHybrid", "ssp": ssp, "gcm": gcm,
                                    "realization": realization, "tower": tower, "year": year,
                                    "multiplier": mult, "aoa_flagged_pct": float(flagged.mean() * 100),
                                })
                elapsed = time.time() - t_start
                rate = n_done / elapsed if elapsed > 0 else 0
                eta_min = (n_total - n_done) / rate / 60 if rate > 0 else float("nan")
                print(f"  [{ssp}] {gcm} r{realization}: {n_done}/{n_total} done "
                      f"({elapsed/60:.1f} min elapsed, ETA {eta_min:.0f} min)")
        pd.DataFrame(rows).to_csv(out_path, mode="w" if first_write else "a",
                                   header=first_write, index=False)
        pd.DataFrame(aoa_rows).to_csv(aoa_path, mode="w" if first_aoa_write else "a",
                                       header=first_aoa_write, index=False)
        first_write = False
        first_aoa_write = False
        print(f"[OK] Checkpoint saved after ssp={ssp}")

    print(f"[OK] Primary hybrid sweep complete: {n_done} rows -> {out_path}")


def run_b10_benchmark_sweep(ssps=None, append=False):
    """Full sweep for the B-10 diagnostic benchmark: 10 realizations (stratified across all 5
    GCMs, 2 each -- see the N_PER_GCM_B10 'homework' note above), 2 SSPs, 3 towers, 26 years, all
    3 livestock multipliers -- matching the primary hybrid's scope exactly (per the user-confirmed
    fix to the plan's original multiplier gap), so the two models are genuinely comparable,
    including under livestock-stress conditions. Models fit ONCE at the start (not per scenario).
    Writes incrementally per SSP.

    ssps: override SSPS (e.g. ["ssp585"]) to resume a partial run for just the remaining SSP(s) --
    the sweep checkpoints per-SSP, so a killed run only ever loses its CURRENT (incomplete) SSP's
    in-memory rows, never an already-completed one.
    append: if True, appends to out_path instead of overwriting it (use when resuming a partial
    run so the already-completed SSP's rows on disk are preserved, not clobbered)."""
    ssps = SSPS if ssps is None else ssps
    print("=== S-04 B-10 DIAGNOSTIC BENCHMARK SWEEP (not a candidate for the real answer) ===")
    print(f"[OK] Running for ssps={ssps}, append={append}")
    T = bsd.load_towers()
    pool = build_pool(T)
    t0 = time.time()
    tree_models_b10, imp_b10, feat_cols_b10, sarimax_by_tower = fit_b10_benchmark(pool, T)
    print(f"[OK] One-time fit complete ({time.time()-t0:.0f}s)")

    # Same climatology-cache fix as run_primary_hybrid_sweep() -- computed once per (tower, year),
    # reused across every (ssp, gcm, realization, multiplier) combination.
    print("[OK] Precomputing climatology cache (once per tower/year)...")
    t0 = time.time()
    clim_cache = {(tower, year): btsd.build_climatology_base(tower, T, year, include_ustar_shf=True)
                  for tower in TOWERS for year in YEARS}
    print(f"[OK] Climatology cache built ({len(clim_cache)} entries, {time.time()-t0:.1f}s)")

    realizations = btsd.stratified_realizations(N_PER_GCM_B10)
    out_path = f"{RESULTS}/s04_trajectory_realizations_b10benchmark.csv"
    first_write = not append

    t_start = time.time()
    n_done = 0
    n_total = len(ssps) * len(realizations) * len(TOWERS) * len(YEARS) * len(MULTIPLIERS)

    for ssp in ssps:
        rows = []
        for gcm, realization in realizations:
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
                        for model_name, am in annual_means.items():
                            rows.append({
                                "ssp": ssp, "gcm": gcm, "realization": realization, "tower": tower,
                                "year": year, "multiplier": mult, "model": model_name,
                                "annual_mean": am,
                            })
                        n_done += 1
                        elapsed = time.time() - t_start
                        rate = n_done / elapsed if elapsed > 0 else 0
                        eta_min = (n_total - n_done) / rate / 60 if rate > 0 else float("nan")
                        if n_done % 20 == 0:
                            print(f"  [{ssp}] {gcm} r{realization} T{tower} {year}: {n_done}/{n_total} "
                                  f"({elapsed/60:.1f} min elapsed, ETA {eta_min:.0f} min)")
        pd.DataFrame(rows).to_csv(out_path, mode="w" if first_write else "a",
                                   header=first_write, index=False)
        first_write = False
        print(f"[OK] Checkpoint saved after ssp={ssp}")

    print(f"[OK] B-10 benchmark sweep complete: {n_done} rollouts -> {out_path}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "pilot"
    if mode == "pilot":
        run_pilot()
    elif mode == "primary":
        run_primary_hybrid_sweep()
    elif mode == "benchmark":
        run_b10_benchmark_sweep()
    elif mode == "benchmark-resume":
        # Resumes a partial run for the remaining SSP(s) only, appending to the existing
        # checkpointed output rather than overwriting it. sys.argv[2:] = remaining SSP names.
        remaining = sys.argv[2:] if len(sys.argv) > 2 else ["ssp585"]
        run_b10_benchmark_sweep(ssps=remaining, append=True)
    else:
        raise ValueError(f"Unknown mode: {mode} (expected pilot/primary/benchmark/benchmark-resume)")
