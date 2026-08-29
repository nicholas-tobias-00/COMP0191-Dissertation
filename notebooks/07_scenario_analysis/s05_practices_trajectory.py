"""S-05 follow-up: farming-practice scenarios -- grazing timing (season extension) and fertilizer
application schedule (rate/frequency) -- run as two SEPARATE experiments (not stacked onto S-05's
27-combo livestock grid), each holding livestock at baseline (1x/1x/1x real climatology) and
sweeping only its own 3 scenario levels. Matches this session's "isolate one axis at a time"
convention (same reasoning as S-05's own species-marginal-response design) and keeps compute
tractable: 3 towers x 2 SSPs x 5 GCMs x 10 realizations x 3 levels = 900 calls per axis, at the
current canonical 2050 horizon (T4/T9: 27yr, T2: 31yr) -- ~1h per axis at the measured ~4s/call
rate, ~2h total for both, versus the 27-combo livestock sweep's ~5-9h.

Expectation set going in (per F-01/F-04/F-05's repeated "redundant on the rich base" finding for
management features on real historical data, and grazing's direct mechanistic tie to livestock
presence vs. fertilizer's weaker one to CH4 specifically): grazing timing more likely to show a
real effect than fertilizer, given both axes' outcome is genuinely an empirical question this
script answers, not assumed going in.

Feature sets: FX_A_SPECIES (13 cols, S-05's own) + 2 grazing cols (`fx_grazing_active`,
`fx_days_since_grazing`) for the grazing run; FX_A_SPECIES + 2 fertN cols
(`fx_mgmt_fertN_recency`, `fx_mgmt_fertN_rate`) for the fertilizer run -- 15 columns each, AOA
threshold recomputed per-tower in that 15-dim space (not reusing S-05's 13-dim precompute, which
would be the wrong feature space for this experiment).

Run from project root:  python notebooks/07_scenario_analysis/s05_practices_trajectory.py
Smoke test: python notebooks/07_scenario_analysis/s05_practices_trajectory.py smoke
"""
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
sys.path.insert(0, ROOT + r"\notebooks\07_scenario_analysis")

import models.recursive_rollout as rr
from build_transient_scenario_drivers_species import FX_A_SPECIES, load_transient_years, stratified_realizations
from build_transient_scenario_drivers_practices import (
    GRAZING_SHIFT_LEVELS, FERT_LEVELS, shifted_species_climatology_base, overlay_transient_practices,
    fertN_recency_frame,
)
from s05_trajectory_2050 import load_towers, tower_anchor, END_YEAR

RESULTS = rf"{ROOT}\results"
TOWERS = [2, 4, 9]
SSPS = ["ssp245", "ssp585"]
GRAZING_COLS = ["fx_grazing_active", "fx_days_since_grazing"]
FERT_COLS = ["fx_mgmt_fertN_recency", "fx_mgmt_fertN_rate"]


def precompute_aoa(dft, feat_cols):
    X_train = dft[feat_cols].dropna().values
    scaler = StandardScaler().fit(X_train)
    Xtr = scaler.transform(X_train)
    d = cdist(Xtr, Xtr)
    np.fill_diagonal(d, np.inf)
    d_loo = d.min(axis=1)
    q1, q3 = np.percentile(d_loo, [25, 75])
    return scaler, Xtr, q3 + 1.5 * (q3 - q1)


def aoa_flagged_frac(scaler, Xtr, threshold, X):
    return float((cdist(scaler.transform(X), Xtr).min(axis=1) > threshold).mean())


def run_axis(feat_cols, levels, build_frame_fn, n_per_gcm=10, run_label=""):
    """`build_frame_fn(tower, T, dft, anchor, years, level, gcm, ssp, real)` -> full-horizon
    covariate frame (index=nominal dates) in `feat_cols` order. Shared driver loop for both axes."""
    T = load_towers()
    realizations = stratified_realizations(n_per_gcm)
    n_total = len(TOWERS) * len(SSPS) * len(realizations) * len(levels)
    print(f"[S-05 practices/{run_label}] {len(TOWERS)} towers x {len(SSPS)} SSPs x "
          f"{len(realizations)} (GCM,realization) pairs x {len(levels)} levels = {n_total} calls")

    all_rows = []
    t0 = time.time()
    n_done = 0

    for tower in TOWERS:
        dft = T[tower]
        anchor = tower_anchor(T, tower)
        hist_target = dft.loc[:anchor, "y_observed"]
        hist_cov = dft.loc[:anchor, feat_cols]
        years = list(range(anchor.year + 1, END_YEAR + 1))

        scaler, Xtr, aoa_thresh = precompute_aoa(dft, feat_cols)
        print(f"\n=== Tower {tower}: anchor={anchor.date()}, years={years[0]}-{years[-1]} ===")

        for ssp in SSPS:
            for gcm, real in realizations:
                t_call0 = time.time()
                try:
                    tyears = load_transient_years(gcm, ssp, real, years)
                except FileNotFoundError as e:
                    print(f"  SKIPPED: {e}")
                    continue

                for level in levels:
                    frame = build_frame_fn(tower, T, dft, anchor, years, level, tyears)[feat_cols]
                    chain = rr.tabicl_forecast(hist_target, hist_cov, frame)

                    chain_df = chain.to_frame("pred")
                    chain_df["nominal_year"] = [d.year for d in frame.index]
                    for yr, g in chain_df.groupby("nominal_year"):
                        yr_frame = frame.loc[g.index]
                        aoa_pct = aoa_flagged_frac(scaler, Xtr, aoa_thresh, yr_frame.values) * 100
                        all_rows.append({"tower": tower, "ssp": ssp, "gcm": gcm, "realization": real,
                                          "level": level, "year": yr, "annual_mean": float(g["pred"].mean()),
                                          "aoa_flagged_pct": aoa_pct})
                    n_done += 1

                dt = time.time() - t_call0
                elapsed = time.time() - t0
                rate = n_done / elapsed if elapsed > 0 else 0
                eta_h = ((n_total - n_done) / rate / 3600) if rate > 0 else float("nan")
                print(f"  T{tower} {ssp} {gcm}/{real}: {len(levels)} levels in {dt:.1f}s "
                      f"({n_done}/{n_total} done, elapsed {elapsed/3600:.2f}h, ETA {eta_h:.2f}h)")

    out = pd.DataFrame(all_rows)
    out.to_csv(f"{RESULTS}/s05_practices_{run_label}.csv", index=False)
    print(f"\n[OK] Saved s05_practices_{run_label}.csv ({len(out)} rows), total {time.time()-t0:.0f}s")
    return out


def build_grazing_frame(tower, T, dft, anchor, years, level, tyears):
    shift = GRAZING_SHIFT_LEVELS[level]
    frames = []
    for yr in years:
        clim = shifted_species_climatology_base(tower, T, yr, shift)
        frames.append(overlay_transient_practices(clim, tyears[yr]))
    return pd.concat(frames)


def build_fertilizer_frame(tower, T, dft, anchor, years, level, tyears):
    shift = 0  # real (unshifted) grazing/species climatology -- fertilizer is the lever here
    frames = []
    for yr in years:
        clim = shifted_species_climatology_base(tower, T, yr, shift)
        frames.append(overlay_transient_practices(clim, tyears[yr]))
    base_frame = pd.concat(frames)
    fert = fertN_recency_frame(tower, base_frame.index, level, anchor)
    return pd.concat([base_frame, fert], axis=1)


def main(n_per_gcm=10):
    grazing_cols = FX_A_SPECIES + GRAZING_COLS
    fert_cols = FX_A_SPECIES + FERT_COLS

    grazing_out = run_axis(grazing_cols, list(GRAZING_SHIFT_LEVELS), build_grazing_frame,
                            n_per_gcm=n_per_gcm, run_label="grazing")
    fert_out = run_axis(fert_cols, list(FERT_LEVELS), build_fertilizer_frame,
                         n_per_gcm=n_per_gcm, run_label="fertilizer")
    return grazing_out, fert_out


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        TOWERS = [4]
        SSPS = ["ssp245"]
        grazing_cols = FX_A_SPECIES + GRAZING_COLS
        fert_cols = FX_A_SPECIES + FERT_COLS
        run_axis(grazing_cols, list(GRAZING_SHIFT_LEVELS), build_grazing_frame, n_per_gcm=1, run_label="grazing_smoketest")
        run_axis(fert_cols, list(FERT_LEVELS), build_fertilizer_frame, n_per_gcm=1, run_label="fertilizer_smoketest")
    else:
        main()
