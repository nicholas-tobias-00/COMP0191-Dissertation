"""S-06b production engine: drop-in replacement for `s05_practices_trajectory.run_axis()` that
calls the B18-derived `Direct_TabICLv2_solo_trend` architecture (S-03b/c/d-validated) instead of
`rr.tabicl_forecast()` (the TS-wrapper). Phase 6 of the additive B18-integration plan (2026-08-20).

Locked-in config, after three rounds of gate checks: S-03b found a POOLED (tower-dummy) + trend
config beating control by 4.4% MASE, but S-03c showed pooling isn't faithful to S-05/S-06's actual
per-tower-anchor pipeline (pooling needs one shared cutoff across towers, which S-05/S-06 don't
have) -- solo per-tower with NO trend feature barely beat control (0.6%). S-03d isolated the cause:
a `b17_days_since_2010` trend feature was the real driver (solo+trend = 2.79% better than control).
**Extrapolation safety confirmed directly** (not assumed): a sanity check fit on real Tower 4 data
and queried at trend values 27-47 years past the training max showed the model SATURATES to a
bounded, plausible value (~8.3 nmol) rather than exploding -- unlike SARIMAX's known explosive
extrapolation (D-63/U-03). Final config: solo per-tower `Direct_TabICLv2` regression on
`feat_cols + [b17_days_since_2010]`, no pooling. Trend column is computed directly from each
frame's own DatetimeIndex (`(date - 2010-01-01).days`) -- works identically for real historical
training rows and synthetic 2050-horizon scenario rows, no dependency on B17's own pipeline.

The one genuine efficiency win this unlocks: `hist_target`/`hist_cov` are FIXED per (tower,
feat_cols), so the model is fit ONCE per tower (not once per call like `tabicl_forecast()`, which
has no separate .fit()/.predict()) and reused across every (ssp, gcm, realization, level) combo for
that tower -- a 3-fits-total-per-axis vs. thousands-of-refits saving, on top of the architecture
change itself.

Same function signature/behaviour as `run_axis()` otherwise (same AOA precompute, same output
schema, same `s05_practices_{run_label}.csv` -- but this writes `s06b_practices_{run_label}.csv` to
stay additive) -- every existing `build_frame_fn` (`build_livestock_frame`, `build_grazing_frame`,
`build_fertilizer_frame`) works unchanged.

Run standalone smoke test:  python notebooks/07_scenario_analysis/s06b_direct_regression_engine.py smoke
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
sys.path.insert(0, str(ROOT / "notebooks" / "07_scenario_analysis"))

import s05_practices_trajectory as spt
from s05_trajectory_2050 import load_towers, tower_anchor, END_YEAR
from build_transient_scenario_drivers_species import stratified_realizations

RESULTS = ROOT / "results"
TOWERS = [2, 4, 9]
SSPS = ["ssp245", "ssp585"]


TREND_COL = "b17_days_since_2010"


def add_trend(df):
    df = df.copy()
    df[TREND_COL] = (df.index - pd.Timestamp("2010-01-01")).days.astype(float)
    return df


def make_model():
    from tabicl import TabICLRegressor

    return TabICLRegressor(n_estimators=8, random_state=42)


def run_axis_b18(feat_cols, levels, build_frame_fn, n_per_gcm=10, run_label=""):
    """Same contract as `run_axis()`: `build_frame_fn(tower, T, dft, anchor, years, level, tyears)`
    -> full-horizon covariate frame (index=nominal dates) in `feat_cols` order. `TREND_COL` is
    added internally (from each frame's own DatetimeIndex) -- callers pass the same `feat_cols`
    they always have (FX_A_SPECIES, +grazing/fert cols), no call-site changes needed."""
    model_features = list(feat_cols) + [TREND_COL]
    T = load_towers()
    realizations = stratified_realizations(n_per_gcm)
    n_total = len(TOWERS) * len(SSPS) * len(realizations) * len(levels)
    print(f"[S-06b/{run_label}] {len(TOWERS)} towers x {len(SSPS)} SSPs x "
          f"{len(realizations)} (GCM,realization) pairs x {len(levels)} levels = {n_total} calls "
          f"(Direct_TabICLv2_solo_trend -- {len(TOWERS)} fits total, not {n_total})")

    all_rows = []
    t0 = time.time()
    n_done = 0

    for tower in TOWERS:
        dft = T[tower]
        anchor = tower_anchor(T, tower)
        hist = dft.loc[:anchor]
        hist = hist.loc[hist["y_observed"].notna()]
        hist = add_trend(hist)
        years = list(range(anchor.year + 1, END_YEAR + 1))

        scaler, Xtr, aoa_thresh = spt.precompute_aoa(dft, feat_cols)

        t_fit0 = time.time()
        imputer = SimpleImputer(strategy="mean")
        x_train = imputer.fit_transform(hist[model_features])
        model = make_model()
        model.fit(x_train, hist["y_observed"].to_numpy())
        print(f"\n=== Tower {tower}: anchor={anchor.date()}, years={years[0]}-{years[-1]}, "
              f"n_train={len(hist)}, trend_range=[{hist[TREND_COL].min():.0f},{hist[TREND_COL].max():.0f}], "
              f"fit={time.time() - t_fit0:.1f}s ===")

        for ssp in SSPS:
            for gcm, real in realizations:
                t_call0 = time.time()
                try:
                    tyears = spt.load_transient_years(gcm, ssp, real, years)
                except FileNotFoundError as e:
                    print(f"  SKIPPED: {e}")
                    continue

                for level in levels:
                    frame_full = build_frame_fn(tower, T, dft, anchor, years, level, tyears)
                    frame_full = add_trend(frame_full)
                    frame = frame_full[feat_cols]  # kept for AOA (feat_cols-only space)
                    x_frame = imputer.transform(frame_full[model_features])
                    prediction = model.predict(x_frame, output_type="median")
                    chain = pd.Series(np.asarray(prediction), index=frame.index)

                    chain_df = chain.to_frame("pred")
                    chain_df["nominal_year"] = [d.year for d in frame.index]
                    for yr, g in chain_df.groupby("nominal_year"):
                        yr_frame = frame.loc[g.index]
                        aoa_pct = spt.aoa_flagged_frac(scaler, Xtr, aoa_thresh, yr_frame.values) * 100
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
    out.to_csv(f"{RESULTS}/s06b_practices_{run_label}.csv", index=False)
    print(f"\n[OK] Saved s06b_practices_{run_label}.csv ({len(out)} rows), total {time.time()-t0:.0f}s")
    return out


if __name__ == "__main__":
    from build_transient_scenario_drivers_species import FX_A_SPECIES
    from build_transient_scenario_drivers_livestock_v2 import COMBOS
    import s05_livestock_v2_trajectory as s05lv2

    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        TOWERS = [4]
        SSPS = ["ssp245"]
        run_axis_b18(FX_A_SPECIES, list(COMBOS), s05lv2.build_livestock_frame, n_per_gcm=1,
                     run_label="livestock_v2_smoketest")
    else:
        print("Import this module's run_axis_b18() from a runner script; no default full-scope action here.")
