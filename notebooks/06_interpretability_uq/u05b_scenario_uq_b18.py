"""U-05b: scenario-analysis UQ for the LOCKED-IN B18-derived architecture -- Phase 4 of the
additive B18-integration plan (2026-08-20), UPDATED after S-03c/S-03d's follow-up gate checks.
U-05 built its calibration set on the OLD architecture (`tabicl_forecast`, the TS-wrapper). S-03b
first found a pooled-across-towers `Direct_TabICLv2_raw` config beating the TS-wrapper by ~4.4%
MASE, but S-03c showed that config isn't faithful to S-05/S-06's actual per-tower-anchor pipeline
(pooling needs one shared cutoff across towers, which S-05/S-06 don't have) -- a solo per-tower
refit with no trend feature barely beat control (0.6%). S-03d isolated the cause: a `days_since_2010`
trend feature was the real driver (solo+trend = 2.79% better than control), confirmed SAFE to
extrapolate to the 2050 horizon via a direct sanity check (TabICLv2 saturates to a bounded plausible
value ~27-47 years past its training range, unlike SARIMAX's known explosive extrapolation, D-63/
U-03). **Final locked-in Phase 6 architecture: solo per-tower `Direct_TabICLv2` regression on
FX_A_SPECIES + `b17_days_since_2010`, no pooling.** This file was originally built against S-03b's
pooled config and is updated here to match the actual final architecture exactly.

Method: mirrors U-05's Steps 1-3 exactly (same 5 anchors x 3 towers, same quantiles, same
leave-one-anchor-out `rr.conformal_margins_by_bin()` via `evaluate_stage()` imported unmodified,
same AOA nearest-neighbour-distance mechanism in FX_A_SPECIES's own 13-dim space, same pre-anchor-
only AOA training set). What changed, forced by the architecture: the model is fit ONCE per anchor
pooled across all 3 towers (not once per tower) -- same fit-once/predict-many pattern as
i03b/u08/s03b. AOA computation is UNCHANGED (per-tower, doesn't depend on model architecture).

Step 4 (apply calibration to S-05/S-06's saved outputs) is DEFERRED to Phase 6 -- the current S-05/
S-06 output files still reflect the OLD architecture; applying this NEW calibration to OLD
predictions would be incoherent. Re-run Step 4 once Phase 6 produces the new scenario outputs.

Run from project root:  python notebooks/06_interpretability_uq/u05b_scenario_uq_b18.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

ROOT = Path(r"C:\Users\Nicholas\Documents\COMP0191 MSc Artificial Intelligence for Sustainable Development Project")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "features"))
sys.path.insert(0, str(ROOT / "notebooks" / "05_benchmarking"))
sys.path.insert(0, str(ROOT / "notebooks" / "06_interpretability_uq"))

import B17_foundation_screen as b17s
import B17_direct_and_recursive_foundations as b17d
from build_transient_scenario_drivers_species import FX_A_SPECIES
from u02_multi_anchor_tower import evaluate_stage, QUANTILES

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

RESULTS = ROOT / "results"
CHAINS_PATH = RESULTS / "u05b_chains.csv"
SUMMARY_PATH = RESULTS / "u05b_summary.csv"
AOA_POOLED_PATH = RESULTS / "u05b_aoa_residual_correlation.csv"
AOA_BY_TOWER_PATH = RESULTS / "u05b_aoa_residual_correlation_by_tower.csv"

TOWERS = b17s.TOWERS
ANCHOR_YEARS = b17s.ANCHOR_YEARS
N_DAYS = b17s.N_DAYS
STATIC_FEATURES = b17d.TOWER_DUMMIES + b17d.TIME_FEATURES


TREND_COL = "b17_days_since_2010"


def fit_stage_b18(frame):
    print(f"[U-05b] FX_A_SPECIES ({len(FX_A_SPECIES)} cols) + {TREND_COL}, solo per-tower "
          f"Direct_TabICLv2 (S-03d-locked final architecture)")
    from tabicl import TabICLRegressor

    features = FX_A_SPECIES + [TREND_COL]
    T = {t: frame.loc[frame["tower"].eq(t)].set_index("Datetime").sort_index() for t in TOWERS}
    rows = []
    t0 = time.time()

    for tower in TOWERS:
        dft = T[tower]
        for yr in ANCHOR_YEARS:
            anchor = pd.Timestamp(f"{yr}-12-16")
            dates = pd.date_range(anchor + pd.Timedelta(days=1), periods=N_DAYS, freq="D")
            train = dft.loc[:anchor]
            train = train.loc[train["y_observed"].notna()]
            future = dft.loc[dft.index.isin(dates)]
            if len(train) > 10_000:
                train = train.sample(10_000, random_state=42).sort_values("Datetime")
            if train.empty or future.empty:
                continue

            t_a = time.time()
            try:
                imputer = SimpleImputer(strategy="mean")
                x_train = imputer.fit_transform(train[features])
                model = TabICLRegressor(n_estimators=8, random_state=42)
                model.fit(x_train, train["y_observed"].to_numpy())

                x_future = imputer.transform(future[features])
                q_array = model.predict(x_future, output_type="quantiles", alphas=list(QUANTILES))
            except Exception as e:
                print(f"    T{tower} {yr} FIT/PREDICT SKIPPED: {str(e)[:150]}")
                continue

            X_train_aoa = train[FX_A_SPECIES].dropna().values
            if len(X_train_aoa) < 5:
                aoa_dist_arr, aoa_threshold = np.full(len(future), np.nan), np.nan
            else:
                scaler = StandardScaler().fit(X_train_aoa)
                Xtr = scaler.transform(X_train_aoa)
                d_train = cdist(Xtr, Xtr)
                np.fill_diagonal(d_train, np.inf)
                d_loo = d_train.min(axis=1)
                q1, q3 = np.percentile(d_loo, [25, 75])
                aoa_threshold = q3 + 1.5 * (q3 - q1)
                aoa_dist_arr = cdist(scaler.transform(future[FX_A_SPECIES].values), Xtr).min(axis=1)

            for i, d in enumerate(future.index):
                aoa_dist = aoa_dist_arr[i]
                rows.append(
                    {
                        "anchor_year": yr,
                        "eval_tower": tower,
                        "model": "Direct_TabICLv2_solo_trend",
                        "date": d,
                        "q05": q_array[i, 0],
                        "median": q_array[i, 1],
                        "q95": q_array[i, 2],
                        "y_true": future["y_observed"].iloc[i],
                        "aoa_dist": aoa_dist,
                        "aoa_flagged": (aoa_dist > aoa_threshold) if np.isfinite(aoa_dist) else np.nan,
                    }
                )
            print(f"  T{tower} anchor {yr} done ({time.time() - t_a:.0f}s, {time.time() - t0:.0f}s elapsed)")

    return pd.DataFrame(rows)


def aoa_residual_check(chains):
    df = chains.copy()
    df["abs_resid"] = (df["y_true"] - df["median"]).abs()
    df = df.dropna(subset=["abs_resid", "aoa_dist"])

    def summarize(g):
        corr = g["abs_resid"].corr(g["aoa_dist"])
        fm = g.groupby("aoa_flagged")["abs_resid"].agg(["mean", "count"])
        return pd.Series(
            {
                "pearson_corr_resid_vs_aoa_dist": corr,
                "mean_abs_resid_in_aoa": fm.loc[False, "mean"] if False in fm.index else np.nan,
                "mean_abs_resid_out_aoa": fm.loc[True, "mean"] if True in fm.index else np.nan,
                "n_in_aoa": fm.loc[False, "count"] if False in fm.index else 0,
                "n_out_aoa": fm.loc[True, "count"] if True in fm.index else 0,
            }
        )

    pooled = summarize(df).to_frame().T
    pooled.to_csv(AOA_POOLED_PATH, index=False)
    print(f"\n[OK] {AOA_POOLED_PATH.name} (pooled)\n{pooled.to_string(index=False)}")

    by_tower = df.groupby("eval_tower").apply(summarize, include_groups=False)
    by_tower.to_csv(AOA_BY_TOWER_PATH)
    print(f"\n[OK] {AOA_BY_TOWER_PATH.name}\n{by_tower.round(3).to_string()}")

    return pooled, by_tower


def main():
    frame = pd.read_csv(b17s.DATA_PATH, low_memory=False)
    frame["Datetime"] = pd.to_datetime(frame["Datetime"], format="mixed")
    frame = b17d.add_b17_features(frame)

    print("=" * 70)
    print("U-05b STEP 1: Direct_TabICLv2_raw FX_A_SPECIES quantile rollout (calibration set)")
    print("=" * 70)
    chains = fit_stage_b18(frame)
    chains.to_csv(CHAINS_PATH, index=False)
    print(f"\n[OK] Saved {CHAINS_PATH.name} ({len(chains)} rows)")

    print("\n" + "=" * 70)
    print("U-05b STEP 2: leave-one-anchor-out conformal calibration")
    print("=" * 70)
    summary = evaluate_stage(chains)
    summary.to_csv(SUMMARY_PATH, index=False)
    print(f"[OK] Saved {SUMMARY_PATH.name} ({len(summary)} rows)")

    def wavg(g, col):
        vals = g[col]
        if vals.isna().all():
            return np.nan
        w = g["n"]
        return (vals * w).sum() / w.sum() if w.sum() > 0 else np.nan

    agg = summary.groupby("eval_tower").apply(
        lambda g: pd.Series(
            {
                "raw_picp": wavg(g, "raw_picp"),
                "raw_mpiw": wavg(g, "raw_mpiw"),
                "conformal_picp": wavg(g, "conformal_picp"),
                "conformal_mpiw": wavg(g, "conformal_mpiw"),
                "conformal_pinball": wavg(g, "conformal_pinball"),
            }
        ),
        include_groups=False,
    ).reset_index()
    print(agg.round(4).to_string(index=False))

    print("\n" + "=" * 70)
    print("U-05b STEP 3: does |residual| correlate with AOA-flagged status? (empirical check)")
    print("=" * 70)
    aoa_residual_check(chains)

    print("\nSTEP 4 (apply to S-05/S-06 outputs) DEFERRED to Phase 6 -- current S-05/S-06 files "
          "still reflect the OLD architecture.")


if __name__ == "__main__":
    main()
