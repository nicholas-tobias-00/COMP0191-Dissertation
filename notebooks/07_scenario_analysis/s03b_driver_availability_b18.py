"""S-03b: driver-availability gate for the NEW (B-18-derived) architecture -- Phase 3 of the
additive B18-integration plan (2026-08-20). This is the step that decides what, if anything,
carries over from B-18 into S-05/S-06.

B-18's actual champion (D-106) cannot run in scenario mode: its `BASE_ALL_52` feature set includes
catchment flow, management-recency, and antecedent real-driver rolling features with no
constructible value in a synthetic 2050 future. S-05/S-06 instead use `FX_A_SPECIES` (13 CMIP6/
scenario-derivable columns: climate + calendar + livestock density) -- established at S-03/D-70 as
the scenario-safe feature set for the OLD (TabICLv2 TS-wrapper) architecture. This script asks the
same question for the NEW architecture: does the direct-regression / spike-gate mechanism still
help once restricted to FX_A_SPECIES, or does it lose its edge?

Method: same 5-anchor (2018-2022) x 3-tower real-anchor backtest as every other B-17/B-18/S-03
script, climatology-scored MASE (D-80 convention). Five candidates, all restricted to FX_A_SPECIES
(+ tower dummies / time features as static covariates for the pooled direct-regression candidates):

  1. CONTROL: current production architecture -- `tabicl_forecast()` (TS-wrapper, per-tower,
     one-shot), i.e. exactly what S-05/S-06 call today.
  2. Direct_TabPFN_raw: B-18's plain direct-regression mechanism (pooled across towers via tower-
     dummy features), no spike-gate.
  3. Direct_TabPFN_tower_robust: same, with B-18's tower-robust (median/IQR) target normalization.
  4. Direct_TabICLv2_raw: architecture-family control -- same mechanism, TabICLv2 instead of TabPFN.
  5. Direct_TabPFN_spikegate: B-18's full champion mechanism (p95 classifier gate + base/normal/
     spike regressors, base_plus_fixed_excess_0.25), restricted to FX_A_SPECIES.

Output: results/s03b_driver_availability_b18_summary.csv (MASE/MAE/RMSE/R2 per candidate, pooled
across towers/anchors) -- whichever candidate beats the control gets carried into Phase 4-6;
otherwise S-05/S-06 stay on the current TabICLv2-forecaster architecture, a legitimate documented
null result (same "B18 checked, doesn't survive the scenario-safe restriction" outcome this
project's rigor norm expects to sometimes find).

Run from project root:  python notebooks/07_scenario_analysis/s03b_driver_availability_b18.py
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

import models.recursive_rollout as rr
import B17_foundation_screen as b17s
import B17_direct_and_recursive_foundations as b17d
import B18_spike_models as b18sm
from build_transient_scenario_drivers_species import FX_A_SPECIES

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

RESULTS = ROOT / "results"
CHAINS_PATH = RESULTS / "s03b_driver_availability_b18_chains.csv"
SUMMARY_PATH = RESULTS / "s03b_driver_availability_b18_summary.csv"

TOWERS = b17s.TOWERS
ANCHOR_YEARS = b17s.ANCHOR_YEARS
N_DAYS = b17s.N_DAYS
STATIC_FEATURES = b17d.TOWER_DUMMIES + b17d.TIME_FEATURES
SPIKE_PERCENTILE = 95
CORRECTION_WEIGHT = 0.25

print(f"[S-03b] FX_A_SPECIES ({len(FX_A_SPECIES)} cols): {FX_A_SPECIES}")


# ---------------------------------------------------------------- candidate 1: current production
def run_control(frame):
    """tabicl_forecast(), per tower, restricted to FX_A_SPECIES -- exactly S-05/S-06's own call."""
    rows = []
    T = {t: frame.loc[frame["tower"].eq(t)].set_index("Datetime").sort_index() for t in TOWERS}
    for yr in ANCHOR_YEARS:
        anchor = pd.Timestamp(f"{yr}-12-16")
        dates = pd.date_range(anchor + pd.Timedelta(days=1), periods=N_DAYS, freq="D")
        for tower in TOWERS:
            dft = T[tower]
            hist = dft.loc[:anchor]
            hist_target = hist["y_observed"]
            hist_cov = hist[FX_A_SPECIES]
            future_cov = dft.loc[dft.index.isin(dates), FX_A_SPECIES]
            try:
                chain = rr.tabicl_forecast(hist_target, hist_cov, future_cov)
                y_true = dft.loc[chain.index, "y_observed"]
                y_gf = dft.loc[chain.index, "y_gapfilled"]
                for d in chain.index:
                    rows.append(
                        {
                            "candidate": "control_tabicl_forecaster",
                            "anchor_year": yr,
                            "tower": tower,
                            "date": d,
                            "y_predict": chain.loc[d],
                            "y_true": y_true.loc[d],
                            "y_gapfilled": y_gf.loc[d],
                        }
                    )
            except Exception as e:
                print(f"    T{tower} {yr} control SKIPPED: {str(e)[:150]}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- candidates 2-5: direct regression
def fit_target_transform_raw(train):
    def center_scale(rows):
        return np.zeros(len(rows)), np.ones(len(rows))

    return center_scale


def fit_target_transform_tower_robust(train):
    tower_median = train.groupby("tower")["y_observed"].median().to_dict()
    quantiles = train.groupby("tower")["y_observed"].quantile([0.25, 0.75]).unstack()
    tower_iqr = (quantiles[0.75] - quantiles[0.25]).replace(0, 1).fillna(1).to_dict()

    def center_scale(rows):
        center = rows["tower"].map(tower_median).to_numpy(float)
        scale = rows["tower"].map(tower_iqr).to_numpy(float)
        return center, scale

    return center_scale


def make_direct_model(model_name):
    if model_name == "TabPFN":
        from tabpfn import TabPFNRegressor

        return TabPFNRegressor(model_path=rr.tabpfn_v2_model_config()["model_path"], n_estimators=8, random_state=137)
    if model_name == "TabICLv2":
        from tabicl import TabICLRegressor

        return TabICLRegressor(n_estimators=8, random_state=42)
    raise ValueError(model_name)


def run_direct(frame, candidate_name, model_name, norm):
    features = FX_A_SPECIES + STATIC_FEATURES
    transform_fn = fit_target_transform_tower_robust if norm == "tower_robust" else fit_target_transform_raw
    rows = []
    for yr in ANCHOR_YEARS:
        anchor = pd.Timestamp(f"{yr}-12-16")
        dates = pd.date_range(anchor + pd.Timedelta(days=1), periods=N_DAYS, freq="D")
        train = frame.loc[frame["Datetime"].le(anchor) & frame["y_observed"].notna()].copy()
        future = frame.loc[frame["Datetime"].isin(dates)].copy()
        if len(train) > 10_000:
            train = train.sample(10_000, random_state=42).sort_values("Datetime")

        t0 = time.time()
        try:
            imputer = SimpleImputer(strategy="mean")
            x_train = imputer.fit_transform(train[features])
            center_train, scale_train = transform_fn(train)(train)
            y_train = (train["y_observed"].to_numpy() - center_train) / scale_train
            model = make_direct_model(model_name)
            model.fit(x_train, y_train)

            x_future = imputer.transform(future[features])
            pred_norm = model.predict(x_future, output_type="median")
            center_future, scale_future = transform_fn(train)(future)
            prediction = center_future + scale_future * np.asarray(pred_norm)
        except Exception as e:
            print(f"    {candidate_name} anchor {yr} SKIPPED: {str(e)[:150]}")
            continue

        future = future.assign(_pred=prediction)
        for tower in TOWERS:
            tf = future.loc[future["tower"].eq(tower)]
            for _, r in tf.iterrows():
                rows.append(
                    {
                        "candidate": candidate_name,
                        "anchor_year": yr,
                        "tower": tower,
                        "date": r["Datetime"],
                        "y_predict": r["_pred"],
                        "y_true": r["y_observed"],
                        "y_gapfilled": r["y_gapfilled"],
                    }
                )
        print(f"    {candidate_name} anchor {yr} done ({time.time() - t0:.0f}s)")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- candidate 5: spike-gate
def run_spikegate(frame):
    features = FX_A_SPECIES + STATIC_FEATURES
    rows = []
    for yr in ANCHOR_YEARS:
        anchor = pd.Timestamp(f"{yr}-12-16")
        dates = pd.date_range(anchor + pd.Timedelta(days=1), periods=N_DAYS, freq="D")
        train = frame.loc[frame["Datetime"].le(anchor) & frame["y_observed"].notna()].copy()
        future = frame.loc[frame["Datetime"].isin(dates)].copy()
        if len(train) > 10_000:
            train = train.sample(10_000, random_state=42).sort_values("Datetime")

        t0 = time.time()
        try:
            tower_thresholds, global_threshold = b18sm.threshold_map(train, SPIKE_PERCENTILE)
            labels, _ = b18sm.event_labels(train, tower_thresholds, global_threshold)
            imputer = SimpleImputer(strategy="mean")
            x_train = imputer.fit_transform(train[features])

            classifier = b18sm.make_classifier({"foundation": "TabPFN", "balance_probabilities": False})
            classifier.fit(x_train, labels)
            base_model = b18sm.make_regressor("TabPFN")
            base_model.fit(x_train, train["y_observed"].to_numpy())
            normal = train.loc[labels == 0]
            spike = train.loc[labels == 1]
            normal_model = b18sm.make_regressor("TabPFN")
            spike_model = b18sm.make_regressor("TabPFN")
            normal_model.fit(imputer.transform(normal[features]), normal["y_observed"].to_numpy())
            spike_model.fit(imputer.transform(spike[features]), spike["y_observed"].to_numpy())

            x_future = imputer.transform(future[features])
            probability = np.asarray(classifier.predict_proba(x_future))[:, 1]
            base_pred = b18sm.median_predict(base_model, x_future)
            normal_pred = b18sm.median_predict(normal_model, x_future)
            spike_pred = b18sm.median_predict(spike_model, x_future)
            excess = np.maximum(spike_pred - normal_pred, 0)
            prediction = base_pred + CORRECTION_WEIGHT * probability * excess
        except Exception as e:
            print(f"    spikegate anchor {yr} SKIPPED: {str(e)[:150]}")
            continue

        future = future.assign(_pred=prediction)
        for tower in TOWERS:
            tf = future.loc[future["tower"].eq(tower)]
            for _, r in tf.iterrows():
                rows.append(
                    {
                        "candidate": "Direct_TabPFN_spikegate",
                        "anchor_year": yr,
                        "tower": tower,
                        "date": r["Datetime"],
                        "y_predict": r["_pred"],
                        "y_true": r["y_observed"],
                        "y_gapfilled": r["y_gapfilled"],
                    }
                )
        print(f"    spikegate anchor {yr} done ({time.time() - t0:.0f}s)")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- scoring
def score(chains):
    chains = chains.copy()
    chains["date"] = pd.to_datetime(chains["date"])
    rows = []
    for (candidate, tower, yr), group in chains.groupby(["candidate", "tower", "anchor_year"]):
        anchor = pd.Timestamp(f"{yr}-12-16")
        frame_t = FULL_FRAME.loc[FULL_FRAME["tower"].eq(tower)].set_index("Datetime").sort_index()
        clim_hist = frame_t.loc[:anchor - pd.Timedelta(days=1), "y_observed"]
        if clim_hist.notna().sum() == 0:
            print(f"    [SKIP score] {candidate} T{tower} {yr}: no pre-anchor climatology history "
                  f"(known Tower-2-style degenerate case)")
            continue
        clim = rr.doy_climatology(clim_hist, pd.DatetimeIndex(group["date"]))
        bm = rr.bin_metrics(
            group["y_true"].to_numpy(),
            group["y_predict"].to_numpy(),
            pd.DatetimeIndex(group["date"]),
            anchor,
            y_persist=clim,
        )
        bm["candidate"] = candidate
        bm["tower"] = tower
        bm["anchor_year"] = yr
        rows.append(bm)
    bins = pd.concat(rows, ignore_index=True)

    summary = []
    for candidate, group in bins.groupby("candidate"):
        ok = group["MASE"].notna() & group["n"].gt(0)
        r2ok = group["R2"].notna() & group["n"].gt(0)
        summary.append(
            {
                "candidate": candidate,
                "n": int(group.loc[ok, "n"].sum()),
                "MASE": float(np.average(group.loc[ok, "MASE"], weights=group.loc[ok, "n"])),
                "MAE": float(np.average(group.loc[ok, "MAE"], weights=group.loc[ok, "n"])),
                "RMSE": float(np.average(group.loc[ok, "RMSE"], weights=group.loc[ok, "n"])),
                "R2": float(np.average(group.loc[r2ok, "R2"], weights=group.loc[r2ok, "n"])) if r2ok.any() else np.nan,
            }
        )
    return pd.DataFrame(summary).sort_values("MASE"), bins


def main():
    global FULL_FRAME
    frame = pd.read_csv(b17s.DATA_PATH, low_memory=False)
    frame["Datetime"] = pd.to_datetime(frame["Datetime"], format="mixed")
    frame = b17d.add_b17_features(frame)
    FULL_FRAME = frame

    print("=" * 70 + "\nCANDIDATE 1: control (current production tabicl_forecast)\n" + "=" * 70)
    c1 = run_control(frame)

    print("=" * 70 + "\nCANDIDATE 2: Direct_TabPFN_raw\n" + "=" * 70)
    c2 = run_direct(frame, "Direct_TabPFN_raw", "TabPFN", "raw")

    print("=" * 70 + "\nCANDIDATE 3: Direct_TabPFN_tower_robust\n" + "=" * 70)
    c3 = run_direct(frame, "Direct_TabPFN_tower_robust", "TabPFN", "tower_robust")

    print("=" * 70 + "\nCANDIDATE 4: Direct_TabICLv2_raw\n" + "=" * 70)
    c4 = run_direct(frame, "Direct_TabICLv2_raw", "TabICLv2", "raw")

    print("=" * 70 + "\nCANDIDATE 5: Direct_TabPFN_spikegate\n" + "=" * 70)
    c5 = run_spikegate(frame)

    chains = pd.concat([c1, c2, c3, c4, c5], ignore_index=True)
    chains.to_csv(CHAINS_PATH, index=False)
    print(f"\n[OK] Saved {CHAINS_PATH.name} ({len(chains)} rows)")

    summary, bins = score(chains)
    summary.to_csv(SUMMARY_PATH, index=False)
    print(f"\n[OK] Saved {SUMMARY_PATH.name}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
