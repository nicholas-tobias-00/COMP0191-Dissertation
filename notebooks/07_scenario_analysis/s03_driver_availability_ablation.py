"""S-03 (proposed D-70): isolates how much forecasting accuracy is lost purely from NOT having
access to real-time sensor variables that a CMIP6 climate scenario can never supply -- distinct
from U-03's distribution-shift test (D-63) and S-01's extrapolation-into-2050 test (D-64), both of
which conflate feature-degradation cost with other effects. Test data here is REAL and HISTORICAL
(the same 2018-2022 anchors as B-10/D-65) -- only the FEATURE SET changes.

Model 1 (B-10's full-feature ensemble) is NOT rerun here -- read directly from
results/b10_b13_rerun_table_all_towers.csv / _by_tower_year.csv (D-65) in the results notebook.

Model 2 = same B-10 architecture/hyperparameters/ensemble, two variants:
  - Variant A (removal): scenario-unavailable columns dropped entirely (never seen in training or
    rollout).
  - Variant B (resample): same columns present and used in TRAINING (real values, identical to
    Model 1's own training), but their values in the ROLLOUT-TIME fx_frame/exog (the 365-day
    target window) are day-of-year-climatology-resampled via rr.doy_climatology(), using
    PRE-ANCHOR-ONLY history -- a deliberate, necessary deviation from build_scenario_drivers.py's
    own call (which correctly uses the full record, since S-01 has no real "anchor", it projects
    from the end of all history to 2050). Using the full record here would leak an anchor's own
    future real values into the climatology used to resample that same anchor's test window.

Degraded-column list is imported directly from build_scenario_drivers.py's RESAMPLED_COLS +
DROPPED_COLS (S-01's own production list) -- not retyped, so this experiment can never silently
drift from what S-01 actually treats as scenario-unavailable. User-confirmed additions beyond the
originally-named variable families: wind direction (fx_wd_sin/cos) and grazing features
(fx_grazing_active/fx_days_since_grazing) are IN scope, matching S-01's own RESAMPLED_COLS exactly.
This 24-column list is the DEFAULT for both variants, but `main()` takes `remove_cols`/
`resample_cols` as independent, overridable parameters (default: DEFAULT_DEGRADED_COLS each) --
the notebook wrapper exposes these as plain editable lists, so e.g. "what if only SWC/TS get
resampled but WS/VPD/USTAR/SHF are just dropped" is a one-line notebook edit, not a script change.

Explicitly OUT of scope (stay real/untouched in both variants):
  - fx_lsu_dens: the scenario LEVER S-01 deliberately manipulates, not a missing-sensor variable.
  - AR features (ar_ch4_dlag*, ar_ch4_drm7, ar_fc_dlag1): S-01 resamples these too, but because no
    real recent CH4/FCO2 history exists in a genuinely blind 2050 future -- a temporal-
    extrapolation problem, not a driver-source problem. This experiment evaluates on real
    historical anchors, where real recent AR history genuinely exists -- touching it would
    reintroduce the exact extrapolation-conflation this experiment exists to avoid.

No new HPO -- every hyperparameter (tree, SARIMAX AIC order-search, ensemble weights) is reused
verbatim from B-10/D-65. TFT/TabPFN are out of scope (Model 1/2 are specifically B-10's 4-model
architecture, not the full I-02/U-02/U-03 8-model roster).
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

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX

import models.recursive_rollout as rr
from build_scenario_drivers import RESAMPLED_COLS, DROPPED_COLS

HOURLY = rf"{ROOT}\data\Hourly"
RESULTS = rf"{ROOT}\results"

TOWERS = [2, 4, 9]
N_DAYS = 365
ANCHOR_YEARS = [2018, 2019, 2020, 2021, 2022]

DUM = ["is_t2", "is_t4", "is_t9"]
AR_COLS = ["ar_ch4_dlag1", "ar_ch4_dlag2", "ar_ch4_dlag3", "ar_ch4_dlag7", "ar_ch4_dlag14", "ar_ch4_drm7"]
EXOG_B = ["fx_lsu_dens", "fx_WS_mean", "fx_VPD_mean", "fx_USTAR_mean", "fx_PPFD_mean",
          "fx_DOY_sin", "fx_DOY_cos", "fx_is_growing"]

# The exact scenario-unavailable column set -- imported from S-01's own production list, not retyped.
# This is the DEFAULT for both variants; main() accepts remove_cols/resample_cols overrides.
DEFAULT_DEGRADED_COLS = list(RESAMPLED_COLS) + list(DROPPED_COLS)  # 22 + 2 = 24 columns

# B-09's frozen multi-anchor mean MASE -- NOT re-derived here, exactly matching B-10's original.
B09_MEAN_MASE = {"XGB": 0.968, "LightGBM": 0.978, "RF": 1.024, "SARIMAX": 1.038}

VARIANTS = ["A_removal", "B_resample"]


def fit_tree(algo, tr, feat_cols):
    """Byte-identical hyperparameters to B10_daily_improvements.ipynb's fit_tree()."""
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


def climatology_substitute(dft, cols, anchor, target_dates):
    """For each column in `cols`, replaces the real target-window values with
    rr.doy_climatology() computed from PRE-ANCHOR-ONLY history (dft.loc[:anchor, col]) --
    the deliberate deviation from build_scenario_drivers.py's full-history call, necessary to
    avoid leaking an anchor's own future real values into its own test-window climatology.
    Returns a DataFrame indexed by target_dates, one column per entry in `cols`."""
    out = {}
    for col in cols:
        hist = dft.loc[:anchor, col].dropna()
        assert hist.index.max() <= anchor, f"{col}: pre-anchor-only history violated"
        out[col] = rr.doy_climatology(hist, target_dates, window=7)
    return pd.DataFrame(out, index=target_dates)


def main(remove_cols=None, resample_cols=None, run_label=""):
    """remove_cols: columns dropped entirely in Variant A. resample_cols: columns
    climatology-substituted (test-window only) in Variant B. Both default to
    DEFAULT_DEGRADED_COLS (S-01's own RESAMPLED_COLS+DROPPED_COLS) if not given -- pass a
    narrower/different list to either independently, e.g. resample_cols=["fx_SWC_mean",
    "fx_TS_mean"] to test resampling only soil moisture/temperature while Variant A still drops
    the full default set. run_label: suffix appended to every output filename (e.g. "_soilonly")
    so a customized run doesn't overwrite the default sweep's outputs."""
    remove_cols = list(DEFAULT_DEGRADED_COLS) if remove_cols is None else list(remove_cols)
    resample_cols = list(DEFAULT_DEGRADED_COLS) if resample_cols is None else list(resample_cols)
    suffix = f"_{run_label}" if run_label else ""

    dv = pd.read_csv(f"{HOURLY}/forecast_daily_v2.csv", low_memory=False)
    dv["Datetime"] = pd.to_datetime(dv["Datetime"], format="mixed")
    FX_B = [c for c in dv.columns if c.startswith("fx")]
    assert len(FX_B) == 34, f"expected 34 fx_ columns, got {len(FX_B)}"
    assert set(remove_cols).issubset(set(FX_B)), "remove_cols has columns not present in FX_B"
    assert set(resample_cols).issubset(set(FX_B)), "resample_cols has columns not present in FX_B"

    FX_A = [c for c in FX_B if c not in remove_cols]  # remaining columns after removal
    EXOG_A = [c for c in EXOG_B if c not in remove_cols]  # reduced SARIMAX exog (variant A)
    exog_resample = [c for c in EXOG_B if c in resample_cols]  # SARIMAX exog resampled (variant B)
    print(f"[S-03] FX_B={len(FX_B)}, remove_cols={len(remove_cols)}, resample_cols={len(resample_cols)}, "
          f"FX_A (removal, remaining)={len(FX_A)}")
    print(f"[S-03] EXOG_B={len(EXOG_B)}, EXOG_A (removal, remaining)={EXOG_A}, exog_resample={exog_resample}")

    T = {t: dv[dv.tower == t].set_index("Datetime").sort_index() for t in TOWERS}

    feat_cols_full = AR_COLS + FX_B + ["ar_fc_dlag1"] + DUM   # variant B trains on this (real)
    feat_cols_A = AR_COLS + FX_A + ["ar_fc_dlag1"] + DUM      # variant A trains on this (reduced)

    all_rows = []
    all_rows_gf = []
    all_chain_rows = []

    for yr in ANCHOR_YEARS:
        anchor = pd.Timestamp(f"{yr}-12-16")
        target_dates = pd.date_range(anchor + pd.Timedelta(days=1), periods=N_DAYS, freq="D")
        print(f"\n{'='*70}\nAnchor {yr}\n{'='*70}")
        t_anchor = time.time()

        # ---- Pooled training frame (identical construction to Model 1) ----
        pool = []
        for t in TOWERS:
            df = T[t].copy()
            df["target"] = df["y_gapfilled"]
            for d in DUM:
                df[d] = 1.0 if d == f"is_t{t}" else 0.0
            pool.append(df[df.index <= anchor])
        tr = pd.concat(pool)
        tr = tr[tr["target"].notna()]

        # ---- Fit trees TWICE per anchor: once on full features (variant B), once reduced (variant A) ----
        tree_models_full, tree_models_A = {}, {}
        for algo in ["RF", "XGB", "LightGBM"]:
            tree_models_full[algo] = fit_tree(algo, tr, feat_cols_full)
            tree_models_A[algo] = fit_tree(algo, tr, feat_cols_A)
        print(f"  Pooled trees fit, both variants ({time.time()-t_anchor:.0f}s)")

        # ---- SARIMAX: fit per tower, per variant (full EXOG_B vs reduced EXOG_A) ----
        for tower in TOWERS:
            t_tower = time.time()
            dft = T[tower]
            history_init = dft.loc[:anchor, "y_gapfilled"].copy()

            y_true = dft["y_observed"].reindex(target_dates).values
            anchor_val = dft.loc[anchor, "y_gapfilled"]
            persist = rr.chain_persistence(anchor_val, N_DAYS)

            y_gf = dft["y_gapfilled"].reindex(target_dates).values
            bin_labels = rr.lead_time_bin(target_dates, anchor)
            real_frac_by_bin = {}
            for lo, hi in ((1, 7), (8, 30), (31, 90), (91, 180), (181, 270), (271, 365)):
                lbl = f"{lo}-{hi}"
                bm_mask = bin_labels == lbl
                real_frac_by_bin[lbl] = float(np.isfinite(y_true[bm_mask]).mean()) if bm_mask.sum() > 0 else np.nan

            y = dft["y_gapfilled"].astype(float)

            for variant in VARIANTS:
                # ---- Build fx_frame + SARIMAX exog for this variant ----
                if variant == "A_removal":
                    fx_frame = dft.loc[target_dates, FX_A + ["ar_fc_dlag1"]].copy()
                    X_full = dft[EXOG_A].astype(float).ffill().bfill()
                    tree_models = tree_models_A
                    feat_cols = feat_cols_A
                else:  # B_resample
                    fx_frame = dft.loc[target_dates, FX_B + ["ar_fc_dlag1"]].copy()
                    clim_sub = climatology_substitute(dft, resample_cols, anchor, target_dates)
                    fx_frame[resample_cols] = clim_sub[resample_cols]
                    X_full = dft[EXOG_B].astype(float).ffill().bfill()
                    tree_models = tree_models_full
                    feat_cols = feat_cols_full

                fx_frame["is_t2"] = 1.0 if tower == 2 else 0.0
                fx_frame["is_t4"] = 1.0 if tower == 4 else 0.0
                fx_frame["is_t9"] = 1.0 if tower == 9 else 0.0

                # ---- Trees: reuse the already-fitted (per-variant) pooled models ----
                tree_chains = {}
                for algo, (model, imp) in tree_models.items():
                    tree_chains[algo] = rr.tree_rollout(model, imp, feat_cols, fx_frame, history_init,
                                                         anchor, n_days=N_DAYS)

                # ---- SARIMAX: fit fresh per tower/variant (B-10's original grid: p in [1,2], q in [0,1], d=1) ----
                X_tr = X_full.loc[:anchor]
                y_tr = y.loc[:anchor]
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
                sarimax_res = best[2]

                if variant == "A_removal":
                    future_X = X_full.loc[target_dates]
                else:
                    future_X = X_full.loc[target_dates].copy()
                    if exog_resample:
                        clim_exog = climatology_substitute(dft, exog_resample, anchor, target_dates)
                        future_X[exog_resample] = clim_exog[exog_resample]

                fc = sarimax_res.get_forecast(steps=N_DAYS, exog=future_X)
                sarimax_chain = pd.Series(fc.predicted_mean.values, index=target_dates)

                # ---- Ensembles (identical construction to Model 1) ----
                ens_df = pd.DataFrame({**tree_chains, "SARIMAX": sarimax_chain})
                ens_unweighted = ens_df.mean(axis=1)
                w = {k: 1.0 / v for k, v in B09_MEAN_MASE.items()}
                wsum = sum(w.values())
                w = {k: v / wsum for k, v in w.items()}
                ens_weighted = sum(ens_df[k] * w[k] for k in ens_df.columns)

                chains_this = {**tree_chains, "SARIMAX": sarimax_chain,
                               "Ensemble_unweighted": ens_unweighted, "Ensemble_MASEweighted": ens_weighted}

                chain_df = pd.DataFrame({name: c.reindex(target_dates) for name, c in chains_this.items()})
                chain_df["y_true"] = y_true
                chain_df["y_gapfilled"] = y_gf
                chain_df["persistence"] = persist
                chain_df["tower"] = tower
                chain_df["anchor_year"] = yr
                chain_df["variant"] = variant
                chain_df.index.name = "date"

                for name, chain in chains_this.items():
                    yp = chain.reindex(target_dates).values
                    bm = rr.bin_metrics(y_true, yp, target_dates, anchor, y_persist=persist)
                    bm["model"] = name
                    bm["anchor_year"] = yr
                    bm["tower"] = tower
                    bm["variant"] = variant
                    all_rows.append(bm)

                    bm_gf = rr.bin_metrics(y_gf, yp, target_dates, anchor, y_persist=persist)
                    bm_gf["model"] = name
                    bm_gf["anchor_year"] = yr
                    bm_gf["tower"] = tower
                    bm_gf["variant"] = variant
                    bm_gf["real_frac"] = bm_gf["bin"].map(real_frac_by_bin)
                    all_rows_gf.append(bm_gf)

                all_chain_rows.append(chain_df.reset_index())

            print(f"  Tower {tower} done, both variants ({time.time()-t_tower:.0f}s)")

        print(f"  Anchor {yr} total ({time.time()-t_anchor:.0f}s)")

    out = pd.concat(all_rows, ignore_index=True)
    out.to_csv(f"{RESULTS}/s03_summary{suffix}.csv", index=False)
    print(f"\n[OK] Saved s03_summary{suffix}.csv ({len(out)} rows)")

    out_gf = pd.concat(all_rows_gf, ignore_index=True)
    out_gf.to_csv(f"{RESULTS}/s03_summary_vs_gapfilled{suffix}.csv", index=False)
    print(f"[OK] Saved s03_summary_vs_gapfilled{suffix}.csv ({len(out_gf)} rows)")

    chains = pd.concat(all_chain_rows, ignore_index=True)
    chains.to_csv(f"{RESULTS}/s03_chains{suffix}.csv", index=False)
    print(f"[OK] Saved s03_chains{suffix}.csv ({len(chains)} rows)")

    return out, out_gf, chains


if __name__ == "__main__":
    main()  # default run: remove_cols=resample_cols=DEFAULT_DEGRADED_COLS (S-01's own list), no suffix
