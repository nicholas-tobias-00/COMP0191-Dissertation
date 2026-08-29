"""S-03 update (D-8x): brings the driver-availability ablation up to speed with two later project
conventions that postdate S-03's original run (D-70, 2026-07-14) and its model-roster-extension
addendum, plus closes the model-roster gap for good in this notebook (not just in a standalone
script never wired into it):

  1. MASE baseline switched from chain-persistence to day-of-year climatology (D-71/D-80,
     2026-08-02) -- climatology_baseline() in s03_driver_availability_ablation.py, reused here and
     applied to every model/variant in this script, including a rescore of the already-computed
     DL chains (no need to touch bin_metrics()/doy_climatology() themselves -- both already generic
     over whatever y_persist array is passed in).

  2. TabICL-sourced gap-filling (D-79/D-80) offers a newer y_gapfilled/AR-feature source for the
     8 of 11 models that read the daily forecast_daily_v2*.csv file (RF/XGB/LightGBM/SARIMAX/2
     ensembles/TabPFN/TabICLv2 -- y_observed and every fx_ driver column are IDENTICAL between
     forecast_daily_v2.csv and forecast_daily_v2_tabicl.csv, confirmed by direct diff; only
     y_gapfilled/ar_ch4_* differ). TFT/DLinear/LSTM read the hourly forecast_features_v2.csv,
     which has no TabICL-sourced sibling anywhere in this project
     (build_forecasting_matrix_v2_tabicl.py's own docstring: "forecast_features_v2.csv (hourly)
     does not depend on the gap-filled CH4 series at all") -- these 3 stay RF-sourced throughout,
     an unavoidable, explicitly-stated data-availability limit, not a gap introduced here.

Also closes a design tension the original S-03 carried: "Model 1" was read from an existing,
never-rerun table (B-10/D-65, persistence-scored). Swapping Variant A/B alone to TabICL-sourced
data while leaving Model 1 on the old RF-sourced/persistence-scored table would reintroduce
exactly the data-source/feature-availability conflation S-03 exists to avoid (S-03's whole reason
for existing is isolating ONE axis of variation at a time). Fixed by recomputing Model 1 too,
wherever a TabICL-sourced daily file makes that possible -- the SAME architecture/hyperparameters,
on the SAME anchors, with a fully real, undegraded feature set (remove_cols=[]/resample_cols=[]/
degraded_cols=[] collapses both variants to that identical config through the EXACT same code
path used for the real ablation, so Model 1 and Variant A/B are guaranteed comparable -- not a
separately-written "similar" computation). For TFT/DLinear/LSTM, Model 1 continues to follow the
original convention of reading from an existing table -- now the climatology-scored one (D-80's
results/b10_b13_climatology_mase_summary.csv and its _gf_ sibling), not the persistence-scored
D-65 one, since no TabICL rebuild is possible for them either way.

No DL (TFT/DLinear/LSTM) retraining happens in this script -- their Variant A/B predictions are
unaffected by the climatology-baseline switch (a scoring-only change) and by the (impossible)
TabICL swap, so the EXISTING saved chains (results/s03_model_roster_extension_chains_dl.csv) are
simply rescored with the new climatology baseline, reusing already-fitted predictions rather than
retraining TFT/DLinear/LSTM from scratch for a metric-only change.

Run from project root:  python notebooks/07_scenario_analysis/s03_climatology_tabicl_update.py
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import models.recursive_rollout as rr
import s03_driver_availability_ablation as s03
import s03_model_roster_extension as ext
import compile_s03_results as cs03

RESULTS = rf"{ROOT}\results"
HOURLY = rf"{ROOT}\data\Hourly"
TOWERS = [2, 4, 9]
ANCHOR_YEARS = [2018, 2019, 2020, 2021, 2022]
TABICL_CSV = "forecast_daily_v2_tabicl.csv"
DL_MODELS = ["TFT", "DLinear", "LSTM"]
BINS = ((1, 7), (8, 30), (31, 90), (91, 180), (181, 270), (271, 365))


# ============================================================ Steps 1-2: trees/SARIMAX/ensembles
def step1_tree_variants():
    print("\n" + "=" * 70 + "\nSTEP 1: tree/SARIMAX/ensemble Variant A/B on TabICL data\n" + "=" * 70)
    return s03.main(daily_csv=TABICL_CSV, run_label="tabicl")


def step2_tree_model1():
    print("\n" + "=" * 70 + "\nSTEP 2: tree/SARIMAX/ensemble Model-1-equivalent on TabICL data\n" + "=" * 70)
    return s03.main(remove_cols=[], resample_cols=[], daily_csv=TABICL_CSV, run_label="model1_tabicl")


# ============================================================ Steps 3-4: TabPFN/TabICLv2
def step3_foundation_variants():
    print("\n" + "=" * 70 + "\nSTEP 3: TabPFN/TabICLv2 Variant A/B on TabICL data\n" + "=" * 70)
    rows, rows_gf, _ = ext.run_foundation_models(ANCHOR_YEARS, daily_csv=TABICL_CSV)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    out_gf = pd.concat(rows_gf, ignore_index=True) if rows_gf else pd.DataFrame()
    out.to_csv(f"{RESULTS}/s03_foundation_tabicl_summary.csv", index=False)
    out_gf.to_csv(f"{RESULTS}/s03_foundation_tabicl_summary_vs_gapfilled.csv", index=False)
    print(f"[OK] Saved s03_foundation_tabicl_summary(.csv/_vs_gapfilled.csv) ({len(out)}/{len(out_gf)} rows)")
    return out, out_gf


def step4_foundation_model1():
    print("\n" + "=" * 70 + "\nSTEP 4: TabPFN/TabICLv2 Model-1-equivalent on TabICL data\n" + "=" * 70)
    rows, rows_gf, _ = ext.run_foundation_models(ANCHOR_YEARS, daily_csv=TABICL_CSV, degraded_cols=[])
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    out_gf = pd.concat(rows_gf, ignore_index=True) if rows_gf else pd.DataFrame()
    out.to_csv(f"{RESULTS}/s03_foundation_model1_tabicl_summary.csv", index=False)
    out_gf.to_csv(f"{RESULTS}/s03_foundation_model1_tabicl_summary_vs_gapfilled.csv", index=False)
    print(f"[OK] Saved s03_foundation_model1_tabicl_summary(.csv/_vs_gapfilled.csv) ({len(out)}/{len(out_gf)} rows)")
    return out, out_gf


# ============================================================ Step 5: DL rescoring (no retrain)
def step5_rescore_dl_variants():
    """TFT/DLinear/LSTM Variant A/B: no retrain (hourly data has no TabICL sibling anywhere in
    this project) -- rescores the EXISTING saved chains (s03_model_roster_extension_chains_dl.csv)
    with the climatology baseline in place of the persistence baseline they were originally scored
    against. Predictions themselves are untouched -- only MASE/RMSSE's denominator changes."""
    print("\n" + "=" * 70 +
          "\nSTEP 5: rescoring existing DL (TFT/DLinear/LSTM) Variant A/B chains with climatology\n" +
          "=" * 70)
    chains = pd.read_csv(f"{RESULTS}/s03_model_roster_extension_chains_dl.csv")
    chains["date"] = pd.to_datetime(chains["date"])

    dv = pd.read_csv(f"{HOURLY}/forecast_daily_v2.csv", low_memory=False)
    dv["Datetime"] = pd.to_datetime(dv["Datetime"], format="mixed")
    T = {t: dv[dv.tower == t].set_index("Datetime").sort_index() for t in TOWERS}

    all_rows, all_rows_gf = [], []
    for (tower, yr, variant), g in chains.groupby(["tower", "anchor_year", "variant"]):
        g = g.sort_values("date")
        anchor = pd.Timestamp(f"{yr}-12-16")
        target_dates = pd.DatetimeIndex(g["date"])
        dft = T[tower]
        clim = s03.climatology_baseline(dft, anchor, target_dates)

        y_true = g["y_true_dl"].values
        y_gf = g["y_gapfilled"].values
        bin_labels = rr.lead_time_bin(target_dates, anchor)
        real_frac_by_bin = {}
        for lo, hi in BINS:
            lbl = f"{lo}-{hi}"
            m = bin_labels == lbl
            real_frac_by_bin[lbl] = float(np.isfinite(y_true[m]).mean()) if m.sum() > 0 else np.nan

        for model in DL_MODELS:
            if model not in g.columns:
                continue
            yp = g[model].values
            if np.isfinite(yp).sum() == 0:
                continue
            bm = rr.bin_metrics(y_true, yp, target_dates, anchor, y_persist=clim)
            bm["model"] = model; bm["anchor_year"] = yr; bm["tower"] = tower; bm["variant"] = variant
            all_rows.append(bm)
            bm_gf = rr.bin_metrics(y_gf, yp, target_dates, anchor, y_persist=clim)
            bm_gf["model"] = model; bm_gf["anchor_year"] = yr; bm_gf["tower"] = tower; bm_gf["variant"] = variant
            bm_gf["real_frac"] = bm_gf["bin"].map(real_frac_by_bin)
            all_rows_gf.append(bm_gf)

    out = pd.concat(all_rows, ignore_index=True)
    out_gf = pd.concat(all_rows_gf, ignore_index=True)
    out.to_csv(f"{RESULTS}/s03_dl_climatology_summary.csv", index=False)
    out_gf.to_csv(f"{RESULTS}/s03_dl_climatology_summary_vs_gapfilled.csv", index=False)
    print(f"[OK] Saved s03_dl_climatology_summary(.csv/_vs_gapfilled.csv) ({len(out)}/{len(out_gf)} rows)")
    return out, out_gf


# ============================================================ Step 6: assemble the 11-model tables
def _load(name):
    return pd.read_csv(f"{RESULTS}/{name}")


def step6_build_tables():
    """Assembles Model1 / VariantA_removal / VariantB_resample raw per-bin frames for the full
    11-model roster, observed + gap-filled targets, then reuses compile_s03_results.py's own
    build_pooled/build_by_tower/three_way (unmodified) -- the SAME aggregation the original S-03
    tables went through, just fed different (climatology-scored, TabICL-updated-where-possible)
    inputs. Model1 for TFT/DLinear/LSTM comes from D-80's existing climatology-scored B-10/B-13
    tables (RF-sourced, since no TabICL rebuild is possible for them) -- NOT recomputed here."""
    print("\n" + "=" * 70 + "\nSTEP 6: assembling 11-model Model1/VariantA/VariantB tables\n" + "=" * 70)

    def build_set(tree_file, found_file, dl_source, dl_filter_col=None):
        tree = _load(tree_file)
        found = _load(found_file)
        if dl_filter_col is None:
            dl = dl_source[dl_source["model"].isin(DL_MODELS)]
        else:
            dl = dl_source
        return pd.concat([tree, found, dl], ignore_index=True)

    clim_model1_obs = _load("b10_b13_climatology_mase_summary.csv")
    clim_model1_gf = _load("b10_b13_climatology_gf_mase_summary.csv")

    tree_m1 = _load("s03_summary_model1_tabicl.csv")
    tree_m1_gf = _load("s03_summary_vs_gapfilled_model1_tabicl.csv")
    tree_A_B = _load("s03_summary_tabicl.csv")
    tree_A_B_gf = _load("s03_summary_vs_gapfilled_tabicl.csv")

    found_m1 = _load("s03_foundation_model1_tabicl_summary.csv")
    found_m1_gf = _load("s03_foundation_model1_tabicl_summary_vs_gapfilled.csv")
    found_A_B = _load("s03_foundation_tabicl_summary.csv")
    found_A_B_gf = _load("s03_foundation_tabicl_summary_vs_gapfilled.csv")

    dl_A_B = _load("s03_dl_climatology_summary.csv")
    dl_A_B_gf = _load("s03_dl_climatology_summary_vs_gapfilled.csv")

    # ---- Model 1 (observed + gapfilled): tree/foundation model1-equivalent (B_resample rows,
    # identical to A_removal by construction when remove_cols=resample_cols=[]) + DL from D-80 ----
    model1_obs = pd.concat([
        tree_m1[tree_m1["variant"] == "B_resample"],
        found_m1[found_m1["variant"] == "B_resample"],
        clim_model1_obs[clim_model1_obs["model"].isin(DL_MODELS)],
    ], ignore_index=True)
    model1_gf = pd.concat([
        tree_m1_gf[tree_m1_gf["variant"] == "B_resample"],
        found_m1_gf[found_m1_gf["variant"] == "B_resample"],
        clim_model1_gf[clim_model1_gf["model"].isin(DL_MODELS)],
    ], ignore_index=True)

    # ---- Variant A/B (observed + gapfilled): tree + foundation (TabICL-sourced) + DL (rescored) ----
    varA_obs = pd.concat([tree_A_B[tree_A_B.variant == "A_removal"],
                           found_A_B[found_A_B.variant == "A_removal"],
                           dl_A_B[dl_A_B.variant == "A_removal"]], ignore_index=True)
    varB_obs = pd.concat([tree_A_B[tree_A_B.variant == "B_resample"],
                           found_A_B[found_A_B.variant == "B_resample"],
                           dl_A_B[dl_A_B.variant == "B_resample"]], ignore_index=True)
    varA_gf = pd.concat([tree_A_B_gf[tree_A_B_gf.variant == "A_removal"],
                          found_A_B_gf[found_A_B_gf.variant == "A_removal"],
                          dl_A_B_gf[dl_A_B_gf.variant == "A_removal"]], ignore_index=True)
    varB_gf = pd.concat([tree_A_B_gf[tree_A_B_gf.variant == "B_resample"],
                          found_A_B_gf[found_A_B_gf.variant == "B_resample"],
                          dl_A_B_gf[dl_A_B_gf.variant == "B_resample"]], ignore_index=True)

    # ---- All-tower pooled ----
    obs_all = cs03.three_way(cs03.build_pooled(model1_obs), cs03.build_pooled(varA_obs), cs03.build_pooled(varB_obs))
    gf_all = cs03.three_way(cs03.build_pooled(model1_gf), cs03.build_pooled(varA_gf), cs03.build_pooled(varB_gf))
    combined_all = pd.concat({"Observed": obs_all, "GapFilled": gf_all}, axis=1)
    combined_all.to_csv(f"{RESULTS}/s03_table_all_towers_climatology_tabicl.csv")
    print(f"[OK] Saved s03_table_all_towers_climatology_tabicl.csv")

    # ---- Per-tower ----
    obs_bt = cs03.three_way(cs03.build_by_tower(model1_obs), cs03.build_by_tower(varA_obs), cs03.build_by_tower(varB_obs))
    gf_bt = cs03.three_way(cs03.build_by_tower(model1_gf), cs03.build_by_tower(varA_gf), cs03.build_by_tower(varB_gf))
    combined_bt = pd.concat({"Observed": obs_bt, "GapFilled": gf_bt}, axis=1)
    combined_bt.to_csv(f"{RESULTS}/s03_table_by_tower_climatology_tabicl.csv")
    print(f"[OK] Saved s03_table_by_tower_climatology_tabicl.csv")

    return combined_all, combined_bt


def main():
    t0 = time.time()
    step1_tree_variants()
    step2_tree_model1()
    step3_foundation_variants()
    step4_foundation_model1()
    step5_rescore_dl_variants()
    step6_build_tables()
    print(f"\n[OK] All steps complete ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
