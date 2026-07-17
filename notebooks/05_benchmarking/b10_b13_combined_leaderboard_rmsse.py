"""D-7x follow-up: adds RMSSE (vs persistence and vs Climatology_gf) to the 11-model combined
leaderboard (`b10_b13_gf_ablation_results.md`'s "Combined leaderboard" section) -- the established
models (RF/XGB/LightGBM/SARIMAX/both Ensembles, already gap-filled-trained, D-36/D-37) plus each
DL-family model's gap-filled-trained `_gf` variant (D-72). Pure recompute over predictions already
stored in `b10_b13_full_chains.csv` via `rr.bin_metrics()` -- no refit needed, RMSSE was only added
to `bin_metrics()` after this leaderboard was first assembled.
"""
import sys

import numpy as np
import pandas as pd

ROOT = r"c:\Users\Nicholas\Documents\COMP0191 MSc Artificial Intelligence for Sustainable Development Project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\src")

import models.recursive_rollout as rr

RESULTS = rf"{ROOT}\results"
TOWERS = [2, 4, 9]
ANCHOR_YEARS = [2018, 2019, 2020, 2021, 2022]
MODEL_ORDER = ["RF", "XGB", "LightGBM", "SARIMAX", "Ensemble_unweighted", "Ensemble_MASEweighted",
               "DLinear_gf", "LSTM_gf", "TFT_gf", "TabPFN_gf", "TabICLv2_gf"]
DL_MODELS = {"DLinear_gf", "LSTM_gf", "TFT_gf"}  # y_true_tft convention
METRICS = ["R2", "RMSE", "MAE", "MASE", "RMSSE", "WAPE", "Correlation"]


def recompute(chains, baseline_col):
    all_rows = []
    for tower in TOWERS:
        for yr in ANCHOR_YEARS:
            sub = chains[(chains.tower == tower) & (chains.anchor_year == yr)].copy()
            sub["date"] = pd.to_datetime(sub["date"])
            sub = sub.sort_values("date")
            anchor = pd.Timestamp(f"{yr}-12-16")
            target_dates = pd.DatetimeIndex(sub["date"])
            baseline = sub[baseline_col].values
            baseline_arg = None if np.isnan(baseline).all() else baseline

            for model in MODEL_ORDER:
                if model not in sub.columns:
                    continue
                yp = sub[model].values
                if np.isfinite(yp).sum() == 0:
                    continue
                y_true = sub["y_true_tft"].values if model in DL_MODELS else sub["y_true"].values
                bm = rr.bin_metrics(y_true, yp, target_dates, anchor, y_persist=baseline_arg)
                bm["model"] = model
                bm["anchor_year"] = yr
                bm["tower"] = tower
                all_rows.append(bm)
    return pd.concat(all_rows, ignore_index=True)


def wavg(g, col):
    d = g.dropna(subset=[col])
    return (d[col] * d["n"]).sum() / d["n"].sum() if d["n"].sum() > 0 else np.nan


def build_pooled(summary_df):
    per_anchor = (summary_df.groupby(["model", "anchor_year"])
                  .apply(lambda g: pd.Series({m: wavg(g, m) for m in METRICS}), include_groups=False)
                  .reset_index())
    return per_anchor.groupby("model")[METRICS].mean().round(3).reindex(MODEL_ORDER)


def main():
    chains = pd.read_csv(f"{RESULTS}/b10_b13_full_chains.csv")

    pooled_persist = build_pooled(recompute(chains, "persistence"))
    pooled_clim = build_pooled(recompute(chains, "Climatology_gf"))

    out = pd.DataFrame({
        "RMSE": pooled_persist["RMSE"],
        "MAE": pooled_persist["MAE"],
        "MASE_persistence": pooled_persist["MASE"],
        "MASE_climatology_gf": pooled_clim["MASE"],
        "RMSSE_persistence": pooled_persist["RMSSE"],
        "RMSSE_climatology_gf": pooled_clim["RMSSE"],
        "WAPE": pooled_persist["WAPE"],
        "Correlation": pooled_persist["Correlation"],
        "R2": pooled_persist["R2"],
    }).reindex(MODEL_ORDER)
    out.to_csv(f"{RESULTS}/b10_b13_latest_combined_leaderboard.csv")
    print("[OK] Updated b10_b13_latest_combined_leaderboard.csv with RMSSE columns")
    print(out.to_string())


if __name__ == "__main__":
    main()
