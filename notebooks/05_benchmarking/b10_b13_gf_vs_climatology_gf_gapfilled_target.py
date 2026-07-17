"""D-7x follow-up: the "scored against gap-filled target vs. observed target" table (D-65 format,
`b10_b13_gf_ablation_results.md`'s "All-tower pooled" section), but with `Climatology_gf` as the
MASE/RMSSE baseline instead of chain-persistence. Same 11-model roster as
`b10_b13_combined_leaderboard_rmsse.py`. Pure recompute over predictions already stored in
`b10_b13_full_chains.csv` -- no refit needed. `y_gapfilled` is used as ground truth for the
"gapfilled" columns (dense, same column for every model); `y_true`/`y_true_tft` for "observed"
(per the project's existing DL-family ground-truth asymmetry).
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


def recompute(chains, truth_col):
    """truth_col: "observed" (y_true/y_true_tft per model) or "gapfilled" (y_gapfilled, same for
    every model). Baseline is always Climatology_gf."""
    all_rows = []
    for tower in TOWERS:
        for yr in ANCHOR_YEARS:
            sub = chains[(chains.tower == tower) & (chains.anchor_year == yr)].copy()
            sub["date"] = pd.to_datetime(sub["date"])
            sub = sub.sort_values("date")
            anchor = pd.Timestamp(f"{yr}-12-16")
            target_dates = pd.DatetimeIndex(sub["date"])
            clim_gf = sub["Climatology_gf"].values
            clim_arg = None if np.isnan(clim_gf).all() else clim_gf
            y_gf = sub["y_gapfilled"].values

            for model in MODEL_ORDER:
                if model not in sub.columns:
                    continue
                yp = sub[model].values
                if np.isfinite(yp).sum() == 0:
                    continue
                if truth_col == "gapfilled":
                    y_true = y_gf
                else:
                    y_true = sub["y_true_tft"].values if model in DL_MODELS else sub["y_true"].values
                bm = rr.bin_metrics(y_true, yp, target_dates, anchor, y_persist=clim_arg)
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

    pooled_obs = build_pooled(recompute(chains, "observed"))
    pooled_gf = build_pooled(recompute(chains, "gapfilled"))

    out = pd.DataFrame({
        "RMSE_gapfilled": pooled_gf["RMSE"], "RMSE_observed": pooled_obs["RMSE"],
        "MAE_gapfilled": pooled_gf["MAE"], "MAE_observed": pooled_obs["MAE"],
        "MASE_gapfilled": pooled_gf["MASE"], "MASE_observed": pooled_obs["MASE"],
        "RMSSE_gapfilled": pooled_gf["RMSSE"], "RMSSE_observed": pooled_obs["RMSSE"],
        "Correlation_gapfilled": pooled_gf["Correlation"], "Correlation_observed": pooled_obs["Correlation"],
        "R2_gapfilled": pooled_gf["R2"], "R2_observed": pooled_obs["R2"],
    }).reindex(MODEL_ORDER)
    out.to_csv(f"{RESULTS}/b10_b13_gf_ablation_vs_climatology_gf_gapfilled_target_table.csv")
    print("[OK] Saved b10_b13_gf_ablation_vs_climatology_gf_gapfilled_target_table.csv")
    print(out.to_string())


if __name__ == "__main__":
    main()
