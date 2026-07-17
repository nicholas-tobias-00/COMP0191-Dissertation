"""D-7x follow-up: full parity for the per-tower and tower x year x model breakdown sections --
crosses the gapfilled/observed evaluation-target axis (D-65) with the persistence/Climatology_gf
MASE-RMSSE-baseline axis (D-71/D-72), for all 11 models in the B-09-B-15 sequence (established
RF/XGB/LightGBM/SARIMAX/both Ensembles, already gap-filled-trained, D-36/D-37, + each DL-family
model's gap-filled-trained `_gf` variant, D-72). Pure recompute over predictions already stored in
`b10_b13_full_chains.csv` -- no refit needed. RMSE/MAE/Correlation/R2 depend only on the evaluation
target (gapfilled/observed), not the baseline, so each is computed once per target and reused for
both baselines; MASE/RMSSE need all 4 (target x baseline) combinations.
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


def recompute(chains, truth_col, baseline_col):
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
                bm = rr.bin_metrics(y_true, yp, target_dates, anchor, y_persist=baseline_arg)
                bm["model"] = model
                bm["anchor_year"] = yr
                bm["tower"] = tower
                all_rows.append(bm)
    return pd.concat(all_rows, ignore_index=True)


def wavg(g, col):
    d = g.dropna(subset=[col])
    return (d[col] * d["n"]).sum() / d["n"].sum() if d["n"].sum() > 0 else np.nan


def build_by_tower(summary_df):
    per_anchor = (summary_df.groupby(["tower", "model", "anchor_year"])
                  .apply(lambda g: pd.Series({m: wavg(g, m) for m in METRICS}), include_groups=False)
                  .reset_index())
    out = per_anchor.groupby(["tower", "model"])[METRICS].mean().round(3)
    return out.reindex(pd.MultiIndex.from_product([TOWERS, MODEL_ORDER], names=["tower", "model"]))


def build_by_tower_year(summary_df):
    per_anchor = (summary_df.groupby(["tower", "model", "anchor_year"])
                  .apply(lambda g: pd.Series({m: wavg(g, m) for m in METRICS}), include_groups=False)
                  .reset_index())
    return per_anchor


def main():
    chains = pd.read_csv(f"{RESULTS}/b10_b13_full_chains.csv")

    combos = {}
    for truth in ("gapfilled", "observed"):
        for baseline, tag in (("persistence", "p"), ("Climatology_gf", "c")):
            combos[(truth, tag)] = recompute(chains, truth, baseline)

    # ---- by-tower ----
    bt = {}
    for (truth, tag), df in combos.items():
        bt[(truth, tag)] = build_by_tower(df)

    by_tower = pd.DataFrame(index=bt[("gapfilled", "p")].index)
    by_tower["RMSE_gapfilled"] = bt[("gapfilled", "p")]["RMSE"]
    by_tower["RMSE_observed"] = bt[("observed", "p")]["RMSE"]
    by_tower["MAE_gapfilled"] = bt[("gapfilled", "p")]["MAE"]
    by_tower["MAE_observed"] = bt[("observed", "p")]["MAE"]
    for truth in ("gapfilled", "observed"):
        for tag, label in (("p", "persistence"), ("c", "climatology_gf")):
            by_tower[f"MASE_{truth}_{label}"] = bt[(truth, tag)]["MASE"]
            by_tower[f"RMSSE_{truth}_{label}"] = bt[(truth, tag)]["RMSSE"]
    by_tower["Correlation_gapfilled"] = bt[("gapfilled", "p")]["Correlation"]
    by_tower["Correlation_observed"] = bt[("observed", "p")]["Correlation"]
    by_tower["R2_gapfilled"] = bt[("gapfilled", "p")]["R2"]
    by_tower["R2_observed"] = bt[("observed", "p")]["R2"]
    by_tower.to_csv(f"{RESULTS}/b10_b13_full_parity_gf_obs_by_tower.csv")
    print("[OK] Saved b10_b13_full_parity_gf_obs_by_tower.csv")
    print(by_tower.to_string())

    # ---- tower x year ----
    ty = {}
    for (truth, tag), df in combos.items():
        ty[(truth, tag)] = build_by_tower_year(df)

    base = ty[("gapfilled", "p")][["tower", "model", "anchor_year", "RMSE", "MAE", "Correlation", "R2"]]
    base = base.rename(columns={"RMSE": "RMSE_gapfilled", "MAE": "MAE_gapfilled",
                                 "Correlation": "Correlation_gapfilled", "R2": "R2_gapfilled"})
    obs_base = ty[("observed", "p")][["tower", "model", "anchor_year", "RMSE", "MAE", "Correlation", "R2"]]
    obs_base = obs_base.rename(columns={"RMSE": "RMSE_observed", "MAE": "MAE_observed",
                                         "Correlation": "Correlation_observed", "R2": "R2_observed"})
    out = base.merge(obs_base, on=["tower", "model", "anchor_year"])
    for truth in ("gapfilled", "observed"):
        for tag, label in (("p", "persistence"), ("c", "climatology_gf")):
            sub = ty[(truth, tag)][["tower", "model", "anchor_year", "MASE", "RMSSE"]].rename(
                columns={"MASE": f"MASE_{truth}_{label}", "RMSSE": f"RMSSE_{truth}_{label}"})
            out = out.merge(sub, on=["tower", "model", "anchor_year"])
    out.to_csv(f"{RESULTS}/b10_b13_full_parity_gf_obs_table_by_tower_year.csv", index=False)
    print(f"\n[OK] Saved b10_b13_full_parity_gf_obs_table_by_tower_year.csv ({len(out)} rows)")


if __name__ == "__main__":
    main()
