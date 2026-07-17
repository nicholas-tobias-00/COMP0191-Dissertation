"""D-7x follow-up: full MASE/RMSSE parity (both persistence and Climatology_gf baselines) for the
per-tower and tower x year x model breakdown sections of `b10_b13_gf_ablation_results.md` -- these
previously only showed MASE vs persistence, no RMSSE at all. Same 10-model DL-family roster
(5 orig + 5 gf pairs) already used in those sections. Pure recompute over predictions already stored
in `b10_b13_full_chains.csv` -- no refit needed.
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
PAIRS = [("DLinear", "DLinear_gf"), ("LSTM", "LSTM_gf"), ("TFT", "TFT_gf"),
         ("TabPFN", "TabPFN_gf"), ("TabICLv2", "TabICLv2_gf")]
MODEL_ORDER = [m for pair in PAIRS for m in pair]
DL_MODELS = {"TFT", "DLinear", "LSTM", "TFT_gf", "DLinear_gf", "LSTM_gf"}
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


def build_by_tower(summary_df):
    per_anchor = (summary_df.groupby(["tower", "model", "anchor_year"])
                  .apply(lambda g: pd.Series({m: wavg(g, m) for m in METRICS}), include_groups=False)
                  .reset_index())
    out = per_anchor.groupby(["tower", "model"])[METRICS].mean().round(3)
    return out.reindex(pd.MultiIndex.from_product([TOWERS, MODEL_ORDER], names=["tower", "model"]))


def build_by_tower_year(summary_df):
    """No cross-anchor averaging -- per-anchor n-weighted value directly, matching the existing
    tower x year x model breakdown's convention."""
    per_anchor = (summary_df.groupby(["tower", "model", "anchor_year"])
                  .apply(lambda g: pd.Series({m: wavg(g, m) for m in METRICS}), include_groups=False)
                  .reset_index())
    return per_anchor


def main():
    chains = pd.read_csv(f"{RESULTS}/b10_b13_full_chains.csv")

    out_p = recompute(chains, "persistence")
    out_c = recompute(chains, "Climatology_gf")

    bt_p = build_by_tower(out_p)[["MASE", "RMSSE"]].rename(columns={"MASE": "MASE_p", "RMSSE": "RMSSE_p"})
    bt_c = build_by_tower(out_c)[["MASE", "RMSSE"]].rename(columns={"MASE": "MASE_c", "RMSSE": "RMSSE_c"})
    bt_other = build_by_tower(out_p)[["RMSE", "MAE", "WAPE", "Correlation", "R2"]]
    by_tower = pd.concat([bt_other, bt_p, bt_c], axis=1)
    by_tower.to_csv(f"{RESULTS}/b10_b13_gf_ablation_by_tower_full_parity.csv")
    print("[OK] Saved b10_b13_gf_ablation_by_tower_full_parity.csv")
    print(by_tower.to_string())

    ty_p = build_by_tower_year(out_p)
    ty_c = build_by_tower_year(out_c)
    ty = ty_p.merge(ty_c[["tower", "model", "anchor_year", "MASE", "RMSSE"]],
                     on=["tower", "model", "anchor_year"], suffixes=("_p", "_c"))
    ty = ty.rename(columns={"MASE_p": "MASE_persistence", "MASE_c": "MASE_climatology_gf",
                             "RMSSE_p": "RMSSE_persistence", "RMSSE_c": "RMSSE_climatology_gf"})
    ty.to_csv(f"{RESULTS}/b10_b13_gf_ablation_table_by_tower_year_full_parity.csv", index=False)
    print(f"\n[OK] Saved b10_b13_gf_ablation_table_by_tower_year_full_parity.csv ({len(ty)} rows)")


if __name__ == "__main__":
    main()
