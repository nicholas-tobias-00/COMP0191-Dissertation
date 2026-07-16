"""D-7x follow-up: reruns the gap-filled-target/-context ablation's MASE comparison (D-72) with
`Climatology_gf` as the MASE denominator instead of chain-persistence -- mirrors
`b10_b13_climatology_gf_baseline.py`'s own `recompute_mase_vs_climatology_gf()` pattern, extended to
the 5 new `_gf`-suffixed models (DLinear_gf/LSTM_gf/TFT_gf/TabPFN_gf/TabICLv2_gf) that script's
MODEL_ORDER predates. Persistence remains this project's primary/default MASE denominator (D-71) --
this is a secondary lens, same caveat-bearing status as the original `Climatology_gf` comparison.
"""
import numpy as np
import pandas as pd
import sys

ROOT = r"c:\Users\Nicholas\Documents\COMP0191 MSc Artificial Intelligence for Sustainable Development Project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\src")

import models.recursive_rollout as rr

RESULTS = rf"{ROOT}\results"
TOWERS = [2, 4, 9]
ANCHOR_YEARS = [2018, 2019, 2020, 2021, 2022]
PAIRS = [("DLinear", "DLinear_gf"), ("LSTM", "LSTM_gf"), ("TFT", "TFT_gf"),
         ("TabPFN", "TabPFN_gf"), ("TabICLv2", "TabICLv2_gf")]
MODELS = [m for pair in PAIRS for m in pair]
DL_MODELS = {"TFT", "DLinear", "LSTM", "TFT_gf", "DLinear_gf", "LSTM_gf"}  # y_true_tft convention
METRICS = ["R2", "RMSE", "MAE", "MASE", "WAPE", "Correlation"]


def recompute(chains):
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

            for model in MODELS:
                if model not in sub.columns:
                    continue
                yp = sub[model].values
                if np.isfinite(yp).sum() == 0:
                    continue
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
    return per_anchor.groupby("model")[METRICS].mean().round(3).reindex(MODELS)


def with_deltas(pooled):
    rows = []
    for base, gf_name in PAIRS:
        row = {"model": base}
        for m in METRICS:
            v_orig = pooled.loc[base, m] if base in pooled.index else np.nan
            v_gf = pooled.loc[gf_name, m] if gf_name in pooled.index else np.nan
            row[f"{m}_orig"] = v_orig
            row[f"{m}_gf"] = v_gf
            row[f"{m}_delta"] = round(v_gf - v_orig, 3) if np.isfinite(v_orig) and np.isfinite(v_gf) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def main():
    chains = pd.read_csv(f"{RESULTS}/b10_b13_full_chains.csv")
    out = recompute(chains)
    out.to_csv(f"{RESULTS}/b10_b13_gf_ablation_vs_climatology_gf_summary.csv", index=False)
    print(f"[OK] Saved b10_b13_gf_ablation_vs_climatology_gf_summary.csv ({len(out)} rows)")

    pooled = build_pooled(out)
    cmp = with_deltas(pooled)
    cmp.to_csv(f"{RESULTS}/b10_b13_gf_ablation_vs_climatology_gf_table.csv")
    print("\n=== Pooled, MASE denominator = Climatology_gf (not persistence) ===")
    print(cmp[["MASE_orig", "MASE_gf", "MASE_delta", "R2_orig", "R2_gf", "R2_delta"]].to_string())


if __name__ == "__main__":
    main()
