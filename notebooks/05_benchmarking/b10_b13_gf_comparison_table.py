"""D-7x: builds the actual answer table for the gap-filled-target/-context ablation -- each DL-family
model's original (y_observed-trained/-context) result next to its _gf (y_gapfilled-trained/-context)
counterpart, pooled and by-tower. Aggregation convention (n-weighted mean per anchor, then mean
across the 5 anchors) matches `b10_b13_climatology_baseline.py`'s `wavg`/`build_pooled`/
`build_by_tower` helpers exactly -- reimplemented here (not imported) since those helpers are
hardcoded to that module's own MODEL_ORDER, which doesn't include the 5 new _gf models.
"""
import numpy as np
import pandas as pd

ROOT = r"c:\Users\Nicholas\Documents\COMP0191 MSc Artificial Intelligence for Sustainable Development Project"
RESULTS = rf"{ROOT}\results"

TOWERS = [2, 4, 9]
METRICS = ["R2", "RMSE", "MAE", "MASE", "WAPE", "Correlation"]
PAIRS = [("DLinear", "DLinear_gf"), ("LSTM", "LSTM_gf"), ("TFT", "TFT_gf"),
         ("TabPFN", "TabPFN_gf"), ("TabICLv2", "TabICLv2_gf")]
MODEL_ORDER = [m for pair in PAIRS for m in pair]


def wavg(g, col):
    d = g.dropna(subset=[col])
    return (d[col] * d["n"]).sum() / d["n"].sum() if d["n"].sum() > 0 else np.nan


def build_pooled(summary_df):
    per_anchor = (summary_df.groupby(["model", "anchor_year"])
                  .apply(lambda g: pd.Series({m: wavg(g, m) for m in METRICS}), include_groups=False)
                  .reset_index())
    return per_anchor.groupby("model")[METRICS].mean().round(3).reindex(MODEL_ORDER)


def build_by_tower(summary_df):
    per_anchor = (summary_df.groupby(["tower", "model", "anchor_year"])
                  .apply(lambda g: pd.Series({m: wavg(g, m) for m in METRICS}), include_groups=False)
                  .reset_index())
    return (per_anchor.groupby(["tower", "model"])[METRICS].mean().round(3)
            .reindex(pd.MultiIndex.from_product([TOWERS, MODEL_ORDER], names=["tower", "model"])))


def load_all():
    rerun = pd.read_csv(f"{RESULTS}/b10_b13_rerun_summary.csv")
    dl = pd.read_csv(f"{RESULTS}/b10_b13_dl_extension_summary.csv")
    tabicl = pd.read_csv(f"{RESULTS}/b10_b13_tabicl_extension_summary.csv")
    dl_gf = pd.read_csv(f"{RESULTS}/b10_b13_dl_gf_extension_summary.csv")
    tft_gf = pd.read_csv(f"{RESULTS}/b10_b13_tft_gf_extension_summary.csv")
    fnd_gf = pd.read_csv(f"{RESULTS}/b10_b13_foundation_gf_extension_summary.csv")

    orig = pd.concat([
        rerun[rerun["model"].isin(["TFT", "TabPFN"])],
        dl[dl["model"].isin(["DLinear", "LSTM"])],
        tabicl,
    ], ignore_index=True)
    gf = pd.concat([dl_gf, tft_gf, fnd_gf], ignore_index=True)
    return pd.concat([orig, gf], ignore_index=True)


def with_deltas(pooled):
    """Reshapes the pooled table into original/gf/delta columns per model pair, per metric."""
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
    combined = load_all()
    combined.to_csv(f"{RESULTS}/b10_b13_gf_ablation_combined_summary.csv", index=False)
    print(f"[OK] Saved b10_b13_gf_ablation_combined_summary.csv ({len(combined)} rows)")

    pooled = build_pooled(combined)
    pooled.to_csv(f"{RESULTS}/b10_b13_gf_ablation_pooled.csv")
    print("\n=== Pooled (all towers/anchors), original vs gf ===")
    print(pooled[["MASE", "R2"]].to_string())

    pooled_cmp = with_deltas(pooled)
    pooled_cmp.to_csv(f"{RESULTS}/b10_b13_gf_ablation_table_all_towers.csv")
    print("\n=== Pooled comparison (MASE/R2 only) ===")
    print(pooled_cmp[["MASE_orig", "MASE_gf", "MASE_delta", "R2_orig", "R2_gf", "R2_delta"]].to_string())

    by_tower = build_by_tower(combined)
    by_tower.to_csv(f"{RESULTS}/b10_b13_gf_ablation_by_tower.csv")

    bt_rows = []
    for t in TOWERS:
        sub = by_tower.loc[t]
        cmp = with_deltas(sub)
        cmp["tower"] = t
        bt_rows.append(cmp.reset_index())
    bt_cmp = pd.concat(bt_rows, ignore_index=True).set_index(["tower", "model"])
    bt_cmp.to_csv(f"{RESULTS}/b10_b13_gf_ablation_table_by_tower.csv")
    print("\n=== By-tower comparison (MASE/R2 only) ===")
    print(bt_cmp[["MASE_orig", "MASE_gf", "MASE_delta", "R2_orig", "R2_gf", "R2_delta"]].to_string())


if __name__ == "__main__":
    main()
