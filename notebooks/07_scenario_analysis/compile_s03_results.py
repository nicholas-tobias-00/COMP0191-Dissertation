"""S-03: builds the 3-way comparison (Model 1 / Variant A removal / Variant B resample) from
s03_driver_availability_ablation.py's + s03_model_roster_extension.py's raw output, reading Model
1's numbers directly from the existing D-65/D-65-addendum files -- NOT recomputed, per the plan's
explicit scope (Model 1 is an existing result, referenced only).

Per-target tables are built first (observed / gap-filled separately: s03_table_all_towers.csv /
_by_tower.csv for observed, s03_table_vs_gapfilled_all_towers.csv / _by_tower.csv for gap-filled --
kept as focused single-target views), then **combined into one primary table** per user request
("add gap-filled FCH4 as a metric as well" -- read as: don't make the reader open a second file to
see it) at `s03_table_all_towers.csv` / `s03_table_by_tower.csv`, which are OVERWRITTEN with the
3-level-column combined version: (target in {Observed, GapFilled}, source in {Model1, VariantA,
VariantB}, metric). The gap-filled column carries D-65's own circularity caveat (y_gapfilled seeds
history_init and shares feature space with the forecasters) -- read directionally, not as validated
accuracy, same as everywhere else this secondary metric appears in this project.

Model roster (D-70 follow-up, model-roster-extension fix): now the FULL 11-model B-10/B-13/B-16
roster -- originally restricted to the 6 rows B-10's own architecture covers, a real scope gap
(Model 2 never fit TFT/TabPFN/DLinear/LSTM/TabICLv2) fixed after direct user challenge. Model 1's
numbers for the 5 extension models were never folded back into `b10_b13_rerun_table_*` -- rather
than reading each family's own pre-built "table" file (three different shapes, and the by-tower-year
granularity doesn't exist at all for the extensions' gap-filled target), Model 1 is now built by
concatenating the three RAW per-bin summary files (`b10_b13_rerun_summary*.csv` +
`b10_b13_dl_extension_summary*.csv` + `b10_b13_tabicl_extension_summary*.csv` -- all three share the
exact same bin/n/METRICS/model/anchor_year/tower schema) and reusing `build_pooled`/`build_by_tower`
-- the SAME aggregation Variant A/B already go through, verified to reproduce the original
`b10_b13_rerun_table_all_towers.csv` bit-for-bit before adopting this approach.
"""
import pandas as pd
import numpy as np

RESULTS = r"c:\Users\Nicholas\Documents\COMP0191 MSc Artificial Intelligence for Sustainable Development Project\results"

METRICS = ["R2", "RMSE", "MAE", "MASE", "WAPE", "Correlation"]
MODEL_ORDER = ["RF", "XGB", "LightGBM", "SARIMAX", "Ensemble_unweighted", "Ensemble_MASEweighted",
               "TFT", "TabPFN", "DLinear", "LSTM", "TabICLv2"]
TOWERS = [2, 4, 9]


def wavg(g, col):
    d = g.dropna(subset=[col])
    return (d[col] * d["n"]).sum() / d["n"].sum() if d["n"].sum() > 0 else np.nan


def build_pooled(summary_df):
    """Matches b10_b13_rerun_multi_anchor.py's own all-tower aggregation exactly:
    per-anchor n-weighted mean across bins+towers, then mean across anchors."""
    per_anchor = (summary_df.groupby(["model", "anchor_year"])
                  .apply(lambda g: pd.Series({m: wavg(g, m) for m in METRICS}), include_groups=False)
                  .reset_index())
    return per_anchor.groupby("model")[METRICS].mean().round(3).reindex(MODEL_ORDER)


def build_by_tower(summary_df):
    """Per-tower, anchors averaged (n-weighted mean across bins per anchor, then mean across
    anchors, per tower)."""
    per_anchor = (summary_df.groupby(["tower", "model", "anchor_year"])
                  .apply(lambda g: pd.Series({m: wavg(g, m) for m in METRICS}), include_groups=False)
                  .reset_index())
    out = per_anchor.groupby(["tower", "model"])[METRICS].mean().round(3)
    out = out.reindex(pd.MultiIndex.from_product([TOWERS, MODEL_ORDER], names=["tower", "model"]))
    return out


def three_way(model1_df, variantA_df, variantB_df):
    """Merges 3 same-shaped (index=model or (tower,model), columns=METRICS) frames into one
    MultiIndex-column comparison table: columns = (source, metric)."""
    parts = {"Model1": model1_df[METRICS], "VariantA_removal": variantA_df[METRICS],
             "VariantB_resample": variantB_df[METRICS]}
    out = pd.concat(parts, axis=1)
    out = out.reorder_levels([0, 1], axis=1)
    return out


def load_model1_raw(target_suffix):
    """Model 1's raw per-bin predictions, all 11 models -- concatenates the 3 files each model
    family was originally computed in (never merged upstream): `b10_b13_rerun_summary*.csv` (8
    models), `b10_b13_dl_extension_summary*.csv` (DLinear/LSTM), `b10_b13_tabicl_extension_summary
    *.csv` (TabICLv2). target_suffix: "" for observed, "_vs_gapfilled" for gap-filled."""
    core = pd.read_csv(f"{RESULTS}/b10_b13_rerun_summary{target_suffix}.csv")
    dl = pd.read_csv(f"{RESULTS}/b10_b13_dl_extension_summary{target_suffix}.csv")
    tabicl = pd.read_csv(f"{RESULTS}/b10_b13_tabicl_extension_summary{target_suffix}.csv")
    return pd.concat([core, dl, tabicl], ignore_index=True)


def run_for_target(summary_path, model1_suffix, suffix):
    df = pd.read_csv(summary_path)
    df_A = df[df["variant"] == "A_removal"]
    df_B = df[df["variant"] == "B_resample"]
    model1_raw = load_model1_raw(model1_suffix)

    # ---- All-towers pooled ----
    model1_all = build_pooled(model1_raw)
    varA_all = build_pooled(df_A)
    varB_all = build_pooled(df_B)
    table_all = three_way(model1_all, varA_all, varB_all)
    out_path = f"{RESULTS}/s03_table{suffix}_all_towers.csv"
    table_all.to_csv(out_path)
    print(f"[OK] Saved {out_path}")

    # ---- Per-tower (anchors averaged) ----
    model1_bt = build_by_tower(model1_raw)
    varA_bt = build_by_tower(df_A)
    varB_bt = build_by_tower(df_B)
    table_bt = three_way(model1_bt, varA_bt, varB_bt)
    out_path2 = f"{RESULTS}/s03_table{suffix}_by_tower.csv"
    table_bt.to_csv(out_path2)
    print(f"[OK] Saved {out_path2}")

    return table_all, table_bt


def combine_and_overwrite(obs_table, gf_table, out_path):
    """Merges the observed-target and gap-filled-target 3-way tables into one 3-level-column
    table (target, source, metric) and overwrites out_path -- the primary/headline file."""
    combined = pd.concat({"Observed": obs_table, "GapFilled": gf_table}, axis=1)
    combined.to_csv(out_path)
    print(f"[OK] Saved combined (Observed+GapFilled) {out_path}")
    return combined


def main():
    print("=== Observed target ===")
    obs_all, obs_bt = run_for_target(f"{RESULTS}/s03_summary.csv", "", "")
    print("\n=== Gap-filled target (secondary, exploratory -- see circularity caveat) ===")
    gf_all, gf_bt = run_for_target(f"{RESULTS}/s03_summary_vs_gapfilled.csv", "_vs_gapfilled", "_vs_gapfilled")

    print("\n=== Combining into primary (headline) tables ===")
    combine_and_overwrite(obs_all, gf_all, f"{RESULTS}/s03_table_all_towers.csv")
    combine_and_overwrite(obs_bt, gf_bt, f"{RESULTS}/s03_table_by_tower.csv")


if __name__ == "__main__":
    main()
