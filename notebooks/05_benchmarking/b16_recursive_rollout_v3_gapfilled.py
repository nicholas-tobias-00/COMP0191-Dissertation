"""F-10 (D-67) Stage 2b/BASE+ALL: secondary metric vs. y_gapfilled, recomputed from the
already-saved raw chains (no refitting needed -- reuses `results/b16_recursive_rollout_v3_chains.csv`
and `_all_chains.csv`, mirroring the D-65 second addendum's own "score every chain a second time"
pattern). Same explicit circularity caveat applies here as everywhere else this pattern is used:
y_gapfilled seeds each chain's history_init AND is a pooled RFm gap-filler's output trained on
features that overlap the tree models' own forecast features -- read directionally, not as
validated accuracy.

Run from project root:  python notebooks/05_benchmarking/b16_recursive_rollout_v3_gapfilled.py
"""
import sys

import numpy as np
import pandas as pd

ROOT = r"c:\Users\Nicholas\Documents\COMP0191 MSc Artificial Intelligence for Sustainable Development Project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\src")

import models.recursive_rollout as rr

RESULTS = rf"{ROOT}\results"

MODELS = ["RF", "XGB", "LightGBM", "SARIMAX", "Ensemble_unweighted", "Ensemble_MASEweighted"]
CONFIGS = ["BASE", "BASE+species", "BASE+arable", "BASE+flow", "BASE+mgmt", "BASE+bodyweight", "BASE+ALL"]


def score_chains(chains):
    """Each (config, tower, anchor) combo was appended as its own chain_df during the original
    sweep, so pd.concat produced a UNION of columns across configs -- for any given row, only the
    columns from the config that produced that row are non-null, every other config's columns are
    NaN on that row. Must filter to the rows belonging to each (config, model) column being scored
    (via `.notna()` on that exact column) before calling bin_metrics -- passing the raw grouped
    frame directly would silently mix in NaN-prediction rows from other configs' rows and crash
    r2_score (confirmed empirically: this was the actual cause of an "Input contains NaN" error
    caught while writing this script)."""
    chains["date"] = pd.to_datetime(chains["date"])
    rows = []
    for (tower, yr), sub in chains.groupby(["tower", "anchor_year"]):
        anchor = pd.Timestamp(f"{yr}-12-16")
        for cfg in CONFIGS:
            for model in MODELS:
                col = f"{cfg}__{model}"
                if col not in sub.columns:
                    continue
                cfg_rows = sub[sub[col].notna()].sort_values("date")
                if len(cfg_rows) == 0:
                    continue
                target_dates = cfg_rows["date"]
                y_true = cfg_rows["y_true"].values
                y_gf = cfg_rows["y_gapfilled"].values
                persist = cfg_rows["persistence"].values
                yp = cfg_rows[col].values
                bm = rr.bin_metrics(y_true, yp, target_dates, anchor, y_persist=persist)
                bm["target"] = "observed"; bm["model"] = model; bm["config"] = cfg
                bm["anchor_year"] = yr; bm["tower"] = tower
                rows.append(bm)
                bm_gf = rr.bin_metrics(y_gf, yp, target_dates, anchor, y_persist=persist)
                bm_gf["target"] = "gapfilled"; bm_gf["model"] = model; bm_gf["config"] = cfg
                bm_gf["anchor_year"] = yr; bm_gf["tower"] = tower
                rows.append(bm_gf)
    return pd.concat(rows, ignore_index=True)


def main():
    c1 = pd.read_csv(f"{RESULTS}/b16_recursive_rollout_v3_chains.csv")
    c2 = pd.read_csv(f"{RESULTS}/b16_recursive_rollout_v3_all_chains.csv")
    all_scored = pd.concat([score_chains(c1), score_chains(c2)], ignore_index=True)
    all_scored.to_csv(f"{RESULTS}/b16_recursive_rollout_v3_summary_vs_gapfilled.csv", index=False)
    print(f"[OK] Saved b16_recursive_rollout_v3_summary_vs_gapfilled.csv ({len(all_scored)} rows)")

    def wavg(g):
        return pd.Series({
            "RMSE": (g.RMSE * g.n).sum() / g.n.sum(), "MASE": (g.MASE * g.n).sum() / g.n.sum(),
            "Correlation": (g.Correlation * g.n).sum() / g.n.sum(), "R2": (g.R2 * g.n).sum() / g.n.sum()})

    per_anchor = all_scored.groupby(["model", "config", "target", "anchor_year"]).apply(wavg, include_groups=False)
    allt = per_anchor.groupby(["model", "config", "target"]).mean().reset_index()
    piv = allt.pivot_table(index=["model", "config"], columns="target", values=["RMSE", "MASE", "Correlation", "R2"])
    piv.columns = [f"{m} ({t})" for m, t in piv.columns]
    piv = piv[["RMSE (gapfilled)", "RMSE (observed)", "MASE (gapfilled)", "MASE (observed)",
               "Correlation (gapfilled)", "Correlation (observed)", "R2 (gapfilled)", "R2 (observed)"]]
    piv.to_csv(f"{RESULTS}/b16_recursive_rollout_v3_table_vs_gapfilled.csv")
    print(f"[OK] Saved b16_recursive_rollout_v3_table_vs_gapfilled.csv")
    print(piv.round(3).to_string())


if __name__ == "__main__":
    main()
