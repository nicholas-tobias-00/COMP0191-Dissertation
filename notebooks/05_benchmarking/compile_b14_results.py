"""Compile B-14 results and generate b14_results.md narrative."""

import pandas as pd
from pathlib import Path
import os

ROOT = Path(r"c:\Users\Nicholas\Documents\COMP0191 MSc Artificial Intelligence for Sustainable Development Project")
RESULTS = ROOT / "results"
NOTEBOOK_DIR = ROOT / "notebooks" / "05_benchmarking"

def compile_results():
    """Load and compile all B-14 results into comprehensive summary."""

    print("="*70)
    print("B-14 RESULTS COMPILATION")
    print("="*70)

    # Load rollout validation results
    try:
        rollout_summary = pd.read_csv(RESULTS/"b14_tuned_rollout_summary.csv")
        print(f"\n[OK] Loaded rollout summary: {len(rollout_summary)} rows")
    except FileNotFoundError:
        print("ERROR: b14_tuned_rollout_summary.csv not found")
        return None, None

    # Calculate per-model aggregates (n-weighted mean across bins)
    def wavg(g, col):
        w = g["n"]
        return (g[col] * w).sum() / w.sum() if w.sum() > 0 else float('nan')

    agg_by_model_anchor = rollout_summary.groupby(["model", "anchor_year"]).apply(
        lambda g: pd.Series({"R2": wavg(g, "R2"), "MASE": wavg(g, "MASE")}),
        include_groups=False
    ).reset_index()

    agg_by_model = agg_by_model_anchor.groupby("model")[["R2", "MASE"]].mean().reset_index()
    agg_by_model.columns = ["model", "mean_R2", "mean_MASE"]
    agg_by_model = agg_by_model.sort_values("mean_R2", ascending=False)

    print("\nPer-model aggregate (5-anchor mean):")
    print(agg_by_model.to_string(index=False))

    # B-10 baseline for comparison -- real per-bin data (not a hardcoded scalar dict, which
    # previously disagreed with the precise numbers used in compile_b15_results.py). Aggregated
    # as n-weighted mean per anchor, then simple mean across anchors -- matches this project's
    # established convention (reproduces D-54's exact published XGB=0.003/LightGBM=-0.014/
    # SARIMAX=-0.039/RF=-0.067), not a pooled mean across all anchor-bin rows.
    b10_raw = pd.read_csv(RESULTS/"b10_ensemble_multi_anchor.csv")
    b10_per_anchor = b10_raw.groupby(["model", "anchor_year"]).apply(
        lambda g: pd.Series({"R2": wavg(g, "R2"), "MASE": wavg(g, "MASE")}),
        include_groups=False
    ).reset_index()
    b10_agg = b10_per_anchor.groupby("model")[["R2", "MASE"]].mean().reset_index()
    B10_BASELINE = {row["model"]: {"mean_r2": row["R2"], "mean_mase": row["MASE"]} for _, row in b10_agg.iterrows()}

    baseline_df = pd.DataFrame([
        {"model": f"{k} (B-10)", "mean_R2": v["mean_r2"], "mean_MASE": v["mean_mase"]}
        for k, v in B10_BASELINE.items()
    ])

    print("\n\nB-10 Baseline for comparison:")
    print(baseline_df.to_string(index=False))

    # Compile summary comparison table
    comparison = pd.concat([agg_by_model, baseline_df], ignore_index=True).sort_values("mean_R2", ascending=False)

    print("\n\nCOMPREHENSIVE COMPARISON: Tuned vs B-10 Baseline")
    print(comparison.to_string(index=False))

    # Find improvements -- match by normalized family, since B-14's own model names
    # (e.g. "RF_tuned", "SARIMAX_widened") never literally match B10_BASELINE's raw names
    # (e.g. "RF", "SARIMAX"); this loop previously never fired for any model as a result.
    def normalize(name):
        return name.replace("_tuned", "").replace("_widened", "")

    baseline_by_family = {normalize(k): v for k, v in B10_BASELINE.items()}

    print("\n\nIMPROVEMENTS OVER B-10 BASELINE:")
    for model in agg_by_model["model"].unique():
        fam = normalize(model)
        if fam in baseline_by_family and not model.startswith("Ensemble"):
            tuned = agg_by_model[agg_by_model["model"] == model].iloc[0]
            baseline = baseline_by_family[fam]
            r2_delta = tuned["mean_R2"] - baseline["mean_r2"]
            mase_delta = tuned["mean_MASE"] - baseline["mean_mase"]
            print(f"\n{model} vs {fam} (B-10):")
            print(f"  R2 change: {tuned['mean_R2']:.4f} vs {baseline['mean_r2']:.4f} (Δ = {r2_delta:+.4f})")
            print(f"  MASE change: {tuned['mean_MASE']:.4f} vs {baseline['mean_mase']:.4f} (Δ = {mase_delta:+.4f})")

    return agg_by_model, comparison, B10_BASELINE

def write_results_markdown(agg_by_model, comparison, b10_baseline):
    """Write b14_results.md markdown narrative."""

    b10_ens_r2 = b10_baseline["Ensemble_unweighted"]["mean_r2"]
    b10_ens_mase = b10_baseline["Ensemble_unweighted"]["mean_mase"]

    md = """# B-14 — Comprehensive Hyperparameter Tuning for Recursive Rollout

**Objective:** Systematically tune tree models (RF/XGB/LightGBM), SARIMAX, via manual grid search on 2020-2021 validation fold, then validate on the full 5-anchor (2018-2022) recursive rollout.

**Design:**
- **Stage 1 (Grid Search):** Manual parameter grid on 2020-2021 validation fold, selecting by R2
- **Stage 2 (Rollout Validation):** Plug winning hyperparameters into B-10's exact 5-anchor mechanism, compare mean R2/MASE directly against B-10 baseline
- **Scope:** Tree models (RF/XGB/LightGBM) and SARIMAX in the recursive-rollout sequence

## Key Findings

### Overall Results (5-anchor mean R2/MASE)

"""

    md += comparison.to_string(index=False)
    md += "\n\n"

    md += "### Interpretation\n\n"

    # Find best tuned model
    tuned_models = agg_by_model[~agg_by_model["model"].str.contains("Ensemble_")]
    if len(tuned_models) > 0:
        best_tuned = tuned_models.iloc[0]
        md += f"**Best tuned single model:** {best_tuned['model']} (R2={best_tuned['mean_R2']:.4f}, MASE={best_tuned['mean_MASE']:.4f})\n\n"

    # Check for ensemble tuned variant
    ensemble_rows = agg_by_model[agg_by_model["model"].str.contains("Ensemble")]
    if len(ensemble_rows) > 0:
        md += f"**Tuned ensemble:** {ensemble_rows.iloc[0]['model']} (R2={ensemble_rows.iloc[0]['mean_R2']:.4f}, MASE={ensemble_rows.iloc[0]['mean_MASE']:.4f})\n\n"

    # Compare to B-10 ensemble baseline
    md += f"**vs. B-10 Ensemble baseline (R2={b10_ens_r2:.4f}, MASE={b10_ens_mase:.4f}):**\n\n"

    # Check if any tuned model beat the baseline
    improvements = []
    for _, row in agg_by_model.iterrows():
        r2_gain = row["mean_R2"] - b10_ens_r2
        mase_change = row["mean_MASE"] - b10_ens_mase
        if r2_gain > 0.001:  # More than 0.001 improvement
            improvements.append((row["model"], r2_gain, mase_change))

    if improvements:
        md += "[OK] **Hyperparameter tuning yielded improvements:**\n"
        for model, r2_gain, mase_change in sorted(improvements, key=lambda x: -x[1])[:3]:
            md += f"  - {model}: +{r2_gain:.4f} R2 (MASE {mase_change:+.4f})\n"
    else:
        md += "[FAIL] Hyperparameter tuning did not improve upon B-10's baseline on the 5-anchor rollout.\n"
        md += "   This is a legitimate finding: one-step CV performance (where tuning was optimized) diverges from 365-day rollout performance.\n"

    md += "\n### Per-Model Results\n\n"
    md += "| Model | Mean R2 | Mean MASE | vs B-10 Ensemble |\n"
    md += "|---|---|---|---|\n"

    for _, row in agg_by_model.iterrows():
        delta_r2 = row["mean_R2"] - b10_ens_r2
        delta_str = f"{delta_r2:+.4f}" if delta_r2 != 0 else "-"
        md += f"| {row['model']} | {row['mean_R2']:.4f} | {row['mean_MASE']:.4f} | {delta_str} |\n"

    md += "\n## Methodology\n\n"
    md += "### Grid Search Stage\n"
    md += "- **RF:** 16 parameter combos (max_features in {0.3,0.5,0.7,1.0} * min_samples_leaf in {5,10,20,50})\n"
    md += "- **XGB:** 36 combos (max_depth in {2,3,4,6} * learning_rate in {0.01,0.02,0.05} * min_child_weight in {1,5,10})\n"
    md += "- **LightGBM:** 36 combos (num_leaves in {7,15,31,63} * min_child_samples in {10,20,50} * learning_rate in {0.01,0.02,0.05})\n"
    md += "- **SARIMAX:** 9 order combos (p in {1,2,3} * q in {0,1,2}, d=1 fixed)\n"
    md += "- **Validation fold:** 2020-2021 (independent of rollout test window)\n\n"

    md += "### Rollout Validation Stage\n"
    md += "- 5-anchor sweep (2018-2022, same as B-09/B-10)\n"
    md += "- Tower 4 evaluation (same as B-10)\n"
    md += "- Lead-time binned metrics (1-7, 8-30, 31-90, 91-180, 181-270, 271-365 days)\n"
    md += "- Direct comparison: tuned configs vs B-10's baseline\n\n"

    md += "## Critical Finding: CV vs Rollout Divergence\n\n"
    md += "The gap between grid-search validation R2 and recursive-rollout R2 is itself a methodological insight:\n"
    md += "- **One-step CV** (where tuning is optimized) scores locally and may overfit the 2020-2021 validation window\n"
    md += "- **365-day rollout** (where verdict is rendered) compounds prediction errors and reveals which hyperparameters stay robust under recursion\n"
    md += "- The **3-model tuned ensemble** (RF+XGB+LightGBM, no SARIMAX) underperforms B-10's 4-model ensemble baseline -- consistent with CV-picked hyperparameters not reliably transferring to rollout\n\n"

    best_single = agg_by_model[~agg_by_model["model"].str.startswith("Ensemble")].iloc[0]
    if best_single["mean_R2"] > b10_ens_r2:
        md += (f"**However**, at the single-model level, `{best_single['model']}` (R2={best_single['mean_R2']:.4f}) actually "
               f"**beats** B-10's own ensemble (R2={b10_ens_r2:.4f}) -- CV tuning did find a genuinely better individual "
               f"tree model here, it just didn't carry through to B-14's own (3-model, SARIMAX-less) ensemble. This nuance "
               f"was missed in an earlier draft of this document that compared against a stale, hand-typed B-10 baseline "
               f"number (0.012) rather than the precise value recomputed from `b10_ensemble_multi_anchor.csv` (0.0026) --"
               f" corrected here.\n\n")

    md += "## Recommendations\n\n"
    if best_single["mean_R2"] > b10_ens_r2:
        md += (f"1. **For production use:** `{best_single['model']}` (R2={best_single['mean_R2']:.4f}) is the best single "
               f"validated model found across B-09-B-14, ahead of B-10's ensemble (R2={b10_ens_r2:.4f}). B-15 (D-59) "
               f"re-examines whether an ensemble built from this tuned model can beat B-10's outright.\n")
    else:
        md += f"1. **For production use:** B-10's unweighted ensemble (R2={b10_ens_r2:.4f}) remains the best validated configuration\n"
    md += "2. **For future tuning:** Focus on features/architecture rather than hyperparameter tweaking; the rollout task is robust to moderate HPO choices\n"
    md += "3. **For next iterations:** Explore ensemble weighting schemes or architecture changes (not parameter tuning alone)\n\n"

    md += "## Files\n\n"
    md += "- `b14_tree_grid_search.csv` — tree model validation fold results (all 88 combos)\n"
    md += "- `b14_sarimax_grid.csv` — SARIMAX order search by AIC\n"
    md += "- `b14_tuned_rollout_summary.csv` — final 5-anchor rollout results (all models * 5 anchors * 6 bins)\n\n"

    md += "## Cross-Reference\n\n"
    md += "- **D-41:** Original manual HPO for forecasting phase\n"
    md += "- **D-53:** B-09 recursive-rollout baseline\n"
    md += "- **D-54:** B-10 improved configuration (current production recommendation)\n"
    md += "- **D-57:** B-13 TFT/TabPFN results\n\n"

    # Write file (UTF-8 encoding to avoid mojibake on Windows cp1252 default)
    with open(NOTEBOOK_DIR / "b14_results.md", "w", encoding="utf-8") as f:
        f.write(md)

    print("\n[OK] Written b14_results.md")

if __name__ == "__main__":
    result = compile_results()
    if result[0] is not None:
        agg, comp, b10_baseline = result
        write_results_markdown(agg, comp, b10_baseline)
        print("\n[OK] B-14 results compilation complete")
