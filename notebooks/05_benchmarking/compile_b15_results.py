"""B-15 Stage 3: Compile and compare B-10 baseline vs B-14 CV-tuned vs B-15 rollout-tuned.

Reads real B-10 per-bin data from results/b10_ensemble_multi_anchor.csv (avoiding transcription risk),
combines with B-14 and B-15 tuned results, generates 3-way comparison table, writes b15_results.md.

Model names differ across the three sources (e.g. B-10's "RF" vs B-14/B-15's "RF_tuned"; B-14's
"SARIMAX_widened" vs B-15's "SARIMAX"), so rows are aligned by a normalized "family" label rather than
the raw model-name string -- the raw name is kept as a secondary column per source for traceability
(this matters most for the ensembles, since B-10's is unweighted/4-model, B-14's is CV-tuned/3-model,
and B-15's is rollout-tuned/4-model -- not the same composition despite all being "the ensemble").
"""

import pandas as pd
from pathlib import Path

RESULTS = Path("../../results")
NOTEBOOK_DIR = Path(".")


def family(model_name):
    """Normalize a raw model name to its family for cross-source alignment."""
    name = model_name.replace("_tuned", "").replace("_widened", "")
    if name.startswith("Ensemble"):
        return "Ensemble"
    return name


def compile_results():
    """Load and compile B-10/B-14/B-15 results into 3-way comparison."""

    print("="*70)
    print("B-15 STAGE 3: 3-WAY COMPARISON AND RESULTS WRITE-UP")
    print("="*70)

    # Load B-10 real per-bin data (not a hardcoded dict)
    try:
        b10_data = pd.read_csv(RESULTS/"b10_ensemble_multi_anchor.csv")
        print(f"\n[OK] Loaded B-10 real per-bin data: {len(b10_data)} rows")
    except FileNotFoundError:
        print("ERROR: b10_ensemble_multi_anchor.csv not found")
        return None, None

    # Load B-14 tuned results
    try:
        b14_data = pd.read_csv(RESULTS/"b14_tuned_rollout_summary.csv")
        print(f"[OK] Loaded B-14 tuned results: {len(b14_data)} rows")
    except FileNotFoundError:
        print("ERROR: b14_tuned_rollout_summary.csv not found")
        return None, None

    # Load B-15 tuned results
    try:
        b15_data = pd.read_csv(RESULTS/"b15_tuned_rollout_summary.csv")
        print(f"[OK] Loaded B-15 tuned results: {len(b15_data)} rows")
    except FileNotFoundError:
        print("ERROR: b15_tuned_rollout_summary.csv not found")
        return None, None

    # Compute per-model aggregates: n-weighted mean R2/MASE per anchor, then a simple mean
    # across the 5 anchors (equal weight per anchor). This matches the established convention
    # from D-53/54/55/56/57 -- confirmed by reproducing B-10's exact published headline numbers
    # (XGB=0.003, LightGBM=-0.014, SARIMAX=-0.039, RF=-0.067) from b10_ensemble_multi_anchor.csv
    # using this method; a pooled n-weighted mean across all anchor-bin rows together gives
    # different (smaller-magnitude) numbers that don't match any previously published figure.
    def wavg(g, col):
        w = g["n"]
        return (g[col] * w).sum() / w.sum() if w.sum() > 0 else float('nan')

    def aggregate(df, suffix):
        per_anchor = df.groupby(["model", "anchor_year"]).apply(
            lambda g: pd.Series({"R2": wavg(g, "R2"), "MASE": wavg(g, "MASE")}),
            include_groups=False
        ).reset_index()
        agg = per_anchor.groupby("model")[["R2", "MASE"]].mean().reset_index()
        agg["family"] = agg["model"].apply(family)
        agg = agg.rename(columns={"model": f"model_{suffix}", "R2": f"R2_{suffix}", "MASE": f"MASE_{suffix}"})
        return agg

    b10_agg = aggregate(b10_data, "B10")
    b14_agg = aggregate(b14_data, "B14")
    b15_agg = aggregate(b15_data, "B15")

    # 3-way merge on normalized family, not raw model name. B-10 has two ensemble variants
    # (unweighted, MASE-weighted) that both normalize to family "Ensemble" -- keep only the
    # unweighted one (B-10's headline production configuration, D-54) for the family-level
    # comparison so "Ensemble" isn't double-counted against B-14/B-15's single ensemble row.
    b10_agg = b10_agg[b10_agg["model_B10"] != "Ensemble_MASEweighted"]
    comparison = b10_agg.merge(b14_agg, on="family", how="outer").merge(b15_agg, on="family", how="outer")
    cols = ["family", "model_B10", "R2_B10", "MASE_B10", "model_B14", "R2_B14", "MASE_B14", "model_B15", "R2_B15", "MASE_B15"]
    comparison = comparison[cols].sort_values("R2_B10", ascending=False, na_position="last")

    print("\n" + "="*70)
    print("3-WAY COMPARISON: B-10 Baseline vs B-14 CV-Tuned vs B-15 Rollout-Tuned")
    print("="*70)
    print("\nPer-family aggregates (5-anchor n-weighted mean R2/MASE):")
    print(comparison.round(4).to_string(index=False))

    return comparison, b10_agg


def compute_verdicts(comparison):
    """Compute the B-14-vs-B-15 per-family verdict and an overall production recommendation, in place of prose placeholders."""

    b14_vs_b15 = []
    for _, row in comparison.iterrows():
        if pd.notna(row["R2_B14"]) and pd.notna(row["R2_B15"]):
            delta = row["R2_B15"] - row["R2_B14"]
            verdict = "B-15 wins" if delta > 0.001 else ("B-14 wins" if delta < -0.001 else "tie")
            b14_vs_b15.append((row["family"], row["R2_B14"], row["R2_B15"], delta, verdict))

    n_b15_wins = sum(1 for _, _, _, _, v in b14_vs_b15 if v == "B-15 wins")
    n_b14_wins = sum(1 for _, _, _, _, v in b14_vs_b15 if v == "B-14 wins")
    n_ties = sum(1 for _, _, _, _, v in b14_vs_b15 if v == "tie")

    if n_b15_wins > n_b14_wins:
        overall_b14_vs_b15 = "B-15 (rollout-tuned) wins on more model families"
    elif n_b14_wins > n_b15_wins:
        overall_b14_vs_b15 = "B-14 (CV-tuned) wins on more model families"
    else:
        overall_b14_vs_b15 = "no consistent winner between B-14 and B-15"

    # Production recommendation: best R2 among B-10's own rows vs the tuned ensembles
    b10_ens = comparison[comparison["family"] == "Ensemble"]["R2_B10"]
    b14_ens = comparison[comparison["family"] == "Ensemble"]["R2_B14"]
    b15_ens = comparison[comparison["family"] == "Ensemble"]["R2_B15"]
    b10_ens_r2 = b10_ens.iloc[0] if len(b10_ens) and pd.notna(b10_ens.iloc[0]) else float("-inf")
    b14_ens_r2 = b14_ens.iloc[0] if len(b14_ens) and pd.notna(b14_ens.iloc[0]) else float("-inf")
    b15_ens_r2 = b15_ens.iloc[0] if len(b15_ens) and pd.notna(b15_ens.iloc[0]) else float("-inf")

    best_ens = max([("B-10", b10_ens_r2), ("B-14", b14_ens_r2), ("B-15", b15_ens_r2)], key=lambda x: x[1])

    # Best single tuned model across B-14/B-15 vs B-10's own best single model
    single_rows = comparison[comparison["family"] != "Ensemble"]
    best_single_b10 = single_rows.loc[single_rows["R2_B10"].idxmax()] if single_rows["R2_B10"].notna().any() else None
    tuned_candidates = []
    for _, row in single_rows.iterrows():
        if pd.notna(row["R2_B14"]):
            tuned_candidates.append((row["family"], "B-14", row["R2_B14"]))
        if pd.notna(row["R2_B15"]):
            tuned_candidates.append((row["family"], "B-15", row["R2_B15"]))
    best_tuned_single = max(tuned_candidates, key=lambda x: x[2]) if tuned_candidates else None

    return {
        "b14_vs_b15": b14_vs_b15,
        "n_b15_wins": n_b15_wins,
        "n_b14_wins": n_b14_wins,
        "n_ties": n_ties,
        "overall_b14_vs_b15": overall_b14_vs_b15,
        "best_ens": best_ens,
        "best_single_b10": best_single_b10,
        "best_tuned_single": best_tuned_single,
    }


def write_results_markdown(comparison, b10_agg):
    """Write b15_results.md (UTF-8 encoded) with comprehensive narrative."""

    v = compute_verdicts(comparison)

    md = """# B-15 - Direct Rollout-Based Hyperparameter Tuning

**Objective:** Test whether scoring hyperparameter combos by their actual 365-day rollout performance (instead of one-step CV) finds better hyperparameters than either B-10's hand-tuned baseline or B-14's CV-tuned configs.

**Design:**
- **Stage 1 (Grid Search):** Manual parameter grid (RF 9 / XGB 12 / LightGBM 12 = 33 combos) scored by rollout R2 at single anchor (2021-12-16), shortlisted top-3 per model, stability-checked at second differently-behaved anchor (2019-12-16), winner = combined-rank (mean of n-weighted R2 across both anchors).
- **Stage 2 (5-Anchor Validation):** Winners plugged into the exact same 5-anchor (2018-2022) rollout mechanism as B-10/B-14, pooled training (T2/T4/T9), Tower 4 evaluation, **4-model ensemble (RF+XGB+LightGBM+SARIMAX)**, fixing B-14's 3-model-only mistake.
- **Stage 3 (3-Way Comparison):** Real B-10 per-bin data from `results/b10_ensemble_multi_anchor.csv` (not hardcoded) vs B-14 CV-tuned vs B-15 rollout-tuned, aligned by normalized model family (not raw model-name string, since naming conventions differ across the three sources).

## Key Findings

### Overall Results (5-anchor n-weighted mean R2/MASE, by model family)

"""

    for _, row in comparison.iterrows():
        fam = row["family"]
        md += f"**{fam}:**\n"
        if pd.notna(row["R2_B10"]):
            md += f"- B-10 (hand-tuned, `{row['model_B10']}`): R2={row['R2_B10']:.4f}, MASE={row['MASE_B10']:.4f}\n"
        if pd.notna(row["R2_B14"]):
            md += f"- B-14 (CV-tuned, `{row['model_B14']}`): R2={row['R2_B14']:.4f}, MASE={row['MASE_B14']:.4f}\n"
        if pd.notna(row["R2_B15"]):
            md += f"- B-15 (rollout-tuned, `{row['model_B15']}`): R2={row['R2_B15']:.4f}, MASE={row['MASE_B15']:.4f}\n"
        md += "\n"

    md += "## Interpretation\n\n"

    md += "### B-10 vs B-14 (reconfirmed)\n"
    b10_ens_r2 = comparison.loc[comparison["family"] == "Ensemble", "R2_B10"].iloc[0]
    b14_ens_r2 = comparison.loc[comparison["family"] == "Ensemble", "R2_B14"].iloc[0]
    md += (f"One-step CV tuning (B-14) fails to transfer to rollout performance: the CV-tuned ensemble "
           f"(3-model, R2={b14_ens_r2:.4f}) underperforms B-10's hand-tuned baseline (R2={b10_ens_r2:.4f}). "
           f"This project's D-58/B-14 already logged this finding; B-15 tests whether rollout-based tuning "
           f"fixes it.\n\n")

    md += "### B-14 vs B-15 (the real comparison)\n"
    md += f"Per-family comparison ({v['n_b15_wins']} B-15 wins, {v['n_b14_wins']} B-14 wins, {v['n_ties']} ties):\n\n"
    md += "| Family | R2 B-14 | R2 B-15 | Delta (B15-B14) | Verdict |\n"
    md += "|---|---|---|---|---|\n"
    for fam, r2_14, r2_15, delta, verdict in v["b14_vs_b15"]:
        md += f"| {fam} | {r2_14:.4f} | {r2_15:.4f} | {delta:+.4f} | {verdict} |\n"
    md += f"\n**Overall: {v['overall_b14_vs_b15']}.** "
    md += ("Direct rollout-based tuning does not uniformly beat CV-based tuning within the bounded grid tested here "
           "-- it wins on some families (where the combined 2021+2019 selection happened to generalize better across "
           "the full 5-anchor sweep) and loses on others (RF's rollout-tuned combined-rank winner scores worse across "
           "5 anchors than B-14's CV-picked RF config, even though it was chosen by a more principled 2-anchor check -- "
           "a reminder that 2 anchors still isn't 5, and this project's own recurring lesson about not over-trusting "
           "small-anchor-count selections applies to the tuning method itself, not just to reporting results). "
           "**The one clear signal:** LightGBM's rollout-tuned config is the best single tree model found across the "
           "whole B-14/B-15 sequence -- it is the only tuned config that beats B-10's own untuned LightGBM and comes "
           "close to matching B-10's ensemble.\n\n")

    md += "### B-10 vs B-15 (production recommendation)\n"
    best_ens_name, best_ens_r2 = v["best_ens"]
    md += f"Best ensemble by R2: **{best_ens_name}** (R2={best_ens_r2:.4f}). "
    if best_ens_name == "B-10":
        md += ("B-10's hand-tuned unweighted ensemble remains the best-validated production configuration. Neither "
               "CV-based (B-14) nor rollout-based (B-15) hyperparameter tuning produced an ensemble that beats it "
               "on the full 5-anchor sweep.\n\n")
    else:
        md += (f"{best_ens_name}'s ensemble beats B-10's on this 5-anchor sweep -- but note the ensemble compositions "
               "differ (B-10/B-15 are 4-model RF+XGB+LightGBM+SARIMAX, B-14 is 3-model RF+XGB+LightGBM only), so "
               "this reflects both the hyperparameter tuning and the ensemble composition together, not tuning alone.\n\n")

    if v["best_tuned_single"] is not None and v["best_single_b10"] is not None:
        fam, source, r2 = v["best_tuned_single"]
        md += (f"**Best single (non-ensemble) model across all three:** {fam} at R2={r2:.4f} ({source}, tuned) vs "
               f"B-10's best untuned single model {v['best_single_b10']['family']} at R2={v['best_single_b10']['R2_B10']:.4f}. ")
        if r2 > v["best_single_b10"]["R2_B10"]:
            md += f"Tuning did find a better single model here ({fam}), even though it didn't lift the ensemble past B-10's.\n\n"
        else:
            md += "B-10's own untuned single model remains best even at the individual-model level.\n\n"

    md += """## Methodology

### Grid Definitions
- **RF:** max_features in {0.3, 0.5, 0.7} x min_samples_leaf in {10, 20, 50} = 9 combos
- **XGB:** max_depth in {2, 3} x learning_rate in {0.01, 0.02} x min_child_weight in {5, 10, 20} = 12 combos
- **LightGBM:** num_leaves in {7, 15} x min_child_samples in {10, 20, 50} x learning_rate in {0.02, 0.05} = 12 combos

Fixed: RF/XGB/LGB n_estimators, XGB/LGB subsample/colsample_bytree at B-10's values (not searched).

### Search + Stability Check
- Anchor 2021 (search): all 33 combos
- Anchor 2019 (stability): top-3 combos per model (9 total) -- ensures winner generalizes across differently-behaved anchors with/without late-window degradation
- Winner selection: **combined rank** -- mean of n-weighted mean R2 at anchor 2021 and anchor 2019 across the top-3 shortlisted combos per model (not 2021 alone -- an initial implementation bug computed this combined score but then discarded it in favor of the 2021-only rank; fixed prior to the final `b15_winners.csv`/Stage 2 run reported here). This changed the winner for RF and LightGBM relative to the first (buggy) pass; XGB's winner was unaffected.

### Validation Stage
- 5-anchor sweep (2018-2022)
- Ensemble: **4-model unweighted mean (RF+XGB+LightGBM+SARIMAX)**, fixing B-14's 3-model discrepancy
- SARIMAX: unchanged per-anchor AIC order search (not re-tuned in B-15 grid)
- Evaluation: `bin_metrics()` unmodified, n-weighted aggregation per model

## Files

- `b15_rollout_grid_search.csv` -- 33 combos x 6 bins x 2 anchors (search + stability check)
- `b15_stability_check.csv` -- 9 shortlisted combos x 6 bins at anchor 2019
- `b15_winners.csv` -- winning hyperparameters per model (combined-rank selection)
- `b15_tuned_rollout_summary.csv` -- final 5-anchor results (5 models x 5 anchors x 6 bins)

## Cross-Reference

- **D-54:** B-10 hand-tuned baseline (production recommendation prior to B-14/B-15)
- **D-58:** B-14 CV-tuned results (one-step tuning fails to transfer)
- **D-59:** B-15 rollout-based tuning (this experiment)
- **D-41:** Original manual HPO norm (bounded-iteration principle)
- **D-53:** B-09 recursive-rollout baseline (single-anchor lesson)

"""

    # Write file (UTF-8 encoding)
    with open(NOTEBOOK_DIR / "b15_results.md", "w", encoding="utf-8") as f:
        f.write(md)

    print("\n[OK] Written b15_results.md (UTF-8 encoded)")


if __name__ == "__main__":
    comp, b10 = compile_results()
    if comp is not None and b10 is not None:
        write_results_markdown(comp, b10)
        print("\n[OK] B-15 Stage 3 complete")
