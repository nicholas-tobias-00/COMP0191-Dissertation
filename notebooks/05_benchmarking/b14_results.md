# B-14 — Comprehensive Hyperparameter Tuning for Recursive Rollout

**Objective:** Systematically tune tree models (RF/XGB/LightGBM), SARIMAX, via manual grid search on 2020-2021 validation fold, then validate on the full 5-anchor (2018-2022) recursive rollout.

**Design:**
- **Stage 1 (Grid Search):** Manual parameter grid on 2020-2021 validation fold, selecting by R2
- **Stage 2 (Rollout Validation):** Plug winning hyperparameters into B-10's exact 5-anchor mechanism, compare mean R2/MASE directly against B-10 baseline
- **Scope:** Tree models (RF/XGB/LightGBM) and SARIMAX in the recursive-rollout sequence

## Key Findings

### Overall Results (5-anchor mean R2/MASE)

                     model   mean_R2  mean_MASE
Ensemble_unweighted (B-10)  0.012000   0.975000
            LightGBM_tuned  0.006006   0.964964
                XGB (B-10)  0.003000   0.968000
                 XGB_tuned  0.002387   0.966796
      Ensemble_tuned_trees -0.005114   0.981981
           LightGBM (B-10) -0.014000   0.978000
            SARIMAX (B-10) -0.039000   1.038000
           SARIMAX_widened -0.054409   1.037878
                 RF (B-10) -0.067000   1.024000
                  RF_tuned -0.072079   1.034774

### Interpretation

**Best tuned single model:** LightGBM_tuned (R2=0.0060, MASE=0.9650)

**Tuned ensemble:** Ensemble_tuned_trees (R2=-0.0051, MASE=0.9820)

**vs. B-10 Ensemble baseline (R2=0.012, MASE=0.975):**

[FAIL] Hyperparameter tuning did not improve upon B-10's baseline on the 5-anchor rollout.
   This is a legitimate finding: one-step CV performance (where tuning was optimized) diverges from 365-day rollout performance.

### Per-Model Results

| Model | Mean R2 | Mean MASE | vs B-10 Ensemble |
|---|---|---|---|
| LightGBM_tuned | 0.0060 | 0.9650 | -0.0060 |
| XGB_tuned | 0.0024 | 0.9668 | -0.0096 |
| Ensemble_tuned_trees | -0.0051 | 0.9820 | -0.0171 |
| SARIMAX_widened | -0.0544 | 1.0379 | -0.0664 |
| RF_tuned | -0.0721 | 1.0348 | -0.0841 |

## Methodology

### Grid Search Stage
- **RF:** 16 parameter combos (max_features in {0.3,0.5,0.7,1.0} * min_samples_leaf in {5,10,20,50})
- **XGB:** 36 combos (max_depth in {2,3,4,6} * learning_rate in {0.01,0.02,0.05} * min_child_weight in {1,5,10})
- **LightGBM:** 36 combos (num_leaves in {7,15,31,63} * min_child_samples in {10,20,50} * learning_rate in {0.01,0.02,0.05})
- **SARIMAX:** 9 order combos (p in {1,2,3} * q in {0,1,2}, d=1 fixed)
- **Validation fold:** 2020-2021 (independent of rollout test window)

### Rollout Validation Stage
- 5-anchor sweep (2018-2022, same as B-09/B-10)
- Tower 4 evaluation (same as B-10)
- Lead-time binned metrics (1-7, 8-30, 31-90, 91-180, 181-270, 271-365 days)
- Direct comparison: tuned configs vs B-10's baseline

## Critical Finding: CV vs Rollout Divergence

The gap between grid-search validation R2 and recursive-rollout R2 is itself a methodological insight:
- **One-step CV** (where tuning is optimized) scores locally and may overfit the 2020-2021 validation window
- **365-day rollout** (where verdict is rendered) compounds prediction errors and reveals which hyperparameters stay robust under recursion
- This divergence is why B-10's hand-tuned baseline (D-41) remains competitive: it was tested on the real task (rollout), not a proxy

## Recommendations

1. **For production use:** B-10's unweighted ensemble (R2=0.012) remains the best validated configuration
2. **For future tuning:** Focus on features/architecture rather than hyperparameter tweaking; the rollout task is robust to moderate HPO choices
3. **For next iterations:** Explore ensemble weighting schemes or architecture changes (not parameter tuning alone)

## Files

- `b14_tree_grid_search.csv` — tree model validation fold results (all 88 combos)
- `b14_sarimax_grid.csv` — SARIMAX order search by AIC
- `b14_tuned_rollout_summary.csv` — final 5-anchor rollout results (all models * 5 anchors * 6 bins)

## Cross-Reference

- **D-41:** Original manual HPO for forecasting phase
- **D-53:** B-09 recursive-rollout baseline
- **D-54:** B-10 improved configuration (current production recommendation)
- **D-57:** B-13 TFT/TabPFN results

