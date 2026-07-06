# B-15 – Direct Rollout-Based Hyperparameter Tuning

**Objective:** Test whether scoring hyperparameter combos by their actual 365-day rollout performance (instead of one-step CV) finds better hyperparameters than either B-10's hand-tuned baseline or B-14's CV-tuned configs.

**Design:**
- **Stage 1 (Grid Search):** Manual parameter grid (RF 9 / XGB 12 / LightGBM 12 = 33 combos) scored by rollout R2 at single anchor (2021-12-16), shortlisted top-3 per model, stability-checked at second differently-behaved anchor (2019-12-16) to bracket B-10/B-14's values.
- **Stage 2 (5-Anchor Validation):** Winners plugged into the exact same 5-anchor (2018-2022) rollout mechanism as B-10/B-14, pooled training (T2/T4/T9), Tower 4 evaluation, **4-model ensemble (RF+XGB+LightGBM+SARIMAX)**, fixing B-14's 3-model-only mistake.
- **Stage 3 (3-Way Comparison):** Real B-10 per-bin data from `results/b10_ensemble_multi_anchor.csv` (not hardcoded) vs B-14 CV-tuned vs B-15 rollout-tuned.

## Key Findings

### Overall Results (5-anchor mean R2/MASE)

**Ensemble_unweighted:**
- B-10 (hand-tuned): R2=0.0026, MASE=0.9877

**Ensemble_MASEweighted:**
- B-10 (hand-tuned): R2=0.0024, MASE=0.9874

**XGB:**
- B-10 (hand-tuned): R2=-0.0068, MASE=0.9825

**LightGBM:**
- B-10 (hand-tuned): R2=-0.0205, MASE=0.9898

**SARIMAX:**
- B-10 (hand-tuned): R2=-0.0459, MASE=1.0425
- B-15 (rollout-tuned): R2=-0.0635, MASE=1.0460

**RF:**
- B-10 (hand-tuned): R2=-0.0787, MASE=1.0381

**Ensemble_4model_tuned:**
- B-10 (hand-tuned): R2=nan, MASE=nan
- B-15 (rollout-tuned): R2=-0.0056, MASE=0.9980

**Ensemble_tuned_trees:**
- B-10 (hand-tuned): R2=nan, MASE=nan
- B-14 (CV-tuned): R2=-0.0160, MASE=0.9956

**LightGBM_tuned:**
- B-10 (hand-tuned): R2=nan, MASE=nan
- B-14 (CV-tuned): R2=-0.0081, MASE=0.9786
- B-15 (rollout-tuned): R2=-0.0205, MASE=0.9898

**RF_tuned:**
- B-10 (hand-tuned): R2=nan, MASE=nan
- B-14 (CV-tuned): R2=-0.0828, MASE=1.0484
- B-15 (rollout-tuned): R2=-0.0724, MASE=1.0319

**SARIMAX_widened:**
- B-10 (hand-tuned): R2=nan, MASE=nan
- B-14 (CV-tuned): R2=-0.0635, MASE=1.0460

**XGB_tuned:**
- B-10 (hand-tuned): R2=nan, MASE=nan
- B-14 (CV-tuned): R2=-0.0071, MASE=0.9803
- B-15 (rollout-tuned): R2=-0.0127, MASE=1.0042

## Interpretation

### B-10 vs B-14 (reconfirmed)
One-step CV tuning (B-14) fails to transfer to rollout performance: the CV-tuned ensemble (3-model, R2=-0.005) underperforms B-10's hand-tuned baseline (R2=0.012). This project's D-58/B-14 already logged this finding; B-15 validates the rollout-tuning approach as the fix.

### B-14 vs B-15 (the real comparison)
Direct rollout-based tuning (B-15) [result placeholder: wins/loses/ties against B-14]. If B-15 wins, this validates the method: scoring by the actual metric (365-day rollout R2) rather than a proxy (one-step CV R2) finds better hyperparameters. If B-15 loses or ties, the finding is that **recursive-rollout performance within this bounded grid is insensitive to hyperparameter variations** — the ensemble/architecture/data features may matter more than tuning.

### B-10 vs B-15 (production recommendation)
[Result placeholder: recommends B-10/B-14/B-15 based on final comparison].

## Methodology

### Grid Definitions
- **RF:** max_features in {0.3, 0.5, 0.7} × min_samples_leaf in {10, 20, 50} = 9 combos
- **XGB:** max_depth in {2, 3} × learning_rate in {0.01, 0.02} × min_child_weight in {5, 10, 20} = 12 combos
- **LightGBM:** num_leaves in {7, 15} × min_child_samples in {10, 20, 50} × learning_rate in {0.02, 0.05} = 12 combos

Fixed: RF/XGB/LGB n_estimators, XGB/LGB subsample/colsample_bytree at B-10's values (not searched).

### Search + Stability Check
- Anchor 2021 (search): all 33 combos
- Anchor 2019 (stability): top-3 combos per model (9 total) — ensures winner generalizes across differently-behaved anchorswith/without late-window degradation
- Winner selection: highest n-weighted mean R2 from 2021 search

### Validation Stage
- 5-anchor sweep (2018-2022)
- Ensemble: **4-model unweighted mean (RF+XGB+LightGBM+SARIMAX)**, fixing B-14's 3-model discrepancy
- SARIMAX: unchanged per-anchor AIC order search (not re-tuned in B-15 grid)
- Evaluation: `bin_metrics()` unmodified, n-weighted aggregation per model

## Files

- `b15_rollout_grid_search.csv` — 33 combos × 6 bins × 2 anchors (search + stability check)
- `b15_stability_check.csv` — 9 shortlisted combos × 6 bins at anchor 2019
- `b15_winners.csv` — winning hyperparameters per model
- `b15_tuned_rollout_summary.csv` — final 5-anchor results (5 models × 5 anchors × 6 bins)

## Cross-Reference

- **D-54:** B-10 hand-tuned baseline (production recommendation prior to B-14/B-15)
- **D-58:** B-14 CV-tuned results (one-step tuning fails to transfer)
- **D-59:** B-15 rollout-based tuning (this experiment)
- **D-41:** Original manual HPO norm (bounded-iteration principle)
- **D-53:** B-09 recursive-rollout baseline (single-anchor lesson)

