# B-15 - Direct Rollout-Based Hyperparameter Tuning

**Objective:** Test whether scoring hyperparameter combos by their actual 365-day rollout performance (instead of one-step CV) finds better hyperparameters than either B-10's hand-tuned baseline or B-14's CV-tuned configs.

**Design:**
- **Stage 1 (Grid Search):** Manual parameter grid (RF 9 / XGB 12 / LightGBM 12 = 33 combos) scored by rollout R2 at single anchor (2021-12-16), shortlisted top-3 per model, stability-checked at second differently-behaved anchor (2019-12-16), winner = combined-rank (mean of n-weighted R2 across both anchors).
- **Stage 2 (5-Anchor Validation):** Winners plugged into the exact same 5-anchor (2018-2022) rollout mechanism as B-10/B-14, pooled training (T2/T4/T9), Tower 4 evaluation, **4-model ensemble (RF+XGB+LightGBM+SARIMAX)**, fixing B-14's 3-model-only mistake.
- **Stage 3 (3-Way Comparison):** Real B-10 per-bin data from `results/b10_ensemble_multi_anchor.csv` (not hardcoded) vs B-14 CV-tuned vs B-15 rollout-tuned, aligned by normalized model family (not raw model-name string, since naming conventions differ across the three sources).

## Key Findings

### Overall Results (5-anchor n-weighted mean R2/MASE, by model family)

**Ensemble:**
- B-10 (hand-tuned, `Ensemble_unweighted`): R2=0.0116, MASE=0.9752
- B-14 (CV-tuned, `Ensemble_tuned_trees`): R2=-0.0051, MASE=0.9820
- B-15 (rollout-tuned, `Ensemble_4model_tuned`): R2=0.0072, MASE=0.9828

**XGB:**
- B-10 (hand-tuned, `XGB`): R2=0.0027, MASE=0.9683
- B-14 (CV-tuned, `XGB_tuned`): R2=0.0024, MASE=0.9668
- B-15 (rollout-tuned, `XGB_tuned`): R2=-0.0086, MASE=0.9949

**LightGBM:**
- B-10 (hand-tuned, `LightGBM`): R2=-0.0138, MASE=0.9784
- B-14 (CV-tuned, `LightGBM_tuned`): R2=0.0060, MASE=0.9650
- B-15 (rollout-tuned, `LightGBM_tuned`): R2=0.0166, MASE=0.9497

**SARIMAX:**
- B-10 (hand-tuned, `SARIMAX`): R2=-0.0392, MASE=1.0377
- B-14 (CV-tuned, `SARIMAX_widened`): R2=-0.0544, MASE=1.0379
- B-15 (rollout-tuned, `SARIMAX`): R2=-0.0544, MASE=1.0379

**RF:**
- B-10 (hand-tuned, `RF`): R2=-0.0673, MASE=1.0238
- B-14 (CV-tuned, `RF_tuned`): R2=-0.0721, MASE=1.0348
- B-15 (rollout-tuned, `RF_tuned`): R2=-0.0783, MASE=1.0405

## Interpretation

### B-10 vs B-14 (reconfirmed)
One-step CV tuning (B-14) fails to transfer to rollout performance: the CV-tuned ensemble (3-model, R2=-0.0051) underperforms B-10's hand-tuned baseline (R2=0.0116). This project's D-58/B-14 already logged this finding; B-15 tests whether rollout-based tuning fixes it.

### B-14 vs B-15 (the real comparison)
Per-family comparison (2 B-15 wins, 2 B-14 wins, 1 ties):

| Family | R2 B-14 | R2 B-15 | Delta (B15-B14) | Verdict |
|---|---|---|---|---|
| Ensemble | -0.0051 | 0.0072 | +0.0123 | B-15 wins |
| XGB | 0.0024 | -0.0086 | -0.0110 | B-14 wins |
| LightGBM | 0.0060 | 0.0166 | +0.0106 | B-15 wins |
| SARIMAX | -0.0544 | -0.0544 | +0.0000 | tie |
| RF | -0.0721 | -0.0783 | -0.0062 | B-14 wins |

**Overall: no consistent winner between B-14 and B-15.** Direct rollout-based tuning does not uniformly beat CV-based tuning within the bounded grid tested here -- it wins on some families (where the combined 2021+2019 selection happened to generalize better across the full 5-anchor sweep) and loses on others (RF's rollout-tuned combined-rank winner scores worse across 5 anchors than B-14's CV-picked RF config, even though it was chosen by a more principled 2-anchor check -- a reminder that 2 anchors still isn't 5, and this project's own recurring lesson about not over-trusting small-anchor-count selections applies to the tuning method itself, not just to reporting results). **The one clear signal:** LightGBM's rollout-tuned config is the best single tree model found across the whole B-14/B-15 sequence -- it is the only tuned config that beats B-10's own untuned LightGBM and comes close to matching B-10's ensemble.

### B-10 vs B-15 (production recommendation)
Best ensemble by R2: **B-10** (R2=0.0116). B-10's hand-tuned unweighted ensemble remains the best-validated production configuration. Neither CV-based (B-14) nor rollout-based (B-15) hyperparameter tuning produced an ensemble that beats it on the full 5-anchor sweep.

**Best single (non-ensemble) model across all three:** LightGBM at R2=0.0166 (B-15, tuned) vs B-10's best untuned single model XGB at R2=0.0027. Tuning did find a better single model here (LightGBM), even though it didn't lift the ensemble past B-10's.

## Methodology

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

## Addendum — Tower 2 / Tower 9 cross-tower evaluation and independent Tower-9 tuning

**Motivation:** the tuning and validation above evaluated exclusively against Tower 4 -- the same
convention B-10 itself used throughout the B-09-B15 sequence. Training pools all three towers
(T2+T4+T9, one-hot dummies), but the "winning" hyperparameters and every R2/MASE number above are
Tower-4-scored. This addendum asks the natural follow-up: **do T4-tuned hyperparameters generalize
to T2/T9, and does tuning separately for T9 do any better?**

**Data-coverage check (done before designing this, not assumed)** -- real `y_observed` coverage in
the 365-day target window, per tower per anchor:

| Anchor | Tower 2 | Tower 4 | Tower 9 |
|---|---|---|---|
| 2018 | 27.9% | 72.1% | 0.0% |
| 2019 | 0.0% | 62.7% | 53.4% |
| 2020 | 0.0% | 72.1% | 69.6% |
| 2021 | 0.0% | 93.4% | 49.3% |
| 2022 | 0.0% | 61.4% | 74.2% |

**Tower 2 is usable at only 1/5 anchors (2018)** -- confirms B-13's TabPFN addendum finding,
reconfirmed independently here. **Tower 9 is usable at 4/5 anchors** -- good enough for its own
independent tuning search, unlike Tower 2.

### Part A — cross-tower evaluation of T4-tuned winners (`b15_cross_tower_eval.py`)

Reused the exact same pooled-fit RF/XGB/LightGBM per anchor (no retraining) and rolled out for
T2/T4/T9 targets; SARIMAX refit separately per tower (not pooled).

| Eval tower | Model | R2 (mean) | MASE (mean) | Usable anchors |
|---|---|---|---|---|
| T4 | LightGBM_tuned | **0.017** | 0.950 | 5/5 |
| T4 | Ensemble_4model_tuned | 0.007 | 0.983 | 5/5 |
| T4 | XGB_tuned | -0.009 | 0.995 | 5/5 |
| T4 | SARIMAX | -0.054 | 1.038 | 5/5 |
| T4 | RF_tuned | -0.078 | 1.041 | 5/5 |
| T9 | Ensemble_4model_tuned | -0.226 | 0.892 | 4/5 |
| T9 | RF_tuned | -0.341 | 0.964 | 4/5 |
| T9 | XGB_tuned | -0.359 | 0.959 | 4/5 |
| T9 | SARIMAX | -0.361 | 0.915 | 4/5 |
| T9 | LightGBM_tuned | **-0.388** | 0.957 | 4/5 |
| T2 | LightGBM_tuned | -0.476 | 0.293 | 1/5 |
| T2 | XGB_tuned | -0.704 | 0.324 | 1/5 |
| T2 | RF_tuned | -0.905 | 0.354 | 1/5 |
| T2 | Ensemble_4model_tuned | -1.073 | 0.377 | 1/5 |
| T2 | SARIMAX | -3.396 | 0.585 | 1/5 |

**Headline reversal:** LightGBM_tuned is Tower 4's *best* single model (R2=0.017, beating even
B-10's ensemble) but Tower 9's *worst* (R2=-0.388) -- T4-tuned hyperparameters do not transfer to
T9, and can actively hurt. This directly motivated Part B. **Tower 2** (single usable anchor, 2018)
is weak evidence by this project's own "don't trust one anchor" standard, but directionally poor
across every model, SARIMAX catastrophically so (-3.396) -- not further diagnosed given the sample size.

### Part B — independent Tower-9 tuning (`b15_t9_rollout_grid_search.py` / `b15_t9_multi_anchor.py`)

Re-ran the identical 33-combo grid search + 2-anchor (2021 search, 2019 stability) combined-rank
selection, this time scored against Tower 9 instead of Tower 4.

**Winners differ from Tower 4's:**

| Model | Tower-4 winner | Tower-9 winner |
|---|---|---|
| RF | max_features=0.3, min_samples_leaf=20 | max_features=0.3, min_samples_leaf=**10** |
| XGB | max_depth=**3**, lr=**0.01**, min_child_weight=**20** | max_depth=**2**, lr=**0.02**, min_child_weight=**10** |
| LightGBM | num_leaves=7, min_child_samples=20, lr=**0.05** | num_leaves=7, min_child_samples=20, lr=**0.02** |

**5-anchor validation (Tower-9-tuned) vs Part A's T4-tuned-on-T9:**

| Model | T9-tuned R2 | T4-tuned-on-T9 R2 (Part A) | Delta |
|---|---|---|---|
| Ensemble_4model_tuned | -0.228 | -0.226 | -0.002 |
| RF_tuned | -0.321 | -0.341 | +0.020 |
| XGB_tuned | -0.358 | -0.359 | +0.001 |
| LightGBM_tuned | -0.359 | -0.388 | +0.029 |
| SARIMAX | -0.361 | -0.361 | 0.000 (same model, not re-tuned) |

**Result: independent Tower-9 tuning barely moves the needle.** Despite genuinely different
winning hyperparameters, the 5-anchor outcome is nearly identical to reusing Tower 4's config on
Tower 9 (all deltas within +/-0.03 R2). **Tower 9's poor recursive-rollout performance is not
primarily a hyperparameter-tuning problem** -- it points to something more structural (feature set,
driver availability, or genuinely different flux dynamics) that this bounded grid cannot fix.

**Anchor-level detail flags one likely driver of the poor mean:** anchor 2020 is a catastrophic
outlier for every model on Tower 9 (R2 -0.79 to -1.38), while 2019/2021 are all positive (R2
0.01-0.19) -- consistent with D-53's "anchor-specific, not universal" degradation pattern, showing
up here as a whole-anchor effect rather than a specific lead-time bin. Not further diagnosed.

### Chain plots (addendum)

`results/figures/b15_chains/`: `T2_anchor{2018-2022}_{model}.png` (Tower 2, T4-tuned, only option),
`T9_anchor{2018-2022}_{model}.png` (Tower 9, Tower-9-tuned), `T9_anchor{2018-2022}_{model}_T4tuned.png`
(Tower 9, T4-tuned, for direct visual comparison) -- 75 plots total.

### Addendum recommendation

- **Tower 4:** use B-15's rollout-tuned LightGBM or B-10's 4-model ensemble (main findings above).
- **Tower 9:** tuning does not help -- reuse whichever config is convenient; the 4-model ensemble
  (R2 ~ -0.23) is the least-bad option found, still clearly worse than Tower 4's result. Anchor
  2020's catastrophic outlier is worth a dedicated look if Tower 9 forecasting becomes a priority.
- **Tower 2:** insufficient real evaluation data (1/5 anchors) for any reliable conclusion -- treat
  as open pending more held-out data (matches this project's standing "held-out 2024 empty" caveat).

## Files

- `b15_rollout_grid_search.csv` -- 33 combos x 6 bins x 2 anchors (search + stability check)
- `b15_stability_check.csv` -- 9 shortlisted combos x 6 bins at anchor 2019
- `b15_winners.csv` -- winning hyperparameters per model (combined-rank selection)
- `b15_tuned_rollout_summary.csv` -- final 5-anchor results (5 models x 5 anchors x 6 bins)
- `b15_cross_tower_summary.csv`, `b15_cross_tower_chains.csv` -- addendum Part A
- `b15_t9_rollout_grid_search.csv`, `b15_t9_stability_check.csv`, `b15_t9_winners.csv`,
  `b15_t9_tuned_rollout_summary.csv`, `b15_t9_chains.csv` -- addendum Part B
- `figures/b15_chains/T{2,4,9}_anchor*.png` -- chain plots (all three towers)

## Cross-Reference

- **D-54:** B-10 hand-tuned baseline (production recommendation prior to B-14/B-15)
- **D-58:** B-14 CV-tuned results (one-step tuning fails to transfer)
- **D-59:** B-15 rollout-based tuning (this experiment, including the Tower 2/9 addendum)
- **D-41:** Original manual HPO norm (bounded-iteration principle)
- **D-53:** B-09 recursive-rollout baseline (single-anchor lesson; source of the anchor-specific-degradation pattern)
- **D-57:** B-13's own Tower 2/9 addendum (same Tower-2 data-scarcity finding, reconfirmed here)

