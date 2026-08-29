# Chapter 5: Forecasting — Finalized Outline

<!-- _Working reference, not chapter prose._ -->

<!-- Point forecasting: **cut entirely**, no mention, no contrast paragraph. The
"architecture wins differently depending on the task" theme survives regardless
— it's already carried by 5.2's feature-enrichment differential (tree/SARIMAX/
ensemble vs. foundation models), which is within-rollout, not point-vs-rollout. -->

## 5.1 Forecasting Methodology (Recursive Nature) and Evaluation Metrics

### 5.1.1 Baseline
- Seasonal mean, SARIMAX.

### 5.1.2 Model roster
- **RF / XGB / LightGBM** — already explained in Ch4 §4.2, reference back only
- **TabPFN & TabICLv2** — already explained in Ch4, 
- **TFT, DLinear** — new, introduced here.

### 5.1.3 Metrics
- Scaled metrics based on seasonal mean, establishing degree of improvement.
- **Observed-vs-gap-filled-target scoring convention, stated here as a
  principle before Section 3's results:** scoring against the gap-filled
  target instead of real observations reverses the ranking (favours tree/
  SARIMAX/ensemble models via circularity — they're trained by directly
  regressing onto that same series). Same category of caveat as Ch4's
  R²-comparability note — state the rule before the table, not after someone
  asks why the gap-filled target wasn't just used for more data points.

## 5.2 Training and Cross-Validation Methodology

- Feature set and CV split methodology.
- **Single-anchor → multi-anchor, one clause, no discovery narrative:** "all
  rollout results use a 5-anchor sweep rather than a single anchor, since
  single-anchor rankings proved unreliable in preliminary checks." No naming
  which model flipped, no methodology-journey narration — single-anchor was
  the very first exploratory run, before the multi-anchor protocol existed,
  not a competing methodology that persisted into any cited result.
- **Feature-enrichment scope, stated precisely:** "The enriched feature set
  (species-disaggregated livestock, Tower-2 land-use flag, catchment flow,
  fertiliser-recency, liveweight density) was constructed uniformly across the
  model roster. Tree/SARIMAX/ensemble models showed no measurable gain in
  preliminary evaluation and were not pursued further given compute cost;
  foundation/attention models were evaluated in full." Precise about
  availability vs. evaluation-depth — not "the feature set is the same for all
  models" (true for construction, not for how thoroughly each model was
  checked against it).

## 5.3 Model Results

- Headline result: TabPFN+species, MASE = 0.715 — this is the origin point for
  *why* species-disaggregation specifically produced the headline number
  rather than being a uniform improvement (per 5.2's differential).
- Production ensemble (unweighted RF/XGB/LightGBM/SARIMAX mean).
- Hyperparameter-tuning negative result — didn't generalize across towers.
- TabICLv2's corrected point-estimate — stated as final numbers only, no
  debugging narrative (same discipline as Ch4's QRF mean→median fix).

### Uncertainty Quantification

- **Scope:** built only for
  top-performing models (TabPFN, TabICLv2) 
<!-- - **No AOA anywhere in this dissertation** (decided and applied to Ch4 as -->
  <!-- well) — PICP/MPIW is the sole UQ vocabulary throughout. -->
<!-- - **Conformal calibration primer, placed before the results table:**
  a distribution-free wrapper around a model's own raw uncertainty estimates.
  Rather than trusting the raw quantiles directly — which, as the raw-PICP
  column shows, can be badly miscalibrated — it uses a held-out calibration
  set to measure how far actual observations fell from the raw predictions,
  then adjusts interval width using the empirical distribution of those errors
  so the result hits the target coverage (0.90) by construction. This project
  uses **leave-one-anchor-out** calibration specifically — each of the five
  rollout anchors takes a turn as the calibration set for the other four —
  not a single fixed split. The guarantee is *marginal* coverage under
  exchangeability: correct on average across the test set, not for any
  individual point or subgroup, and known to break down for genuinely
  out-of-distribution inputs — directly relevant once Chapter 6 asks these
  same models to extrapolate into scenario conditions they've never seen. -->
- **Results:**

  | Model | Tower | Raw PICP | Raw MPIW | Conformal PICP | Conformal MPIW | Conformal pinball |
  |---|---|---|---|---|---|---|
  | TabPFN | T2 | 0.804 | 65.5 | NaN | NaN | NaN |
  | TabPFN | T4 | 0.867 | 111.6 | 0.898 | 149.5 | 10.56 |
  | TabPFN | T9 | 0.724 | 125.3 | 0.889 | 188.9 | 12.86 |
  | TabICLv2 | T2 | 0.804 | 192.4 | NaN | NaN | NaN |
  | TabICLv2 | T4 | 0.964 | 301.7 | 0.895 | 154.7 | 10.63 |
  | TabICLv2 | T9 | 0.771 | 290.4 | 0.894 | 195.3 | 13.04 |

- **T2 uncalibratable (NaN throughout)** — same structural issue established
  elsewhere in this project: too few real observed values in T2's anchor
  windows to build a usable calibration set. Not a new finding here, just
  resurfacing.
- **Raw miscalibration direction isn't a clean per-model story, worth stating
  as such:** TabPFN under-covers at both T4/T9 (needs widening). TabICLv2
  over-covers at T4 (needs sharpening, 301.7→154.7, more than halved) but
  under-covers at T9 like TabPFN does — the direction flips by tower even
  within TabICLv2 alone.
- **After calibration**, both land close to 0.90 at T4/T9 (0.895–0.898).
  TabPFN comes out marginally sharper and better-scored than TabICLv2 (MPIW
  149.5/188.9 vs 154.7/195.3; pinball 10.56/12.86 vs 10.63/13.04) — a small
  complementary result given TabPFN is already the MASE champion: not just
  more accurate, its calibrated intervals are marginally tighter too.