# Chapter 4: Gap-Filling — Finalized Outline

<!-- _Working reference, not chapter prose._ -->

## 4.1 Overview and Necessity of Gap-Filling

## 4.2 Implementation Methodology

Models: MDS, RFm, Bi-LSTM, TabICL, TabPFN, HyperImpute. XGB, ANN, SVM, LightGBM (?)
UQ Implemented for top performing models: 
- TabICL quantile regression
- Conformal calibration

## 4.3 Implementation Metrics and Evaluation Protocol

- Metrics: MAE, R², etc.
- Evaluation protocol: blocked, calendar-gap cross-validation with h-block
  buffering, applied uniformly across all three towers — same paradigm as
  Irvin/Zhu/Kim.
- Caveat: R² is computed differently across papers (sklearn residual-based vs.
  scipy correlation-squared vs. statsmodels OLS) — state this before 4.4.3's
  comparison claims.

## 4.4 Replication of Baseline (MDS, RFm, XGB, ANN)

1. Overview of Models used
2. Overview of Features used
3. Results and comparison with cited numbers in said papers

## 4.5 Improving the Baseline

1. Overview of Models used
2. Overview of Features used
   1. Pooling approach
   2. Additional Features / Columns used
3. Results and comparison with baseline
   - Caveat: Tower 2's higher R² reflects a discriminable livestock-on/off
     signal, not a genuinely easier prediction task.
4. Uncertainty Quantification for the Production Champion and Best Alternative
   <!-- - **RFm:** AOA/dissimilarity-index layer (weak-but-real error correlation,
     Pearson +0.11–0.16) — same technique reused in Ch8 for scenario
     extrapolation-risk flagging. Plus QRF_RFm quantile intervals: -->

     | Model | Tower | Coverage (target 0.90) | Mean width |
     |---|---|---|---|
     | QRF_RFm | T2/T4/T9 | 0.941/0.919/0.898 | 162.7/197.1/213.7 |
     | TabICL_quantile | T2/T4/T9 | 0.918/0.901/0.893 | 141.7/184.2/220.8 |

   <!-- - **Verdict:** TabICL ahead at 2/3 towers (sharper, tighter to nominal
     coverage); QRF conservative/wasteful rather than wrong.
   - Separate, weaker check: width vs. real gap length (not error) — weak,
     inconsistent sign, both models.
   - Gap-length-stratified conformal calibration fixes coverage cleanly (QRF
     0.899, TabICL 0.898, held-out test-half) but only partially fixes the
     width-vs-gap-length relationship — honest partial fix, not a full one.
   - Not yet detailed: §24–26 (structural fix via distance-to-real-observation
     as a feature, consolidated per-hour UQ export, separate daily-resolution -->
     <!-- conformal calibration). -->

---
<!-- 
## Open items

- TabICL UQ result exists locally, not yet pushed to GitHub — recommend pushing
  before drafting so the repo stays source of truth.
- TabICL-solo's stated production-adoption blocker ("no equivalent UQ tooling")
  may no longer hold now that this UQ result exists — 4.5.3's "benchmark-only,
  not production-adopted" framing may need revisiting. Your call, not decided.
- §24–26 not yet walked through. -->
