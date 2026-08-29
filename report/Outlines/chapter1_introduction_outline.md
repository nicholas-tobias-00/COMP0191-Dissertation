# Chapter 1: Introduction — Finalized Outline

_Working reference, not chapter prose. Captures decisions made through outline discussion._

## 1.1 Motivation

Beat order (funnel: global → sectoral → hard → resource → payoff → roadmap):

1. Methane as a powerful greenhouse gas — GWP, global scale.
2. Agriculture as a major methane contributor — % figures, UK **and** global.
3. Difficulty of prediction (multi-driver, non-stationary, livestock-dominant in UK
   grassland).
4. NWFP's EC dataset as a rare, underexploited resource — seven years of continuous
   half-hourly flux data at a managed grassland site. Use "underexploited" /
   "underutilised," not "fascinating" — keep it evidence-driven, not subjective.
5. **Explicit gap-statement bridge** (load-bearing sentence, not optional): *"to my
   knowledge, no prior study has applied ML-based forecasting to EC CH₄ flux at a
   managed temperate grassland."* Sits between beats 4 and 6 — without it, the
   pipeline roadmap (beat 6) arrives unearned.
6. Uses of accurate prediction and scenario projection — **named, not generic**.
   Candidates: farm-level management decision support (grazing timing, stocking
   density); scenario/policy planning toward 2050 targets.
7. Pipeline roadmap: gap-filling → forecasting → scenario projection. Bridges
   directly into 1.2.

## 1.2 Research Questions

_(renamed from "Research Objective" — plural content needs a plural header. No
Hypothesis subsection — several reference dissertations reviewed don't carry one,
and RQ1–5 stand on their own without needing a formal hypothesis-testing frame.)_

| ID | Question |
|---|---|
| RQ1 | What ML approaches exist for CH₄ flux prediction in agricultural/grassland contexts; do any address ecosystem-scale EC data; how does gap-filling differ from multi-step forecasting as a modelling constraint? |
| RQ2 | How do statistical baselines, tree-based ensembles, and deep learning compare for half-hourly EC CH₄ forecasting under temporal variability and non-stationarity? |
| RQ3 | How transferable are findings from adjacent domains (wetland gap-filling, animal-scale prediction) to managed grassland EC forecasting? |
| RQ4 | How can XAI (SHAP) and UQ (quantile ML / conformal prediction) decode interactions between continuous environmental variables and discrete management interventions? |
| RQ5 | What structural requirements are needed to integrate forecasting models into a digital shadow with "what-if" scenario analysis at farm scale? |

## 1.3 Aims and Objectives

_(renamed from "Aims and Goals" — matches the term used project-wide: slide deck,
DECISIONS.md, COMP0190 CW)_

Five objectives — **no mention of a sixth anywhere in the dissertation.** The
original Streamlit digital-shadow-interface objective was cut on supervisor advice
early in the project; this is a resolved scoping decision, not a late descope, and
does not need explaining or flagging in Limitations.

1. Systematic literature review — time-series forecasting, quantile ML, EC sensor
   methodology, agricultural digital twin frameworks.
2. Acquire, preprocess, and document NWFP EC CH₄ data (2018–present) — gap
   imputation, QC, feature engineering.
3. Benchmarking pipeline — statistical baselines, tree ensembles, deep learning,
   temporal cross-validation.
4. Interpretability (SHAP) + uncertainty quantification (conformal / quantile ML).
5. Synthetic management/climate scenario generation and evaluation.

## 1.4 Contributions

No IPCC Tier 1 comparison framing. Contribution is stated as: improved methane
prediction accuracy (gap-filling + forecasting) plus construction of models capable
of scenario projection. Include metrics selectively — comparative framing, not bare
numbers:

- **Gap-filling:** "daily-resolution R² of 0.40–0.58 across all three towers,
  against a published ceiling of R²<0.1 at hourly resolution for the same
  ecosystem type (Zhu et al. 2023a)."
- **Forecasting:** beats day-of-year climatology baseline by a wide margin
  (MASE = 0.715).
- **Interpretability:** livestock density as the dominant driver — consistency
  across SHAP, native importance, SARIMAX coefficients, and TabPFN permutation
  importance *is* the finding; no single metric improves on stating that.
- **Scenario projection:** models constructed to produce distinguishable,
  interpretable forecasts under contrasting management/climate scenarios — state
  as a ratio, not the raw per-tower percentage cluster — "roughly double a
  tree-only diagnostic baseline's scenario response," not the full
  "+38.6%/+156.4%/+120.3% vs +20.4%/+76.6%/+62.0%" breakdown (that belongs in the
  Scenario chapter's own table).

**Write this section and the Abstract last, in the same sitting**, off whatever
`BEST_RESULTS.md` says at that point — headline numbers have already moved more
than once this project (D-77 gap-filling revision, D-80 MASE convention change),
and another revision before submission is plausible.

## 1.5 Report Overview

Standard one-paragraph chapter roadmap. **Needs writing once the full dissertation
chapter list is finalized** — do not reuse the old draft's roadmap, which described
a 9-chapter structure that no longer matches current decisions.

---

## Open items (not yet resolved — flagged for traceability)

- **Actual COMP0191 word limit unconfirmed.** Only the COMP0190 prep-module brief is
  in project knowledge; the dissertation module handbook itself isn't. Affects how
  much room Motivation/Contributions can take.
- **Title page still needs the digital-shadow fix** (D-10) — currently reads
  "Towards Digital Twins," should read "Towards a Digital Shadow" — contradicts the
  Background chapter's own argument otherwise. Unrelated to the hypothesis cut.
- **Status of the IPCC Tier 1 comparison itself is unresolved.** Framing language
  is cut from Ch1, but it's not yet confirmed whether the underlying analysis
  (gap-filled annual sums vs. Tier 1 EF estimates) is still being run somewhere in
  the dissertation, reframed as a plain finding rather than a hypothesis test, or
  dropped as an activity entirely. Affects whether a dedicated chapter/section is
  still needed downstream.
