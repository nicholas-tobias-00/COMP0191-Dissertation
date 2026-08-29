# Chapter 2: Background and Related Work — Finalized Outline

_Working reference, not chapter prose. Captures decisions made through outline discussion._

**PRISMA / search-methodology process is cut entirely** — no query counts, no
768→44→29 funnel, no search-strategy tables. That process lives in the COMP0190 CW
document only. What survives into the dissertation is the *evidence itself*
(the 29 studies and their findings), not the review protocol that found them.

## 2.1 North Wyke Farm Platform

- What and where the experiment is conducted, where the data comes from —
  **headline only**. Full dataset detail is Chapter 3's job, not this section's.
- Site description: three catchments/towers (2, 4, 9), EC instrumentation since
  2017/2018, Rothamsted Research, Devon.
- **Register check vs. Chapter 1:** Ch1's NWFP mention is a *motivating claim*
  ("rare, underexploited resource"). This section is a *factual description*
  (what it physically is). No repeated framing language between the two.
- **New claim to source, not yet cited:** NWFP is representative of UK grassland
  farming and therefore Western Europe more broadly — this is what makes the
  dissertation's findings generalizable rather than single-site trivia. Needs a
  real citation (Defra grassland statistics, or literature characterizing the
  farmlet systems as representative), not an assertion.
- Prior NWFP-grounded publications for context: Cardenas et al. 2022 (CO₂ flux
  EC), Oulaid et al. 2025 (GWQML soil moisture, supervisor co-authored), Wu et al.
  (SPACSYS process model), Partridge et al. 2024 (GreenFeed cattle CH₄).

## 2.2 Eddy Covariance Measurement of FCH₄ and Other Gases

### 2.2.1 Gap-filling methodology and its difficulty
- Irvin et al. 2021 (17-site FLUXNET-CH4 wetland benchmark, RF/XGBoost dominant,
  raw ML uncertainty systematically underestimated), Kim et al. 2020 (RF over
  ANN/SVM/MDS, lagged variables help CH4 more than CO2), Zhu et al. 2023a (RF over
  MDS at UK managed pastures specifically — the direct comparator site type),
  Kheradmand et al. 2023.
- Models used across this literature are almost exclusively simple ANN and
  tree-based ensembles — sets up Section 2.4's contrast.

### 2.2.2 Lack of forecasting/scenario-projection work in this sector
- **This is the chapter's central load-bearing argument** — does the real case-
  building, in full, not a one-liner. Gap-filling (within-distribution
  interpolation) is structurally distinct from forecasting (sequential
  extrapolation); the literature above answers the former, not the latter.
- To the author's knowledge, no prior study has applied ML-based forecasting to
  EC CH₄ flux at a managed temperate grassland.

## 2.3 Towards Digital Twins and the What-If Simulation Gap

**All supporting arguments stay one line each — not thorough treatment.** This
section motivates why the four components matter; it doesn't re-argue the
evidence, which lives elsewhere (2.2.2 for forecasting; interpretability/UQ
literature briefly cited here for the first and only time).

- **Gap citations:** Purcell & Neubauer (2023a) — what-if simulation critically
  absent from agricultural digital twins generally. Fakeye et al. (2024) —
  NWFP-specific three-tier DT framework, names CH₄ forecasting as a missing
  module directly.
- **Four components, not three** (forecasting and scenario projection stay
  separate — they answer different gaps):
  1. **Forecasting** — one-line callback to 2.2.2 only ("as established above,
     this gap exists independent of any digital-twin framing").
  2. **Scenario Projection** — the DT-specific what-if capability
     (Purcell & Neubauer).
  3. **Interpretability** — Buzacott et al. 2024 (SHAP reveals entangled
     multi-driver dependency that raw accuracy masks), Sharma et al. 2026,
     Fuchs et al. 2020 (missed episodic high-emission events without
     site-calibrated, driver-aware models). **CH₄ driver literature lives here,
     not as its own section.**
  4. **Uncertainty Quantification** — Irvin et al.'s raw-ML-uncertainty-
     underestimated finding anchors this specifically (not the DT-gap citation,
     despite being introduced in the same chapter as gap-filling).

## 2.4 Recent Developments in Tabular Modelling Approaches

- Motivated directly by 2.2.1: that literature is predominantly simple ANN and
  tree-based ensembles.
- TabPFN, TabICL, HyperImpute, TFT, DLinear — newer architectures not yet applied
  in this domain.
- "Will be experimented with in Section 3/4" in the original notes refers to
  later **dissertation chapters** (Gap-Filling, Forecasting), not Chapter 2
  sub-sections — flagged so this doesn't cause confusion when drafting.

### 2.4.1 Tabular Foundation Models
### 2.4.2 Sequence Architectures and the Attention-vs-Linear Debate

## 2.5 Synthesis and Gap Statement

- Ties 2.1–2.4 together into the single explicit sentence Chapter 1 already
  promised: no prior study has applied ML-based forecasting to EC CH₄ flux at a
  managed temperate grassland.
- Closes by stating why this project's specific combination — tabular/foundation
  models + forecasting + scenario projection + interpretability + UQ, at a site
  representative of UK/Western European grassland systems — is the necessary and
  appropriate response to the gap just established.

---

## To-do

<!-- - **Build `report/references.bib` from scratch.** Current file is unrelated
  computer-vision/SLAM boilerplate (ORB-SLAM, COLMAP, PointNet, etc.) left over
  from a different dissertation template — none of this chapter's citations
  exist in it. The IEEE-format reference list for the 29 COMP0190 studies
  already exists in that PDF and is the starting point; needs converting to
  BibTeX, not re-researched. Not resolvable from this chat — no write access to
  the actual GitHub repo, only a local read clone. -->
- **Source the NWFP-representativeness claim (2.1)** — currently an assertion,
  needs a real citation (Defra grassland stats, or literature characterizing the
  farmlet systems as representative).
<!-- - **Confirmed: Hoo et al. 2025, Semenov et al. 2025, and Wu et al. (SPACSYS) are
  NOT among the original 29 studies.** Checked directly against Table 6, the
  references list, and the search-strategy tables — none appear. Wu et al. only
  shows up in a separate, informal background document, not the structured
  PRISMA process. All three need adding as a small top-up (not a new search) —
  they became methodologically load-bearing only after the COMP0190 review
  closed (TabPFN-TS extrapolation mechanism; the CMIP6 dataset the Scenario
  chapter runs on; the process-model future-work anchor). -->
