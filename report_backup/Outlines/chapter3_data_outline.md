# Chapter 3: Data — Finalized Outline

_Working reference, not chapter prose. Captures decisions made through outline discussion._

## 3.1 Dataset

### 3.1.1 Site Description and Overview
- **Third touchpoint on NWFP — deliberately different register from Ch1/Ch2.**
  Ch1 = motivating claim ("rare, underexploited resource"). Ch2.1 = factual/
  narrative description (location, representativeness). Ch3.1.1 = technical/
  quantitative specs needed for methodology: catchment areas (hectares, feeds
  LSU/ha calculations used throughout), sensor specifications, coordinates.
  No repeated framing language across all three.

### 3.1.2 Captured Metrics / Data Sources
- **Scope principle (locked):** present data that ends up used as features in
  gap-filling, not the full raw inventory. Water-quality (17-parameter catchment
  chemistry) and raw condition-score records are excluded on this basis — explored
  during EDA, never adopted as features.
- **Working assumption, not yet explicitly re-confirmed:** literally read,
  "features during gap-filling" is narrower than "features used anywhere in the
  project" — bodyweight, fertiliser-recency, and catchment flow aren't gap-filling
  features, they only enter in Forecasting/Scenario. Default plan is to introduce
  those locally in the chapters that first use them, rather than cataloguing
  everything upfront here. Flag if the intent was actually a full upfront catalog.
- Small detail worth including: the `SWIN_1_1_1` naming surprise — shortwave
  radiation was expected under a `SW_IN_` pattern, found under a different column
  name instead. Good concrete due-diligence detail.

## 3.2 Exploratory Data Analysis

_Should immediately highlight data availability / gap-filling as the first
challenge — sets up Section 3.3 directly._

### 3.2.1 Availability of data per year
- **Confirmed accurate against the repo, safe to state as fact:** Tower 2's
  1,675-day gap (May 2019–Jan 2024) is a genuine, confirmed permanent-pasture→
  arable land-use conversion (Red farmlet, field NW002) — not a sensor fault.
- Include: the empty 2024 held-out window is itself a per-year availability
  fact and belongs here — also worth planting early since it resurfaces in
  Limitations later in the dissertation.

### 3.2.2 Different levels of granularity within the dataset
- 30-min EC flux, 15-min meteorological/soil measurements, daily livestock
  counts — consolidated to a common 1-hour index.

### 3.2.3 Temporal Analysis and Dimensionality Reduction (Correlation, ACF/PACF, UMAP/TICA)
- Something about what ACF / PACF in addition to UMAP / TICA
<!-- - **Weighting:** correlation and ACF/PACF need no explanation (standard enough
  that defining them reads as padding). UMAP gets a sentence or two. **TICA gets
  the real explanatory work** — most readers won't know it. Good hook available:
  TICA comes from molecular dynamics (finding slow collective coordinates in
  trajectory data), and works by maximizing autocorrelation over a time lag
  rather than preserving static neighborhood structure the way UMAP does — this
  is simultaneously the explanation of what TICA is *and* the justification for
  preferring it over UMAP on temporal data, so it's one paragraph, not two. -->
- TICA preferred over UMAP as the dimensionality-reduction lens — due to temporal nature

## 3.3 Data Preprocessing Done

### 3.3.1 Gap-filling
- **Corrected:** not "gap-filling with an R package." Confirmed via direct
  codebase check — no `rpy2`, no `Rscript`, no literal REddyProc R-package
  invocation anywhere. The actual implementation is `src/data/
  reddyproc_pipeline.py` (`mdc_gapfill()`), a **Python reimplementation** of
  REddyProc-style logic (MDS interpolation + mean-diurnal-course fallback).

### 3.3.2 Data Correction, Truncation, and Filtering
- **Expanded beyond "very short" — this is a real correction narrative, not a
  one-line QC filter.** Covers: the USTAR contamination bug (readings up to
  1039.9 m/s, physically impossible, silently corrupting the mean-based
  gap-fill fallback during extended blackouts), a matching VPD contamination
  pattern, and the staged WS/TA truncation fix that followed. Framed as **data
  correction** (a discovery-and-fix story), not just filtering.

---

## Open items (not yet resolved — flagged for traceability)

- **EDA/cleaning chronology (3.2.3):** was correlation/UMAP/TICA analysis run
  before or after the contamination fixes in 3.3.2? Determines whether the
  chapter needs an explicit caveat about early findings predating cleanup.
- **Feature-catalog scope (3.1.2):** confirm whether bodyweight, fertiliser-
  recency, and catchment flow should be introduced locally in their first-use
  chapters (current working assumption) or catalogued upfront in Chapter 3
  regardless of when they're actually used.
