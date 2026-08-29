# Chapter 3: Data — Current Outline

_Working reference, not chapter prose. Captures decisions made through outline discussion._

## 3.1 Dataset Overview

- Introduces the five source families used by the project: MET station, soil moisture stations,
  eddy-covariance greenhouse gases, livestock locations and field events.
- Table 3.1 reports their date ranges, temporal resolution and principal contents.
- Records that hourly consolidation produced 70,153 timestamps and 449 variables, while the common
  EC modelling window contains 61,344 hours from 1 January 2017 to 1 January 2024.
- Keeps the comprehensive raw and derived data dictionaries in Appendix A.

## 3.2 Exploratory Data Analysis

### 3.2.1 Availability of Data per Year

- Distinguishes raw FCH4 availability from QC-valid target availability after SSITC and plausibility
  screening.
- Reports post-QC target unavailability of 92.0% at T2, 68.3% at T4 and 81.7% at T9.
- Figure 3.1 compares daily FCH4 behaviour and annual completeness across all three towers.
- Treats T2's terminal absence after analyser relocation as a structural gap rather than ordinary
  intermittent missingness.

## 3.3 Data Preparation

### 3.3.1 Quality Control and Environmental Variable Gap-Filling

- Combines the former quality-control and environmental-gap-filling subsections because screening
  determines which observations can anchor reconstruction.
- Reports SSITC and plausibility screening for FCH4 and FCO2, plus pre-fill bounds for friction
  velocity and VPD. The low-u* result remains diagnostic and is not used to reject methane values.
- Describes the active external SMS/MET sourcing policy used by the gap-filling experiments.
- Describes the Python REddyProc-style hierarchy accurately: linear interpolation for gaps up to two
  hours, same-hour mean diurnal course over expanding 7/14/28/60-day windows, hourly median
  climatology and a global-median last resort. This is not an invocation of the R package or a full
  implementation of REddyProc MDS.
- Table 3.3 reports availability for 11 environmental drivers and confirms that all 33 tower-driver
  series reach 100% numerical coverage after reconstruction.
- Distinguishes completeness from imputation accuracy, which is evaluated in Chapter 4.
- Treats FCO2 separately: accepted observations are retained and missing values are reconstructed by
  a tower-specific random forest, producing complete predictors for the common window.

### 3.3.2 Temporal Analysis and Dimensionality Reduction

- Correlation and ACF/PACF characterise cross-variable association and temporal dependence.
- Figure 3.2 presents the 10-variable union obtained from the five strongest absolute Spearman
  correlations within each tower; the complete 30-variable matrix remains in Appendix A.
- Figure 3.3 embeds the gap-filling pipeline's TICA and UMAP outputs: TICA loadings and tower
  projection, followed by UMAP season and target-availability views.
- TICA uses a 24-hour lag and contiguous tower-specific trajectories; UMAP uses the saved balanced
  3,000-hour D5 sample.
- Local-neighbour agreement shows meaningful tower and seasonal structure but only a small
  above-chance separation between observed and missing FCH4 hours.
- ACF/PACF material remains to be activated and tightened if it is retained in the final chapter.

## Open items

- Verify the provider unit for VPD before final submission: the processing path applies a `[0,15]`
  threshold to the stored values, while repository documentation records an unresolved hPa versus
  kPa inconsistency.
- Confirm whether bodyweight, fertiliser recency and catchment flow should be introduced only in the
  chapters where they first become active, as currently planned.
