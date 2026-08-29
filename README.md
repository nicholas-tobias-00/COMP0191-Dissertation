# NWFP methane-flux modelling and scenario projection

This repository contains the code and report for a UCL MSc project on reconstructing,
forecasting, interpreting, and projecting ecosystem-scale methane flux
(FCH4) at the North Wyke Farm Platform (NWFP). The analysis covers Towers 2, 4, and 9.

## Start here

- [Canonical latest workflows](workflows/latest/README.md) — the shortest route through the
  currently promoted processes.
- [Current best results](BEST_RESULTS.md) — authoritative champion configurations and headline
  metrics.
- [Project context](CONTEXT.md) — detailed status and experiment history.
- [Methodological decisions](DECISIONS.md) — chronological decision log and caveats.
- [Data dictionary](DATA_DICTIONARY.md) — variables, units, sources, and spatial alignment.
- [Compiled dissertation](report/Report.pdf) — current report output.

Historical notebooks and experiment scripts remain in their numbered stage directories. They are
kept for auditability, but their filename date or experiment number does not by itself make them
the current recommendation. Promotion is determined by `BEST_RESULTS.md` and represented in
`workflows/latest/manifest.json`.

## Current validated endpoints

| Phase | Promoted endpoint | Status |
|---|---|---|
| Gap filling | TabICLv2 solo, per tower, 30 predictors | Benchmark-best; RFm remains the production reference |
| 365-day forecasting | Equal mean of the p95-corrected, 1,095-day, and 1,460-day direct TabPFN forecasts | Lowest exploratory MASE: 0.6908 |
| Interpretability | Permutation sensitivity for the direct TabPFN forecasting architecture | `fx_lsu_dens` remains dominant |
| Forecast UQ | Native quantiles followed by CQR and LSU-stratified CQR | B18-calibrated workflow |
| Scenario projection | Solo per-tower Direct TabICLv2 with `FX_A_SPECIES` and a trend feature | S-06b, bias-corrected climate drivers |

MASE uses the day-of-year climatology denominator; persistence is retained only as a historical
diagnostic. Forecast evaluation is performed against QC-valid observed FCH4 unless explicitly
labelled otherwise.

## Repository layout

```text
data/                 local raw and derived data; intentionally not versioned
notebooks/            historical experiments and current implementation scripts
src/                  reusable data, feature, model, and evaluation code
workflows/latest/     canonical stage order and promoted entry-point manifest
results/              metrics and figures; large/generated artifacts are mostly ignored
report/               dissertation source, appendices, references, and compiled PDF
```

## Reproducing the promoted workflow

Use Python 3.13 from the repository root. The local data directories are required and are not
included in Git.

```powershell
python workflows/latest/validate.py
```

Then follow the stage runbooks in order:

1. [Data preparation](workflows/latest/01_data_preparation.md)
2. [Gap filling](workflows/latest/02_gap_filling.md)
3. [Forecasting](workflows/latest/03_forecasting.md)
4. [Interpretability and uncertainty](workflows/latest/04_interpretability_uq.md)
5. [Scenario projection](workflows/latest/05_scenario_projection.md)
6. [Report build](workflows/latest/06_report.md)

Several historical scripts contain machine-specific Windows paths. The runbooks identify the
promoted entry points and assume execution from this repository root. Data/model runs can require
CUDA and several hours; use the documented smoke tests before full scenario sweeps.

## Provenance rules

- All final evaluations cover Towers 2, 4, and 9 unless a documented data-coverage limitation
  prevents scoring.
- Temporal context is restricted to information available at the forecast origin.
- Raw observations, gap-filled targets, and forecast predictions are kept distinct.
- `BEST_RESULTS.md` is the champion index; `DECISIONS.md` contains the supporting evidence.
- Generated data and large result tables are not assumed to be available from Git alone.

