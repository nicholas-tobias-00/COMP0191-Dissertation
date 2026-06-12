# CONTEXT.md
_Read at the start of every session. Update the "Current status" and "Next task" fields at the end of each session._

---

## Project

**UCL MSc dissertation** — COMP0191: AI for Sustainable Development  
**Student:** Nicholas Tobias (ucabnt1@ucl.ac.uk)  
**Data source:** North Wyke Farm Platform (NWFP), Rothamsted Research, Devon UK  
**Period covered:** 2017–2024 (8 annual slices per dataset type)

## Research question

> _[To be defined. Working direction: apply ML to understand relationships between livestock management practices, land use, and environmental outcomes — specifically GHG fluxes (CH₄, CO₂) and water/soil quality — at the North Wyke Farm Platform.]_

## Data summary

All analysis-ready data lives in `data/`. It is gitignored (too large).

| Location | Contents |
|---|---|
| `data/Consolidated/` | Raw annual CSVs from NWFP portal (original naming, one file per year per type) |
| `data/Compiled/` | Merged multi-year files produced by `notebooks/01_data_compilation/` |

**Key compiled outputs:**

| File | What it is | Frequency |
|---|---|---|
| `measurements.csv` | Water flow + soil moisture across ≤15 catchments | 15-min |
| `greenhouse.csv` | Eddy covariance: CH₄, CO₂, H₂O, H, LE (Tower 2, from 2018) | 30-min |
| `livestock_weight_long.csv` | Cattle + sheep + lamb weighings (long format) | Event |
| `livestock_condition_score_long.csv` | Body condition scores (long format) | Event |
| `Animal_location_counts_*.csv` | Daily head-count per field per species | Daily |
| `Field_Event_Data_Format_1.csv` | Fertiliser, spraying, reseeding events | Event |
| `Field_Survey_Data_Format_1_*.csv` | Botanical, herbage, silage, soil, grain surveys | Periodic |

**Data quality notes:**
- `measurements_` and `greenhouse_` columns carry sibling quality-flag columns (values: `"Acceptable"`, `"Not set"`). Always filter on these before analysis.
- High sparsity is normal — not all catchments/sensors active in all years.
- `greenhouse_` starts from 2018 (no 2017 file).

## Repository layout

```
notebooks/          # numbered analysis notebooks
  01_data_compilation/   COMPLETE — merges Consolidated → Compiled
  02_eda/                IN PROGRESS
src/                # reusable Python modules (data/, features/, models/, evaluation/)
results/            # figures and tables committed here; benchmarks.csv is append-only
prompts/            # LLM session templates
data/               # gitignored
DECISIONS.md        # log of every methodological choice
REPLICATIONS.md     # log of paper replications
```

## Current status

- **Phase:** EDA (exploratory data analysis)
- **Completed:** Data compilation pipeline (`01_data_compilation`) — all 23 compiled files verified
- **In progress:** `02_eda` — only a skeleton exists (two `head()` calls on raw files, not yet on compiled data)

## Next task

> _[One sentence — update this at the end of each session before committing]_

---
_Last updated: 2026-06-12_
