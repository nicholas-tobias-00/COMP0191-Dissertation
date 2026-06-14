# CLAUDE.md

Guidance for Claude Code when working in this repository.
Read CONTEXT.md at the start of every session for current project status and next task.

---

## Project Overview

UCL MSc dissertation (COMP0191: AI for Sustainable Development).  
**Goal:** Develop and evaluate ML approaches for EC CH₄ flux forecasting at the North Wyke Farm Platform (NWFP), Rothamsted Research, and demonstrate integration with a digital shadow architecture for scenario analysis and uncertainty quantification.  
**Supervisor:** Prof. Paul Harris, Rothamsted Research.  
**Student:** Nicholas Tobias (ucabnt1@ucl.ac.uk)

---

## Environment

- **Python 3.13.5** with JupyterLab 4.3.4
- **ML stack**: PyTorch 2.6.0 (CUDA 12.4), TensorFlow 2.20.0, PyTorch Lightning 2.6.0, scikit-learn 1.6.1
- **Data stack**: pandas 2.2.3, numpy 2.1.3, matplotlib 3.10.0, seaborn 0.13.2

Run JupyterLab:
```powershell
jupyter lab
```

Execute a notebook non-interactively:
```powershell
jupyter nbconvert --to notebook --execute --inplace "notebooks/02_eda/Dataset EDA.ipynb"
```

---

## Repository Layout

```
data/                           # gitignored — too large for git
  Consolidated/                 # Raw annual CSVs from NWFP portal
  Compiled/                     # Merged multi-year files (23 files, from 01_data_compilation)
  Hourly/                       # 1-hour resampled outputs (from src/data/consolidate_hourly.py)
    greenhouse_hourly.csv       # EC fluxes, 61,345 rows x 147 cols
    measurements_hourly.csv     # Flow + soil moisture, 70,153 rows x 239 cols
    livestock_hourly.csv        # Head counts per location, 70,129 rows x 63 cols
    consolidated_hourly.csv     # All sources merged, 70,153 rows x 449 cols
documents/                      # gitignored — PDFs (literature, presentations)
notebooks/
  01_data_compilation/          # COMPLETE — compiles Consolidated -> Compiled (23 files)
  02_eda/                       # COMPLETE — full EDA; figures in results/figures/
  03_gap_filling/               # PLANNED — R-01 through R-04 replications
  04_feature_engineering/       # PLANNED
  05_benchmarking/              # PLANNED
  06_interpretability_uq/       # PLANNED
  07_scenario_analysis/         # PLANNED
results/
  benchmarks.csv                # Append-only results ledger
  figures/                      # EDA and model figures
src/
  data/
    consolidate_hourly.py       # Resample all compiled data to 1h resolution
  features/                     # Aggregation, lag construction, quality filtering
  models/                       # Model wrappers
  evaluation/                   # Metrics, plotting
prompts/
  templates.md                  # Session start / end checklists
CONTEXT.md                      # Current project status + replication tracker — read every session
DECISIONS.md                    # Log of all methodological decisions
```

---

## Data

### Paths

| Layer | Path | Notes |
|---|---|---|
| Raw annual slices | `data/Consolidated/` | One CSV per year per type |
| Compiled multi-year | `data/Compiled/` | Run `01_data_compilation` notebook once |
| Hourly consolidated | `data/Hourly/` | Run `src/data/consolidate_hourly.py` once |

### Key compiled files

| File | Frequency | Notes |
|---|---|---|
| `greenhouse.csv` | 30-min | EC fluxes 2017–2024; Towers 2, 4, 9 |
| `measurements.csv` | 15-min | Flow + soil moisture, 2017–2025, 718 cols |
| `Animal_location_counts_*.csv` | Daily | Head counts per catchment per species |
| `livestock_weight_long.csv` | Event | Long-format weights |
| `livestock_condition_score_long.csv` | Event | Long-format condition scores |
| `Field_Event_Data_Format_1.csv` | Event | Fertiliser, spraying, reseeding |

### Confirmed target columns (from EDA)

- **Primary target:** `FCH4_1_1_1 [Tower 4]` — EC CH₄ flux (nmol m⁻² s⁻¹), 44.6% valid
- **Secondary targets:** `FCH4_1_1_1 [Tower 2]` (12.1%), `FCH4_1_1_1 [Tower 9]` (25.6%)
- **Quality flag:** `FCH4_SSITC_TEST_1_1_1 [Tower N]` — 0=best, 1=acceptable, 2=reject
- **Note:** `CH4_1_1_1 [Tower N]` is mole fraction (nmol/mol), NOT flux — do not confuse

### Spatial alignment rule (CRITICAL — D-18)

**Tower N = Catchment N.** Each tower's model must use only its own catchment's data:

| Tower | Soil moisture column | Met/flux columns |
|---|---|---|
| Tower 2 | `Soil Moisture @ 10cm Depth (%) [Catchment 2]` | `* [Tower 2]` |
| Tower 4 | `Soil Moisture @ 10cm Depth (%) [Catchment 4 After  2013/08/13]` | `* [Tower 4]` |
| Tower 9 | `Soil Moisture @ 10cm Depth (%) [Catchment 9]` | `* [Tower 9]` |

**Never average soil moisture across catchments from different towers.** Shortwave radiation column is `SWIN_1_1_1 [Tower N]` (not `SW_IN_`). Soil temperature exception: if `TS_1_1_1 [Tower N]` < 20% available, cross-tower use is permitted (document in DECISIONS.md).

### Quality filtering

`measurements_` and `greenhouse_` columns carry sibling quality-flag columns:
- String flags: `"Acceptable"` / `"Not set"` (column name = data col + `" Quality"`)
- Numeric SSITC flags: 0/1/2 for EC flux columns
- Timestamp cols: column name ends in `" Quality Last Modified"` — metadata only

`src/data/consolidate_hourly.py` drops non-numeric columns automatically. When loading compiled CSVs directly, filter before analysis.

### Loading compiled data

```python
import pandas as pd

# Sub-hourly — load compiled file directly
gh = pd.read_csv("data/Compiled/greenhouse.csv", parse_dates=["Datetime"], low_memory=False)

# Hourly consolidated — best starting point for modelling
df = pd.read_csv("data/Hourly/consolidated_hourly.csv", parse_dates=["Datetime"], index_col="Datetime")
```

---

## Temporal split

| Split | Years | Purpose |
|---|---|---|
| Train | 2018–2021 | Model fitting |
| Test | 2022–2023 | Held-out evaluation |
| Held-out | 2024 | Final benchmark only |

**No random splits anywhere** — temporal ordering must be respected (D-04).

---

## Session workflow

1. Read `CONTEXT.md` — confirms current phase and next task.
2. Pick one task from `CONTEXT.md` "Next task".
3. Work. Use `prompts/templates.md` as session scaffolding.
4. End of session: update `CONTEXT.md` (status + replications table) + `DECISIONS.md`, append to `results/benchmarks.csv`, commit.

---

## Coding guidelines

### Think before coding

- State assumptions explicitly before implementing. If uncertain, ask.
- If multiple interpretations exist, surface them — don't pick silently.
- If something is unclear, stop and name what's confusing.

### Simplicity first

- Minimum code that solves the problem. Nothing speculative.
- No abstractions for single-use code. No configurability that wasn't requested.
- If you write 200 lines and it could be 50, rewrite it.

### Surgical changes

- Touch only what the task requires. Don't improve adjacent code.
- Match existing style even if you'd do it differently.
- Every changed line should trace directly to the user's request.

### Goal-driven execution

- For multi-step tasks, state a brief plan with verifiable checkpoints before coding.
- Run notebooks with `nbconvert --execute` to verify output, not just syntax.
