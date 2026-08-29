# 1. Data preparation

Run commands from the repository root. Raw NWFP and simulated-climate data are local-only and must
already exist under `data/`.

## Core build order

```powershell
jupyter nbconvert --to notebook --execute --inplace "notebooks/01_data_compilation/Compile Datasets.ipynb"
python src/data/consolidate_hourly.py
python src/data/build_sms_met_dataset.py
python src/data/reddyproc_pipeline.py
python src/features/build_management_features.py
python src/features/build_bodyweight_density.py
```

Build both methane reconstructions because they serve different roles:

```powershell
python src/data/build_fch4_gapfilled.py
python src/data/build_fch4_gapfilled_tabicl.py
```

- `fch4_gapfilled.csv`: production RFm reference and historical forecasting source.
- `fch4_gapfilled_tabicl.csv`: benchmark-best TabICLv2-solo alternative.

Build the report-facing daily forecasting matrix:

```powershell
python src/features/build_forecasting_matrix_v2.py
python src/features/build_forecasting_matrix_v3.py
```

The promoted B18 forecast reads `data/Hourly/forecast_daily_v3.csv`. The TabICL-sourced sibling
matrix is retained for target-source sensitivity checks but is not the primary observed-target B18
input.

## Invariants

- Tower N uses Catchment N data; never average soil variables across catchments.
- Keep `y_observed` and `y_gapfilled` separate.
- Do not use post-origin methane observations or methane-derived AR values in strict forecasts.

