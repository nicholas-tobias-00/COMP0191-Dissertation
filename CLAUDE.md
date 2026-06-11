# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

UCL MSc dissertation (COMP0191: AI for Sustainable Development) analysing multi-year agricultural and environmental data from a UK farm research station. The goal is to apply AI/ML to understand relationships between livestock management, land use, and environmental outcomes (water quality, greenhouse gas fluxes, soil health).

## Environment

- **Python 3.13.5** with JupyterLab 4.3.4
- **ML stack**: PyTorch 2.6.0 (CUDA 12.4), TensorFlow 2.20.0, PyTorch Lightning 2.6.0, scikit-learn 1.6.1
- **Data stack**: pandas 2.2.3, numpy 2.1.3, matplotlib 3.10.0, seaborn 0.13.2

## Running Notebooks

```powershell
jupyter lab
```

Run a notebook non-interactively:
```powershell
jupyter nbconvert --to notebook --execute "Dataset EDA.ipynb"
```

## Dataset Structure

All data lives in `Dataset/Consolidated/`. Files follow the naming pattern:

```
<DataType>_<Format>_<YYYY-MM-DD>_<YYYY-MM-DD>.csv
```

Each file covers one calendar year. Data spans **2017–2024** (8 annual slices per type). Loading all years requires concatenating across slices.

### Dataset Types

| File prefix | Description | Frequency |
|---|---|---|
| `measurements_` | Environmental sensor data — water flow (l/s) and soil moisture (%) across up to 15 catchments; ~718 columns including quality flags and last-modified timestamps | 15-min |
| `greenhouse_` | Eddy covariance tower data — CO₂, H₂O, CH₄ fluxes; sensible heat (H), latent energy (LE), turbulent statistics from Tower 2; ~295 columns | 30-min |
| `Cattle Basic Data_` | Individual cattle records (ID, breed, sex, dates) | Event |
| `Cattle Location Data_` | Cattle field/location assignments over time | Event |
| `Cattle Weight Data_` | Cattle weighing records | Event |
| `Cattle Condition Score Data_` | Body condition score assessments | Event |
| `Cattle Sales Data_` | Sales transactions | Event |
| `Breeding Sheep Basic/Location/Weight/Condition Score/Sales Data_` | Same structure for breeding ewes | Event |
| `Lamb Basic/Location/Weight/Sales Data_` | Same structure for lambs | Event |
| `Animal_location_counts_<Type>_` | Summarised head-count per field per day | Daily |
| `Feed Type Data_` | Supplementary feed records | Event |
| `Field Event Data_` | Farm management events (fertiliser, spraying, reseeding, etc.) | Event |
| `Field Survey Data_Format 1_Botanical Survey_` | Plant species composition surveys | Periodic |
| `Field Survey Data_Format 1_Herbage Survey_` | Herbage mass / quality measurements | Periodic |
| `Field Survey Data_Format 1_Silage Cut Survey_` | Silage cut records | Periodic |
| `Field Survey Data_Format 1_Soil Chemistry & Physics_` | Soil nutrient and physical property samples | Periodic |
| `Field Survey Data_Format 1_Grain Survey_` | Grain yield/quality records | Periodic |

### Loading Multi-Year Data

```python
import pandas as pd
import glob

files = sorted(glob.glob("Dataset/Consolidated/measurements_*.csv"))
df = pd.concat([pd.read_csv(f, parse_dates=["Datetime"]) for f in files], ignore_index=True)
```

### Data Quality Notes

- `measurements_` and `greenhouse_` columns come with sibling quality-flag columns (values: `"Acceptable"`, `"Not set"`, etc.) and `"Quality Last Modified"` timestamp columns — filter on these before analysis.
- High sparsity is normal: not all catchments or sensors were active in all years.
- `greenhouse_` data starts from 2018 (no file for 2017).

## Repository Layout

```
Dataset/
  Consolidated/    # All analysis-ready CSVs (annual slices)
  Archive/         # Original ZIP downloads from the data portal
Previous Coursework/  # Literature review and presentation PDFs (COMP0190)
Scripts/           # Intended for reusable processing scripts
Update Deck/       # Supervisor update presentations
Dataset EDA.ipynb  # Initial exploratory analysis notebook
```
