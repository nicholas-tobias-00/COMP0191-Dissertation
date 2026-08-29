# 5. Scenario projection

## Promoted architecture

S-06b fits `TabICLRegressor(n_estimators=8, random_state=42)` separately for each tower using the 13
scenario-constructible `FX_A_SPECIES` predictors plus days since 2010. Each tower model is fitted
once and reused across all SSP, GCM, realization, and intervention combinations.

Climate inputs are bias-corrected against NWFP observations before scenario construction. The
regulation-ceiling livestock level is 2.5 LSU/ha.

## Run order

Create the bias-corrected climate-driver files:

```powershell
python src/features/build_bias_corrected_cmip6.py
```

Optional validation of the selected solo-plus-trend model:

```powershell
python notebooks/07_scenario_analysis/s03d_solo_trend_check.py
```

Smoke-test before the full sweep:

```powershell
python notebooks/07_scenario_analysis/s06b_master_runner.py smoke
```

Run the full livestock, grazing, and fertiliser grid, then replace only the regulation-ceiling
rows with the corrected 2.5 LSU/ha implementation:

```powershell
python notebooks/07_scenario_analysis/s06b_master_runner.py
python notebooks/07_scenario_analysis/s06b_lit_ceil_fix.py
```

Generate representative daily chains, plots, and annual methane summaries:

```powershell
python notebooks/07_scenario_analysis/s06b_livestock_v2_daily_chains_subset.py
python notebooks/07_scenario_analysis/s06b_practices_daily_chains_subset.py
python notebooks/07_scenario_analysis/s06b_livestock_v2_daily_chains_plots.py
python notebooks/07_scenario_analysis/s06b_practices_daily_chains_plots.py
python notebooks/07_scenario_analysis/s06b_annual_ch4_generation.py
```

The full sweep is multi-hour and writes large ignored CSVs. Preserve raw realization-level outputs
locally; Git should contain the code, manifest, concise summaries, and report-facing products.

