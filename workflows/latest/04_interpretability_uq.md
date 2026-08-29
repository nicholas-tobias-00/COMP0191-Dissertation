# 4. Interpretability and uncertainty quantification

These scripts recalibrate interpretation and UQ for the direct B18 architecture rather than
silently reusing the earlier TabPFN-TS wrapper outputs.

## Forecast interpretation

```powershell
python notebooks/06_interpretability_uq/i03b_champion_interpretability_b18.py
python notebooks/06_interpretability_uq/i03b_plots.py
```

The method is prediction-time permutation sensitivity over five anchors and all three towers. The
model bundle is fitted once per anchor; only `fx_*` query columns are permuted.

## Forecast and scenario UQ

```powershell
python notebooks/06_interpretability_uq/u08_champion_uq_b18.py
python notebooks/06_interpretability_uq/u08_fanchart_plots.py
python notebooks/06_interpretability_uq/u05b_scenario_uq_b18.py
python notebooks/06_interpretability_uq/u06b_u07b_cqr_b18.py
python notebooks/06_interpretability_uq/u06b_u07b_plots.py
python notebooks/06_interpretability_uq/u06c_report_cqr_gallery.py
```

U-08 covers the full-feature direct TabPFN champion. U-05b covers the scenario-safe solo Direct
TabICLv2 plus trend architecture. U-06b/U-07b apply CQR and LSU-stratified CQR to their saved native
quantile chains without new model fitting.

Report raw and calibrated coverage separately. Average 90% coverage does not imply adequate spike
coverage; retain the spike-stratified diagnostics.

