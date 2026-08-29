# Canonical latest workflows

This directory is the stable navigation layer for the current promoted analysis. It does not copy
the large notebooks or model implementations: each runbook points to the authoritative source
file and `manifest.json` records those paths in a machine-checkable form.

This avoids two common failure modes:

1. a `latest_*` copy silently diverging from the implementation that produced the reported result;
2. a high experiment number being mistaken for the adopted method despite a negative verdict.

## Stage order

| Order | Stage | Runbook | Promoted endpoint |
|---:|---|---|---|
| 1 | Data preparation | [01](01_data_preparation.md) | Hourly data, RF/TabICL fills, daily v3 forecasting matrix |
| 2 | Gap filling | [02](02_gap_filling.md) | TabICLv2-solo benchmark and calibrated UQ products |
| 3 | Forecasting | [03](03_forecasting.md) | Three-component direct TabPFN forecast |
| 4 | Interpretability and UQ | [04](04_interpretability_uq.md) | B18 permutation sensitivity and CQR |
| 5 | Scenario projection | [05](05_scenario_projection.md) | S-06b Direct TabICLv2 scenario engine |
| 6 | Dissertation | [06](06_report.md) | `report/Report.pdf` |

## Status vocabulary

- **Canonical:** the promoted process behind a current report-facing result.
- **Supporting:** required input preparation, evaluation, calibration, or plotting.
- **Validation:** confirms why a configuration was promoted; not normally rerun for production.
- **Historical:** retained elsewhere for provenance but deliberately absent from this folder.

## Validation

Run from the repository root:

```powershell
python workflows/latest/validate.py
```

The validator checks that every runbook and source entry in `manifest.json` exists. It does not
require ignored data/results to be present and does not execute expensive models.

## Updating this layer

When a result is genuinely promoted:

1. update `BEST_RESULTS.md` and `DECISIONS.md` first;
2. update the relevant runbook and `manifest.json` in the same change;
3. run `validate.py`;
4. keep the superseded implementation in its historical stage directory.

