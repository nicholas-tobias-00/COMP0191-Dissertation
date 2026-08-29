# 3. Forecasting

## Promoted endpoint

The current long-horizon numerical champion is an equal mean of three direct TabPFN v2 forecasts:

1. p95 event correction: base forecast plus 25% of probability-weighted spike excess;
2. raw target with the most recent 1,095 days of context;
3. tower-robust target scaling with the most recent 1,460 days.

All variants use observed pre-origin labels, known-future `fx_*` covariates, three tower indicators,
calendar year, and days since 2010. The predictive median is used. Headline MASE is 0.6908 against
the day-of-year climatology baseline; the final equal-mean increment remains exploratory.

## Reproduction order

```powershell
python notebooks/05_benchmarking/B17_foundation_screen.py
python notebooks/05_benchmarking/B18_direct_structure.py
python notebooks/05_benchmarking/B18_spike_models.py
python notebooks/05_benchmarking/B18_evaluate_and_plot.py
python notebooks/05_benchmarking/B18_blend_validation.py
python notebooks/05_benchmarking/B18_final_triple_chain.py
python notebooks/05_benchmarking/B18_spike_timing_plots.py
```

The screen supplies the same-protocol B16 controls. The evaluation step must precede blend
validation because it defines and aligns the candidate registry. The final script writes raw daily
chains, metrics, and the 15 tower-anchor figures.

## Evaluation contract

- Towers 2, 4, and 9; anchor years 2018–2022; 365-day horizons.
- Primary target: QC-valid `y_observed`.
- Primary metric: climatology-scaled MASE; report R2 alongside it.
- Preserve all tower-anchor chains, including blocks without observed scoring rows.
- Spike periods use tower-specific pre-anchor percentiles only.

