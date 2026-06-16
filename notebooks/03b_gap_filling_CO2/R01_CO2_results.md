# R-01-CO2 Results: Irvin (2021) with gap-filled FCO₂

**Notebook:** `R01_CO2_Irvin2021_RF_XGBoost.ipynb`
**Base:** R-01 (`../03_gap_filling/R01_results.md`)
**Executed:** 2026-06-16
**Change vs base:** the `FC_1_1_1 [Tower N]` feature is replaced by the **gap-filled FCO₂** series (`src/data/fco2_gapfill.py`, D-26): observed FC after QC (SSITC∈{0,1} + plausibility [−100,100] µmol m⁻² s⁻¹, D-25), reconstructed by RFm where FC is missing. Everything else (model, split, 40% empirical gaps, 5 perms) is identical.

---

## Results — median R² over 5 permutations

| Tower | Model | R-01 | R-01-CO2 | Δ |
|-------|-------|------|----------|---|
| Tower 4 | RF | +0.144 | **+0.049** | −0.095 |
| Tower 4 | XGBoost | +0.086 | +0.020 | −0.066 |
| Tower 9 | RF | −0.027 | −0.026 | +0.001 |
| Tower 9 | XGBoost | −0.089 | −0.095 | −0.006 |
| Tower 2 | RF | −16.9 | **−4.85** | +12.1 |
| Tower 2 | XGBoost | −55.9 | **−8.34** | +47.6 |

---

## Interpretation

- **Tower 4 drops (+0.144 → +0.049).** Base R-01 used *raw* `FC_1_1_1` (no FC-specific QC). Replacing it with **QC'd** FC removes gross CO₂ spikes and SSITC-rejected values that co-occurred with extreme FCH₄ (shared EC-artifact periods). That spurious "extreme-period" signal was inflating base R-01; the cleaner FC is more honest but less predictive of FCH₄ outliers.
- **Tower 9 unchanged.** FC contributes little at Tower 9 in this masking regime.
- **Tower 2 improves dramatically (−16.9 → −4.85).** The catastrophic base value was driven by raw FC outliers compounding the broken seasonal split (D-15); QC + reconstruction tames them. Still strongly negative — the split design remains the real problem (D-19).

**Caveat (D-22):** because only FCH₄ is masked, FC is *observed* at the gap points, so this remains an optimistic upper bound (co-observed flux). See `co2_augmented_summary.md`.

30 rows tagged `R-01-CO2` in `results/benchmarks.csv`.
