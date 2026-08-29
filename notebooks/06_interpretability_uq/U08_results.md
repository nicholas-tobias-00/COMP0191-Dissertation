# U-08: UQ recalibrated for the actual B-18 champion

Phase 2 of the additive B18-integration plan (2026-08-20). U-04 (D-88) targeted the prior
"TabPFN+species" TS-wrapper champion. This recalibrates for B-18's actual architecture (direct
regression + p95 spike-gate, `BASE_ALL_52`).

**Method**: base regressor's native quantile spread (`TabPFNRegressor.predict(output_type=
"quantiles")`) shifted by the champion's own deterministic spike-excess correction (uniform shift
across q05/median/q95), then leave-one-anchor-out conformal calibration (`evaluate_stage`, imported
unchanged from `u02_multi_anchor_tower.py`, same as U-04's own precedent). Runtime: 38s total
(5 anchor-level fits, pooled across towers).

## Result: same ~0.89-0.90 PICP convergence, 5th/6th replication

| Tower | Raw PICP | Raw MPIW | Conformal PICP | Conformal MPIW | Conformal pinball |
|---|---:|---:|---:|---:|---:|
| T2 | 0.7255 | 86.04 | NaN | NaN | NaN |
| T4 | 0.9294 | 187.13 | 0.8976 | 144.45 | 10.09 |
| T9 | 0.9277 | 192.84 | 0.8943 | 166.36 | 11.42 |

Confirms U-02/U-04/U-05's central finding (calibration converges to ~0.88-0.90 PICP regardless of
raw coverage, or raw architecture) under a completely different model mechanism this time. Raw PICP
is notably HIGHER here than prior TabPFN passes (0.93 vs. ~0.72-0.92 previously) -- plausibly
because the base regressor's raw quantile spread, built on the richer `BASE_ALL_52` feature set,
starts closer to well-calibrated before any conformal correction. **Tower 2 remains uncalibratable**
(NaN conformal columns, consistent with every prior UQ pass -- U-02/U-04/U-05, now U-08).

## Status

No change to any standing recommendation. Feeds into Phase 3 (S-03b): if the direct-regression
architecture is adopted for scenario work, this confirms its UQ behaviour is not worse than the
prior architecture's, and Phase 4 (U-05b) can reuse this same construction restricted to
`FX_A_SPECIES`.

Full outputs: `results/u08_chains.csv`, `results/u08_summary.csv`.
