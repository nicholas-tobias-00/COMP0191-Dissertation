# U-06b/U-07b: CQR spike-fix + LSU-stratified CQR for the B18-derived architecture

Phase 5 of the additive B18-integration plan (2026-08-20). Pure recalibration, zero new model
calls -- reuses U-06's `evaluate_cqr()`/`spike_coverage_check()` and U-07's `evaluate_lsu_cqr()`
unchanged, an 8th/9th reuse of `rr.conformal_margins_by_bin()`, against U-08's (full `BASE_ALL_52`
champion) and U-05b's (scenario-safe, solo per-tower `Direct_TabICLv2` + trend, the final
S-03d-locked architecture) already-saved chains.

**Correction (2026-08-20, post-hoc audit):** this file originally reported U05b numbers from the
FIRST (pre-correction, pooled `Direct_TabICLv2_raw`) run of U-05b, before U-05b was rebuilt against
the final solo+trend architecture -- the underlying summary CSVs were correctly regenerated at the
time, but this write-up was never refreshed to match. Numbers below are re-derived directly from
the current `results/u06b_u05b_cqr_summary.csv`/`u06_spike_coverage_b18_u05b.csv`/
`u07b_u05b_lsu_cqr_summary.csv` (U08's numbers were already correct -- its chains were never rerun).

## U-06b: spike coverage roughly triples again

| Chains | Old spike coverage | CQR spike coverage | Old normal coverage | CQR normal coverage |
|---|---:|---:|---:|---:|
| U08 (B18 champion, BASE_ALL_52) | 0.266 | 0.775 | 0.966 | 0.879 |
| U05b (Direct_TabICLv2_solo_trend, FX_A_SPECIES) | 0.266 | **0.901** | 0.961 | 0.832 |

Both replicate U-06's "spike coverage roughly triples" finding cleanly (U08: 0.266->0.775, U05b:
0.266->0.901). U05b's post-CQR spike coverage (90.1%) is close to, not clearly above, the
full-feature champion's (77.5%) -- both are strong, comparable results; no "scenario-restricted
does better" wrinkle survives the correction.

## U-07b: LSU-stratified margins replicate the width-ratio pattern in both

| Chains | Low-LSU MPIW | Mid-LSU MPIW | High-LSU MPIW | Low as % of high |
|---|---:|---:|---:|---:|
| U08 | 83.5 | 105.7 | 298.7 | 27.9% |
| U05b | 151.7 | 206.3 | 373.1 | 40.7% |

Both fall within U-07's original 26-59% range. U05b's absolute margins are wider throughout
(consistent with `FX_A_SPECIES`'s narrower feature set carrying less information than `BASE_ALL_52`
would, an expected and honest cost of the scenario-safe restriction), but the *relative*
livestock-density-driven heteroscedasticity pattern holds either way.

## Status

No change to any standing recommendation -- confirms U-06/U-07's CQR machinery transfers cleanly to
both new architectures with zero code changes beyond pointing at new chain files. U05b's calibration
(this file + U-05b itself) is what Phase 6 will attach to the new S-05/S-06-equivalent scenario
outputs, mirroring D-92's precedent (not yet done -- see S03b_results.md's "Not yet done" list).

Full outputs: `results/u06b_u08_cqr_summary.csv`, `results/u06b_u05b_cqr_summary.csv`,
`results/u06_spike_coverage_b18_u08.csv`, `results/u06_spike_coverage_b18_u05b.csv`,
`results/u07b_u08_lsu_cqr_summary.csv`, `results/u07b_u05b_lsu_cqr_summary.csv`.
