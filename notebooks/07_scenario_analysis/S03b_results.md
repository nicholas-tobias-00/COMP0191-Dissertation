# S-03b: driver-availability gate for the B-18-derived architecture

Phase 3 (the gate) of the additive B18-integration plan (2026-08-20). Answers: does B-18's
direct-regression architecture still beat the current S-05/S-06 production architecture
(`tabicl_forecast`, TS-wrapper) once restricted to `FX_A_SPECIES` (13 CMIP6/scenario-derivable
columns) -- B-18's own winning feature set (`BASE_ALL_52`) cannot run in scenario mode at all.

**Method**: 5-anchor (2018-2022) x 3-tower real-anchor backtest, climatology-scored MASE (D-80
convention). Five candidates, all restricted to `FX_A_SPECIES` (+ tower dummies/time features for
the pooled direct-regression candidates): (1) control -- current production `tabicl_forecast()`,
per-tower TS-wrapper; (2) `Direct_TabPFN_raw` -- B-18's plain direct-regression mechanism, pooled
across towers; (3) `Direct_TabPFN_tower_robust` -- same + tower-robust (median/IQR) target
normalization; (4) `Direct_TabICLv2_raw` -- architecture-family control (TabICLv2 instead of
TabPFN); (5) `Direct_TabPFN_spikegate` -- B-18's full champion mechanism (p95 classifier gate +
base/normal/spike regressors), restricted to `FX_A_SPECIES`. Runtime: under 2 minutes total for all
5 candidates (model calls are cheap at this restricted feature-set/anchor scale).

**One data-availability finding, symmetric across all candidates (not a bug)**: Tower 9's 2018 and
2019 anchors have insufficient pre-anchor climatology history (`doy_climatology()`'s history window
is empty/all-NaN) and were excluded from scoring for every candidate equally -- confirmed via
`n=2124` identical across all 5 rows in the summary, so the comparison stays fair. Not previously
documented at this exact granularity; flagged for anyone extending T9's anchor coverage further
back.

## Result: every B-18-derived candidate beats the control, by a real margin

| Candidate | n | MASE | MAE | RMSE | R2 |
|---|---:|---:|---:|---:|---:|
| Direct_TabPFN_spikegate | 2124 | 0.7249 | 31.11 | 52.48 | -0.042 |
| Direct_TabPFN_tower_robust | 2124 | 0.7259 | 31.23 | 53.32 | -0.074 |
| **Direct_TabICLv2_raw** | 2124 | 0.7260 | 31.02 | 52.49 | -0.065 |
| Direct_TabPFN_raw | 2124 | 0.7268 | 31.25 | 53.40 | -0.066 |
| control_tabicl_forecaster (current production) | 2124 | 0.7595 | 32.57 | 54.55 | -0.171 |

Every direct-regression variant beats the control by 4.3-4.6% MASE -- a genuinely meaningful margin,
comparable to or larger than B-16-to-B-18's own headline improvement (3.0%) in the full-feature
setting. **The four direct-regression variants are statistically indistinguishable from each other**
(MASE spread of only 0.0019) -- the real win here is the architecture change itself (direct tabular
regression vs. the TS-forecaster wrapper), not any specific normalization or spike-gate refinement.

## Decision at this point: `Direct_TabICLv2_raw` (pooled) provisionally chosen

Ties with `Direct_TabPFN_tower_robust`/`Direct_TabPFN_spikegate` within noise, and beats
`Direct_TabPFN_raw` narrowly. Chosen over the other three on secondary grounds: (a) single model
fit per anchor, no spike-gate's ~3-4x per-call cost; (b) keeps the same TabICLv2 foundation-model
family S-05/S-06 already depend on.

**SUPERSEDED (see S-03c/S-03d, same day) -- this pooled config was NOT what Phases 4-6 actually
used.** Pooling assumes one shared training cutoff across all 3 towers; S-05/S-06's real pipeline
uses per-tower anchors (`tower_anchor()`, each tower's own last-real-data date), so this config
isn't deployable as-is. S-03c tested the faithful solo per-tower adaptation and found it barely
beat control (+0.6%, most of this table's 4.4% margin came from pooling and/or the trend feature,
not the architecture change alone). S-03d isolated the cause and found solo per-tower **+ a trend
feature** recovers most of the margin (+2.79%) with confirmed-safe extrapolation to 2050 --
**`Direct_TabICLv2_solo_trend` is the actual config Phases 4-6 (U-05b, S-06b) use**, not
`Direct_TabICLv2_raw`. See DECISIONS.md D-108 for the full account; this file is kept as an
accurate record of the S-03b experiment itself, not of the final locked-in architecture.

## Status

This table's headline finding stands (every direct-regression variant beats the control) but the
specific "locked in for Phases 4-6" claim above does not -- superseded by S-03c/S-03d same-day
follow-ups. Still the useful evidence that the architecture family (direct regression) genuinely
carries over, which is the opposite of the "B18 checked, doesn't survive the restriction" null
result this project's rigor norm was prepared to accept.

Full outputs: `results/s03b_driver_availability_b18_chains.csv` (27,375 raw prediction rows),
`results/s03b_driver_availability_b18_summary.csv`.
