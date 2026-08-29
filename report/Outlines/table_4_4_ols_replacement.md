# Table 4.4: proposed OLS R2 replacement

_Prepared 2026-08-21. Result note only: `report/4 Gap Filling.tex` has not been edited._

## Scope

Table 4.4 is the gap-length robustness table in Chapter 4. Its current entries are residual-based
`sklearn.metrics.r2_score` values. The proposed replacement below uses Zhu-style OLS R2: the
squared Pearson correlation between actual and predicted held-out values, computed within each
fold and then median-aggregated across the two repetitions. Each table cell is ordered
**Tower 2 / Tower 4 / Tower 9**.

## Complete replacement values

| Model | 1 h | 4 h | 32 h | 288 h | Mixed |
|---|---|---|---|---|---|
| LightGBM | .583/.452/.453 | .684/.418/.406 | .702/.443/.428 | .341/.273/.207 | .509/.314/.425 |
| XGBoost | .619/.410/.454 | .675/.385/.375 | .661/.380/.394 | .307/.188/.179 | .427/.283/.359 |
| TabPFN | .569/.465/.465 | .753/.443/.405 | .736/.464/.552 | .342/.345/.253 | .510/.377/.401 |
| TabICLv2, pooled | .587/.463/.462 | .763/.446/.368 | .736/.425/.470 | .384/.313/.246 | .481/.350/.338 |
| SAITS | .562/.457/.371 | .645/.318/.331 | .619/.289/.304 | .287/.217/.169 | .447/.284/.290 |
| Bidirectional LSTM | .459/.181/.205 | .280/.204/.189 | .592/.306/.189 | .206/.214/.101 | .209/.191/.128 |
| TabICLv2, solo | .686/.525/.516 | .756/.468/.430 | .706/.429/.467 | .440/.289/.180 | .473/.333/.327 |

## LaTeX-ready rows

```tex
LightGBM & .583/.452/.453 & .684/.418/.406 & .702/.443/.428 & .341/.273/.207 & .509/.314/.425 \\
XGBoost & .619/.410/.454 & .675/.385/.375 & .661/.380/.394 & .307/.188/.179 & .427/.283/.359 \\
TabPFN & .569/.465/.465 & .753/.443/.405 & .736/.464/.552 & .342/.345/.253 & .510/.377/.401 \\
TabICLv2, pooled & .587/.463/.462 & .763/.446/.368 & .736/.425/.470 & .384/.313/.246 & .481/.350/.338 \\
SAITS & .562/.457/.371 & .645/.318/.331 & .619/.289/.304 & .287/.217/.169 & .447/.284/.290 \\
Bidirectional LSTM & .459/.181/.205 & .280/.204/.189 & .592/.306/.189 & .206/.214/.101 & .209/.191/.128 \\
TabICLv2, solo & .686/.525/.516 & .756/.468/.430 & .706/.429/.467 & .440/.289/.180 & .473/.333/.327 \\
```

## Provenance and verification

- **LightGBM, XGBoost, pooled TabICLv2, SAITS, and Bidirectional LSTM:** recomputed from
  `notebooks/03c_gap_filling_revisited/_data/d100_ols_recalc_raw_predictions.csv`.
- **Solo TabICLv2:** taken from the `baseline` arm of
  `notebooks/03c_gap_filling_revisited/_data/d5_tabicl_solo_results.csv`, which already stores
  scenario-level `R2_OLS`.
- **TabPFN:** its original experiment retained aggregate sklearn metrics but not raw predictions,
  so an exact 60-fold rerun was required. The rerun reproduced all 15 original tower-scenario
  sklearn values within 0.0005 (verification tolerance: 0.002). Results are in
  `notebooks/03c_gap_filling_revisited/_data/table44_tabpfn_ols_summary.csv`; raw predictions are in
  `table44_tabpfn_ols_raw_predictions.csv` in the same directory.
- The consolidated review table is
  `notebooks/03c_gap_filling_revisited/_data/table44_ols_replacement_values.csv`.

## Caveat: SAITS

The original SAITS run also did not retain raw predictions. Its OLS values therefore come from the
existing D-103 stochastic rerun. That rerun did not exactly reproduce the earlier sklearn row
because SAITS validation masking depends partly on global NumPy RNG state. These are nevertheless
the same rerun-derived OLS values already used for Chapter 4's overall OLS comparison; they should
be described as rerun values rather than an algebraic conversion of the original table.

## Accompanying report changes required later

Replacing the numeric rows alone would leave two contradictions:

1. The metric-convention paragraph in Section 4.3 currently says the gap-length robustness
   analysis remains in sklearn R2. That exception must be removed or narrowed.
2. The paragraph immediately below Table 4.4 says several 288-hour results become negative.
   OLS R2 is bounded at zero, so that statement becomes false. The defensible replacement finding
   is that **288 hours remains the weakest scenario for nearly every model, despite all OLS R2
   values being positive**.

The table header/caption should explicitly identify the entries as OLS R2 and retain the note that
each slash-delimited cell is ordered T2/T4/T9.
