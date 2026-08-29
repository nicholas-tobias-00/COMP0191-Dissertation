# results/latest — snapshot of current best-validated results

Curated backup, built 2026-08-19. A representative selection of the latest/best metrics, tables,
and figures for the four project phases, pulled from `BEST_RESULTS.md` (the authoritative source —
consult it and `DECISIONS.md` for full detail/caveats; this is a snapshot, not a replacement).

---

## 1. Gap-filling + its UQ

**Best model: TabICL-solo (D-79)** — per-tower solo training (not pooled — TabICL's fixed
10,000-row context cap makes pooling actively dilute it, unlike RF; D5.5's finding), on the exact
same champion feature set as RFm (30 cols: `lsu_dens` + `FEATURES`). Beats RFm at 2 of 3 towers,
essentially ties at the third. Full metrics (median across 5 gap-length scenarios × 2 reps):

| Tower | R² | RMSE | MAE | nMAE | R²_OLS (scipy, Zhu et al. 2023a convention) |
|---|---|---|---|---|---|
| T2 | **0.676** | 77.60 | 30.03 | 0.213 | 0.686 |
| T4 | **0.428** | 99.18 | 41.57 | 0.324 | 0.429 |
| T9 | 0.423 | 110.01 | 50.48 | 0.346 | 0.430 |

nMAE = MAE / std(all real, QC'd observed FCH4) — a fixed per-tower constant (T2=140.9, T4=128.5,
T9=146.0 nmol m⁻² s⁻¹), not test-set std, so it's comparable across every experiment in this
project's gap-filling phase (D-79 convention, chosen over range-normalization since FCH4's range is
set by 1-2 extreme spikes per tower). R²_OLS is scipy `linregress`'s squared Pearson r, bounded
[0,1] — the metric Zhu et al. (2023a) actually report, as distinct from this project's standing
(unbounded-below) sklearn R² above.

**Standing production config (RFm pooled, D-77) — kept for reference, not the best result:**

| Tower | R² | RMSE | MAE | nMAE |
|---|---|---|---|---|
| T2 | 0.576 | 75.0 | 31.2 | 0.221 |
| T4 | 0.404 | 100.3 | 42.8 | 0.333 |
| T9 | **0.426** | 107.1 | 49.3 | 0.338 |

RFm remains the only config with full UQ/production-fill tooling built around it (that's a tooling
gap for TabICL, not evidence RFm is more accurate) and edges TabICL back out at T9 specifically.

**UQ — hourly (QRF + TabICL native quantiles, 178,560 held-out points, D-78):**

| Model | Tower | Raw coverage (target 0.90) | Mean width |
|---|---|---|---|
| QRF | T2/T4/T9 | 0.941/0.919/0.898 | 162.7/197.2/213.7 |
| TabICL | T2/T4/T9 | 0.918/0.901/0.893 | 141.7/184.3/220.8 |

**UQ — gap-length-stratified conformal calibration (raw → calibrated, pooled):** QRF 0.919→0.899,
TabICL 0.902→0.898. Fixes aggregate coverage but not the width-vs-gap-length sharpness relationship
(honest partial fix — see `gap_filling/data/conformal_calibration_summary.csv` for the full
per-tower/per-scenario breakdown).

**UQ — daily resolution (§10):** naive normal-approximation band does as well as or better than
conformal at this resolution (daily averaging smooths the hourly spike-skew) — margins ±52.4/±80.8/
±62.3 nmol at T2/T4/T9.

**Figures:** `gap_filling/figures/` — `fanchart_T{2,4,9}.png` (actual/predicted + interval),
`calibration_reliability.png`, `production_interval_T{2,4,9}.png` (raw+calibrated bands on the real
series), `daily_calibration_reliability.png`, `daily_production_intervals.png`.

**Data:** `gap_filling/data/` — `tabicl_solo_champion_full_metrics.csv` (the table above),
`d5_tabicl_solo_results.csv` (per-scenario detail behind the median), `model_comparison.csv` (full
model roster R², pooled configs), calibration summary CSVs, `d100_ols_recalc_summary.csv` (R²_OLS
for the other D-78 challenger models).

---

## 2. Forecasting + its UQ

**Champion (recursive rollout, the project's headline forecasting result):** `TabPFN+species`
(F-10/D-67, rescored D-80) — **MASE = 0.715** (climatology-scored, all-tower pooled — this
project's primary metric, CLAUDE.md), R² = −0.084 (all-tower pooled). Zero-shot, zero training,
beats every other model/config tested in this project on MASE.

**Full per-tower metrics (`BASE+species` config, observed target):**

| Tower | R² | RMSE | MAE | MASE (persistence-scored) | WAPE | R²_OLS | nMAE (test-set std) |
|---|---|---|---|---|---|---|---|
| T2 | −0.068 | 17.47 | 12.44 | 0.229 | 0.910 | 0.003 | 0.686 |
| T4 | −0.080 | 54.90 | 31.11 | 0.883 | 0.869 | 0.218 | 0.460 |
| T9 | −0.171 | 61.30 | 37.95 | 0.897 | 0.909 | 0.082 | 0.528 |

R²/RMSE/MAE/MASE/WAPE are the project's standard n-weighted mean across lead-time bins and anchors
(`bin_metrics()` convention). **MASE column here uses the original chain-persistence baseline
(D-37)**, not the climatology baseline (D-80) — the two aren't directly comparable; the headline
0.715 above is the climatology-scored all-tower pooled number specifically, computed separately.
R²_OLS/nMAE were computed directly from raw (median, y_true) prediction pairs
(`u04_chains.csv`) using this project's shared `src/evaluation/metrics.py` — note the resulting raw
R² there (T4=0.103, T9=0.015, T2=−0.049) differs from the bin-weighted R² column, since it's an
unweighted flat pool across all days/anchors rather than the bin-then-anchor-weighted convention;
both are legitimate, differently-computed views, not a contradiction. T2's small sample (n=102, real
coverage ends May 2019) makes its numbers the least stable of the three.

**Point/direct alternative (different task, not directly comparable):** B-03 enriched RF/XGB,
daily track — R² T4 h1=0.365/h14=0.280, T9 h14=0.359.

**UQ — champion recalibration (U-04, leave-one-anchor-out conformal, `BASE+species` config):**

| Model | Tower | Raw PICP | Conformal PICP (target 0.90) | Conformal MPIW | Conformal pinball |
|---|---|---|---|---|---|
| TabPFN | T4 | 0.867 | 0.898 | 149.5 | 10.56 |
| TabPFN | T9 | 0.724 | 0.889 | 188.9 | 12.86 |
| TabICLv2 | T4 | 0.964 | 0.895 | 154.7 | 10.63 |
| TabICLv2 | T9 | 0.771 | 0.894 | 195.3 | 13.04 |

T2 cannot support calibration (real coverage ends May 2019).

**UQ — CQR spike fix (U-06/U-07, standing method):** flat symmetric margins missed 75% of top-10%-
magnitude days despite good average PICP; CQR (quantile-based nonconformity) roughly triples spike
coverage (TabICLv2 22–24%→~79%, TabPFN 24%→57%). U-07 layers livestock-density-stratified bins on
top — low-LSU intervals are 26–59% the width of high-LSU intervals (genuine win-win, not a
trade-off).

**Figures:** `forecasting/figures/champion_fancharts/` — TabPFN actual/predicted + calibrated 90%
band at representative anchors/towers. `forecasting/figures/cqr_examples/` — U-04's symmetric
margin vs. U-07's LSU-stratified CQR side by side (same anchor/tower), showing the visual effect of
the tighter, spike-aware interval.

**Data:** `forecasting/data/` — `tabpfn_species_champion_full_metrics.csv` (the table above),
`u04_summary.csv` (full PICP/MPIW/pinball table), `u06_u04_cqr_summary.csv`,
`u07_u04_lsu_cqr_summary.csv`, `b10_b13_rerun_table_all_towers.csv` (full 11-model roster
comparison).

---

## 3. Interpretability

**Current (I-03, D-102, 2026-08-18 — recalibrated for the actual champion, TabPFN+species):**

| Rank | Feature | Mean importance |
|---|---|---|
| 1 | `fx_lsu_dens` | 1.1456 |
| 2 | `fx_cattle_dens` | 0.8043 |
| 3 | `fx_grazing_active` | 0.3750 |
| 4 | `fx_total_liveweight_dens` | 0.2807 |

`fx_sheep_dens`/`fx_lamb_dens` rank near the bottom (0.0143/0.0320) — the species-split gain is a
cattle effect specifically, independently corroborating S-05's scenario cattle-dominance finding.
**Tower 2 has zero livestock features in its top 10** (all TA/TS/SWIN) — a fourth independent
confirmation of T2's livestock-blindness.

**Figures:** `interpretability/figures/` — `i03_overall_ranking.png` (top-15, livestock highlighted),
`i03_per_tower_top8.png` (T2 vs. T4/T9 contrast), `i03_species_split.png` (cattle vs. sheep/lamb).

**Data:** `interpretability/data/` — full ranking + per-tower breakdown CSVs.

**Caveat:** I-03 covers TabPFN only; TabICLv2's interpretability is a flagged, not-yet-executed
follow-up. I-02 (D-61, the prior comprehensive pass across 8 models) predates the champion
architecture — kept for the SHAP/native-importance/SARIMAX-coefficient methods I-03 doesn't cover.

---

## 4. Scenario projection (Phase 07)

**Current (S-05, TabICLv2+Variant A+species, 2050 horizon, bias-corrected D-100):** cattle
dominates the FCH4 response far beyond its LSU-weight share. Raw vs. bias-corrected 3×-cattle-alone
response:

| Tower | Raw | Bias-corrected (D-100) |
|---|---|---|
| T4 | +214.5% | +101.8% |
| T9 | +187.3% | +110.4% |

Bias correction (delta-method, anchoring to the real historical mean) roughly **halves** the raw
headline — S-05's own baseline check found 40–80% underprediction of real history at every tower
(far larger than S-01/S-04's 9–20% gap). Direction of the cattle-dominance finding is unaffected;
exact magnitude is genuinely uncertain between raw/corrected, stated as such.

**S-04 (transient annual 2025–2050, both SSPs, full realization scale) — bias-corrected 1×→3×
livestock response:** T2 +40.7%, T4 +135.4%, T9 +114.4% (was +38.6%/+156.4%/+120.3% pre-correction).
SSP2-4.5 vs SSP5-8.5 divergence is real, grows toward 2050, stays under 1% of the mean throughout —
a minor lever next to livestock. AOA-flagged extrapolation risk: 9–15% (S-04) vs. 62–68% (S-05,
wider TabICLv2 feature space) — absolute level tracks feature-space breadth, not new risk.

**Management levers (S-05, D-86):** grazing-season length is a real, monotonic driver (+18.9% at T4
for +4 weeks) — livestock-linked, consistent with the project's central thesis. Fertilizer schedule
is null (<5%, sign-inconsistent across towers) — twice-confirmed, matches F-01/F-04/F-05's
real-data "redundant on the rich base" finding.

**UQ attached to actual scenario trajectories (D-92):** U-06/U-07's CQR calibration applied
directly to S-05's 2050 output — >99% coverage at T4/T9 (T2 0%, pre-established degeneracy).

**Figures:** `scenario_projection/figures/` — `s05_trajectory_bands_2050.png` (full 2050 trajectory
with realization spread), `s05_species_response_2050.png` (cattle/sheep/lamb dose-response),
`s05_aoa_trend_2050.png`, `s04_ssp_divergence.png`, `s04_hybrid_vs_benchmark.png` (hybrid vs.
tree-only diagnostic — the core "fixes U-03's flattening" result), `s04_trajectory_bands.png`,
`s05_uq_cqr_livestock_ssp245.png` (calibrated intervals on the actual scenario).

**Data:** `scenario_projection/data/` — bias-corrected summary tables (S-01/S-04), S-05's 2050
trajectory/species/AOA/SSP summaries. Note: S-05's full realization-level CSVs (~42MB each) were
**not** copied here — see `results/s05_trajectory_realizations_2050_bias_corrected.csv` in the main
`results/` directory for the raw, uncompressed sweep.

**In progress, not yet finalized (D-101/S-06):** bias-correcting the CMIP6 driver data itself (not
just the model's output) — precipitation is ~4.3× too wet, temperature 2–3.5°C too cool in the raw
simulated data vs. real NWFP history. Full corrected sweep launched but write-up still pending as
of this snapshot — not included above, check `DECISIONS.md` D-101 for current status.

---

## What's NOT in this snapshot

This is a curated subset, not a full mirror — large raw sweep outputs (S-05's 42MB realization-level
CSVs, the full `u04_fancharts`/`s05_summary` figure sets with all towers×anchors×axes) were left in
their original `results/`/`notebooks/*/` locations to keep this backup small and legible. Use
`BEST_RESULTS.md` + the decision IDs cited above to trace back to full detail.
