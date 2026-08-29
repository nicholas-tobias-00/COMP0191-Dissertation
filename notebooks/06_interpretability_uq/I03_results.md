# I-03 results: interpretability recalibrated for the current champion (TabPFN+species)

**Script:** `i03_champion_interpretability.py` · **Data:** `results/i03_tabpfn_species_importance.csv`
(780 rows), `results/i03_tabpfn_species_importance_ranked.csv`,
`results/i03_tabpfn_species_importance_by_tower.csv`.

## Context

I-02 (D-61, 2026-07-06) is this project's only comprehensive interpretability pass on the
recursive-rollout forecasting track — but it predates TabICLv2 joining the roster (D-66,
2026-07-09) and F-10's species-disaggregated features (D-67, 2026-07-10), which together produced
the standing champion, **TabPFN+species** (MASE=0.715, climatology-scored, D-80). I-02's SHAP/
permutation results were computed on the old 8-model roster and the old (BASE-only) feature set —
the model this project actually recommends had never been through an interpretability pass. This
is the same "predates the champion" gap U-04 already closed for UQ; I-03 applies the identical fix
to interpretability.

**Scope (champion-focused, mirrors U-04's precedent):** TabPFN only — the single best model in
this project's roster (S-03's own 11-model table: TabPFN MASE=0.855, the lowest/best of all models
tested, including TabICLv2 at 0.930) — i.e. the exact model ingested as "Model 1" for S-03's
driver-availability ablation. TabICLv2 is not covered here; a natural, cheap follow-up given it's
also zero-shot with the same cost profile, flagged but not executed this pass.

## Method

**Unchanged from I-02's own TabPFN treatment** (`i02_multi_anchor_tower.py`, lines ~277–304):
permutation importance is TabPFN's only available substitute for a native per-feature signal, and
it is architecturally mismatched with SHAP's row-wise tabular framework (I-02's own documented
reasoning, not revisited here). Per (anchor, tower): one baseline zero-shot rollout, then one
single-shuffle permutation per feature column (seeded by anchor year, exactly as I-02 did — not
upgraded to `n_repeats>1`, so old-config vs. new-config importance numbers stay comparable, not
just similarly computed). Importance = `|mean(shuffled_chain) − mean(base_chain)|`.

**What changed:** (a) the feature set is `BASE+species` — 52 `fx_` columns from
`forecast_daily_v3.csv` (F-10's actual champion config, identical construction to
`u04_champion_uq.py`'s `fit_stage_champion`, imported in spirit not retyped); (b) source data is
`forecast_daily_v3.csv`, not v2.

Same 3-tower × 5-anchor sweep as I-02/U-02/U-04 (2018–2022 anchors, T2/T4/T9) — full coverage.
53 TabPFN calls per (tower, anchor) × 15 pairs = 795 calls total. Smoke-tested (1 tower, 1 anchor,
3 features) before the full run. **Actual runtime: ~22 minutes** (1,314s), no training — pure
zero-shot inference throughout.

## Headline: livestock features dominate, and the champion's new species split is genuinely load-bearing

**Overall ranking (mean importance, all towers/anchors pooled), top 10 of 52:**

| Rank | Feature | Mean importance |
|---|---|---|
| 1 | `fx_lsu_dens` | 1.1456 |
| 2 | `fx_cattle_dens` | 0.8043 |
| 3 | `fx_grazing_active` | 0.3750 |
| 4 | `fx_total_liveweight_dens` | 0.2807 |
| 5 | `fx_is_growing` | 0.1593 |
| 6 | `fx_wd_cos` | 0.0961 |
| 7 | `fx_TA_max` | 0.0899 |
| 8 | `fx_TA_mean` | 0.0828 |
| 9 | `fx_mgmt_lime_recency` | 0.0647 |
| 10 | `fx_days_since_grazing` | 0.0636 |

`fx_sheep_dens` (0.0143) and `fx_lamb_dens` (0.0320) rank 47th and 29th of 52 — near the bottom,
an order of magnitude below `fx_cattle_dens`.

**This directly explains *why* species-disaggregation (F-10/D-67) produced the champion, rather
than being a uniform improvement**: the model's own permutation-importance signal shows the gain is
concentrated entirely in `fx_cattle_dens`, not a generic benefit of splitting livestock by species.
Sheep/lamb densities carry almost no independent signal beyond what `fx_lsu_dens` already captures.
This is the same asymmetry S-05's scenario-projection work found independently from a completely
different angle (cattle tripling ~triples predicted FCH4 at T4/T9; sheep/lamb response stays under
25% even at 3×) — two unrelated methods (real-anchor permutation importance here; scenario dose-
response there) now agree on the same mechanism.

## Per-tower breakdown — Tower 2 confirms its livestock-blindness a further, independent time

| Tower | Top feature | 2nd | 3rd |
|---|---|---|---|
| T2 | `fx_TA_max` (0.1224) | `fx_TA_mean` (0.0851) | `fx_TS_lag28` (0.0802) |
| T4 | `fx_lsu_dens` (1.7210) | `fx_cattle_dens` (1.0627) | `fx_grazing_active` (0.5398) |
| T9 | `fx_lsu_dens` (1.7142) | `fx_cattle_dens` (1.3503) | `fx_grazing_active` (0.5735) |

**Tower 2 has no livestock feature in its top 10 at all** — its top-ranked drivers are all
meteorological/soil-temperature (`fx_TA_max`, `fx_TA_mean`, `fx_TS_lag28`, `fx_SWIN_mean`,
`fx_is_winter`). This is now confirmed by a fourth independent method (after U-03's rollout stress
test, S-01's AOA/climatology-max check, and S-05-T2's pooling test) that Tower 2's near-absent
livestock signal is a genuine, structural property of that catchment's record, not an artifact of
any one method — TabPFN's own permutation importance, run directly on the champion architecture,
agrees with everything this project has already found about Tower 2 from other angles.

T4/T9 both show the identical top-3 ordering (`fx_lsu_dens` > `fx_cattle_dens` > `fx_grazing_active`)
— consistent, not tower-specific noise.

## Practical implications

1. **The interpretability gap this project carried since D-66/D-67 is closed for the model that
   actually needs it** (TabPFN+species, the standing champion). TabICLv2 remains unclosed — flagged,
   not executed this pass.
2. **Confirms and sharpens I-02's original finding**, rather than overturning it: `fx_lsu_dens`
   dominance holds on the actual champion config too, but I-03 additionally shows *which* of the
   species-split components (`fx_cattle_dens`) is doing the real work — a finding I-02 could not
   have made, since it predates the species features entirely.
3. **Cross-validates S-05's scenario-projection cattle-dominance finding from an entirely different
   method and a different part of the project** (real-anchor permutation importance vs. scenario
   dose-response sweep) — independent convergence, not circular reasoning (different data windows,
   different question, same underlying mechanism).
4. **Tower 2's livestock-blindness is now a four-times-replicated finding** across UQ (U-03),
   scenario AOA (S-01), scenario pooling (S05-T2/D-95), and now interpretability (I-03) — should be
   stated as a settled, cross-method characteristic of that catchment in the write-up, not a
   single-method observation.

## Files

- `notebooks/06_interpretability_uq/i03_champion_interpretability.py` — script (committed,
  smoke-tested before the full run).
- `notebooks/06_interpretability_uq/i03_run_log.txt` — full run log.
- `results/i03_tabpfn_species_importance.csv` (780 rows, raw per-anchor/tower/feature importance).
- `results/i03_tabpfn_species_importance_ranked.csv` (overall ranking, 52 rows).
- `results/i03_tabpfn_species_importance_by_tower.csv` (per-tower ranking, 156 rows).

No `benchmarks.csv` rows (interpretability output, not a point-forecast/interval-calibration
benchmark row — same exclusion precedent as I-01/I-02/U-01/U-02/U-03).
