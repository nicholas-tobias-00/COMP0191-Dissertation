# I-03b: interpretability recalibrated for the actual B-18 champion

Phase 1 of the additive B18-integration plan (2026-08-20, D-107 series). I-03 (D-102) targeted the
prior "TabPFN+species" TS-wrapper champion. B-18 (D-106, 2026-08-19) replaced that champion with a
structurally different architecture: direct tabular regression (`TabPFNRegressor.fit/predict`,
pooled across all 3 towers via tower-dummy features) plus a two-stage p95 spike-gate
(`base + 0.25 * P(spike) * (spike_pred - normal_pred)`), on the `BASE_ALL_52` feature set. I-03's
SHAP/permutation results were never computed on this architecture. This closes that gap.

**Method**: same 5-anchor (2018-2022) x 3-tower sweep, same permutation-importance definition
(`|mean(shuffled) - mean(base)|`, single shuffle per feature, seeded by anchor year) as I-02/I-03.
Champion bundle (classifier + base/normal/spike TabPFN regressors) fit ONCE per anchor (B-18's
model is pooled, not per-tower), then predicted from multiple times (baseline + one shuffle per of
52 `fx_` columns, restricted to that tower's own future rows) -- predict-only, no refit, the
methodologically correct form of permutation importance. Runtime: ~52 min (5 anchor fits + 5 x 53 x
3 predict-only calls), vs. I-03's own 795 full-forecast calls.

## Result: `fx_lsu_dens` dominance holds and strengthens

Overall ranking (mean importance across all towers/anchors), top 10:

| Rank | Feature | Mean importance |
|---|---|---:|
| 1 | `fx_lsu_dens` | 1.746 |
| 2 | `fx_total_liveweight_dens` | 1.652 |
| 3 | `fx_cattle_dens` | 1.550 |
| 4 | `fx_grazing_active` | 0.595 |
| 5 | `fx_WS_mean` | 0.498 |
| 6 | `fx_USTAR_mean` | 0.314 |
| 7 | `fx_wd_sin` | 0.293 |
| 8 | `fx_lamb_dens` | 0.280 |
| 9 | `fx_SWIN_mean` | 0.248 |
| 10 | `fx_days_since_grazing` | 0.195 |

`fx_lsu_dens` stays #1 by a wide margin, and its magnitude is actually higher than I-03's own
number (1.746 vs. 1.1456) -- consistent with, not contradicting, I-03's finding. `fx_cattle_dens`
stays a clear top-3 feature (1.550, was 0.80/#2 in I-03), now alongside a genuinely new feature
`BASE_ALL_52` carries that I-03's config didn't (`fx_total_liveweight_dens`, F-10-era bodyweight
density, #2 here at 1.652) -- reconfirms the cattle-driven livestock story a second, independent
way. `fx_is_arable` stays negligible (0.0002, was ~0 in I-03) -- unchanged.

**One genuine new wrinkle, not present in I-03**: `fx_lamb_dens` jumps to #8 overall (0.280) here,
vs. #near-bottom (0.0320) in I-03. `fx_sheep_dens` stays low (0.034). Worth flagging in any
species-comparison write-up, though it doesn't change the headline cattle-dominance conclusion --
plausibly an artifact of the pooled-fit architecture (dummy-encoded cross-tower fit can pick up
different sensitivity patterns than three independent per-tower TS-forecasts) rather than a new
substantive finding; not investigated further here.

## Per-tower breakdown

**Tower 2** (top 10): `fx_WS_mean` (0.252), `fx_USTAR_mean` (0.168), `fx_VPD_mean` (0.080),
`fx_flow_roll7` (0.079), `fx_mgmt_fertN_rate` (0.075), `fx_SWIN_mean` (0.069), `fx_SWC_lag28`
(0.067), `fx_flow_roll14` (0.065), `fx_mgmt_fertN_recency` (0.063), `fx_DOY_cos` (0.060).
**Zero livestock features in the top 10** -- the 5th independent confirmation of Tower 2's
livestock-blindness (after U-03, S-01, S05-T2/D-95, I-03), now shown to hold under a completely
different model architecture too.

**Tower 4** (top 3): `fx_lsu_dens` (2.864), `fx_total_liveweight_dens` (2.618), `fx_cattle_dens`
(2.433) -- overwhelming livestock dominance, consistent with every prior pass.

**Tower 9** (top 3): `fx_lsu_dens` (2.372), `fx_total_liveweight_dens` (2.334), `fx_cattle_dens`
(2.216) -- same pattern as Tower 4.

## Status

No change to any standing recommendation -- this reconfirms, does not revise, the project's central
livestock/cattle-dominance thesis, now validated against the actual B-18 forecasting champion
rather than the superseded TS-wrapper architecture. Feeds into Phase 3 (S-03b) as evidence that the
livestock-density features remain the dominant signal even in the new architecture, supporting
their central role in any scenario-safe (`FX_A_SPECIES`-restricted) version of this model.

Full outputs: `results/i03b_b18champion_importance.csv` (raw, per anchor/tower/feature),
`results/i03b_b18champion_importance_ranked.csv` (overall ranking),
`results/i03b_b18champion_importance_by_tower.csv` (per-tower top-10s).
