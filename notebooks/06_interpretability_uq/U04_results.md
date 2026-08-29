# U-04 — UQ recalibrated for the current forecasting champion (TabPFN+species, TabICLv2)

**Script:** `u04_champion_uq.py` · **Data:** `results/u04_chains.csv` (10,950 rows), `results/u04_summary.csv` (94 rows).

## Context

U-02 (D-62, 2026-07-06) built leave-one-anchor-out conformal calibration for an 8-model roster on
`forecast_daily_v2.csv` — but that predates **TabICLv2 joining the roster** (D-66, 2026-07-09),
**F-10's species-disaggregated features** (D-67, 2026-07-10), and the standing champion becoming
**TabPFN+species** (D-67, reconfirmed under climatology-MASE at D-80). U-02's "TabPFN" interval is
calibrated for a superseded feature configuration, not the model this project actually recommends.
TabICLv2 has never had UQ built for it at all. U-04 closes this gap directly.

**Scope, user-confirmed (champion-focused, not the full 11-model roster)**: TabPFN and TabICLv2
only, on `forecast_daily_v3.csv`'s `BASE+species` config (F-10's own `FAMILIES`/`BASE_FX`
construction, imported in spirit from `b16_foundation_models_v3.py`, not retyped independently).
Both are zero-shot with native quantile support (`tabpfn_forecast(...,quantiles=...)`,
`tabicl_forecast(...,quantiles=...)`) — no retraining, no new adapters. The other 6 models in
U-02's original roster (RF/XGB/LightGBM/SARIMAX/TFT/2 ensembles) need real refitting per anchor to
recalibrate and their point-accuracy/conformal behavior on the *old* feature set is already on
record — only the two models whose production config actually changed needed rebuilding here.

**Method: unchanged from U-02.** Same 3-tower × 5-anchor sweep (2018–2022), same quantiles
(0.05, 0.5, 0.95), same leave-one-anchor-out split-conformal calibration
(`rr.conformal_margins_by_bin()`, per lead-time bin), same metrics (PICP/MPIW/pinball,
`src/evaluation/metrics.py`). `evaluate_stage()` is imported **unmodified** from
`u02_multi_anchor_tower.py` — only `fit_stage` differs (zero-shot rollout instead of U-02's pooled
tree/SARIMAX/TFT fitting) — so U-02 and U-04's numbers are directly comparable, not just
similarly-computed. Smoke-tested (1 tower/1 anchor, then 1 tower/2 anchors to validate the
calibration path needs ≥2 anchors) before the full run. **Actual runtime: 25 seconds** — no model
training at all, pure zero-shot inference.

## Headline: calibration converges to ~0.89–0.90 at T4/T9, essentially unchanged from U-02's old config

| Model | Tower | Raw PICP | Raw MPIW | Conformal PICP | Conformal MPIW | Conformal pinball |
|---|---|---|---|---|---|---|
| TabPFN | T2 | 0.804 | 65.5 | NaN | NaN | NaN |
| TabPFN | T4 | 0.867 | 111.6 | **0.898** | 149.5 | 10.56 |
| TabPFN | T9 | 0.724 | 125.3 | **0.889** | 188.9 | 12.86 |
| TabICLv2 | T2 | 0.804 | 192.4 | NaN | NaN | NaN |
| TabICLv2 | T4 | 0.964 | 301.7 | **0.895** | 154.7 | 10.63 |
| TabICLv2 | T9 | 0.771 | 290.4 | **0.894** | 195.3 | 13.04 |

Both models' raw (pre-calibration) coverage is imperfect and inconsistent across towers, but
conformal calibration pulls both to ~0.89–0.90 PICP at T4/T9 — replicating U-02's own headline
finding ("every model converges to ~0.88–0.90 regardless of raw coverage") on the new feature
config too.

**Tower 2 still cannot support calibration** (NaN conformal columns) — same pre-existing,
documented limitation U-02 already found ("Tower 2 cannot support calibration at all — real
`y_observed` in only 1/5 anchor windows"), not a new issue introduced by the feature-set change.
T2's real coverage ends May 2019, leaving only the 2018 anchor with usable ground truth — leave-
one-anchor-out calibration needs at least one *other* anchor with real residuals to pool from, and
T2 has none.

## The actual finding: species enrichment improved point accuracy, not calibration quality

Direct comparison, TabPFN, old (U-02, BASE-only) vs. new (U-04, BASE+species):

| Tower | Conformal MPIW (old) | Conformal MPIW (new) | Conformal pinball (old) | Conformal pinball (new) |
|---|---|---|---|---|
| T4 | 148.44 | 149.52 | 10.56 | 10.56 |
| T9 | 190.31 | 188.93 | 13.00 | 12.86 |

Essentially identical — within noise. **F-10/D-67's species features improved TabPFN's point
accuracy (the MASE gain already on record) without materially changing its uncertainty
calibration.** This makes mechanistic sense: conformal margins track the *distribution* of
residuals, and D-80's own numbers show the species-config MASE gain over BASE was modest
(0.715 vs. nearby configs, not a dramatic jump) — a modest point-accuracy change shouldn't be
expected to move interval width much either. Not a surprising result, but a genuinely new,
directly-measured one — the "closes the gap" framing this experiment was built for turned out to
also answer "did closing it matter for UQ specifically" (no), which wasn't guaranteed in advance.

## Practical implications

1. **The UQ gap this project carried since D-66/D-67 is closed for the two models that actually
   need it** (TabPFN+species, the standing champion; TabICLv2, its closest zero-shot competitor).
2. **No recalibration is needed for the other 6 models in U-02's roster** — their feature
   configuration never changed, so their existing U-02 numbers remain valid as-is.
3. **T2's calibration limitation is confirmed to persist independent of feature-set changes** — a
   structural data-scarcity issue (real coverage ends May 2019), not something any feature
   enrichment can fix.
4. **This is the foundation for the next step (scenario-analysis UQ, S-05)** — the same
   `conformal_margins_by_bin()`/leave-one-anchor-out machinery, now validated on the champion's
   actual feature family, is what gets extended (AOA-stratified) for scenario points next.

## Figures

`u04_fanchart_plots.py` — same visual convention as `u02_fanchart_plots.py` (actual/gap-filled
FCH4 + predicted median + shaded 90% conformal-calibrated band, hatched raw-interval fallback
where no calibration margin exists for that day's lead-time bin). Every (tower, anchor, model)
combination plotted: 3 towers × 5 anchors × 2 models = **30 figures**,
`results/figures/u04_fancharts/T{tower}_anchor{year}_{model}.png`.

## Files

- `notebooks/06_interpretability_uq/u04_champion_uq.py` — script (committed, smoke-tested before
  the full run).
- `notebooks/06_interpretability_uq/u04_fanchart_plots.py` — figures (committed).
- `results/u04_chains.csv` (10,950 rows, raw per-day quantile predictions).
- `results/u04_summary.csv` (94 rows, per-model/tower/anchor/bin raw + calibrated metrics).
- `results/figures/u04_fancharts/` (30 figures).

No `benchmarks.csv` rows (UQ output, not a point-forecast benchmark row — same exclusion
precedent as U-01/U-02/U-03).
