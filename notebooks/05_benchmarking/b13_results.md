# B-13 — TFT and TabPFN for recursive rollout (+ DLinear/LSTM chain-plot extension)

**Notebook:** `B13_tft_tabpfn.ipynb` (single-anchor smoke test, 2021-12-16, Tower 4, both TFT and
TabPFN). **Results:** `results/b13a_tft_summary.csv`, `results/b13b_tabpfn_summary.csv`
(single-anchor); `results/b13a_tft_multi_anchor.csv`, `results/b13b_tabpfn_multi_anchor.csv`
(5-anchor extension, 2018–2022 — the headline verdicts below). **Extended:**
`src/models/recursive_rollout.py` gained `tabpfn_forecast` (new, one-shot, not a rollout).
**Visualization:** `results/figures/b10_chains/T{t}_anchor{yr}_{DLinear,LSTM}.png` (30 new plots,
Part A) + `T4_anchor2021_{TFT,TabPFN}.png` (single-anchor spot-checks).

Fills in the two stretch items B-09 (D-53) originally deferred — TFT (known instability,
D-45/D-48) and TabPFN (unresearched, never installed) — plus a visualization gap the user
flagged: DLinear/LSTM had never been plotted across the full 3-tower x 5-year grid the tree
models got.

## Part A — DLinear/LSTM chain-plot extension

Mirrors `b10_chain_plots.py`'s loop exactly, reusing `forecasting_dl`/`dl_rollout` unmodified.
30 new plots (2 models x 3 towers x 5 years), spot-checked against B-09's original single-anchor
chain — consistent with B-09's documented behaviour (tracks peak *timing* reasonably, misses
peak *magnitude*, some erratic dips in DLinear specifically).

## Part B — TFT: D-45's fix generalizes to recursive rollout

**Adaptation from D-45's recipe**: validation carve-out shortened from a full held-out year to
the **last 90 days before the anchor** (early anchors don't have a spare year — anchor=2018 has
only 715 pre-anchor days total). Same `d_model=32/n_heads=4/weight_decay=1e-3/patience=5`
otherwise, no new HPO.

**Result: no reproduction of the original unregularized-TFT pathology.** Single-anchor sanity
check: no NaNs, sanely-scaled predictions (mean 22.9 vs real 32.1, std 33.2 vs real 64.6).
Multi-anchor (5-anchor, n-weighted mean):

| Model | Mean R² | Mean MASE |
|---|---|---|
| XGB | 0.003 | 0.968 |
| Ensemble_unweighted (B-10) | 0.012 | 0.975 |
| LightGBM | -0.014 | 0.978 |
| SARIMAX | -0.039 | 1.038 |
| RF | -0.067 | 1.024 |
| **TFT** | **-0.237** | **1.055** |
| LSTM | -0.438 | 1.104 |
| DLinear | -1.460 | 1.580 |

**TFT is the best deep-learning model in the entire B-09–B13 sequence** — clearly ahead of LSTM
and dramatically ahead of DLinear — but still behind every tree/SARIMAX model. This directly
mirrors D-45's own conclusion for the original forecasting-phase TFT: regularization converts it
from "catastrophic" to "genuinely reasonable, non-competitive" — the same pattern transfers
cleanly to the recursive-rollout context, a useful confirmation that the fix is general, not a
one-off patch for B-03b's specific setup.

## Part C — TabPFN: a genuinely competitive zero-shot result

**Design**: `tabpfn-time-series` is **not autoregressive** — one forward pass predicts the full
365-day horizon from a `context_df` (history) + `future_df` (known future covariates),
architecturally closer to SARIMAX's single `get_forecast` call than to any rollout loop. Runs in
**local inference mode** (one-time browser license acceptance at ux.priorlabs.ai, `TABPFN_TOKEN`
env var — all inference then runs on the local GPU, no per-call data transmission). Context uses
real `y_observed` (gaps as NaN, handled internally by TabPFN) rather than `y_gapfilled` —
deliberately avoiding the diffuse gap-filler-optimism caveat that applies to every other model's
training target, since TabPFN's context isn't supervised training in the same sense. Per-tower
only (no static-covariate/pooling support in the simple `predict_df` API).

**Result: the best mean MASE of any model tested in the entire B-09–B13 sequence.**

| Model | Mean R² | Mean MASE |
|---|---|---|
| Ensemble_unweighted (B-10) | 0.012 | 0.975 |
| XGB | 0.003 | 0.968 |
| **TabPFN** | **-0.006** | **0.862** |
| LightGBM | -0.014 | 0.978 |
| SARIMAX | -0.039 | 1.038 |

TabPFN's mean R² (-0.006) sits between LightGBM and XGB — competitive with, though not quite
beating, B-10's ensemble. Its mean MASE (0.862) beats **every single model tested this session**,
including the ensemble (0.975) and XGB (0.968) — and every one of its 5 per-anchor MASE values is
comfortably below 1 (0.809–0.891), the most *consistently* sub-1 result of any model. By bin, it
is also the only model with **positive mean R² in both late-window bins** (181-270: 0.069,
271-365: 0.058) — the seasonal-echo degradation that hits most other models' late window (D-53's
addendum) does not clearly appear here.

**This is achieved with zero training or hyperparameter search** — TabPFN is a pretrained
foundation model used purely via in-context learning (the context + future covariates *are* the
"training," supplied at inference time). That a zero-shot approach is essentially competitive
with the carefully-tuned tree ensemble that took the whole B-09→B-12 sequence to arrive at is the
single most notable finding of this session.

**Full 3-tower x 5-year grid** (extending beyond the initially-approved Tower-4-only scope, since
TabPFN runs in ~1-2s per anchor — cheap enough to match the same grid every other model in this
sequence got): `results/b13b_tabpfn_full_grid.csv`, 15 chain plots in `results/figures/b10_chains/
T{2,4,9}_anchor{2018-2022}_TabPFN.png`.

| Tower | Mean R² | Mean MASE | Usable anchors |
|---|---|---|---|
| 4 | -0.006 | 0.862 | 5/5 |
| 9 | -0.264 | 0.925 | 4/5 (2018 anchor: 0% real test-window coverage, NaN) |
| 2 | -0.240 | 0.243 | 1/5 (only 2018; 2019-2022 have 0% real test-window coverage) |

Tower 4's full-grid numbers exactly reproduce the Tower-4-only sweep above (internal consistency
check passed). Tower 9's 4 usable anchors show the same pattern as Tower 4 (spot-checked
visually: T9/anchor2022 tracks the July-October high-flux season correctly, misses the ~320 spike
entirely — same strength, same limitation). **Tower 2's single data point (MASE=0.243) should not
be read as "TabPFN is exceptional at Tower 2"** — it reflects Tower 2's severe, already-documented
real-data scarcity (only 404 real days across the whole 2017-2025 record, concentrated in an
early window) making the persistence baseline unusually bad in that one window, not a
meaningfully large evaluation. This mirrors the same Tower-2/9 data-coverage caveat already raised
for the B-10 chain-plot extension.

## Revised picture

| Model | Mean R² | Mean MASE | Training required |
|---|---|---|---|
| Ensemble_unweighted (B-10) | **0.012** | 0.975 | 4 tuned models |
| XGB | 0.003 | 0.968 | 1 tuned model |
| **TabPFN** | -0.006 | **0.862** | **none (zero-shot)** |
| LightGBM | -0.014 | 0.978 | 1 tuned model |
| SARIMAX | -0.039 | 1.038 | 1 tuned model |
| RF | -0.067 | 1.024 | 1 tuned model |
| TFT | -0.237 | 1.055 | 1 regularized model |
| LSTM | -0.438 | 1.104 | 1 model |
| DLinear | -1.460 | 1.580 | 1 model |

R² and MASE point in slightly different directions here, the same pattern this project has named
repeatedly (D-44b's "spike-tail signature"): B-10's ensemble keeps a narrow R² edge, but TabPFN
has the most robust absolute-error performance of anything tested, for no training cost at all.

## Recommendation

- **B-10's ensemble remains the headline R² recommendation** (D-56) — this doesn't change.
- **TabPFN is worth a genuine, standing mention as an alternative or complementary choice** —
  particularly attractive if compute/engineering cost matters (no training pipeline to maintain)
  or if MASE/robustness is the priority metric. Not a replacement recommendation, but not a
  footnote either.
- **TFT confirms D-45's fix is a general pattern, not a one-off** — worth citing as corroborating
  evidence if this project's write-up discusses the "regularization reverses an apparent
  architecture verdict" methodological point (D-45's own conclusion).
- **A natural follow-up** (not attempted here, out of this bounded session's scope): TabPFN's
  strong single-shot result invites testing whether B-10's ensemble idea extends to include it
  (an ensemble of TabPFN + the tree models) — flagged as a legitimate future extension, not
  executed to avoid re-opening the closed B-09→B-12 sequence with new scope.

## Caveats

- TFT's single-anchor smoke-test numbers showed minor run-to-run variation between the prototype
  script and the notebook's own execution (e.g. 1-7 bin R² -9.893 vs -11.391) — same GPU/attention
  non-determinism already noted for tree ensembles in B-12; doesn't affect the multi-anchor
  conclusion, which is computed once per run and internally consistent.
- TabPFN's `max_context_length` default (32,768) comfortably covers this project's longest
  pre-anchor history (2,176 days for the 2022 anchor) — not a binding constraint here, but would
  become one for the eventual long-range (2030) scenario work if reused directly.
- Same ground-truth and single-calendar-date-anchor caveats as B-09–B-12.
- TabPFN and TFT results are Tower 4 only — no multi-tower validation attempted (matching every
  other B-09–B12 model's committed scope).

## Files / scope

New: `notebooks/05_benchmarking/B13_tft_tabpfn.ipynb`, `results/b13a_tft_summary.csv`,
`results/b13b_tabpfn_summary.csv` (single-anchor), `results/b13a_tft_multi_anchor.csv`,
`results/b13b_tabpfn_multi_anchor.csv` (5-anchor, Tower 4), `results/b13b_tabpfn_full_grid.csv`
(3-tower x 5-year grid), `results/figures/b10_chains/T{t}_anchor{yr}_{DLinear,LSTM}.png` (30
files), `T4_anchor2021_TFT.png`, `T{2,4,9}_anchor{2018-2022}_TabPFN.png` (16 files). Extended:
`src/models/recursive_rollout.py` (`tabpfn_forecast`, new function; `dl_rollout`/`build_model`
reused unmodified for TFT). `.env` (gitignored, not committed) holds `TABPFN_TOKEN`. No `benchmarks.csv`
rows appended for B-13 (single-anchor smoke test only, matching the "diagnostic, not headline
ledger" precedent already used for every B-09–B12 multi-anchor sweep).

*Source: `B13_tft_tabpfn.ipynb`, `results/b13a_tft_*.csv`, `results/b13b_tabpfn_*.csv`.
Cross-ref D-45 (TFT regularization recipe this reuses), D-53/54/55/56 (the B-09→B-12 baseline
this compares against).*
