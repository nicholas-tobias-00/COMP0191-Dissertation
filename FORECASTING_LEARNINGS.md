# Time Series Forecasting — Learnings from the M5 Replication

Notes from replicating M5's 1st-place Accuracy (A1) and 1st-place Uncertainty (U1)
solutions locally, on real data, on Windows, in 2026 — six years after the original
code was written. Kept for reuse on future forecasting projects.

## Modeling learnings

- **Recursive vs. non-recursive multi-step forecasting.** A1's winning ensemble
  blends both: recursive models predict one day at a time and feed predictions back
  in as lag features (compounds error but captures short-term dynamics); non-recursive
  models predict all 28 days directly from a single fit (no error compounding, but
  needs the horizon as a feature). Blending both styles was a deliberate part of the
  winning design, not incidental.
- **Partition by hierarchy for tractability and accuracy.** Rather than one global
  model, both winners train many small models sliced by store / category / department.
  This keeps each model's data homogeneous (a store in Texas and a store in California
  have different demand patterns) and keeps training sets small enough to iterate fast.
- **Tweedie loss suits intermittent, zero-inflated demand.** Retail unit sales are
  mostly small non-negative integers with many zeros — Tweedie (`tweedie_variance_power`
  ~1.1) outperforms plain regression losses here; worth defaulting to it for any
  similar sparse-count demand series before reaching for standard MSE.
- **Feature importance was dominated by**: `item_id` (highest single feature — model
  leans heavily on per-item identity/history), rolling std/mean at multiple windows
  (7/14/30/60/180 day), calendar position (`tm_w` week, `tm_d` day-of-month, `tm_dw`
  day-of-week), price statistics (rolling min/mean/std, momentum), and target-encoded
  category/department/item means. If engineering features for a new demand-forecasting
  task, this set is a strong starting checklist.
- **Quantile ensembles for uncertainty, scored hierarchically.** U1 trains a separate
  LightGBM quantile regressor per quantile level (9 levels: 0.005…0.995) per
  aggregation level (12 levels: Total → Item-Store). Loss (WSPL) is naturally lowest
  at coarse aggregates (Total ≈ 0.10) and highest at the finest grain (item-store ≈
  0.25) — expected and not a bug: aggregation cancels idiosyncratic noise. Don't
  benchmark a single "accuracy" number across a hierarchy without splitting by level.
- **Always carve out a real validation window.** Both winners' code defaults to
  training through the last known day and forecasting into the truly blind
  competition window — which by construction has no accessible ground truth once
  you're replicating outside the original competition. Rerouting both to train
  through day *T−28* and forecast *T−27…T* (the last 28 known days) let every result
  in this project be checked against real actuals. Default to this pattern for any
  forecasting replication/backtest: hold out the last known horizon, don't just
  point at "the future."

## Engineering gotchas hit (with fixes)

Legacy notebooks (this code was ~2020-era) accumulate real rot against modern
library versions. Every one of these was a hard crash, not a warning:

| Issue | Fix |
|---|---|
| LightGBM 4.x removed `verbose_eval` kwarg from `lgb.train()` | Use `callbacks=[lgb.log_evaluation(n)]` |
| pandas 2.0 removed `Series.iteritems()` | Use `.items()` |
| pandas 2.x made `DataFrame.pivot()` reject positional args | Use `pivot(index=..., columns=..., values=...)` |
| sklearn `make_scorer(fn, False, ...)` — `greater_is_better` is keyword-only now | `make_scorer(fn, greater_is_better=False, ...)` |
| sklearn `RandomizedSearchCV.fit(x, y, groups)` — `groups` is keyword-only now | `.fit(x, y, groups=groups)` |
| **Windows-only**: `sklearn.model_selection.ParameterSampler` overflows int32 (`ValueError: high is out of bounds for int32`) on huge parameter grids (~10^11 combinations) | numpy's default integer width is 32-bit on Windows vs. 64-bit on Linux/Mac — this class of bug is invisible on the original authors' machines. Monkeypatch `sample_without_replacement` in the calling module's namespace (not just `sklearn.utils`) with a `numpy.random.default_rng().choice(..., replace=False)`-based version. Preserves search semantics exactly. |
| **Windows-only**: console crashes printing a non-ASCII character (e.g. `μ`) — `UnicodeEncodeError: 'charmap' codec can't encode character` | Set `PYTHONIOENCODING=utf-8` before invoking python |
| Stale sanity-check `assert`s tied to an old data format (e.g. an ID-matching check against a sample-submission schema that no longer matches the current file) | Don't trust an inherited assertion blindly — test the actual intersection/overlap against real current data before deciding whether to fix the logic or the data, versus just disabling a check that no longer reflects reality |
| Wrong reference file bundled — U1's data folder held the **Accuracy** competition's `sample_submission.csv` format (60,980 item-level rows) where the **Uncertainty** competition needs a much larger multi-hierarchy-level format (~770K rows) | Competitions/datasets that look similar (same file name, same host competition family) can silently carry an incompatible schema. Check row counts and id patterns against what the code expects, don't assume from the filename. |
| Kaggle API auth: newer `kaggle` CLI (2.x) doesn't recognize the classic `kaggle.json` (username+key) format, returns "Authentication required" even with valid credentials | Pin `kaggle==1.6.x` for classic-credential workflows, or migrate to the new OAuth/token flow deliberately |
| `df[col][mask] = values` (chained assignment) still works in pandas 2.2 without Copy-on-Write enabled, but is fragile against pandas 3.0 | Prefer `df.loc[mask, col] = values` in any code meant to last; verify empirically (`FutureWarning`/`SettingWithCopyWarning` text tells you directly) rather than assuming either way |

## Operational / process learnings

- **Windows buffers stdout fully when redirected to a file.** A backgrounded
  `python script.py > log.txt` on Windows shows *nothing* in the log until the
  process exits or the buffer fills — looks identical to a hang. Use `python -u`
  (or `PYTHONIOENCODING`/`-u` together) to get live line-by-line output, especially
  important when monitoring multi-hour background jobs.
- **When logs are silently buffered, use process CPU-time deltas to confirm liveness**,
  not just log growth or wall-clock elapsed. A process with climbing CPU-seconds
  across two checks is definitely working even with zero new log lines.
- **CPU contention between concurrent heavy jobs is real and large.** Running two
  `n_jobs=-1` LightGBM/sklearn jobs at once on a 12-thread machine roughly doubled
  wall-clock time for both, compared to running them one after another. On a
  core-constrained machine, chain heavy parallel jobs sequentially (`a && b && c`)
  rather than backgrounding them concurrently, unless the machine has real headroom.
- **Smoke-test the cheapest variant first.** Before committing to the ~10-hour
  configuration of a pipeline, run the smallest/fastest variant (fewest series,
  smallest hierarchy level) first — every bug in this project surfaced and was fixed
  in an 11-minute run rather than hours in.
- **A vendor-provided "fast mode" is worth trusting if the authors quantified the
  tradeoff.** U1's `SPEED=True` flag was documented by the original team as "~20x
  faster, 0% CV difference" — worth using as the default rather than the literal
  200-hour original setting, but only because that specific claim was in the code's
  own comments. Don't assume an undocumented speed flag (this repo also had an
  untested `SUPER_SPEED`) is safe without similar evidence.
- **Scope the compute budget explicitly before starting**, especially for a "replicate
  the winning solution" style task: a literal full ensemble (this project: 220 models
  for Accuracy, 45 training runs for Uncertainty) can be a multi-day undertaking.
  Decide up front whether a single representative ensemble member / the vendor's own
  fast-replication path is an acceptable substitute for the literal full pipeline —
  cheaper to agree on this before spending hours of compute than to discover it after.

## Results achieved (for reference)

- **A1** (recursive-by-store LightGBM, all 10 stores): 28-day aggregate forecast
  1,214,697 vs. actual 1,231,764 units (**-1.39% bias**); item-day MAE 1.04, RMSE 2.06.
- **U1** (quantile LightGBM, all 12 hierarchy levels): mean WSPL **0.167** across all
  levels/quantiles; ranged from 0.10 (Total level) to 0.26 (item-store level),
  matching the expected coarse-vs-fine-grain pattern.

## Recommendations for the next forecasting task

1. Start with a validation-window backtest (train through day *T*, forecast *T+1…T+h*)
   before ever pointing a pipeline at a truly-blind future window — you need ground
   truth to know if anything works.
2. Partition models by natural hierarchy (region/store/category) before reaching for
   one global model, especially for retail/demand data with heterogeneous sub-populations.
3. Default to Tweedie or a similar zero-inflated loss for sparse count data rather
   than plain regression losses.
4. For uncertainty/quantile work, always report loss broken out by aggregation level —
   a single blended number hides that coarse levels are inherently easier.
5. Budget for legacy-code archaeology when reviving any pre-2023 ML notebook — pandas
   2.x, sklearn's keyword-only argument changes, and LightGBM 4.x are the most common
   breakage sources, all listed above with fixes.
6. On Windows specifically, watch for the int32-vs-int64 default integer width
   difference (breaks large combinatorial sampling) and default `cp1252` console
   encoding (breaks any non-ASCII print) — both are invisible on Mac/Linux.
7. Get explicit agreement on compute scope before launching multi-hour training,
   and smoke-test the cheapest configuration first regardless.
