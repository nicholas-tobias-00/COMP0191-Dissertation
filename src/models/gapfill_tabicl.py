"""TabICL-solo FCH4 gap-filler (D-79) -- the first model this project has found that beats the
RFm champion at more than one tower.

TabICL (Tabular In-Context Learning) is a foundation model: no training in the usual sense, just
an in-context fit on a subsample of rows (`FOUNDATION_MODEL_ROW_CAP`, fixed regardless of how
much data is actually available). Two findings from D5-D8 (see
`notebooks/03c_gap_filling_revisited/summary.md` S15.5, S18) shape this module:

1. **Solo (per-tower), not pooled.** `TabICLRegressor` always subsamples to a fixed row cap --
   pooling across all 3 towers dilutes that fixed budget, whereas solo spends the whole budget on
   one tower. Unlike RF (uncapped, so pooling is unambiguously more data for free), TabICL-solo
   beats TabICL-pooled at every tower (T2/T4/T9: +0.118/+0.005/+0.059 R2). Adopted here as the
   only mode this module offers -- there is no `pooled` argument.
2. **Row-cap bagging helps, but only at Tower 4.** Fitting `n_bags` independent models on
   different random subsamples (instead of one fixed-seed subsample) and averaging predictions
   increases how much of a large tower's domain the model actually sees. Real gain specific to
   Tower 4 (~55k domain hours, only ~18% ever seen by a single subsample): +0.012-0.013 R2 at
   n_bags=5-8 (plateaus by n_bags=5). Flat/negative at T2 (already fits under the cap) and T9
   (intermediate coverage) -- **do not default n_bags>1 project-wide**, pass it explicitly per
   tower if wanted.

Result (headline, all 3 towers, champion FEATURES, n_bags=1): T2=0.676, T4=0.428, T9=0.423, vs.
RFm's 0.576/0.404/0.426 -- beats RFm at T2/T4, ties at T9. Flagged as a validated **benchmark**
result in `BEST_RESULTS.md` S1, not yet the production-adopted config (no UQ/production-fill
tooling existed for it before this refactor).

Run standalone for a quick per-tower gap-CV check:  python src/models/gapfill_tabicl.py [n_bags]
"""
from pathlib import Path
import sys

import numpy as np
from sklearn.impute import SimpleImputer

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from gapfill_rfm import load_ext, frame, feat_list, TOWERS  # noqa: E402
from gap_cv import dom_mask, SCENARIOS, insert_calendar_gaps, gapfilling_metrics, median_metrics, headline  # noqa: E402

FEATURES = feat_list()
FOUNDATION_MODEL_ROW_CAP = 10_000


def fit(feat, trd, n_bags=1):
    """Fits TabICL-solo. n_bags=1 (default) reproduces the original single-fixed-subsample
    behaviour (random_state=42, matching D5-D7's established config) exactly. n_bags>1 bags
    n_bags independent random-subsample fits (distinct random_state per bag) and averages their
    predictions at predict time -- see module docstring for when this actually helps (Tower 4
    only)."""
    from tabicl import TabICLRegressor

    if n_bags == 1:
        # Matches the original fit_tabicl exactly: imputer fit on the subsample only, not the
        # full pre-subsample trd (unlike the n_bags>1 branch below, where one shared imputer
        # across multiple different subsample draws requires fitting on the full trd instead).
        imp = SimpleImputer(strategy="mean")
        trd_sub = trd.sample(n=min(FOUNDATION_MODEL_ROW_CAP, len(trd)), random_state=42)
        m = TabICLRegressor(random_state=42)
        m.fit(imp.fit_transform(trd_sub[feat].values), trd_sub["target"].values)
        return [m], imp
    imp = SimpleImputer(strategy="mean")
    imp.fit(trd[feat].values)
    models = []
    for b in range(n_bags):
        trd_sub = trd.sample(n=min(FOUNDATION_MODEL_ROW_CAP, len(trd)), random_state=1000 + b)
        m = TabICLRegressor(random_state=1000 + b)
        m.fit(imp.transform(trd_sub[feat].values), trd_sub["target"].values)
        models.append(m)
    return models, imp


def predict(models, imp, X):
    """X: raw (unimputed) feature matrix. Averages predictions across all bagged models
    (n_bags=1 -> a single model, i.e. a plain predict)."""
    Xt = imp.transform(X)
    preds = np.stack([m.predict(Xt) for m in models], axis=0)
    return preds.mean(axis=0)


def evaluate_tower(t, d_all, n_bags=1):
    """Full gap-CV sweep (5 scenarios x N_REPS) for tower t, TabICL-solo (never pooled -- see
    module docstring)."""
    g = frame(t, pooled=False, d=d_all)
    dm = dom_mask(g.index, t)
    train_std = float(g.loc[dm, "target"].std())
    out = {}
    for sc, gh in SCENARIOS.items():
        rows = []
        for gt in insert_calendar_gaps(g, "target", dm, gh):
            if len(gt) < 5:
                continue
            train_rows = g.loc[dm & g["target"].notna().values].drop(index=gt, errors="ignore")
            test_rows = g.loc[gt]
            models, imp = fit(FEATURES, train_rows, n_bags=n_bags)
            yp = predict(models, imp, test_rows[FEATURES].values)
            rows.append(gapfilling_metrics(test_rows["target"].values, yp, train_std))
        out[sc] = median_metrics(rows)
    return out


if __name__ == "__main__":
    n_bags = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    d_all = load_ext()
    for t in TOWERS:
        res = evaluate_tower(t, d_all, n_bags=n_bags)
        h = headline(res)
        print(f"[n_bags={n_bags}] Tower {t}: MAE={h['MAE']:.2f}  nMAE={h['nMAE']:.3f}  "
              f"RMSE={h['RMSE']:.2f}  R2={h['R2']:.3f}  R2_OLS={h['R2_OLS']:.3f}")
