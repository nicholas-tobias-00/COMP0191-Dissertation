"""Statistical/imputation-only FCH4 gap-filling baselines: Mean, MICE, HyperImpute (D-79).

Reference-floor models for the RFm champion (`gapfill_rfm.py`) -- not the recommended production
model, but included so the full model roster this project evaluated can be run/scored
consistently. Reuses `gapfill_rfm.py`'s `frame`/`feat_list`/`DUM`/`TOWERS`/`load_ext` as the
single source of truth for features (same cross-import convention as `build_fch4_gapfilled.py`).

MICE = `sklearn.impute.IterativeImputer` (default `BayesianRidge`, one fixed model for every
column). HyperImpute (van der Schaar lab) = same chained-equations structure, but each column's
imputation model is chosen by an internal per-iteration AutoML search over a regressor/classifier
pool instead of one fixed model -- columns with <=5 unique values (tower dummies, `graze`) are
auto-treated as classification targets, a real and expected difference from MICE. Both mask this
fold's held-out target values to NaN first, `fit_transform` the whole (feature + target) matrix
jointly, then read the target column's imputed values back out at the held-out rows' *positional*
index (not label -- pooling stacks 3 towers sharing the same hourly `Datetime` index, so a
label-based lookup on the concatenated frame would match every tower's row at that timestamp, not
just tower t's own row).

Run standalone for a quick per-tower gap-CV check:  python src/models/gapfill_baselines.py mean|mice|hyperimpute
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  registers IterativeImputer
from sklearn.impute import IterativeImputer

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from gapfill_rfm import load_ext, frame, fit, feat_list, DUM, TOWERS, cfg, ts_col_for, PLAUS_LOW, PLAUS_HIGH  # noqa: E402
from gap_cv import dom_mask, SCENARIOS, insert_calendar_gaps, gapfilling_metrics, median_metrics, headline  # noqa: E402

FEATURES = feat_list()
MET_ONLY_FEATURES = FEATURES[:11]   # feat_list()'s own construction order: the 11 raw met/soil
                                    # columns come first, before fc/AUX/livestock/lags/mgmt/gpp-reco


def _pooled_gap_frame(t, pooled, d_all, gt, feat):
    """Builds the concatenated (target-masked-at-gt, feature) matrix for one fold, and returns
    (matrix, absolute_row_positions_of_gt). Shared by run_mice/run_hyperimpute."""
    towers = TOWERS if pooled else [t]
    frames = {tt: frame(tt, pooled, d_all) for tt in towers}
    gt_ts = pd.DatetimeIndex(gt)
    parts = []; t_positions = None; t_part_i = None
    for i, tt in enumerate(towers):
        gg = frames[tt]; dmt = dom_mask(gg.index, tt)
        sub = gg.loc[dmt, ["target"] + feat].copy()
        if tt == t:
            t_part_i = i
            t_positions = sub.index.get_indexer(gt_ts)
            sub.iloc[t_positions, sub.columns.get_loc("target")] = np.nan
        parts.append(sub.reset_index(drop=True))
    offset = sum(len(parts[i]) for i in range(t_part_i))
    abs_positions = offset + t_positions
    mat = pd.concat(parts, ignore_index=True) if pooled else parts[0]
    return frames, mat, abs_positions


def evaluate_tower_met_only(t, d_all):
    """RFm trained on only the 11 raw met/micromet drivers -- no livestock, lags, management, or
    GPP/Reco. A reference floor showing how much of the champion's skill is 'raw weather' versus
    everything else (D-79): met-only R2=0.05/0.04/0.06 vs. champion 0.58/0.40/0.43. Pooled, like
    the champion (`gapfill_rfm.frame`/`gapfill_rfm.fit`, same RF hyperparameters -- only the
    feature list differs)."""
    feat = MET_ONLY_FEATURES + DUM
    g = frame(t, pooled=True, d=d_all); dm = dom_mask(g.index, t)
    train_std = float(g.loc[dm, "target"].std())
    frames = {tt: frame(tt, pooled=True, d=d_all) for tt in TOWERS}
    out = {}
    for sc, gh in SCENARIOS.items():
        rows = []
        for gt in insert_calendar_gaps(g, "target", dm, gh):
            if len(gt) < 5:
                continue
            trd = pd.concat([frames[tt][dom_mask(frames[tt].index, tt) & frames[tt]["target"].notna().values]
                              .drop(index=gt, errors="ignore") for tt in TOWERS], ignore_index=True)
            rf, imp = fit(feat, trd)
            yp = rf.predict(imp.transform(g.loc[gt, feat].values))
            rows.append(gapfilling_metrics(g.loc[gt, "target"].values, yp, train_std))
        out[sc] = median_metrics(rows)
    return out


def evaluate_tower_mean(t, d_all):
    """Simplest possible baseline: fill every held-out point with that fold's own training-set
    mean -- the standard 'trivial' floor."""
    c = cfg(t, ts_col_for(t)); d = d_all.copy(); tgt = c["tgt"]
    d.loc[~d[c["ssitc"]].isin([0, 1]), tgt] = np.nan
    d.loc[d[tgt].notna() & ~d[tgt].between(PLAUS_LOW, PLAUS_HIGH, inclusive="both"), tgt] = np.nan
    dm = dom_mask(d.index, t)
    train_std = float(d.loc[dm, tgt].std())
    out = {}
    for sc, gh in SCENARIOS.items():
        rows = []
        for gt in insert_calendar_gaps(d, tgt, dm, gh):
            if len(gt) < 5:
                continue
            gt_ts = pd.DatetimeIndex(gt)
            train_mean = d.loc[dm & d[tgt].notna() & ~d.index.isin(gt_ts), tgt].mean()
            yp = np.full(len(gt), train_mean)
            rows.append(gapfilling_metrics(d.loc[gt, tgt].values, yp, train_std))
        out[sc] = median_metrics(rows)
    return out


def evaluate_tower_mice(t, pooled, d_all):
    """MICE (sklearn IterativeImputer, default BayesianRidge) on the champion's own FEATURES +
    target, jointly imputed."""
    feat = FEATURES + (DUM if pooled else [])
    g = frame(t, pooled, d_all); dm = dom_mask(g.index, t)
    train_std = float(g.loc[dm, "target"].std())
    out = {}
    for sc, gh in SCENARIOS.items():
        rows = []
        for gt in insert_calendar_gaps(g, "target", dm, gh):
            if len(gt) < 5:
                continue
            _, mat, abs_positions = _pooled_gap_frame(t, pooled, d_all, gt, feat)
            imputer = IterativeImputer(random_state=42)
            filled = imputer.fit_transform(mat.values)
            target_col = mat.columns.get_loc("target")
            yp = filled[abs_positions, target_col]
            rows.append(gapfilling_metrics(g.loc[gt, "target"].values, yp, train_std))
        out[sc] = median_metrics(rows)
    return out


def evaluate_tower_hyperimpute(t, pooled, d_all):
    """HyperImpute (van der Schaar lab): same structure as evaluate_tower_mice, but each column's
    imputation model is chosen via an internal per-iteration AutoML search rather than one fixed
    model. `Imputers().get("hyperimpute")` is imported lazily -- an optional heavy dependency,
    only needed if this specific baseline is actually run."""
    from hyperimpute.plugins.imputers import Imputers

    feat = FEATURES + (DUM if pooled else [])
    g = frame(t, pooled, d_all); dm = dom_mask(g.index, t)
    train_std = float(g.loc[dm, "target"].std())
    out = {}
    for sc, gh in SCENARIOS.items():
        rows = []
        for gt in insert_calendar_gaps(g, "target", dm, gh):
            if len(gt) < 5:
                continue
            _, mat, abs_positions = _pooled_gap_frame(t, pooled, d_all, gt, feat)
            plugin = Imputers().get("hyperimpute", random_state=42)
            filled_df = plugin.fit_transform(mat)
            yp = filled_df.iloc[abs_positions][filled_df.columns[filled_df.columns.get_loc("target")]].values
            rows.append(gapfilling_metrics(g.loc[gt, "target"].values, yp, train_std))
        out[sc] = median_metrics(rows)
    return out


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "mean"
    d_all = load_ext()
    for t in TOWERS:
        if which == "mean":
            res = evaluate_tower_mean(t, d_all)
        elif which == "mice":
            res = evaluate_tower_mice(t, pooled=True, d_all=d_all)
        elif which == "hyperimpute":
            res = evaluate_tower_hyperimpute(t, pooled=True, d_all=d_all)
        elif which == "met_only":
            res = evaluate_tower_met_only(t, d_all)
        else:
            raise ValueError(f"unknown baseline {which!r}, expected mean|mice|hyperimpute|met_only")
        h = headline(res)
        print(f"[{which}] Tower {t}: MAE={h['MAE']:.2f}  nMAE={h['nMAE']:.3f}  RMSE={h['RMSE']:.2f}  "
              f"R2={h['R2']:.3f}  R2_OLS={h['R2_OLS']:.3f}")
