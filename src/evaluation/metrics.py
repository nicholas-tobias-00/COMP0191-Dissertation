"""Shared forecasting/gap-filling metrics, used across all benchmarking notebooks (D-44).

FCH4 is a signed flux that crosses zero (uptake periods), which makes MAPE mathematically
unstable (division by near-zero/zero actuals) -- documented here rather than silently
computed. WAPE and MASE are the scale-free metrics recommended for this data; sMAPE/MAPE
are provided for completeness/comparability but should be read with that caveat in mind.
"""
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


def mae(y, p):
    return float(mean_absolute_error(y, p))


def r2(y, p):
    y = np.asarray(y, float)
    return float(r2_score(y, p)) if (len(y) > 1 and np.var(y) > 0) else np.nan


def mbe(y, p):
    return float(np.mean(np.asarray(p, float) - np.asarray(y, float)))


def wape(y, p):
    """Weighted Absolute Percentage Error = sum|y-p| / sum|y|.
    Aggregates before dividing, so unlike MAPE a few near-zero actuals can't blow it up.
    Returns NaN if sum|y| == 0 (degenerate test set)."""
    y = np.asarray(y, float); p = np.asarray(p, float)
    denom = np.sum(np.abs(y))
    return float(np.sum(np.abs(y - p)) / denom) if denom > 0 else np.nan


def mase(y, p, y_naive):
    """MASE, test-set relative-MAE form: MAE(model) / MAE(persistence forecast).
    Scaled against the same out-of-sample persistence baseline already used for
    skill_persist (D-37), rather than Hyndman-Koehler's in-sample naive -- keeps this
    consistent with the project's existing baseline convention. <1 = beats persistence,
    1 = ties it, >1 = worse than persistence. Returns NaN if the persistence MAE is 0."""
    denom = mae(y, y_naive)
    return float(mae(y, p) / denom) if denom > 0 else np.nan


def smape(y, p, eps=1e-6):
    """Symmetric MAPE. CAVEAT: still unstable when |y|+|p| ~ 0 (quiet flux periods,
    common for FCH4) -- eps floors the denominator rather than producing inf/nan, but
    treat sMAPE on this data as indicative, not precise."""
    y = np.asarray(y, float); p = np.asarray(p, float)
    denom = np.abs(y) + np.abs(p)
    return float(np.mean(2 * np.abs(y - p) / np.maximum(denom, eps)))


def mape(y, p, min_abs_y=1.0):
    """Mean Absolute Percentage Error, computed only over |y| >= min_abs_y (default 1
    nmol m-2 s-1). CAVEAT: FCH4 is signed and crosses zero, so MAPE is fundamentally
    unstable on this data even with filtering -- WAPE is the recommended alternative.
    Returns (mape, n_excluded) -- n_excluded = rows dropped for |y| < min_abs_y."""
    y = np.asarray(y, float); p = np.asarray(p, float)
    mask = np.abs(y) >= min_abs_y
    n_excluded = int((~mask).sum())
    val = float(np.mean(np.abs((y[mask] - p[mask]) / y[mask]))) if mask.sum() > 0 else np.nan
    return val, n_excluded


def full_metrics(y, p, y_naive=None):
    """All metrics in one call, for direct use in notebook emit()/metrics() helpers.
    y_naive (persistence-forecast array on the same test rows) is required for MASE;
    omit to get NaN for that key only."""
    y = np.asarray(y, float); p = np.asarray(p, float)
    mp, mp_n = mape(y, p)
    out = dict(
        RMSE=rmse(y, p), MAE=mae(y, p), R2=r2(y, p), MBE=mbe(y, p),
        WAPE=wape(y, p), sMAPE=smape(y, p), MAPE=mp, MAPE_n_excluded=mp_n,
        MASE=mase(y, p, y_naive) if y_naive is not None else np.nan,
    )
    return out
