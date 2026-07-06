"""B-09: shared helpers for the recursive 365-day daily rollout backtest.

Tests whether autoregressive (feed-your-own-prediction-back-in) forecasting stays usable
over a full-year horizon, or compounds/degrades -- a direct precursor question for the
long-range (D-46/D-52) scenario work, and directly informed by FORECASTING_LEARNINGS.md's
"always validate a rollout mechanism against a real held-out window" lesson.

Single fixed anchor date, one continuous 365-day recursive chain per model -- NOT a
walk-forward evaluation across many re-originating origins (that never lets error compound
past a few days, since it re-observes real ground truth each stride). Every model must be
fit fresh on data strictly <= the anchor date; reusing B03/B03a/B03b/B04's existing results
would leak the anchor-adjacent real observations into "training".

Perfect-foresight exogenous drivers throughout (real historical fx_ values) -- this isolates
the recursive-mechanism question from driver-forecast realism (B-08/D-47's separate, still-
queued scope).
"""
import numpy as np
import pandas as pd

AR_LAGS = [1, 2, 3, 7, 14]


# ---------------------------------------------------------------- baselines

def chain_persistence(anchor_value, n_days):
    """Repeat the anchor day's real value for every day of the chain."""
    return np.full(n_days, float(anchor_value))


def doy_climatology(history_series, target_dates, window=7):
    """Historical mean by day-of-year with a +/- `window`-day circular window, computed from
    `history_series` (real y_observed strictly before the anchor). Finer than the project's
    existing monthly climatology() helpers (B03/forecasting_dl.py) -- appropriate for a
    day-level chain rather than a per-horizon CV comparison."""
    doy = np.asarray(history_series.index.dayofyear)
    vals = history_series.values.astype(float)
    global_mean = np.nanmean(vals)
    preds = []
    for d in target_dates:
        td = d.dayofyear
        circ = np.abs(((doy - td + 182) % 365) - 182)
        mask = circ <= window
        v = vals[mask]
        preds.append(np.nanmean(v) if mask.any() and np.isfinite(np.nanmean(v)) else global_mean)
    return np.array(preds)


# ---------------------------------------------------------------- tree-model rollout
# Mirrors build_forecasting_matrix_v2.py's daily_table() shift/rolling math exactly
# (ar_ch4_dlag{1,2,3,7,14} = yg.shift(L); ar_ch4_drm7 = yg.shift(1).rolling(7,min_periods=1).mean())
# -- only these 6 columns are recursion-dependent; everything else (fx_*, ar_fc_dlag1) is real,
# perfect-foresight history read directly from forecast_daily_v2.csv.

def ar_features_for_day(history, day):
    """history: pandas Series indexed by date, containing real+predicted daily CH4 up to (not
    including) `day`. Returns the 6 ar_ch4_* feature values for predicting `day`."""
    out = {}
    for L in AR_LAGS:
        prior = day - pd.Timedelta(days=L)
        out[f"ar_ch4_dlag{L}"] = history.get(prior, np.nan)
    window_start = day - pd.Timedelta(days=7)
    window_end = day - pd.Timedelta(days=1)
    wsub = history.loc[window_start:window_end]
    out["ar_ch4_drm7"] = wsub.mean() if len(wsub) else np.nan
    return out


def tree_rollout(model, imp, feat_cols, fx_frame, history_init, anchor, n_days=365,
                  alpha=1.0, clim_series=None):
    """model/imp: fitted (RandomForestRegressor|XGBRegressor|LGBMRegressor, SimpleImputer) pair.
    feat_cols: full ordered feature list the model was trained on (ar_ch4_* + fx_*/ar_fc_dlag1 + DUM).
    fx_frame: DataFrame indexed by date, real historical values for every non-AR feature in
        feat_cols (fx_*, ar_fc_dlag1, is_t2/is_t4/is_t9 dummies), covering the full 365-day
        target window (perfect foresight).
    history_init: pandas Series of real y_gapfilled up to and including `anchor`.
    alpha/clim_series (B-10, D-54): optional blended-AR memory. Default alpha=1.0,
        clim_series=None reproduces B-09's exact original behaviour bit-for-bit (pure recursive
        memory) -- existing callers are unaffected. When clim_series (a pandas Series of
        day-of-year climatology values, e.g. from doy_climatology, aligned to the target dates)
        is given, what gets FED BACK INTO MEMORY for day d is
        `alpha*pred + (1-alpha)*clim_series[d]` -- alpha=1 is pure-recursive, alpha=0 is
        pure-climatological-anchor, alpha in (0,1) blends the two. Note this only changes what
        future days' AR features read from `history` -- the reported/evaluated prediction for
        day d itself is always the model's own raw `pred`, never the blended value.
    Returns a pandas Series of the 365-day recursive prediction chain (raw model predictions)."""
    history = history_init.copy()
    dates = pd.date_range(anchor + pd.Timedelta(days=1), periods=n_days, freq="D")
    preds = []
    for d in dates:
        row = ar_features_for_day(history, d)
        for c in fx_frame.columns:
            row[c] = fx_frame.loc[d, c]
        X = pd.DataFrame([row])[feat_cols]
        Xi = imp.transform(X.values)
        pred = float(model.predict(Xi)[0])
        preds.append(pred)
        if alpha >= 1.0 or clim_series is None:
            history.loc[d] = pred
        else:
            history.loc[d] = alpha * pred + (1 - alpha) * float(clim_series.loc[d])
    return pd.Series(preds, index=dates)


# ---------------------------------------------------------------- DL-model rollout
# Slides a fixed L-day encoder window forward one day at a time; encoder feature 0 (CH4) is
# substituted with the growing real+predicted series, every other encoder/decoder feature is
# real historical fx_/flux_t data. Reuses the model's *trained* H=14 weights and static scaler
# verbatim -- only `pred[:, 0]` (the first predicted day) is kept and fed back in each step.

def dl_rollout(model, scaler_enc, scaler_dec, ch4_mu, ch4_sd, device, static_vec,
               enc_ex_full, dec_ex_full, dates_full, history_init, anchor, L=28, H=14, n_days=365):
    """enc_ex_full: (N, F_enc-1) real fx_+flux_t values aligned with `dates_full` (feature 0 of
        the encoder, CH4, is NOT included here -- it's substituted at each step below).
    dec_ex_full: (N, F_dec) real fx_ values aligned with `dates_full`, must extend at least H
        days past the last target day (dates_full must cover anchor..anchor+n_days+H).
    dates_full: DatetimeIndex, contiguous daily, covering pre-anchor history through the target
        window + an H-day buffer.
    history_init: array of real gap-filled CH4 values for the pre-anchor portion of dates_full.
    static_vec: the tower's one-hot static vector (forecasting_dl.TOW[tower]).
    """
    import torch
    ch4 = np.full(len(dates_full), np.nan, dtype=np.float32)
    anchor_idx = dates_full.get_loc(anchor)
    ch4[:anchor_idx + 1] = history_init
    static = np.asarray(static_vec, dtype=np.float32)
    preds = []
    model.eval()
    with torch.no_grad():
        for step in range(n_days):
            t_idx = anchor_idx + 1 + step
            enc_ch4 = ch4[t_idx - L:t_idx]
            enc_feat = np.concatenate([enc_ch4[:, None], enc_ex_full[t_idx - L:t_idx]], axis=1)
            enc_feat = np.nan_to_num(enc_feat, nan=0.0)
            dec_feat = np.nan_to_num(dec_ex_full[t_idx:t_idx + H], nan=0.0)
            enc_s = scaler_enc.tf(enc_feat[None, ...])
            dec_s = scaler_dec.tf(dec_feat[None, ...])
            xe = torch.tensor(enc_s, dtype=torch.float32).to(device)
            xd = torch.tensor(dec_s, dtype=torch.float32).to(device)
            xs = torch.tensor(static[None, :], dtype=torch.float32).to(device)
            pred_scaled = model(xe, xd, xs).cpu().numpy()[0]
            pred_day1 = float(pred_scaled[0]) * ch4_sd + ch4_mu
            preds.append(pred_day1)
            ch4[t_idx] = pred_day1
    return pd.Series(preds, index=dates_full[anchor_idx + 1:anchor_idx + 1 + n_days])


# ---------------------------------------------------------------- monthly rollout (B-11, D-55)
# Coarser companion to the daily tree_rollout above -- dampens any single missed spike-day's
# influence on evaluation (the M5-hierarchy lesson: coarser aggregates score better). Uses
# build_forecasting_matrix_monthly.py's forecast_monthly_v2.csv as its only feature source.

AR_LAGS_MONTHLY = [1, 2, 3]


def ar_features_for_month(history, month):
    """history: pandas Series indexed by month-start Timestamp, real+predicted monthly CH4 up to
    (not including) `month`. Returns the 3 ar_ch4_mlag* feature values for predicting `month`."""
    out = {}
    for L in AR_LAGS_MONTHLY:
        prior = month - pd.DateOffset(months=L)
        out[f"ar_ch4_mlag{L}"] = history.get(prior, np.nan)
    return out


def moy_climatology(history_series, target_months, window=1):
    """Month-of-year analogue of doy_climatology -- circular +/- `window`-month window over
    calendar month instead of day-of-year."""
    moy = np.asarray(history_series.index.month)
    vals = history_series.values.astype(float)
    global_mean = np.nanmean(vals)
    preds = []
    for d in target_months:
        tm = d.month
        circ = np.abs(((moy - tm + 6) % 12) - 6)
        mask = circ <= window
        v = vals[mask]
        preds.append(np.nanmean(v) if mask.any() and np.isfinite(np.nanmean(v)) else global_mean)
    return np.array(preds)


def monthly_rollout(model, imp, feat_cols, fx_frame, history_init, anchor_month, n_months=13,
                     alpha=1.0, clim_series=None):
    """Month-step analogue of tree_rollout -- same alpha/clim_series blended-memory contract.
    anchor_month/fx_frame/history_init must all use month-start (MS) Timestamps."""
    history = history_init.copy()
    dates = pd.date_range(anchor_month + pd.DateOffset(months=1), periods=n_months, freq="MS")
    preds = []
    for d in dates:
        row = ar_features_for_month(history, d)
        for c in fx_frame.columns:
            row[c] = fx_frame.loc[d, c]
        X = pd.DataFrame([row])[feat_cols]
        Xi = imp.transform(X.values)
        pred = float(model.predict(Xi)[0])
        preds.append(pred)
        if alpha >= 1.0 or clim_series is None:
            history.loc[d] = pred
        else:
            history.loc[d] = alpha * pred + (1 - alpha) * float(clim_series.loc[d])
    return pd.Series(preds, index=dates)


def downscale_monthly_to_daily(monthly_pred_series, daily_template_series):
    """Hybrid-calibration downscaling (B-11): reuses `daily_template_series` (e.g. a B-09/B-10
    daily chain for the same anchor/tower) as the within-month *shape*, recentered so its own
    monthly mean matches `monthly_pred_series`'s independently-derived prediction for that month:
    `daily_synth[d] = daily_template[d] - mean(daily_template over month) + monthly_pred[month]`.
    NOT independent of the daily template's own shape errors -- state this caveat in any write-up.
    monthly_pred_series must be indexed by month-start (MS) Timestamps covering every month
    touched by daily_template_series's index."""
    months = daily_template_series.index.to_period("M")
    month_means = daily_template_series.groupby(months).transform("mean")
    monthly_aligned = pd.Series(
        [monthly_pred_series.loc[pd.Timestamp(p.start_time)] for p in months],
        index=daily_template_series.index,
    )
    return daily_template_series - month_means + monthly_aligned


# ---------------------------------------------------------------- evaluation

def bin_metrics(y_true, y_pred, dates, anchor, y_persist=None,
                 bins=((1, 7), (8, 30), (31, 90), (91, 180), (181, 270), (271, 365))):
    """y_true/y_pred/y_persist: arrays aligned with `dates` (may contain NaN in y_true for
    ungapfilled days -- those rows are dropped per bin). y_persist (the chain-persistence
    baseline, same length) is used as MASE's y_naive if given. Returns a DataFrame, one row
    per lead-time bin -- the direct M5-lesson analogue of "don't blend across the hierarchy",
    here binning by lead-time-within-the-chain instead of store/category."""
    from sklearn.metrics import r2_score, mean_absolute_error
    from evaluation.metrics import mase as mase_fn
    lead = np.array([(d - anchor).days for d in dates])
    rows = []
    for lo, hi in bins:
        m = (lead >= lo) & (lead <= hi) & np.isfinite(y_true)
        if m.sum() < 3:
            rows.append(dict(bin=f"{lo}-{hi}", n=int(m.sum()), R2=np.nan, MAE=np.nan, MASE=np.nan))
            continue
        yt, yp = y_true[m], y_pred[m]
        r2 = r2_score(yt, yp) if np.var(yt) > 0 else np.nan
        mae_v = mean_absolute_error(yt, yp)
        mase_v = mase_fn(yt, yp, y_persist[m]) if y_persist is not None else np.nan
        rows.append(dict(bin=f"{lo}-{hi}", n=int(m.sum()), R2=round(r2, 3) if np.isfinite(r2) else np.nan,
                          MAE=round(float(mae_v), 3),
                          MASE=round(float(mase_v), 4) if np.isfinite(mase_v) else np.nan))
    return pd.DataFrame(rows)
