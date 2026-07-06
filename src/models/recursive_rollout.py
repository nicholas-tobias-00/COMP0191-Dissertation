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


def dl_rollout_quantile(model, scaler_enc, scaler_dec, ch4_mu, ch4_sd, device, static_vec,
                         enc_ex_full, dec_ex_full, dates_full, history_init, anchor,
                         quantiles=(0.05, 0.5, 0.95), L=28, H=14, n_days=365):
    """Quantile analogue of `dl_rollout` (U-02) -- for a model like TFTQuantile whose forward()
    returns (B, H, Q) instead of (B, H). Same sliding-window mechanism as `dl_rollout` (identical
    docstring for enc_ex_full/dec_ex_full/dates_full/history_init/static_vec applies), but at each
    step takes `pred_scaled[0]` as a (Q,) array (day-1's quantiles, not a scalar), sorts it to
    enforce non-crossing (matching `predict_quantile`'s own convention), and feeds back ONLY the
    median (quantiles must include 0.5) into the shared `ch4` history -- the same single-coherent-
    history principle `tree_rollout_quantile` already established, for the same reason: three
    diverging quantile-specific histories would compound inconsistently, one shared median-anchored
    history does not. Returns a DataFrame indexed by date, one column per quantile level (float)
    plus 'median' (identical to the 0.5 column) -- same contract as `tree_rollout_quantile`."""
    if 0.5 not in quantiles:
        raise ValueError("0.5 (median) must be included in quantiles -- it is what's fed back into ch4 history")
    import torch
    median_idx = list(quantiles).index(0.5)
    ch4 = np.full(len(dates_full), np.nan, dtype=np.float32)
    anchor_idx = dates_full.get_loc(anchor)
    ch4[:anchor_idx + 1] = history_init
    static = np.asarray(static_vec, dtype=np.float32)
    rows = []
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
            pred_scaled = model(xe, xd, xs).cpu().numpy()[0]   # (H, Q)
            day1_q = np.sort(pred_scaled[0]) * ch4_sd + ch4_mu  # (Q,), non-crossing enforced
            rec = {q: float(day1_q[i]) for i, q in enumerate(quantiles)}
            rows.append(rec)
            ch4[t_idx] = day1_q[median_idx]
    dates = dates_full[anchor_idx + 1:anchor_idx + 1 + n_days]
    df = pd.DataFrame(rows, index=dates)
    df["median"] = df[0.5]
    return df


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


# ---------------------------------------------------------------- quantile rollout (U-02)
# Model adapters give tree_rollout_quantile a uniform `.predict_quantiles(Xi, quantiles)` interface
# despite RF/XGB/LightGBM having genuinely different native quantile mechanisms (quantile-forest
# trick vs separately-fit quantile-objective models) -- see U02_results.md for why each was chosen.

class RFQuantileAdapter:
    """Wraps an ALREADY-FITTED point-forecast RandomForestRegressor -- no retraining. Quantiles come
    from the empirical distribution of the forest's individual tree predictions at each row (the
    standard "quantile regression forest" trick), reusing the exact same fitted model B-10 already
    validated as its point forecaster."""
    def __init__(self, rf_model):
        self.rf = rf_model

    def predict_quantiles(self, Xi, quantiles):
        tree_preds = np.array([est.predict(Xi) for est in self.rf.estimators_])  # (n_trees, n_rows)
        return {q: np.quantile(tree_preds, q, axis=0) for q in quantiles}


class MultiModelQuantileAdapter:
    """Wraps N separately-fit quantile-objective models (e.g. XGBRegressor(objective=
    'reg:quantileerror', quantile_alpha=q) or LGBMRegressor(objective='quantile', alpha=q)), one
    fitted model per requested quantile level."""
    def __init__(self, models_by_quantile):
        self.models_by_quantile = models_by_quantile  # {quantile_level: fitted_model}

    def predict_quantiles(self, Xi, quantiles):
        return {q: self.models_by_quantile[q].predict(Xi) for q in quantiles}


def tree_rollout_quantile(adapter, imp, feat_cols, fx_frame, history_init, anchor, n_days=365,
                           quantiles=(0.05, 0.5, 0.95)):
    """Quantile analogue of `tree_rollout`. `adapter` exposes `.predict_quantiles(Xi, quantiles) ->
    {q: array_of_len_1}` (see RFQuantileAdapter/MultiModelQuantileAdapter above). At each recursive
    step, computes every requested quantile but feeds ONLY the median (0.5, required to be present
    in `quantiles`) back into the shared AR history -- keeps one coherent recursive state rather than
    letting each quantile diverge into its own history (three separately-fed-back chains would be
    internally inconsistent: the 0.05 chain would keep compounding an artificially-low history against
    itself, biasing its own future predictions further down every step, not just reflecting genuine
    lower-tail uncertainty). Returns a DataFrame indexed by date, one column per quantile level
    (float) plus a 'median' column (identical to the 0.5 column, provided for readability)."""
    if 0.5 not in quantiles:
        raise ValueError("0.5 (median) must be included in quantiles -- it is what's fed back into AR history")
    history = history_init.copy()
    dates = pd.date_range(anchor + pd.Timedelta(days=1), periods=n_days, freq="D")
    rows = []
    for d in dates:
        row = ar_features_for_day(history, d)
        for c in fx_frame.columns:
            row[c] = fx_frame.loc[d, c]
        X = pd.DataFrame([row])[feat_cols]
        Xi = imp.transform(X.values)
        qpreds = adapter.predict_quantiles(Xi, quantiles)
        rec = {q: float(qpreds[q][0]) for q in quantiles}
        rows.append(rec)
        history.loc[d] = rec[0.5]
    df = pd.DataFrame(rows, index=dates)
    df["median"] = df[0.5]
    return df


def sarimax_quantile(sarimax_res, target_dates, exog, alpha=0.10):
    """Thin wrapper around SARIMAX's own predictive distribution -- `get_forecast().conf_int()`
    already gives a (1-alpha) interval essentially for free from the fitted state-space model, no
    new fitting cost beyond the point-forecast fit every other B-09-B-15 SARIMAX call already pays.
    Returns a DataFrame indexed by target_dates with columns 'median', alpha/2, 1-alpha/2 (e.g.
    0.05/0.95 for the default alpha=0.10, a 90% interval)."""
    fc = sarimax_res.get_forecast(steps=len(target_dates), exog=exog)
    ci = fc.conf_int(alpha=alpha)
    lo_col, hi_col = ci.columns[0], ci.columns[1]
    out = pd.DataFrame(index=target_dates)
    out["median"] = fc.predicted_mean.values
    out[round(alpha / 2, 4)] = ci[lo_col].values
    out[round(1 - alpha / 2, 4)] = ci[hi_col].values
    return out


def conformal_margins_by_bin(residuals_by_bin, alpha=0.10):
    """Standard split-conformal finite-sample-corrected quantile of absolute residuals, computed
    separately per lead-time bin (necessary because rollout error is heteroscedastic by lead time,
    confirmed throughout B-09-B-15 -- a single global margin would under-cover the far end of the
    chain and over-cover the near end). `residuals_by_bin`: dict {bin_label: array of abs residuals
    pooled from calibration anchors} (e.g. leave-one-anchor-out: every anchor except the one being
    tested). Returns {bin_label: margin}; the calibrated interval for a point forecast `p` in that bin
    is [p - margin, p + margin]. Uses the standard finite-sample conformal correction
    level = min(1, ceil((n+1)*(1-alpha))/n) (Lei et al. 2018 / Romano et al. 2019's well-known
    split-conformal quantile adjustment, not specific to any prior work in this repo)."""
    margins = {}
    for b, res in residuals_by_bin.items():
        res = np.asarray(res, dtype=float)
        res = res[np.isfinite(res)]
        n = len(res)
        if n == 0:
            margins[b] = np.nan
            continue
        level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        margins[b] = float(np.quantile(res, level))
    return margins


def lead_time_bin(dates, anchor, bins=((1, 7), (8, 30), (31, 90), (91, 180), (181, 270), (271, 365))):
    """Maps each date to its lead-time bin label (matching bin_metrics's bins exactly), or None if
    outside every bin. Used to group residuals for conformal_margins_by_bin."""
    lead = np.array([(d - anchor).days for d in dates])
    labels = np.full(len(dates), None, dtype=object)
    for lo, hi in bins:
        m = (lead >= lo) & (lead <= hi)
        labels[m] = f"{lo}-{hi}"
    return labels


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


# ---------------------------------------------------------------- TabPFN one-shot forecast (B-13b)
# NOT a rollout in the iterative sense -- tabpfn-time-series predicts the entire horizon in a
# single forward pass from a context+future-covariates dataframe (architecturally closer to
# SARIMAX's one-shot get_forecast(steps=H) than to tree_rollout/dl_rollout's day-by-day loop).
# Per-tower only -- the simple predict_df API has no static-covariate/pooling support.

def tabpfn_forecast(hist_target, hist_covariates, future_covariates, mode="local", quantiles=None):
    """hist_target: pandas Series, real y_observed history (gaps allowed as NaN -- TabPFN handles
    missing context values internally, so this deliberately does NOT use y_gapfilled, avoiding the
    diffuse globally-trained-gap-filler optimism flagged for every other model's training target).
    hist_covariates/future_covariates: DataFrames indexed by date, same columns in both (only
    covariates present in both are used) -- real historical / perfect-foresight future fx_ drivers.
    mode: "local" (default, user-confirmed -- requires TABPFN_TOKEN env var set once via
    https://ux.priorlabs.ai) or "client" (cloud; sends data to Prior Labs per call).
    quantiles (U-02): optional list of quantile levels (e.g. [0.05, 0.5, 0.95]) -- tabpfn-time-series's
    own `predict_df(..., quantiles=[...])` already supports this natively (confirmed via its
    signature), returning one column per level named as the quantile's string (e.g. '0.05'). Default
    (None) preserves the exact original point-only behavior (a single Series of the median forecast).
    When quantiles is given, returns a DataFrame instead (columns = quantile levels as floats, plus
    'median'), indexed by future_covariates.index."""
    import tabpfn_time_series as tts

    pipeline = tts.TabPFNTSPipeline(
        tabpfn_mode=tts.TabPFNMode.LOCAL if mode == "local" else tts.TabPFNMode.CLIENT
    )

    context_df = hist_covariates.copy()
    context_df["timestamp"] = context_df.index
    context_df["target"] = hist_target.reindex(context_df.index).values
    context_df = context_df.reset_index(drop=True)

    future_df = future_covariates.copy()
    future_df["timestamp"] = future_df.index
    future_df = future_df.reset_index(drop=True)

    if quantiles is None:
        preds = pipeline.predict_df(context_df, future_df=future_df)
        preds = preds.reset_index()
        return pd.Series(preds["target"].values, index=pd.to_datetime(preds["timestamp"]))

    preds = pipeline.predict_df(context_df, future_df=future_df, quantiles=list(quantiles))
    preds = preds.reset_index()
    idx = pd.to_datetime(preds["timestamp"])
    out = pd.DataFrame(index=idx)
    out["median"] = preds["target"].values
    # predict_df's returned columns are the float quantile VALUES themselves (e.g. 0.05), not their
    # string representation -- confirmed empirically (a naive str(q)-based lookup silently returned
    # all-NaN, since '0.05' was never a column name; 0.05 the float was).
    for q in quantiles:
        out[q] = preds[q].values if q in preds.columns else np.nan
    return out
