"""S-01: level-residual hybrid model for scenario-conditional prediction.

Design (per the S-01 plan, informed by this session's deep-research pass): an explicit, stable
PARAMETRIC trend model (Ridge, fit once on the full real historical record -- not SARIMAX, which
U-02/U-03 found unstable across per-anchor refits, 58-380% overshoot range) carries the two
extrapolation-risk axes (temperature, livestock density) into the scenario region. A tree ensemble
(RF/XGB/LightGBM, B-10's exact hyperparameters) then corrects only the RESIDUAL around that trend --
residuals stay closer to stationary/in-range, which is the whole point: it sidesteps the
tree-extrapolation-ceiling U-03 found for the raw target. XGB/LightGBM get a monotonic constraint
forcing the residual non-decreasing in livestock density (I-02's own finding: FCH4 rises with
stocking density) as a cheap guardrail against spurious reversals beyond the training range.

Also provides a lightweight, from-scratch Python `dissimilarity_index` in the spirit of Meyer &
Pebesma (2021)'s Area of Applicability -- NOT a port of the R `CAST` package, just the same core
idea (nearest-neighbour distance to training data in scaled feature space, thresholded via a robust
outlier rule on the training data's own leave-one-out distances) implemented directly in Python.
"""
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

TREND_FEATURES = ["fx_TA_mean", "fx_lsu_dens", "fx_DOY_sin", "fx_DOY_cos"]
DUM = ["is_t2", "is_t4", "is_t9"]
AR_COLS = ["ar_ch4_dlag1", "ar_ch4_dlag2", "ar_ch4_dlag3", "ar_ch4_dlag7", "ar_ch4_dlag14", "ar_ch4_drm7"]
DROPPED_FX = ("fx_USTAR_mean", "fx_SHF_mean")


def s01_feat_cols(dv_columns):
    """AR_COLS + every fx_ column EXCEPT fx_USTAR_mean/fx_SHF_mean + ar_fc_dlag1 + tower dummies --
    identical to B-10/U-02/U-03's feat_cols construction, minus the two dropped columns."""
    fx_cols = [c for c in dv_columns if c.startswith("fx") and c not in DROPPED_FX]
    return AR_COLS + fx_cols + ["ar_fc_dlag1"] + DUM


def fit_trend_model(pool_df):
    """Ridge trend model, fit ONCE on the full pooled real historical record (all 3 towers, tower
    dummies included) -- deliberately the ONLY fit, unlike SARIMAX's per-anchor refits which U-03
    found unstable. Feature set deliberately narrow (TREND_FEATURES + DUM): the trend's only job is
    to extrapolate sensibly along temperature/livestock density, not to explain day-to-day or
    soil/wind variation (that's the residual tree model's job)."""
    cols = TREND_FEATURES + DUM
    X = pool_df[cols].values
    y = pool_df["y_gapfilled"].values
    model = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
    model.fit(X, y)
    return model, cols


def trend_predict(trend_model, trend_cols, df):
    return trend_model.predict(df[trend_cols].values)


def build_monotone_tuple(feat_cols, positive_cols=("fx_lsu_dens",)):
    """1 at each column in `positive_cols` (residual forced non-decreasing in that feature), 0
    elsewhere. Returned as a tuple (XGBoost) -- LightGBM takes the same values as a plain list."""
    return tuple(1 if c in positive_cols else 0 for c in feat_cols)


def fit_residual_trees(pool_df, trend_model, trend_cols, feat_cols):
    """Fits RF/XGB/LightGBM on the RESIDUAL (y_gapfilled - trend_prediction), using B-10's exact
    hyperparameters (D-41/D-54, no new HPO) plus a monotonic constraint on fx_lsu_dens for XGB/
    LightGBM (RF has no native monotonic-constraint support in scikit-learn -- a known asymmetry,
    not a bug). Returns (imputer, {model_name: fitted_model})."""
    resid = pool_df["y_gapfilled"].values - trend_predict(trend_model, trend_cols, pool_df)

    imp = SimpleImputer(strategy="mean")
    Xi = imp.fit_transform(pool_df[feat_cols].values)

    mono = build_monotone_tuple(feat_cols)

    rf = RandomForestRegressor(n_estimators=500, max_features=0.5, min_samples_leaf=10,
                                n_jobs=-1, random_state=42)
    rf.fit(Xi, resid)

    xgb = XGBRegressor(n_estimators=400, max_depth=2, learning_rate=0.02, min_child_weight=10,
                        subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42,
                        monotone_constraints=mono)
    xgb.fit(Xi, resid)

    lgb = LGBMRegressor(n_estimators=400, num_leaves=7, min_child_samples=10, learning_rate=0.02,
                         subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42, verbosity=-1,
                         monotone_constraints=list(mono))
    lgb.fit(Xi, resid)

    return imp, {"RF": rf, "XGB": xgb, "LightGBM": lgb}


def predict_scenario(trend_model, trend_cols, imp, tree_models, feat_cols, scenario_df):
    """Combined trend + mean-of-tree-residual prediction for a scenario frame. Returns
    (combined_pred, trend_pred, {model_name: residual_pred})."""
    trend_pred = trend_predict(trend_model, trend_cols, scenario_df)
    Xi = imp.transform(scenario_df[feat_cols].values)
    resid_preds = {name: m.predict(Xi) for name, m in tree_models.items()}
    mean_resid = np.mean(list(resid_preds.values()), axis=0)
    return trend_pred + mean_resid, trend_pred, resid_preds


def dissimilarity_index(X_train, X_scenario):
    """Lightweight from-scratch Python implementation in the spirit of Meyer & Pebesma (2021)'s
    Area of Applicability -- normalized nearest-neighbour distance (in scaled feature space) from
    each scenario row to the real training data. Threshold derived from the training data's OWN
    leave-one-out nearest-neighbour distances via a standard Tukey IQR fence (Q3 + 1.5*IQR) -- a
    robust-outlier convention, not an arbitrary constant. Returns (d_scenario, threshold, flagged)."""
    scaler = StandardScaler().fit(X_train)
    Xtr = scaler.transform(X_train)
    Xsc = scaler.transform(X_scenario)

    d_scenario = cdist(Xsc, Xtr).min(axis=1)

    d_train_matrix = cdist(Xtr, Xtr)
    np.fill_diagonal(d_train_matrix, np.inf)
    d_train_loo = d_train_matrix.min(axis=1)
    q1, q3 = np.percentile(d_train_loo, [25, 75])
    threshold = q3 + 1.5 * (q3 - q1)

    flagged = d_scenario > threshold
    return d_scenario, threshold, flagged
