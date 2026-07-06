"""I-02: feature-importance dispatch (native / SHAP / LIME) for the B-10/B-13 recursive-rollout
models. Fresh methodology -- deliberately NOT modeled on the old I-01 notebook (which targeted a
different, unrelated forecasting harness and is explicitly not used as precedent here).

Three importance families, each with its own honest scope:
- native_importance_*: whatever "for free" signal a model type already exposes (tree impurity/
  gain, SARIMAX exogenous coefficients, TFT's VSN gate weights, or -- for TabPFN, which has no
  native per-feature signal -- a permutation-importance substitute, explicitly flagged as such).
- shap_importance_*: shap.TreeExplainer for RF/XGB/LightGBM (fast, exact); shap.KernelExplainer
  (bounded background) for anything else, treated as a black box via a supplied predict_fn.
- lime_explain_instance: local, per-instance only (LIME's actual intended use) -- never aggregated
  into a global ranking, unlike native/SHAP.
"""
import numpy as np
import pandas as pd


# ---------------------------------------------------------------- native importance

def native_importance_tree(model, feat_cols):
    """RF/XGB/LightGBM: `.feature_importances_`, paired with feat_cols. Returns a Series sorted
    descending."""
    vals = np.asarray(model.feature_importances_, dtype=float)
    return pd.Series(vals, index=feat_cols).sort_values(ascending=False)


def native_importance_sarimax(sarimax_res, exog_cols):
    """SARIMAX: exogenous-regressor coefficient magnitude + p-value, restricted to the exog
    columns (excludes the model's own AR/MA/trend terms, which aren't "features" in the same
    sense). Returns a DataFrame with columns coef, abs_coef, pvalue, sorted by abs_coef
    descending."""
    params = sarimax_res.params
    pvalues = sarimax_res.pvalues
    rows = []
    for c in exog_cols:
        if c in params.index:
            rows.append({
                "feature": c,
                "coef": float(params[c]),
                "abs_coef": abs(float(params[c])),
                "pvalue": float(pvalues[c]) if c in pvalues.index else np.nan,
            })
    return pd.DataFrame(rows).set_index("feature").sort_values("abs_coef", ascending=False)


def native_importance_tft(enc_vsn_weights, dec_vsn_weights, enc_cols, dec_cols):
    """TFT: VSN gate weights (computed during forward(), stored on the model as
    last_enc_vsn_w/last_dec_vsn_w -- shape (B,T,n_vars) each) -- averaged over batch and time to
    give one weight per encoder/decoder feature. Returns a Series (encoder + decoder features
    combined, prefixed 'enc:'/'dec:' to disambiguate), sorted descending."""
    enc_w = enc_vsn_weights.mean(dim=(0, 1)).cpu().numpy()
    dec_w = dec_vsn_weights.mean(dim=(0, 1)).cpu().numpy()
    idx = [f"enc:{c}" for c in enc_cols] + [f"dec:{c}" for c in dec_cols]
    vals = np.concatenate([enc_w, dec_w])
    return pd.Series(vals, index=idx).sort_values(ascending=False)


def permutation_importance_generic(predict_fn, X, feat_cols, n_repeats=3, seed=0):
    """Model-agnostic permutation importance, for models with no native per-feature signal
    (TabPFN) or as a cross-check for any other model. Shuffles one feature column of X at a time
    and measures the mean absolute change in predict_fn(X)'s output vs. the unshuffled baseline.
    X: (n_rows, n_features) array matching feat_cols order. Returns a Series sorted descending."""
    rng = np.random.default_rng(seed)
    base_pred = np.asarray(predict_fn(X), dtype=float)
    scores = {}
    for j, col in enumerate(feat_cols):
        deltas = []
        for _ in range(n_repeats):
            Xp = X.copy()
            rng.shuffle(Xp[:, j])
            pred = np.asarray(predict_fn(Xp), dtype=float)
            deltas.append(float(np.mean(np.abs(pred - base_pred))))
        scores[col] = float(np.mean(deltas))
    return pd.Series(scores).sort_values(ascending=False)


def combine_ensemble_importance(importances_by_model, weights):
    """Weighted-average combination of constituent models' native/SHAP importances, using the SAME
    weights as the point-forecast ensemble itself (e.g. B-10's equal 0.25 each, or the frozen
    MASE-weighted set) -- not re-derived from this importance analysis, to avoid circularity.
    importances_by_model: dict {model_name: pd.Series indexed by feature}. Returns a combined
    Series (union of all features; a model missing a feature contributes 0 for it)."""
    all_feats = sorted(set().union(*[set(s.index) for s in importances_by_model.values()]))
    combined = pd.Series(0.0, index=all_feats)
    for name, s in importances_by_model.items():
        w = weights.get(name, 0.0)
        combined = combined.add(s.reindex(all_feats).fillna(0.0) * w, fill_value=0.0)
    return combined.sort_values(ascending=False)


# ---------------------------------------------------------------- SHAP

def shap_importance_tree(model, X_explain, feat_cols):
    """RF/XGB/LightGBM: shap.TreeExplainer -- fast, exact, no background sampling needed. Returns
    (shap_values array shape (n,F), mean|SHAP| Series sorted descending)."""
    import shap
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_explain)
    mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=feat_cols).sort_values(ascending=False)
    return sv, mean_abs


def shap_importance_blackbox(predict_fn, X_background, X_explain, feat_cols, nsamples="auto"):
    """SARIMAX/TFT/TabPFN/Ensemble: shap.KernelExplainer treating predict_fn as a black box.
    X_background is summarized (shap.kmeans) if larger than 50 rows -- KernelExplainer's cost
    scales with background size x nsamples, so both are kept bounded. Returns (shap_values array
    shape (n,F), mean|SHAP| Series sorted descending)."""
    import shap
    background = shap.kmeans(X_background, min(50, len(X_background))) if len(X_background) > 50 else X_background
    explainer = shap.KernelExplainer(predict_fn, background)
    sv = np.asarray(explainer.shap_values(X_explain, nsamples=nsamples))
    mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=feat_cols).sort_values(ascending=False)
    return sv, mean_abs


# ---------------------------------------------------------------- LIME (local, per-instance only)

def lime_explain_instance(predict_fn, X_background, instance_row, feat_cols, num_features=None,
                           mode="regression", seed=0):
    """LIME's actual intended use: a LOCAL explanation for ONE instance, not a global ranking
    (native_importance_*/shap_importance_* are for that). Returns a Series of per-feature local
    weights for this one instance, sorted by |weight| descending.
    predict_fn: X (n,F) -> y (n,) black-box callable (same contract as shap_importance_blackbox).
    X_background: (n_bg, F) array used to fit LIME's local perturbation distribution.
    instance_row: (F,) array, the single row being explained."""
    import lime.lime_tabular
    num_features = num_features or len(feat_cols)
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.asarray(X_background), feature_names=list(feat_cols),
        mode=mode, random_state=seed, discretize_continuous=False,
    )
    exp = explainer.explain_instance(np.asarray(instance_row), predict_fn, num_features=num_features)
    weights = dict(exp.as_list())
    # LIME's as_list() keys are string feature descriptions (e.g. "fx_lsu_dens <= 0.12"), not raw
    # feature names -- map back to the closest matching feat_cols entry for consistent indexing.
    out = {}
    for desc, w in weights.items():
        matched = next((c for c in feat_cols if c in desc), desc)
        out[matched] = w
    result = pd.Series(out).reindex(feat_cols).fillna(0.0)
    return result.reindex(result.abs().sort_values(ascending=False).index)
