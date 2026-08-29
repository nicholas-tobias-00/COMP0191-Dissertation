"""I-03b: interpretability recalibrated for the ACTUAL B-18 champion (D-106), closing the gap
I-03 (D-102) left behind. I-03 targeted the prior "TabPFN+species" TS-wrapper champion
(`rr.tabpfn_forecast`, per-tower-only, BASE+species). B-18 (2026-08-19) replaced that champion
with a structurally different architecture: direct tabular regression (TabPFNRegressor.fit/predict
on flat rows, pooled across all 3 towers via tower-dummy features, NOT the TS-forecaster wrapper)
plus a two-stage p95 spike-gate (`base + 0.25 * P(spike) * (spike_pred - normal_pred)`), on the
BASE_ALL_52 feature set (same forecast_daily_v3.csv fx_ columns I-03 already used, confirmed via
B17_foundation_screen.DATA_PATH == I-03's source). This is Phase 1 of the additive B18-integration
plan (2026-08-20): I-03b (this file) -> U-08 -> S-03b (driver-availability gate for the new
architecture, restricted to FX_A_SPECIES) -> U-05b/06b/07b -> S-03/S-06 replication.

Method: mirrors I-03/I-02's permutation-importance protocol as closely as the new architecture
allows -- same 5 anchors (2018-2022) x 3 towers, same seed convention (np.random.default_rng(yr)),
same importance definition (|mean(shuffled) - mean(base)|), same "permute fx_ columns only, not
static/dummy features" scope. What changed, forced by the architecture: B-18's model is fit ONCE
per anchor pooled across all 3 towers (not once per tower), so this fits the champion bundle
(classifier + base/normal/spike regressors) once per anchor and predicts from it multiple times
(baseline + one shuffle per feature) instead of refitting -- permutation importance is a
prediction-time-sensitivity measure, not a refit-per-feature measure, so this is the more correct
implementation of the same idea, not a methodology drift. Per-tower shuffles are still applied only
within that tower's own future rows (not mixed across towers), so per-tower importance numbers stay
directly interpretable the same way I-03's were, even though the underlying fit is now pooled.

Cost: 5 anchor-level fits (4 TabPFN .fit calls each = 20 fits) + 5 x (1 baseline + 52 feature
shuffles) x 3 towers' worth of predict-only calls -- predict-only, no refit, so much cheaper than
I-03's own 795 full-forecast calls despite covering the same 52-feature x 5-anchor x 3-tower grid.

Run from project root:  python notebooks/06_interpretability_uq/i03b_champion_interpretability_b18.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

ROOT = Path(r"C:\Users\Nicholas\Documents\COMP0191 MSc Artificial Intelligence for Sustainable Development Project")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notebooks" / "05_benchmarking"))

import B17_foundation_screen as b17s
import B17_direct_and_recursive_foundations as b17d
import B18_spike_models as b18sm

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

RESULTS = ROOT / "results"
OUT_RAW = RESULTS / "i03b_b18champion_importance.csv"
OUT_RANKED = RESULTS / "i03b_b18champion_importance_ranked.csv"
OUT_BY_TOWER = RESULTS / "i03b_b18champion_importance_by_tower.csv"

TOWERS = b17s.TOWERS
ANCHOR_YEARS = b17s.ANCHOR_YEARS
N_DAYS = b17s.N_DAYS
STATIC_FEATURES = b17d.TOWER_DUMMIES + b17d.TIME_FEATURES
SPIKE_PERCENTILE = 95
CORRECTION_WEIGHT = 0.25  # B18S04 base_plus_fixed_excess_0.25 -- the actual champion method


def fit_champion_bundle(train, fx_cols):
    """Fit once: classifier (p95 event gate) + base/normal/spike TabPFN regressors, pooled across
    all 3 towers. Mirrors B18_spike_models.fit_bundle's fit steps exactly, decomposed so
    predict_champion() can be called many times against the SAME fit (permutation importance
    perturbs inputs at predict time, it does not refit)."""
    features = fx_cols + STATIC_FEATURES
    tower_thresholds, global_threshold = b18sm.threshold_map(train, SPIKE_PERCENTILE)
    labels, _ = b18sm.event_labels(train, tower_thresholds, global_threshold)
    if labels.min() == labels.max():
        raise ValueError("Spike labels contain only one class")

    imputer = SimpleImputer(strategy="mean")
    x_train = imputer.fit_transform(train[features])

    classifier = b18sm.make_classifier({"foundation": "TabPFN", "balance_probabilities": False})
    classifier.fit(x_train, labels)

    base_model = b18sm.make_regressor("TabPFN")
    base_model.fit(x_train, train["y_observed"].to_numpy())

    normal = train.loc[labels == 0]
    spike = train.loc[labels == 1]
    if len(spike) < 10:
        raise ValueError(f"Only {len(spike)} spike rows")
    normal_model = b18sm.make_regressor("TabPFN")
    spike_model = b18sm.make_regressor("TabPFN")
    normal_model.fit(imputer.transform(normal[features]), normal["y_observed"].to_numpy())
    spike_model.fit(imputer.transform(spike[features]), spike["y_observed"].to_numpy())

    return {
        "features": features,
        "imputer": imputer,
        "classifier": classifier,
        "base": base_model,
        "normal": normal_model,
        "spike": spike_model,
        "tower_thresholds": tower_thresholds,
        "global_threshold": global_threshold,
    }


def predict_champion(bundle, rows):
    """Predict-only (no refit): base + 0.25 * P(spike) * max(spike_pred - normal_pred, 0) --
    B18S04's exact base_plus_fixed_excess_0.25 method."""
    x = bundle["imputer"].transform(rows[bundle["features"]])
    probability = np.asarray(bundle["classifier"].predict_proba(x))[:, 1]
    base_pred = b18sm.median_predict(bundle["base"], x)
    normal_pred = b18sm.median_predict(bundle["normal"], x)
    spike_pred = b18sm.median_predict(bundle["spike"], x)
    excess = np.maximum(spike_pred - normal_pred, 0)
    return base_pred + CORRECTION_WEIGHT * probability * excess


def run():
    frame = pd.read_csv(b17s.DATA_PATH, low_memory=False)
    frame["Datetime"] = pd.to_datetime(frame["Datetime"], format="mixed")
    frame = b17d.add_b17_features(frame)
    fx_cols = b17s.build_configs(frame)["BASE_ALL_52"]
    print(f"[I-03b] BASE_ALL_52: {len(fx_cols)} fx_ columns (B18's actual champion feature set)")

    rows = []
    t0 = time.time()
    for yr in ANCHOR_YEARS:
        anchor = pd.Timestamp(f"{yr}-12-16")
        dates = pd.date_range(anchor + pd.Timedelta(days=1), periods=N_DAYS, freq="D")
        train = frame.loc[frame["Datetime"].le(anchor) & frame["y_observed"].notna()].copy()
        future = frame.loc[frame["Datetime"].isin(dates)].copy()
        if len(train) > 10_000:
            train = train.sample(10_000, random_state=42).sort_values("Datetime")

        print(f"\n{'=' * 70}\nAnchor {yr} (n_train={len(train)})\n{'=' * 70}")
        t_anchor = time.time()
        try:
            bundle = fit_champion_bundle(train, fx_cols)
        except Exception as e:
            print(f"  ANCHOR {yr} FIT SKIPPED: {str(e)[:150]}")
            continue

        base_pred_all = predict_champion(bundle, future)
        future = future.assign(_base_pred=base_pred_all)

        for tower in TOWERS:
            t_tower = time.time()
            tower_mask = future["tower"].eq(tower)
            tower_future = future.loc[tower_mask]
            base_mean = tower_future["_base_pred"].mean()

            perm_rng = np.random.default_rng(yr)
            for col in fx_cols:
                shuffled = future.copy()
                idx = shuffled.index[tower_mask]
                shuffled.loc[idx, col] = perm_rng.permutation(shuffled.loc[idx, col].to_numpy())
                shuffled_pred = predict_champion(bundle, shuffled.loc[idx])
                importance = abs(shuffled_pred.mean() - base_mean)
                rows.append(
                    {
                        "anchor_year": yr,
                        "eval_tower": tower,
                        "model": "B18_TabPFN_champion",
                        "feature": col,
                        "importance": importance,
                    }
                )
            print(f"  Tower {tower} done ({time.time() - t_tower:.0f}s, {time.time() - t0:.0f}s elapsed)")
        print(f"  Anchor {yr} total: {time.time() - t_anchor:.0f}s")

    return pd.DataFrame(rows)


def main():
    print("=" * 70)
    print("I-03b: B18 champion (direct TabPFN + p95 spike-gate) permutation importance")
    print("=" * 70)
    importance_rows = run()
    if importance_rows.empty:
        print("[FAIL] No rows produced -- aborting.")
        return

    importance_rows.to_csv(OUT_RAW, index=False)
    print(f"\n[OK] Saved {OUT_RAW.name} ({len(importance_rows)} rows)")

    print("\nOverall ranking (mean importance across all towers/anchors):")
    overall = importance_rows.groupby("feature")["importance"].mean().sort_values(ascending=False)
    print(overall.round(4).to_string())
    overall.round(6).to_csv(OUT_RANKED, header=["mean_importance"])

    print("\nPer-tower ranking (mean importance across 5 anchors, top 10 each):")
    by_tower = importance_rows.groupby(["eval_tower", "feature"])["importance"].mean().reset_index()
    by_tower.to_csv(OUT_BY_TOWER, index=False)
    for t in TOWERS:
        sub = by_tower[by_tower.eval_tower == t].sort_values("importance", ascending=False).head(10)
        print(f"\n  Tower {t}:")
        print(sub[["feature", "importance"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
