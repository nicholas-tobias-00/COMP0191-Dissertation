"""U-08: UQ recalibrated for the ACTUAL B-18 champion (D-106), closing the gap U-04 (D-88) left
behind -- same "predates the champion" framing U-04 itself used against U-02. Phase 2 of the
additive B18-integration plan (2026-08-20): I-03b -> U-08 (this file) -> S-03b (driver-availability
gate, restricted to FX_A_SPECIES) -> U-05b/06b/07b -> S-03/S-06 replication.

U-04 built its band from TabPFN/TabICLv2's own native TS-wrapper quantile output
(`rr.tabpfn_forecast(..., quantiles=...)`). B-18's champion has no equivalent native-quantile call
-- it is a point-estimate architecture (classifier-gated base+excess correction, direct tabular
regression). This constructs the closest faithful analogue: the BASE regressor's own native
quantile spread (`TabPFNRegressor.predict(x, output_type="quantiles", quantiles=[0.05,0.5,0.95])`,
confirmed via the installed tabpfn package's own signature/docstring) gives the raw band shape, then
the SAME deterministic spike-excess correction the champion's point forecast uses
(`0.25 * P(spike) * max(spike_pred - normal_pred, 0)`) is added as a uniform shift to all three
quantile levels -- so the calibrated median here is bit-for-bit the champion's actual point
forecast, and q05/q95 preserve the base regressor's own uncertainty shape around it. Leave-one-
anchor-out conformal calibration (`evaluate_stage`, imported UNCHANGED from
`u02_multi_anchor_tower.py`, same as U-04's own precedent) is what actually fixes calibration
regardless of how the raw band was built -- consistent with U-02's own finding that calibration
converges models to ~0.88-0.90 PICP "regardless of raw coverage".

Same fit-once-per-anchor / predict-many pooled-across-towers architecture as I-03b (B-18's model is
fit once per anchor across all 3 towers via tower-dummy features, not once per tower).

Run from project root:  python notebooks/06_interpretability_uq/u08_champion_uq_b18.py
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
sys.path.insert(0, str(ROOT / "notebooks" / "06_interpretability_uq"))

import B17_foundation_screen as b17s
import B17_direct_and_recursive_foundations as b17d
import B18_spike_models as b18sm
from u02_multi_anchor_tower import evaluate_stage, QUANTILES

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

RESULTS = ROOT / "results"
CHAINS_PATH = RESULTS / "u08_chains.csv"
SUMMARY_PATH = RESULTS / "u08_summary.csv"

TOWERS = b17s.TOWERS
ANCHOR_YEARS = b17s.ANCHOR_YEARS
N_DAYS = b17s.N_DAYS
STATIC_FEATURES = b17d.TOWER_DUMMIES + b17d.TIME_FEATURES
SPIKE_PERCENTILE = 95
CORRECTION_WEIGHT = 0.25


def fit_champion_bundle(train, fx_cols):
    """Identical to i03b_champion_interpretability_b18.fit_champion_bundle -- not imported cross-
    directory to keep this file runnable standalone; kept byte-for-byte equivalent on purpose."""
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
    }


def predict_champion_quantiles(bundle, rows, quantiles=QUANTILES):
    """Base regressor's native quantile spread, shifted by the champion's own deterministic
    spike-excess correction (uniform shift -- the correction term does not depend on quantile
    level, only on the row's own features). Returns a dict {quantile_level: array} plus
    'correction' (for diagnostics)."""
    x = bundle["imputer"].transform(rows[bundle["features"]])
    probability = np.asarray(bundle["classifier"].predict_proba(x))[:, 1]
    normal_pred = b18sm.median_predict(bundle["normal"], x)
    spike_pred = b18sm.median_predict(bundle["spike"], x)
    excess = np.maximum(spike_pred - normal_pred, 0)
    correction = CORRECTION_WEIGHT * probability * excess

    base_quantiles = bundle["base"].predict(x, output_type="quantiles", quantiles=list(quantiles))
    out = {q: np.asarray(base_quantiles[i]) + correction for i, q in enumerate(quantiles)}
    return out


def fit_stage_b18(frame):
    """Returns the same long-format chain schema u02's fit_stage()/evaluate_stage() expect:
    anchor_year, eval_tower, model, date, q05, median, q95, y_true."""
    fx_cols = b17s.build_configs(frame)["BASE_ALL_52"]
    print(f"[U-08] BASE_ALL_52: {len(fx_cols)} fx_ columns (B18's actual champion feature set)")

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

        q_preds = predict_champion_quantiles(bundle, future)
        future = future.assign(
            _q05=q_preds[0.05], _median=q_preds[0.5], _q95=q_preds[0.95]
        )

        for tower in TOWERS:
            t_tower = time.time()
            tower_future = future.loc[future["tower"].eq(tower)].sort_values("Datetime")
            for _, r in tower_future.iterrows():
                rows.append(
                    {
                        "anchor_year": yr,
                        "eval_tower": tower,
                        "model": "B18_TabPFN_champion",
                        "date": r["Datetime"],
                        "q05": r["_q05"],
                        "median": r["_median"],
                        "q95": r["_q95"],
                        "y_true": r["y_observed"],
                    }
                )
            print(f"  Tower {tower} done ({time.time() - t_tower:.0f}s, {time.time() - t0:.0f}s elapsed)")
        print(f"  Anchor {yr} total: {time.time() - t_anchor:.0f}s")

    return pd.DataFrame(rows)


def main():
    frame = pd.read_csv(b17s.DATA_PATH, low_memory=False)
    frame["Datetime"] = pd.to_datetime(frame["Datetime"], format="mixed")
    frame = b17d.add_b17_features(frame)

    print("=" * 70)
    print("U-08 STAGE A: B18 champion (direct TabPFN + p95 spike-gate) quantile rollout")
    print("=" * 70)
    chains = fit_stage_b18(frame)
    if chains.empty:
        print("[FAIL] No rows produced -- aborting.")
        return
    chains.to_csv(CHAINS_PATH, index=False)
    print(f"\n[OK] Saved {CHAINS_PATH.name} ({len(chains)} rows)")

    print("\n" + "=" * 70)
    print("U-08 STAGE B: leave-one-anchor-out conformal calibration + evaluation (reuses U-02's evaluate_stage unchanged)")
    print("=" * 70)
    summary = evaluate_stage(chains)
    summary.to_csv(SUMMARY_PATH, index=False)
    print(f"\n[OK] Saved {SUMMARY_PATH.name} ({len(summary)} rows)")

    def wavg(g, col):
        vals = g[col]
        if vals.isna().all():
            return np.nan
        w = g["n"]
        return (vals * w).sum() / w.sum() if w.sum() > 0 else float("nan")

    print("\nPer-tower aggregate (n-weighted mean across bins):")
    agg = summary.groupby(["model", "eval_tower"]).apply(
        lambda g: pd.Series(
            {
                "raw_picp": wavg(g, "raw_picp") if "raw_picp" in g else float("nan"),
                "raw_mpiw": wavg(g, "raw_mpiw") if "raw_mpiw" in g else float("nan"),
                "raw_pinball": wavg(g, "raw_pinball") if "raw_pinball" in g else float("nan"),
                "conformal_picp": wavg(g, "conformal_picp"),
                "conformal_mpiw": wavg(g, "conformal_mpiw"),
                "conformal_pinball": wavg(g, "conformal_pinball"),
            }
        ),
        include_groups=False,
    ).reset_index()
    print(agg.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
