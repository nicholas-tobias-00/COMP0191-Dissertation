"""F-10 (D-67), Stage 1 signal check: cheap, bounded leave-one-group-in RF ablation over
forecast_daily_v3.csv's 5 new feature families, all 3 towers, h in {1, 14}.

Deliberately a smoke-test tier of rigor (single seed, no HPO, no cross-validation) -- the point
is to decide which families are worth carrying into a real Stage 2 rerun (B-16, point-forecast
and/or recursive-rollout), not to produce a final reported number. Matches this project's
established two-stage pattern (F-01's ablation-before-integration, B-14's CV-search-before-
B-15's-real-validation).

Direct-forecast harness mirrors B03_enriched_ML.ipynb's Track B (daily) `aligned()`/`fit_model()`
structure: fx_ columns are shifted -h (future, perfect-foresight covariates, matching this
project's established convention for all fx_ features); AR_ columns stay at their origin-time
value; target = y_observed.shift(-h); RF hyperparameters are B-03/B-10's exact daily-track config
(n_estimators=500, min_samples_leaf=10, max_features=0.5, random_state=42) -- no new HPO.
Pooled training (all 3 towers, 2018-2021), evaluated directly per tower on 2022-2023 (matching
B-03's Tower 4/9 "MAIN" evaluation; Tower 2 evaluated the same way and reported honestly if too
thin, rather than replicating B-03's separate expanding-window T2 fold machinery -- not needed
for a bounded smoke-test-tier check).

Run from project root:  python notebooks/04_feature_engineering/f10_signal_check.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
HOURLY = ROOT / "data" / "Hourly"

TOWERS = [2, 4, 9]
DUM = ["is_t2", "is_t4", "is_t9"]
HORIZONS = [1, 14]

FAMILIES = {
    "species": ["fx_cattle_dens", "fx_sheep_dens", "fx_lamb_dens"],
    "arable": ["fx_is_arable"],
    "flow": ["fx_flow_mean", "fx_flow_lag7", "fx_flow_lag14", "fx_flow_lag21", "fx_flow_lag28",
             "fx_flow_roll7", "fx_flow_roll14"],
    "mgmt": ["fx_mgmt_fertN_recency", "fx_mgmt_fertN_rate", "fx_mgmt_lime_recency",
             "fx_mgmt_cultiv_recency", "fx_mgmt_cut_recency", "fx_mgmt_manure_recency"],
    "bodyweight": ["fx_total_liveweight_dens"],
}
ALL_NEW = sorted({c for cols in FAMILIES.values() for c in cols})


def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


def aligned(T, h, ar_cols, fx_cols):
    """Mirrors B03_enriched_ML.ipynb's aligned(): fx_ = future covariates (shift -h), AR_ = origin
    time, target = y_observed h days ahead."""
    parts = []
    for t, df in T.items():
        f = df[ar_cols + DUM].copy()
        for c in fx_cols:
            f[c] = df[c].shift(-h)
        f["target"] = df["y_observed"].shift(-h)
        f["tower"] = t
        f["ttime"] = df.index + pd.Timedelta(days=h)
        parts.append(f)
    return pd.concat(parts)


def fit_rf(tr, feat):
    imp = SimpleImputer(strategy="mean")
    Xi = imp.fit_transform(tr[feat].values)
    m = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=42,
                               min_samples_leaf=10, max_features=0.5)
    m.fit(Xi, tr["target"].values)
    return m, imp


def main():
    dv = pd.read_csv(HOURLY / "forecast_daily_v3.csv", low_memory=False)
    dv["Datetime"] = pd.to_datetime(dv["Datetime"], format="mixed")
    AR_COLS = [c for c in dv.columns if c.startswith("ar_")]
    fx_all = [c for c in dv.columns if c.startswith("fx")]
    BASE_FX = [c for c in fx_all if c not in ALL_NEW]
    print(f"BASE_FX ({len(BASE_FX)}): {BASE_FX}")
    print(f"New columns by family: {FAMILIES}")

    T = {t: dv[dv.tower == t].set_index("Datetime").sort_index() for t in TOWERS}

    feat_by_config = {"BASE": AR_COLS + BASE_FX + DUM}
    for fam, cols in FAMILIES.items():
        feat_by_config[f"BASE+{fam}"] = AR_COLS + BASE_FX + cols + DUM
    feat_by_config["BASE+ALL"] = AR_COLS + BASE_FX + ALL_NEW + DUM
    # Follow-up swap test: fx_cattle_dens/fx_lsu_dens correlate at r=0.97 (species collinear with
    # the existing combined density) -- "BASE+species" ADDS species alongside fx_lsu_dens, so any
    # SHAP attention on fx_cattle_dens could just be credit-splitting with a redundant feature,
    # not new information. This variant instead REPLACES fx_lsu_dens with the 3-way species split,
    # a cleaner test of whether disaggregation itself helps.
    base_fx_no_lsu = [c for c in BASE_FX if c != "fx_lsu_dens"]
    feat_by_config["SWAP_species_for_lsu"] = AR_COLS + base_fx_no_lsu + FAMILIES["species"] + DUM

    rows = []
    shap_rows = []
    for h in HORIZONS:
        A = aligned(T, h, AR_COLS, fx_all)
        tr = A[(A.ttime.dt.year.between(2018, 2021)) & A.target.notna()]
        print(f"\n=== h={h}: train n={len(tr)} ===")

        for cfg_name, feat in feat_by_config.items():
            m, imp = fit_rf(tr, feat)

            for t in TOWERS:
                te = A[(A.tower == t) & (A.ttime.dt.year.isin([2022, 2023])) & A.target.notna()]
                if len(te) < 5:
                    print(f"  {cfg_name} h={h} T{t}: skipped, only {len(te)} real test rows")
                    continue
                pred = m.predict(imp.transform(te[feat].values))
                y = te["target"].values
                r2 = r2_score(y, pred) if (len(y) > 1 and np.var(y) > 0) else np.nan
                rows.append(dict(config=cfg_name, horizon=h, tower=t, n_test=len(te),
                                  RMSE=rmse(y, pred), MAE=float(mean_absolute_error(y, pred)),
                                  R2=float(r2) if np.isfinite(r2) else np.nan))

            if cfg_name == "BASE+ALL":
                # SHAP mean|value| on ALL_NEW columns, evaluated over the pooled training rows'
                # own feature distribution (matches F-01/I-02's own SHAP usage pattern).
                import shap
                Xi = imp.transform(tr[feat].values)
                sample = Xi[np.random.RandomState(0).choice(len(Xi), size=min(500, len(Xi)), replace=False)]
                explainer = shap.TreeExplainer(m)
                sv = explainer.shap_values(sample)
                mean_abs = np.abs(sv).mean(axis=0)
                for col, val in zip(feat, mean_abs):
                    if col in ALL_NEW:
                        shap_rows.append(dict(horizon=h, column=col, mean_abs_shap=float(val)))

        print(f"h={h} done")

    res = pd.DataFrame(rows)
    res.to_csv(HOURLY.parent.parent / "results" / "f10_signal_check_summary.csv", index=False)
    print(f"\nWrote results/f10_signal_check_summary.csv ({len(res)} rows)")

    shap_df = pd.DataFrame(shap_rows).sort_values(["horizon", "mean_abs_shap"], ascending=[True, False])
    shap_df.to_csv(HOURLY.parent.parent / "results" / "f10_signal_check_shap.csv", index=False)
    print(f"Wrote results/f10_signal_check_shap.csv ({len(shap_df)} rows)")

    # ---- ablation deltas vs BASE, per family, per tower/horizon ----
    piv = res.pivot_table(index=["horizon", "tower"], columns="config", values="R2")
    print("\nR2 by config (rows=horizon/tower):")
    print(piv.round(4).to_string())

    print("\nDelta R2 vs BASE, per family:")
    delta_rows = []
    for fam in FAMILIES:
        col = f"BASE+{fam}"
        if col not in piv.columns:
            continue
        d = piv[col] - piv["BASE"]
        for (h, t), val in d.items():
            delta_rows.append(dict(family=fam, horizon=h, tower=t, delta_R2=val))
    delta_df = pd.DataFrame(delta_rows)
    delta_df.to_csv(HOURLY.parent.parent / "results" / "f10_signal_check_deltas.csv", index=False)
    print(delta_df.pivot_table(index=["family", "horizon"], columns="tower", values="delta_R2").round(4).to_string())


if __name__ == "__main__":
    main()
