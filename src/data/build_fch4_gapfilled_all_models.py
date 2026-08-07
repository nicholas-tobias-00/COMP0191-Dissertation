"""Production gap-filled FCH4 series for every model this project validated (D-79), not just the
RFm champion. Generalizes `build_fch4_gapfilled.py`'s pattern (fit once on ALL real, domain-
restricted data -- no held-out rows, this is production not a CV fold -- predict every domain
timestamp) to MDS, Mean, MICE, HyperImpute, RFm met-only, and TabICL-solo. The RFm champion
column is **reused directly from the existing `fch4_gapfilled.csv`**, not refit -- run
`build_fch4_gapfilled.py` first if that file doesn't exist yet.

Output format: long, not wide -- `Datetime, tower, model, y_observed, y_gapfilled` -- matching
this project's own `all_raw_predictions.csv`/`fch4_gapfilled_all_models.csv` (notebook)
conventions, one `model` column distinguishing stacked rows.

Output: data/Hourly/fch4_gapfilled_all_models.csv
Run from project root (after build_fch4_gapfilled.py has produced the champion file):
    python src/data/build_fch4_gapfilled_all_models.py
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  registers IterativeImputer
from sklearn.impute import IterativeImputer, SimpleImputer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from gapfill_rfm import load_ext, frame, fit, feat_list, DUM, TOWERS, cfg, ts_col_for  # noqa: E402
from gapfill_mds import mds_fill_batch  # noqa: E402
from gapfill_baselines import MET_ONLY_FEATURES  # noqa: E402
from gap_cv import dom_mask  # noqa: E402

HOURLY = Path(__file__).resolve().parents[2] / "data" / "Hourly"
FEATURES = feat_list()
PRED_BATCH = 2000   # TabICL's in-context embedding buffer OOMs on an unbatched whole-domain
                    # predict for wide feature sets -- harmless no-op for RF/sklearn imputers


def _domain_rows(t, d_all):
    g = frame(t, pooled=True, d=d_all)
    dm = dom_mask(g.index, t)
    return g.loc[dm]


def production_fill_mds(d_all):
    parts = []
    for t in TOWERS:
        c = cfg(t, ts_col_for(t))
        g = _domain_rows(t, d_all)
        obs = g["target"].notna()
        gap_ts = list(g.index[~obs])
        sw_col, ta_col = c["sw"], c["ta"]
        vpd_col = f"VPD_0_0_1 [Tower {t}]"
        # mds_fill_batch reads sw/ta/vpd straight from d_all -- rebuild a domain-restricted frame
        # with the same target masking frame() already applied (QC'd/plausibility-filtered target)
        d_t = d_all.copy(); d_t[c["tgt"]] = np.nan
        d_t.loc[g.index[obs], c["tgt"]] = g.loc[obs, "target"].values
        preds = mds_fill_batch(d_t, c["tgt"], sw_col, ta_col, gap_ts, vpd_col=vpd_col)
        filled = g["target"].copy()
        for ts, v in preds.items():
            filled.loc[ts] = v
        n_filled = filled.notna().sum()
        print(f"  MDS   Tower {t}: observed {int(obs.sum()):>6} -> gap-filled {int(n_filled):>6} "
              f"({100*n_filled/len(g):.0f}%, {len(gap_ts) - len(preds)} unfillable)")
        parts.append(pd.DataFrame({"Datetime": g.index, "tower": t, "model": "MDS",
                                    "y_observed": g["target"].values, "y_gapfilled": filled.values}))
    return pd.concat(parts, ignore_index=True)


def production_fill_mean(d_all):
    parts = []
    for t in TOWERS:
        g = _domain_rows(t, d_all)
        obs = g["target"].notna()
        train_mean = g.loc[obs, "target"].mean()
        filled = g["target"].where(obs, train_mean)
        print(f"  Mean  Tower {t}: observed {int(obs.sum()):>6} -> gap-filled {int(filled.notna().sum()):>6} "
              f"(constant={train_mean:.2f})")
        parts.append(pd.DataFrame({"Datetime": g.index, "tower": t, "model": "Mean",
                                    "y_observed": g["target"].values, "y_gapfilled": filled.values}))
    return pd.concat(parts, ignore_index=True)


def _pooled_impute(d_all, feat, plugin_fit_transform, label):
    frames = {t: _domain_rows(t, d_all) for t in TOWERS}
    parts = []
    for t in TOWERS:
        g = frames[t]; obs = g["target"].notna()
        mat_parts = [frames[tt][["target"] + feat].reset_index(drop=True) for tt in TOWERS]
        offset = sum(len(mat_parts[i]) for i in range(TOWERS.index(t)))
        mat = pd.concat(mat_parts, ignore_index=True)
        filled_vals = plugin_fit_transform(mat)
        filled = pd.Series(filled_vals[offset:offset + len(g)], index=g.index)
        print(f"  {label:5s} Tower {t}: observed {int(obs.sum()):>6} -> gap-filled {int(filled.notna().sum()):>6}")
        parts.append(pd.DataFrame({"Datetime": g.index, "tower": t, "model": label,
                                    "y_observed": g["target"].values, "y_gapfilled": filled.values}))
    return pd.concat(parts, ignore_index=True)


def production_fill_mice(d_all):
    def fit_transform(mat):
        imputer = IterativeImputer(random_state=42)
        filled = imputer.fit_transform(mat.values)
        return filled[:, mat.columns.get_loc("target")]
    return _pooled_impute(d_all, FEATURES, fit_transform, "MICE")


def production_fill_hyperimpute(d_all):
    from hyperimpute.plugins.imputers import Imputers

    def fit_transform(mat):
        plugin = Imputers().get("hyperimpute", random_state=42)
        filled_df = plugin.fit_transform(mat)
        return filled_df.iloc[:, filled_df.columns.get_loc("target")].values
    return _pooled_impute(d_all, FEATURES, fit_transform, "HyperImpute")


def _production_fill_rf(d_all, feat_base, pooled, label, extra_dum=True):
    feat = feat_base + (DUM if (pooled and extra_dum) else [])
    frames = {t: _domain_rows(t, d_all) for t in TOWERS}
    parts = []
    for t in TOWERS:
        g = frames[t]; obs = g["target"].notna()
        if pooled:
            trd = pd.concat([frames[tt][frames[tt]["target"].notna()] for tt in TOWERS], ignore_index=True)
        else:
            trd = g[obs]
        rf, imp = fit(feat, trd)
        pred = rf.predict(imp.transform(g[feat].values))
        filled = g["target"].where(obs, pd.Series(pred, index=g.index))
        print(f"  {label:9s} Tower {t}: observed {int(obs.sum()):>6} -> gap-filled {int(filled.notna().sum()):>6}")
        parts.append(pd.DataFrame({"Datetime": g.index, "tower": t, "model": label,
                                    "y_observed": g["target"].values, "y_gapfilled": filled.values}))
    return pd.concat(parts, ignore_index=True)


def production_fill_met_only(d_all):
    return _production_fill_rf(d_all, MET_ONLY_FEATURES, pooled=True, label="RFm_metOnly")


def production_fill_tabicl(d_all):
    from tabicl import TabICLRegressor
    from gapfill_tabicl import FOUNDATION_MODEL_ROW_CAP

    parts = []
    for t in TOWERS:
        g = frame(t, pooled=False, d=d_all)
        dm = dom_mask(g.index, t)
        g = g.loc[dm]
        obs = g["target"].notna()
        trd = g[obs]
        imp = SimpleImputer(strategy="mean")
        trd_sub = trd.sample(n=min(FOUNDATION_MODEL_ROW_CAP, len(trd)), random_state=42)
        m = TabICLRegressor(random_state=42)
        m.fit(imp.fit_transform(trd_sub[FEATURES].values), trd_sub["target"].values)
        X = imp.transform(g[FEATURES].values)
        preds = np.concatenate([m.predict(X[i:i + PRED_BATCH]) for i in range(0, len(X), PRED_BATCH)])
        filled = g["target"].where(obs, pd.Series(preds, index=g.index))
        print(f"  TabICL    Tower {t}: observed {int(obs.sum()):>6} -> gap-filled {int(filled.notna().sum()):>6}")
        parts.append(pd.DataFrame({"Datetime": g.index, "tower": t, "model": "TabICL_solo",
                                    "y_observed": g["target"].values, "y_gapfilled": filled.values}))
    return pd.concat(parts, ignore_index=True)


def production_fill_champion():
    """Reuses the already-computed RFm champion series directly -- does NOT refit. Run
    build_fch4_gapfilled.py first if data/Hourly/fch4_gapfilled.csv doesn't exist yet."""
    champ_path = HOURLY / "fch4_gapfilled.csv"
    if not champ_path.exists():
        raise FileNotFoundError(f"{champ_path} not found -- run build_fch4_gapfilled.py first")
    champ = pd.read_csv(champ_path, parse_dates=["Datetime"]).set_index("Datetime")
    parts = []
    for t in TOWERS:
        filled = champ[f"FCH4_gapfilled [Tower {t}]"]
        obs_mask = champ[f"FCH4_observed_mask [Tower {t}]"].astype(bool)
        y_obs = filled.where(obs_mask)
        parts.append(pd.DataFrame({"Datetime": filled.index, "tower": t, "model": "RFm_champion",
                                    "y_observed": y_obs.values, "y_gapfilled": filled.values}))
    print("  RFm_champion: reused directly from fch4_gapfilled.csv (not refit)")
    return pd.concat(parts, ignore_index=True)


def main():
    d_all = load_ext()
    print(f"Loaded EXT layer {d_all.shape}\n")

    all_parts = [
        production_fill_champion(),
        production_fill_met_only(d_all),
        production_fill_mds(d_all),
        production_fill_mean(d_all),
        production_fill_mice(d_all),
        production_fill_hyperimpute(d_all),
        production_fill_tabicl(d_all),
    ]
    out = pd.concat(all_parts, ignore_index=True)

    dest = HOURLY / "fch4_gapfilled_all_models.csv"
    out.to_csv(dest, index=False)
    print(f"\nWrote {dest}  ({out.shape[0]:,} rows x {out.shape[1]} cols, "
          f"{out['model'].nunique()} models x {out['tower'].nunique()} towers)")


if __name__ == "__main__":
    main()
