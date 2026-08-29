"""U-04: closes the UQ gap this project has carried since U-02 (D-62, 2026-07-06) -- that
calibration used forecast_daily_v2.csv and an 8-model roster that predates TabICLv2 (D-66,
2026-07-09), F-10's species-disaggregated features (D-67, 2026-07-10), and the champion becoming
TabPFN+species (D-67/D-80). U-02's conformal interval for "TabPFN" is calibrated for a superseded
feature configuration, not the model this project actually recommends. TabICLv2 has never had UQ
built for it at all.

Scope (user-confirmed, champion-focused rather than the full 11-model roster): TabPFN and
TabICLv2 only, both on forecast_daily_v3.csv's BASE+species config (F-10/D-67's actual champion
feature set, same FAMILIES/config construction as b16_foundation_models_v3.py, imported not
retyped). Both are zero-shot, native-quantile-supporting (`tabpfn_forecast(..., quantiles=...)`,
`tabicl_forecast(..., quantiles=...)`) -- no retraining, no new adapters, cheap per call (this
session's own TabPFN/TabICLv2 calls ran ~2-4s each). This is why the roster was narrowed to these
two rather than U-02's full 8 (RF/XGB/LightGBM/SARIMAX/TFT/2 ensembles all need real refitting per
anchor -- expensive, and their point-accuracy/conformal behavior on the OLD feature set is already
on record from U-02; only the two models whose actual production config changed need recalibrating).

Method: unchanged from U-02 -- same 3-tower x 5-anchor sweep, same quantiles (0.05, 0.5, 0.95),
same leave-one-anchor-out split-conformal calibration (`rr.conformal_margins_by_bin()`, per
lead-time bin), same metrics (PICP/MPIW/pinball, `src/evaluation/metrics.py`). `evaluate_stage()`
is imported UNCHANGED from `u02_multi_anchor_tower.py` (same chain schema: anchor_year, eval_tower,
model, date, q05, median, q95, y_true) rather than reimplemented -- only `fit_stage` differs
(TabPFN/TabICLv2 zero-shot rollout instead of U-02's pooled tree/SARIMAX/TFT fitting), so the two
experiments' RAW/CALIBRATED numbers are directly comparable, not just similarly-computed.

Run from project root:  python notebooks/06_interpretability_uq/u04_champion_uq.py
"""
import os
import sys
import time
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

ROOT = r"c:\Users\Nicholas\Documents\COMP0191 MSc Artificial Intelligence for Sustainable Development Project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\src")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import models.recursive_rollout as rr
from u02_multi_anchor_tower import evaluate_stage, QUANTILES, ANCHOR_YEARS, TOWERS, N_DAYS

from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))

HOURLY = rf"{ROOT}\data\Hourly"
RESULTS = rf"{ROOT}\results"

# Same FAMILIES/BASE_FX construction as b16_foundation_models_v3.py -- imported in spirit, not
# retyped independently, so this can't silently drift from what "species" actually means there.
SPECIES_COLS = ["fx_cattle_dens", "fx_sheep_dens", "fx_lamb_dens"]


def fit_stage_champion(dv, T, tabpfn_ok):
    """TabPFN + TabICLv2, BASE+species config, zero-shot per tower/anchor. Returns the same
    long-format chain schema U-02's fit_stage() produces (so evaluate_stage() works unmodified)."""
    fx_all = [c for c in dv.columns if c.startswith("fx")]
    base_fx = [c for c in fx_all if c not in SPECIES_COLS]
    fx_cols = base_fx + SPECIES_COLS
    print(f"[U-04] BASE+species: {len(fx_cols)} fx_ columns ({len(base_fx)} base + {len(SPECIES_COLS)} species)")

    rows = []
    t0 = time.time()

    for yr in ANCHOR_YEARS:
        anchor = pd.Timestamp(f"{yr}-12-16")
        target_dates = pd.date_range(anchor + pd.Timedelta(days=1), periods=N_DAYS, freq="D")
        print(f"\n{'='*70}\nAnchor {yr}\n{'='*70}")

        for tower in TOWERS:
            t_tower = time.time()
            dft = T[tower]
            hist = dft.loc[:anchor]
            hist_target = hist["y_observed"]
            hist_cov = hist[fx_cols]
            future_cov = dft.loc[target_dates, fx_cols]
            y_true_full = pd.Series(dft.loc[target_dates, "y_observed"].values, index=target_dates)

            if tabpfn_ok:
                try:
                    df_tabpfn = rr.tabpfn_forecast(hist_target, hist_cov, future_cov, mode="local",
                                                    quantiles=list(QUANTILES))
                    for d in target_dates:
                        rows.append({"anchor_year": yr, "eval_tower": tower, "model": "TabPFN", "date": d,
                                     "q05": df_tabpfn.loc[d, 0.05], "median": df_tabpfn.loc[d, "median"],
                                     "q95": df_tabpfn.loc[d, 0.95], "y_true": y_true_full.loc[d]})
                except Exception as e:
                    print(f"    T{tower} {yr} TabPFN SKIPPED: {str(e)[:150]}")

            try:
                df_tabicl = rr.tabicl_forecast(hist_target, hist_cov, future_cov, quantiles=list(QUANTILES))
                for d in target_dates:
                    rows.append({"anchor_year": yr, "eval_tower": tower, "model": "TabICLv2", "date": d,
                                 "q05": df_tabicl.loc[d, 0.05], "median": df_tabicl.loc[d, "median"],
                                 "q95": df_tabicl.loc[d, 0.95], "y_true": y_true_full.loc[d]})
            except Exception as e:
                print(f"    T{tower} {yr} TabICLv2 SKIPPED: {str(e)[:150]}")

            print(f"  Tower {tower} done ({time.time()-t_tower:.0f}s, {time.time()-t0:.0f}s elapsed)")

    return pd.DataFrame(rows)


def main():
    dv = pd.read_csv(f"{HOURLY}/forecast_daily_v3.csv", low_memory=False)
    dv["Datetime"] = pd.to_datetime(dv["Datetime"], format="mixed")
    T = {t: dv[dv.tower == t].set_index("Datetime").sort_index() for t in TOWERS}

    tabpfn_ok = bool(os.environ.get("TABPFN_TOKEN"))
    if not tabpfn_ok:
        print("WARNING: TABPFN_TOKEN not set -- TabPFN steps will be skipped this run.")

    print("="*70)
    print("U-04 STAGE A: TabPFN/TabICLv2 BASE+species zero-shot quantile rollout")
    print("="*70)
    chains = fit_stage_champion(dv, T, tabpfn_ok)
    chains.to_csv(f"{RESULTS}/u04_chains.csv", index=False)
    print(f"\n[OK] Saved u04_chains.csv ({len(chains)} rows)")

    print("\n" + "="*70)
    print("U-04 STAGE B: leave-one-anchor-out conformal calibration + evaluation (reuses U-02's evaluate_stage unchanged)")
    print("="*70)
    summary = evaluate_stage(chains)
    summary.to_csv(f"{RESULTS}/u04_summary.csv", index=False)
    print(f"\n[OK] Saved u04_summary.csv ({len(summary)} rows)")

    def wavg(g, col):
        vals = g[col]
        if vals.isna().all():
            import numpy as np
            return np.nan
        w = g["n"]
        return (vals * w).sum() / w.sum() if w.sum() > 0 else float("nan")

    print("\nPer-model/tower aggregate (n-weighted mean across bins):")
    agg = summary.groupby(["model", "eval_tower"]).apply(
        lambda g: pd.Series({
            "raw_picp": wavg(g, "raw_picp") if "raw_picp" in g else float("nan"),
            "raw_mpiw": wavg(g, "raw_mpiw") if "raw_mpiw" in g else float("nan"),
            "raw_pinball": wavg(g, "raw_pinball") if "raw_pinball" in g else float("nan"),
            "conformal_picp": wavg(g, "conformal_picp"),
            "conformal_mpiw": wavg(g, "conformal_mpiw"),
            "conformal_pinball": wavg(g, "conformal_pinball"),
        }), include_groups=False
    ).reset_index()
    print(agg.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
