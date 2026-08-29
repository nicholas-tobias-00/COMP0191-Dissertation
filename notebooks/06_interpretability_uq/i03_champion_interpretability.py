"""I-03: closes the interpretability gap this project has carried since I-02 (D-61, 2026-07-06) --
that pass predates TabICLv2 joining the roster (D-66, 2026-07-09) and F-10's species-disaggregated
features (D-67, 2026-07-10), which together produced the standing champion, TabPFN+species
(D-67/D-80, MASE=0.715 climatology-scored). I-02's SHAP/permutation results were computed on the
OLD 8-model roster and OLD (BASE-only) feature set -- the model this project actually recommends
has never been through an interpretability pass. This is the exact gap U-04 already closed for UQ
(same "predates the champion" framing); I-03 applies the same fix to interpretability.

Scope (mirrors U-04's champion-focused precedent): TabPFN only -- the single best model in this
project's roster (S-03's own table: TabPFN MASE=0.855, the lowest/best of all 11 models tested,
including TabICLv2 at 0.930), i.e. the model actually ingested as "Model 1" for S-03's
driver-availability ablation. TabICLv2 is not covered here (a natural, cheap follow-up given it's
also zero-shot, same cost profile -- flagged, not executed this pass, matching U-04's own
TabPFN+TabICLv2 vs U-05's TabICLv2-only precedent for narrowing scope to what's actually needed).

Method: UNCHANGED from I-02's own TabPFN treatment (`i02_multi_anchor_tower.py` lines ~277-304) --
permutation importance is TabPFN's only available substitute (it has no native per-feature signal,
and is architecturally mismatched with SHAP's row-wise tabular framework, per I-02's own documented
reasoning). Per (anchor, tower): one baseline zero-shot rollout, then one single-shuffle permutation
per feature column (seeded by anchor year, exactly as I-02 did -- not re-derived, not upgraded to
n_repeats>1, so old-config vs new-config importance numbers stay comparable, not just similarly
computed), importance = |mean(shuffled_chain) - mean(base_chain)|. Only what changed: (a) the
feature set is BASE+species (52 fx_ columns from forecast_daily_v3.csv, F-10's actual champion
config, same construction as u04_champion_uq.py's fit_stage_champion -- imported in spirit, not
retyped) instead of I-02's old BASE-only FX_B; (b) source data is forecast_daily_v3.csv, not v2.

Same 3-tower x 5-anchor sweep as I-02/U-02/U-04 (2018-2022 anchors, T2/T4/T9) for full coverage.
52 features + 1 baseline = 53 TabPFN calls per (tower, anchor) x 15 (tower, anchor) pairs = 795
calls total, ~2-4s each per U-04's own measured cost -- run in background, not interactively.

Run from project root:  python notebooks/06_interpretability_uq/i03_champion_interpretability.py
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = r"c:\Users\Nicholas\Documents\COMP0191 MSc Artificial Intelligence for Sustainable Development Project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\src")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import models.recursive_rollout as rr
from u02_multi_anchor_tower import ANCHOR_YEARS, TOWERS, N_DAYS

from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))

HOURLY = rf"{ROOT}\data\Hourly"
RESULTS = rf"{ROOT}\results"

# Identical construction to u04_champion_uq.py's fit_stage_champion -- BASE+species, F-10's actual
# champion feature set. Not retyped independently so this can't silently drift from what "species"
# means there.
SPECIES_COLS = ["fx_cattle_dens", "fx_sheep_dens", "fx_lamb_dens"]


def tabpfn_permutation_importance(dv, T, tabpfn_ok):
    """Per (anchor, tower): baseline TabPFN rollout + one single-shuffle permutation per fx_
    column, method unchanged from I-02. Returns long-format rows: anchor_year, eval_tower, model,
    feature, importance."""
    fx_all = [c for c in dv.columns if c.startswith("fx")]
    base_fx = [c for c in fx_all if c not in SPECIES_COLS]
    fx_cols = base_fx + SPECIES_COLS
    print(f"[I-03] BASE+species: {len(fx_cols)} fx_ columns ({len(base_fx)} base + {len(SPECIES_COLS)} species)")

    if not tabpfn_ok:
        print("WARNING: TABPFN_TOKEN not set -- I-03 cannot run without it (TabPFN is the only model in scope).")
        return pd.DataFrame()

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

            try:
                base_chain = rr.tabpfn_forecast(hist_target, hist_cov, future_cov, mode="local")
                base_mean = base_chain.mean()

                perm_rng = np.random.default_rng(yr)
                perm_scores = {}
                for col in fx_cols:
                    shuffled = future_cov.copy()
                    shuffled[col] = perm_rng.permutation(shuffled[col].values)
                    shuffled_chain = rr.tabpfn_forecast(hist_target, hist_cov, shuffled, mode="local")
                    perm_scores[col] = abs(shuffled_chain.mean() - base_mean)

                for feat, val in perm_scores.items():
                    rows.append({"anchor_year": yr, "eval_tower": tower, "model": "TabPFN",
                                 "feature": feat, "importance": val})
                print(f"  Tower {tower} done ({time.time()-t_tower:.0f}s, {time.time()-t0:.0f}s elapsed)")
            except Exception as e:
                print(f"    T{tower} {yr} TabPFN SKIPPED: {str(e)[:150]}")

    return pd.DataFrame(rows)


def main():
    dv = pd.read_csv(f"{HOURLY}/forecast_daily_v3.csv", low_memory=False)
    dv["Datetime"] = pd.to_datetime(dv["Datetime"], format="mixed")
    T = {t: dv[dv.tower == t].set_index("Datetime").sort_index() for t in TOWERS}

    tabpfn_ok = bool(os.environ.get("TABPFN_TOKEN"))

    print("=" * 70)
    print("I-03: TabPFN+species permutation importance, BASE+species config, forecast_daily_v3.csv")
    print("=" * 70)
    importance_rows = tabpfn_permutation_importance(dv, T, tabpfn_ok)
    if importance_rows.empty:
        print("[FAIL] No rows produced -- aborting.")
        return

    importance_rows.to_csv(f"{RESULTS}/i03_tabpfn_species_importance.csv", index=False)
    print(f"\n[OK] Saved i03_tabpfn_species_importance.csv ({len(importance_rows)} rows)")

    print("\nOverall ranking (mean importance across all towers/anchors):")
    overall = importance_rows.groupby("feature")["importance"].mean().sort_values(ascending=False)
    print(overall.round(4).to_string())
    overall.round(6).to_csv(f"{RESULTS}/i03_tabpfn_species_importance_ranked.csv", header=["mean_importance"])

    print("\nPer-tower ranking (mean importance across 5 anchors, top 10 each):")
    by_tower = importance_rows.groupby(["eval_tower", "feature"])["importance"].mean().reset_index()
    by_tower.to_csv(f"{RESULTS}/i03_tabpfn_species_importance_by_tower.csv", index=False)
    for t in TOWERS:
        sub = by_tower[by_tower.eval_tower == t].sort_values("importance", ascending=False).head(10)
        print(f"\n  Tower {t}:")
        print(sub[["feature", "importance"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
