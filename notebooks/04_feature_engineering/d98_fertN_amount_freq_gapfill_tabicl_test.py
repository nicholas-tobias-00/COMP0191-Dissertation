"""D-98 additive test (Process 1/3, TabICL variant -- user redirected away from RF): does adding
corrected fertN_amount (true kg N/ha) + fertN_freq (trailing-365d true-N event count) to the
gap-filling BENCHMARK-BEST model (TabICL-solo, D-79, champion FEATURES = same feat_list() as RFm)
help or hurt? RFm's own additive test (same features) found a real, consistent degradation at all
3 towers (T2 0.576->0.556, T4 0.404->0.373, T9 0.426->0.382) -- this checks whether that holds for
a structurally different model (foundation/in-context, solo-per-tower, not a fitted RF), or is
specific to RF's splitting behavior.

Fully additive: imports gapfill_tabicl.py's fit/predict/evaluate_tower pattern unchanged, defines
its own frame_plus_fertN()/feat_list_plus_fertN() (shared with the RF test script) rather than
editing gapfill_tabicl.py in place. TabICL-solo baseline for comparison (D-79, champion features,
n_bags=1): T2=0.676, T4=0.428, T9=0.423.

Run from project root:  python notebooks/04_feature_engineering/d98_fertN_amount_freq_gapfill_tabicl_test.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"c:\Users\Nicholas\Documents\COMP0191 MSc Artificial Intelligence for Sustainable Development Project")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "models"))
sys.path.insert(0, str(ROOT / "src" / "evaluation"))

import gapfill_rfm as G
import gapfill_tabicl as TI
from gap_cv import dom_mask, SCENARIOS, insert_calendar_gaps, gapfilling_metrics, median_metrics

HOURLY = ROOT / "data" / "Hourly"
RESULTS = ROOT / "results"


def load_ext_plus_fertN():
    d = G.load_ext()
    fq = pd.read_csv(HOURLY / "fertN_amount_freq_features.csv", low_memory=False)
    fq["Datetime"] = pd.to_datetime(fq["Datetime"], format="mixed")
    fq = fq.set_index("Datetime")
    return d.join(fq, how="left")


def frame_plus_fertN(t, pooled, d):
    g = G.frame(t, pooled, d)
    g["fertN_amount"] = d[f"fertN_amount_t{t}"].reindex(g.index)
    g["fertN_freq"] = d[f"fertN_freq_t{t}"].reindex(g.index)
    return g


def feat_list_plus_fertN():
    return G.feat_list() + ["fertN_amount", "fertN_freq"]


def evaluate_tower_tabicl_plus_fertN(t, d_all, n_bags=1):
    """Mirrors gapfill_tabicl.evaluate_tower() exactly (solo, not pooled -- D-79's own finding),
    swapping FEATURES for the +fertN extension."""
    feat = feat_list_plus_fertN()
    g = frame_plus_fertN(t, pooled=False, d=d_all)
    dm = dom_mask(g.index, t)
    train_std = float(g.loc[dm, "target"].std())
    out = {}
    for sc, gh in SCENARIOS.items():
        rows = []
        for gt in insert_calendar_gaps(g, "target", dm, gh):
            if len(gt) < 5:
                continue
            train_rows = g.loc[dm & g["target"].notna().values].drop(index=gt, errors="ignore")
            test_rows = g.loc[gt]
            models, imp = TI.fit(feat, train_rows, n_bags=n_bags)
            yp = TI.predict(models, imp, test_rows[feat].values)
            rows.append(gapfilling_metrics(test_rows["target"].values, yp, train_std))
        out[sc] = median_metrics(rows)
    return out


CHAMPION_R2 = {2: 0.676, 4: 0.428, 9: 0.423}  # D-79 TabICL-solo, champion features


def main():
    d_all = load_ext_plus_fertN()
    print(f"[OK] loaded, {d_all.shape}")

    rows = []
    for t in G.TOWERS:
        res = evaluate_tower_tabicl_plus_fertN(t, d_all)
        r2_by_scenario = {sc: m["R2"] for sc, m in res.items()}
        r2_mean = float(np.mean(list(r2_by_scenario.values())))
        print(f"Tower {t}: {r2_by_scenario}  mean={r2_mean:.4f}  (TabICL-solo baseline={CHAMPION_R2[t]})")
        for sc, m in res.items():
            rows.append({"tower": t, "scenario": sc, **m})

    out = pd.DataFrame(rows)
    out.to_csv(f"{RESULTS}/d98_gapfill_tabicl_fertN_amount_freq_test.csv", index=False)
    print(f"[OK] Saved d98_gapfill_tabicl_fertN_amount_freq_test.csv")


if __name__ == "__main__":
    main()
