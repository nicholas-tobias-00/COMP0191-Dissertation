"""D-98 additive test (Process 1/3): does adding corrected fertN_amount (true kg N/ha,
recency-weighted) + the new fertN_freq (trailing-365d true-N event count) to the production
gap-filling champion (RFm, F-08/D-35/D-77) help or hurt? F-01 excluded fertN entirely from this
model for overfitting (Tower 9 collapse, -0.86 R2) -- but that was raw product-mass rate, no
frequency signal, tested before D-97's units fix existed. Re-tested here on the corrected features,
not assumed to repeat the old result.

Fully additive: imports gapfill_rfm.py's cfg/fit/TOWERS/etc. unchanged, defines its own
frame_plus_fertN()/feat_list_plus_fertN() rather than editing frame()/feat_list() in place.
Champion baseline for comparison: T2=0.576, T4=0.404, T9=0.426 (D-77).

Run from project root:  python notebooks/04_feature_engineering/d98_fertN_amount_freq_gapfill_test.py
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


def evaluate_tower_plus_fertN(t, d_all):
    feat = feat_list_plus_fertN() + G.DUM
    frames = {tt: frame_plus_fertN(tt, pooled=True, d=d_all) for tt in G.TOWERS}
    g = frames[t]; dm = dom_mask(g.index, t)
    train_std = float(g.loc[dm, "target"].std())
    out = {}
    for sc, gh in SCENARIOS.items():
        rows = []
        for gt in insert_calendar_gaps(g, "target", dm, gh):
            if len(gt) < 5:
                continue
            trd = pd.concat([frames[tt][dom_mask(frames[tt].index, tt) & frames[tt]["target"].notna().values]
                              .drop(index=gt, errors="ignore") for tt in G.TOWERS], ignore_index=True)
            rf, imp = G.fit(feat, trd)
            yp = rf.predict(imp.transform(g.loc[gt, feat].values))
            rows.append(gapfilling_metrics(g.loc[gt, "target"].values, yp, train_std))
        out[sc] = median_metrics(rows)
    return out


CHAMPION_R2 = {2: 0.576, 4: 0.404, 9: 0.426}


def main():
    d_all = load_ext_plus_fertN()
    print(f"[OK] loaded, {d_all.shape}")

    rows = []
    for t in G.TOWERS:
        res = evaluate_tower_plus_fertN(t, d_all)
        r2_by_scenario = {sc: m["R2"] for sc, m in res.items()}
        r2_mean = float(np.mean(list(r2_by_scenario.values())))
        print(f"Tower {t}: {r2_by_scenario}  mean={r2_mean:.4f}  (champion baseline={CHAMPION_R2[t]})")
        for sc, m in res.items():
            rows.append({"tower": t, "scenario": sc, **m})

    out = pd.DataFrame(rows)
    out.to_csv(f"{RESULTS}/d98_gapfill_fertN_amount_freq_test.csv", index=False)
    print(f"[OK] Saved d98_gapfill_fertN_amount_freq_test.csv")


if __name__ == "__main__":
    main()
