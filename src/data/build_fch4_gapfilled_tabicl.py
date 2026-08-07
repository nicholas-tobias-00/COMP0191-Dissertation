"""TabICL-solo production gap-filled FCH4 series (D-79) -- same wide-format schema as
`build_fch4_gapfilled.py`'s `fch4_gapfilled.csv` (drop-in-compatible column names), so downstream
forecasting code can consume it as an alternate/additional AR source with a filename swap, not a
restructuring. See `src/models/gapfill_tabicl.py`'s module docstring for why this is solo
(per-tower, not pooled) and why TabICL-solo is worth having alongside the RFm champion: it beats
RFm at Tower 2 (0.676 vs 0.576) and Tower 4 (0.428 vs 0.404) on the held-out gap-CV benchmark.

For each tower: train one TabICL-solo model on that tower's own real, domain-restricted CH4
(no held-out rows -- this is production, not a CV fold), predict every timestamp of that tower's
own DOMAIN window. Output = observed where genuinely measured, else the TabICL prediction, plus
an observed-mask column -- identical shape to `fch4_gapfilled.csv`.

Output: data/Hourly/fch4_gapfilled_tabicl.csv
Run from project root:  python src/data/build_fch4_gapfilled_tabicl.py
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from gapfill_rfm import load_ext, frame, feat_list, TOWERS  # noqa: E402
from gapfill_tabicl import FOUNDATION_MODEL_ROW_CAP  # noqa: E402
from gap_cv import dom_mask  # noqa: E402

HOURLY = Path(__file__).resolve().parents[2] / "data" / "Hourly"
FEATURES = feat_list()
PRED_BATCH = 2000   # TabICL's in-context embedding buffer OOMs on an unbatched whole-domain
                    # predict for wide feature sets (D-79) -- harmless no-op below this size


def main():
    from tabicl import TabICLRegressor

    d = load_ext()
    print(f"Loaded EXT layer {d.shape}")

    out = pd.DataFrame(index=d.index); out.index.name = "Datetime"
    for t in TOWERS:
        g = frame(t, pooled=False, d=d)   # full raw hourly span -- NOT domain-restricted, so the
                                            # output covers exactly the same rows as
                                            # fch4_gapfilled.csv (drop-in compatibility is the
                                            # point of this file; DOMAIN is a gap-CV evaluation
                                            # concept, not a production-fill one -- the RFm
                                            # champion's own build_fch4_gapfilled.py fills the
                                            # full span the same way)
        dm = dom_mask(g.index, t)
        obs = g["target"].notna()
        trd = g.loc[dm & obs]   # train only on domain-restricted, QC'd real rows -- matches how
                                # TabICL-solo was actually evaluated (gapfill_tabicl.evaluate_tower)

        imp = SimpleImputer(strategy="mean")
        trd_sub = trd.sample(n=min(FOUNDATION_MODEL_ROW_CAP, len(trd)), random_state=42)
        m = TabICLRegressor(random_state=42)
        m.fit(imp.fit_transform(trd_sub[FEATURES].values), trd_sub["target"].values)

        X = imp.transform(g[FEATURES].values)
        pred = np.concatenate([m.predict(X[i:i + PRED_BATCH]) for i in range(0, len(X), PRED_BATCH)])
        filled = g["target"].where(obs, pd.Series(pred, index=g.index))

        out.loc[g.index, f"FCH4_gapfilled [Tower {t}]"] = filled.round(4)
        out.loc[g.index, f"FCH4_observed_mask [Tower {t}]"] = obs.astype(int)
        print(f"  Tower {t}: observed {int(obs.sum()):>6} -> gap-filled {int(filled.notna().sum()):>6} "
              f"({100*filled.notna().mean():.0f}%)  pred range [{pred.min():.1f}, {pred.max():.1f}]")

    dest = HOURLY / "fch4_gapfilled_tabicl.csv"
    out.to_csv(dest)
    print(f"\nWrote {dest}  ({out.shape[0]:,} rows x {out.shape[1]} cols)")


if __name__ == "__main__":
    main()
