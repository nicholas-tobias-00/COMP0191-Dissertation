"""Stage 0a (forecasting, D-36): continuous gap-filled CH4 per tower.

Produces a regularly-sampled CH4 series for each tower so the forecasting phase has a
continuous target + autoregressive features. Uses the F-08 best partially-pooled RFm
gap-filler on the **External SMS/MET** data layer (D-35).

For each tower t: train one pooled RFm on ALL QC'd observed CH4 across towers (tower
dummies), predict every timestamp of t. Output = observed where genuinely measured,
else the RFm prediction, plus an observed-mask column.

Caveat (documented): the gap-filler is globally trained, so gap-filled values later used
as AR features carry minor optimism — an accepted preprocessing simplification; the
forecasting models are evaluated on observed timestamps only.

Output: data/Hourly/fch4_gapfilled.csv  (70,153 rows)
Run from project root:  python src/data/build_fch4_gapfilled.py
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
from gapfill_rfm import (  # noqa: E402  single source of truth
    load_ext, frame, fit, feat_list, DUM, TOWERS,
)

HOURLY = Path(__file__).resolve().parents[2] / "data" / "Hourly"


def main():
    d = load_ext()
    print(f"Loaded EXT layer {d.shape}")
    feat = feat_list() + DUM
    frames = {t: frame(t, pooled=True, d=d) for t in TOWERS}

    out = pd.DataFrame(index=d.index); out.index.name = "Datetime"
    for t in TOWERS:
        g = frames[t]
        obs = g["target"].notna()
        # pooled training: this tower's observed + the other towers' observed rows
        trd = pd.concat([frames[tt][frames[tt]["target"].notna()] for tt in TOWERS],
                        ignore_index=True)
        rf, imp = fit(feat, trd)
        pred = rf.predict(imp.transform(g[feat].values))
        filled = g["target"].where(obs, pd.Series(pred, index=g.index))
        out[f"FCH4_gapfilled [Tower {t}]"] = filled.round(4)
        out[f"FCH4_observed_mask [Tower {t}]"] = obs.astype(int)
        print(f"  Tower {t}: observed {int(obs.sum()):>6} -> gap-filled {int(filled.notna().sum()):>6} "
              f"({100*filled.notna().mean():.0f}%)  pred range [{pred.min():.1f}, {pred.max():.1f}]")

    dest = HOURLY / "fch4_gapfilled.csv"
    out.to_csv(dest)
    print(f"\nWrote {dest}  ({out.shape[0]:,} rows x {out.shape[1]} cols)")


if __name__ == "__main__":
    main()
