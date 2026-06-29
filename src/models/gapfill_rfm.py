"""Shared F-08 RFm gap-filler (External SMS/MET sourcing, D-35).

Single source of truth for the partially-pooled RandomForest CH4 gap-filler used by
F-08 and by the forecasting precompute (`build_fch4_gapfilled.py`). Encapsulates the
per-tower config, the F-06/F-08 feature frame, and a fit helper, so the logic is not
copied a third time.

EXT (external) variant only: gap-filled met from `reddyproc_processed_SMS_MET.csv`
(SWIN/TA/WS = Site sources after the swap), soil temperature = per-catchment external,
FCO2 from `fco2_gapfilled.csv` (no external twin), management from `management_features.csv`.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

HOURLY = Path(__file__).resolve().parents[2] / "data" / "Hourly"

PLAUS_LOW, PLAUS_HIGH = -500, 3000
LSU = {"cattle": 1.0, "sheep": 0.1, "lamb": 0.05}
AUX = ["_hs", "_hc", "_ds", "_dc"]
LAG_HOURS = [168, 336, 504, 672]
AREA = {2: 6.65, 4: 7.75, 9: 7.75}
C4 = "Catchment 4 After  2013/08/13"
TOWERS = [2, 4, 9]


def ts_col_for(t):
    """EXT: own-catchment external soil temperature (D-35)."""
    cat = C4 if t == 4 else f"Catchment {t}"
    return f"Soil Temperature @ 15cm Depth (oC) [{cat}]"


def cfg(t, ts_col):
    cat = C4 if t == 4 else f"Catchment {t}"
    met = [f"SWIN_1_1_1 [Tower {t}]", f"TA_0_0_1 [Tower {t}]", f"VPD_0_0_1 [Tower {t}]",
           f"PPFD_1_1_1 [Tower {t}]", f"RN_1_1_1 [Tower {t}]", f"WS_0_0_1 [Tower {t}]",
           f"USTAR_0_0_1 [Tower {t}]", f"SHF_1_1_1 [Tower {t}]",
           f"Precipitation (mm) [{cat}]", ts_col, f"Soil Moisture @ 10cm Depth (%) [{cat}]"]
    return dict(t=t, cat=cat, area=AREA[t], tgt=f"FCH4_1_1_1 [Tower {t}]",
                ssitc=f"FCH4_SSITC_TEST_1_1_1 [Tower {t}]",
                sw=f"SWIN_1_1_1 [Tower {t}]", ta=f"TA_0_0_1 [Tower {t}]", met=met,
                fc=f"FC_1_1_1 [Tower {t}]", gpp=f"GPP [Tower {t}]", reco=f"Reco [Tower {t}]",
                swc=f"Soil Moisture @ 10cm Depth (%) [{cat}]", ts=ts_col,
                liv={s: f"{s}_{cat}" for s in LSU},
                mc=f"mgmt_t{t}_cut_recency", mm=f"mgmt_t{t}_manure_recency")


def feat_list():
    tc = ts_col_for(2)
    return ([m.split(" [")[0] for m in cfg(2, tc)["met"]] + ["fc"] + AUX + ["lsu_dens", "graze"]
            + [f"swc_l{l}" for l in LAG_HOURS] + [f"ts_l{l}" for l in LAG_HOURS]
            + ["mgmt_cut", "mgmt_manure", "gpp", "reco"])


DUM = ["is_t2", "is_t4", "is_t9"]


def load_ext():
    """Load the External SMS/MET data layer joined with FCO2 + management + REddyProc."""
    d = pd.read_csv(HOURLY / "consolidated_hourly_SMS_MET.csv", low_memory=False)
    d["Datetime"] = pd.to_datetime(d["Datetime"], format="mixed"); d = d.set_index("Datetime")
    for f in ["fco2_gapfilled.csv", "management_features.csv", "reddyproc_processed_SMS_MET.csv"]:
        e = pd.read_csv(HOURLY / f, low_memory=False)
        e["Datetime"] = pd.to_datetime(e["Datetime"], format="mixed")
        d = d.join(e.set_index("Datetime"), how="left")
    for t in TOWERS:
        d[f"FC_1_1_1 [Tower {t}]"] = d[f"FC_gapfilled [Tower {t}]"]   # FCO2 kept EC (no external twin)
    return d


def frame(t, pooled, d):
    """F-06/F-08 feature frame for tower t (EXT variant)."""
    c = cfg(t, ts_col_for(t)); d = d.copy(); tgt = c["tgt"]
    d.loc[~d[c["ssitc"]].isin([0, 1]), tgt] = np.nan
    d.loc[d[tgt].notna() & ~d[tgt].between(PLAUS_LOW, PLAUS_HIGH, inclusive="both"), tgt] = np.nan
    h, doy = d.index.hour, d.index.dayofyear
    d["_hs"] = np.sin(2 * np.pi * h / 24); d["_hc"] = np.cos(2 * np.pi * h / 24)
    d["_ds"] = np.sin(2 * np.pi * doy / 365); d["_dc"] = np.cos(2 * np.pi * doy / 365)
    g = pd.DataFrame(index=d.index); g["target"] = d[tgt]
    for k in c["met"]:
        nm = k.split(" [")[0]; g[nm] = d[k + "__f"] if (k + "__f") in d.columns else d[k]
    g["fc"] = d[c["fc"]]
    for a in AUX: g[a] = d[a]
    lsu = sum(d[c["liv"][s]].fillna(0) * w for s, w in LSU.items())
    g["lsu_dens"] = lsu / c["area"]
    g["graze"] = (sum(d[c["liv"][s]].fillna(0) for s in LSU) > 0).astype(float)
    swc = d[c["swc"] + "__f"] if (c["swc"] + "__f") in d.columns else d[c["swc"]]
    ts = d[c["ts"] + "__f"] if (c["ts"] + "__f") in d.columns else d[c["ts"]]
    for lag in LAG_HOURS:
        g[f"swc_l{lag}"] = swc.shift(lag); g[f"ts_l{lag}"] = ts.shift(lag)
    g["mgmt_cut"] = d[c["mc"]]; g["mgmt_manure"] = d[c["mm"]]
    g["gpp"] = d[c["gpp"]]; g["reco"] = d[c["reco"]]
    if pooled:
        for tt in TOWERS: g[f"is_t{tt}"] = 1.0 if tt == t else 0.0
    g["_y"] = d.index.year
    return g


def fit(feat, trd):
    imp = SimpleImputer(strategy="mean")
    rf = RandomForestRegressor(n_estimators=500, min_samples_leaf=5, n_jobs=-1, random_state=42)
    rf.fit(imp.fit_transform(trd[feat].values), trd["target"].values)
    return rf, imp
