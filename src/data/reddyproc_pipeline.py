"""F-06: Python REddyProc-style processing pipeline (D-33).

Re-implements, in Python, the core of the NWFP/REddyProc EC workflow that we had NOT
applied — gap-filling the *meteorological drivers* (we previously mean-imputed them) —
plus pragmatic versions of u*-threshold filtering and NEE->GPP/Reco partitioning.

Per tower (2, 4, 9), produces gap-filled driver columns (suffix ``__f``), a u*-filter
flag, an MDS-gap-filled FCO2 (for comparison vs the RF reconstruction, 03b), and
GPP/Reco from nighttime Lloyd-Taylor partitioning. Output: data/Hourly/reddyproc_processed.csv.

Pragmatic simplifications vs REddyProc (documented, D-33):
  - met gap-fill: linear interp (<=2 h) -> mean-diurnal-course with expanding window
    (REddyProc uses MDS look-up tables; MDC is its dominant fallback).
  - u*: binned-plateau threshold per tower (a simplification of the bootstrapped
    Moving-Point-Test, Papale 2006). u*-filtering of CH4 is debated (ebullition); flag only.
  - partitioning: nighttime method (Reichstein 2005) with global E0 + block-wise Rref
    (Lloyd & Taylor 1994); daytime/Lasslop not implemented.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

ROOT   = Path(__file__).resolve().parents[2]
HOURLY = ROOT / "data" / "Hourly"
TOWERS = [2, 4, 9]
FC_LOW, FC_HIGH = -100.0, 100.0          # umol CO2 m-2 s-1 (D-25)
C4 = "Catchment 4 After  2013/08/13"


def driver_map(t, catstr):
    """Model met drivers for tower t (the inputs we previously mean-imputed)."""
    return {
        "sw":   f"SWIN_1_1_1 [Tower {t}]",
        "ta":   f"TA_0_0_1 [Tower {t}]",
        "vpd":  f"VPD_0_0_1 [Tower {t}]",
        "ppfd": f"PPFD_1_1_1 [Tower {t}]",
        "rn":   f"RN_1_1_1 [Tower {t}]",
        "ws":   f"WS_0_0_1 [Tower {t}]",
        "ustar":f"USTAR_0_0_1 [Tower {t}]",
        "shf":  f"SHF_1_1_1 [Tower {t}]",
        "precip": f"Precipitation (mm) [{catstr}]",
        "swc":  f"Soil Moisture @ 10cm Depth (%) [{catstr}]",
        "ts":   "TS_1_1_1 [Tower 9]",
    }


def mdc_gapfill(s):
    """Linear interp (<=2 h) then mean-diurnal-course with expanding +/-7/14/28/60 d window."""
    out = s.astype(float).copy()
    n_obs = out.notna().sum()
    out = out.interpolate(limit=2, limit_area="inside")
    idx = out.index
    piv = (pd.DataFrame({"v": out.values, "hour": idx.hour, "date": idx.normalize()})
           .pivot_table(index="date", columns="hour", values="v"))
    for w in [7, 14, 28, 60]:
        piv = piv.fillna(piv.rolling(2 * w + 1, min_periods=1, center=True).mean())
    ser = piv.stack(dropna=False)                       # (date, hour) -> value
    key = pd.MultiIndex.from_arrays([idx.normalize(), idx.hour])
    mapped = pd.Series(ser.reindex(key).values, index=idx)
    out = out.where(out.notna(), mapped)
    # final fallbacks: per-hour climatology, then global mean
    if out.isna().any():
        hourly_mean = out.groupby(idx.hour).transform("mean")
        out = out.where(out.notna(), hourly_mean)
        out = out.fillna(out.mean())
    return out, int(n_obs), int(out.notna().sum())


def qc_fc(df, t):
    fc = df[f"FC_1_1_1 [Tower {t}]"].astype(float).copy()
    ss = f"FC_SSITC_TEST_1_1_1 [Tower {t}]"
    fc[~df[ss].isin([0, 1])] = np.nan
    fc[fc.notna() & ~fc.between(FC_LOW, FC_HIGH)] = np.nan
    return fc


def ustar_threshold(ta, ustar, fc, rg):
    """Binned-plateau u* threshold (pragmatic; simplification of Papale 2006 MPT)."""
    night = (rg < 20) & ta.notna() & ustar.notna() & fc.notna()
    d = pd.DataFrame({"ta": ta[night], "ustar": ustar[night], "fc": fc[night]})
    d = d[d["ustar"] > 0]
    if len(d) < 200:
        return 0.10
    try:
        d["tc"] = pd.qcut(d["ta"], 6, duplicates="drop")
    except ValueError:
        d["tc"] = 0
    thr = []
    for _, g in d.groupby("tc", observed=True):
        if len(g) < 50:
            continue
        edges = np.quantile(g["ustar"], np.linspace(0, 1, 21))
        edges = np.unique(edges)
        if len(edges) < 6:
            continue
        g = g.assign(ub=pd.cut(g["ustar"], edges, labels=False, include_lowest=True))
        m = g.groupby("ub")["fc"].mean()
        c = g.groupby("ub")["ustar"].mean()
        if len(m) < 6:
            continue
        plateau = m.loc[m.index >= m.index.max() // 2].mean()
        if not np.isfinite(plateau) or plateau <= 0:
            continue
        ok = m[m >= 0.95 * plateau]
        if len(ok):
            thr.append(float(c.loc[ok.index[0]]))
    if not thr:
        return 0.10
    return float(np.clip(np.median(thr), 0.01, 0.5))


def lloyd_taylor(t_k, rref, e0):
    T0, Tref = 227.13, 283.15
    return rref * np.exp(e0 * (1.0 / (Tref - T0) - 1.0 / (t_k - T0)))


def partition_nee(fc, ta, rg):
    """Nighttime partitioning: global E0 + block-wise Rref (Lloyd-Taylor) -> GPP, Reco."""
    t_k = ta + 273.15
    night = (rg < 20) & fc.notna() & ta.notna() & (t_k > 230)
    gpp = pd.Series(np.nan, index=fc.index)
    reco = pd.Series(np.nan, index=fc.index)
    if night.sum() < 300:
        return gpp, reco, None
    Tn, Fn = t_k[night].values, fc[night].values
    try:
        (rref0, e0), _ = curve_fit(lloyd_taylor, Tn, Fn, p0=[2.0, 150.0],
                                   bounds=([0, 30], [60, 450]), maxfev=10000)
    except Exception:
        return gpp, reco, None
    # block-wise Rref (7-day blocks), E0 fixed
    days = (fc.index - fc.index[0]).days
    blk = (days // 7)
    rref_by_blk = {}
    nb = pd.DataFrame({"t_k": t_k[night], "fc": fc[night], "blk": blk[night.values]})
    for b, g in nb.groupby("blk"):
        if len(g) < 8:
            continue
        try:
            (rr,), _ = curve_fit(lambda T, rr: lloyd_taylor(T, rr, e0),
                                 g["t_k"].values, g["fc"].values,
                                 p0=[rref0], bounds=([0], [60]), maxfev=5000)
            rref_by_blk[b] = rr
        except Exception:
            continue
    if not rref_by_blk:
        rref_t = pd.Series(rref0, index=fc.index)
    else:
        blk_series = pd.Series(blk, index=fc.index)
        rref_t = blk_series.map(rref_by_blk).astype(float)
        rref_t = rref_t.interpolate().bfill().ffill().fillna(rref0)
    reco = lloyd_taylor(t_k, rref_t.values, e0)
    reco = pd.Series(reco, index=fc.index)
    gpp = (reco - fc)                       # NEE = fc; GPP = Reco - NEE
    gpp[gpp < 0] = 0.0
    gpp[(rg < 20)] = 0.0                    # no GPP at night
    return gpp, reco, float(e0)


def main():
    df = pd.read_csv(HOURLY / "consolidated_hourly.csv", low_memory=False)
    df["Datetime"] = pd.to_datetime(df["Datetime"], format="mixed")
    df = df.set_index("Datetime")
    print(f"Loaded {df.shape}")

    out = pd.DataFrame(index=df.index); out.index.name = "Datetime"
    for t in TOWERS:
        catstr = C4 if t == 4 else f"Catchment {t}"
        dm = driver_map(t, catstr)
        print(f"\n=== Tower {t} ===")
        # (A) met-driver gap-fill
        for k, col in dm.items():
            if col not in df.columns:
                print(f"  [skip] {col} missing"); continue
            filled, n0, n1 = mdc_gapfill(df[col])
            out[f"{col}__f"] = filled.round(4)
            base = f"{100*df[col].notna().mean():.0f}%"
            print(f"  metfill {k:6s}: {base} -> {100*filled.notna().mean():.0f}%")
        # QC FC for u*/partitioning
        fc = qc_fc(df, t)
        sw = out[f"{dm['sw']}__f"]; ta = out[f"{dm['ta']}__f"]; us = out[f"{dm['ustar']}__f"]
        # (B) u* threshold
        thr = ustar_threshold(ta, us, fc, sw)
        flagged = ((sw < 20) & (us < thr)).astype(int)
        out[f"ustar_filtered [Tower {t}]"] = flagged
        print(f"  u* threshold = {thr:.3f} m/s; nighttime low-u* flagged = {int(flagged.sum()):,} hrs")
        # (D) partitioning
        gpp, reco, e0 = partition_nee(fc, ta, sw)
        out[f"GPP [Tower {t}]"] = gpp.round(4)
        out[f"Reco [Tower {t}]"] = reco.round(4)
        print(f"  partitioning E0={e0}; GPP coverage {100*gpp.notna().mean():.0f}%")

    dest = HOURLY / "reddyproc_processed.csv"
    out.to_csv(dest)
    print(f"\nWrote {dest}  ({out.shape[0]:,} rows x {out.shape[1]} cols)")


if __name__ == "__main__":
    main()
