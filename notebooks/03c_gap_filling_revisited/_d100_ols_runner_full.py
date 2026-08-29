# ===== CELL 2 (id=ec998827) =====
from pathlib import Path
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score

pd.set_option("display.max_columns", 80)
pd.set_option("display.width", 160)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["figure.facecolor"] = "white"

HOURLY = Path("../../data/Hourly")
COMPILED = Path("../../data/Compiled")
RESULTS = Path("../../results")

TOWERS = [2, 4, 9]
C4 = "Catchment 4 After  2013/08/13"           # Tower 4's catchment column suffix (note double space)
CAT = {2: "Catchment 2", 4: C4, 9: "Catchment 9"}   # Tower N = Catchment N (D-18)
AREA = {2: 6.65, 4: 7.75, 9: 7.75}             # fenced catchment area, hectares (Appendix D)

print("Setup complete.")

# ===== CELL 4 (id=2375d7d7) =====
raw = pd.read_csv(HOURLY / "consolidated_hourly.csv", low_memory=False)
raw["Datetime"] = pd.to_datetime(raw["Datetime"], format="mixed")
raw = raw.set_index("Datetime")

print(f"Shape: {raw.shape[0]:,} rows x {raw.shape[1]} cols")
print(f"Span:  {raw.index.min()} -> {raw.index.max()}")
raw.iloc[:3, :6]

# ===== CELL 6 (id=141e44a2) =====
full_range = pd.date_range(raw.index.min(), raw.index.max(), freq="h")
print("Monotonic increasing:      ", raw.index.is_monotonic_increasing)
print("Duplicate timestamps:      ", raw.index.duplicated().sum())
print("Missing hourly timestamps: ", len(full_range.difference(raw.index)))
print("Overall NaN share:         ", f"{100 * raw.isna().mean().mean():.1f}%")

# ===== CELL 8 (id=f6043f44) =====
for t in TOWERS:
    tgt, ssitc = f"FCH4_1_1_1 [Tower {t}]", f"FCH4_SSITC_TEST_1_1_1 [Tower {t}]"
    valid_pct = 100 * raw[tgt].notna().mean()
    flag_counts = raw[ssitc].value_counts(dropna=False).sort_index()
    print(f"Tower {t}: {valid_pct:5.1f}% raw-valid FCH4  |  SSITC flag counts: {flag_counts.to_dict()}")

# ===== CELL 10 (id=15722ecf) =====
PLAUS_LOW, PLAUS_HIGH = -500, 3000   # nmol m-2 s-1 plausibility bound for FCH4 (D-13)

fig, axes = plt.subplots(1, 3, figsize=(13, 3.2))
for ax, t in zip(axes, TOWERS):
    tgt, ssitc = f"FCH4_1_1_1 [Tower {t}]", f"FCH4_SSITC_TEST_1_1_1 [Tower {t}]"
    qc = raw[tgt].where(raw[ssitc].isin([0, 1]))
    n_extreme = int((qc.notna() & ~qc.between(PLAUS_LOW, PLAUS_HIGH)).sum())
    print(f"Tower {t}: SSITC-passed range [{qc.min():.0f}, {qc.max():.0f}] nmol m-2 s-1, "
          f"{n_extreme} points outside [{PLAUS_LOW}, {PLAUS_HIGH}]")
    ax.hist(qc.clip(-1000, 3000).dropna(), bins=80, color="#4C72B0")
    ax.axvline(PLAUS_LOW, color="crimson", ls="--", lw=1); ax.axvline(PLAUS_HIGH, color="crimson", ls="--", lw=1)
    ax.set_title(f"Tower {t}"); ax.set_xlabel("FCH4 (nmol m-2 s-1)")
plt.suptitle("SSITC-passed FCH4 distribution, with the [-500, 3000] plausibility band")
plt.tight_layout(); plt.show()

# ===== CELL 12 (id=c0be42c4) =====
for t in TOWERS:
    cat = CAT[t]
    swc_col = f"Soil Moisture @ 10cm Depth (%) [{cat}]"
    ts_ec_col = f"TS_1_1_1 [Tower {t}]"                          # on-tower EC soil temp sensor
    ts_ext_col = f"Soil Temperature @ 15cm Depth (oC) [{cat}]"   # own-catchment external SMS sensor
    print(f"Tower {t} <-> {cat}")
    print(f"  own-catchment soil moisture (10cm, external): {100*raw[swc_col].notna().mean():5.1f}% coverage")
    print(f"  on-tower EC soil temperature (TS_1_1_1):      {100*raw[ts_ec_col].notna().mean():5.1f}% coverage")
    print(f"  own-catchment soil temperature (external):    {100*raw[ts_ext_col].notna().mean():5.1f}% coverage")

# ===== CELL 14 (id=3a37087b) =====
SITE_SW = "Solar Radiation (W/m2) [Site]"
SITE_TA = "Air Temperature (oC) [Site]"
SITE_WS = "Wind Speed (km/h) [Site]"

for t in TOWERS:
    for ec_col, site_col, label in [(f"SWIN_1_1_1 [Tower {t}]", SITE_SW, "SWIN/solar"),
                                     (f"TA_0_0_1 [Tower {t}]", SITE_TA, "TA/air temp"),
                                     (f"WS_0_0_1 [Tower {t}]", SITE_WS, "WS/wind")]:
        print(f"Tower {t} {label:10s}: EC-tower {100*raw[ec_col].notna().mean():5.1f}%  |  "
              f"Site-external {100*raw[site_col].notna().mean():5.1f}%")

# ===== CELL 16 (id=a6603825) =====
# Met-driver plausibility bounds (D-48). Only USTAR and VPD show real outlier contamination
# across all towers (audited); PPFD/WS/SHF were checked and are clean.
MET_PLAUS = {"USTAR_0_0_1": (0.0, 3.0), "VPD_0_0_1": (0.0, 15.0)}

for base, (lo, hi) in MET_PLAUS.items():
    print(f"--- {base} plausibility bound [{lo}, {hi}] ---")
    for t in TOWERS:
        s = raw[f"{base} [Tower {t}]"].astype(float)
        n_bad = int((s.notna() & ~s.between(lo, hi)).sum())
        pct_bad = 100 * n_bad / s.notna().sum() if s.notna().sum() else np.nan
        print(f"  Tower {t}: n={s.notna().sum():6,}  mean={s.mean():7.3f}  median={s.median():7.3f}  "
              f"max={s.max():10.1f}  |  {n_bad} points ({pct_bad:.1f}%) outside bound")

# ===== CELL 18 (id=33b43c4e) =====
tgt2, ssitc2 = "FCH4_1_1_1 [Tower 2]", "FCH4_SSITC_TEST_1_1_1 [Tower 2]"
qc2 = raw[tgt2].where(raw[ssitc2].isin([0, 1]))
by_year_valid = qc2.notna().groupby(raw.index.year).sum()
by_year_mean = qc2.groupby(raw.index.year).mean()
cattle2 = raw[f"cattle_{CAT[2]}"]
by_year_cattle = cattle2.groupby(raw.index.year).mean()

summary = pd.DataFrame({"n_valid_FCH4": by_year_valid, "mean_FCH4": by_year_mean.round(1),
                         "mean_cattle_head": by_year_cattle.round(1)})
print("Tower 2 (Catchment 2) by year:")
print(summary.loc[summary["n_valid_FCH4"] > 0].to_string())

# ===== CELL 20 (id=db66b542) =====
FC_LOW, FC_HIGH = -100.0, 100.0   # umol CO2 m-2 s-1 plausibility bound (D-25)

for t in TOWERS:
    fc, ssitc = f"FC_1_1_1 [Tower {t}]", f"FC_SSITC_TEST_1_1_1 [Tower {t}]"
    qc_fc_raw = raw[fc].where(raw[ssitc].isin([0, 1]))
    n_bad = int((qc_fc_raw.notna() & ~qc_fc_raw.between(FC_LOW, FC_HIGH)).sum())
    print(f"Tower {t}: raw {100*raw[fc].notna().mean():5.1f}% valid, "
          f"SSITC-passed {100*qc_fc_raw.notna().mean():5.1f}% valid, "
          f"{n_bad} points outside [{FC_LOW}, {FC_HIGH}] umol m-2 s-1")

# ===== CELL 23 (id=30d3c461) =====
# driver_m (met-only) per tower -- identical to R-02 (D-21); Tower 2 uses its only usable years (D-15).
FCO2_AUX = ["_hour_sin", "_hour_cos", "_doy_sin", "_doy_cos"]

FCO2_TOWER_CONFIGS = {
    2: {"fc": "FC_1_1_1 [Tower 2]", "ssitc": "FC_SSITC_TEST_1_1_1 [Tower 2]",
        "driver_m": ["SWIN_1_1_1 [Tower 2]", "TA_0_0_1 [Tower 2]", "VPD_0_0_1 [Tower 2]",
                     "PPFD_1_1_1 [Tower 2]", "USTAR_0_0_1 [Tower 2]", "WS_0_0_1 [Tower 2]",
                     "RN_1_1_1 [Tower 2]", "Precipitation (mm) [Catchment 2]",
                     "TS_1_1_1 [Tower 9]", "Soil Moisture @ 10cm Depth (%) [Catchment 2]",
                     "SHF_1_1_1 [Tower 2]"],
        "train_yrs": [2018], "test_yrs": [2019]},
    4: {"fc": "FC_1_1_1 [Tower 4]", "ssitc": "FC_SSITC_TEST_1_1_1 [Tower 4]",
        "driver_m": [f"SWIN_1_1_1 [Tower 4]", "TA_0_0_1 [Tower 4]", "VPD_0_0_1 [Tower 4]",
                     "PPFD_1_1_1 [Tower 4]", "USTAR_0_0_1 [Tower 4]", "WS_0_0_1 [Tower 4]",
                     "RN_1_1_1 [Tower 4]", f"Precipitation (mm) [{C4}]",
                     "TS_1_1_1 [Tower 9]", f"Soil Moisture @ 10cm Depth (%) [{C4}]",
                     "SHF_1_1_1 [Tower 4]"],
        "train_yrs": list(range(2018, 2022)), "test_yrs": list(range(2022, 2024))},
    9: {"fc": "FC_1_1_1 [Tower 9]", "ssitc": "FC_SSITC_TEST_1_1_1 [Tower 9]",
        "driver_m": ["SWIN_1_1_1 [Tower 9]", "TA_0_0_1 [Tower 9]", "VPD_0_0_1 [Tower 9]",
                     "PPFD_1_1_1 [Tower 9]", "USTAR_0_0_1 [Tower 9]", "WS_0_0_1 [Tower 9]",
                     "RN_1_1_1 [Tower 9]", "Precipitation (mm) [Catchment 9]",
                     "TS_1_1_1 [Tower 9]", "Soil Moisture @ 10cm Depth (%) [Catchment 9]",
                     "SHF_1_1_1 [Tower 9]"],
        "train_yrs": list(range(2018, 2022)), "test_yrs": list(range(2022, 2024))},
}


def add_cyclical(df):
    hour, doy = df.index.hour, df.index.dayofyear
    df["_hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["_hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["_doy_sin"] = np.sin(2 * np.pi * doy / 365)
    df["_doy_cos"] = np.cos(2 * np.pi * doy / 365)


def reconstruct_fc(df, tower, cfg):
    """RF-reconstruct a complete FC series for one tower; return (fc_gapfilled, stats)."""
    fc_col, ssitc_col, feats = cfg["fc"], cfg["ssitc"], cfg["driver_m"] + FCO2_AUX

    fc_obs = df[fc_col].copy()
    fc_obs[~df[ssitc_col].isin([0, 1])] = np.nan
    fc_obs[fc_obs.notna() & ~fc_obs.between(FC_LOW, FC_HIGH)] = np.nan

    train_mask = df.index.year.isin(cfg["train_yrs"]) & fc_obs.notna()
    imputer = SimpleImputer(strategy="mean")
    X_train = imputer.fit_transform(df.loc[train_mask, feats].values)
    y_train = fc_obs.loc[train_mask].values

    rf = RandomForestRegressor(n_estimators=500, min_samples_leaf=5, n_jobs=1, random_state=42)
    # n_jobs=1, not -1: parallel RF training is NOT bit-reproducible across separate process
    # runs even with random_state fixed (confirmed empirically) -- FCO2 is the one RF step not
    # itself cached, so its cross-run drift would otherwise cascade into every downstream
    # cache key (via the "fc" feature) and silently defeat the section 11.0 model cache.
    rf.fit(X_train, y_train)

    X_all = imputer.transform(df[feats].values)
    fc_recon = pd.Series(rf.predict(X_all), index=df.index)

    fc_gapfilled = fc_obs.copy()
    fc_gapfilled[fc_gapfilled.isna()] = fc_recon[fc_gapfilled.isna()]

    test_mask = df.index.year.isin(cfg["test_yrs"]) & fc_obs.notna()
    if test_mask.sum() >= 20:
        val_r2 = r2_score(fc_obs.loc[test_mask].values, fc_recon.loc[test_mask].values)
    else:
        val_r2 = np.nan
    stats = {"tower": tower, "n_train": int(train_mask.sum()),
              "n_filled_by_recon": int(fc_obs.isna().sum()), "recon_test_r2": val_r2,
              "recon_test_n": int(test_mask.sum())}
    return fc_gapfilled, stats


add_cyclical(raw)
fco2_gapfilled = pd.DataFrame(index=raw.index)
fco2_stats = []
for t in TOWERS:
    filled, stats = reconstruct_fc(raw, t, FCO2_TOWER_CONFIGS[t])
    fco2_gapfilled[f"FC_gapfilled [Tower {t}]"] = filled
    fco2_stats.append(stats)
    print(f"Tower {t}: n_train={stats['n_train']:,}  filled-by-reconstruction={stats['n_filled_by_recon']:,}  "
          f"recon test R2={stats['recon_test_r2']:.3f} (n={stats['recon_test_n']:,})")

# ===== CELL 25 (id=1537f3b7) =====
df_ext = raw.copy()
for t in TOWERS:
    for ec_col, site_col, conv in [(f"SWIN_1_1_1 [Tower {t}]", SITE_SW, 1.0),
                                    (f"TA_0_0_1 [Tower {t}]", SITE_TA, 1.0),
                                    (f"WS_0_0_1 [Tower {t}]", SITE_WS, 1.0 / 3.6)]:   # km/h -> m/s
        src = pd.to_numeric(raw[site_col], errors="coerce") * conv
        n_ec, n_ext = raw[ec_col].notna().sum(), src.notna().sum()
        df_ext[ec_col] = src
        print(f"  T{t} {ec_col.split(' [')[0]:12s} <- {site_col:28s}  "
              f"cov {100*n_ec/len(raw):.0f}% -> {100*n_ext/len(raw):.0f}%")

print(f"\ndf_ext shape: {df_ext.shape}")

# ===== CELL 27 (id=75f89b92) =====
def plausibility_filter(s, colname):
    """Set values outside MET_PLAUS[colname] to NaN before gap-filling (D-48). No-op if
    colname has no bound registered. Applied ahead of mdc_gapfill so contaminated raw
    outliers never enter the interpolation/MDC/fallback chain in the first place."""
    base = colname.split(" [")[0]
    if base not in MET_PLAUS:
        return s
    lo, hi = MET_PLAUS[base]
    s = s.astype(float).copy()
    s[s.notna() & ~s.between(lo, hi, inclusive="both")] = np.nan
    return s


# Variables where mdc_gapfill's per-hour-of-day pivot averaging is a poor fit: smooth,
# slowly-drifting signals with weak diurnal structure. Found by the section 6.4/16.1 diagnostic --
# plain linear interpolation *beats* the MDC pipeline on long (288h/12-day) gaps for these 4
# variables, consistently across all 3 towers (mdc_l vs interp_l e.g. swc: -0.14/0.97, ts:
# -0.01/0.93, ta: 0.14/0.41, vpd: 0.05/0.56). The reverse holds for radiation-driven variables
# (SWIN/PPFD/RN) and SHF/WS/precip, where a straight line across a multi-day gap misses the
# day/night cycle entirely (interp R2 as low as -1.15 there) -- those keep the original MDC-first
# behavior. USTAR is a wash either way (near-zero/negative for both) and is also left unchanged.
LONG_INTERP_VARS = {"TA_0_0_1", "VPD_0_0_1", "Soil Moisture @ 10cm Depth (%)",
                     "Soil Temperature @ 15cm Depth (oC)", "WS_0_0_1"}
# WS_0_0_1 added after section 6.4's forced-interp confirmation check found a consistent,
# non-trivial overall win (0.199 -> 0.458, all 3 towers) missed by the original l-scenario-only
# diagnostic -- l-scenario alone still favors mdc for wind speed (0.004 vs -0.143), but the
# OVERALL (5-scenario median) metric, which is what actually determines met_filled quality, favors
# interpolation once the shorter scenarios are included.


def mdc_gapfill(s, colname=None, fallback="median"):
    """Linear interp (<=2h, or <=288h for LONG_INTERP_VARS) then mean-diurnal-course with
    expanding +/-7/14/28/60d window, then a per-hour and finally a global fallback (median by
    default, D-48; mean pre-fix)."""
    out = s.astype(float).copy()
    n_obs = out.notna().sum()
    interp_limit = 288 if (colname is not None and colname.split(" [")[0] in LONG_INTERP_VARS) else 2
    out = out.interpolate(limit=interp_limit, limit_area="inside")
    idx = out.index
    piv = (pd.DataFrame({"v": out.values, "hour": idx.hour, "date": idx.normalize()})
           .pivot_table(index="date", columns="hour", values="v"))
    for w in [7, 14, 28, 60]:
        piv = piv.fillna(piv.rolling(2 * w + 1, min_periods=1, center=True).mean())
    ser = piv.stack(dropna=False)
    key = pd.MultiIndex.from_arrays([idx.normalize(), idx.hour])
    mapped = pd.Series(ser.reindex(key).values, index=idx)
    out = out.where(out.notna(), mapped)
    if out.isna().any():
        agg = "median" if fallback == "median" else "mean"
        hourly_fb = out.groupby(idx.hour).transform(agg)
        out = out.where(out.notna(), hourly_fb)
        out = out.fillna(out.median() if fallback == "median" else out.mean())
    return out, int(n_obs), int(out.notna().sum())

# ===== CELL 29 (id=ef9130f9) =====
ustar2_raw = df_ext["USTAR_0_0_1 [Tower 2]"]
blackout = slice("2019-06-01", "2019-12-31")

before, *_ = mdc_gapfill(ustar2_raw, fallback="mean")                                   # bug reproduced: no filter, mean fallback
after, *_ = mdc_gapfill(plausibility_filter(ustar2_raw, "USTAR_0_0_1"), fallback="median")  # the fix

print(f"Blackout-window USTAR__f, BEFORE fix: mean={before.loc[blackout].mean():.3f} m/s "
      f"(vs. sane range roughly 0-1 m/s)")
print(f"Blackout-window USTAR__f, AFTER  fix: mean={after.loc[blackout].mean():.3f} m/s")

fig, ax = plt.subplots(figsize=(11, 3))
ax.plot(before.loc["2019"], label="before fix (unfiltered, mean fallback)", color="crimson", lw=0.8)
ax.plot(after.loc["2019"], label="after fix (plausibility-filtered, median fallback)", color="#4C72B0", lw=0.8)
ax.axvspan(pd.Timestamp("2019-06-01"), pd.Timestamp("2019-12-31"), color="grey", alpha=0.15, label="total blackout")
ax.set_ylabel("USTAR__f (m/s)"); ax.set_title("Tower 2 gap-filled USTAR, 2019 -- D-48 fix"); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()

# ===== CELL 31 (id=b28e5baa) =====
def ts_col_for(t):
    """Own-catchment external soil temperature (D-35), replacing the old Tower-9 EC proxy (D-16)."""
    return f"Soil Temperature @ 15cm Depth (oC) [{CAT[t]}]"


def driver_map(t):
    cat = CAT[t]
    return {"sw": f"SWIN_1_1_1 [Tower {t}]", "ta": f"TA_0_0_1 [Tower {t}]",
            "vpd": f"VPD_0_0_1 [Tower {t}]", "ppfd": f"PPFD_1_1_1 [Tower {t}]",
            "rn": f"RN_1_1_1 [Tower {t}]", "ws": f"WS_0_0_1 [Tower {t}]",
            "ustar": f"USTAR_0_0_1 [Tower {t}]", "shf": f"SHF_1_1_1 [Tower {t}]",
            "precip": f"Precipitation (mm) [{cat}]",
            "swc": f"Soil Moisture @ 10cm Depth (%) [{cat}]", "ts": ts_col_for(t)}


met_filled = pd.DataFrame(index=df_ext.index)
for t in TOWERS:
    print(f"=== Tower {t} ===")
    for k, col in driver_map(t).items():
        filled, n0, n1 = mdc_gapfill(plausibility_filter(df_ext[col], col), colname=col)
        met_filled[f"{col}__f"] = filled.round(4)
        print(f"  {k:6s}: {100*df_ext[col].notna().mean():5.1f}% -> {100*filled.notna().mean():5.1f}%  [{col}]")

# ===== CELL 33 (id=1b60c829) =====
MASK_FRAC = 0.25
SCENARIOS = {"vs": 1, "s": 4, "m": 32, "l": 288, "m1": "mixed"}
N_REPS = 2   # matches F-09a's exact scope, not F-08's original 5 -- see section 11 for the full story


def insert_calendar_gaps(df_qc, target, domain_mask, gap_hours, n_reps=N_REPS, seed=0):
    """Randomly place n_reps independent sets of non-overlapping calendar-shaped gaps within the
    domain, each covering ~MASK_FRAC of the domain's valid target points."""
    dom_ts = df_qc.index[domain_mask]; valid = df_qc.loc[domain_mask, target].notna().values
    n = len(dom_ts); target_n = max(1, int(valid.sum() * MASK_FRAC)); rb = np.random.default_rng(seed); reps = []
    for _ in range(n_reps):
        rng = np.random.default_rng(int(rb.integers(0, 2**31))); occ = np.zeros(n, bool); m = 0
        for sp in rng.permutation(n):
            if m >= target_n:
                break
            gh = int(rng.choice([1, 4, 32, 288])) if gap_hours == "mixed" else gap_hours
            ep = min(int(sp) + gh, n)
            if occ[sp:ep].any():
                continue
            occ[sp:ep] = True; m += int(valid[sp:ep].sum())
        reps.append(dom_ts[occ & valid])
    return reps


def mets(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    r2 = r2_score(y, p) if np.var(y) > 0 else np.nan
    return r2, float(np.sqrt(np.mean((p - y) ** 2))), float(np.mean(np.abs(p - y))), float(np.mean(p - y))


def med_metrics(rows):
    if not rows:
        return {k: np.nan for k in ["R2", "RMSE", "MAE", "MBE"]}
    a = np.array(rows, float)
    return {"R2": np.nanmedian(a[:, 0]), "RMSE": np.median(a[:, 1]),
            "MAE": np.median(a[:, 2]), "MBE": np.median(a[:, 3])}

print("Shared gap-testing toolkit defined (reused in sections 6.4 and 11).")

# ===== CELL 35 (id=8165585f) =====
def eval_driver_gapfill(colname):
    """Full-period gap-CV for one gap-filled driver: mdc_gapfill()'s own reconstruction, AND a
    naive constant-median baseline (median of the remaining real domain values, per fold) scored
    on the exact same held-out points -- a direct check of whether mdc_gapfill actually beats the
    simplest possible imputation, not just whether its R2 crosses zero (R2=0 is "as good as the
    MEAN", not the median -- for skewed variables like USTAR/precipitation those can differ)."""
    s_raw = df_ext[colname]
    valid_idx = s_raw.index[s_raw.notna()]
    dm = (s_raw.index >= valid_idx[0]) & (s_raw.index <= valid_idx[-1])
    dfq = pd.DataFrame({colname: s_raw})
    out, out_base = {}, {}
    for sc, gh in SCENARIOS.items():
        rr, rr_base = [], []
        for gt in insert_calendar_gaps(dfq, colname, dm, gh, n_reps=N_REPS):
            if len(gt) < 5:
                continue
            sav = s_raw.loc[gt].copy()
            s_masked = s_raw.copy(); s_masked.loc[gt] = np.nan
            filled, _, _ = mdc_gapfill(plausibility_filter(s_masked, colname), colname=colname)
            rr.append(mets(sav.values, filled.loc[gt].values))
            train_median = s_masked.loc[dm].median()
            rr_base.append(mets(sav.values, np.full(len(gt), train_median)))
        out[sc] = med_metrics(rr)
        out_base[sc] = med_metrics(rr_base)
    return out, out_base


driver_rows = []
for t in TOWERS:
    for var_key, colname in driver_map(t).items():
        scn, scn_base = eval_driver_gapfill(colname)
        ov = {k: np.nanmedian([scn[s][k] for s in SCENARIOS]) for k in ["R2", "RMSE", "MAE", "MBE"]}
        ov_base = {k: np.nanmedian([scn_base[s][k] for s in SCENARIOS]) for k in ["R2", "RMSE", "MAE", "MBE"]}
        driver_rows.append({"tower": t, "driver": var_key, "column": colname,
                             **{f"{k}_overall": round(ov[k], 3) for k in ov},
                             "R2_median_baseline": round(ov_base["R2"], 3),
                             "RMSE_median_baseline": round(ov_base["RMSE"], 3),
                             "beats_median_baseline": ov["R2"] > ov_base["R2"],
                             **{f"R2_{s}": round(scn[s]["R2"], 3) if pd.notna(scn[s]["R2"]) else np.nan
                                for s in SCENARIOS}})
        beats = "beats" if ov["R2"] > ov_base["R2"] else "LOSES TO"
        print(f"Tower {t} {var_key:6s}: mdc_gapfill R2={ov['R2']:+.3f}  vs  median-baseline R2={ov_base['R2']:+.3f}"
              f"  ({beats} the naive median baseline)")

MET_GAPFILL_R2 = pd.DataFrame(driver_rows)
MET_GAPFILL_R2

# ===== CELL 37 (id=36e24f63) =====
def eval_forced_interp(colname):
    """Same held-out mechanism as eval_driver_gapfill, but always plain linear interpolation
    (plausibility-filtered first) instead of mdc_gapfill -- for checking variables NOT in
    LONG_INTERP_VARS, to confirm they were correctly left on the MDC-first path."""
    s_raw = df_ext[colname]
    valid_idx = s_raw.index[s_raw.notna()]
    dm = (s_raw.index >= valid_idx[0]) & (s_raw.index <= valid_idx[-1])
    dfq = pd.DataFrame({colname: s_raw})
    out = {}
    for sc, gh in SCENARIOS.items():
        rr = []
        for gt in insert_calendar_gaps(dfq, colname, dm, gh, n_reps=N_REPS):
            if len(gt) < 5:
                continue
            s_masked = s_raw.copy(); s_masked.loc[gt] = np.nan
            filled = plausibility_filter(s_masked, colname).interpolate(method="linear", limit_direction="both")
            rr.append(mets(s_raw.loc[gt].values, filled.loc[gt].values))
        out[sc] = med_metrics(rr)
    return out


other_keys = [k for k in driver_map(4).keys() if k not in ("ta", "vpd", "swc", "ts")]
check_rows = []
for t in TOWERS:
    dm_map = driver_map(t)
    for key in other_keys:
        colname = dm_map[key]
        interp_scn = eval_forced_interp(colname)
        interp_ov = np.nanmedian([interp_scn[s]["R2"] for s in SCENARIOS])
        prod_r2 = MET_GAPFILL_R2.loc[(MET_GAPFILL_R2.tower == t) & (MET_GAPFILL_R2.driver == key),
                                      "R2_overall"].iloc[0]
        would_help = pd.notna(interp_ov) and interp_ov > prod_r2
        check_rows.append({"tower": t, "driver": key, "production_mdc_R2": prod_r2,
                            "forced_interp_R2": round(interp_ov, 3) if pd.notna(interp_ov) else np.nan,
                            "would_interp_help": would_help})
        verdict = "would help -- reconsider" if would_help else "confirmed worse, correctly left alone"
        print(f"Tower {t} {key:6s}: production(mdc)={prod_r2:+.3f}  forced-interp={interp_ov:+.3f}   ({verdict})")

CHECK_OTHER_VARS = pd.DataFrame(check_rows)
CHECK_OTHER_VARS

# ===== CELL 40 (id=30ad7bcc) =====
t_demo_soil = 4
col_demo = f"Soil Moisture @ 10cm Depth (%) [{CAT[t_demo_soil]}]"
s_raw_demo = df_ext[col_demo]
valid_idx = s_raw_demo.index[s_raw_demo.notna()]
dm_demo = (s_raw_demo.index >= valid_idx[0]) & (s_raw_demo.index <= valid_idx[-1])
dfq_demo = pd.DataFrame({col_demo: s_raw_demo})
gap_reps_soil = insert_calendar_gaps(dfq_demo, col_demo, dm_demo, gap_hours=288, n_reps=1, seed=123)
gt_soil = gap_reps_soil[0]

sav_soil = s_raw_demo.loc[gt_soil].copy()
s_masked_demo = s_raw_demo.copy(); s_masked_demo.loc[gt_soil] = np.nan
filled_demo, _, _ = mdc_gapfill(plausibility_filter(s_masked_demo, col_demo), colname=col_demo)
r2_soil_demo = r2_score(sav_soil.values, filled_demo.loc[gt_soil].values)

fig, ax = plt.subplots(figsize=(11, 3.5))
ax.plot(gt_soil, sav_soil.values, "o-", label="actual (real, held out)", color="black", ms=3)
ax.plot(gt_soil, filled_demo.loc[gt_soil].values, "o-", label="mdc_gapfill reconstruction", color="#4C72B0", ms=3)
ax.set_ylabel("Soil moisture (%)")
ax.set_title(f"Tower {t_demo_soil} soil moisture, one 288h synthetic gap -- R2={r2_soil_demo:.3f}")
ax.legend(); plt.tight_layout(); plt.show()

# ===== CELL 43 (id=033ef18a) =====
fig, axes = plt.subplots(2, 2, figsize=(14, 7))
t_show = 4
ustar_col = f"USTAR_0_0_1 [Tower {t_show}]"
vpd_col = f"VPD_0_0_1 [Tower {t_show}]"
ustar_clean = plausibility_filter(df_ext[ustar_col], ustar_col)
vpd_clean = plausibility_filter(df_ext[vpd_col], vpd_col)
zoom = slice("2021-06-01", "2021-07-15")

axes[0, 0].plot(ustar_clean.resample("7D").mean(), color="#DD8452", lw=1)
axes[0, 0].set_title(f"Tower {t_show} USTAR -- full period, 7-day rolling mean")
axes[0, 0].set_ylabel("USTAR (m/s)")

axes[1, 0].plot(vpd_clean.resample("7D").mean(), color="#4C72B0", lw=1)
axes[1, 0].set_title(f"Tower {t_show} VPD -- full period, 7-day rolling mean")
axes[1, 0].set_ylabel("VPD (kPa)")

axes[0, 1].plot(ustar_clean.loc[zoom], color="#DD8452", lw=0.8, marker=".", ms=2)
axes[0, 1].set_title(f"Tower {t_show} USTAR -- hourly, {zoom.start} to {zoom.stop}")
axes[0, 1].set_ylabel("USTAR (m/s)")

axes[1, 1].plot(vpd_clean.loc[zoom], color="#4C72B0", lw=0.8, marker=".", ms=2)
axes[1, 1].set_title(f"Tower {t_show} VPD -- hourly, {zoom.start} to {zoom.stop}")
axes[1, 1].set_ylabel("VPD (kPa)")

plt.suptitle(f"Tower {t_show}: real, plausibility-filtered USTAR and VPD -- why one gap-fills far worse than the other")
plt.tight_layout(); plt.show()

# ===== CELL 46 (id=27fb94fb) =====
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
        edges = np.unique(np.quantile(g["ustar"], np.linspace(0, 1, 21)))
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
    days = (fc.index - fc.index[0]).days
    blk = days // 7
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
    reco_vals = lloyd_taylor(t_k, rref_t.values, e0)
    reco = pd.Series(reco_vals, index=fc.index)
    gpp = reco - fc
    gpp[gpp < 0] = 0.0
    gpp[rg < 20] = 0.0
    return gpp, reco, float(e0)


gpp_reco = pd.DataFrame(index=df_ext.index)
for t in TOWERS:
    dm = driver_map(t)
    fc = qc_fc(df_ext, t)
    sw = met_filled[f"{dm['sw']}__f"]; ta = met_filled[f"{dm['ta']}__f"]; us = met_filled[f"{dm['ustar']}__f"]
    thr = ustar_threshold(ta, us, fc, sw)
    gpp, reco, e0 = partition_nee(fc, ta, sw)
    gpp_reco[f"GPP [Tower {t}]"] = gpp.round(4)
    gpp_reco[f"Reco [Tower {t}]"] = reco.round(4)
    print(f"Tower {t}: u* threshold={thr:.3f} m/s, E0={e0}, GPP coverage={100*gpp.notna().mean():.0f}%")

# ===== CELL 48 (id=b2ca37fc) =====
TAU = {"manure": 30.0, "cut": 21.0}   # e-folding decay, days (pruned to the two channels the CH4 model uses)

# Field -> catchment map (NWFP_UG_Design_Develop.pdf, Appendix D). Tower N = Catchment N (D-18).
CATCHMENT_FIELDS = {2: {"NW002"}, 4: {"NW005", "NW006"}, 9: {"NW013", "NW039"}}

CUT_OPS = {"Mow", "Rowing up", "Forage harvest", "Trailers (silage)", "Top",
           "Baling (silage)", "Hay turning"}


def classify(row):
    app = str(row.get("Application", "")).lower()
    op = str(row.get("Field_Operation", "")).strip()
    if "inorganic fertiliser" in app:
        # Checked (and discarded) before "organic fertiliser": "inorganic fertiliser" contains
        # "organic fertiliser" as a literal substring ("in" + "organic fertiliser"), so this guard
        # must run first or every inorganic-N event gets misclassified as manure.
        return "fertN"
    if "organic fertiliser" in app:
        return "manure"
    if op in CUT_OPS:
        return "cut"
    return None


def recency_series(index, event_times, tau):
    idx_ns = index.values.astype("datetime64[ns]")
    if len(event_times) == 0:
        return np.zeros(len(index))
    ev = np.sort(np.array(event_times, dtype="datetime64[ns]"))
    pos = np.searchsorted(ev, idx_ns, side="right") - 1
    rec = np.zeros(len(index))
    valid = pos >= 0
    days = (idx_ns[valid] - ev[pos[valid]]) / np.timedelta64(1, "D")
    rec[valid] = np.exp(-days / tau)
    return rec


fe = pd.read_csv(COMPILED / "Field_Event_Data_Format_1.csv", low_memory=False)
fe["dt"] = pd.to_datetime(fe["Event_Date"], errors="coerce")
fe["channel"] = fe.apply(classify, axis=1)
fe["field"] = fe["Field"].astype(str).str.strip()
fe = fe.dropna(subset=["dt", "channel"])

mgmt = pd.DataFrame(index=df_ext.index)
for t in TOWERS:
    sub = fe[fe["field"].isin(CATCHMENT_FIELDS[t])]
    for ch, tau in TAU.items():
        evs = sub.loc[sub["channel"] == ch, "dt"].tolist()
        mgmt[f"mgmt_t{t}_{ch}_recency"] = np.round(recency_series(df_ext.index, evs, tau), 5)
    print(f"Tower {t}: manure events={len(sub[sub.channel=='manure'])}, cut events={len(sub[sub.channel=='cut'])}")

mgmt.describe().loc[["mean", "max"]]

# ===== CELL 50 (id=7f155d80) =====
LSU = {"cattle": 1.0, "sheep": 0.1, "lamb": 0.05}

livestock_density = pd.DataFrame(index=df_ext.index)
for t in TOWERS:
    cat = CAT[t]
    lsu = sum(df_ext[f"{s}_{cat}"].fillna(0) * w for s, w in LSU.items())
    livestock_density[f"lsu_dens [Tower {t}]"] = lsu / AREA[t]
    livestock_density[f"graze [Tower {t}]"] = (sum(df_ext[f"{s}_{cat}"].fillna(0) for s in LSU) > 0).astype(float)
    print(f"Tower {t} ({cat}, {AREA[t]} ha): mean LSU/ha={livestock_density[f'lsu_dens [Tower {t}]'].mean():.3f}, "
          f"max={livestock_density[f'lsu_dens [Tower {t}]'].max():.3f}, "
          f"% hours grazed={100*livestock_density[f'graze [Tower {t}]'].mean():.1f}%")

fig, ax = plt.subplots(figsize=(11, 3))
for t in TOWERS:
    ax.plot(livestock_density[f"lsu_dens [Tower {t}]"].resample("7D").mean(), label=f"Tower {t}", lw=1)
ax.set_ylabel("LSU/ha (7-day mean)"); ax.set_title("Livestock stocking density by tower"); ax.legend()
plt.tight_layout(); plt.show()

# ===== CELL 52 (id=69420f9b) =====
AUX = ["_hs", "_hc", "_ds", "_dc"]
LAG_HOURS = [168, 336, 504, 672]
DUM = ["is_t2", "is_t4", "is_t9"]

d_all = df_ext.join(met_filled).join(fco2_gapfilled).join(mgmt).join(gpp_reco)
for t in TOWERS:
    d_all[f"FC_1_1_1 [Tower {t}]"] = d_all[f"FC_gapfilled [Tower {t}]"]   # FC kept EC-sourced (section 4)
print(f"d_all shape: {d_all.shape}")


def cfg(t):
    cat = CAT[t]
    met = [f"SWIN_1_1_1 [Tower {t}]", f"TA_0_0_1 [Tower {t}]", f"VPD_0_0_1 [Tower {t}]",
           f"PPFD_1_1_1 [Tower {t}]", f"RN_1_1_1 [Tower {t}]", f"WS_0_0_1 [Tower {t}]",
           f"USTAR_0_0_1 [Tower {t}]", f"SHF_1_1_1 [Tower {t}]",
           f"Precipitation (mm) [{cat}]", ts_col_for(t), f"Soil Moisture @ 10cm Depth (%) [{cat}]"]
    return dict(t=t, cat=cat, area=AREA[t], tgt=f"FCH4_1_1_1 [Tower {t}]",
                ssitc=f"FCH4_SSITC_TEST_1_1_1 [Tower {t}]", met=met,
                fc=f"FC_1_1_1 [Tower {t}]", gpp=f"GPP [Tower {t}]", reco=f"Reco [Tower {t}]",
                swc=f"Soil Moisture @ 10cm Depth (%) [{cat}]", ts=ts_col_for(t),
                mc=f"mgmt_t{t}_cut_recency", mm=f"mgmt_t{t}_manure_recency")


def feat_list():
    c = cfg(2)
    return ([m.split(" [")[0] for m in c["met"]] + ["fc"] + AUX + ["lsu_dens", "graze"]
            + [f"swc_l{l}" for l in LAG_HOURS] + [f"ts_l{l}" for l in LAG_HOURS]
            + ["mgmt_cut", "mgmt_manure", "gpp", "reco"])


def frame(t, pooled, d):
    """Per-tower FCH4 gap-filling feature frame (target + all imputed/engineered features)."""
    c = cfg(t); d = d.copy(); tgt = c["tgt"]
    d.loc[~d[c["ssitc"]].isin([0, 1]), tgt] = np.nan
    d.loc[d[tgt].notna() & ~d[tgt].between(PLAUS_LOW, PLAUS_HIGH, inclusive="both"), tgt] = np.nan
    h, doy = d.index.hour, d.index.dayofyear
    d["_hs"] = np.sin(2 * np.pi * h / 24); d["_hc"] = np.cos(2 * np.pi * h / 24)
    d["_ds"] = np.sin(2 * np.pi * doy / 365); d["_dc"] = np.cos(2 * np.pi * doy / 365)
    g = pd.DataFrame(index=d.index); g["target"] = d[tgt]
    for k in c["met"]:
        nm = k.split(" [")[0]
        g[nm] = d[k + "__f"] if (k + "__f") in d.columns else d[k]
    g["fc"] = d[c["fc"]]
    for a in AUX:
        g[a] = d[a]
    g["lsu_dens"] = livestock_density[f"lsu_dens [Tower {t}]"]
    g["graze"] = livestock_density[f"graze [Tower {t}]"]
    swc = d[c["swc"] + "__f"] if (c["swc"] + "__f") in d.columns else d[c["swc"]]
    ts = d[c["ts"] + "__f"] if (c["ts"] + "__f") in d.columns else d[c["ts"]]
    for lag in LAG_HOURS:
        g[f"swc_l{lag}"] = swc.shift(lag); g[f"ts_l{lag}"] = ts.shift(lag)
    g["mgmt_cut"] = d[c["mc"]]; g["mgmt_manure"] = d[c["mm"]]
    g["gpp"] = d[c["gpp"]]; g["reco"] = d[c["reco"]]
    if pooled:
        for tt in TOWERS:
            g[f"is_t{tt}"] = 1.0 if tt == t else 0.0
    g["_y"] = d.index.year
    return g


FEATURES = feat_list()
print(f"{len(FEATURES)} base features (+3 tower dummies when pooled): {FEATURES}")

g4_preview = frame(4, pooled=True, d=d_all)
print(f"\nTower 4 frame shape: {g4_preview.shape}, target valid: {g4_preview['target'].notna().sum():,}")
g4_preview[["target", "fc", "lsu_dens", "gpp", "reco", "is_t2", "is_t4", "is_t9"]].dropna(subset=["target"]).head()

# ===== injected DATA_DIR =====
from pathlib import Path
DATA_DIR = Path("_data")
DATA_DIR.mkdir(exist_ok=True)


# ===== CELL 56 (id=75894044) =====
DOMAIN = {2: ("2017-10-01", "2019-06-30"), 4: ("2017-10-01", "2023-12-31"), 9: ("2020-02-01", "2023-12-31")}


def dom_mask(idx, t):
    a, b = DOMAIN[t]
    return (idx >= pd.Timestamp(a)) & (idx <= pd.Timestamp(b))


def mds_fill_batch(df_obs, target, sw_col, ta_col, gap_ts):
    """Marginal distribution sampling: average real values within an expanding similarity
    window (time-of-day, +/- days, radiation/temperature tolerance), literature baseline."""
    SW_TOL, TA_TOL = 50.0, 2.5
    WINDOWS = [pd.Timedelta(days=d) for d in [7, 14, 28, 91]]
    av = df_obs[df_obs[target].notna()]; ay = av[target].values.astype(float)
    ahr = av.index.hour.to_numpy(); adoy = av.index.dayofyear.to_numpy(); ats = av.index.to_numpy()
    asw = av[sw_col].values.astype(float); ata = av[ta_col].values.astype(float)
    gi = pd.DatetimeIndex(gap_ts)
    gsw = df_obs.reindex(gi)[sw_col].values.astype(float); gta = df_obs.reindex(gi)[ta_col].values.astype(float)
    preds = {}
    for i, t in enumerate(gap_ts):
        tt = np.datetime64(t); hr = t.hour; doy = t.dayofyear; sv = gsw[i]; tv = gta[i]
        day = (not np.isnan(sv)) and sv > 10.0; filled = False
        for wd in WINDOWS:
            w = wd.to_timedelta64()
            m = (ats >= tt - w) & (ats <= tt + w) & (np.abs(ahr - hr) <= 1)
            if not np.isnan(tv):
                m &= (np.abs(ata - tv) <= TA_TOL) | np.isnan(ata)
            if day and not np.isnan(sv):
                m &= (np.abs(asw - sv) <= SW_TOL) | np.isnan(asw)
            c = ay[m]
            if len(c) >= 1:
                preds[t] = float(np.nanmean(c)); filled = True; break
        if not filled:
            sh = np.abs(ahr - hr) <= 1
            dd = np.minimum(np.abs(adoy - doy), 365 - np.abs(adoy - doy))
            c = ay[sh & (dd <= 7)]
            if len(c) >= 1:
                preds[t] = float(np.nanmean(c))
    return preds


# insert_calendar_gaps/mets/med_metrics/SCENARIOS/N_REPS/MASK_FRAC were already defined in
# section 6.3 (reused here, not redefined). fit() is defined in section 11.0 above (with
# model caching).

print("FCH4-specific harness pieces defined (DOMAIN, dom_mask, mds_fill_batch).")

# ===== CELL 103 (id=749f32ba) =====
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

def fit_lgbm(feat, trd):
    imp = SimpleImputer(strategy="mean")
    m = LGBMRegressor(n_estimators=500, min_child_samples=5, random_state=42, n_jobs=-1, verbosity=-1)
    m.fit(imp.fit_transform(trd[feat].values), trd["target"].values)
    return m, imp


def fit_xgb(feat, trd):
    imp = SimpleImputer(strategy="mean")
    m = XGBRegressor(n_estimators=500, min_child_weight=5, random_state=42, n_jobs=-1, verbosity=0)
    m.fit(imp.fit_transform(trd[feat].values), trd["target"].values)
    return m, imp


def run_model(t, pooled, fit_fn, verbose=False):
    """Same held-out mechanism as run_rf (section 11), with a swappable fit function -- for
    comparing alternative estimators against the RFm champion without touching run_rf itself.
    verbose=True prints per-fold timing: fold count x fit cost is opaque up front for slow
    zero-shot models like TabPFN/TabICL -- without this, a slow fold is indistinguishable from a
    hang until the whole tower finishes."""
    feat = FEATURES + (DUM if pooled else [])
    towers = TOWERS if pooled else [t]
    frames = {tt: frame(tt, pooled, d_all) for tt in towers}
    g = frames[t]; idx = g.index; out = {}; dm = dom_mask(idx, t)
    t0v = time.time()
    for sc, gh in SCENARIOS.items():
        rr = []
        for fi, gt in enumerate(insert_calendar_gaps(g, "target", dm, gh)):
            if len(gt) < 5:
                continue
            base = g[dm & g["target"].notna().values]; trd = base.drop(index=gt, errors="ignore")
            if pooled:
                others = []
                for tt in towers:
                    if tt == t:
                        continue
                    gg = frames[tt]; dmt = dom_mask(gg.index, tt)
                    others.append(gg[dmt & gg["target"].notna().values])
                trd = pd.concat([trd] + others, ignore_index=True)
            m, imp = fit_fn(feat, trd)
            yp = m.predict(imp.transform(g.loc[gt, feat].values))
            rr.append(mets(g.loc[gt, "target"].values, yp))
            if verbose:
                print(f"    tower {t} scenario={sc} fold={fi} train_n={len(trd)} gap_n={len(gt)} "
                      f"[{time.time()-t0v:.0f}s elapsed]", flush=True)
        out[sc] = med_metrics(rr)
    return out

print("fit_lgbm / fit_xgb / run_model defined.")

# ===== CELL 107 (id=24020cfa) =====
from tabpfn import TabPFNRegressor
from tabicl import TabICLRegressor

FOUNDATION_MODEL_ROW_CAP = 10_000

def fit_tabpfn(feat, trd):
    imp = SimpleImputer(strategy="mean")
    trd_sub = trd.sample(n=min(FOUNDATION_MODEL_ROW_CAP, len(trd)), random_state=42)
    m = TabPFNRegressor(random_state=42)
    m.fit(imp.fit_transform(trd_sub[feat].values), trd_sub["target"].values)
    return m, imp


def fit_tabicl(feat, trd):
    imp = SimpleImputer(strategy="mean")
    trd_sub = trd.sample(n=min(FOUNDATION_MODEL_ROW_CAP, len(trd)), random_state=42)
    m = TabICLRegressor(random_state=42)
    m.fit(imp.fit_transform(trd_sub[feat].values), trd_sub["target"].values)
    return m, imp

print("fit_tabpfn / fit_tabicl defined.")

# ===== CELL 110 (id=saits18202) =====
import torch
from pypots.imputation import SAITS
from pypots.nn.modules.loss import Criterion
from pygrinder import mcar

WINDOW, STRIDE = 336, 24  # 14-day windows, daily stride -- matches F-11
SAITS_FEAT_COLS = [f for f in FEATURES if not f.startswith("swc_l") and not f.startswith("ts_l")]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{len(SAITS_FEAT_COLS)} covariate channels + 1 target channel, device={DEVICE}")


class SpikeWeightedMAE(Criterion):
    """MAE with points upweighted by |target| (standardized units) -- F-11's single largest lever
    (EXP_D), fixing FCH4's systematic under-prediction under a symmetric loss on a right-skewed
    target. Reproduced verbatim from F11_SAITS_Implementation.ipynb."""

    def __init__(self, weight_scale=1.0):
        super().__init__()
        self.weight_scale = weight_scale

    def forward(self, logits, targets, masks=None):
        weights = 1.0 + self.weight_scale * torch.abs(targets)
        err = torch.abs(logits - targets) * weights
        if masks is not None:
            return (err * masks).sum() / (masks.sum() + 1e-12)
        return err.mean()


def make_windows(g, feat_cols, domain, window=WINDOW, stride=STRIDE):
    """Sliding windows of feat_cols restricted to `domain=(a,b)`. Returns (X, idx, starts)."""
    a, b = domain
    gd = g.loc[a:b]
    idx = gd.index
    n = len(idx)
    starts = list(range(0, max(1, n - window + 1), stride))
    X = np.stack([gd.iloc[s:s + window][feat_cols].values for s in starts]).astype(np.float32)
    return X, idx, starts


def extract_at_timestamps(imputation, idx, starts, window, channel_idx, timestamps):
    """Average every overlapping window's reconstruction for each requested timestamp."""
    pos_of_ts = {ts: i for i, ts in enumerate(idx)}
    starts_arr = np.array(starts)
    out = {}
    for ts in timestamps:
        p = pos_of_ts.get(ts)
        if p is None:
            continue
        lo = max(0, p - window + 1)
        ws = np.where((starts_arr >= lo) & (starts_arr <= p))[0]
        if len(ws) == 0:
            continue
        vals = [imputation[w, p - starts_arr[w], channel_idx] for w in ws]
        out[ts] = float(np.mean(vals))
    return out


def build_train_val(X_windows, val_frac=0.1, mcar_rate=0.1):
    """Chronological split; val_set gets extra MCAR masking for SAITS's early-stopping metric."""
    n = X_windows.shape[0]
    n_val = max(1, int(n * val_frac))
    X_train = X_windows[:n - n_val]
    X_val_ori = X_windows[n - n_val:].copy()
    X_val = mcar(X_val_ori, p=mcar_rate)
    return {"X": X_train}, {"X": X_val, "X_ori": X_val_ori}

print("SAITS helpers defined.")

# ===== CELL 113 (id=bilstm18302) =====
import torch.nn as nn


class BiLSTMImputer(nn.Module):
    """2-layer bidirectional LSTM over a window of [covariates, target_filled, observed_mask]
    channels; predicts the target at every timestep from the concatenated fwd/bwd hidden states."""

    def __init__(self, n_covariates, hidden_size=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_covariates + 2, hidden_size=hidden_size, num_layers=num_layers,
                             batch_first=True, bidirectional=True,
                             dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out).squeeze(-1)


def build_model_input(X_windows, extra_mask=None):
    """X_windows: (n, WINDOW, n_channels) with target as the LAST channel, NaN where missing.
    extra_mask: optional (n, WINDOW) array in [0,1] of additional positions to artificially hide
    (the MIT task) -- 1 means "pretend this observed point is missing" for this forward pass."""
    covariates = np.nan_to_num(X_windows[..., :-1], nan=0.0)
    target_col = X_windows[..., -1]
    mask_obs = (~np.isnan(target_col)).astype(np.float32)
    target_filled = np.nan_to_num(target_col, nan=0.0)
    if extra_mask is not None:
        mask_obs = mask_obs * (1.0 - extra_mask)
        target_filled = target_filled * (1.0 - extra_mask)
    x = np.concatenate([covariates, target_filled[..., None], mask_obs[..., None]], axis=-1)
    return torch.tensor(x, dtype=torch.float32)


def spike_weighted_mae(pred, true, mask):
    w = 1.0 + torch.abs(true)
    err = torch.abs(pred - true) * w * mask
    return err.sum() / (mask.sum() + 1e-12)


def train_bilstm(X_train, X_val, epochs=50, patience=8, batch_size=32, mit_rate=0.2, seed=42, lr=1e-3):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n_covariates = X_train.shape[-1] - 1
    model = BiLSTMImputer(n_covariates).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = X_train.shape[0]
    best_val, best_state, bad_epochs = np.inf, None, 0

    for epoch in range(epochs):
        model.train()
        order = rng.permutation(n)
        for s in range(0, n, batch_size):
            idx = order[s:s + batch_size]
            Xb = X_train[idx]
            obs_mask = (~np.isnan(Xb[..., -1])).astype(np.float32)
            mit_mask = (rng.random(obs_mask.shape) < mit_rate).astype(np.float32) * obs_mask
            x_in = build_model_input(Xb, extra_mask=mit_mask).to(DEVICE)
            true_target = torch.tensor(np.nan_to_num(Xb[..., -1], nan=0.0), dtype=torch.float32).to(DEVICE)
            mit_mask_t = torch.tensor(mit_mask, dtype=torch.float32).to(DEVICE)
            pred = model(x_in)
            loss = spike_weighted_mae(pred, true_target, mit_mask_t)
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            obs_mask_v = (~np.isnan(X_val[..., -1])).astype(np.float32)
            mit_mask_v = (rng.random(obs_mask_v.shape) < mit_rate).astype(np.float32) * obs_mask_v
            x_in_v = build_model_input(X_val, extra_mask=mit_mask_v).to(DEVICE)
            true_v = torch.tensor(np.nan_to_num(X_val[..., -1], nan=0.0), dtype=torch.float32).to(DEVICE)
            mit_v_t = torch.tensor(mit_mask_v, dtype=torch.float32).to(DEVICE)
            pred_v = model(x_in_v)
            val_loss = spike_weighted_mae(pred_v, true_v, mit_v_t).item()

        if val_loss < best_val - 1e-5:
            best_val, best_state, bad_epochs = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model


def predict_bilstm(model, X_windows, batch_size=64):
    model.eval()
    n = X_windows.shape[0]
    outs = []
    with torch.no_grad():
        for s in range(0, n, batch_size):
            Xb = X_windows[s:s + batch_size]
            x_in = build_model_input(Xb).to(DEVICE)
            outs.append(model(x_in).cpu().numpy())
    return np.concatenate(outs, axis=0)[..., None]  # add trailing channel dim for extract_at_timestamps

print("BiLSTMImputer / train_bilstm / predict_bilstm defined.")

import time
import gc
from scipy.stats import linregress
from sklearn.preprocessing import StandardScaler

CKPT_DIR = DATA_DIR / "d100_checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

CHAMPION_R2 = {2: 0.576, 4: 0.404, 9: 0.426}   # standing production champion (D-77), hardcoded reference only

# sklearn-R2 numbers already on disk (_data/model_comparison.csv) -- used as the verification
# checkpoint (this run's own sklearn R2 must reproduce these bit-for-bit before R2_OLS is trusted).
ORIGINAL_R2 = {
    ("LightGBM", 2): 0.522, ("LightGBM", 4): 0.410, ("LightGBM", 9): 0.422,
    ("XGBoost", 2): 0.551, ("XGBoost", 4): 0.349, ("XGBoost", 9): 0.369,
    ("TabICL", 2): 0.558, ("TabICL", 4): 0.423, ("TabICL", 9): 0.364,
    ("SAITS", 2): 0.358, ("SAITS", 4): 0.293, ("SAITS", 9): 0.285,
    ("BI-LSTM", 2): 0.237, ("BI-LSTM", 4): 0.155, ("BI-LSTM", 9): 0.146,
}


def mets_ols(y, p):
    """mets() + R2_OLS (squared Pearson r, Zhu et al. 2023a convention, bounded [0,1]) -- same
    formula as temp_gap_filling_pipeline.ipynb's own mets(), ported here since this notebook's
    mets() predates that addition."""
    r2, rmse, mae, mbe = mets(y, p)
    y = np.asarray(y, float); p = np.asarray(p, float)
    if np.var(y) > 0 and np.var(p) > 0:
        slope, _, r, _, _ = linregress(y, p)
        r2_ols = r ** 2
    else:
        slope, r2_ols = np.nan, np.nan
    return r2, rmse, mae, mbe, r2_ols, slope


def med_metrics_ols(rows):
    if not rows:
        return {k: np.nan for k in ["R2", "RMSE", "MAE", "MBE", "R2_OLS", "OLS_slope"]}
    a = np.array(rows, float)
    return {"R2": np.nanmedian(a[:, 0]), "RMSE": np.median(a[:, 1]), "MAE": np.median(a[:, 2]),
            "MBE": np.median(a[:, 3]), "R2_OLS": np.nanmedian(a[:, 4]), "OLS_slope": np.nanmedian(a[:, 5])}


def run_model_capture(t, fit_fn, model_name, pooled=True):
    """Verbatim copy of cell 103's run_model(), extended to also stash raw (Datetime, actual,
    pred) rows and score with mets_ols instead of mets. Same insert_calendar_gaps(seed=0) fold
    structure -- must reproduce ORIGINAL_R2[(model_name, t)] exactly."""
    feat = FEATURES + (DUM if pooled else [])
    towers = TOWERS if pooled else [t]
    frames = {tt: frame(tt, pooled, d_all) for tt in towers}
    g = frames[t]; dm = dom_mask(g.index, t)
    raw_parts = []; scn_metrics = {}
    for sc, gh in SCENARIOS.items():
        rr = []
        for fi, gt in enumerate(insert_calendar_gaps(g, "target", dm, gh)):
            if len(gt) < 5:
                continue
            base = g[dm & g["target"].notna().values]; trd = base.drop(index=gt, errors="ignore")
            if pooled:
                others = []
                for tt in towers:
                    if tt == t:
                        continue
                    gg = frames[tt]; dmt = dom_mask(gg.index, tt)
                    others.append(gg[dmt & gg["target"].notna().values])
                trd = pd.concat([trd] + others, ignore_index=True)
            m, imp = fit_fn(feat, trd)
            yp = m.predict(imp.transform(g.loc[gt, feat].values))
            y_true = g.loc[gt, "target"].values
            rr.append(mets_ols(y_true, yp))
            raw_parts.append(pd.DataFrame({"actual": y_true, "pred": yp, "rep": fi, "scenario": sc}, index=gt))
        scn_metrics[sc] = med_metrics_ols(rr)
    raw_df = pd.concat(raw_parts) if raw_parts else pd.DataFrame(columns=["actual", "pred", "rep", "scenario"])
    raw_df = raw_df.reset_index().rename(columns={"index": "Datetime"})
    raw_df["tower"] = t; raw_df["model"] = model_name
    return scn_metrics, raw_df


def fit_score_saits_capture(t, sc, gh, seed=42):
    """Verbatim copy of cell 111's fit_score_saits_scenario(), extended to also stash raw pairs
    and score with mets_ols."""
    channels = SAITS_FEAT_COLS + ["target"]
    target_idx = channels.index("target")
    torch.manual_seed(seed)

    g_t = frame(t, pooled=False, d=d_all)
    dm_t = dom_mask(g_t.index, t)
    reps = insert_calendar_gaps(g_t, "target", dm_t, gh, n_reps=N_REPS, seed=0)
    union_ts = pd.DatetimeIndex(sorted(set().union(*[set(r) for r in reps])))

    g_gapped = g_t.copy()
    y_true_all = g_gapped["target"].copy()
    g_gapped.loc[union_ts, "target"] = np.nan

    scaler = StandardScaler().fit(g_gapped.loc[DOMAIN[t][0]:DOMAIN[t][1], channels])
    g_scaled = g_gapped.copy()
    g_scaled[channels] = scaler.transform(g_gapped[channels])
    X_t, idx_t, starts_t = make_windows(g_scaled, channels, DOMAIN[t])

    train_set, val_set = build_train_val(X_t)
    model = SAITS(
        n_steps=WINDOW, n_features=len(channels), n_layers=3, d_model=256, n_heads=4,
        d_k=64, d_v=64, d_ffn=512, dropout=0.1, epochs=100, patience=10, batch_size=32,
        device=DEVICE, saving_path=None, model_saving_strategy=None, verbose=False,
        training_loss=SpikeWeightedMAE,
    )
    model.fit(train_set, val_set)
    result = model.predict({"X": X_t})
    imputation = np.asarray(result["imputation"])   # detach from any GPU-backed object before model is freed
    del model, result, train_set, val_set, X_t
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    t_mean, t_std = scaler.mean_[target_idx], scaler.scale_[target_idx]
    rep_metrics = []; raw_parts = []
    for rep_i, rep_ts in enumerate(reps):
        preds_scaled = extract_at_timestamps(imputation, idx_t, starts_t, WINDOW, target_idx, rep_ts)
        ts_scored = [ts for ts in rep_ts if ts in preds_scaled]
        if len(ts_scored) < 5:
            continue
        y_pred = np.array([preds_scaled[ts] for ts in ts_scored]) * t_std + t_mean
        y_obs = y_true_all.loc[ts_scored].values
        rep_metrics.append(mets_ols(y_obs, y_pred))
        raw_parts.append(pd.DataFrame({"actual": y_obs, "pred": y_pred, "rep": rep_i, "scenario": sc},
                                       index=pd.DatetimeIndex(ts_scored)))
    raw_df = pd.concat(raw_parts) if raw_parts else pd.DataFrame(columns=["actual", "pred", "rep", "scenario"])
    return med_metrics_ols(rep_metrics), raw_df


def fit_score_bilstm_capture(t, sc, gh, seed=42):
    """Verbatim copy of cell 114's fit_score_bilstm_scenario(), extended to also stash raw pairs
    and score with mets_ols."""
    channels = SAITS_FEAT_COLS + ["target"]
    target_idx = channels.index("target")

    g_t = frame(t, pooled=False, d=d_all)
    dm_t = dom_mask(g_t.index, t)
    reps = insert_calendar_gaps(g_t, "target", dm_t, gh, n_reps=N_REPS, seed=0)
    union_ts = pd.DatetimeIndex(sorted(set().union(*[set(r) for r in reps])))

    g_gapped = g_t.copy()
    y_true_all = g_gapped["target"].copy()
    g_gapped.loc[union_ts, "target"] = np.nan

    scaler = StandardScaler().fit(g_gapped.loc[DOMAIN[t][0]:DOMAIN[t][1], channels])
    g_scaled = g_gapped.copy()
    g_scaled[channels] = scaler.transform(g_gapped[channels])
    X_t, idx_t, starts_t = make_windows(g_scaled, channels, DOMAIN[t])

    n = X_t.shape[0]
    n_val = max(1, int(n * 0.1))
    X_train, X_val = X_t[:n - n_val], X_t[n - n_val:]

    model = train_bilstm(X_train, X_val, seed=seed)
    imputation = predict_bilstm(model, X_t)   # already .cpu().numpy() inside predict_bilstm
    del model, X_train, X_val
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    t_mean, t_std = scaler.mean_[target_idx], scaler.scale_[target_idx]
    rep_metrics = []; raw_parts = []
    for rep_i, rep_ts in enumerate(reps):
        preds_scaled = extract_at_timestamps(imputation, idx_t, starts_t, WINDOW, 0, rep_ts)
        ts_scored = [ts for ts in rep_ts if ts in preds_scaled]
        if len(ts_scored) < 5:
            continue
        y_pred = np.array([preds_scaled[ts] for ts in ts_scored]) * t_std + t_mean
        y_obs = y_true_all.loc[ts_scored].values
        rep_metrics.append(mets_ols(y_obs, y_pred))
        raw_parts.append(pd.DataFrame({"actual": y_obs, "pred": y_pred, "rep": rep_i, "scenario": sc},
                                       index=pd.DatetimeIndex(ts_scored)))
    raw_df = pd.concat(raw_parts) if raw_parts else pd.DataFrame(columns=["actual", "pred", "rep", "scenario"])
    return med_metrics_ols(rep_metrics), raw_df


# ============================== driver ==============================
RESULTS_ROWS = []
ALL_RAW = []


def _report(model_name, t, scn):
    ov = {k: np.nanmedian([scn[s][k] for s in SCENARIOS]) for k in ["R2", "RMSE", "MAE", "MBE", "R2_OLS", "OLS_slope"]}
    orig = ORIGINAL_R2[(model_name, t)]
    match = "OK" if abs(ov["R2"] - orig) < 0.002 else "MISMATCH"
    print(f"{model_name:10s} T{t}: R2_sklearn={ov['R2']:+.3f} (orig {orig:+.3f}, {match})  "
          f"R2_OLS={ov['R2_OLS']:.3f}  champion_R2={CHAMPION_R2[t]:.3f}", flush=True)
    RESULTS_ROWS.append({"model": model_name, "tower": t, **ov, "orig_R2_sklearn": orig,
                          "verification": match, "champion_R2": CHAMPION_R2[t]})


print("=== D-100: recalculating R2_OLS (scipy/Zhu-style) for D-78's 5 non-TabPFN models ===", flush=True)


def _ckpt_load_or_run(key, compute_fn):
    """Checkpoint-gated: if (raw_df, scn_dict) was already saved for this key, load it instead of
    recomputing -- so a killed/restarted run doesn't repay already-finished work."""
    raw_path = CKPT_DIR / f"{key}_raw.csv"
    scn_path = CKPT_DIR / f"{key}_scn.csv"
    if raw_path.exists() and scn_path.exists():
        print(f"  [checkpoint hit] {key}", flush=True)
        raw_df = pd.read_csv(raw_path, parse_dates=["Datetime"])
        scn = pd.read_csv(scn_path).set_index("scenario").to_dict("index")
        return scn, raw_df
    scn, raw_df = compute_fn()
    raw_df.to_csv(raw_path, index=False)
    pd.DataFrame(scn).T.reset_index().rename(columns={"index": "scenario"}).to_csv(scn_path, index=False)
    return scn, raw_df


t0 = time.time()
for model_name, fit_fn in [("LightGBM", fit_lgbm), ("XGBoost", fit_xgb), ("TabICL", fit_tabicl)]:
    for t in TOWERS:
        scn, raw_df = _ckpt_load_or_run(f"{model_name}_{t}", lambda: run_model_capture(t, fit_fn, model_name, pooled=True))
        _report(model_name, t, scn)
        raw_df.to_csv(DATA_DIR / f"d100_raw_{model_name}_{t}.csv", index=False)
        ALL_RAW.append(raw_df)
print(f"Tree/foundation models done [{time.time()-t0:.0f}s elapsed]", flush=True)

for t in TOWERS:
    def _run_saits_tower(t=t):
        scn_t = {}; raw_parts_t = []
        for sc, gh in SCENARIOS.items():
            m, raw = fit_score_saits_capture(t, sc, gh)
            scn_t[sc] = m; raw["scenario"] = sc; raw_parts_t.append(raw)
            print(f"    SAITS T{t} scenario={sc}: R2={m['R2']:+.3f} R2_OLS={m['R2_OLS']:.3f} [{time.time()-t0:.0f}s elapsed]", flush=True)
        raw_df_t = pd.concat(raw_parts_t).reset_index().rename(columns={"index": "Datetime"})
        raw_df_t["tower"] = t; raw_df_t["model"] = "SAITS"
        return scn_t, raw_df_t
    scn, raw_df = _ckpt_load_or_run(f"SAITS_{t}", _run_saits_tower)
    _report("SAITS", t, scn)
    raw_df.to_csv(DATA_DIR / f"d100_raw_SAITS_{t}.csv", index=False)
    ALL_RAW.append(raw_df)
print(f"SAITS done [{time.time()-t0:.0f}s elapsed]", flush=True)

for t in TOWERS:
    def _run_bilstm_tower(t=t):
        scn_t = {}; raw_parts_t = []
        for sc, gh in SCENARIOS.items():
            m, raw = fit_score_bilstm_capture(t, sc, gh)
            scn_t[sc] = m; raw["scenario"] = sc; raw_parts_t.append(raw)
            print(f"    BI-LSTM T{t} scenario={sc}: R2={m['R2']:+.3f} R2_OLS={m['R2_OLS']:.3f} [{time.time()-t0:.0f}s elapsed]", flush=True)
        raw_df_t = pd.concat(raw_parts_t).reset_index().rename(columns={"index": "Datetime"})
        raw_df_t["tower"] = t; raw_df_t["model"] = "BI-LSTM"
        return scn_t, raw_df_t
    scn, raw_df = _ckpt_load_or_run(f"BILSTM_{t}", _run_bilstm_tower)
    _report("BI-LSTM", t, scn)
    raw_df.to_csv(DATA_DIR / f"d100_raw_BILSTM_{t}.csv", index=False)
    ALL_RAW.append(raw_df)
print(f"BI-LSTM done [{time.time()-t0:.0f}s elapsed]", flush=True)

RESULTS_DF = pd.DataFrame(RESULTS_ROWS)
RESULTS_DF.to_csv(DATA_DIR / "d100_ols_recalc_summary.csv", index=False)
ALL_RAW_DF = pd.concat(ALL_RAW, ignore_index=True)
ALL_RAW_DF.to_csv(DATA_DIR / "d100_ols_recalc_raw_predictions.csv", index=False)
print(f"\nSaved d100_ols_recalc_summary.csv ({len(RESULTS_DF)} rows) and "
      f"d100_ols_recalc_raw_predictions.csv ({len(ALL_RAW_DF):,} rows) [{time.time()-t0:.0f}s total]", flush=True)

print("\n=== Final table: sklearn R2 vs R2_OLS (scipy/Zhu-style), per model/tower ===")
print(RESULTS_DF[["model", "tower", "R2", "orig_R2_sklearn", "verification", "R2_OLS", "champion_R2"]]
      .to_string(index=False))
