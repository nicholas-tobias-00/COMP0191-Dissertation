"""Literature-correct MDS (Marginal Distribution Sampling) gap-filler for FCH4 (D-79).

Reichstein et al. (2005)/REddyProc's real algorithm, reconstructed and validated in
`notebooks/03c_gap_filling_revisited/temp_mds.ipynb` against REddyProc's own R source
(`EddyGapfilling.R`) after auditing this project's original implementation and finding 3 bugs:
(1) an hour-of-day +/-1h restriction was wrongly applied to every case instead of only the final
fallback; (2) an intermediate SW-only look-up case (Case 2) was missing entirely; (3) the
fallback was a single fixed +/-7-day box with no meteorological constraint, not the real
algorithm's expanding mean-diurnal-course window. This is the floor reference for the RFm
champion (`gapfill_rfm.py`) -- not the recommended production model, but included here so the
full model roster this project evaluated can be run/scored consistently (D-79).

Reuses `gapfill_rfm.py`'s tower config (target/QC/catchment columns) as the single source of
truth for column names, matching this project's existing cross-import convention
(`build_fch4_gapfilled.py` -> `gapfill_rfm.py`).

Run standalone for a quick per-tower gap-CV check:  python src/models/gapfill_mds.py
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from gapfill_rfm import load_ext, cfg, ts_col_for, PLAUS_LOW, PLAUS_HIGH, TOWERS  # noqa: E402
from gap_cv import dom_mask, SCENARIOS, insert_calendar_gaps, gapfilling_metrics, median_metrics, headline  # noqa: E402


def mds_fill_batch(df_obs, target, sw_col, ta_col, gap_ts, vpd_col=None):
    """3-case hierarchy, each tried in order until a match is found:
    Case 1 (SW+TA+VPD look-up): +/-7d, then +/-14d; no hour restriction.
    Case 2 (SW-only look-up, day-gated SW>10): +/-7d stepping to +/-70d; no hour restriction.
    Case 3 (meteo-free mean diurnal course, fallback): +/-1h restriction; 0/1/2d, then +/-7d
    stepping to +/-210d.
    Returns {timestamp: filled_value} -- may omit timestamps genuinely unfillable within the
    largest window (rare; report the fill rate, don't silently assume 100%)."""
    SW_TOL, TA_TOL, VPD_TOL = 50.0, 2.5, 5.0
    CASE1_WINDOWS = [pd.Timedelta(days=d) for d in [7, 14]]
    CASE2_WINDOWS = [pd.Timedelta(days=d) for d in range(7, 71, 7)]
    CASE3_WINDOWS = [pd.Timedelta(days=d) for d in [0, 1, 2]] + [pd.Timedelta(days=d) for d in range(7, 211, 7)]

    av = df_obs[df_obs[target].notna()]; ay = av[target].values.astype(float)
    ahr = av.index.hour.to_numpy(); ats = av.index.to_numpy()
    asw = av[sw_col].values.astype(float); ata = av[ta_col].values.astype(float)
    avpd = av[vpd_col].values.astype(float) if vpd_col is not None else None

    gi = pd.DatetimeIndex(gap_ts)
    gsw = df_obs.reindex(gi)[sw_col].values.astype(float)
    gta = df_obs.reindex(gi)[ta_col].values.astype(float)
    gvpd = df_obs.reindex(gi)[vpd_col].values.astype(float) if vpd_col is not None else None

    preds = {}
    for i, t in enumerate(gap_ts):
        tt = np.datetime64(t); hr = t.hour
        sv = gsw[i]; tv = gta[i]; vv = gvpd[i] if gvpd is not None else np.nan
        day = (not np.isnan(sv)) and sv > 10.0
        filled = False

        if not np.isnan(sv) and not np.isnan(tv) and (avpd is None or not np.isnan(vv)):
            for wd in CASE1_WINDOWS:
                w = wd.to_timedelta64()
                m = (ats >= tt - w) & (ats <= tt + w)
                m &= (np.abs(ata - tv) <= TA_TOL) | np.isnan(ata)
                if avpd is not None:
                    m &= (np.abs(avpd - vv) <= VPD_TOL) | np.isnan(avpd)
                m &= (np.abs(asw - sv) <= SW_TOL) | np.isnan(asw)
                c = ay[m]
                if len(c) >= 1:
                    preds[t] = float(np.nanmean(c)); filled = True; break

        if not filled and day and not np.isnan(sv):
            for wd in CASE2_WINDOWS:
                w = wd.to_timedelta64()
                m = (ats >= tt - w) & (ats <= tt + w)
                m &= (np.abs(asw - sv) <= SW_TOL) | np.isnan(asw)
                c = ay[m]
                if len(c) >= 1:
                    preds[t] = float(np.nanmean(c)); filled = True; break

        if not filled:
            sh = np.abs(ahr - hr) <= 1
            for wd in CASE3_WINDOWS:
                w = wd.to_timedelta64()
                m = sh & (ats >= tt - w) & (ats <= tt + w)
                c = ay[m]
                if len(c) >= 1:
                    preds[t] = float(np.nanmean(c)); filled = True; break
    return preds


def _qc_frame(t, d):
    c = cfg(t, ts_col_for(t)); d = d.copy(); tgt = c["tgt"]
    d.loc[~d[c["ssitc"]].isin([0, 1]), tgt] = np.nan
    d.loc[d[tgt].notna() & ~d[tgt].between(PLAUS_LOW, PLAUS_HIGH, inclusive="both"), tgt] = np.nan
    return d, c


def evaluate_tower(t, d):
    """Full gap-CV sweep (5 scenarios x N_REPS) for tower t. Returns
    {scenario: median_metrics_dict}; headline() gives the single-number summary."""
    d, c = _qc_frame(t, d)
    tgt, sw_col, ta_col = c["tgt"], c["sw"], c["ta"]
    vpd_col = f"VPD_0_0_1 [Tower {t}]"
    dm = dom_mask(d.index, t)
    train_std = float(d.loc[dm, tgt].std())

    out = {}
    for sc, gh in SCENARIOS.items():
        rows = []
        for gt in insert_calendar_gaps(d, tgt, dm, gh):
            if len(gt) < 5:
                continue
            sav = d.loc[gt, tgt].copy(); d.loc[gt, tgt] = np.nan
            preds = mds_fill_batch(d, tgt, sw_col, ta_col, list(gt), vpd_col=vpd_col)
            d.loc[gt, tgt] = sav
            filled_ts = [x for x in gt if x in preds]
            if len(filled_ts) >= 5:
                rows.append(gapfilling_metrics(sav.loc[filled_ts].values,
                                                np.array([preds[x] for x in filled_ts]), train_std))
        out[sc] = median_metrics(rows)
    return out


if __name__ == "__main__":
    d_all = load_ext()
    for t in TOWERS:
        res = evaluate_tower(t, d_all)
        h = headline(res)
        print(f"Tower {t}: MAE={h['MAE']:.2f}  nMAE={h['nMAE']:.3f}  RMSE={h['RMSE']:.2f}  "
              f"R2={h['R2']:.3f}  R2_OLS={h['R2_OLS']:.3f}")
