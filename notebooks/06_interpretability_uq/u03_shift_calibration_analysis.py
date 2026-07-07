"""U-03 Part A: does U-02's leave-one-anchor-out conformal calibration hold up worse for anchors
that are more DIFFERENT (in scenario-relevant driver distribution) from the other anchors used to
calibrate them? Pure re-analysis -- no new model fitting, no new rollout. Reuses:
  - results/u02_summary.csv (per anchor/tower/model/bin conformal_picp etc., from u02_multi_anchor_tower.py)
  - results/u02_chains.csv (per-day chains, for the highlighted-anchor fan charts)
  - data/Hourly/forecast_daily_v2.csv (real historical fx_ driver values, perfect foresight)

Shift score (per anchor, tower): mean |z-score| of 4 scenario-relevant drivers (fx_lsu_dens,
fx_TA_mean, fx_PRECIP_sum, fx_SWIN_mean), each anchor's real 365-day target-window mean compared
against the pooled mean/std of the SAME 4 drivers over the OTHER 4 anchors' target windows for that
tower -- i.e. the exact reference set U-02's own leave-one-anchor-out calibration used for that
anchor. This directly tests, with real ground truth, whether more-different anchors show worse
realized calibration (evidence for/against the exchangeability concern flagged in U02_results.md
as an undocumented gap).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = r"c:\Users\Nicholas\Documents\COMP0191 MSc Artificial Intelligence for Sustainable Development Project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\notebooks\06_interpretability_uq")

HOURLY = rf"{ROOT}\data\Hourly"
RESULTS = rf"{ROOT}\results"

TOWERS = [2, 4, 9]
ANCHOR_YEARS = [2018, 2019, 2020, 2021, 2022]
SHIFT_FEATURES = ["fx_lsu_dens", "fx_TA_mean", "fx_PRECIP_sum", "fx_SWIN_mean"]


def anchor_window_means(dft, anchor, n_days=365):
    """Real historical mean of each SHIFT_FEATURES driver over the anchor's 365-day target window
    (perfect-foresight fx_ values, identical source to u02_multi_anchor_tower.py's fx_frame)."""
    target_dates = pd.date_range(anchor + pd.Timedelta(days=1), periods=n_days, freq="D")
    sub = dft.reindex(target_dates)
    return {f: float(sub[f].mean()) for f in SHIFT_FEATURES}


def compute_shift_scores(T):
    """For each (anchor, tower), z-score its own window-mean against the pooled distribution of the
    OTHER 4 anchors' window-means for that tower -- same reference set U-02's leave-one-anchor-out
    calibration used. Returns a DataFrame: anchor_year, eval_tower, shift_score, plus each feature's
    raw z-score for transparency."""
    rows = []
    for tower in TOWERS:
        dft = T[tower]
        means_by_anchor = {}
        for yr in ANCHOR_YEARS:
            anchor = pd.Timestamp(f"{yr}-12-16")
            means_by_anchor[yr] = anchor_window_means(dft, anchor)

        for yr in ANCHOR_YEARS:
            other_yrs = [y for y in ANCHOR_YEARS if y != yr]
            z_scores = {}
            for f in SHIFT_FEATURES:
                other_vals = np.array([means_by_anchor[y][f] for y in other_yrs])
                ref_mean, ref_std = other_vals.mean(), other_vals.std()
                this_val = means_by_anchor[yr][f]
                z = (this_val - ref_mean) / ref_std if ref_std > 1e-9 else 0.0
                z_scores[f] = z
            shift_score = float(np.mean([abs(z) for z in z_scores.values()]))
            row = {"anchor_year": yr, "eval_tower": tower, "shift_score": shift_score}
            row.update({f"z_{f}": z_scores[f] for f in SHIFT_FEATURES})
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    dv = pd.read_csv(f"{HOURLY}/forecast_daily_v2.csv", low_memory=False)
    dv["Datetime"] = pd.to_datetime(dv["Datetime"], format="mixed")
    T = {t: dv[dv.tower == t].set_index("Datetime").sort_index() for t in TOWERS}

    print("=" * 70)
    print("U-03 Part A: shift score (real historical) vs conformal PICP")
    print("=" * 70)

    shift_df = compute_shift_scores(T)
    print("\nShift scores per anchor/tower:")
    print(shift_df.round(3).to_string(index=False))

    summary = pd.read_csv(f"{RESULTS}/u02_summary.csv")

    def wavg(g, col):
        vals = g[col]
        if vals.isna().all():
            return np.nan
        w = g["n"]
        return (vals * w).sum() / w.sum() if w.sum() > 0 else np.nan

    picp_agg = summary.groupby(["anchor_year", "eval_tower", "model"]).apply(
        lambda g: pd.Series({
            "conformal_picp": wavg(g, "conformal_picp"),
            "conformal_mpiw": wavg(g, "conformal_mpiw"),
            "conformal_pinball": wavg(g, "conformal_pinball"),
            "raw_picp": wavg(g, "raw_picp") if "raw_picp" in g else np.nan,
        }), include_groups=False
    ).reset_index()

    merged = picp_agg.merge(shift_df, on=["anchor_year", "eval_tower"], how="left")
    merged.to_csv(f"{RESULTS}/u03_shift_calibration_summary.csv", index=False)
    print(f"\n[OK] Saved u03_shift_calibration_summary.csv ({len(merged)} rows)")

    # Headline correlation: shift_score vs conformal_picp, per tower (T2 reported separately --
    # its own already-documented calibration-availability failure would confound a pooled number).
    print("\nCorrelation(shift_score, conformal_picp) per tower (n-weighted mean across models):")
    per_anchor_tower = merged.groupby(["anchor_year", "eval_tower"]).agg(
        shift_score=("shift_score", "first"),
        conformal_picp=("conformal_picp", "mean"),
        raw_picp=("raw_picp", "mean"),
    ).reset_index()
    for tower in TOWERS:
        sub = per_anchor_tower[per_anchor_tower.eval_tower == tower].dropna(subset=["conformal_picp"])
        if len(sub) >= 3:
            corr = sub["shift_score"].corr(sub["conformal_picp"])
            print(f"  Tower {tower}: n={len(sub)}, corr={corr:.3f}")
        else:
            print(f"  Tower {tower}: n={len(sub)} (too few non-NaN points for a correlation)")

    per_anchor_tower.to_csv(f"{RESULTS}/u03_shift_vs_picp_per_anchor_tower.csv", index=False)
    print(f"\n[OK] Saved u03_shift_vs_picp_per_anchor_tower.csv ({len(per_anchor_tower)} rows)")

    # Identify the single most-shifted (anchor, tower) among T4/T9 only (T2 excluded -- confounded
    # by its own separate, already-documented calibration-availability failure).
    candidates = per_anchor_tower[per_anchor_tower.eval_tower.isin([4, 9])].dropna(subset=["shift_score"])
    top = candidates.sort_values("shift_score", ascending=False).iloc[0]
    print(f"\nMost-shifted (anchor, tower) among T4/T9: anchor={int(top.anchor_year)}, "
          f"tower={int(top.eval_tower)}, shift_score={top.shift_score:.3f}, "
          f"conformal_picp={top.conformal_picp:.3f}, raw_picp={top.raw_picp:.3f}")

    # Highlighted fan charts for this anchor/tower, every model, reusing u02_fanchart_plots.py's
    # exact plotting function/visual convention unmodified.
    import u02_fanchart_plots as ufp
    ufp.FIG_DIR = Path(RESULTS) / "figures" / "u03_fancharts"
    ufp.FIG_DIR.mkdir(parents=True, exist_ok=True)

    chains = pd.read_csv(f"{RESULTS}/u02_chains.csv", parse_dates=["date"])
    hl_tower, hl_yr = int(top.eval_tower), int(top.anchor_year)
    hl_anchor = pd.Timestamp(f"{hl_yr}-12-16")
    n_saved = 0
    for model, chain_sub in chains[(chains.eval_tower == hl_tower) & (chains.anchor_year == hl_yr)].groupby("model"):
        m_sub = summary[(summary.eval_tower == hl_tower) & (summary.anchor_year == hl_yr) & (summary.model == model)]
        margins_by_bin = dict(zip(m_sub["bin"], m_sub["conformal_margin"]))
        ufp.plot_fanchart(T[hl_tower], chain_sub, margins_by_bin, hl_tower, hl_yr, model, hl_anchor)
        n_saved += 1
    print(f"\n[OK] Saved {n_saved} highlighted most-shifted-anchor fan charts to {ufp.FIG_DIR}")


if __name__ == "__main__":
    main()
