"""U-07: livestock-density-stratified CQR -- direct follow-up to a user question ("can't the
margin be thinner where livestock presence is smaller?"), checked empirically before building
anything (matching this session's standing practice). Confirmed a MUCH stronger signal than the
AOA-distance check U-05 ran: corr(|residual|, fx_lsu_dens) = 0.43-0.45 (vs. AOA-distance's weak
0.09-0.15), and residuals are ~3.2x larger on above-median-LSU days (51.0 vs. 15.9-16.2 mean |resid|).
fx_cattle_dens correlates almost identically (0.427) -- consistent with S-05's own species finding
that cattle dominates; sheep/lamb correlate at noise level (0.03-0.05), so fx_lsu_dens (the
aggregate, standard, interpretable variable) is used directly, not a species-specific one.

U-06's CQR already partially captures this implicitly -- TabPFN/TabICLv2's raw q05/q95 are
themselves functions of fx_lsu_dens (a model input), so raw spread already correlates with it
(0.46-0.63, checked directly). What's NOT yet livestock-density-aware is the ADDITIVE calibration
margin layered on top, which U-06 binned only by lead-time (U-04) or lead-time x AOA-flagged (U-05)
-- never by the covariate driving the actual heteroscedasticity most strongly. This adds that axis.

Method: identical CQR machinery to U-06 (nonconformity = max(q05-y_true, y_true-q95),
`rr.conformal_margins_by_bin()` reused unchanged a fifth time) -- only the bin KEY changes, from
lead-time-bin alone to lead-time-bin x LSU-tertile (combined string key, e.g. "31-90_low" --
conformal_margins_by_bin() needs no code change at all to support this, it was already generic
over arbitrary dict keys). Tertile boundaries computed from the LEAVE-IN (calibration) anchors only
per test fold, never the held-out test anchor -- same no-leakage discipline as every other
leave-one-anchor-out step in this project.

No new model calls -- recalibrates U-04's/U-05's already-saved chains a second time (first was
U-06's CQR margin; this is CQR + LSU-density stratification, a direct extension, not a restart).

Run from project root:  python notebooks/06_interpretability_uq/u07_lsu_stratified_cqr.py
"""
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = r"c:\Users\Nicholas\Documents\COMP0191 MSc Artificial Intelligence for Sustainable Development Project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\src")

import models.recursive_rollout as rr
from evaluation.metrics import pinball, picp, mpiw

HOURLY = rf"{ROOT}\data\Hourly"
RESULTS = rf"{ROOT}\results"
BINS = ((1, 7), (8, 30), (31, 90), (91, 180), (181, 270), (271, 365))
ALPHA = 0.10
QUANTILES = (0.05, 0.5, 0.95)
LSU_TIERS = ["low", "mid", "high"]


def load_lsu():
    dv = pd.read_csv(f"{HOURLY}/forecast_daily_v3.csv", low_memory=False)
    dv["Datetime"] = pd.to_datetime(dv["Datetime"], format="mixed")
    return dv[["Datetime", "tower", "fx_lsu_dens"]].rename(columns={"Datetime": "date", "tower": "eval_tower"})


def lsu_tier(vals, edges):
    """edges: (low/mid boundary, mid/high boundary), from calibration data only."""
    return np.where(vals <= edges[0], "low", np.where(vals <= edges[1], "mid", "high"))


def evaluate_lsu_cqr(chains, lsu, anchor_years, towers):
    chains = chains.copy()
    chains["date"] = pd.to_datetime(chains["date"])
    chains = chains.merge(lsu, on=["date", "eval_tower"], how="left")
    summary_rows = []

    for model in chains["model"].unique():
        for tower in towers:
            sub_all = chains[(chains.model == model) & (chains.eval_tower == tower)]
            if sub_all.empty:
                continue

            for test_yr in anchor_years:
                anchor = pd.Timestamp(f"{test_yr}-12-16")
                test_sub = sub_all[sub_all.anchor_year == test_yr].copy()
                if test_sub.empty:
                    continue
                test_sub["lead_bin"] = rr.lead_time_bin(test_sub["date"].values, anchor, BINS)

                calib_sub = sub_all[sub_all.anchor_year != test_yr].dropna(subset=["fx_lsu_dens"])
                if calib_sub.empty:
                    continue
                edges = calib_sub["fx_lsu_dens"].quantile([1/3, 2/3]).values
                calib_sub = calib_sub.copy()
                calib_sub["lsu_tier"] = lsu_tier(calib_sub["fx_lsu_dens"].values, edges)

                calib_scores_by_bin = {}
                for calib_yr in calib_sub["anchor_year"].unique():
                    c_anchor = pd.Timestamp(f"{calib_yr}-12-16")
                    c_rows = calib_sub[calib_sub.anchor_year == calib_yr].copy()
                    c_rows["lead_bin"] = rr.lead_time_bin(c_rows["date"].values, c_anchor, BINS)
                    score = np.maximum(c_rows["q05"].values - c_rows["y_true"].values,
                                        c_rows["y_true"].values - c_rows["q95"].values)
                    for lo, hi in BINS:
                        lbl = f"{lo}-{hi}"
                        for tier in LSU_TIERS:
                            key = f"{lbl}_{tier}"
                            mask = (c_rows["lead_bin"] == lbl) & (c_rows["lsu_tier"] == tier) & np.isfinite(score)
                            calib_scores_by_bin.setdefault(key, []).extend(score[mask].tolist())

                margins = rr.conformal_margins_by_bin(calib_scores_by_bin, alpha=ALPHA)

                test_sub = test_sub.dropna(subset=["fx_lsu_dens"])
                test_sub["lsu_tier"] = lsu_tier(test_sub["fx_lsu_dens"].values, edges)

                for lo, hi in BINS:
                    lbl = f"{lo}-{hi}"
                    for tier in LSU_TIERS:
                        m = (test_sub["lead_bin"] == lbl) & (test_sub["lsu_tier"] == tier)
                        if m.sum() < 3:
                            continue
                        rows = test_sub[m]
                        yt = rows["y_true"].values
                        real_mask = np.isfinite(yt)
                        if real_mask.sum() < 3:
                            continue
                        yt_r = yt[real_mask]
                        med_r = rows["median"].values[real_mask]
                        q05_r = rows["q05"].values[real_mask]
                        q95_r = rows["q95"].values[real_mask]

                        margin = margins.get(f"{lbl}_{tier}", np.nan)
                        lo_b, hi_b = q05_r - margin, q95_r + margin

                        summary_rows.append({
                            "anchor_year": test_yr, "eval_tower": tower, "model": model,
                            "bin": lbl, "lsu_tier": tier, "n": int(real_mask.sum()),
                            "lsu_cqr_picp": picp(yt_r, lo_b, hi_b),
                            "lsu_cqr_mpiw": mpiw(lo_b, hi_b),
                            "lsu_cqr_pinball": pinball(yt_r, {0.05: lo_b, 0.5: med_r, 0.95: hi_b}, QUANTILES),
                            "lsu_cqr_margin": margin,
                        })

    return pd.DataFrame(summary_rows)


def main():
    lsu = load_lsu()

    for label, chains_file in [("U04", "u04_chains.csv"), ("U05", "u05_chains.csv")]:
        print("=" * 70)
        print(f"U-07: LSU-density-stratified CQR recalibration of {label}'s chains")
        print("=" * 70)
        chains = pd.read_csv(f"{RESULTS}/{chains_file}")
        summary = evaluate_lsu_cqr(chains, lsu, [2018, 2019, 2020, 2021, 2022], [2, 4, 9])
        summary.to_csv(f"{RESULTS}/u07_{label.lower()}_lsu_cqr_summary.csv", index=False)
        print(f"[OK] Saved u07_{label.lower()}_lsu_cqr_summary.csv ({len(summary)} rows)")

        def wavg(g, col):
            vals = g[col]
            if vals.isna().all():
                return np.nan
            return (vals * g["n"]).sum() / g["n"].sum() if g["n"].sum() > 0 else np.nan

        agg = summary.groupby(["model", "lsu_tier"]).apply(
            lambda g: pd.Series({
                "picp": wavg(g, "lsu_cqr_picp"), "mpiw": wavg(g, "lsu_cqr_mpiw"),
                "pinball": wavg(g, "lsu_cqr_pinball"), "n": g["n"].sum(),
            }), include_groups=False
        ).reset_index()
        agg["lsu_tier"] = pd.Categorical(agg["lsu_tier"], categories=LSU_TIERS, ordered=True)
        print(agg.sort_values(["model", "lsu_tier"]).round(3).to_string(index=False))
        print()


if __name__ == "__main__":
    main()
