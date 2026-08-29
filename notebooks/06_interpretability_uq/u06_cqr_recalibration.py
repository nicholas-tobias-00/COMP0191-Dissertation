"""U-06: Conformalized Quantile Regression (CQR, Romano et al. 2019) recalibration of U-04/U-05's
already-saved chains -- direct follow-up to the finding that split-conformal's flat, symmetric
per-bin margin (anchored to the MEDIAN) catastrophically undercovers spike days: checked directly
on U-04's chains, 75% of the top-10%-magnitude days fell entirely outside the "90%" interval, vs.
3.3% for the bottom 90% -- the median (~35 nmol) massively undershoots spikes (~193 nmol mean), and
a constant additive margin around it can't fix that.

Motivated by a second direct check before building anything: does the model's own RAW (uncalibrated)
q95 already track spike magnitude better than the median does? Yes -- on spike days, raw q95 sits
close to (TabPFN) or already exceeds (TabICLv2) the actual mean spike value, while raw spread
(q95-q05) widens 1.3-1.8x on spike days vs. normal days. CQR exploits exactly this: instead of
conformalizing a symmetric margin around the median, conformalize the model's own ASYMMETRIC raw
quantiles directly -- nonconformity score = max(q05-y_true, y_true-q95), calibrated interval =
[q05-margin, q95+margin]. Because q05/q95 already vary day-to-day with the model's own confidence,
the calibrated interval inherits that adaptivity instead of using one flat width regardless of day.

Reuses `rr.conformal_margins_by_bin()` COMPLETELY UNCHANGED -- it's already generic over whatever
nonconformity-score array it's given; CQR only changes what's computed and how the resulting margin
is applied (asymmetric bounds instead of median +/- margin), not the calibration function itself.
No new model calls anywhere -- this recalibrates U-04's and U-05's already-saved chains
(`u04_chains.csv`, `u05_chains.csv`), a pure re-scoring exercise.

Run from project root:  python notebooks/06_interpretability_uq/u06_cqr_recalibration.py
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

RESULTS = rf"{ROOT}\results"
BINS = ((1, 7), (8, 30), (31, 90), (91, 180), (181, 270), (271, 365))
ALPHA = 0.10
QUANTILES = (0.05, 0.5, 0.95)


def evaluate_cqr(chains, anchor_years, towers):
    """Leave-one-anchor-out CQR calibration + evaluation, per model/tower/bin. Mirrors
    u02_multi_anchor_tower.py's evaluate_stage() structure exactly, but the nonconformity score is
    CQR's max(q05-y_true, y_true-q95) instead of |y_true-median|, and the calibrated interval is
    asymmetric [q05-margin, q95+margin] instead of [median-margin, median+margin]."""
    chains = chains.copy()
    chains["date"] = pd.to_datetime(chains["date"])
    summary_rows = []

    for model in chains["model"].unique():
        for tower in towers:
            sub_all = chains[(chains.model == model) & (chains.eval_tower == tower)]
            if sub_all.empty:
                continue

            for test_yr in anchor_years:
                anchor = pd.Timestamp(f"{test_yr}-12-16")
                test_sub = sub_all[sub_all.anchor_year == test_yr]
                if test_sub.empty:
                    continue
                test_bins = rr.lead_time_bin(test_sub["date"].values, anchor, BINS)

                calib_sub = sub_all[sub_all.anchor_year != test_yr]
                calib_scores_by_bin = {}
                for calib_yr in calib_sub["anchor_year"].unique():
                    c_anchor = pd.Timestamp(f"{calib_yr}-12-16")
                    c_rows = calib_sub[calib_sub.anchor_year == calib_yr]
                    c_bins = rr.lead_time_bin(c_rows["date"].values, c_anchor, BINS)
                    # CQR nonconformity score -- the only substantive change from split-conformal
                    score = np.maximum(c_rows["q05"].values - c_rows["y_true"].values,
                                        c_rows["y_true"].values - c_rows["q95"].values)
                    for lo, hi in BINS:
                        label = f"{lo}-{hi}"
                        mask = (c_bins == label) & np.isfinite(score)
                        calib_scores_by_bin.setdefault(label, []).extend(score[mask].tolist())

                # SAME function as split-conformal -- already generic, not modified
                margins = rr.conformal_margins_by_bin(calib_scores_by_bin, alpha=ALPHA)

                for lo, hi in BINS:
                    label = f"{lo}-{hi}"
                    m = test_bins == label
                    if m.sum() < 3:
                        continue
                    yt = test_sub["y_true"].values[m]
                    med = test_sub["median"].values[m]
                    q05 = test_sub["q05"].values[m]
                    q95 = test_sub["q95"].values[m]
                    real_mask = np.isfinite(yt)
                    if real_mask.sum() < 3:
                        continue
                    yt_r, med_r, q05_r, q95_r = yt[real_mask], med[real_mask], q05[real_mask], q95[real_mask]

                    margin = margins.get(label, np.nan)
                    cqr_lo, cqr_hi = q05_r - margin, q95_r + margin

                    summary_rows.append({
                        "anchor_year": test_yr, "eval_tower": tower, "model": model, "bin": label,
                        "n": int(real_mask.sum()),
                        "cqr_picp": picp(yt_r, cqr_lo, cqr_hi),
                        "cqr_mpiw": mpiw(cqr_lo, cqr_hi),
                        "cqr_pinball": pinball(yt_r, {0.05: cqr_lo, 0.5: med_r, 0.95: cqr_hi}, QUANTILES),
                        "cqr_margin": margin,
                    })

    return pd.DataFrame(summary_rows)


def spike_coverage_check(chains, cqr_summary, old_summary, anchor_years, towers, label):
    """Direct before/after comparison on the exact question that motivated this: what fraction of
    top-10%-magnitude (spike) days fall outside the interval, old (symmetric) vs. new (CQR)?"""
    chains = chains.copy()
    chains["date"] = pd.to_datetime(chains["date"])
    rows = []

    for model in chains["model"].unique():
        for tower in towers:
            sub_all = chains[(chains.model == model) & (chains.eval_tower == tower)]
            if sub_all.empty:
                continue
            p90 = sub_all["y_true"].quantile(0.9)

            for test_yr in anchor_years:
                anchor = pd.Timestamp(f"{test_yr}-12-16")
                test_sub = sub_all[sub_all.anchor_year == test_yr].copy()
                if test_sub.empty:
                    continue
                test_sub["bin"] = rr.lead_time_bin(test_sub["date"].values, anchor, BINS)

                old_m = old_summary[(old_summary.eval_tower == tower) & (old_summary.anchor_year == test_yr) & (old_summary.model == model)]
                cqr_m = cqr_summary[(cqr_summary.eval_tower == tower) & (cqr_summary.anchor_year == test_yr) & (cqr_summary.model == model)]
                old_margins = dict(zip(old_m["bin"], old_m["conformal_margin"]))
                cqr_margins = dict(zip(cqr_m["bin"], cqr_m["cqr_margin"]))

                test_sub["old_margin"] = test_sub["bin"].map(old_margins)
                test_sub["cqr_margin"] = test_sub["bin"].map(cqr_margins)
                test_sub = test_sub.dropna(subset=["y_true", "old_margin", "cqr_margin"])
                if test_sub.empty:
                    continue

                test_sub["old_lo"] = test_sub["median"] - test_sub["old_margin"]
                test_sub["old_hi"] = test_sub["median"] + test_sub["old_margin"]
                test_sub["cqr_lo"] = test_sub["q05"] - test_sub["cqr_margin"]
                test_sub["cqr_hi"] = test_sub["q95"] + test_sub["cqr_margin"]

                spike = test_sub[test_sub.y_true >= p90]
                normal = test_sub[test_sub.y_true < p90]
                if len(spike) == 0:
                    continue

                rows.append({
                    "model": model, "tower": tower, "anchor_year": test_yr,
                    "n_spike": len(spike), "n_normal": len(normal),
                    "old_spike_coverage": 1 - ((spike.y_true < spike.old_lo) | (spike.y_true > spike.old_hi)).mean(),
                    "cqr_spike_coverage": 1 - ((spike.y_true < spike.cqr_lo) | (spike.y_true > spike.cqr_hi)).mean(),
                    "old_normal_coverage": 1 - ((normal.y_true < normal.old_lo) | (normal.y_true > normal.old_hi)).mean() if len(normal) else np.nan,
                    "cqr_normal_coverage": 1 - ((normal.y_true < normal.cqr_lo) | (normal.y_true > normal.cqr_hi)).mean() if len(normal) else np.nan,
                    "old_mpiw_mean": (spike.old_hi - spike.old_lo).mean(),
                    "cqr_mpiw_mean": (spike.cqr_hi - spike.cqr_lo).mean(),
                })

    out = pd.DataFrame(rows)
    out.to_csv(f"{RESULTS}/u06_spike_coverage_{label}.csv", index=False)

    def wavg(col, wcol):
        d = out.dropna(subset=[col])
        return (d[col] * d[wcol]).sum() / d[wcol].sum() if d[wcol].sum() > 0 else np.nan

    agg = out.groupby("model").apply(lambda g: pd.Series({
        "old_spike_coverage": wavg("old_spike_coverage", "n_spike") if len(g) else np.nan,
        "cqr_spike_coverage": (g["cqr_spike_coverage"] * g["n_spike"]).sum() / g["n_spike"].sum(),
        "old_normal_coverage": (g["old_normal_coverage"] * g["n_normal"]).sum() / g["n_normal"].sum(),
        "cqr_normal_coverage": (g["cqr_normal_coverage"] * g["n_normal"]).sum() / g["n_normal"].sum(),
        "old_mpiw_spike_mean": g["old_mpiw_mean"].mean(),
        "cqr_mpiw_spike_mean": g["cqr_mpiw_mean"].mean(),
    }), include_groups=False)
    print(f"\n[{label}] Spike (top 10%) vs. normal-day coverage, OLD (symmetric) vs. CQR:")
    print(agg.round(3).to_string())
    return out, agg


def main():
    for label, chains_file, summary_file, anchor_years, towers in [
        ("U04", "u04_chains.csv", "u04_summary.csv", [2018, 2019, 2020, 2021, 2022], [2, 4, 9]),
        ("U05", "u05_chains.csv", "u05_summary.csv", [2018, 2019, 2020, 2021, 2022], [2, 4, 9]),
    ]:
        print("=" * 70)
        print(f"U-06: CQR recalibration of {label}'s chains")
        print("=" * 70)
        chains = pd.read_csv(f"{RESULTS}/{chains_file}")
        old_summary = pd.read_csv(f"{RESULTS}/{summary_file}")

        cqr_summary = evaluate_cqr(chains, anchor_years, towers)
        cqr_summary.to_csv(f"{RESULTS}/u06_{label.lower()}_cqr_summary.csv", index=False)
        print(f"[OK] Saved u06_{label.lower()}_cqr_summary.csv ({len(cqr_summary)} rows)")

        def wavg(g, col):
            # Explicit all-NaN guard (same footgun u02_multi_anchor_tower.py's own wavg() already
            # flags): pandas' .sum() on an all-NaN column silently returns 0.0 (sum of nothing),
            # not NaN -- without this, T2's fully-missing calibration would misleadingly report as
            # a confident 0.0 rather than "no data" (caught directly: per-bin cqr_picp rows are
            # correctly NaN for T2, but the aggregate printed 0.0 before this fix).
            vals = g[col]
            if vals.isna().all():
                return np.nan
            return (vals * g["n"]).sum() / g["n"].sum() if g["n"].sum() > 0 else np.nan

        agg = cqr_summary.groupby(["model", "eval_tower"]).apply(
            lambda g: pd.Series({"cqr_picp": wavg(g, "cqr_picp"), "cqr_mpiw": wavg(g, "cqr_mpiw"),
                                  "cqr_pinball": wavg(g, "cqr_pinball")}), include_groups=False
        ).reset_index()
        print(agg.round(4).to_string(index=False))

        spike_coverage_check(chains, cqr_summary, old_summary, anchor_years, towers, label)


if __name__ == "__main__":
    main()
