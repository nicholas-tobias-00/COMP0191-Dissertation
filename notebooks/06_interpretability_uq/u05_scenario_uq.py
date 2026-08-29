"""U-05: scenario-analysis UQ ("Option B" of the two-part plan; Option A = U-04). Builds a
conformal calibration set in S-05's OWN architecture/feature space -- TabICLv2 zero-shot,
`FX_A_SPECIES` (13 cols: S-03's Variant A + F-10's species split), NOT U-04's `BASE+species` (52
cols) -- because S-05 deliberately uses a narrower feature set (S-03's whole point) and a
different feature space means a genuinely different model with different error characteristics.
U-04's calibration is not reusable here; this repeats its METHOD on the correct architecture.

Method: same 5 real historical anchors (2018-2022) x 3 towers as U-02/U-03/U-04, same quantiles
(0.05, 0.5, 0.95), same leave-one-anchor-out `rr.conformal_margins_by_bin()`, same PICP/MPIW/
pinball metrics. `evaluate_stage()` imported unmodified from `u02_multi_anchor_tower.py` (third
reuse of this function -- U-04 was the first). This calibration set is a legitimate result on its
own (does TabICLv2 calibrate reasonably on Variant A's reduced feature space, on real data) before
being used for the scenario question.

Step 3 (the actual design question from the plan): does residual magnitude correlate with
AOA-flagged-%? Checked empirically, not assumed -- reuses the EXACT same AOA mechanism S-05 already
computes (`precompute_aoa`/`aoa_flagged_frac`, `scenario_hybrid.dissimilarity_index()`-style
nearest-neighbour distance in FX_A_SPECIES's own 13-dim space, per-tower threshold), so the
stratifier is the same quantity already saved in every S-05 output file, not a new abstraction.

Step 4: applies the resulting calibration to S-05's existing livestock/grazing/fertilizer outputs
-- pure post-processing (a join against each file's own already-saved `aoa_flagged_pct`), no new
model calls, since Step 1's calibration set is the only new inference this experiment needs.

Run from project root:  python notebooks/06_interpretability_uq/u05_scenario_uq.py
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = r"c:\Users\Nicholas\Documents\COMP0191 MSc Artificial Intelligence for Sustainable Development Project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\src")
sys.path.insert(0, ROOT + r"\src\features")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT + r"\notebooks\06_interpretability_uq")

import models.recursive_rollout as rr
from build_transient_scenario_drivers_species import FX_A_SPECIES
from u02_multi_anchor_tower import evaluate_stage, QUANTILES, ANCHOR_YEARS, TOWERS, N_DAYS

from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))

HOURLY = rf"{ROOT}\data\Hourly"
RESULTS = rf"{ROOT}\results"


# ============================================================ Step 1-2: calibration set
def fit_stage_scenario(dv, T):
    """TabICLv2, FX_A_SPECIES config, zero-shot per tower/anchor -- same chain schema as U-02/U-04
    (anchor_year, eval_tower, model, date, q05, median, q95, y_true) so evaluate_stage() works
    unmodified, PLUS an aoa_dist column (nearest-neighbour distance to real training data in
    FX_A_SPECIES's own space) for Step 3's correlation check.

    AOA training set is PRE-ANCHOR-ONLY, recomputed fresh per anchor -- unlike S-05's own
    precompute_aoa() (which correctly uses the FULL historical record, since S-05's scenario dates
    are all genuinely future and never overlap it), U-05 tests on REAL HISTORICAL anchors, so a
    test point can be a literal row already inside an unrestricted training set (distance-to-self
    = 0) -- caught directly via a smoke test showing aoa_dist uniformly 0.0 before this fix."""
    print(f"[U-05] FX_A_SPECIES: {len(FX_A_SPECIES)} columns")
    rows = []
    t0 = time.time()

    for tower in TOWERS:
        dft = T[tower]

        for yr in ANCHOR_YEARS:
            anchor = pd.Timestamp(f"{yr}-12-16")
            target_dates = pd.date_range(anchor + pd.Timedelta(days=1), periods=N_DAYS, freq="D")
            t_a = time.time()

            hist = dft.loc[:anchor]
            hist_target = hist["y_observed"]
            hist_cov = hist[FX_A_SPECIES]
            future_cov = dft.loc[target_dates, FX_A_SPECIES]
            y_true_full = pd.Series(dft.loc[target_dates, "y_observed"].values, index=target_dates)

            X_train = hist[FX_A_SPECIES].dropna().values
            scaler = StandardScaler().fit(X_train)
            Xtr = scaler.transform(X_train)
            d_train = cdist(Xtr, Xtr)
            np.fill_diagonal(d_train, np.inf)
            d_loo = d_train.min(axis=1)
            q1, q3 = np.percentile(d_loo, [25, 75])
            aoa_threshold = q3 + 1.5 * (q3 - q1)

            d_scenario = cdist(scaler.transform(future_cov.values), Xtr).min(axis=1)
            aoa_dist_by_date = dict(zip(target_dates, d_scenario))

            try:
                df_tabicl = rr.tabicl_forecast(hist_target, hist_cov, future_cov, quantiles=list(QUANTILES))
                for d in target_dates:
                    rows.append({"anchor_year": yr, "eval_tower": tower, "model": "TabICLv2", "date": d,
                                 "q05": df_tabicl.loc[d, 0.05], "median": df_tabicl.loc[d, "median"],
                                 "q95": df_tabicl.loc[d, 0.95], "y_true": y_true_full.loc[d],
                                 "aoa_dist": aoa_dist_by_date[d], "aoa_flagged": aoa_dist_by_date[d] > aoa_threshold})
            except Exception as e:
                print(f"    T{tower} {yr} TabICLv2 SKIPPED: {str(e)[:150]}")

            print(f"  T{tower} anchor {yr} done ({time.time()-t_a:.0f}s, {time.time()-t0:.0f}s elapsed)")

    return pd.DataFrame(rows)


# ============================================================ Step 3: AOA-residual correlation
def aoa_residual_check(chains):
    """Does |residual| correlate with AOA distance/flagged status, in REAL historical data? The
    actual empirical question the plan flagged -- resolved here, not assumed either way. Computed
    BOTH pooled (for the headline finding) and per-tower (Step 4 needs per-tower margins -- T2 is
    known-degenerate and shouldn't be pooled into T4/T9's numbers)."""
    df = chains.copy()
    df["abs_resid"] = (df["y_true"] - df["median"]).abs()
    df = df.dropna(subset=["abs_resid", "aoa_dist"])

    def summarize(g):
        corr = g["abs_resid"].corr(g["aoa_dist"])
        fm = g.groupby("aoa_flagged")["abs_resid"].agg(["mean", "count"])
        return pd.Series({
            "pearson_corr_resid_vs_aoa_dist": corr,
            "mean_abs_resid_in_aoa": fm.loc[False, "mean"] if False in fm.index else np.nan,
            "mean_abs_resid_out_aoa": fm.loc[True, "mean"] if True in fm.index else np.nan,
            "n_in_aoa": fm.loc[False, "count"] if False in fm.index else 0,
            "n_out_aoa": fm.loc[True, "count"] if True in fm.index else 0,
        })

    pooled = summarize(df).to_frame().T
    pooled.to_csv(f"{RESULTS}/u05_aoa_residual_correlation.csv", index=False)
    print(f"\n[OK] u05_aoa_residual_correlation.csv (pooled)\n{pooled.to_string(index=False)}")

    by_tower = df.groupby("eval_tower").apply(summarize, include_groups=False)
    by_tower.to_csv(f"{RESULTS}/u05_aoa_residual_correlation_by_tower.csv")
    print(f"\n[OK] u05_aoa_residual_correlation_by_tower.csv\n{by_tower.round(3).to_string()}")

    return pooled, by_tower


# ============================================================ Step 4: apply to S-05 outputs
def apply_to_s05_outputs(margin_in_pct_by_tower, margin_out_pct_by_tower):
    """Attaches a calibrated interval to S-05's existing annual-mean outputs (livestock/grazing/
    fertilizer), joined purely on each row's own already-saved aoa_flagged_pct -- no new model
    calls. Two-tier margin (Step 3's actual empirical finding, not a flat one): interpolates
    linearly between the in-AOA and out-of-AOA margin-as-%-of-mean, weighted by that row's OWN
    aoa_flagged_pct (0-100) -- a row flagged 0% uses the in-AOA margin, 100% uses the out-of-AOA
    margin, in between blends continuously by its own flagged fraction. Interval width is a
    %-of-mean conversion since S-05's outputs are annual means, not the daily points the
    calibration margins were measured on -- stated explicitly, not silently assumed equivalent."""
    files = {
        "livestock": "s05_trajectory_realizations_2050.csv",
        "grazing": "s05_practices_grazing.csv",
        "fertilizer": "s05_practices_fertilizer.csv",
    }
    out_paths = {}
    for name, fname in files.items():
        path = f"{RESULTS}/{fname}"
        if not os.path.exists(path):
            print(f"  [SKIP] {fname} not found")
            continue
        df = pd.read_csv(path)
        m_in = df["tower"].map(margin_in_pct_by_tower)
        m_out = df["tower"].map(margin_out_pct_by_tower)
        frac = (df["aoa_flagged_pct"] / 100.0).clip(0, 1)
        df["uq_margin_pct"] = m_in + frac * (m_out - m_in)
        df["uq_lo"] = df["annual_mean"] * (1 - df["uq_margin_pct"])
        df["uq_hi"] = df["annual_mean"] * (1 + df["uq_margin_pct"])
        df["uq_valid"] = df["tower"].map(lambda t: t in margin_in_pct_by_tower and pd.notna(margin_in_pct_by_tower[t]))
        out_path = f"{RESULTS}/u05_{name}_with_uq.csv"
        df.to_csv(out_path, index=False)
        out_paths[name] = out_path
        print(f"  [OK] {out_path} ({len(df)} rows)")
    return out_paths


def main():
    dv = pd.read_csv(f"{HOURLY}/forecast_daily_v3.csv", low_memory=False)
    dv["Datetime"] = pd.to_datetime(dv["Datetime"], format="mixed")
    T = {t: dv[dv.tower == t].set_index("Datetime").sort_index() for t in TOWERS}

    print("="*70)
    print("U-05 STEP 1: TabICLv2 FX_A_SPECIES zero-shot quantile rollout (calibration set)")
    print("="*70)
    chains = fit_stage_scenario(dv, T)
    chains.to_csv(f"{RESULTS}/u05_chains.csv", index=False)
    print(f"\n[OK] Saved u05_chains.csv ({len(chains)} rows)")

    print("\n" + "="*70)
    print("U-05 STEP 2: leave-one-anchor-out conformal calibration")
    print("="*70)
    summary = evaluate_stage(chains.rename(columns={}))
    summary.to_csv(f"{RESULTS}/u05_summary.csv", index=False)
    print(f"[OK] Saved u05_summary.csv ({len(summary)} rows)")

    def wavg(g, col):
        vals = g[col]
        if vals.isna().all():
            return np.nan
        w = g["n"]
        return (vals * w).sum() / w.sum() if w.sum() > 0 else np.nan

    agg = summary.groupby("eval_tower").apply(
        lambda g: pd.Series({
            "raw_picp": wavg(g, "raw_picp"), "raw_mpiw": wavg(g, "raw_mpiw"),
            "conformal_picp": wavg(g, "conformal_picp"), "conformal_mpiw": wavg(g, "conformal_mpiw"),
            "conformal_pinball": wavg(g, "conformal_pinball"),
        }), include_groups=False).reset_index()
    print(agg.round(4).to_string(index=False))

    print("\n" + "="*70)
    print("U-05 STEP 3: does |residual| correlate with AOA-flagged status? (empirical check)")
    print("="*70)
    aoa_pooled, aoa_by_tower = aoa_residual_check(chains)

    print("\n" + "="*70)
    print("U-05 STEP 4: apply two-tier calibration to S-05's existing livestock/grazing/fertilizer outputs")
    print("="*70)
    margin_in_pct_by_tower, margin_out_pct_by_tower = {}, {}
    for tower in TOWERS:
        # Gate on Step 2's own conformal_mpiw, not just whether AOA-stratified residuals exist --
        # T2 already failed proper leave-one-anchor-out calibration in Step 2 (only 1 anchor has
        # real y_observed at all, so there's no "other anchor" to pool calibration residuals from).
        # Falling back to a cruder raw-residual split for T2 here would quietly re-introduce a
        # weaker standard than Step 2 already established was invalid -- and T2's out-of-AOA sample
        # is n=4, too small to trust regardless (its mean even reverses direction vs. T4/T9, almost
        # certainly noise). Kept consistently NaN, matching Step 2's own honest finding.
        conformal_ok = summary[(summary.eval_tower == tower)]["conformal_mpiw"].notna().any()
        mean_level = chains[chains.eval_tower == tower]["y_true"].mean()
        if conformal_ok and tower in aoa_by_tower.index and mean_level:
            row = aoa_by_tower.loc[tower]
            margin_in_pct_by_tower[tower] = row["mean_abs_resid_in_aoa"] / mean_level if pd.notna(row["mean_abs_resid_in_aoa"]) else np.nan
            margin_out_pct_by_tower[tower] = row["mean_abs_resid_out_aoa"] / mean_level if pd.notna(row["mean_abs_resid_out_aoa"]) else np.nan
        else:
            margin_in_pct_by_tower[tower] = np.nan
            margin_out_pct_by_tower[tower] = np.nan
    print("In-AOA margin-as-%-of-mean:", margin_in_pct_by_tower)
    print("Out-of-AOA margin-as-%-of-mean:", margin_out_pct_by_tower)
    out_paths = apply_to_s05_outputs(margin_in_pct_by_tower, margin_out_pct_by_tower)

    return chains, summary, (aoa_pooled, aoa_by_tower), out_paths


if __name__ == "__main__":
    main()
