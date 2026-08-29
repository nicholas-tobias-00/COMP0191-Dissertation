"""S-05 + UQ: attaches U-06's flat CQR and U-07's LSU-density-stratified CQR intervals to the new
quantile daily chains from `s05_uq_daily_chains_subset.py`. Pure post-processing -- no new model
calls, no new calibration fitting; reuses U-06's/U-07's already-fitted `u06_u05_cqr_summary.csv` /
`u07_u05_lsu_cqr_summary.csv` margins exactly as computed (TabICLv2, FX_A_SPECIES architecture,
same as S-05's own model -- unlike U-04's BASE+species margins, these are the correct-architecture
ones per D-89's finding that feature space determines error characteristics).

Two things U-05's Step 4 (`apply_to_s05_outputs`) didn't have to handle, because it worked in
%-of-mean space on annual means: this works in raw q05/q95 units on daily chains, so needs an
actual lead-time-bin mapping and (for U-07) an LSU-tier assignment per scenario day.

  1. **Lead-time-bin margins pooled across anchor years.** U-06/U-07's margins are per
     (tower, anchor_year, bin); S-05's scenario points don't have their own anchor_year, so this
     averages across the 5 available anchor years per (tower, bin) -- a flat "typical" margin for
     that tower/lead-bin, not a per-year one.
  2. **Lead times beyond 365 days extrapolate the widest bin's margin flat.** S-05's horizon runs
     to 2050 (thousands of days out); the conformal bins only go to 271-365 days (U-02's original
     design, sized for the forecasting-phase 1-year evaluation window). No calibration evidence
     exists for multi-year-ahead lead times, so the 271-365 margin is held flat rather than
     extrapolated by any trend -- an explicit, stated assumption, not a hidden one. This likely
     UNDERSTATES true uncertainty at year 20+ of the horizon (if anything, error should grow, not
     plateau) -- treat far-horizon CQR bands as a floor, not a ceiling.
  3. **LSU tiers use FULL pooled edges, not leave-one-anchor-out ones.** U-07's own comparison plot
     recomputed edges per anchor (leave-one-out, since it tested real historical anchors that could
     leak). S-05's scenario points are all genuinely future and never overlap the calibration
     window, so -- same reasoning as `precompute_aoa()`'s own choice in `s05_trajectory_2050.py`
     -- the full pooled U-05 calibration set's `fx_lsu_dens` distribution per tower is the right
     edge basis here.

Run from project root:  python notebooks/07_scenario_analysis/s05_uq_cqr_apply.py
"""
import sys

import numpy as np
import pandas as pd

ROOT = r"c:\Users\Nicholas\Documents\COMP0191 MSc Artificial Intelligence for Sustainable Development Project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\src")

import models.recursive_rollout as rr

HOURLY = rf"{ROOT}\data\Hourly"
RESULTS = rf"{ROOT}\results"

TOWERS = [2, 4, 9]
BINS = ((1, 7), (8, 30), (31, 90), (91, 180), (181, 270), (271, 365))
WIDEST_BIN = "271-365"
LSU_TIERS = ["low", "mid", "high"]


def lsu_tier(vals, edges):
    return np.where(vals <= edges[0], "low", np.where(vals <= edges[1], "mid", "high"))


def tower_anchor(dft):
    return dft.loc[dft["y_observed"].notna()].index.max()


def load_pooled_margins():
    flat = pd.read_csv(f"{RESULTS}/u06_u05_cqr_summary.csv")
    flat_margin = flat.groupby(["eval_tower", "bin"])["cqr_margin"].mean().to_dict()

    lsu = pd.read_csv(f"{RESULTS}/u07_u05_lsu_cqr_summary.csv")
    lsu_margin = lsu.groupby(["eval_tower", "bin", "lsu_tier"])["lsu_cqr_margin"].mean().to_dict()
    return flat_margin, lsu_margin


def load_lsu_edges():
    """Per-tower LSU tertile edges from U-05's own pooled calibration set (all 5 anchors, full
    historical fx_lsu_dens -- see module docstring point 3)."""
    dv = pd.read_csv(f"{HOURLY}/forecast_daily_v3.csv", low_memory=False)
    dv["Datetime"] = pd.to_datetime(dv["Datetime"], format="mixed")
    T = {t: dv[dv.tower == t].set_index("Datetime").sort_index() for t in TOWERS}

    chains = pd.read_csv(f"{RESULTS}/u05_chains.csv", parse_dates=["date"])
    edges = {}
    for tower in TOWERS:
        sub = chains[chains.eval_tower == tower].merge(
            T[tower][["fx_lsu_dens"]].reset_index().rename(columns={"Datetime": "date"}), on="date", how="left")
        edges[tower] = sub["fx_lsu_dens"].dropna().quantile([1 / 3, 2 / 3]).values
    return edges


def anchors_by_tower():
    dv = pd.read_csv(f"{HOURLY}/forecast_daily_v3.csv", low_memory=False)
    dv["Datetime"] = pd.to_datetime(dv["Datetime"], format="mixed")
    T = {t: dv[dv.tower == t].set_index("Datetime").sort_index() for t in TOWERS}
    return {t: tower_anchor(T[t]) for t in TOWERS}


def apply_cqr(df, flat_margin, lsu_margin, lsu_edges, anchors):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    lead_bins = []
    for tower, ts in zip(df["tower"], df["timestamp"]):
        lead = (ts - anchors[tower]).days
        label = None
        for lo, hi in BINS:
            if lo <= lead <= hi:
                label = f"{lo}-{hi}"
                break
        if label is None and lead > BINS[-1][1]:
            label = WIDEST_BIN  # flat extrapolation beyond calibrated horizon (see module docstring)
        lead_bins.append(label)
    df["lead_bin"] = lead_bins

    df["lsu_tier"] = [lsu_tier(np.array([v]), lsu_edges[t])[0] if pd.notna(v) else np.nan
                       for v, t in zip(df["fx_lsu_dens"], df["tower"])]

    df["flat_margin"] = [flat_margin.get((t, b), np.nan) for t, b in zip(df["tower"], df["lead_bin"])]
    df["lsu_cqr_margin"] = [lsu_margin.get((t, b, tier), np.nan)
                             for t, b, tier in zip(df["tower"], df["lead_bin"], df["lsu_tier"])]

    df["u06_lo"] = df["q05"] - df["flat_margin"]
    df["u06_hi"] = df["q95"] + df["flat_margin"]
    df["u07_lo"] = df["q05"] - df["lsu_cqr_margin"]
    df["u07_hi"] = df["q95"] + df["lsu_cqr_margin"]
    return df


def main():
    flat_margin, lsu_margin = load_pooled_margins()
    lsu_edges = load_lsu_edges()
    anchors = anchors_by_tower()

    files = {
        "livestock": "s05_livestock_daily_chains_subset_uq.csv",
        "grazing": "s05_practices_grazing_daily_chains_subset_uq.csv",
        "fertilizer": "s05_practices_fertilizer_daily_chains_subset_uq.csv",
    }
    for name, fname in files.items():
        df = pd.read_csv(f"{RESULTS}/{fname}")
        out = apply_cqr(df, flat_margin, lsu_margin, lsu_edges, anchors)
        out_path = f"{RESULTS}/s05_{name}_with_cqr.csv"
        out.to_csv(out_path, index=False)
        n_valid = out["u07_lo"].notna().sum()
        print(f"[OK] {out_path} ({len(out)} rows, {n_valid} with valid CQR margin "
              f"[{n_valid/len(out)*100:.0f}%, T2 expected all-NaN])")


if __name__ == "__main__":
    main()
