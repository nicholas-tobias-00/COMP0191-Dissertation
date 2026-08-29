"""S-05 analysis: summarizes the already-computed 10-year transient trajectory
(`s05_trajectory_10yr.py`, TabICLv2 + S-03's Variant A feature set + F-10's species split, run
against real CMIP6 transient weather with independent per-species livestock multipliers). Read-only
against the existing output CSV -- runs no models, fits nothing new.

Produces (mirrors S-04's own analysis structure where the comparison is apples-to-apples; adds a
genuinely new piece -- per-species marginal response -- that only Option B's independent
multipliers make possible):
  1. Trajectory summary (mean + realization spread) per tower/ssp/multiplier-combo/year.
  2. Realization-level spread (p10/p90 band) as a fraction of the mean, pooled across years, at
     baseline (1x/1x/1x) -- same question S-04's Finding 1 asked, now for TabICLv2/Variant A.
  3. SSP2-4.5 vs SSP5-8.5 divergence at baseline livestock, early (years 1-5) vs late (years 6-10)
     window POST-ANCHOR -- year-offset-based, not absolute calendar year, since T2's anchor
     (2019-05-31) and T4/T9's (2023-12-29) put them on different absolute year ranges.
  4. AOA-flagged-% trend over the 10-year post-anchor horizon.
  5. NEW: per-species marginal response -- holding the other two species at 1x, how much does
     scaling cattle/sheep/lamb alone move predicted FCH4? Answers this session's actual question
     ("does livestock type matter, not just aggregate density") directly, and checks it against
     each species' real historical LSU-weight contribution (cattle dominates baseline LSU by
     construction -- 74-88% across towers -- so a naive prior would expect cattle scaling to
     dominate the response too; whether the model's response ordering actually matches that prior,
     or diverges from it, is the real question F-10's species-split features were added to let a
     model answer that a single aggregate fx_lsu_dens structurally cannot).
  6. NEW: joint vs. additive check -- is the response to scaling all three species together
     (3x/3x/3x) close to the sum of each species' individual marginal effect, or does the model
     show synergistic/sub-additive behavior? Only checkable because Option B ran the full 27-combo
     grid rather than a single shared multiplier.

Figures: trajectory + spread band (per tower, both SSPs, baseline), AOA-over-time line chart,
per-species marginal response bar chart, joint-vs-additive check.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

ROOT = r"c:\Users\Nicholas\Documents\COMP0191 MSc Artificial Intelligence for Sustainable Development Project"
RESULTS = rf"{ROOT}\results"
FIG_DIR = rf"{RESULTS}\figures\s05_summary"
os.makedirs(FIG_DIR, exist_ok=True)

TOWERS = [2, 4, 9]
SSPS = ["ssp245", "ssp585"]
SPECIES = ["cattle", "sheep", "lamb"]
MULT_COLS = {"cattle": "mult_cattle", "sheep": "mult_sheep", "lamb": "mult_lamb"}
TOWER_ANCHOR_YEAR = {2: 2019, 4: 2023, 9: 2023}  # anchor.year (last real y_observed), per script
TOWER_COLORS = {2: "tab:blue", 4: "tab:orange", 9: "tab:green"}
SPECIES_COLORS = {"cattle": "tab:brown", "sheep": "tab:purple", "lamb": "tab:pink"}


def load_data():
    df = pd.read_csv(f"{RESULTS}/s05_trajectory_realizations.csv")
    df["year_offset"] = df.apply(lambda r: r["year"] - TOWER_ANCHOR_YEAR[r["tower"]], axis=1)
    print(f"[OK] s05_trajectory_realizations.csv: {len(df):,} rows, "
          f"year_offset range {df.year_offset.min()}-{df.year_offset.max()}")
    return df


def trajectory_summary(df):
    g = df.groupby(["tower", "ssp", "mult_cattle", "mult_sheep", "mult_lamb", "year_offset"])["annual_mean"]
    out = g.agg(mean="mean", std="std", n="count",
                p10=lambda s: s.quantile(0.10), p90=lambda s: s.quantile(0.90)).reset_index()
    out.to_csv(f"{RESULTS}/s05_trajectory_summary.csv", index=False)
    print(f"[OK] s05_trajectory_summary.csv ({len(out)} rows)")
    return out


def realization_spread(df):
    """Baseline (1x/1x/1x) only. TWO decompositions, deliberately kept separate -- pooling
    year+realization+GCM together (as S-04's own Finding 1 metric does, for direct comparability)
    conflates two very different sources of variation, and a spot-check (single GCM/realization/
    SSP's own 10-year trajectory: 9.98-15.46, already a ~55% range on its own) confirmed the pooled
    number is dominated by year-to-year weather variability WITHIN a decade, not by which
    realization/GCM was drawn:
      (a) year+realization+GCM pooled (S-04-comparable headline number, but CAVEAT above applies)
      (b) realization+GCM ONLY, at fixed year -- isolates the actual "which of the 50 weather
          sequences did I draw" question S-04's framing intends, computed per year then averaged."""
    base = df[(df.mult_cattle == 1) & (df.mult_sheep == 1) & (df.mult_lamb == 1)]

    g = base.groupby(["tower", "ssp"])["annual_mean"]
    pooled = g.agg(mean="mean", std="std", p10=lambda s: s.quantile(0.10),
                    p90=lambda s: s.quantile(0.90), n="count").reset_index()
    pooled["band_pct_of_mean"] = (pooled["p90"] - pooled["p10"]) / pooled["mean"] * 100
    pooled.to_csv(f"{RESULTS}/s05_realization_spread_pooled.csv", index=False)

    per_year = base.groupby(["tower", "ssp", "year"])["annual_mean"].agg(
        mean="mean", p10=lambda s: s.quantile(0.10), p90=lambda s: s.quantile(0.90)).reset_index()
    per_year["band_pct_of_mean"] = (per_year["p90"] - per_year["p10"]) / per_year["mean"] * 100
    isolated = per_year.groupby(["tower", "ssp"])["band_pct_of_mean"].mean().reset_index()
    isolated.columns = ["tower", "ssp", "realization_only_band_pct_of_mean"]
    isolated.to_csv(f"{RESULTS}/s05_realization_spread_isolated.csv", index=False)

    print(f"[OK] s05_realization_spread_pooled.csv (year+realization+GCM conflated, "
          f"S-04-comparable but see caveat)\n{pooled.to_string(index=False)}")
    print(f"\n[OK] s05_realization_spread_isolated.csv (realization+GCM ONLY, fixed year -- the "
          f"actual apples-to-apples question)\n{isolated.to_string(index=False)}")
    return pooled, isolated


def ssp_divergence(traj):
    """Baseline livestock, early (offset 1-5) vs late (offset 6-10) window."""
    sub = traj[(traj.mult_cattle == 1) & (traj.mult_sheep == 1) & (traj.mult_lamb == 1)].copy()
    sub["window"] = np.where(sub["year_offset"] <= 5, "early_yr1_5", "late_yr6_10")
    win = sub.groupby(["tower", "ssp", "window"])["mean"].mean().reset_index()
    piv = win.pivot_table(index=["tower", "window"], columns="ssp", values="mean").reset_index()
    piv["ssp585_pct_of_ssp245"] = (piv["ssp585"] / piv["ssp245"] - 1.0) * 100
    piv.to_csv(f"{RESULTS}/s05_ssp_divergence.csv", index=False)
    print(f"[OK] s05_ssp_divergence.csv\n{piv.to_string(index=False)}")
    return piv


def aoa_trend(df):
    g = df.groupby(["tower", "ssp", "mult_cattle", "mult_sheep", "mult_lamb", "year_offset"])["aoa_flagged_pct"].mean().reset_index()
    g.to_csv(f"{RESULTS}/s05_aoa_trend.csv", index=False)
    base = g[(g.mult_cattle == 1) & (g.mult_sheep == 1) & (g.mult_lamb == 1)]
    print(f"[OK] s05_aoa_trend.csv\nBaseline (1x/1x/1x) AOA-flagged %% by tower/offset:")
    print(base.groupby(["tower", "year_offset"])["aoa_flagged_pct"].mean().unstack("year_offset").round(1))
    return g


def species_marginal_response(df):
    """Holding the other two species at 1x, how much does scaling cattle/sheep/lamb ALONE move
    predicted FCH4? The core new question Option B was built to answer."""
    base = df[(df.mult_cattle == 1) & (df.mult_sheep == 1) & (df.mult_lamb == 1)].groupby("tower")["annual_mean"].mean()

    rows = []
    for sp in SPECIES:
        other = [s for s in SPECIES if s != sp]
        for mult in [1.0, 2.0, 3.0]:
            mask = (df[MULT_COLS[sp]] == mult)
            for o in other:
                mask &= (df[MULT_COLS[o]] == 1.0)
            g = df[mask].groupby("tower")["annual_mean"].mean()
            for t in TOWERS:
                rows.append({"tower": t, "species": sp, "multiplier": mult,
                             "annual_mean": g.get(t, np.nan),
                             "pct_change_vs_baseline": (g.get(t, np.nan) / base[t] - 1) * 100})
    out = pd.DataFrame(rows)
    out.to_csv(f"{RESULTS}/s05_species_marginal_response.csv", index=False)
    print(f"[OK] s05_species_marginal_response.csv")
    print(out.pivot_table(index=["tower", "species"], columns="multiplier",
                           values="pct_change_vs_baseline").round(1))
    return out


def joint_vs_additive(df):
    """Is the response to scaling ALL THREE species together close to the SUM of each species'
    individual marginal effect (additive), or does the model show synergistic/sub-additive
    behavior? Only checkable because the full 27-combo grid exists."""
    base = df[(df.mult_cattle == 1) & (df.mult_sheep == 1) & (df.mult_lamb == 1)].groupby("tower")["annual_mean"].mean()
    joint3x = df[(df.mult_cattle == 3) & (df.mult_sheep == 3) & (df.mult_lamb == 3)].groupby("tower")["annual_mean"].mean()

    rows = []
    for t in TOWERS:
        individual_deltas = {}
        for sp in SPECIES:
            other = [s for s in SPECIES if s != sp]
            mask = (df["tower"] == t) & (df[MULT_COLS[sp]] == 3.0)
            for o in other:
                mask &= (df[MULT_COLS[o]] == 1.0)
            individual_deltas[sp] = df[mask]["annual_mean"].mean() - base[t]
        additive_prediction = base[t] + sum(individual_deltas.values())
        actual_joint = joint3x[t]
        rows.append({"tower": t, "baseline": base[t], "actual_joint_3x3x3": actual_joint,
                     "additive_prediction": additive_prediction,
                     "cattle_delta": individual_deltas["cattle"], "sheep_delta": individual_deltas["sheep"],
                     "lamb_delta": individual_deltas["lamb"],
                     "synergy_pct": (actual_joint / additive_prediction - 1) * 100})
    out = pd.DataFrame(rows)
    out.to_csv(f"{RESULTS}/s05_joint_vs_additive.csv", index=False)
    print(f"[OK] s05_joint_vs_additive.csv\n{out.round(2).to_string(index=False)}")
    return out


def plot_trajectory_bands(traj):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
    for ax, tower in zip(axes, TOWERS):
        sub_all = traj[(traj.tower == tower) & (traj.mult_cattle == 1) &
                       (traj.mult_sheep == 1) & (traj.mult_lamb == 1)]
        for ssp, ls in zip(SSPS, ["-", "--"]):
            sub = sub_all[sub_all.ssp == ssp].sort_values("year_offset")
            ax.plot(sub["year_offset"], sub["mean"], ls, color=TOWER_COLORS[tower],
                    label=f"{ssp} mean", linewidth=1.8)
            ax.fill_between(sub["year_offset"], sub["p10"], sub["p90"],
                             color=TOWER_COLORS[tower], alpha=0.15 if ssp == "ssp245" else 0.08)
        ax.set_title(f"Tower {tower} (1x/1x/1x baseline, p10-p90 band)")
        ax.set_xlabel("Years post-anchor")
        ax.set_ylabel("Predicted annual mean FCH4 (nmol m-2 s-1)")
        ax.legend(fontsize=8)
    fig.suptitle("S-05: 10-year transient trajectory, TabICLv2+Variant A+species, realization-level spread")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/s05_trajectory_bands.png", dpi=120)
    plt.close(fig)
    print("[OK] figure: s05_trajectory_bands.png")


def plot_aoa_trend(aoa_g):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, tower in zip(axes, TOWERS):
        sub_all = aoa_g[(aoa_g.tower == tower) & (aoa_g.mult_cattle == 1) &
                        (aoa_g.mult_sheep == 1) & (aoa_g.mult_lamb == 1)]
        for ssp, ls in zip(SSPS, ["-", "--"]):
            sub = sub_all[sub_all.ssp == ssp].sort_values("year_offset")
            ax.plot(sub["year_offset"], sub["aoa_flagged_pct"], ls, color=TOWER_COLORS[tower], linewidth=1.5, label=ssp)
        ax.set_title(f"Tower {tower}")
        ax.set_xlabel("Years post-anchor")
        ax.set_ylabel("AOA-flagged %")
        ax.legend(fontsize=8)
    fig.suptitle("S-05: Area-of-Applicability flagged %, 1x/1x/1x baseline, 10 years post-anchor")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/s05_aoa_trend.png", dpi=120)
    plt.close(fig)
    print("[OK] figure: s05_aoa_trend.png")


def plot_species_response(resp):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
    for ax, tower in zip(axes, TOWERS):
        sub = resp[resp.tower == tower]
        for sp in SPECIES:
            s = sub[sub.species == sp].sort_values("multiplier")
            ax.plot(s["multiplier"], s["pct_change_vs_baseline"], "o-", color=SPECIES_COLORS[sp], label=sp)
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_title(f"Tower {tower}")
        ax.set_xlabel("Species multiplier (other 2 species held at 1x)")
        ax.set_ylabel("% change in predicted FCH4 vs. baseline")
        ax.set_xticks([1, 2, 3])
        ax.legend()
    fig.suptitle("S-05: per-species marginal response to livestock scaling")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/s05_species_response.png", dpi=120)
    plt.close(fig)
    print("[OK] figure: s05_species_response.png")


def main():
    df = load_data()
    traj = trajectory_summary(df)
    spread = realization_spread(df)
    div = ssp_divergence(traj)
    aoa_g = aoa_trend(df)
    resp = species_marginal_response(df)
    joint = joint_vs_additive(df)

    plot_trajectory_bands(traj)
    plot_aoa_trend(aoa_g)
    plot_species_response(resp)

    print("\n[DONE] S-05 analysis complete.")


if __name__ == "__main__":
    main()
