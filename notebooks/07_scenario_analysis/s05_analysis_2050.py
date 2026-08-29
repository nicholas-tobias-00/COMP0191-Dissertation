"""S-05, 2050-horizon analysis: summarizes `s05_trajectory_2050.py`'s output (same 8,100-call grid
as the original 10-year `s05_trajectory_10yr.py`/`s05_analysis.py`, but the horizon now runs from
each tower's anchor to 2050 -- T4/T9: 27 years, T2: 31 years -- matching S-04's own endpoint).
Read-only, runs no models. SUPERSEDES the 10-year analysis's findings (not just extends them) --
mirrors its structure exactly so the two are directly comparable, but every table/figure here is
recomputed against the new horizon and written under a distinct `_2050` suffix so the original
10-year outputs are never overwritten (both stay on record).

Early/late windowing is offset-based (relative to each tower's own anchor), not a fixed calendar
window like S-04's 2025-29/2046-50 -- T2 (anchor 2019) and T4/T9 (anchor 2023) reach 2050 via
different total year counts (31 vs 27), so "final 5 years of the trajectory" is defined per-tower
as (max_offset-4) to max_offset, not a hardcoded absolute range.

Same 6 questions as the 10-year analysis (see s05_analysis.py's own docstring for the full
rationale): trajectory summary, realization spread (pooled + isolated -- the pooled-vs-isolated
distinction matters even more here, since more years pooled together means more year-to-year
weather variability to conflate with realization/GCM choice if not separated), SSP divergence,
AOA trend, per-species marginal response, joint-vs-additive check.
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
TOWER_ANCHOR_YEAR = {2: 2019, 4: 2023, 9: 2023}
TOWER_COLORS = {2: "tab:blue", 4: "tab:orange", 9: "tab:green"}
SPECIES_COLORS = {"cattle": "tab:brown", "sheep": "tab:purple", "lamb": "tab:pink"}


def load_data():
    df = pd.read_csv(f"{RESULTS}/s05_trajectory_realizations_2050.csv")
    df["year_offset"] = df.apply(lambda r: r["year"] - TOWER_ANCHOR_YEAR[r["tower"]], axis=1)
    print(f"[OK] s05_trajectory_realizations_2050.csv: {len(df):,} rows, "
          f"year_offset range {df.year_offset.min()}-{df.year_offset.max()}")
    print(df.groupby("tower")["year_offset"].agg(["min", "max"]))
    return df


def trajectory_summary(df):
    g = df.groupby(["tower", "ssp", "mult_cattle", "mult_sheep", "mult_lamb", "year_offset"])["annual_mean"]
    out = g.agg(mean="mean", std="std", n="count",
                p10=lambda s: s.quantile(0.10), p90=lambda s: s.quantile(0.90)).reset_index()
    out.to_csv(f"{RESULTS}/s05_trajectory_summary_2050.csv", index=False)
    print(f"[OK] s05_trajectory_summary_2050.csv ({len(out)} rows)")
    return out


def realization_spread(df):
    base = df[(df.mult_cattle == 1) & (df.mult_sheep == 1) & (df.mult_lamb == 1)]

    g = base.groupby(["tower", "ssp"])["annual_mean"]
    pooled = g.agg(mean="mean", std="std", p10=lambda s: s.quantile(0.10),
                    p90=lambda s: s.quantile(0.90), n="count").reset_index()
    pooled["band_pct_of_mean"] = (pooled["p90"] - pooled["p10"]) / pooled["mean"] * 100
    pooled.to_csv(f"{RESULTS}/s05_realization_spread_pooled_2050.csv", index=False)

    per_year = base.groupby(["tower", "ssp", "year"])["annual_mean"].agg(
        mean="mean", p10=lambda s: s.quantile(0.10), p90=lambda s: s.quantile(0.90)).reset_index()
    per_year["band_pct_of_mean"] = (per_year["p90"] - per_year["p10"]) / per_year["mean"] * 100
    isolated = per_year.groupby(["tower", "ssp"])["band_pct_of_mean"].mean().reset_index()
    isolated.columns = ["tower", "ssp", "realization_only_band_pct_of_mean"]
    isolated.to_csv(f"{RESULTS}/s05_realization_spread_isolated_2050.csv", index=False)

    print(f"[OK] s05_realization_spread_pooled_2050.csv (year+realization+GCM conflated)\n"
          f"{pooled.to_string(index=False)}")
    print(f"\n[OK] s05_realization_spread_isolated_2050.csv (realization+GCM ONLY, fixed year)\n"
          f"{isolated.to_string(index=False)}")
    return pooled, isolated


def ssp_divergence(traj):
    """Early = offset 1-5 (post-anchor). Late = each tower's own final 5 years of the trajectory
    (offset (max-4) to max) -- T4/T9 reach 2050 at offset 27, T2 at offset 31, so "late" is a
    different absolute offset range per tower, always ending at 2050."""
    sub = traj[(traj.mult_cattle == 1) & (traj.mult_sheep == 1) & (traj.mult_lamb == 1)].copy()
    max_offset = sub.groupby("tower")["year_offset"].transform("max")
    sub["window"] = np.where(sub["year_offset"] <= 5, "early_yr1_5",
                       np.where(sub["year_offset"] > max_offset - 5, "late_final5yr", None))
    sub = sub.dropna(subset=["window"])
    win = sub.groupby(["tower", "ssp", "window"])["mean"].mean().reset_index()
    piv = win.pivot_table(index=["tower", "window"], columns="ssp", values="mean").reset_index()
    piv["ssp585_pct_of_ssp245"] = (piv["ssp585"] / piv["ssp245"] - 1.0) * 100
    piv.to_csv(f"{RESULTS}/s05_ssp_divergence_2050.csv", index=False)
    print(f"[OK] s05_ssp_divergence_2050.csv\n{piv.to_string(index=False)}")
    return piv


def aoa_trend(df):
    g = df.groupby(["tower", "ssp", "mult_cattle", "mult_sheep", "mult_lamb", "year_offset"])["aoa_flagged_pct"].mean().reset_index()
    g.to_csv(f"{RESULTS}/s05_aoa_trend_2050.csv", index=False)
    base = g[(g.mult_cattle == 1) & (g.mult_sheep == 1) & (g.mult_lamb == 1)]
    print(f"[OK] s05_aoa_trend_2050.csv\nBaseline (1x/1x/1x) AOA-flagged %% by tower/offset (first 5, last 5):")
    piv = base.groupby(["tower", "year_offset"])["aoa_flagged_pct"].mean().unstack("year_offset").round(1)
    for t in TOWERS:
        row = piv.loc[t].dropna()
        print(f"  T{t}: offset 1-5 = {row.iloc[:5].values}, offset (last 5) = {row.iloc[-5:].values}")
    return g


def species_marginal_response(df):
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
    out.to_csv(f"{RESULTS}/s05_species_marginal_response_2050.csv", index=False)
    print(f"[OK] s05_species_marginal_response_2050.csv")
    print(out.pivot_table(index=["tower", "species"], columns="multiplier",
                           values="pct_change_vs_baseline").round(1))
    return out


def joint_vs_additive(df):
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
    out.to_csv(f"{RESULTS}/s05_joint_vs_additive_2050.csv", index=False)
    print(f"[OK] s05_joint_vs_additive_2050.csv\n{out.round(2).to_string(index=False)}")
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
        ax.set_xlabel("Years post-anchor (to 2050)")
        ax.set_ylabel("Predicted annual mean FCH4 (nmol m-2 s-1)")
        ax.legend(fontsize=8)
    fig.suptitle("S-05 (2050 horizon): transient trajectory, TabICLv2+Variant A+species, realization-level spread")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/s05_trajectory_bands_2050.png", dpi=120)
    plt.close(fig)
    print("[OK] figure: s05_trajectory_bands_2050.png")


def plot_aoa_trend(aoa_g):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, tower in zip(axes, TOWERS):
        sub_all = aoa_g[(aoa_g.tower == tower) & (aoa_g.mult_cattle == 1) &
                        (aoa_g.mult_sheep == 1) & (aoa_g.mult_lamb == 1)]
        for ssp, ls in zip(SSPS, ["-", "--"]):
            sub = sub_all[sub_all.ssp == ssp].sort_values("year_offset")
            ax.plot(sub["year_offset"], sub["aoa_flagged_pct"], ls, color=TOWER_COLORS[tower], linewidth=1.2, label=ssp)
        ax.set_title(f"Tower {tower}")
        ax.set_xlabel("Years post-anchor (to 2050)")
        ax.set_ylabel("AOA-flagged %")
        ax.legend(fontsize=8)
    fig.suptitle("S-05 (2050 horizon): Area-of-Applicability flagged %, 1x/1x/1x baseline")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/s05_aoa_trend_2050.png", dpi=120)
    plt.close(fig)
    print("[OK] figure: s05_aoa_trend_2050.png")


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
    fig.suptitle("S-05 (2050 horizon): per-species marginal response to livestock scaling")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/s05_species_response_2050.png", dpi=120)
    plt.close(fig)
    print("[OK] figure: s05_species_response_2050.png")


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

    print("\n[DONE] S-05 (2050 horizon) analysis complete.")


if __name__ == "__main__":
    main()
