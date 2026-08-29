"""S05-T2: does pooling Tower 2 with Tower 4/9 (real historical context) let TabICLv2 borrow
T4/T9's learned cattle-CH4 sensitivity when projecting T2's livestock scenarios, instead of T2
getting essentially no response (D-92-era finding: cattle 3x = only +1.8-2.3% at T2, vs.
+186-215% at T4/T9)? User's own framing: T2's real fx_lsu_dens never exceeds ~0.71 (T4/T9's own
1x baseline is ~5), so the zero-shot model solo-per-tower has no historical livestock->CH4
covariation to learn from at T2 -- pooling gives it T4/T9's real relationship to draw on.

Mechanics, verified before building (not assumed): TabICLForecaster's multi-item_id API requires
EVERY item_id present in context_df to also appear in future_df (confirmed directly -- an
asymmetric context-only-T4/T9 attempt raised a KeyError deep in its batch dispatcher). So T4/T9
get a minimal (30-day) placeholder future window at their own baseline (1x/1x/1x) scenario --
their own predictions are discarded, this padding exists purely to satisfy the API. The pooling
benefit comes from T4/T9's REAL rows being present in context_df (shared in-context learning
across items in one forward pass), not from anything in their future_df.

Scope: T2 only (the tower in question), full 2050 horizon, 3 combos (baseline/cattle3x_alone/
all_3x) x 2 SSPs x 1 representative GCM/realization (ACCESS-ESM1-5/1, matching every other S-05
subset test this session) = 6 calls total, TabICLv2 only.

Run from project root:  python notebooks/07_scenario_analysis/s05_t2_pooled_test.py
"""
import sys
import time
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

ROOT = r"c:\Users\Nicholas\Documents\COMP0191 MSc Artificial Intelligence for Sustainable Development Project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\src")
sys.path.insert(0, ROOT + r"\src\features")
sys.path.insert(0, ROOT + r"\notebooks\07_scenario_analysis")

from build_transient_scenario_drivers_species import (
    FX_A_SPECIES, build_climatology_base_species, overlay_transient_species, load_transient_years,
)
from s05_trajectory_2050 import load_towers, tower_anchor, END_YEAR

RESULTS = rf"{ROOT}\results"
TOWERS = [2, 4, 9]
DUM = ["is_t2", "is_t4", "is_t9"]
COLS = FX_A_SPECIES + DUM
GCM, REAL = "ACCESS-ESM1-5", 1
SSPS = ["ssp245", "ssp585"]
COMBOS = {
    "baseline_1x1x1x": (1.0, 1.0, 1.0),
    "cattle3x_alone": (3.0, 1.0, 1.0),
    "all_3x3x3x": (3.0, 3.0, 3.0),
}


def add_dummies(df, tower):
    df = df.copy()
    df["is_t2"] = 1 if tower == 2 else 0
    df["is_t4"] = 1 if tower == 4 else 0
    df["is_t9"] = 1 if tower == 9 else 0
    return df


def build_context(T):
    """Real pre-anchor history, all 3 towers, tagged by item_id -- this is the pooling itself."""
    frames = []
    for t in TOWERS:
        dft = T[t]
        anchor = tower_anchor(T, t)
        hist = dft.loc[:anchor]
        cdf = add_dummies(hist[FX_A_SPECIES], t)
        cdf["timestamp"] = cdf.index
        cdf["target"] = hist["y_observed"].values
        cdf["item_id"] = t
        frames.append(cdf.reset_index(drop=True))
    return pd.concat(frames, ignore_index=True)


def build_t2_future(T, ssp, cattle, sheep, lamb):
    """T2's real scenario future (full 2050 horizon) -- the actual test."""
    anchor = tower_anchor(T, 2)
    years = list(range(anchor.year + 1, END_YEAR + 1))
    clim_cache = {yr: build_climatology_base_species(2, T, yr) for yr in years}
    tyears = load_transient_years(GCM, ssp, REAL, years)
    year_frames = [overlay_transient_species(clim_cache[yr], tyears[yr], cattle, sheep, lamb) for yr in years]
    frame = pd.concat(year_frames)
    fdf = add_dummies(frame[FX_A_SPECIES], 2)
    fdf["timestamp"] = fdf.index
    fdf["item_id"] = 2
    return fdf.reset_index(drop=True)


def build_padding_future(T, tower, ssp, n_days=30):
    """Minimal placeholder future window for T4/T9 -- required by the API (every context item_id
    must appear in future_df), predictions discarded, baseline (1x/1x/1x) scenario, short horizon
    to keep compute minimal since these predictions are never used."""
    anchor = tower_anchor(T, tower)
    years = [anchor.year + 1]
    clim = build_climatology_base_species(tower, T, years[0])
    tyears = load_transient_years(GCM, ssp, REAL, years)
    frame = overlay_transient_species(clim, tyears[years[0]], 1.0, 1.0, 1.0).iloc[:n_days]
    fdf = add_dummies(frame[FX_A_SPECIES], tower)
    fdf["timestamp"] = fdf.index
    fdf["item_id"] = tower
    return fdf.reset_index(drop=True)


def run_pooled(T, context_df, model_name, forecast_fn):
    """Returns (annual_df, daily_df) -- daily chains kept this time (not just the annual
    aggregate) so a daily-resolution figure can be built in the same style as
    s05_livestock_daily_chains_plots.py's full-horizon figures."""
    rows, daily_rows = [], []
    t0 = time.time()
    for ssp in SSPS:
        for combo_name, (cattle, sheep, lamb) in COMBOS.items():
            t2_future = build_t2_future(T, ssp, cattle, sheep, lamb)
            pad4 = build_padding_future(T, 4, ssp)
            pad9 = build_padding_future(T, 9, ssp)
            future_df = pd.concat([t2_future, pad4, pad9], ignore_index=True)

            preds = forecast_fn(context_df, future_df).reset_index()
            t2_preds = preds[preds.item_id == 2].copy()
            t2_preds["timestamp"] = pd.to_datetime(t2_preds["timestamp"])
            t2_preds["year"] = t2_preds["timestamp"].dt.year

            for _, r in t2_preds.iterrows():
                daily_rows.append({"model": model_name, "ssp": ssp, "combo": combo_name,
                                    "timestamp": r["timestamp"], "pred_pooled": r[0.5]})
            for yr, g in t2_preds.groupby("year"):
                rows.append({"model": model_name, "ssp": ssp, "combo": combo_name, "mult_cattle": cattle,
                             "mult_sheep": sheep, "mult_lamb": lamb, "year": yr, "annual_mean_pooled": g[0.5].mean()})
            print(f"  [{model_name} pooled] {ssp} {combo_name}: done ({time.time()-t0:.0f}s elapsed)")
    return pd.DataFrame(rows), pd.DataFrame(daily_rows)


def run_solo_tabpfn(T, forecast_fn):
    """No existing TabPFN solo baseline exists for FX_A_SPECIES (S-05 only ever used TabICLv2) --
    computed fresh here, same combos/SSPs, for a fair TabPFN pooled-vs-solo comparison. Returns
    (annual_df, daily_df)."""
    rows, daily_rows = [], []
    t0 = time.time()
    for ssp in SSPS:
        for combo_name, (cattle, sheep, lamb) in COMBOS.items():
            anchor = tower_anchor(T, 2)
            hist = T[2].loc[:anchor]
            future = build_t2_future(T, ssp, cattle, sheep, lamb).set_index("timestamp")
            context_df = hist[FX_A_SPECIES].copy()
            context_df["timestamp"] = context_df.index
            context_df["target"] = hist["y_observed"].values
            context_df = context_df.reset_index(drop=True)
            future_df = future[FX_A_SPECIES].reset_index()

            preds = forecast_fn(context_df, future_df).reset_index()
            preds["timestamp"] = pd.to_datetime(preds["timestamp"])
            for _, r in preds.iterrows():
                daily_rows.append({"model": "TabPFN", "ssp": ssp, "combo": combo_name,
                                    "timestamp": r["timestamp"], "pred_solo": r[0.5]})
            preds["year"] = preds["timestamp"].dt.year
            for yr, g in preds.groupby("year"):
                rows.append({"model": "TabPFN", "ssp": ssp, "combo": combo_name, "mult_cattle": cattle,
                             "mult_sheep": sheep, "mult_lamb": lamb, "year": yr, "annual_mean_solo": g[0.5].mean()})
            print(f"  [TabPFN solo] {ssp} {combo_name}: done ({time.time()-t0:.0f}s elapsed)")
    return pd.DataFrame(rows), pd.DataFrame(daily_rows)


def main():
    import os
    from tabicl import TabICLForecaster
    import tabpfn_time_series as tts
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
    tabpfn_ok = bool(os.environ.get("TABPFN_TOKEN"))

    T = load_towers()
    context_df = build_context(T)
    print(f"[OK] Pooled context built: {len(context_df)} rows, towers {sorted(context_df.item_id.unique())}")

    def tabicl_fn(ctx, fut):
        return TabICLForecaster().predict_df(ctx, future_df=fut, quantiles=[0.5])

    icl_annual, icl_daily = run_pooled(T, context_df, "TabICLv2", tabicl_fn)

    all_annual = [icl_annual]
    all_daily = [icl_daily]
    tabpfn_solo_annual, tabpfn_solo_daily = None, None
    if tabpfn_ok:
        def tabpfn_fn(ctx, fut):
            return tts.TabPFNTSPipeline(tabpfn_mode=tts.TabPFNMode.LOCAL).predict_df(ctx, future_df=fut, quantiles=[0.5])
        pfn_annual, pfn_daily = run_pooled(T, context_df, "TabPFN", tabpfn_fn)
        all_annual.append(pfn_annual)
        all_daily.append(pfn_daily)
        tabpfn_solo_annual, tabpfn_solo_daily = run_solo_tabpfn(T, tabpfn_fn)
    else:
        print("WARNING: TABPFN_TOKEN not set -- TabPFN steps skipped.")

    out = pd.concat(all_annual, ignore_index=True)
    out.to_csv(f"{RESULTS}/s05_t2_pooled_trajectory.csv", index=False)
    daily_out = pd.concat(all_daily, ignore_index=True)
    daily_out.to_csv(f"{RESULTS}/s05_t2_pooled_daily_chains.csv", index=False)
    print(f"\n[OK] Saved s05_t2_pooled_trajectory.csv ({len(out)} rows), "
          f"s05_t2_pooled_daily_chains.csv ({len(daily_out)} rows)")
    if tabpfn_solo_daily is not None:
        tabpfn_solo_daily.to_csv(f"{RESULTS}/s05_t2_tabpfn_solo_daily_chains.csv", index=False)
    if tabpfn_solo_annual is not None:
        tabpfn_solo_annual.to_csv(f"{RESULTS}/s05_t2_tabpfn_solo_trajectory.csv", index=False)

    # ---- TabICLv2: pooled (fresh) vs. solo (already on record, s05_trajectory_realizations_2050.csv) ----
    icl_solo = pd.read_csv(f"{RESULTS}/s05_trajectory_realizations_2050.csv")
    icl_solo = icl_solo[(icl_solo.tower == 2) & (icl_solo.gcm == GCM) & (icl_solo.realization == REAL)]

    print("\n" + "=" * 70)
    print("T2: pooled vs. solo, cattle/all-species response (%, vs. that SSP's own baseline)")
    print("=" * 70)
    compare_rows = []

    for ssp in SSPS:
        icl_out = out[out.model == "TabICLv2"]
        base_pooled = icl_out[(icl_out.ssp == ssp) & (icl_out.combo == "baseline_1x1x1x")]["annual_mean_pooled"].mean()
        base_solo = icl_solo[(icl_solo.ssp == ssp) & (icl_solo.mult_cattle == 1) & (icl_solo.mult_sheep == 1) &
                              (icl_solo.mult_lamb == 1)]["annual_mean"].mean()
        for combo_name, (c, s, l) in COMBOS.items():
            if combo_name == "baseline_1x1x1x":
                continue
            p = icl_out[(icl_out.ssp == ssp) & (icl_out.combo == combo_name)]["annual_mean_pooled"].mean()
            so = icl_solo[(icl_solo.ssp == ssp) & (icl_solo.mult_cattle == c) & (icl_solo.mult_sheep == s) &
                           (icl_solo.mult_lamb == l)]["annual_mean"].mean()
            pct_pooled = (p / base_pooled - 1) * 100
            pct_solo = (so / base_solo - 1) * 100
            compare_rows.append({"model": "TabICLv2", "ssp": ssp, "combo": combo_name,
                                  "solo_pct": pct_solo, "pooled_pct": pct_pooled,
                                  "delta_pct_points": pct_pooled - pct_solo})
            print(f"  [TabICLv2] {ssp} {combo_name}: solo {pct_solo:+.1f}%  ->  pooled {pct_pooled:+.1f}%  "
                  f"(delta {pct_pooled-pct_solo:+.1f} pp)")

    # ---- TabPFN: pooled (fresh) vs. solo (fresh, no prior record exists) ----
    if tabpfn_solo_annual is not None:
        pfn_out = out[out.model == "TabPFN"]
        for ssp in SSPS:
            base_pooled = pfn_out[(pfn_out.ssp == ssp) & (pfn_out.combo == "baseline_1x1x1x")]["annual_mean_pooled"].mean()
            base_solo = tabpfn_solo_annual[(tabpfn_solo_annual.ssp == ssp) & (tabpfn_solo_annual.combo == "baseline_1x1x1x")]["annual_mean_solo"].mean()
            for combo_name in ("cattle3x_alone", "all_3x3x3x"):
                p = pfn_out[(pfn_out.ssp == ssp) & (pfn_out.combo == combo_name)]["annual_mean_pooled"].mean()
                so = tabpfn_solo_annual[(tabpfn_solo_annual.ssp == ssp) & (tabpfn_solo_annual.combo == combo_name)]["annual_mean_solo"].mean()
                pct_pooled = (p / base_pooled - 1) * 100
                pct_solo = (so / base_solo - 1) * 100
                compare_rows.append({"model": "TabPFN", "ssp": ssp, "combo": combo_name,
                                      "solo_pct": pct_solo, "pooled_pct": pct_pooled,
                                      "delta_pct_points": pct_pooled - pct_solo})
                print(f"  [TabPFN] {ssp} {combo_name}: solo {pct_solo:+.1f}%  ->  pooled {pct_pooled:+.1f}%  "
                      f"(delta {pct_pooled-pct_solo:+.1f} pp)")

    pd.DataFrame(compare_rows).to_csv(f"{RESULTS}/s05_t2_pooled_vs_solo_compare.csv", index=False)
    print(f"\n[OK] Saved s05_t2_pooled_vs_solo_compare.csv")
    return out, compare_rows


if __name__ == "__main__":
    main()
