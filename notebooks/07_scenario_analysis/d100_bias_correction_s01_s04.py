"""D-100: delta-method bias correction for S-01/S-04's level-residual hybrid scenario outputs.

S-01's own baseline-reconstruction sanity check (`s01_results.md`, Finding 1) already documents
that the 1.0x (no-perturbation) scenario does not reconstruct each tower's real historical mean
exactly -- T2 +20.2%, T4 -2.0%, T9 +9.2% -- and concluded this was small enough to trust the
perturbed-scenario comparison as-is. This script does not treat that as a newly-discovered bug (it
was already known and accepted); it applies a more rigorous standard climate-impact-modelling
correction on top of that same known gap, so the reported scenario changes are anchored to the real
observed mean rather than the model's own (imperfectly reconstructed) baseline, and reports both
raw and corrected numbers side by side rather than replacing one with the other.

Delta-method / bias-correction (climate impact modelling convention): trust the model for the SHAPE
of the change, anchor the LEVEL to the real observed baseline --
    corrected_scenario = real_historical_mean + (predicted_scenario - predicted_baseline_1x)
Per-tower bias offset = predicted_baseline_1x - real_historical_mean, derived directly from S-01's
own saved summary (not hardcoded/re-typed), then applied identically to S-04 (which reuses S-01's
exact frozen model artifacts unmodified -- same model, same bias mechanism, so the same offset
transfers directly, no independent re-derivation needed).

For S-04 (a genuine transient trajectory, not a single snapshot), % change is recomputed as a
PAIRED comparison within each (tower, ssp, gcm, realization, year) group -- corrected multiplier=Nx
vs. the SAME group's corrected multiplier=1x -- rather than against the single S-01 snapshot value,
since S-04's own realization/year structure gives a more statistically sound paired baseline than
reusing one fixed external reference point.

Fully additive: reads results/s01_scenario_summary.csv and results/s04_trajectory_realizations.csv
+ results/s04_trajectory_summary.csv unchanged, writes new files with a `_bias_corrected` suffix
that carry BOTH raw and corrected columns side by side (nothing overwritten, nothing hidden).

Run from project root:  python notebooks/07_scenario_analysis/d100_bias_correction_s01_s04.py
"""
import pandas as pd

ROOT = r"c:\Users\Nicholas\Documents\COMP0191 MSc Artificial Intelligence for Sustainable Development Project"
RESULTS = rf"{ROOT}\results"

TOWERS = [2, 4, 9]


def derive_bias_offsets():
    """Per-tower bias offset (predicted_1x - real_historical_mean), re-derived directly from
    S-01's own saved summary -- not hardcoded, so this stays correct if s01_scenario_summary.csv
    is ever regenerated."""
    s01 = pd.read_csv(f"{RESULTS}/s01_scenario_summary.csv")
    base = s01[s01.multiplier == 1.0].set_index("tower")
    offsets = {}
    for t in TOWERS:
        pred_1x = float(base.loc[t, "annual_mean"])
        real_mean = float(base.loc[t, "real_historical_mean"])
        offsets[t] = {"predicted_baseline_1x": pred_1x, "real_historical_mean": real_mean,
                      "bias_offset": pred_1x - real_mean}
    return offsets


def correct_s01(offsets):
    s01 = pd.read_csv(f"{RESULTS}/s01_scenario_summary.csv")
    s01["bias_offset"] = s01["tower"].map(lambda t: offsets[t]["bias_offset"])
    s01["annual_mean_bias_corrected"] = s01["annual_mean"] - s01["bias_offset"]
    # % change vs. the model's own raw 1x baseline (what BEST_RESULTS.md/s01_results.md currently
    # report) alongside % change vs. the real historical mean, using the bias-corrected series --
    # both computed explicitly, not inferred, so the difference is auditable row by row.
    rows = []
    for t in TOWERS:
        sub = s01[s01.tower == t].copy()
        pred_1x = offsets[t]["predicted_baseline_1x"]
        real_mean = offsets[t]["real_historical_mean"]
        sub["pct_change_vs_model_1x_raw"] = sub["annual_mean"] / pred_1x - 1
        sub["pct_change_vs_real_mean_corrected"] = sub["annual_mean_bias_corrected"] / real_mean - 1
        rows.append(sub)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(f"{RESULTS}/s01_scenario_summary_bias_corrected.csv", index=False)
    print(f"[OK] Saved s01_scenario_summary_bias_corrected.csv ({len(out)} rows)")
    print(out[out.multiplier.isin([1.0, 3.0])][
        ["tower", "multiplier", "annual_mean", "annual_mean_bias_corrected",
         "pct_change_vs_model_1x_raw", "pct_change_vs_real_mean_corrected"]
    ].round(4).to_string(index=False))
    return out


def correct_s04(offsets):
    s04 = pd.read_csv(f"{RESULTS}/s04_trajectory_realizations.csv")
    s04["bias_offset"] = s04["tower"].map(lambda t: offsets[t]["bias_offset"])
    s04["annual_mean_bias_corrected"] = s04["annual_mean"] - s04["bias_offset"]

    # Paired % change: within each (tower, ssp, gcm, realization, year, model) group, compare the
    # corrected value at each multiplier against the SAME group's corrected multiplier=1.0 value.
    key = ["tower", "ssp", "gcm", "realization", "year", "model"]
    base_raw = s04[s04.multiplier == 1.0].set_index(key)["annual_mean"]
    base_corr = s04[s04.multiplier == 1.0].set_index(key)["annual_mean_bias_corrected"]
    idx = pd.MultiIndex.from_frame(s04[key])
    s04["baseline_1x_raw"] = base_raw.reindex(idx).values
    s04["baseline_1x_corrected"] = base_corr.reindex(idx).values
    s04["pct_change_vs_model_1x_raw"] = s04["annual_mean"] / s04["baseline_1x_raw"] - 1
    s04["pct_change_vs_real_mean_corrected"] = (
        s04["annual_mean_bias_corrected"] / s04["baseline_1x_corrected"] - 1)

    s04.to_csv(f"{RESULTS}/s04_trajectory_realizations_bias_corrected.csv", index=False)
    print(f"\n[OK] Saved s04_trajectory_realizations_bias_corrected.csv ({len(s04)} rows)")

    # Pooled summary (mirrors s04_trajectory_summary.csv's own grouping), both raw and corrected.
    summ = s04.groupby(["tower", "ssp", "multiplier"]).agg(
        annual_mean_raw=("annual_mean", "mean"),
        annual_mean_corrected=("annual_mean_bias_corrected", "mean"),
        pct_change_vs_model_1x_raw=("pct_change_vs_model_1x_raw", "mean"),
        pct_change_vs_real_mean_corrected=("pct_change_vs_real_mean_corrected", "mean"),
        n=("annual_mean", "size"),
    ).reset_index()
    summ.to_csv(f"{RESULTS}/s04_trajectory_summary_bias_corrected.csv", index=False)
    print(f"[OK] Saved s04_trajectory_summary_bias_corrected.csv ({len(summ)} rows)")
    print(summ[summ.multiplier == 3.0].round(4).to_string(index=False))
    return s04, summ


def main():
    offsets = derive_bias_offsets()
    print("Per-tower bias offsets (predicted_1x - real_historical_mean), derived from s01_scenario_summary.csv:")
    for t in TOWERS:
        o = offsets[t]
        print(f"  T{t}: predicted_1x={o['predicted_baseline_1x']:.2f}  "
              f"real_historical_mean={o['real_historical_mean']:.2f}  "
              f"bias_offset={o['bias_offset']:+.2f}")
    print()
    correct_s01(offsets)
    correct_s04(offsets)


if __name__ == "__main__":
    main()
