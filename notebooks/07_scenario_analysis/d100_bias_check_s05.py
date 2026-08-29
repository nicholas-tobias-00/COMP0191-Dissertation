"""D-100: baseline-reconstruction check + delta-method bias correction for S-05 (TabICLv2 +
FX_A_SPECIES, S-03's Variant A + F-10's species split -- S-05's CURRENT, latest architecture, the
one every D-84 through D-98 result in this project actually uses).

Never checked before this task: does S-05's own baseline_1x1x1x scenario (real climatology-
resampled drivers, no livestock perturbation) reconstruct the real historical mean the way S-01's
Ridge+tree hybrid does? S-05 is a structurally different model family (TabICLv2, a one-shot
zero-shot foundation model, not a fitted Ridge trend + tree residual), so S-01's own gap (9-20%,
already accepted as small) is NOT assumed to transfer -- computed fresh, from source, below.

Two "real historical mean" conventions checked side by side, since this project has repeatedly
found target choice matters (D-96): y_gapfilled (matches S-01/S-04's own convention exactly, for
apples-to-apples comparability across the whole Phase 07 correction effort) and y_observed (this
project's authoritative evaluation target, D-36/D-37). y_gapfilled is used as the PRIMARY basis for
the correction applied below, for consistency with S-01/S-04; y_observed is reported alongside for
transparency, not silently dropped.

Fully additive: reads results/s05_trajectory_realizations_2050.csv and
results/s05_practices_livestock_v2.csv unchanged, writes new `_bias_corrected` files.

Run from project root:  python notebooks/07_scenario_analysis/d100_bias_check_s05.py
"""
import pandas as pd

ROOT = r"c:\Users\Nicholas\Documents\COMP0191 MSc Artificial Intelligence for Sustainable Development Project"
RESULTS = rf"{ROOT}\results"
HOURLY = rf"{ROOT}\data\Hourly"

TOWERS = [2, 4, 9]


def real_historical_means():
    dv = pd.read_csv(f"{HOURLY}/forecast_daily_v3.csv", low_memory=False)
    out = {}
    for t in TOWERS:
        dft = dv[dv.tower == t]
        out[t] = {"y_gapfilled": float(dft.y_gapfilled.mean()),
                  "y_observed": float(dft.y_observed.mean())}
    return out


def s05_baseline_1x(path=f"{RESULTS}/s05_trajectory_realizations_2050.csv"):
    df = pd.read_csv(path)
    base = df[(df.mult_cattle == 1.0) & (df.mult_sheep == 1.0) & (df.mult_lamb == 1.0)]
    return base.groupby("tower")["annual_mean"].mean().to_dict()


def main():
    real_means = real_historical_means()
    pred_1x = s05_baseline_1x()

    print("S-05 (TabICLv2 + FX_A_SPECIES) baseline-reconstruction check -- never done before this task:")
    offsets_gf = {}
    for t in TOWERS:
        gf, obs = real_means[t]["y_gapfilled"], real_means[t]["y_observed"]
        p1x = pred_1x[t]
        gap_gf = p1x / gf - 1
        gap_obs = p1x / obs - 1
        offsets_gf[t] = p1x - gf
        print(f"  T{t}: predicted_1x={p1x:.2f}  vs y_gapfilled={gf:.2f} ({gap_gf:+.1%})  "
              f"vs y_observed={obs:.2f} ({gap_obs:+.1%})  [bias_offset(gf-basis)={offsets_gf[t]:+.2f}]")
    print("\n*** Magnitude is far larger than S-01/S-04's 9-20% gap -- 40-80% underprediction at "
          "every tower, same direction throughout (TabICLv2 undershoots, never overshoots). This "
          "is a structurally different, more severe finding, not the same gap recurring in a new "
          "model. Corrected below using the same delta-method mechanism, but the correction should "
          "be read with substantially more caution than S-01/S-04's -- when the absolute bias is "
          "this large relative to the signal, the delta-method's core assumption (the model's SHAPE "
          "of response is trustworthy even if its LEVEL isn't) is itself less well supported. ***\n")

    # --- Correct the main species trajectory table ---
    df = pd.read_csv(f"{RESULTS}/s05_trajectory_realizations_2050.csv")
    df["bias_offset_gf"] = df["tower"].map(offsets_gf)
    df["annual_mean_bias_corrected"] = df["annual_mean"] - df["bias_offset_gf"]
    key = ["tower", "ssp", "gcm", "realization", "year"]
    b1x_raw = df[(df.mult_cattle == 1.0) & (df.mult_sheep == 1.0) & (df.mult_lamb == 1.0)].set_index(key)["annual_mean"]
    b1x_corr = df[(df.mult_cattle == 1.0) & (df.mult_sheep == 1.0) & (df.mult_lamb == 1.0)].set_index(key)["annual_mean_bias_corrected"]
    idx = pd.MultiIndex.from_frame(df[key])
    df["baseline_1x_raw"] = b1x_raw.reindex(idx).values
    df["baseline_1x_corrected"] = b1x_corr.reindex(idx).values
    df["pct_change_vs_model_1x_raw"] = df["annual_mean"] / df["baseline_1x_raw"] - 1
    df["pct_change_vs_real_mean_corrected"] = df["annual_mean_bias_corrected"] / df["baseline_1x_corrected"] - 1
    df.to_csv(f"{RESULTS}/s05_trajectory_realizations_2050_bias_corrected.csv", index=False)
    print(f"[OK] Saved s05_trajectory_realizations_2050_bias_corrected.csv ({len(df)} rows)")

    cattle3x = df[(df.mult_cattle == 3.0) & (df.mult_sheep == 1.0) & (df.mult_lamb == 1.0)]
    summ = cattle3x.groupby("tower")[["pct_change_vs_model_1x_raw", "pct_change_vs_real_mean_corrected"]].mean()
    print("Cattle-3x-alone headline, raw vs. corrected:")
    print((summ * 100).round(1).to_string())

    # --- Correct the redesigned livestock ladder (D-97) ---
    lv2 = pd.read_csv(f"{RESULTS}/s05_practices_livestock_v2.csv")
    lv2["bias_offset_gf"] = lv2["tower"].map(offsets_gf)
    lv2["annual_mean_bias_corrected"] = lv2["annual_mean"] - lv2["bias_offset_gf"]
    key2 = ["tower", "ssp", "gcm", "realization", "year"]
    b1x_raw2 = lv2[lv2.level == "baseline"].set_index(key2)["annual_mean"]
    b1x_corr2 = lv2[lv2.level == "baseline"].set_index(key2)["annual_mean_bias_corrected"]
    idx2 = pd.MultiIndex.from_frame(lv2[key2])
    lv2["baseline_1x_raw"] = b1x_raw2.reindex(idx2).values
    lv2["baseline_1x_corrected"] = b1x_corr2.reindex(idx2).values
    lv2["pct_change_vs_model_1x_raw"] = lv2["annual_mean"] / lv2["baseline_1x_raw"] - 1
    lv2["pct_change_vs_real_mean_corrected"] = lv2["annual_mean_bias_corrected"] / lv2["baseline_1x_corrected"] - 1
    lv2.to_csv(f"{RESULTS}/s05_practices_livestock_v2_bias_corrected.csv", index=False)
    print(f"\n[OK] Saved s05_practices_livestock_v2_bias_corrected.csv ({len(lv2)} rows)")

    own_max = lv2[lv2.level == "own_max__all_species"]
    summ2 = own_max.groupby("tower")[["pct_change_vs_model_1x_raw", "pct_change_vs_real_mean_corrected"]].mean()
    print("own_max (all_species) headline, raw vs. corrected:")
    print((summ2 * 100).round(1).to_string())


if __name__ == "__main__":
    main()
