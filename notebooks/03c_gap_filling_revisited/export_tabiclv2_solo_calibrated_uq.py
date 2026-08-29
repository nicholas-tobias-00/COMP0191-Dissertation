"""Calibrate and plot uncertainty for the exact D5.5 TabICLv2-solo gap filler.

The earlier calibrated gap-filling artifact predates the D5.5 per-tower solo
champion.  By default this script reuses D9's persisted, exact D5.5 held-out
point predictions and native 90% interval widths.  It fits tower- and
gap-duration-specific scaled-residual split-conformal factors on repetition 0,
evaluates them on repetition 1, and applies them to the persisted latest solo
production chain.  This requires no model refit and does not borrow intervals
from the earlier pooled TabICL configuration.

An optional full native-endpoint CQR rerun is retained behind ``--full-cqr``.
Every such fit uses the unchanged D5.5 protocol: champion 30 FEATURES,
per-tower training, mean imputation, a fixed random sample capped at 10,000
observed rows, and TabICLRegressor(random_state=42).  Per-fold checkpoints make
that slower rerun safe to resume.
"""

from __future__ import annotations

import json
import math
import shutil
import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score

from export_latest_tabicl_uq_chains import FEATURES, RANDOM_STATE, ROW_CAP, TOWERS


ALPHA = 0.10
MASK_FRAC = 0.25
N_REPS = 2
SCENARIOS = {"vs": 1, "s": 4, "m": 32, "l": 288}
DOMAIN = {
    2: ("2017-10-01", "2019-06-30"),
    4: ("2017-10-01", "2023-12-31"),
    9: ("2020-02-01", "2023-12-31"),
}


def _insert_calendar_gaps(
    frame: pd.DataFrame,
    gap_hours: int,
    n_reps: int = N_REPS,
    seed: int = 0,
) -> list[pd.DatetimeIndex]:
    """Exact calendar-gap generator used by the D5.5 experiment."""
    valid = frame["target"].notna().to_numpy()
    timestamps = frame.index
    target_n = max(1, int(valid.sum() * MASK_FRAC))
    base_rng = np.random.default_rng(seed)
    repetitions = []
    for _ in range(n_reps):
        rng = np.random.default_rng(int(base_rng.integers(0, 2**31)))
        occupied = np.zeros(len(frame), dtype=bool)
        masked_valid = 0
        for start in rng.permutation(len(frame)):
            if masked_valid >= target_n:
                break
            end = min(int(start) + gap_hours, len(frame))
            if occupied[start:end].any():
                continue
            occupied[start:end] = True
            masked_valid += int(valid[start:end].sum())
        repetitions.append(timestamps[occupied & valid])
    return repetitions


def _fit_fold(training: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    from tabicl import TabICLRegressor

    sampled = training.sample(n=min(ROW_CAP, len(training)), random_state=RANDOM_STATE)
    imputer = SimpleImputer(strategy="mean")
    x_train = imputer.fit_transform(sampled[FEATURES].to_numpy())
    x_test = imputer.transform(test[FEATURES].to_numpy())
    model = TabICLRegressor(random_state=RANDOM_STATE)
    model.fit(x_train, sampled["target"].to_numpy(dtype=float))
    mean = np.asarray(model.predict(x_test), dtype=float).reshape(-1)
    quantiles = np.asarray(
        model.predict(x_test, output_type="quantiles", alphas=[0.05, 0.95]),
        dtype=float,
    )
    if quantiles.shape != (len(test), 2):
        raise RuntimeError(f"Unexpected native quantile shape: {quantiles.shape}")
    if not np.all(quantiles[:, 0] <= quantiles[:, 1]):
        raise RuntimeError("TabICLv2 returned crossed native quantiles")
    return pd.DataFrame(
        {
            "Datetime": test.index,
            "actual": test["target"].to_numpy(dtype=float),
            "pred_mean": mean,
            "q05": quantiles[:, 0],
            "q95": quantiles[:, 1],
        }
    )


def _finite_sample_quantile(values: np.ndarray, alpha: float = ALPHA) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError("Cannot calibrate an empty conformity-score array")
    level = min(1.0, math.ceil((len(values) + 1) * (1 - alpha)) / len(values))
    return float(np.quantile(values, level, method="higher"))


def _coverage(y: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean((y >= lower) & (y <= upper)))


def _run_or_load_folds(
    feature_frame: pd.DataFrame,
    checkpoint_dir: Path,
) -> pd.DataFrame:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    for tower in TOWERS:
        start, end = DOMAIN[tower]
        tower_frame = (
            feature_frame.loc[feature_frame["tower"].eq(tower)]
            .set_index("Datetime")
            .sort_index()
            .loc[pd.Timestamp(start) : pd.Timestamp(end)]
        )
        for scenario, gap_hours in SCENARIOS.items():
            gaps = _insert_calendar_gaps(tower_frame, gap_hours)
            for rep, held_out in enumerate(gaps):
                checkpoint = checkpoint_dir / f"T{tower}_{scenario}_rep{rep}.csv"
                if checkpoint.exists():
                    result = pd.read_csv(checkpoint, parse_dates=["Datetime"])
                    print(f"Checkpoint: T{tower} {scenario} rep {rep} ({len(result):,} rows)", flush=True)
                else:
                    training = tower_frame.loc[tower_frame["target"].notna()].drop(
                        index=held_out, errors="ignore"
                    )
                    test = tower_frame.loc[held_out]
                    print(
                        f"Fit: T{tower} {scenario} rep {rep}; "
                        f"train {min(ROW_CAP, len(training)):,}/{len(training):,}, "
                        f"test {len(test):,}",
                        flush=True,
                    )
                    result = _fit_fold(training, test)
                    result.to_csv(checkpoint, index=False)
                result["tower"] = tower
                result["scenario"] = scenario
                result["rep"] = rep
                parts.append(result)
    return pd.concat(parts, ignore_index=True)


def _calibrate(folds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries = []
    evaluated = []
    for (tower, scenario), group in folds.groupby(["tower", "scenario"], sort=True):
        calibration = group.loc[group["rep"].eq(0)].copy()
        test = group.loc[group["rep"].eq(1)].copy()
        scores = np.maximum(
            calibration["q05"].to_numpy() - calibration["actual"].to_numpy(),
            calibration["actual"].to_numpy() - calibration["q95"].to_numpy(),
        )
        margin = _finite_sample_quantile(scores)
        test["q05_cal"] = test["q05"] - margin
        test["q95_cal"] = test["q95"] + margin
        y = test["actual"].to_numpy()
        raw_lower = test["q05"].to_numpy()
        raw_upper = test["q95"].to_numpy()
        cal_lower = test["q05_cal"].to_numpy()
        cal_upper = test["q95_cal"].to_numpy()
        summaries.append(
            {
                "model": "TabICLv2-solo",
                "tower": tower,
                "scenario": scenario,
                "n_cal": len(calibration),
                "n_test": len(test),
                "margin": margin,
                "raw_coverage_test": _coverage(y, raw_lower, raw_upper),
                "calibrated_coverage_test": _coverage(y, cal_lower, cal_upper),
                "raw_width_test": float(np.mean(raw_upper - raw_lower)),
                "calibrated_width_test": float(np.mean(cal_upper - cal_lower)),
            }
        )
        evaluated.append(test)
    return pd.DataFrame(summaries), pd.concat(evaluated, ignore_index=True)


def _load_exact_solo_width_folds(data_dir: Path) -> pd.DataFrame:
    """Load D9 baseline rows: exact D5.5 point predictions plus native q90 width."""
    parts = []
    for tower in TOWERS:
        path = data_dir / "d9_checkpoints" / f"tower_{tower}.csv"
        frame = pd.read_csv(path, parse_dates=["Datetime"])
        frame = frame.loc[frame["arm"].eq("baseline")].copy()
        frame["tower"] = tower
        parts.append(frame)
    folds = pd.concat(parts, ignore_index=True)
    if (folds["width1"] <= 0).any():
        raise RuntimeError("D9 contains non-positive native interval widths")
    return folds


def _verify_d5_point_reproduction(folds: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    """Confirm D9's baseline rows reproduce the persisted D5.5 point results."""
    rows = []
    for (tower, scenario), group in folds.groupby(["tower", "scenario"]):
        rep_scores = [
            r2_score(rep_frame["actual"], rep_frame["pred_pass1"])
            for _, rep_frame in group.groupby("rep")
        ]
        rows.append(
            {
                "tower": tower,
                "scenario": scenario,
                "reproduced_R2": float(np.median(rep_scores)),
            }
        )
    verification = pd.DataFrame(rows)
    reference = pd.read_csv(data_dir / "d5_tabicl_solo_results.csv")
    reference = reference.loc[reference["feature_arm"].eq("baseline"), ["tower", "scenario", "R2"]]
    verification = verification.merge(reference, on=["tower", "scenario"], validate="one_to_one")
    verification["absolute_difference"] = (
        verification["reproduced_R2"] - verification["R2"]
    ).abs()
    if verification["absolute_difference"].max() > 1e-6:
        raise RuntimeError("D9 baseline does not reproduce D5.5 point predictions")
    return verification


def _calibrate_scaled_width(folds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split conformal using |y-mean|/(native q90 width/2) as conformity score."""
    summaries = []
    evaluated = []
    folds = folds.loc[folds["scenario"].isin(SCENARIOS)].copy()
    for (tower, scenario), group in folds.groupby(["tower", "scenario"], sort=True):
        calibration = group.loc[group["rep"].eq(0)].copy()
        test = group.loc[group["rep"].eq(1)].copy()
        half_width_cal = calibration["width1"].to_numpy(dtype=float) / 2
        scores = np.abs(
            calibration["actual"].to_numpy(dtype=float)
            - calibration["pred_pass1"].to_numpy(dtype=float)
        ) / half_width_cal
        factor = _finite_sample_quantile(scores)

        half_width_test = test["width1"].to_numpy(dtype=float) / 2
        center = test["pred_pass1"].to_numpy(dtype=float)
        y = test["actual"].to_numpy(dtype=float)
        test["lower_raw_symmetric"] = center - half_width_test
        test["upper_raw_symmetric"] = center + half_width_test
        test["lower_calibrated"] = center - factor * half_width_test
        test["upper_calibrated"] = center + factor * half_width_test
        summaries.append(
            {
                "model": "TabICLv2-solo",
                "tower": tower,
                "scenario": scenario,
                "n_cal": len(calibration),
                "n_test": len(test),
                "width_scale_factor": factor,
                "raw_symmetric_coverage_test": _coverage(
                    y,
                    test["lower_raw_symmetric"].to_numpy(),
                    test["upper_raw_symmetric"].to_numpy(),
                ),
                "calibrated_coverage_test": _coverage(
                    y,
                    test["lower_calibrated"].to_numpy(),
                    test["upper_calibrated"].to_numpy(),
                ),
                "raw_width_test": float(test["width1"].mean()),
                "calibrated_width_test": float((factor * test["width1"]).mean()),
            }
        )
        evaluated.append(test)
    return pd.DataFrame(summaries), pd.concat(evaluated, ignore_index=True)


def _gap_run_lengths(is_gap: pd.Series) -> pd.Series:
    groups = is_gap.ne(is_gap.shift(fill_value=False)).cumsum()
    lengths = is_gap.groupby(groups).transform("sum").astype(int)
    return lengths.where(is_gap, 0)


def _gap_bucket(hours: pd.Series) -> pd.Series:
    return pd.Series(
        np.select(
            [hours.le(2), hours.le(11), hours.le(96)],
            ["vs", "s", "m"],
            default="l",
        ),
        index=hours.index,
        dtype="object",
    )


def _apply_to_production_chain(chain: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    margins = summary.set_index(["tower", "scenario"])["margin"]
    outputs = []
    for tower, tower_chain in chain.groupby("tower", sort=True):
        tower_chain = tower_chain.sort_values("Datetime").copy()
        is_gap = tower_chain["is_gap"].astype(bool)
        tower_chain["gap_run_length_h"] = _gap_run_lengths(is_gap)
        tower_chain["gap_bucket"] = ""
        tower_chain.loc[is_gap, "gap_bucket"] = _gap_bucket(
            tower_chain.loc[is_gap, "gap_run_length_h"]
        )
        tower_chain["conformal_margin"] = np.nan
        for scenario in SCENARIOS:
            selected = is_gap & tower_chain["gap_bucket"].eq(scenario)
            tower_chain.loc[selected, "conformal_margin"] = margins.loc[(tower, scenario)]
        tower_chain["q05_cal"] = tower_chain["q05"] - tower_chain["conformal_margin"]
        tower_chain["q95_cal"] = tower_chain["q95"] + tower_chain["conformal_margin"]
        outputs.append(tower_chain)
    return pd.concat(outputs, ignore_index=True)


def _apply_scaled_width_to_production_chain(
    chain: pd.DataFrame,
    summary: pd.DataFrame,
) -> pd.DataFrame:
    factors = summary.set_index(["tower", "scenario"])["width_scale_factor"]
    outputs = []
    for tower, tower_chain in chain.groupby("tower", sort=True):
        tower_chain = tower_chain.sort_values("Datetime").copy()
        is_gap = tower_chain["is_gap"].astype(bool)
        tower_chain["gap_run_length_h"] = _gap_run_lengths(is_gap)
        tower_chain["gap_bucket"] = ""
        tower_chain.loc[is_gap, "gap_bucket"] = _gap_bucket(
            tower_chain.loc[is_gap, "gap_run_length_h"]
        )
        tower_chain["conformal_width_scale"] = np.nan
        for scenario in SCENARIOS:
            selected = is_gap & tower_chain["gap_bucket"].eq(scenario)
            tower_chain.loc[selected, "conformal_width_scale"] = factors.loc[(tower, scenario)]
        half_native_width = (tower_chain["q95"] - tower_chain["q05"]) / 2
        calibrated_half_width = tower_chain["conformal_width_scale"] * half_native_width
        tower_chain["q05_cal"] = tower_chain["y_predict_mean"] - calibrated_half_width
        tower_chain["q95_cal"] = tower_chain["y_predict_mean"] + calibrated_half_width
        outputs.append(tower_chain)
    return pd.concat(outputs, ignore_index=True)


def _plot_three_month(chain: pd.DataFrame, output_path: Path) -> None:
    end = chain["Datetime"].max()
    window = chain.loc[chain["Datetime"].ge(end - pd.DateOffset(months=3))].copy()
    is_gap = window["is_gap"].astype(bool).to_numpy()
    lower = np.where(is_gap, window["q05_cal"], np.nan)
    upper = np.where(is_gap, window["q95_cal"], np.nan)
    prediction = np.where(is_gap, window["y_predict_mean"], np.nan)

    fig, ax = plt.subplots(figsize=(15, 5.2))
    ax.fill_between(
        window["Datetime"],
        lower,
        upper,
        color="#357ABD",
        alpha=0.20,
        linewidth=0,
        label="TabICLv2 conformalized 90% interval",
    )
    ax.plot(
        window["Datetime"],
        prediction,
        color="#0066CC",
        linestyle=":",
        linewidth=1.0,
        label="TabICLv2 gap-filled mean",
    )
    ax.plot(
        window["Datetime"],
        window["y_observed"],
        color="black",
        linewidth=0.9,
        label="Observed FCH4",
    )
    tower = int(window["tower"].iloc[0])
    observed_pct = 100 * window["y_observed"].notna().mean()
    ax.set_title(
        f"Tower {tower}: latest three months of hourly FCH4 "
        f"({observed_pct:.1f}% observed)"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel(r"FCH4 (nmol m$^{-2}$ s$^{-1}$)")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.grid(axis="y", alpha=0.18)
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    fig.autofmt_xdate(rotation=0, ha="center")
    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=180,
        metadata={
            "Software": Path(__file__).name,
            "Description": (
                "Exact D5.5 TabICLv2-solo hourly gap-filled FCH4 with a 90% interval "
                "calibrated on held-out, tower- and gap-duration-specific synthetic gaps."
            ),
        },
    )
    plt.close(fig)


def export_tabiclv2_solo_calibrated_uq(
    data_dir: str | Path,
    fig_dir: str | Path,
    report_dir: str | Path,
    full_cqr: bool = False,
) -> list[Path]:
    data_dir = Path(data_dir)
    fig_dir = Path(fig_dir)
    report_dir = Path(report_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    feature_frame = pd.read_csv(data_dir / "feature_frame.csv", parse_dates=["Datetime"])
    required = set(FEATURES + ["Datetime", "target", "tower"])
    missing = sorted(required - set(feature_frame.columns))
    if missing:
        raise KeyError(f"feature_frame.csv is missing: {missing}")

    if full_cqr:
        folds = _run_or_load_folds(feature_frame, data_dir / "tabiclv2_solo_cqr_checkpoints")
        folds.to_csv(data_dir / "tabiclv2_solo_cqr_raw_predictions.csv", index=False)
        summary, evaluated = _calibrate(folds)
        summary_name = "tabiclv2_solo_cqr_summary.csv"
        evaluation_name = "tabiclv2_solo_cqr_evaluation_rows.csv"
        calibration_label = "Tower- and gap-duration-specific 90% split CQR"
        calibrated_chain = None
    else:
        folds = _load_exact_solo_width_folds(data_dir)
        verification = _verify_d5_point_reproduction(folds, data_dir)
        verification.to_csv(data_dir / "tabiclv2_solo_scaled_conformal_point_verification.csv", index=False)
        summary, evaluated = _calibrate_scaled_width(folds)
        summary_name = "tabiclv2_solo_scaled_conformal_summary.csv"
        evaluation_name = "tabiclv2_solo_scaled_conformal_evaluation_rows.csv"
        calibration_label = (
            "Tower- and gap-duration-specific 90% scaled-residual split conformal"
        )

    summary.to_csv(data_dir / summary_name, index=False)
    evaluated.to_csv(data_dir / evaluation_name, index=False)

    chain = pd.read_csv(data_dir / "latest_tabicl_uq_6month_chains.csv", parse_dates=["Datetime"])
    if set(chain["config"].dropna().unique()) != {"FEATURES"}:
        raise RuntimeError("The production chain is not the D5.5 FEATURES configuration")
    calibrated_chain = (
        _apply_to_production_chain(chain, summary)
        if full_cqr
        else _apply_scaled_width_to_production_chain(chain, summary)
    )
    calibrated_chain.to_csv(data_dir / "latest_tabiclv2_solo_uq_6month_chains_calibrated.csv", index=False)

    outputs = []
    for tower in TOWERS:
        tower_chain = calibrated_chain.loc[calibrated_chain["tower"].eq(tower)]
        output = fig_dir / f"latest_tabiclv2_solo_uq_hourly_T{tower}_3month_calibrated.png"
        _plot_three_month(tower_chain, output)
        outputs.append(output)
        shutil.copy2(
            output,
            report_dir / f"ch4_latest_tabiclv2_solo_uq_hourly_T{tower}_3month_calibrated.png",
        )

    manifest = {
        "model": "TabICLv2-solo / TabICLRegressor(random_state=42)",
        "configuration": "Exact D5.5 champion FEATURES; per tower; 10,000-row cap",
        "model_refit": bool(full_cqr),
        "validation": "D5.5 synthetic calendar gaps; rep 0 calibration, rep 1 test",
        "calibration": calibration_label,
        "point_provenance_check": (
            None
            if full_cqr
            else "tabiclv2_solo_scaled_conformal_point_verification.csv"
        ),
        "production_input": "latest_tabicl_uq_6month_chains.csv",
        "production_output": "latest_tabiclv2_solo_uq_6month_chains_calibrated.csv",
        "summary": summary_name,
        "evaluation_rows": evaluation_name,
        "figures": [path.name for path in outputs],
        "report_copies": [
            f"ch4_latest_tabiclv2_solo_uq_hourly_T{tower}_3month_calibrated.png"
            for tower in TOWERS
        ],
    }
    (fig_dir / "latest_tabiclv2_solo_calibrated_uq_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full-cqr",
        action="store_true",
        help="Rerun native q05/q95 for every D5.5 validation fold instead of using D9 widths.",
    )
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[2]
    paths = export_tabiclv2_solo_calibrated_uq(
        data_dir=Path(__file__).with_name("_data"),
        fig_dir=Path(__file__).with_name("_figures"),
        report_dir=project_root / "report" / "Figures",
        full_cqr=args.full_cqr,
    )
    print("\n".join(f"Saved {path}" for path in paths))
