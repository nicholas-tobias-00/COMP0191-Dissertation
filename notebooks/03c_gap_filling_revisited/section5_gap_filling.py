"""Leakage-safe Section 5 benchmark used by temp_modeling_focus.ipynb.

The notebook keeps the configuration, execution, interpretation, and plots visible.
This module contains the longer reusable mechanics so that the notebook remains readable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import hashlib
import json
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler


ENV11 = [
    "SWIN_1_1_1",
    "TA_0_0_1",
    "VPD_0_0_1",
    "PPFD_1_1_1",
    "RN_1_1_1",
    "WS_0_0_1",
    "USTAR_0_0_1",
    "SHF_1_1_1",
    "Precipitation (mm)",
    "Soil Temperature @ 15cm Depth (oC)",
    "Soil Moisture @ 10cm Depth (%)",
]
CLOCK4 = ["_hs", "_hc", "_ds", "_dc"]
TOWER_DUMMIES = ["tower_2", "tower_4", "tower_9"]
TARGET_MEMORY_HOURS = [1, 2, 3, 6, 24, 48, 168]
TARGET_MEMORY_COLUMNS = (
    [f"fold_target_lag{h}" for h in TARGET_MEMORY_HOURS]
    + [f"fold_target_lead{h}" for h in TARGET_MEMORY_HOURS]
    + [
        "fold_y_prev",
        "fold_h_prev",
        "fold_has_prev",
        "fold_y_next",
        "fold_h_next",
        "fold_has_next",
        "fold_y_linear",
        "fold_gap_width",
        "fold_relative_gap_position",
    ]
)
TICA_COLUMNS = ["fold_TIC1", "fold_TIC2", "fold_TIC3"]


def default_scenarios() -> dict[str, int | str]:
    return {"vs": 1, "s": 4, "m": 32, "l": 288, "m1": "mixed"}


@dataclass
class Section5Config:
    """All expensive choices are explicit and can be changed in one notebook cell."""

    scenarios: dict[str, int | str] = field(default_factory=default_scenarios)
    mask_fraction: float = 0.25
    mds_rf_repetitions: int = 2
    tabicl_repetitions: int = 1
    random_seed: int = 42

    rf_n_estimators: int = 500
    rf_min_samples_leaf: int = 5
    rf_max_features: float = 1.0
    rf_n_jobs: int = -1
    use_model_cache: bool = True

    run_tabicl: bool = False
    tabicl_row_cap: int = 10_000
    tabicl_feature_arms: tuple[str, ...] = ("compact24",)

    target_neighbour_horizon: int = 72
    target_memory_hours: tuple[int, ...] = tuple(TARGET_MEMORY_HOURS)
    training_augmentation_fraction: float = 0.25

    tica_lag_hours: int = 24
    n_tics: int = 3


def stable_seed(base_seed: int, *parts: object) -> int:
    payload = "|".join([str(base_seed), *map(str, parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def calendar_gap_fold(
    target: pd.Series,
    gap_hours: int | str,
    mask_fraction: float,
    seed: int,
) -> pd.DataFrame:
    """Place non-overlapping calendar gaps and retain metadata for observed held-out hours."""
    if not 0 < mask_fraction < 1:
        raise ValueError("mask_fraction must lie strictly between 0 and 1.")
    if not isinstance(target.index, pd.DatetimeIndex):
        raise TypeError("target must use a DatetimeIndex.")

    n = len(target)
    valid = target.notna().to_numpy()
    target_n = max(1, int(valid.sum() * mask_fraction))
    rng = np.random.default_rng(seed)
    occupied = np.zeros(n, dtype=bool)
    gap_id = np.full(n, -1, dtype=int)
    nominal_width = np.full(n, np.nan)
    calendar_position = np.full(n, np.nan)
    n_selected = 0
    next_gap_id = 0

    for start in rng.permutation(n):
        if n_selected >= target_n:
            break
        width = int(rng.choice([1, 4, 32, 288])) if gap_hours == "mixed" else int(gap_hours)
        stop = min(start + width, n)
        if occupied[start:stop].any() or not valid[start:stop].any():
            continue
        occupied[start:stop] = True
        gap_id[start:stop] = next_gap_id
        nominal_width[start:stop] = stop - start
        calendar_position[start:stop] = np.arange(stop - start)
        n_selected += int(valid[start:stop].sum())
        next_gap_id += 1

    selected = occupied & valid
    if not selected.any():
        raise RuntimeError("Synthetic masking selected no observed target values.")

    result = pd.DataFrame(
        {
            "Datetime": target.index[selected],
            "synthetic_gap_id": gap_id[selected],
            "gap_length_hours": nominal_width[selected].astype(int),
            "gap_position_hours": calendar_position[selected].astype(int),
        }
    )
    denominator = np.maximum(result["gap_length_hours"] - 1, 1)
    result["relative_gap_position"] = result["gap_position_hours"] / denominator
    result["fold_seed"] = int(seed)
    return result.set_index("Datetime").sort_index()


def _simple_imputer(strategy: str) -> SimpleImputer:
    """Keep all columns even if a small smoke-test fold makes one entirely empty."""
    try:
        return SimpleImputer(strategy=strategy, keep_empty_features=True)
    except TypeError:  # compatibility with older scikit-learn
        return SimpleImputer(strategy=strategy)


def _metric_values(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(predicted)
    y = actual[valid]
    p = predicted[valid]
    if len(y) == 0:
        return {"R2": np.nan, "RMSE": np.nan, "MAE": np.nan, "MBE": np.nan}
    return {
        "R2": r2_score(y, p) if len(y) >= 2 and np.var(y) > 0 else np.nan,
        "RMSE": float(np.sqrt(np.mean((p - y) ** 2))),
        "MAE": float(np.mean(np.abs(p - y))),
        "MBE": float(np.mean(p - y)),
    }


class Section5Benchmark:
    """Partially pooled, target-tower-specific blocked gap cross-validation."""

    def __init__(
        self,
        prepared: pd.DataFrame,
        domain: dict[int, tuple[str, str]],
        output_dir: str | Path,
        cache_dir: str | Path,
        real_gaps: pd.DataFrame | None = None,
        config: Section5Config | None = None,
    ) -> None:
        self.config = config or Section5Config()
        self.domain = {int(k): v for k, v in domain.items()}
        self.towers = sorted(self.domain)
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.real_gaps = real_gaps.copy() if real_gaps is not None else None
        self.frames = self._prepare_frames(prepared)
        self.feature_arms = self._define_feature_arms()
        self._validate_feature_arms()
        self.folds: dict[tuple[int, str, int], pd.DataFrame] = {}

    def _prepare_frames(self, prepared: pd.DataFrame) -> dict[int, pd.DataFrame]:
        required = {"Datetime", "tower", "target", *ENV11, *CLOCK4}
        missing = sorted(required - set(prepared.columns))
        if missing:
            raise KeyError(f"Prepared data lacks required columns: {missing}")
        if prepared.duplicated(["tower", "Datetime"]).any():
            raise ValueError("Prepared data contains duplicate tower-hour keys.")

        data = prepared.copy()
        data["Datetime"] = pd.to_datetime(data["Datetime"])
        data["tower"] = data["tower"].astype(int)
        livestock = ["cattle_dens", "sheep_dens", "lamb_dens"]
        missing_livestock = sorted(set(livestock) - set(data.columns))
        if missing_livestock:
            raise KeyError(f"Cannot derive lsu_dens; missing {missing_livestock}")
        data["lsu_dens"] = (
            data["cattle_dens"].fillna(0)
            + 0.1 * data["sheep_dens"].fillna(0)
            + 0.05 * data["lamb_dens"].fillna(0)
        )
        for tower in self.towers:
            data[f"tower_{tower}"] = data["tower"].eq(tower).astype("int8")

        frames: dict[int, pd.DataFrame] = {}
        for tower in self.towers:
            frame = (
                data.loc[data["tower"].eq(tower)]
                .set_index("Datetime")
                .sort_index()
                .copy()
            )
            start, final_day = self.domain[tower]
            expected = pd.date_range(start, pd.Timestamp(final_day), freq="h", name="Datetime")
            if not frame.index.equals(expected):
                absent = expected.difference(frame.index)
                extra = frame.index.difference(expected)
                raise ValueError(
                    f"Tower {tower} does not match its inclusive midnight domain: "
                    f"{len(absent)} absent, {len(extra)} extra timestamps."
                )
            frames[tower] = frame
        return frames

    def _define_feature_arms(self) -> dict[str, list[str]]:
        champion30 = (
            ENV11
            + ["fc"]
            + CLOCK4
            + ["lsu_dens", "graze"]
            + [f"swc_lagQ{h}" for h in [168, 336, 504, 672]]
            + [f"ts_lagQ{h}" for h in [168, 336, 504, 672]]
            + ["mgmt_cut", "mgmt_manure", "gpp", "reco"]
        )
        compact24 = (
            ENV11
            + ["fc"]
            + CLOCK4
            + ["lsu_dens", "swc_rollD168", "ts_rollD24", "ts_rollD168"]
            + ["mgmt_cut", "mgmt_manure", "gpp", "reco"]
        )
        compact22 = [name for name in compact24 if name not in {"PPFD_1_1_1", "RN_1_1_1"}]
        compact_tica24 = [
            name
            for name in compact24
            if name not in {"swc_rollD168", "ts_rollD24", "ts_rollD168"}
        ] + TICA_COLUMNS
        short_memory47 = compact24 + TARGET_MEMORY_COLUMNS
        return {
            "met_only11": ENV11.copy(),
            "champion30_reference": champion30,
            "compact24": compact24,
            "compact22_no_rad_dupes": compact22,
            "compact_tica24": compact_tica24,
            "short_memory47": short_memory47,
        }

    def _validate_feature_arms(self) -> None:
        expected_widths = {
            "met_only11": 11,
            "champion30_reference": 30,
            "compact24": 24,
            "compact22_no_rad_dupes": 22,
            "compact_tica24": 24,
            "short_memory47": 47,
        }
        prepared_columns = set(next(iter(self.frames.values())).columns)
        dynamic = set(TICA_COLUMNS + TARGET_MEMORY_COLUMNS)
        for arm, features in self.feature_arms.items():
            if len(features) != len(set(features)):
                raise AssertionError(f"{arm} contains duplicate features.")
            if len(features) != expected_widths[arm]:
                raise AssertionError(
                    f"{arm} has {len(features)} features, expected {expected_widths[arm]}."
                )
            missing = sorted(set(features) - prepared_columns - dynamic)
            if missing:
                raise KeyError(f"{arm} cannot be built; missing columns: {missing}")
            if "target" in features:
                raise AssertionError(f"{arm} directly contains the target.")

    def feature_audit(self) -> pd.DataFrame:
        rows = []
        combined = pd.concat(self.frames, names=["tower_key", "Datetime"])
        dynamic = set(TICA_COLUMNS + TARGET_MEMORY_COLUMNS)
        for arm, features in self.feature_arms.items():
            for feature in features:
                if feature in TICA_COLUMNS:
                    family, source, missing_pct = "TICA", "fold-fitted", np.nan
                elif feature in TARGET_MEMORY_COLUMNS:
                    family, source, missing_pct = "target_memory", "fold-derived", np.nan
                elif feature in TOWER_DUMMIES:
                    family, source = "tower", "prepared"
                    missing_pct = 100 * combined[feature].isna().mean()
                elif feature.startswith(("swc_", "ts_")):
                    family, source = "soil_memory", "prepared"
                    missing_pct = 100 * combined[feature].isna().mean()
                elif feature in {"lsu_dens", "graze", "mgmt_cut", "mgmt_manure"}:
                    family, source = "management", "prepared/derived"
                    missing_pct = 100 * combined[feature].isna().mean()
                elif feature in CLOCK4:
                    family, source = "clock", "prepared"
                    missing_pct = 100 * combined[feature].isna().mean()
                else:
                    family, source = "environment_or_flux_covariate", "prepared"
                    missing_pct = 100 * combined[feature].isna().mean()
                rows.append(
                    {
                        "feature_arm": arm,
                        "feature": feature,
                        "family": family,
                        "source": source,
                        "dynamic_inside_fold": feature in dynamic,
                        "target_derived": feature in TARGET_MEMORY_COLUMNS,
                        "missing_pct": missing_pct,
                    }
                )
        return pd.DataFrame(rows)

    def build_folds(
        self,
        towers: list[int] | tuple[int, ...] | None = None,
        scenarios: dict[str, int | str] | None = None,
        repetitions: int | None = None,
        mask_fraction: float | None = None,
    ) -> dict[tuple[int, str, int], pd.DataFrame]:
        towers = list(towers or self.towers)
        scenarios = scenarios or self.config.scenarios
        repetitions = repetitions or max(
            self.config.mds_rf_repetitions,
            self.config.tabicl_repetitions if self.config.run_tabicl else 0,
        )
        mask_fraction = mask_fraction or self.config.mask_fraction
        folds = {}
        for tower in towers:
            for scenario, width in scenarios.items():
                for repetition in range(repetitions):
                    seed = stable_seed(
                        self.config.random_seed, "evaluation", tower, scenario, repetition
                    )
                    fold = calendar_gap_fold(
                        self.frames[tower]["target"], width, mask_fraction, seed
                    )
                    fold["tower"] = tower
                    fold["scenario"] = scenario
                    fold["repetition"] = repetition
                    folds[(tower, scenario, repetition)] = fold
        self.folds.update(folds)
        return folds

    def fold_audit(self) -> pd.DataFrame:
        if not self.folds:
            self.build_folds()
        rows = []
        for (tower, scenario, repetition), fold in sorted(self.folds.items()):
            rows.append(
                {
                    "tower": tower,
                    "scenario": scenario,
                    "repetition": repetition,
                    "held_out_observations": len(fold),
                    "unique_synthetic_gaps": fold["synthetic_gap_id"].nunique(),
                    "mean_gap_length_hours": fold["gap_length_hours"].mean(),
                    "fold_seed": int(fold["fold_seed"].iloc[0]),
                }
            )
        return pd.DataFrame(rows)

    def _mds_predictions(self, tower: int, held_out: pd.DatetimeIndex) -> pd.Series:
        """Canonical two-driver MDS with every failed prediction retained as NaN."""
        frame = self.frames[tower]
        source = frame["target"].copy()
        source.loc[held_out] = np.nan
        available = source.notna()
        candidates = frame.loc[available]
        candidate_y = source.loc[available].to_numpy(dtype=float)
        candidate_time = candidates.index.to_numpy()
        candidate_hour = candidates.index.hour.to_numpy()
        candidate_doy = candidates.index.dayofyear.to_numpy()
        candidate_swin = candidates["SWIN_1_1_1"].to_numpy(dtype=float)
        candidate_ta = candidates["TA_0_0_1"].to_numpy(dtype=float)
        windows = [pd.Timedelta(days=d).to_timedelta64() for d in [7, 14, 28, 91]]

        predictions = pd.Series(np.nan, index=held_out, dtype=float)
        queries = frame.reindex(held_out)
        for timestamp, row in queries.iterrows():
            query_time = np.datetime64(timestamp)
            query_hour = timestamp.hour
            query_doy = timestamp.dayofyear
            query_swin = float(row["SWIN_1_1_1"])
            query_ta = float(row["TA_0_0_1"])
            is_day = np.isfinite(query_swin) and query_swin > 10.0
            for window in windows:
                mask = (
                    (candidate_time >= query_time - window)
                    & (candidate_time <= query_time + window)
                    & (np.abs(candidate_hour - query_hour) <= 1)
                )
                if np.isfinite(query_ta):
                    mask &= (
                        (np.abs(candidate_ta - query_ta) <= 2.5)
                        | ~np.isfinite(candidate_ta)
                    )
                if is_day:
                    mask &= (
                        (np.abs(candidate_swin - query_swin) <= 50.0)
                        | ~np.isfinite(candidate_swin)
                    )
                if mask.any():
                    predictions.loc[timestamp] = float(np.nanmean(candidate_y[mask]))
                    break
            if np.isfinite(predictions.loc[timestamp]):
                continue
            same_hour = np.abs(candidate_hour - query_hour) <= 1
            doy_distance = np.minimum(
                np.abs(candidate_doy - query_doy),
                365 - np.abs(candidate_doy - query_doy),
            )
            fallback = same_hour & (doy_distance <= 7)
            if fallback.any():
                predictions.loc[timestamp] = float(np.nanmean(candidate_y[fallback]))
        return predictions

    def _target_memory(
        self,
        source: pd.Series,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Derive strict lag/lead and nearest-observation features from a masked source."""
        index = source.index
        output = pd.DataFrame(index=index)
        for hours in self.config.target_memory_hours:
            output[f"fold_target_lag{hours}"] = source.shift(hours)
            output[f"fold_target_lead{hours}"] = source.shift(-hours)

        positions = pd.Series(np.arange(len(source), dtype=float), index=index)
        previous_position = (
            positions.where(source.notna())
            .shift(1)
            .ffill(limit=self.config.target_neighbour_horizon)
        )
        next_position = (
            positions.where(source.notna())
            .shift(-1)
            .bfill(limit=self.config.target_neighbour_horizon)
        )
        previous_value = (
            source.shift(1).ffill(limit=self.config.target_neighbour_horizon)
        )
        next_value = (
            source.shift(-1).bfill(limit=self.config.target_neighbour_horizon)
        )
        h_previous = positions - previous_position
        h_next = next_position - positions
        has_previous = previous_position.notna()
        has_next = next_position.notna()
        both = has_previous & has_next
        width = h_previous + h_next

        output["fold_y_prev"] = previous_value
        output["fold_h_prev"] = h_previous
        output["fold_has_prev"] = has_previous.astype(float)
        output["fold_y_next"] = next_value
        output["fold_h_next"] = h_next
        output["fold_has_next"] = has_next.astype(float)
        output["fold_y_linear"] = np.nan
        output.loc[both, "fold_y_linear"] = (
            previous_value.loc[both]
            + (next_value.loc[both] - previous_value.loc[both])
            * (h_previous.loc[both] / width.loc[both])
        )
        output["fold_gap_width"] = width.where(both)
        output["fold_relative_gap_position"] = (
            h_previous / width
        ).where(both)

        source_times = pd.Series(index, index=index)
        audit = pd.DataFrame(index=index)
        audit["previous_source"] = (
            source_times.where(source.notna())
            .shift(1)
            .ffill(limit=self.config.target_neighbour_horizon)
        )
        audit["next_source"] = (
            source_times.where(source.notna())
            .shift(-1)
            .bfill(limit=self.config.target_neighbour_horizon)
        )
        assert (output.loc[has_previous, "fold_h_prev"] >= 1).all()
        assert (output.loc[has_next, "fold_h_next"] >= 1).all()
        return output, audit

    def _assert_memory_safe(
        self,
        features: pd.DataFrame,
        source_audit: pd.DataFrame,
        forbidden: pd.DatetimeIndex,
        context: str,
    ) -> None:
        forbidden = pd.DatetimeIndex(forbidden)
        for hours in self.config.target_memory_hours:
            lag_source = features.index - pd.Timedelta(hours=hours)
            lead_source = features.index + pd.Timedelta(hours=hours)
            bad_lag = pd.Series(lag_source, index=features.index).isin(forbidden)
            bad_lag &= features[f"fold_target_lag{hours}"].notna()
            bad_lead = pd.Series(lead_source, index=features.index).isin(forbidden)
            bad_lead &= features[f"fold_target_lead{hours}"].notna()
            if bad_lag.any() or bad_lead.any():
                raise AssertionError(f"Target lag/lead leakage in {context}, h={hours}.")
        if source_audit["previous_source"].isin(forbidden).any():
            raise AssertionError(f"Previous-neighbour leakage in {context}.")
        if source_audit["next_source"].isin(forbidden).any():
            raise AssertionError(f"Next-neighbour leakage in {context}.")

    def _augmentation_mask(
        self,
        tower: int,
        scenario: str,
        repetition: int,
        unavailable: pd.DatetimeIndex,
    ) -> pd.DatetimeIndex:
        source = self.frames[tower]["target"].copy()
        source.loc[source.index.intersection(unavailable)] = np.nan
        width = self.config.scenarios[scenario]
        seed = stable_seed(
            self.config.random_seed, "training_augmentation", tower, scenario, repetition
        )
        fold = calendar_gap_fold(
            source,
            width,
            self.config.training_augmentation_fraction,
            seed,
        )
        return pd.DatetimeIndex(fold.index)

    def _memory_views(
        self,
        target_tower: int,
        scenario: str,
        repetition: int,
        held_out: pd.DatetimeIndex,
    ) -> tuple[dict[int, pd.DataFrame], dict[int, pd.DataFrame]]:
        evaluation_frames = {tower: frame.copy() for tower, frame in self.frames.items()}
        training_frames = {tower: frame.copy() for tower, frame in self.frames.items()}

        for tower in self.towers:
            fold_unavailable = held_out if tower == target_tower else pd.DatetimeIndex([])

            evaluation_source = self.frames[tower]["target"].copy()
            evaluation_source.loc[
                evaluation_source.index.intersection(fold_unavailable)
            ] = np.nan
            evaluation_memory, evaluation_audit = self._target_memory(evaluation_source)
            if tower == target_tower:
                self._assert_memory_safe(
                    evaluation_memory,
                    evaluation_audit,
                    held_out,
                    f"evaluation tower={tower}, scenario={scenario}, rep={repetition}",
                )
            evaluation_frames[tower][TARGET_MEMORY_COLUMNS] = evaluation_memory[
                TARGET_MEMORY_COLUMNS
            ]

            augmentation = self._augmentation_mask(
                tower, scenario, repetition, fold_unavailable
            )
            training_forbidden = fold_unavailable.union(augmentation)
            training_source = self.frames[tower]["target"].copy()
            training_source.loc[
                training_source.index.intersection(training_forbidden)
            ] = np.nan
            training_memory, training_audit = self._target_memory(training_source)
            self._assert_memory_safe(
                training_memory,
                training_audit,
                training_forbidden,
                f"training tower={tower}, scenario={scenario}, rep={repetition}",
            )
            training_frames[tower][TARGET_MEMORY_COLUMNS] = training_memory[
                TARGET_MEMORY_COLUMNS
            ]
        return evaluation_frames, training_frames

    def _tica_views(
        self,
        target_tower: int,
        held_out: pd.DatetimeIndex,
    ) -> tuple[dict[int, pd.DataFrame], dict[int, pd.DataFrame]]:
        """Fit within the fold using target-free features and separate tower trajectories."""
        try:
            from deeptime.decomposition import TICA
        except ImportError as exc:
            raise ImportError(
                "compact_tica24 requires deeptime; run Section 4's install/import cell."
            ) from exc

        scaled_all: dict[int, np.ndarray] = {}
        eligible_by_tower: dict[int, np.ndarray] = {}
        for tower, frame in self.frames.items():
            eligible = frame["target"].notna().to_numpy()
            if tower == target_tower:
                eligible &= ~frame.index.isin(held_out)
            imputer = _simple_imputer("median")
            scaler = StandardScaler()
            x_train = imputer.fit_transform(frame.loc[eligible, ENV11])
            scaler.fit(x_train)
            scaled_all[tower] = scaler.transform(imputer.transform(frame[ENV11]))
            eligible_by_tower[tower] = eligible

        lag = self.config.tica_lag_hours
        estimator = TICA(lagtime=lag, dim=self.config.n_tics, scaling=None)
        n_pairs = 0
        for tower, frame in self.frames.items():
            x = scaled_all[tower]
            eligible = eligible_by_tower[tower]
            exact_lag = (
                frame.index[lag:].to_numpy() - frame.index[:-lag].to_numpy()
                == np.timedelta64(lag, "h")
            )
            pair_mask = eligible[:-lag] & eligible[lag:] & exact_lag
            if pair_mask.any():
                estimator.partial_fit((x[:-lag][pair_mask], x[lag:][pair_mask]))
                n_pairs += int(pair_mask.sum())
        if n_pairs <= self.config.n_tics:
            raise RuntimeError(f"Only {n_pairs} eligible lagged TICA pairs were found.")

        model = estimator.fetch_model()
        evaluation_frames = {tower: frame.copy() for tower, frame in self.frames.items()}
        for tower in self.towers:
            projected = np.asarray(model.transform(scaled_all[tower]))
            if projected.shape[1] < self.config.n_tics:
                raise RuntimeError(
                    f"TICA produced {projected.shape[1]} components, "
                    f"expected {self.config.n_tics}."
                )
            evaluation_frames[tower][TICA_COLUMNS] = projected[:, : self.config.n_tics]
        training_frames = {
            tower: frame.copy() for tower, frame in evaluation_frames.items()
        }
        return evaluation_frames, training_frames

    def _fold_views(
        self,
        feature_arm: str,
        target_tower: int,
        scenario: str,
        repetition: int,
        held_out: pd.DatetimeIndex,
    ) -> tuple[dict[int, pd.DataFrame], dict[int, pd.DataFrame]]:
        if feature_arm == "short_memory47":
            return self._memory_views(target_tower, scenario, repetition, held_out)
        if feature_arm == "compact_tica24":
            return self._tica_views(target_tower, held_out)
        return self.frames, self.frames

    def _training_frame(
        self,
        training_frames: dict[int, pd.DataFrame],
        target_tower: int,
        held_out: pd.DatetimeIndex,
        features: list[str],
    ) -> pd.DataFrame:
        parts = []
        for tower, frame in training_frames.items():
            selected = frame["target"].notna()
            if tower == target_tower:
                selected &= ~frame.index.isin(held_out)
            part = frame.loc[selected, features + ["target"]].copy()
            part["training_tower"] = tower
            part["training_datetime"] = part.index
            parts.append(part)
        training = pd.concat(parts, ignore_index=True)
        heldout_keys = pd.MultiIndex.from_arrays(
            [
                np.full(len(held_out), target_tower),
                held_out,
            ]
        )
        training_keys = pd.MultiIndex.from_arrays(
            [training["training_tower"], training["training_datetime"]]
        )
        if training_keys.isin(heldout_keys).any():
            raise AssertionError("Held-out target-tower rows entered model training.")
        return training

    def _rf_cache_path(
        self,
        feature_arm: str,
        training: pd.DataFrame,
        features: list[str],
    ) -> Path:
        params = {
            "n_estimators": self.config.rf_n_estimators,
            "min_samples_leaf": self.config.rf_min_samples_leaf,
            "max_features": self.config.rf_max_features,
            "random_state": self.config.random_seed,
        }
        hashed = pd.util.hash_pandas_object(
            training[features + ["target"]], index=False
        ).to_numpy()
        digest = hashlib.sha256()
        digest.update(feature_arm.encode("utf-8"))
        digest.update(hashed.tobytes())
        digest.update(json.dumps(params, sort_keys=True).encode("utf-8"))
        return self.cache_dir / f"section5_rf_{digest.hexdigest()[:24]}.joblib"

    def _fit_rf(
        self,
        feature_arm: str,
        training: pd.DataFrame,
        features: list[str],
    ) -> tuple[RandomForestRegressor, SimpleImputer]:
        cache_path = self._rf_cache_path(feature_arm, training, features)
        if self.config.use_model_cache and cache_path.exists():
            return joblib.load(cache_path)
        imputer = _simple_imputer("mean")
        x_train = imputer.fit_transform(training[features])
        model = RandomForestRegressor(
            n_estimators=self.config.rf_n_estimators,
            min_samples_leaf=self.config.rf_min_samples_leaf,
            max_features=self.config.rf_max_features,
            n_jobs=self.config.rf_n_jobs,
            random_state=self.config.random_seed,
        )
        model.fit(x_train, training["target"].to_numpy(dtype=float))
        if self.config.use_model_cache:
            joblib.dump((model, imputer), cache_path)
        return model, imputer

    def _fit_tabicl(
        self,
        training: pd.DataFrame,
        features: list[str],
    ):
        try:
            from tabicl import TabICLRegressor
        except ImportError as exc:
            raise ImportError("TabICL is not installed in this environment.") from exc
        sample_n = min(self.config.tabicl_row_cap, len(training))
        sampled = training.sample(n=sample_n, random_state=self.config.random_seed)
        imputer = _simple_imputer("mean")
        x_train = imputer.fit_transform(sampled[features])
        model = TabICLRegressor(random_state=self.config.random_seed)
        model.fit(x_train, sampled["target"].to_numpy(dtype=float))
        return model, imputer

    def tabicl_version(self) -> str | None:
        try:
            return version("tabicl")
        except PackageNotFoundError:
            return None

    def _prediction_records(
        self,
        model_name: str,
        feature_arm: str,
        tower: int,
        scenario: str,
        repetition: int,
        fold: pd.DataFrame,
        predicted: pd.Series,
    ) -> pd.DataFrame:
        held_out = pd.DatetimeIndex(fold.index)
        actual = self.frames[tower].loc[held_out, "target"].astype(float)
        records = fold.copy()
        records["model"] = model_name
        records["feature_arm"] = feature_arm
        records["y_actual"] = actual.reindex(records.index).to_numpy()
        records["y_pred"] = predicted.reindex(records.index).to_numpy()
        records["residual"] = records["y_pred"] - records["y_actual"]
        records["prediction_available"] = np.isfinite(records["y_pred"])
        return records.reset_index()

    def run(
        self,
        *,
        run_mds: bool = True,
        rf_feature_arms: tuple[str, ...] | list[str] | None = None,
        run_tabicl: bool | None = None,
        tabicl_feature_arms: tuple[str, ...] | list[str] | None = None,
        towers: tuple[int, ...] | list[int] | None = None,
        scenarios: dict[str, int | str] | None = None,
        mds_rf_repetitions: int | None = None,
        tabicl_repetitions: int | None = None,
        mask_fraction: float | None = None,
    ) -> pd.DataFrame:
        """Run selected methods and return one row for every requested held-out prediction."""
        towers = list(towers or self.towers)
        scenarios = scenarios or self.config.scenarios
        rf_feature_arms = list(rf_feature_arms or self.feature_arms)
        run_tabicl = self.config.run_tabicl if run_tabicl is None else run_tabicl
        tabicl_feature_arms = list(
            tabicl_feature_arms or self.config.tabicl_feature_arms
        )
        mds_rf_repetitions = (
            mds_rf_repetitions or self.config.mds_rf_repetitions
        )
        tabicl_repetitions = tabicl_repetitions or self.config.tabicl_repetitions
        total_repetitions = max(
            mds_rf_repetitions,
            tabicl_repetitions if run_tabicl else 0,
        )

        unknown = sorted(
            (set(rf_feature_arms) | set(tabicl_feature_arms))
            - set(self.feature_arms)
        )
        if unknown:
            raise KeyError(f"Unknown feature arms: {unknown}")

        requested_folds = self.build_folds(
            towers=towers,
            scenarios=scenarios,
            repetitions=total_repetitions,
            mask_fraction=mask_fraction,
        )
        all_records = []
        for tower in towers:
            for scenario in scenarios:
                for repetition in range(total_repetitions):
                    fold = requested_folds[(tower, scenario, repetition)]
                    held_out = pd.DatetimeIndex(fold.index)
                    if run_mds and repetition < mds_rf_repetitions:
                        predicted = self._mds_predictions(tower, held_out)
                        all_records.append(
                            self._prediction_records(
                                "MDS",
                                "mds_swin_ta",
                                tower,
                                scenario,
                                repetition,
                                fold,
                                predicted,
                            )
                        )
                        print(
                            f"done {'MDS':6s} {'mds_swin_ta':24s} "
                            f"T{tower} {scenario} rep={repetition}",
                            flush=True,
                        )

                    arms_this_fold = []
                    if repetition < mds_rf_repetitions:
                        arms_this_fold.extend(("RF", arm) for arm in rf_feature_arms)
                    if run_tabicl and repetition < tabicl_repetitions:
                        arms_this_fold.extend(("TabICL", arm) for arm in tabicl_feature_arms)

                    view_cache = {}
                    for model_name, feature_arm in arms_this_fold:
                        if feature_arm not in view_cache:
                            view_cache[feature_arm] = self._fold_views(
                                feature_arm,
                                tower,
                                scenario,
                                repetition,
                                held_out,
                            )
                        evaluation_frames, training_frames = view_cache[feature_arm]
                        features = self.feature_arms[feature_arm] + TOWER_DUMMIES
                        training = self._training_frame(
                            training_frames, tower, held_out, features
                        )
                        if model_name == "RF":
                            model, imputer = self._fit_rf(
                                feature_arm, training, features
                            )
                        else:
                            model, imputer = self._fit_tabicl(training, features)
                        x_query = imputer.transform(
                            evaluation_frames[tower].loc[held_out, features]
                        )
                        y_pred = np.asarray(model.predict(x_query), dtype=float)
                        if not np.isfinite(y_pred).all():
                            raise AssertionError(
                                f"{model_name}/{feature_arm} produced non-finite predictions."
                            )
                        predicted = pd.Series(y_pred, index=held_out)
                        all_records.append(
                            self._prediction_records(
                                model_name,
                                feature_arm,
                                tower,
                                scenario,
                                repetition,
                                fold,
                                predicted,
                            )
                        )
                        print(
                            f"done {model_name:6s} {feature_arm:24s} "
                            f"T{tower} {scenario} rep={repetition}",
                            flush=True,
                        )
        if not all_records:
            return pd.DataFrame()
        result = pd.concat(all_records, ignore_index=True)
        columns = [
            "model",
            "feature_arm",
            "tower",
            "scenario",
            "repetition",
            "Datetime",
            "synthetic_gap_id",
            "gap_length_hours",
            "gap_position_hours",
            "relative_gap_position",
            "fold_seed",
            "y_actual",
            "y_pred",
            "residual",
            "prediction_available",
        ]
        return result[columns].sort_values(
            ["model", "feature_arm", "tower", "scenario", "repetition", "Datetime"]
        ).reset_index(drop=True)

    def metrics_by_repetition(self, raw_predictions: pd.DataFrame) -> pd.DataFrame:
        rows = []
        group_columns = ["model", "feature_arm", "tower", "scenario", "repetition"]
        for keys, group in raw_predictions.groupby(group_columns, sort=True):
            available = group["prediction_available"].astype(bool)
            values = _metric_values(
                group.loc[available, "y_actual"],
                group.loc[available, "y_pred"],
            )
            rows.append(
                {
                    **dict(zip(group_columns, keys)),
                    **values,
                    "count": int(available.sum()),
                    "requested_count": int(len(group)),
                    "coverage": float(available.mean()),
                }
            )
        return pd.DataFrame(rows)

    def _real_gap_weights(self) -> pd.DataFrame:
        if self.real_gaps is None or self.real_gaps.empty:
            return pd.DataFrame()
        gaps = self.real_gaps.copy()
        gaps["scenario"] = np.select(
            [
                gaps["n_hours"].eq(1),
                gaps["n_hours"].between(2, 4),
                gaps["n_hours"].between(5, 32),
            ],
            ["vs", "s", "m"],
            default="l",
        )
        weights = (
            gaps.groupby(["tower", "scenario"], as_index=False)["n_hours"]
            .sum()
            .rename(columns={"n_hours": "real_missing_hours"})
        )
        weights["weight"] = weights["real_missing_hours"] / weights.groupby(
            "tower"
        )["real_missing_hours"].transform("sum")
        return weights

    def summary(self, metrics: pd.DataFrame) -> pd.DataFrame:
        group_columns = ["model", "feature_arm", "tower", "scenario"]
        metric_columns = ["R2", "RMSE", "MAE", "MBE", "count", "coverage"]
        summary = metrics.groupby(group_columns, as_index=False)[metric_columns].agg(
            ["median", "mean", "std"]
        )
        summary.columns = [
            "_".join(str(part) for part in column if part)
            if isinstance(column, tuple)
            else column
            for column in summary.columns
        ]
        summary = summary.rename(
            columns={
                "model_": "model",
                "feature_arm_": "feature_arm",
                "tower_": "tower",
                "scenario_": "scenario",
            }
        )
        summary["aggregate_type"] = "tower_scenario"

        flat_rows = []
        for keys, group in summary.groupby(["model", "feature_arm", "tower"]):
            row = dict(zip(["model", "feature_arm", "tower"], keys))
            row["scenario"] = "all"
            row["aggregate_type"] = "flat_scenario_median"
            for metric in metric_columns:
                for statistic in ["median", "mean", "std"]:
                    values = group[f"{metric}_{statistic}"]
                    row[f"{metric}_{statistic}"] = (
                        values.median() if values.notna().any() else np.nan
                    )
            flat_rows.append(row)

        weighted_rows = []
        weights = self._real_gap_weights()
        if not weights.empty:
            weighted = summary.merge(weights, on=["tower", "scenario"], how="inner")
            for keys, group in weighted.groupby(["model", "feature_arm", "tower"]):
                row = dict(zip(["model", "feature_arm", "tower"], keys))
                row["scenario"] = "real_gap_mix"
                row["aggregate_type"] = "real_gap_hour_weighted"
                normaliser = group["weight"].sum()
                for statistic in ["median", "mean"]:
                    row[f"R2_{statistic}"] = (
                        group[f"R2_{statistic}"] * group["weight"]
                    ).sum() / normaliser
                    row[f"RMSE_{statistic}"] = np.sqrt(
                        (
                            group[f"RMSE_{statistic}"].pow(2) * group["weight"]
                        ).sum()
                        / normaliser
                    )
                    for metric in ["MAE", "MBE", "count", "coverage"]:
                        row[f"{metric}_{statistic}"] = (
                            group[f"{metric}_{statistic}"] * group["weight"]
                        ).sum() / normaliser
                for metric in metric_columns:
                    row[f"{metric}_std"] = np.nan
                weighted_rows.append(row)

        return pd.concat(
            [summary, pd.DataFrame(flat_rows), pd.DataFrame(weighted_rows)],
            ignore_index=True,
            sort=False,
        )

    def save_results(
        self,
        raw_predictions: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        metrics = self.metrics_by_repetition(raw_predictions)
        summary = self.summary(metrics)
        audit = self.feature_audit()
        raw_predictions.to_csv(
            self.output_dir / "section5_raw_predictions.csv", index=False
        )
        metrics.to_csv(
            self.output_dir / "section5_metrics_by_rep.csv", index=False
        )
        summary.to_csv(self.output_dir / "section5_summary.csv", index=False)
        audit.to_csv(
            self.output_dir / "section5_feature_audit.csv", index=False
        )
        return metrics, summary, audit

    def champion_reference_check(
        self,
        summary: pd.DataFrame,
        tolerance: float = 0.03,
    ) -> pd.DataFrame:
        historical = {2: 0.576, 4: 0.404, 9: 0.426}
        rows = summary.loc[
            summary["model"].eq("RF")
            & summary["feature_arm"].eq("champion30_reference")
            & summary["aggregate_type"].eq("flat_scenario_median")
        ].copy()
        rows["historical_R2"] = rows["tower"].map(historical)
        rows["delta_R2"] = rows["R2_median"] - rows["historical_R2"]
        rows["within_tolerance"] = rows["delta_R2"].abs().le(tolerance)
        if not rows.empty and not rows["within_tolerance"].all():
            warnings.warn(
                "The reconstructed champion differs from the historical R2 by more "
                f"than ±{tolerance:.2f} for at least one tower. Check fold seeds, "
                "current prepared features, and lag definitions before interpreting it."
            )
        return rows[
            [
                "tower",
                "R2_median",
                "historical_R2",
                "delta_R2",
                "within_tolerance",
            ]
        ]


def plot_section5_summary(summary: pd.DataFrame):
    """Compact comparison of R2 and MDS coverage; returns the Matplotlib figure."""
    scenario_rows = summary.loc[summary["aggregate_type"].eq("tower_scenario")].copy()
    if scenario_rows.empty:
        raise ValueError("No tower-scenario summary rows to plot.")
    labels = (
        scenario_rows["model"].astype(str)
        + " / "
        + scenario_rows["feature_arm"].astype(str)
    )
    scenario_rows["label"] = labels
    models = scenario_rows["label"].drop_duplicates().tolist()

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    x = np.arange(len(scenario_rows["scenario"].unique()))
    scenarios = scenario_rows["scenario"].unique().tolist()
    width = 0.8 / max(len(models), 1)
    for i, label in enumerate(models):
        group = (
            scenario_rows.loc[scenario_rows["label"].eq(label)]
            .groupby("scenario")["R2_median"]
            .median()
            .reindex(scenarios)
        )
        axes[0].bar(x + (i - (len(models) - 1) / 2) * width, group, width, label=label)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_xticks(x, scenarios)
    axes[0].set_ylabel("Median R2 across towers")
    axes[0].set_title("Blocked gap-CV predictive skill")
    axes[0].legend(fontsize=7)

    mds = scenario_rows.loc[scenario_rows["model"].eq("MDS")]
    for tower, group in mds.groupby("tower"):
        axes[1].plot(
            group["scenario"],
            group["coverage_median"],
            marker="o",
            label=f"Tower {tower}",
        )
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Prediction coverage")
    axes[1].set_title("MDS coverage")
    axes[1].legend()
    fig.tight_layout()
    return fig


def plot_section5_diagnostics(
    raw_predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    tower: int = 4,
    scenario: str = "l",
):
    """Observation, residual, residual-ACF, and gap-position diagnostics for one RF arm."""
    candidate_metrics = metrics.loc[
        metrics["model"].eq("RF")
        & metrics["tower"].eq(tower)
        & metrics["scenario"].eq(scenario)
    ]
    if candidate_metrics.empty:
        raise ValueError(f"No RF predictions for Tower {tower}, scenario {scenario}.")
    best_arm = (
        candidate_metrics.groupby("feature_arm")["R2"].median().idxmax()
    )
    subset = raw_predictions.loc[
        raw_predictions["model"].eq("RF")
        & raw_predictions["feature_arm"].eq(best_arm)
        & raw_predictions["tower"].eq(tower)
        & raw_predictions["scenario"].eq(scenario)
        & raw_predictions["repetition"].eq(
            candidate_metrics.loc[
                candidate_metrics["feature_arm"].eq(best_arm), "repetition"
            ].min()
        )
    ].dropna(subset=["y_pred"])

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes[0, 0].scatter(subset["y_actual"], subset["y_pred"], s=8, alpha=0.35)
    low = min(subset["y_actual"].min(), subset["y_pred"].min())
    high = max(subset["y_actual"].max(), subset["y_pred"].max())
    axes[0, 0].plot([low, high], [low, high], color="black", linestyle="--")
    axes[0, 0].set(xlabel="Observed FCH4", ylabel="Predicted FCH4", title="Observed vs predicted")

    axes[0, 1].hist(subset["residual"], bins=50, alpha=0.8)
    axes[0, 1].axvline(0, color="black", linestyle="--")
    axes[0, 1].set(title="Residual distribution", xlabel="Prediction - observation")

    residual = subset.set_index("Datetime")["residual"].sort_index()
    acf_rows = []
    for lag in range(169):
        shifted = residual.copy()
        shifted.index = shifted.index + pd.Timedelta(hours=lag)
        pairs = pd.concat([residual.rename("left"), shifted.rename("right")], axis=1).dropna()
        acf_rows.append(pairs["left"].corr(pairs["right"]) if len(pairs) >= 3 else np.nan)
    axes[1, 0].plot(range(169), acf_rows)
    axes[1, 0].axhline(0, color="black", linewidth=0.8)
    axes[1, 0].axvline(24, color="0.6", linestyle=":")
    axes[1, 0].set(title="Pairwise residual ACF", xlabel="Lag (hours)", ylabel="Correlation")

    subset = subset.assign(abs_error=subset["residual"].abs())
    position_error = subset.groupby("relative_gap_position")["abs_error"].mean()
    axes[1, 1].scatter(position_error.index, position_error.values, s=10, alpha=0.6)
    axes[1, 1].set(
        title="Error through synthetic gaps",
        xlabel="Relative position in gap",
        ylabel="Mean absolute error",
    )
    fig.suptitle(f"Tower {tower}, {scenario}, best RF arm: {best_arm}", y=1.01)
    fig.tight_layout()
    return fig
