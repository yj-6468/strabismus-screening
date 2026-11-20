"""Learn and infer per-eye cover state from raw-video appearance features."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable

# The estimator is deliberately single-process.  Give joblib a deterministic
# logical-core fallback on managed macOS hosts where physical-core discovery
# is unavailable, avoiding an irrelevant warning during fitting.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix

from .config import EventConfig
from .types import Eye


FACE_FEATURES = (
    "face_center_x",
    "face_center_y",
    "face_width",
    "face_height",
    "face_area",
    "interocular_px",
    "yaw_proxy",
    "pitch_proxy",
    "roll_radians",
)

EYE_FEATURE_SUFFIXES = (
    "eye_width_px",
    "iris_diameter_px",
    "nasal_u",
    "vertical_v",
    "aperture",
    "skin_fraction",
    "skin_distance",
    "texture_ratio",
    "dark_score",
    "smooth_score",
    "skin_cover_score",
    "neutral_cover_score",
    "cover_proxy",
    "glint_reliability",
    "eye_gray_mean",
    "eye_gray_std",
    "eye_gray_q10",
    "eye_gray_q50",
    "eye_gray_q90",
    "eye_saturation_mean",
    "eye_laplacian_variance",
    "eye_edge_density",
    "cheek_gray_mean",
    "cheek_gray_std",
    "cheek_gray_q10",
    "cheek_gray_q50",
    "cheek_gray_q90",
    "cheek_saturation_mean",
    "cheek_laplacian_variance",
    "cheek_edge_density",
)

LIMBUS_FEATURE_SUFFIXES = (
    "limbus_conf",
    "limbus_status",
    "limbus_step_luma",
    "limbus_temporal_score",
    "limbus_nasal_score",
    "limbus_r_px",
)

LIMBUS_TEMPORAL_SUFFIXES = ("limbus_conf", "limbus_step_luma")

TEMPORAL_SUFFIXES = (
    "cover_proxy",
    "eye_gray_mean",
    "eye_gray_q50",
    "eye_laplacian_variance",
    "eye_edge_density",
    "skin_fraction",
    "texture_ratio",
    "aperture",
)


def _robust_scale(values: pd.Series) -> tuple[float, float]:
    array = values.to_numpy(dtype=float)
    median = float(np.nanmedian(array)) if np.any(np.isfinite(array)) else 0.0
    mad = (
        float(np.nanmedian(np.abs(array - median)))
        if np.any(np.isfinite(array))
        else 1.0
    )
    return median, max(1.4826 * mad, 1e-6)


def long_eye_features(
    observations: pd.DataFrame, *, include_limbus: bool = False
) -> pd.DataFrame:
    """Convert patient-track wide rows into a shared per-eye feature table.

    ``include_limbus`` adds the native limbus evidence columns (cache v4) so the
    cover-state model can time transitions from the eye itself; it is opt-in so
    older caches and the legacy bundles keep their exact feature contract.
    """

    if observations.empty:
        return pd.DataFrame()
    eye_suffixes = EYE_FEATURE_SUFFIXES + (LIMBUS_FEATURE_SUFFIXES if include_limbus else ())
    temporal_suffixes = TEMPORAL_SUFFIXES + (LIMBUS_TEMPORAL_SUFFIXES if include_limbus else ())
    data = observations
    if "is_patient_track" in data:
        selected = data[data["is_patient_track"].astype(bool)]
        if not selected.empty:
            data = selected
    rows: list[pd.DataFrame] = []
    for eye in (Eye.RIGHT, Eye.LEFT):
        other = Eye.LEFT if eye is Eye.RIGHT else Eye.RIGHT
        base = data[["video", "frame", "timestamp_ms", "track_id"]].copy()
        base["eye"] = eye.value
        base["is_right_eye"] = float(eye is Eye.RIGHT)
        for feature in FACE_FEATURES:
            if feature in data:
                base[feature] = data[feature].to_numpy()
        for suffix in eye_suffixes:
            own = f"{eye.value}_{suffix}"
            counterpart = f"{other.value}_{suffix}"
            if own in data:
                base[suffix] = data[own].to_numpy()
            if own in data and counterpart in data:
                base[f"intereye_{suffix}"] = (
                    data[own].to_numpy(dtype=float)
                    - data[counterpart].to_numpy(dtype=float)
                )
        rows.append(base)
    result = pd.concat(rows, ignore_index=True)
    result.sort_values(["video", "eye", "frame"], inplace=True)

    for (_, _), indices in result.groupby(["video", "eye"], sort=False).groups.items():
        group = result.loc[indices]
        for suffix in temporal_suffixes:
            if suffix not in group:
                continue
            median, scale = _robust_scale(group[suffix])
            result.loc[indices, f"video_z_{suffix}"] = (
                group[suffix].to_numpy(dtype=float) - median
            ) / scale
            local = group[suffix].rolling(11, center=True, min_periods=2).median()
            result.loc[indices, f"local_delta_{suffix}"] = (
                group[suffix].to_numpy(dtype=float) - local.to_numpy(dtype=float)
            )
    return result.reset_index(drop=True)


def attach_cover_targets(
    long_features: pd.DataFrame,
    frame_annotations: pd.DataFrame,
) -> pd.DataFrame:
    """Join trusted anatomical cover labels to sampled feature rows."""

    required = {"video", "frame", "right_eye_state", "left_eye_state"}
    missing = required - set(frame_annotations)
    if missing:
        raise ValueError(f"Frame annotations missing columns: {sorted(missing)}")
    parts: list[pd.DataFrame] = []
    for eye in (Eye.RIGHT, Eye.LEFT):
        state_column = f"{eye.value}_eye_state"
        truth = frame_annotations[["video", "frame", state_column]].copy()
        truth.rename(columns={state_column: "eye_state"}, inplace=True)
        truth["eye"] = eye.value
        parts.append(truth)
    truth_long = pd.concat(parts, ignore_index=True)
    merged = long_features.merge(
        truth_long,
        on=["video", "frame", "eye"],
        how="inner",
        validate="many_to_one",
    )
    merged = merged[merged["eye_state"].isin(["covered", "uncovered"])].copy()
    merged["covered"] = (merged["eye_state"] == "covered").astype(int)
    return merged


def _model_columns(table: pd.DataFrame) -> list[str]:
    excluded = {
        "video",
        "frame",
        "timestamp_ms",
        "track_id",
        "eye",
        "eye_state",
        "covered",
        "patient_id",
        "patient_id_group",
        "subject_uid",
        "video_uid",
    }
    return [
        column
        for column in table.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(table[column])
    ]


@dataclass
class CoverModelDiagnostics:
    confusion: list[list[int]]
    report: dict[str, object]
    n_rows: int
    n_videos: int
    covered_fraction: float


class CoverEventModel:
    """A shallow, nonlinear appearance model with temporal self-normalization."""

    def __init__(self, random_seed: int = 20260824, *, include_limbus: bool = False) -> None:
        self.random_seed = random_seed
        self.include_limbus = bool(include_limbus)
        self.feature_columns: list[str] = []
        self.estimator = HistGradientBoostingClassifier(
            learning_rate=0.055,
            max_iter=180,
            max_leaf_nodes=9,
            max_depth=3,
            min_samples_leaf=24,
            l2_regularization=0.35,
            random_state=random_seed,
        )

    def fit(self, training_rows: pd.DataFrame) -> CoverModelDiagnostics:
        if "covered" not in training_rows:
            raise ValueError("Training rows require a binary 'covered' column")
        self.feature_columns = _model_columns(training_rows)
        if not self.feature_columns:
            raise ValueError("No numeric cover-model features were found")
        target = training_rows["covered"].to_numpy(dtype=int)
        # Frames are correlated within a clip.  First give each video equal
        # total mass, then balance the two cover states.  Otherwise a long clip
        # (or one sampled densely around many transitions) can dominate this
        # frame-level nuisance model and leak the sampling policy into it.
        if "video" in training_rows:
            rows_per_video = training_rows.groupby("video")["video"].transform("size")
            unit_weights = 1.0 / rows_per_video.to_numpy(dtype=float)
            n_videos = int(training_rows["video"].nunique())
        else:
            unit_weights = np.ones(len(target), dtype=float)
            n_videos = 1
        class_mass = {
            label: float(np.sum(unit_weights[target == label])) for label in (0, 1)
        }
        if min(class_mass.values()) <= 0.0:
            raise ValueError("Cover-event training requires covered and uncovered rows")
        total_mass = sum(class_mass.values())
        weights = unit_weights * np.asarray(
            [total_mass / (2.0 * class_mass[int(label)]) for label in target],
            dtype=float,
        )
        weights *= len(weights) / np.sum(weights)
        self.estimator.fit(training_rows[self.feature_columns], target, sample_weight=weights)
        prediction = self.estimator.predict(training_rows[self.feature_columns])
        return CoverModelDiagnostics(
            confusion=confusion_matrix(target, prediction, labels=[0, 1]).tolist(),
            report=classification_report(
                target,
                prediction,
                labels=[0, 1],
                target_names=["uncovered", "covered"],
                output_dict=True,
                zero_division=0,
            ),
            n_rows=len(training_rows),
            n_videos=n_videos,
            covered_fraction=float(np.mean(target)),
        )

    def predict_probability(self, feature_rows: pd.DataFrame) -> np.ndarray:
        if not self.feature_columns:
            raise RuntimeError("CoverEventModel is not fitted")
        missing = set(self.feature_columns) - set(feature_rows)
        if missing:
            raise ValueError(f"Cover features missing columns: {sorted(missing)}")
        return self.estimator.predict_proba(feature_rows[self.feature_columns])[:, 1]

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, destination)

    @classmethod
    def load(cls, path: str | Path) -> "CoverEventModel":
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError(f"Expected {cls.__name__}, found {type(model).__name__}")
        return model


def _hysteresis(probabilities: np.ndarray, on: float, off: float) -> np.ndarray:
    """Threshold probabilities without converting missing evidence to uncovered."""

    states = np.full(len(probabilities), np.nan, dtype=float)
    active: bool | None = None
    for index, probability in enumerate(probabilities):
        if not np.isfinite(probability):
            # A long/no-face gap breaks continuity.  The next observed sample
            # must establish state independently, and no transition is emitted
            # across this unknown interval.
            active = None
            continue
        if active is None:
            if probability >= on:
                active = True
            elif probability <= off:
                active = False
            else:
                continue
        elif active and probability < off:
            active = False
        elif not active and probability >= on:
            active = True
        states[index] = float(active)
    return states


def _remove_short_runs(states: np.ndarray, minimum: int) -> np.ndarray:
    result = states.copy()
    if not len(result):
        return result
    start = 0
    while start < len(result):
        value = result[start]
        stop = start + 1
        while stop < len(result) and (
            (np.isnan(value) and np.isnan(result[stop])) or result[stop] == value
        ):
            stop += 1
        if value == 1.0 and stop - start < minimum:
            result[start:stop] = 0.0
        start = stop
    return result


def _transition_flags(states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    covered = np.zeros(len(states), dtype=int)
    uncovered = np.zeros(len(states), dtype=int)
    if len(states) > 1:
        previous = states[:-1]
        current = states[1:]
        known = np.isfinite(previous) & np.isfinite(current)
        covered_indices = np.flatnonzero(known & (previous == 0.0) & (current == 1.0)) + 1
        uncovered_indices = np.flatnonzero(known & (previous == 1.0) & (current == 0.0)) + 1
        covered[covered_indices] = 1
        uncovered[uncovered_indices] = 1
    return covered, uncovered


def _fill_only_supported_probability_gaps(
    values: pd.Series, maximum_gap_frames: int
) -> pd.Series:
    """Interpolate short internal gaps and hold only a bounded trailing gap."""

    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    if maximum_gap_frames < 0:
        raise ValueError("maximum_gap_frames must be non-negative")
    if numeric.notna().sum() == 0:
        return numeric
    positions = np.arange(len(numeric), dtype=float)
    observed = numeric.notna().to_numpy()
    observed_positions = np.where(observed, positions, np.nan)
    previous = pd.Series(observed_positions).ffill().to_numpy(dtype=float)
    following = pd.Series(observed_positions).bfill().to_numpy(dtype=float)
    internal = (
        np.isfinite(previous)
        & np.isfinite(following)
        & ((following - previous - 1.0) <= maximum_gap_frames)
    )
    trailing = (
        np.isfinite(previous)
        & ~np.isfinite(following)
        & ((positions - previous) <= maximum_gap_frames)
    )
    supported = observed | internal | trailing
    filled = numeric.interpolate(limit_area="inside").ffill()
    filled.loc[~supported] = np.nan
    return filled


def infer_cover_timeline(
    observations: pd.DataFrame,
    model: CoverEventModel,
    *,
    frame_count: int,
    config: EventConfig | None = None,
) -> pd.DataFrame:
    """Return dense OD/OS probabilities, states, and transition flags."""

    config = config or EventConfig()
    long = long_eye_features(
        observations, include_limbus=bool(getattr(model, "include_limbus", False))
    )
    if long.empty:
        timeline = pd.DataFrame({"frame": np.arange(frame_count, dtype=int)})
        for eye in (Eye.RIGHT, Eye.LEFT):
            timeline[f"{eye.value}_cover_probability"] = np.nan
            timeline[f"{eye.value}_covered"] = np.nan
            timeline[f"{eye.value}_gets_covered"] = 0
            timeline[f"{eye.value}_gets_uncovered"] = 0
        return timeline
    long["cover_probability"] = model.predict_probability(long)
    dense = pd.DataFrame({"frame": np.arange(frame_count, dtype=int)})
    for eye in (Eye.RIGHT, Eye.LEFT):
        eye_rows = long[long["eye"] == eye.value][["frame", "cover_probability"]]
        eye_rows = eye_rows.groupby("frame", as_index=False).mean(numeric_only=True)
        series = dense[["frame"]].merge(eye_rows, on="frame", how="left")[
            "cover_probability"
        ]
        # Fill only gaps actually bounded by nearby observations.  Chaining a
        # bidirectional interpolation and forward fill can accidentally bridge
        # a gap several times longer than the configured hold limit.
        series = _fill_only_supported_probability_gaps(
            series, config.max_landmark_hold_frames
        )
        smooth = series.rolling(
            config.median_window_frames,
            center=True,
            min_periods=1,
        ).median()
        smooth[series.isna()] = np.nan
        dense[f"{eye.value}_cover_probability"] = smooth.to_numpy(dtype=float)

    right_probability = dense["right_cover_probability"].to_numpy(dtype=float)
    left_probability = dense["left_cover_probability"].to_numpy(dtype=float)
    right_state = _hysteresis(
        right_probability, config.cover_on_threshold, config.cover_off_threshold
    )
    left_state = _hysteresis(
        left_probability, config.cover_on_threshold, config.cover_off_threshold
    )
    both = (right_state == 1.0) & (left_state == 1.0)
    right_state[both & (right_probability < left_probability)] = False
    left_state[both & (left_probability <= right_probability)] = False
    right_state = _remove_short_runs(right_state, config.minimum_cover_frames)
    left_state = _remove_short_runs(left_state, config.minimum_cover_frames)
    for eye, state in ((Eye.RIGHT, right_state), (Eye.LEFT, left_state)):
        gets_covered, gets_uncovered = _transition_flags(state)
        dense[f"{eye.value}_covered"] = state
        dense[f"{eye.value}_gets_covered"] = gets_covered
        dense[f"{eye.value}_gets_uncovered"] = gets_uncovered
    return dense


def compact_cover_events(timeline: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for eye in (Eye.RIGHT, Eye.LEFT):
        for event in ("gets_covered", "gets_uncovered"):
            column = f"{eye.value}_{event}"
            if column not in timeline:
                continue
            for frame in timeline.loc[timeline[column].astype(bool), "frame"]:
                records.append(
                    {
                        "frame": int(frame),
                        "event": f"{eye.value}_{event}",
                        "eye": eye.value,
                    }
                )
    return pd.DataFrame.from_records(records).sort_values("frame").reset_index(drop=True)
