"""Event-aligned ocular motion features for all three video-level tasks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .config import MotionConfig
from .types import Eye, TestType


@dataclass(frozen=True)
class MotionCycle:
    event_frame: int
    event_kind: str
    covered_eye: str
    observed_eye: str
    pre_or_early_position: float
    plateau_position: float
    delta_nasal: float
    eso_signed_motion: float
    absolute_motion: float
    head_pose_change: float
    aperture_fraction: float
    n_initial_samples: int
    n_plateau_samples: int
    quality: float
    usable: bool


def trusted_timeline(frame_annotations: pd.DataFrame) -> pd.DataFrame:
    """Normalize the workbook's trusted event flags to the inference schema."""

    columns = ["frame"]
    for eye in (Eye.RIGHT, Eye.LEFT):
        state = f"{eye.value}_eye_state"
        covered = f"{eye.value}_covered"
        if state in frame_annotations:
            # ``no_face`` is unknown, not evidence that the eye was uncovered.
            # Keeping NaN prevents training-only annotations from inventing a
            # binocular baseline that would be unavailable in the raw video.
            result = frame_annotations[state].map({"covered": 1.0, "uncovered": 0.0})
            frame_annotations = frame_annotations.copy()
            frame_annotations[covered] = result
        columns.extend(
            [
                covered,
                f"{eye.value}_gets_covered",
                f"{eye.value}_gets_uncovered",
            ]
        )
    available = [column for column in columns if column in frame_annotations]
    result = frame_annotations[available].copy()
    for column in columns:
        if column not in result:
            result[column] = 0
    return result[columns].sort_values("frame").reset_index(drop=True)


def _patient_rows(observations: pd.DataFrame) -> pd.DataFrame:
    if observations.empty:
        return observations.copy()
    if "is_patient_track" in observations:
        patient = observations[observations["is_patient_track"].astype(bool)]
        if not patient.empty:
            return patient.sort_values("frame").drop_duplicates("frame", keep="first")
    counts = observations.groupby("track_id").size()
    if counts.empty:
        return observations.iloc[0:0].copy()
    track_id = counts.idxmax()
    return (
        observations[observations["track_id"] == track_id]
        .sort_values("frame")
        .drop_duplicates("frame", keep="first")
    )


def _window_values(
    rows: pd.DataFrame,
    column: str,
    start: int,
    stop: int,
) -> np.ndarray:
    if column not in rows:
        return np.empty(0, dtype=float)
    mask = (rows["frame"] >= start) & (rows["frame"] <= stop)
    values = rows.loc[mask, column].to_numpy(dtype=float)
    return values[np.isfinite(values)]


def _median_or_nan(values: np.ndarray) -> float:
    return float(np.median(values)) if len(values) else float("nan")


def _pose_change(rows: pd.DataFrame, event_frame: int, before: int, after: int) -> float:
    mask = (rows["frame"] >= event_frame - before) & (
        rows["frame"] <= event_frame + after
    )
    window = rows.loc[mask]
    changes: list[float] = []
    for column in ("yaw_proxy", "pitch_proxy", "roll_radians"):
        if column not in window:
            continue
        values = window[column].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if len(values):
            changes.append(float(np.percentile(values, 90) - np.percentile(values, 10)))
    return max(changes, default=float("inf"))


def _aperture_baseline(rows: pd.DataFrame, eye: Eye) -> float:
    column = f"{eye.value}_aperture"
    if column not in rows:
        return float("nan")
    values = rows[column].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return float(np.percentile(values, 80)) if len(values) else float("nan")


def _motion_cycle(
    rows: pd.DataFrame,
    event_frame: int,
    event_kind: str,
    covered_eye: Eye,
    observed_eye: Eye,
    config: MotionConfig,
) -> MotionCycle:
    position_column = f"{observed_eye.value}_nasal_u"
    if event_kind == "gets_covered":
        initial = _window_values(
            rows,
            position_column,
            event_frame - config.pre_window_frames,
            event_frame - 2,
        )
    else:
        initial = _window_values(
            rows,
            position_column,
            event_frame,
            event_frame + config.early_post_frames,
        )
    plateau = _window_values(
        rows,
        position_column,
        event_frame + config.plateau_start_frames,
        event_frame + config.plateau_end_frames,
    )
    initial_position = _median_or_nan(initial)
    plateau_position = _median_or_nan(plateau)
    delta_nasal = float(plateau_position - initial_position)
    # Nasal refixation means exodeviation; temporal refixation means eso.
    eso_signed_motion = float(-delta_nasal)
    head_change = _pose_change(
        rows,
        event_frame,
        config.pre_window_frames,
        config.plateau_end_frames,
    )
    baseline = _aperture_baseline(rows, observed_eye)
    aperture_values = _window_values(
        rows,
        f"{observed_eye.value}_aperture",
        event_frame,
        event_frame + config.plateau_end_frames,
    )
    aperture_fraction = float(
        _median_or_nan(aperture_values) / max(baseline, 1e-6)
        if np.isfinite(baseline)
        else float("nan")
    )
    sample_quality = min(1.0, len(initial) / 3.0) * min(1.0, len(plateau) / 4.0)
    pose_quality = float(
        np.clip(1.0 - head_change / max(config.maximum_head_pose_change, 1e-6), 0.0, 1.0)
    )
    blink_quality = float(np.clip((aperture_fraction - 0.35) / 0.45, 0.0, 1.0))
    quality = float(sample_quality * (0.35 + 0.65 * pose_quality) * blink_quality)
    usable = bool(
        len(initial) >= 2
        and len(plateau) >= 2
        and np.isfinite(delta_nasal)
        and quality > 0.08
    )
    return MotionCycle(
        event_frame=int(event_frame),
        event_kind=event_kind,
        covered_eye=covered_eye.value,
        observed_eye=observed_eye.value,
        pre_or_early_position=initial_position,
        plateau_position=plateau_position,
        delta_nasal=delta_nasal,
        eso_signed_motion=eso_signed_motion,
        absolute_motion=float(abs(delta_nasal)),
        head_pose_change=head_change,
        aperture_fraction=aperture_fraction,
        n_initial_samples=len(initial),
        n_plateau_samples=len(plateau),
        quality=quality,
        usable=usable,
    )


def extract_motion_cycles(
    observations: pd.DataFrame,
    timeline: pd.DataFrame,
    *,
    test_type: str | TestType,
    config: MotionConfig | None = None,
) -> pd.DataFrame:
    """Measure fellow-eye cover-on and newly-uncovered-eye refixations."""

    config = config or MotionConfig()
    rows = _patient_rows(observations)
    if rows.empty or timeline.empty:
        return pd.DataFrame(columns=list(MotionCycle.__dataclass_fields__))
    test_type = TestType(str(test_type))
    cycles: list[MotionCycle] = []
    for covered_eye in (Eye.RIGHT, Eye.LEFT):
        fellow = Eye.LEFT if covered_eye is Eye.RIGHT else Eye.RIGHT
        cover_column = f"{covered_eye.value}_gets_covered"
        uncover_column = f"{covered_eye.value}_gets_uncovered"
        # In ACT a switch normally produces a cover-on flag for one eye and an
        # uncover flag for the other within a couple of frames.  Both point to
        # the same newly uncovered eye movement, so suppress the paired cover-
        # on duplicate.  Retain an unpaired cover-on (notably the first cover
        # when a clip begins binocularly) because fellow-eye motion there is
        # genuine manifest evidence.  CU always needs both phases.
        if cover_column in timeline:
            fellow_uncover_column = f"{fellow.value}_gets_uncovered"
            fellow_uncover_frames = (
                timeline.loc[
                    timeline[fellow_uncover_column].astype(bool), "frame"
                ].to_numpy(dtype=int)
                if fellow_uncover_column in timeline
                else np.empty(0, dtype=int)
            )
            for event_frame in timeline.loc[timeline[cover_column].astype(bool), "frame"]:
                paired_act_switch = bool(
                    test_type is TestType.ALTERNATE_COVER
                    and len(fellow_uncover_frames)
                    and np.min(np.abs(fellow_uncover_frames - int(event_frame))) <= 2
                )
                if paired_act_switch:
                    continue
                # Cover-on refixation of the fellow eye identifies manifest tropia.
                cycles.append(
                    _motion_cycle(
                        rows,
                        int(event_frame),
                        "gets_covered",
                        covered_eye,
                        fellow,
                        config,
                    )
                )
        if uncover_column in timeline:
            for event_frame in timeline.loc[timeline[uncover_column].astype(bool), "frame"]:
                # The just-uncovered eye reveals latent/total deviation.  In ACT this
                # is the principal PD signal; in CU it is auxiliary recovery evidence.
                cycles.append(
                    _motion_cycle(
                        rows,
                        int(event_frame),
                        "gets_uncovered",
                        covered_eye,
                        covered_eye,
                        config,
                    )
                )
    if not cycles:
        return pd.DataFrame(columns=list(MotionCycle.__dataclass_fields__))
    result = pd.DataFrame.from_records([asdict(cycle) for cycle in cycles])
    result["test_type"] = test_type.value
    return result.sort_values(["event_frame", "event_kind", "covered_eye"]).reset_index(
        drop=True
    )


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[mask]
    weights = weights[mask]
    if not len(values):
        return float("nan")
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights) / np.sum(weights)
    return float(np.interp(q, cumulative, values))


def _aggregate_cycles(cycles: pd.DataFrame, prefix: str) -> dict[str, float]:
    if cycles.empty:
        return {
            f"{prefix}_n": 0.0,
            f"{prefix}_usable_n": 0.0,
            f"{prefix}_motion_median": float("nan"),
            f"{prefix}_motion_abs_median": float("nan"),
            f"{prefix}_motion_iqr": float("nan"),
            f"{prefix}_motion_max_abs": float("nan"),
            f"{prefix}_eso_fraction": float("nan"),
            f"{prefix}_exo_fraction": float("nan"),
            f"{prefix}_sign_consistency": float("nan"),
            f"{prefix}_quality_median": float("nan"),
        }
    usable = cycles[cycles["usable"].astype(bool)]
    if usable.empty:
        usable = cycles.iloc[0:0]
    signed = usable["eso_signed_motion"].to_numpy(dtype=float)
    absolute = np.abs(signed)
    weights = usable["quality"].to_numpy(dtype=float)
    median = _weighted_quantile(signed, weights, 0.5)
    q25 = _weighted_quantile(signed, weights, 0.25)
    q75 = _weighted_quantile(signed, weights, 0.75)
    nontrivial = absolute > 0.005
    if np.any(nontrivial):
        positive = np.mean(signed[nontrivial] > 0)
        negative = np.mean(signed[nontrivial] < 0)
        consistency = float(max(positive, negative))
    else:
        consistency = 0.5 if len(signed) else float("nan")
    return {
        f"{prefix}_n": float(len(cycles)),
        f"{prefix}_usable_n": float(len(usable)),
        f"{prefix}_motion_median": median,
        f"{prefix}_motion_abs_median": _weighted_quantile(absolute, weights, 0.5),
        f"{prefix}_motion_iqr": float(q75 - q25),
        f"{prefix}_motion_max_abs": float(np.nanmax(absolute)) if len(absolute) else float("nan"),
        f"{prefix}_eso_fraction": float(np.mean(signed > 0.005)) if len(signed) else float("nan"),
        f"{prefix}_exo_fraction": float(np.mean(signed < -0.005)) if len(signed) else float("nan"),
        f"{prefix}_sign_consistency": consistency,
        f"{prefix}_quality_median": float(np.nanmedian(weights)) if len(weights) else float("nan"),
    }


def _numeric_summary(rows: pd.DataFrame, column: str) -> tuple[float, float]:
    if column not in rows:
        return float("nan"), float("nan")
    values = rows[column].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan")
    return float(np.median(values)), float(np.std(values))


def _state_numeric_summary(
    values: pd.Series | np.ndarray,
    prefix: str,
) -> dict[str, float]:
    """Summarize one ocular coordinate over a stable inferred cover state.

    Event-window jumps can be only one or two pixels in the legacy videos.  A
    stable-state summary uses all available samples in the same cover epoch and
    is therefore much less sensitive to a single landmark or transition-frame
    error.  Counts remain explicit so downstream models can distinguish a true
    small movement from an unobserved state.
    """

    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {
            f"{prefix}_n": 0.0,
            f"{prefix}_median": float("nan"),
            f"{prefix}_std": float("nan"),
            f"{prefix}_iqr": float("nan"),
            f"{prefix}_p10": float("nan"),
            f"{prefix}_p90": float("nan"),
        }
    p10, q25, median, q75, p90 = np.percentile(array, (10, 25, 50, 75, 90))
    return {
        f"{prefix}_n": float(len(array)),
        f"{prefix}_median": float(median),
        f"{prefix}_std": float(np.std(array)),
        f"{prefix}_iqr": float(q75 - q25),
        f"{prefix}_p10": float(p10),
        f"{prefix}_p90": float(p90),
    }


def _state_alignment_features(
    rows: pd.DataFrame,
    timeline: pd.DataFrame,
) -> dict[str, float]:
    """Aggregate binocular and unilateral-cover ocular alignment evidence.

    In addition to the iris-to-canthus coordinate, this derives a
    pupil/corneal-reflex (PCR) vector when a sufficiently reliable glint is
    visible.  The PCR coordinate is retained for the standardized
    camera-associated-light protocol while remaining missing, rather than
    fabricated, in legacy frames without a usable reflex.
    """

    state_names = ("binoc", "right_cov", "left_cov")
    eye_names = ("right", "left")
    coordinate_names = ("nasal", "pcr")
    result: dict[str, float] = {}

    if rows.empty or timeline.empty:
        aligned = rows.iloc[0:0].copy()
        aligned["right_covered"] = np.nan
        aligned["left_covered"] = np.nan
    else:
        state = timeline[
            ["frame", "right_covered", "left_covered"]
        ].drop_duplicates("frame", keep="last")
        aligned = rows.merge(state, on="frame", how="left", validate="one_to_one")

    for eye in eye_names:
        nasal_column = f"{eye}_nasal_u"
        if nasal_column not in aligned:
            aligned[nasal_column] = np.nan
        required = {
            "nose_x_px",
            f"{eye}_eye_mid_x_px",
            f"{eye}_glint_x",
            f"{eye}_glint_reliability",
        }
        if required.issubset(aligned.columns):
            nasal_sign = np.sign(
                aligned["nose_x_px"].to_numpy(dtype=float)
                - aligned[f"{eye}_eye_mid_x_px"].to_numpy(dtype=float)
            )
            nasal_sign[nasal_sign == 0.0] = 1.0
            pcr = -nasal_sign * aligned[f"{eye}_glint_x"].to_numpy(dtype=float)
            reliability = aligned[f"{eye}_glint_reliability"].to_numpy(dtype=float)
            pcr[~np.isfinite(reliability) | (reliability < 0.50)] = np.nan
            aligned[f"{eye}_pcr_u"] = pcr
        else:
            aligned[f"{eye}_pcr_u"] = np.nan

    known = (
        aligned.get("right_covered", pd.Series(np.nan, index=aligned.index)).notna()
        & aligned.get("left_covered", pd.Series(np.nan, index=aligned.index)).notna()
    )
    masks = {
        "binoc": known
        & (aligned["right_covered"] == 0.0)
        & (aligned["left_covered"] == 0.0),
        "right_cov": known
        & (aligned["right_covered"] == 1.0)
        & (aligned["left_covered"] == 0.0),
        "left_cov": known
        & (aligned["right_covered"] == 0.0)
        & (aligned["left_covered"] == 1.0),
    }

    for state_name in state_names:
        mask = masks[state_name]
        for eye in eye_names:
            result.update(
                _state_numeric_summary(
                    aligned.loc[mask, f"{eye}_nasal_u"],
                    f"{state_name}_{eye}_nasal",
                )
            )
            result.update(
                _state_numeric_summary(
                    aligned.loc[mask, f"{eye}_pcr_u"],
                    f"{state_name}_{eye}_pcr",
                )
            )
        result.update(
            _state_numeric_summary(
                aligned.loc[mask, "right_nasal_u"]
                - aligned.loc[mask, "left_nasal_u"],
                f"{state_name}_nasal_diff",
            )
        )
        result.update(
            _state_numeric_summary(
                aligned.loc[mask, "right_nasal_u"]
                + aligned.loc[mask, "left_nasal_u"],
                f"{state_name}_nasal_sum",
            )
        )
        result.update(
            _state_numeric_summary(
                aligned.loc[mask, "right_pcr_u"]
                - aligned.loc[mask, "left_pcr_u"],
                f"{state_name}_pcr_diff",
            )
        )

    # For each eye, compare its binocular position with its position while the
    # fellow eye is covered.  This is the state-aggregated cover-on refixation.
    for eye, fellow_covered_state in (
        ("right", "left_cov"),
        ("left", "right_cov"),
    ):
        for coordinate in coordinate_names:
            covered_position = result[f"{fellow_covered_state}_{eye}_{coordinate}_median"]
            binocular_position = result[f"binoc_{eye}_{coordinate}_median"]
            delta = float(covered_position - binocular_position)
            signed = float(-delta) if np.isfinite(delta) else float("nan")
            result[f"state_cover_on_{eye}_{coordinate}_eso_signed"] = signed
            result[f"state_cover_on_{eye}_{coordinate}_abs"] = (
                float(abs(signed)) if np.isfinite(signed) else float("nan")
            )

    for coordinate in coordinate_names:
        values = np.asarray(
            [
                result[f"state_cover_on_right_{coordinate}_eso_signed"],
                result[f"state_cover_on_left_{coordinate}_eso_signed"],
            ],
            dtype=float,
        )
        finite = values[np.isfinite(values)]
        result[f"state_cover_on_{coordinate}_available"] = float(bool(len(finite)))
        result[f"state_cover_on_{coordinate}_max_abs"] = (
            float(np.max(np.abs(finite))) if len(finite) else float("nan")
        )
        result[f"state_cover_on_{coordinate}_median"] = (
            float(np.median(finite)) if len(finite) else float("nan")
        )
        result[f"state_cover_on_{coordinate}_dominant_signed"] = (
            float(finite[np.argmax(np.abs(finite))])
            if len(finite)
            else float("nan")
        )
    return result


def _signed_displacement_summary(
    values: Iterable[float],
    prefix: str,
) -> dict[str, float]:
    """Fixed-size summary for anatomically signed paired iris displacement."""

    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {
            f"{prefix}_n": 0.0,
            f"{prefix}_median": float("nan"),
            f"{prefix}_p10": float("nan"),
            f"{prefix}_p90": float("nan"),
            f"{prefix}_dominant": float("nan"),
            f"{prefix}_positive_fraction": float("nan"),
        }
    return {
        f"{prefix}_n": float(len(array)),
        f"{prefix}_median": float(np.median(array)),
        f"{prefix}_p10": float(np.percentile(array, 10)),
        f"{prefix}_p90": float(np.percentile(array, 90)),
        f"{prefix}_dominant": float(array[np.argmax(np.abs(array))]),
        f"{prefix}_positive_fraction": float(np.mean(array > 0.0)),
    }


def _paired_iris_displacement_features(
    rows: pd.DataFrame,
    timeline: pd.DataFrame,
    *,
    test_type: TestType,
    fps: float,
) -> dict[str, float]:
    """Compare the same iris before and after inferred cover transitions.

    The original motion cycle used one early plateau ending about 0.5 seconds
    after a transition.  Legacy clips often show a slower refixation.  These
    features retain two later, non-overlapping clinical comparisons and both a
    raw and a three-sample median-smoothed iris-to-canthus coordinate.  Every
    displacement is patient-anatomical and eso-positive (temporal movement),
    never screen-coordinate based.

    Trusted frame annotations are not accepted here: ``timeline`` is the same
    inferred cover state available during raw-video inference.
    """

    event_kinds = ("cover_on", "uncover")
    groups = ("all", "right", "left")
    variants = ("raw", "smooth")
    horizons = ("late", "verylate")
    collected: dict[tuple[str, str, str, str], list[float]] = {
        (kind, group, variant, horizon): []
        for kind in event_kinds
        for group in groups
        for variant in variants
        for horizon in horizons
    }

    def emit() -> dict[str, float]:
        output: dict[str, float] = {}
        for key, values in collected.items():
            kind, group, variant, horizon = key
            output.update(
                _signed_displacement_summary(
                    values,
                    f"iris_pair_{kind}_{group}_{variant}_{horizon}",
                )
            )
        return output

    if rows.empty or timeline.empty:
        return emit()

    working = rows.sort_values("frame").drop_duplicates("frame", keep="first").copy()
    for eye in ("right", "left"):
        column = f"{eye}_nasal_u"
        if column not in working:
            working[column] = np.nan
        values = pd.to_numeric(working[column], errors="coerce")
        values = values.where(values.abs() <= 2.0)
        working[f"{eye}_iris_pair_raw"] = values
        working[f"{eye}_iris_pair_smooth"] = values.rolling(
            3, center=True, min_periods=1
        ).median()

    rate_scale = float(fps / 30.0) if np.isfinite(fps) and fps > 0 else 1.0

    def scaled(frame_offset: int) -> int:
        return int(round(frame_offset * rate_scale))

    windows = {
        "late": (scaled(10), scaled(20)),
        "verylate": (scaled(18), scaled(32)),
    }

    def position(column: str, start: int, stop: int) -> float:
        selected = working.loc[
            (working["frame"] >= start) & (working["frame"] <= stop),
            column,
        ].to_numpy(dtype=float)
        selected = selected[np.isfinite(selected)]
        return float(np.median(selected)) if len(selected) else float("nan")

    for covered_eye in ("right", "left"):
        fellow = "left" if covered_eye == "right" else "right"
        for kind, event_column, observed_eye in (
            ("cover_on", f"{covered_eye}_gets_covered", fellow),
            ("uncover", f"{covered_eye}_gets_uncovered", covered_eye),
        ):
            if event_column not in timeline:
                continue
            event_frames = timeline.loc[
                timeline[event_column].astype(bool), "frame"
            ].to_numpy(dtype=int)
            if kind == "cover_on" and test_type is TestType.ALTERNATE_COVER:
                paired_column = f"{fellow}_gets_uncovered"
                paired_frames = (
                    timeline.loc[
                        timeline[paired_column].astype(bool), "frame"
                    ].to_numpy(dtype=int)
                    if paired_column in timeline
                    else np.empty(0, dtype=int)
                )
                event_frames = np.asarray(
                    [
                        event
                        for event in event_frames
                        if not len(paired_frames)
                        or np.min(np.abs(paired_frames - event)) > scaled(2)
                    ],
                    dtype=int,
                )
            for event_frame in event_frames:
                baseline_start, baseline_stop = (
                    (event_frame - scaled(12), event_frame - scaled(2))
                    if kind == "cover_on"
                    else (event_frame, event_frame + scaled(3))
                )
                for variant in variants:
                    column = f"{observed_eye}_iris_pair_{variant}"
                    baseline = position(column, baseline_start, baseline_stop)
                    for horizon, (start_offset, stop_offset) in windows.items():
                        plateau = position(
                            column,
                            event_frame + start_offset,
                            event_frame + stop_offset,
                        )
                        signed = float(-(plateau - baseline))
                        if np.isfinite(signed):
                            collected[(kind, observed_eye, variant, horizon)].append(
                                signed
                            )
                            collected[(kind, "all", variant, horizon)].append(signed)
    return emit()


def _compact_uncover_iris_displacement_features(
    rows: pd.DataFrame,
    timeline: pd.DataFrame,
    *,
    fps: float,
) -> dict[str, float]:
    """Summarize delayed refixation of the newly uncovered eye.

    A covered iris is not a trustworthy geometric reference.  This comparison
    therefore starts only after the inferred uncover transition, using the
    first visible samples as that event's baseline, and compares them with the
    same anatomical iris-to-canthus coordinate 0.6--1.1 seconds later.  Values
    are pooled across inferred uncover events; positive displacement follows
    the package-wide eso sign convention.
    """

    prefix = "uncover_pair_nasal_u_late18_32_all"
    empty = {
        f"{prefix}_median": float("nan"),
        f"{prefix}_mean": float("nan"),
        f"{prefix}_p25": float("nan"),
        f"{prefix}_p75": float("nan"),
        f"{prefix}_dominant": float("nan"),
        f"{prefix}_positive_fraction": float("nan"),
    }
    if rows.empty or timeline.empty:
        return empty

    working = rows.sort_values("frame").drop_duplicates("frame", keep="first").copy()
    state_columns = ["frame", "right_covered", "left_covered"]
    if any(column not in timeline for column in state_columns):
        return empty
    working = working.merge(
        timeline[state_columns].drop_duplicates("frame", keep="first"),
        on="frame",
        how="left",
        validate="one_to_one",
    )
    rate_scale = float(fps / 30.0) if np.isfinite(fps) and fps > 0 else 1.0

    def scaled(frame_offset: int) -> int:
        return int(round(frame_offset * rate_scale))

    displacements: list[float] = []
    for uncovered_eye in ("right", "left"):
        event_column = f"{uncovered_eye}_gets_uncovered"
        position_column = f"{uncovered_eye}_nasal_u"
        if event_column not in timeline or position_column not in working:
            continue
        event_frames = timeline.loc[
            timeline[event_column].astype(bool), "frame"
        ].to_numpy(dtype=int)
        for event_frame in event_frames:
            both_uncovered = (
                (working["right_covered"] == 0.0)
                & (working["left_covered"] == 0.0)
            )
            early = working.loc[
                (working["frame"] >= event_frame)
                & (working["frame"] <= event_frame + scaled(8))
                & both_uncovered
            ].head(4)
            if len(early) < 2:
                early = working.loc[
                    (working["frame"] >= event_frame)
                    & (working["frame"] <= event_frame + scaled(12))
                    & (working[f"{uncovered_eye}_covered"] == 0.0)
                ].head(4)
            baseline_values = pd.to_numeric(
                early[position_column], errors="coerce"
            ).to_numpy(dtype=float)
            baseline_values = baseline_values[
                np.isfinite(baseline_values) & (np.abs(baseline_values) <= 2.0)
            ]
            if len(baseline_values) < 2:
                continue
            late = working.loc[
                (working["frame"] >= event_frame + scaled(18))
                & (working["frame"] <= event_frame + scaled(32))
                & (working[f"{uncovered_eye}_covered"] == 0.0),
                position_column,
            ]
            late_values = pd.to_numeric(late, errors="coerce").to_numpy(dtype=float)
            late_values = late_values[
                np.isfinite(late_values) & (np.abs(late_values) <= 2.0)
            ]
            if not len(late_values):
                continue
            baseline = float(np.median(baseline_values))
            displacements.extend((-(late_values - baseline)).tolist())

    values = np.asarray(displacements, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return empty
    return {
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_p25": float(np.percentile(values, 25)),
        f"{prefix}_p75": float(np.percentile(values, 75)),
        f"{prefix}_dominant": float(values[np.argmax(np.abs(values))]),
        f"{prefix}_positive_fraction": float(np.mean(values > 0.0)),
    }


def aggregate_video_features(
    observations: pd.DataFrame,
    timeline: pd.DataFrame,
    cycles: pd.DataFrame,
    *,
    video_uid: str,
    test_type: str | TestType,
    distance: str,
    correction: str,
    fps: float,
    frame_count: int,
    sampled_frame_count: int | None = None,
) -> dict[str, float | str]:
    """Produce the shared, fixed-size input used by all three task heads."""

    rows = _patient_rows(observations)
    test_type = TestType(str(test_type))
    if sampled_frame_count is None:
        sampled_frame_count = int(rows["frame"].nunique()) if "frame" in rows else 0
    sampled_frame_count = max(int(sampled_frame_count), 0)
    observed_frame_count = int(rows["frame"].nunique()) if "frame" in rows else 0
    starts_covered = float("nan")
    initial_binocular_baseline_observed = 0.0
    initial_state_observation_delay_s = float("nan")
    initial_state_from_delayed_observation = 0.0
    cover_state_known_fraction = 0.0
    binocular_uncovered_fraction = 0.0
    verified_binocular_segment_observed = 0.0
    if len(timeline):
        right_state = pd.to_numeric(
            timeline.get("right_covered", pd.Series(np.nan, index=timeline.index)),
            errors="coerce",
        ).to_numpy(dtype=float)
        left_state = pd.to_numeric(
            timeline.get("left_covered", pd.Series(np.nan, index=timeline.index)),
            errors="coerce",
        ).to_numpy(dtype=float)
        known = np.isfinite(right_state) & np.isfinite(left_state)
        binocular = known & (right_state == 0.0) & (left_state == 0.0)
        cover_state_known_fraction = float(np.mean(known))
        binocular_uncovered_fraction = float(np.mean(binocular))
        padded = np.r_[False, binocular, False].astype(int)
        changes = np.diff(padded)
        starts = np.flatnonzero(changes == 1)
        stops = np.flatnonzero(changes == -1)
        sampled_patient_frames = (
            set(rows["frame"].astype(int).tolist()) if "frame" in rows else set()
        )
        binocular_observed_counts = [
            sum(
                int(frame) in sampled_patient_frames
                for frame in timeline.iloc[int(start) : int(stop)]["frame"]
            )
            for start, stop in zip(starts, stops)
        ]
        # Dense inferred states may be interpolated between sparse landmark
        # samples.  Verification therefore requires three actual patient-face
        # observations inside one binocular run, not merely three dense rows.
        verified_binocular_segment_observed = float(
            max(binocular_observed_counts, default=0) >= 3
        )
        # Estimate the left-censored starting state from actual patient-face
        # samples.  Frame zero may be missing, especially in handheld clips;
        # a delayed estimate is retained and flagged rather than inventing a
        # cover transition at frame zero.
        actual_state_rows = timeline[
            timeline["frame"].astype(int).isin(sampled_patient_frames) & known
        ][["frame", "right_covered", "left_covered"]]
        if not actual_state_rows.empty:
            first_frame = int(actual_state_rows.iloc[0]["frame"])
            window_stop = first_frame + max(3, int(round(0.25 * fps)))
            initial_samples = actual_state_rows[
                actual_state_rows["frame"] <= window_stop
            ].head(3)
            state_pairs = initial_samples[
                ["right_covered", "left_covered"]
            ].to_numpy(dtype=float)
            stable = bool(
                len(state_pairs) >= (1 if first_frame == 0 else 2)
                and np.all(state_pairs == state_pairs[0])
            )
            if stable:
                right_initial, left_initial = state_pairs[0]
                starts_covered = float(bool(right_initial or left_initial))
                initial_binocular_baseline_observed = float(
                    right_initial == 0.0 and left_initial == 0.0
                )
                initial_state_observation_delay_s = float(
                    first_frame / fps if fps > 0 else np.nan
                )
                initial_state_from_delayed_observation = float(first_frame > 0)
    record: dict[str, float | str] = {
        "video_uid": video_uid,
        "test_type": test_type.value,
        "distance": distance,
        "correction": correction,
        "is_cover_uncover": float(test_type is TestType.COVER_UNCOVER),
        "is_near": float(str(distance).lower().startswith("n")),
        "with_correction": float(
            str(correction).lower() in {"with correction", "cc", "with_correction"}
        ),
        "fps": float(fps),
        "duration_s": float(frame_count / fps) if fps > 0 else float("nan"),
        "frame_count": float(frame_count),
        # Coverage must be relative to frames actually sent through the landmark
        # model.  Dividing by total video frames would make the training sampling
        # policy (dense transition windows versus coarse background frames) a
        # label-correlated predictor and would shift at deployment.
        "face_observation_fraction": float(
            observed_frame_count / max(sampled_frame_count, 1)
        ),
        "sampling_fraction": float(sampled_frame_count / max(frame_count, 1)),
        "starts_covered": starts_covered,
        "initial_binocular_baseline_observed": initial_binocular_baseline_observed,
        "initial_state_observation_delay_s": initial_state_observation_delay_s,
        "initial_state_from_delayed_observation": (
            initial_state_from_delayed_observation
        ),
        "cover_state_known_fraction": cover_state_known_fraction,
        "binocular_uncovered_fraction": binocular_uncovered_fraction,
        "verified_binocular_segment_observed": verified_binocular_segment_observed,
    }
    for subset_name, subset in (
        ("all", cycles),
        ("cover_on", cycles[cycles["event_kind"] == "gets_covered"] if not cycles.empty else cycles),
        (
            "uncover",
            cycles[cycles["event_kind"] == "gets_uncovered"] if not cycles.empty else cycles,
        ),
    ):
        record.update(_aggregate_cycles(subset, subset_name))
    if not cycles.empty:
        for eye in (Eye.RIGHT, Eye.LEFT):
            record.update(
                _aggregate_cycles(
                    cycles[cycles["observed_eye"] == eye.value],
                    f"observed_{eye.value}",
                )
            )
    for column in ("yaw_proxy", "pitch_proxy", "roll_radians"):
        median, spread = _numeric_summary(rows, column)
        record[f"{column}_median"] = median
        record[f"{column}_std"] = spread
    for eye in (Eye.RIGHT, Eye.LEFT):
        for suffix in (
            "eye_width_px",
            "iris_diameter_px",
            "aperture",
            "glint_reliability",
            "nasal_u",
        ):
            median, spread = _numeric_summary(rows, f"{eye.value}_{suffix}")
            record[f"{eye.value}_{suffix}_median"] = median
            record[f"{eye.value}_{suffix}_std"] = spread
    record.update(_state_alignment_features(rows, timeline))
    record.update(
        _paired_iris_displacement_features(
            rows,
            timeline,
            test_type=test_type,
            fps=fps,
        )
    )
    record.update(
        _compact_uncover_iris_displacement_features(
            rows,
            timeline,
            fps=fps,
        )
    )
    return record


def clinical_motion_evidence(features: dict[str, float | str]) -> dict[str, float]:
    """Transparent evidence scores used alongside learned probabilities."""

    cover_on = float(features.get("cover_on_motion_median", np.nan))
    uncover = float(features.get("uncover_motion_median", np.nan))
    cover_quality = float(features.get("cover_on_quality_median", 0.0))
    uncover_quality = float(features.get("uncover_quality_median", 0.0))
    return {
        "manifest_eso_evidence": float(max(cover_on, 0.0) * cover_quality)
        if np.isfinite(cover_on)
        else 0.0,
        "manifest_exo_evidence": float(max(-cover_on, 0.0) * cover_quality)
        if np.isfinite(cover_on)
        else 0.0,
        "latent_or_total_eso_evidence": float(max(uncover, 0.0) * uncover_quality)
        if np.isfinite(uncover)
        else 0.0,
        "latent_or_total_exo_evidence": float(max(-uncover, 0.0) * uncover_quality)
        if np.isfinite(uncover)
        else 0.0,
    }
