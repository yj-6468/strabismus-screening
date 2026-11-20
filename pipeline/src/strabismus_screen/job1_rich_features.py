"""Job 1-only event-centred horizontal iris comparisons.

These features consume only raw-video observations and a cover timeline inferred
from pixels.  They deliberately omit uncover-recovery summaries from the
manifest-presence gate because recovery can also be caused by a phoria.
"""

from __future__ import annotations

from collections.abc import Iterable
import math

import numpy as np
import pandas as pd


COORDINATES = ("raw", "roll", "pose", "pcr")
HORIZONS = ("early", "mid", "late")
EYES = ("right", "left")


def _patient_rows(observations: pd.DataFrame) -> pd.DataFrame:
    if observations.empty:
        return observations.copy()
    rows = observations
    if "is_patient_track" in rows:
        selected = rows[rows["is_patient_track"].astype(bool)]
        if not selected.empty:
            rows = selected
    return rows.sort_values("frame").drop_duplicates("frame", keep="first").copy()


def _finite(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    return array[np.isfinite(array)]


def _median(values: Iterable[float]) -> float:
    array = _finite(values)
    return float(np.median(array)) if len(array) else float("nan")


def _trimmed_mean(values: np.ndarray, fraction: float = 0.15) -> float:
    values = np.sort(values[np.isfinite(values)])
    if not len(values):
        return float("nan")
    trim = int(math.floor(fraction * len(values)))
    if trim and len(values) > 2 * trim:
        values = values[trim:-trim]
    return float(np.mean(values))


def _summary(values: list[float], weights: list[float], prefix: str) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    weight = np.asarray(weights, dtype=float)
    valid = np.isfinite(array) & np.isfinite(weight) & (weight > 0)
    array = array[valid]
    weight = weight[valid]
    if not len(array):
        return {
            f"{prefix}_n": 0.0,
            f"{prefix}_median": float("nan"),
            f"{prefix}_trimmed_mean": float("nan"),
            f"{prefix}_weighted_mean": float("nan"),
            f"{prefix}_p20": float("nan"),
            f"{prefix}_p80": float("nan"),
            f"{prefix}_dominant": float("nan"),
            f"{prefix}_median_abs": float("nan"),
            f"{prefix}_iqr": float("nan"),
            f"{prefix}_positive_fraction": float("nan"),
            f"{prefix}_sign_consistency": float("nan"),
            f"{prefix}_quality_median": float("nan"),
        }
    p20, q25, median, q75, p80 = np.percentile(array, (20, 25, 50, 75, 80))
    nontrivial = np.abs(array) > 0.0025
    positive_fraction = (
        float(np.mean(array[nontrivial] > 0)) if np.any(nontrivial) else 0.5
    )
    return {
        f"{prefix}_n": float(len(array)),
        f"{prefix}_median": float(median),
        f"{prefix}_trimmed_mean": _trimmed_mean(array),
        f"{prefix}_weighted_mean": float(np.average(array, weights=weight)),
        f"{prefix}_p20": float(p20),
        f"{prefix}_p80": float(p80),
        f"{prefix}_dominant": float(array[np.argmax(np.abs(array))]),
        f"{prefix}_median_abs": float(np.median(np.abs(array))),
        f"{prefix}_iqr": float(q75 - q25),
        f"{prefix}_positive_fraction": positive_fraction,
        f"{prefix}_sign_consistency": float(
            max(positive_fraction, 1.0 - positive_fraction)
        ),
        f"{prefix}_quality_median": float(np.median(weight)),
    }


def _coordinate_columns(rows: pd.DataFrame, quiet_mask: np.ndarray) -> pd.DataFrame:
    result = rows.copy()
    roll = pd.to_numeric(result.get("roll_radians"), errors="coerce").to_numpy(float)
    yaw = pd.to_numeric(result.get("yaw_proxy"), errors="coerce").to_numpy(float)
    pitch = pd.to_numeric(result.get("pitch_proxy"), errors="coerce").to_numpy(float)
    pose_matrix = np.column_stack((yaw, pitch, roll))
    pose_matrix[np.abs(pose_matrix) > 4.0] = np.nan

    for eye in EYES:
        width = pd.to_numeric(
            result.get(f"{eye}_eye_width_px"), errors="coerce"
        ).to_numpy(float)
        dx = (
            pd.to_numeric(result.get(f"{eye}_iris_x_px"), errors="coerce").to_numpy(float)
            - pd.to_numeric(result.get(f"{eye}_eye_mid_x_px"), errors="coerce").to_numpy(float)
        )
        dy = (
            pd.to_numeric(result.get(f"{eye}_iris_y_px"), errors="coerce").to_numpy(float)
            - pd.to_numeric(result.get(f"{eye}_eye_mid_y_px"), errors="coerce").to_numpy(float)
        )
        nasal_sign = np.sign(
            pd.to_numeric(result.get("nose_x_px"), errors="coerce").to_numpy(float)
            - pd.to_numeric(result.get(f"{eye}_eye_mid_x_px"), errors="coerce").to_numpy(float)
        )
        nasal_sign[nasal_sign == 0] = 1.0
        valid_width = np.isfinite(width) & (width > 1.0)
        raw = np.full(len(result), np.nan)
        raw[valid_width] = nasal_sign[valid_width] * dx[valid_width] / width[valid_width]
        corrected = np.full(len(result), np.nan)
        corrected[valid_width] = nasal_sign[valid_width] * (
            dx[valid_width] * np.cos(roll[valid_width])
            + dy[valid_width] * np.sin(roll[valid_width])
        ) / width[valid_width]

        pose_corrected = corrected.copy()
        fit_mask = (
            quiet_mask
            & np.isfinite(corrected)
            & (np.abs(corrected) <= 2.0)
            & np.all(np.isfinite(pose_matrix), axis=1)
        )
        if int(np.sum(fit_mask)) >= 8:
            design = pose_matrix[fit_mask]
            center = np.median(design, axis=0)
            design = design - center
            target = corrected[fit_mask]
            target_center = float(np.median(target))
            ridge = 0.05 * np.eye(design.shape[1])
            try:
                beta = np.linalg.solve(
                    design.T @ design + ridge,
                    design.T @ (target - target_center),
                )
                beta = np.clip(beta, -1.0, 1.0)
                all_valid = (
                    np.all(np.isfinite(pose_matrix), axis=1)
                    & np.isfinite(corrected)
                    & (np.abs(corrected) <= 2.0)
                )
                if np.all(np.isfinite(beta)):
                    with np.errstate(all="ignore"):
                        nuisance = (pose_matrix[all_valid] - center) @ beta
                    finite_nuisance = np.isfinite(nuisance)
                    valid_indices = np.flatnonzero(all_valid)[finite_nuisance]
                    pose_corrected[valid_indices] = (
                        corrected[valid_indices] - nuisance[finite_nuisance]
                    )
            except np.linalg.LinAlgError:
                pass

        reliability = pd.to_numeric(
            result.get(f"{eye}_glint_reliability"), errors="coerce"
        ).to_numpy(float)
        glint = pd.to_numeric(
            result.get(f"{eye}_glint_x"), errors="coerce"
        ).to_numpy(float)
        pcr = -nasal_sign * glint
        pcr[~np.isfinite(reliability) | (reliability < 0.50)] = np.nan

        result[f"{eye}_rich_raw"] = raw
        result[f"{eye}_rich_roll"] = corrected
        result[f"{eye}_rich_pose"] = pose_corrected
        result[f"{eye}_rich_pcr"] = pcr
    return result


def _position(
    rows: pd.DataFrame,
    column: str,
    start: int,
    stop: int,
    *,
    uncovered_column: str | None = None,
) -> tuple[float, int, float]:
    mask = (rows["frame"] >= start) & (rows["frame"] <= stop)
    if uncovered_column is not None:
        mask &= rows[uncovered_column] == 0.0
    selected = rows.loc[mask]
    values = pd.to_numeric(selected[column], errors="coerce").to_numpy(float)
    valid = np.isfinite(values) & (np.abs(values) <= 2.0)
    values = values[valid]
    if not len(values):
        return float("nan"), 0, float("nan")
    selected = selected.iloc[np.flatnonzero(valid)]
    yaw = pd.to_numeric(selected.get("yaw_proxy"), errors="coerce").to_numpy(float)
    pitch = pd.to_numeric(selected.get("pitch_proxy"), errors="coerce").to_numpy(float)
    pose = float(np.nanmedian(np.hypot(yaw, pitch))) if len(yaw) else float("nan")
    return float(np.median(values)), int(len(values)), pose


def _stable_mask(
    aligned: pd.DataFrame, event_frames: np.ndarray, guard: int
) -> np.ndarray:
    mask = np.ones(len(aligned), dtype=bool)
    frames = aligned["frame"].to_numpy(int)
    for event in event_frames:
        mask &= np.abs(frames - int(event)) > guard
    return mask


def job1_rich_features(
    observations: pd.DataFrame,
    timeline: pd.DataFrame,
    *,
    fps: float,
) -> dict[str, float]:
    """Return the fixed Job 1 rich-feature schema for one inferred timeline."""

    rows = _patient_rows(observations)
    result: dict[str, float] = {}
    for coordinate in COORDINATES:
        for horizon in HORIZONS:
            result.update(_summary([], [], f"rich_cover_{coordinate}_{horizon}"))
            for eye in EYES:
                result[f"rich_cover_{coordinate}_{horizon}_{eye}_median"] = float("nan")
                result[f"rich_cover_{coordinate}_{horizon}_{eye}_n"] = 0.0
        for name in (
            "quiet_right_median",
            "quiet_left_median",
            "quiet_sum_median",
            "quiet_diff_median",
            "quiet_sum_iqr",
            "quiet_diff_iqr",
            "stable_cover_on_right",
            "stable_cover_on_left",
            "stable_cover_on_median",
            "stable_cover_on_dominant",
            "stable_cover_on_abs_median",
        ):
            result[f"rich_{coordinate}_{name}"] = float("nan")
        result[f"rich_{coordinate}_quiet_n"] = 0.0
        result[f"rich_{coordinate}_stable_cover_on_n"] = 0.0

    if rows.empty or timeline.empty:
        result.update(
            {
                "rich_transition_n": 0.0,
                "rich_binocular_quiet_fraction": 0.0,
                "rich_event_density": 0.0,
            }
        )
        return result

    state_columns = (
        "frame",
        "right_covered",
        "left_covered",
        "right_gets_covered",
        "right_gets_uncovered",
        "left_gets_covered",
        "left_gets_uncovered",
    )
    states = timeline[list(state_columns)].drop_duplicates("frame", keep="last")
    aligned = rows.merge(states, on="frame", how="left", validate="one_to_one")
    transition_frames = np.sort(
        np.unique(
            np.concatenate(
                [
                    timeline.loc[timeline[column].astype(bool), "frame"].to_numpy(int)
                    for column in (
                        "right_gets_covered",
                        "right_gets_uncovered",
                        "left_gets_covered",
                        "left_gets_uncovered",
                    )
                ]
            )
        )
    )
    rate = float(fps / 30.0) if np.isfinite(fps) and fps > 0 else 1.0

    def scaled(value: int) -> int:
        return max(1, int(round(value * rate)))

    stable = _stable_mask(aligned, transition_frames, scaled(14))
    binocular = (
        (aligned["right_covered"] == 0.0)
        & (aligned["left_covered"] == 0.0)
    ).to_numpy(bool)
    quiet_mask = stable & binocular
    aligned = _coordinate_columns(aligned, quiet_mask)
    result["rich_transition_n"] = float(len(transition_frames))
    result["rich_binocular_quiet_fraction"] = float(np.mean(quiet_mask))
    duration = max(
        float(aligned["frame"].max() - aligned["frame"].min()) / max(fps, 1e-6),
        1e-6,
    )
    result["rich_event_density"] = float(len(transition_frames) / duration)

    for coordinate in COORDINATES:
        right = pd.to_numeric(aligned[f"right_rich_{coordinate}"], errors="coerce")
        left = pd.to_numeric(aligned[f"left_rich_{coordinate}"], errors="coerce")
        quiet_right = right[quiet_mask].to_numpy(float)
        quiet_left = left[quiet_mask].to_numpy(float)
        quiet_sum = (right + left)[quiet_mask].to_numpy(float)
        quiet_diff = (right - left)[quiet_mask].to_numpy(float)
        result[f"rich_{coordinate}_quiet_n"] = float(np.sum(np.isfinite(quiet_sum)))
        result[f"rich_{coordinate}_quiet_right_median"] = _median(quiet_right)
        result[f"rich_{coordinate}_quiet_left_median"] = _median(quiet_left)
        result[f"rich_{coordinate}_quiet_sum_median"] = _median(quiet_sum)
        result[f"rich_{coordinate}_quiet_diff_median"] = _median(quiet_diff)
        finite_sum = _finite(quiet_sum)
        finite_diff = _finite(quiet_diff)
        result[f"rich_{coordinate}_quiet_sum_iqr"] = (
            float(np.subtract(*np.percentile(finite_sum, [75, 25])))
            if len(finite_sum)
            else float("nan")
        )
        result[f"rich_{coordinate}_quiet_diff_iqr"] = (
            float(np.subtract(*np.percentile(finite_diff, [75, 25])))
            if len(finite_diff)
            else float("nan")
        )
        stable_motions: list[float] = []
        for eye, fellow_state in (("right", "left_covered"), ("left", "right_covered")):
            eye_values = pd.to_numeric(
                aligned[f"{eye}_rich_{coordinate}"], errors="coerce"
            )
            covered_mask = stable & (aligned[fellow_state] == 1.0).to_numpy(bool)
            baseline = _median(eye_values[quiet_mask].to_numpy(float))
            covered = _median(eye_values[covered_mask].to_numpy(float))
            signed = (
                float(-(covered - baseline))
                if np.isfinite(baseline) and np.isfinite(covered)
                else float("nan")
            )
            result[f"rich_{coordinate}_stable_cover_on_{eye}"] = signed
            if np.isfinite(signed):
                stable_motions.append(signed)
        result[f"rich_{coordinate}_stable_cover_on_n"] = float(len(stable_motions))
        result[f"rich_{coordinate}_stable_cover_on_median"] = _median(stable_motions)
        result[f"rich_{coordinate}_stable_cover_on_dominant"] = (
            float(stable_motions[int(np.argmax(np.abs(stable_motions)))])
            if stable_motions
            else float("nan")
        )
        result[f"rich_{coordinate}_stable_cover_on_abs_median"] = (
            _median(np.abs(stable_motions)) if stable_motions else float("nan")
        )

    horizon_windows = {
        "early": ((-12, -3), (3, 10)),
        "mid": ((-16, -3), (8, 18)),
        "late": ((-18, -3), (16, 32)),
    }
    values = {
        (coordinate, horizon): []
        for coordinate in COORDINATES
        for horizon in HORIZONS
    }
    weights = {key: [] for key in values}
    by_eye = {
        (coordinate, horizon, eye): []
        for coordinate in COORDINATES
        for horizon in HORIZONS
        for eye in EYES
    }
    for covered_eye in EYES:
        observed_eye = "left" if covered_eye == "right" else "right"
        event_frames = timeline.loc[
            timeline[f"{covered_eye}_gets_covered"].astype(bool), "frame"
        ].to_numpy(int)
        for event_frame in event_frames:
            for horizon, (baseline_offsets, plateau_offsets) in horizon_windows.items():
                baseline_start = event_frame - scaled(abs(baseline_offsets[0]))
                baseline_stop = event_frame - scaled(abs(baseline_offsets[1]))
                plateau_start = event_frame + scaled(plateau_offsets[0])
                plateau_stop = event_frame + scaled(plateau_offsets[1])
                for coordinate in COORDINATES:
                    column = f"{observed_eye}_rich_{coordinate}"
                    uncovered_column = f"{observed_eye}_covered"
                    baseline, n0, pose0 = _position(
                        aligned,
                        column,
                        baseline_start,
                        baseline_stop,
                        uncovered_column=uncovered_column,
                    )
                    plateau, n1, pose1 = _position(
                        aligned,
                        column,
                        plateau_start,
                        plateau_stop,
                        uncovered_column=uncovered_column,
                    )
                    if (
                        n0 < 2
                        or n1 < 2
                        or not np.isfinite(baseline)
                        or not np.isfinite(plateau)
                    ):
                        continue
                    signed = float(-(plateau - baseline))
                    pose_delta = (
                        abs(pose1 - pose0)
                        if np.isfinite(pose0) and np.isfinite(pose1)
                        else 0.25
                    )
                    quality = float(
                        min(1.0, n0 / 4.0)
                        * min(1.0, n1 / 4.0)
                        * np.exp(-pose_delta / 0.12)
                    )
                    key = (coordinate, horizon)
                    values[key].append(signed)
                    weights[key].append(max(quality, 1e-3))
                    by_eye[(coordinate, horizon, observed_eye)].append(signed)

    for (coordinate, horizon), event_values in values.items():
        prefix = f"rich_cover_{coordinate}_{horizon}"
        result.update(_summary(event_values, weights[(coordinate, horizon)], prefix))
        for eye in EYES:
            eye_values = by_eye[(coordinate, horizon, eye)]
            result[f"{prefix}_{eye}_median"] = _median(eye_values)
            result[f"{prefix}_{eye}_n"] = float(len(_finite(eye_values)))
    return result


def job1_rich_feature_names(table: pd.DataFrame) -> tuple[str, ...]:
    # Preserve extractor column order.  ExtraTrees samples feature indices at
    # random; sorting the same feature set would therefore change the fitted
    # model even though the names and values were identical to the validated
    # experimental finalist.
    return tuple(
        column
        for column in table.columns
        if column.startswith("rich_")
        and pd.api.types.is_numeric_dtype(table[column])
    )


__all__ = ["job1_rich_feature_names", "job1_rich_features"]
