"""Pixel-derived iris/pupil displacement features for alternate-cover tests.

Face landmarks localize and normalize each eye crop, but the horizontal center
used here is measured independently from dark pixels in the crop.  Features
summarize movement around cover transitions inferred by the event model.  The
output schema is fixed so failed localization remains a valid forced-output
input rather than changing the trained feature contract.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


KINDS = ("cover", "uncover")
HORIZONS = ("early", "mid", "late")
EYES = ("right", "left")
TARGET_SECONDS = {
    "cover": {"early": 0.12, "mid": 0.28, "late": 0.50},
    "uncover": {"early": 0.25, "mid": 0.48, "late": 0.75},
}
SOURCE_SECONDS = {"cover": -0.18, "uncover": 0.08}
SUMMARY_NAMES = (
    "n",
    "median",
    "weighted_mean",
    "p20",
    "p80",
    "median_abs",
    "positive_fraction",
    "sign_consistency",
    "correlation_median",
    "cached_median",
    "template_cached_sign_agreement",
)


def dark_iris_feature_schema() -> dict[str, float]:
    """Return the fixed feature schema with neutral/missing defaults."""

    result: dict[str, float] = {}
    for kind in KINDS:
        for horizon in HORIZONS:
            for name in SUMMARY_NAMES:
                result[f"dark_{kind}_{horizon}_{name}"] = (
                    0.0 if name == "n" else float("nan")
                )
            for eye in EYES:
                result[f"dark_{kind}_{horizon}_{eye}_median"] = float("nan")
                result[f"dark_{kind}_{horizon}_{eye}_n"] = 0.0
    result["dark_requested_pair_n"] = 0.0
    result["dark_usable_pair_n"] = 0.0
    return result


def dark_iris_feature_names() -> tuple[str, ...]:
    """Return the stable ordered names emitted by :func:`dark_iris_features`."""

    return tuple(dark_iris_feature_schema())


def _patient_rows(observations: pd.DataFrame) -> pd.DataFrame:
    rows = observations
    if "is_patient_track" in rows:
        selected = rows[rows["is_patient_track"].astype(bool)]
        if not selected.empty:
            rows = selected
    if rows.empty:
        return rows.copy()
    return rows.sort_values("frame").drop_duplicates("frame", keep="first").copy()


def _nearest_row(
    rows: pd.DataFrame,
    *,
    target_frame: float,
    eye: str,
    maximum_distance: float,
) -> pd.Series | None:
    required = [
        f"{eye}_iris_x_px",
        f"{eye}_iris_y_px",
        f"{eye}_eye_mid_x_px",
        f"{eye}_eye_mid_y_px",
        f"{eye}_eye_width_px",
    ]
    if not set(required).issubset(rows.columns):
        return None
    valid = rows[required].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
    valid &= pd.to_numeric(rows[f"{eye}_eye_width_px"], errors="coerce") > 8.0
    candidates = rows[valid]
    if candidates.empty:
        return None
    distance = np.abs(candidates["frame"].to_numpy(float) - float(target_frame))
    index = int(np.argmin(distance))
    if float(distance[index]) > maximum_distance:
        return None
    return candidates.iloc[index]


def _read_selected_frames(path: Path, indices: set[int]) -> dict[int, np.ndarray]:
    if not indices:
        return {}
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return {}
    wanted = sorted(indices)
    result: dict[int, np.ndarray] = {}
    cursor = 0
    target_index = 0
    try:
        while target_index < len(wanted):
            ok, frame = capture.read()
            if not ok:
                break
            target = wanted[target_index]
            if cursor == target:
                result[target] = frame
                target_index += 1
            cursor += 1
    finally:
        capture.release()
    return result


def _normalized_patch(
    frame: np.ndarray,
    row: pd.Series,
    eye: str,
) -> tuple[np.ndarray, tuple[float, float]] | None:
    cx = float(row[f"{eye}_eye_mid_x_px"])
    cy = float(row[f"{eye}_eye_mid_y_px"])
    width = float(row[f"{eye}_eye_width_px"])
    iris_x = float(row[f"{eye}_iris_x_px"])
    iris_y = float(row[f"{eye}_iris_y_px"])
    roll = float(row.get("roll_radians", 0.0))
    if not all(np.isfinite(value) for value in (cx, cy, width, iris_x, iris_y, roll)):
        return None
    if width <= 8.0:
        return None
    rotation = cv2.getRotationMatrix2D((cx, cy), float(np.degrees(roll)), 1.0)
    aligned = cv2.warpAffine(
        frame,
        rotation,
        (frame.shape[1], frame.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    crop_width = max(24, int(round(1.55 * width)))
    crop_height = max(14, int(round(0.82 * width)))
    crop = cv2.getRectSubPix(aligned, (crop_width, crop_height), (cx, cy))
    if crop is None or crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (96, 48), interpolation=cv2.INTER_CUBIC)
    gray = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(6, 4)).apply(gray)
    cos_value = float(np.cos(roll))
    sin_value = float(np.sin(roll))
    dx = iris_x - cx
    dy = iris_y - cy
    aligned_dx = dx * cos_value + dy * sin_value
    aligned_dy = -dx * sin_value + dy * cos_value
    iris_u = 48.0 + 96.0 * aligned_dx / (1.55 * width)
    iris_v = 24.0 + 48.0 * aligned_dy / (0.82 * width)
    return gray, (iris_u, iris_v)


def _dark_center(
    patch: np.ndarray,
    expected: tuple[float, float],
) -> tuple[float, float, float] | None:
    expected_x, expected_y = expected
    x0 = max(0, int(round(expected_x)) - 19)
    x1 = min(patch.shape[1], int(round(expected_x)) + 20)
    y0 = max(0, int(round(expected_y)) - 12)
    y1 = min(patch.shape[0], int(round(expected_y)) + 13)
    roi = patch[y0:y1, x0:x1]
    if roi.shape[0] < 12 or roi.shape[1] < 18:
        return None
    smooth = cv2.GaussianBlur(roi, (5, 5), 0)
    yy, xx = np.mgrid[: roi.shape[0], : roi.shape[1]]
    center_x = expected_x - x0
    center_y = expected_y - y0
    ellipse = ((xx - center_x) / 17.0) ** 2 + ((yy - center_y) / 10.0) ** 2 <= 1.0
    values = smooth[ellipse]
    if values.size < 80:
        return None
    dark = float(np.percentile(values, 32.0))
    light = float(np.percentile(values, 78.0))
    if light - dark < 4.0:
        return None
    binary = ((smooth <= dark) & ellipse).astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, labels, statistics, centroids = cv2.connectedComponentsWithStats(binary, 8)
    best: tuple[float, int] | None = None
    for component in range(1, count):
        area = int(statistics[component, cv2.CC_STAT_AREA])
        if not 5 <= area <= 260:
            continue
        cx, cy = centroids[component]
        distance = ((cx - center_x) / 14.0) ** 2 + ((cy - center_y) / 8.0) ** 2
        score = float(distance - 0.012 * min(area, 80))
        if best is None or score < best[0]:
            best = (score, component)
    if best is None:
        return None
    selected = labels == best[1]
    weight = np.maximum(light - smooth.astype(float), 1.0) * selected
    total = float(np.sum(weight))
    if total <= 0.0:
        return None
    measured_x = x0 + float(np.sum(xx * weight) / total)
    measured_y = y0 + float(np.sum(yy * weight) / total)
    contrast_quality = float(np.clip((light - dark) / 80.0, 0.0, 1.0))
    proximity_quality = float(
        np.exp(
            -0.5
            * (
                ((measured_x - expected_x) / 10.0) ** 2
                + ((measured_y - expected_y) / 6.0) ** 2
            )
        )
    )
    return measured_x, measured_y, contrast_quality * proximity_quality


def _summarize(
    values: list[float],
    qualities: list[float],
    cached: list[float],
) -> dict[str, float]:
    value = np.asarray(values, dtype=float)
    quality = np.asarray(qualities, dtype=float)
    cache = np.asarray(cached, dtype=float)
    valid = np.isfinite(value) & np.isfinite(quality)
    value = value[valid]
    quality = quality[valid]
    cache = cache[valid]
    if not len(value):
        return {
            name: (0.0 if name == "n" else float("nan"))
            for name in SUMMARY_NAMES
        }
    p20, p80 = np.percentile(value, (20, 80))
    nontrivial = np.abs(value) > 0.004
    positive = float(np.mean(value[nontrivial] > 0)) if np.any(nontrivial) else 0.5
    weight = np.maximum(quality - 0.05, 0.01)
    comparable = np.isfinite(cache) & (np.abs(cache) > 0.004) & nontrivial
    agreement = (
        float(np.mean(np.sign(value[comparable]) == np.sign(cache[comparable])))
        if np.any(comparable)
        else float("nan")
    )
    return {
        "n": float(len(value)),
        "median": float(np.median(value)),
        "weighted_mean": float(np.average(value, weights=weight)),
        "p20": float(p20),
        "p80": float(p80),
        "median_abs": float(np.median(np.abs(value))),
        "positive_fraction": positive,
        "sign_consistency": max(positive, 1.0 - positive),
        "correlation_median": float(np.median(quality)),
        "cached_median": (
            float(np.nanmedian(cache)) if np.any(np.isfinite(cache)) else float("nan")
        ),
        "template_cached_sign_agreement": agreement,
    }


def dark_iris_features(
    video_path: str | Path,
    observations: pd.DataFrame,
    timeline: pd.DataFrame,
    *,
    fps: float,
) -> dict[str, float]:
    """Measure and aggregate horizontal dark-iris movement around transitions."""

    result = dark_iris_feature_schema()
    rows = _patient_rows(observations)
    if rows.empty or timeline.empty:
        return result
    maximum_distance = max(2.0, 0.12 * fps)
    specs: list[dict[str, Any]] = []
    for covered_eye, fellow_eye in (("right", "left"), ("left", "right")):
        for kind, event_column, observed_eye in (
            ("cover", f"{covered_eye}_gets_covered", fellow_eye),
            ("uncover", f"{covered_eye}_gets_uncovered", covered_eye),
        ):
            if event_column not in timeline:
                continue
            events = timeline.loc[
                timeline[event_column].astype(bool), "frame"
            ].to_numpy(dtype=int)
            for event_frame in events:
                source = _nearest_row(
                    rows,
                    target_frame=event_frame + SOURCE_SECONDS[kind] * fps,
                    eye=observed_eye,
                    maximum_distance=maximum_distance,
                )
                if source is None:
                    continue
                for horizon, seconds in TARGET_SECONDS[kind].items():
                    target = _nearest_row(
                        rows,
                        target_frame=event_frame + seconds * fps,
                        eye=observed_eye,
                        maximum_distance=maximum_distance,
                    )
                    if target is not None and int(target["frame"]) != int(source["frame"]):
                        specs.append(
                            {
                                "kind": kind,
                                "horizon": horizon,
                                "eye": observed_eye,
                                "source": source,
                                "target": target,
                            }
                        )
    result["dark_requested_pair_n"] = float(len(specs))
    frames = _read_selected_frames(
        Path(video_path),
        {
            int(spec[key]["frame"])
            for spec in specs
            for key in ("source", "target")
        },
    )
    grouped: dict[tuple[str, str], dict[str, list[Any]]] = defaultdict(
        lambda: {"value": [], "quality": [], "cached": [], "eye": []}
    )
    for spec in specs:
        source_index = int(spec["source"]["frame"])
        target_index = int(spec["target"]["frame"])
        if source_index not in frames or target_index not in frames:
            continue
        eye = str(spec["eye"])
        source_normalized = _normalized_patch(
            frames[source_index], spec["source"], eye
        )
        target_normalized = _normalized_patch(
            frames[target_index], spec["target"], eye
        )
        if source_normalized is None or target_normalized is None:
            continue
        source_center = _dark_center(*source_normalized)
        target_center = _dark_center(*target_normalized)
        if source_center is None or target_center is None:
            continue
        shift = (target_center[0] - source_center[0]) * 1.55 / 96.0
        quality = float(np.sqrt(source_center[2] * target_center[2]))
        source = spec["source"]
        target = spec["target"]
        nasal_sign = np.sign(
            float(source["nose_x_px"])
            - float(source[f"{eye}_eye_mid_x_px"])
        )
        if nasal_sign == 0 or not np.isfinite(nasal_sign):
            nasal_sign = 1.0
        key = (str(spec["kind"]), str(spec["horizon"]))
        grouped[key]["value"].append(float(-nasal_sign * shift))
        grouped[key]["quality"].append(quality)
        grouped[key]["cached"].append(
            float(-(target[f"{eye}_nasal_u"] - source[f"{eye}_nasal_u"]))
        )
        grouped[key]["eye"].append(eye)
    result["dark_usable_pair_n"] = float(
        sum(len(values["value"]) for values in grouped.values())
    )
    for (kind, horizon), values in grouped.items():
        for name, value in _summarize(
            values["value"], values["quality"], values["cached"]
        ).items():
            result[f"dark_{kind}_{horizon}_{name}"] = value
        eyes = np.asarray(values["eye"], dtype=object)
        shifts = np.asarray(values["value"], dtype=float)
        for eye in EYES:
            selected = shifts[eyes == eye]
            result[f"dark_{kind}_{horizon}_{eye}_median"] = (
                float(np.median(selected)) if len(selected) else float("nan")
            )
            result[f"dark_{kind}_{horizon}_{eye}_n"] = float(len(selected))
    return result


__all__ = [
    "dark_iris_feature_names",
    "dark_iris_feature_schema",
    "dark_iris_features",
]
