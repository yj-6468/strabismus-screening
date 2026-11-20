"""Raw-video decoding and patient-centric ocular feature extraction.

The module deliberately emits auditable geometric and appearance features
instead of an opaque video embedding.  All horizontal coordinates are made
patient-centric: positive ``nasal_u`` means motion toward the tracked nose for
either eye, so mirroring does not reverse the clinical sign.
"""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import cv2
import numpy as np
import pandas as pd

from .native_limbus import STATUS_CODES, LimbusTracker, localize_limbus
from .types import Eye


# MediaPipe's refined face mesh has 478 points.  These indices follow the
# patient's anatomy, not screen-left/screen-right.
RIGHT_IRIS = (468, 469, 470, 471, 472)
LEFT_IRIS = (473, 474, 475, 476, 477)
RIGHT_EYE_CONTOUR = (
    33,
    7,
    163,
    144,
    145,
    153,
    154,
    155,
    133,
    173,
    157,
    158,
    159,
    160,
    161,
    246,
)
LEFT_EYE_CONTOUR = (
    263,
    249,
    390,
    373,
    374,
    380,
    381,
    382,
    362,
    398,
    384,
    385,
    386,
    387,
    388,
    466,
)

EYE_GEOMETRY = {
    Eye.RIGHT: {
        "iris": RIGHT_IRIS,
        "contour": RIGHT_EYE_CONTOUR,
        "corners": (33, 133),
        "lids": (159, 145),
        "cheek": 205,
    },
    Eye.LEFT: {
        "iris": LEFT_IRIS,
        "contour": LEFT_EYE_CONTOUR,
        "corners": (263, 362),
        "lids": (386, 374),
        "cheek": 425,
    },
}


@dataclass(frozen=True)
class VideoMetadata:
    path: str
    frame_count: int
    fps: float
    width: int
    height: int
    orientation_degrees: int
    decoded_frames: int
    sampled_frames: int
    patient_track_id: int | None
    # None means native pixels throughout; 160 identifies historical texture caches.
    texture_max_dimension: int | None = None


@dataclass
class VisionResult:
    metadata: VideoMetadata
    observations: pd.DataFrame

    @property
    def patient_observations(self) -> pd.DataFrame:
        if self.metadata.patient_track_id is None:
            return self.observations.iloc[0:0].copy()
        return self.observations[
            self.observations["track_id"] == self.metadata.patient_track_id
        ].copy()


def _point_array(landmarks: Sequence[object], width: int, height: int) -> np.ndarray:
    points = np.empty((len(landmarks), 3), dtype=np.float32)
    for index, landmark in enumerate(landmarks):
        points[index] = (
            float(landmark.x) * width,
            float(landmark.y) * height,
            float(landmark.z) * width,
        )
    return points


def _safe_crop(
    image: np.ndarray,
    center_x: float,
    center_y: float,
    half_width: float,
    half_height: float,
) -> np.ndarray:
    height, width = image.shape[:2]
    x0 = max(0, int(np.floor(center_x - half_width)))
    x1 = min(width, int(np.ceil(center_x + half_width)))
    y0 = max(0, int(np.floor(center_y - half_height)))
    y1 = min(height, int(np.ceil(center_y + half_height)))
    if x1 <= x0 or y1 <= y0:
        return np.empty((0, 0, 3), dtype=np.uint8)
    return image[y0:y1, x0:x1]


def _patch_stats(patch: np.ndarray, *, texture_max_dimension: int | None = None) -> dict[str, float]:
    if patch.size == 0 or min(patch.shape[:2]) < 3:
        names = (
            "gray_mean",
            "gray_std",
            "gray_q10",
            "gray_q50",
            "gray_q90",
            "saturation_mean",
            "laplacian_variance",
            "edge_density",
        )
        return {name: float("nan") for name in names}

    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    small = gray
    if texture_max_dimension is not None and max(gray.shape) > texture_max_dimension:
        scale = float(texture_max_dimension) / max(gray.shape)
        small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    laplacian = cv2.Laplacian(small, cv2.CV_32F)
    sobel_x = cv2.Sobel(small, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(small, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(sobel_x, sobel_y)
    q10, q50, q90 = np.percentile(gray, (10, 50, 90))
    return {
        "gray_mean": float(np.mean(gray)),
        "gray_std": float(np.std(gray)),
        "gray_q10": float(q10),
        "gray_q50": float(q50),
        "gray_q90": float(q90),
        "saturation_mean": float(np.mean(hsv[..., 1])),
        "laplacian_variance": float(np.var(laplacian) / (255.0**2)),
        "edge_density": float(np.mean(gradient > 28.0)),
    }


def _skin_similarity(eye_patch: np.ndarray, cheek_patch: np.ndarray) -> tuple[float, float]:
    if eye_patch.size == 0 or cheek_patch.size == 0:
        return float("nan"), float("nan")
    eye_lab = cv2.cvtColor(eye_patch, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    cheek_lab = cv2.cvtColor(cheek_patch, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(
        np.float32
    )
    reference = np.median(cheek_lab, axis=0)
    distances = np.linalg.norm(eye_lab - reference[None, :], axis=1)
    return float(np.mean(distances < 30.0)), float(np.median(distances))


def _glint_features(
    frame: np.ndarray,
    iris_center: np.ndarray,
    iris_diameter: float,
) -> tuple[float, float, float]:
    """Return glint x/y relative to iris and a conservative reliability score."""

    if not np.isfinite(iris_diameter) or iris_diameter < 3.0:
        return float("nan"), float("nan"), 0.0
    radius = max(3, int(round(iris_diameter * 0.65)))
    patch = _safe_crop(frame, iris_center[0], iris_center[1], radius, radius)
    if patch.size == 0:
        return float("nan"), float("nan"), 0.0
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    threshold = max(215.0, float(np.percentile(gray, 99.0)))
    mask = (gray >= threshold).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return float("nan"), float("nan"), 0.0
    candidates: list[tuple[float, int]] = []
    patch_center = np.array([(patch.shape[1] - 1) / 2, (patch.shape[0] - 1) / 2])
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if not 1 <= area <= max(3, int(0.15 * patch.size / 3)):
            continue
        distance = float(np.linalg.norm(centroids[label] - patch_center))
        if distance <= radius * 0.9:
            candidates.append((distance - 0.05 * area, label))
    if not candidates:
        return float("nan"), float("nan"), 0.0
    _, label = min(candidates)
    centroid = centroids[label]
    dx = float((centroid[0] - patch_center[0]) / iris_diameter)
    dy = float((centroid[1] - patch_center[1]) / iris_diameter)
    contrast = float((np.max(gray) - np.median(gray)) / 255.0)
    reliability = float(np.clip(contrast * 1.8, 0.0, 1.0))
    return dx, dy, reliability


def _eye_features(
    frame: np.ndarray,
    points: np.ndarray,
    eye: Eye,
    nose_x: float,
    *,
    texture_max_dimension: int | None = None,
) -> dict[str, float]:
    definition = EYE_GEOMETRY[eye]
    corner_a, corner_b = (points[index, :2] for index in definition["corners"])
    eye_midpoint = (corner_a + corner_b) / 2.0
    eye_width = float(np.linalg.norm(corner_a - corner_b))
    iris_points = points[list(definition["iris"]), :2]
    iris_center = np.mean(iris_points, axis=0)
    pairwise = iris_points[:, None, :] - iris_points[None, :, :]
    iris_diameter = float(np.max(np.linalg.norm(pairwise, axis=-1)))
    nasal_sign = float(np.sign(nose_x - eye_midpoint[0]))
    if nasal_sign == 0.0:
        nasal_sign = 1.0
    denominator = max(eye_width, 1e-6)
    nasal_u = float(nasal_sign * (iris_center[0] - eye_midpoint[0]) / denominator)
    vertical_v = float((iris_center[1] - eye_midpoint[1]) / denominator)
    lid_a, lid_b = (points[index, :2] for index in definition["lids"])
    aperture = float(np.linalg.norm(lid_a - lid_b) / denominator)

    contour = points[list(definition["contour"]), :2]
    contour_min = np.min(contour, axis=0)
    contour_max = np.max(contour, axis=0)
    contour_center = (contour_min + contour_max) / 2.0
    contour_size = np.maximum(contour_max - contour_min, 1.0)
    eye_patch = _safe_crop(
        frame,
        contour_center[0],
        contour_center[1] - contour_size[1] * 0.35,
        contour_size[0] * 0.85,
        max(contour_size[1] * 2.2, eye_width * 0.42),
    )
    cheek_point = points[int(definition["cheek"]), :2]
    cheek_patch = _safe_crop(
        frame,
        cheek_point[0],
        cheek_point[1],
        eye_width * 0.45,
        eye_width * 0.45,
    )
    eye_stats = _patch_stats(eye_patch, texture_max_dimension=texture_max_dimension)
    cheek_stats = _patch_stats(cheek_patch, texture_max_dimension=texture_max_dimension)
    skin_fraction, skin_distance = _skin_similarity(eye_patch, cheek_patch)
    cheek_mean = cheek_stats["gray_mean"]
    eye_median = eye_stats["gray_q50"]
    dark_score = float(
        np.clip((cheek_mean - eye_median) / max(cheek_mean, 30.0), 0.0, 1.0)
    )
    texture_ratio = float(
        eye_stats["laplacian_variance"]
        / max(cheek_stats["laplacian_variance"], 1e-5)
    )
    smooth_score = float(np.clip(1.0 - texture_ratio, 0.0, 1.0))
    skin_cover_score = float(np.nan_to_num(skin_fraction) * smooth_score)
    neutral_cover_score = float(
        smooth_score * (1.0 - np.nan_to_num(skin_fraction)) * (1.0 - dark_score)
    )
    cover_proxy = float(max(dark_score, 0.75 * skin_cover_score, 0.65 * neutral_cover_score))
    glint_x, glint_y, glint_reliability = _glint_features(
        frame, iris_center, iris_diameter
    )

    features: dict[str, float] = {
        "iris_x_px": float(iris_center[0]),
        "iris_y_px": float(iris_center[1]),
        "eye_mid_x_px": float(eye_midpoint[0]),
        "eye_mid_y_px": float(eye_midpoint[1]),
        "eye_width_px": eye_width,
        "iris_diameter_px": iris_diameter,
        "nasal_u": nasal_u,
        "vertical_v": vertical_v,
        "aperture": aperture,
        "skin_fraction": skin_fraction,
        "skin_distance": skin_distance,
        "texture_ratio": texture_ratio,
        "dark_score": dark_score,
        "smooth_score": smooth_score,
        "skin_cover_score": skin_cover_score,
        "neutral_cover_score": neutral_cover_score,
        "cover_proxy": cover_proxy,
        "glint_x": glint_x,
        "glint_y": glint_y,
        "glint_reliability": glint_reliability,
    }
    features.update({f"eye_{key}": value for key, value in eye_stats.items()})
    features.update({f"cheek_{key}": value for key, value in cheek_stats.items()})
    return features


def _face_features(points: np.ndarray, width: int, height: int) -> dict[str, float]:
    xy = points[:, :2]
    minimum = np.min(xy, axis=0)
    maximum = np.max(xy, axis=0)
    center = (minimum + maximum) / 2.0
    span = np.maximum(maximum - minimum, 1.0)
    right_mid = np.mean(points[[33, 133], :2], axis=0)
    left_mid = np.mean(points[[263, 362], :2], axis=0)
    eye_axis = left_mid - right_mid
    interocular = float(np.linalg.norm(eye_axis))
    eye_center = (left_mid + right_mid) / 2.0
    nose = points[1, :2]
    yaw_proxy = float((nose[0] - eye_center[0]) / max(interocular, 1.0))
    pitch_proxy = float((nose[1] - eye_center[1]) / max(interocular, 1.0))
    roll_radians = float(np.arctan2(eye_axis[1], eye_axis[0]))
    return {
        "face_center_x": float(center[0] / width),
        "face_center_y": float(center[1] / height),
        "face_width": float(span[0] / width),
        "face_height": float(span[1] / height),
        "face_area": float((span[0] * span[1]) / (width * height)),
        "interocular_px": interocular,
        "yaw_proxy": yaw_proxy,
        "pitch_proxy": pitch_proxy,
        "roll_radians": roll_radians,
        "nose_x_px": float(nose[0]),
        "nose_y_px": float(nose[1]),
    }


RIGID_LANDMARKS = (6, 197, 195, 5, 4, 168, 234, 454, 10, 152, 50, 280)


def _head_pose_from_matrix(matrix: np.ndarray | None) -> dict[str, float]:
    names = ("head_yaw_deg", "head_pitch_deg", "head_roll_deg", "head_tx", "head_ty", "head_tz")
    if matrix is None:
        return {name: float("nan") for name in names}
    m = np.asarray(matrix, dtype=float)
    if m.shape != (4, 4) or not np.all(np.isfinite(m)):
        return {name: float("nan") for name in names}
    yaw = math.degrees(math.atan2(-m[2, 0], math.hypot(m[2, 1], m[2, 2])))
    pitch = math.degrees(math.atan2(m[2, 1], m[2, 2]))
    roll = math.degrees(math.atan2(m[1, 0], m[0, 0]))
    return {
        "head_yaw_deg": float(yaw),
        "head_pitch_deg": float(pitch),
        "head_roll_deg": float(roll),
        "head_tx": float(m[0, 3]),
        "head_ty": float(m[1, 3]),
        "head_tz": float(m[2, 3]),
    }


def _rigid_landmark_columns(points: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    for index in RIGID_LANDMARKS:
        out[f"lm{index}_x_px"] = float(points[index, 0])
        out[f"lm{index}_y_px"] = float(points[index, 1])
    return out


def _native_limbus_features(
    frame: np.ndarray,
    points: np.ndarray,
    eye: Eye,
    tracker: LimbusTracker,
    frame_index: int,
) -> dict[str, float]:
    """Measure the limbus on native pixels; landmarks only place the crop and seed."""

    definition = EYE_GEOMETRY[eye]
    temporal_idx, nasal_idx = definition["corners"]
    temporal = points[temporal_idx, :2]
    nasal = points[nasal_idx, :2]
    mid = (temporal + nasal) / 2.0
    width = float(np.linalg.norm(nasal - temporal))
    result: dict[str, float] = {
        "temporal_canthus_x_px": float(temporal[0]),
        "temporal_canthus_y_px": float(temporal[1]),
        "nasal_canthus_x_px": float(nasal[0]),
        "nasal_canthus_y_px": float(nasal[1]),
        "limbus_x_px": float("nan"),
        "limbus_y_px": float("nan"),
        "limbus_r_px": float("nan"),
        "limbus_conf": 0.0,
        "limbus_step_luma": float("nan"),
        "limbus_status": 0.0,
        "limbus_temporal_score": float("nan"),
        "limbus_nasal_score": float("nan"),
        "limbus_nasal_u": float("nan"),
    }
    contour_points = points[list(definition["contour"]), :2]
    for k, (px, py) in enumerate(contour_points):
        result[f"contour{k}_x_px"] = float(px)
        result[f"contour{k}_y_px"] = float(py)
    if not np.isfinite(width) or width < 12.0:
        return result
    hw, hh = 1.4 * width, 0.85 * width
    x0 = max(0, int(math.floor(mid[0] - hw)))
    x1 = min(frame.shape[1], int(math.ceil(mid[0] + hw)))
    y0 = max(0, int(math.floor(mid[1] - hh)))
    y1 = min(frame.shape[0], int(math.ceil(mid[1] + hh)))
    if x1 - x0 < 24 or y1 - y0 < 16:
        return result
    crop = frame[y0:y1, x0:x1]
    iris_points = points[list(definition["iris"]), :2]
    iris_center = np.mean(iris_points, axis=0)
    pairwise = iris_points[:, None, :] - iris_points[None, :, :]
    iris_radius = float(np.max(np.linalg.norm(pairwise, axis=-1))) / 2.0
    seed_x, seed_y = tracker.seed(frame_index, float(iris_center[0]), float(iris_center[1]))
    r_prior = tracker.r_prior
    contour = points[list(definition["contour"]), :2] - np.array([x0, y0], dtype=float)
    fit = localize_limbus(
        crop,
        cx0=seed_x - x0,
        cy0=seed_y - y0,
        r0=iris_radius,
        r_prior=r_prior if np.isfinite(r_prior) else None,
        contour=contour,
    )
    tracker.update(frame_index, fit, float(x0), float(y0))
    if np.isfinite(float(fit["cx"])):
        cx = float(fit["cx"]) + x0
        cy = float(fit["cy"]) + y0
        axis = nasal - temporal
        axis_n = axis / max(np.linalg.norm(axis), 1e-6)
        nasal_u = float((np.array([cx, cy]) - mid) @ axis_n / max(width, 1e-6))
        result.update(
            limbus_x_px=cx,
            limbus_y_px=cy,
            limbus_r_px=float(fit["r"]),
            limbus_conf=float(fit["conf"]),
            limbus_step_luma=float(fit["step_luma"]),
            limbus_status=float(STATUS_CODES.get(str(fit["status"]), 0)),
            limbus_temporal_score=float(fit["temporal_score"]),
            limbus_nasal_score=float(fit["nasal_score"]),
            limbus_nasal_u=nasal_u,
        )
    return result


class _TrackAssigner:
    def __init__(self, maximum_distance: float = 0.28, maximum_gap: int = 20):
        self.maximum_distance = maximum_distance
        self.maximum_gap = maximum_gap
        self._tracks: dict[int, tuple[np.ndarray, int]] = {}
        self._next_id = 0

    def assign(self, centers: list[np.ndarray], frame_index: int) -> list[int]:
        active = {
            track_id: state
            for track_id, state in self._tracks.items()
            if frame_index - state[1] <= self.maximum_gap
        }
        assignments: list[int] = []
        used: set[int] = set()
        for center in centers:
            choices = [
                (float(np.linalg.norm(center - state[0])), track_id)
                for track_id, state in active.items()
                if track_id not in used
            ]
            if choices and min(choices)[0] <= self.maximum_distance:
                _, track_id = min(choices)
            else:
                track_id = self._next_id
                self._next_id += 1
            assignments.append(track_id)
            used.add(track_id)
            self._tracks[track_id] = (center, frame_index)
        return assignments


def select_patient_track(observations: pd.DataFrame, sampled_frames: int) -> int | None:
    """Choose the face undergoing unilateral occlusion rather than a bystander."""

    if observations.empty:
        return None
    scores: list[tuple[float, int]] = []
    for track_id, group in observations.groupby("track_id", sort=True):
        right = group["right_cover_proxy"].to_numpy(dtype=float)
        left = group["left_cover_proxy"].to_numpy(dtype=float)
        difference = np.nan_to_num(right - left)
        if len(difference):
            p90 = float(np.percentile(np.abs(difference), 90))
            activity = float(np.std(difference))
            changes = float(np.mean(np.abs(np.diff(difference)) > 0.18)) if len(difference) > 1 else 0.0
        else:
            p90 = activity = changes = 0.0
        coverage = min(1.0, len(group) / max(sampled_frames, 1))
        size = float(np.nanmedian(group["face_area"]))
        dx = group["face_center_x"].to_numpy(dtype=float) - 0.5
        dy = group["face_center_y"].to_numpy(dtype=float) - 0.5
        centrality = float(1.0 - np.clip(np.nanmedian(np.hypot(dx, dy)) / 0.71, 0.0, 1.0))
        score = (
            3.2 * activity
            + 2.0 * p90
            + 1.4 * changes
            + 0.8 * coverage
            + 1.3 * np.sqrt(max(size, 0.0))
            + 0.2 * centrality
        )
        scores.append((score, int(track_id)))
    return max(scores)[1] if scores else None


class FaceLandmarkerBackend(AbstractContextManager["FaceLandmarkerBackend"]):
    """Thin CPU-delegate wrapper around the official MediaPipe task model."""

    def __init__(
        self,
        model_path: str | Path,
        maximum_faces: int = 2,
        detection_confidence: float = 0.30,
        presence_confidence: float = 0.30,
        tracking_confidence: float = 0.30,
    ) -> None:
        self.model_path = Path(model_path)
        self.maximum_faces = maximum_faces
        self.detection_confidence = detection_confidence
        self.presence_confidence = presence_confidence
        self.tracking_confidence = tracking_confidence
        self._landmarker = None
        self._last_inference_timestamp_ms = -1
        self.last_matrices: list[np.ndarray] = []

    def __enter__(self) -> "FaceLandmarkerBackend":
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Face Landmarker model not found at {self.model_path}. "
                "Download the official MediaPipe face_landmarker.task asset."
            )
        import mediapipe as mp

        base_options = mp.tasks.BaseOptions(
            model_asset_path=str(self.model_path),
            delegate=mp.tasks.BaseOptions.Delegate.CPU,
        )
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=self.maximum_faces,
            min_face_detection_confidence=self.detection_confidence,
            min_face_presence_confidence=self.presence_confidence,
            min_tracking_confidence=self.tracking_confidence,
            output_facial_transformation_matrixes=True,
        )
        try:
            self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        except RuntimeError as error:
            if "kGpuService" in str(error) or "NSOpenGLPixelFormat" in str(error):
                raise RuntimeError(
                    "MediaPipe could not access the local macOS graphics service, even "
                    "with CPU inference selected. Run the extraction command from a "
                    "normal desktop terminal rather than a headless sandbox."
                ) from error
            raise
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def begin_video(self, separation_ms: int = 10_000) -> int:
        """Return a safe timestamp offset for another independent clip.

        MediaPipe's VIDEO API requires timestamps to increase for the lifetime
        of a landmarker.  A batch extractor can therefore reuse one expensive
        model instance while keeping the timestamps stored in the feature table
        local to each clip.  The deliberate gap also forces the tracking graph
        to reacquire the face at a clip boundary.
        """

        if separation_ms < 1:
            raise ValueError("separation_ms must be positive")
        if self._landmarker is None:
            raise RuntimeError("FaceLandmarkerBackend must be used as a context manager")
        return 0 if self._last_inference_timestamp_ms < 0 else (
            self._last_inference_timestamp_ms + int(separation_ms)
        )

    def detect(self, rgb_frame: np.ndarray, timestamp_ms: int) -> list[Sequence[object]]:
        if self._landmarker is None:
            raise RuntimeError("FaceLandmarkerBackend must be used as a context manager")
        if timestamp_ms <= self._last_inference_timestamp_ms:
            raise ValueError(
                "MediaPipe VIDEO timestamps must be strictly increasing; call "
                "begin_video() before extracting another clip"
            )
        import mediapipe as mp

        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        self._last_inference_timestamp_ms = int(timestamp_ms)
        self.last_matrices = [
            np.asarray(matrix, dtype=np.float32)
            for matrix in (result.facial_transformation_matrixes or [])
        ]
        return list(result.face_landmarks)


def _wanted_frames(
    frame_count: int,
    stride: int,
    frame_indices: Iterable[int] | None,
) -> set[int] | None:
    if frame_indices is None:
        return None
    return {int(index) for index in frame_indices if 0 <= int(index) < frame_count}


def extract_video_observations(
    video_path: str | Path,
    model_path: str | Path,
    *,
    maximum_faces: int = 2,
    stride: int = 1,
    frame_indices: Iterable[int] | None = None,
    max_frames: int | None = None,
    backend: FaceLandmarkerBackend | None = None,
    native_limbus: bool = True,
    texture_max_dimension: int | None = None,
) -> VisionResult:
    """Extract one wide feature row per tracked face and sampled frame.

    With ``native_limbus`` the limbus of each eye is additionally measured on
    the native frame pixels (see :mod:`native_limbus`), and the anatomical eye
    corners, a rigid landmark subset, and the Face Landmarker head-pose matrix
    are stored so downstream features can build head-stabilized coordinates.

    OpenCV's orientation auto-rotation is explicitly enabled so iPhone portrait
    clips are analyzed in display orientation.  Presentation timestamps are
    retained when the container exposes them; a monotonic FPS fallback is used
    otherwise.
    """

    path = Path(video_path)
    if stride < 1:
        raise ValueError("stride must be at least 1")
    if texture_max_dimension is not None and texture_max_dimension < 3:
        raise ValueError("texture_max_dimension must be at least 3 or None")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video: {path}")
    if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
        capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    orientation = 0
    if hasattr(cv2, "CAP_PROP_ORIENTATION_META"):
        orientation = int(round(capture.get(cv2.CAP_PROP_ORIENTATION_META)))
    wanted = _wanted_frames(frame_count, stride, frame_indices)
    rows: list[dict[str, float | int | str]] = []
    assigner = _TrackAssigner(maximum_gap=max(20, stride * 5))
    trackers: dict[tuple[int, str], LimbusTracker] = {}
    decoded_frames = 0
    sampled_frames = 0
    last_timestamp = -1

    owns_backend = backend is None
    backend_context = (
        FaceLandmarkerBackend(model_path, maximum_faces=maximum_faces)
        if owns_backend
        else nullcontext(backend)
    )
    try:
        with backend_context as active_backend:
            if active_backend is None:  # pragma: no cover - nullcontext typing guard
                raise RuntimeError("No face-landmarker backend is available")
            inference_timestamp_offset = active_backend.begin_video()
            frame_index = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                decoded_frames += 1
                should_sample = (
                    frame_index in wanted if wanted is not None else frame_index % stride == 0
                )
                if should_sample:
                    sampled_frames += 1
                    presentation_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
                    fallback_ms = frame_index * 1000.0 / fps
                    if not np.isfinite(presentation_ms) or presentation_ms <= last_timestamp:
                        presentation_ms = fallback_ms
                    timestamp_ms = max(last_timestamp + 1, int(round(presentation_ms)))
                    last_timestamp = timestamp_ms
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    detected = active_backend.detect(
                        rgb, inference_timestamp_offset + timestamp_ms
                    )
                    matrices = list(getattr(active_backend, "last_matrices", []))
                    point_sets = []
                    matrix_sets = []
                    for face_index, landmarks in enumerate(detected):
                        if len(landmarks) < 478:
                            continue
                        point_sets.append(
                            _point_array(landmarks, frame.shape[1], frame.shape[0])
                        )
                        matrix_sets.append(
                            matrices[face_index] if face_index < len(matrices) else None
                        )
                    face_data = [
                        _face_features(points, frame.shape[1], frame.shape[0])
                        for points in point_sets
                    ]
                    centers = [
                        np.array([data["face_center_x"], data["face_center_y"]])
                        for data in face_data
                    ]
                    track_ids = assigner.assign(centers, frame_index)
                    for track_id, points, face_features, matrix in zip(
                        track_ids, point_sets, face_data, matrix_sets, strict=True
                    ):
                        row: dict[str, float | int | str] = {
                            "video": str(path),
                            "frame": frame_index,
                            "timestamp_ms": timestamp_ms,
                            "track_id": track_id,
                        }
                        row.update(face_features)
                        for eye in (Eye.RIGHT, Eye.LEFT):
                            features = _eye_features(
                                frame, points, eye, face_features["nose_x_px"],
                                texture_max_dimension=texture_max_dimension,
                            )
                            row.update({f"{eye.value}_{key}": value for key, value in features.items()})
                            if native_limbus:
                                tracker = trackers.setdefault(
                                    (int(track_id), eye.value), LimbusTracker()
                                )
                                native = _native_limbus_features(
                                    frame, points, eye, tracker, frame_index
                                )
                                row.update(
                                    {f"{eye.value}_{key}": value for key, value in native.items()}
                                )
                        if native_limbus:
                            row.update(_head_pose_from_matrix(matrix))
                            row.update(_rigid_landmark_columns(points))
                        rows.append(row)
                    if max_frames is not None and sampled_frames >= max_frames:
                        break
                frame_index += 1
    finally:
        capture.release()

    observations = pd.DataFrame.from_records(rows)
    patient_track_id = select_patient_track(observations, sampled_frames)
    if not observations.empty:
        observations["is_patient_track"] = observations["track_id"] == patient_track_id
    metadata = VideoMetadata(
        path=str(path),
        # CAP_PROP_FRAME_COUNT is frequently only a nominal container value for
        # these variable-duration MOV files.  All downstream dense timelines,
        # duration, and sampling coverage must use the number of frames that
        # were actually decoded, otherwise a synthetic trailing no-face segment
        # is appended to many clips.
        frame_count=decoded_frames,
        fps=fps,
        width=width,
        height=height,
        orientation_degrees=orientation,
        decoded_frames=decoded_frames,
        sampled_frames=sampled_frames,
        patient_track_id=patient_track_id,
        texture_max_dimension=texture_max_dimension,
    )
    return VisionResult(metadata=metadata, observations=observations)


def frame_windows(
    transitions: Iterable[int],
    frame_count: int,
    before: int = 10,
    after: int = 20,
) -> list[int]:
    """Return sorted unique dense-frame indices around annotated transitions."""

    indices: set[int] = set()
    for transition in transitions:
        start = max(0, int(transition) - before)
        stop = min(frame_count, int(transition) + after + 1)
        indices.update(range(start, stop))
    return sorted(indices)
