"""Physical calibration for horizontal ocular motion and prism diopters.

This module intentionally has no ``pixels_to_prism_diopters`` function.  A raw
pixel displacement is not an ocular angle.  Valid paths are:

* patient-centric eye coordinates calibrated against known target angles;
* undistorted gaze-ray pixels interpreted through camera intrinsics; or
* an explicitly opted-in dataset-learned mapping of normalized motion.

Every path first estimates a signed ocular angle (temporal/eso positive), then
uses the exact prism definition ``delta = 100 * tan(theta)`` from ``prism.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from .prism import (
    angle_normal_prediction_interval_in_prism_diopters,
    degrees_to_prism_diopters,
    prism_diopters_to_degrees,
    prism_uncertainty_from_angle,
)


CALIBRATION_SCHEMA_VERSION = 1


class CalibrationMode(StrEnum):
    """Supported sources of angular calibration."""

    KNOWN_TARGET = "known_target"
    CAMERA_INTRINSICS = "camera_intrinsics"
    DATASET_LEARNED = "dataset_learned"


class CalibrationInputSpace(StrEnum):
    """Feature space a calibration profile is allowed to consume."""

    PATIENT_TEMPORAL_COORDINATE = "patient_temporal_coordinate"
    PATIENT_TEMPORAL_MOTION = "patient_temporal_motion"
    UNDISTORTED_GAZE_RAY_PIXEL = "undistorted_gaze_ray_pixel"


class CalibrationWarningCode(StrEnum):
    NO_PER_DEVICE_CALIBRATION = "no_per_device_calibration"
    DATASET_LEARNED_FALLBACK = "dataset_learned_fallback"
    OUTSIDE_CALIBRATED_DOMAIN = "outside_calibrated_domain"


@dataclass(frozen=True, slots=True)
class CalibrationWarning:
    code: CalibrationWarningCode
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code.value, "message": self.message}


class CalibrationRequiredError(RuntimeError):
    """Raised instead of silently interpreting uncalibrated pixels as PD."""

    def __init__(self, warning: CalibrationWarning):
        self.warning = warning
        super().__init__(f"{warning.code.value}: {warning.message}")


@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    """Pinhole intrinsics for already-undistorted image coordinates."""

    fx_px: float
    fy_px: float
    cx_px: float
    cy_px: float
    width_px: int
    height_px: int
    distortion_coefficients: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        numeric = (self.fx_px, self.fy_px, self.cx_px, self.cy_px)
        if not all(math.isfinite(float(value)) for value in numeric):
            raise ValueError("camera intrinsic values must be finite")
        if self.fx_px <= 0.0 or self.fy_px <= 0.0:
            raise ValueError("camera focal lengths must be greater than zero")
        if self.width_px <= 0 or self.height_px <= 0:
            raise ValueError("camera image dimensions must be positive")
        if not all(math.isfinite(float(value)) for value in self.distortion_coefficients):
            raise ValueError("distortion coefficients must be finite")

    def horizontal_ray_angle_degrees(self, undistorted_x_px: float) -> float:
        """Angle of an undistorted camera ray relative to the optical axis."""

        x = _finite_float(undistorted_x_px, "undistorted_x_px")
        return math.degrees(math.atan2(x - self.cx_px, self.fx_px))

    def horizontal_ray_angle_sd_degrees(
        self, undistorted_x_px: float, pixel_standard_deviation: float
    ) -> float:
        """First-order propagation from horizontal pixel SD to ray-angle SD."""

        x = _finite_float(undistorted_x_px, "undistorted_x_px")
        sigma = _nonnegative_float(pixel_standard_deviation, "pixel_standard_deviation")
        derivative = (
            self.fx_px
            / (self.fx_px**2 + (x - self.cx_px) ** 2)
            * 180.0
            / math.pi
        )
        return abs(derivative) * sigma

    def to_dict(self) -> dict[str, Any]:
        return {
            "fx_px": self.fx_px,
            "fy_px": self.fy_px,
            "cx_px": self.cx_px,
            "cy_px": self.cy_px,
            "width_px": self.width_px,
            "height_px": self.height_px,
            "distortion_coefficients": list(self.distortion_coefficients),
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "CameraIntrinsics":
        return cls(
            fx_px=float(values["fx_px"]),
            fy_px=float(values["fy_px"]),
            cx_px=float(values["cx_px"]),
            cy_px=float(values["cy_px"]),
            width_px=int(values["width_px"]),
            height_px=int(values["height_px"]),
            distortion_coefficients=tuple(
                float(value) for value in values.get("distortion_coefficients", ())
            ),
        )


@dataclass(frozen=True, slots=True)
class NormalizedEyeCoordinate:
    """Patient-temporal coordinate normalized by the eye-aperture width."""

    value: float
    standard_deviation: float = 0.0

    def __post_init__(self) -> None:
        _finite_float(self.value, "value")
        _nonnegative_float(self.standard_deviation, "standard_deviation")


@dataclass(frozen=True, slots=True)
class PatientCentricMotion:
    """Head-translation/scale-normalized horizontal ocular motion."""

    eye: str
    before: NormalizedEyeCoordinate
    after: NormalizedEyeCoordinate

    def __post_init__(self) -> None:
        eye = _normalize_eye(self.eye)
        object.__setattr__(self, "eye", eye)

    @property
    def temporal_motion(self) -> float:
        """Positive is temporal refixation (therefore positive/eso internally)."""

        return self.after.value - self.before.value

    @property
    def standard_deviation(self) -> float:
        return math.hypot(
            self.before.standard_deviation,
            self.after.standard_deviation,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "eye": self.eye,
            "before_coordinate": self.before.value,
            "after_coordinate": self.after.value,
            "temporal_motion": self.temporal_motion,
            "standard_deviation": self.standard_deviation,
        }


def patient_temporal_coordinate(
    feature_x_px: float,
    medial_canthus_x_px: float,
    lateral_canthus_x_px: float,
    *,
    minimum_eye_width_px: float = 4.0,
) -> float:
    """Normalize an eye feature so positive always points toward the temple.

    ``medial_canthus_x_px`` and ``lateral_canthus_x_px`` are anatomical labels,
    not screen-left/screen-right labels.  Their signed separation automatically
    handles eye laterality, mirroring, camera rotation, head translation, and
    uniform image scale.  The result is dimensionless.
    """

    x = _finite_float(feature_x_px, "feature_x_px")
    medial = _finite_float(medial_canthus_x_px, "medial_canthus_x_px")
    lateral = _finite_float(lateral_canthus_x_px, "lateral_canthus_x_px")
    minimum_width = _positive_float(minimum_eye_width_px, "minimum_eye_width_px")
    signed_width = lateral - medial
    if abs(signed_width) < minimum_width:
        raise ValueError("eye-aperture width is too small for stable normalization")
    midpoint = (medial + lateral) / 2.0
    return (x - midpoint) / signed_width


def patient_temporal_coordinate_with_uncertainty(
    feature_x_px: float,
    medial_canthus_x_px: float,
    lateral_canthus_x_px: float,
    *,
    feature_x_standard_deviation_px: float = 0.0,
    medial_canthus_standard_deviation_px: float = 0.0,
    lateral_canthus_standard_deviation_px: float = 0.0,
    minimum_eye_width_px: float = 4.0,
) -> NormalizedEyeCoordinate:
    """Patient-temporal coordinate with first-order landmark uncertainty."""

    value = patient_temporal_coordinate(
        feature_x_px,
        medial_canthus_x_px,
        lateral_canthus_x_px,
        minimum_eye_width_px=minimum_eye_width_px,
    )
    medial = float(medial_canthus_x_px)
    lateral = float(lateral_canthus_x_px)
    signed_width = lateral - medial
    sigma_x = _nonnegative_float(
        feature_x_standard_deviation_px,
        "feature_x_standard_deviation_px",
    )
    sigma_medial = _nonnegative_float(
        medial_canthus_standard_deviation_px,
        "medial_canthus_standard_deviation_px",
    )
    sigma_lateral = _nonnegative_float(
        lateral_canthus_standard_deviation_px,
        "lateral_canthus_standard_deviation_px",
    )
    derivative_x = 1.0 / signed_width
    derivative_medial = (value - 0.5) / signed_width
    derivative_lateral = -(value + 0.5) / signed_width
    variance = (
        (derivative_x * sigma_x) ** 2
        + (derivative_medial * sigma_medial) ** 2
        + (derivative_lateral * sigma_lateral) ** 2
    )
    return NormalizedEyeCoordinate(value=value, standard_deviation=math.sqrt(variance))


def normalize_patient_horizontal_motion(
    *,
    eye: str,
    before_feature_x_px: float,
    before_medial_canthus_x_px: float,
    before_lateral_canthus_x_px: float,
    after_feature_x_px: float,
    after_medial_canthus_x_px: float,
    after_lateral_canthus_x_px: float,
    feature_x_standard_deviation_px: float = 0.0,
    canthus_standard_deviation_px: float = 0.0,
    minimum_eye_width_px: float = 4.0,
) -> PatientCentricMotion:
    """Create a patient-centric motion record from two eye-feature frames."""

    coordinate_kwargs = {
        "feature_x_standard_deviation_px": feature_x_standard_deviation_px,
        "medial_canthus_standard_deviation_px": canthus_standard_deviation_px,
        "lateral_canthus_standard_deviation_px": canthus_standard_deviation_px,
        "minimum_eye_width_px": minimum_eye_width_px,
    }
    before = patient_temporal_coordinate_with_uncertainty(
        before_feature_x_px,
        before_medial_canthus_x_px,
        before_lateral_canthus_x_px,
        **coordinate_kwargs,
    )
    after = patient_temporal_coordinate_with_uncertainty(
        after_feature_x_px,
        after_medial_canthus_x_px,
        after_lateral_canthus_x_px,
        **coordinate_kwargs,
    )
    return PatientCentricMotion(eye=eye, before=before, after=after)


@dataclass(frozen=True, slots=True)
class CalibrationProfile:
    """Serializable contract mapping a declared feature space to ocular angle."""

    profile_id: str
    device_id: str
    eye: str
    mode: CalibrationMode
    input_space: CalibrationInputSpace
    residual_std_degrees: float
    coefficients: tuple[float, ...] = ()
    coefficient_covariance: tuple[tuple[float, ...], ...] = ()
    valid_input_range: tuple[float, float] | None = None
    camera_intrinsics: CameraIntrinsics | None = None
    temporal_axis_sign: int = 1
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = CALIBRATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.profile_id.strip() or not self.device_id.strip():
            raise ValueError("profile_id and device_id must be non-empty")
        object.__setattr__(self, "eye", _normalize_eye(self.eye))
        object.__setattr__(self, "mode", CalibrationMode(self.mode))
        object.__setattr__(self, "input_space", CalibrationInputSpace(self.input_space))
        _nonnegative_float(self.residual_std_degrees, "residual_std_degrees")
        if self.schema_version != CALIBRATION_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported calibration schema version: {self.schema_version}"
            )
        if self.temporal_axis_sign not in (-1, 1):
            raise ValueError("temporal_axis_sign must be -1 or +1")

        coefficients = tuple(float(value) for value in self.coefficients)
        covariance = tuple(
            tuple(float(value) for value in row)
            for row in self.coefficient_covariance
        )
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "coefficient_covariance", covariance)
        if covariance and (
            len(covariance) != len(coefficients)
            or any(len(row) != len(coefficients) for row in covariance)
        ):
            raise ValueError("coefficient covariance must be square and match coefficients")
        if self.valid_input_range is not None:
            lower, upper = (float(value) for value in self.valid_input_range)
            if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
                raise ValueError("valid_input_range must contain increasing finite bounds")
            object.__setattr__(self, "valid_input_range", (lower, upper))

        mapping_modes = {
            CalibrationMode.KNOWN_TARGET,
            CalibrationMode.DATASET_LEARNED,
        }
        if self.mode in mapping_modes and not coefficients:
            raise ValueError("polynomial calibration profiles require coefficients")
        if self.mode is CalibrationMode.KNOWN_TARGET:
            if self.input_space is not CalibrationInputSpace.PATIENT_TEMPORAL_COORDINATE:
                raise ValueError("known-target profiles require patient coordinates")
            if self.camera_intrinsics is not None:
                raise ValueError("known-target profiles must not include camera intrinsics")
        elif self.mode is CalibrationMode.DATASET_LEARNED:
            if self.input_space is not CalibrationInputSpace.PATIENT_TEMPORAL_MOTION:
                raise ValueError("dataset profiles require patient-normalized motion")
            if self.camera_intrinsics is not None:
                raise ValueError("dataset profiles must not include camera intrinsics")
        elif self.mode is CalibrationMode.CAMERA_INTRINSICS:
            if self.input_space is not CalibrationInputSpace.UNDISTORTED_GAZE_RAY_PIXEL:
                raise ValueError("intrinsic profiles require undistorted gaze-ray pixels")
            if self.camera_intrinsics is None:
                raise ValueError("intrinsic profiles require CameraIntrinsics")
            if coefficients:
                raise ValueError("intrinsic profiles must not contain polynomial coefficients")
        json.dumps(dict(self.metadata), sort_keys=True, allow_nan=False)

    @property
    def is_geometric(self) -> bool:
        return self.mode is not CalibrationMode.DATASET_LEARNED

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "device_id": self.device_id,
            "eye": self.eye,
            "mode": self.mode.value,
            "input_space": self.input_space.value,
            "residual_std_degrees": self.residual_std_degrees,
            "coefficients": list(self.coefficients),
            "coefficient_covariance": [
                list(row) for row in self.coefficient_covariance
            ],
            "valid_input_range": (
                list(self.valid_input_range)
                if self.valid_input_range is not None
                else None
            ),
            "camera_intrinsics": (
                self.camera_intrinsics.to_dict()
                if self.camera_intrinsics is not None
                else None
            ),
            "temporal_axis_sign": self.temporal_axis_sign,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "CalibrationProfile":
        intrinsics = values.get("camera_intrinsics")
        valid_range = values.get("valid_input_range")
        return cls(
            schema_version=int(values.get("schema_version", 0)),
            profile_id=str(values["profile_id"]),
            device_id=str(values["device_id"]),
            eye=str(values["eye"]),
            mode=CalibrationMode(values["mode"]),
            input_space=CalibrationInputSpace(values["input_space"]),
            residual_std_degrees=float(values["residual_std_degrees"]),
            coefficients=tuple(float(value) for value in values.get("coefficients", ())),
            coefficient_covariance=tuple(
                tuple(float(value) for value in row)
                for row in values.get("coefficient_covariance", ())
            ),
            valid_input_range=(
                tuple(float(value) for value in valid_range)
                if valid_range is not None
                else None
            ),
            camera_intrinsics=(
                CameraIntrinsics.from_dict(intrinsics)
                if intrinsics is not None
                else None
            ),
            temporal_axis_sign=int(values.get("temporal_axis_sign", 1)),
            metadata=dict(values.get("metadata", {})),
        )

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(
            self.to_dict(),
            indent=indent,
            sort_keys=True,
            allow_nan=False,
        )

    @classmethod
    def from_json(cls, payload: str) -> "CalibrationProfile":
        values = json.loads(payload)
        if not isinstance(values, dict):
            raise ValueError("calibration JSON must contain an object")
        return cls.from_dict(values)

    def save(self, path: str | os.PathLike[str]) -> None:
        """Atomically save this profile as human-auditable JSON."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_path = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(self.to_json())
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, destination)
        except Exception:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass
            raise

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "CalibrationProfile":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


@dataclass(frozen=True, slots=True)
class CalibrationContract:
    available: bool
    geometric: bool
    mode: CalibrationMode | None
    warnings: tuple[CalibrationWarning, ...]


@dataclass(frozen=True, slots=True)
class CalibratedDeviation:
    """Signed calibrated angle, exact PD conversion, and uncertainty."""

    profile_id: str
    calibration_mode: CalibrationMode
    geometric_calibration: bool
    angle_degrees: float
    angle_standard_deviation_degrees: float
    prism_diopters: float
    prism_standard_deviation: float
    prism_prediction_interval: tuple[float, float]
    coverage: float
    warnings: tuple[CalibrationWarning, ...] = ()

    @property
    def direction(self) -> str:
        if self.prism_diopters > 0.0:
            return "eso"
        if self.prism_diopters < 0.0:
            return "exo"
        return "normal"

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "calibration_mode": self.calibration_mode.value,
            "geometric_calibration": self.geometric_calibration,
            "angle_degrees": self.angle_degrees,
            "angle_standard_deviation_degrees": (
                self.angle_standard_deviation_degrees
            ),
            "prism_diopters": self.prism_diopters,
            "prism_standard_deviation": self.prism_standard_deviation,
            "prism_prediction_interval": list(self.prism_prediction_interval),
            "coverage": self.coverage,
            "direction": self.direction,
            "warnings": [warning.to_dict() for warning in self.warnings],
        }


def calibration_contract(
    profile: CalibrationProfile | None,
) -> CalibrationContract:
    """Expose whether calibrated PD is available without performing conversion."""

    if profile is None:
        warning = CalibrationWarning(
            CalibrationWarningCode.NO_PER_DEVICE_CALIBRATION,
            "No calibration profile is available; raw pixels must not be reported as PD.",
        )
        return CalibrationContract(False, False, None, (warning,))
    if profile.mode is CalibrationMode.DATASET_LEARNED:
        warning = _dataset_fallback_warning()
        return CalibrationContract(True, False, profile.mode, (warning,))
    return CalibrationContract(True, True, profile.mode, ())


def fit_known_target_profile(
    normalized_coordinates: Sequence[float] | np.ndarray,
    target_temporal_angles_degrees: Sequence[float] | np.ndarray,
    *,
    profile_id: str,
    device_id: str,
    eye: str,
    polynomial_degree: int = 1,
    uncertainty_floor_degrees: float = 0.25,
    metadata: Mapping[str, Any] | None = None,
) -> CalibrationProfile:
    """Fit patient-coordinate to angle using known fixation targets.

    Target angles must be patient-temporal positive for the calibrated eye.
    At least ``polynomial_degree + 2`` samples are required so residual
    uncertainty can be estimated.
    """

    coordinates = np.asarray(normalized_coordinates, dtype=float)
    angles = np.asarray(target_temporal_angles_degrees, dtype=float)
    if np.any(np.abs(angles) >= 90.0):
        raise ValueError("target angles must lie strictly between -90 and 90")
    coefficients, covariance, residual_std, valid_range = _fit_monotonic_polynomial(
        coordinates,
        angles,
        degree=polynomial_degree,
        uncertainty_floor=uncertainty_floor_degrees,
    )
    profile_metadata = dict(metadata or {})
    profile_metadata.update(
        {
            "calibration_samples": int(coordinates.size),
            "polynomial_degree": polynomial_degree,
            "target_angle_convention": "patient_temporal_positive",
        }
    )
    return CalibrationProfile(
        profile_id=profile_id,
        device_id=device_id,
        eye=eye,
        mode=CalibrationMode.KNOWN_TARGET,
        input_space=CalibrationInputSpace.PATIENT_TEMPORAL_COORDINATE,
        residual_std_degrees=residual_std,
        coefficients=coefficients,
        coefficient_covariance=covariance,
        valid_input_range=valid_range,
        metadata=profile_metadata,
    )


def fit_dataset_learned_profile(
    normalized_temporal_motion: Sequence[float] | np.ndarray,
    signed_prism_diopters: Sequence[float] | np.ndarray,
    *,
    profile_id: str,
    device_id: str,
    eye: str = "both",
    polynomial_degree: int = 1,
    uncertainty_floor_degrees: float = 0.5,
    metadata: Mapping[str, Any] | None = None,
) -> CalibrationProfile:
    """Fit an explicit non-geometric fallback from normalized motion.

    PD labels are first converted to angles using the exact inverse tangent;
    the learned model never fits a raw-pixel-to-PD constant.
    """

    motion = np.asarray(normalized_temporal_motion, dtype=float)
    prism = np.asarray(signed_prism_diopters, dtype=float)
    angles = np.asarray(prism_diopters_to_degrees(prism), dtype=float)
    coefficients, covariance, residual_std, valid_range = _fit_monotonic_polynomial(
        motion,
        angles,
        degree=polynomial_degree,
        uncertainty_floor=uncertainty_floor_degrees,
    )
    profile_metadata = dict(metadata or {})
    profile_metadata.update(
        {
            "calibration_samples": int(motion.size),
            "polynomial_degree": polynomial_degree,
            "fallback": True,
            "sign_convention": "eso_positive_exo_negative",
        }
    )
    return CalibrationProfile(
        profile_id=profile_id,
        device_id=device_id,
        eye=eye,
        mode=CalibrationMode.DATASET_LEARNED,
        input_space=CalibrationInputSpace.PATIENT_TEMPORAL_MOTION,
        residual_std_degrees=residual_std,
        coefficients=coefficients,
        coefficient_covariance=covariance,
        valid_input_range=valid_range,
        metadata=profile_metadata,
    )


def create_camera_intrinsics_profile(
    intrinsics: CameraIntrinsics,
    *,
    profile_id: str,
    device_id: str,
    eye: str,
    patient_temporal_image_x_sign: int,
    residual_std_degrees: float = 0.25,
    metadata: Mapping[str, Any] | None = None,
) -> CalibrationProfile:
    """Create a profile for *gaze-ray* pixels, never raw iris displacement.

    ``patient_temporal_image_x_sign`` is +1 when increasing image x is temporal
    for the calibrated eye and -1 otherwise.
    """

    return CalibrationProfile(
        profile_id=profile_id,
        device_id=device_id,
        eye=eye,
        mode=CalibrationMode.CAMERA_INTRINSICS,
        input_space=CalibrationInputSpace.UNDISTORTED_GAZE_RAY_PIXEL,
        residual_std_degrees=residual_std_degrees,
        camera_intrinsics=intrinsics,
        temporal_axis_sign=patient_temporal_image_x_sign,
        metadata=dict(metadata or {}),
    )


def estimate_from_patient_motion(
    profile: CalibrationProfile | None,
    motion: PatientCentricMotion,
    *,
    coverage: float = 0.95,
    allow_dataset_fallback: bool = False,
) -> CalibratedDeviation:
    """Estimate signed deviation from normalized patient-centric motion."""

    profile = _require_profile(profile, allow_dataset_fallback)
    _require_eye_match(profile, motion.eye)
    warnings = list(calibration_contract(profile).warnings)

    if profile.mode is CalibrationMode.KNOWN_TARGET:
        before = motion.before.value
        after = motion.after.value
        theta = _polynomial_value(profile, after) - _polynomial_value(profile, before)
        measurement_variance = (
            _polynomial_derivative(profile, before)
            * motion.before.standard_deviation
        ) ** 2 + (
            _polynomial_derivative(profile, after)
            * motion.after.standard_deviation
        ) ** 2
        basis_difference = _polynomial_basis(profile, after) - _polynomial_basis(
            profile, before
        )
        parameter_variance = _parameter_variance(profile, basis_difference)
        # Two absolute coordinate estimates contribute to the angular difference.
        residual_variance = 2.0 * profile.residual_std_degrees**2
        checked_inputs = (before, after)
    elif profile.mode is CalibrationMode.DATASET_LEARNED:
        value = motion.temporal_motion
        theta = _polynomial_value(profile, value)
        measurement_variance = (
            _polynomial_derivative(profile, value) * motion.standard_deviation
        ) ** 2
        parameter_variance = _parameter_variance(
            profile, _polynomial_basis(profile, value)
        )
        residual_variance = profile.residual_std_degrees**2
        checked_inputs = (value,)
    else:
        raise ValueError(
            "camera-intrinsic profiles require estimate_from_camera_gaze_rays; "
            "raw eye-feature pixels are not gaze-ray pixels"
        )

    if _outside_profile_domain(profile, checked_inputs):
        warnings.append(_extrapolation_warning())
    sigma_theta = math.sqrt(
        max(0.0, measurement_variance + parameter_variance + residual_variance)
    )
    return _build_result(profile, theta, sigma_theta, coverage, warnings)


def estimate_from_camera_gaze_rays(
    profile: CalibrationProfile | None,
    *,
    before_undistorted_gaze_x_px: float,
    after_undistorted_gaze_x_px: float,
    before_pixel_standard_deviation: float = 0.0,
    after_pixel_standard_deviation: float = 0.0,
    coverage: float = 0.95,
) -> CalibratedDeviation:
    """Estimate deviation from calibrated, undistorted gaze-ray coordinates.

    The inputs must be gaze-ray intersections produced by a calibrated gaze
    model.  Passing raw pupil or iris centers is a category error and is not a
    supported conversion path.
    """

    profile = _require_profile(profile, allow_dataset_fallback=False)
    if profile.mode is not CalibrationMode.CAMERA_INTRINSICS:
        raise ValueError("profile is not a camera-intrinsics gaze-ray profile")
    intrinsics = profile.camera_intrinsics
    if intrinsics is None:  # Guard for static type checkers and corrupt objects.
        raise ValueError("camera intrinsics are missing")
    before_x = _finite_float(
        before_undistorted_gaze_x_px,
        "before_undistorted_gaze_x_px",
    )
    after_x = _finite_float(
        after_undistorted_gaze_x_px,
        "after_undistorted_gaze_x_px",
    )
    theta_camera = intrinsics.horizontal_ray_angle_degrees(
        after_x
    ) - intrinsics.horizontal_ray_angle_degrees(before_x)
    theta = profile.temporal_axis_sign * theta_camera
    before_sigma = intrinsics.horizontal_ray_angle_sd_degrees(
        before_x, before_pixel_standard_deviation
    )
    after_sigma = intrinsics.horizontal_ray_angle_sd_degrees(
        after_x, after_pixel_standard_deviation
    )
    sigma_theta = math.sqrt(
        before_sigma**2
        + after_sigma**2
        + profile.residual_std_degrees**2
    )
    warnings: list[CalibrationWarning] = []
    if not (0.0 <= before_x < intrinsics.width_px) or not (
        0.0 <= after_x < intrinsics.width_px
    ):
        warnings.append(_extrapolation_warning())
    return _build_result(profile, theta, sigma_theta, coverage, warnings)


def _build_result(
    profile: CalibrationProfile,
    angle_degrees: float,
    angle_standard_deviation: float,
    coverage: float,
    warnings: Sequence[CalibrationWarning],
) -> CalibratedDeviation:
    prism = float(degrees_to_prism_diopters(angle_degrees))
    prism_sigma = float(
        prism_uncertainty_from_angle(angle_degrees, angle_standard_deviation)
    )
    lower, upper = angle_normal_prediction_interval_in_prism_diopters(
        angle_degrees,
        angle_standard_deviation,
        coverage=coverage,
    )
    return CalibratedDeviation(
        profile_id=profile.profile_id,
        calibration_mode=profile.mode,
        geometric_calibration=profile.is_geometric,
        angle_degrees=float(angle_degrees),
        angle_standard_deviation_degrees=float(angle_standard_deviation),
        prism_diopters=prism,
        prism_standard_deviation=prism_sigma,
        prism_prediction_interval=(float(lower), float(upper)),
        coverage=float(coverage),
        warnings=tuple(warnings),
    )


def _fit_monotonic_polynomial(
    x_values: np.ndarray,
    angle_values: np.ndarray,
    *,
    degree: int,
    uncertainty_floor: float,
) -> tuple[
    tuple[float, ...],
    tuple[tuple[float, ...], ...],
    float,
    tuple[float, float],
]:
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(angle_values, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError("calibration inputs must be equal-length one-dimensional arrays")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("calibration inputs must be finite")
    if not isinstance(degree, int) or not 1 <= degree <= 3:
        raise ValueError("polynomial_degree must be an integer from 1 to 3")
    if x.size < degree + 2:
        raise ValueError(
            f"at least {degree + 2} calibration samples are required for degree {degree}"
        )
    floor = _nonnegative_float(uncertainty_floor, "uncertainty_floor")
    lower = float(np.min(x))
    upper = float(np.max(x))
    if lower == upper:
        raise ValueError("calibration inputs must span more than one value")

    design = np.vander(x, N=degree + 1, increasing=False)
    coefficients, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    if rank != degree + 1:
        raise ValueError("calibration design is rank deficient")
    residuals = y - design @ coefficients
    degrees_of_freedom = x.size - (degree + 1)
    residual_variance = float(np.sum(residuals**2) / degrees_of_freedom)
    variance = max(residual_variance, floor**2)
    covariance = variance * np.linalg.pinv(design.T @ design)

    derivative = np.polyder(coefficients)
    grid = np.linspace(lower, upper, 101)
    if np.any(np.polyval(derivative, grid) <= 0.0):
        raise ValueError(
            "calibration must be strictly increasing in patient-temporal coordinates"
        )
    return (
        tuple(float(value) for value in coefficients),
        tuple(tuple(float(value) for value in row) for row in covariance),
        math.sqrt(variance),
        (lower, upper),
    )


def _polynomial_value(profile: CalibrationProfile, value: float) -> float:
    return float(np.polyval(np.asarray(profile.coefficients), value))


def _polynomial_derivative(profile: CalibrationProfile, value: float) -> float:
    derivative = np.polyder(np.asarray(profile.coefficients))
    return float(np.polyval(derivative, value))


def _polynomial_basis(profile: CalibrationProfile, value: float) -> np.ndarray:
    return np.asarray(
        [value**power for power in range(len(profile.coefficients) - 1, -1, -1)],
        dtype=float,
    )


def _parameter_variance(profile: CalibrationProfile, basis: np.ndarray) -> float:
    if not profile.coefficient_covariance:
        return 0.0
    covariance = np.asarray(profile.coefficient_covariance, dtype=float)
    return max(0.0, float(basis @ covariance @ basis))


def _outside_profile_domain(
    profile: CalibrationProfile, values: Sequence[float]
) -> bool:
    if profile.valid_input_range is None:
        return False
    lower, upper = profile.valid_input_range
    return any(value < lower or value > upper for value in values)


def _require_profile(
    profile: CalibrationProfile | None,
    allow_dataset_fallback: bool,
) -> CalibrationProfile:
    contract = calibration_contract(profile)
    if profile is None:
        raise CalibrationRequiredError(contract.warnings[0])
    if profile.mode is CalibrationMode.DATASET_LEARNED and not allow_dataset_fallback:
        raise CalibrationRequiredError(contract.warnings[0])
    return profile


def _require_eye_match(profile: CalibrationProfile, eye: str) -> None:
    normalized = _normalize_eye(eye)
    if profile.eye != "both" and profile.eye != normalized:
        raise ValueError(
            f"calibration profile is for {profile.eye} eye, not {normalized} eye"
        )


def _dataset_fallback_warning() -> CalibrationWarning:
    return CalibrationWarning(
        CalibrationWarningCode.DATASET_LEARNED_FALLBACK,
        "Using a dataset-learned normalized-motion calibration, not a physical "
        "per-device calibration; explicit fallback opt-in is required.",
    )


def _extrapolation_warning() -> CalibrationWarning:
    return CalibrationWarning(
        CalibrationWarningCode.OUTSIDE_CALIBRATED_DOMAIN,
        "Input lies outside the calibration domain; uncertainty may be understated.",
    )


def _normalize_eye(value: str) -> str:
    token = str(value).strip().lower()
    aliases = {"od": "right", "os": "left", "right": "right", "left": "left"}
    if token == "both":
        return token
    if token not in aliases:
        raise ValueError("eye must be right/OD, left/OS, or both")
    return aliases[token]


def _finite_float(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _nonnegative_float(value: float, name: str) -> float:
    result = _finite_float(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _positive_float(value: float, name: str) -> float:
    result = _finite_float(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be greater than zero")
    return result


__all__ = [
    "CALIBRATION_SCHEMA_VERSION",
    "CalibratedDeviation",
    "CalibrationContract",
    "CalibrationInputSpace",
    "CalibrationMode",
    "CalibrationProfile",
    "CalibrationRequiredError",
    "CalibrationWarning",
    "CalibrationWarningCode",
    "CameraIntrinsics",
    "NormalizedEyeCoordinate",
    "PatientCentricMotion",
    "calibration_contract",
    "create_camera_intrinsics_profile",
    "estimate_from_camera_gaze_rays",
    "estimate_from_patient_motion",
    "fit_dataset_learned_profile",
    "fit_known_target_profile",
    "normalize_patient_horizontal_motion",
    "patient_temporal_coordinate",
    "patient_temporal_coordinate_with_uncertainty",
]
