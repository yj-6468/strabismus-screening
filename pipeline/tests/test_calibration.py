from __future__ import annotations

import math
from pathlib import Path
import tempfile

import numpy as np
import pytest

from strabismus_screen.calibration import (
    CalibrationMode,
    CalibrationProfile,
    CalibrationRequiredError,
    CalibrationWarningCode,
    CameraIntrinsics,
    calibration_contract,
    create_camera_intrinsics_profile,
    estimate_from_camera_gaze_rays,
    estimate_from_patient_motion,
    fit_dataset_learned_profile,
    fit_known_target_profile,
    normalize_patient_horizontal_motion,
    patient_temporal_coordinate,
    patient_temporal_coordinate_with_uncertainty,
)
from strabismus_screen.prism import degrees_to_prism_diopters


def _known_target_profile() -> CalibrationProfile:
    return fit_known_target_profile(
        [-0.4, 0.0, 0.4],
        [-10.0, 0.0, 10.0],
        profile_id="device-a-right-v1",
        device_id="device-a",
        eye="OD",
        uncertainty_floor_degrees=0.1,
    )


def test_patient_temporal_coordinate_handles_laterality_mirroring_and_scale() -> None:
    # Feature 20% of an aperture width toward the anatomical lateral canthus.
    right_facing = patient_temporal_coordinate(24.0, 10.0, 30.0)
    mirrored = patient_temporal_coordinate(16.0, 30.0, 10.0)
    translated_and_scaled = patient_temporal_coordinate(128.0, 100.0, 140.0)
    assert right_facing == pytest.approx(0.2)
    assert mirrored == pytest.approx(0.2)
    assert translated_and_scaled == pytest.approx(0.2)


def test_patient_motion_is_temporal_positive_and_propagates_landmark_error() -> None:
    motion = normalize_patient_horizontal_motion(
        eye="right",
        before_feature_x_px=20.0,
        before_medial_canthus_x_px=10.0,
        before_lateral_canthus_x_px=30.0,
        after_feature_x_px=28.0,
        after_medial_canthus_x_px=10.0,
        after_lateral_canthus_x_px=30.0,
        feature_x_standard_deviation_px=0.5,
        canthus_standard_deviation_px=0.25,
    )
    assert motion.before.value == pytest.approx(0.0)
    assert motion.after.value == pytest.approx(0.4)
    assert motion.temporal_motion == pytest.approx(0.4)
    assert motion.standard_deviation > 0.0

    coordinate = patient_temporal_coordinate_with_uncertainty(
        20.0,
        10.0,
        30.0,
        feature_x_standard_deviation_px=1.0,
    )
    assert coordinate.standard_deviation == pytest.approx(0.05)


def test_known_target_profile_converts_angle_then_exact_prism() -> None:
    profile = _known_target_profile()
    motion = normalize_patient_horizontal_motion(
        eye="OD",
        before_feature_x_px=16.0,
        before_medial_canthus_x_px=10.0,
        before_lateral_canthus_x_px=30.0,
        after_feature_x_px=24.0,
        after_medial_canthus_x_px=10.0,
        after_lateral_canthus_x_px=30.0,
    )
    result = estimate_from_patient_motion(profile, motion)
    assert result.angle_degrees == pytest.approx(10.0)
    assert result.prism_diopters == pytest.approx(degrees_to_prism_diopters(10.0))
    assert result.direction == "eso"
    assert result.geometric_calibration is True
    assert result.prism_standard_deviation > 0.0
    assert result.prism_prediction_interval[0] < result.prism_diopters
    assert result.prism_prediction_interval[1] > result.prism_diopters
    assert result.warnings == ()


def test_known_target_extrapolation_is_explicit() -> None:
    profile = _known_target_profile()
    motion = normalize_patient_horizontal_motion(
        eye="right",
        before_feature_x_px=20.0,
        before_medial_canthus_x_px=10.0,
        before_lateral_canthus_x_px=30.0,
        after_feature_x_px=32.0,
        after_medial_canthus_x_px=10.0,
        after_lateral_canthus_x_px=30.0,
    )
    result = estimate_from_patient_motion(profile, motion)
    assert [warning.code for warning in result.warnings] == [
        CalibrationWarningCode.OUTSIDE_CALIBRATED_DOMAIN
    ]


def test_dataset_fallback_requires_opt_in_and_always_warns() -> None:
    angles = np.array([-10.0, 0.0, 10.0])
    profile = fit_dataset_learned_profile(
        [-0.4, 0.0, 0.4],
        degrees_to_prism_diopters(angles),
        profile_id="legacy-dataset-v1",
        device_id="mixed-legacy",
        uncertainty_floor_degrees=0.2,
    )
    motion = normalize_patient_horizontal_motion(
        eye="left",
        before_feature_x_px=20.0,
        before_medial_canthus_x_px=30.0,
        before_lateral_canthus_x_px=10.0,
        after_feature_x_px=16.0,
        after_medial_canthus_x_px=30.0,
        after_lateral_canthus_x_px=10.0,
    )
    with pytest.raises(CalibrationRequiredError) as error:
        estimate_from_patient_motion(profile, motion)
    assert error.value.warning.code is CalibrationWarningCode.DATASET_LEARNED_FALLBACK

    result = estimate_from_patient_motion(
        profile, motion, allow_dataset_fallback=True
    )
    assert result.calibration_mode is CalibrationMode.DATASET_LEARNED
    assert result.geometric_calibration is False
    assert result.angle_degrees == pytest.approx(5.0)
    assert [warning.code for warning in result.warnings] == [
        CalibrationWarningCode.DATASET_LEARNED_FALLBACK
    ]


def test_missing_profile_never_converts_motion_or_pixels() -> None:
    contract = calibration_contract(None)
    assert contract.available is False
    assert contract.geometric is False
    assert contract.warnings[0].code is CalibrationWarningCode.NO_PER_DEVICE_CALIBRATION

    motion = normalize_patient_horizontal_motion(
        eye="right",
        before_feature_x_px=20.0,
        before_medial_canthus_x_px=10.0,
        before_lateral_canthus_x_px=30.0,
        after_feature_x_px=22.0,
        after_medial_canthus_x_px=10.0,
        after_lateral_canthus_x_px=30.0,
    )
    with pytest.raises(CalibrationRequiredError) as error:
        estimate_from_patient_motion(None, motion)
    assert error.value.warning.code is CalibrationWarningCode.NO_PER_DEVICE_CALIBRATION


def test_camera_intrinsics_path_requires_gaze_ray_pixels() -> None:
    intrinsics = CameraIntrinsics(
        fx_px=1000.0,
        fy_px=1000.0,
        cx_px=500.0,
        cy_px=300.0,
        width_px=1000,
        height_px=600,
    )
    profile = create_camera_intrinsics_profile(
        intrinsics,
        profile_id="camera-a-left-v1",
        device_id="camera-a",
        eye="left",
        patient_temporal_image_x_sign=1,
        residual_std_degrees=0.1,
    )
    after_x = 500.0 + 1000.0 * math.tan(math.radians(10.0))
    result = estimate_from_camera_gaze_rays(
        profile,
        before_undistorted_gaze_x_px=500.0,
        after_undistorted_gaze_x_px=after_x,
        before_pixel_standard_deviation=0.5,
        after_pixel_standard_deviation=0.5,
    )
    assert result.angle_degrees == pytest.approx(10.0)
    assert result.prism_diopters == pytest.approx(degrees_to_prism_diopters(10.0))

    motion = normalize_patient_horizontal_motion(
        eye="left",
        before_feature_x_px=20.0,
        before_medial_canthus_x_px=30.0,
        before_lateral_canthus_x_px=10.0,
        after_feature_x_px=18.0,
        after_medial_canthus_x_px=30.0,
        after_lateral_canthus_x_px=10.0,
    )
    with pytest.raises(ValueError, match="gaze-ray pixels"):
        estimate_from_patient_motion(profile, motion)


def test_profile_json_round_trip_and_file_serialization() -> None:
    profile = _known_target_profile()
    restored = CalibrationProfile.from_json(profile.to_json())
    assert restored == profile

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "calibration.json"
        profile.save(path)
        assert CalibrationProfile.load(path) == profile
