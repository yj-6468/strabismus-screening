from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strabismus_screen.config import EventConfig
from strabismus_screen.events import CoverEventModel, infer_cover_timeline
from strabismus_screen.features import (
    aggregate_video_features,
    extract_motion_cycles,
    trusted_timeline,
)
from strabismus_screen.types import Eye, TestType as CoverTestType
from strabismus_screen.vision import (
    EYE_GEOMETRY,
    FaceLandmarkerBackend,
    _eye_features,
    frame_windows,
)


def _synthetic_eye_points(eye: Eye, *, iris_x: float, eye_mid_x: float) -> np.ndarray:
    points = np.zeros((478, 3), dtype=np.float32)
    definition = EYE_GEOMETRY[eye]
    for index in definition["contour"]:
        points[index, :2] = (eye_mid_x, 50.0)
    corner_a, corner_b = definition["corners"]
    points[corner_a, :2] = (eye_mid_x - 10.0, 50.0)
    points[corner_b, :2] = (eye_mid_x + 10.0, 50.0)
    lid_a, lid_b = definition["lids"]
    points[lid_a, :2] = (eye_mid_x, 47.0)
    points[lid_b, :2] = (eye_mid_x, 53.0)
    for index, offset in zip(definition["iris"], (-2, -1, 0, 1, 2), strict=True):
        points[index, :2] = (iris_x + offset, 50.0)
    points[definition["cheek"], :2] = (eye_mid_x, 70.0)
    return points


def test_patient_centric_nasal_coordinate_has_same_anatomical_sign() -> None:
    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    right = _eye_features(
        frame,
        _synthetic_eye_points(Eye.RIGHT, iris_x=34.0, eye_mid_x=30.0),
        Eye.RIGHT,
        nose_x=60.0,
    )
    left = _eye_features(
        frame,
        _synthetic_eye_points(Eye.LEFT, iris_x=86.0, eye_mid_x=90.0),
        Eye.LEFT,
        nose_x=60.0,
    )
    assert right["nasal_u"] == pytest.approx(0.2)
    assert left["nasal_u"] == pytest.approx(0.2)


def test_frame_windows_are_clipped_and_deduplicated() -> None:
    assert frame_windows([1, 4], frame_count=7, before=2, after=2) == list(range(7))


def test_reused_backend_allocates_separate_monotonic_epochs() -> None:
    backend = FaceLandmarkerBackend("unused.task")
    backend._landmarker = object()
    assert backend.begin_video() == 0
    backend._last_inference_timestamp_ms = 321
    assert backend.begin_video() == 10_321


def _motion_observations() -> pd.DataFrame:
    frames = np.arange(50)
    left = np.zeros(50, dtype=float)
    # Covering OD at frame 10 makes OS refixate temporally: eso-positive.
    left[15:26] = -0.20
    right = np.zeros(50, dtype=float)
    # Immediately after uncovering OD, it is still nasal then refixates temporal.
    right[30:34] = 0.20
    return pd.DataFrame(
        {
            "video": "clip",
            "frame": frames,
            "timestamp_ms": frames * 33,
            "track_id": 0,
            "is_patient_track": True,
            "right_nasal_u": right,
            "left_nasal_u": left,
            "right_aperture": 0.30,
            "left_aperture": 0.30,
            "yaw_proxy": 0.0,
            "pitch_proxy": 0.0,
            "roll_radians": 0.0,
        }
    )


def test_motion_cycles_apply_clinical_eso_positive_sign() -> None:
    timeline = pd.DataFrame(
        {
            "frame": np.arange(50),
            "right_covered": np.r_[np.zeros(10), np.ones(20), np.zeros(20)],
            "left_covered": 0,
            "right_gets_covered": np.arange(50) == 10,
            "right_gets_uncovered": np.arange(50) == 30,
            "left_gets_covered": False,
            "left_gets_uncovered": False,
        }
    )
    cycles = extract_motion_cycles(
        _motion_observations(), timeline, test_type=CoverTestType.COVER_UNCOVER
    )
    assert list(cycles["event_kind"]) == ["gets_covered", "gets_uncovered"]
    assert np.all(cycles["eso_signed_motion"].to_numpy() > 0)
    assert np.all(cycles["usable"].to_numpy())


def test_alternate_cover_switch_is_not_counted_twice() -> None:
    frames = np.arange(50)
    timeline = pd.DataFrame(
        {
            "frame": frames,
            "right_covered": np.r_[np.zeros(10), np.ones(20), np.zeros(20)],
            "left_covered": np.r_[np.ones(10), np.zeros(20), np.ones(20)],
            "right_gets_covered": frames == 10,
            "right_gets_uncovered": frames == 30,
            "left_gets_covered": frames == 30,
            "left_gets_uncovered": frames == 10,
        }
    )
    cycles = extract_motion_cycles(
        _motion_observations(), timeline, test_type=CoverTestType.ALTERNATE_COVER
    )
    assert len(cycles) == 2
    assert set(cycles["event_kind"]) == {"gets_uncovered"}


def test_alternate_cover_retains_unpaired_initial_binocular_cover() -> None:
    frames = np.arange(50)
    timeline = pd.DataFrame(
        {
            "frame": frames,
            "right_covered": np.r_[np.zeros(10), np.ones(40)],
            "left_covered": np.zeros(50),
            "right_gets_covered": frames == 10,
            "right_gets_uncovered": False,
            "left_gets_covered": False,
            "left_gets_uncovered": False,
        }
    )
    cycles = extract_motion_cycles(
        _motion_observations(), timeline, test_type=CoverTestType.ALTERNATE_COVER
    )
    assert len(cycles) == 1
    assert cycles.loc[0, "event_kind"] == "gets_covered"
    assert cycles.loc[0, "observed_eye"] == "left"


class _CoverProbabilityStub:
    def predict_probability(self, rows: pd.DataFrame) -> np.ndarray:
        return np.asarray(
            [0.9 if eye == "right" and 3 <= frame <= 7 else 0.1 for eye, frame in zip(rows.eye, rows.frame)],
            dtype=float,
        )


def test_cover_inference_preserves_initial_state_and_transitions() -> None:
    frames = np.arange(12)
    observations = pd.DataFrame(
        {
            "video": "clip",
            "frame": frames,
            "timestamp_ms": frames * 33,
            "track_id": 0,
            "is_patient_track": True,
            "right_cover_proxy": 0.0,
            "left_cover_proxy": 0.0,
        }
    )
    timeline = infer_cover_timeline(
        observations,
        _CoverProbabilityStub(),
        frame_count=12,
        config=EventConfig(
            median_window_frames=1,
            cover_on_threshold=0.6,
            cover_off_threshold=0.4,
            minimum_cover_frames=2,
        ),
    )
    assert timeline.loc[3, "right_gets_covered"] == 1
    assert timeline.loc[8, "right_gets_uncovered"] == 1
    assert timeline["left_covered"].sum() == 0


def test_no_face_and_no_observations_remain_unknown() -> None:
    trusted = trusted_timeline(
        pd.DataFrame(
            {
                "frame": [0, 1],
                "right_eye_state": ["no_face", "uncovered"],
                "left_eye_state": ["no_face", "uncovered"],
            }
        )
    )
    assert np.isnan(trusted.loc[0, "right_covered"])
    assert np.isnan(trusted.loc[0, "left_covered"])

    inferred = infer_cover_timeline(
        pd.DataFrame(), _CoverProbabilityStub(), frame_count=5
    )
    assert inferred["right_covered"].isna().all()
    assert inferred["left_covered"].isna().all()
    assert inferred.filter(like="gets_").to_numpy().sum() == 0

    sparse = pd.DataFrame(
        {
            "video": "clip",
            "frame": np.arange(5, 10),
            "timestamp_ms": np.arange(5, 10) * 33,
            "track_id": 0,
            "is_patient_track": True,
            "right_cover_proxy": 0.0,
            "left_cover_proxy": 0.0,
        }
    )
    inferred_sparse = infer_cover_timeline(
        sparse,
        _CoverProbabilityStub(),
        frame_count=10,
        config=EventConfig(median_window_frames=5, minimum_cover_frames=2),
    )
    assert inferred_sparse.loc[:4, "right_covered"].isna().all()
    assert inferred_sparse.loc[5, "right_gets_covered"] == 0

    long_gap = sparse.iloc[[0, -1]].copy()
    long_gap["frame"] = [0, 100]
    long_gap["timestamp_ms"] = [0, 3300]
    inferred_gap = infer_cover_timeline(
        long_gap,
        _CoverProbabilityStub(),
        frame_count=101,
        config=EventConfig(
            median_window_frames=1,
            minimum_cover_frames=2,
            max_landmark_hold_frames=10,
        ),
    )
    assert inferred_gap.loc[11:89, "right_covered"].isna().all()
    assert inferred_gap.loc[11:89, "left_covered"].isna().all()


def test_cover_model_counts_independent_videos_and_requires_both_states() -> None:
    rows = pd.DataFrame(
        {
            "video": ["short"] * 4 + ["long"] * 8,
            "covered": [0, 0, 1, 1] + [0, 0, 0, 0, 1, 1, 1, 1],
            "appearance": np.linspace(0.0, 1.0, 12),
        }
    )
    diagnostics = CoverEventModel().fit(rows)
    assert diagnostics.n_videos == 2
    assert diagnostics.n_rows == 12

    with pytest.raises(ValueError, match="covered and uncovered"):
        CoverEventModel().fit(rows.assign(covered=0))


def test_video_aggregation_marks_clip_that_starts_covered() -> None:
    timeline = pd.DataFrame(
        {
            "frame": [0, 1],
            "right_covered": [1, 1],
            "left_covered": [0, 0],
        }
    )
    features = aggregate_video_features(
        _motion_observations(),
        timeline,
        pd.DataFrame(),
        video_uid="clip",
        test_type="Cover-Uncover",
        distance="distance",
        correction="without correction",
        fps=30.0,
        frame_count=50,
    )
    assert features["starts_covered"] == 1.0
    assert features["initial_binocular_baseline_observed"] == 0.0
    assert features["verified_binocular_segment_observed"] == 0.0
    assert features["is_cover_uncover"] == 1.0


def test_video_aggregation_does_not_invent_unknown_initial_baseline() -> None:
    timeline = pd.DataFrame(
        {
            "frame": [0, 1],
            "right_covered": [np.nan, 0.0],
            "left_covered": [np.nan, 0.0],
        }
    )
    features = aggregate_video_features(
        _motion_observations(),
        timeline,
        pd.DataFrame(),
        video_uid="clip",
        test_type="Alternate-Cover",
        distance="near",
        correction="with correction",
        fps=30.0,
        frame_count=50,
    )
    assert np.isnan(features["starts_covered"])
    assert features["initial_binocular_baseline_observed"] == 0.0


def test_binocular_verification_requires_three_observed_face_frames() -> None:
    observations = _motion_observations().iloc[[0, 9]].copy()
    timeline = pd.DataFrame(
        {
            "frame": np.arange(10),
            "right_covered": np.zeros(10),
            "left_covered": np.zeros(10),
        }
    )
    features = aggregate_video_features(
        observations,
        timeline,
        pd.DataFrame(),
        video_uid="clip",
        test_type="Cover-Uncover",
        distance="near",
        correction="without_correction",
        fps=30.0,
        frame_count=10,
        sampled_frame_count=5,
    )
    assert features["verified_binocular_segment_observed"] == 0.0


def test_stable_cover_state_features_capture_fellow_eye_refixation() -> None:
    frames = np.arange(50)
    timeline = pd.DataFrame(
        {
            "frame": frames,
            "right_covered": np.r_[np.zeros(10), np.ones(20), np.zeros(20)],
            "left_covered": np.r_[np.zeros(30), np.ones(20)],
        }
    )
    features = aggregate_video_features(
        _motion_observations(),
        timeline,
        pd.DataFrame(),
        video_uid="clip",
        test_type="Cover-Uncover",
        distance="distance",
        correction="without_correction",
        fps=30.0,
        frame_count=50,
    )

    assert features["binoc_left_nasal_n"] == 10.0
    assert features["right_cov_left_nasal_n"] == 20.0
    assert features["state_cover_on_left_nasal_eso_signed"] == pytest.approx(0.2)
    assert features["state_cover_on_nasal_max_abs"] == pytest.approx(0.2)
    assert features["state_cover_on_nasal_dominant_signed"] > 0.0
    assert features["state_cover_on_pcr_available"] == 0.0


def test_paired_iris_features_compare_late_pre_and_post_transition_frames() -> None:
    frames = np.arange(50)
    timeline = pd.DataFrame(
        {
            "frame": frames,
            "right_covered": np.r_[np.zeros(10), np.ones(20), np.zeros(20)],
            "left_covered": np.zeros(50),
            "right_gets_covered": frames == 10,
            "right_gets_uncovered": frames == 30,
            "left_gets_covered": False,
            "left_gets_uncovered": False,
        }
    )
    features = aggregate_video_features(
        _motion_observations(),
        timeline,
        pd.DataFrame(),
        video_uid="clip",
        test_type="Cover-Uncover",
        distance="distance",
        correction="without_correction",
        fps=30.0,
        frame_count=50,
    )

    assert features["iris_pair_cover_on_all_raw_late_median"] == pytest.approx(0.2)
    assert features["iris_pair_uncover_all_raw_verylate_median"] == pytest.approx(0.2)
    assert features["iris_pair_cover_on_right_raw_late_n"] == 0.0
    assert features["iris_pair_cover_on_left_raw_late_n"] == 1.0
    assert features["uncover_pair_nasal_u_late18_32_all_median"] == pytest.approx(
        0.2
    )
    assert features["uncover_pair_nasal_u_late18_32_all_positive_fraction"] == 1.0
