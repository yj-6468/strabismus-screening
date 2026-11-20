from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import strabismus_screen.pipeline as pipeline_module
from strabismus_screen.config import EventConfig, MotionConfig
from strabismus_screen.pipeline import (
    OBSERVATION_CACHE_VERSION,
    PipelineModelBundle,
    analyze_raw_video,
    load_observation_cache,
    save_observation_cache,
)
from strabismus_screen.vision import VideoMetadata, VisionResult


def test_observation_cache_round_trip_rewrites_video_to_opaque_uid(tmp_path: Path) -> None:
    metadata = VideoMetadata(
        path="identifiable/source.mov",
        frame_count=10,
        fps=30.0,
        width=100,
        height=80,
        orientation_degrees=0,
        decoded_frames=10,
        sampled_frames=5,
        patient_track_id=0,
    )
    observations = pd.DataFrame(
        {
            "video": ["identifiable/source.mov"],
            "frame": [0],
            "track_id": [0],
            "is_patient_track": [True],
            "right_nasal_u": [0.1],
        }
    )
    result = VisionResult(metadata=metadata, observations=observations)
    path = tmp_path / "cache.joblib"
    save_observation_cache(
        path, video_uid="vid_opaque", signature="signature", result=result
    )
    restored = load_observation_cache(
        path,
        expected_video_uid="vid_opaque",
        expected_signature="signature",
    )
    assert restored.observations.loc[0, "video"] == "vid_opaque"
    assert restored.metadata.path == "vid_opaque"
    assert restored.metadata.sampled_frames == 5
    assert restored.metadata.frame_count == restored.metadata.decoded_frames == 10
    assert OBSERVATION_CACHE_VERSION >= 1

    with pytest.raises(ValueError, match="signature"):
        load_observation_cache(path, expected_signature="stale")


def test_cache_rejects_non_dataframe_payload(tmp_path: Path) -> None:
    import joblib

    path = tmp_path / "bad.joblib"
    joblib.dump(
        {
            "cache_version": OBSERVATION_CACHE_VERSION,
            "video_uid": "vid",
            "signature": "sig",
            "metadata": {
                "path": "x",
                "frame_count": 1,
                "fps": 1.0,
                "width": 1,
                "height": 1,
                "orientation_degrees": 0,
                "decoded_frames": 1,
                "sampled_frames": 1,
                "patient_track_id": None,
            },
            "observations": np.zeros((1, 1)),
        },
        path,
    )
    with pytest.raises(TypeError, match="DataFrame"):
        load_observation_cache(path)


def test_cache_uses_decoded_not_nominal_container_frame_count(tmp_path: Path) -> None:
    import joblib

    path = tmp_path / "legacy.joblib"
    joblib.dump(
        {
            "cache_version": OBSERVATION_CACHE_VERSION,
            "video_uid": "vid",
            "signature": "sig",
            "metadata": {
                "path": "x",
                "frame_count": 56,
                "fps": 30.0,
                "width": 100,
                "height": 80,
                "orientation_degrees": 0,
                "decoded_frames": 26,
                "sampled_frames": 13,
                "patient_track_id": None,
            },
            "observations": pd.DataFrame(),
        },
        path,
    )
    assert load_observation_cache(path).metadata.frame_count == 26


def test_raw_prediction_contract_and_act_observability_guard(monkeypatch) -> None:
    metadata = VideoMetadata(
        path="source.mov",
        frame_count=10,
        fps=30.0,
        width=100,
        height=80,
        orientation_degrees=0,
        decoded_frames=10,
        sampled_frames=5,
        patient_track_id=0,
    )
    result = VisionResult(metadata=metadata, observations=pd.DataFrame())

    class TaskStubs:
        def predict_job1(self, X, *, quality):
            return [{"forced_label": "exo_tropia", "warnings": []}]

        def predict_job2(self, X, *, quality):
            return [
                {
                    "direction": "exo",
                    "forced_magnitude_pd": 12.0,
                    "prediction_interval_pd": [7.0, 17.0],
                    "warnings": [],
                }
            ]

        def predict_job3(self, X, *, quality):
            return [
                {
                    "forced_label": "intermittent_tropia",
                    "clinical_class": "intermittent_tropia",
                    "observable_state": "yes",
                    "observable_in_clip": True,
                    "indeterminate": False,
                    "warnings": [],
                }
            ]

    bundle = PipelineModelBundle(
        cover_event_model=object(),
        task_models=TaskStubs(),
        feature_names=("feature",),
        event_config=EventConfig(),
        motion_config=MotionConfig(),
        extraction_stride=2,
        maximum_faces=2,
        face_landmarker_sha256="expected",
        training_metadata={},
    )
    monkeypatch.setattr(
        pipeline_module, "extract_video_observations", lambda *args, **kwargs: result
    )
    monkeypatch.setattr(
        pipeline_module,
        "infer_cover_timeline",
        lambda *args, **kwargs: pd.DataFrame({"frame": range(10)}),
    )
    monkeypatch.setattr(
        pipeline_module,
        "feature_record_from_timeline",
        lambda *args, **kwargs: {
            "feature": 1.0,
            "technical_quality_score": 0.9,
            "starts_covered": 0.0,
            "all_usable_n": 1.0,
            "cover_state_known_fraction": 1.0,
            "initial_binocular_baseline_observed": 1.0,
            "verified_binocular_segment_observed": 1.0,
        },
    )
    monkeypatch.setattr(
        pipeline_module, "_file_content_sha256", lambda path: "expected"
    )
    output = analyze_raw_video(
        bundle,
        video_path="source.mov",
        test_type="Alternate-Cover",
        distance="near",
        correction="sc",
        face_landmarker_model="landmarker.task",
    )
    assert output["inputs"]["correction"] == "without_correction"
    assert output["job3_manifestation"]["clinical_class"] == "intermittent_tropia"
    assert output["observable_in_clip"] == "indeterminate"
    assert "observability_requires_cover_uncover" in output["job3_manifestation"]["warnings"]

    with pytest.raises(ValueError, match="distance"):
        analyze_raw_video(
            bundle,
            video_path="source.mov",
            test_type="Cover-Uncover",
            distance="typo",
            correction="sc",
            face_landmarker_model="landmarker.task",
        )


def test_raw_prediction_with_no_detected_face_still_returns_all_jobs(monkeypatch) -> None:
    metadata = VideoMetadata(
        path="source.mov",
        frame_count=12,
        fps=30.0,
        width=100,
        height=80,
        orientation_degrees=0,
        decoded_frames=12,
        sampled_frames=6,
        patient_track_id=None,
    )
    result = VisionResult(metadata=metadata, observations=pd.DataFrame())

    class TaskStubs:
        def predict_job1(self, X, *, quality):
            assert quality == 0.0
            return [{"forced_label": "no_manifest_horizontal_deviation", "warnings": []}]

        def predict_job2(self, X, *, quality):
            return [
                {
                    "direction": "normal",
                    "forced_magnitude_pd": 0.0,
                    "prediction_interval_pd": [0.0, 30.0],
                    "warnings": [],
                }
            ]

        def predict_job3(self, X, *, quality):
            return [
                {
                    "forced_label": "normal",
                    "clinical_class": "normal",
                    "observable_state": "no",
                    "observable_in_clip": False,
                    "indeterminate": False,
                    "warnings": [],
                }
            ]

    bundle = PipelineModelBundle(
        cover_event_model=object(),
        task_models=TaskStubs(),
        feature_names=("face_observation_fraction", "all_usable_n"),
        event_config=EventConfig(),
        motion_config=MotionConfig(),
        extraction_stride=2,
        maximum_faces=2,
        face_landmarker_sha256="expected",
        training_metadata={},
    )
    monkeypatch.setattr(
        pipeline_module, "extract_video_observations", lambda *args, **kwargs: result
    )
    monkeypatch.setattr(
        pipeline_module, "_file_content_sha256", lambda path: "expected"
    )

    output = analyze_raw_video(
        bundle,
        video_path="source.mov",
        test_type="Cover-Uncover",
        distance="distance",
        correction="cc",
        face_landmarker_model="landmarker.task",
    )
    assert output["job1_manifest_direction"]["forced_label"]
    assert output["job2_total_horizontal_deviation"]["forced_magnitude_pd"] == 0.0
    assert output["job3_manifestation"]["clinical_class"] == "normal"
    assert output["observable_in_clip"] == "indeterminate"
    assert output["technical_quality_score"] == 0.0
    assert "no_patient_face_track" in output["common_warnings"]
    assert "no_usable_refixation_cycle" in output["common_warnings"]


def test_load_observation_cache_tolerates_extra_metadata_keys(tmp_path):
    import joblib
    import pandas as pd
    from strabismus_screen.pipeline import OBSERVATION_CACHE_VERSION, load_observation_cache

    payload = {
        "cache_version": OBSERVATION_CACHE_VERSION,
        "video_uid": "vid_x",
        "signature": "sig",
        "metadata": {
            "path": "vid_x", "frame_count": 3, "fps": 30.0, "width": 4, "height": 4, "orientation_degrees": 0,
            "decoded_frames": 3, "sampled_frames": 3, "patient_track_id": 0, "limbus_localizer": "radialprofile",
        },
        "observations": pd.DataFrame({"video": ["vid_x"], "frame": [0], "track_id": [0]}),
    }
    path = tmp_path / "vid_x.joblib"
    joblib.dump(payload, path)
    result = load_observation_cache(path, expected_video_uid="vid_x")
    assert result.metadata.decoded_frames == 3
