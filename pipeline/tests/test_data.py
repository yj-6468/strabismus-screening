from __future__ import annotations

from collections import Counter
from dataclasses import fields
from functools import lru_cache
from itertools import islice
from pathlib import Path
import sys


import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
# Regression checks on the final dataset at <Strabismus>/data/final_dataset (774 clips).
DATASET_ROOT = PROJECT_ROOT.parent / "data" / "final_dataset"
FINAL_VIDEO_COUNT = 774

from strabismus_screen.data import (  # noqa: E402
    VideoRecord,
    load_dataset,
    make_video_uid,
    parse_condition,
)
from strabismus_screen.labels import (  # noqa: E402
    Job1ManifestLabel,
)


@lru_cache(maxsize=1)
def _dataset():
    if not DATASET_ROOT.is_dir():
        pytest.skip(f"final dataset not found at {DATASET_ROOT}")
    return load_dataset(DATASET_ROOT, expected_video_count=FINAL_VIDEO_COUNT)


def test_exact_one_video_one_subject_manifest_and_joins() -> None:
    dataset = _dataset()
    records = dataset.manifest
    assert len(records) == FINAL_VIDEO_COUNT
    assert len({record.video_uid for record in records}) == FINAL_VIDEO_COUNT
    assert all(record.subject_uid == record.video_uid for record in records)
    assert all(record.video_path.is_file() for record in records)
    assert [record.relative_path for record in records] == sorted(
        record.relative_path for record in records
    )
    assert all(
        make_video_uid(record.relative_path) == record.video_uid for record in records
    )
    assert all(
        record.video_uid.startswith("vid_")
        and len(record.video_uid) == 68
        and record.filename not in record.video_uid
        for record in records
    )

    assert dataset.validation.video_count == FINAL_VIDEO_COUNT
    assert dataset.validation.event_count == 8_287
    assert dataset.validation.frame_count == 261_311
    assert len(dataset.records_by_uid) == FINAL_VIDEO_COUNT
    assert len(dataset.events_by_uid) == FINAL_VIDEO_COUNT
    assert all(
        len(dataset.events_by_uid[record.video_uid]) == record.summary.total_events
        for record in records
    )


def test_protocol_and_condition_decoding() -> None:
    records = _dataset().manifest
    assert Counter(record.test_type for record in records) == {
        "Cover-Uncover": 456,
        "Alternate-Cover": 318,
    }
    # 25 clips carry a clinician-confirmed acquisition correction (near -> distance,
    # configs/distance_overrides_v2.csv): 18 sc and 7 cc.  Filename-derived counts
    # are recovered with apply_distance_overrides=False.
    assert Counter((record.distance, record.correction) for record in records) == {
        ("near", "without_correction"): 269,
        ("distance", "without_correction"): 223,
        ("near", "with_correction"): 143,
        ("distance", "with_correction"): 139,
    }
    raw = load_dataset(DATASET_ROOT, expected_video_count=FINAL_VIDEO_COUNT, apply_distance_overrides=False).manifest
    assert Counter((record.distance, record.correction) for record in raw) == {
        ("near", "without_correction"): 287,
        ("distance", "without_correction"): 205,
        ("near", "with_correction"): 150,
        ("distance", "with_correction"): 132,
    }
    assert {r.distance_source for r in raw} == {"filename"}
    assert sum(r.distance_source == "clinician_override" for r in records) == 25
    assert parse_condition("Dcc") == ("distance", "with_correction")
    assert parse_condition("Nsc") == ("near", "without_correction")


def test_approved_job1_manifest_targets_have_exact_counts() -> None:
    records = _dataset().manifest
    counts = Counter(record.job1_manifest_label for record in records)
    # Phorias and well-controlled intermittent tropias (control <= 2) count as orthotropic.
    # None = the 318 alternate-cover clips.
    assert counts == {
        Job1ManifestLabel.EXO_TROPIA: 193,
        Job1ManifestLabel.ESO_TROPIA: 77,
        Job1ManifestLabel.NO_MANIFEST_HORIZONTAL_DEVIATION: 186,
        None: 318,
    }
    well_controlled = [
        r for r in records
        if r.test_type == "Cover-Uncover" and r.raw_horizontal_type in {"X(T)", "E(T)"}
        and r.control is not None and r.control <= 2.0
    ]
    assert well_controlled and all(
        r.job1_manifest_label is Job1ManifestLabel.NO_MANIFEST_HORIZONTAL_DEVIATION
        for r in well_controlled
    )
    assert all(
        r.job1_manifest_label is not Job1ManifestLabel.NO_MANIFEST_HORIZONTAL_DEVIATION
        for r in records
        if r.test_type == "Cover-Uncover" and r.raw_horizontal_type in {"ET", "XT"}
    )
    eligible = [record for record in records if record.job1_manifest_label is not None]
    assert len(eligible) == 456  # all CU clips; flick phorias are no longer excluded
    assert all(record.test_type == "Cover-Uncover" for record in eligible)
    assert all(
        record.observable_in_clip is None
        for record in eligible
        if record.raw_horizontal_type in {"E(T)", "X(T)"}
    )


def test_control_is_never_a_feature() -> None:
    # Control defines the Job 1 target, so it is on the record, but it must never be a feature.
    assert "control" in {field.name.casefold() for field in fields(VideoRecord)}
    import pandas as pd  # noqa: PLC0415

    from strabismus_screen.pipeline import select_task_feature_names  # noqa: PLC0415

    probe = pd.DataFrame({"control": [1.0], "Control": [1.0], "binoc_nasal_sum_median": [0.1]})
    assert "control" not in {c.casefold() for c in select_task_feature_names(probe)}


def test_pd_intervals_and_normal_zero_are_lossless() -> None:
    records = _dataset().manifest
    assert sum(record.horizontal_pd is None for record in records) == 11
    assert (
        sum(
            record.horizontal_pd is not None and record.horizontal_pd.is_exact
            for record in records
        )
        == 591
    )  # 476 scalar source labels plus 115 explicit normal zeros.
    assert (
        sum(
            record.horizontal_pd is not None and not record.horizontal_pd.is_exact
            for record in records
        )
        == 172
    )
    for record in records:
        if record.raw_horizontal_type == "normal":
            assert record.horizontal_pd is not None
            assert (record.horizontal_pd.lower, record.horizontal_pd.upper) == (
                0.0,
                0.0,
            )
            assert (record.horizontal_pd_lower, record.horizontal_pd_upper) == (
                0.0,
                0.0,
            )


def test_beginning_covered_and_right_censored_events_are_explicit() -> None:
    dataset = _dataset()
    records = dataset.manifest
    assert dataset.validation.clips_beginning_covered == 79
    assert (
        sum(
            record.begins_right_covered is True or record.begins_left_covered is True
            for record in records
        )
        == 79
    )
    assert dataset.validation.right_censored_cover_events == 238
    assert (
        sum(
            record.right_censored_right_cover_events
            + record.right_censored_left_cover_events
            for record in records
        )
        == 238
    )
    assert all(
        episode.uncover_frame is None
        for episode in dataset.cover_episodes
        if episode.right_censored
    )
    assert sum(episode.begins_covered for episode in dataset.cover_episodes) == 79


def test_public_frame_stream_is_joined_to_opaque_uid() -> None:
    dataset = _dataset()
    sample = tuple(islice(dataset.iter_frames(), 5))
    assert [frame.frame for frame in sample] == [0, 1, 2, 3, 4]
    assert all(frame.video_uid in dataset.records_by_uid for frame in sample)
