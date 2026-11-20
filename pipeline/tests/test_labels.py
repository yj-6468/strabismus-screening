from __future__ import annotations

from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from strabismus_screen.labels import (  # noqa: E402
    HorizontalDirection,
    Job1ManifestLabel,
    LabelError,
    horizontal_direction,
    job1_manifest_label,
    observable_in_clip,
    parse_horizontal_pd,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("ET", HorizontalDirection.ESO),
        ("E(T)", HorizontalDirection.ESO),
        ("E'", HorizontalDirection.ESO),
        ("XT", HorizontalDirection.EXO),
        ("X(T)", HorizontalDirection.EXO),
        ("X", HorizontalDirection.EXO),
        ("normal", HorizontalDirection.NORMAL),
        ("Flick E' to Flick X'", HorizontalDirection.MIXED),
    ],
)
def test_horizontal_direction(raw: str, expected: HorizontalDirection) -> None:
    assert horizontal_direction(raw) is expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("ET", Job1ManifestLabel.ESO_TROPIA),
        ("E(T)", Job1ManifestLabel.ESO_TROPIA),
        ("XT", Job1ManifestLabel.EXO_TROPIA),
        ("X(T)", Job1ManifestLabel.EXO_TROPIA),
        ("E", Job1ManifestLabel.NO_MANIFEST_HORIZONTAL_DEVIATION),
        ("E'", Job1ManifestLabel.NO_MANIFEST_HORIZONTAL_DEVIATION),
        ("X", Job1ManifestLabel.NO_MANIFEST_HORIZONTAL_DEVIATION),
        ("normal", Job1ManifestLabel.NO_MANIFEST_HORIZONTAL_DEVIATION),
        ("Flick E' to Flick X'", Job1ManifestLabel.NO_MANIFEST_HORIZONTAL_DEVIATION),
        ("flick X' to E'2", Job1ManifestLabel.NO_MANIFEST_HORIZONTAL_DEVIATION),
    ],
)
def test_job1_cover_uncover_policy(
    raw: str, expected: Job1ManifestLabel | None
) -> None:
    assert job1_manifest_label(raw, "Cover-Uncover") is expected
    assert job1_manifest_label(raw, "Alternate-Cover") is None


def test_observable_in_clip_does_not_overclaim_intermittency() -> None:
    assert observable_in_clip("ET", "Cover-Uncover") is True
    assert observable_in_clip("normal", "Cover-Uncover") is False
    assert observable_in_clip("X", "Cover-Uncover") is False
    assert observable_in_clip("X(T)", "Cover-Uncover") is None
    assert observable_in_clip("ET", "Alternate-Cover") is None


def test_prism_diopter_scalar_interval_and_normal_zero() -> None:
    scalar = parse_horizontal_pd("18", "XT")
    assert scalar is not None
    assert (scalar.lower, scalar.upper, scalar.midpoint, scalar.is_exact) == (
        18.0,
        18.0,
        18.0,
        True,
    )

    interval = parse_horizontal_pd("16-20", "ET")
    assert interval is not None
    assert (interval.lower, interval.upper, interval.midpoint, interval.is_exact) == (
        16.0,
        20.0,
        18.0,
        False,
    )

    normal = parse_horizontal_pd(None, "normal")
    assert normal is not None
    assert (normal.lower, normal.upper) == (0.0, 0.0)
    assert parse_horizontal_pd(None, "Flick E' to Flick X'") is None


@pytest.mark.parametrize("value", ["20-16", "abc", "-4", "4--8"])
def test_invalid_prism_diopter_values_fail(value: str) -> None:
    with pytest.raises(LabelError):
        parse_horizontal_pd(value, "XT")


def test_normal_cannot_carry_nonzero_pd() -> None:
    with pytest.raises(LabelError):
        parse_horizontal_pd("2", "normal")
