"""Synthetic-eye tests for the native limbus localizer and native event features."""
from __future__ import annotations

import math

import cv2
import numpy as np
import pandas as pd
import pytest

from strabismus_screen.native_features import (
    build_track,
    native_feature_names,
    native_feature_schema,
    native_features,
)
from strabismus_screen.native_limbus import LimbusTracker, localize_limbus
from strabismus_screen.vision import _head_pose_from_matrix


def _render_eye(cx: float, cy: float, r: float, width: int = 280, height: int = 170, *, occluded: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Anti-aliased synthetic eye: skin background, bright sclera inside an almond fissure, dark iris disc."""

    scale = 4
    W, H = width * scale, height * scale
    img = np.full((H, W), 95, np.uint8)
    # INTER_AREA maps supersampled coordinate X to (X - 1.5) / scale, so draw at scale * c + 1.5
    # for the downsampled disc to be centred exactly at (cx, cy).
    sx, sy = cx * scale + 1.5, cy * scale + 1.5
    fissure = np.array(
        [[int((cx - 2.2 * r) * scale + 1.5), int(cy * scale + 1.5)], [int(cx * scale + 1.5), int((cy - 1.15 * r) * scale + 1.5)], [int((cx + 2.2 * r) * scale + 1.5), int(cy * scale + 1.5)], [int(cx * scale + 1.5), int((cy + 1.15 * r) * scale + 1.5)]],
        np.int32,
    )
    contour_pts = np.array([[cx - 2.2 * r, cy], [cx - 1.1 * r, cy - 1.0 * r], [cx, cy - 1.15 * r], [cx + 1.1 * r, cy - 1.0 * r], [cx + 2.2 * r, cy], [cx + 1.1 * r, cy + 1.0 * r], [cx, cy + 1.15 * r], [cx - 1.1 * r, cy + 1.0 * r]], float)
    if not occluded:
        cv2.fillPoly(img, [fissure], 135)
        cv2.circle(img, (int(round(sx)), int(round(sy))), int(round(r * scale)), 45, -1)
        cv2.circle(img, (int(round(sx)), int(round(sy))), int(round(0.42 * r * scale)), 25, -1)
    small = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    rng = np.random.default_rng(3)
    noisy = np.clip(small.astype(float) + rng.normal(0, 2.0, small.shape), 0, 255).astype(np.uint8)
    return cv2.cvtColor(noisy, cv2.COLOR_GRAY2BGR), contour_pts


@pytest.mark.parametrize("offset", [0.0, 0.3, -0.45])
def test_localizer_recovers_subpixel_centre(offset: float) -> None:
    cx, cy, r = 140.0 + offset, 85.0, 23.0
    crop, contour = _render_eye(cx, cy, r)
    out = localize_limbus(crop, cx0=cx + 4.0, cy0=cy - 2.0, r0=0.9 * r, r_prior=None, contour=contour)
    assert out["status"] == "both"
    assert out["conf"] > 0.5
    assert abs(float(out["cx"]) - cx) < 0.2
    assert abs(float(out["cy"]) - cy) < 0.6
    assert abs(float(out["r"]) - r) < 1.5


def test_localizer_reports_no_confidence_on_uniform_occluder() -> None:
    crop, contour = _render_eye(140.0, 85.0, 23.0, occluded=True)
    out = localize_limbus(crop, cx0=140.0, cy0=85.0, r0=23.0, contour=contour)
    assert out["conf"] < 0.05


def test_tracker_prior_and_seed() -> None:
    tracker = LimbusTracker()
    assert math.isnan(tracker.r_prior)
    for frame in range(5):
        tracker.update(frame, {"conf": 0.9, "status": "both", "r": 22.0 + 0.1 * frame, "cx": 10.0, "cy": 5.0}, 100.0, 50.0)
    assert abs(tracker.r_prior - 22.2) < 0.2
    sx, sy = tracker.seed(5, 112.0, 56.0)
    assert 110.0 < sx < 112.0 and 55.0 < sy < 56.0
    far_x, far_y = tracker.seed(5, 140.0, 56.0)
    assert (far_x, far_y) == (140.0, 56.0)
    late_x, _ = tracker.seed(40, 112.0, 56.0)
    assert late_x == 112.0


def test_head_pose_from_matrix_identity() -> None:
    pose = _head_pose_from_matrix(np.eye(4))
    assert pose["head_yaw_deg"] == 0.0 and pose["head_pitch_deg"] == 0.0 and pose["head_roll_deg"] == 0.0
    missing = _head_pose_from_matrix(None)
    assert all(math.isnan(v) for v in missing.values())


def _synthetic_observations(n_frames: int = 90, step_px: float = 8.0) -> pd.DataFrame:
    """Right eye uncovered at frame 30 in the deviated (nasal) position, refixating temporally at frame 36."""

    rows = []
    for frame in range(n_frames):
        base = {"video": "v", "frame": frame, "timestamp_ms": frame * 33, "track_id": 0, "is_patient_track": True, "nose_x_px": 960.0, "head_yaw_deg": 0.0}
        for i, index in enumerate((6, 197, 195, 5, 4, 168, 234, 454, 10, 152, 50, 280)):
            base[f"lm{index}_x_px"] = 900.0 + 15.0 * i
            base[f"lm{index}_y_px"] = 400.0 + 9.0 * (i % 4)
        for eye, mid_x, nasal_sign in (("right", 850.0, 1.0), ("left", 1070.0, -1.0)):
            x = mid_x
            if eye == "right":
                x = mid_x + nasal_sign * (step_px if frame < 36 else 0.0)
            covered = 1.0 if (eye == "right" and frame < 30) else 0.0
            base.update({
                f"{eye}_limbus_x_px": x, f"{eye}_limbus_y_px": 500.0, f"{eye}_limbus_r_px": 23.0, f"{eye}_limbus_conf": 0.0 if covered else 0.9,
                f"{eye}_temporal_canthus_x_px": mid_x - nasal_sign * 50.0, f"{eye}_temporal_canthus_y_px": 500.0,
                f"{eye}_nasal_canthus_x_px": mid_x + nasal_sign * 50.0, f"{eye}_nasal_canthus_y_px": 500.0,
                f"{eye}_eye_width_px": 100.0, f"{eye}_iris_diameter_px": 46.0,
            })
        rows.append(base)
    return pd.DataFrame(rows)


def _timeline(n_frames: int = 90) -> pd.DataFrame:
    tl = pd.DataFrame({"frame": np.arange(n_frames)})
    tl["right_covered"] = (tl["frame"] < 30).astype(float)
    tl["left_covered"] = 0.0
    for col in ("right_gets_covered", "right_gets_uncovered", "left_gets_covered", "left_gets_uncovered"):
        tl[col] = 0
    tl.loc[tl["frame"] == 30, "right_gets_uncovered"] = 1
    return tl


def test_native_features_measure_uncover_step_in_prism_equivalents() -> None:
    observations = _synthetic_observations()
    timeline = _timeline()
    feats = native_features(observations, timeline, fps=30.0, test_type="Cover-Uncover")
    assert set(native_feature_names()) <= set(feats)
    px_per_pd = feats["nat_q_px_per_pd"]
    assert abs(px_per_pd - (46.0 / 11.7) * 0.105) < 1e-6
    # nasal -> temporal refixation of 8 px is an eso movement of 8 / px_per_pd prism-diopter equivalents
    assert feats["nat_unc_corner_all_step_n"] == 1.0
    assert abs(feats["nat_unc_corner_all_step_median"] - 8.0 / px_per_pd) < 0.5
    assert feats["nat_unc_corner_all_step_median"] > 0
    assert abs(feats["nat_unc_corner_all_latency_median_s"] - 6 / 30.0) < 0.05
    track = build_track(observations, timeline, fps=30.0)
    assert track is not None and track.frames[-1] == 89


def test_native_feature_schema_is_stable() -> None:
    schema = native_feature_schema()
    assert tuple(schema) == native_feature_names()
    empty = native_features(pd.DataFrame(), pd.DataFrame(), fps=30.0, test_type="Alternate-Cover")
    assert set(empty) == set(native_feature_names())
