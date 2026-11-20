from __future__ import annotations

import numpy as np
import pandas as pd

from strabismus_screen.job1_rich_features import job1_rich_features


def test_job1_rich_features_have_fixed_cover_only_schema_and_motion() -> None:
    frames = np.arange(60)
    observations = pd.DataFrame(
        {
            "frame": frames,
            "is_patient_track": True,
            "roll_radians": 0.0,
            "yaw_proxy": 0.0,
            "pitch_proxy": 0.0,
            "nose_x_px": 50.0,
            "right_eye_width_px": 20.0,
            "left_eye_width_px": 20.0,
            "right_eye_mid_x_px": 40.0,
            "left_eye_mid_x_px": 60.0,
            "right_eye_mid_y_px": 30.0,
            "left_eye_mid_y_px": 30.0,
            "right_iris_x_px": 40.0,
            "left_iris_x_px": np.where(frames >= 20, 58.0, 60.0),
            "right_iris_y_px": 30.0,
            "left_iris_y_px": 30.0,
            "right_glint_reliability": 1.0,
            "left_glint_reliability": 1.0,
            "right_glint_x": 0.0,
            "left_glint_x": 0.0,
        }
    )
    right_covered = frames >= 20
    timeline = pd.DataFrame(
        {
            "frame": frames,
            "right_covered": right_covered.astype(float),
            "left_covered": 0.0,
            "right_gets_covered": frames == 20,
            "right_gets_uncovered": False,
            "left_gets_covered": False,
            "left_gets_uncovered": False,
        }
    )
    features = job1_rich_features(observations, timeline, fps=30.0)
    assert len(features) == 247
    assert not any(name.startswith("rich_uncover_") for name in features)
    assert features["rich_cover_raw_early_n"] == 1.0
    assert np.isfinite(features["rich_cover_raw_early_median"])

