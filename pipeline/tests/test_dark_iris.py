from __future__ import annotations

import numpy as np
import pandas as pd

from strabismus_screen.dark_iris import (
    dark_iris_feature_names,
    dark_iris_feature_schema,
    dark_iris_features,
)
from strabismus_screen.pipeline import (
    select_job2_feature_names,
    select_task_feature_names,
)


def test_dark_iris_schema_is_fixed_and_empty_input_is_forced_output_safe() -> None:
    schema = dark_iris_feature_schema()
    assert tuple(schema) == dark_iris_feature_names()
    assert len(schema) == 92
    assert schema["dark_requested_pair_n"] == 0.0
    assert schema["dark_usable_pair_n"] == 0.0
    assert np.isnan(schema["dark_cover_early_median"])

    output = dark_iris_features(
        "unused.mov",
        pd.DataFrame(),
        pd.DataFrame(),
        fps=30.0,
    )
    assert output.keys() == schema.keys()
    assert output["dark_usable_pair_n"] == 0.0


def test_dark_features_are_job2_only() -> None:
    table = pd.DataFrame(
        {
            "is_near": [1.0],
            "all_usable_n": [2.0],
            "dark_cover_early_median": [0.1],
            "dark_usable_pair_n": [3.0],
        }
    )
    base = select_task_feature_names(table)
    job2 = select_job2_feature_names(table)
    assert "dark_cover_early_median" not in base
    assert "dark_usable_pair_n" not in base
    assert "dark_cover_early_median" in job2
    assert "dark_usable_pair_n" in job2
