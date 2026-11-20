from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from strabismus_screen.evaluation import (  # noqa: E402
    classification_metrics,
    make_evaluation_splits,
    pd_prediction_interval_metrics,
    pd_regression_metrics,
)
from strabismus_screen.modeling import (  # noqa: E402
    DirectionHead,
    FeatureSchema,
    ManifestationHead,
    PDMagnitudeHead,
    StrabismusModelBundle,
    direction_label_from_diagnosis,
    manifestation_label_from_diagnosis,
    parse_pd_interval,
)


def test_dataset_label_mappings_are_explicit_and_mixed_labels_need_adjudication():
    assert direction_label_from_diagnosis("ET") == "eso_tropia"
    assert direction_label_from_diagnosis("E(T)") == "eso_tropia"
    assert direction_label_from_diagnosis("XT") == "exo_tropia"
    assert direction_label_from_diagnosis("X(T)") == "exo_tropia"
    assert direction_label_from_diagnosis("E") == "no_manifest_horizontal_deviation"
    assert direction_label_from_diagnosis("normal") == "no_manifest_horizontal_deviation"
    # A phoria-only patient is orthotropic for Job 1; the class needs no
    # direction, so a mixed-direction flick phoria is normal on a CU clip.
    assert (
        direction_label_from_diagnosis("Flick E' to Flick X'")
        == "no_manifest_horizontal_deviation"
    )
    assert direction_label_from_diagnosis("Flick E' to Flick X'", "Alternate-Cover") is None

    assert manifestation_label_from_diagnosis("ET") == "tropia"
    assert manifestation_label_from_diagnosis("X(T)") == "intermittent_tropia"
    assert manifestation_label_from_diagnosis("E'") == "phoria"
    assert manifestation_label_from_diagnosis("normal") == "normal"
    assert manifestation_label_from_diagnosis("flick X' to E'2") == "phoria"
    assert direction_label_from_diagnosis("ET", "Alternate-Cover") is None


def test_parse_pd_interval_preserves_censoring_and_zero_is_opt_in():
    assert parse_pd_interval(20) == (20.0, 20.0)
    assert parse_pd_interval("20-25") == (20.0, 25.0)
    assert parse_pd_interval("30–35") == (30.0, 35.0)
    assert parse_pd_interval("") is None
    assert parse_pd_interval("", normal_is_zero=True) == (0.0, 0.0)
    with pytest.raises(ValueError):
        parse_pd_interval("unknown")


def _direction_data(seed: int = 8):
    rng = np.random.default_rng(seed)
    eso = rng.normal(loc=(-2.0, 0.0, 0.2), scale=0.35, size=(30, 3))
    exo = rng.normal(loc=(2.0, 0.0, -0.2), scale=0.35, size=(30, 3))
    normal = rng.normal(loc=(0.0, 2.0, 0.0), scale=0.35, size=(30, 3))
    X = np.vstack((eso, exo, normal))
    y = np.asarray(
        ["eso_tropia"] * 30
        + ["exo_tropia"] * 30
        + ["no_manifest_horizontal_deviation"] * 30,
        dtype=object,
    )
    return X, y


def test_direction_head_forces_outputs_and_serializes_deterministically():
    X, y = _direction_data()
    model = DirectionHead(random_state=42).fit(X, y, feature_names=("dx", "dy", "velocity"))
    probabilities = model.predict_proba(X)
    assert probabilities.shape == (90, 3)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    assert np.mean(model.predict(X) == y) > 0.95

    outputs = model.predict_with_warnings(X[:2], quality=[0.2, 0.9])
    assert all(
        output["forced_label"]
        in {"eso_tropia", "exo_tropia", "no_manifest_horizontal_deviation"}
        for output in outputs
    )
    assert "low_quality" in outputs[0]["warnings"]
    assert outputs[1]["quality"] == 0.9

    first = model.dumps()
    second = model.dumps()
    assert first == second
    restored = DirectionHead.loads(first)
    np.testing.assert_array_equal(restored.predict(X), model.predict(X))


def test_direction_threshold_calibration_targets_deviation_sensitivity():
    X, y = _direction_data()
    model = DirectionHead(random_state=3, deviation_probability_threshold=0.9).fit(X, y)
    threshold = model.calibrate_deviation_threshold(X, y, target_sensitivity=0.95)
    prediction = model.predict(X)
    no_manifest = "no_manifest_horizontal_deviation"
    sensitivity = np.mean(prediction[y != no_manifest] != no_manifest)
    assert 0 <= threshold <= 1
    assert sensitivity >= 0.95


def test_direction_head_keeps_uncover_pairs_out_of_manifest_gate() -> None:
    X, y = _direction_data()
    names = (
        "cover_state_signal",
        "iris_pair_broad_motion",
        "uncover_pair_nasal_u_late18_32_all_median",
    )
    model = DirectionHead(random_state=4).fit(X, y, feature_names=names)

    manifest_names = {
        names[index] for index in model.manifest_feature_indices_.tolist()
    }
    compact_direction_names = {
        names[index] for index in model.direction_feature_indices_.tolist()
    }
    broad_direction_names = {
        names[index] for index in model.direction_broad_feature_indices_.tolist()
    }
    assert "uncover_pair_nasal_u_late18_32_all_median" not in manifest_names
    assert "uncover_pair_nasal_u_late18_32_all_median" in compact_direction_names
    assert "iris_pair_broad_motion" not in compact_direction_names
    assert "iris_pair_broad_motion" in broad_direction_names
    assert "uncover_pair_nasal_u_late18_32_all_median" not in broad_direction_names


def test_direction_head_uses_clinical_family_prior_and_isolates_rich_features() -> None:
    values, y = _direction_data()
    X = pd.DataFrame(
        {
            "cover_state_signal": values[:, 0],
            "all_motion_summary": values[:, 1],
            "uncover_motion_median": values[:, 2],
            "rich_cover_raw_early_median": values[:, 0] * 0.1,
            "rich_uncover_raw_early_median": values[:, 2] * 0.1,
        }
    )
    clinical = np.asarray(
        ["tropia"] * 30
        + ["intermittent_tropia"] * 30
        + ["normal"] * 30,
        dtype=object,
    )
    model = DirectionHead(random_state=9).fit(X, y, clinical_y=clinical)
    names = model.feature_schema_.names
    manifest_names = {
        names[index] for index in model.manifest_feature_indices_.tolist()
    }
    clinical_names = {
        names[index]
        for index in model.manifest_clinical_feature_indices_.tolist()
    }
    direction_names = {
        names[index] for index in model.direction_feature_indices_.tolist()
    }
    assert "rich_cover_raw_early_median" in manifest_names
    assert "rich_uncover_raw_early_median" not in manifest_names
    assert "all_motion_summary" not in manifest_names
    assert not any(name.startswith("rich_") for name in clinical_names)
    assert not any(name.startswith("rich_") for name in direction_names)
    np.testing.assert_allclose(model.predict_proba(X).sum(axis=1), 1.0)


def test_pd_head_uses_ranges_returns_nonnegative_interval_and_keeps_direction_separate():
    rng = np.random.default_rng(14)
    X = rng.normal(size=(72, 4))
    latent = np.maximum(0.0, 18.0 + 7.0 * X[:, 0] - 3.0 * X[:, 1])
    lower = np.maximum(0.0, np.floor((latent - 2.0) / 2.0) * 2.0)
    upper = np.ceil((latent + 2.0) / 2.0) * 2.0
    lower[:3] = 0.0
    upper[:3] = 0.0
    groups = np.repeat(np.arange(18), 4)

    model = PDMagnitudeHead(
        random_state=21,
        n_estimators=24,
        calibration_folds=3,
        min_samples_leaf=1,
    ).fit(X, lower, upper, groups=groups, feature_names=("a", "b", "c", "d"))
    point = model.predict(X[:6])
    interval = model.predict_interval(X[:6])
    assert np.all(point >= 0)
    assert interval.shape == (6, 2)
    assert np.all(interval[:, 0] >= 0)
    assert np.all(interval[:, 0] <= point)
    assert np.all(interval[:, 1] >= point)
    assert model.calibration_radius_pd_ >= 0
    assert model.calibration_method_ == "grouped_oof"

    unspecified = model.predict_with_warnings(X[:1])[0]
    assert unspecified["forced_magnitude_pd"] >= 0
    assert unspecified["direction"] is None
    assert "direction_not_provided" in unspecified["warnings"]
    specified = model.predict_with_warnings(X[:1], direction="exo")[0]
    assert specified["direction"] == "exo"
    assert "direction_not_provided" not in specified["warnings"]

    total_direction = np.where(upper == 0, "normal", np.where(X[:, 0] < 0, "eso", "exo"))
    with_direction = PDMagnitudeHead(
        random_state=22, n_estimators=16, calibration_folds=2, min_samples_leaf=1
    ).fit(X, lower, upper, direction_y=total_direction)
    directed = with_direction.predict_with_warnings(X[:3])
    assert all(output["direction"] in {"eso", "exo", "normal"} for output in directed)
    assert all(output["direction_probabilities"] is not None for output in directed)
    assert all("direction_not_provided" not in output["warnings"] for output in directed)

    normal_row = int(np.flatnonzero(total_direction == "normal")[0])
    normal_output = with_direction.predict_with_warnings(X[[normal_row]])[0]
    if normal_output["direction"] == "normal":
        assert normal_output["forced_magnitude_pd"] == 0.0
        assert normal_output["prediction_interval_pd"][0] == 0.0

    with pytest.raises(ValueError, match="incoherent"):
        PDMagnitudeHead(random_state=23, n_estimators=8).fit(
            X[:4],
            np.ones(4),
            np.ones(4),
            direction_y=["normal"] * 4,
        )


def test_bundle_enforces_one_engineered_feature_contract():
    X, y = _direction_data()
    direction = DirectionHead(random_state=2).fit(X, y, feature_names=("a", "b", "c"))
    pd = PDMagnitudeHead(random_state=2, n_estimators=12, calibration_folds=2).fit(
        X,
        np.full(len(X), 10.0),
        np.full(len(X), 12.0),
        feature_names=("a", "b", "c"),
    )
    clinical = np.resize(
        np.asarray(["tropia", "intermittent_tropia", "phoria", "normal"]), len(X)
    )
    manifestation = ManifestationHead(random_state=2).fit(
        X,
        clinical,
        np.resize(np.asarray([0, 1]), len(X)),
        feature_names=("a", "b", "c"),
    )
    bundle = StrabismusModelBundle(
        direction_head=direction, pd_head=pd, manifestation_head=manifestation
    )
    assert bundle.validate_shared_feature_schema() == FeatureSchema(("a", "b", "c"))


def test_bundle_contract_is_ordered_union_when_job1_has_extra_features():
    X, y = _direction_data()
    direction = DirectionHead(random_state=5).fit(
        X, y, feature_names=("a", "b", "job1_rich")
    )
    pd = PDMagnitudeHead(
        random_state=5, n_estimators=8, calibration_folds=2
    ).fit(
        X[:, :2],
        np.full(len(X), 8.0),
        np.full(len(X), 10.0),
        feature_names=("a", "b"),
    )
    clinical = np.resize(
        np.asarray(["tropia", "intermittent_tropia", "phoria", "normal"]),
        len(X),
    )
    manifestation = ManifestationHead(random_state=5).fit(
        X[:, :2],
        clinical,
        np.resize(np.asarray([0, 1]), len(X)),
        feature_names=("a", "b"),
    )
    bundle = StrabismusModelBundle(
        direction_head=direction,
        pd_head=pd,
        manifestation_head=manifestation,
    )
    assert bundle.validate_shared_feature_schema() == FeatureSchema(
        ("a", "b", "job1_rich")
    )


def test_bundle_can_preserve_base_job2_direction_with_richer_magnitude_head():
    class MagnitudeStub:
        def predict_with_warnings(self, X, *, quality, direction):
            assert list(direction) == ["exo", "normal"]
            return [
                {
                    "direction": str(value),
                    "direction_confidence": None,
                    "direction_probabilities": None,
                    "forced_magnitude_pd": 9.0 if value != "normal" else 0.0,
                    "warnings": [],
                }
                for value in direction
            ]

    class DirectionStub:
        def predict_direction(self, X):
            return np.asarray(["exo", "normal"], dtype=object)

        def predict_direction_proba(self, X):
            return np.asarray(
                [[0.10, 0.80, 0.10], [0.10, 0.15, 0.75]], dtype=float
            )

    bundle = StrabismusModelBundle(
        direction_head=object(),
        pd_head=MagnitudeStub(),
        pd_direction_head=DirectionStub(),
        manifestation_head=object(),
    )
    outputs = bundle.predict_job2(pd.DataFrame({"feature": [1.0, 2.0]}), quality=0.9)
    assert [output["direction"] for output in outputs] == ["exo", "normal"]
    assert outputs[0]["direction_confidence"] == 0.80
    assert outputs[1]["direction_probabilities"]["normal"] == 0.75


def test_classifier_sanitizes_extreme_fold_specific_inputs() -> None:
    X = pd.DataFrame(
        {
            "nearly_constant": np.linspace(0.0, 1e-300, 12),
            "signal": np.tile([0.0, 1.0, 2.0], 4),
        }
    )
    y = np.tile(
        ["eso_tropia", "exo_tropia", "no_manifest_horizontal_deviation"], 4
    )
    model = DirectionHead(random_state=91).fit(X, y)
    probabilities = model.predict_proba(
        pd.DataFrame({"nearly_constant": [np.inf], "signal": [1e300]})
    )
    assert np.isfinite(probabilities).all()
    assert np.isclose(probabilities.sum(), 1.0)


def test_video_primary_and_patient_grouped_audit_are_deterministic_and_leakage_free():
    y = np.tile(
        np.asarray(
            ["eso_tropia", "exo_tropia", "no_manifest_horizontal_deviation"],
            dtype=object,
        ),
        12,
    )
    patients = np.repeat(np.arange(12), 3)
    first = make_evaluation_splits(y, patient_ids=patients, n_splits=3, random_state=99)
    second = make_evaluation_splits(y, patient_ids=patients, n_splits=3, random_state=99)
    for name in first:
        for (train_a, test_a), (train_b, test_b) in zip(first[name], second[name]):
            np.testing.assert_array_equal(train_a, train_b)
            np.testing.assert_array_equal(test_a, test_b)
    for train_index, test_index in first["patient_grouped_audit"]:
        assert set(patients[train_index]).isdisjoint(set(patients[test_index]))


def test_metrics_include_required_classification_pd_and_bland_altman_outputs():
    class_result = classification_metrics(
        ["eso", "exo", "normal", "eso"],
        ["eso", "normal", "normal", "exo"],
        labels=("eso", "exo", "normal"),
        normal_label="normal",
    )
    assert np.asarray(class_result["confusion_matrix"]).shape == (3, 3)
    assert 0 <= class_result["macro_f1"] <= 1
    assert 0 <= class_result["accuracy"] <= 1
    assert 0 <= class_result["balanced_accuracy"] <= 1
    assert set(class_result["sensitivity_by_class"]) == {"eso", "exo", "normal"}
    assert 0 <= class_result["deviation_sensitivity"] <= 1
    assert class_result["no_deviation_specificity"] == 1.0

    pd_result = pd_regression_metrics(
        [0.0, 10.0, 20.0], [0.0, 14.0, 24.0], [1.0, 12.0, 30.0]
    )
    required = {
        "mae_pd",
        "rmse_pd",
        "within_5_pd",
        "within_10_pd",
        "bland_altman_bias_pd",
        "bland_altman_lower_loa_pd",
        "bland_altman_upper_loa_pd",
        "interval_mae_pd",
    }
    assert required.issubset(pd_result)
    assert pd_result["interval_mae_pd"] <= pd_result["mae_pd"]

    interval_result = pd_prediction_interval_metrics(
        [0.0, 10.0], [0.0, 14.0], [0.0, 11.0], [2.0, 13.0]
    )
    assert interval_result["midpoint_coverage"] == 1.0
    assert interval_result["reference_interval_overlap_rate"] == 1.0
    assert interval_result["reference_interval_containment_rate"] == 0.5
    assert interval_result["mean_prediction_interval_width_pd"] == 2.0
