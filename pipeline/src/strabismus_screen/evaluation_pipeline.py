"""Leakage-safe end-to-end evaluation from uniformly sampled video observations.

The cover-state detector is a learned part of the pipeline, not preprocessing
that may be fitted once before cross-validation.  Consequently every outer
fold in this module fits :class:`CoverEventModel` using frame annotations from
the outer-training videos only.  Both training and held-out task features are
then constructed from that fold's *predicted* cover timeline.  Trusted
held-out frame annotations are joined only after feature construction, and
only to report a separately labelled event-detector diagnostic.

The resulting estimates therefore model deployment from raw video plus the
declared test metadata.  No task head receives an oracle event flag.
"""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .data import IngestedDataset
from .evaluation import (
    classification_metrics,
    make_evaluation_splits,
    pd_prediction_interval_metrics,
    pd_regression_metrics,
    warning_summary,
)
from .events import CoverEventModel
from .modeling import (
    DIRECTION_CLASSES,
    MANIFESTATION_CLASSES,
    TOTAL_DIRECTION_CLASSES,
    DirectionHead,
    ManifestationHead,
    PDMagnitudeHead,
)
from .pipeline import (
    build_cover_training_rows,
    build_predicted_feature_table,
    frame_annotation_table,
    load_cached_observations,
    select_job1_feature_names,
    select_task_feature_names,
    validate_observation_sampling,
)


EVALUATION_REPORT_VERSION = 1


def _enum_value(value: Any) -> str:
    return str(value.value if hasattr(value, "value") else value)


def _json_safe(value: Any) -> Any:
    """Convert numpy values and non-finite floats into strict JSON values."""

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _uid_digest(video_uids: Iterable[str]) -> str:
    joined = "\n".join(sorted(str(uid) for uid in video_uids)).encode("utf-8")
    return sha256(joined).hexdigest()


def _manifest_strata(dataset: IngestedDataset) -> np.ndarray:
    """Build useful strata without consulting PD magnitude or frame events."""

    combined = np.asarray(
        [
            f"{record.test_type}|{_enum_value(record.job3_clinical_class)}"
            for record in dataset.manifest
        ],
        dtype=object,
    )
    # The validated dataset has adequate counts for the combined strata.  The
    # fallbacks make small synthetic/development subsets evaluable as well.
    candidates = (
        combined,
        np.asarray([record.test_type for record in dataset.manifest], dtype=object),
        np.asarray(
            [_enum_value(record.job3_clinical_class) for record in dataset.manifest],
            dtype=object,
        ),
    )
    for candidate in candidates:
        _, counts = np.unique(candidate, return_counts=True)
        if len(counts) >= 2 and int(counts.min()) >= 2:
            return candidate
    raise ValueError("Evaluation needs at least two strata with two videos each")


def make_end_to_end_splits(
    dataset: IngestedDataset,
    *,
    n_splits: int = 5,
    random_state: int = 20260824,
    include_video_level: bool = True,
    include_patient_grouped: bool = True,
    include_cohort_holdout: bool = True,
    group_recordings: bool = False,
    stratify_by_distance: bool = False,
) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    """Create outer video, patient, and optional temporal-cohort splits.

    ``group_recordings`` keeps clips cut from one source recording in the same
    video-level fold; ``stratify_by_distance`` adds near/distance to the fold
    strata.  Both default to off so earlier partitions stay reproducible.
    """

    splits: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    if include_video_level or include_patient_grouped:
        strata = _manifest_strata(dataset)
        if stratify_by_distance:
            with_distance = np.asarray(
                [f"{s}|{r.distance}" for s, r in zip(strata, dataset.manifest)],
                dtype=object,
            )
            _, counts = np.unique(with_distance, return_counts=True)
            if int(counts.min()) >= n_splits:
                strata = with_distance
        patients = (
            [record.patient_id for record in dataset.manifest]
            if include_patient_grouped
            else None
        )
        recordings = (
            [
                f"{r.patient_id}|{r.exam_date.isoformat()}|{r.condition}|{r.test_type}|{r.repeat}"
                for r in dataset.manifest
            ]
            if group_recordings
            else None
        )
        cross_validation = make_evaluation_splits(
            strata,
            patient_ids=patients,
            recording_ids=recordings,
            n_splits=n_splits,
            random_state=random_state,
        )
        if include_video_level:
            splits["video_level_stratified"] = cross_validation[
                "video_level_stratified"
            ]
        if include_patient_grouped:
            splits["patient_grouped_audit"] = cross_validation[
                "patient_grouped_audit"
            ]
    if include_cohort_holdout:
        cohorts = np.asarray([record.cohort for record in dataset.manifest], dtype=object)
        older = np.flatnonzero(cohorts == "older")
        new = np.flatnonzero(cohorts == "new")
        if len(older) and len(new):
            splits["older_to_new_cohort_holdout"] = [(older, new)]
    return splits


def _tune_direction_threshold(
    X: pd.DataFrame,
    y: np.ndarray,
    clinical_y: np.ndarray | None = None,
    *,
    patient_ids: np.ndarray | None,
    random_state: int,
    target_sensitivity: float,
) -> tuple[float, float, dict[str, Any]]:
    """Tune Job 1 only inside the current outer-training partition."""

    default = 0.35
    positive = y != "no_manifest_horizontal_deviation"
    if len(y) < 4 or not np.any(positive) or len(np.unique(y)) < 2:
        return default, 0.50, {
            "method": "fixed_default_insufficient_inner_training_data",
            "deviation_threshold": default,
            "direction_threshold": 0.50,
            "target_deviation_sensitivity": target_sensitivity,
        }
    try:
        split_sets = make_evaluation_splits(
            y,
            patient_ids=patient_ids,
            n_splits=3,
            random_state=random_state,
        )
        split_name = (
            "patient_grouped_audit"
            if patient_ids is not None
            else "video_level_stratified"
        )
        splits = split_sets[split_name]
    except ValueError:
        return default, 0.50, {
            "method": "fixed_default_inner_split_unavailable",
            "deviation_threshold": default,
            "direction_threshold": 0.50,
            "target_deviation_sensitivity": target_sensitivity,
        }

    manifest_splits = splits
    if clinical_y is not None:
        manifest_split_sets = make_evaluation_splits(
            y,
            patient_ids=patient_ids,
            n_splits=3,
            random_state=random_state + 101,
        )
        manifest_splits = manifest_split_sets[split_name]

    oof_probability = np.full(len(y), np.nan, dtype=float)
    oof_eso_probability = np.full(len(y), np.nan, dtype=float)
    for inner_fold, (inner_train, inner_validation) in enumerate(splits):
        model = DirectionHead(random_state=random_state + inner_fold + 1)
        model.fit(
            X.iloc[inner_train],
            y[inner_train],
        )
        if clinical_y is None:
            probabilities = model.predict_proba(X.iloc[inner_validation])
            oof_probability[inner_validation] = probabilities[:, :2].sum(axis=1)
        oof_eso_probability[inner_validation] = (
            model.predict_conditional_eso_probability(X.iloc[inner_validation])
        )
    if clinical_y is not None:
        for inner_fold, (inner_train, inner_validation) in enumerate(
            manifest_splits
        ):
            model = DirectionHead(random_state=random_state + inner_fold + 1)
            model.fit(
                X.iloc[inner_train],
                y[inner_train],
                clinical_y=clinical_y[inner_train],
            )
            probabilities = model.predict_proba(X.iloc[inner_validation])
            oof_probability[inner_validation] = probabilities[:, :2].sum(axis=1)
    if not np.all(np.isfinite(oof_probability[positive])):
        return default, 0.50, {
            "method": "fixed_default_incomplete_inner_predictions",
            "deviation_threshold": default,
            "direction_threshold": 0.50,
            "target_deviation_sensitivity": target_sensitivity,
        }
    candidates = np.unique(np.r_[0.0, oof_probability[np.isfinite(oof_probability)], 1.0])
    feasible = [
        float(threshold)
        for threshold in candidates
        if np.mean(oof_probability[positive] >= threshold) >= target_sensitivity
    ]
    threshold = max(feasible, default=0.0)
    direction_truth = y[positive] == "eso_tropia"
    direction_score = oof_eso_probability[positive]
    direction_candidates = np.unique(np.r_[0.0, direction_score, 1.0])
    direction_best: tuple[float, float, float, float] | None = None
    direction_threshold = 0.50
    for candidate_threshold in direction_candidates:
        direction_prediction = direction_score >= candidate_threshold
        eso_recall = (
            float(np.mean(direction_prediction[direction_truth]))
            if np.any(direction_truth)
            else 0.0
        )
        exo_recall = (
            float(np.mean(~direction_prediction[~direction_truth]))
            if np.any(~direction_truth)
            else 0.0
        )
        balanced = (eso_recall + exo_recall) / 2.0
        accuracy = float(np.mean(direction_prediction == direction_truth))
        score = (
            balanced,
            accuracy,
            -abs(float(candidate_threshold) - 0.5),
            float(candidate_threshold),
        )
        if direction_best is None or score > direction_best:
            direction_best = score
            direction_threshold = float(candidate_threshold)
    return threshold, direction_threshold, {
        "method": (
            "patient_grouped_inner_oof" if patient_ids is not None else "video_inner_oof"
        ),
        "deviation_threshold": threshold,
        "direction_threshold": direction_threshold,
        "target_deviation_sensitivity": target_sensitivity,
        "achieved_inner_deviation_sensitivity": float(
            np.mean(oof_probability[positive] >= threshold)
        ),
        "achieved_inner_conditional_direction_balanced_accuracy": (
            None if direction_best is None else float(direction_best[0])
        ),
        "inner_fold_count": len(splits),
    }


def _feature_matrices(
    table: pd.DataFrame,
    train_uids: set[str],
) -> tuple[tuple[str, ...], pd.DataFrame, pd.DataFrame]:
    if "video_uid" not in table:
        raise ValueError("Predicted feature table is missing video_uid")
    if table["video_uid"].duplicated().any():
        duplicates = table.loc[table["video_uid"].duplicated(), "video_uid"].tolist()
        raise ValueError(f"Predicted feature table has duplicate videos: {duplicates[:5]}")
    training_table = table[table["video_uid"].isin(train_uids)].copy()
    test_table = table[~table["video_uid"].isin(train_uids)].copy()
    names = select_task_feature_names(training_table)
    return names, training_table, test_table


def _event_test_diagnostic(
    cover_model: CoverEventModel,
    heldout_rows: pd.DataFrame,
) -> dict[str, Any]:
    truth = heldout_rows["covered"].to_numpy(dtype=int)
    probability = cover_model.predict_probability(heldout_rows)
    prediction = (probability >= 0.5).astype(int)
    metrics = classification_metrics(truth, prediction, labels=(0, 1))
    metrics.update(
        {
            "unit": "sampled_eye_frame",
            "threshold": 0.5,
            "heldout_video_count": int(heldout_rows["video"].nunique()),
            "diagnostic_only_not_a_video_level_head": True,
        }
    )
    return metrics


def _job1_records(
    training_table: pd.DataFrame,
    test_table: pd.DataFrame,
    feature_names: Sequence[str],
    *,
    grouped_inner_threshold: bool,
    random_state: int,
    target_sensitivity: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    train = training_table[training_table["job1_label"].notna()].copy()
    test = test_table[test_table["job1_label"].notna()].copy()
    if train.empty:
        raise ValueError("Outer training fold contains no eligible Job 1 videos")
    if test.empty:
        return [], {"status": "no_eligible_test_videos"}
    job1_feature_names = select_job1_feature_names(training_table)
    X_train = train.loc[:, list(job1_feature_names)]
    y_train = train["job1_label"].to_numpy(dtype=object)
    clinical_train = train["job3_label"].to_numpy(dtype=object)
    threshold, direction_threshold, tuning = _tune_direction_threshold(
        X_train.reset_index(drop=True),
        y_train,
        clinical_train,
        patient_ids=(
            train["patient_id_group"].to_numpy(dtype=object)
            if grouped_inner_threshold
            else None
        ),
        random_state=random_state,
        target_sensitivity=target_sensitivity,
    )
    model = DirectionHead(
        random_state=random_state,
        manifest_random_state_offset=77,
    )
    model.fit(X_train, y_train, clinical_y=clinical_train)
    model.deviation_probability_threshold = threshold
    model.direction_probability_threshold = direction_threshold
    quality = test["technical_quality_score"].to_numpy(dtype=float)
    outputs = model.predict_with_warnings(
        test.loc[:, list(job1_feature_names)], quality=quality
    )
    records = []
    for (_, row), output in zip(test.iterrows(), outputs):
        records.append(
            {
                "video_uid": str(row["video_uid"]),
                "truth": str(row["job1_label"]),
                "prediction": output["forced_label"],
                "deviation_probability": output["deviation_probability"],
                "conditional_eso_probability": output[
                    "conditional_eso_probability"
                ],
                "conditional_direction_prediction": (
                    "eso_tropia"
                    if output["conditional_eso_probability"] >= direction_threshold
                    else "exo_tropia"
                ),
                "direction_probability_threshold": direction_threshold,
                "probabilities": output["probabilities"],
                "warnings": output["warnings"],
            }
        )
    return records, tuning


def _job2_records(
    training_table: pd.DataFrame,
    test_table: pd.DataFrame,
    feature_names: Sequence[str],
    *,
    use_patient_groups: bool,
    random_state: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    def eligible(table: pd.DataFrame) -> pd.Series:
        return (
            (table["test_type"] == "Alternate-Cover")
            & table["pd_lower"].notna()
            & table["pd_upper"].notna()
            & table["total_direction_label"].notna()
        )

    train = training_table[eligible(training_table)].copy()
    test = test_table[eligible(test_table)].copy()
    if train.empty:
        raise ValueError("Outer training fold contains no eligible Job 2 videos")
    if test.empty:
        return [], {"status": "no_eligible_test_videos"}
    lower = train["pd_lower"].to_numpy(dtype=float)
    upper = train["pd_upper"].to_numpy(dtype=float)
    direction = train["total_direction_label"].to_numpy(dtype=object)
    model = PDMagnitudeHead(random_state=random_state)
    model.fit(
        train.loc[:, list(feature_names)],
        lower,
        upper,
        groups=(
            train["patient_id_group"].to_numpy(dtype=object)
            if use_patient_groups
            else None
        ),
        direction_y=direction,
    )
    quality = test["technical_quality_score"].to_numpy(dtype=float)
    outputs = model.predict_with_warnings(
        test.loc[:, list(feature_names)], quality=quality
    )
    records: list[dict[str, Any]] = []
    for (_, row), output in zip(test.iterrows(), outputs):
        records.append(
            {
                "video_uid": str(row["video_uid"]),
                "truth_lower_pd": float(row["pd_lower"]),
                "truth_upper_pd": float(row["pd_upper"]),
                "truth_direction": str(row["total_direction_label"]),
                # These are the coherent values actually emitted by the pipeline.
                "predicted_pd": float(output["forced_magnitude_pd"]),
                "unreconciled_predicted_pd": float(
                    output["unreconciled_magnitude_pd"]
                ),
                "predicted_lower_pd": float(output["prediction_interval_pd"][0]),
                "predicted_upper_pd": float(output["prediction_interval_pd"][1]),
                "predicted_direction": str(output["direction"]),
                "direction_probabilities": output["direction_probabilities"],
                "warnings": output["warnings"],
            }
        )
    return records, {
        "status": "fitted",
        "training_video_count": len(train),
        "calibration_method": model.calibration_method_,
        "point_calibration_method": model.point_calibration_method_,
        "point_calibration_slope": float(model.point_calibration_slope_),
        "point_calibration_intercept_pd": float(
            model.point_calibration_intercept_pd_
        ),
        "calibration_radius_pd": float(model.calibration_radius_pd_),
        "interval_coverage_target": float(model.interval_coverage),
        "normal_probability_threshold": float(
            model.normal_probability_threshold_
        ),
        "normal_magnitude_limit_pd": float(model.normal_magnitude_limit_pd_),
        "joint_policy_oof_direction_accuracy": float(
            model.joint_policy_oof_direction_accuracy_
        ),
        "joint_policy_oof_direction_balanced_accuracy": float(
            model.joint_policy_oof_direction_balanced_accuracy_
        ),
        "joint_policy_oof_emitted_mae_pd": float(
            model.joint_policy_oof_emitted_mae_pd_
        ),
    }


def _job3_records(
    training_table: pd.DataFrame,
    test_table: pd.DataFrame,
    feature_names: Sequence[str],
    *,
    use_patient_groups: bool,
    random_state: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    train = training_table[
        (training_table["test_type"] == "Cover-Uncover")
        & training_table["job3_label"].notna()
    ].copy()
    test = test_table[
        (test_table["test_type"] == "Cover-Uncover")
        & test_table["job3_label"].notna()
    ].copy()
    if train.empty:
        raise ValueError("Outer training fold contains no eligible Job 3 videos")
    if test.empty:
        return [], {"status": "no_eligible_test_videos"}
    model = ManifestationHead(random_state=random_state)
    model.fit(
        train.loc[:, list(feature_names)],
        train["job3_label"].to_numpy(dtype=object),
        train["observable_label"].to_numpy(dtype=object),
        groups=(
            train["patient_id_group"].to_numpy(dtype=object)
            if use_patient_groups
            else None
        ),
    )
    quality = test["technical_quality_score"].to_numpy(dtype=float)
    outputs = model.predict_with_warnings(
        test.loc[:, list(feature_names)], quality=quality
    )
    records: list[dict[str, Any]] = []
    for (_, row), output in zip(test.iterrows(), outputs):
        raw_truth = row["observable_label"]
        truth_observable = (
            None
            if raw_truth is None or pd.isna(raw_truth)
            else ("yes" if bool(raw_truth) else "no")
        )
        records.append(
            {
                "video_uid": str(row["video_uid"]),
                "truth_clinical_class": str(row["job3_label"]),
                "predicted_clinical_class": output["forced_label"],
                "clinical_probabilities": output["probabilities"],
                "truth_observable_in_clip": truth_observable,
                "predicted_observable_in_clip": output["observable_state"],
                "forced_observable_in_clip": (
                    "yes" if output["forced_observable_in_clip"] else "no"
                ),
                "observable_probability": output["observable_probability"],
                "warnings": output["warnings"],
            }
        )
    return records, {
        "status": "fitted",
        "training_video_count": len(train),
        "observable_training_video_count": int(model.observable_training_count_),
        "observable_calibration_method": model.observable_calibration_method_,
        "observable_no_threshold": float(model.observable_no_threshold),
        "observable_yes_threshold": float(model.observable_yes_threshold),
        "observable_oof_resolved_fraction": float(
            model.observable_oof_resolved_fraction_
        ),
        "observable_oof_resolved_accuracy": float(
            model.observable_oof_resolved_accuracy_
        ),
    }


def _job1_summary(records: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    if not records:
        return None
    result = classification_metrics(
        [record["truth"] for record in records],
        [record["prediction"] for record in records],
        labels=DIRECTION_CLASSES,
        normal_label="no_manifest_horizontal_deviation",
    )
    manifest = [
        record
        for record in records
        if record["truth"] != "no_manifest_horizontal_deviation"
        and record.get("conditional_direction_prediction") is not None
    ]
    if manifest:
        result["conditional_eso_exo_on_true_tropias"] = classification_metrics(
            [record["truth"] for record in manifest],
            [record["conditional_direction_prediction"] for record in manifest],
            labels=("eso_tropia", "exo_tropia"),
        )
    result["warnings"] = warning_summary(records)
    return result


def _job2_summary(records: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    if not records:
        return None
    lower = [record["truth_lower_pd"] for record in records]
    upper = [record["truth_upper_pd"] for record in records]
    emitted = [record["predicted_pd"] for record in records]
    raw = [record["unreconciled_predicted_pd"] for record in records]
    predicted_lower = [record["predicted_lower_pd"] for record in records]
    predicted_upper = [record["predicted_upper_pd"] for record in records]
    return {
        "emitted_magnitude": pd_regression_metrics(lower, upper, emitted),
        "prediction_intervals": pd_prediction_interval_metrics(
            lower, upper, predicted_lower, predicted_upper
        ),
        "direction": classification_metrics(
            [record["truth_direction"] for record in records],
            [record["predicted_direction"] for record in records],
            labels=TOTAL_DIRECTION_CLASSES,
            normal_label="normal",
        ),
        "unreconciled_regressor_diagnostic": pd_regression_metrics(lower, upper, raw),
        "warnings": warning_summary(records),
        "target_semantics": "interval_censored_prism_diopters",
        "point_reference_for_mae_and_bland_altman": "annotated_interval_midpoint",
    }


def _observable_summary(records: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    known = [record for record in records if record["truth_observable_in_clip"] is not None]
    if not known:
        return None
    truth_order = ("no", "yes")
    prediction_order = ("no", "yes", "indeterminate")
    matrix = [
        [
            sum(
                record["truth_observable_in_clip"] == truth
                and record["predicted_observable_in_clip"] == prediction
                for record in known
            )
            for prediction in prediction_order
        ]
        for truth in truth_order
    ]
    forced = classification_metrics(
        [record["truth_observable_in_clip"] for record in known],
        [record["forced_observable_in_clip"] for record in known],
        labels=truth_order,
    )
    resolved = [
        record
        for record in known
        if record["predicted_observable_in_clip"] != "indeterminate"
    ]
    return {
        "known_truth_video_count": len(known),
        "truth_labels": list(truth_order),
        "prediction_labels": list(prediction_order),
        "tri_state_confusion_matrix": matrix,
        "indeterminate_rate_on_known_truth": float(
            np.mean(
                [
                    record["predicted_observable_in_clip"] == "indeterminate"
                    for record in known
                ]
            )
        ),
        "resolved_accuracy": (
            float(
                np.mean(
                    [
                        record["truth_observable_in_clip"]
                        == record["predicted_observable_in_clip"]
                        for record in resolved
                    ]
                )
            )
            if resolved
            else None
        ),
        "forced_binary_diagnostic": forced,
    }


def _job3_summary(records: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    if not records:
        return None
    return {
        "clinical_class": classification_metrics(
            [record["truth_clinical_class"] for record in records],
            [record["predicted_clinical_class"] for record in records],
            labels=MANIFESTATION_CLASSES,
            normal_label="normal",
        ),
        "observable_in_clip": _observable_summary(records),
        "all_video_indeterminate_rate": float(
            np.mean(
                [
                    record["predicted_observable_in_clip"] == "indeterminate"
                    for record in records
                ]
            )
        ),
        "warnings": warning_summary(records),
    }


def _check_pooled_unique(records: Sequence[dict[str, Any]], *, task: str) -> None:
    identifiers = [record["video_uid"] for record in records]
    if len(identifiers) != len(set(identifiers)):
        raise AssertionError(f"{task} received duplicate out-of-fold predictions")


def _evaluate_split_mode(
    dataset: IngestedDataset,
    observations: dict[str, Any],
    annotations: pd.DataFrame,
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    split_name: str,
    config: PipelineConfig,
    target_deviation_sensitivity: float,
    progress: Callable[[dict[str, Any]], None] | None,
) -> dict[str, Any]:
    manifest = list(dataset.manifest)
    pooled_job1: list[dict[str, Any]] = []
    pooled_job2: list[dict[str, Any]] = []
    pooled_job3: list[dict[str, Any]] = []
    fold_reports: list[dict[str, Any]] = []
    grouped = split_name == "patient_grouped_audit"

    for fold_number, (train_index, test_index) in enumerate(splits):
        train_records = [manifest[int(index)] for index in train_index]
        test_records = [manifest[int(index)] for index in test_index]
        train_uids = {record.video_uid for record in train_records}
        test_uids = {record.video_uid for record in test_records}
        if train_uids & test_uids:
            raise AssertionError("Outer evaluation train/test video leakage")
        train_patients = {record.patient_id for record in train_records}
        test_patients = {record.patient_id for record in test_records}
        if grouped and train_patients & test_patients:
            raise AssertionError("Patient leakage in patient-grouped outer fold")

        cover_training_rows = build_cover_training_rows(
            dataset,
            observations,
            annotations,
            allowed_video_uids=train_uids,
        )
        annotation_training_uids = set(
            cover_training_rows["video"].astype(str).unique()
        )
        annotation_leakage = annotation_training_uids & test_uids
        if annotation_leakage:
            raise AssertionError(
                f"Held-out frame annotations entered event training: {annotation_leakage}"
            )
        cover_model = CoverEventModel(random_seed=config.random_seed + fold_number)
        cover_training_diagnostics = cover_model.fit(cover_training_rows)

        # This is the decisive no-oracle boundary: both outer-train and
        # outer-test features are generated from the fold-fitted predictions.
        feature_table = build_predicted_feature_table(
            dataset,
            observations,
            cover_model,
            event_config=config.event,
            motion_config=config.motion,
            video_uids=train_uids | test_uids,
        )
        feature_names, training_table, test_table = _feature_matrices(
            feature_table, train_uids
        )
        if set(test_table["video_uid"].astype(str)) != test_uids:
            missing = test_uids - set(test_table["video_uid"].astype(str))
            raise ValueError(f"Predicted feature table omitted held-out videos: {missing}")

        fold_seed = config.random_seed + 1_000 * (fold_number + 1)
        job1, threshold_tuning = _job1_records(
            training_table,
            test_table,
            feature_names,
            grouped_inner_threshold=grouped,
            random_state=fold_seed + 1,
            target_sensitivity=target_deviation_sensitivity,
        )
        job2, pd_fitting = _job2_records(
            training_table,
            test_table,
            feature_names,
            use_patient_groups=grouped,
            random_state=fold_seed + 2,
        )
        job3, manifestation_fitting = _job3_records(
            training_table,
            test_table,
            feature_names,
            use_patient_groups=grouped,
            random_state=fold_seed + 3,
        )

        # Test annotations are joined only now, after all test feature and task
        # predictions have been made.  They never influence pipeline output.
        heldout_cover_rows = build_cover_training_rows(
            dataset,
            observations,
            annotations,
            allowed_video_uids=test_uids,
        )
        event_test = _event_test_diagnostic(cover_model, heldout_cover_rows)

        pooled_job1.extend(job1)
        pooled_job2.extend(job2)
        pooled_job3.extend(job3)
        fold_report = {
            "fold": fold_number,
            "outer_training_video_count": len(train_uids),
            "outer_test_video_count": len(test_uids),
            "outer_training_patient_count": len(train_patients),
            "outer_test_patient_count": len(test_patients),
            "outer_patient_overlap_count": len(train_patients & test_patients),
            "training_video_uid_digest": _uid_digest(train_uids),
            "test_video_uid_digest": _uid_digest(test_uids),
            "event_annotation_training_video_count": len(annotation_training_uids),
            "event_annotation_test_leakage_count": 0,
            "cover_event_training_diagnostics_resubstitution_only": asdict(
                cover_training_diagnostics
            ),
            "cover_event_heldout_diagnostic": event_test,
            "approved_feature_count": len(feature_names),
            "job1": {
                "test_video_count": len(job1),
                "threshold_tuning": threshold_tuning,
                "metrics": _job1_summary(job1),
            },
            "job2": {
                "test_video_count": len(job2),
                "fitting": pd_fitting,
                "metrics": _job2_summary(job2),
            },
            "job3": {
                "test_video_count": len(job3),
                "fitting": manifestation_fitting,
                "metrics": _job3_summary(job3),
            },
        }
        fold_reports.append(fold_report)
        if progress is not None:
            progress(
                {
                    "status": "completed_evaluation_fold",
                    "split": split_name,
                    "fold": fold_number,
                    "fold_count": len(splits),
                    "training_video_count": len(train_uids),
                    "test_video_count": len(test_uids),
                }
            )

    _check_pooled_unique(pooled_job1, task="Job 1")
    _check_pooled_unique(pooled_job2, task="Job 2")
    _check_pooled_unique(pooled_job3, task="Job 3")
    return {
        "split_role": (
            "primary"
            if split_name == "video_level_stratified"
            else (
                "stricter_patient_sensitivity_analysis"
                if split_name == "patient_grouped_audit"
                else "acquisition_domain_shift_analysis"
            )
        ),
        "fold_count": len(splits),
        "job1_direction_of_manifest_tropia": {
            "overall": _job1_summary(pooled_job1),
            "oof_predictions": pooled_job1,
        },
        "job2_total_horizontal_prism_diopters": {
            "overall": _job2_summary(pooled_job2),
            "oof_predictions": pooled_job2,
        },
        "job3_manifestation_and_observability": {
            "overall": _job3_summary(pooled_job3),
            "oof_predictions": pooled_job3,
        },
        "folds": fold_reports,
    }


def evaluate_pipeline(
    dataset: IngestedDataset,
    *,
    cache_directory: str | Path,
    config: PipelineConfig | None = None,
    n_splits: int = 5,
    include_patient_grouped: bool = True,
    include_cohort_holdout: bool = True,
    modes: Sequence[str] | None = None,
    target_deviation_sensitivity: float = 0.95,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Evaluate all three jobs with the cover model nested in every outer fold."""

    if n_splits < 2:
        raise ValueError("n_splits must be at least two")
    if not 0 < target_deviation_sensitivity <= 1:
        raise ValueError("target_deviation_sensitivity must be in (0, 1]")
    aliases = {
        "video": "video_level_stratified",
        "patient": "patient_grouped_audit",
        "cohort": "older_to_new_cohort_holdout",
        "video_level_stratified": "video_level_stratified",
        "patient_grouped_audit": "patient_grouped_audit",
        "older_to_new_cohort_holdout": "older_to_new_cohort_holdout",
    }
    if modes is None:
        selected_modes = {"video_level_stratified"}
        if include_patient_grouped:
            selected_modes.add("patient_grouped_audit")
        if include_cohort_holdout:
            selected_modes.add("older_to_new_cohort_holdout")
    else:
        unknown = sorted(set(modes) - set(aliases))
        if unknown:
            raise ValueError(f"Unknown evaluation modes: {unknown}")
        selected_modes = {aliases[mode] for mode in modes}
        if not selected_modes:
            raise ValueError("At least one evaluation mode is required")
    config = config or PipelineConfig()
    observations = load_cached_observations(
        dataset,
        cache_directory,
        model_path=config.face_landmarker_model,
        stride=config.event.sample_stride,
        maximum_faces=config.maximum_faces,
    )
    validate_observation_sampling(
        observations, expected_stride=config.event.sample_stride
    )
    annotations = frame_annotation_table(dataset)
    splits = make_end_to_end_splits(
        dataset,
        n_splits=n_splits,
        random_state=config.random_seed,
        include_video_level="video_level_stratified" in selected_modes,
        include_patient_grouped="patient_grouped_audit" in selected_modes,
        include_cohort_holdout="older_to_new_cohort_holdout" in selected_modes,
    )
    results: dict[str, Any] = {}
    for split_name, mode_splits in splits.items():
        results[split_name] = _evaluate_split_mode(
            dataset,
            observations,
            annotations,
            mode_splits,
            split_name=split_name,
            config=config,
            target_deviation_sensitivity=target_deviation_sensitivity,
            progress=progress,
        )
    report = {
        "report_version": EVALUATION_REPORT_VERSION,
        "analysis_unit": "video",
        "evaluation_mode": "raw_video_inferred_events_no_frame_annotations_at_inference",
        "primary_analysis": (
            "video_level_stratified"
            if "video_level_stratified" in results
            else None
        ),
        "secondary_patient_sensitivity_analysis": (
            "patient_grouped_audit" if "patient_grouped_audit" in results else None
        ),
        "optional_temporal_domain_shift_analysis": (
            "older_to_new_cohort_holdout"
            if "older_to_new_cohort_holdout" in results
            else None
        ),
        "video_count": len(dataset.manifest),
        "video_uid_manifest_digest": _uid_digest(
            record.video_uid for record in dataset.manifest
        ),
        "configuration": config.to_dict(),
        "event_model_nesting": (
            "refit_on_outer_training_video_frame_annotations_for_every_fold"
        ),
        "task_feature_event_source": "fold_fitted_cover_event_predictions_only",
        "outer_training_task_feature_event_mode": (
            "resubstitution_predictions_from_outer_training_fitted_event_model"
        ),
        "oracle_event_features_used": False,
        "heldout_frame_annotations_used_as_task_inputs": False,
        "heldout_frame_annotations_use": (
            "post_prediction_cover_event_diagnostic_only"
        ),
        "job1_threshold_tuning": (
            "inner_oof_within_each_outer_training_partition_only"
        ),
        "target_deviation_sensitivity": target_deviation_sensitivity,
        "normal_blank_horizontal_pd_semantics": "zero_prism_diopters",
        "pd_range_target_semantics": "interval_censored_not_replaced_in_scoring",
        "pd_calibration_mode": "dataset_learned_fallback_not_geometric",
        "results": results,
    }
    return _json_safe(report)


def write_evaluation_report(report: dict[str, Any], path: str | Path) -> Path:
    """Atomically write a strict-JSON evaluation artifact."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(_json_safe(report), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return destination


def evaluation_console_summary(report: dict[str, Any]) -> dict[str, Any]:
    """Return headline metrics without repeating all video-level predictions."""

    modes: dict[str, Any] = {}
    for name, result in report["results"].items():
        modes[name] = {
            "split_role": result["split_role"],
            "fold_count": result["fold_count"],
            "job1": result["job1_direction_of_manifest_tropia"]["overall"],
            "job2": result["job2_total_horizontal_prism_diopters"]["overall"],
            "job3": result["job3_manifestation_and_observability"]["overall"],
        }
    return {
        "evaluation_mode": report["evaluation_mode"],
        "oracle_event_features_used": report["oracle_event_features_used"],
        "primary_analysis": report["primary_analysis"],
        "results": modes,
    }


__all__ = [
    "EVALUATION_REPORT_VERSION",
    "evaluate_pipeline",
    "evaluation_console_summary",
    "make_end_to_end_splits",
    "write_evaluation_report",
]
