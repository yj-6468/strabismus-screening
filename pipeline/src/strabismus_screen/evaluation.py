"""Split construction and metrics for the three horizontal cover-test jobs.

The primary analysis treats every video as the requested analysis unit and uses
video-level stratified folds.  A second patient-grouped audit is always exposed
when patient identifiers are supplied; this estimates robustness to completely
unseen faces and visits and prevents repeated clips from creating a falsely
optimistic headline result.  Nested end-to-end cross-validation lives in
``evaluation_pipeline``; this module only builds splits and scores predictions.
"""

from __future__ import annotations

from collections.abc import Sequence
import math
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    recall_score,
)
from sklearn.model_selection import GroupKFold, StratifiedKFold

try:  # Available in supported modern scikit-learn, with a safe older fallback.
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # pragma: no cover - exercised only on old sklearn.
    StratifiedGroupKFold = None  # type: ignore[assignment]



def _one_dimensional(values: Any, *, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return array


def _safe_video_fold_count(y: np.ndarray, requested: int) -> int:
    _, counts = np.unique(y, return_counts=True)
    if len(counts) < 2:
        raise ValueError("Stratified evaluation requires at least two classes/strata")
    folds = min(int(requested), int(counts.min()))
    if folds < 2:
        raise ValueError("Each class/stratum needs at least two videos for CV")
    return folds


def _safe_group_fold_count(y: np.ndarray, groups: np.ndarray, requested: int) -> tuple[int, bool]:
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("Patient-grouped audit requires at least two patient IDs")
    group_counts_per_class = [len(np.unique(groups[y == label])) for label in np.unique(y)]
    stratified_folds = min(int(requested), min(group_counts_per_class))
    if stratified_folds >= 2:
        return stratified_folds, True
    return min(int(requested), len(unique_groups)), False


def make_evaluation_splits(
    strata: Sequence[Any],
    *,
    patient_ids: Sequence[Any] | None = None,
    recording_ids: Sequence[Any] | None = None,
    n_splits: int = 5,
    random_state: int = 1729,
) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    """Build the primary video CV and optional patient-grouped audit splits.

    ``recording_ids`` names the source recording of every clip.  Clips cut
    from one recording (the v2 ``_split1``/``_split2`` halves) are near
    duplicates, so when any recording holds more than one clip the video-level
    folds are grouped by recording.  With all-singleton recordings the original
    ungrouped partition is reproduced exactly.
    """

    y = _one_dimensional(strata, name="strata")
    video_folds = _safe_video_fold_count(y, n_splits)
    dummy = np.zeros((len(y), 1), dtype=float)
    recordings = None
    if recording_ids is not None:
        recordings = _one_dimensional(recording_ids, name="recording_ids")
        if recordings.shape != y.shape:
            raise ValueError("recording_ids must have one value per video")
        if len(np.unique(recordings)) == len(recordings):
            recordings = None
    if recordings is not None and StratifiedGroupKFold is not None:
        primary = list(
            StratifiedGroupKFold(
                n_splits=video_folds, shuffle=True, random_state=random_state
            ).split(dummy, y, recordings)
        )
        for train_index, test_index in primary:
            overlap = set(recordings[train_index]).intersection(recordings[test_index])
            if overlap:
                raise AssertionError(
                    f"Source recording straddles video folds: {sorted(overlap)!r}"
                )
    else:
        primary = list(
            StratifiedKFold(
                n_splits=video_folds, shuffle=True, random_state=random_state
            ).split(dummy, y)
        )
    result = {"video_level_stratified": primary}
    if patient_ids is None:
        return result

    groups = _one_dimensional(patient_ids, name="patient_ids")
    if groups.shape != y.shape:
        raise ValueError("patient_ids must have one value per video")
    group_folds, can_stratify = _safe_group_fold_count(y, groups, n_splits)
    if can_stratify and StratifiedGroupKFold is not None:
        splitter = StratifiedGroupKFold(
            n_splits=group_folds, shuffle=True, random_state=random_state
        )
        audit = list(splitter.split(dummy, y, groups))
    else:
        audit = list(GroupKFold(n_splits=group_folds).split(dummy, y, groups))
    for train_index, test_index in audit:
        overlap = set(groups[train_index]).intersection(groups[test_index])
        if overlap:
            raise AssertionError(f"Patient leakage in grouped audit: {sorted(overlap)!r}")
    result["patient_grouped_audit"] = audit
    return result


def classification_metrics(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    *,
    labels: Sequence[Any],
    normal_label: Any | None = None,
) -> dict[str, Any]:
    """Confusion, macro F1, balanced accuracy, and class sensitivities."""

    truth = _one_dimensional(y_true, name="y_true")
    prediction = _one_dimensional(y_pred, name="y_pred")
    if truth.shape != prediction.shape:
        raise ValueError("y_true and y_pred must have equal length")
    ordered_labels = list(labels)
    matrix = confusion_matrix(truth, prediction, labels=ordered_labels)
    sensitivities = recall_score(
        truth, prediction, labels=ordered_labels, average=None, zero_division=0
    )
    observed_truth_classes = matrix.sum(axis=1) > 0
    balanced_accuracy = (
        float(np.mean(sensitivities[observed_truth_classes]))
        if np.any(observed_truth_classes)
        else math.nan
    )
    result: dict[str, Any] = {
        "n_videos": int(len(truth)),
        "labels": ordered_labels,
        "confusion_matrix": matrix.astype(int).tolist(),
        "macro_f1": float(
            f1_score(
                truth,
                prediction,
                labels=ordered_labels,
                average="macro",
                zero_division=0,
            )
        ),
        "accuracy": float(accuracy_score(truth, prediction)),
        # Mean recall over classes actually present in this test partition.
        # This is algebraically the standard balanced accuracy while avoiding
        # sklearn's warning when a small domain-shift set contains predictions
        # from classes absent in its truth labels.
        "balanced_accuracy": balanced_accuracy,
        "sensitivity_by_class": {
            str(label): float(sensitivity)
            for label, sensitivity in zip(ordered_labels, sensitivities)
        },
        "macro_sensitivity": float(np.mean(sensitivities)),
    }
    if normal_label is not None:
        true_deviation = truth != normal_label
        predicted_deviation = prediction != normal_label
        positives = int(np.sum(true_deviation))
        negatives = int(np.sum(~true_deviation))
        result["deviation_sensitivity"] = (
            float(np.sum(true_deviation & predicted_deviation) / positives)
            if positives
            else math.nan
        )
        # In the binary screen abstraction, correctly retaining the normal/no-
        # manifest class is specificity.  The multiclass confusion matrix above
        # still preserves which deviation direction was confused.
        result["no_deviation_specificity"] = (
            float(np.sum((~true_deviation) & (~predicted_deviation)) / negatives)
            if negatives
            else math.nan
        )
    return result


def pd_regression_metrics(
    lower_pd: Sequence[float],
    upper_pd: Sequence[float] | None,
    predicted_pd: Sequence[float],
) -> dict[str, Any]:
    """Standard midpoint metrics plus errors respecting annotated PD ranges."""

    lower = np.asarray(lower_pd, dtype=float)
    upper = lower.copy() if upper_pd is None else np.asarray(upper_pd, dtype=float)
    prediction = np.asarray(predicted_pd, dtype=float)
    if lower.ndim != 1 or upper.shape != lower.shape or prediction.shape != lower.shape:
        raise ValueError("PD arrays must be one-dimensional and equally sized")
    if not (
        np.all(np.isfinite(lower))
        and np.all(np.isfinite(upper))
        and np.all(np.isfinite(prediction))
    ):
        raise ValueError("PD arrays must be finite")
    if np.any(lower < 0) or np.any(upper < lower):
        raise ValueError("PD intervals require 0 <= lower <= upper")

    midpoint = (lower + upper) / 2.0
    difference = prediction - midpoint
    interval_error = np.where(
        prediction < lower,
        lower - prediction,
        np.where(prediction > upper, prediction - upper, 0.0),
    )
    bias = float(np.mean(difference))
    difference_sd = float(np.std(difference, ddof=1)) if len(difference) > 1 else 0.0
    return {
        "n_videos": int(len(midpoint)),
        "mae_pd": float(mean_absolute_error(midpoint, prediction)),
        "rmse_pd": float(math.sqrt(mean_squared_error(midpoint, prediction))),
        "within_5_pd": float(np.mean(np.abs(difference) <= 5.0)),
        "within_10_pd": float(np.mean(np.abs(difference) <= 10.0)),
        "bland_altman_bias_pd": bias,
        "bland_altman_lower_loa_pd": float(bias - 1.96 * difference_sd),
        "bland_altman_upper_loa_pd": float(bias + 1.96 * difference_sd),
        "bland_altman_sd_pd": difference_sd,
        "interval_mae_pd": float(np.mean(interval_error)),
        "interval_rmse_pd": float(math.sqrt(np.mean(interval_error**2))),
        "within_annotated_interval": float(np.mean(interval_error == 0.0)),
        "interval_within_5_pd": float(np.mean(interval_error <= 5.0)),
        "interval_within_10_pd": float(np.mean(interval_error <= 10.0)),
        "mae_vs_interval_midpoint_pd": float(
            mean_absolute_error(midpoint, prediction)
        ),
        "within_5_vs_interval_midpoint": float(
            np.mean(np.abs(difference) <= 5.0)
        ),
        "within_10_vs_interval_midpoint": float(
            np.mean(np.abs(difference) <= 10.0)
        ),
        "bland_altman_reference": "annotated_interval_midpoint",
    }


def pd_prediction_interval_metrics(
    reference_lower_pd: Sequence[float],
    reference_upper_pd: Sequence[float],
    predicted_lower_pd: Sequence[float],
    predicted_upper_pd: Sequence[float],
) -> dict[str, Any]:
    """Held-out coverage and sharpness for model prediction intervals.

    The clinical reference may itself be interval censored, so midpoint
    coverage, any overlap, and containment of the whole annotated interval are
    reported separately rather than conflated.
    """

    reference_lower = np.asarray(reference_lower_pd, dtype=float)
    reference_upper = np.asarray(reference_upper_pd, dtype=float)
    predicted_lower = np.asarray(predicted_lower_pd, dtype=float)
    predicted_upper = np.asarray(predicted_upper_pd, dtype=float)
    shapes = {
        reference_lower.shape,
        reference_upper.shape,
        predicted_lower.shape,
        predicted_upper.shape,
    }
    if len(shapes) != 1 or reference_lower.ndim != 1:
        raise ValueError("Prediction/reference interval arrays must be equally sized")
    if not all(
        np.all(np.isfinite(values))
        for values in (
            reference_lower,
            reference_upper,
            predicted_lower,
            predicted_upper,
        )
    ):
        raise ValueError("Prediction/reference interval arrays must be finite")
    if np.any(reference_upper < reference_lower) or np.any(
        predicted_upper < predicted_lower
    ):
        raise ValueError("Interval upper bounds must not be below lower bounds")
    midpoint = (reference_lower + reference_upper) / 2.0
    contains_midpoint = (predicted_lower <= midpoint) & (
        midpoint <= predicted_upper
    )
    overlaps_reference = (predicted_lower <= reference_upper) & (
        predicted_upper >= reference_lower
    )
    contains_reference = (predicted_lower <= reference_lower) & (
        predicted_upper >= reference_upper
    )
    widths = predicted_upper - predicted_lower
    return {
        "n_videos": int(len(midpoint)),
        "midpoint_coverage": float(np.mean(contains_midpoint)),
        "reference_interval_overlap_rate": float(np.mean(overlaps_reference)),
        "reference_interval_containment_rate": float(np.mean(contains_reference)),
        "mean_prediction_interval_width_pd": float(np.mean(widths)),
        "median_prediction_interval_width_pd": float(np.median(widths)),
    }


def warning_summary(outputs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for output in outputs:
        for warning in output.get("warnings", []):
            counts[str(warning)] = counts.get(str(warning), 0) + 1
    n = len(outputs)
    return {
        "n_videos": n,
        "forced_output_rate": 1.0 if n else math.nan,
        "any_warning_rate": (
            float(np.mean([bool(output.get("warnings", [])) for output in outputs]))
            if n
            else math.nan
        ),
        "warning_counts": dict(sorted(counts.items())),
    }
