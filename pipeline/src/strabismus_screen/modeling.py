"""Deterministic models for horizontal cover-test screening.

The models in this module deliberately consume *engineered, event-aligned numeric
features*.  Video decoding, eye tracking, and feature extraction live upstream.
Keeping that boundary explicit prevents these small-data models from learning
patient identity, room, examiner, or occluder appearance directly from pixels.

All public prediction methods return a forced prediction.  Low confidence, poor
input quality, or lack of an observable movement is reported as a warning (and,
for Job 3, an ``indeterminate`` interpretation), never by silently dropping a
video.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import pickle
import re
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline

from .labels import (
    HorizontalDirection,
    LabelError,
    horizontal_direction,
    job1_manifest_label,
    job3_clinical_class,
)


MODEL_FORMAT_VERSION = 2
DIRECTION_CLASSES = (
    "eso_tropia",
    "exo_tropia",
    "no_manifest_horizontal_deviation",
)
TOTAL_DIRECTION_CLASSES = ("eso", "exo", "normal")
MANIFESTATION_CLASSES = (
    "tropia",
    "intermittent_tropia",
    "phoria",
    "normal",
)


def direction_label_from_diagnosis(
    horizontal_type: Any,
    test_type: Any = "Cover-Uncover",
) -> str | None:
    """Map a trusted horizontal diagnosis to the Job 1 target.

    Job 1 asks for the direction of *tropia*.  Consequently, a phoria is in the
    no-tropia/``normal`` class for this head; Job 3 retains the phoria label.
    Mixed-direction flick phorias are phorias too, so they are ``normal`` here.
    """

    try:
        label = job1_manifest_label(horizontal_type, test_type)
    except LabelError:
        return None
    return None if label is None else label.value


def manifestation_label_from_diagnosis(horizontal_type: Any) -> str | None:
    """Map dataset notation to the four Job 3 clinical classes."""

    try:
        label = job3_clinical_class(horizontal_type)
    except LabelError:
        return None
    return None if label is None else label.value


def total_direction_label_from_diagnosis(horizontal_type: Any) -> str | None:
    """Map any horizontal deviation (manifest or latent) to eso/exo/normal."""

    try:
        direction = horizontal_direction(horizontal_type)
    except LabelError:
        return None
    if direction is HorizontalDirection.MIXED:
        return None
    return direction.value


def parse_pd_interval(value: Any, *, normal_is_zero: bool = False) -> tuple[float, float] | None:
    """Parse a scalar or range prism-diopter label into ``(lower, upper)``.

    Examples include ``20``, ``"20"``, and ``"20-25"``.  Blank values remain
    missing unless the caller has already established that the diagnosis is
    normal and explicitly sets ``normal_is_zero=True``.
    """

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return (0.0, 0.0) if normal_is_zero else None
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        if number < 0 or not math.isfinite(number):
            raise ValueError(f"PD must be a finite non-negative magnitude, got {value!r}")
        return number, number
    text = str(value).strip()
    if not text:
        return (0.0, 0.0) if normal_is_zero else None
    text = text.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(?:-\s*(\d+(?:\.\d+)?))?\s*", text)
    if not match:
        raise ValueError(f"Cannot parse horizontal PD interval {value!r}")
    lower = float(match.group(1))
    upper = float(match.group(2)) if match.group(2) is not None else lower
    if upper < lower:
        raise ValueError(f"PD interval has upper < lower: {value!r}")
    return lower, upper


@dataclass(frozen=True)
class FeatureSchema:
    """Ordered numeric feature contract shared by the three task heads."""

    names: tuple[str, ...]

    @classmethod
    def infer(cls, X: Any, feature_names: Sequence[str] | None = None) -> "FeatureSchema":
        if feature_names is not None:
            names = tuple(str(name) for name in feature_names)
        elif hasattr(X, "columns"):
            names = tuple(str(name) for name in X.columns)
        else:
            array = np.asarray(X)
            if array.ndim != 2:
                raise ValueError("Engineered features must be a two-dimensional table")
            names = tuple(f"feature_{index}" for index in range(array.shape[1]))
        if not names or len(set(names)) != len(names):
            raise ValueError("Feature names must be non-empty and unique")
        return cls(names)

    def transform(self, X: Any) -> np.ndarray:
        if hasattr(X, "columns") and hasattr(X, "loc"):
            present = {str(name) for name in X.columns}
            missing = [name for name in self.names if name not in present]
            if missing:
                raise ValueError(f"Missing engineered features: {missing}")
            array = np.asarray(X.loc[:, list(self.names)], dtype=float)
        else:
            array = np.asarray(X, dtype=float)
        if array.ndim != 2 or array.shape[1] != len(self.names):
            raise ValueError(
                f"Expected {len(self.names)} engineered features, got shape {array.shape}"
            )
        return array


class DeterministicSerializable:
    """Stable-in-process pickle serialization with atomic file replacement."""

    model_format_version = MODEL_FORMAT_VERSION

    def dumps(self) -> bytes:
        return pickle.dumps(self, protocol=4)

    @classmethod
    def loads(cls, payload: bytes) -> Any:
        loaded = pickle.loads(payload)
        if not isinstance(loaded, cls):
            raise TypeError(f"Serialized payload contains {type(loaded)!r}, expected {cls!r}")
        return loaded

    def save(self, path: str | os.PathLike[str]) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = self.dumps()
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, destination)
        except Exception:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> Any:
        return cls.loads(Path(path).read_bytes())

    def metadata_json(self) -> str:
        metadata = {
            "class": type(self).__name__,
            "model_format_version": self.model_format_version,
            "random_state": getattr(self, "random_state", None),
        }
        return json.dumps(metadata, sort_keys=True, separators=(",", ":"))


class _ConstantClassifier:
    """Fold-safe classifier for a training subset containing one class."""

    def __init__(self, value: Any):
        self.value = value
        self.classes_ = np.asarray([value])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_ConstantClassifier":
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.ones((len(X), 1), dtype=float)


def _classifier_pipeline(
    *, random_state: int, class_weight: str | Mapping[Any, float] | None
) -> Pipeline:
    return Pipeline(
        [
            (
                "sanitize_raw_features",
                FunctionTransformer(
                    _sanitize_raw_features,
                    validate=False,
                    feature_names_out="one-to-one",
                ),
            ),
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            (
                "classifier",
                RandomForestClassifier(
                    class_weight=class_weight,
                    n_estimators=160,
                    min_samples_leaf=4,
                    max_depth=8,
                    max_features="sqrt",
                    random_state=random_state,
                    n_jobs=1,
                ),
            ),
        ]
    )


def _extra_trees_classifier_pipeline(
    *,
    random_state: int,
    class_weight: str | Mapping[Any, float] | None,
    min_samples_leaf: int = 5,
) -> Pipeline:
    """Stable high-dimensional classifier used by the Job 3 clinical head."""

    return Pipeline(
        [
            (
                "sanitize_raw_features",
                FunctionTransformer(
                    _sanitize_raw_features,
                    validate=False,
                    feature_names_out="one-to-one",
                ),
            ),
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            (
                "classifier",
                ExtraTreesClassifier(
                    class_weight=class_weight,
                    n_estimators=300,
                    min_samples_leaf=min_samples_leaf,
                    max_features=0.7,
                    random_state=random_state,
                    n_jobs=1,
                ),
            ),
        ]
    )


def _job1_manifest_classifier_pipeline(*, random_state: int) -> Pipeline:
    """Prespecified high-sensitivity Job 1 manifest-family classifier."""

    return Pipeline(
        [
            (
                "sanitize_raw_features",
                FunctionTransformer(
                    _sanitize_raw_features,
                    validate=False,
                    feature_names_out="one-to-one",
                ),
            ),
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                    keep_empty_features=True,
                ),
            ),
            (
                "classifier",
                ExtraTreesClassifier(
                    class_weight="balanced",
                    n_estimators=450,
                    min_samples_leaf=5,
                    max_features=0.65,
                    random_state=random_state,
                    n_jobs=1,
                ),
            ),
        ]
    )


def _hist_classifier_pipeline(
    *,
    random_state: int,
    class_weight: str | Mapping[Any, float] | None = "balanced",
) -> Pipeline:
    """Regularized nonlinear classifier for conditional eso/exo direction."""

    return Pipeline(
        [
            (
                "sanitize_raw_features",
                FunctionTransformer(
                    _sanitize_raw_features,
                    validate=False,
                    feature_names_out="one-to-one",
                ),
            ),
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    max_iter=130,
                    max_leaf_nodes=15,
                    min_samples_leaf=12,
                    learning_rate=0.05,
                    l2_regularization=5.0,
                    class_weight=class_weight,
                    random_state=random_state,
                ),
            ),
        ]
    )


def _positive_class_probability(estimator: Any, matrix: np.ndarray) -> np.ndarray:
    """Return P(class=1) for a fold-safe binary classifier."""

    raw = estimator.predict_proba(matrix)
    classes = list(estimator.classes_)
    if 1 not in classes:
        return np.zeros(len(matrix), dtype=float)
    return np.asarray(raw[:, classes.index(1)], dtype=float)


def _sanitize_raw_features(values: Any) -> np.ndarray:
    """Preserve missingness while bounding impossible engineered-feature values.

    The approved feature contract is dimensionless or a small count and valid
    values are orders of magnitude below this ceiling.  Clipping before model
    fitting also prevents malformed-but-finite inputs from overflowing an
    estimator's internal float32 representation.  Non-finite values remain
    missing so that the fold-fitted imputer, rather than a hard-coded zero,
    supplies them.
    """

    array = np.asarray(values, dtype=float)
    finite = np.isfinite(array)
    result = np.full(array.shape, np.nan, dtype=float)
    result[finite] = np.clip(array[finite], -1_000_000.0, 1_000_000.0)
    return result


def _quality_values(quality: Any, n_samples: int) -> np.ndarray:
    if quality is None:
        return np.full(n_samples, np.nan)
    values = np.asarray(quality, dtype=float)
    if values.ndim == 0:
        values = np.full(n_samples, float(values))
    if values.shape != (n_samples,):
        raise ValueError(f"quality must be scalar or length {n_samples}")
    return values


class _FixedClassHead(DeterministicSerializable):
    classes: tuple[str, ...]

    def _fit_classifier(
        self,
        X: Any,
        y: Sequence[str],
        *,
        feature_names: Sequence[str] | None,
        class_weight: str | Mapping[str, float] | None,
        sample_weight: Sequence[float] | None = None,
    ) -> None:
        self.feature_schema_ = FeatureSchema.infer(X, feature_names)
        matrix = self.feature_schema_.transform(X)
        labels = np.asarray(
            [value.value if hasattr(value, "value") else str(value) for value in y],
            dtype=object,
        )
        if labels.shape != (len(matrix),):
            raise ValueError("One class label is required per video")
        unknown = sorted(set(labels) - set(self.classes))
        if unknown:
            raise ValueError(f"Unknown labels: {unknown}; expected {list(self.classes)}")
        unique = np.unique(labels)
        if len(unique) == 1:
            self.estimator_ = _ConstantClassifier(unique[0]).fit(matrix, labels)
        else:
            if isinstance(class_weight, Mapping):
                class_weight = {label: float(class_weight.get(label, 1.0)) for label in unique}
            self.estimator_ = _classifier_pipeline(
                random_state=self.random_state, class_weight=class_weight
            )
            fit_kwargs: dict[str, Any] = {}
            if sample_weight is not None:
                weights = np.asarray(sample_weight, dtype=float)
                if weights.shape != (len(matrix),):
                    raise ValueError("sample_weight must have one value per video")
                fit_kwargs["classifier__sample_weight"] = weights
            self.estimator_.fit(matrix, labels, **fit_kwargs)

    def predict_proba(self, X: Any) -> np.ndarray:
        if not hasattr(self, "estimator_"):
            raise RuntimeError("Model is not fitted")
        matrix = self.feature_schema_.transform(X)
        raw = self.estimator_.predict_proba(matrix)
        result = np.zeros((len(matrix), len(self.classes)), dtype=float)
        class_to_index = {label: index for index, label in enumerate(self.classes)}
        for raw_index, label in enumerate(self.estimator_.classes_):
            if label in class_to_index:
                result[:, class_to_index[label]] = raw[:, raw_index]
        return result


class DirectionHead(_FixedClassHead):
    """Job 1: hierarchical manifest screen followed by eso/exo direction.

    Manifest presence and deviation direction have different class imbalance
    and different useful signals.  Keeping them as separate binary decisions
    prevents the former single three-class forest from collapsing toward the
    prevalent exo label while preserving an explicitly tunable screening
    sensitivity.
    """

    classes = DIRECTION_CLASSES

    def __init__(
        self,
        *,
        random_state: int = 1729,
        deviation_probability_threshold: float = 0.35,
        direction_probability_threshold: float = 0.50,
        deviation_class_weight: float = 2.0,
        manifest_binary_weight: float = 0.50,
        manifest_random_state_offset: int = 0,
        compact_direction_weight: float = 0.50,
        confidence_warning_threshold: float = 0.50,
        quality_warning_threshold: float = 0.55,
    ) -> None:
        if not 0 <= deviation_probability_threshold <= 1:
            raise ValueError("deviation_probability_threshold must be in [0, 1]")
        if not 0 <= direction_probability_threshold <= 1:
            raise ValueError("direction_probability_threshold must be in [0, 1]")
        if not 0 <= manifest_binary_weight <= 1:
            raise ValueError("manifest_binary_weight must be in [0, 1]")
        if not 0 <= compact_direction_weight <= 1:
            raise ValueError("compact_direction_weight must be in [0, 1]")
        self.random_state = int(random_state)
        self.deviation_probability_threshold = float(deviation_probability_threshold)
        self.direction_probability_threshold = float(direction_probability_threshold)
        # Retained in the public constructor for bundle/API compatibility.  The
        # hierarchical manifest component now uses fold-balanced weights.
        self.deviation_class_weight = float(deviation_class_weight)
        self.manifest_binary_weight = float(manifest_binary_weight)
        self.manifest_random_state_offset = int(manifest_random_state_offset)
        self.compact_direction_weight = float(compact_direction_weight)
        self.confidence_warning_threshold = float(confidence_warning_threshold)
        self.quality_warning_threshold = float(quality_warning_threshold)

    def fit(
        self,
        X: Any,
        y: Sequence[str],
        *,
        feature_names: Sequence[str] | None = None,
        sample_weight: Sequence[float] | None = None,
        clinical_y: Sequence[str] | None = None,
    ) -> "DirectionHead":
        self.feature_schema_ = FeatureSchema.infer(X, feature_names)
        matrix = self.feature_schema_.transform(X)
        labels = np.asarray(
            [value.value if hasattr(value, "value") else str(value) for value in y],
            dtype=object,
        )
        if labels.shape != (len(matrix),):
            raise ValueError("One class label is required per video")
        unknown = sorted(set(labels) - set(self.classes))
        if unknown:
            raise ValueError(f"Unknown labels: {unknown}; expected {list(self.classes)}")
        weights = None
        if sample_weight is not None:
            weights = np.asarray(sample_weight, dtype=float)
            if weights.shape != (len(matrix),):
                raise ValueError("sample_weight must have one value per video")

        manifest = (labels != "no_manifest_horizontal_deviation").astype(int)
        clinical: np.ndarray | None = None
        if clinical_y is not None:
            clinical = np.asarray(
                [
                    value.value if hasattr(value, "value") else str(value)
                    for value in clinical_y
                ],
                dtype=object,
            )
            if clinical.shape != (len(matrix),):
                raise ValueError("clinical_y must contain one label per video")
            unknown_clinical = sorted(set(clinical) - set(MANIFESTATION_CLASSES))
            if unknown_clinical:
                raise ValueError(
                    f"Unknown clinical labels: {unknown_clinical}; "
                    f"expected {list(MANIFESTATION_CLASSES)}"
                )

        # The dedicated binary gate excludes recovery/latent summaries that
        # can also be caused by a phoria.  Stable cover-on and same-iris rich
        # comparisons remain eligible.  The broader four-class auxiliary view
        # is fitted separately and supplies a diagnosis prior.
        if clinical is None:
            self.manifest_feature_indices_ = np.asarray(
                [
                    index
                    for index, name in enumerate(self.feature_schema_.names)
                    if not name.startswith("uncover_pair_")
                ],
                dtype=int,
            )
        else:
            forbidden_manifest_prefixes = (
                "uncover_",
                "uncover_pair_",
                "iris_pair_uncover_",
                "latent_or_total_",
                "all_",
                "observed_",
                "rich_uncover_",
            )
            self.manifest_feature_indices_ = np.asarray(
                [
                    index
                    for index, name in enumerate(self.feature_schema_.names)
                    if not name.startswith(forbidden_manifest_prefixes)
                ],
                dtype=int,
            )
        manifest_matrix = matrix[:, self.manifest_feature_indices_]
        if len(np.unique(manifest)) == 1:
            self.manifest_estimator_ = _ConstantClassifier(int(manifest[0])).fit(
                manifest_matrix, manifest
            )
        else:
            self.manifest_estimator_ = (
                _classifier_pipeline(
                    random_state=self.random_state,
                    class_weight="balanced",
                )
                if clinical is None
                else _job1_manifest_classifier_pipeline(
                    random_state=(
                        self.random_state + self.manifest_random_state_offset
                    )
                )
            )
            kwargs: dict[str, Any] = {}
            if weights is not None:
                kwargs["classifier__sample_weight"] = weights
            self.manifest_estimator_.fit(manifest_matrix, manifest, **kwargs)

        if clinical is not None:
            self.manifest_clinical_feature_indices_ = np.asarray(
                [
                    index
                    for index, name in enumerate(self.feature_schema_.names)
                    if not name.startswith("rich_")
                ],
                dtype=int,
            )
            clinical_matrix = matrix[:, self.manifest_clinical_feature_indices_]
            if len(np.unique(clinical)) == 1:
                self.manifest_clinical_estimator_ = _ConstantClassifier(
                    clinical[0]
                ).fit(clinical_matrix, clinical)
            else:
                self.manifest_clinical_estimator_ = (
                    _job1_manifest_classifier_pipeline(
                        random_state=(
                            self.random_state
                            + self.manifest_random_state_offset
                            + 19
                        )
                    )
                )
                kwargs = {}
                if weights is not None:
                    kwargs["classifier__sample_weight"] = weights
                self.manifest_clinical_estimator_.fit(
                    clinical_matrix, clinical, **kwargs
                )

        deviation_rows = manifest == 1
        eso = (labels[deviation_rows] == "eso_tropia").astype(int)
        # The compact delayed-uncover comparison is deliberately retained,
        # while the broad experimental ``iris_pair_`` expansion is excluded
        # from this low-prevalence conditional head.  The latter is useful to
        # the other tasks but diluted direction generalization with many
        # correlated early/late summaries.
        self.direction_feature_indices_ = np.asarray(
            [
                index
                for index, name in enumerate(self.feature_schema_.names)
                if not name.startswith(("iris_pair_", "rich_"))
            ],
            dtype=int,
        )
        direction_matrix = matrix[deviation_rows][
            :, self.direction_feature_indices_
        ]
        self.direction_broad_feature_indices_ = np.asarray(
            [
                index
                for index, name in enumerate(self.feature_schema_.names)
                if not name.startswith(("uncover_pair_", "rich_"))
            ],
            dtype=int,
        )
        broad_direction_matrix = matrix[deviation_rows][
            :, self.direction_broad_feature_indices_
        ]
        if not len(eso):
            self.direction_estimator_ = _ConstantClassifier(0).fit(
                direction_matrix, eso
            )
            self.direction_broad_estimator_ = _ConstantClassifier(0).fit(
                broad_direction_matrix, eso
            )
        elif len(np.unique(eso)) == 1:
            self.direction_estimator_ = _ConstantClassifier(int(eso[0])).fit(
                direction_matrix, eso
            )
            self.direction_broad_estimator_ = _ConstantClassifier(int(eso[0])).fit(
                broad_direction_matrix, eso
            )
        else:
            self.direction_estimator_ = _extra_trees_classifier_pipeline(
                random_state=self.random_state + 1,
                class_weight="balanced",
                min_samples_leaf=5,
            )
            kwargs = {}
            if weights is not None:
                kwargs["classifier__sample_weight"] = weights[deviation_rows]
            self.direction_estimator_.fit(direction_matrix, eso, **kwargs)
            self.direction_broad_estimator_ = _extra_trees_classifier_pipeline(
                random_state=self.random_state + 2,
                class_weight="balanced",
                min_samples_leaf=3,
            )
            self.direction_broad_estimator_.fit(
                broad_direction_matrix,
                eso,
                **kwargs,
            )
        return self

    def predict_manifest_probability(self, X: Any) -> np.ndarray:
        if not hasattr(self, "manifest_estimator_"):
            raise RuntimeError("Model is not fitted")
        matrix = self.feature_schema_.transform(X)
        indices = getattr(
            self,
            "manifest_feature_indices_",
            np.arange(matrix.shape[1], dtype=int),
        )
        binary_probability = _positive_class_probability(
            self.manifest_estimator_, matrix[:, indices]
        )
        if not hasattr(self, "manifest_clinical_estimator_"):
            return binary_probability
        clinical_indices = getattr(
            self,
            "manifest_clinical_feature_indices_",
            np.arange(matrix.shape[1], dtype=int),
        )
        clinical_matrix = matrix[:, clinical_indices]
        raw = self.manifest_clinical_estimator_.predict_proba(clinical_matrix)
        classes = list(self.manifest_clinical_estimator_.classes_)
        family_probability = np.zeros(len(matrix), dtype=float)
        for label in ("tropia", "intermittent_tropia"):
            if label in classes:
                family_probability += raw[:, classes.index(label)]
        weight = float(getattr(self, "manifest_binary_weight", 0.50))
        return weight * binary_probability + (1.0 - weight) * family_probability

    def predict_conditional_eso_probability(self, X: Any) -> np.ndarray:
        if not hasattr(self, "direction_estimator_"):
            raise RuntimeError("Model is not fitted")
        matrix = self.feature_schema_.transform(X)
        # Bundles written before the compact-pair feature revision have no
        # stored subset and continue to consume their original full schema.
        indices = getattr(
            self,
            "direction_feature_indices_",
            np.arange(matrix.shape[1], dtype=int),
        )
        compact_probability = _positive_class_probability(
            self.direction_estimator_, matrix[:, indices]
        )
        if not hasattr(self, "direction_broad_estimator_"):
            return compact_probability
        broad_indices = getattr(
            self,
            "direction_broad_feature_indices_",
            np.arange(matrix.shape[1], dtype=int),
        )
        broad_probability = _positive_class_probability(
            self.direction_broad_estimator_, matrix[:, broad_indices]
        )
        compact_weight = float(getattr(self, "compact_direction_weight", 1.0))
        return (
            compact_weight * compact_probability
            + (1.0 - compact_weight) * broad_probability
        )

    def predict_proba(self, X: Any) -> np.ndarray:
        manifest = self.predict_manifest_probability(X)
        eso_given_manifest = self.predict_conditional_eso_probability(X)
        return np.column_stack(
            (
                manifest * eso_given_manifest,
                manifest * (1.0 - eso_given_manifest),
                1.0 - manifest,
            )
        )

    def calibrate_deviation_threshold(
        self,
        X_validation: Any,
        y_validation: Sequence[str],
        *,
        target_sensitivity: float = 0.95,
    ) -> float:
        """Choose the most specific threshold meeting validation sensitivity."""

        if not 0 < target_sensitivity <= 1:
            raise ValueError("target_sensitivity must be in (0, 1]")
        probabilities = self.predict_proba(X_validation)
        labels = np.asarray(
            [
                value.value if hasattr(value, "value") else str(value)
                for value in y_validation
            ],
            dtype=object,
        )
        deviation = labels != "no_manifest_horizontal_deviation"
        if not np.any(deviation):
            raise ValueError("Validation labels contain no tropia examples")
        score = probabilities[:, 0] + probabilities[:, 1]
        candidates = np.unique(np.concatenate(([0.0], score, [1.0])))
        feasible = [
            threshold
            for threshold in candidates
            if np.mean(score[deviation] >= threshold) >= target_sensitivity
        ]
        self.deviation_probability_threshold = float(max(feasible))
        return self.deviation_probability_threshold

    def calibrate_direction_threshold(
        self,
        X_validation: Any,
        y_validation: Sequence[str],
    ) -> float:
        """Maximize balanced eso/exo recall on a validation partition."""

        labels = np.asarray(
            [
                value.value if hasattr(value, "value") else str(value)
                for value in y_validation
            ],
            dtype=object,
        )
        deviation = labels != "no_manifest_horizontal_deviation"
        if not np.any(deviation):
            raise ValueError("Validation labels contain no tropia examples")
        score = self.predict_conditional_eso_probability(X_validation)[deviation]
        truth = labels[deviation] == "eso_tropia"
        candidates = np.unique(np.concatenate(([0.0], score, [1.0])))
        best: tuple[float, float, float, float] | None = None
        for threshold in candidates:
            prediction = score >= threshold
            eso_recall = float(np.mean(prediction[truth])) if np.any(truth) else 0.0
            exo_recall = (
                float(np.mean(~prediction[~truth])) if np.any(~truth) else 0.0
            )
            balanced = (eso_recall + exo_recall) / 2.0
            accuracy = float(np.mean(prediction == truth))
            candidate = (balanced, accuracy, -abs(float(threshold) - 0.5), float(threshold))
            if best is None or candidate > best:
                best = candidate
        self.direction_probability_threshold = 0.5 if best is None else best[-1]
        return self.direction_probability_threshold

    def predict(self, X: Any) -> np.ndarray:
        probabilities = self.predict_proba(X)
        deviation_probability = probabilities[:, 0] + probabilities[:, 1]
        eso_probability = self.predict_conditional_eso_probability(X)
        labels = np.full(
            len(probabilities), "no_manifest_horizontal_deviation", dtype=object
        )
        selected = deviation_probability >= self.deviation_probability_threshold
        labels[selected] = np.where(
            eso_probability[selected] >= self.direction_probability_threshold,
            "eso_tropia",
            "exo_tropia",
        )
        return labels

    def predict_with_warnings(self, X: Any, *, quality: Any = None) -> list[dict[str, Any]]:
        probabilities = self.predict_proba(X)
        labels = self.predict(X)
        qualities = _quality_values(quality, len(labels))
        outputs: list[dict[str, Any]] = []
        for index, label in enumerate(labels):
            label_index = self.classes.index(str(label))
            confidence = float(probabilities[index, label_index])
            warnings: list[str] = []
            if confidence < self.confidence_warning_threshold:
                warnings.append("low_confidence")
            if np.isfinite(qualities[index]) and qualities[index] < self.quality_warning_threshold:
                warnings.append("low_quality")
            outputs.append(
                {
                    "forced_label": str(label),
                    "confidence": confidence,
                    "deviation_probability": float(probabilities[index, 0:2].sum()),
                    "conditional_eso_probability": float(
                        probabilities[index, 0]
                        / max(probabilities[index, 0:2].sum(), 1e-12)
                    ),
                    "probabilities": {
                        class_name: float(probabilities[index, class_index])
                        for class_index, class_name in enumerate(self.classes)
                    },
                    "quality": None if not np.isfinite(qualities[index]) else float(qualities[index]),
                    "warnings": warnings,
                }
            )
        return outputs


def _pd_estimator(*, random_state: int, n_estimators: int, min_samples_leaf: int) -> Pipeline:
    return Pipeline(
        [
            (
                "sanitize_raw_features",
                FunctionTransformer(
                    _sanitize_raw_features,
                    validate=False,
                    feature_names_out="one-to-one",
                ),
            ),
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            (
                "regressor",
                ExtraTreesRegressor(
                    n_estimators=n_estimators,
                    min_samples_leaf=min_samples_leaf,
                    max_features=0.7,
                    random_state=random_state,
                    n_jobs=1,
                ),
            ),
        ]
    )


def _higher_quantile(values: np.ndarray, quantile: float) -> float:
    try:
        return float(np.quantile(values, quantile, method="higher"))
    except TypeError:  # NumPy < 1.22
        return float(np.quantile(values, quantile, interpolation="higher"))


class PDMagnitudeHead(DeterministicSerializable):
    """Job 2: non-negative PD magnitude with interval-aware calibration.

    The regressor learns interval midpoints.  Out-of-fold residuals calibrate a
    symmetric prediction interval using the farthest annotated endpoint, so a
    range label such as 20--25 PD is not collapsed to a falsely exact target.
    Direction remains a separate categorical output.
    """

    def __init__(
        self,
        *,
        random_state: int = 1729,
        n_estimators: int = 240,
        min_samples_leaf: int = 3,
        calibration_folds: int = 5,
        interval_coverage: float = 0.90,
        quality_warning_threshold: float = 0.55,
        wide_interval_warning_pd: float = 20.0,
    ) -> None:
        if not 0 < interval_coverage < 1:
            raise ValueError("interval_coverage must be in (0, 1)")
        self.random_state = int(random_state)
        self.n_estimators = int(n_estimators)
        self.min_samples_leaf = int(min_samples_leaf)
        self.calibration_folds = int(calibration_folds)
        self.interval_coverage = float(interval_coverage)
        self.quality_warning_threshold = float(quality_warning_threshold)
        self.wide_interval_warning_pd = float(wide_interval_warning_pd)

    def _new_estimator(self, seed_offset: int = 0) -> Pipeline:
        return _pd_estimator(
            random_state=self.random_state + seed_offset,
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
        )

    def _new_deviation_estimator(self, seed_offset: int = 0) -> Pipeline:
        return _extra_trees_classifier_pipeline(
            random_state=self.random_state + 10_000 + seed_offset,
            class_weight="balanced",
            min_samples_leaf=3,
        )

    def _new_eso_estimator(self, seed_offset: int = 0) -> Pipeline:
        return _hist_classifier_pipeline(
            random_state=self.random_state + 20_000 + seed_offset,
            class_weight="balanced",
        )

    @staticmethod
    def _fit_binary_component(
        estimator: Any,
        matrix: np.ndarray,
        labels: np.ndarray,
        weights: np.ndarray | None,
    ) -> Any:
        """Fit a fold-safe binary component, including one-class folds."""

        if not len(labels):
            return _ConstantClassifier(0).fit(matrix, labels)
        unique = np.unique(labels)
        if len(unique) == 1:
            return _ConstantClassifier(int(unique[0])).fit(matrix, labels)
        kwargs: dict[str, Any] = {}
        if weights is not None:
            kwargs["classifier__sample_weight"] = weights
        estimator.fit(matrix, labels, **kwargs)
        return estimator

    @staticmethod
    def _direction_probabilities_from_components(
        matrix: np.ndarray,
        deviation_estimator: Any,
        eso_estimator: Any,
    ) -> np.ndarray:
        deviation = _positive_class_probability(deviation_estimator, matrix)
        eso_given_deviation = _positive_class_probability(eso_estimator, matrix)
        return np.column_stack(
            (
                deviation * eso_given_deviation,
                deviation * (1.0 - eso_given_deviation),
                1.0 - deviation,
            )
        )

    def _calibrate_joint_normal_policy(
        self,
        probabilities: np.ndarray,
        point_pd: np.ndarray,
        directions: np.ndarray,
        midpoint: np.ndarray,
    ) -> None:
        """Tune a coherent normal/0-delta gate on training-only OOF outputs.

        Direction probability and magnitude contain complementary evidence.  A
        clip is called normal only when both agree; all other clips retain the
        conditional eso/exo decision.  Accuracy is optimized first, followed
        by balanced recall and magnitude MAE, matching the deployment output.
        """

        best: tuple[float, float, float, float, float, float] | None = None
        best_values = (0.52, 12.5)
        for threshold in np.linspace(0.30, 0.90, 61):
            for magnitude_limit in (3.0, 5.0, 7.5, 10.0, 12.5, 15.0, 20.0):
                normal = (
                    probabilities[:, 2] >= threshold
                ) & (point_pd <= magnitude_limit)
                prediction = np.where(
                    normal,
                    "normal",
                    np.where(probabilities[:, 0] >= probabilities[:, 1], "eso", "exo"),
                )
                accuracy = float(np.mean(prediction == directions))
                recalls = [
                    float(np.mean(prediction[directions == label] == label))
                    for label in TOTAL_DIRECTION_CLASSES
                    if np.any(directions == label)
                ]
                balanced = float(np.mean(recalls)) if recalls else 0.0
                emitted = np.where(normal, 0.0, np.maximum(point_pd, 1.0))
                mae = float(np.mean(np.abs(emitted - midpoint)))
                score = (
                    accuracy,
                    balanced,
                    -mae,
                    -abs(float(threshold) - 0.52),
                    -abs(float(magnitude_limit) - 12.5),
                    float(threshold),
                )
                if best is None or score > best:
                    best = score
                    best_values = (float(threshold), float(magnitude_limit))
                    self.joint_policy_oof_direction_accuracy_ = accuracy
                    self.joint_policy_oof_direction_balanced_accuracy_ = balanced
                    self.joint_policy_oof_emitted_mae_pd_ = mae
        self.normal_probability_threshold_, self.normal_magnitude_limit_pd_ = best_values

    def fit(
        self,
        X: Any,
        lower_pd: Sequence[float],
        upper_pd: Sequence[float] | None = None,
        *,
        groups: Sequence[Any] | None = None,
        feature_names: Sequence[str] | None = None,
        sample_weight: Sequence[float] | None = None,
        direction_y: Sequence[str] | None = None,
    ) -> "PDMagnitudeHead":
        self.feature_schema_ = FeatureSchema.infer(X, feature_names)
        matrix = self.feature_schema_.transform(X)
        lower = np.asarray(lower_pd, dtype=float)
        upper = lower.copy() if upper_pd is None else np.asarray(upper_pd, dtype=float)
        if lower.shape != (len(matrix),) or upper.shape != lower.shape:
            raise ValueError("PD endpoints must have one value per video")
        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
            raise ValueError("PD endpoints must be finite")
        if np.any(lower < 0) or np.any(upper < lower):
            raise ValueError("PD intervals require 0 <= lower <= upper")
        midpoint = (lower + upper) / 2.0

        if sample_weight is not None:
            weights = np.asarray(sample_weight, dtype=float)
            if weights.shape != midpoint.shape:
                raise ValueError("sample_weight must have one value per video")
        else:
            weights = None

        n_samples = len(matrix)
        oof = np.empty(n_samples, dtype=float)
        use_oof = n_samples >= 4 and self.calibration_folds >= 2
        splits: list[tuple[np.ndarray, np.ndarray]] = []
        if use_oof and groups is not None:
            group_values = np.asarray(groups)
            if group_values.shape != (n_samples,):
                raise ValueError("groups must have one value per video")
            folds = min(self.calibration_folds, len(np.unique(group_values)))
            use_oof = folds >= 2
            if use_oof:
                splits = list(
                    GroupKFold(n_splits=folds).split(
                        matrix, midpoint, group_values
                    )
                )
                self.calibration_method_ = "grouped_oof"
        if use_oof and groups is None:
            folds = min(self.calibration_folds, n_samples)
            splits = list(
                KFold(
                    n_splits=folds,
                    shuffle=True,
                    random_state=self.random_state,
                ).split(matrix)
            )
            self.calibration_method_ = "video_oof"

        if use_oof:
            for fold, (train_index, validation_index) in enumerate(splits):
                estimator = self._new_estimator(seed_offset=fold + 1)
                kwargs: dict[str, Any] = {}
                if weights is not None:
                    kwargs["regressor__sample_weight"] = weights[train_index]
                estimator.fit(matrix[train_index], midpoint[train_index], **kwargs)
                oof[validation_index] = np.maximum(
                    0.0, estimator.predict(matrix[validation_index])
                )
        else:
            provisional = self._new_estimator(seed_offset=1)
            kwargs = {}
            if weights is not None:
                kwargs["regressor__sample_weight"] = weights
            provisional.fit(matrix, midpoint, **kwargs)
            oof[:] = np.maximum(0.0, provisional.predict(matrix))
            self.calibration_method_ = "in_sample_small_n"

        # Tree ensembles shrink the upper range on this modest dataset.  Learn
        # one robust affine correction exclusively from out-of-fold training
        # predictions, then use the same transformation at inference.
        self.point_calibration_slope_ = 1.0
        self.point_calibration_intercept_pd_ = 0.0
        self.point_calibration_method_ = "identity"
        if n_samples >= 8 and float(np.std(oof)) > 1e-8:
            try:
                calibrator = HuberRegressor(epsilon=1.5, alpha=1.0)
                calibrator.fit(
                    oof.reshape(-1, 1),
                    midpoint,
                    sample_weight=weights,
                )
                slope = float(calibrator.coef_[0])
                intercept = float(calibrator.intercept_)
                if np.isfinite(slope) and slope > 0.0 and np.isfinite(intercept):
                    self.point_calibration_slope_ = slope
                    self.point_calibration_intercept_pd_ = intercept
                    self.point_calibration_method_ = "oof_huber_affine"
            except (ValueError, FloatingPointError):
                pass
        oof = np.maximum(
            0.0,
            self.point_calibration_intercept_pd_
            + self.point_calibration_slope_ * oof,
        )

        # Radius contains the *whole* annotated target interval, not merely its midpoint.
        calibration_scores = np.maximum(np.abs(oof - lower), np.abs(oof - upper))
        conformal_level = min(
            1.0,
            math.ceil((n_samples + 1) * self.interval_coverage) / n_samples,
        )
        self.calibration_radius_pd_ = _higher_quantile(calibration_scores, conformal_level)
        self.oof_interval_coverage_ = float(
            np.mean(
                (lower >= np.maximum(0.0, oof - self.calibration_radius_pd_))
                & (upper <= oof + self.calibration_radius_pd_)
            )
        )

        self.estimator_ = self._new_estimator()
        kwargs = {}
        if weights is not None:
            kwargs["regressor__sample_weight"] = weights
        self.estimator_.fit(matrix, midpoint, **kwargs)

        if direction_y is not None:
            directions = np.asarray(
                [
                    value.value if hasattr(value, "value") else str(value)
                    for value in direction_y
                ],
                dtype=object,
            )
            if directions.shape != (n_samples,):
                raise ValueError("direction_y must have one value per video")
            unknown = sorted(set(directions) - set(TOTAL_DIRECTION_CLASSES))
            if unknown:
                raise ValueError(
                    f"Unknown total-deviation directions: {unknown}; "
                    f"expected {list(TOTAL_DIRECTION_CLASSES)}"
                )
            incoherent_normal = (directions == "normal") & (
                (lower != 0.0) | (upper != 0.0)
            )
            incoherent_deviation = (directions != "normal") & (upper == 0.0)
            if np.any(incoherent_normal) or np.any(incoherent_deviation):
                raise ValueError(
                    "Job 2 targets are incoherent: normal requires [0, 0] PD and "
                    "eso/exo require a non-zero upper bound"
                )
            deviation = (directions != "normal").astype(int)
            eso = (directions == "eso").astype(int)
            deviation_rows = deviation == 1

            oof_direction_probability = np.full(
                (n_samples, len(TOTAL_DIRECTION_CLASSES)), np.nan, dtype=float
            )
            if use_oof:
                for fold, (train_index, validation_index) in enumerate(splits):
                    fold_deviation = deviation[train_index]
                    fold_deviation_model = self._fit_binary_component(
                        self._new_deviation_estimator(seed_offset=fold + 1),
                        matrix[train_index],
                        fold_deviation,
                        None if weights is None else weights[train_index],
                    )
                    fold_deviation_rows = fold_deviation == 1
                    fold_eso_model = self._fit_binary_component(
                        self._new_eso_estimator(seed_offset=fold + 1),
                        matrix[train_index][fold_deviation_rows],
                        eso[train_index][fold_deviation_rows],
                        (
                            None
                            if weights is None
                            else weights[train_index][fold_deviation_rows]
                        ),
                    )
                    oof_direction_probability[validation_index] = (
                        self._direction_probabilities_from_components(
                            matrix[validation_index],
                            fold_deviation_model,
                            fold_eso_model,
                        )
                    )

            self.deviation_estimator_ = self._fit_binary_component(
                self._new_deviation_estimator(),
                matrix,
                deviation,
                weights,
            )
            self.eso_estimator_ = self._fit_binary_component(
                self._new_eso_estimator(),
                matrix[deviation_rows],
                eso[deviation_rows],
                None if weights is None else weights[deviation_rows],
            )
            if not use_oof:
                oof_direction_probability = (
                    self._direction_probabilities_from_components(
                        matrix,
                        self.deviation_estimator_,
                        self.eso_estimator_,
                    )
                )
            if not np.all(np.isfinite(oof_direction_probability)):
                raise RuntimeError("Incomplete Job 2 OOF direction probabilities")
            self._calibrate_joint_normal_policy(
                oof_direction_probability,
                oof,
                directions,
                midpoint,
            )
        else:
            self.normal_probability_threshold_ = 0.52
            self.normal_magnitude_limit_pd_ = 12.5
        return self

    def predict(self, X: Any) -> np.ndarray:
        if not hasattr(self, "estimator_"):
            raise RuntimeError("Model is not fitted")
        matrix = self.feature_schema_.transform(X)
        raw = np.asarray(self.estimator_.predict(matrix), dtype=float)
        return np.maximum(
            0.0,
            self.point_calibration_intercept_pd_
            + self.point_calibration_slope_ * raw,
        )

    def predict_interval(self, X: Any) -> np.ndarray:
        point = self.predict(X)
        return np.column_stack(
            (
                np.maximum(0.0, point - self.calibration_radius_pd_),
                point + self.calibration_radius_pd_,
            )
        )

    def predict_direction_proba(self, X: Any) -> np.ndarray:
        if not hasattr(self, "deviation_estimator_") or not hasattr(
            self, "eso_estimator_"
        ):
            raise RuntimeError("Total-deviation direction component is not fitted")
        matrix = self.feature_schema_.transform(X)
        return self._direction_probabilities_from_components(
            matrix,
            self.deviation_estimator_,
            self.eso_estimator_,
        )

    def predict_direction(self, X: Any) -> np.ndarray:
        probabilities = self.predict_direction_proba(X)
        point = self.predict(X)
        normal = (
            probabilities[:, 2] >= self.normal_probability_threshold_
        ) & (point <= self.normal_magnitude_limit_pd_)
        return np.where(
            normal,
            "normal",
            np.where(probabilities[:, 0] >= probabilities[:, 1], "eso", "exo"),
        ).astype(object)

    def predict_with_warnings(
        self,
        X: Any,
        *,
        quality: Any = None,
        direction: Sequence[str | None] | str | None = None,
    ) -> list[dict[str, Any]]:
        point = self.predict(X)
        intervals = self.predict_interval(X)
        qualities = _quality_values(quality, len(point))
        direction_probabilities: np.ndarray | None = None
        if direction is None and hasattr(self, "deviation_estimator_"):
            direction_probabilities = self.predict_direction_proba(X)
            directions = self.predict_direction(X).tolist()
        elif direction is None:
            directions = [None] * len(point)
        elif isinstance(direction, str):
            directions = [direction] * len(point)
        else:
            directions = list(direction)
            if len(directions) != len(point):
                raise ValueError("direction must have one value per video")
        outputs: list[dict[str, Any]] = []
        for index, raw_magnitude in enumerate(point):
            reported_magnitude = float(raw_magnitude)
            reported_interval = intervals[index].astype(float).copy()
            if directions[index] == "normal":
                # The point estimate is a joint clinical output.  Preserve the
                # regressor's upper uncertainty bound, but never emit the
                # incoherent combination `normal` plus a non-zero point value.
                reported_magnitude = 0.0
                reported_interval[0] = 0.0
            elif directions[index] in {"eso", "exo"} and raw_magnitude <= 0.5:
                # The clinical output must not pair a non-normal direction with
                # 0Δ.  One prism diopter is the smallest non-zero target in the
                # current development data; retain the raw regressor value and
                # flag this joint-output reconciliation explicitly.
                reported_magnitude = 1.0
                reported_interval[0] = min(float(reported_interval[0]), 1.0)
                reported_interval[1] = max(float(reported_interval[1]), 1.0)
            width = float(reported_interval[1] - reported_interval[0])
            confidence = float(1.0 / (1.0 + width / 10.0))
            warnings: list[str] = []
            if directions[index] is None:
                warnings.append("direction_not_provided")
            direction_confidence = (
                None
                if direction_probabilities is None
                else float(
                    direction_probabilities[
                        index,
                        TOTAL_DIRECTION_CLASSES.index(str(directions[index])),
                    ]
                )
            )
            if direction_confidence is not None and direction_confidence < 0.50:
                warnings.append("low_direction_confidence")
            if directions[index] == "normal" and raw_magnitude > 0.5:
                warnings.append("direction_magnitude_reconciled")
                warnings.append("joint_normal_gate_applied")
            elif directions[index] in {"eso", "exo"} and raw_magnitude <= 0.5:
                warnings.append("direction_magnitude_reconciled")
            if width > self.wide_interval_warning_pd:
                warnings.append("wide_prediction_interval")
            if np.isfinite(qualities[index]) and qualities[index] < self.quality_warning_threshold:
                warnings.append("low_quality")
            outputs.append(
                {
                    "forced_magnitude_pd": reported_magnitude,
                    "unreconciled_magnitude_pd": float(raw_magnitude),
                    "prediction_interval_pd": [
                        float(reported_interval[0]),
                        float(reported_interval[1]),
                    ],
                    "interval_coverage_target": self.interval_coverage,
                    "direction": directions[index],
                    "direction_confidence": direction_confidence,
                    "direction_probabilities": (
                        None
                        if direction_probabilities is None
                        else {
                            label: float(direction_probabilities[index, class_index])
                            for class_index, label in enumerate(TOTAL_DIRECTION_CLASSES)
                        }
                    ),
                    "confidence": confidence,
                    "quality": None if not np.isfinite(qualities[index]) else float(qualities[index]),
                    "warnings": warnings,
                }
            )
        return outputs


class ManifestationHead(_FixedClassHead):
    """Job 3: four clinical classes plus movement observability."""

    classes = MANIFESTATION_CLASSES

    def __init__(
        self,
        *,
        random_state: int = 1729,
        observable_no_threshold: float = 0.35,
        observable_yes_threshold: float = 0.65,
        observable_target_resolved_accuracy: float = 0.80,
        observable_calibration_folds: int = 5,
        confidence_warning_threshold: float = 0.50,
        quality_warning_threshold: float = 0.55,
    ) -> None:
        if not 0 <= observable_no_threshold < observable_yes_threshold <= 1:
            raise ValueError(
                "observable thresholds require 0 <= no < yes <= 1"
            )
        if not 0 < observable_target_resolved_accuracy <= 1:
            raise ValueError("observable_target_resolved_accuracy must be in (0, 1]")
        if observable_calibration_folds < 2:
            raise ValueError("observable_calibration_folds must be at least 2")
        self.random_state = int(random_state)
        self.observable_no_threshold = float(observable_no_threshold)
        self.observable_yes_threshold = float(observable_yes_threshold)
        self.observable_target_resolved_accuracy = float(
            observable_target_resolved_accuracy
        )
        self.observable_calibration_folds = int(observable_calibration_folds)
        self.confidence_warning_threshold = float(confidence_warning_threshold)
        self.quality_warning_threshold = float(quality_warning_threshold)

    def _calibrate_observable_thresholds(
        self,
        probabilities: np.ndarray,
        truth: np.ndarray,
    ) -> None:
        """Maximize resolved coverage at a requested training-only accuracy."""

        quantiles = np.quantile(probabilities, np.linspace(0.0, 1.0, 101))
        candidates = np.unique(np.r_[0.0, quantiles, 0.5, 1.0])
        lows = candidates[candidates <= 0.5]
        highs = candidates[candidates >= 0.5]
        feasible_best: tuple[float, float, float, float, float] | None = None
        fallback_best: tuple[float, float, float, float, float] | None = None
        chosen = (self.observable_no_threshold, self.observable_yes_threshold)
        for low in lows:
            for high in highs:
                if high <= low:
                    continue
                resolved = (probabilities <= low) | (probabilities >= high)
                coverage = float(np.mean(resolved))
                if not np.any(resolved):
                    continue
                prediction = probabilities[resolved] >= high
                accuracy = float(np.mean(prediction == truth[resolved]))
                tie = -abs(float(low) - 0.42) - abs(float(high) - 0.51)
                fallback_score = (accuracy, coverage, -float(high - low), tie, float(low))
                if fallback_best is None or fallback_score > fallback_best:
                    fallback_best = fallback_score
                if accuracy + 1e-12 < self.observable_target_resolved_accuracy:
                    continue
                score = (coverage, accuracy, -float(high - low), tie, float(low))
                if feasible_best is None or score > feasible_best:
                    feasible_best = score
                    chosen = (float(low), float(high))
        if feasible_best is None and fallback_best is not None:
            # If the target cannot be attained, select the most accurate
            # selective rule rather than silently retaining arbitrary defaults.
            best_accuracy = fallback_best[0]
            best_coverage = fallback_best[1]
            for low in lows:
                for high in highs:
                    if high <= low:
                        continue
                    resolved = (probabilities <= low) | (probabilities >= high)
                    if not np.any(resolved):
                        continue
                    prediction = probabilities[resolved] >= high
                    accuracy = float(np.mean(prediction == truth[resolved]))
                    coverage = float(np.mean(resolved))
                    if np.isclose(accuracy, best_accuracy) and np.isclose(
                        coverage, best_coverage
                    ):
                        chosen = (float(low), float(high))
                        break
                else:
                    continue
                break
        self.observable_no_threshold, self.observable_yes_threshold = chosen
        resolved = (probabilities <= chosen[0]) | (probabilities >= chosen[1])
        prediction = probabilities[resolved] >= chosen[1]
        self.observable_oof_resolved_fraction_ = float(np.mean(resolved))
        self.observable_oof_resolved_accuracy_ = (
            float(np.mean(prediction == truth[resolved]))
            if np.any(resolved)
            else float("nan")
        )

    def fit(
        self,
        X: Any,
        clinical_y: Sequence[str],
        observable_y: Sequence[int | bool],
        *,
        groups: Sequence[Any] | None = None,
        feature_names: Sequence[str] | None = None,
        sample_weight: Sequence[float] | None = None,
    ) -> "ManifestationHead":
        self.feature_schema_ = FeatureSchema.infer(X, feature_names)
        matrix = self.feature_schema_.transform(X)
        clinical = np.asarray(
            [
                value.value if hasattr(value, "value") else str(value)
                for value in clinical_y
            ],
            dtype=object,
        )
        if clinical.shape != (len(matrix),):
            raise ValueError("clinical_y must contain one label per video")
        unknown = sorted(set(clinical) - set(self.classes))
        if unknown:
            raise ValueError(
                f"Unknown labels: {unknown}; expected {list(self.classes)}"
            )
        weights = None
        if sample_weight is not None:
            weights = np.asarray(sample_weight, dtype=float)
            if weights.shape != (len(matrix),):
                raise ValueError("sample_weight must have one value per video")
        if len(np.unique(clinical)) == 1:
            self.estimator_ = _ConstantClassifier(clinical[0]).fit(matrix, clinical)
        else:
            self.estimator_ = _extra_trees_classifier_pipeline(
                random_state=self.random_state,
                class_weight="balanced",
                min_samples_leaf=5,
            )
            kwargs: dict[str, Any] = {}
            if weights is not None:
                kwargs["classifier__sample_weight"] = weights
            self.estimator_.fit(matrix, clinical, **kwargs)

        raw_observable = np.asarray(observable_y, dtype=object)
        if raw_observable.shape != (len(matrix),):
            raise ValueError("observable_y must contain one label per video")
        labelled = np.asarray(
            [
                value is not None
                and not (isinstance(value, (float, np.floating)) and math.isnan(value))
                for value in raw_observable
            ],
            dtype=bool,
        )
        if not np.any(labelled):
            # Conservative fold-safe fallback: the clinical prediction remains
            # available, while observability is forced to false and explicitly
            # warned at inference instead of borrowing information from a test fold.
            self.observable_estimator_ = _ConstantClassifier(0).fit(
                matrix[:0], np.asarray([], dtype=int)
            )
            self.observable_training_count_ = 0
            self.observable_calibration_method_ = "untrained_no_labels"
            self.observable_oof_resolved_fraction_ = 0.0
            self.observable_oof_resolved_accuracy_ = float("nan")
            return self
        observable = np.asarray(raw_observable[labelled], dtype=int)
        if not set(observable).issubset({0, 1}):
            raise ValueError("Known observable_y values must be binary")
        observable_matrix = matrix[labelled]
        observable_weights = None if weights is None else weights[labelled]
        observable_groups = None
        if groups is not None:
            group_values = np.asarray(groups)
            if group_values.shape != (len(matrix),):
                raise ValueError("groups must have one value per video")
            observable_groups = group_values[labelled]
        unique = np.unique(observable)
        if len(unique) == 1:
            self.observable_estimator_ = _ConstantClassifier(int(unique[0])).fit(
                observable_matrix, observable
            )
            self.observable_calibration_method_ = "constant_class"
            self.observable_oof_resolved_fraction_ = 1.0
            self.observable_oof_resolved_accuracy_ = 1.0
        else:
            splits: list[tuple[np.ndarray, np.ndarray]] = []
            if (
                observable_groups is not None
                and len(np.unique(observable_groups)) >= 2
            ):
                folds = min(
                    self.observable_calibration_folds,
                    len(np.unique(observable_groups)),
                )
                splits = list(
                    GroupKFold(n_splits=folds).split(
                        observable_matrix, observable, observable_groups
                    )
                )
                self.observable_calibration_method_ = "grouped_oof"
            else:
                class_counts = np.unique(observable, return_counts=True)[1]
                folds = min(
                    self.observable_calibration_folds,
                    int(class_counts.min()),
                )
                if folds >= 2:
                    splits = list(
                        StratifiedKFold(
                            n_splits=folds,
                            shuffle=True,
                            random_state=self.random_state,
                        ).split(observable_matrix, observable)
                    )
                    self.observable_calibration_method_ = "video_oof"
            if splits:
                oof_probability = np.full(len(observable), np.nan, dtype=float)
                for fold, (train_index, validation_index) in enumerate(splits):
                    fold_labels = observable[train_index]
                    if len(np.unique(fold_labels)) == 1:
                        fold_model: Any = _ConstantClassifier(
                            int(fold_labels[0])
                        ).fit(observable_matrix[train_index], fold_labels)
                    else:
                        fold_model = _classifier_pipeline(
                            random_state=self.random_state + fold + 1,
                            class_weight="balanced",
                        )
                        fold_kwargs: dict[str, Any] = {}
                        if observable_weights is not None:
                            fold_kwargs["classifier__sample_weight"] = (
                                observable_weights[train_index]
                            )
                        fold_model.fit(
                            observable_matrix[train_index],
                            fold_labels,
                            **fold_kwargs,
                        )
                    oof_probability[validation_index] = _positive_class_probability(
                        fold_model, observable_matrix[validation_index]
                    )
                if np.all(np.isfinite(oof_probability)):
                    self._calibrate_observable_thresholds(
                        oof_probability, observable.astype(bool)
                    )
            else:
                self.observable_calibration_method_ = "fixed_small_n"
            self.observable_estimator_ = _classifier_pipeline(
                random_state=self.random_state + 1, class_weight="balanced"
            )
            kwargs: dict[str, Any] = {}
            if observable_weights is not None:
                kwargs["classifier__sample_weight"] = observable_weights
            self.observable_estimator_.fit(observable_matrix, observable, **kwargs)
        self.observable_training_count_ = int(np.sum(labelled))
        return self

    def predict(self, X: Any) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return np.asarray(self.classes, dtype=object)[np.argmax(probabilities, axis=1)]

    def predict_observable_proba(self, X: Any) -> np.ndarray:
        if not hasattr(self, "observable_estimator_"):
            raise RuntimeError("Model is not fitted")
        matrix = self.feature_schema_.transform(X)
        raw = self.observable_estimator_.predict_proba(matrix)
        classes = list(self.observable_estimator_.classes_)
        if 1 not in classes:
            return np.zeros(len(matrix), dtype=float)
        return np.asarray(raw[:, classes.index(1)], dtype=float)

    def predict_with_warnings(self, X: Any, *, quality: Any = None) -> list[dict[str, Any]]:
        probabilities = self.predict_proba(X)
        labels = self.predict(X)
        observable_probability = self.predict_observable_proba(X)
        qualities = _quality_values(quality, len(labels))
        evidence_sufficient = np.ones(len(labels), dtype=bool)
        direct_evidence_features_present = False
        if hasattr(X, "columns") and hasattr(X, "loc"):
            required = {
                "verified_binocular_segment_observed",
                "cover_on_usable_n",
            }
            if required.issubset({str(column) for column in X.columns}):
                direct_evidence_features_present = True
                binocular = np.nan_to_num(
                    np.asarray(X["verified_binocular_segment_observed"], dtype=float),
                    nan=0.0,
                )
                usable_cover_on = np.nan_to_num(
                    np.asarray(X["cover_on_usable_n"], dtype=float), nan=0.0
                )
                state_evidence = np.zeros(len(labels), dtype=bool)
                state_columns = {
                    "binoc_right_nasal_n",
                    "left_cov_right_nasal_n",
                    "binoc_left_nasal_n",
                    "right_cov_left_nasal_n",
                }
                if state_columns.issubset(
                    {str(column) for column in X.columns}
                ):
                    binoc_right = np.nan_to_num(
                        np.asarray(X["binoc_right_nasal_n"], dtype=float), nan=0.0
                    )
                    left_covered_right = np.nan_to_num(
                        np.asarray(X["left_cov_right_nasal_n"], dtype=float),
                        nan=0.0,
                    )
                    binoc_left = np.nan_to_num(
                        np.asarray(X["binoc_left_nasal_n"], dtype=float), nan=0.0
                    )
                    right_covered_left = np.nan_to_num(
                        np.asarray(X["right_cov_left_nasal_n"], dtype=float),
                        nan=0.0,
                    )
                    state_evidence = (
                        (binoc_right >= 3.0) & (left_covered_right >= 3.0)
                    ) | ((binoc_left >= 3.0) & (right_covered_left >= 3.0))
                evidence_sufficient = (binocular >= 1.0) & (
                    (usable_cover_on >= 1.0) | state_evidence
                )
        outputs: list[dict[str, Any]] = []
        for index, label in enumerate(labels):
            label_index = self.classes.index(str(label))
            confidence = float(probabilities[index, label_index])
            warnings: list[str] = []
            forced_observable = bool(observable_probability[index] >= 0.50)
            if observable_probability[index] <= self.observable_no_threshold:
                observable_state = "no"
                observable: bool | None = False
            elif observable_probability[index] >= self.observable_yes_threshold:
                observable_state = "yes"
                observable = True
            else:
                observable_state = "indeterminate"
                observable = None
                warnings.append("observable_probability_indeterminate")
            if self.observable_training_count_ == 0:
                observable_state = "indeterminate"
                observable = None
                warnings.append("observable_model_untrained")
            if not evidence_sufficient[index]:
                observable_state = "indeterminate"
                observable = None
                warnings.append("insufficient_direct_manifest_observation_evidence")
            if confidence < self.confidence_warning_threshold:
                warnings.append("low_confidence")
            if np.isfinite(qualities[index]) and qualities[index] < self.quality_warning_threshold:
                warnings.append("low_quality")
                if not direct_evidence_features_present:
                    observable_state = "indeterminate"
                    observable = None
            indeterminate = observable_state == "indeterminate"
            outputs.append(
                {
                    "forced_label": str(label),
                    "clinical_class": str(label),
                    "interpretation": str(label),
                    "indeterminate": indeterminate,
                    "confidence": confidence,
                    "probabilities": {
                        class_name: float(probabilities[index, class_index])
                        for class_index, class_name in enumerate(self.classes)
                    },
                    "observable_in_clip": observable,
                    "forced_observable_in_clip": forced_observable,
                    "observable_state": observable_state,
                    "observable_probability": float(observable_probability[index]),
                    "observable_evidence_sufficient": bool(
                        evidence_sufficient[index]
                    ),
                    "quality": None if not np.isfinite(qualities[index]) else float(qualities[index]),
                    "warnings": warnings,
                }
            )
        return outputs


class ManifestationClinicalHead(DeterministicSerializable):
    """Clinical-only Job 3 head used in a probability blend.

    Observability remains owned by :class:`ManifestationHead`.  This head uses
    the exact compact-rich ExtraTrees specification evaluated in the Phase 2
    raw-nested experiments.
    """

    classes = MANIFESTATION_CLASSES

    def __init__(
        self,
        *,
        random_state: int = 1729,
        n_estimators: int = 400,
        min_samples_leaf: int = 5,
        max_features: float = 0.6,
    ) -> None:
        self.random_state = int(random_state)
        self.n_estimators = int(n_estimators)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = float(max_features)

    def fit(
        self,
        X: Any,
        clinical_y: Sequence[str],
        *,
        feature_names: Sequence[str] | None = None,
    ) -> "ManifestationClinicalHead":
        self.feature_schema_ = FeatureSchema.infer(X, feature_names)
        matrix = self.feature_schema_.transform(X)
        clinical = np.asarray(
            [
                value.value if hasattr(value, "value") else str(value)
                for value in clinical_y
            ],
            dtype=object,
        )
        if clinical.shape != (len(matrix),):
            raise ValueError("clinical_y must contain one label per video")
        unknown = sorted(set(clinical) - set(self.classes))
        if unknown:
            raise ValueError(
                f"Unknown labels: {unknown}; expected {list(self.classes)}"
            )
        if len(np.unique(clinical)) == 1:
            self.estimator_ = _ConstantClassifier(clinical[0]).fit(matrix, clinical)
        else:
            self.estimator_ = Pipeline(
                [
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median",
                            keep_empty_features=True,
                        ),
                    ),
                    (
                        "classifier",
                        ExtraTreesClassifier(
                            n_estimators=self.n_estimators,
                            min_samples_leaf=self.min_samples_leaf,
                            max_features=self.max_features,
                            class_weight="balanced",
                            random_state=self.random_state,
                            n_jobs=1,
                        ),
                    ),
                ]
            )
            self.estimator_.fit(matrix, clinical)
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        if not hasattr(self, "estimator_"):
            raise RuntimeError("Model is not fitted")
        matrix = self.feature_schema_.transform(X)
        raw = np.asarray(self.estimator_.predict_proba(matrix), dtype=float)
        present = [str(value) for value in self.estimator_.classes_]
        result = np.zeros((len(matrix), len(self.classes)), dtype=float)
        for index, label in enumerate(self.classes):
            if str(label) in present:
                result[:, index] = raw[:, present.index(str(label))]
        return result


class StrabismusModelBundle(DeterministicSerializable):
    """Three task heads bound to compatible engineered-feature schemas."""

    def __init__(
        self,
        *,
        direction_head: DirectionHead | None = None,
        pd_head: PDMagnitudeHead | None = None,
        pd_direction_head: PDMagnitudeHead | None = None,
        manifestation_head: ManifestationHead | None = None,
        manifestation_clinical_head: ManifestationClinicalHead | None = None,
        manifestation_blend_weight: float = 0.0,
        random_state: int = 1729,
    ) -> None:
        if not 0.0 <= manifestation_blend_weight <= 1.0:
            raise ValueError("manifestation_blend_weight must be in [0, 1]")
        self.random_state = int(random_state)
        self.direction_head = direction_head or DirectionHead(random_state=random_state)
        self.pd_head = pd_head or PDMagnitudeHead(random_state=random_state)
        # When present, this base-feature head supplies Job 2 direction while
        # ``pd_head`` may consume a richer magnitude-only feature schema.  Use
        # getattr at prediction time so bundles serialized before this field
        # was introduced remain loadable.
        self.pd_direction_head = pd_direction_head
        self.manifestation_head = manifestation_head or ManifestationHead(
            random_state=random_state
        )
        self.manifestation_clinical_head = manifestation_clinical_head
        self.manifestation_blend_weight = float(manifestation_blend_weight)

    def validate_shared_feature_schema(self) -> FeatureSchema:
        heads = [self.direction_head, self.pd_head, self.manifestation_head]
        pd_direction_head = getattr(self, "pd_direction_head", None)
        if pd_direction_head is not None:
            heads.append(pd_direction_head)
        manifestation_clinical_head = getattr(
            self, "manifestation_clinical_head", None
        )
        if manifestation_clinical_head is not None:
            heads.append(manifestation_clinical_head)
        schemas = [
            head.feature_schema_
            for head in heads
            if hasattr(head, "feature_schema_")
        ]
        if not schemas:
            raise RuntimeError("No task head is fitted")
        # A task may consume a strict superset (currently Job 1's richer
        # same-iris comparisons).  DataFrame-based FeatureSchema.transform
        # selects each head's declared columns, so a stable ordered union is a
        # safe bundle-level inference contract without exposing Job 1 features
        # to the Job 2/3 estimators.
        union = tuple(
            dict.fromkeys(
                name for schema in schemas for name in schema.names
            )
        )
        return FeatureSchema(union)

    def predict_job1(self, X: Any, *, quality: Any = None) -> list[dict[str, Any]]:
        return self.direction_head.predict_with_warnings(X, quality=quality)

    def predict_job2(
        self,
        X: Any,
        *,
        quality: Any = None,
        direction: Sequence[str | None] | str | None = None,
    ) -> list[dict[str, Any]]:
        pd_direction_head = getattr(self, "pd_direction_head", None)
        if direction is not None or pd_direction_head is None:
            return self.pd_head.predict_with_warnings(
                X, quality=quality, direction=direction
            )
        directions = pd_direction_head.predict_direction(X)
        probabilities = pd_direction_head.predict_direction_proba(X)
        outputs = self.pd_head.predict_with_warnings(
            X,
            quality=quality,
            direction=directions,
        )
        for index, output in enumerate(outputs):
            direction_label = str(directions[index])
            confidence = float(
                probabilities[
                    index,
                    TOTAL_DIRECTION_CLASSES.index(direction_label),
                ]
            )
            output["direction_confidence"] = confidence
            output["direction_probabilities"] = {
                label: float(probabilities[index, class_index])
                for class_index, label in enumerate(TOTAL_DIRECTION_CLASSES)
            }
            if confidence < 0.50 and "low_direction_confidence" not in output["warnings"]:
                output["warnings"].append("low_direction_confidence")
        return outputs

    def predict_job3(self, X: Any, *, quality: Any = None) -> list[dict[str, Any]]:
        outputs = self.manifestation_head.predict_with_warnings(X, quality=quality)
        clinical_head = getattr(self, "manifestation_clinical_head", None)
        weight = float(getattr(self, "manifestation_blend_weight", 0.0))
        if clinical_head is None or weight <= 0.0:
            return outputs
        baseline = self.manifestation_head.predict_proba(X)
        contender = clinical_head.predict_proba(X)
        probabilities = (1.0 - weight) * baseline + weight * contender
        labels = np.asarray(MANIFESTATION_CLASSES, dtype=object)[
            np.argmax(probabilities, axis=1)
        ]
        for index, output in enumerate(outputs):
            label = str(labels[index])
            confidence = float(np.max(probabilities[index]))
            output["forced_label"] = label
            output["clinical_class"] = label
            output["interpretation"] = label
            output["confidence"] = confidence
            output["probabilities"] = {
                class_name: float(probabilities[index, class_index])
                for class_index, class_name in enumerate(MANIFESTATION_CLASSES)
            }
            output["clinical_probability_blend_weight"] = weight
            output["clinical_probability_blend"] = "compact_rich"
            output["warnings"] = [
                warning for warning in output["warnings"] if warning != "low_confidence"
            ]
            if confidence < self.manifestation_head.confidence_warning_threshold:
                output["warnings"].append("low_confidence")
        return outputs


# Descriptive aliases for callers that prefer job-numbered names.
Job1DirectionModel = DirectionHead
Job2PDMagnitudeModel = PDMagnitudeHead
Job3ManifestationModel = ManifestationHead
