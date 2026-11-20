"""End-to-end orchestration helpers for the research screening pipeline.

The raw observation cache is deliberately upstream of every annotation.  Each
video is sampled at one fixed inference stride; trusted event labels are joined
later, during model development, and therefore cannot influence which frames a
held-out clip contributes.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Iterable

import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

from .config import EventConfig, MotionConfig, PipelineConfig
from .data import FrameAnnotation, IngestedDataset, VideoRecord, load_dataset
from .dark_iris import dark_iris_feature_names, dark_iris_feature_schema, dark_iris_features
from .events import CoverEventModel, attach_cover_targets, infer_cover_timeline, long_eye_features
from .features import (
    aggregate_video_features,
    clinical_motion_evidence,
    extract_motion_cycles,
)
from .job1_rich_features import job1_rich_feature_names, job1_rich_features
from .labels import HorizontalDirection
from .native_features import native_feature_names, native_feature_schema, native_features
from .modeling import (
    DirectionHead,
    ManifestationClinicalHead,
    ManifestationHead,
    PDMagnitudeHead,
    StrabismusModelBundle,
)
from .vision import (
    FaceLandmarkerBackend,
    VideoMetadata,
    VisionResult,
    extract_video_observations,
)


OBSERVATION_CACHE_VERSION = 5
SUPPORTED_OBSERVATION_CACHE_VERSIONS = (3, 4, 5)
PIPELINE_BUNDLE_VERSION = 4


def _enum_value(value: Any) -> Any:
    return value.value if hasattr(value, "value") else value


def _normalize_distance(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized not in {"near", "distance"}:
        raise ValueError("distance must be 'near' or 'distance'")
    return normalized


def _normalize_correction(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "cc": "with_correction",
        "with_correction": "with_correction",
        "sc": "without_correction",
        "without_correction": "without_correction",
    }
    if normalized not in aliases:
        raise ValueError(
            "correction must be 'with_correction'/'cc' or "
            "'without_correction'/'sc'"
        )
    return aliases[normalized]


def dataset_audit(dataset: IngestedDataset) -> dict[str, Any]:
    """Return a JSON-safe audit of the validated source joins and targets."""

    manifest = dataset.manifest
    job1 = Counter(
        _enum_value(record.job1_manifest_label)
        for record in manifest
        if record.job1_manifest_label is not None
    )
    job3 = Counter(
        _enum_value(record.job3_clinical_class)
        for record in manifest
        if record.job3_clinical_class is not None
    )
    tests = Counter(record.test_type for record in manifest)
    cohorts = Counter(record.cohort for record in manifest)
    pd_eligible = [
        record
        for record in manifest
        if record.test_type == "Alternate-Cover"
        and record.horizontal_pd is not None
        and _enum_value(record.horizontal_direction) != "mixed"
    ]
    ranged_pd = [
        record for record in pd_eligible if record.horizontal_pd_lower != record.horizontal_pd_upper
    ]
    return {
        "analysis_unit": "video",
        "video_count": dataset.validation.video_count,
        "unique_subject_uid_count": len({record.subject_uid for record in manifest}),
        "patient_id_count_for_grouped_audit": len(
            {record.patient_id for record in manifest}
        ),
        "event_count": dataset.validation.event_count,
        "annotated_frame_count": dataset.validation.frame_count,
        "cover_episode_count": dataset.validation.cover_episode_count,
        "clips_beginning_covered": dataset.validation.clips_beginning_covered,
        "right_censored_cover_events": (
            dataset.validation.right_censored_cover_events
        ),
        "test_type_counts": dict(sorted(tests.items())),
        "cohort_counts": dict(sorted(cohorts.items())),
        "job1_eligible_video_count": int(sum(job1.values())),
        "job1_label_counts": dict(sorted(job1.items())),
        "job2_primary_eligible_video_count": len(pd_eligible),
        "job2_interval_censored_video_count": len(ranged_pd),
        "job3_primary_cover_uncover_video_count": int(
            sum(record.test_type == "Cover-Uncover" for record in manifest)
        ),
        "job3_all_clinical_label_counts": dict(sorted(job3.items())),
        "mixed_flick_policy": "job3_phoria_job1_job2_excluded",
        "normal_blank_horizontal_pd": "0_delta",
    }


def audit_dataset_root(root: str | Path) -> dict[str, Any]:
    return dataset_audit(load_dataset(root))


def observation_cache_path(cache_directory: str | Path, video_uid: str) -> Path:
    return Path(cache_directory) / f"{video_uid}.joblib"


def _file_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _file_content_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def observation_signature(
    record: VideoRecord,
    model_path: str | Path,
    *,
    stride: int,
    maximum_faces: int,
) -> str:
    model = Path(model_path)
    payload = {
        "cache_version": OBSERVATION_CACHE_VERSION,
        "video_uid": record.video_uid,
        "video": _file_fingerprint(record.video_path),
        "model": _file_fingerprint(model),
        "stride": int(stride),
        "maximum_faces": int(maximum_faces),
        "orientation_auto": True,
        "sampling_policy": "uniform_fixed_stride_no_annotation_guidance",
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return sha256(encoded).hexdigest()


def _atomic_joblib_dump(value: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    try:
        joblib.dump(value, temporary, compress=3)
        os.replace(temporary, destination)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def save_observation_cache(
    path: str | Path,
    *,
    video_uid: str,
    signature: str,
    result: VisionResult,
) -> None:
    observations = result.observations.copy()
    if not observations.empty:
        observations["video"] = video_uid
    metadata = asdict(result.metadata)
    metadata["path"] = video_uid
    payload = {
        "cache_version": OBSERVATION_CACHE_VERSION,
        "video_uid": video_uid,
        "signature": signature,
        "metadata": metadata,
        "observations": observations,
    }
    _atomic_joblib_dump(payload, Path(path))


def load_observation_cache(
    path: str | Path,
    *,
    expected_video_uid: str | None = None,
    expected_signature: str | None = None,
) -> VisionResult:
    payload = joblib.load(path)
    if (
        not isinstance(payload, dict)
        or payload.get("cache_version") not in SUPPORTED_OBSERVATION_CACHE_VERSIONS
    ):
        raise ValueError(f"Unsupported observation cache: {path}")
    if expected_video_uid is not None and payload.get("video_uid") != expected_video_uid:
        raise ValueError(f"Observation cache video UID mismatch: {path}")
    if expected_signature is not None and payload.get("signature") != expected_signature:
        raise ValueError(f"Stale observation cache signature: {path}")
    metadata_values = dict(payload["metadata"])
    if payload["cache_version"] < 5:
        metadata_values.setdefault("texture_max_dimension", 160)
    metadata_values["path"] = str(payload.get("video_uid", expected_video_uid or "video"))
    decoded_frames = int(metadata_values.get("decoded_frames", 0))
    if decoded_frames > 0:
        # Caches produced before the effective-frame-count fix retained
        # OpenCV's often inflated nominal container count.  Raw observations
        # were already decoded correctly, so normalize the metadata at read
        # time without re-running the face landmarker.
        metadata_values["frame_count"] = decoded_frames
    known_fields = set(VideoMetadata.__dataclass_fields__)
    metadata = VideoMetadata(
        **{key: value for key, value in metadata_values.items() if key in known_fields}
    )
    observations = payload["observations"]
    if not isinstance(observations, pd.DataFrame):
        raise TypeError(f"Observation cache does not contain a DataFrame: {path}")
    return VisionResult(metadata=metadata, observations=observations)


def _select_manifest_shard(
    manifest: Iterable[VideoRecord], start: int, limit: int | None
) -> list[VideoRecord]:
    records = list(manifest)
    if start < 0:
        raise ValueError("start must be non-negative")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    stop = None if limit is None else start + limit
    return records[start:stop]


def extract_observation_cache(
    dataset: IngestedDataset,
    *,
    model_path: str | Path,
    cache_directory: str | Path,
    stride: int = 1,
    maximum_faces: int = 2,
    start: int = 0,
    limit: int | None = None,
    overwrite: bool = False,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Uniformly sample a deterministic manifest shard into resumable caches."""

    if stride < 1:
        raise ValueError("stride must be at least one")
    model = Path(model_path)
    if not model.exists():
        raise FileNotFoundError(f"Face-landmarker model not found: {model}")
    records = _select_manifest_shard(dataset.manifest, start, limit)
    cache_root = Path(cache_directory)
    pending: list[tuple[VideoRecord, str, Path]] = []
    reused = 0
    stale = 0
    for record in records:
        signature = observation_signature(
            record, model, stride=stride, maximum_faces=maximum_faces
        )
        destination = observation_cache_path(cache_root, record.video_uid)
        if destination.exists() and not overwrite:
            try:
                load_observation_cache(
                    destination,
                    expected_video_uid=record.video_uid,
                    expected_signature=signature,
                )
                reused += 1
                continue
            except (ValueError, TypeError, OSError, EOFError):
                stale += 1
        pending.append((record, signature, destination))

    failures: list[dict[str, str]] = []
    extracted = 0
    sampled_frames = 0
    patient_observation_frames = 0
    if pending:
        with FaceLandmarkerBackend(model, maximum_faces=maximum_faces) as backend:
            for position, (record, signature, destination) in enumerate(pending, 1):
                try:
                    result = extract_video_observations(
                        record.video_path,
                        model,
                        maximum_faces=maximum_faces,
                        stride=stride,
                        backend=backend,
                    )
                    save_observation_cache(
                        destination,
                        video_uid=record.video_uid,
                        signature=signature,
                        result=result,
                    )
                    extracted += 1
                    sampled_frames += result.metadata.sampled_frames
                    patient_observation_frames += int(
                        result.patient_observations["frame"].nunique()
                        if not result.patient_observations.empty
                        else 0
                    )
                    if progress is not None:
                        progress(
                            {
                                "status": "extracted",
                                "position": position,
                                "pending": len(pending),
                                "video_uid": record.video_uid,
                                "sampled_frames": result.metadata.sampled_frames,
                                "patient_track_id": result.metadata.patient_track_id,
                            }
                        )
                except Exception as error:  # Continue so one corrupt clip is auditable.
                    failure = {
                        "video_uid": record.video_uid,
                        "error": (
                            f"{type(error).__name__}: {error}"
                            .replace(str(record.video_path), "<video>")
                            .replace(record.filename, "<video>")
                        ),
                    }
                    failures.append(failure)
                    if progress is not None:
                        progress({"status": "failed", **failure})
    return {
        "sampling_policy": "uniform_fixed_stride_no_annotation_guidance",
        "stride": stride,
        "shard_start": start,
        "shard_size": len(records),
        "reused_cache_count": reused,
        "stale_cache_count": stale,
        "extracted_count": extracted,
        "sampled_frame_count": sampled_frames,
        "patient_observation_frame_count": patient_observation_frames,
        "failure_count": len(failures),
        "failures": failures,
    }


@dataclass
class PipelineModelBundle:
    """The learned event detector, task heads, and exact feature contract."""

    cover_event_model: CoverEventModel
    task_models: StrabismusModelBundle
    feature_names: tuple[str, ...]
    event_config: EventConfig
    motion_config: MotionConfig
    extraction_stride: int
    maximum_faces: int
    face_landmarker_sha256: str
    training_metadata: dict[str, Any]
    bundle_version: int = PIPELINE_BUNDLE_VERSION

    def save(self, path: str | Path) -> None:
        _atomic_joblib_dump(self, Path(path))

    @classmethod
    def load(cls, path: str | Path) -> "PipelineModelBundle":
        value = joblib.load(path)
        if not isinstance(value, cls):
            raise TypeError(f"Expected {cls.__name__}, found {type(value).__name__}")
        if value.bundle_version not in (3, PIPELINE_BUNDLE_VERSION):
            raise ValueError(f"Unsupported pipeline bundle version: {value.bundle_version}")
        return value


def frame_annotation_table(dataset: IngestedDataset) -> pd.DataFrame:
    """Materialize trusted annotations for development only, keyed by video UID."""

    rows = [asdict(annotation) for annotation in dataset.iter_frames()]
    table = pd.DataFrame.from_records(rows)
    return table.rename(columns={"video_uid": "video"})


def load_cached_observations(
    dataset: IngestedDataset,
    cache_directory: str | Path,
    *,
    model_path: str | Path | None = None,
    stride: int | None = None,
    maximum_faces: int | None = None,
) -> dict[str, VisionResult]:
    validate_signatures = any(
        value is not None for value in (model_path, stride, maximum_faces)
    )
    if validate_signatures and (
        model_path is None or stride is None or maximum_faces is None
    ):
        raise ValueError(
            "model_path, stride, and maximum_faces are all required for cache "
            "signature validation"
        )
    results: dict[str, VisionResult] = {}
    missing: list[str] = []
    for record in dataset.manifest:
        path = observation_cache_path(cache_directory, record.video_uid)
        if not path.exists():
            missing.append(record.video_uid)
            continue
        expected_signature = (
            observation_signature(
                record,
                model_path,
                stride=int(stride),
                maximum_faces=int(maximum_faces),
            )
            if validate_signatures
            else None
        )
        results[record.video_uid] = load_observation_cache(
            path,
            expected_video_uid=record.video_uid,
            expected_signature=expected_signature,
        )
    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(
            f"Missing {len(missing)} observation caches (first: {preview}); run extract"
        )
    return results


def validate_observation_sampling(
    observations: dict[str, VisionResult], *, expected_stride: int
) -> None:
    """Reject caches produced with a different uniform sampling contract."""

    if expected_stride < 1:
        raise ValueError("expected_stride must be at least one")
    mismatches: list[str] = []
    for video_uid, result in observations.items():
        decoded = int(result.metadata.decoded_frames)
        expected = (decoded + expected_stride - 1) // expected_stride
        if int(result.metadata.sampled_frames) != expected:
            mismatches.append(video_uid)
    if mismatches:
        raise ValueError(
            f"{len(mismatches)} observation caches do not match configured "
            f"sample_stride={expected_stride}; re-run extract with that stride"
        )


def build_cover_training_rows(
    dataset: IngestedDataset,
    observations: dict[str, VisionResult],
    annotations: pd.DataFrame,
    *,
    allowed_video_uids: set[str] | None = None,
    include_limbus: bool = False,
) -> pd.DataFrame:
    """Join trusted cover states only after raw uniform observation extraction."""

    annotation_groups = {
        str(video_uid): group
        for video_uid, group in annotations.groupby("video", sort=False)
    }
    parts: list[pd.DataFrame] = []
    for record in dataset.manifest:
        if allowed_video_uids is not None and record.video_uid not in allowed_video_uids:
            continue
        result = observations[record.video_uid]
        if result.observations.empty:
            continue
        long = long_eye_features(result.observations, include_limbus=include_limbus)
        truth = annotation_groups.get(record.video_uid)
        if truth is None:
            continue
        attached = attach_cover_targets(long, truth)
        if not attached.empty:
            attached["patient_id_group"] = record.patient_id
            parts.append(attached)
    if not parts:
        raise ValueError("No landmark rows could be joined to trusted cover states")
    return pd.concat(parts, ignore_index=True)


def _technical_quality(features: dict[str, float | str]) -> float:
    def bounded(name: str, default: float = 0.0) -> float:
        try:
            value = float(features.get(name, default))
        except (TypeError, ValueError):
            value = default
        return float(np.clip(value if np.isfinite(value) else default, 0.0, 1.0))

    usable_cycles = float(features.get("all_usable_n", 0.0) or 0.0)
    cycle_presence = float(np.clip(usable_cycles / 2.0, 0.0, 1.0))
    cycle_quality = bounded("all_quality_median")
    return float(
        0.35 * bounded("face_observation_fraction")
        + 0.25 * bounded("cover_state_known_fraction")
        + 0.20 * cycle_presence
        + 0.20 * cycle_quality
    )


def feature_record_from_timeline(
    record: VideoRecord,
    result: VisionResult,
    timeline: pd.DataFrame,
    *,
    motion_config: MotionConfig | None = None,
    include_dark_iris: bool = False,
) -> dict[str, Any]:
    cycles = extract_motion_cycles(
        result.observations,
        timeline,
        test_type=record.test_type,
        config=motion_config,
    )
    features = aggregate_video_features(
        result.observations,
        timeline,
        cycles,
        video_uid=record.video_uid,
        test_type=record.test_type,
        distance=record.distance,
        correction=record.correction,
        fps=result.metadata.fps,
        frame_count=result.metadata.frame_count,
        sampled_frame_count=result.metadata.sampled_frames,
    )
    features.update(clinical_motion_evidence(features))
    if record.test_type == "Cover-Uncover":
        features.update(
            job1_rich_features(
                result.observations,
                timeline,
                fps=result.metadata.fps,
            )
        )
    if "right_limbus_x_px" in result.observations.columns:
        features.update(
            native_features(
                result.observations,
                timeline,
                fps=result.metadata.fps,
                test_type=record.test_type,
            )
        )
    else:
        features.update(native_feature_schema())
    if include_dark_iris:
        if record.test_type == "Alternate-Cover":
            video_path = getattr(record, "video_path", None)
            if video_path is None:
                raise ValueError(
                    "Dark-iris feature extraction requires a source video path"
                )
            features.update(
                dark_iris_features(
                    video_path,
                    result.observations,
                    timeline,
                    fps=result.metadata.fps,
                )
            )
        else:
            features.update(dark_iris_feature_schema())
    features["technical_quality_score"] = _technical_quality(features)
    return features


def build_predicted_feature_table(
    dataset: IngestedDataset,
    observations: dict[str, VisionResult],
    cover_model: CoverEventModel,
    *,
    event_config: EventConfig | None = None,
    motion_config: MotionConfig | None = None,
    video_uids: set[str] | None = None,
    include_dark_iris: bool = False,
) -> pd.DataFrame:
    """Build task features exclusively from event states inferred from pixels."""

    records: list[dict[str, Any]] = []
    for source in dataset.manifest:
        if video_uids is not None and source.video_uid not in video_uids:
            continue
        result = observations[source.video_uid]
        timeline = infer_cover_timeline(
            result.observations,
            cover_model,
            frame_count=result.metadata.frame_count,
            config=event_config,
        )
        feature = feature_record_from_timeline(
            source,
            result,
            timeline,
            motion_config=motion_config,
            include_dark_iris=include_dark_iris,
        )
        feature.update(
            {
                "patient_id_group": source.patient_id,
                "cohort": source.cohort,
                "job1_label": _enum_value(source.job1_manifest_label),
                "job3_label": _enum_value(source.job3_clinical_class),
                "observable_label": source.observable_in_clip,
                "pd_lower": source.horizontal_pd_lower,
                "pd_upper": source.horizontal_pd_upper,
                "total_direction_label": (
                    None
                    if source.horizontal_direction is HorizontalDirection.MIXED
                    else source.horizontal_direction.value
                ),
            }
        )
        records.append(feature)
    return pd.DataFrame.from_records(records)


_TASK_EXACT_FEATURES = {
    "is_cover_uncover",
    "is_near",
    "with_correction",
    "face_observation_fraction",
    "sampling_fraction",
    "starts_covered",
    "initial_binocular_baseline_observed",
    "cover_state_known_fraction",
    "binocular_uncovered_fraction",
    "verified_binocular_segment_observed",
    "technical_quality_score",
    "yaw_proxy_std",
    "pitch_proxy_std",
    "roll_radians_std",
    "right_aperture_std",
    "left_aperture_std",
    "right_glint_reliability_std",
    "left_glint_reliability_std",
    "right_nasal_u_std",
    "left_nasal_u_std",
    "manifest_eso_evidence",
    "manifest_exo_evidence",
    "latent_or_total_eso_evidence",
    "latent_or_total_exo_evidence",
}
_TASK_PREFIXES = (
    "all_",
    "binoc_",
    "cover_on_",
    "iris_pair_",
    "left_cov_",
    "right_cov_",
    "state_cover_on_",
    "uncover_",
    "observed_right_",
    "observed_left_",
)


def select_task_feature_names(table: pd.DataFrame) -> tuple[str, ...]:
    """Explicit allowlist excluding identity, camera size, path, and static anatomy."""

    selected = [
        column
        for column in table.columns
        if (column in _TASK_EXACT_FEATURES or column.startswith(_TASK_PREFIXES))
        and pd.api.types.is_numeric_dtype(table[column])
    ]
    if not selected:
        raise ValueError("No approved task features are present")
    return tuple(sorted(set(selected)))


def select_native_feature_names(table: pd.DataFrame) -> tuple[str, ...]:
    """Native-limbus event features (cache v4); opt-in for experimental heads."""

    return tuple(
        name
        for name in native_feature_names()
        if name in table and pd.api.types.is_numeric_dtype(table[name])
    )


def select_job1_feature_names(table: pd.DataFrame) -> tuple[str, ...]:
    """Base task features plus Job 1-only rich same-iris comparisons."""

    return tuple(
        dict.fromkeys(
            (*select_task_feature_names(table), *job1_rich_feature_names(table))
        )
    )


def select_job2_feature_names(table: pd.DataFrame) -> tuple[str, ...]:
    """Base task features plus validated pixel-derived iris displacement."""

    dark = [
        name
        for name in dark_iris_feature_names()
        if name in table and pd.api.types.is_numeric_dtype(table[name])
    ]
    return tuple(dict.fromkeys((*select_task_feature_names(table), *dark)))


def select_job3_blend_feature_names(table: pd.DataFrame) -> tuple[str, ...]:
    """Base features plus the compact raw/roll rich subset for Job 3."""

    stable_stats = (
        "_median",
        "_weighted_mean",
        "_dominant",
        "_median_abs",
        "_iqr",
        "_positive_fraction",
        "_sign_consistency",
        "_n",
    )
    compact = [
        name
        for name in job1_rich_feature_names(table)
        if name.startswith("rich_")
        and name.endswith(stable_stats)
        and (
            "_raw_" in name
            or "_roll_" in name
            or name.startswith(("rich_raw_", "rich_roll_"))
        )
    ]
    return tuple(dict.fromkeys((*select_task_feature_names(table), *compact)))


def _tune_final_direction_threshold(
    X: pd.DataFrame,
    y: np.ndarray,
    clinical_y: np.ndarray | None = None,
    *,
    random_state: int,
    target_sensitivity: float = 0.95,
) -> tuple[float, float]:
    class_counts = pd.Series(y).value_counts()
    folds = min(5, int(class_counts.min()))
    if folds < 2:
        return 0.35, 0.50
    splitter = StratifiedKFold(
        n_splits=folds, shuffle=True, random_state=random_state
    )
    splits = list(splitter.split(X, y))
    manifest_splits = splits
    if clinical_y is not None:
        manifest_splits = list(
            StratifiedKFold(
                n_splits=folds,
                shuffle=True,
                random_state=random_state + 101,
            ).split(X, y)
        )
    oof = np.empty(len(y), dtype=float)
    oof_eso = np.empty(len(y), dtype=float)
    for fold, (train, validation) in enumerate(splits):
        model = DirectionHead(random_state=random_state + fold + 1).fit(
            X.iloc[train],
            y[train],
        )
        if clinical_y is None:
            oof[validation] = model.predict_proba(X.iloc[validation])[:, :2].sum(axis=1)
        oof_eso[validation] = model.predict_conditional_eso_probability(
            X.iloc[validation]
        )
    if clinical_y is not None:
        for fold, (train, validation) in enumerate(manifest_splits):
            model = DirectionHead(random_state=random_state + fold + 1).fit(
                X.iloc[train],
                y[train],
                clinical_y=clinical_y[train],
            )
            oof[validation] = model.predict_proba(X.iloc[validation])[:, :2].sum(axis=1)
    positive = y != "no_manifest_horizontal_deviation"
    candidates = np.unique(np.r_[0.0, oof, 1.0])
    feasible = [
        value
        for value in candidates
        if np.mean(oof[positive] >= value) >= target_sensitivity
    ]
    deviation_threshold = float(max(feasible))
    deviation_rows = positive
    truth_eso = y[deviation_rows] == "eso_tropia"
    direction_scores = oof_eso[deviation_rows]
    direction_candidates = np.unique(np.r_[0.0, direction_scores, 1.0])
    best: tuple[float, float, float, float] | None = None
    direction_threshold = 0.50
    for threshold in direction_candidates:
        prediction = direction_scores >= threshold
        eso_recall = (
            float(np.mean(prediction[truth_eso])) if np.any(truth_eso) else 0.0
        )
        exo_recall = (
            float(np.mean(~prediction[~truth_eso])) if np.any(~truth_eso) else 0.0
        )
        balanced = (eso_recall + exo_recall) / 2.0
        accuracy = float(np.mean(prediction == truth_eso))
        score = (
            balanced,
            accuracy,
            -abs(float(threshold) - 0.5),
            float(threshold),
        )
        if best is None or score > best:
            best = score
            direction_threshold = float(threshold)
    return deviation_threshold, direction_threshold


def train_pipeline_bundle(
    dataset: IngestedDataset,
    *,
    cache_directory: str | Path,
    output_path: str | Path,
    config: PipelineConfig | None = None,
    feature_table_path: str | Path | None = None,
) -> dict[str, Any]:
    """Fit the event detector and the three requested video-level task heads."""

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
    cover_rows = build_cover_training_rows(dataset, observations, annotations)
    cover_model = CoverEventModel(random_seed=config.random_seed)
    cover_diagnostics = cover_model.fit(cover_rows)
    feature_table = build_predicted_feature_table(
        dataset,
        observations,
        cover_model,
        event_config=config.event,
        motion_config=config.motion,
        include_dark_iris=True,
    )
    base_feature_names = select_task_feature_names(feature_table)
    job1_feature_names = select_job1_feature_names(feature_table)
    job2_feature_names = select_job2_feature_names(feature_table)
    job3_blend_feature_names = select_job3_blend_feature_names(feature_table)
    X = feature_table.loc[:, list(base_feature_names)]
    X_job1 = feature_table.loc[:, list(job1_feature_names)]
    X_job2 = feature_table.loc[:, list(job2_feature_names)]
    X_job3_blend = feature_table.loc[:, list(job3_blend_feature_names)]

    job1_mask = feature_table["job1_label"].notna().to_numpy()
    job1_y = feature_table.loc[job1_mask, "job1_label"].to_numpy(dtype=object)
    job1_clinical_y = feature_table.loc[
        job1_mask, "job3_label"
    ].to_numpy(dtype=object)
    direction_head = DirectionHead(
        random_state=config.random_seed,
        manifest_random_state_offset=77,
    )
    direction_head.fit(
        X_job1.loc[job1_mask],
        job1_y,
        clinical_y=job1_clinical_y,
    )
    (
        direction_head.deviation_probability_threshold,
        direction_head.direction_probability_threshold,
    ) = _tune_final_direction_threshold(
        X_job1.loc[job1_mask].reset_index(drop=True),
        job1_y,
        job1_clinical_y,
        random_state=config.random_seed,
    )

    job2_mask = (
        (feature_table["test_type"] == "Alternate-Cover")
        & feature_table["pd_lower"].notna()
        & feature_table["pd_upper"].notna()
        & feature_table["total_direction_label"].notna()
    ).to_numpy()
    pd_direction_head = PDMagnitudeHead(random_state=config.random_seed)
    pd_direction_head.fit(
        X.loc[job2_mask],
        feature_table.loc[job2_mask, "pd_lower"].to_numpy(dtype=float),
        feature_table.loc[job2_mask, "pd_upper"].to_numpy(dtype=float),
        direction_y=feature_table.loc[
            job2_mask, "total_direction_label"
        ].to_numpy(dtype=object),
    )
    pd_head = PDMagnitudeHead(random_state=config.random_seed)
    pd_head.fit(
        X_job2.loc[job2_mask],
        feature_table.loc[job2_mask, "pd_lower"].to_numpy(dtype=float),
        feature_table.loc[job2_mask, "pd_upper"].to_numpy(dtype=float),
        direction_y=feature_table.loc[
            job2_mask, "total_direction_label"
        ].to_numpy(dtype=object),
    )

    job3_mask = (feature_table["test_type"] == "Cover-Uncover").to_numpy()
    manifestation_head = ManifestationHead(random_state=config.random_seed)
    manifestation_head.fit(
        X.loc[job3_mask],
        feature_table.loc[job3_mask, "job3_label"].to_numpy(dtype=object),
        feature_table.loc[job3_mask, "observable_label"].to_numpy(dtype=object),
    )
    manifestation_clinical_head = ManifestationClinicalHead(
        random_state=config.random_seed + 110
    ).fit(
        X_job3_blend.loc[job3_mask],
        feature_table.loc[job3_mask, "job3_label"].to_numpy(dtype=object),
    )
    task_models = StrabismusModelBundle(
        direction_head=direction_head,
        pd_head=pd_head,
        pd_direction_head=pd_direction_head,
        manifestation_head=manifestation_head,
        manifestation_clinical_head=manifestation_clinical_head,
        manifestation_blend_weight=0.65,
        random_state=config.random_seed,
    )
    feature_names = task_models.validate_shared_feature_schema().names
    training_metadata = {
        "texture_max_dimension": None,
        "analysis_unit": "video",
        "training_video_count": len(feature_table),
        "job1_video_count": int(job1_mask.sum()),
        "job2_act_video_count": int(job2_mask.sum()),
        "job3_cu_video_count": int(job3_mask.sum()),
        "cover_training_row_count": len(cover_rows),
        "cover_training_video_count": int(cover_rows["video"].nunique()),
        "cover_diagnostics_resubstitution_only": asdict(cover_diagnostics),
        "job1_deviation_probability_threshold": (
            direction_head.deviation_probability_threshold
        ),
        "job1_direction_probability_threshold": (
            direction_head.direction_probability_threshold
        ),
        "job1_threshold_target_sensitivity": 0.95,
        "pd_calibration_mode": "dataset_learned_fallback",
        "pd_point_calibration_method": pd_head.point_calibration_method_,
        "pd_point_calibration_slope": pd_head.point_calibration_slope_,
        "pd_point_calibration_intercept_pd": (
            pd_head.point_calibration_intercept_pd_
        ),
        "pd_normal_probability_threshold": (
            pd_direction_head.normal_probability_threshold_
        ),
        "pd_normal_magnitude_limit_pd": (
            pd_direction_head.normal_magnitude_limit_pd_
        ),
        "pd_magnitude_head_normal_probability_threshold_unused": (
            pd_head.normal_probability_threshold_
        ),
        "pd_magnitude_head_normal_magnitude_limit_pd_unused": (
            pd_head.normal_magnitude_limit_pd_
        ),
        "pd_geometric_calibration": False,
        "pd_magnitude_uses_dark_iris_features": True,
        "pd_direction_uses_dark_iris_features": False,
        "pd_magnitude_feature_count": len(job2_feature_names),
        "pd_direction_feature_count": len(base_feature_names),
        "job3_observable_no_threshold": (
            manifestation_head.observable_no_threshold
        ),
        "job3_observable_yes_threshold": (
            manifestation_head.observable_yes_threshold
        ),
        "job3_clinical_probability_blend": "compact_rich",
        "job3_clinical_probability_blend_weight": 0.65,
        "job3_clinical_blend_feature_count": len(job3_blend_feature_names),
        "feature_names": list(feature_names),
        "event_training_uses_annotations": True,
        "raw_inference_requires_annotations": False,
        "extraction_stride": config.event.sample_stride,
        "maximum_faces": config.maximum_faces,
        "face_landmarker_sha256": _file_content_sha256(
            config.face_landmarker_model
        ),
    }
    bundle = PipelineModelBundle(
        cover_event_model=cover_model,
        task_models=task_models,
        feature_names=feature_names,
        event_config=config.event,
        motion_config=config.motion,
        extraction_stride=config.event.sample_stride,
        maximum_faces=config.maximum_faces,
        face_landmarker_sha256=_file_content_sha256(
            config.face_landmarker_model
        ),
        training_metadata=training_metadata,
    )
    bundle.save(output_path)
    if feature_table_path is not None:
        destination = Path(feature_table_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        feature_table.drop(columns=["patient_id_group"], errors="ignore").to_csv(
            destination, index=False
        )
    return training_metadata


def _append_warning(output: dict[str, Any], warning: str) -> None:
    warnings = output.setdefault("warnings", [])
    if warning not in warnings:
        warnings.append(warning)


def analyze_raw_video(
    bundle: PipelineModelBundle,
    *,
    video_path: str | Path,
    test_type: str,
    distance: str,
    correction: str,
    face_landmarker_model: str | Path,
    maximum_faces: int | None = None,
    stride: int | None = None,
) -> dict[str, Any]:
    """Run all three forced-output jobs from raw video and declared metadata."""

    if test_type not in {"Cover-Uncover", "Alternate-Cover"}:
        raise ValueError("test_type must be Cover-Uncover or Alternate-Cover")
    distance = _normalize_distance(distance)
    correction = _normalize_correction(correction)
    if stride is None:
        stride = bundle.extraction_stride
    elif stride != bundle.extraction_stride:
        raise ValueError(
            f"Bundle requires extraction stride {bundle.extraction_stride}, got {stride}"
        )
    if maximum_faces is None:
        maximum_faces = bundle.maximum_faces
    elif maximum_faces != bundle.maximum_faces:
        raise ValueError(
            f"Bundle requires maximum_faces={bundle.maximum_faces}, got {maximum_faces}"
        )
    supplied_landmarker_sha256 = _file_content_sha256(face_landmarker_model)
    if supplied_landmarker_sha256 != bundle.face_landmarker_sha256:
        raise ValueError(
            "Face Landmarker asset does not match the model bundle preprocessing "
            "contract"
        )
    path = Path(video_path)
    result = extract_video_observations(
        path,
        face_landmarker_model,
        maximum_faces=maximum_faces,
        stride=stride,
        texture_max_dimension=bundle.training_metadata.get("texture_max_dimension", 160),
    )
    timeline = infer_cover_timeline(
        result.observations,
        bundle.cover_event_model,
        frame_count=result.metadata.frame_count,
        config=bundle.event_config,
    )
    synthetic_record = type(
        "InferenceRecord",
        (),
        {
            "video_uid": sha256(str(path.resolve()).encode()).hexdigest(),
            "video_path": path,
            "test_type": test_type,
            "distance": distance,
            "correction": correction,
        },
    )()
    feature = feature_record_from_timeline(
        synthetic_record,
        result,
        timeline,
        motion_config=bundle.motion_config,
        include_dark_iris=any(
            name.startswith("dark_") for name in bundle.feature_names
        ),
    )
    X = pd.DataFrame([feature]).reindex(columns=list(bundle.feature_names))
    quality = float(feature["technical_quality_score"])
    job1 = bundle.task_models.predict_job1(X, quality=quality)[0]
    job2 = bundle.task_models.predict_job2(X, quality=quality)[0]
    job3 = bundle.task_models.predict_job3(X, quality=quality)[0]

    common_warnings: list[str] = []
    if result.metadata.patient_track_id is None:
        common_warnings.append("no_patient_face_track")
    if not np.isfinite(float(feature.get("starts_covered", np.nan))):
        common_warnings.append("initial_cover_state_unknown")
    elif float(feature["starts_covered"]) == 1.0:
        common_warnings.append("clip_starts_covered_no_initial_binocular_baseline")
    if float(feature.get("initial_state_from_delayed_observation", 0.0)) == 1.0:
        common_warnings.append("initial_cover_state_left_censored_delayed_observation")
    if float(feature.get("all_usable_n", 0.0)) < 1.0:
        common_warnings.append("no_usable_refixation_cycle")
    if float(feature.get("cover_state_known_fraction", 0.0)) < 0.5:
        common_warnings.append("limited_cover_state_evidence")

    if test_type != "Cover-Uncover":
        _append_warning(job1, "job1_requires_cover_uncover")
    if test_type != "Alternate-Cover":
        _append_warning(job2, "job2_primary_requires_alternate_cover")
    _append_warning(job2, "dataset_learned_fallback")
    job2["calibration_mode"] = "dataset_learned_fallback"
    job2["geometric_calibration"] = False

    verified_binocular = bool(
        float(feature.get("verified_binocular_segment_observed", 0.0)) == 1.0
    )
    if test_type != "Cover-Uncover":
        _append_warning(job3, "clinical_class_low_evidence_act")
        job3["observable_state"] = "indeterminate"
        job3["observable_in_clip"] = None
        job3["indeterminate"] = True
        _append_warning(job3, "observability_requires_cover_uncover")
    elif not verified_binocular:
        job3["observable_state"] = "indeterminate"
        job3["observable_in_clip"] = None
        job3["indeterminate"] = True
        _append_warning(job3, "binocular_segment_not_verified")
    for warning in common_warnings:
        _append_warning(job1, warning)
        _append_warning(job2, warning)
        _append_warning(job3, warning)

    return {
        "video": str(path),
        "inputs": {
            "test_type": test_type,
            "distance": distance,
            "correction": correction,
        },
        "job1_manifest_direction": job1,
        "job2_total_horizontal_deviation": job2,
        "job3_manifestation": job3,
        "technical_quality_score": quality,
        "observable_in_clip": job3["observable_state"],
        "initial_binocular_baseline_observed": bool(
            float(feature.get("initial_binocular_baseline_observed", 0.0)) == 1.0
        ),
        "initial_state_observation_delay_s": (
            None
            if not np.isfinite(
                float(feature.get("initial_state_observation_delay_s", np.nan))
            )
            else float(feature["initial_state_observation_delay_s"])
        ),
        "verified_binocular_segment_observed": verified_binocular,
        "video_metadata": asdict(result.metadata),
        "common_warnings": common_warnings,
    }


__all__ = [
    "OBSERVATION_CACHE_VERSION",
    "audit_dataset_root",
    "analyze_raw_video",
    "build_cover_training_rows",
    "build_predicted_feature_table",
    "dataset_audit",
    "extract_observation_cache",
    "feature_record_from_timeline",
    "frame_annotation_table",
    "load_observation_cache",
    "load_cached_observations",
    "observation_cache_path",
    "observation_signature",
    "PipelineModelBundle",
    "save_observation_cache",
    "select_task_feature_names",
    "select_native_feature_names",
    "select_job1_feature_names",
    "select_job2_feature_names",
    "select_job3_blend_feature_names",
    "train_pipeline_bundle",
    "validate_observation_sampling",
]
