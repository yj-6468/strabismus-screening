from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import strabismus_screen.evaluation_pipeline as evaluation_pipeline
from strabismus_screen.cli import build_parser
from strabismus_screen.config import PipelineConfig
from strabismus_screen.events import CoverModelDiagnostics
from strabismus_screen.modeling import PDMagnitudeHead as RealPDMagnitudeHead


@dataclass(frozen=True)
class _Record:
    video_uid: str
    patient_id: str
    cohort: str
    test_type: str
    job1_manifest_label: str | None
    job3_clinical_class: str
    observable_in_clip: bool | None
    horizontal_pd_lower: float
    horizontal_pd_upper: float
    horizontal_direction: str


def _synthetic_dataset() -> SimpleNamespace:
    records: list[_Record] = []
    clinical_classes = ("tropia", "intermittent_tropia", "phoria", "normal")
    for index in range(32):
        test_type = "Cover-Uncover" if index % 2 == 0 else "Alternate-Cover"
        clinical = clinical_classes[(index // 2) % len(clinical_classes)]
        direction = ("eso", "exo")[(index // 8) % 2]
        if clinical in {"phoria", "normal"}:
            job1 = "no_manifest_horizontal_deviation"
        else:
            job1 = f"{direction}_tropia"
        if test_type != "Cover-Uncover":
            job1 = None
        observable = True if clinical == "tropia" else False
        if clinical == "intermittent_tropia" or test_type != "Cover-Uncover":
            observable = None
        total_direction = "normal" if clinical == "normal" else direction
        lower = 0.0 if total_direction == "normal" else float(5 + index % 12)
        upper = lower if index % 5 else lower + (0.0 if lower == 0 else 5.0)
        records.append(
            _Record(
                video_uid=f"vid_{index:03d}",
                patient_id=f"patient_{index // 2:03d}",
                cohort="new" if index >= 24 else "older",
                test_type=test_type,
                job1_manifest_label=job1,
                job3_clinical_class=clinical,
                observable_in_clip=observable,
                horizontal_pd_lower=lower,
                horizontal_pd_upper=upper,
                horizontal_direction=total_direction,
            )
        )
    return SimpleNamespace(manifest=tuple(records))


def _feature_table(dataset: SimpleNamespace, video_uids: set[str]) -> pd.DataFrame:
    rows = []
    for index, record in enumerate(dataset.manifest):
        if record.video_uid not in video_uids:
            continue
        direction_sign = 1.0 if record.horizontal_direction == "eso" else -1.0
        if record.horizontal_direction == "normal":
            direction_sign = 0.0
        rows.append(
            {
                "video_uid": record.video_uid,
                "test_type": record.test_type,
                "patient_id_group": record.patient_id,
                "cohort": record.cohort,
                "job1_label": record.job1_manifest_label,
                "job3_label": record.job3_clinical_class,
                "observable_label": record.observable_in_clip,
                "pd_lower": record.horizontal_pd_lower,
                "pd_upper": record.horizontal_pd_upper,
                "total_direction_label": record.horizontal_direction,
                "is_cover_uncover": float(record.test_type == "Cover-Uncover"),
                "technical_quality_score": 0.9,
                "all_motion_median": direction_sign * (0.1 + index / 100.0),
                "all_absolute_motion_median": (
                    (record.horizontal_pd_lower + record.horizontal_pd_upper) / 40.0
                ),
                "cover_state_known_fraction": 0.95,
            }
        )
    return pd.DataFrame.from_records(rows)


def test_outer_splits_keep_patients_disjoint_and_add_temporal_holdout() -> None:
    dataset = _synthetic_dataset()
    splits = evaluation_pipeline.make_end_to_end_splits(
        dataset, n_splits=2, include_patient_grouped=True, include_cohort_holdout=True
    )
    assert set(splits) == {
        "video_level_stratified",
        "patient_grouped_audit",
        "older_to_new_cohort_holdout",
    }
    all_indices = set(range(len(dataset.manifest)))
    primary_test = set()
    for train, test in splits["video_level_stratified"]:
        assert set(train).isdisjoint(test)
        primary_test.update(int(index) for index in test)
    assert primary_test == all_indices

    for train, test in splits["patient_grouped_audit"]:
        train_patients = {dataset.manifest[int(i)].patient_id for i in train}
        test_patients = {dataset.manifest[int(i)].patient_id for i in test}
        assert train_patients.isdisjoint(test_patients)

    temporal_train, temporal_test = splits["older_to_new_cohort_holdout"][0]
    assert {dataset.manifest[int(i)].cohort for i in temporal_train} == {"older"}
    assert {dataset.manifest[int(i)].cohort for i in temporal_test} == {"new"}

    patient_only = evaluation_pipeline.make_end_to_end_splits(
        dataset,
        n_splits=2,
        include_video_level=False,
        include_patient_grouped=True,
        include_cohort_holdout=False,
    )
    assert set(patient_only) == {"patient_grouped_audit"}


def test_evaluation_refits_event_model_without_heldout_annotations(
    monkeypatch, tmp_path: Path
) -> None:
    dataset = _synthetic_dataset()
    event_fit_video_sets: list[set[str]] = []

    class FakeCoverModel:
        def __init__(self, random_seed: int):
            self.random_seed = random_seed

        def fit(self, rows: pd.DataFrame) -> CoverModelDiagnostics:
            event_fit_video_sets.append(set(rows["video"].astype(str)))
            return CoverModelDiagnostics(
                confusion=[[len(rows) // 2, 0], [0, len(rows) // 2]],
                report={},
                n_rows=len(rows),
                n_videos=rows["video"].nunique(),
                covered_fraction=0.5,
            )

        def predict_probability(self, rows: pd.DataFrame) -> np.ndarray:
            return np.where(rows["covered"].to_numpy(dtype=int) == 1, 0.9, 0.1)

    def fake_cover_rows(
        dataset_arg,
        observations,
        annotations,
        *,
        allowed_video_uids=None,
    ) -> pd.DataFrame:
        assert dataset_arg is dataset
        rows = []
        for video_uid in sorted(allowed_video_uids or set()):
            rows.extend(
                [
                    {"video": video_uid, "covered": 0, "feature": 0.0},
                    {"video": video_uid, "covered": 1, "feature": 1.0},
                ]
            )
        return pd.DataFrame.from_records(rows)

    def fake_predicted_features(
        dataset_arg,
        observations,
        cover_model,
        *,
        event_config,
        motion_config,
        video_uids,
    ) -> pd.DataFrame:
        assert dataset_arg is dataset
        return _feature_table(dataset, set(video_uids))

    monkeypatch.setattr(
        evaluation_pipeline,
        "load_cached_observations",
        lambda dataset, cache, **kwargs: {},
    )
    monkeypatch.setattr(
        evaluation_pipeline,
        "validate_observation_sampling",
        lambda observations, *, expected_stride: None,
    )
    monkeypatch.setattr(
        evaluation_pipeline, "frame_annotation_table", lambda dataset: pd.DataFrame()
    )
    monkeypatch.setattr(evaluation_pipeline, "CoverEventModel", FakeCoverModel)
    monkeypatch.setattr(
        evaluation_pipeline, "build_cover_training_rows", fake_cover_rows
    )
    monkeypatch.setattr(
        evaluation_pipeline, "build_predicted_feature_table", fake_predicted_features
    )
    monkeypatch.setattr(
        evaluation_pipeline,
        "PDMagnitudeHead",
        lambda random_state: RealPDMagnitudeHead(
            random_state=random_state, n_estimators=4, calibration_folds=2
        ),
    )

    report = evaluation_pipeline.evaluate_pipeline(
        dataset,
        cache_directory=tmp_path,
        config=PipelineConfig(random_seed=19),
        n_splits=2,
        include_patient_grouped=True,
        include_cohort_holdout=True,
    )
    assert report["oracle_event_features_used"] is False
    assert report["heldout_frame_annotations_used_as_task_inputs"] is False
    assert len(event_fit_video_sets) == 5  # 2 video + 2 patient + 1 cohort folds.

    expected_splits = evaluation_pipeline.make_end_to_end_splits(
        dataset,
        n_splits=2,
        random_state=19,
        include_patient_grouped=True,
        include_cohort_holdout=True,
    )
    fit_offset = 0
    for mode_name, mode in report["results"].items():
        mode_fit_sets = event_fit_video_sets[
            fit_offset : fit_offset + mode["fold_count"]
        ]
        for fit_videos, fold, (train_indices, test_indices) in zip(
            mode_fit_sets, mode["folds"], expected_splits[mode_name]
        ):
            expected_train = {
                dataset.manifest[int(index)].video_uid for index in train_indices
            }
            expected_test = {
                dataset.manifest[int(index)].video_uid for index in test_indices
            }
            assert fit_videos == expected_train
            assert fit_videos.isdisjoint(expected_test)
            assert fold["event_annotation_test_leakage_count"] == 0
            assert len(fit_videos) == fold["event_annotation_training_video_count"]
        fit_offset += mode["fold_count"]

    primary = report["results"]["video_level_stratified"]
    for job_key in (
        "job1_direction_of_manifest_tropia",
        "job2_total_horizontal_prism_diopters",
        "job3_manifestation_and_observability",
    ):
        predictions = primary[job_key]["oof_predictions"]
        assert len({row["video_uid"] for row in predictions}) == len(predictions)
    pd_metrics = primary["job2_total_horizontal_prism_diopters"]["overall"]
    assert "within_5_pd" in pd_metrics["emitted_magnitude"]
    assert "within_10_pd" in pd_metrics["emitted_magnitude"]
    assert "bland_altman_bias_pd" in pd_metrics["emitted_magnitude"]
    assert "reference_interval_overlap_rate" in pd_metrics["prediction_intervals"]
    observable = primary["job3_manifestation_and_observability"]["overall"][
        "observable_in_clip"
    ]
    assert observable["prediction_labels"] == ["no", "yes", "indeterminate"]

    destination = evaluation_pipeline.write_evaluation_report(
        report, tmp_path / "evaluation.json"
    )
    loaded = json.loads(destination.read_text())
    assert loaded["evaluation_mode"].startswith("raw_video_inferred_events")


def test_cli_exposes_nested_evaluate_command() -> None:
    args = build_parser().parse_args(
        [
            "evaluate",
            "--folds",
            "3",
            "--mode",
            "video",
            "--output",
            "report.json",
        ]
    )
    assert args.command == "evaluate"
    assert args.folds == 3
    assert args.mode == ["video"]
