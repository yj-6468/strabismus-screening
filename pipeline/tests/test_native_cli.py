"""Fast CLI dispatch tests; fitting and video inference are replaced by stubs."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np

from strabismus_screen import cli
from strabismus_screen.config import EventConfig, MotionConfig, PipelineConfig
from strabismus_screen.native_bundle import HeadSpec, NativeBundle
from strabismus_screen.pipeline import PIPELINE_BUNDLE_VERSION, PipelineModelBundle


PREDICT = ["predict", "example.mov", "--test-type", "Alternate-Cover", "--distance", "near", "--correction", "sc"]


def native_bundle():
    return NativeBundle(cover_model=None, config=PipelineConfig(), spec=HeadSpec(),
                        job1=None, job1_no_phoria=None, job2=None, job3=None,
                        face_landmarker_sha256="test", training_metadata={"texture_max_dimension": None})


def legacy_bundle(version=PIPELINE_BUNDLE_VERSION):
    return PipelineModelBundle(cover_event_model=None, task_models=None, feature_names=(),
                               event_config=EventConfig(sample_stride=2), motion_config=MotionConfig(),
                               extraction_stride=2, maximum_faces=3, face_landmarker_sha256="test",
                               training_metadata={}, bundle_version=version)


def test_cli_defaults_use_strict_native_training_and_prediction():
    parser = cli.build_parser()
    train = parser.parse_args(["train-native"])
    predict = parser.parse_args(PREDICT)
    assert train.cache_dir == "artifacts/observations_v14_merged"
    assert train.output == predict.bundle == "artifacts/final_model_v14.joblib"
    assert train.config == "configs/stride1.yaml"
    assert train.seed == 20260901
    assert train.job1_model == "ensemble"
    assert train.job2_features == "phys"
    assert parser.parse_args(["extract"]).stride == 1
    assert parser.parse_args(["train"]).output == "artifacts/strabismus_pipeline.joblib"


def test_train_native_dispatches_config_spec_and_paths(monkeypatch, tmp_path, capsys):
    captured = {}
    dataset = object()
    config = PipelineConfig()
    def load_dataset(path):
        captured["dataset_root"] = path
        return dataset
    def load_config(path):
        captured["config_path"] = path
        return config
    def train(data, **kwargs):
        captured["dataset"] = data
        captured.update(kwargs)
        return SimpleNamespace(training_metadata={"videos": 12, "stride": 1})
    monkeypatch.setattr(cli, "load_dataset", load_dataset)
    monkeypatch.setattr(cli, "load_config", load_config)
    monkeypatch.setattr(cli, "train_native_bundle", train)
    output = tmp_path / "trained.joblib"
    assert cli.main(["--dataset-root", str(tmp_path), "train-native", "--cache-dir", "cache-native",
                     "--output", str(output), "--config", "custom.yaml", "--seed", "17",
                     "--job3-model", "extratrees"]) == 0
    assert captured["dataset_root"] == tmp_path.resolve()
    assert captured["dataset"] is dataset
    assert captured["config"] is config
    assert captured["config_path"] == Path("custom.yaml").resolve()
    assert captured["cache_directory"] == "cache-native"
    assert captured["output_path"] == str(output)
    assert captured["seed"] == 17
    assert captured["spec"].job3_model == "extratrees"
    assert captured["spec"].job2_features == "phys"
    assert json.loads(capsys.readouterr().out) == {"output": str(output), "stride": 1, "videos": 12}


def test_train_native_rejects_stride_two_before_loading_data(monkeypatch):
    config = PipelineConfig(event=EventConfig(sample_stride=2))
    monkeypatch.setattr(cli, "load_config", lambda _: config)
    monkeypatch.setattr(cli, "load_dataset", lambda _: pytest.fail("must reject config before data loading"))
    with pytest.raises(ValueError, match="requires sample_stride=1"):
        cli.main(["train-native"])


@pytest.mark.parametrize("options", [[], ["--stride", "1", "--maximum-faces", "2"], ["--bundle", "old-native.joblib"]])
def test_predict_native_dispatch_and_matching_overrides(monkeypatch, capsys, options):
    bundle = native_bundle()
    captured = {}
    def load(path):
        captured["bundle_path"] = path
        return bundle
    def predict(model, **kwargs):
        assert model is bundle
        captured.update(kwargs)
        return {"native_result": True}
    monkeypatch.setattr(cli.joblib, "load", load)
    monkeypatch.setattr(cli, "predict_native", predict)
    monkeypatch.setattr(cli, "analyze_raw_video", lambda *a, **k: pytest.fail("native model routed to legacy predictor"))
    assert cli.main(PREDICT + options) == 0
    assert captured["bundle_path"] == ("old-native.joblib" if "--bundle" in options else cli.NATIVE_MODEL)
    assert captured["video_path"] == "example.mov"
    assert captured["correction"] == "sc"
    assert "stride" not in captured and "maximum_faces" not in captured  # native inference uses stored config
    assert json.loads(capsys.readouterr().out) == {"native_result": True}


@pytest.mark.parametrize("version", [3, PIPELINE_BUNDLE_VERSION])
def test_predict_explicit_legacy_bundle_preserves_dispatch(monkeypatch, capsys, version):
    bundle = legacy_bundle(version)
    captured = {}
    def load(path):
        assert path == "legacy.joblib"
        return bundle
    def predict(model, **kwargs):
        assert model is bundle
        captured.update(kwargs)
        return {"legacy_result": True}
    monkeypatch.setattr(cli.joblib, "load", load)
    monkeypatch.setattr(cli, "analyze_raw_video", predict)
    monkeypatch.setattr(cli, "predict_native", lambda *a, **k: pytest.fail("legacy model routed to native predictor"))
    assert cli.main(PREDICT + ["--bundle", "legacy.joblib", "--stride", "2", "--maximum-faces", "3"]) == 0
    assert captured["stride"] == 2
    assert captured["maximum_faces"] == 3
    assert json.loads(capsys.readouterr().out) == {"legacy_result": True}


@pytest.mark.parametrize("native,option,value,error", [
    (True, "--stride", "2", "requires extraction stride 1"),
    (True, "--maximum-faces", "3", "requires maximum_faces=2"),
    (False, "--stride", "1", "requires extraction stride 2"),
    (False, "--maximum-faces", "2", "requires maximum_faces=3"),
])
def test_predict_rejects_mismatched_extraction_contract(monkeypatch, native, option, value, error):
    bundle = native_bundle() if native else legacy_bundle()
    monkeypatch.setattr(cli.joblib, "load", lambda _: bundle)
    monkeypatch.setattr(cli, "predict_native", lambda *a, **k: pytest.fail("inference must not run"))
    monkeypatch.setattr(cli, "analyze_raw_video", lambda *a, **k: pytest.fail("inference must not run"))
    with pytest.raises(ValueError, match=error):
        cli.main(PREDICT + [option, value])


def test_predict_rejects_unsupported_bundle_type_and_legacy_version(monkeypatch):
    monkeypatch.setattr(cli.joblib, "load", lambda _: {})
    with pytest.raises(TypeError, match="Expected NativeBundle or PipelineModelBundle"):
        cli.main(PREDICT)
    monkeypatch.setattr(cli.joblib, "load", lambda _: legacy_bundle(999))
    with pytest.raises(ValueError, match="Unsupported pipeline bundle version: 999"):
        cli.main(PREDICT)


def test_native_prediction_prints_missing_measurements_as_valid_json(monkeypatch, capsys):
    monkeypatch.setattr(cli.joblib, "load", lambda _: native_bundle())
    monkeypatch.setattr(cli, "predict_native", lambda *a, **k: {
        "native_measurement": {"px_per_prism_diopter": float("nan"), "uncover_events": np.int64(0)}
    })
    assert cli.main(PREDICT) == 0
    assert json.loads(capsys.readouterr().out) == {
        "native_measurement": {"px_per_prism_diopter": None, "uncover_events": 0}
    }
