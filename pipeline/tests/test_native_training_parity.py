"""Job 2 recipe parity, training boundaries, and metric-direction regression."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from strabismus_screen import native_bundle as native


ROOT = Path(__file__).resolve().parents[1]


def load_experiment(name):
    path = ROOT / "scripts/evaluation" / f"{name}.py"
    if not path.exists():  # retired experiment scripts were removed in the 19 Oct 2026 cleanup
        pytest.skip(f"experiment script not present: {path.name}")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def training_rows():
    ids = np.arange(36)
    labels = np.asarray(native.J2D)[ids % 3]
    lower = np.where(labels == "normal", 0, 3 + (ids * 7) % 29).astype(float)
    width = np.where(labels == "normal", 0, (ids % 4) * 2).astype(float)
    return pd.DataFrame({
        "video_uid": [f"v{i}" for i in ids], "patient_id_group": [f"p{i//2}" for i in ids],
        "test_type": "Alternate-Cover", "split": "train", "fold": 2,
        "x": ids.astype(float), "motion": np.sin(ids / 3),
        "pd_lower": lower, "pd_upper": lower + width, "total_direction_label": labels,
    })


class RecordingModel:
    def __init__(self, direction, seed, log):
        self.direction, self.seed, self.log = direction, seed, log
        step = "clf" if direction else "reg"
        estimator = DecisionTreeClassifier(max_depth=3, random_state=seed) if direction else DecisionTreeRegressor(max_depth=3, random_state=seed)
        self.pipeline = Pipeline([(step, estimator)])
        self.predicted_ids = []

    @property
    def named_steps(self):
        return self.pipeline.named_steps

    def fit(self, X, y, **kwargs):
        self.train_ids = X[:, 0].copy()
        self.kwargs = kwargs
        self.log.append(self)
        self.pipeline.fit(X, y, **kwargs)
        return self

    def predict(self, X):
        self.predicted_ids.append(X[:, 0].copy())
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        self.predicted_ids.append(X[:, 0].copy())
        return self.pipeline.predict_proba(X)


def factories(log):
    return (lambda kind, seed, n: RecordingModel(True, seed, log),
            lambda kind, seed, n: RecordingModel(False, seed, log))


@pytest.mark.parametrize("grouped", [False, True])
def test_job2_seed_weight_and_group_recipe(grouped):
    tr = training_rows()
    cu = tr.iloc[[0, 1]].copy()
    cu["video_uid"] = ["aux0", "aux1"]
    cu["x"] += 1000
    cu["test_type"] = "Cover-Uncover"
    magnitude = pd.concat([tr, cu], ignore_index=True)
    log = []
    cf, rf = factories(log)
    head = native.fit_magnitude_head(tr, magnitude_table=magnitude, features=["x", "motion"],
                                     seed=1000, fold=2, grouped=grouped,
                                     classifier_factory=cf, regressor_factory=rf,
                                     source_provenance={"feature_cache": "test-native"})
    classifiers = [m for m in log if m.direction]
    regressors = [m for m in log if not m.direction]
    assert [m.seed for m in classifiers] == [1020, 1021, 1022, 1200]
    assert [m.seed for m in regressors] == [1400, 1600, 1601, 1602]
    assert all(m.kwargs == {} for m in regressors[1:])  # evaluated inner recipe
    expected = 1 / (1 + (magnitude.pd_upper-magnitude.pd_lower).to_numpy()/5)
    expected[-2:] *= .5
    np.testing.assert_array_equal(regressors[0].kwargs["reg__sample_weight"], expected)
    assert all(np.max(m.train_ids) < 1000 for m in classifiers + regressors[1:])
    for model in classifiers[:-1] + regressors[1:]:
        assert len(model.predicted_ids) == 1
        validation = model.predicted_ids[0]
        assert not set(model.train_ids) & set(validation)
        if grouped:
            assert not set(model.train_ids // 2) & set(validation // 2)
    metadata = head.training_metadata
    assert metadata["grouped_inner"] == grouped
    assert metadata["source"] == {"feature_cache": "test-native"}
    assert metadata["direction_training_videos"] == 36
    assert metadata["magnitude_training_videos"] == 38
    assert metadata["direction_training_sha256"] != metadata["magnitude_training_sha256"]


@pytest.mark.parametrize("grouped", [False, True])
def test_evaluator_and_public_head_have_identical_predictions(grouped, monkeypatch):
    runner = load_experiment("run_heads")
    tr = training_rows()
    te = tr.iloc[:6].copy()
    te["video_uid"] = [f"heldout{i}" for i in range(len(te))]
    te["patient_id_group"] = "heldout_patient"
    te["split"] = "test"
    te["x"] += 100
    table = pd.concat([tr, te], ignore_index=True)
    cf, rf = factories([])
    monkeypatch.setattr(runner, "make_classifier", cf)
    monkeypatch.setattr(runner, "make_regressor", rf)
    evaluated = runner.run_job2([table], feature_list_fn=lambda t: ["x", "motion"],
                                dir_model="catboost", mag_model="catboost", grouped=grouped,
                                seed=1000, use_cu_aux=False)
    head = native.fit_magnitude_head(tr, features=["x", "motion"], seed=1000, fold=2,
                                     grouped=grouped, classifier_factory=cf, regressor_factory=rf)
    predicted = head.predict(te)
    oof = evaluated["oof"]
    assert oof.pred_dir.tolist() == predicted["direction"]
    for column, key in (("raw_pd", "raw_pd"), ("cal_pd", "calibrated_pd"), ("emitted_pd", "emitted_pd")):
        np.testing.assert_array_equal(oof[column], predicted[key])
    np.testing.assert_array_equal(np.asarray(oof.probs.tolist()), predicted["direction_probabilities"])
    provenance = evaluated["metrics"]["training_provenance"][0]
    assert provenance["direction_training_sha256"] == head.training_metadata["direction_training_sha256"]
    assert provenance["direction_training_videos"] == len(tr)


def test_bundle_private_entrypoint_uses_public_recipe(monkeypatch):
    captured = {}
    marker = object()
    def fit(table, **kwargs):
        captured.update(kwargs)
        return marker
    monkeypatch.setattr(native, "fit_magnitude_head", fit)
    result = native._fit_magnitude_head(training_rows(), features=["x"], kind="catboost", seed=41, inner_folds=3)
    assert result is marker
    assert captured["seed"] == 41
    assert captured["direction_kind"] == captured["magnitude_kind"] == "catboost"


def test_public_fit_rejects_outer_test_rows_before_fitting():
    tr = training_rows()
    tr.loc[0, "split"] = "test"
    with pytest.raises(ValueError, match="training rows only"):
        native.fit_magnitude_head(tr, features=["x"])


def test_bootstrap_reports_lower_mae_as_better(tmp_path, monkeypatch, capsys):
    bootstrap = load_experiment("paired_bootstrap")
    truth = ["eso", "exo", "normal"] * 8
    rows = [dict(video_uid=f"v{i}", patient=f"p{i//2}", truth_lower=10.0, truth_upper=10.0,
                 truth_dir=cls, pred_dir=cls, emitted_pd=15.0) for i, cls in enumerate(truth)]
    candidate = [dict(row, emitted_pd=11.0) for row in rows]
    path = tmp_path / "paired.json"
    path.write_text(json.dumps({"results": {"job2|test|reference": {"oof": rows}, "job2|test|candidate": {"oof": candidate}}}))
    monkeypatch.setattr(sys, "argv", ["paired_bootstrap", str(path), "job2|test|candidate", "job2|test|reference", "--n", "20"])
    assert bootstrap.main() == 0
    output = capsys.readouterr().out
    result = json.loads(output[output.index("{"):])
    assert result["mae"]["diff"] == -4.0
    assert result["mae"]["p_candidate_better"] == 1.0
    assert result["dir_ba"]["p_candidate_better"] == 0.0  # ties excluded
    assert bootstrap.probability_candidate_better(np.array([-.2, .3, .4, 0]), "ba") == .5
    assert bootstrap.probability_candidate_better(np.array([-.2, .3, .4, 0]), "interval_mae") == .25
