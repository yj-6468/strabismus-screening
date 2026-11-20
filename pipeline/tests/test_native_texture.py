"""Native texture provenance and raw/fast-cache-upgrade equivalence."""
from __future__ import annotations

from dataclasses import asdict
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import warnings

import cv2
import joblib
import numpy as np
import pandas as pd
import pytest

from strabismus_screen import native_bundle, pipeline, vision
from strabismus_screen.types import Eye


@pytest.fixture
def upgrader():
    path = Path(__file__).resolve().parents[1] / "experiments/performance_v6/upgrade_native_texture.py"
    if not path.exists():  # retired experiment script, removed in the 19 Oct 2026 cleanup
        pytest.skip(f"experiment script not present: {path.name}")
    spec = importlib.util.spec_from_file_location("native_texture_upgrade_for_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeCapture:
    def __init__(self, frames):
        self.frames = frames
        self.position = 0
        self.released = False

    def isOpened(self):
        return True

    def set(self, *_):
        return True

    def read(self):
        if self.position == len(self.frames):
            return False, None
        frame = self.frames[self.position]
        self.position += 1
        return True, frame.copy()

    def get(self, prop):
        values = {cv2.CAP_PROP_FRAME_COUNT: len(self.frames), cv2.CAP_PROP_FPS: 30.0,
                  cv2.CAP_PROP_FRAME_WIDTH: self.frames[0].shape[1], cv2.CAP_PROP_FRAME_HEIGHT: self.frames[0].shape[0],
                  cv2.CAP_PROP_POS_MSEC: max(0, self.position - 1) * 1000 / 30}
        return values.get(prop, 0)

    def release(self):
        self.released = True


def _metadata(frames, patient=0, texture=160):
    return vision.VideoMetadata(path="synthetic.mov", frame_count=len(frames), fps=30.0,
                                width=frames[0].shape[1], height=frames[0].shape[0], orientation_degrees=0,
                                decoded_frames=len(frames), sampled_frames=len(frames),
                                patient_track_id=patient, texture_max_dimension=texture)


def _points(shift=0.0):
    points = np.zeros((478, 3), dtype=np.float32)
    # This valid float32 contour has x1=538 with raw arithmetic, but x1=539
    # if its cached coordinates are promoted to float64 before ROI construction.
    lo, hi = np.float32(328.1621 + shift), np.float32(483.5976 + shift)
    for eye in (Eye.RIGHT, Eye.LEFT):
        definition = vision.EYE_GEOMETRY[eye]
        offset = 0 if eye is Eye.RIGHT else 100
        mid = (lo + hi) / 2 + offset
        for index in definition["contour"]:
            points[index, :2] = (mid, 180)
        a, b = definition["corners"]
        points[a, :2], points[b, :2] = (lo + offset, 180), (hi + offset, 180)
        a, b = definition["lids"]
        points[a, :2], points[b, :2] = (mid, 160), (mid, 200)
        for index, delta in zip(definition["iris"], (-4, -2, 0, 2, 4), strict=True):
            points[index, :2] = (mid + delta, 180)
        points[definition["cheek"], :2] = (mid, 350)
    return points


def test_native_patch_texture_uses_every_pixel_and_legacy_mode_is_explicit(monkeypatch):
    patch = np.random.default_rng(8).integers(0, 256, (196, 284, 3), dtype=np.uint8)
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    expected_variance = float(np.var(cv2.Laplacian(gray, cv2.CV_32F)) / 255.0**2)
    expected_edges = float(np.mean(cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0),
                                              cv2.Sobel(gray, cv2.CV_32F, 0, 1)) > 28))
    original_resize = cv2.resize
    calls = []

    def resize(*args, **kwargs):
        calls.append(args[0].shape)
        return original_resize(*args, **kwargs)

    monkeypatch.setattr(cv2, "resize", resize)
    native = vision._patch_stats(patch)
    assert calls == []
    assert native["laplacian_variance"] == expected_variance
    assert native["edge_density"] == expected_edges
    legacy = vision._patch_stats(patch, texture_max_dimension=160)
    assert calls == [gray.shape]
    assert legacy["laplacian_variance"] != native["laplacian_variance"]
    # The old cap only affected texture; preserving these cached values is valid.
    for name in ("gray_mean", "gray_std", "gray_q10", "gray_q50", "gray_q90", "saturation_mean"):
        assert native[name] == legacy[name]


def test_fast_upgrade_matches_raw_features_for_all_faces_and_frames(upgrader, monkeypatch):
    rng = np.random.default_rng(91)
    frames = [rng.integers(0, 256, (500, 800, 3), dtype=np.uint8) for _ in range(3)]
    cached_rows, expected_rows = [], []
    for frame_index, frame in enumerate(frames):
        for track in (0, 1):
            points = _points(shift=track * 5)
            row = {"video": "vid_synthetic", "frame": frame_index, "track_id": track,
                   "face_area": .06 + track * .02, "face_center_x": .45 + track * .05, "face_center_y": .5}
            expected = row.copy()
            for eye in (Eye.RIGHT, Eye.LEFT):
                definition = vision.EYE_GEOMETRY[eye]
                old = vision._eye_features(frame, points, eye, nose_x=450, texture_max_dimension=160)
                native = vision._eye_features(frame, points, eye, nose_x=450)
                row.update({f"{eye.value}_{k}": v for k, v in old.items()})
                expected.update({f"{eye.value}_{k}": v for k, v in native.items()})
                for k, (x, y) in enumerate(points[list(definition["contour"]), :2]):
                    row[f"{eye.value}_contour{k}_x_px"] = expected[f"{eye.value}_contour{k}_x_px"] = float(x)
                    row[f"{eye.value}_contour{k}_y_px"] = expected[f"{eye.value}_contour{k}_y_px"] = float(y)
                # Large eye crop; every cheek is below the historical texture cap.
                assert old["eye_width_px"] * .9 + 2 < 160
                assert old["eye_laplacian_variance"] != native["eye_laplacian_variance"]
            cached_rows.append(row)
            expected_rows.append(expected)
    expected = pd.DataFrame(expected_rows)
    selected = vision.select_patient_track(expected, len(frames))
    # A change in texture can change the selected patient. Start with the other face.
    old_patient = 1 - selected
    cached = pd.DataFrame(cached_rows)
    cached["is_patient_track"] = cached.track_id == old_patient
    expected["is_patient_track"] = expected.track_id == selected
    capture = FakeCapture(frames)
    monkeypatch.setattr(upgrader.cv2, "VideoCapture", lambda _: capture)
    assert not upgrader.needs_full_extraction(cached)
    result, changed = upgrader.upgrade(SimpleNamespace(video_path=Path("synthetic.mov"), video_uid="vid_synthetic"),
                                       vision.VisionResult(_metadata(frames, old_patient), cached))
    assert capture.position == len(frames) and capture.released
    assert changed == len(frames) * 2 * 2
    assert result.metadata.decoded_frames == result.metadata.sampled_frames == len(frames)
    assert result.metadata.texture_max_dimension is None
    assert result.metadata.patient_track_id == selected
    pd.testing.assert_frame_equal(result.observations, expected, check_exact=True)
    # Upgrade is a copy; it must not mutate the source cache in memory.
    assert cached["is_patient_track"].tolist() == (cached.track_id == old_patient).tolist()


@pytest.mark.parametrize("width,expected", [(100, False), (175, False), (176, True), (250, True)])
def test_cheek_cap_routes_large_faces_to_full_reextraction(upgrader, width, expected):
    table = pd.DataFrame({"right_eye_width_px": [width], "left_eye_width_px": [100]})
    assert bool(upgrader.needs_full_extraction(table)) is expected


@pytest.mark.parametrize("cache_version,expected_cap", [(3, 160), (4, 160), (5, None)])
def test_cache_texture_provenance_is_versioned(tmp_path, cache_version, expected_cap):
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    metadata = asdict(_metadata([frame]))
    metadata.pop("texture_max_dimension")
    path = tmp_path / "cache.joblib"
    joblib.dump({"cache_version": cache_version, "video_uid": "vid", "signature": "sig", "metadata": metadata,
                 "observations": pd.DataFrame()}, path)
    result = pipeline.load_observation_cache(path)
    assert result.metadata.texture_max_dimension == expected_cap


@pytest.mark.parametrize("kind", ["native", "legacy"])
@pytest.mark.parametrize("metadata,expected_cap", [({}, 160), ({"texture_max_dimension": None}, None)])
def test_bundle_inference_preserves_training_texture_contract(monkeypatch, kind, metadata, expected_cap):
    module = native_bundle if kind == "native" else pipeline
    bundle = SimpleNamespace(training_metadata=metadata, face_landmarker_sha256="hash", extraction_stride=1,
                             maximum_faces=2, config=SimpleNamespace(maximum_faces=2, event=SimpleNamespace(sample_stride=1)))
    observed = {}

    class StopAfterExtractionContract(Exception):
        pass

    def capture_extraction(*_, **kwargs):
        observed.update(kwargs)
        raise StopAfterExtractionContract

    monkeypatch.setattr(module, "_file_content_sha256", lambda _: "hash")
    monkeypatch.setattr(module, "extract_video_observations", capture_extraction)
    function = native_bundle.predict_native if kind == "native" else pipeline.analyze_raw_video
    with pytest.raises(StopAfterExtractionContract):
        function(bundle, video_path="synthetic.mov", test_type="Cover-Uncover", distance="near", correction="cc",
                 face_landmarker_model="model.task")
    assert observed["texture_max_dimension"] == expected_cap
    assert observed["stride"] == 1


def test_extraction_decodes_every_native_frame_by_default(monkeypatch):
    frames = [np.full((184, 326, 3), (i, i + 11, i + 22), dtype=np.uint8) for i in range(4)]
    capture = FakeCapture(frames)
    monkeypatch.setattr(vision.cv2, "VideoCapture", lambda _: capture)
    seen = []

    class Backend:
        last_matrices = []

        def begin_video(self):
            return 10000

        def detect(self, frame, timestamp):
            seen.append((frame.copy(), timestamp))
            return []

    result = vision.extract_video_observations("synthetic.mov", "model.task", backend=Backend())
    assert result.metadata.sampled_frames == result.metadata.decoded_frames == len(frames)
    assert result.metadata.texture_max_dimension is None
    assert result.metadata.width == 326 and result.metadata.height == 184
    assert capture.released
    for (seen_frame, _), source in zip(seen, frames, strict=True):
        np.testing.assert_array_equal(seen_frame, source[..., ::-1])
    assert len({timestamp for _, timestamp in seen}) == len(frames)


def test_numeric_patient_identifiers_cannot_enter_cover_model_features():
    from strabismus_screen.events import _model_columns

    rows = pd.DataFrame({"video": [101, 102], "frame": [0, 1], "timestamp_ms": [0, 33], "track_id": [0, 1],
                         "patient_id": [500, 501], "patient_id_group": [500, 501], "subject_uid": [1, 2], "video_uid": [11, 12],
                         "eye": [0, 1], "eye_state": [0, 1], "covered": [0, 1],
                         "aperture": [.2, .1], "cover_proxy": [.1, .8], "is_right_eye": [1., 0.]})
    assert set(_model_columns(rows)) == {"aperture", "cover_proxy", "is_right_eye"}


def test_classifier_training_matches_outer_fold_seed_and_partition_recipe(monkeypatch):
    # Compare the production refactor with the independently retained evaluation
    # harness, using tiny deterministic classifiers instead of expensive fits.
    path = Path(__file__).resolve().parents[1] / "scripts/evaluation/run_heads.py"
    spec = importlib.util.spec_from_file_location("native_v4_classifier_reference", path)
    reference = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    with warnings.catch_warnings():
        spec.loader.exec_module(reference)
    calls = []

    class RecordingClassifier:
        def __init__(self, seed):
            self.seed = seed
            self.named_steps = {"clf": self}

        def fit(self, X, y):
            self.classes_ = np.unique(y)
            calls.append((self.seed, np.asarray(X).tolist(), np.asarray(y).tolist()))
            return self

        def predict_proba(self, X):
            return np.full((len(X), len(self.classes_)), 1.0 / len(self.classes_))

    monkeypatch.setattr(native_bundle, "make_classifier", lambda kind, seed: RecordingClassifier(seed))
    monkeypatch.setattr(reference, "make_classifier", lambda kind, seed, n_features: RecordingClassifier(seed))
    table = pd.DataFrame({"motion": np.arange(24, dtype=float), "job1_label": list(native_bundle.J1) * 8,
                          "video_uid": [f"vid_{i}" for i in range(24)], "patient_id_group": np.arange(24),
                          "test_type": "Cover-Uncover", "raw_type": "unused", "cohort": "synthetic",
                          "fold": 2, "split": ["train"] * 18 + ["test"] * 6})
    seed = 194
    head = native_bundle._fit_classifier_head(table.iloc[:18], label_col="job1_label", classes=native_bundle.J1,
                                              features=["motion"], kind="recording", seed=seed, inner_folds=3, fold=2)
    production_calls = list(calls)
    calls.clear()
    result = reference.run_classification([table], label_col="job1_label", classes=native_bundle.J1,
                                          eligible=lambda t: t.job1_label.notna(), feature_list_fn=lambda t: ["motion"],
                                          model_name="recording", grouped=False, seed=seed)
    assert calls == production_calls
    assert head.predict(table.iloc[18:])[0] == result["oof"]["pred"].tolist()
