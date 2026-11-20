"""Train and apply the native-resolution research heads as one predictor bundle.

The bundle holds a cover-state model fitted on all training frame annotations, the
feature contract, per-job heads with balanced-accuracy decision weights tuned on inner
out-of-fold predictions, and the abstention layer.

Heads (configurable through :class:`HeadSpec`):

* Job 1, manifest direction (cover-uncover): ensemble of CatBoost, ExtraTrees and
  logistic regression on base + native features.
* Job 2, total horizontal deviation (alternate-cover): CatBoost direction classifier
  and CatBoost regressor on the physics feature set, with affine calibration.
* Job 3, clinical class (all clips): CatBoost on base + native features.

Inference input: one raw video plus test type, near/distance and correction status.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any, Callable, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import PipelineConfig
from .data import IngestedDataset
from .events import CoverEventModel, infer_cover_timeline
from .job1_rich_features import job1_rich_feature_names
from .pipeline import (
    _file_content_sha256,
    build_cover_training_rows,
    build_predicted_feature_table,
    feature_record_from_timeline,
    frame_annotation_table,
    load_cached_observations,
    select_native_feature_names,
    select_task_feature_names,
    validate_observation_sampling,
    OBSERVATION_CACHE_VERSION,
    _normalize_distance,
    _normalize_correction,
)
from .vision import extract_video_observations
from .track_merge import apply_to_result as merge_patient_tracks

NATIVE_BUNDLE_VERSION = 2

# Abstention layer: a hard gradability gate, then the least measurable 20% of
# alternate-cover clips are declined for magnitude.
GATE_FACE_MIN = 0.50          # patient's face observed on at least half the frames
GATE_COVER_KNOWN_MIN = 0.10   # cover state resolved on at least 10% of frames
MEASURABILITY_COLUMNS = ("face_observation_fraction", "nat_q_right_conf_fraction", "nat_q_left_conf_fraction",
                         "all_usable_n", "cover_state_known_fraction")
ABSTENTION_COVERAGE = 0.80


def _numeric_column(table: pd.DataFrame, name: str) -> np.ndarray:
    if name not in table:
        return np.full(len(table), np.nan)
    return pd.to_numeric(table[name], errors="coerce").to_numpy(float)


def gradability(table: pd.DataFrame) -> tuple[np.ndarray, list[list[str]]]:
    """Hard gate: did the front end see the patient well enough to measure anything?"""
    face, usable, known = (_numeric_column(table, c) for c in ("face_observation_fraction", "all_usable_n", "cover_state_known_fraction"))
    no_face = ~np.isfinite(face) | (face < GATE_FACE_MIN)
    no_sample = ~np.isfinite(usable) | (usable <= 0)
    no_cover = ~np.isfinite(known) | (known < GATE_COVER_KNOWN_MIN)
    reasons = [[text for text, bad in (("face observed on fewer than half the frames", a), ("no usable eye-movement sample", b),
                                       ("cover state never resolved", c)) if bad] for a, b, c in zip(no_face, no_sample, no_cover)]
    return ~(no_face | no_sample | no_cover), reasons


def measurability_index(table: pd.DataFrame, reference: dict[str, Sequence[float]]) -> np.ndarray:
    """Mean percentile rank of the front-end quality columns against the training clips (higher = better measured)."""
    per = []
    for name in MEASURABILITY_COLUMNS:
        ref = np.asarray(reference[name], float)
        v = _numeric_column(table, name)
        r = np.searchsorted(ref, v, side="right") / max(len(ref), 1)
        r[~np.isfinite(v)] = 0.0
        per.append(r)
    return np.mean(per, axis=0)
J1 = ("eso_tropia", "exo_tropia", "no_manifest_horizontal_deviation")
J2D = ("eso", "exo", "normal")
J3 = ("tropia", "intermittent_tropia", "phoria", "normal")
PHORIA_TYPES = {"E", "E'", "X"}
PHYS_FEATURES = (
    "nat_unc_corner_long_latefirst_median", "nat_unc_corner_all_latefirst_median", "nat_unc_corner_long_devprefix_median", "nat_unc_corner_all_devprefix_median",
    "nat_unc_rigid_long_latefirst_median", "nat_unc_rigid_all_latefirst_median", "nat_unc_corner_long_step_median", "nat_unc_corner_all_step_median",
    "nat_unc_corner_all_latefirst_abs_median", "nat_unc_corner_long_latefirst_abs_median", "nat_unc_corner_all_step_abs_median",
    "nat_unc_corner_all_latefirst_sign_consistency", "nat_unc_corner_all_latefirst_n", "nat_unc_corner_long_latefirst_n", "nat_unc_corner_all_cover_dur_median_s",
    "nat_cov_corner_all_step_median", "nat_cov_corner_all_latefirst_median", "nat_bin_corner_vergence_median", "nat_q_px_per_pd", "nat_q_head_motion_p90",
    "nat_best_latefirst", "nat_best_latefirst_abs", "nat_best_source", "nat_best_devprefix", "iris_pair_uncover_all_smooth_verylate_median", "uncover_pair_nasal_u_late18_32_all_median",
    "is_cover_uncover", "is_near", "with_correction", "duration_s", "technical_quality_score",
)


# ----------------------------------------------------------------------------- helpers shared with scripts/evaluation/run_heads.py
def add_derived(table: pd.DataFrame) -> pd.DataFrame:
    t = table.copy()
    long_ = pd.to_numeric(t.get("nat_unc_corner_long_latefirst_median"), errors="coerce")
    all_ = pd.to_numeric(t.get("nat_unc_corner_all_latefirst_median"), errors="coerce")
    legacy = pd.to_numeric(t.get("iris_pair_uncover_all_smooth_verylate_median"), errors="coerce")
    scale = pd.to_numeric(t.get("nat_q_px_per_pd"), errors="coerce")
    ew = pd.to_numeric(t.get("nat_q_eye_width_px"), errors="coerce")
    legacy_pd = legacy * ew / scale
    best = long_.where(long_.notna(), all_.where(all_.notna(), legacy_pd))
    t["nat_best_latefirst"] = best
    t["nat_best_latefirst_abs"] = best.abs()
    t["nat_best_source"] = np.where(long_.notna(), 2.0, np.where(all_.notna(), 1.0, np.where(legacy_pd.notna(), 0.0, np.nan)))
    dev_ = pd.to_numeric(t.get("nat_unc_corner_long_devprefix_median"), errors="coerce")
    t["nat_best_devprefix"] = dev_.where(dev_.notna(), pd.to_numeric(t.get("nat_unc_corner_all_devprefix_median"), errors="coerce"))
    return t


def _sanitize(X: pd.DataFrame) -> np.ndarray:
    a = X.to_numpy(dtype=float)
    a[~np.isfinite(a)] = np.nan
    return np.clip(a, -1e6, 1e6)


def bal_acc_weights(P: np.ndarray, y_idx: np.ndarray, n_classes: int, grid=(0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0)) -> np.ndarray:
    w = np.ones(n_classes)

    def ba(wv):
        pred = np.argmax(P * wv[None, :], axis=1)
        return float(np.mean([np.mean(pred[y_idx == k] == k) for k in range(n_classes) if np.any(y_idx == k)]))

    best = ba(w)
    for _ in range(3):
        improved = False
        for k in range(n_classes):
            for g in grid:
                trial = w.copy(); trial[k] = g
                v = ba(trial)
                if v > best + 1e-9:
                    best, w, improved = v, trial, True
        if not improved:
            break
    return w


def _catboost_classifier(seed: int):
    from catboost import CatBoostClassifier
    return Pipeline([("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
                     ("clf", CatBoostClassifier(iterations=400, depth=5, learning_rate=0.05, l2_leaf_reg=6, boosting_type="Plain", rsm=0.5,
                                                auto_class_weights="Balanced", random_seed=seed, verbose=False, thread_count=6, allow_writing_files=False))])


def _extratrees_classifier(seed: int):
    return Pipeline([("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
                     ("clf", ExtraTreesClassifier(n_estimators=400, min_samples_leaf=4, max_features=0.6, class_weight="balanced", random_state=seed, n_jobs=4))])


def _logreg_classifier(seed: int):
    return Pipeline([("imp", SimpleImputer(strategy="median", keep_empty_features=True)), ("sc", StandardScaler()),
                     ("clf", LogisticRegression(C=0.2, class_weight="balanced", max_iter=2000))])


class ProbaEnsemble:
    def __init__(self, members):
        self.members = members

    def fit(self, X, y):
        for m in self.members:
            m.fit(X, y)
        self.classes_ = self.members[0].named_steps["clf"].classes_
        return self

    def predict_proba(self, X):
        out = None
        for m in self.members:
            P = m.predict_proba(X)
            aligned = np.zeros((len(X), len(self.classes_)))
            for j, c in enumerate(m.named_steps["clf"].classes_):
                aligned[:, list(self.classes_).index(c)] = P[:, j]
            out = aligned if out is None else out + aligned
        return out / len(self.members)


def make_classifier(kind: str, seed: int):
    if kind == "catboost":
        return _catboost_classifier(seed)
    if kind == "extratrees":
        return _extratrees_classifier(seed)
    if kind == "logreg":
        return _logreg_classifier(seed)
    if kind == "ensemble":
        return ProbaEnsemble([_catboost_classifier(seed), _extratrees_classifier(seed + 7), _logreg_classifier(seed)])
    raise ValueError(kind)


def make_regressor(kind: str, seed: int):
    if kind == "catboost":
        from catboost import CatBoostRegressor
        return Pipeline([("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
                         ("reg", CatBoostRegressor(iterations=500, depth=4, learning_rate=0.05, l2_leaf_reg=6, loss_function="MAE", rsm=0.5, random_seed=seed, verbose=False, thread_count=6, allow_writing_files=False))])
    if kind == "extratrees":
        return Pipeline([("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
                         ("reg", ExtraTreesRegressor(n_estimators=400, min_samples_leaf=3, max_features=0.7, random_state=seed, n_jobs=4))])
    raise ValueError(kind)


def _align(P: np.ndarray, classes_, n: int) -> np.ndarray:
    out = np.zeros((P.shape[0], n))
    for j, c in enumerate(classes_):
        out[:, int(c)] = P[:, j]
    return out


def _classes_of(model) -> Sequence[Any]:
    return model.classes_ if isinstance(model, ProbaEnsemble) else model.named_steps["clf"].classes_


# ----------------------------------------------------------------------------- specs and bundle
@dataclass
class HeadSpec:
    job1_model: str = "ensemble"
    job1_features: str = "base+nat"
    job2_model: str = "catboost"
    job2_features: str = "phys"
    job3_model: str = "catboost"
    job3_features: str = "base+nat"
    inner_folds: int = 3


@dataclass
class ClassifierHead:
    classes: tuple[str, ...]
    features: tuple[str, ...]
    model: Any
    weights: np.ndarray
    inner_oof_balanced_accuracy: float

    def predict_proba(self, table: pd.DataFrame) -> np.ndarray:
        X = _sanitize(table.reindex(columns=list(self.features)))
        return _align(self.model.predict_proba(X), _classes_of(self.model), len(self.classes))

    def predict(self, table: pd.DataFrame) -> tuple[list[str], np.ndarray]:
        P = self.predict_proba(table)
        idx = np.argmax(P * self.weights[None, :], axis=1)
        return [self.classes[i] for i in idx], P


@dataclass
class MagnitudeHead:
    features: tuple[str, ...]
    direction: ClassifierHead
    regressor: Any
    slope: float
    intercept: float
    inner_oof_mae: float
    training_metadata: dict[str, Any] = field(default_factory=dict)

    def predict(self, table: pd.DataFrame) -> dict[str, Any]:
        X = _sanitize(table.reindex(columns=list(self.features)))
        raw = np.maximum(0.0, self.regressor.predict(X))
        cal = np.maximum(0.0, self.intercept + self.slope * raw)
        dirs, P = self.direction.predict(table)
        emitted = np.where(np.array(dirs) == "normal", 0.0, np.maximum(cal, 1.0))
        return {"direction": dirs, "direction_probabilities": P, "raw_pd": raw, "calibrated_pd": cal, "emitted_pd": emitted}


@dataclass
class NativeBundle:
    cover_model: CoverEventModel
    config: PipelineConfig
    spec: HeadSpec
    job1: ClassifierHead
    job1_no_phoria: ClassifierHead
    job2: MagnitudeHead
    job3: ClassifierHead
    face_landmarker_sha256: str
    training_metadata: dict[str, Any] = field(default_factory=dict)
    bundle_version: int = NATIVE_BUNDLE_VERSION
    # Training-clip reference distributions and threshold for the abstention layer (empty = none).
    abstention: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)

    @classmethod
    def load(cls, path: str | Path) -> "NativeBundle":
        value = joblib.load(path)
        if not isinstance(value, cls):
            raise TypeError(f"Expected {cls.__name__}, found {type(value).__name__}")
        return value


# ----------------------------------------------------------------------------- feature sets
def feature_list(table: pd.DataFrame, name: str) -> list[str]:
    base = list(select_task_feature_names(table))
    nat = list(select_native_feature_names(table)) + [c for c in ("nat_best_latefirst", "nat_best_latefirst_abs", "nat_best_source", "nat_best_devprefix") if c in table]
    rich = list(job1_rich_feature_names(table))
    phys = [c for c in PHYS_FEATURES if c in table]
    sets = {"base": base, "nat": nat, "rich": rich, "phys": phys}
    return list(dict.fromkeys(c for part in name.split("+") for c in sets[part]))


# ----------------------------------------------------------------------------- training
def _fit_classifier_head(table: pd.DataFrame, *, label_col: str, classes: tuple[str, ...], features: list[str], kind: str, seed: int, inner_folds: int, fold: int = 0, grouped: bool = False) -> ClassifierHead:
    cls_idx = {c: i for i, c in enumerate(classes)}
    X = _sanitize(table[features]); y = np.array([cls_idx[v] for v in table[label_col]])
    P_in = np.full((len(y), len(classes)), np.nan)
    groups = table["patient_id_group"].to_numpy() if grouped else None
    for k, (a, b) in enumerate(_job2_inner_splits(y, groups, inner_folds, seed + fold)):
        m = make_classifier(kind, seed + 10 * fold + k); m.fit(X[a], y[a]); P_in[b] = _align(m.predict_proba(X[b]), _classes_of(m), len(classes))
    w = bal_acc_weights(P_in, y, len(classes))
    pred = np.argmax(P_in * w[None, :], axis=1)
    inner_ba = float(np.mean([np.mean(pred[y == k] == k) for k in range(len(classes)) if np.any(y == k)]))
    model = make_classifier(kind, seed + 100 * fold); model.fit(X, y)
    return ClassifierHead(classes=classes, features=tuple(features), model=model, weights=w, inner_oof_balanced_accuracy=inner_ba)


def _job2_inner_splits(y: np.ndarray, groups: np.ndarray | None, n: int, seed: int):
    """The native-v4 Job 2 split recipe, shared by training and evaluation."""
    n = max(2, min(n, int(np.unique(y, return_counts=True)[1].min())))
    if groups is not None:
        if len(np.unique(groups)) < n:
            raise ValueError("Grouped Job 2 training needs enough distinct patient groups for the inner folds")
        return list(GroupKFold(n_splits=n).split(np.zeros(len(y)), y, groups))
    return list(StratifiedKFold(n_splits=n, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))


def _job2_training_fingerprint(table: pd.DataFrame, features: Sequence[str]) -> str:
    columns = list(dict.fromkeys([*features, *[c for c in ("video_uid", "patient_id_group", "test_type", "pd_lower", "pd_upper", "total_direction_label") if c in table]]))
    digest = sha256(json.dumps(columns).encode())
    digest.update(pd.util.hash_pandas_object(table.loc[:, columns], index=False).to_numpy().tobytes())
    return digest.hexdigest()


def fit_magnitude_head(
    table: pd.DataFrame,
    *,
    features: list[str],
    direction_kind: str = "catboost",
    magnitude_kind: str = "catboost",
    seed: int = 20260901,
    fold: int = 0,
    grouped: bool = False,
    inner_folds: int = 3,
    magnitude_table: pd.DataFrame | None = None,
    classifier_factory: Callable[[str, int, int], Any] | None = None,
    regressor_factory: Callable[[str, int, int], Any] | None = None,
    source_provenance: dict[str, Any] | None = None,
) -> MagnitudeHead:
    """Fit the evaluated native Job 2 recipe on training rows only.

    ``table`` supplies ACT rows for direction and inner magnitude calibration.
    Optional ``magnitude_table`` adds CU training rows to the final regressor at
    half weight. Final magnitude fits use 1/(1 + interval width/5) weights;
    inner magnitude fits are deliberately unweighted, matching native-v4.
    Factory hooks preserve the evaluator's exploratory estimator choices.

    Seed offsets are explicit: inner direction +10*fold+k, final direction
    +100*fold, inner magnitude +300*fold+k, final magnitude +200*fold. Both
    inner splitters use seed+fold. No test rows may enter this function.
    """
    if fold < 0 or inner_folds < 2:
        raise ValueError("fold must be nonnegative and inner_folds must be at least two")
    mag = table if magnitude_table is None else magnitude_table
    for name, training in (("direction", table), ("magnitude", mag)):
        if training.empty:
            raise ValueError(f"No {name} training rows")
        if "split" in training and not training["split"].eq("train").all():
            raise ValueError("Job 2 fitting accepts training rows only; filter outer test rows before fitting")
        if "video_uid" in training and training["video_uid"].duplicated().any():
            raise ValueError("Job 2 requires one independent row per video_uid")
        intervals = training[["pd_lower", "pd_upper"]].to_numpy(float)
        if not np.all(np.isfinite(intervals)) or np.any(intervals[:, 0] < 0) or np.any(intervals[:, 1] < intervals[:, 0]):
            raise ValueError("Job 2 requires finite, nonnegative, ordered PD intervals")
    if "test_type" in table and not table["test_type"].eq("Alternate-Cover").all():
        raise ValueError("Direction and inner magnitude calibration require Alternate-Cover training rows")
    if grouped and ("patient_id_group" not in table or table["patient_id_group"].isna().any()):
        raise ValueError("Grouped Job 2 training requires nonmissing patient_id_group values")
    cf = classifier_factory or (lambda kind, state, _: make_classifier(kind, state))
    rf = regressor_factory or (lambda kind, state, _: make_regressor(kind, state))
    X, Xm = _sanitize(table[features]), _sanitize(mag[features])
    yd = np.array([J2D.index(v) for v in table["total_direction_label"]])
    groups = table["patient_id_group"].to_numpy() if grouped else None
    direction_splits = _job2_inner_splits(yd, groups, inner_folds, seed + fold)
    P_in = np.full((len(table), len(J2D)), np.nan)
    for k, (a, b) in enumerate(direction_splits):
        model = cf(direction_kind, seed + 10 * fold + k, X.shape[1])
        model.fit(X[a], yd[a])
        P_in[b] = _align(model.predict_proba(X[b]), _classes_of(model), len(J2D))
    weights = bal_acc_weights(P_in, yd, len(J2D))
    pred = np.argmax(P_in * weights, axis=1)
    inner_ba = float(np.mean([np.mean(pred[yd == k] == k) for k in range(len(J2D)) if np.any(yd == k)]))
    model = cf(direction_kind, seed + 100 * fold, X.shape[1])
    model.fit(X, yd)
    direction = ClassifierHead(J2D, tuple(features), model, weights, inner_ba)

    mid = 0.5 * (mag["pd_lower"].to_numpy(float) + mag["pd_upper"].to_numpy(float))
    width = mag["pd_upper"].to_numpy(float) - mag["pd_lower"].to_numpy(float)
    acquisition_weight = np.where(mag["test_type"] == "Alternate-Cover", 1.0, 0.5) if "test_type" in mag else 1.0
    sw = 1.0 / (1.0 + width / 5.0) * acquisition_weight
    reg = rf(magnitude_kind, seed + 200 * fold, Xm.shape[1])
    used_final_weights = magnitude_kind != "tabpfn"
    try:
        reg.fit(Xm, mid, **({"reg__sample_weight": sw} if used_final_weights else {}))
    except TypeError:
        reg.fit(Xm, mid)
        used_final_weights = False

    mid_a = 0.5 * (table["pd_lower"].to_numpy(float) + table["pd_upper"].to_numpy(float))
    oof = np.full(len(table), np.nan)
    magnitude_splits = _job2_inner_splits((mid_a > 10).astype(int), groups, inner_folds, seed + fold)
    for k, (a, b) in enumerate(magnitude_splits):
        inner_reg = rf(magnitude_kind, seed + 300 * fold + k, X.shape[1])
        inner_reg.fit(X[a], mid_a[a])
        oof[b] = np.maximum(0.0, inner_reg.predict(X[b]))
    valid = np.isfinite(oof)
    slope, intercept = 1.0, 0.0
    if valid.sum() >= 10 and np.std(oof[valid]) > 1e-6:
        hb = HuberRegressor(epsilon=1.5, alpha=1.0).fit(oof[valid].reshape(-1, 1), mid_a[valid])
        if hb.coef_[0] > 0:
            slope, intercept = float(hb.coef_[0]), float(hb.intercept_)
    inner_mae = float(np.mean(np.abs(np.maximum(0.0, intercept + slope * oof[valid]) - mid_a[valid])))
    metadata = {
        "recipe": "native_job2_v1_evaluator_parity", "seed": int(seed), "outer_fold": int(fold),
        "inner_split_seed": int(seed + fold), "grouped_inner": bool(grouped),
        "direction_inner_seeds": [seed + 10 * fold + k for k in range(len(direction_splits))],
        "magnitude_inner_seeds": [seed + 300 * fold + k for k in range(len(magnitude_splits))],
        "direction_final_seed": int(seed + 100 * fold), "magnitude_final_seed": int(seed + 200 * fold),
        "direction_model": direction_kind, "magnitude_model": magnitude_kind,
        "magnitude_inner_weighting": "unweighted", "magnitude_final_weighting": "1/(1+interval_width/5); CU multiplier 0.5" if used_final_weights else "unsupported_by_estimator",
        "direction_training_videos": len(table), "magnitude_training_videos": len(mag),
        "direction_training_sha256": _job2_training_fingerprint(table, features),
        "magnitude_training_sha256": _job2_training_fingerprint(mag, features),
        "source": dict(source_provenance or {}),
    }
    return MagnitudeHead(tuple(features), direction, reg, slope, intercept, inner_mae, metadata)


def _fit_magnitude_head(table: pd.DataFrame, *, features: list[str], kind: str, seed: int, inner_folds: int) -> MagnitudeHead:
    return fit_magnitude_head(table, features=features, direction_kind=kind, magnitude_kind=kind,
                              seed=seed, inner_folds=inner_folds, source_provenance={"entrypoint": "train_native_bundle"})


def train_native_bundle(dataset: IngestedDataset, *, cache_directory: str | Path, config: PipelineConfig, spec: HeadSpec | None = None, output_path: str | Path | None = None, seed: int = 20260901, track_merge: bool = True) -> NativeBundle:
    spec = spec or HeadSpec()
    if config.event.sample_stride != 1:
        raise ValueError("Native training requires sample_stride=1")
    observations = load_cached_observations(dataset, cache_directory, model_path=config.face_landmarker_model, stride=config.event.sample_stride, maximum_faces=config.maximum_faces)
    validate_observation_sampling(observations, expected_stride=config.event.sample_stride)
    for uid, result in observations.items():
        if result.metadata.texture_max_dimension is not None:
            raise ValueError(f"Native training requires native texture statistics: {uid}")
        if not result.observations.empty and not {"right_limbus_x_px", "left_limbus_x_px"}.issubset(result.observations):
            raise ValueError(f"Native limbus observations missing: {uid}")
    if track_merge:  # the identity merge the final model's cache was built with; idempotent on that cache
        observations = {uid: merge_patient_tracks(result) for uid, result in observations.items()}
    annotations = frame_annotation_table(dataset)
    cover_model = CoverEventModel(random_seed=config.random_seed)
    cover_model.fit(build_cover_training_rows(dataset, observations, annotations))
    table = add_derived(build_predicted_feature_table(dataset, observations, cover_model, event_config=config.event, motion_config=config.motion))
    rec = {r.video_uid: r for r in dataset.manifest}
    table["raw_type"] = [rec[u].raw_horizontal_type for u in table["video_uid"]]
    # Job 1
    j1 = table[table["job1_label"].notna()].reset_index(drop=True)
    f1 = feature_list(table, spec.job1_features)
    job1 = _fit_classifier_head(j1, label_col="job1_label", classes=J1, features=f1, kind=spec.job1_model, seed=seed, inner_folds=spec.inner_folds)
    j1b = j1[~j1["raw_type"].isin(PHORIA_TYPES)].reset_index(drop=True)
    job1_no_phoria = _fit_classifier_head(j1b, label_col="job1_label", classes=J1, features=f1, kind=spec.job1_model, seed=seed, inner_folds=spec.inner_folds)
    # Job 2
    j2 = table[(table["test_type"] == "Alternate-Cover") & table["pd_lower"].notna() & table["total_direction_label"].notna()].reset_index(drop=True)
    f2 = feature_list(table, spec.job2_features)
    job2 = _fit_magnitude_head(j2, features=f2, kind=spec.job2_model, seed=seed, inner_folds=spec.inner_folds)
    # Job 3
    j3 = table[table["job3_label"].notna()].reset_index(drop=True)
    f3 = feature_list(table, spec.job3_features)
    job3 = _fit_classifier_head(j3, label_col="job3_label", classes=J3, features=f3, kind=spec.job3_model, seed=seed, inner_folds=spec.inner_folds)
    # Abstention: rank against all training clips; decline the least measurable 20% of magnitude clips.
    reference = {c: np.sort(_numeric_column(table, c)[np.isfinite(_numeric_column(table, c))]).tolist() for c in MEASURABILITY_COLUMNS}
    job2_threshold = float(np.quantile(measurability_index(j2, reference), 1.0 - ABSTENTION_COVERAGE))
    j2_gate, _ = gradability(j2)
    j2_reported = j2_gate & (measurability_index(j2, reference) >= job2_threshold)
    abstention = {"columns": list(MEASURABILITY_COLUMNS), "reference": reference, "job2_threshold": job2_threshold,
                  "coverage_target": ABSTENTION_COVERAGE,
                  "gate": {"face_observation_fraction_min": GATE_FACE_MIN, "all_usable_n_min_exclusive": 0,
                           "cover_state_known_fraction_min": GATE_COVER_KNOWN_MIN}}
    bundle = NativeBundle(
        cover_model=cover_model, config=config, spec=spec, job1=job1, job1_no_phoria=job1_no_phoria, job2=job2, job3=job3, abstention=abstention,
        face_landmarker_sha256=_file_content_sha256(config.face_landmarker_model),
        training_metadata={
            "training_seed": seed, "classifier_recipe": "native_v4_outer_fold0_seed_schedule",
            "videos": int(len(table)), "job1_videos": int(len(j1)), "job1_no_phoria_videos": int(len(j1b)), "job2_videos": int(len(j2)), "job3_videos": int(len(j3)),
            "job1_features": len(f1), "job2_features": len(f2), "job3_features": len(f3), "spec": asdict(spec),
            "ordered_feature_sha256": {
                "job1": sha256(json.dumps(f1).encode()).hexdigest(),
                "job2": sha256(json.dumps(f2).encode()).hexdigest(),
                "job3": sha256(json.dumps(f3).encode()).hexdigest(),
            },
            "job1_inner_oof_ba": job1.inner_oof_balanced_accuracy, "job1_no_phoria_inner_oof_ba": job1_no_phoria.inner_oof_balanced_accuracy,
            "job2_inner_oof_mae": job2.inner_oof_mae, "job2_direction_inner_oof_ba": job2.direction.inner_oof_balanced_accuracy, "job3_inner_oof_ba": job3.inner_oof_balanced_accuracy,
            "job1_weights": job1.weights.tolist(), "job3_weights": job3.weights.tolist(), "job2_direction_weights": job2.direction.weights.tolist(),
            "job2_calibration": {"slope": job2.slope, "intercept": job2.intercept},
            "job2_training_recipe": job2.training_metadata,
            "cache_version": OBSERVATION_CACHE_VERSION, "texture_max_dimension": None,
            "stride": config.event.sample_stride, "note": "inner-OOF figures are training-set diagnostics, not held-out performance",
            "track_merge": bool(track_merge),
            "abstention_summary": {"job2_threshold": job2_threshold, "coverage_target": ABSTENTION_COVERAGE,
                                   "job2_training_clips_reported": int(j2_reported.sum()), "job2_training_clips": int(len(j2))},
        },
    )
    if output_path is not None:
        bundle.save(output_path)
    return bundle


# ----------------------------------------------------------------------------- inference
def predict_native(bundle: NativeBundle, *, video_path: str | Path, test_type: str, distance: str, correction: str, face_landmarker_model: str | Path) -> dict[str, Any]:
    if test_type not in {"Cover-Uncover", "Alternate-Cover"}:
        raise ValueError("test_type must be Cover-Uncover or Alternate-Cover")
    distance = _normalize_distance(distance)
    correction = _normalize_correction(correction)
    if _file_content_sha256(face_landmarker_model) != bundle.face_landmarker_sha256:
        raise ValueError("Face Landmarker asset does not match the bundle")
    result = extract_video_observations(video_path, face_landmarker_model, maximum_faces=bundle.config.maximum_faces, stride=bundle.config.event.sample_stride, native_limbus=True,
                                       texture_max_dimension=bundle.training_metadata.get("texture_max_dimension", 160))
    if bundle.training_metadata.get("track_merge"):
        result = merge_patient_tracks(result)
    timeline = infer_cover_timeline(result.observations, bundle.cover_model, frame_count=result.metadata.frame_count, config=bundle.config.event)
    record = type("InferenceRecord", (), {"video_uid": sha256(str(Path(video_path).resolve()).encode()).hexdigest(), "video_path": Path(video_path), "test_type": test_type,
                                          "distance": "near" if str(distance).lower().startswith("n") else "distance",
                                          "correction": "with_correction" if str(correction).lower() in {"cc", "with_correction"} else "without_correction"})()
    feature = feature_record_from_timeline(record, result, timeline, motion_config=bundle.config.motion, include_dark_iris=False)
    table = add_derived(pd.DataFrame([feature]))
    warnings: list[str] = []
    if result.metadata.patient_track_id is None:
        warnings.append("no_patient_face_track")
    if float(feature.get("nat_unc_corner_all_step_n", 0.0) or 0.0) + float(feature.get("nat_cov_corner_all_step_n", 0.0) or 0.0) < 1.0:
        warnings.append("no_usable_native_event")
    abstention = getattr(bundle, "abstention", None) or {}
    gate_ok, gate_reasons = gradability(table)
    gradable = bool(gate_ok[0])
    measurability = float(measurability_index(table, abstention["reference"])[0]) if abstention else None
    if abstention and not gradable:
        warnings.append("not_gradable")
    label1, P1 = bundle.job1.predict(table)
    label1b, P1b = bundle.job1_no_phoria.predict(table)
    j2 = bundle.job2.predict(table)
    j2_decision, j2_reason = "report", None
    if abstention and not gradable:
        j2_decision, j2_reason = "not gradable - needs examination", "; ".join(gate_reasons[0])
    elif abstention and measurability < abstention["job2_threshold"]:
        j2_decision = "uncertain - needs examination"
        j2_reason = f"measurability {measurability:.3f} below the training threshold {abstention['job2_threshold']:.3f}"
    out = {
        "video": str(video_path),
        "inputs": {"test_type": test_type, "distance": distance, "correction": correction},
        "job1_manifest_direction": {"label": label1[0], "probabilities": dict(zip(J1, P1[0].round(4).tolist())), "label_phoria_excluded_model": label1b[0],
                                    "probabilities_phoria_excluded_model": dict(zip(J1, P1b[0].round(4).tolist())), "warnings": (["job1_requires_cover_uncover"] if test_type != "Cover-Uncover" else []) + warnings},
        "job2_total_horizontal_deviation": {"direction": j2["direction"][0], "direction_probabilities": dict(zip(J2D, j2["direction_probabilities"][0].round(4).tolist())),
                                            "decision": j2_decision, "abstain_reason": j2_reason,
                                            "magnitude_pd": float(j2["emitted_pd"][0]) if j2_decision == "report" else None,
                                            "magnitude_pd_if_forced": float(j2["emitted_pd"][0]),
                                            "unreconciled_magnitude_pd": float(j2["calibrated_pd"][0]), "calibration_mode": "dataset_learned_fallback", "geometric_calibration": False,
                                            "warnings": (["job2_primary_requires_alternate_cover"] if test_type != "Alternate-Cover" else []) + warnings},
        "native_measurement": {"px_per_prism_diopter": feature.get("nat_q_px_per_pd"), "uncover_events": feature.get("nat_unc_corner_all_step_n"), "cover_on_events": feature.get("nat_cov_corner_all_step_n"),
                               "best_uncover_displacement_pd_equiv": float(table["nat_best_latefirst"].iloc[0]) if np.isfinite(table["nat_best_latefirst"].iloc[0]) else None},
        "gradability": {"gradable": gradable, "reasons": gate_reasons[0], "measurability_index": measurability,
                        "job2_threshold": abstention.get("job2_threshold")},
        "video_metadata": asdict(result.metadata),
        "bundle": {"version": bundle.bundle_version, "spec": asdict(bundle.spec)},
    }
    return out


__all__ = ["HeadSpec", "NativeBundle", "PHYS_FEATURES", "add_derived", "feature_list", "fit_magnitude_head", "gradability",
           "measurability_index", "predict_native", "train_native_bundle"]
