"""Evaluate candidate heads for the three jobs on the per-fold feature tables.

All candidates are trained on the outer-training partition of each fold and
predicted on the outer-test partition; decision policies (balanced-accuracy
class weights, normal/deviation reconciliation) are tuned only on inner
out-of-fold predictions of the outer-training partition.  Out-of-fold
predictions are pooled across folds and scored once per candidate.

Objectives: balanced accuracy for Jobs 1 and 3; for Job 2, MAE against the
interval midpoint plus direction balanced accuracy.

Usage: PYTHONPATH=src .venv/bin/python scripts/evaluation/run_heads.py --mode video --jobs 1 2 3 --out reports/native_v4
"""
from __future__ import annotations

import argparse, json, math, sys, time, warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, LogisticRegression
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from strabismus_screen.evaluation import classification_metrics, pd_regression_metrics  # noqa: E402
from strabismus_screen.pipeline import select_job1_feature_names, select_native_feature_names, select_task_feature_names  # noqa: E402
from strabismus_screen.job1_rich_features import job1_rich_feature_names  # noqa: E402
from strabismus_screen import evaluation_pipeline  # noqa: E402

warnings.filterwarnings("ignore")
TABLES = ROOT / "artifacts" / "native_v4"
PHORIA_TYPES = {"E", "E'", "X"}
J1 = ("eso_tropia", "exo_tropia", "no_manifest_horizontal_deviation")
J2D = ("eso", "exo", "normal")
J3 = ("tropia", "intermittent_tropia", "phoria", "normal")


# ----------------------------------------------------------------------------- utilities
def _sanitize(X: pd.DataFrame) -> np.ndarray:
    a = X.to_numpy(dtype=float)
    a[~np.isfinite(a)] = np.nan
    return np.clip(a, -1e6, 1e6)


def _inner_splits(y: np.ndarray, groups: np.ndarray | None, n: int, seed: int):
    classes, counts = np.unique(y, return_counts=True)
    n = max(2, min(n, int(counts.min())))
    if groups is not None and len(np.unique(groups)) >= n:
        return list(GroupKFold(n_splits=n).split(np.zeros(len(y)), y, groups))
    return list(StratifiedKFold(n_splits=n, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))


def bal_acc_weights(P: np.ndarray, y_idx: np.ndarray, n_classes: int, grid=(0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0)) -> np.ndarray:
    """Multiplicative class weights on probabilities maximizing balanced accuracy (coordinate ascent)."""
    w = np.ones(n_classes)
    def ba(wv):
        pred = np.argmax(P * wv[None, :], axis=1)
        recalls = [np.mean(pred[y_idx == k] == k) for k in range(n_classes) if np.any(y_idx == k)]
        return float(np.mean(recalls))
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


# ----------------------------------------------------------------------------- model factories
class _ProbaEnsemble:
    """Average class probabilities of several fitted sklearn pipelines (same class set)."""

    def __init__(self, members):
        self.members = members

    def fit(self, X, y, **kw):
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

    @property
    def named_steps(self):
        return {"clf": self}


def make_classifier(name: str, seed: int, n_features: int):
    if name == "catboost_ms":
        return _ProbaEnsemble([make_classifier("catboost", seed + 1000 * k, n_features) for k in range(3)])
    if name == "ensemble_ms":
        return _ProbaEnsemble([make_classifier("catboost", seed + 1000 * k, n_features) for k in range(3)] + [make_classifier("extratrees", seed + 7 + 1000 * k, n_features) for k in range(2)] + [make_classifier("logreg", seed, n_features)])
    if name in ("stack", "stack_ms"):
        # the legacy-probability columns are appended by _legacy_augment before fitting; the learner itself is an ensemble
        return make_classifier("ensemble_ms" if name == "stack_ms" else "ensemble", seed, n_features)
    if name == "ensemble":
        return _ProbaEnsemble([make_classifier("catboost", seed, n_features), make_classifier("extratrees", seed + 7, n_features), make_classifier("logreg", seed, n_features)])
    if name == "extratrees":
        return Pipeline([("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
                         ("clf", ExtraTreesClassifier(n_estimators=400, min_samples_leaf=4, max_features=0.6, class_weight="balanced", random_state=seed, n_jobs=4))])
    if name == "rf":
        return Pipeline([("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
                         ("clf", RandomForestClassifier(n_estimators=400, min_samples_leaf=3, max_depth=10, max_features="sqrt", class_weight="balanced", random_state=seed, n_jobs=4))])
    if name in ("logreg", "physlin"):
        return Pipeline([("imp", SimpleImputer(strategy="median", keep_empty_features=True)), ("sc", StandardScaler()),
                         ("clf", LogisticRegression(C=0.2 if name == "logreg" else 1.0, class_weight="balanced", max_iter=2000))])
    if name == "catboost":
        from catboost import CatBoostClassifier
        return Pipeline([("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
                         ("clf", CatBoostClassifier(iterations=400, depth=5, learning_rate=0.05, l2_leaf_reg=6, boosting_type="Plain", rsm=0.5,
                                                    auto_class_weights="Balanced", random_seed=seed, verbose=False, thread_count=6, allow_writing_files=False))])
    if name == "lgbm":
        from lightgbm import LGBMClassifier
        return Pipeline([("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
                         ("clf", LGBMClassifier(n_estimators=400, num_leaves=15, learning_rate=0.03, min_child_samples=12, subsample=0.8, subsample_freq=1, colsample_bytree=0.6,
                                                reg_lambda=5.0, class_weight="balanced", random_state=seed, verbose=-1, n_jobs=4))])
    if name == "tabpfn":
        from tabpfn import TabPFNClassifier
        return Pipeline([("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
                         ("sel", SelectKBest(f_classif, k=min(300, n_features))),
                         ("clf", TabPFNClassifier(device="cpu", random_state=seed))])
    raise ValueError(name)


class _MeanRegressor:
    def __init__(self, members):
        self.members = members
    def fit(self, X, y, **kw):
        for m in self.members:
            try:
                m.fit(X, y, **kw)
            except TypeError:
                m.fit(X, y)
        return self
    def predict(self, X):
        return np.mean([m.predict(X) for m in self.members], axis=0)


def make_regressor(name: str, seed: int, n_features: int = 100000):
    if name == "catboost_ms":
        return _MeanRegressor([make_regressor("catboost", seed + 1000 * k, n_features) for k in range(3)])
    if name == "extratrees":
        return Pipeline([("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
                         ("reg", ExtraTreesRegressor(n_estimators=400, min_samples_leaf=3, max_features=0.7, random_state=seed, n_jobs=4))])
    if name == "catboost":
        from catboost import CatBoostRegressor
        return Pipeline([("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
                         ("reg", CatBoostRegressor(iterations=500, depth=4, learning_rate=0.05, l2_leaf_reg=6, loss_function="MAE", rsm=0.5, random_seed=seed, verbose=False, thread_count=6, allow_writing_files=False))])
    if name == "huber":
        return Pipeline([("imp", SimpleImputer(strategy="median", keep_empty_features=True)), ("sc", StandardScaler()), ("reg", HuberRegressor(epsilon=1.35, alpha=1.0, max_iter=500))])
    if name == "tabpfn":
        from tabpfn import TabPFNRegressor
        return Pipeline([("imp", SimpleImputer(strategy="median", keep_empty_features=True)), ("sel", SelectKBest(f_regression, k=min(300, n_features))), ("reg", TabPFNRegressor(device="cpu", random_state=seed))])
    raise ValueError(name)


# ----------------------------------------------------------------------------- derived columns
def add_derived(table: pd.DataFrame) -> pd.DataFrame:
    """Best-available signed uncover displacement (long-cover native -> any native -> legacy verylate), plus flags."""
    t = table.copy()
    long_ = pd.to_numeric(t.get("nat_unc_corner_long_latefirst_median"), errors="coerce")
    all_ = pd.to_numeric(t.get("nat_unc_corner_all_latefirst_median"), errors="coerce")
    legacy = pd.to_numeric(t.get("iris_pair_uncover_all_smooth_verylate_median"), errors="coerce")
    scale = pd.to_numeric(t.get("nat_q_px_per_pd"), errors="coerce")
    ew = pd.to_numeric(t.get("nat_q_eye_width_px"), errors="coerce")
    legacy_pd = legacy * ew / scale  # eye-width units -> px -> prism-diopter equivalents
    best = long_.where(long_.notna(), all_.where(all_.notna(), legacy_pd))
    t["nat_best_latefirst"] = best
    t["nat_best_latefirst_abs"] = best.abs()
    t["nat_best_source"] = np.where(long_.notna(), 2.0, np.where(all_.notna(), 1.0, np.where(legacy_pd.notna(), 0.0, np.nan)))
    dev_ = pd.to_numeric(t.get("nat_unc_corner_long_devprefix_median"), errors="coerce")
    t["nat_best_devprefix"] = dev_.where(dev_.notna(), pd.to_numeric(t.get("nat_unc_corner_all_devprefix_median"), errors="coerce"))
    return t


# ----------------------------------------------------------------------------- feature sets
def feature_sets(table: pd.DataFrame) -> dict[str, list[str]]:
    base = list(select_task_feature_names(table))
    rich = [c for c in job1_rich_feature_names(table)]
    nat = list(select_native_feature_names(table)) + [c for c in ("nat_best_latefirst", "nat_best_latefirst_abs", "nat_best_source", "nat_best_devprefix") if c in table]
    nat_compact = [c for c in nat if any(s in c for s in ("_step_median", "_step_dominant", "_step_absmax", "_step_abs_p75", "_latefirst_median", "_devprefix_median", "_step_n",
                                                            "_sign_consistency", "vergence_median", "_latency_median_s", "_drift_median", "cover_dur", "nat_q_"))]
    meta = [c for c in ("is_cover_uncover", "is_near", "with_correction", "duration_s", "technical_quality_score") if c in table]
    phys = [c for c in ("nat_unc_corner_long_latefirst_median", "nat_unc_corner_all_latefirst_median", "nat_unc_corner_long_devprefix_median", "nat_unc_corner_all_devprefix_median",
                        "nat_unc_rigid_long_latefirst_median", "nat_unc_rigid_all_latefirst_median", "nat_unc_corner_long_step_median", "nat_unc_corner_all_step_median",
                        "nat_unc_corner_all_latefirst_abs_median", "nat_unc_corner_long_latefirst_abs_median", "nat_unc_corner_all_step_abs_median",
                        "nat_unc_corner_all_latefirst_sign_consistency", "nat_unc_corner_all_latefirst_n", "nat_unc_corner_long_latefirst_n", "nat_unc_corner_all_cover_dur_median_s",
                        "nat_cov_corner_all_step_median", "nat_cov_corner_all_latefirst_median", "nat_bin_corner_vergence_median", "nat_q_px_per_pd", "nat_q_head_motion_p90",
                        "nat_best_latefirst", "nat_best_latefirst_abs", "nat_best_source", "nat_best_devprefix", "iris_pair_uncover_all_smooth_verylate_median", "uncover_pair_nasal_u_late18_32_all_median") if c in table] + meta
    extra = {pref: [c for c in table.columns if c.startswith(pref + "_") and pd.api.types.is_numeric_dtype(table[c])] for pref in ("ph", "ct", "mil")}
    return {"base": base, "rich": rich, "nat": nat, "nat_compact": nat_compact, "meta": meta, "phys": phys, **{k: v for k, v in extra.items() if v}}


# ----------------------------------------------------------------------------- legacy-probability stacking
def _legacy_probs(train: pd.DataFrame, test: pd.DataFrame, job: int, grouped: bool, seed: int) -> np.ndarray:
    """Probabilities of the legacy head fitted on ``train`` and applied to ``test`` (Job 1: DirectionHead; Job 3: ManifestationHead)."""
    from strabismus_screen.modeling import DirectionHead, ManifestationHead
    names = list(select_task_feature_names(train))
    if job == 1:
        j1n = list(select_job1_feature_names(train))
        m = DirectionHead(random_state=seed, manifest_random_state_offset=77)
        m.fit(train.loc[:, j1n], train["job1_label"].to_numpy(dtype=object), clinical_y=train["job3_label"].to_numpy(dtype=object))
        return m.predict_proba(test.loc[:, j1n])
    m = ManifestationHead(random_state=seed)
    m.fit(train.loc[:, names], train["job3_label"].to_numpy(dtype=object), train["observable_label"].to_numpy(dtype=object),
          groups=train["patient_id_group"].to_numpy(dtype=object) if grouped else None)
    return m.predict_proba(test.loc[:, names])


def _legacy_augment(tr: pd.DataFrame, te: pd.DataFrame, job: int, grouped: bool, seed: int, label_col: str):
    """Return (tr, te) with lg_p_k columns: inner-OOF legacy probabilities for tr, full-train legacy probabilities for te."""
    y = tr[label_col].to_numpy(dtype=object); classes_ = np.unique(y); y_idx = np.searchsorted(classes_, y)
    groups = tr["patient_id_group"].to_numpy() if grouped else None
    P_in = np.full((len(tr), 3 if job == 1 else 4), np.nan)
    for k, (a, b) in enumerate(_inner_splits(y_idx, groups, 3, seed + 5)):
        P_in[b] = _legacy_probs(tr.iloc[a].reset_index(drop=True), tr.iloc[b].reset_index(drop=True), job, grouped, seed + 50 + k)
    P_te = _legacy_probs(tr.reset_index(drop=True), te.reset_index(drop=True), job, grouped, seed + 99)
    tr = tr.copy(); te = te.copy()
    for k in range(P_in.shape[1]):
        tr[f"lg_p_{k}"] = P_in[:, k]; te[f"lg_p_{k}"] = P_te[:, k]
    return tr, te


# ----------------------------------------------------------------------------- generic classification runner
def run_classification(tables: list[pd.DataFrame], *, label_col: str, classes: tuple[str, ...], eligible, feature_list_fn, model_name: str,
                       grouped: bool, seed: int, tune_ba: bool = True) -> dict[str, Any]:
    cls_idx = {c: i for i, c in enumerate(classes)}
    oof_rows = []
    for table in tables:
        fold = int(table["fold"].iloc[0])
        tr = table[(table["split"] == "train") & eligible(table)].reset_index(drop=True)
        te = table[(table["split"] == "test") & eligible(table)].reset_index(drop=True)
        if te.empty:
            continue
        feats = feature_list_fn(table)
        if model_name in ("stack", "stack_ms") and label_col in ("job1_label", "job3_label"):
            tr, te = _legacy_augment(tr, te, 1 if label_col == "job1_label" else 3, grouped, seed + fold, label_col)
            feats = list(feats) + [c for c in tr.columns if c.startswith("lg_p_")]
        Xtr, Xte = _sanitize(tr[feats]), _sanitize(te[feats])
        ytr = np.array([cls_idx[v] for v in tr[label_col]])
        groups = tr["patient_id_group"].to_numpy() if grouped else None
        w = np.ones(len(classes))
        if tune_ba:
            P_in = np.full((len(tr), len(classes)), np.nan)
            for k, (a, b) in enumerate(_inner_splits(ytr, groups, 3, seed + fold)):
                m = make_classifier(model_name, seed + 10 * fold + k, Xtr.shape[1])
                m.fit(Xtr[a], ytr[a])
                pr = m.predict_proba(Xtr[b]); P_in[b] = _align(pr, m.named_steps["clf"].classes_, len(classes))
            ok = np.all(np.isfinite(P_in), axis=1)
            w = bal_acc_weights(P_in[ok], ytr[ok], len(classes))
        model = make_classifier(model_name, seed + 100 * fold, Xtr.shape[1])
        model.fit(Xtr, ytr)
        P = _align(model.predict_proba(Xte), model.named_steps["clf"].classes_, len(classes))
        pred = np.argmax(P * w[None, :], axis=1)
        print(json.dumps({"progress": f"{label_col}:{model_name}", "fold": fold, "train": int(len(tr)), "test": int(len(te)), "features": int(Xtr.shape[1])}), flush=True)
        for i, (_, row) in enumerate(te.iterrows()):
            oof_rows.append({"video_uid": row["video_uid"], "truth": row[label_col], "pred": classes[pred[i]], "pred_argmax": classes[int(np.argmax(P[i]))],
                             "probs": P[i].tolist(), "fold": fold, "patient": row["patient_id_group"], "cohort": row.get("cohort", ""), "test_type": row["test_type"], "raw_type": row.get("raw_type", "")})
    df = pd.DataFrame(oof_rows)
    return {"oof": df, "metrics": _cls_metrics(df, classes), "metrics_argmax": _cls_metrics(df, classes, col="pred_argmax")}


def _align(pr: np.ndarray, model_classes, n: int) -> np.ndarray:
    out = np.zeros((pr.shape[0], n))
    for j, c in enumerate(model_classes):
        out[:, int(c)] = pr[:, j]
    return out


def _cls_metrics(df: pd.DataFrame, classes, col: str = "pred", normal=None) -> dict[str, Any]:
    if df.empty:
        return {}
    normal_label = normal or ("no_manifest_horizontal_deviation" if "no_manifest_horizontal_deviation" in classes else "normal")
    m = classification_metrics(df["truth"].tolist(), df[col].tolist(), labels=list(classes), normal_label=normal_label)
    return {k: m[k] for k in ("n_videos", "accuracy", "balanced_accuracy", "macro_f1", "sensitivity_by_class", "confusion_matrix", "deviation_sensitivity", "no_deviation_specificity")}


# ----------------------------------------------------------------------------- Job 2 runner
def run_job2(tables: list[pd.DataFrame], *, feature_list_fn, dir_model: str, mag_model: str, grouped: bool, seed: int, use_cu_aux: bool) -> dict[str, Any]:
    from strabismus_screen.native_bundle import fit_magnitude_head

    rows = []
    training_provenance = []
    for table in tables:
        fold = int(table["fold"].iloc[0])
        elig = table["pd_lower"].notna() & table["total_direction_label"].notna()
        act = elig & (table["test_type"] == "Alternate-Cover")
        tr_dir = table[(table["split"] == "train") & act].reset_index(drop=True)
        tr_mag = table[(table["split"] == "train") & (act | (elig & use_cu_aux))].reset_index(drop=True)
        te = table[(table["split"] == "test") & act].reset_index(drop=True)
        if te.empty:
            continue
        feats = feature_list_fn(table)
        head = fit_magnitude_head(
            tr_dir, magnitude_table=tr_mag, features=feats,
            direction_kind=dir_model, magnitude_kind=mag_model,
            seed=seed, fold=fold, grouped=grouped, inner_folds=3,
            classifier_factory=make_classifier, regressor_factory=make_regressor,
            source_provenance={"entrypoint": "native_v4.run_job2", "mode": "patient" if grouped else "video"},
        )
        output = head.predict(te)
        Pd = output["direction_probabilities"]
        pred_dir = [J2D.index(v) for v in output["direction"]]
        raw, cal, emitted = output["raw_pd"], output["calibrated_pd"], output["emitted_pd"]
        slope = head.slope
        training_provenance.append(head.training_metadata)
        for i, (_, row) in enumerate(te.iterrows()):
            rows.append({"video_uid": row["video_uid"], "truth_lower": float(row["pd_lower"]), "truth_upper": float(row["pd_upper"]), "truth_dir": row["total_direction_label"],
                         "pred_dir": J2D[pred_dir[i]], "pred_dir_argmax": J2D[int(np.argmax(Pd[i]))], "raw_pd": float(raw[i]), "cal_pd": float(cal[i]), "emitted_pd": float(emitted[i]),
                         "probs": Pd[i].tolist(), "fold": fold, "patient": row["patient_id_group"], "cohort": row.get("cohort", ""), "slope": slope})
    df = pd.DataFrame(rows)
    metrics = _job2_metrics(df)
    metrics["training_provenance"] = training_provenance
    return {"oof": df, "metrics": metrics}


def _job2_metrics(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {}
    out = {}
    for col in ("emitted_pd", "cal_pd", "raw_pd"):
        m = pd_regression_metrics(df["truth_lower"].tolist(), df["truth_upper"].tolist(), df[col].tolist())
        out[col] = {k: m[k] for k in ("mae_pd", "interval_mae_pd", "within_5_pd", "within_10_pd", "bland_altman_bias_pd", "bland_altman_lower_loa_pd", "bland_altman_upper_loa_pd", "within_annotated_interval")}
    mid = 0.5 * (df["truth_lower"] + df["truth_upper"])
    for lo, hi in ((0, 0), (1, 10), (11, 20), (21, 30), (31, 45), (46, 99)):
        sel = (mid >= lo) & (mid <= hi)
        if sel.any():
            out[f"emitted_mae_{lo}_{hi}"] = float(np.mean(np.abs(df.loc[sel, "emitted_pd"] - mid[sel])))
            out[f"n_{lo}_{hi}"] = int(sel.sum())
    dm = classification_metrics(df["truth_dir"].tolist(), df["pred_dir"].tolist(), labels=list(J2D), normal_label="normal")
    out["direction"] = {k: dm[k] for k in ("accuracy", "balanced_accuracy", "sensitivity_by_class", "confusion_matrix", "deviation_sensitivity", "no_deviation_specificity")}
    return out


# ----------------------------------------------------------------------------- legacy reference heads
def run_legacy(tables: list[pd.DataFrame], job: int, grouped: bool, seed: int) -> dict[str, Any]:
    rows = []
    for table in tables:
        fold = int(table["fold"].iloc[0])
        tr = table[table["split"] == "train"].copy(); te = table[table["split"] == "test"].copy()
        names = select_task_feature_names(tr)
        fseed = seed + 1000 * (fold + 1)
        if job == 1:
            recs, _ = evaluation_pipeline._job1_records(tr, te, names, grouped_inner_threshold=grouped, random_state=fseed + 1, target_sensitivity=0.95)
            for r in recs:
                p = r["probabilities"]
                rows.append({"video_uid": r["video_uid"], "truth": r["truth"], "pred": r["prediction"], "pred_argmax": max(p, key=p.get), "probs": [p[c] for c in J1], "fold": fold})
        elif job == 2:
            recs, _ = evaluation_pipeline._job2_records(tr, te, names, use_patient_groups=grouped, random_state=fseed + 2)
            for r in recs:
                rows.append({"video_uid": r["video_uid"], "truth_lower": r["truth_lower_pd"], "truth_upper": r["truth_upper_pd"], "truth_dir": r["truth_direction"], "pred_dir": r["predicted_direction"],
                             "pred_dir_argmax": r["predicted_direction"], "raw_pd": r["unreconciled_predicted_pd"], "cal_pd": r["unreconciled_predicted_pd"], "emitted_pd": r["predicted_pd"], "probs": None, "fold": fold})
        else:
            recs, _ = evaluation_pipeline._job3_records(tr, te, names, use_patient_groups=grouped, random_state=fseed + 3)
            for r in recs:
                p = r["clinical_probabilities"]
                rows.append({"video_uid": r["video_uid"], "truth": r["truth_clinical_class"], "pred": r["predicted_clinical_class"], "pred_argmax": r["predicted_clinical_class"], "probs": [p[c] for c in J3], "fold": fold})
    df = pd.DataFrame(rows)
    meta_cols = pd.concat(tables)[["video_uid", "patient_id_group", "test_type", "raw_type", "cohort"]].drop_duplicates("video_uid")
    df = df.merge(meta_cols.rename(columns={"patient_id_group": "patient"}), on="video_uid", how="left")
    if job == 2:
        return {"oof": df, "metrics": _job2_metrics(df)}
    classes = J1 if job == 1 else J3
    return {"oof": df, "metrics": _cls_metrics(df, classes), "metrics_argmax": _cls_metrics(df, classes, col="pred_argmax")}


# ----------------------------------------------------------------------------- driver
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", action="append", choices=("video", "patient"), required=True)
    ap.add_argument("--jobs", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--models", nargs="+", default=["extratrees", "catboost", "logreg"])
    ap.add_argument("--featsets", nargs="+", default=["base", "base+nat", "nat", "base+rich+nat"])
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "native_v4")
    ap.add_argument("--seed", type=int, default=20260901)
    ap.add_argument("--tag", default="")
    ap.add_argument("--tables", type=Path, default=TABLES)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for mode in args.mode:
        tables = [add_derived(pd.read_csv(p, low_memory=False)) for p in sorted(args.tables.glob(f"{mode}_fold*.csv"))]
        if not tables:
            print("no tables for", mode); continue
        grouped = mode == "patient"
        fs = feature_sets(tables[0])
        def fl(name):
            parts = name.split("+")
            return lambda t: list(dict.fromkeys(c for p in parts for c in feature_sets(t)[p]))
        report: dict[str, Any] = {"mode": mode, "results": {}}
        for job in args.jobs:
            if "legacy" in args.models:
                t0 = time.time()
                try:
                    res = run_legacy(tables, job, grouped, args.seed)
                    if job == 1:
                        cond = res["oof"][res["oof"]["truth"] != J1[2]]
                        entry = {"phoria_as_normal": res["metrics"], "phoria_as_normal_argmax": res["metrics_argmax"],
                                 "phoria_excluded_from_scoring": _cls_metrics(res["oof"][~res["oof"]["raw_type"].isin(PHORIA_TYPES)], J1),
                                 "phoria_excluded_train_and_test": _cls_metrics(res["oof"][~res["oof"]["raw_type"].isin(PHORIA_TYPES)], J1),
                                 "conditional_eso_exo": _cls_metrics(cond.assign(pred=np.where(cond["probs"].apply(lambda p: p[0] >= p[1]), J1[0], J1[1])), J1[:2])}
                    elif job == 2:
                        entry = res["metrics"]
                    else:
                        entry = {"pooled": res["metrics"], "pooled_argmax": res["metrics_argmax"], "cover_uncover_only": _cls_metrics(res["oof"][res["oof"]["test_type"] == "Cover-Uncover"], J3),
                                 "alternate_cover_only": _cls_metrics(res["oof"][res["oof"]["test_type"] == "Alternate-Cover"], J3)}
                    entry["seconds"] = round(time.time() - t0, 1)
                    report["results"][f"job{job}|legacy|legacy"] = {"metrics": entry, "oof": res["oof"].to_dict(orient="records")}
                    print(json.dumps({"mode": mode, "key": f"job{job}|legacy", "headline": _headline(job, entry), "sec": entry["seconds"]}), flush=True)
                except Exception as exc:
                    print(json.dumps({"mode": mode, "key": f"job{job}|legacy", "error": str(exc)[:300]}), flush=True)
            for featset in args.featsets:
                for model in [m for m in args.models if m != "legacy"]:
                    key = f"job{job}|{featset}|{model}"
                    t0 = time.time()
                    try:
                        if job == 1:
                            res_a = run_classification(tables, label_col="job1_label", classes=J1, eligible=lambda t: t["job1_label"].notna(), feature_list_fn=fl(featset), model_name=model, grouped=grouped, seed=args.seed)
                            res_b_df = res_a["oof"][~res_a["oof"]["raw_type"].isin(PHORIA_TYPES)]
                            entry = {"phoria_as_normal": res_a["metrics"], "phoria_as_normal_argmax": res_a["metrics_argmax"],
                                     "phoria_excluded_from_scoring": _cls_metrics(res_b_df, J1)}
                            # variant trained without phoria videos
                            res_c = run_classification(tables, label_col="job1_label", classes=J1, eligible=lambda t: t["job1_label"].notna() & ~t["raw_type"].isin(PHORIA_TYPES), feature_list_fn=fl(featset), model_name=model, grouped=grouped, seed=args.seed)
                            entry["phoria_excluded_train_and_test"] = res_c["metrics"]
                            cond = res_a["oof"][res_a["oof"]["truth"] != J1[2]]
                            entry["conditional_eso_exo"] = _cls_metrics(cond.assign(pred=np.where(cond["probs"].apply(lambda p: p[0] >= p[1]), J1[0], J1[1])), J1[:2])
                            oof = {"phoria_as_normal": res_a["oof"].to_dict(orient="records"), "phoria_excluded": res_c["oof"].to_dict(orient="records")}
                        elif job == 2:
                            res = run_job2(tables, feature_list_fn=fl(featset), dir_model=model, mag_model=("catboost_ms" if model in ("catboost_ms", "ensemble_ms", "stack_ms") else "catboost" if model in ("catboost", "lgbm", "ensemble", "stack") else "extratrees" if model == "extratrees" else "tabpfn" if model == "tabpfn" else "huber"), grouped=grouped, seed=args.seed, use_cu_aux=False)
                            entry = res["metrics"]; oof = res["oof"].to_dict(orient="records")
                        else:
                            res = run_classification(tables, label_col="job3_label", classes=J3, eligible=lambda t: t["job3_label"].notna(), feature_list_fn=fl(featset), model_name=model, grouped=grouped, seed=args.seed)
                            entry = {"pooled": res["metrics"], "pooled_argmax": res["metrics_argmax"],
                                     "cover_uncover_only": _cls_metrics(res["oof"][res["oof"]["test_type"] == "Cover-Uncover"], J3),
                                     "alternate_cover_only": _cls_metrics(res["oof"][res["oof"]["test_type"] == "Alternate-Cover"], J3)}
                            oof = res["oof"].to_dict(orient="records")
                        entry["seconds"] = round(time.time() - t0, 1)
                        report["results"][key] = {"metrics": entry, "oof": oof}
                        headline = _headline(job, entry)
                        print(json.dumps({"mode": mode, "key": key, "headline": headline, "sec": entry["seconds"]}), flush=True)
                    except Exception as exc:  # keep going; record the failure
                        report["results"][key] = {"error": f"{type(exc).__name__}: {exc}"}
                        print(json.dumps({"mode": mode, "key": key, "error": str(exc)[:200]}), flush=True)
            (args.out / f"heads_{mode}{args.tag}.json").write_text(json.dumps(report, indent=1, default=float))
    return 0


def _headline(job: int, entry: dict[str, Any]) -> dict[str, Any]:
    if job == 1:
        a = entry["phoria_as_normal"]; c = entry["phoria_excluded_train_and_test"]; d = entry["conditional_eso_exo"]
        return {"BA_phoria_normal": round(a["balanced_accuracy"], 4), "acc": round(a["accuracy"], 4), "BA_phoria_excluded": round(c["balanced_accuracy"], 4), "cond_esoexo_acc": round(d.get("accuracy", float("nan")), 4)}
    if job == 2:
        return {"MAE": round(entry["emitted_pd"]["mae_pd"], 3), "iMAE": round(entry["emitted_pd"]["interval_mae_pd"], 3), "w5": round(entry["emitted_pd"]["within_5_pd"], 3), "w10": round(entry["emitted_pd"]["within_10_pd"], 3),
                "dirBA": round(entry["direction"]["balanced_accuracy"], 4), "dir_acc": round(entry["direction"]["accuracy"], 4), "mae_31_45": round(entry.get("emitted_mae_31_45", float("nan")), 2)}
    p = entry["pooled"]; cu = entry["cover_uncover_only"]
    return {"BA_pooled": round(p["balanced_accuracy"], 4), "acc_pooled": round(p["accuracy"], 4), "BA_cu": round(cu["balanced_accuracy"], 4), "acc_cu": round(cu["accuracy"], 4)}


if __name__ == "__main__":
    raise SystemExit(main())
