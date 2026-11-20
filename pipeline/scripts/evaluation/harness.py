"""Evaluation harness: one fold loop, one report, one shortcut battery.

Every candidate model ("arm") is scored through ``run_arm``, so rows, folds, slices and metrics are
identical for every arm, test rows reach a head only as label-free features, and every run is
appended to the multiplicity log.  This module does not select or tune models.

Public names:

    load(partition, mode)                      -> copies of the 5 fold tables (evaluation metadata under ``meta_``)
    ELIGIBLE / CLASSES / eligible(table, job)  -> row eligibility and class order per job
    run_arm(name, job, mode, partition, fit_predict, view, featurize=None, extra_features=None, ...)
    report(oof, job)                           -> metrics for every reporting slice
    paired_bootstrap(oof_a, oof_b, job, metric)
    g_edit / nan_mask_probe / level_shift_counterfactual / discordant_stratum
    STATIC_LEVEL_PATTERNS / SCALE_CONTAINER_PATTERNS / view_minus / static_cols
    frozen_anchor(job)                         -> the delivered heads, reproduced exactly

New columns come in only two ways (never write into the tables returned by load()):
  featurize(train_df, test_X, ctx) -> (train_df, test_X)   per-fold hook; anything fitted is fit on train_df only.
  extra_features="path/{partition}_{mode}_fold{fold}.csv"  label-free per-video measurements read from disk.
A callable view sees only the fold's training rows.

Evaluation-only metadata (``meta_*``) is used for slicing and fold checks only; it is stripped from
every frame an arm receives, so it cannot become a feature.

Run as:  cd pipeline && PYTHONPATH=src:scripts/evaluation .venv/bin/python <script>
"""
from __future__ import annotations

import hashlib, json, re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score

import common
import run_heads as ref
from strabismus_screen import native_bundle

SEED = 20260901
PARTITIONS = {"v3h0": common.ROOT / "artifacts" / "v3_folds_h0",  # retired track re-election experiment
              "h0": common.ROOT / "artifacts" / "v2_folds_h0",   # SELECTION
              "h1": common.ROOT / "artifacts" / "v2_folds_h1"}   # CONFIRMATION (a re-split of the same videos, not an external test set)
ARMS_DIR = common.ROOT / "reports" / "v2" / "arms"
MULTIPLICITY_LOG = ARMS_DIR / "MULTIPLICITY_LOG.jsonl"
BASELINE_DIR = {"h0": common.ROOT / "reports" / "v2" / "baseline_h0", "h1": common.ROOT / "reports" / "v2" / "baseline_h1"}

CU, ACT = "Cover-Uncover", "Alternate-Cover"
CLASSES: dict[int, tuple[str, ...]] = {1: ref.J1, 2: ref.J2D, 3: ref.J3}
ELIGIBLE: dict[int, Callable[[pd.DataFrame], pd.Series]] = {
    1: lambda t: t["test_type"].eq(CU) & t["job1_label"].notna(),                                   # phoria-only = normal (common.job1_label_from_raw)
    2: lambda t: t["test_type"].eq(ACT) & t["pd_lower"].notna() & t["total_direction_label"].notna(),  # mixed-flick ACT stays excluded
    3: lambda t: t["job3_label"].notna(),
}
LABEL_COL = {1: "job1_label", 2: "total_direction_label", 3: "job3_label"}
# Known at inference (typed by the examiner before the clip is analysed), so they may accompany test rows.
CONTEXT_COLS = ["is_near", "with_correction", "is_cover_uncover"]
META_PREFIX = "meta_"

# ----------------------------------------------------------------------------- view guard
# Never a feature: labels, identifiers, fold bookkeeping, protocol strings and the evaluation-only metadata
# under both its raw and its prefixed names.  ``lg_p_*`` (legacy-probability stacking) is a protocol dead end.
DENY_ALWAYS = frozenset({
    "job1_label", "job3_label", "total_direction_label", "observable_label", "pd_lower", "pd_upper", "pd_mid", "raw_type",
    "split", "fold", "mode", "video_uid", "patient_id_group", "patient_id", "patient", "cohort", "occluder", "distance", "correction",
    "test_type", "source_video_key", "condition", "edited", "pseudo", "new_cohort", "recording_group", "Control", "control",
    "truth", "truth_dir", "truth_lower", "truth_upper", "pred", "pred_dir", "probs",
})
_DENY_REGEX = re.compile(r"^(meta_|lg_p_)|(_label$)|(^label_)|(^truth_)|(^pd_(lower|upper|mid)$)")

# Static eye-position columns (can read pseudo-strabismus appearance instead of eye movement).
STATIC_LEVEL_PATTERNS: tuple[str, ...] = (
    r"^(binoc|right_cov|left_cov)_(right|left)_(nasal|pcr)_(median|p10|p90)$",
    r"^(binoc|right_cov|left_cov)_(nasal_(diff|sum)|pcr_diff)_(median|p10|p90)$",
    r"^(right|left)_nasal_u_median$",
    r"^rich_(raw|roll|pose|pcr)_quiet_(right|left|sum|diff)_median$",
    r"^nat_bin_(corner|rigid)_(vergence|right|left|diff)_median$",
    r"^nat_bin_rigid_",
)
# Pixel-scale and container columns that can detect re-exported (zoomed, re-encoded) clips.
SCALE_CONTAINER_PATTERNS: tuple[str, ...] = (
    r"^nat_q_eye_width_px$",
    r"^nat_q_px_per_pd$",
    r"^nat_bin_rigid_",
    r"^(right|left)_eye_width_px_(median|std)$",
    r"^(right|left)_iris_diameter_px_(median|std)$",
    r"^fps$",
    r"^frame_count$",
    r"^duration_s$",
    r"(^|_)(bitrate|brightness|luma|gray_mean|sharpness|laplacian)(_|$)",   # none present today; guards future columns
)


def _matches(column: str, patterns: Sequence[str]) -> bool:
    return any(re.search(p, column) for p in patterns)


def view_minus(view: Sequence[str], patterns: Sequence[str]) -> list[str]:
    """The view without every column matching any of the regex ``patterns`` (order preserved)."""
    return [c for c in view if not _matches(c, patterns)]


def static_cols(view: Sequence[str]) -> list[str]:
    """Columns of the view that are static levels under STATIC_LEVEL_PATTERNS."""
    return [c for c in view if _matches(c, STATIC_LEVEL_PATTERNS)]


# Module-level alias: inside level_shift_counterfactual the parameter ``static_cols``
# shadows the function above, and that function is needed there to reconcile a
# caller-supplied list against the view.
_static_cols_of_view = static_cols


def check_view(view: Sequence[str], table: pd.DataFrame) -> list[str]:
    """Raise if any view column is metadata, a label, an identifier, missing or non-numeric; return the de-duplicated list."""
    cols = list(dict.fromkeys(view))
    bad = [c for c in cols if c in DENY_ALWAYS or _DENY_REGEX.search(c)]
    if bad:
        raise ValueError(f"view leak-guard: forbidden columns {bad[:8]}{' ...' if len(bad) > 8 else ''}")
    missing = [c for c in cols if c not in table.columns]
    if missing:
        raise KeyError(f"view columns not in the fold table: {missing[:8]}")
    non_numeric = [c for c in cols if not pd.api.types.is_numeric_dtype(table[c])]
    if non_numeric:
        raise TypeError(f"view columns must be numeric: {non_numeric[:8]}")
    if not cols:
        raise ValueError("empty view")
    return cols


def resolve_view(view, table: pd.DataFrame) -> list[str]:
    """``view`` is a list of column names, a native_bundle view name ('base+nat', 'phys', ...) or a callable(train_table) -> list.

    A callable receives only the fold's training rows, so feature selection never sees test rows.
    Named views are resolved from column names alone.
    """
    if isinstance(view, str):
        view = native_bundle.feature_list(table, view)
    elif callable(view):
        train = table[table["split"] == "train"] if "split" in table.columns else table
        view = view(_strip_meta(train).reset_index(drop=True))
    return check_view(list(view), table)


# ----------------------------------------------------------------------------- data
_META_CACHE: pd.DataFrame | None = None
_PRISTINE: dict[tuple[str, str], list[pd.DataFrame]] = {}      # private; never handed out (load() returns deep copies)
_FINGERPRINT: dict[tuple[str, str], list[str]] = {}            # sha256 of columns + values as read from disk
_EXTRA_CACHE: dict[tuple[str, float], tuple[pd.DataFrame, str]] = {}


def _metadata() -> pd.DataFrame:
    """common.video_metadata() plus the source-recording key used to group the video-mode folds (evaluation-only)."""
    global _META_CACHE
    if _META_CACHE is None:
        from strabismus_screen.data import load_dataset

        meta = common.video_metadata()
        # Same key as evaluation_pipeline.make_end_to_end_splits(group_recordings=True): halves of a split recording share it.
        recording = {r.video_uid: f"{r.patient_id}|{r.exam_date.isoformat()}|{r.condition}|{r.test_type}|{r.repeat}"
                     for r in load_dataset(common.DATASET_ROOT).manifest}
        meta["recording_group"] = meta["video_uid"].map(recording)
        if int(meta["edited"].sum()) != 95 or int((meta["pseudo"] != "").sum()) != 11:
            raise ValueError("byte-verified edited list (95) or clinician pseudo list (11) changed")
        _META_CACHE = meta
    return _META_CACHE


def _fingerprint(table: pd.DataFrame) -> str:
    digest = hashlib.sha256(json.dumps([str(c) for c in table.columns]).encode())
    digest.update(pd.util.hash_pandas_object(table, index=False).to_numpy().tobytes())
    return digest.hexdigest()


def _pristine(partition: str, mode: str) -> list[pd.DataFrame]:
    """The fold tables exactly as read from disk (+ meta_), for the harness's own use.  Verified against their fingerprint on every
    call, so a column written into them (a renamed label, a feature standardised on all 778 rows) stops the run instead of being scored."""
    if partition not in PARTITIONS or mode not in ("video", "patient"):
        raise ValueError("partition in {'g0','g1'}, mode in {'video','patient'}")
    key = (partition, mode)
    if key not in _PRISTINE:
        meta = _metadata().rename(columns={c: META_PREFIX + c for c in ("edited", "pseudo", "condition", "source_video_key", "recording_group")})
        tables = []
        for table in common.load_tables(PARTITIONS[partition], mode):
            merged = table.merge(meta, on="video_uid", how="left", validate="one_to_one")   # left merge keeps the table's row order
            if merged[META_PREFIX + "condition"].isna().any():
                raise ValueError("fold table holds videos that are not in the dataset manifest")
            merged[META_PREFIX + "new_cohort"] = merged["cohort"].eq("new")
            tables.append(merged)
        _PRISTINE[key], _FINGERPRINT[key] = tables, [_fingerprint(t) for t in tables]
    for fold, (table, fingerprint) in enumerate(zip(_PRISTINE[key], _FINGERPRINT[key])):
        if _fingerprint(table) != fingerprint:
            raise AssertionError(f"fold table {partition}/{mode}/fold{fold} no longer equals what was read from disk; new columns go through "
                                 "featurize= (per fold, fit on training rows) or extra_features= (a file on disk), never into the tables")
    return _PRISTINE[key]


def load(partition: str, mode: str) -> list[pd.DataFrame]:
    """Copies of the five fold tables (``split`` in {train, test}) with meta_* evaluation columns.

    Writing to them has no effect: run_arm and the battery read private, fingerprinted originals.
    """
    return [table.copy(deep=True) for table in _pristine(partition, mode)]


_ALIAS_LABELS = ("job1_label", "job3_label", "total_direction_label", "raw_type")


def _assert_not_label_alias(table: pd.DataFrame, columns: Sequence[str]) -> None:
    """Tripwire (not a proof) for extra_features: a low-cardinality column that determines a label, or a column rank-identical to the PD."""
    for c in columns:
        x = table[c]
        for label in _ALIAS_LABELS:
            ok = table[label].notna()
            if ok.sum() >= 30 and table.loc[ok, label].nunique() >= 2:
                xv = x[ok].fillna(-9.0e99)
                if xv.nunique() <= 24 and bool((table.loc[ok, label].groupby(xv.to_numpy()).nunique() == 1).all()):
                    raise ValueError(f"extra_features column {c!r} determines {label}: it is a label alias, not a measurement")
        for label in ("pd_lower", "pd_upper"):
            ok = table[label].notna() & x.notna()
            if ok.sum() >= 30 and x[ok].nunique() > 1 and table.loc[ok, label].nunique() > 1 and abs(x[ok].corr(table.loc[ok, label], method="spearman")) > 0.98:
                raise ValueError(f"extra_features column {c!r} is rank-identical to {label}")


def _tables(partition: str, mode: str, extra_features: str | Path | None = None, *, allow_global_extra: bool = False) -> tuple[list[pd.DataFrame], list[dict[str, Any]]]:
    """Pristine fold tables, optionally left-joined with label-free per-video measurements read from disk.

    ``extra_features`` is a csv path; '{partition}', '{mode}', '{fold}' placeholders select a per-fold file.  It holds
    ``video_uid`` plus numeric columns.  Anything fitted on rows belongs in featurize instead.

    A path without '{fold}' uses one file for every fold, which is only valid for measurements from the video pixels
    alone, so it requires ``allow_global_extra=True``.  Its sha256 is logged.
    """
    tables = _pristine(partition, mode)
    if extra_features is None:
        return tables, []
    if "{fold}" not in str(extra_features) and not allow_global_extra:
        raise ValueError(
            f"extra_features={extra_features!r} has no '{{fold}}' placeholder, so one globally-computed file would be joined to the "
            "test rows of every fold. Use a per-fold file, move the computation into featurize, or pass allow_global_extra=True to "
            "declare on the record that this measurement is computed from the video pixels alone and never from labels."
        )
    out, provenance = [], []
    for table in tables:
        fold = int(table["fold"].iloc[0])
        path = Path(str(extra_features).format(partition=partition, mode=mode, fold=fold)).resolve()
        key = (str(path), path.stat().st_mtime)
        if key not in _EXTRA_CACHE:
            _EXTRA_CACHE[key] = (pd.read_csv(path, low_memory=False), hashlib.sha256(path.read_bytes()).hexdigest())
        extra, sha = _EXTRA_CACHE[key]
        columns = [c for c in extra.columns if c != "video_uid"]
        if "video_uid" not in extra.columns or extra["video_uid"].duplicated().any() or not set(extra["video_uid"]) <= set(table["video_uid"]):
            raise ValueError(f"{path.name}: needs a unique video_uid column holding only videos of the fold table")
        bad = [c for c in columns if c in DENY_ALWAYS or _DENY_REGEX.search(c) or c in table.columns or not pd.api.types.is_numeric_dtype(extra[c])]
        if bad or not columns:
            raise ValueError(f"{path.name}: columns forbidden by the view guard, shadowing a fold-table column, or non-numeric: {bad[:8]}")
        merged = table.merge(extra, on="video_uid", how="left", validate="one_to_one")
        _assert_not_label_alias(merged, columns)
        out.append(merged); provenance.append({"fold": fold, "path": str(path), "sha256": sha, "columns": columns})
    return out, provenance


def eligible(table: pd.DataFrame, job: int) -> pd.Series:
    return ELIGIBLE[job](table)


def _strip_meta(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(columns=[c for c in frame.columns if c.startswith(META_PREFIX)])


def _assert_fold_integrity(table: pd.DataFrame, mode: str) -> None:
    tr, te = table[table["split"] == "train"], table[table["split"] == "test"]
    if len(tr) + len(te) != len(table) or set(tr["video_uid"]) & set(te["video_uid"]):
        raise AssertionError("train/test video overlap or unknown split value")
    if mode == "patient" and set(tr["patient_id_group"]) & set(te["patient_id_group"]):
        raise AssertionError("patient overlap between train and test in patient mode")
    if mode == "video" and set(tr[META_PREFIX + "recording_group"]) & set(te[META_PREFIX + "recording_group"]):
        raise AssertionError("source-recording overlap between train and test in video mode")


def _fold_frames(table: pd.DataFrame, job: int, feats: list[str], train_rows: str):
    """(train frame with labels and no meta_, label-free test_X, full eligible test rows kept by the harness only)."""
    elig = eligible(table, job)
    if train_rows == "eligible":
        tr = table[(table["split"] == "train") & elig]
    elif train_rows == "all":
        tr = table[table["split"] == "train"]       # the arm decides what to use; harness.eligible(train_df, job) still works
    else:
        raise ValueError("train_rows in {'eligible','all'}")
    te = table[(table["split"] == "test") & elig].reset_index(drop=True)
    context = [c for c in CONTEXT_COLS if c in table.columns and c not in feats]
    return _strip_meta(tr).reset_index(drop=True), te[feats + context].copy(), te


def _apply_featurize(featurize, train_df: pd.DataFrame, test_X: pd.DataFrame, feats: list[str], ctx: dict) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Run the per-fold hook and return (train_df, test_X, view).  test_X is label-free and id-free, so whatever the hook fits can only
    come from train_df.  It may add, replace or drop feature columns and may drop TRAINING rows; test rows keep their number and order."""
    index, context = test_X.index, [c for c in test_X.columns if c not in feats]
    result = featurize(train_df, test_X.copy(), dict(ctx))
    if not (isinstance(result, tuple) and len(result) == 2 and all(isinstance(f, pd.DataFrame) for f in result)):
        raise TypeError("featurize(train_df, test_X, ctx) must return (train_df, test_X) DataFrames")
    train_df, test_X = result
    if len(test_X) != len(index) or not test_X.index.equals(index):
        raise ValueError("featurize must keep every test row, in order, with its index")
    new_view = [c for c in test_X.columns if c not in context]
    bad = [c for c in new_view if c in DENY_ALWAYS or _DENY_REGEX.search(c) or c not in train_df.columns
           or not pd.api.types.is_numeric_dtype(test_X[c]) or not pd.api.types.is_numeric_dtype(train_df[c])]
    if bad or not new_view:
        raise ValueError(f"featurize columns must be numeric, present in both frames and pass the view guard: {bad[:8]}")
    return train_df, test_X, new_view


# ----------------------------------------------------------------------------- the fold loop
def _labels_from(pred, classes) -> np.ndarray:
    pred = np.asarray(pred)
    if pred.dtype.kind in "iu":
        return np.array([classes[int(i)] for i in pred], dtype=object)
    out = pred.astype(object)
    if not set(out) <= set(classes):
        raise ValueError(f"predicted labels outside {classes}")
    return out


def _check_proba(P, n: int, k: int) -> np.ndarray:
    P = np.asarray(P, dtype=float)
    if P.shape != (n, k):
        raise ValueError(f"probabilities have shape {P.shape}, expected {(n, k)}")
    if not np.all(np.isfinite(P)) or np.any(P < -1e-9) or not np.allclose(P.sum(axis=1), 1.0, atol=1e-3):
        raise ValueError("probabilities must be finite, nonnegative and sum to 1 per row")
    return P


def _oof_base(te: pd.DataFrame, fold: int) -> pd.DataFrame:
    """Evaluation columns of the OOF table."""
    return pd.DataFrame({
        "video_uid": te["video_uid"].to_numpy(), "fold": fold, "patient": te["patient_id_group"].to_numpy(), "cohort": te["cohort"].to_numpy(),
        "test_type": te["test_type"].to_numpy(), "raw_type": te["raw_type"].to_numpy(),
        "source_video_key": te[META_PREFIX + "source_video_key"].to_numpy(), "condition": te[META_PREFIX + "condition"].to_numpy(),
        "edited": te[META_PREFIX + "edited"].to_numpy(bool), "pseudo": te[META_PREFIX + "pseudo"].to_numpy(), "new_cohort": te[META_PREFIX + "new_cohort"].to_numpy(bool),
        "is_near": te["is_near"].to_numpy(float), "with_correction": te["with_correction"].to_numpy(float)})


def _predict_folds(job: int, mode: str, partition: str, fit_predict, view, *, train_rows: str = "eligible", seed: int = SEED,
                   test_transform: Callable[[pd.DataFrame], pd.DataFrame] | None = None, featurize=None,
                   extra_features: str | Path | None = None, allow_global_extra: bool = False) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run the 5 folds and return (OOF table, info).  UNLOGGED: for run_arm, predict_folds and the battery only.
    ``test_transform`` is for the battery only (counterfactual test rows); it runs before ``featurize``."""
    (tables, provenance), classes, pieces, n_features, names = _tables(partition, mode, extra_features, allow_global_extra=allow_global_extra), CLASSES[job], [], 0, []
    for table in tables:
        fold = int(table["fold"].iloc[0])
        _assert_fold_integrity(table, mode)
        feats = resolve_view(view, table)
        n_features = len(feats)
        train_df, test_X, te = _fold_frames(table, job, feats, train_rows)
        if te.empty:
            continue
        if test_transform is not None:
            test_X = test_transform(test_X)
        ctx = {"fold": fold, "seed": seed, "mode": mode, "job": job, "classes": classes, "view": list(feats),
               "groups_train": train_df["patient_id_group"].to_numpy(), "train_rows": train_rows}
        if featurize is not None:
            train_df, test_X, feats = _apply_featurize(featurize, train_df, test_X, feats, ctx)
            ctx.update(view=list(feats), groups_train=train_df["patient_id_group"].to_numpy())
            n_features = len(feats)
        names = sorted(set(names) | set(feats))
        leaked = [c for c in test_X.columns if c in DENY_ALWAYS or _DENY_REGEX.search(c)]
        if leaked or any(c.startswith(META_PREFIX) for c in train_df.columns):
            raise AssertionError(f"structural leakage guard failed: {leaked}")
        out = fit_predict(train_df, test_X, ctx)
        piece = _oof_base(te, fold)
        if job == 2:
            if not isinstance(out, dict) or "pd" not in out or "dir_proba" not in out:
                raise ValueError("job 2 arms return {'pd': ndarray, 'dir_proba': ndarray[n,3]} (+ optional 'dir_pred', 'extra')")
            P = _check_proba(out["dir_proba"], len(te), 3)
            pd_hat = np.asarray(out["pd"], dtype=float)
            if pd_hat.shape != (len(te),) or not np.all(np.isfinite(pd_hat)) or np.any(pd_hat < 0):
                raise ValueError("job 2 'pd' must be finite, nonnegative, one value per test row")
            piece["truth_lower"], piece["truth_upper"] = te["pd_lower"].to_numpy(float), te["pd_upper"].to_numpy(float)
            piece["truth_dir"] = te["total_direction_label"].to_numpy()
            piece["pred_dir_argmax"] = _labels_from(np.argmax(P, axis=1), classes)
            piece["pred_dir"] = _labels_from(out["dir_pred"], classes) if out.get("dir_pred") is not None else piece["pred_dir_argmax"]
            piece["emitted_pd"] = pd_hat
        else:
            P = _check_proba(out["proba"] if isinstance(out, dict) else out, len(te), len(classes))
            piece["truth"] = te[LABEL_COL[job]].to_numpy()
            piece["pred_argmax"] = _labels_from(np.argmax(P, axis=1), classes)
            piece["pred"] = _labels_from(out["pred"], classes) if isinstance(out, dict) and out.get("pred") is not None else piece["pred_argmax"]
        for k, c in enumerate(classes):
            piece[f"p_{c}"] = P[:, k]
        piece["probs"] = [json.dumps(row.tolist()) for row in P]
        for name, values in (out.get("extra", {}) if isinstance(out, dict) else {}).items():
            piece[name] = np.asarray(values)
        pieces.append(piece)
    oof = pd.concat(pieces, ignore_index=True)
    expected = set(tables[0].loc[eligible(tables[0], job), "video_uid"])
    if oof["video_uid"].duplicated().any() or set(oof["video_uid"]) != expected:
        raise AssertionError("every eligible video must be predicted exactly once across the 5 test folds")
    _pristine(partition, mode)   # re-verify: an arm must not have written into the harness's tables while it ran
    return oof, {"n_features": n_features, "view_sha1": hashlib.sha1(json.dumps(names).encode()).hexdigest()[:12], "extra_features": provenance}


def _describe_view(view) -> str:
    return view if isinstance(view, str) else (f"callable:{getattr(view, '__name__', 'view')}" if callable(view) else f"list[{len(list(view))}]")


def _append_log(record: dict[str, Any]) -> None:
    MULTIPLICITY_LOG.parent.mkdir(parents=True, exist_ok=True)
    with MULTIPLICITY_LOG.open("a") as handle:
        handle.write(json.dumps(record, default=float) + "\n")


def _run_logged(record: dict[str, Any], body: Callable[[dict[str, Any]], Any]):
    """Log every arm that is run, including failures.  ``body`` fills the record; the line is written in ``finally``."""
    record.update(status="failed", error=None)
    try:
        result = body(record)
        record["status"] = "ok"
        return result
    except BaseException as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"[:500]
        raise
    finally:
        record["timestamp"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        _append_log(record)


def predict_folds(job: int, mode: str, partition: str, fit_predict, view, *, train_rows: str = "eligible", seed: int = SEED,
                  featurize=None, extra_features: str | Path | None = None, allow_global_extra: bool = False, name: str = "(direct predict_folds)") -> tuple[pd.DataFrame, int]:
    """(OOF table, n_features) without files.  The OOF gives headline metrics, so the call is logged like an arm (dry_run=True), failures included."""
    record = {"name": name, "job": job, "mode": mode, "partition": partition, "n_features": None, "train_rows": train_rows, "seed": seed, "headline": {},
              "notes": "direct predict_folds call (no files written)", "dry_run": True, "view": _describe_view(view), "featurize": featurize is not None,
              "allow_global_extra": bool(allow_global_extra)}

    def body(rec):
        oof, info = _predict_folds(job, mode, partition, fit_predict, view, train_rows=train_rows, seed=seed, featurize=featurize, extra_features=extra_features, allow_global_extra=allow_global_extra)
        rec.update(info, headline=_headline(report(oof, job), job))
        return oof, info["n_features"]
    return _run_logged(record, body)


def run_arm(name: str, job: int, mode: str, partition: str, fit_predict, view, *, train_rows: str = "eligible", seed: int = SEED,
            notes: str = "", out_dir: Path | None = None, log: bool = True, featurize=None, extra_features: str | Path | None = None,
            allow_global_extra: bool = False) -> dict[str, Any]:
    """Score one arm on one (job, mode, partition); write OOF + report and append to the multiplicity log.

    fit_predict(train_df, test_X, ctx):
      train_df  training-fold rows WITH labels (eligible rows, or every training-fold row if train_rows='all'); no meta_ columns.
      test_X    ONLY the view's columns plus CONTEXT_COLS: no labels, ids or metadata, so a head cannot peek or route on them.
      ctx       {'fold','seed','mode','job','classes','groups_train','view','train_rows'}
      returns   jobs 1/3: ndarray [n_test, n_classes] in CLASSES[job] order, or {'proba':..., 'pred': labels};
                job 2: {'pd': emitted PD, 'dir_proba': [n,3]} (+ 'dir_pred'); any arm may add {'extra': {column: array}}.
    featurize / extra_features: see the module docstring; ctx['view'] is the view after featurize.
    Every call logs one line (status 'ok' or 'failed').  ``log=False`` is a test-only dry run and needs ``out_dir``.
    """
    if not log and out_dir is None:
        raise ValueError("log=False is a test-only dry run and needs out_dir; real arms keep the defaults")
    record = {"name": name, "job": job, "mode": mode, "partition": partition, "n_features": None, "train_rows": train_rows, "seed": seed, "headline": {},
              "notes": notes, "dry_run": not log, "view": _describe_view(view), "featurize": featurize is not None,
              "allow_global_extra": bool(allow_global_extra)}

    def body(rec):
        oof, info = _predict_folds(job, mode, partition, fit_predict, view, train_rows=train_rows, seed=seed, featurize=featurize, extra_features=extra_features, allow_global_extra=allow_global_extra)
        rep = report(oof, job)
        rec.update(info, headline=_headline(rep, job))
        directory = Path(out_dir) if out_dir is not None else ARMS_DIR / name
        directory.mkdir(parents=True, exist_ok=True)
        stem = f"{partition}_{mode}_job{job}"
        oof.to_csv(directory / f"{stem}_oof.csv", index=False)
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        (directory / f"{stem}_report.json").write_text(json.dumps({**rec, "status": "ok", "timestamp": stamp, "report": rep}, indent=1, default=float))
        return rep, oof, {"oof": str(directory / f"{stem}_oof.csv"), "report": str(directory / f"{stem}_report.json")}

    rep, oof, paths = _run_logged(record, body)
    return {**record, "report": rep, "oof": oof, "paths": paths}


def _headline(rep: dict, job: int) -> dict[str, float]:
    """Headline metrics for all clips, near, distance and unedited clips."""
    out = {}
    for name in ("all", "near", "distance", "unedited"):
        s = rep.get(name, {})
        if not s.get("n"):
            continue
        if job == 2:
            out[f"mae_pd_{name}"] = round(s["mae_pd"], 4); out[f"direction_ba_{name}"] = round(s["direction"]["balanced_accuracy"], 4)
        else:
            out[f"ba_{name}"] = round(s["balanced_accuracy"], 4)
    return out


# ----------------------------------------------------------------------------- report
def load_oof(path: str | Path) -> pd.DataFrame:
    """Read an OOF csv written by run_arm (``probs`` is a JSON list string)."""
    oof = pd.read_csv(path, keep_default_na=False, na_values=[""], dtype={"pseudo": str})
    oof["pseudo"] = oof["pseudo"].fillna("")
    return oof


def _probs(oof: pd.DataFrame) -> np.ndarray:
    return np.array([json.loads(p) if isinstance(p, str) else list(p) for p in oof["probs"]], dtype=float)


def _ba(truth: np.ndarray, pred: np.ndarray, classes) -> float:
    recalls = [np.mean(pred[truth == c] == c) for c in classes if np.any(truth == c)]
    return float(np.mean(recalls)) if recalls else float("nan")


def _cls_block(truth, pred, classes) -> dict[str, Any]:
    truth, pred = np.asarray(truth, dtype=object), np.asarray(pred, dtype=object)
    if len(truth) == 0:
        return {"n": 0}
    return {"n": int(len(truth)), "balanced_accuracy": _ba(truth, pred, classes), "accuracy": float(np.mean(truth == pred)),
            "recall": {c: (float(np.mean(pred[truth == c] == c)) if np.any(truth == c) else None) for c in classes},
            "n_by_class": {c: int(np.sum(truth == c)) for c in classes},
            "confusion_matrix": {"labels": list(classes), "rows_truth_cols_pred": [[int(np.sum((truth == a) & (pred == b))) for b in classes] for a in classes]}}


_PD_BINS = (("0", lambda m: m == 0), ("1-10", lambda m: (m > 0) & (m <= 10)), ("11-20", lambda m: (m > 10) & (m <= 20)),
            ("21-30", lambda m: (m > 20) & (m <= 30)), ("31-45", lambda m: (m > 30) & (m <= 45)), (">45", lambda m: m > 45))


def _job2_block(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {"n": 0}
    mid = 0.5 * (df["truth_lower"] + df["truth_upper"]).to_numpy(float)
    pd_hat = df["emitted_pd"].to_numpy(float)
    err = pd_hat - mid
    interval_err = np.maximum(0, np.maximum(df["truth_lower"].to_numpy(float) - pd_hat, pd_hat - df["truth_upper"].to_numpy(float)))
    out = {"n": int(len(df)), "mae_pd": float(np.mean(np.abs(err))), "interval_mae_pd": float(np.mean(interval_err)), "bias_pd": float(np.mean(err)),
           "within_5": float(np.mean(np.abs(err) <= 5)), "within_10": float(np.mean(np.abs(err) <= 10)),
           "mae_by_true_magnitude": {name: {"n": int(sel(mid).sum()), "mae_pd": (float(np.mean(np.abs(err[sel(mid)]))) if sel(mid).any() else None)} for name, sel in _PD_BINS}}
    out["direction"] = _cls_block(df["truth_dir"], df["pred_dir"], CLASSES[2])
    return out


def _slice_masks(oof: pd.DataFrame) -> list[tuple[str, np.ndarray]]:
    near, cc = oof["is_near"].to_numpy(float) == 1, oof["with_correction"].to_numpy(float) == 1
    # Built from the effective near/distance values, not the filename code, because the
    # 25 distance corrections change is_near without changing the filename.
    cond = np.array([("N" if n else "D") + ("cc" if c else "sc") for n, c in zip(near, cc)], dtype=object)
    edited, pseudo = oof["edited"].to_numpy(bool), oof["pseudo"].fillna("").to_numpy(object) != ""
    return [("all", np.ones(len(oof), bool)), ("near", near), ("distance", ~near), *[(c, cond == c) for c in ("Nsc", "Ncc", "Dsc", "Dcc")],
            ("sc", ~cc), ("cc", cc), ("edited", edited), ("unedited", ~edited), ("new_cohort_excluded", oof["cohort"].to_numpy(object) != "new"), ("pseudo_listed", pseudo)]


def report(oof: pd.DataFrame, job: int) -> dict[str, Any]:
    """Metrics for every reporting slice of one OOF table."""
    oof = oof.reset_index(drop=True)
    block = _job2_block if job == 2 else (lambda d: _cls_block(d["truth"], d["pred"], CLASSES[job]))
    out: dict[str, Any] = {name: block(oof[mask]) for name, mask in _slice_masks(oof)}
    if job != 2:
        classes, P = list(CLASSES[job]), _probs(oof)
        out["all_argmax"] = _cls_block(oof["truth"], oof["pred_argmax"], CLASSES[job])
        listed = np.flatnonzero(oof["pseudo"].fillna("").to_numpy(object) != "")
        # Report-only (n=11): P(true class) shows whether a hit is supported by the probabilities or only by decision weights.
        out["pseudo_clips"] = [{"source_video_key": oof.at[i, "source_video_key"], "pseudo": oof.at[i, "pseudo"], "truth": oof.at[i, "truth"], "pred": oof.at[i, "pred"],
                                "pred_argmax": oof.at[i, "pred_argmax"], "p_true": float(P[i, classes.index(oof.at[i, "truth"])]), "probs": [round(float(v), 4) for v in P[i]]} for i in listed]
    return out


# ----------------------------------------------------------------------------- paired patient-cluster bootstrap
_METRICS = {"balanced_accuracy": (1, 3), "mae_pd": (2,), "direction_ba": (2,)}


def paired_bootstrap(oof_a: pd.DataFrame, oof_b: pd.DataFrame, job: int, metric: str, n_boot: int = 2000, seed: int = 0) -> dict[str, float]:
    """delta = metric(b) - metric(a) in natural units; p_better = P(b better than a) (LOWER is better for mae_pd; ties are not better).

    Whole patients are resampled with replacement and both arms are scored on the same resampled rows.
    """
    if metric not in _METRICS or job not in _METRICS[metric]:
        raise ValueError(f"metric {metric!r} is not defined for job {job}")
    if oof_a["video_uid"].duplicated().any() or oof_b["video_uid"].duplicated().any() or set(oof_a["video_uid"]) != set(oof_b["video_uid"]):
        raise ValueError("paired bootstrap needs the same video set in both OOF tables")
    a = oof_a.sort_values("video_uid").reset_index(drop=True)
    b = oof_b.sort_values("video_uid").reset_index(drop=True)
    if metric == "mae_pd":
        mid = 0.5 * (a["truth_lower"] + a["truth_upper"]).to_numpy(float)
        if not np.allclose(mid, 0.5 * (b["truth_lower"] + b["truth_upper"]).to_numpy(float)):
            raise ValueError("truth differs between the two OOF tables")
        ea, eb = np.abs(a["emitted_pd"].to_numpy(float) - mid), np.abs(b["emitted_pd"].to_numpy(float) - mid)
        score = lambda idx: (float(ea[idx].mean()), float(eb[idx].mean()))
    else:
        tcol, pcol = ("truth_dir", "pred_dir") if metric == "direction_ba" else ("truth", "pred")
        classes = list(CLASSES[job])
        if not (a[tcol].to_numpy(object) == b[tcol].to_numpy(object)).all():
            raise ValueError("truth differs between the two OOF tables")
        truth = np.array([classes.index(v) for v in a[tcol]])
        hit_a = (np.array([classes.index(v) for v in a[pcol]]) == truth).astype(float)
        hit_b = (np.array([classes.index(v) for v in b[pcol]]) == truth).astype(float)
        onehot = np.eye(len(classes))[truth]

        def score(idx):
            n_k = onehot[idx].sum(axis=0); keep = n_k > 0
            return (float(np.mean((onehot[idx] * hit_a[idx, None]).sum(axis=0)[keep] / n_k[keep])), float(np.mean((onehot[idx] * hit_b[idx, None]).sum(axis=0)[keep] / n_k[keep])))
    patients = a["patient"].to_numpy()
    members = [np.flatnonzero(patients == p) for p in np.unique(patients)]
    full_a, full_b = score(np.arange(len(a)))
    rng, deltas = np.random.default_rng(seed), np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([members[j] for j in rng.integers(0, len(members), len(members))])
        sa, sb = score(idx)
        deltas[i] = sb - sa
    better = deltas < 0 if metric == "mae_pd" else deltas > 0
    return {"metric": metric, "a": full_a, "b": full_b, "delta": full_b - full_a, "ci_low": float(np.percentile(deltas, 2.5)), "ci_high": float(np.percentile(deltas, 97.5)),
            "p_better": float(np.mean(better)), "n_videos": int(len(a)), "n_patients": int(len(members)), "n_boot": int(n_boot)}


# ----------------------------------------------------------------------------- shortcut battery
def _extratrees(seed: int, min_samples_leaf: int = 2):
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    return Pipeline([("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
                     ("clf", ExtraTreesClassifier(n_estimators=400, min_samples_leaf=min_samples_leaf, class_weight="balanced", random_state=seed, n_jobs=3))])


def g_edit(view, partition: str = "h0", threshold: float = 0.65, *, featurize=None, extra_features: str | Path | None = None,
           allow_global_extra: bool = False) -> dict[str, Any]:
    """Can the view tell that a CU clip was re-exported by the editor?  Patient-grouped 5-fold OOF AUC (the harness's own patient folds,
    so the fold-refit features are the ones a head sees).  The label-only reference is ALSO out-of-fold: P(edited | Job 1 class)
    estimated on the training folds.  meta_edited is the probe's TARGET here and never a feature (the view guard still applies)."""
    score, reference, truth, n_features = [], [], [], 0
    for table in _tables(partition, "patient", extra_features, allow_global_extra=allow_global_extra)[0]:
        feats = resolve_view(view, table); n_features = len(feats)
        cu = table[table["test_type"].eq(CU)]
        tr, te = cu[cu["split"] == "train"].reset_index(drop=True), cu[cu["split"] == "test"].reset_index(drop=True)
        y_tr = tr[META_PREFIX + "edited"].to_numpy(int)
        if featurize is not None:   # the probe must see the columns the head sees; the hook gets the same label-free, meta-free test frame
            context = [c for c in CONTEXT_COLS if c in table.columns and c not in feats]
            ctx = {"fold": int(table["fold"].iloc[0]), "seed": SEED, "mode": "patient", "job": 1, "classes": CLASSES[1], "view": list(feats),
                   "groups_train": tr["patient_id_group"].to_numpy(), "train_rows": "all"}
            tr_f, te_f, feats = _apply_featurize(featurize, _strip_meta(tr), te[feats + context].copy(), feats, ctx)
            if len(tr_f) != len(tr):
                raise ValueError("g_edit needs a featurize hook that keeps every training row")
            tr, te, n_features = tr.assign(**{c: tr_f[c].to_numpy() for c in feats}), te.assign(**{c: te_f[c].to_numpy() for c in feats}), len(feats)
        model = _extratrees(0).fit(ref._sanitize(tr[feats]), y_tr)
        score.append(model.predict_proba(ref._sanitize(te[feats]))[:, 1])
        rate = pd.Series(y_tr).groupby(tr["job1_label"].fillna("unlabelled").to_numpy()).mean()
        reference.append(te["job1_label"].fillna("unlabelled").map(rate).fillna(y_tr.mean()).to_numpy(float))
        truth.append(te[META_PREFIX + "edited"].to_numpy(int))
    y, s, r = np.concatenate(truth), np.concatenate(score), np.concatenate(reference)
    auc = float(roc_auc_score(y, s))
    return {"probe": "g_edit", "partition": partition, "n_features": n_features, "n_cu": int(len(y)), "n_edited": int(y.sum()), "oof_auc_edited": auc,
            "label_only_reference_auc_oof": float(roc_auc_score(y, r)), "threshold": threshold, "pass": bool(auc <= threshold)}


def nan_mask_probe(view, job: int, partition: str = "h0", mode: str = "patient", margin: float = 0.07, *, featurize=None,
                   extra_features: str | Path | None = None, allow_global_extra: bool = False) -> dict[str, Any]:
    """Fit on the view's missingness mask only (which eyes were covered, which channels failed).

    Passes if BA <= chance + margin.  Uses the raw view's mask, since a fill hook would hide the pattern.
    A ``featurize`` hook's mask is reported too, and the verdict takes the worse of the two.
    """
    def fit_predict(train_df, test_X, ctx):
        feats = ctx["view"]
        mask_tr = np.isnan(ref._sanitize(train_df[feats])).astype(float)
        mask_te = np.isnan(ref._sanitize(test_X[feats])).astype(float)
        y = np.array([ctx["classes"].index(v) for v in train_df[LABEL_COL[ctx["job"]]]])
        model = ExtraTreesClassifier(n_estimators=400, min_samples_leaf=4, class_weight="balanced", random_state=ctx["seed"], n_jobs=3).fit(mask_tr, y)
        P = ref._align(model.predict_proba(mask_te), model.classes_, len(ctx["classes"]))
        return {"pd": np.zeros(len(test_X)), "dir_proba": P} if ctx["job"] == 2 else P

    tcol, pcol = ("truth_dir", "pred_dir") if job == 2 else ("truth", "pred")
    chance = 1.0 / len(CLASSES[job])

    def measure(hook):   # a probe, not an arm: unlogged
        oof, info = _predict_folds(job, mode, partition, fit_predict, view, featurize=hook, extra_features=extra_features, allow_global_extra=allow_global_extra)
        near = oof["is_near"].to_numpy(float) == 1
        truth, pred = oof[tcol].to_numpy(object), oof[pcol].to_numpy(object)
        return {"n_features": info["n_features"], "n": int(len(oof)), "balanced_accuracy": _ba(truth, pred, CLASSES[job]),
                "ba_near": _ba(truth[near], pred[near], CLASSES[job]), "ba_distance": _ba(truth[~near], pred[~near], CLASSES[job])}

    def post_hook_mask_is_vacuous() -> bool:
        """True when the hook leaves (almost) no NaNs, so the post-hook mask carries nothing and its score is meaningless."""
        table = _tables(partition, mode, extra_features, allow_global_extra=allow_global_extra)[0][0]
        feats = resolve_view(view, table)
        tr, te = table[table["split"] == "train"], table[table["split"] == "test"]
        train_df, test_X = _strip_meta(tr).reset_index(drop=True), te[feats].copy()
        ctx = {"fold": int(table["fold"].iloc[0]), "seed": SEED, "mode": mode, "job": job,
               "classes": CLASSES[job], "groups_train": tr["patient_id_group"].to_numpy(), "view": feats, "train_rows": "eligible"}
        _, post, _ = _apply_featurize(featurize, train_df, test_X, feats, dict(ctx))
        before = int(test_X.isna().to_numpy().sum())
        after = int(post.isna().to_numpy().sum())
        return before > 0 and after <= 0.02 * before

    raw = measure(None)
    first = _tables(partition, mode, extra_features, allow_global_extra=allow_global_extra)[0][0]
    n_varying = int(sum(first[c].isna().any() for c in resolve_view(view, first)))   # raw view columns (before any featurize)
    out = {"probe": "nan_mask_only", "job": job, "partition": partition, "mode": mode, "mask_source": "raw_view_before_featurize",
           "n_columns_with_any_nan": n_varying, **raw, "chance": chance, "threshold": chance + margin}
    worst = raw["balanced_accuracy"]
    if featurize is not None:
        after = measure(featurize)
        if post_hook_mask_is_vacuous():
            # A fill hook leaves no NaNs, so this reading is uninformative; see censoring_efficacy().
            after.update(vacuous=True, note="the hook leaves no NaNs, so this score is chance by construction and is excluded from the verdict; run censoring_efficacy()")
        else:
            after["vacuous"] = False
            worst = max(worst, after["balanced_accuracy"])
        out["after_featurize"] = after
        out["verdict_balanced_accuracy"] = worst
    out["pass"] = bool(worst <= chance + margin)
    return out


def censoring_efficacy(view, featurize, job: int = 1, partition: str = "h0", mode: str = "patient", *, extra_features: str | Path | None = None,
                       allow_global_extra: bool = False, threshold: float = 0.30) -> dict[str, Any]:
    """Did a fill-style hook actually destroy the missingness pattern, or only hide the NaN sentinel?

    Returns the fraction of originally missing test cells recovered by the rule "equals the column's most
    common filled value".  A constant fill (e.g. the median) scores high; a hot-deck fill scores low.
    """
    recovered = missing = false_positive = observed = 0
    post_nan = before_nan = 0
    for table in _tables(partition, mode, extra_features, allow_global_extra=allow_global_extra)[0]:
        feats = resolve_view(view, table)
        tr, te = table[table["split"] == "train"], table[table["split"] == "test"]
        train_df, test_X = _strip_meta(tr).reset_index(drop=True), te[feats].copy()
        ctx = {"fold": int(table["fold"].iloc[0]), "seed": SEED, "mode": mode, "job": job,
               "classes": CLASSES[job], "groups_train": tr["patient_id_group"].to_numpy(), "view": feats, "train_rows": "eligible"}
        before_nan += int(test_X.isna().to_numpy().sum())
        _, post, _ = _apply_featurize(featurize, train_df, test_X, feats, dict(ctx))
        shared = [c for c in feats if c in post.columns]
        post_values = post[shared].to_numpy(float)
        sub = test_X[shared].isna().to_numpy()
        post_nan += int(post[shared].isna().to_numpy().sum())
        for j, column in enumerate(shared):
            filled = post_values[sub[:, j], j]
            if filled.size == 0:
                continue
            finite = filled[np.isfinite(filled)]
            if finite.size == 0:
                continue
            values, counts = np.unique(finite, return_counts=True)
            constant = values[int(np.argmax(counts))]          # the fill value, whatever it was
            hit = np.isclose(post_values[:, j], constant, rtol=0, atol=0)
            recovered += int((hit & sub[:, j]).sum())
            missing += int(sub[:, j].sum())
            false_positive += int((hit & ~sub[:, j]).sum())
            observed += int((~sub[:, j]).sum())
    recall = float(recovered / missing) if missing else 0.0
    return {"probe": "censoring_efficacy", "partition": partition, "mode": mode, "n_missing_cells": missing,
            "trivial_recovery_recall": recall, "false_positive_rate": float(false_positive / observed) if observed else 0.0,
            "nan_cells_before": before_nan, "nan_cells_after": post_nan, "threshold": threshold,
            "pass": bool(recall <= threshold),
            "note": "recall near 1 means the hook only hid the NaN sentinel; the missingness is still readable from the filled values"}


def level_shift_counterfactual(fit_predict, view, job: int, static_cols: Sequence[str] | None = None, shift: float = 20.0, *, partition: str = "h0",
                               mode: str = "patient", train_rows: str = "eligible", seed: int = SEED, eye_width_equiv: float | None = 0.05, featurize=None,
                               extra_features: str | Path | None = None, allow_global_extra: bool = False) -> dict[str, Any]:
    """Fraction of test predictions that change when the static levels of the TEST rows are offset.

    A direction head should ignore apparent eye position (pseudo-strabismus).  Passes ``plus`` and ``minus`` shift
    every test row; ``mixed`` shifts each row randomly, which catches heads that centre on the test batch.
    ``pass`` requires all three to be zero.

    ``shift`` is in PD-equivalents for ``nat_*`` columns; legacy columns use ``eye_width_equiv`` (0.05 eye widths
    ~ 20 PD-eq), or pass None to add the raw shift everywhere.  ``static_cols`` defaults to the view's static
    columns; a supplied list that misses one of them raises.
    """
    in_any_fold = set().union(*[resolve_view(view, table) for table in _tables(partition, mode, extra_features, allow_global_extra=allow_global_extra)[0]])   # a callable view may differ per fold
    derived = _static_cols_of_view(sorted(in_any_fold))
    if static_cols is None:
        requested = list(derived)
    else:
        requested = list(static_cols)
        omitted = sorted(set(derived) - set(requested))
        if omitted:
            raise ValueError(
                f"static_cols omits {len(omitted)} static level(s) the view contains, so the result would understate the "
                f"head's sensitivity: {omitted[:8]}{' ...' if len(omitted) > 8 else ''}. Pass static_cols=None to use the view's own."
            )
    present = [c for c in requested if c in in_any_fold]
    base = {"probe": "level_shift_counterfactual", "job": job, "partition": partition, "mode": mode, "shift_pd_eq": shift, "eye_width_equiv": eye_width_equiv,
            "n_static_cols_requested": len(requested), "n_static_cols_in_view": len(present), "n_static_cols_derived_from_view": len(derived)}
    if not present:
        return {**base, "trivial": True, "fraction_changed": 0.0, "pass": True,
                "note": "the view holds none of the static columns, so predictions cannot change: 0.0 by construction, not measured"}

    def offset_for(column: str) -> float:
        return shift if (eye_width_equiv is None or column.startswith("nat_")) else shift * eye_width_equiv / 20.0

    def shifted(sign: float | None):
        """sign=+1/-1 offsets the whole batch; sign=None draws an independent +-1 per ROW (batch-centring cannot cancel it)."""
        def transform(test_X: pd.DataFrame) -> pd.DataFrame:
            out = test_X.copy()
            signs = (np.full(len(out), sign) if sign is not None
                     else np.where(np.random.default_rng(seed).random(len(out)) < 0.5, -1.0, 1.0))
            for c in (c for c in present if c in out.columns):
                out[c] = out[c] + signs * offset_for(c)
            return out
        return transform

    runs = {name: _predict_folds(job, mode, partition, fit_predict, view, train_rows=train_rows, seed=seed, test_transform=t, featurize=featurize,
                                 extra_features=extra_features, allow_global_extra=allow_global_extra)[0].set_index("video_uid")   # the shift is applied BEFORE featurize; unlogged (no metric is returned)
            for name, t in (("base", None), ("plus", shifted(+1.0)), ("minus", shifted(-1.0)), ("mixed", shifted(None)))}
    pcol = "pred_dir" if job == 2 else "pred"
    ref_pred = runs["base"][pcol]
    passes = ("plus", "minus", "mixed")
    changed = {name: (runs[name][pcol].reindex(ref_pred.index) != ref_pred).to_numpy() for name in passes}
    out = {**base, "trivial": False, "n": int(len(ref_pred)),
           **{f"fraction_changed_{name}": float(changed[name].mean()) for name in passes},
           "fraction_changed": float(np.logical_or.reduce([changed[name] for name in passes]).mean())}
    if job == 2:
        dpd = np.max([np.abs(runs[name]["emitted_pd"].reindex(ref_pred.index) - runs["base"]["emitted_pd"]).to_numpy() for name in passes], axis=0)
        out.update({"pd_mean_abs_change": float(dpd.mean()), "pd_fraction_changed_over_0.5": float(np.mean(dpd > 0.5))})
        out["pass"] = bool(out["fraction_changed"] == 0.0 and out["pd_fraction_changed_over_0.5"] == 0.0)
    else:
        out["pass"] = bool(out["fraction_changed"] == 0.0)
    return out


DISCORDANT_COLUMN = "binoc_nasal_sum_median"   # static vergence level; defines a REPORTING stratum only, never a feature here


def discordant_stratum(oof: pd.DataFrame, partition: str, mode: str) -> dict[str, Any]:
    """Job 1 errors where static appearance contradicts the diagnosis (pseudo-strabismus test).

    eso-looking = top tercile of binoc_nasal_sum_median (both irises nasal), exo-looking = bottom tercile, terciles computed WITHIN
    distance over the eligible CU rows.  Each video's value is read from the fold table in which it is a test row.
    """
    rows = pd.concat([t.loc[(t["split"] == "test") & eligible(t, 1), ["video_uid", "is_near", DISCORDANT_COLUMN]] for t in _pristine(partition, mode)])
    df = oof.merge(rows[["video_uid", DISCORDANT_COLUMN]], on="video_uid", how="left", validate="one_to_one")
    value, looks = df[DISCORDANT_COLUMN].to_numpy(float), np.full(len(df), "", dtype=object)
    for near in (0.0, 1.0):
        sel = (df["is_near"].to_numpy(float) == near) & np.isfinite(value)
        if sel.sum() >= 3:
            lo, hi = np.quantile(value[sel], [1 / 3, 2 / 3])
            looks[sel & (value >= hi)] = "eso_looking"; looks[sel & (value <= lo)] = "exo_looking"
    eso, exo, normal = CLASSES[1]
    truth, pred = df["truth"].to_numpy(object), df["pred"].to_numpy(object)

    def block(true_class, appearance, opposite):
        sel = (truth == true_class) & (looks == appearance)
        n = int(sel.sum())
        out = {"n": n, "wrong_direction_rate": float(np.mean(pred[sel] == opposite)) if n else None, "missed_rate": float(np.mean(pred[sel] == normal)) if n else None,
               "recall": float(np.mean(pred[sel] == true_class)) if n else None}
        for name, near in (("near", 1.0), ("distance", 0.0)):
            s = sel & (df["is_near"].to_numpy(float) == near)
            out[name] = {"n": int(s.sum()), "wrong_direction_rate": float(np.mean(pred[s] == opposite)) if s.any() else None}
        return out

    concordant = lambda true_class, appearance, opposite: {k: v for k, v in block(true_class, appearance, opposite).items() if k in ("n", "wrong_direction_rate", "missed_rate")}
    return {"probe": "discordant_stratum", "partition": partition, "mode": mode, "stratum_column": DISCORDANT_COLUMN, "n_without_static_value": int((~np.isfinite(value)).sum()),
            "exo_tropic_in_eso_looking_tercile": block(exo, "eso_looking", eso), "eso_tropic_in_exo_looking_tercile": block(eso, "exo_looking", exo),
            "reference_exo_tropic_in_exo_looking_tercile": concordant(exo, "exo_looking", eso), "reference_eso_tropic_in_eso_looking_tercile": concordant(eso, "eso_looking", exo)}


# ----------------------------------------------------------------------------- frozen anchor (delivered 6 September heads)
FROZEN = {1: ("ensemble", "base+nat"), 2: ("catboost", "phys"), 3: ("catboost", "base+nat")}


def frozen_anchor(job: int):
    """(fit_predict, view) for the frozen production heads.

    BA decision weights are tuned on inner folds (GroupKFold by patient in patient mode); seeds are
    seed + 10*fold + k for inner members and seed + 100*fold for the final model.  Job 2 uses
    native_bundle.fit_magnitude_head.
    """
    model_name, view_name = FROZEN[job]
    view = lambda table: native_bundle.feature_list(table, view_name)

    def fit_predict_classification(train_df, test_X, ctx):
        feats, classes, fold, seed = ctx["view"], ctx["classes"], ctx["fold"], ctx["seed"]
        Xtr, Xte = ref._sanitize(train_df[feats]), ref._sanitize(test_X[feats])
        ytr = np.array([classes.index(v) for v in train_df[LABEL_COL[ctx["job"]]]])
        groups = ctx["groups_train"] if ctx["mode"] == "patient" else None
        P_in = np.full((len(train_df), len(classes)), np.nan)
        for k, (a, b) in enumerate(ref._inner_splits(ytr, groups, 3, seed + fold)):
            member = ref.make_classifier(model_name, seed + 10 * fold + k, Xtr.shape[1])
            member.fit(Xtr[a], ytr[a])
            P_in[b] = ref._align(member.predict_proba(Xtr[b]), member.named_steps["clf"].classes_, len(classes))
        ok = np.all(np.isfinite(P_in), axis=1)
        weights = ref.bal_acc_weights(P_in[ok], ytr[ok], len(classes))
        model = ref.make_classifier(model_name, seed + 100 * fold, Xtr.shape[1])
        model.fit(Xtr, ytr)
        P = ref._align(model.predict_proba(Xte), model.named_steps["clf"].classes_, len(classes))
        return {"proba": P, "pred": np.argmax(P * weights[None, :], axis=1), "extra": {"ba_weights": [json.dumps(weights.tolist())] * len(test_X)}}

    def fit_predict_job2(train_df, test_X, ctx):
        head = native_bundle.fit_magnitude_head(train_df, magnitude_table=train_df, features=ctx["view"], direction_kind=model_name, magnitude_kind=model_name,
                                                seed=ctx["seed"], fold=ctx["fold"], grouped=ctx["mode"] == "patient", inner_folds=3,
                                                classifier_factory=ref.make_classifier, regressor_factory=ref.make_regressor,
                                                source_provenance={"entrypoint": "v2.harness.frozen_anchor", "mode": ctx["mode"]})
        out = head.predict(test_X)
        return {"pd": out["emitted_pd"], "dir_proba": out["direction_probabilities"], "dir_pred": np.array(out["direction"], dtype=object),
                "extra": {"raw_pd": out["raw_pd"], "cal_pd": out["calibrated_pd"], "slope": np.full(len(test_X), head.slope)}}

    return (fit_predict_job2 if job == 2 else fit_predict_classification), view


def compare_to_baseline(oof: pd.DataFrame, job: int, mode: str, partition: str = "h0") -> dict[str, Any]:
    """Per-video comparison of an OOF table with a saved baseline OOF table."""
    base = load_oof(BASELINE_DIR[partition] / f"job{job}_{mode}_oof.csv")
    truth_json = json.loads((BASELINE_DIR[partition] / f"job{job}_{mode}.json").read_text())
    if set(base["video_uid"]) != set(oof["video_uid"]):
        return {"same_video_set": False, "n_baseline": int(len(base)), "n_arm": int(len(oof))}
    a, b = base.set_index("video_uid"), oof.set_index("video_uid").reindex(base["video_uid"])
    pcols = ("pred_dir", "pred_dir_argmax") if job == 2 else ("pred", "pred_argmax")
    differing = np.zeros(len(a), bool)
    for c in pcols:
        differing |= a[c].to_numpy(object) != b[c].to_numpy(object)
    out = {"same_video_set": True, "n": int(len(a)), "max_abs_probability_diff": float(np.max(np.abs(_probs(a) - _probs(b))))}
    mine, theirs = report(oof, job)["all"], truth_json["all"]
    if job == 2:
        dpd = np.abs(a["emitted_pd"].to_numpy(float) - b["emitted_pd"].to_numpy(float))
        differing |= dpd > 1e-9
        out["max_abs_pd_diff"] = float(dpd.max())
        pairs = [(mine[k], theirs[k]) for k in ("mae_pd", "interval_mae_pd", "bias_pd", "within_5", "within_10")] + [(mine["direction"][k], theirs["direction"][k]) for k in ("balanced_accuracy", "accuracy")]
    else:
        pairs = [(mine[k], theirs[k]) for k in ("balanced_accuracy", "accuracy")] + [(mine["recall"][c], theirs["recall"][c]) for c in CLASSES[job]]
    # every slice the baseline JSON holds, not only 'all'
    rep = report(oof, job)
    for name, block in truth_json.items():
        if isinstance(block, dict) and name in rep and block.get("n") and name != "all":
            key = "mae_pd" if job == 2 else "balanced_accuracy"
            pairs.append((rep[name][key], block[key]))
    out["max_abs_metric_diff"] = float(max(abs(x - y) for x, y in pairs))
    out["n_videos_with_different_prediction"] = int(differing.sum())
    return out
