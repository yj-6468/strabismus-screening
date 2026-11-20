"""Build leakage-safe per-fold feature tables from the cached eye tracking.

In each outer fold the cover-state model is fitted on training-fold annotations only,
then features for all videos in the fold are computed from its predicted timeline.
Output: <out>/<mode>_fold<k>.csv with labels and split membership.

Usage: PYTHONPATH=src .venv/bin/python scripts/final/build_fold_tables.py --mode video --mode patient --folds 5
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from strabismus_screen.config import load_config  # noqa: E402
from strabismus_screen.data import load_dataset  # noqa: E402
from strabismus_screen.events import CoverEventModel  # noqa: E402
from strabismus_screen import evaluation_pipeline  # noqa: E402
from strabismus_screen.pipeline import (  # noqa: E402
    build_cover_training_rows,
    build_predicted_feature_table,
    frame_annotation_table,
    load_cached_observations,
    validate_observation_sampling,
)

MODES = {"video": "video_level_stratified", "patient": "patient_grouped_audit"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", action="append", choices=list(MODES), required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--dataset-root", type=Path, default=ROOT, help="root containing Older data/New data (e.g. the v2 release folder)")
    ap.add_argument("--group-recordings", action="store_true", help="keep clips cut from one source recording in the same video-level fold")
    ap.add_argument("--stratify-distance", action="store_true", help="add near/distance to the fold strata")
    ap.add_argument("--seed-offset", type=int, default=0, help="added to the config random seed to draw a different fold partition (repeated CV)")
    ap.add_argument("--cache-dir", type=Path, default=ROOT / "artifacts" / "observations_native")
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "stride1.yaml")
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "native_v4")
    ap.add_argument("--fold-limit", type=int)
    ap.add_argument("--fold", type=int, help="build only this fold index (for parallel shards)")
    ap.add_argument("--cover-features", choices=("appearance", "limbus"), default="appearance", help="limbus = add native limbus evidence to the cover-state model")
    ap.add_argument("--extra-features", type=Path, help="label-free per-video CSV (video_uid + columns) merged into every fold table")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(args.dataset_root, expected_video_count=774)
    config = load_config(args.config)
    observations = load_cached_observations(dataset, args.cache_dir, model_path=config.face_landmarker_model, stride=config.event.sample_stride, maximum_faces=config.maximum_faces)
    validate_observation_sampling(observations, expected_stride=config.event.sample_stride)
    annotations = frame_annotation_table(dataset)
    split_map = evaluation_pipeline.make_end_to_end_splits(dataset, n_splits=args.folds, random_state=config.random_seed + args.seed_offset,
                                                           include_video_level="video" in args.mode, include_patient_grouped="patient" in args.mode, include_cohort_holdout=False, group_recordings=args.group_recordings, stratify_by_distance=args.stratify_distance)
    manifest = list(dataset.manifest)
    meta = {"folds": {}, "config": config.to_dict(), "cache_dir": str(args.cache_dir)}
    for mode in args.mode:
        split_name = MODES[mode]
        splits = split_map[split_name]
        if args.fold_limit:
            splits = splits[: args.fold_limit]
        for fold, (train_index, test_index) in enumerate(splits):
            if args.fold is not None and fold != args.fold:
                continue
            t0 = time.time()
            train_uids = {manifest[int(i)].video_uid for i in train_index}
            test_uids = {manifest[int(i)].video_uid for i in test_index}
            assert not (train_uids & test_uids)
            use_limbus = args.cover_features == "limbus"
            cover_rows = build_cover_training_rows(dataset, observations, annotations, allowed_video_uids=train_uids, include_limbus=use_limbus)
            assert not (set(cover_rows["video"].astype(str)) & test_uids)
            cover_model = CoverEventModel(random_seed=config.random_seed + fold, include_limbus=use_limbus)
            cover_model.fit(cover_rows)
            heldout_rows = build_cover_training_rows(dataset, observations, annotations, allowed_video_uids=test_uids, include_limbus=use_limbus)
            cover_diag = evaluation_pipeline._event_test_diagnostic(cover_model, heldout_rows)
            table = build_predicted_feature_table(dataset, observations, cover_model, event_config=config.event, motion_config=config.motion, video_uids=train_uids | test_uids)
            table["split"] = np.where(table["video_uid"].isin(train_uids), "train", "test")
            table["fold"] = fold
            table["mode"] = mode
            # extra labels for the two Job 1 variants and pooled Job 3
            rec = {r.video_uid: r for r in manifest}
            table["raw_type"] = [rec[u].raw_horizontal_type for u in table["video_uid"]]
            table["cohort"] = [rec[u].cohort for u in table["video_uid"]]
            table["occluder"] = [rec[u].summary.occluder_type for u in table["video_uid"]]
            if args.extra_features is not None:
                extra = pd.read_csv(args.extra_features).drop_duplicates("video_uid")
                table = table.merge(extra, on="video_uid", how="left", validate="one_to_one")
            out = args.out / f"{mode}_fold{fold}.csv"
            table.to_csv(out, index=False)
            meta["folds"][f"{mode}_fold{fold}"] = {"train": len(train_uids), "test": len(test_uids), "seconds": round(time.time() - t0, 1), "columns": len(table.columns),
                                                   "cover_features": args.cover_features, "heldout_cover_balanced_accuracy": cover_diag["balanced_accuracy"], "heldout_cover_accuracy": cover_diag["accuracy"]}
            print(json.dumps({"mode": mode, "fold": fold, "train": len(train_uids), "test": len(test_uids), "sec": round(time.time() - t0, 1), "cols": len(table.columns),
                              "cover_features": args.cover_features, "heldout_cover_ba": round(cover_diag["balanced_accuracy"], 4)}), flush=True)
    suffix = "" if args.fold is None else f"_fold{args.fold}"
    (args.out / f"build_meta{suffix}.json").write_text(json.dumps(meta, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
