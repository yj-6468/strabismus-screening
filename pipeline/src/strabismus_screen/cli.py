"""Command-line interface for the horizontal strabismus research pipeline."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np

from .config import load_config
from .data import load_dataset
from .evaluation_pipeline import (
    evaluate_pipeline,
    evaluation_console_summary,
    write_evaluation_report,
)
from .pipeline import (
    PIPELINE_BUNDLE_VERSION,
    PipelineModelBundle,
    analyze_raw_video,
    dataset_audit,
    extract_observation_cache,
    train_pipeline_bundle,
)
from .native_bundle import HeadSpec, NativeBundle, predict_native, train_native_bundle


# The final (v14) model: trained on the 774 clips of ~/Desktop/CHLA Data with the merged-identity cache.
NATIVE_CACHE = "artifacts/observations_v14_merged"
NATIVE_MODEL = "artifacts/final_model_v14.joblib"


def _native_json_value(value: Any) -> Any:
    """Represent unavailable native measurements as JSON null, not NaN."""
    if isinstance(value, dict):
        return {key: _native_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_native_json_value(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    return None if isinstance(value, float) and not math.isfinite(value) else value


def _print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), flush=True)


def _progress(value: dict[str, Any]) -> None:
    print(json.dumps(value, sort_keys=True, allow_nan=False), flush=True)


def _load_dataset(dataset_root: Path | str, args: argparse.Namespace):
    if args.expected_videos is None:
        return load_dataset(dataset_root)
    return load_dataset(dataset_root, expected_video_count=args.expected_videos)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="strabismus-screen",
        description="Research pipeline for horizontal cover-test videos",
    )
    parser.add_argument("--dataset-root", default=".", help="Root containing Older data/New data")
    parser.add_argument("--expected-videos", type=int, default=None,
                        help="clip count the dataset root must hold (774 for CHLA Data); default: the release's own count")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("audit", help="Validate all source joins and print target counts")

    extract = subparsers.add_parser(
        "extract", help="Uniformly extract raw-video observations into a local cache"
    )
    extract.add_argument("--model", default="models/face_landmarker.task")
    extract.add_argument("--cache-dir", default="artifacts/observations_native_fulltexture")
    extract.add_argument("--stride", type=int, default=1)
    extract.add_argument("--maximum-faces", type=int, default=2)
    extract.add_argument("--start", type=int, default=0)
    extract.add_argument("--limit", type=int)
    extract.add_argument("--overwrite", action="store_true")
    extract.add_argument("--quiet", action="store_true")

    train = subparsers.add_parser(
        "train", help="Fit the cover detector and three video-level task heads"
    )
    train.add_argument("--cache-dir", default="artifacts/observations")
    train.add_argument("--output", default="artifacts/strabismus_pipeline.joblib")
    train.add_argument("--feature-table", default="artifacts/video_features.csv")
    train.add_argument("--config", default="configs/default.yaml")

    train_native = subparsers.add_parser(
        "train-native", help="Fit the native-pixel, stride-1 cover detector and three task heads"
    )
    train_native.add_argument("--cache-dir", default=NATIVE_CACHE)
    train_native.add_argument("--output", default=NATIVE_MODEL)
    train_native.add_argument("--config", default="configs/stride1.yaml")
    train_native.add_argument("--seed", type=int, default=20260901)
    defaults = HeadSpec()
    for job in (1, 2, 3):
        train_native.add_argument(f"--job{job}-model", default=getattr(defaults, f"job{job}_model"))
        train_native.add_argument(f"--job{job}-features", default=getattr(defaults, f"job{job}_features"))

    evaluate = subparsers.add_parser(
        "evaluate",
        help=(
            "Nested raw-video evaluation with the cover detector refit in every "
            "outer fold"
        ),
    )
    evaluate.add_argument("--cache-dir", default="artifacts/observations")
    evaluate.add_argument("--output", default="reports/evaluation.json")
    evaluate.add_argument("--config", default="configs/default.yaml")
    evaluate.add_argument("--folds", type=int, default=5)
    evaluate.add_argument("--target-deviation-sensitivity", type=float, default=0.95)
    evaluate.add_argument(
        "--mode",
        action="append",
        choices=("video", "patient", "cohort"),
        help=(
            "Evaluation mode; repeat for multiple modes. Defaults to all three. "
            "Separate --output paths allow modes to run in parallel."
        ),
    )
    evaluate.add_argument("--skip-patient-grouped", action="store_true")
    evaluate.add_argument("--skip-cohort-holdout", action="store_true")
    evaluate.add_argument("--quiet", action="store_true")

    predict = subparsers.add_parser(
        "predict", help="Run direction and deviation from one raw video without annotations"
    )
    predict.add_argument("video")
    predict.add_argument("--bundle", default=NATIVE_MODEL,
                         help="Trained native bundle (default); an explicit legacy pipeline bundle is also supported")
    predict.add_argument("--model", default="models/face_landmarker.task")
    predict.add_argument(
        "--test-type", choices=("Cover-Uncover", "Alternate-Cover"), required=True
    )
    predict.add_argument("--distance", choices=("near", "distance"), required=True)
    predict.add_argument(
        "--correction",
        choices=("with_correction", "without_correction", "cc", "sc"),
        required=True,
    )
    predict.add_argument(
        "--stride",
        type=int,
        help="Advanced override; must match the stride stored in the trained bundle",
    )
    predict.add_argument(
        "--maximum-faces",
        type=int,
        help="Advanced override; must match the value stored in the trained bundle",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset_root = Path(args.dataset_root).resolve()
    if args.command == "audit":
        dataset = _load_dataset(dataset_root, args)
        _print_json(dataset_audit(dataset))
        return 0
    if args.command == "extract":
        dataset = _load_dataset(dataset_root, args)
        model = Path(args.model)
        if not model.is_absolute():
            model = Path.cwd() / model
        cache_directory = Path(args.cache_dir)
        if not cache_directory.is_absolute():
            cache_directory = Path.cwd() / cache_directory
        summary = extract_observation_cache(
            dataset,
            model_path=model,
            cache_directory=cache_directory,
            stride=args.stride,
            maximum_faces=args.maximum_faces,
            start=args.start,
            limit=args.limit,
            overwrite=args.overwrite,
            progress=None if args.quiet else _progress,
        )
        _print_json(summary)
        return 1 if summary["failure_count"] else 0
    if args.command == "train":
        dataset = _load_dataset(dataset_root, args)
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = Path.cwd() / config_path
        config = load_config(config_path)
        metadata = train_pipeline_bundle(
            dataset,
            cache_directory=args.cache_dir,
            output_path=args.output,
            config=config,
            feature_table_path=args.feature_table,
        )
        _print_json(metadata)
        return 0
    if args.command == "train-native":
        config = load_config(Path(args.config).resolve())
        if config.event.sample_stride != 1:
            raise ValueError("train-native requires sample_stride=1; use configs/stride1.yaml")
        spec = HeadSpec(**{
            f"job{job}_{option}": getattr(args, f"job{job}_{option}")
            for job in (1, 2, 3) for option in ("model", "features")
        })
        bundle = train_native_bundle(
            _load_dataset(dataset_root, args),
            cache_directory=args.cache_dir,
            output_path=args.output,
            config=config,
            spec=spec,
            seed=args.seed,
        )
        _print_json({**bundle.training_metadata, "output": str(Path(args.output).resolve())})
        return 0
    if args.command == "evaluate":
        dataset = _load_dataset(dataset_root, args)
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = Path.cwd() / config_path
        config = load_config(config_path)
        selected_modes = args.mode
        if selected_modes is not None:
            selected_modes = [
                mode
                for mode in selected_modes
                if not (mode == "patient" and args.skip_patient_grouped)
                and not (mode == "cohort" and args.skip_cohort_holdout)
            ]
            if not selected_modes:
                raise ValueError("The selected evaluation modes were all skipped")
        report = evaluate_pipeline(
            dataset,
            cache_directory=args.cache_dir,
            config=config,
            n_splits=args.folds,
            include_patient_grouped=not args.skip_patient_grouped,
            include_cohort_holdout=not args.skip_cohort_holdout,
            modes=selected_modes,
            target_deviation_sensitivity=args.target_deviation_sensitivity,
            progress=None if args.quiet else _progress,
        )
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = Path.cwd() / output_path
        write_evaluation_report(report, output_path)
        summary = evaluation_console_summary(report)
        summary["report_path"] = str(output_path)
        _print_json(summary)
        return 0
    if args.command == "predict":
        bundle = joblib.load(args.bundle)
        if isinstance(bundle, NativeBundle):
            trained_stride, trained_faces = bundle.config.event.sample_stride, bundle.config.maximum_faces
        elif isinstance(bundle, PipelineModelBundle):
            if bundle.bundle_version not in (3, PIPELINE_BUNDLE_VERSION):
                raise ValueError(f"Unsupported pipeline bundle version: {bundle.bundle_version}")
            trained_stride, trained_faces = bundle.extraction_stride, bundle.maximum_faces
        else:
            raise TypeError(f"Expected NativeBundle or PipelineModelBundle, found {type(bundle).__name__}")
        if args.stride is not None and args.stride != trained_stride:
            raise ValueError(f"Bundle requires extraction stride {trained_stride}, got {args.stride}")
        if args.maximum_faces is not None and args.maximum_faces != trained_faces:
            raise ValueError(f"Bundle requires maximum_faces={trained_faces}, got {args.maximum_faces}")
        inputs = dict(video_path=args.video, test_type=args.test_type, distance=args.distance,
                      correction=args.correction, face_landmarker_model=args.model)
        if isinstance(bundle, NativeBundle):
            output = _native_json_value(predict_native(bundle, **inputs))
        else:
            output = analyze_raw_video(bundle, **inputs, maximum_faces=args.maximum_faces, stride=args.stride)
        _print_json(output)
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
