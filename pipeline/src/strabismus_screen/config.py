"""Configuration loading with conservative, auditable defaults."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class EventConfig:
    sample_stride: int = 1
    median_window_frames: int = 5
    cover_on_threshold: float = 0.58
    cover_off_threshold: float = 0.40
    minimum_cover_frames: int = 4
    transition_guard_frames: int = 2
    max_landmark_hold_frames: int = 45


@dataclass
class MotionConfig:
    pre_window_frames: int = 6
    early_post_frames: int = 3
    plateau_start_frames: int = 5
    plateau_end_frames: int = 15
    minimum_valid_cycles: int = 1
    maximum_head_pose_change: float = 0.12


@dataclass
class CalibrationConfig:
    enabled: bool = False
    mode: str = "dataset_supervised"
    target_angles_degrees: list[float] = field(default_factory=list)
    camera_light_available: bool = False


@dataclass
class PipelineConfig:
    random_seed: int = 20260824
    face_landmarker_model: str = "models/face_landmarker.task"
    maximum_faces: int = 2
    event: EventConfig = field(default_factory=EventConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _update_dataclass(instance: Any, values: dict[str, Any]) -> Any:
    for key, value in values.items():
        if not hasattr(instance, key):
            raise ValueError(f"Unknown configuration key: {key}")
        current = getattr(instance, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _update_dataclass(current, value)
        else:
            setattr(instance, key, value)
    return instance


def load_config(path: str | Path | None = None) -> PipelineConfig:
    config = PipelineConfig()
    if path is None:
        return config
    with Path(path).open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle) or {}
    if not isinstance(values, dict):
        raise ValueError("Pipeline configuration must be a YAML mapping")
    return _update_dataclass(config, values)
