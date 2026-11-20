"""Shared enums and small immutable records used across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class TestType(StrEnum):
    COVER_UNCOVER = "Cover-Uncover"
    ALTERNATE_COVER = "Alternate-Cover"


class Eye(StrEnum):
    RIGHT = "right"
    LEFT = "left"


class DirectionClass(StrEnum):
    ESO_TROPIA = "eso_tropia"
    EXO_TROPIA = "exo_tropia"
    NO_MANIFEST = "no_manifest_horizontal_deviation"


class ClinicalClass(StrEnum):
    TROPIA = "tropia"
    INTERMITTENT_TROPIA = "intermittent_tropia"
    PHORIA = "phoria"
    NORMAL = "normal"


class ObservableState(StrEnum):
    YES = "yes"
    NO = "no"
    INDETERMINATE = "indeterminate"


@dataclass(frozen=True)
class VideoInput:
    path: Path
    test_type: TestType
    distance: str
    correction: str


@dataclass(frozen=True)
class PredictionWarning:
    code: str
    message: str


@dataclass
class ScreeningPrediction:
    video: str
    test_type: str
    direction: str
    direction_probability: dict[str, float]
    clinical_class: str
    clinical_probability: dict[str, float]
    observable_in_clip: str
    horizontal_pd: float
    horizontal_pd_interval: tuple[float, float]
    quality_score: float
    warnings: list[PredictionWarning] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result = {
            "video": self.video,
            "test_type": self.test_type,
            "direction": self.direction,
            "direction_probability": self.direction_probability,
            "clinical_class": self.clinical_class,
            "clinical_probability": self.clinical_probability,
            "observable_in_clip": self.observable_in_clip,
            "horizontal_pd": self.horizontal_pd,
            "horizontal_pd_interval": list(self.horizontal_pd_interval),
            "quality_score": self.quality_score,
            "warnings": [
                {"code": warning.code, "message": warning.message}
                for warning in self.warnings
            ],
            "metadata": self.metadata,
        }
        return result
