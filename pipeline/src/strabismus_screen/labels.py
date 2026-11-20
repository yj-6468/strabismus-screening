"""Clinical-label normalization for horizontal strabismus screening.

The source tables use ophthalmic shorthand.  This module is deliberately the
only place where that shorthand is converted into model targets.  In
particular, the ``Control`` column is not accepted by any function here and is
therefore unable to leak into a target accidentally.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import re
from typing import Any


class LabelError(ValueError):
    """Base class for invalid or unsupported clinical labels."""


class UnknownHorizontalTypeError(LabelError):
    """Raised when a horizontal diagnosis is outside the documented schema."""


class HorizontalDirection(str, Enum):
    ESO = "eso"
    EXO = "exo"
    NORMAL = "normal"
    MIXED = "mixed"


class Job1ManifestLabel(str, Enum):
    """Cover-uncover targets approved for Job 1."""

    ESO_TROPIA = "eso_tropia"
    EXO_TROPIA = "exo_tropia"
    NO_MANIFEST_HORIZONTAL_DEVIATION = "no_manifest_horizontal_deviation"


class ClinicalManifestClass(str, Enum):
    """Clip-level clinical classes used by Job 3."""

    TROPIA = "tropia"
    INTERMITTENT_TROPIA = "intermittent_tropia"
    PHORIA = "phoria"
    NORMAL = "normal"


class MixedFlickPolicy(str, Enum):
    """How mixed-direction flick phorias are handled for Job 3."""

    PHORIA = "phoria"
    EXCLUDE = "exclude"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class PrismDiopterInterval:
    """An interval-valued horizontal deviation magnitude in prism diopters."""

    lower: float
    upper: float
    raw: str

    def __post_init__(self) -> None:
        if not (math.isfinite(self.lower) and math.isfinite(self.upper)):
            raise LabelError("prism-diopter bounds must be finite")
        if self.lower < 0 or self.upper < 0:
            raise LabelError("prism-diopter bounds cannot be negative")
        if self.lower > self.upper:
            raise LabelError("prism-diopter lower bound exceeds upper bound")

    @property
    def is_exact(self) -> bool:
        return self.lower == self.upper

    @property
    def midpoint(self) -> float:
        return (self.lower + self.upper) / 2.0


_CANONICAL_TYPES = {
    "et": "ET",
    "xt": "XT",
    "e(t)": "E(T)",
    "x(t)": "X(T)",
    "e": "E",
    "e'": "E'",
    "x": "X",
    "normal": "normal",
    "flick e' to flick x'": "Flick E' to Flick X'",
    # This spelling (including the trailing 2) occurs in two source rows.
    "flick x' to e'2": "flick X' to E'2",
}

_MIXED_TYPES = {"Flick E' to Flick X'", "flick X' to E'2"}
_ESO_TYPES = {"ET", "E(T)", "E", "E'"}
_EXO_TYPES = {"XT", "X(T)", "X"}
_TROPIA_TYPES = {"ET", "XT"}
_INTERMITTENT_TYPES = {"E(T)", "X(T)"}
_PHORIA_TYPES = {"E", "E'", "X"}


def normalize_horizontal_type(value: Any) -> str:
    """Return the canonical source spelling for a horizontal diagnosis."""

    if value is None:
        raise UnknownHorizontalTypeError("horizontal type is missing")
    text = " ".join(str(value).strip().split())
    canonical = _CANONICAL_TYPES.get(text.casefold())
    if canonical is None:
        raise UnknownHorizontalTypeError(f"unsupported horizontal type: {value!r}")
    return canonical


def horizontal_direction(value: Any) -> HorizontalDirection:
    """Map the clinical shorthand to direction without implying manifestation."""

    horizontal_type = normalize_horizontal_type(value)
    if horizontal_type in _ESO_TYPES:
        return HorizontalDirection.ESO
    if horizontal_type in _EXO_TYPES:
        return HorizontalDirection.EXO
    if horizontal_type == "normal":
        return HorizontalDirection.NORMAL
    return HorizontalDirection.MIXED


# An intermittent tropia with chart control score <= this value is well controlled and counts
# as orthotropic for Job 1 (clinician's rule; control scale: constant 5, intermittent 1-4, phoria 0).
WELL_CONTROLLED_MAXIMUM = 2.0


def job1_manifest_label(
    horizontal_type: Any,
    test_type: Any,
    control: Any = None,
    *,
    well_controlled_maximum: float = WELL_CONTROLLED_MAXIMUM,
) -> Job1ManifestLabel | None:
    """Return the approved Job 1 target, or ``None`` when the clip is excluded.

    Only cover-uncover clips are eligible.  Job 1 asks whether a *tropic*
    component is on view, so the class ``no_manifest_horizontal_deviation``
    means orthotropic.  Constant tropias keep their eso/exo direction.  A
    patient with only a phoria has no tropic component, so phorias (including
    the mixed-direction flick phorias, which need no direction for this class)
    and normal diagnoses are orthotropic.  An intermittent tropia keeps its
    direction unless ``control`` shows it is well controlled.

    ``control`` is the chart control score.  It defines the target only and must
    never be a model feature.  ``None`` leaves an intermittent tropia tropic.
    """

    if str(test_type).strip() != "Cover-Uncover":
        return None
    diagnosis = normalize_horizontal_type(horizontal_type)
    if diagnosis in _INTERMITTENT_TYPES and _is_well_controlled(
        control, well_controlled_maximum
    ):
        return Job1ManifestLabel.NO_MANIFEST_HORIZONTAL_DEVIATION
    if diagnosis in {"ET", "E(T)"}:
        return Job1ManifestLabel.ESO_TROPIA
    if diagnosis in {"XT", "X(T)"}:
        return Job1ManifestLabel.EXO_TROPIA
    if diagnosis in _PHORIA_TYPES or diagnosis in _MIXED_TYPES or diagnosis == "normal":
        return Job1ManifestLabel.NO_MANIFEST_HORIZONTAL_DEVIATION
    return None


def _is_well_controlled(control: Any, maximum: float) -> bool:
    if _is_missing(control):
        return False
    try:
        score = float(str(control).strip())
    except ValueError as exc:
        raise LabelError(f"invalid control score: {control!r}") from exc
    if not math.isfinite(score) or score < 0:
        raise LabelError(f"invalid control score: {control!r}")
    return score <= maximum


def observable_in_clip(horizontal_type: Any, test_type: Any) -> bool | None:
    """Encode what the clinical label alone establishes about observability.

    ``True`` means a constant tropia is expected to be observable, ``False``
    means the label explicitly denotes normal alignment or a latent phoria,
    and ``None`` means the table cannot establish clip-level visibility (all
    alternate-cover clips, intermittent tropias, and mixed-direction labels).
    """

    if str(test_type).strip() != "Cover-Uncover":
        return None
    diagnosis = normalize_horizontal_type(horizontal_type)
    if diagnosis in _TROPIA_TYPES:
        return True
    if diagnosis in _PHORIA_TYPES or diagnosis == "normal":
        return False
    return None


def job3_clinical_class(
    horizontal_type: Any,
    *,
    mixed_flick_policy: MixedFlickPolicy | str = MixedFlickPolicy.PHORIA,
) -> ClinicalManifestClass | None:
    """Map a source diagnosis to the four Job 3 clinical classes."""

    diagnosis = normalize_horizontal_type(horizontal_type)
    try:
        policy = MixedFlickPolicy(mixed_flick_policy)
    except ValueError as exc:
        raise LabelError(
            f"unsupported mixed-flick policy: {mixed_flick_policy!r}"
        ) from exc

    if diagnosis in _TROPIA_TYPES:
        return ClinicalManifestClass.TROPIA
    if diagnosis in _INTERMITTENT_TYPES:
        return ClinicalManifestClass.INTERMITTENT_TROPIA
    if diagnosis in _PHORIA_TYPES:
        return ClinicalManifestClass.PHORIA
    if diagnosis == "normal":
        return ClinicalManifestClass.NORMAL
    if policy is MixedFlickPolicy.PHORIA:
        return ClinicalManifestClass.PHORIA
    if policy is MixedFlickPolicy.EXCLUDE:
        return None
    raise LabelError(
        f"mixed-direction horizontal type {diagnosis!r} is forbidden by policy"
    )


_PD_PATTERN = re.compile(
    r"^(?P<lower>\d+(?:\.\d+)?)"
    r"(?:\s*[-\N{EN DASH}\N{EM DASH}]\s*(?P<upper>\d+(?:\.\d+)?))?$"
)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return str(value).strip() == ""


def parse_horizontal_pd(
    value: Any,
    horizontal_type: Any,
) -> PrismDiopterInterval | None:
    """Parse a scalar/range PD label; normal clips are explicitly ``[0, 0]``.

    The source stores unsigned magnitudes.  Direction remains a separate label
    and must not be inferred from the sign of this interval.
    """

    diagnosis = normalize_horizontal_type(horizontal_type)
    if diagnosis == "normal":
        if not _is_missing(value):
            match = _PD_PATTERN.fullmatch(str(value).strip())
            if match is None or any(
                float(group) != 0 for group in match.groups() if group
            ):
                raise LabelError(
                    "a normal diagnosis cannot have a non-zero horizontal PD"
                )
        return PrismDiopterInterval(0.0, 0.0, "0")

    if _is_missing(value):
        # The only such source records are the documented mixed flick phorias.
        return None
    raw = str(value).strip()
    match = _PD_PATTERN.fullmatch(raw)
    if match is None:
        raise LabelError(f"invalid horizontal PD value: {value!r}")
    lower = float(match.group("lower"))
    upper_text = match.group("upper")
    upper = lower if upper_text is None else float(upper_text)
    return PrismDiopterInterval(lower, upper, raw)


__all__ = [
    "WELL_CONTROLLED_MAXIMUM",
    "ClinicalManifestClass",
    "HorizontalDirection",
    "Job1ManifestLabel",
    "LabelError",
    "MixedFlickPolicy",
    "PrismDiopterInterval",
    "UnknownHorizontalTypeError",
    "horizontal_direction",
    "job1_manifest_label",
    "job3_clinical_class",
    "normalize_horizontal_type",
    "observable_in_clip",
    "parse_horizontal_pd",
]
