"""Prism-diopter geometry, interval targets, and measurement metrics.

The internal signed convention is positive for esodeviation, negative for
exodeviation, and zero for normal alignment.  Prism diopters are converted
from ocular rotation with the exact tangent definition::

    prism_diopters = 100 * tan(angle_radians)

The functions accept scalars or NumPy-compatible arrays.  Geometry helpers
preserve scalar inputs as ``float`` values; array inputs return arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from statistics import NormalDist
from typing import Literal, Sequence

import numpy as np


Reduction = Literal["none", "mean", "sum"]
IntervalLossKind = Literal["absolute", "squared", "huber"]


def _scalarize(value: np.ndarray, original: object) -> float | np.ndarray:
    """Return a Python float when ``original`` was scalar-like."""

    if np.ndim(original) == 0:
        return float(np.asarray(value))
    return np.asarray(value)


def _require_finite_or_nan(values: np.ndarray, name: str) -> None:
    if np.any(np.isinf(values)):
        raise ValueError(f"{name} must not contain infinity")


def degrees_to_prism_diopters(
    angle_degrees: float | Sequence[float] | np.ndarray,
) -> float | np.ndarray:
    """Convert a signed ocular deviation angle in degrees to prism diopters.

    Positive angles produce positive (eso) prism values and negative angles
    produce negative (exo) values.  Angles must lie strictly between -90 and
    +90 degrees.  NaNs are preserved so callers can mask missing observations.
    """

    angle = np.asarray(angle_degrees, dtype=float)
    _require_finite_or_nan(angle, "angle_degrees")
    finite = np.isfinite(angle)
    if np.any(np.abs(angle[finite]) >= 90.0):
        raise ValueError("angle_degrees must lie strictly between -90 and 90")
    result = 100.0 * np.tan(np.deg2rad(angle))
    return _scalarize(result, angle_degrees)


def radians_to_prism_diopters(
    angle_radians: float | Sequence[float] | np.ndarray,
) -> float | np.ndarray:
    """Convert signed ocular deviation in radians to exact prism diopters."""

    angle = np.asarray(angle_radians, dtype=float)
    _require_finite_or_nan(angle, "angle_radians")
    finite = np.isfinite(angle)
    if np.any(np.abs(angle[finite]) >= np.pi / 2.0):
        raise ValueError("angle_radians must lie strictly between -pi/2 and pi/2")
    result = 100.0 * np.tan(angle)
    return _scalarize(result, angle_radians)


def prism_diopters_to_degrees(
    prism_diopters: float | Sequence[float] | np.ndarray,
) -> float | np.ndarray:
    """Convert signed prism diopters to ocular deviation angle in degrees."""

    prism = np.asarray(prism_diopters, dtype=float)
    _require_finite_or_nan(prism, "prism_diopters")
    result = np.rad2deg(np.arctan(prism / 100.0))
    return _scalarize(result, prism_diopters)


def prism_diopters_to_radians(
    prism_diopters: float | Sequence[float] | np.ndarray,
) -> float | np.ndarray:
    """Convert signed prism diopters to ocular deviation angle in radians."""

    prism = np.asarray(prism_diopters, dtype=float)
    _require_finite_or_nan(prism, "prism_diopters")
    result = np.arctan(prism / 100.0)
    return _scalarize(result, prism_diopters)


# Explicit aliases make the physical quantity clear at call sites.
angle_degrees_to_prism_diopters = degrees_to_prism_diopters
prism_diopters_to_angle_degrees = prism_diopters_to_degrees


_ESO_DIRECTIONS = {
    "e",
    "et",
    "e(t)",
    "eso",
    "esodeviation",
    "esophoria",
    "esotropia",
}
_EXO_DIRECTIONS = {
    "x",
    "xt",
    "x(t)",
    "exo",
    "exodeviation",
    "exophoria",
    "exotropia",
}
_NORMAL_DIRECTIONS = {"0", "normal", "ortho", "orthophoria", "orthotropia"}


def deviation_sign(direction: str) -> int:
    """Return +1 for eso, -1 for exo, or 0 for normal alignment.

    Common clinical abbreviations used in the CHLA labels are accepted.
    Mixed or unquantified labels such as ``"flick E to X"`` intentionally
    raise rather than being silently assigned a direction.
    """

    token = str(direction).strip().lower().replace("'", "")
    token = re.sub(r"[\s_-]+", "", token)
    if token in _ESO_DIRECTIONS:
        return 1
    if token in _EXO_DIRECTIONS:
        return -1
    if token in _NORMAL_DIRECTIONS:
        return 0
    raise ValueError(f"unsupported or ambiguous horizontal direction: {direction!r}")


def signed_prism_diopters(magnitude: float, direction: str) -> float:
    """Apply the internal eso-positive/exo-negative sign convention."""

    value = float(magnitude)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("magnitude must be a finite, non-negative number")
    sign = deviation_sign(direction)
    if sign == 0 and value != 0.0:
        raise ValueError("normal alignment must have zero prism magnitude")
    return float(sign * value)


@dataclass(frozen=True, slots=True)
class PrismInterval:
    """Closed signed interval for a prism-diopter target."""

    lower: float
    upper: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.lower) or not math.isfinite(self.upper):
            raise ValueError("interval bounds must be finite")
        if self.lower > self.upper:
            raise ValueError("lower interval bound must not exceed upper bound")

    @property
    def midpoint(self) -> float:
        return (self.lower + self.upper) / 2.0

    @property
    def width(self) -> float:
        return self.upper - self.lower

    def project(self, prediction: float) -> float:
        """Return the closest target value in this interval."""

        return float(np.clip(float(prediction), self.lower, self.upper))

    @classmethod
    def from_magnitude(
        cls, lower_magnitude: float, upper_magnitude: float, direction: str
    ) -> "PrismInterval":
        """Construct a correctly ordered signed interval from magnitudes."""

        lower = float(lower_magnitude)
        upper = float(upper_magnitude)
        if not math.isfinite(lower) or not math.isfinite(upper):
            raise ValueError("magnitude bounds must be finite")
        if lower < 0.0 or upper < lower:
            raise ValueError("magnitude bounds must satisfy 0 <= lower <= upper")
        sign = deviation_sign(direction)
        if sign == 0:
            if upper != 0.0:
                raise ValueError("normal alignment must have a zero interval")
            return cls(0.0, 0.0)
        if sign > 0:
            return cls(lower, upper)
        return cls(-upper, -lower)


_MAGNITUDE_INTERVAL_RE = re.compile(
    r"^\s*(\d+(?:\.\d+)?)\s*(?:[-\u2013\u2014]\s*(\d+(?:\.\d+)?))?\s*(?:\u0394|pd)?\s*$",
    re.IGNORECASE,
)


def parse_prism_interval(
    value: str | float | int | None,
    direction: str,
) -> PrismInterval:
    """Parse an exact or range-valued magnitude into a signed interval.

    Empty values are accepted only when ``direction`` is normal, in which
    case ``[0, 0]`` is returned.  This prevents blank, unquantified labels from
    being accidentally treated as normal observations.
    """

    sign = deviation_sign(direction)
    if value is None or (isinstance(value, str) and not value.strip()):
        if sign == 0:
            return PrismInterval(0.0, 0.0)
        raise ValueError("a non-normal direction requires a prism magnitude")

    if isinstance(value, (int, float, np.integer, np.floating)):
        magnitude = float(value)
        return PrismInterval.from_magnitude(magnitude, magnitude, direction)

    match = _MAGNITUDE_INTERVAL_RE.fullmatch(str(value))
    if match is None:
        raise ValueError(f"invalid prism magnitude or interval: {value!r}")
    lower = float(match.group(1))
    upper = float(match.group(2)) if match.group(2) is not None else lower
    return PrismInterval.from_magnitude(lower, upper, direction)


def _broadcast_interval_inputs(
    prediction: float | Sequence[float] | np.ndarray,
    lower: float | Sequence[float] | np.ndarray,
    upper: float | Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred, lo, hi = np.broadcast_arrays(
        np.asarray(prediction, dtype=float),
        np.asarray(lower, dtype=float),
        np.asarray(upper, dtype=float),
    )
    _require_finite_or_nan(pred, "prediction")
    _require_finite_or_nan(lo, "lower")
    _require_finite_or_nan(hi, "upper")
    valid_bounds = np.isfinite(lo) & np.isfinite(hi)
    if np.any(lo[valid_bounds] > hi[valid_bounds]):
        raise ValueError("lower interval bounds must not exceed upper bounds")
    return pred, lo, hi


def interval_midpoint(
    lower: float | Sequence[float] | np.ndarray,
    upper: float | Sequence[float] | np.ndarray,
) -> float | np.ndarray:
    """Return midpoint point targets as an explicit fallback for intervals."""

    lo, hi = np.broadcast_arrays(np.asarray(lower, dtype=float), np.asarray(upper, dtype=float))
    _require_finite_or_nan(lo, "lower")
    _require_finite_or_nan(hi, "upper")
    valid = np.isfinite(lo) & np.isfinite(hi)
    if np.any(lo[valid] > hi[valid]):
        raise ValueError("lower interval bounds must not exceed upper bounds")
    result = (lo + hi) / 2.0
    if np.ndim(lower) == 0 and np.ndim(upper) == 0:
        return float(result)
    return result


midpoint_fallback = interval_midpoint


def project_to_interval(
    prediction: float | Sequence[float] | np.ndarray,
    lower: float | Sequence[float] | np.ndarray,
    upper: float | Sequence[float] | np.ndarray,
) -> float | np.ndarray:
    """Project predictions to the closest values allowed by target intervals."""

    pred, lo, hi = _broadcast_interval_inputs(prediction, lower, upper)
    result = np.minimum(np.maximum(pred, lo), hi)
    return float(result) if result.ndim == 0 else result


def interval_residual(
    prediction: float | Sequence[float] | np.ndarray,
    lower: float | Sequence[float] | np.ndarray,
    upper: float | Sequence[float] | np.ndarray,
) -> float | np.ndarray:
    """Signed distance to an interval; zero denotes an acceptable prediction."""

    pred, lo, hi = _broadcast_interval_inputs(prediction, lower, upper)
    residual = pred - np.minimum(np.maximum(pred, lo), hi)
    return float(residual) if residual.ndim == 0 else residual


def _reduce(values: np.ndarray, reduction: Reduction) -> float | np.ndarray:
    if reduction == "none":
        return values
    if reduction == "mean":
        return float(np.nanmean(values))
    if reduction == "sum":
        return float(np.nansum(values))
    raise ValueError(f"unsupported reduction: {reduction!r}")


def interval_loss(
    prediction: float | Sequence[float] | np.ndarray,
    lower: float | Sequence[float] | np.ndarray,
    upper: float | Sequence[float] | np.ndarray,
    *,
    kind: IntervalLossKind = "squared",
    huber_delta: float = 1.0,
    reduction: Reduction = "mean",
) -> float | np.ndarray:
    """Minimum loss between each prediction and any value in its target interval.

    The loss is zero for predictions inside the closed interval.  ``kind`` may
    be absolute distance, squared distance, or Huber loss.
    """

    residual = np.asarray(interval_residual(prediction, lower, upper), dtype=float)
    absolute = np.abs(residual)
    if kind == "absolute":
        values = absolute
    elif kind == "squared":
        values = residual**2
    elif kind == "huber":
        if not math.isfinite(huber_delta) or huber_delta <= 0.0:
            raise ValueError("huber_delta must be finite and greater than zero")
        values = np.where(
            absolute <= huber_delta,
            0.5 * residual**2,
            huber_delta * (absolute - 0.5 * huber_delta),
        )
    else:
        raise ValueError(f"unsupported interval loss kind: {kind!r}")
    reduced = _reduce(np.asarray(values), reduction)
    if reduction == "none" and np.ndim(reduced) == 0:
        return float(np.asarray(reduced))
    return reduced


def interval_aware_mae(
    prediction: float | Sequence[float] | np.ndarray,
    lower: float | Sequence[float] | np.ndarray,
    upper: float | Sequence[float] | np.ndarray,
    *,
    reduction: Reduction = "mean",
) -> float | np.ndarray:
    """Mean absolute distance to the nearest acceptable target value."""

    return interval_loss(
        prediction, lower, upper, kind="absolute", reduction=reduction
    )


@dataclass(frozen=True, slots=True)
class BlandAltmanStats:
    """Summary of paired prediction-minus-reference differences."""

    n: int
    bias: float
    sd_difference: float
    loa_lower: float
    loa_upper: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "n": self.n,
            "bias": self.bias,
            "sd_difference": self.sd_difference,
            "loa_lower": self.loa_lower,
            "loa_upper": self.loa_upper,
        }


def _paired_values(
    prediction: float | Sequence[float] | np.ndarray,
    reference: float | Sequence[float] | np.ndarray,
    sample_indices: Sequence[int] | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    pred, ref = np.broadcast_arrays(
        np.asarray(prediction, dtype=float), np.asarray(reference, dtype=float)
    )
    pred = pred.ravel()
    ref = ref.ravel()
    if sample_indices is not None:
        indices = np.asarray(sample_indices, dtype=int).ravel()
        pred = pred[indices]
        ref = ref[indices]
    finite = np.isfinite(pred) & np.isfinite(ref)
    return pred[finite], ref[finite]


def bland_altman_differences(
    prediction: float | Sequence[float] | np.ndarray,
    reference: float | Sequence[float] | np.ndarray,
    *,
    sample_indices: Sequence[int] | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return paired means and prediction-minus-reference differences.

    ``sample_indices`` may contain repeated indices, so bootstrap callers can
    resample at the video level without reimplementing filtering or pairing.
    """

    pred, ref = _paired_values(prediction, reference, sample_indices)
    return (pred + ref) / 2.0, pred - ref


def bland_altman(
    prediction: float | Sequence[float] | np.ndarray,
    reference: float | Sequence[float] | np.ndarray,
    *,
    sample_indices: Sequence[int] | np.ndarray | None = None,
    ddof: int = 1,
    loa_z: float = 1.96,
) -> BlandAltmanStats:
    """Compute deterministic, bootstrap-compatible Bland-Altman statistics."""

    if ddof < 0:
        raise ValueError("ddof must be non-negative")
    if not math.isfinite(loa_z) or loa_z <= 0.0:
        raise ValueError("loa_z must be finite and greater than zero")
    _, differences = bland_altman_differences(
        prediction, reference, sample_indices=sample_indices
    )
    if differences.size <= ddof:
        raise ValueError(f"at least {ddof + 1} finite pairs are required")
    bias = float(np.mean(differences))
    sd = float(np.std(differences, ddof=ddof))
    return BlandAltmanStats(
        n=int(differences.size),
        bias=bias,
        sd_difference=sd,
        loa_lower=bias - loa_z * sd,
        loa_upper=bias + loa_z * sd,
    )


def within_tolerance_rate(
    prediction: float | Sequence[float] | np.ndarray,
    reference: float | Sequence[float] | np.ndarray,
    tolerance: float,
) -> float:
    """Fraction of finite point predictions within an absolute PD tolerance."""

    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and non-negative")
    pred, ref = _paired_values(prediction, reference)
    if pred.size == 0:
        raise ValueError("at least one finite prediction/reference pair is required")
    return float(np.mean(np.abs(pred - ref) <= tolerance))


def interval_within_tolerance_rate(
    prediction: float | Sequence[float] | np.ndarray,
    lower: float | Sequence[float] | np.ndarray,
    upper: float | Sequence[float] | np.ndarray,
    tolerance: float,
) -> float:
    """Fraction within a tolerance of the nearest point in each target interval."""

    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and non-negative")
    residual = np.asarray(interval_residual(prediction, lower, upper), dtype=float).ravel()
    residual = residual[np.isfinite(residual)]
    if residual.size == 0:
        raise ValueError("at least one finite interval residual is required")
    return float(np.mean(np.abs(residual) <= tolerance))


def within_5_10(
    prediction: float | Sequence[float] | np.ndarray,
    reference: float | Sequence[float] | np.ndarray | None = None,
    *,
    lower: float | Sequence[float] | np.ndarray | None = None,
    upper: float | Sequence[float] | np.ndarray | None = None,
) -> dict[str, float]:
    """Return standard proportions within +/-5 PD and +/-10 PD.

    Supply either point ``reference`` values or both interval bounds.
    """

    using_point = reference is not None
    using_interval = lower is not None or upper is not None
    if using_point == using_interval:
        raise ValueError("supply either reference or both lower and upper bounds")
    if using_point:
        rate = lambda tolerance: within_tolerance_rate(  # noqa: E731
            prediction, reference, tolerance
        )
    else:
        if lower is None or upper is None:
            raise ValueError("both lower and upper interval bounds are required")
        rate = lambda tolerance: interval_within_tolerance_rate(  # noqa: E731
            prediction, lower, upper, tolerance
        )
    return {"within_5_pd": rate(5.0), "within_10_pd": rate(10.0)}


def _validate_coverage(coverage: float) -> float:
    coverage = float(coverage)
    if not math.isfinite(coverage) or not 0.0 < coverage < 1.0:
        raise ValueError("coverage must lie strictly between zero and one")
    return coverage


def normal_prediction_interval(
    mean: float | Sequence[float] | np.ndarray,
    standard_deviation: float | Sequence[float] | np.ndarray,
    *,
    coverage: float = 0.95,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Central prediction interval under a Gaussian uncertainty model."""

    coverage = _validate_coverage(coverage)
    mu, sigma = np.broadcast_arrays(
        np.asarray(mean, dtype=float), np.asarray(standard_deviation, dtype=float)
    )
    _require_finite_or_nan(mu, "mean")
    _require_finite_or_nan(sigma, "standard_deviation")
    finite_sigma = np.isfinite(sigma)
    if np.any(sigma[finite_sigma] < 0.0):
        raise ValueError("standard_deviation must be non-negative")
    z = NormalDist().inv_cdf(0.5 + coverage / 2.0)
    lower = mu - z * sigma
    upper = mu + z * sigma
    if np.ndim(mean) == 0 and np.ndim(standard_deviation) == 0:
        return float(lower), float(upper)
    return lower, upper


def empirical_prediction_interval(
    samples: Sequence[float] | np.ndarray,
    *,
    coverage: float = 0.95,
    axis: int | tuple[int, ...] | None = 0,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Central percentile interval from ensemble, bootstrap, or MC samples."""

    coverage = _validate_coverage(coverage)
    values = np.asarray(samples, dtype=float)
    if values.size == 0:
        raise ValueError("samples must not be empty")
    alpha = (1.0 - coverage) / 2.0
    lower = np.nanquantile(values, alpha, axis=axis)
    upper = np.nanquantile(values, 1.0 - alpha, axis=axis)
    if np.ndim(lower) == 0:
        return float(lower), float(upper)
    return lower, upper


def prediction_interval_coverage(
    target: float | Sequence[float] | np.ndarray,
    lower: float | Sequence[float] | np.ndarray,
    upper: float | Sequence[float] | np.ndarray,
) -> float:
    """Empirical inclusive coverage of point targets by prediction intervals."""

    target_values, lo, hi = _broadcast_interval_inputs(target, lower, upper)
    finite = np.isfinite(target_values) & np.isfinite(lo) & np.isfinite(hi)
    if not np.any(finite):
        raise ValueError("at least one finite target and interval are required")
    covered = (target_values[finite] >= lo[finite]) & (target_values[finite] <= hi[finite])
    return float(np.mean(covered))


def mean_prediction_interval_width(
    lower: float | Sequence[float] | np.ndarray,
    upper: float | Sequence[float] | np.ndarray,
) -> float:
    """Mean finite width of prediction intervals."""

    lo, hi = np.broadcast_arrays(np.asarray(lower, dtype=float), np.asarray(upper, dtype=float))
    finite = np.isfinite(lo) & np.isfinite(hi)
    if np.any(lo[finite] > hi[finite]):
        raise ValueError("lower interval bounds must not exceed upper bounds")
    if not np.any(finite):
        raise ValueError("at least one finite prediction interval is required")
    return float(np.mean(hi[finite] - lo[finite]))


def prism_uncertainty_from_angle(
    angle_degrees: float | Sequence[float] | np.ndarray,
    angle_standard_deviation_degrees: float | Sequence[float] | np.ndarray,
) -> float | np.ndarray:
    """First-order propagation of angular SD into prism-diopter SD.

    The derivative is ``(100*pi/180) * sec(angle)**2`` PD per degree.
    """

    angle, sigma = np.broadcast_arrays(
        np.asarray(angle_degrees, dtype=float),
        np.asarray(angle_standard_deviation_degrees, dtype=float),
    )
    _require_finite_or_nan(angle, "angle_degrees")
    _require_finite_or_nan(sigma, "angle_standard_deviation_degrees")
    finite_angle = np.isfinite(angle)
    if np.any(np.abs(angle[finite_angle]) >= 90.0):
        raise ValueError("angle_degrees must lie strictly between -90 and 90")
    finite_sigma = np.isfinite(sigma)
    if np.any(sigma[finite_sigma] < 0.0):
        raise ValueError("angle standard deviation must be non-negative")
    derivative = (100.0 * np.pi / 180.0) / np.cos(np.deg2rad(angle)) ** 2
    result = derivative * sigma
    if np.ndim(angle_degrees) == 0 and np.ndim(angle_standard_deviation_degrees) == 0:
        return float(result)
    return result


def angle_normal_prediction_interval_in_prism_diopters(
    mean_angle_degrees: float | Sequence[float] | np.ndarray,
    angle_standard_deviation_degrees: float | Sequence[float] | np.ndarray,
    *,
    coverage: float = 0.95,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Gaussian angle interval transformed through the exact tangent geometry."""

    lower_angle, upper_angle = normal_prediction_interval(
        mean_angle_degrees,
        angle_standard_deviation_degrees,
        coverage=coverage,
    )
    lower_prism = degrees_to_prism_diopters(lower_angle)
    upper_prism = degrees_to_prism_diopters(upper_angle)
    return lower_prism, upper_prism


__all__ = [
    "BlandAltmanStats",
    "PrismInterval",
    "angle_degrees_to_prism_diopters",
    "angle_normal_prediction_interval_in_prism_diopters",
    "bland_altman",
    "bland_altman_differences",
    "degrees_to_prism_diopters",
    "deviation_sign",
    "empirical_prediction_interval",
    "interval_aware_mae",
    "interval_loss",
    "interval_midpoint",
    "interval_residual",
    "interval_within_tolerance_rate",
    "mean_prediction_interval_width",
    "midpoint_fallback",
    "normal_prediction_interval",
    "parse_prism_interval",
    "prediction_interval_coverage",
    "prism_diopters_to_angle_degrees",
    "prism_diopters_to_degrees",
    "prism_diopters_to_radians",
    "prism_uncertainty_from_angle",
    "project_to_interval",
    "radians_to_prism_diopters",
    "signed_prism_diopters",
    "within_5_10",
    "within_tolerance_rate",
]
