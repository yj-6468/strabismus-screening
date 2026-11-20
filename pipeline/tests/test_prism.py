from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from strabismus_screen.prism import (  # noqa: E402
    PrismInterval,
    angle_normal_prediction_interval_in_prism_diopters,
    bland_altman,
    bland_altman_differences,
    degrees_to_prism_diopters,
    empirical_prediction_interval,
    interval_aware_mae,
    interval_loss,
    interval_midpoint,
    interval_residual,
    interval_within_tolerance_rate,
    mean_prediction_interval_width,
    normal_prediction_interval,
    parse_prism_interval,
    prediction_interval_coverage,
    prism_diopters_to_degrees,
    prism_diopters_to_radians,
    prism_uncertainty_from_angle,
    project_to_interval,
    radians_to_prism_diopters,
    signed_prism_diopters,
    within_5_10,
    within_tolerance_rate,
)


class PrismGeometryTests(unittest.TestCase):
    def test_exact_tangent_definition_and_sign(self) -> None:
        self.assertAlmostEqual(degrees_to_prism_diopters(45.0), 100.0, places=12)
        self.assertAlmostEqual(degrees_to_prism_diopters(-45.0), -100.0, places=12)
        self.assertAlmostEqual(degrees_to_prism_diopters(0.0), 0.0, places=12)
        self.assertAlmostEqual(radians_to_prism_diopters(math.pi / 4.0), 100.0)
        self.assertAlmostEqual(prism_diopters_to_radians(100.0), math.pi / 4.0)

    def test_round_trip_vector(self) -> None:
        prism = np.array([-65.0, -20.0, 0.0, 10.0, 65.0])
        recovered = degrees_to_prism_diopters(prism_diopters_to_degrees(prism))
        np.testing.assert_allclose(recovered, prism, rtol=0.0, atol=1e-12)

    def test_angle_domain_is_guarded(self) -> None:
        with self.assertRaises(ValueError):
            degrees_to_prism_diopters(90.0)
        with self.assertRaises(ValueError):
            degrees_to_prism_diopters(-90.0)

    def test_signed_convention(self) -> None:
        self.assertEqual(signed_prism_diopters(18.0, "ET"), 18.0)
        self.assertEqual(signed_prism_diopters(18.0, "X(T)"), -18.0)
        self.assertEqual(signed_prism_diopters(0.0, "normal"), 0.0)
        with self.assertRaises(ValueError):
            signed_prism_diopters(2.0, "normal")


class PrismTargetTests(unittest.TestCase):
    def test_signed_interval_construction_and_parsing(self) -> None:
        self.assertEqual(parse_prism_interval("16-20", "ET"), PrismInterval(16.0, 20.0))
        self.assertEqual(parse_prism_interval("16–20 Δ", "XT"), PrismInterval(-20.0, -16.0))
        self.assertEqual(parse_prism_interval("", "normal"), PrismInterval(0.0, 0.0))
        with self.assertRaises(ValueError):
            parse_prism_interval("", "XT")
        with self.assertRaises(ValueError):
            parse_prism_interval("", "Flick E to Flick X")

    def test_midpoint_and_projection(self) -> None:
        self.assertEqual(interval_midpoint(-20.0, -16.0), -18.0)
        prediction = np.array([8.0, 15.0, 25.0])
        projected = project_to_interval(prediction, 10.0, 20.0)
        residual = interval_residual(prediction, 10.0, 20.0)
        np.testing.assert_array_equal(projected, np.array([10.0, 15.0, 20.0]))
        np.testing.assert_array_equal(residual, np.array([-2.0, 0.0, 5.0]))

        # A scalar prediction broadcasts against interval arrays without being
        # incorrectly coerced back to a scalar.
        np.testing.assert_array_equal(
            project_to_interval(12.0, np.array([0.0, 15.0]), np.array([10.0, 20.0])),
            np.array([10.0, 15.0]),
        )

    def test_interval_losses(self) -> None:
        prediction = np.array([8.0, 15.0, 25.0])
        np.testing.assert_array_equal(
            interval_loss(prediction, 10.0, 20.0, kind="absolute", reduction="none"),
            np.array([2.0, 0.0, 5.0]),
        )
        np.testing.assert_array_equal(
            interval_loss(prediction, 10.0, 20.0, kind="squared", reduction="none"),
            np.array([4.0, 0.0, 25.0]),
        )
        np.testing.assert_array_equal(
            interval_loss(
                prediction,
                10.0,
                20.0,
                kind="huber",
                huber_delta=1.0,
                reduction="none",
            ),
            np.array([1.5, 0.0, 4.5]),
        )
        self.assertAlmostEqual(interval_aware_mae(prediction, 10.0, 20.0), 7.0 / 3.0)


class PrismMetricTests(unittest.TestCase):
    def test_bland_altman_and_resampling_indices(self) -> None:
        prediction = np.array([1.0, 2.0, 4.0])
        reference = np.array([0.0, 2.0, 3.0])
        means, differences = bland_altman_differences(prediction, reference)
        np.testing.assert_allclose(means, np.array([0.5, 2.0, 3.5]))
        np.testing.assert_allclose(differences, np.array([1.0, 0.0, 1.0]))

        stats = bland_altman(prediction, reference)
        self.assertEqual(stats.n, 3)
        self.assertAlmostEqual(stats.bias, 2.0 / 3.0)
        self.assertAlmostEqual(stats.sd_difference, math.sqrt(1.0 / 3.0))
        self.assertAlmostEqual(
            stats.loa_lower, stats.bias - 1.96 * math.sqrt(1.0 / 3.0)
        )
        self.assertAlmostEqual(
            stats.loa_upper, stats.bias + 1.96 * math.sqrt(1.0 / 3.0)
        )

        bootstrap_stats = bland_altman(
            prediction, reference, sample_indices=np.array([0, 0, 2])
        )
        self.assertEqual(bootstrap_stats.bias, 1.0)
        self.assertEqual(bootstrap_stats.sd_difference, 0.0)

    def test_point_and_interval_tolerance_rates(self) -> None:
        prediction = np.array([0.0, 7.0, 20.0])
        reference = np.array([0.0, 0.0, 8.0])
        self.assertAlmostEqual(within_tolerance_rate(prediction, reference, 5.0), 1 / 3)
        self.assertAlmostEqual(within_tolerance_rate(prediction, reference, 10.0), 2 / 3)
        self.assertEqual(
            within_5_10(prediction, reference),
            {"within_5_pd": 1 / 3, "within_10_pd": 2 / 3},
        )

        interval_prediction = np.array([8.0, 15.0, 27.0])
        self.assertAlmostEqual(
            interval_within_tolerance_rate(interval_prediction, 10.0, 20.0, 5.0),
            2 / 3,
        )
        self.assertEqual(
            within_5_10(interval_prediction, lower=10.0, upper=20.0),
            {"within_5_pd": 2 / 3, "within_10_pd": 1.0},
        )


class PredictionIntervalTests(unittest.TestCase):
    def test_gaussian_and_empirical_intervals(self) -> None:
        lower, upper = normal_prediction_interval(10.0, 2.0, coverage=0.95)
        self.assertAlmostEqual(lower, 10.0 - 2.0 * 1.959963984540054)
        self.assertAlmostEqual(upper, 10.0 + 2.0 * 1.959963984540054)

        samples = np.arange(101.0)
        empirical_lower, empirical_upper = empirical_prediction_interval(
            samples, coverage=0.80, axis=0
        )
        self.assertAlmostEqual(empirical_lower, 10.0)
        self.assertAlmostEqual(empirical_upper, 90.0)

    def test_coverage_and_width(self) -> None:
        target = np.array([0.0, 5.0, 10.0])
        lower = np.array([-1.0, 5.0, 11.0])
        upper = np.array([1.0, 6.0, 12.0])
        self.assertAlmostEqual(prediction_interval_coverage(target, lower, upper), 2 / 3)
        self.assertAlmostEqual(mean_prediction_interval_width(lower, upper), 4 / 3)

    def test_angle_uncertainty_and_nonlinear_transform(self) -> None:
        self.assertAlmostEqual(
            prism_uncertainty_from_angle(0.0, 1.0), 100.0 * math.pi / 180.0
        )
        lower, upper = angle_normal_prediction_interval_in_prism_diopters(
            0.0, 1.0, coverage=0.95
        )
        expected = degrees_to_prism_diopters(1.959963984540054)
        self.assertAlmostEqual(lower, -expected)
        self.assertAlmostEqual(upper, expected)


if __name__ == "__main__":
    unittest.main()
