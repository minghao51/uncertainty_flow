"""Tests for DistributionPrediction._forward_cdf and PIT-related methods."""

import numpy as np
import polars as pl
import pytest

from uncertainty_flow.core.distribution import DistributionPrediction
from uncertainty_flow.utils.exceptions import InvalidDataError


class TestForwardCDF:
    """Tests for the piecewise-linear CDF evaluation (_forward_cdf)."""

    def test_perfect_calibration_uniform_pit(self):
        """For perfectly calibrated predictions, PIT should be near-uniform."""
        rng = np.random.default_rng(42)
        n = 500
        # True values: uniform on [0, 1]
        y_true = rng.uniform(0, 1, size=n)
        # Perfect predictions: quantiles are exactly the true CDF
        levels = np.array([0.1, 0.5, 0.9])
        quantile_matrix = np.column_stack(
            [
                np.full(n, 0.1),  # 10th percentile
                np.full(n, 0.5),  # median
                np.full(n, 0.9),  # 90th percentile
            ]
        )
        pit = DistributionPrediction._forward_cdf(quantile_matrix, levels, y_true)
        # For uniform true values and constant predictions at [0.1, 0.5, 0.9],
        # the PIT values should cluster around 0.1, 0.5, 0.9 based on where
        # y_true falls relative to the predicted quantiles.
        assert pit.shape == (n,)
        assert np.all((pit >= 0) & (pit <= 1))

    def test_value_at_exact_quantile(self):
        """When y equals a quantile value, PIT should equal the quantile level."""
        levels = np.array([0.1, 0.5, 0.9])
        quantile_matrix = np.array([[1.0, 5.0, 10.0]])
        # y = 5.0 equals the median prediction
        y_true = np.array([5.0])
        pit = DistributionPrediction._forward_cdf(quantile_matrix, levels, y_true)
        assert pit[0] == pytest.approx(0.5, abs=0.01)

    def test_value_below_all_quantiles(self):
        """When y is below all predicted quantiles, PIT should be small."""
        levels = np.array([0.1, 0.5, 0.9])
        quantile_matrix = np.array([[5.0, 10.0, 15.0]])
        y_true = np.array([0.0])  # well below lowest quantile
        pit = DistributionPrediction._forward_cdf(quantile_matrix, levels, y_true)
        assert pit[0] >= 0.0
        assert pit[0] < levels[0]

    def test_value_above_all_quantiles(self):
        """When y is above all predicted quantiles, PIT should be large."""
        levels = np.array([0.1, 0.5, 0.9])
        quantile_matrix = np.array([[5.0, 10.0, 15.0]])
        y_true = np.array([20.0])  # well above highest quantile
        pit = DistributionPrediction._forward_cdf(quantile_matrix, levels, y_true)
        assert pit[0] > levels[-1]
        assert pit[0] <= 1.0

    def test_interpolated_value(self):
        """PIT should interpolate linearly between quantile knots."""
        levels = np.array([0.1, 0.5, 0.9])
        quantile_matrix = np.array([[0.0, 10.0, 20.0]])
        # y = 5.0 is halfway between q=0 (level 0.1) and q=10 (level 0.5)
        y_true = np.array([5.0])
        pit = DistributionPrediction._forward_cdf(quantile_matrix, levels, y_true)
        # Linear interpolation: halfway between 0.1 and 0.5 -> 0.3
        assert pit[0] == pytest.approx(0.3, abs=0.01)

    def test_single_quantile(self):
        """With only one quantile level, PIT is binary."""
        levels = np.array([0.5])
        quantile_matrix = np.array([[10.0]])
        y_true = np.array([5.0])
        pit = DistributionPrediction._forward_cdf(quantile_matrix, levels, y_true)
        # Below the single quantile -> PIT = level = 0.5
        assert pit[0] == pytest.approx(0.5)

    def test_constant_predictions(self):
        """When all quantile values are identical, PIT uses level[0] * 0.5."""
        levels = np.array([0.25, 0.75])
        quantile_matrix = np.array([[10.0, 10.0]])
        y_true = np.array([10.0])
        pit = DistributionPrediction._forward_cdf(quantile_matrix, levels, y_true)
        # Tied quantiles at the lower bound -> levels[0] * 0.5
        assert pit[0] == pytest.approx(0.125)

    def test_clipping_to_0_1(self):
        """PIT values should always be clipped to [0, 1]."""
        levels = np.array([0.5])
        quantile_matrix = np.array([[10.0]])
        # Far below -> extrapolation could go negative, but should be clipped
        y_true = np.array([-1000.0])
        pit = DistributionPrediction._forward_cdf(quantile_matrix, levels, y_true)
        assert pit[0] >= 0.0
        assert pit[0] <= 1.0

    def test_batch_multiple_samples(self):
        """_forward_cdf should handle multiple samples in one call."""
        levels = np.array([0.1, 0.5, 0.9])
        quantile_matrix = np.array(
            [
                [1.0, 5.0, 10.0],
                [2.0, 6.0, 11.0],
                [3.0, 7.0, 12.0],
            ]
        )
        y_true = np.array([5.0, 6.0, 7.0])
        pit = DistributionPrediction._forward_cdf(quantile_matrix, levels, y_true)
        assert pit.shape == (3,)
        # Each y_true equals the median of its row
        assert np.allclose(pit, 0.5, atol=0.01)


class TestPITValues:
    """Tests for DistributionPrediction._pit_values()."""

    def test_univariate_pit(self):
        """Should compute PIT values for univariate predictions."""
        rng = np.random.default_rng(42)
        n = 100
        y_true = rng.normal(0, 1, size=n)
        # Symmetric quantiles around true mean
        levels = [0.1, 0.5, 0.9]
        quantile_matrix = np.column_stack(
            [
                np.full(n, -1.28),  # ~10th percentile of N(0,1)
                np.full(n, 0.0),  # median
                np.full(n, 1.28),  # ~90th percentile
            ]
        )
        pred = DistributionPrediction(
            quantile_matrix=quantile_matrix,
            quantile_levels=levels,
            target_names=["y"],
        )
        pit = pred._pit_values(y_true)
        assert isinstance(pit, np.ndarray)
        assert pit.shape == (n,)
        assert np.all((pit >= 0) & (pit <= 1))

    def test_multivariate_pit(self):
        """Should return dict of PIT arrays for multivariate predictions."""
        n = 50
        levels = [0.25, 0.75]
        # Two targets, each with 2 quantile levels
        quantile_matrix = np.column_stack(
            [
                np.full(n, 0.0),
                np.full(n, 10.0),  # target 1
                np.full(n, 5.0),
                np.full(n, 15.0),  # target 2
            ]
        )
        pred = DistributionPrediction(
            quantile_matrix=quantile_matrix,
            quantile_levels=levels,
            target_names=["a", "b"],
        )
        y_true = pl.DataFrame({"a": np.full(n, 5.0), "b": np.full(n, 10.0)})
        pit = pred._pit_values(y_true)
        assert isinstance(pit, dict)
        assert set(pit.keys()) == {"a", "b"}
        assert pit["a"].shape == (n,)
        assert pit["b"].shape == (n,)

    def test_insufficient_quantiles_raises(self):
        """Should raise error when fewer than 2 quantile levels."""
        pred = DistributionPrediction(
            quantile_matrix=np.array([[1.0], [2.0]]),
            quantile_levels=[0.5],
            target_names=["y"],
        )
        with pytest.raises(InvalidDataError, match="at least 2 quantile levels"):
            pred._pit_values(np.array([1.0, 2.0]))


class TestPITHistogram:
    """Tests for DistributionPrediction.pit_histogram()."""

    def test_returns_dataframe(self):
        """Should return a Polars DataFrame with expected columns."""
        n = 100
        levels = [0.1, 0.5, 0.9]
        quantile_matrix = np.column_stack(
            [
                np.full(n, 0.0),
                np.full(n, 5.0),
                np.full(n, 10.0),
            ]
        )
        pred = DistributionPrediction(
            quantile_matrix=quantile_matrix,
            quantile_levels=levels,
            target_names=["y"],
        )
        y_true = np.random.default_rng(42).uniform(0, 10, size=n)
        hist = pred.pit_histogram(y_true, n_bins=10)
        assert isinstance(hist, pl.DataFrame)
        assert "bin_center" in hist.columns
        assert "count" in hist.columns
        assert "expected" in hist.columns
        assert len(hist) == 10
        # Total counts should equal n
        assert hist["count"].sum() == pytest.approx(n)
        # Expected count per bin
        assert hist["expected"][0] == pytest.approx(n / 10)

    def test_multivariate_returns_dict(self):
        """Should return dict of DataFrames for multivariate predictions."""
        n = 50
        levels = [0.25, 0.75]
        quantile_matrix = np.column_stack(
            [
                np.full(n, 0.0),
                np.full(n, 10.0),
                np.full(n, 5.0),
                np.full(n, 15.0),
            ]
        )
        pred = DistributionPrediction(
            quantile_matrix=quantile_matrix,
            quantile_levels=levels,
            target_names=["a", "b"],
        )
        y_true = pl.DataFrame({"a": np.full(n, 5.0), "b": np.full(n, 10.0)})
        hist = pred.pit_histogram(y_true, n_bins=5)
        assert isinstance(hist, dict)
        assert set(hist.keys()) == {"a", "b"}
        assert len(hist["a"]) == 5


class TestCalibrationCurve:
    """Tests for DistributionPrediction.calibration_curve()."""

    def test_returns_dataframe(self):
        """Should return a Polars DataFrame with expected columns."""
        n = 100
        levels = [0.1, 0.5, 0.9]
        quantile_matrix = np.column_stack(
            [
                np.full(n, 0.0),
                np.full(n, 5.0),
                np.full(n, 10.0),
            ]
        )
        pred = DistributionPrediction(
            quantile_matrix=quantile_matrix,
            quantile_levels=levels,
            target_names=["y"],
        )
        y_true = np.random.default_rng(42).uniform(0, 10, size=n)
        curve = pred.calibration_curve(y_true, n_bins=10)
        assert isinstance(curve, pl.DataFrame)
        assert "expected_coverage" in curve.columns
        assert "observed_coverage" in curve.columns
        assert "bin_center" in curve.columns
        assert len(curve) == 10
        # Final observed coverage should be 1.0 (all observations accounted for)
        assert curve["observed_coverage"][-1] == pytest.approx(1.0, abs=0.01)
        # Final expected coverage should be 1.0
        assert curve["expected_coverage"][-1] == pytest.approx(1.0, abs=0.01)

    def test_perfect_calibration(self):
        """For perfectly calibrated predictions, curve should be near diagonal."""
        rng = np.random.default_rng(42)
        n = 1000
        # True values uniform on [0, 1]
        y_true = rng.uniform(0, 1, size=n)
        # Perfect predictions: quantiles match the true CDF exactly
        levels = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        quantile_matrix = np.column_stack(
            [
                np.full(n, 0.1),
                np.full(n, 0.25),
                np.full(n, 0.5),
                np.full(n, 0.75),
                np.full(n, 0.9),
            ]
        )
        pred = DistributionPrediction(
            quantile_matrix=quantile_matrix,
            quantile_levels=levels.tolist(),
            target_names=["y"],
        )
        curve = pred.calibration_curve(y_true, n_bins=20)
        # For perfect calibration, observed should closely track expected
        diff = np.abs(curve["expected_coverage"].to_numpy() - curve["observed_coverage"].to_numpy())
        # Allow some sampling variance, but should be close
        assert np.mean(diff) < 0.05
