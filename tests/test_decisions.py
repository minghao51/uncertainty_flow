"""Tests for uncertainty_flow.decisions module."""

import numpy as np
import pytest

from uncertainty_flow.core.distribution import DistributionPrediction
from uncertainty_flow.decisions import InventoryOptimiser, TargetOptimizer, ThresholdAction


def dummy_prediction_multi_target(n_samples=10, q_levels=[0.1, 0.5, 0.9]):
    """Create a multi-target DistributionPrediction for testing."""
    q_matrix = np.zeros((n_samples, len(q_levels) * 2))
    for i in range(n_samples):
        q_matrix[i, :] = np.array([-1.28, 0, 1.28, -1.28, 0, 1.28]) + i
    return DistributionPrediction(
        quantile_matrix=q_matrix,
        quantile_levels=q_levels,
        target_names=["y1", "y2"],
    )


def dummy_prediction_1_target(n_samples=5, q_levels=[0.1, 0.5, 0.9]):
    """Create a single-target DistributionPrediction for testing."""
    q_matrix = np.zeros((n_samples, len(q_levels)))
    for i in range(n_samples):
        q_matrix[i, :] = np.array([-1.28, 0, 1.28]) + i
    return DistributionPrediction(
        quantile_matrix=q_matrix,
        quantile_levels=q_levels,
        target_names=["y"],
    )


class TestTargetOptimizer:
    """Test TargetOptimizer decision strategy."""

    def test_rejects_invalid_confidence(self):
        """Should raise ValueError for invalid confidence levels."""
        with pytest.raises(ValueError, match="confidence must be in"):
            TargetOptimizer(target=0.0, confidence=0.0)
        with pytest.raises(ValueError, match="confidence must be in"):
            TargetOptimizer(target=0.0, confidence=1.0)
        with pytest.raises(ValueError, match="confidence must be in"):
            TargetOptimizer(target=0.0, confidence=-0.5)

    def test_rejects_invalid_confidence_boundary(self):
        """Should accept confidence=0.001 and confidence=0.999."""
        TargetOptimizer(target=0.0, confidence=0.001)
        TargetOptimizer(target=0.0, confidence=0.999)

    def test_multivariate_returns_per_target_optimal(self):
        """Multi-target predictions should return optimal values for each target."""
        pred = dummy_prediction_multi_target(n_samples=5, q_levels=[0.1, 0.5, 0.9])

        strategy = TargetOptimizer(target=5.0, confidence=0.9)
        result = strategy.resolve(pred)

        assert result.optimal_value.width == 2
        assert "y1" in result.optimal_value.columns
        assert "y2" in result.optimal_value.columns
        assert "target" in result.metadata
        assert "requested_confidence" in result.metadata
        assert "actual_probability" in result.metadata

    def test_interpolation_fallback_on_duplicate_quantiles(self):
        """When quantiles are duplicate, interpolation should fall back to nearest neighbor."""
        q_levels = [0.1, 0.2, 0.2, 0.5, 0.8]
        q_matrix = np.array([
            [0.0, 0.0, 5.0],
            [1.0, 1.0, 5.0],
            [2.0, 2.0, 5.0],
            [3.0, 3.0, 5.0],
            [4.0, 4.0, 5.0],
            [5.0, 5.0, 5.0],
        ])

        pred = DistributionPrediction(
            quantile_matrix=q_matrix,
            quantile_levels=q_levels,
            target_names=["y"],
        )

        strategy = TargetOptimizer(target=3.0, confidence=0.8)
        result = strategy.resolve(pred)

        assert result.optimal_value[0] == 3.0


class TestInventoryOptimiser:
    """Test InventoryOptimiser decision strategy."""

    def test_rejects_nonpositive_costs(self):
        """Should raise ValueError for non-positive costs."""
        with pytest.raises(ValueError, match="Costs must be positive"):
            InventoryOptimiser(stockout_cost=0, overstock_cost=1)
        with pytest.raises(ValueError, match="Costs must be positive"):
            InventoryOptimiser(stockout_cost=1, overstock_cost=0)
        with pytest.raises(ValueError, match="Costs must be positive"):
            InventoryOptimiser(stockout_cost=-1, overstock_cost=1)

    def test_multivariate_returns_optimal_quantity_per_target(self):
        """Multi-target predictions should return optimal quantity for each target."""
        pred = dummy_prediction_multi_target(n_samples=5, q_levels=[0.1, 0.5, 0.9])

        strategy = InventoryOptimiser(stockout_cost=10, overstock_cost=2)
        result = strategy.resolve(pred)

        assert result.optimal_value.width == 2
        assert "y1" in result.optimal_value.columns
        assert "y2" in result.optimal_value.columns
        assert "target_quantile" in result.metadata
        assert pytest.approx(result.metadata["target_quantile"]) == 10 / 12

    def test_single_target_returns_series(self):
        """Single-target predictions should return a Series."""
        pred = dummy_prediction_1_target(n_samples=5, q_levels=[0.1, 0.5, 0.9])

        strategy = InventoryOptimiser(stockout_cost=10, overstock_cost=2)
        result = strategy.resolve(pred)

        assert isinstance(result.optimal_value, pl.Series)
        assert result.optimal_value.name == "optimal_quantity"


class TestThresholdAction:
    """Test ThresholdAction decision strategy."""

    def test_rejects_invalid_probability(self):
        """Should raise ValueError for invalid probability."""
        with pytest.raises(ValueError, match="min_probability must be in"):
            ThresholdAction(threshold=5.0, min_probability=0.0)
        with pytest.raises(ValueError, match="min_probability must be in"):
            ThresholdAction(threshold=5.0, min_probability=1.0)

    def test_rejects_invalid_probability_boundary(self):
        """Should accept min_probability=0.001 and min_probability=0.999."""
        ThresholdAction(threshold=5.0, min_probability=0.001)
        ThresholdAction(threshold=5.0, min_probability=0.999)

    def test_multivariate_returns_per_column_exceeded(self):
        """Multi-target predictions should return per-column threshold exceeded."""
        pred = dummy_prediction_multi_target(n_samples=5, q_levels=[0.1, 0.5, 0.9])

        strategy = ThresholdAction(threshold=3.0, min_probability=0.8)
        result = strategy.resolve(pred)

        assert result.optimal_value.width == 2
        assert "y1_exceeded" in result.optimal_value.columns
        assert "y2_exceeded" in result.optimal_value.columns
        assert "threshold" in result.metadata
        assert "min_probability" in result.metadata

    def test_single_target_returns_dataframe(self):
        """Single-target predictions should return a DataFrame."""
        pred = dummy_prediction_1_target(n_samples=5, q_levels=[0.1, 0.5, 0.9])

        strategy = ThresholdAction(threshold=3.0, min_probability=0.8)
        result = strategy.resolve(pred)

        assert isinstance(result.optimal_value, pl.DataFrame)
        assert "threshold_exceeded" in result.optimal_value.columns
