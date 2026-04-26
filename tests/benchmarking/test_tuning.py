"""Tests for uncertainty_flow.benchmarking.tuning module."""

import numpy as np
import polars as pl
import pytest

from uncertainty_flow.benchmarking.tuning import (
    coverage_objective,
    tune_quantile_forest,
)
from uncertainty_flow.models import QuantileForestForecaster


@pytest.fixture
def sample_tuning_data():
    """Create sample data for tuning tests."""
    np.random.seed(42)
    n = 120
    return pl.DataFrame(
        {
            "date": range(n),
            "price": 10 + np.arange(n) * 0.5 + np.random.randn(n) * 0.5,
        }
    )


class TestCoverageObjective:
    """Test coverage objective function."""

    def test_lower_is_better(self):
        """Lower coverage error should increase objective."""
        base = coverage_objective(coverage=0.9, sharpness=0.1)
        worse = coverage_objective(coverage=0.8, sharpness=0.1)

        assert worse < base

    def test_sharpness_penalty(self):
        """Higher sharpness should penalize more."""
        base = coverage_objective(coverage=0.9, sharpness=0.1)
        penalized = coverage_objective(coverage=0.9, sharpness=0.2)

        assert penalized > base

    def test_large_coverage_error_penalizes_more(self):
        """Large coverage error should penalize heavily."""
        base = coverage_objective(coverage=0.9, sharpness=0.1)
        large_error = coverage_objective(coverage=0.7, sharpness=0.1)

        assert large_error < base


class TestTuneQuantileForest:
    """Test tune_quantile_forest function."""

    def test_tune_quantile_forest_returns_metrics(self, sample_tuning_data):
        """Should return coverage, sharpness, and training time."""
        cov, sharp, train_time = tune_quantile_forest(
            df=sample_tuning_data(),
            target="price",
            horizon=3,
            n_estimators=10,
            random_state=42,
        )

        assert isinstance(cov, float)
        assert isinstance(sharp, float)
        assert isinstance(train_time, float)
        assert 0 < cov < 1
        assert sharp > 0
        assert train_time > 0

    def test_tune_quantile_forest_with_tuned_params(self, sample_tuning_data):
        """Should respect provided tuned parameters."""
        tuned_params = tune_quantile_forest(
            df=sample_tuning_data(),
            target="price",
            horizon=3,
            n_estimators=10,
            random_state=42,
        )

        assert "n_estimators" in tuned_params.best_params
        assert "coverage" in tuned_params.best_params
        assert "sharpness" in tuned_params.best_params

    def test_auto_tune_model(self, sample_tuning_data):
        """Should integrate with auto_tune_model."""
        from uncertainty_flow.benchmarking.tuning import auto_tune_model

        config = {
            "target_coverage": 0.9,
            "n_samples": 100,
            "timeout": 300,
        }
        result = auto_tune_model(
            model_factory=QuantileForestForecaster,
            data=sample_tuning_data(),
            target="price",
            horizon=3,
            config=config,
        )

        assert result.model_name == "quantile-forest"
        assert result.best_params is not None
        assert "n_estimators" in result.best_params
