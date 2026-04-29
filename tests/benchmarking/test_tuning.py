"""Tests for uncertainty_flow.benchmarking.tuning module."""

import numpy as np
import polars as pl

from uncertainty_flow.benchmarking.tuning import (
    TuningConfig,
    auto_tune_model,
    tune_quantile_forest,
)


def sample_tuning_data() -> pl.DataFrame:
    """Create sample data for tuning tests."""
    np.random.seed(42)
    n = 120
    return pl.DataFrame(
        {
            "date": range(n),
            "price": 10 + np.arange(n) * 0.5 + np.random.randn(n) * 0.5,
        }
    )


class TestTuneQuantileForest:
    """Test tune_quantile_forest function."""

    def test_tune_quantile_forest_returns_metrics(self):
        """Should return coverage, sharpness, winkler and training time."""
        cov, sharp, wink, train_time = tune_quantile_forest(
            df=sample_tuning_data(),
            target="price",
            horizon=3,
            n_estimators=10,
        )

        assert isinstance(cov, float)
        assert isinstance(sharp, float)
        assert isinstance(wink, float)
        assert isinstance(train_time, float)
        assert 0 <= cov <= 1
        assert sharp > 0
        assert wink >= 0
        assert train_time >= 0


class TestAutoTuneModel:
    """Test auto_tune_model integration."""

    def test_auto_tune_model_quantile_forest(self):
        """Should return a populated TuningResult for quantile-forest."""
        config = TuningConfig(
            target_coverage=0.9,
            n_samples=100,
            timeout=30,
        )
        result = auto_tune_model(
            model_name="quantile-forest",
            df=sample_tuning_data(),
            target="price",
            horizon=3,
            config=config,
        )

        assert result.model_name == "quantile-forest"
        assert isinstance(result.best_params, dict)
        assert "n_estimators" in result.best_params
        assert "horizon" in result.best_params
        assert result.trials > 0
        assert 0 <= result.coverage_90 <= 1
        assert result.sharpness_90 >= 0
        assert result.winkler_90 >= 0
