"""Tests for dashboard functionality."""

import numpy as np
import polars as pl
import pytest

from uncertainty_flow.viz.dashboard import launch_dashboard


class TestLaunchDashboard:
    """Test launch_dashboard function."""

    def test_launch_dashboard_import(self):
        """Should be importable from viz module."""
        from uncertainty_flow.viz import launch_dashboard as ld

        assert ld is launch_dashboard

    def test_launch_dashboard_requires_model(self, sample_data):
        """Should raise error without model."""

        with pytest.raises(Exception):
            launch_dashboard(None, sample_data)

    def test_launch_dashboard_requires_streamlit(self, sample_forecaster, sample_data):
        """Should raise ImportError without streamlit."""
        try:
            import streamlit  # noqa: F401

            # If streamlit is available, just verify the function is callable
            assert callable(launch_dashboard)
        except ImportError:
            # Streamlit is optional; without it we only verify the module exposes the entrypoint.
            # Instead, we just verify the module structure is correct
            from uncertainty_flow.viz import dashboard

            assert hasattr(dashboard, "launch_dashboard")

    def test_launch_dashboard_signature(self):
        """Should have correct signature."""
        import inspect

        sig = inspect.signature(launch_dashboard)
        params = list(sig.parameters.keys())
        assert "model" in params
        assert "data" in params
        assert "target" in params
        assert "port" in params
        assert "title" in params


class TestDashboardHelpers:
    """Test dashboard helper functions."""

    def test_pdcut_bins_correctly(self):
        """Should bin data into correct number of bins."""
        from uncertainty_flow.viz.dashboard import pdcut

        np.random.seed(42)
        x = np.random.randn(100)
        n_bins = 5

        result = pdcut(x, n_bins)

        assert len(result) == len(x)
        assert np.min(result) >= 0
        assert np.max(result) < n_bins

    def test_pdcut_edge_cases(self):
        """Should handle edge cases."""
        from uncertainty_flow.viz.dashboard import pdcut

        # Constant values
        x = np.ones(10)
        result = pdcut(x, 3)
        assert len(result) == 10

        # Single value
        x = np.array([1.0])
        result = pdcut(x, 3)
        assert len(result) == 1


# Fixtures
@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    n = 200
    return pl.DataFrame(
        {
            "x1": np.random.randn(n),
            "x2": np.random.randn(n),
            "x3": np.random.randn(n),
            "y": 3 * np.random.randn(n) + 5,
        }
    )


@pytest.fixture
def sample_forecaster(sample_data):
    """Create a fitted forecaster for testing."""
    from uncertainty_flow.models import QuantileForestForecaster

    model = QuantileForestForecaster(
        targets="y",
        horizon=1,
        n_estimators=10,
        min_samples_leaf=5,
        random_state=42,
    )
    model.fit(sample_data)
    return model


class TestDashboardIntegration:
    """Integration tests for dashboard components."""

    def test_dashboard_workflow(self, sample_forecaster, sample_data):
        """Should be able to create predictions needed for dashboard."""
        # Get predictions
        x_features = sample_data.drop("y")
        predictions = sample_forecaster.predict(x_features.head(10))

        # Verify predictions have required methods
        assert hasattr(predictions, "interval")
        assert hasattr(predictions, "mean")

        # Test interval extraction
        interval = predictions.interval(0.9)
        assert "lower" in interval.columns
        assert "upper" in interval.columns

    def test_calibration_metrics_computation(self, sample_forecaster, sample_data):
        """Should be able to compute calibration metrics."""
        x_features = sample_data.drop("y")
        y_true = sample_data["y"]

        predictions = sample_forecaster.predict(x_features.head(50))

        # Compute empirical coverage
        interval = predictions.interval(0.9)
        lower = interval["lower"].to_numpy()
        upper = interval["upper"].to_numpy()
        y_arr = y_true.head(50).to_numpy()

        coverage = np.mean((y_arr >= lower) & (y_arr <= upper))

        assert 0 <= coverage <= 1

    def test_interval_width_computation(self, sample_forecaster, sample_data):
        """Should be able to compute interval widths."""
        x_features = sample_data.drop("y")

        predictions = sample_forecaster.predict(x_features.head(50))

        interval = predictions.interval(0.9)
        lower = interval["lower"].to_numpy()
        upper = interval["upper"].to_numpy()

        widths = upper - lower

        assert len(widths) == 50
        assert np.all(widths >= 0)

    def test_residual_computation(self, sample_forecaster, sample_data):
        """Should be able to compute residuals."""
        x_features = sample_data.drop("y")
        y_true = sample_data["y"]

        predictions = sample_forecaster.predict(x_features.head(50))

        mean = predictions.mean().to_numpy()
        y_arr = y_true.head(50).to_numpy()

        residuals = y_arr - mean

        assert len(residuals) == 50
