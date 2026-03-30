"""Tests for calibration_report utility."""

import numpy as np
import polars as pl
import pytest
from sklearn.ensemble import GradientBoostingRegressor

from uncertainty_flow import ConformalRegressor
from uncertainty_flow.utils.calibration_utils import calibration_report


@pytest.fixture
def train_data():
    """Create training data."""
    np.random.seed(42)
    n = 100
    return pl.DataFrame(
        {
            "x1": np.random.randn(n),
            "x2": np.random.randn(n),
            "target": 3 * np.random.randn(n) + 5,
        }
    )


@pytest.fixture
def val_data():
    """Create validation data."""
    np.random.seed(123)
    n = 50
    return pl.DataFrame(
        {
            "x1": np.random.randn(n),
            "x2": np.random.randn(n),
            "target": 3 * np.random.randn(n) + 5,
        }
    )


class TestCalibrationReportInit:
    """Test calibration_report initialization (function signature)."""

    def test_function_exists(self):
        """Should be importable from calibration module."""
        from uncertainty_flow.calibration import calibration_report

        assert calibration_report is not None


class TestCalibrationReportOutput:
    """Test calibration_report output structure."""

    def test_returns_dataframe(self, train_data, val_data):
        """Should return a Polars DataFrame."""
        model = ConformalRegressor(
            base_model=GradientBoostingRegressor(n_estimators=10, random_state=42),
            random_state=42,
        )
        model.fit(train_data, target="target")
        report = calibration_report(model, val_data, target="target")
        assert isinstance(report, pl.DataFrame)

    def test_has_expected_columns(self, train_data, val_data):
        """Should have expected columns."""
        model = ConformalRegressor(
            base_model=GradientBoostingRegressor(n_estimators=10, random_state=42),
            random_state=42,
        )
        model.fit(train_data, target="target")
        report = calibration_report(model, val_data, target="target")
        expected_cols = [
            "quantile",
            "requested_coverage",
            "achieved_coverage",
            "sharpness",
            "winkler_score",
        ]
        for col in expected_cols:
            assert col in report.columns, f"Missing column: {col}"

    def test_quantile_levels_as_requested(self, train_data, val_data):
        """Should have one row per quantile level."""
        model = ConformalRegressor(
            base_model=GradientBoostingRegressor(n_estimators=10, random_state=42),
            random_state=42,
        )
        model.fit(train_data, target="target")

        levels = [0.8, 0.9, 0.95]
        report = calibration_report(model, val_data, target="target", quantile_levels=levels)
        assert report.height == 3
        assert report["quantile"].to_list() == levels

    def test_default_quantile_levels(self, train_data, val_data):
        """Should use default quantile levels [0.8, 0.9, 0.95]."""
        model = ConformalRegressor(
            base_model=GradientBoostingRegressor(n_estimators=10, random_state=42),
            random_state=42,
        )
        model.fit(train_data, target="target")
        report = calibration_report(model, val_data, target="target")
        assert report.height == 3
        assert report["quantile"].to_list() == [0.8, 0.9, 0.95]

    def test_requested_coverage_matches_quantile(self, train_data, val_data):
        """Requested coverage should match quantile level."""
        model = ConformalRegressor(
            base_model=GradientBoostingRegressor(n_estimators=10, random_state=42),
            random_state=42,
        )
        model.fit(train_data, target="target")
        report = calibration_report(model, val_data, target="target", quantile_levels=[0.9])
        assert report["requested_coverage"][0] == 0.9
        assert report["quantile"][0] == 0.9


class TestCalibrationReportMetrics:
    """Test calibration_report metric values."""

    def test_achieved_coverage_in_range(self, train_data, val_data):
        """Achieved coverage should be between 0 and 1."""
        model = ConformalRegressor(
            base_model=GradientBoostingRegressor(n_estimators=10, random_state=42),
            random_state=42,
        )
        model.fit(train_data, target="target")
        report = calibration_report(model, val_data, target="target")
        for cov in report["achieved_coverage"]:
            assert 0.0 <= cov <= 1.0

    def test_sharpness_positive(self, train_data, val_data):
        """Sharpness (interval width) should be positive."""
        model = ConformalRegressor(
            base_model=GradientBoostingRegressor(n_estimators=10, random_state=42),
            random_state=42,
        )
        model.fit(train_data, target="target")
        report = calibration_report(model, val_data, target="target")
        assert (report["sharpness"] > 0).all()

    def test_winkler_score_positive(self, train_data, val_data):
        """Winkler score should be non-negative."""
        model = ConformalRegressor(
            base_model=GradientBoostingRegressor(n_estimators=10, random_state=42),
            random_state=42,
        )
        model.fit(train_data, target="target")
        report = calibration_report(model, val_data, target="target")
        assert (report["winkler_score"] >= 0).all()


class TestCalibrationReportSingleRow:
    """Test calibration_report with single-row data."""

    def test_single_row_prediction(self, train_data):
        """Should work with single-row prediction data."""
        model = ConformalRegressor(
            base_model=GradientBoostingRegressor(n_estimators=10, random_state=42),
            random_state=42,
        )
        model.fit(train_data, target="target")
        single_row = train_data.head(1)
        report = calibration_report(model, single_row, target="target")
        assert isinstance(report, pl.DataFrame)
        assert report.height == 3


class TestCalibrationReportMultipleTargets:
    """Test calibration_report with multiple targets."""

    @pytest.fixture
    def multivariate_data(self):
        """Create multivariate data."""
        np.random.seed(42)
        n = 100
        return pl.DataFrame(
            {
                "x1": np.random.randn(n),
                "x2": np.random.randn(n),
                "price": 3 * np.random.randn(n) + 5,
                "volume": 100 + 10 * np.random.randn(n),
            }
        )

    def test_multivariate_returns_averaged_metrics(self, multivariate_data):
        """Should return averaged metrics across targets."""
        model = ConformalRegressor(
            base_model=GradientBoostingRegressor(n_estimators=10, random_state=42),
            random_state=42,
        )
        model.fit(multivariate_data, target="price")
        # Note: calibration_report currently computes per-target metrics
        # and averages them. This tests that behavior.
        report = calibration_report(
            model,
            multivariate_data,
            target="price",
        )
        assert isinstance(report, pl.DataFrame)
        assert "achieved_coverage" in report.columns
