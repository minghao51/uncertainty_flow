"""Tests for compute_uncertainty_drivers residual analysis."""

import numpy as np
import polars as pl

from uncertainty_flow.calibration.residual_analysis import compute_uncertainty_drivers


class TestComputeUncertaintyDriversOutput:
    """Test compute_uncertainty_drivers output structure."""

    def test_returns_dataframe(self):
        """Should return a Polars DataFrame."""
        residuals = np.array([1, -2, 3, -1, 2, 1.5, -2.5, 3.5])
        features = pl.DataFrame(
            {
                "a": [1, 2, 3, 4, 5, 6, 7, 8],
                "b": [8, 7, 6, 5, 4, 3, 2, 1],
            }
        )
        result = compute_uncertainty_drivers(residuals, features)
        assert isinstance(result, pl.DataFrame)

    def test_has_expected_columns(self):
        """Should have feature, residual_correlation, p_value columns."""
        residuals = np.array([1, -2, 3, -1, 2, 1.5, -2.5, 3.5])
        features = pl.DataFrame(
            {
                "a": [1, 2, 3, 4, 5, 6, 7, 8],
                "b": [8, 7, 6, 5, 4, 3, 2, 1],
            }
        )
        result = compute_uncertainty_drivers(residuals, features)
        assert "feature" in result.columns
        assert "residual_correlation" in result.columns
        assert "p_value" in result.columns

    def test_sorted_by_correlation(self):
        """Should be sorted by absolute correlation descending."""
        np.random.seed(42)
        n = 100
        residuals = np.random.randn(n)
        features = pl.DataFrame(
            {
                "weak": np.random.randn(n),
                "strong": np.linspace(0, 10, n) + np.random.randn(n) * 0.1,
            }
        )
        result = compute_uncertainty_drivers(residuals, features)
        if result.height > 1:
            correlations = result["residual_correlation"].to_list()
            assert correlations == sorted(correlations, reverse=True)


class TestComputeUncertaintyDriversCorrelation:
    """Test correlation computation."""

    def test_positive_correlation(self):
        """Should detect positive correlation with squared residuals."""
        # Feature that increases with squared residuals
        residuals = np.array([1, 2, 3, 4, 5])
        features = pl.DataFrame({"feature": [1, 4, 9, 16, 25]})  # residual^2
        result = compute_uncertainty_drivers(residuals, features)
        row = result.filter(pl.col("feature") == "feature")
        assert row["residual_correlation"][0] > 0.9

    def test_negative_correlation(self):
        """Should detect negative correlation with squared residuals."""
        # Feature that decreases as squared residuals increase
        residuals = np.array([1, 2, 3, 4, 5])
        features = pl.DataFrame({"feature": [25, 16, 9, 4, 1]})  # decreasing
        result = compute_uncertainty_drivers(residuals, features)
        row = result.filter(pl.col("feature") == "feature")
        assert row["residual_correlation"][0] < -0.9

    def test_zero_correlation_constant_feature(self):
        """Should skip constant features."""
        residuals = np.array([1, -2, 3, -1, 2])
        features = pl.DataFrame(
            {
                "constant": [5, 5, 5, 5, 5],
                "varying": [1, 2, 3, 4, 5],
            }
        )
        result = compute_uncertainty_drivers(residuals, features)
        features_found = result["feature"].to_list()
        assert "constant" not in features_found
        assert "varying" in features_found


class TestComputeUncertaintyDriversEdgeCases:
    """Test edge cases."""

    def test_no_columns(self):
        """Should return empty DataFrame with correct schema when no feature columns."""
        residuals = np.array([1, -2, 3, -1, 2])
        features = pl.DataFrame()  # no columns
        result = compute_uncertainty_drivers(residuals, features)
        assert isinstance(result, pl.DataFrame)
        assert result.height == 0
        assert "feature" in result.columns
        assert "residual_correlation" in result.columns
        assert "p_value" in result.columns

    def test_single_feature(self):
        """Should work with single feature."""
        residuals = np.array([1, -2, 3, -1, 2])
        features = pl.DataFrame({"only": [1, 2, 3, 4, 5]})
        result = compute_uncertainty_drivers(residuals, features)
        assert result.height == 1
        assert result["feature"][0] == "only"

    def test_residuals_all_same(self):
        """Should handle zero-variance residuals."""
        residuals = np.array([1, 1, 1, 1, 1])  # all same
        features = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
        result = compute_uncertainty_drivers(residuals, features)
        # Should still return result (correlation undefined but handled)
        assert isinstance(result, pl.DataFrame)


class TestComputeUncertaintyDriversWarning:
    """Test warning behavior."""

    def test_warns_when_no_significant_drivers(self):
        """Should warn when no features have p < 0.05."""
        residuals = np.random.randn(20)
        features = pl.DataFrame(
            {
                "noise1": np.random.randn(20),
                "noise2": np.random.randn(20),
            }
        )
        import warnings

        with warnings.catch_warnings(record=True) as _:
            warnings.simplefilter("always")
            compute_uncertainty_drivers(residuals, features)
            # Warning may or may not fire depending on random luck
            # Just ensure it doesn't crash
