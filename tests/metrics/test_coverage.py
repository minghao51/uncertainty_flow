"""Tests for coverage_score metric."""

import numpy as np
import polars as pl
import pytest

from uncertainty_flow.metrics import coverage_score


class TestCoverageScore:
    """Test coverage score calculation."""

    def test_numpy_arrays(self):
        """Should work with numpy arrays."""
        y_true = np.array([1, 2, 3, 4, 5])
        lower = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
        upper = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
        coverage = coverage_score(y_true, lower, upper)
        # All values are inside: 1 in [0.5,1.5], 2 in [1.5,2.5], etc.
        assert coverage == 1.0

    def test_polars_series(self):
        """Should work with polars Series."""
        y_true = pl.Series([1, 2, 3, 4, 5])
        lower = pl.Series([0.5, 1.5, 2.5, 3.5, 4.5])
        upper = pl.Series([1.5, 2.5, 3.5, 4.5, 5.5])
        coverage = coverage_score(y_true, lower, upper)
        assert coverage == 1.0

    def test_all_inside_interval(self):
        """Should return 1.0 when all values are inside."""
        y_true = np.array([1, 2, 3, 4, 5])
        lower = np.array([0, 0, 0, 0, 0])
        upper = np.array([10, 10, 10, 10, 10])
        coverage = coverage_score(y_true, lower, upper)
        assert coverage == 1.0

    def test_none_inside_interval(self):
        """Should return 0.0 when no values are inside."""
        y_true = np.array([5, 6, 7])
        lower = np.array([0, 0, 0])
        upper = np.array([1, 1, 1])
        coverage = coverage_score(y_true, lower, upper)
        assert coverage == 0.0

    def test_on_boundary_counts(self):
        """Values exactly on boundary should count as inside."""
        y_true = np.array([1, 2, 3])
        lower = np.array([1, 2, 3])  # Exactly equal
        upper = np.array([1, 2, 3])
        coverage = coverage_score(y_true, lower, upper)
        assert coverage == 1.0

    def test_validates_bounds(self):
        """Should raise error if lower > upper."""
        y_true = np.array([5, 5, 5])
        lower = np.array([6, 6, 6])
        upper = np.array([4, 4, 4])

        with pytest.raises(ValueError, match="lower bound must be <= upper bound"):
            coverage_score(y_true, lower, upper)

    def test_returns_float(self):
        """Should always return a float."""
        y_true = np.array([1, 2, 3])
        lower = np.array([0, 0, 0])
        upper = np.array([10, 10, 10])
        coverage = coverage_score(y_true, lower, upper)
        assert isinstance(coverage, float)
