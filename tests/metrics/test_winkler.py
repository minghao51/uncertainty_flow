"""Tests for winkler_score metric."""

import numpy as np
import polars as pl
import pytest

from uncertainty_flow.metrics import winkler_score


class TestWinklerScore:
    """Test Winkler score calculation."""

    def test_numpy_arrays(self):
        """Should work with numpy arrays."""
        y_true = np.array([5, 5, 5])
        lower = np.array([4, 4, 4])
        upper = np.array([6, 6, 6])
        score = winkler_score(y_true, lower, upper, 0.9)
        # All within interval, only width penalty
        assert score == 2.0  # Average width = 2

    def test_polars_series(self):
        """Should work with polars Series."""
        y_true = pl.Series([5, 5, 5])
        lower = pl.Series([4, 4, 4])
        upper = pl.Series([6, 6, 6])
        score = winkler_score(y_true, lower, upper, 0.9)
        assert score == 2.0

    def test_miss_penalty(self):
        """Should penalize values outside interval."""
        y_true = np.array([7, 5, 3])  # One above, one inside, one below
        lower = np.array([4, 4, 4])
        upper = np.array([6, 6, 6])
        score = winkler_score(y_true, lower, upper, 0.9)

        # Score should be higher than perfect case due to misses
        assert score > 2.0

    def test_wider_intervals_higher_penalty(self):
        """Wider intervals should have higher penalty."""
        y_true = np.array([5, 5, 5])

        narrow_lower = np.array([4.5, 4.5, 4.5])
        narrow_upper = np.array([5.5, 5.5, 5.5])

        wide_lower = np.array([3, 3, 3])
        wide_upper = np.array([7, 7, 7])

        score_narrow = winkler_score(y_true, narrow_lower, narrow_upper, 0.9)
        score_wide = winkler_score(y_true, wide_lower, wide_upper, 0.9)

        assert score_narrow < score_wide

    def test_validates_confidence(self):
        """Should raise error for invalid confidence."""
        y_true = np.array([1, 2, 3])
        lower = np.array([0, 1, 2])
        upper = np.array([2, 3, 4])

        with pytest.raises(ValueError, match="confidence must be in \\(0, 1\\)"):
            winkler_score(y_true, lower, upper, 1.5)

    def test_validates_bounds(self):
        """Should raise error if lower > upper."""
        y_true = np.array([5, 5, 5])
        lower = np.array([6, 6, 6])  # Higher than upper
        upper = np.array([4, 4, 4])

        with pytest.raises(ValueError, match="lower bound must be <= upper bound"):
            winkler_score(y_true, lower, upper, 0.9)

    def test_all_inside_interval(self):
        """When all values are inside, only width penalty applies."""
        y_true = np.array([5, 5, 5])
        lower = np.array([4, 4, 4])
        upper = np.array([6, 6, 6])
        score = winkler_score(y_true, lower, upper, 0.9)
        assert score == 2.0  # Width = 6-4 = 2
