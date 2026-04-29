"""Tests for crps_score metric."""

import numpy as np
import polars as pl
import pytest

from uncertainty_flow.metrics import crps_score


class TestCRPSScore:
    def test_numpy_arrays(self):
        y_true = np.array([5.0, 5.0, 5.0])
        lower = np.array([4.0, 4.0, 4.0])
        upper = np.array([6.0, 6.0, 6.0])
        crps = crps_score(y_true, lower, upper, confidence=0.9)
        assert isinstance(crps, float)
        assert crps >= 0

    def test_polars_series(self):
        y_true = pl.Series([5.0, 5.0, 5.0])
        lower = pl.Series([4.0, 4.0, 4.0])
        upper = pl.Series([6.0, 6.0, 6.0])
        crps = crps_score(y_true, lower, upper, confidence=0.9)
        assert isinstance(crps, float)
        assert crps >= 0

    def test_perfect_interval_zero_crps(self):
        y_true = np.array([5.0])
        lower = np.array([5.0])
        upper = np.array([5.0])
        crps = crps_score(y_true, lower, upper, confidence=0.9)
        assert crps == pytest.approx(0.0, abs=1e-10)

    def test_wider_intervals_higher_crps(self):
        y_true = np.array([5.0, 5.0, 5.0])

        narrow_lower = np.array([4.5, 4.5, 4.5])
        narrow_upper = np.array([5.5, 5.5, 5.5])

        wide_lower = np.array([0.0, 0.0, 0.0])
        wide_upper = np.array([10.0, 10.0, 10.0])

        crps_narrow = crps_score(y_true, narrow_lower, narrow_upper, confidence=0.9)
        crps_wide = crps_score(y_true, wide_lower, wide_upper, confidence=0.9)

        assert crps_narrow < crps_wide

    def test_returns_float(self):
        y_true = np.array([1.0, 2.0, 3.0])
        lower = np.array([0.0, 1.0, 2.0])
        upper = np.array([2.0, 3.0, 4.0])
        crps = crps_score(y_true, lower, upper)
        assert isinstance(crps, float)

    def test_shifted_predictions_higher_crps(self):
        y_true = np.array([5.0, 5.0, 5.0])

        centered_lower = np.array([4.0, 4.0, 4.0])
        centered_upper = np.array([6.0, 6.0, 6.0])

        shifted_lower = np.array([10.0, 10.0, 10.0])
        shifted_upper = np.array([12.0, 12.0, 12.0])

        crps_centered = crps_score(y_true, centered_lower, centered_upper, confidence=0.9)
        crps_shifted = crps_score(y_true, shifted_lower, shifted_upper, confidence=0.9)

        assert crps_shifted > crps_centered
