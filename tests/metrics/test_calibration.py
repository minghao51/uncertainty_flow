"""Tests for calibration_error metric."""

import numpy as np
import polars as pl
import pytest

from uncertainty_flow.metrics import calibration_error


class TestCalibrationError:
    def test_numpy_arrays(self):
        y_true = np.array([5.0, 5.0, 5.0])
        lower = np.array([4.0, 4.0, 4.0])
        upper = np.array([6.0, 6.0, 6.0])
        error = calibration_error(y_true, lower, upper, nominal_coverage=0.9)
        assert isinstance(error, float)
        assert error >= 0

    def test_polars_series(self):
        y_true = pl.Series([5.0, 5.0, 5.0])
        lower = pl.Series([4.0, 4.0, 4.0])
        upper = pl.Series([6.0, 6.0, 6.0])
        error = calibration_error(y_true, lower, upper, nominal_coverage=0.9)
        assert isinstance(error, float)

    def test_perfect_calibration(self):
        y_true = np.array([5.0, 5.0, 5.0, 5.0, 5.0,
                           5.0, 5.0, 5.0, 5.0, 5.0])
        lower = np.array([4.0] * 9 + [6.0])
        upper = np.array([6.0] * 9 + [8.0])
        error = calibration_error(y_true, lower, upper, nominal_coverage=0.9)
        assert error == pytest.approx(0.0, abs=0.01)

    def test_worst_case_no_coverage(self):
        y_true = np.array([10.0, 10.0, 10.0])
        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([1.0, 1.0, 1.0])
        error = calibration_error(y_true, lower, upper, nominal_coverage=0.9)
        assert error == pytest.approx(0.9)

    def test_worst_case_full_coverage(self):
        y_true = np.array([5.0, 5.0, 5.0])
        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([10.0, 10.0, 10.0])
        error = calibration_error(y_true, lower, upper, nominal_coverage=0.5)
        assert error == pytest.approx(0.5)

    def test_returns_float(self):
        y_true = np.array([1.0, 2.0, 3.0])
        lower = np.array([0.0, 1.0, 2.0])
        upper = np.array([2.0, 3.0, 4.0])
        error = calibration_error(y_true, lower, upper, nominal_coverage=0.9)
        assert isinstance(error, float)

    def test_boundary_values_count_as_inside(self):
        y_true = np.array([4.0, 6.0])
        lower = np.array([4.0, 4.0])
        upper = np.array([6.0, 6.0])
        error = calibration_error(y_true, lower, upper, nominal_coverage=1.0)
        assert error == pytest.approx(0.0)
