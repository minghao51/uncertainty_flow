"""Tests for mae_score and rmse_score metrics."""

import numpy as np
import polars as pl
import pytest

from uncertainty_flow.metrics import mae_score, rmse_score


class TestMAEScore:
    def test_numpy_arrays(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.5, 2.5, 2.5, 4.5, 4.5])
        mae = mae_score(y_true, y_pred)
        assert mae == pytest.approx(0.5)

    def test_polars_series(self):
        y_true = pl.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = pl.Series([1.5, 2.5, 2.5, 4.5, 4.5])
        mae = mae_score(y_true, y_pred)
        assert mae == pytest.approx(0.5)

    def test_perfect_prediction_zero_error(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        mae = mae_score(y_true, y_pred)
        assert mae == 0.0

    def test_known_value(self):
        y_true = np.array([3.0, -0.5, 2.0, 7.0])
        y_pred = np.array([2.5, 0.0, 2.0, 8.0])
        expected = np.mean(np.abs(y_true - y_pred))
        mae = mae_score(y_true, y_pred)
        assert mae == pytest.approx(expected)

    def test_returns_float(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        mae = mae_score(y_true, y_pred)
        assert isinstance(mae, float)

    def test_non_negative(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([10.0, 20.0, 30.0])
        mae = mae_score(y_true, y_pred)
        assert mae >= 0


class TestRMSEScore:
    def test_numpy_arrays(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        rmse = rmse_score(y_true, y_pred)
        assert rmse == 0.0

    def test_polars_series(self):
        y_true = pl.Series([1.0, 2.0, 3.0])
        y_pred = pl.Series([1.0, 2.0, 3.0])
        rmse = rmse_score(y_true, y_pred)
        assert rmse == 0.0

    def test_known_value(self):
        y_true = np.array([3.0, -0.5, 2.0, 7.0])
        y_pred = np.array([2.5, 0.0, 2.0, 8.0])
        expected = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        rmse = rmse_score(y_true, y_pred)
        assert rmse == pytest.approx(expected)

    def test_rmse_geq_mae(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([0.0, 3.0, 2.0, 5.0])
        mae = mae_score(y_true, y_pred)
        rmse = rmse_score(y_true, y_pred)
        assert rmse >= mae

    def test_returns_float(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 3.5])
        rmse = rmse_score(y_true, y_pred)
        assert isinstance(rmse, float)

    def test_non_negative(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([10.0, 20.0, 30.0])
        rmse = rmse_score(y_true, y_pred)
        assert rmse >= 0
