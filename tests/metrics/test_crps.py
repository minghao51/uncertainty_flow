"""Tests for CRPS metrics."""

import warnings

import numpy as np
import pytest

from uncertainty_flow.metrics import crps_quantile, crps_score


class TestCRPSQuantile:
    def test_perfect_prediction_zero_crps(self):
        y = np.array([5.0])
        q = np.array([[5.0, 5.0, 5.0, 5.0, 5.0]])
        levels = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        assert crps_quantile(y, q, levels) == pytest.approx(0.0, abs=1e-10)

    def test_symmetric_spread_positive_crps(self):
        y = np.array([3.0])
        q = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        levels = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        assert crps_quantile(y, q, levels) > 0

    def test_batch_mean(self):
        y = np.array([3.0, 3.0])
        q = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]] * 2)
        levels = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        result = crps_quantile(y, q, levels)
        assert isinstance(result, float)
        assert result > 0

    def test_wider_spread_higher_crps(self):
        y = np.array([5.0])
        narrow = np.array([[4.0, 4.5, 5.0, 5.5, 6.0]])
        wide = np.array([[0.0, 2.5, 5.0, 7.5, 10.0]])
        levels = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        assert crps_quantile(y, narrow, levels) < crps_quantile(y, wide, levels)

    def test_requires_at_least_two_levels(self):
        y = np.array([5.0])
        q = np.array([[5.0]])
        levels = np.array([0.5])
        with pytest.raises(ValueError, match="at least 2"):
            crps_quantile(y, q, levels)

    def test_uniform_distribution_analytical(self):
        levels = np.linspace(0.01, 0.99, 99)
        q = levels.reshape(1, -1)
        y = np.array([0.5])
        result = crps_quantile(y, q, levels)
        expected = 1.0 / 12.0
        assert result == pytest.approx(expected, rel=0.02)

    def test_non_monotone_quantiles_warns(self):
        y = np.array([3.0])
        q = np.array([[5.0, 1.0, 3.0]])
        levels = np.array([0.1, 0.5, 0.9])
        with pytest.warns(UserWarning, match="non-monotone"):
            crps_quantile(y, q, levels)


class TestCRPSScoreDeprecated:
    def test_emits_future_warning(self):
        y_true = np.array([5.0])
        lower = np.array([4.0])
        upper = np.array([6.0])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            crps_score(y_true, lower, upper, confidence=0.9)
            assert len(w) == 1
            assert issubclass(w[0].category, FutureWarning)
            assert "v0.3.0" in str(w[0].message)

    def test_still_returns_valid_result(self):
        y_true = np.array([5.0, 5.0, 5.0])
        lower = np.array([4.0, 4.0, 4.0])
        upper = np.array([6.0, 6.0, 6.0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            result = crps_score(y_true, lower, upper, confidence=0.9)
        assert isinstance(result, float)
        assert result >= 0
