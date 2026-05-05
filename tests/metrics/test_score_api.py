"""Tests for the unified score() metric dispatcher."""

import numpy as np
import pytest

from uncertainty_flow.core.distribution import DistributionPrediction
from uncertainty_flow.metrics import score


def _make_pred():
    q = np.array([[1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 3.0, 4.0, 5.0, 6.0]])
    return DistributionPrediction(q, [0.1, 0.25, 0.5, 0.75, 0.9], ["y"])


class TestScoreDispatcher:
    def test_score_crps(self):
        dp = _make_pred()
        result = score(dp, np.array([3.5, 4.0]), "crps")
        assert isinstance(result, float)
        assert result > 0

    def test_score_mae(self):
        dp = _make_pred()
        result = score(dp, np.array([3.5, 4.0]), "mae")
        assert isinstance(result, float)
        assert result >= 0

    def test_score_rmse(self):
        dp = _make_pred()
        result = score(dp, np.array([3.5, 4.0]), "rmse")
        assert isinstance(result, float)
        assert result >= 0

    def test_score_coverage(self):
        dp = _make_pred()
        result = score(dp, np.array([3.0, 4.0]), "coverage")
        assert isinstance(result, float)
        assert 0 <= result <= 1

    def test_score_winkler(self):
        dp = _make_pred()
        result = score(dp, np.array([3.0, 4.0]), "winkler", confidence=0.8)
        assert isinstance(result, float)
        assert result > 0

    def test_score_calibration_error(self):
        dp = _make_pred()
        result = score(dp, np.array([3.0, 4.0]), "calibration_error", confidence=0.8)
        assert isinstance(result, float)
        assert result >= 0

    def test_score_pinball(self):
        dp = _make_pred()
        result = score(dp, np.array([3.0, 4.0]), "pinball")
        assert isinstance(result, float)
        assert result >= 0

    def test_score_callable(self):
        dp = _make_pred()
        result = score(dp, np.array([3.0, 4.0]), lambda p, y: 42.0)
        assert result == 42.0

    def test_score_unknown_raises(self):
        dp = _make_pred()
        with pytest.raises(ValueError, match="Unknown metric"):
            score(dp, np.array([3.0, 4.0]), "invalid")

    def test_score_matches_direct_coverage(self):
        from uncertainty_flow.metrics import coverage_score

        dp = _make_pred()
        y = np.array([3.0, 4.0])
        via_score = score(dp, y, "coverage")
        interval = dp.interval(0.9)
        direct = coverage_score(y, interval["lower"].to_numpy(), interval["upper"].to_numpy())
        assert via_score == pytest.approx(direct)


class TestScoreDispatcherMultivariate:
    def _make_multi_pred(self):
        matrix = np.array([[1, 2, 3, 4, 5, 10, 11, 12, 13, 14]])
        return DistributionPrediction(matrix, [0.1, 0.25, 0.5, 0.75, 0.9], ["a", "b"])

    def test_multivariate_coverage_returns_dict(self):
        dp = self._make_multi_pred()
        y = np.array([[3.0, 12.0]])
        result = score(dp, y, "coverage")
        assert isinstance(result, dict)
        assert "a" in result and "b" in result

    def test_multivariate_mae_returns_dict(self):
        dp = self._make_multi_pred()
        y = np.array([[3.0, 12.0]])
        result = score(dp, y, "mae")
        assert isinstance(result, dict)
        assert all(isinstance(v, float) for v in result.values())

    def test_multivariate_rmse_returns_dict(self):
        dp = self._make_multi_pred()
        y = np.array([[3.0, 12.0]])
        result = score(dp, y, "rmse")
        assert isinstance(result, dict)

    def test_multivariate_winkler_returns_dict(self):
        dp = self._make_multi_pred()
        y = np.array([[3.0, 12.0]])
        result = score(dp, y, "winkler", confidence=0.8)
        assert isinstance(result, dict)

    def test_multivariate_pinball_returns_dict(self):
        dp = self._make_multi_pred()
        y = np.array([[3.0, 12.0]])
        result = score(dp, y, "pinball")
        assert isinstance(result, dict)

    def test_multivariate_calibration_error_returns_dict(self):
        dp = self._make_multi_pred()
        y = np.array([[3.0, 12.0]])
        result = score(dp, y, "calibration_error", confidence=0.8)
        assert isinstance(result, dict)

    def test_multivariate_crps_returns_dict(self):
        dp = self._make_multi_pred()
        y = np.array([[3.0, 12.0]])
        result = score(dp, y, "crps")
        assert isinstance(result, dict)
        assert "a" in result and "b" in result
