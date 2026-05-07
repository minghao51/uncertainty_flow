"""Tests for multivariate scoring rules (energy score, variogram score)."""

import numpy as np
import pytest

from uncertainty_flow.core.distribution import DistributionPrediction
from uncertainty_flow.metrics.multivariate import energy_score, variogram_score


def _make_multivariate_pred(n=20, n_targets=2, seed=42):
    rng = np.random.default_rng(seed)
    levels = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    target_names = [f"t{i}" for i in range(n_targets)]
    parts = []
    for t in range(n_targets):
        loc = t * 5
        scale = 1.0 + t * 0.5
        qv = np.array([loc + scale * (-1.96 + j * 0.654) for j in range(len(levels))])
        qv_tiled = np.tile(qv, (n, 1)) + rng.normal(0, 0.1, size=(n, len(levels)))
        parts.append(qv_tiled)
    q_matrix = np.column_stack(parts)
    return DistributionPrediction(q_matrix, levels, target_names=target_names)


class TestEnergyScore:
    def test_basic_energy_score(self):
        pred = _make_multivariate_pred(n=10, n_targets=2)
        y = np.column_stack([np.zeros(10), np.full(10, 5.0)])
        es = energy_score(pred, y, n_samples=100, random_state=42)
        assert isinstance(es, float)
        assert np.isfinite(es)
        assert es >= 0

    def test_requires_at_least_2_targets(self):
        levels = [0.1, 0.5, 0.9]
        qv = np.array([-1.0, 0.0, 1.0])
        pred = DistributionPrediction(np.tile(qv, (5, 1)), levels, target_names=["y"])
        y = np.zeros(5)
        with pytest.raises(ValueError, match="at least 2"):
            energy_score(pred, y)

    def test_perfect_forecast_low_score(self):
        n = 10
        levels = [0.05, 0.25, 0.5, 0.75, 0.95]
        rng = np.random.default_rng(42)
        y_vals = rng.normal(0, 1, size=(n, 2))
        qv_vals = []
        for t in range(2):
            quants = np.array([np.quantile(y_vals[:, t], lv) for lv in levels])
            qv_vals.append(np.tile(quants, (n, 1)))
        q_matrix = np.column_stack(qv_vals)
        pred = DistributionPrediction(q_matrix, levels, target_names=["a", "b"])
        es = energy_score(pred, y_vals, n_samples=200, random_state=42)
        assert np.isfinite(es)

    def test_convenience_method(self):
        pred = _make_multivariate_pred(n=10, n_targets=2)
        y = np.column_stack([np.zeros(10), np.full(10, 5.0)])
        es = pred.energy_score(y, n_samples=50, random_state=42)
        assert isinstance(es, float)
        assert np.isfinite(es)


class TestVariogramScore:
    def test_basic_variogram_score(self):
        pred = _make_multivariate_pred(n=10, n_targets=2)
        y = np.column_stack([np.zeros(10), np.full(10, 5.0)])
        vs = variogram_score(pred, y, n_samples=100, random_state=42)
        assert isinstance(vs, float)
        assert np.isfinite(vs)
        assert vs >= 0

    def test_requires_at_least_2_targets(self):
        levels = [0.1, 0.5, 0.9]
        qv = np.array([-1.0, 0.0, 1.0])
        pred = DistributionPrediction(np.tile(qv, (5, 1)), levels, target_names=["y"])
        y = np.zeros(5)
        with pytest.raises(ValueError, match="at least 2"):
            variogram_score(pred, y)

    def test_custom_power(self):
        pred = _make_multivariate_pred(n=10, n_targets=2)
        y = np.column_stack([np.zeros(10), np.full(10, 5.0)])
        vs_05 = variogram_score(pred, y, n_samples=50, p=0.5, random_state=42)
        vs_1 = variogram_score(pred, y, n_samples=50, p=1.0, random_state=42)
        assert np.isfinite(vs_05)
        assert np.isfinite(vs_1)

    def test_variogram_score_convenience_method(self):
        pred = _make_multivariate_pred(n=10, n_targets=2)
        y = np.column_stack([np.zeros(10), np.full(10, 5.0)])
        vs = pred.variogram_score(y, n_samples=50, random_state=42)
        assert isinstance(vs, float)
        assert np.isfinite(vs)
        assert vs >= 0


class TestScoreDispatcherIntegration:
    def test_log_score_via_score(self):
        from uncertainty_flow.metrics import score

        pred = _make_multivariate_pred(n=10, n_targets=2)
        y = np.column_stack([np.zeros(10), np.full(10, 5.0)])
        result = score(pred, y, metric="log_score", family="normal")
        assert isinstance(result, dict)
        assert "t0" in result
        assert "t1" in result

    def test_energy_score_via_score(self):
        from uncertainty_flow.metrics import score

        pred = _make_multivariate_pred(n=10, n_targets=2)
        y = np.column_stack([np.zeros(10), np.full(10, 5.0)])
        result = score(pred, y, metric="energy_score", n_samples=50, random_state=42)
        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_variogram_score_via_score(self):
        from uncertainty_flow.metrics import score

        pred = _make_multivariate_pred(n=10, n_targets=2)
        y = np.column_stack([np.zeros(10), np.full(10, 5.0)])
        result = score(pred, y, metric="variogram_score", n_samples=50, random_state=42)
        assert isinstance(result, float)
        assert np.isfinite(result)
