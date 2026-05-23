"""Tests for counterfactual search bug fixes: L1/L2 proximals, param capping."""

import numpy as np
import polars as pl
import pytest

from uncertainty_flow.counterfactual.search import EvolutionarySearch, GradientSearch, SearchResult
from uncertainty_flow.models import QuantileForestForecaster


@pytest.fixture
def _fitted_forecaster():
    rng = np.random.default_rng(42)
    n = 200
    df = pl.DataFrame(
        {
            "x1": rng.standard_normal(n),
            "x2": rng.standard_normal(n),
            "x3": rng.standard_normal(n),
            "y": 3 * rng.standard_normal(n) + 5,
        }
    )
    model = QuantileForestForecaster(
        targets="y", horizon=1, n_estimators=10, min_samples_leaf=5, random_state=42
    )
    model.fit(df)
    return model


@pytest.fixture
def _single_row():
    return pl.DataFrame({"x1": [0.5], "x2": [-0.3], "x3": [1.2]})


class TestEvolutionarySearchNoCap:
    def test_n_generations_above_old_cap(self, _fitted_forecaster, _single_row):
        searcher = EvolutionarySearch(_fitted_forecaster, n_generations=50, random_state=42)
        assert searcher.n_generations == 50
        assert not hasattr(searcher, "_max_effective_generations")

    def test_evolutionary_search_uses_full_generations(self, _fitted_forecaster, _single_row):
        searcher = EvolutionarySearch(_fitted_forecaster, n_generations=30, random_state=42)
        result = searcher.search(
            _single_row,
            target_reduction=0.2,
            feature_bounds={"x1": (-5, 5), "x2": (-5, 5), "x3": (-5, 5)},
        )
        assert isinstance(result, SearchResult)


class TestGradientSearchNoCap:
    def test_n_iterations_above_old_cap(self, _fitted_forecaster):
        searcher = GradientSearch(_fitted_forecaster, n_iterations=200, random_state=42)
        assert searcher.n_iterations == 200
        assert not hasattr(searcher, "_max_effective_iterations")

    def test_gradient_search_uses_full_iterations(self, _fitted_forecaster, _single_row):
        searcher = GradientSearch(_fitted_forecaster, n_iterations=150, random_state=42)
        result = searcher.search(
            _single_row,
            target_reduction=0.2,
            feature_bounds={"x1": (-5, 5), "x2": (-5, 5), "x3": (-5, 5)},
        )
        assert isinstance(result, SearchResult)


class TestL1L2ProximalFormulas:
    """Specification tests documenting the correct L1/L2 proximal formulas."""

    def test_l1_soft_thresholding_zero_small_change(self):
        change = np.array([0.001, -0.001, 0.5, -0.5])
        lr = 0.01
        l1 = 0.1

        result = np.sign(change) * np.maximum(np.abs(change) - lr * l1, 0.0)

        assert result[0] == 0.0
        assert result[1] == 0.0
        assert result[2] == pytest.approx(0.499)
        assert result[3] == pytest.approx(-0.499)

    def test_l2_proximal_shrinkage(self):
        change = np.array([1.0, -2.0, 0.5])
        lr = 0.01
        l2 = 0.1

        result = change / (1 + lr * l2)

        expected = np.array([1.0 / 1.001, -2.0 / 1.001, 0.5 / 1.001])
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_l1_preserves_sign(self):
        change = np.array([3.0, -3.0, 0.5, -0.5])
        lr = 0.01
        l1 = 0.01

        result = np.sign(change) * np.maximum(np.abs(change) - lr * l1, 0.0)

        assert np.sign(result[0]) == 1.0
        assert np.sign(result[1]) == -1.0
        assert result[2] > 0
        assert result[3] < 0

    def test_l2_shrinks_but_does_not_flip_sign(self):
        change = np.array([1.0, -1.0])
        lr = 0.01
        l2 = 100.0

        result = change / (1 + lr * l2)

        assert np.sign(result[0]) == 1.0
        assert np.sign(result[1]) == -1.0
        assert abs(result[0]) < abs(change[0])

    def test_old_l1_formula_wrong_for_small_change(self):
        change = 0.05
        l1 = 0.1

        old_result = change
        if abs(old_result) < l1:
            old_result = 0
        else:
            old_result -= l1 * np.sign(old_result)

        lr = 0.01
        new_result = np.sign(change) * max(abs(change) - lr * l1, 0.0)

        assert old_result == 0.0
        assert new_result > 0.0

    def test_old_l2_formula_can_flip_sign(self):
        change = np.array([0.5])
        l2 = 2.0

        old_result = change * (1 - l2)

        lr = 0.01
        new_result = change / (1 + lr * l2)

        assert old_result[0] < 0
        assert new_result[0] > 0


class TestProximalEndToEnd:
    """End-to-end tests verifying GradientSearch applies L1/L2 penalties correctly."""

    def test_high_l1_produces_sparser_changes(self, _fitted_forecaster, _single_row):
        bounds = {"x1": (-5, 5), "x2": (-5, 5), "x3": (-5, 5)}
        searcher = GradientSearch(
            _fitted_forecaster,
            n_iterations=50,
            l1_penalty=10.0,
            l2_penalty=0.0,
            learning_rate=0.01,
            random_state=42,
        )
        result = searcher.search(
            _single_row,
            target_reduction=0.1,
            feature_bounds=bounds,
        )
        n_near_zero = sum(1 for v in result.changes.values() if abs(v) < 1e-6)
        assert n_near_zero >= 1

    def test_high_l2_produces_smaller_changes(self, _fitted_forecaster, _single_row):
        bounds = {"x1": (-5, 5), "x2": (-5, 5), "x3": (-5, 5)}
        no_l2 = GradientSearch(
            _fitted_forecaster,
            n_iterations=50,
            l1_penalty=0.0,
            l2_penalty=0.0,
            learning_rate=0.01,
            random_state=42,
        )
        high_l2 = GradientSearch(
            _fitted_forecaster,
            n_iterations=50,
            l1_penalty=0.0,
            l2_penalty=100.0,
            learning_rate=0.01,
            random_state=42,
        )
        result_no_l2 = no_l2.search(
            _single_row,
            target_reduction=0.1,
            feature_bounds=bounds,
        )
        result_high_l2 = high_l2.search(
            _single_row,
            target_reduction=0.1,
            feature_bounds=bounds,
        )
        mag_no_l2 = sum(abs(v) for v in result_no_l2.changes.values())
        mag_high_l2 = sum(abs(v) for v in result_high_l2.changes.values())
        assert mag_high_l2 <= mag_no_l2

    def test_changes_preserve_sign(self, _fitted_forecaster, _single_row):
        bounds = {"x1": (-5, 5), "x2": (-5, 5), "x3": (-5, 5)}
        searcher = GradientSearch(
            _fitted_forecaster,
            n_iterations=50,
            l1_penalty=0.1,
            l2_penalty=10.0,
            learning_rate=0.01,
            random_state=42,
        )
        result = searcher.search(
            _single_row,
            target_reduction=0.1,
            feature_bounds=bounds,
        )
        for feature, change in result.changes.items():
            orig = _single_row[feature][0]
            cf = result.counterfactual[feature][0]
            if abs(change) > 1e-8:
                assert np.sign(cf - orig) == np.sign(change)
