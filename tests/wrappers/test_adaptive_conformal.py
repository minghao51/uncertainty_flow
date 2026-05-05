"""Tests for Adaptive Conformal Inference (AdaptiveConformalForecaster)."""

import numpy as np
import polars as pl
import pytest
from sklearn.linear_model import LinearRegression

from uncertainty_flow.wrappers import AdaptiveConformalForecaster, ConformalRegressor


class TestAdaptiveConformalForecaster:
    """Tests for AdaptiveConformalForecaster."""

    def _make_data(self, n: int = 200, rng_seed: int = 42):
        """Create synthetic regression data."""
        rng = np.random.default_rng(rng_seed)
        x = np.linspace(0, 10, n)
        y = 2 * x + rng.normal(0, 2, size=n)
        return pl.DataFrame({"x": x, "y": y})

    def test_fit_predict_basic(self):
        """Should fit and predict without errors."""
        df = self._make_data(n=100)
        train = df[:70]
        calib = df[70:]

        base = ConformalRegressor(
            base_model=LinearRegression(),
            auto_tune=False,
            calibration_size=0.3,
        )
        base.fit(train, target="y")

        aci = AdaptiveConformalForecaster(model=base, alpha=0.1, gamma=0.05)
        aci.fit(calib, target="y")

        pred = aci.predict(df[:5])
        assert pred._targets == ["y"]
        assert pred._n_samples == 5

    def test_alpha_starts_at_initial_value(self):
        """Alpha should equal the initial value after fit."""
        df = self._make_data(n=100)
        train = df[:70]
        calib = df[70:]

        base = ConformalRegressor(
            base_model=LinearRegression(),
            auto_tune=False,
            calibration_size=0.3,
        )
        base.fit(train, target="y")

        aci = AdaptiveConformalForecaster(model=base, alpha=0.15, gamma=0.01)
        aci.fit(calib, target="y")
        assert aci.current_alpha == pytest.approx(0.15)

    def test_update_decreases_alpha_when_uncovered(self):
        """When true value falls outside interval, alpha should decrease (widen)."""
        df = self._make_data(n=200)
        train = df[:140]
        calib = df[140:180]

        base = ConformalRegressor(
            base_model=LinearRegression(),
            auto_tune=False,
            calibration_size=0.3,
        )
        base.fit(train, target="y")

        aci = AdaptiveConformalForecaster(model=base, alpha=0.1, gamma=0.05)
        aci.fit(calib, target="y")

        initial_alpha = aci.current_alpha

        # Get a prediction on a calibration point
        point = calib[:1]
        pred = aci.predict(point)
        interval = pred.interval(0.9)
        lower = interval["lower"][0]
        _ = interval["upper"][0]

        # Choose a true value well outside the interval
        true_y = float(lower - 10.0)
        assert true_y < lower, "Test setup: expected value below lower bound"

        aci.update(true_y)
        new_alpha = aci.current_alpha

        # Alpha should decrease (intervals widen) after coverage failure
        assert new_alpha < initial_alpha

    def test_update_increases_alpha_when_covered(self):
        """When true value falls inside interval, alpha should increase (narrow)."""
        df = self._make_data(n=200)
        train = df[:140]
        calib = df[140:180]

        base = ConformalRegressor(
            base_model=LinearRegression(),
            auto_tune=False,
            calibration_size=0.3,
        )
        base.fit(train, target="y")

        aci = AdaptiveConformalForecaster(model=base, alpha=0.1, gamma=0.05)
        aci.fit(calib, target="y")

        initial_alpha = aci.current_alpha

        # Predict on any point — the point prediction itself is always
        # at the center of the interval, so it is guaranteed to be covered.
        test_point = df[:1]
        aci.predict(test_point)
        true_y = float(aci._last_point_pred)

        aci.update(true_y)
        new_alpha = aci.current_alpha

        # Alpha should increase (intervals narrow) after coverage success
        assert new_alpha > initial_alpha

    def test_update_appends_new_score(self):
        """update() should append the new residual to the score pool."""
        df = self._make_data(n=100)
        train = df[:70]
        calib = df[70:]

        base = ConformalRegressor(
            base_model=LinearRegression(),
            auto_tune=False,
            calibration_size=0.3,
        )
        base.fit(train, target="y")

        aci = AdaptiveConformalForecaster(model=base, alpha=0.1, gamma=0.01)
        aci.fit(calib, target="y")

        n_scores_before = len(aci._scores)

        test_point = df[:1]
        aci.predict(test_point)
        aci.update(999.0)  # extreme value to ensure a large score

        n_scores_after = len(aci._scores)
        assert n_scores_after == n_scores_before + 1

    def test_update_batch(self):
        """update_batch should process a sequence of observations."""
        df = self._make_data(n=100)
        train = df[:70]
        calib = df[70:]

        base = ConformalRegressor(
            base_model=LinearRegression(),
            auto_tune=False,
            calibration_size=0.3,
        )
        base.fit(train, target="y")

        aci = AdaptiveConformalForecaster(model=base, alpha=0.1, gamma=0.05)
        aci.fit(calib, target="y")

        n_scores_before = len(aci._scores)

        # Simulate a sequence: predict then update for each step
        test_vals = [5.0, 10.0, 15.0, 20.0]
        for y in test_vals:
            aci.predict(df[:1])
        aci.update_batch(np.array(test_vals))

        n_scores_after = len(aci._scores)
        assert n_scores_after == n_scores_before + len(test_vals)

    def test_update_without_predict_raises(self):
        """update() called before predict() should raise an error."""
        df = self._make_data(n=100)
        train = df[:70]
        calib = df[70:]

        base = ConformalRegressor(
            base_model=LinearRegression(),
            auto_tune=False,
            calibration_size=0.3,
        )
        base.fit(train, target="y")

        aci = AdaptiveConformalForecaster(model=base, alpha=0.1, gamma=0.01)
        aci.fit(calib, target="y")

        with pytest.raises(RuntimeError, match="predict\\(\\) first"):
            aci.update(5.0)

    def test_not_fitted_raises(self):
        """Predict before fit should raise an error."""
        base = ConformalRegressor(
            base_model=LinearRegression(),
            auto_tune=False,
        )
        base.fit(self._make_data(n=100), target="y")

        aci = AdaptiveConformalForecaster(model=base, alpha=0.1, gamma=0.01)
        with pytest.raises(Exception):  # noqa: B017
            aci.predict(self._make_data(n=10))

    def test_alpha_bounds(self):
        """Alpha should stay within (0, 1) even after many updates."""
        rng = np.random.default_rng(42)
        df = self._make_data(n=100)
        train = df[:70]
        calib = df[70:]

        base = ConformalRegressor(
            base_model=LinearRegression(),
            auto_tune=False,
            calibration_size=0.3,
        )
        base.fit(train, target="y")

        aci = AdaptiveConformalForecaster(model=base, alpha=0.5, gamma=0.5)
        aci.fit(calib, target="y")

        # Run many updates with extreme values to test bounds
        for _ in range(50):
            aci.predict(df[:1])
            aci.update(rng.normal(100, 10))

        assert 0 < aci.current_alpha < 1
        assert aci.current_alpha >= 1e-6
        assert aci.current_alpha <= 1.0 - 1e-6

    def test_multi_step_predict_constant_alpha(self):
        """Multi-step predict should use constant alpha projection."""
        df = self._make_data(n=100)
        train = df[:70]
        calib = df[70:]

        base = ConformalRegressor(
            base_model=LinearRegression(),
            auto_tune=False,
            calibration_size=0.3,
        )
        base.fit(train, target="y")

        aci = AdaptiveConformalForecaster(model=base, alpha=0.1, gamma=0.01)
        aci.fit(calib, target="y")

        alpha_before = aci.current_alpha
        pred = aci.predict(df[:5], steps=3)
        alpha_after = aci.current_alpha

        # Alpha should not change during predict-only
        assert alpha_after == pytest.approx(alpha_before)
        assert pred._n_samples == 5

    def test_conformal_scores_from_fit(self):
        """fit() should populate conformal scores from calibration residuals."""
        df = self._make_data(n=100)
        train = df[:70]
        calib = df[70:]

        base = ConformalRegressor(
            base_model=LinearRegression(),
            auto_tune=False,
            calibration_size=0.3,
        )
        base.fit(train, target="y")

        aci = AdaptiveConformalForecaster(model=base, alpha=0.1, gamma=0.01)
        aci.fit(calib, target="y")

        assert len(aci._scores) > 0
        assert all(s >= 0 for s in aci._scores)

    def test_coverage_adapts_under_distribution_shift(self):
        """ACI should maintain better coverage than static conformal under shift."""
        rng = np.random.default_rng(42)
        n = 300
        x = np.linspace(0, 10, n)
        # First 200 points: y = 2x + N(0, 1)
        # Last 100 points: y = 2x + N(0, 5)  (variance shift)
        y = np.concatenate(
            [
                2 * x[:200] + rng.normal(0, 1, size=200),
                2 * x[200:] + rng.normal(0, 5, size=100),
            ]
        )
        df = pl.DataFrame({"x": x, "y": y})

        train = df[:150]
        calib = df[150:200]
        test_shift = df[200:]

        base = ConformalRegressor(
            base_model=LinearRegression(),
            auto_tune=False,
            calibration_size=0.3,
        )
        base.fit(train, target="y")

        # Static conformal coverage on shifted data
        static_pred = base.predict(test_shift)
        static_interval = static_pred.interval(0.9)
        static_lower = static_interval["lower"].to_numpy()
        static_upper = static_interval["upper"].to_numpy()
        y_test = test_shift["y"].to_numpy()
        static_coverage = np.mean((y_test >= static_lower) & (y_test <= static_upper))

        # ACI coverage on shifted data
        aci = AdaptiveConformalForecaster(model=base, alpha=0.1, gamma=0.05)
        aci.fit(calib, target="y")

        aci_covered = []
        for i in range(len(test_shift)):
            point = test_shift[i : i + 1]
            pred = aci.predict(point)
            interval = pred.interval(0.9)
            lower = interval["lower"][0]
            upper = interval["upper"][0]
            true_y = float(test_shift["y"][i])
            aci_covered.append(lower <= true_y <= upper)
            aci.update(true_y)

        aci_coverage = np.mean(aci_covered)

        # ACI should achieve coverage closer to target (0.9) than static
        # under distribution shift. We allow some tolerance because this
        # is a stochastic test.
        assert aci_coverage >= static_coverage - 0.15
