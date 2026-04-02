"""Tests for BayesianQuantileRegressor."""

from importlib.util import find_spec

import numpy as np
import polars as pl
import pytest

NUMPYRO_AVAILABLE = find_spec("numpyro") is not None
pytestmark = pytest.mark.skipif(not NUMPYRO_AVAILABLE, reason="NumPyro not installed")


@pytest.fixture
def regression_data():
    """Create a simple regression dataset: 50 samples, 2 features."""
    rng = np.random.default_rng(42)
    n = 50
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    y = 2 * x1 + 5 * x2 + rng.standard_normal(n) * 0.5
    return pl.DataFrame({"x1": x1, "x2": x2, "y": y})


class TestBayesianQuantileRegressorInit:
    """Test BayesianQuantileRegressor initialization."""

    def test_init_defaults(self):
        """Should initialize with default parameters."""
        from uncertainty_flow.bayesian import BayesianQuantileRegressor

        model = BayesianQuantileRegressor()
        assert model.quantiles == [0.1, 0.5, 0.9]
        assert model.n_warmup == 500
        assert model.n_samples == 1000
        assert model.kernel == "nuts"
        assert model.prior_width == 1.0
        assert model.random_state is None
        assert model._fitted is False

    def test_init_custom_params(self):
        """Should initialize with custom parameters."""
        from uncertainty_flow.bayesian import BayesianQuantileRegressor

        model = BayesianQuantileRegressor(
            quantiles=[0.05, 0.5, 0.95],
            n_warmup=100,
            n_samples=200,
            kernel="nuts",
            prior_width=2.0,
            random_state=123,
        )
        assert model.quantiles == [0.05, 0.5, 0.95]
        assert model.n_warmup == 100
        assert model.n_samples == 200
        assert model.prior_width == 2.0
        assert model.random_state == 123
        assert model._fitted is False

    def test_is_base_uncertainty_model(self):
        """Should inherit from BaseUncertaintyModel."""
        from uncertainty_flow.bayesian import BayesianQuantileRegressor
        from uncertainty_flow.core.base import BaseUncertaintyModel

        assert issubclass(BayesianQuantileRegressor, BaseUncertaintyModel)


class TestBayesianQuantileRegressorFit:
    """Test BayesianQuantileRegressor fit method."""

    def test_fit_returns_self(self, regression_data):
        """Should return self after fitting."""
        from uncertainty_flow.bayesian import BayesianQuantileRegressor

        model = BayesianQuantileRegressor(n_warmup=10, n_samples=10, random_state=0)
        result = model.fit(regression_data, target="y")
        assert result is model

    def test_fit_sets_fitted_flag(self, regression_data):
        """Should set _fitted to True after fitting."""
        from uncertainty_flow.bayesian import BayesianQuantileRegressor

        model = BayesianQuantileRegressor(n_warmup=10, n_samples=10, random_state=0)
        assert model._fitted is False
        model.fit(regression_data, target="y")
        assert model._fitted is True

    def test_fit_stores_feature_cols(self, regression_data):
        """Should store feature column names."""
        from uncertainty_flow.bayesian import BayesianQuantileRegressor

        model = BayesianQuantileRegressor(n_warmup=10, n_samples=10, random_state=0)
        model.fit(regression_data, target="y")
        assert model._feature_cols_ == ["x1", "x2"]
        assert model._target_col_ == "y"

    def test_fit_stores_posterior(self, regression_data):
        """Should store posterior samples after fitting."""
        from uncertainty_flow.bayesian import BayesianQuantileRegressor

        model = BayesianQuantileRegressor(n_warmup=10, n_samples=10, random_state=0)
        model.fit(regression_data, target="y")
        assert model._posterior_samples_ is not None
        assert model._posterior_samples_.shape[0] == 10  # n_samples

    def test_fit_with_lazyframe(self, regression_data):
        """Should work with LazyFrame input."""
        from uncertainty_flow.bayesian import BayesianQuantileRegressor

        model = BayesianQuantileRegressor(n_warmup=10, n_samples=10, random_state=0)
        lf = regression_data.lazy()
        model.fit(lf, target="y")
        assert model._fitted is True

    def test_fit_requires_target(self, regression_data):
        """Should raise ValueError if target is not specified."""
        from uncertainty_flow.bayesian import BayesianQuantileRegressor

        model = BayesianQuantileRegressor(n_warmup=10, n_samples=10, random_state=0)
        with pytest.raises(ValueError, match="target must be specified"):
            model.fit(regression_data)


class TestBayesianQuantileRegressorPredict:
    """Test BayesianQuantileRegressor predict method."""

    def test_predict_before_fit_raises(self, regression_data):
        """Should raise ModelNotFittedError if predict called before fit."""
        from uncertainty_flow.bayesian import BayesianQuantileRegressor
        from uncertainty_flow.utils.exceptions import ModelNotFittedError

        model = BayesianQuantileRegressor()
        with pytest.raises(ModelNotFittedError, match="not fitted"):
            model.predict(regression_data)

    def test_predict_returns_distribution_prediction(self, regression_data):
        """Should return a DistributionPrediction."""
        from uncertainty_flow.bayesian import BayesianQuantileRegressor
        from uncertainty_flow.core.distribution import DistributionPrediction

        model = BayesianQuantileRegressor(n_warmup=10, n_samples=10, random_state=0)
        model.fit(regression_data, target="y")
        pred = model.predict(regression_data)
        assert isinstance(pred, DistributionPrediction)

    def test_predict_has_posterior(self, regression_data):
        """Should have posterior samples accessible via posterior_samples()."""
        from uncertainty_flow.bayesian import BayesianQuantileRegressor

        model = BayesianQuantileRegressor(n_warmup=10, n_samples=10, random_state=0)
        model.fit(regression_data, target="y")
        pred = model.predict(regression_data)
        posterior = pred.posterior_samples()
        assert isinstance(posterior, np.ndarray)
        assert posterior.shape[0] == 10

    def test_predict_output_shape(self, regression_data):
        """Predicted quantile matrix should have correct shape."""
        from uncertainty_flow.bayesian import BayesianQuantileRegressor

        model = BayesianQuantileRegressor(n_warmup=10, n_samples=10, random_state=0)
        model.fit(regression_data, target="y")
        pred = model.predict(regression_data)

        n_data = regression_data.height
        n_quantiles = len(model.quantiles)
        q_df = pred.quantile(model.quantiles)
        assert q_df.shape == (n_data, n_quantiles)

    def test_predict_with_lazyframe(self, regression_data):
        """Should work with LazyFrame input for predict."""
        from uncertainty_flow.bayesian import BayesianQuantileRegressor

        model = BayesianQuantileRegressor(n_warmup=10, n_samples=10, random_state=0)
        model.fit(regression_data, target="y")
        lf = regression_data.lazy()
        pred = model.predict(lf)
        assert pred is not None

    def test_predict_quantile_levels(self, regression_data):
        """Should return the configured quantile levels."""
        from uncertainty_flow.bayesian import BayesianQuantileRegressor

        quantiles = [0.05, 0.5, 0.95]
        model = BayesianQuantileRegressor(
            quantiles=quantiles,
            n_warmup=10,
            n_samples=10,
            random_state=0,
        )
        model.fit(regression_data, target="y")
        pred = model.predict(regression_data)
        q_df = pred.quantile(quantiles)
        assert q_df.shape == (regression_data.height, 3)


class TestBayesianPosteriorMethods:
    """Test Bayesian posterior methods on DistributionPrediction."""

    def test_credible_interval(self, regression_data):
        """credible_interval() should return a DataFrame with lower/upper."""
        from uncertainty_flow.bayesian import BayesianQuantileRegressor

        model = BayesianQuantileRegressor(n_warmup=10, n_samples=10, random_state=0)
        model.fit(regression_data, target="y")
        pred = model.predict(regression_data)
        ci = pred.credible_interval(0.9)
        assert isinstance(ci, pl.DataFrame)
        assert "lower" in ci.columns
        assert "upper" in ci.columns

    def test_rhat(self, regression_data):
        """rhat() should return an array of convergence diagnostics."""
        from uncertainty_flow.bayesian import BayesianQuantileRegressor

        model = BayesianQuantileRegressor(n_warmup=10, n_samples=10, random_state=0)
        model.fit(regression_data, target="y")
        pred = model.predict(regression_data)
        rhat = pred.rhat(n_chains=2)
        assert isinstance(rhat, np.ndarray)
        assert rhat.ndim == 1

    def test_posterior_summary(self, regression_data):
        """posterior_summary() should return a DataFrame with stats."""
        from uncertainty_flow.bayesian import BayesianQuantileRegressor

        model = BayesianQuantileRegressor(n_warmup=10, n_samples=10, random_state=0)
        model.fit(regression_data, target="y")
        pred = model.predict(regression_data)
        summary = pred.posterior_summary()
        assert isinstance(summary, pl.DataFrame)
        assert "param" in summary.columns
        assert "mean" in summary.columns
        assert "std" in summary.columns
        assert "q025" in summary.columns
        assert "q50" in summary.columns
        assert "q975" in summary.columns
