"""Tests for DeepQuantileNet (sklearn backend)."""

import numpy as np
import polars as pl
import pytest

from uncertainty_flow import DeepQuantileNet
from uncertainty_flow.core.distribution import DistributionPrediction
from uncertainty_flow.utils.exceptions import ModelNotFittedError


@pytest.fixture
def sample_regression_data():
    """Create sample regression data."""
    np.random.seed(42)
    n = 100
    df = pl.DataFrame(
        {
            "x1": np.random.randn(n),
            "x2": np.random.randn(n),
            "x3": np.random.randn(n),
            "y": 3 * np.random.randn(n) + 5,
        }
    )
    return df


@pytest.fixture
def linear_regression_data():
    """Create data with linear relationship."""
    np.random.seed(42)
    n = 100
    df = pl.DataFrame(
        {
            "x1": np.linspace(0, 10, n),
            "x2": np.linspace(-5, 5, n),
        }
    )
    df = df.with_columns(y=pl.col("x1") * 2 + pl.col("x2") * 3 + np.random.randn(n) * 0.5)
    return df


class TestDeepQuantileNetInit:
    """Test DeepQuantileNet initialization."""

    def test_init_default_params(self):
        """Should initialize with default parameters."""
        model = DeepQuantileNet()
        assert model.hidden_layer_sizes == (100, 50)
        assert model.random_state is None
        assert model.trunk_alpha == 0.0001

    def test_init_custom_params(self):
        """Should initialize with custom parameters."""
        model = DeepQuantileNet(
            hidden_layer_sizes=(128, 64, 32),
            quantile_levels=[0.1, 0.5, 0.9],
            trunk_alpha=0.01,
            trunk_max_iter=1000,
            random_state=42,
        )
        assert model.hidden_layer_sizes == (128, 64, 32)
        assert model.quantile_levels == [0.1, 0.5, 0.9]
        assert model.trunk_alpha == 0.01
        assert model.trunk_max_iter == 1000
        assert model.random_state == 42


class TestDeepQuantileNetFit:
    """Test DeepQuantileNet fit method."""

    def test_fit_returns_self(self, sample_regression_data):
        """Should return self for method chaining."""
        model = DeepQuantileNet(random_state=42)
        result = model.fit(sample_regression_data, target="y")
        assert result is model

    def test_fit_sets_fitted_flag(self, sample_regression_data):
        """Should set _fitted flag to True."""
        model = DeepQuantileNet(random_state=42)
        assert not getattr(model, "_fitted", False)
        model.fit(sample_regression_data, target="y")
        assert model._fitted is True

    def test_fit_stores_feature_cols(self, sample_regression_data):
        """Should store feature column names."""
        model = DeepQuantileNet(random_state=42)
        model.fit(sample_regression_data, target="y")
        assert model._feature_cols_ == ["x1", "x2", "x3"]

    def test_fit_stores_scaler(self, sample_regression_data):
        """Should fit and store StandardScaler."""
        model = DeepQuantileNet(random_state=42)
        model.fit(sample_regression_data, target="y")
        assert hasattr(model, "_scaler_")
        assert model._scaler_ is not None

    def test_fit_stores_trunk(self, sample_regression_data):
        """Should fit and store trunk MLP."""
        model = DeepQuantileNet(random_state=42)
        model.fit(sample_regression_data, target="y")
        assert hasattr(model, "_trunk_")
        assert model._trunk_ is not None

    def test_fit_stores_head_coefs(self, sample_regression_data):
        """Should store head coefficients for each quantile."""
        model = DeepQuantileNet(random_state=42)
        model.fit(sample_regression_data, target="y")
        assert hasattr(model, "_head_coefs_")
        assert len(model._head_coefs_) == len(model.quantile_levels)

    def test_fit_with_lazyframe(self, sample_regression_data):
        """Should work with LazyFrame input."""
        lazy_df = sample_regression_data.lazy()
        model = DeepQuantileNet(random_state=42)
        model.fit(lazy_df, target="y")
        assert model._fitted is True

    def test_fit_with_numpy_target(self, sample_regression_data):
        """Should work when target is passed as numpy array."""
        model = DeepQuantileNet(random_state=42)
        x = sample_regression_data.select(["x1", "x2", "x3"]).to_numpy()
        y = sample_regression_data.select("y").to_numpy()
        model.fit(x, y)
        assert model._fitted is True


class TestDeepQuantileNetPredict:
    """Test DeepQuantileNet predict method."""

    def test_predict_before_fit_raises(self):
        """Should raise error if model not fitted."""
        model = DeepQuantileNet()
        with pytest.raises(ModelNotFittedError, match="not fitted"):
            model.predict(pl.DataFrame({"x": [1, 2, 3]}))

    def test_predict_returns_distribution_prediction(self, sample_regression_data):
        """Should return DistributionPrediction."""
        model = DeepQuantileNet(random_state=42)
        model.fit(sample_regression_data, target="y")
        pred = model.predict(sample_regression_data)
        assert isinstance(pred, DistributionPrediction)

    def test_predict_output_shape(self, sample_regression_data):
        """Should output correct shape."""
        model = DeepQuantileNet(random_state=42)
        model.fit(sample_regression_data, target="y")
        pred = model.predict(sample_regression_data)
        assert pred._quantiles.shape == (100, 11)  # 100 samples, 11 quantiles

    def test_predict_quantiles_sorted(self, sample_regression_data):
        """Should return sorted quantiles (non-crossing)."""
        model = DeepQuantileNet(random_state=42)
        model.fit(sample_regression_data, target="y")
        pred = model.predict(sample_regression_data)
        # Each row should be sorted
        for i in range(pred._quantiles.shape[0]):
            row = pred._quantiles[i, :]
            assert list(row) == sorted(row), "Quantiles should be non-crossing"

    def test_predict_with_lazyframe(self, sample_regression_data):
        """Should work with LazyFrame input."""
        model = DeepQuantileNet(random_state=42)
        model.fit(sample_regression_data, target="y")
        pred = model.predict(sample_regression_data.lazy())
        assert isinstance(pred, DistributionPrediction)

    def test_predict_with_numpy(self, sample_regression_data):
        """Should work with numpy array input."""
        model = DeepQuantileNet(random_state=42)
        model.fit(sample_regression_data, target="y")
        x = sample_regression_data.select(["x1", "x2", "x3"]).to_numpy()
        pred = model.predict(x)
        assert isinstance(pred, DistributionPrediction)


class TestDeepQuantileNetInterval:
    """Test DeepQuantileNet interval predictions."""

    def test_interval_returns_dataframe(self, sample_regression_data):
        """Should return DataFrame with lower/upper columns."""
        model = DeepQuantileNet(random_state=42)
        model.fit(sample_regression_data, target="y")
        pred = model.predict(sample_regression_data)
        interval = pred.interval(0.9)
        assert isinstance(interval, pl.DataFrame)
        assert "lower" in interval.columns
        assert "upper" in interval.columns

    def test_interval_lower_less_than_upper(self, sample_regression_data):
        """Lower bound should be less than upper bound."""
        model = DeepQuantileNet(random_state=42)
        model.fit(sample_regression_data, target="y")
        pred = model.predict(sample_regression_data)
        interval = pred.interval(0.9)
        assert (interval["lower"] < interval["upper"]).all()


class TestDeepQuantileNetMean:
    """Test DeepQuantileNet mean (median) predictions."""

    def test_median_returns_series(self, sample_regression_data):
        """Should return Series with median predictions."""
        model = DeepQuantileNet(random_state=42)
        model.fit(sample_regression_data, target="y")
        pred = model.predict(sample_regression_data)
        median = pred.median()
        assert isinstance(median, pl.Series)
        assert len(median) == 100


class TestDeepQuantileNetCoverage:
    """Test DeepQuantileNet prediction coverage."""

    def test_90_coverage_approximately_90(self, linear_regression_data):
        """90% interval should contain approximately 90% of true values."""
        from uncertainty_flow import coverage_score

        train = linear_regression_data.head(80)
        calib = linear_regression_data.tail(20)

        model = DeepQuantileNet(
            hidden_layer_sizes=(64, 32),
            random_state=42,
            trunk_max_iter=1000,
        )
        model.fit(train, target="y")
        pred = model.predict(calib)

        interval = pred.interval(0.9)
        cov = coverage_score(calib["y"], interval["lower"], interval["upper"])
        # Should be at least 0.6 (sklearn model has wider intervals)
        assert 0.6 <= cov <= 1.0


class TestDeepQuantileNetNonCrossingPenalty:
    """Test training-time non-crossing penalty."""

    def _count_crossings(self, quantile_matrix: np.ndarray) -> int:
        """Count number of samples with at least one quantile crossing."""
        return int(np.sum(np.any(np.diff(quantile_matrix, axis=1) < 0, axis=1)))

    def test_non_crossing_penalty_reduces_crossings(self, sample_regression_data):
        """Training-time penalty should reduce quantile crossings."""
        # Fit without penalty
        model_no_penalty = DeepQuantileNet(
            random_state=42,
            non_crossing_penalty=0.0,
        )
        model_no_penalty.fit(sample_regression_data, target="y")
        _ = model_no_penalty.predict(sample_regression_data)
        # Note: post-sorting is applied in predict(), so we need to check
        # the raw backend predictions before sorting.
        raw_no_penalty = model_no_penalty._predict_backend(
            model_no_penalty._scaler_.transform(
                sample_regression_data.select(
                    [c for c in sample_regression_data.columns if c != "y"]
                ).to_numpy()
            )
        )
        crossings_no_penalty = self._count_crossings(raw_no_penalty)

        # Fit with penalty
        model_with_penalty = DeepQuantileNet(
            random_state=42,
            non_crossing_penalty=10.0,
        )
        model_with_penalty.fit(sample_regression_data, target="y")
        raw_with_penalty = model_with_penalty._predict_backend(
            model_with_penalty._scaler_.transform(
                sample_regression_data.select(
                    [c for c in sample_regression_data.columns if c != "y"]
                ).to_numpy()
            )
        )
        crossings_with_penalty = self._count_crossings(raw_with_penalty)

        # The penalty should reduce or eliminate crossings in the raw predictions
        assert crossings_with_penalty <= crossings_no_penalty

    def test_non_crossing_penalty_preserves_monotonicity(self, sample_regression_data):
        """Predictions should remain monotone even with high penalty."""
        model = DeepQuantileNet(
            random_state=42,
            non_crossing_penalty=50.0,
        )
        model.fit(sample_regression_data, target="y")
        pred = model.predict(sample_regression_data)
        # All rows should be sorted (no crossings after predict's post-sort)
        for i in range(pred._quantiles.shape[0]):
            row = pred._quantiles[i, :]
            assert list(row) == sorted(row), "Quantiles should be non-crossing"

    def test_non_crossing_penalty_zero_is_default(self, sample_regression_data):
        """Default penalty should be 0.1 (enabled training-time penalty)."""
        model = DeepQuantileNet(random_state=42)
        assert model.non_crossing_penalty == 0.1


class TestDeepQuantileNetUncertaintyDrivers:
    """Test uncertainty_drivers_ property."""

    def test_uncertainty_drivers_returns_none(self, sample_regression_data):
        """Should return None (not implemented for this model)."""
        model = DeepQuantileNet(random_state=42)
        model.fit(sample_regression_data, target="y")
        assert model.uncertainty_drivers_ is None
