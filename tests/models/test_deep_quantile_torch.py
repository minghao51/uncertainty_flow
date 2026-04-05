"""Tests for DeepQuantileNetTorch (PyTorch backend)."""

from importlib.util import find_spec

import numpy as np
import polars as pl
import pytest

from uncertainty_flow import DeepQuantileNetTorch
from uncertainty_flow.core.distribution import DistributionPrediction

TORCH_AVAILABLE = find_spec("torch") is not None


pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE, reason="PyTorch not installed (pip install uncertainty-flow[torch])"
)


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


class TestDeepQuantileNetTorchInit:
    """Test DeepQuantileNetTorch initialization."""

    def test_init_default_params(self):
        """Should initialize with default parameters."""
        model = DeepQuantileNetTorch()
        assert model.hidden_layer_sizes == (100, 50)
        assert model.random_state is None
        assert model.epochs == 100
        assert model.learning_rate == 0.001

    def test_init_custom_params(self):
        """Should initialize with custom parameters."""
        model = DeepQuantileNetTorch(
            hidden_layer_sizes=(128, 64, 32),
            quantile_levels=[0.1, 0.5, 0.9],
            epochs=50,
            batch_size=32,
            learning_rate=0.01,
            weight_decay=0.0001,
            monotonicity_weight=0.1,
            device="cpu",
            random_state=42,
            verbose=True,
        )
        assert model.hidden_layer_sizes == (128, 64, 32)
        assert model.quantile_levels == [0.1, 0.5, 0.9]
        assert model.epochs == 50
        assert model.batch_size == 32
        assert model.learning_rate == 0.01
        assert model.monotonicity_weight == 0.1

    def test_resolve_device_auto_cpu(self):
        """Should resolve to CPU when cuda not available."""
        model = DeepQuantileNetTorch(device="auto")
        assert model.device.type == "cpu"


class TestDeepQuantileNetTorchFit:
    """Test DeepQuantileNetTorch fit method."""

    def test_fit_returns_self(self, sample_regression_data):
        """Should return self for method chaining."""
        model = DeepQuantileNetTorch(
            epochs=20,
            batch_size=32,
            random_state=42,
            device="cpu",
        )
        result = model.fit(sample_regression_data, target="y")
        assert result is model

    def test_fit_sets_fitted_flag(self, sample_regression_data):
        """Should set _fitted flag to True."""
        model = DeepQuantileNetTorch(
            epochs=20,
            batch_size=32,
            random_state=42,
            device="cpu",
        )
        assert not getattr(model, "_fitted", False)
        model.fit(sample_regression_data, target="y")
        assert model._fitted is True

    def test_fit_stores_feature_cols(self, sample_regression_data):
        """Should store feature column names."""
        model = DeepQuantileNetTorch(
            epochs=20,
            batch_size=32,
            random_state=42,
            device="cpu",
        )
        model.fit(sample_regression_data, target="y")
        assert model._feature_cols_ == ["x1", "x2", "x3"]

    def test_fit_stores_models(self, sample_regression_data):
        """Should store fitted PyTorch models."""
        model = DeepQuantileNetTorch(
            epochs=20,
            batch_size=32,
            n_estimators=2,
            random_state=42,
            device="cpu",
        )
        model.fit(sample_regression_data, target="y")
        assert hasattr(model, "_models")
        assert len(model._models) == 2

    def test_fit_with_lazyframe(self, sample_regression_data):
        """Should work with LazyFrame input."""
        lazy_df = sample_regression_data.lazy()
        model = DeepQuantileNetTorch(
            epochs=20,
            batch_size=32,
            random_state=42,
            device="cpu",
        )
        model.fit(lazy_df, target="y")
        assert model._fitted is True


class TestDeepQuantileNetTorchPredict:
    """Test DeepQuantileNetTorch predict method."""

    def test_predict_before_fit_raises(self):
        """Should raise error if model not fitted."""
        model = DeepQuantileNetTorch(device="cpu")
        with pytest.raises(ValueError, match="not fitted"):
            model.predict(pl.DataFrame({"x": [1, 2, 3]}))

    def test_predict_returns_distribution_prediction(self, sample_regression_data):
        """Should return DistributionPrediction."""
        model = DeepQuantileNetTorch(
            epochs=20,
            batch_size=32,
            random_state=42,
            device="cpu",
        )
        model.fit(sample_regression_data, target="y")
        pred = model.predict(sample_regression_data)
        assert isinstance(pred, DistributionPrediction)

    def test_predict_output_shape(self, sample_regression_data):
        """Should output correct shape."""
        model = DeepQuantileNetTorch(
            epochs=20,
            batch_size=32,
            random_state=42,
            device="cpu",
        )
        model.fit(sample_regression_data, target="y")
        pred = model.predict(sample_regression_data)
        assert pred._quantiles.shape == (100, 11)  # 100 samples, 11 quantiles

    def test_predict_quantiles_sorted(self, sample_regression_data):
        """Should return sorted quantiles (non-crossing)."""
        model = DeepQuantileNetTorch(
            epochs=20,
            batch_size=32,
            random_state=42,
            device="cpu",
        )
        model.fit(sample_regression_data, target="y")
        pred = model.predict(sample_regression_data)
        for i in range(pred._quantiles.shape[0]):
            row = pred._quantiles[i, :]
            assert list(row) == sorted(row), "Quantiles should be non-crossing"

    def test_predict_with_lazyframe(self, sample_regression_data):
        """Should work with LazyFrame input."""
        model = DeepQuantileNetTorch(
            epochs=20,
            batch_size=32,
            random_state=42,
            device="cpu",
        )
        model.fit(sample_regression_data, target="y")
        pred = model.predict(sample_regression_data.lazy())
        assert isinstance(pred, DistributionPrediction)


class TestDeepQuantileNetTorchInterval:
    """Test DeepQuantileNetTorch interval predictions."""

    def test_interval_returns_dataframe(self, sample_regression_data):
        """Should return DataFrame with lower/upper columns."""
        model = DeepQuantileNetTorch(
            epochs=20,
            batch_size=32,
            random_state=42,
            device="cpu",
        )
        model.fit(sample_regression_data, target="y")
        pred = model.predict(sample_regression_data)
        interval = pred.interval(0.9)
        assert isinstance(interval, pl.DataFrame)
        assert "lower" in interval.columns
        assert "upper" in interval.columns

    def test_interval_lower_less_than_upper(self, sample_regression_data):
        """Lower bound should be less than upper bound."""
        model = DeepQuantileNetTorch(
            epochs=20,
            batch_size=32,
            random_state=42,
            device="cpu",
        )
        model.fit(sample_regression_data, target="y")
        pred = model.predict(sample_regression_data)
        interval = pred.interval(0.9)
        assert (interval["lower"] < interval["upper"]).all()


class TestDeepQuantileNetTorchMedian:
    """Test DeepQuantileNetTorch median predictions."""

    def test_median_returns_series(self, sample_regression_data):
        """Should return Series with median predictions."""
        model = DeepQuantileNetTorch(
            epochs=20,
            batch_size=32,
            random_state=42,
            device="cpu",
        )
        model.fit(sample_regression_data, target="y")
        pred = model.predict(sample_regression_data)
        median = pred.median()
        assert isinstance(median, pl.Series)
        assert len(median) == 100


class TestDeepQuantileNetTorchCoverage:
    """Test DeepQuantileNetTorch prediction coverage."""

    def test_90_coverage_approximately_90(self, linear_regression_data):
        """90% interval should contain approximately 90% of true values."""
        from uncertainty_flow import coverage_score

        train = linear_regression_data.head(80)
        calib = linear_regression_data.tail(20)

        model = DeepQuantileNetTorch(
            hidden_layer_sizes=(64, 32),
            epochs=50,
            batch_size=32,
            random_state=42,
            device="cpu",
        )
        model.fit(train, target="y")
        pred = model.predict(calib)

        interval = pred.interval(0.9)
        cov = coverage_score(calib["y"], interval["lower"], interval["upper"])
        # Should be at least 0.6 (can be 1.0 on small samples)
        assert 0.6 <= cov <= 1.0


class TestDeepQuantileNetTorchMonotonicity:
    """Test monotonicity penalty."""

    def test_monotonicity_weight_parameter(self, sample_regression_data):
        """Should accept monotonicity_weight parameter."""
        model = DeepQuantileNetTorch(
            epochs=20,
            batch_size=32,
            monotonicity_weight=0.5,
            random_state=42,
            device="cpu",
        )
        model.fit(sample_regression_data, target="y")
        assert model.monotonicity_weight == 0.5


class TestDeepQuantileNetTorchPinballScores:
    """Test pinball_scores method."""

    def test_pinball_scores(self, sample_regression_data):
        """Should compute pinball scores for each quantile."""
        model = DeepQuantileNetTorch(
            epochs=20,
            batch_size=32,
            random_state=42,
            device="cpu",
        )
        model.fit(sample_regression_data, target="y")
        scores = model.pinball_scores(sample_regression_data, target="y")
        assert isinstance(scores, dict)
        assert len(scores) == len(model.quantile_levels)
        for q in model.quantile_levels:
            assert q in scores
            assert scores[q] >= 0


class TestDeepQuantileNetTorchUncertaintyDrivers:
    """Test uncertainty_drivers_ property."""

    def test_uncertainty_drivers_returns_none(self, sample_regression_data):
        """Should return None (not implemented for this model)."""
        model = DeepQuantileNetTorch(
            epochs=20,
            batch_size=32,
            random_state=42,
            device="cpu",
        )
        model.fit(sample_regression_data, target="y")
        assert model.uncertainty_drivers_ is None
