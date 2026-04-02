"""Tests for ConformalForecaster time series wrapper."""

import numpy as np
import polars as pl
import pytest
from sklearn.ensemble import GradientBoostingRegressor

from uncertainty_flow import ConformalForecaster
from uncertainty_flow.core.distribution import DistributionPrediction


@pytest.fixture
def time_series_data():
    """Create sample time series DataFrame for testing."""
    np.random.seed(42)
    n = 150
    return pl.DataFrame(
        {
            "date": range(n),
            "price": [10 + i * 0.5 + np.sin(i / 3) + np.random.randn() * 0.5 for i in range(n)],
            "volume": [100 + i * 2 + np.cos(i / 2) + np.random.randn() * 5 for i in range(n)],
        }
    )


@pytest.fixture
def univariate_time_series():
    """Create univariate time series DataFrame."""
    np.random.seed(42)
    n = 150
    return pl.DataFrame(
        {
            "date": range(n),
            "target": [10 + i * 0.5 + np.sin(i / 3) + np.random.randn() * 0.5 for i in range(n)],
        }
    )


class TestConformalForecasterInit:
    """Test ConformalForecaster initialization."""

    def test_init_default_params(self):
        """Should initialize with default parameters."""
        model = ConformalForecaster(
            base_model=GradientBoostingRegressor(),
            horizon=3,
            targets="price",
        )
        assert model.horizon == 3
        assert model.targets == ["price"]
        assert model.copula_family == "auto"
        assert model.calibration_method == "holdout"
        assert model.calibration_size == 0.2
        assert model.auto_tune is True

    def test_init_custom_params(self):
        """Should initialize with custom parameters."""
        model = ConformalForecaster(
            base_model=GradientBoostingRegressor(),
            horizon=5,
            targets=["price", "volume"],
            copula_family="gaussian",
            lags=2,
            calibration_method="cross",
            calibration_size=0.3,
            auto_tune=False,
            uncertainty_features=["volume"],
            random_state=42,
        )
        assert model.horizon == 5
        assert model.targets == ["price", "volume"]
        assert model.copula_family == "gaussian"
        assert model.lags == [2]
        assert model.calibration_method == "cross"
        assert model.calibration_size == 0.3
        assert model.auto_tune is False
        assert model.uncertainty_features == ["volume"]
        assert model.random_state == 42

    def test_init_single_target_as_string(self):
        """Should accept single target as string."""
        model = ConformalForecaster(
            base_model=GradientBoostingRegressor(),
            horizon=3,
            targets="price",
        )
        assert model.targets == ["price"]

    def test_init_multiple_targets_as_list(self):
        """Should accept multiple targets as list."""
        model = ConformalForecaster(
            base_model=GradientBoostingRegressor(),
            horizon=3,
            targets=["price", "volume"],
        )
        assert model.targets == ["price", "volume"]

    def test_init_multiple_lags_as_list(self):
        """Should accept multiple lags as list."""
        model = ConformalForecaster(
            base_model=GradientBoostingRegressor(),
            horizon=3,
            targets="price",
            lags=[1, 3, 7],
        )
        assert model.lags == [1, 3, 7]


class TestConformalForecasterFit:
    """Test ConformalForecaster fit method."""

    def test_fit_returns_self(self, univariate_time_series):
        """Should return self for method chaining."""
        model = ConformalForecaster(
            base_model=GradientBoostingRegressor(n_estimators=10, random_state=42),
            horizon=3,
            targets="target",
            random_state=42,
        )
        result = model.fit(univariate_time_series)
        assert result is model

    def test_fit_sets_fitted_flag(self, univariate_time_series):
        """Should set _fitted flag to True."""
        model = ConformalForecaster(
            base_model=GradientBoostingRegressor(n_estimators=10, random_state=42),
            horizon=3,
            targets="target",
            random_state=42,
        )
        assert not model._fitted
        model.fit(univariate_time_series)
        assert model._fitted is True

    def test_fit_stores_models_per_target(self, time_series_data):
        """Should store fitted model for each target."""
        model = ConformalForecaster(
            base_model=GradientBoostingRegressor(n_estimators=10, random_state=42),
            horizon=3,
            targets=["price", "volume"],
            random_state=42,
        )
        model.fit(time_series_data)
        assert len(model._models_) == 2
        assert "price" in model._models_
        assert "volume" in model._models_

    def test_fit_stores_quantiles_per_target(self, univariate_time_series):
        """Should store conformal quantiles for each target."""
        model = ConformalForecaster(
            base_model=GradientBoostingRegressor(n_estimators=10, random_state=42),
            horizon=3,
            targets="target",
            random_state=42,
        )
        model.fit(univariate_time_series)
        assert "target" in model._quantiles_
        assert len(model._quantiles_["target"]) == 11  # DEFAULT_QUANTILES

    def test_fit_with_lazyframe(self, univariate_time_series):
        """Should work with LazyFrame input."""
        lazy_df = univariate_time_series.lazy()
        model = ConformalForecaster(
            base_model=GradientBoostingRegressor(n_estimators=10, random_state=42),
            horizon=3,
            targets="target",
            random_state=42,
        )
        model.fit(lazy_df)
        assert model._fitted is True

    def test_fit_multivariate_with_copula(self, time_series_data):
        """Should fit copula for multivariate targets."""
        model = ConformalForecaster(
            base_model=GradientBoostingRegressor(n_estimators=10, random_state=42),
            horizon=3,
            targets=["price", "volume"],
            copula_family="gaussian",
            random_state=42,
        )
        model.fit(time_series_data)
        assert model._copula is not None

    def test_fit_multivariate_independent(self, time_series_data):
        """Should not fit copula when copula_family is independent."""
        model = ConformalForecaster(
            base_model=GradientBoostingRegressor(n_estimators=10, random_state=42),
            horizon=3,
            targets=["price", "volume"],
            copula_family="independent",
            random_state=42,
        )
        model.fit(time_series_data)
        assert model._copula is None


class TestConformalForecasterPredict:
    """Test ConformalForecaster predict method."""

    def test_predict_before_fit_raises(self):
        """Should raise error if model not fitted."""
        model = ConformalForecaster(
            base_model=GradientBoostingRegressor(),
            horizon=3,
            targets="price",
        )
        with pytest.raises(ValueError, match="not fitted"):
            model.predict(pl.DataFrame({"date": [1, 2, 3], "price": [1, 2, 3]}))

    def test_predict_returns_distribution_prediction(self, univariate_time_series):
        """Should return DistributionPrediction."""
        model = ConformalForecaster(
            base_model=GradientBoostingRegressor(n_estimators=10, random_state=42),
            horizon=3,
            targets="target",
            random_state=42,
        )
        model.fit(univariate_time_series)
        pred = model.predict(univariate_time_series)
        assert isinstance(pred, DistributionPrediction)

    def test_predict_univariate_shape(self, univariate_time_series):
        """Should output correct shape for univariate."""
        model = ConformalForecaster(
            base_model=GradientBoostingRegressor(n_estimators=10, random_state=42),
            horizon=3,
            targets="target",
            random_state=42,
        )
        model.fit(univariate_time_series)
        pred = model.predict(univariate_time_series)
        assert pred._quantiles.shape[1] == 11  # 11 quantiles

    def test_predict_multivariate_shape(self, time_series_data):
        """Should output correct shape for multivariate."""
        model = ConformalForecaster(
            base_model=GradientBoostingRegressor(n_estimators=10, random_state=42),
            horizon=3,
            targets=["price", "volume"],
            random_state=42,
        )
        model.fit(time_series_data)
        pred = model.predict(time_series_data)
        # 2 targets * 11 quantiles = 22 columns
        assert pred._quantiles.shape[1] == 22

    def test_predict_quantiles_sorted(self, univariate_time_series):
        """Should return sorted quantiles (non-crossing)."""
        model = ConformalForecaster(
            base_model=GradientBoostingRegressor(n_estimators=10, random_state=42),
            horizon=3,
            targets="target",
            random_state=42,
        )
        model.fit(univariate_time_series)
        pred = model.predict(univariate_time_series)
        for i in range(pred._quantiles.shape[0]):
            row = pred._quantiles[i, :]
            assert list(row) == sorted(row), "Quantiles should be non-crossing"

    def test_predict_with_lazyframe(self, univariate_time_series):
        """Should work with LazyFrame input."""
        model = ConformalForecaster(
            base_model=GradientBoostingRegressor(n_estimators=10, random_state=42),
            horizon=3,
            targets="target",
            random_state=42,
        )
        model.fit(univariate_time_series)
        pred = model.predict(univariate_time_series.lazy())
        assert isinstance(pred, DistributionPrediction)

    def test_predict_with_custom_steps(self, univariate_time_series):
        """Should respect custom steps parameter."""
        model = ConformalForecaster(
            base_model=GradientBoostingRegressor(n_estimators=10, random_state=42),
            horizon=5,
            targets="target",
            random_state=42,
        )
        model.fit(univariate_time_series)
        pred = model.predict(univariate_time_series, steps=10)
        assert isinstance(pred, DistributionPrediction)


class TestConformalForecasterInterval:
    """Test ConformalForecaster interval predictions."""

    def test_interval_univariate_returns_dataframe(self, univariate_time_series):
        """Should return DataFrame with lower/upper columns for univariate."""
        model = ConformalForecaster(
            base_model=GradientBoostingRegressor(n_estimators=10, random_state=42),
            horizon=3,
            targets="target",
            random_state=42,
        )
        model.fit(univariate_time_series)
        pred = model.predict(univariate_time_series)
        interval = pred.interval(0.9)
        assert isinstance(interval, pl.DataFrame)
        assert "lower" in interval.columns
        assert "upper" in interval.columns

    def test_interval_multivariate_returns_dataframe(self, time_series_data):
        """Should return DataFrame with per-target lower/upper columns."""
        model = ConformalForecaster(
            base_model=GradientBoostingRegressor(n_estimators=10, random_state=42),
            horizon=3,
            targets=["price", "volume"],
            random_state=42,
        )
        model.fit(time_series_data)
        pred = model.predict(time_series_data)
        interval = pred.interval(0.9)
        assert "price_lower" in interval.columns
        assert "price_upper" in interval.columns
        assert "volume_lower" in interval.columns
        assert "volume_upper" in interval.columns

    def test_interval_lower_less_than_upper(self, univariate_time_series):
        """Lower bound should be less than upper bound."""
        model = ConformalForecaster(
            base_model=GradientBoostingRegressor(n_estimators=10, random_state=42),
            horizon=3,
            targets="target",
            random_state=42,
        )
        model.fit(univariate_time_series)
        pred = model.predict(univariate_time_series)
        interval = pred.interval(0.9)
        assert (interval["lower"] < interval["upper"]).all()


class TestConformalForecasterMean:
    """Test ConformalForecaster mean predictions."""

    def test_mean_returns_series_univariate(self, univariate_time_series):
        """Should return Series with median predictions for univariate."""
        model = ConformalForecaster(
            base_model=GradientBoostingRegressor(n_estimators=10, random_state=42),
            horizon=3,
            targets="target",
            random_state=42,
        )
        model.fit(univariate_time_series)
        pred = model.predict(univariate_time_series)
        mean = pred.mean()
        assert isinstance(mean, pl.Series)

    def test_mean_returns_dataframe_multivariate(self, time_series_data):
        """Should return DataFrame with per-target means for multivariate."""
        model = ConformalForecaster(
            base_model=GradientBoostingRegressor(n_estimators=10, random_state=42),
            horizon=3,
            targets=["price", "volume"],
            random_state=42,
        )
        model.fit(time_series_data)
        pred = model.predict(time_series_data)
        mean = pred.mean()
        assert isinstance(mean, pl.DataFrame)
        assert "price" in mean.columns
        assert "volume" in mean.columns


class TestConformalForecasterCoverage:
    """Test ConformalForecaster coverage."""

    def test_90_coverage_approximately_90(self, univariate_time_series):
        """90% interval should contain approximately 90% of true values."""
        from uncertainty_flow import coverage_score

        train = univariate_time_series.head(110)
        calib = univariate_time_series.tail(30)

        model = ConformalForecaster(
            base_model=GradientBoostingRegressor(n_estimators=50, random_state=42),
            horizon=3,
            targets="target",
            random_state=42,
        )
        model.fit(train)
        pred = model.predict(calib)

        interval = pred.interval(0.9)
        cov = coverage_score(
            calib["target"].to_numpy()[1:],
            interval["lower"].to_numpy(),
            interval["upper"].to_numpy(),
        )
        assert 0.0 <= cov <= 1.0


class TestConformalForecasterCopulaFamily:
    """Test copula family selection."""

    def test_auto_select_copula(self, time_series_data):
        """Should auto-select copula when family is auto."""
        model = ConformalForecaster(
            base_model=GradientBoostingRegressor(n_estimators=10, random_state=42),
            horizon=3,
            targets=["price", "volume"],
            copula_family="auto",
            random_state=42,
        )
        model.fit(time_series_data)
        assert model._copula is not None

    def test_invalid_copula_family_raises(self, time_series_data):
        """Should raise error for invalid copula family."""
        model = ConformalForecaster(
            base_model=GradientBoostingRegressor(n_estimators=10, random_state=42),
            horizon=3,
            targets=["price", "volume"],
            copula_family="invalid",
            random_state=42,
        )
        with pytest.raises(Exception, match="Unknown copula_family"):
            model.fit(time_series_data)
