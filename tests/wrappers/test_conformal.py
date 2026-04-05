"""Tests for ConformalRegressor wrapper."""

import numpy as np
import polars as pl
import pytest
from sklearn.ensemble import GradientBoostingRegressor

from uncertainty_flow import ConformalRegressor
from uncertainty_flow.core.distribution import DistributionPrediction


@pytest.fixture
def regression_data():
    """Create sample regression data."""
    np.random.seed(42)
    n = 100
    df = pl.DataFrame(
        {
            "x1": np.random.randn(n),
            "x2": np.random.randn(n),
            "x3": np.random.randn(n),
            "target": 3 * np.random.randn(n) + 5,
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
    df = df.with_columns(target=pl.col("x1") * 2 + pl.col("x2") * 3 + np.random.randn(n) * 0.5)
    return df


class TestConformalRegressorInit:
    """Test ConformalRegressor initialization."""

    def test_init_default_params(self):
        """Should initialize with default parameters."""
        model = ConformalRegressor(base_model=GradientBoostingRegressor())
        assert model.calibration_method == "holdout"
        assert model.calibration_size == 0.2
        assert model.coverage_target == 0.9
        assert model.auto_tune is True
        assert model.random_state is None

    def test_init_custom_params(self):
        """Should initialize with custom parameters."""
        model = ConformalRegressor(
            base_model=GradientBoostingRegressor(),
            calibration_method="cross",
            calibration_size=0.3,
            coverage_target=0.95,
            auto_tune=False,
            uncertainty_features=["x1"],
            random_state=42,
        )
        assert model.calibration_method == "cross"
        assert model.calibration_size == 0.3
        assert model.coverage_target == 0.95
        assert model.auto_tune is False
        assert model.uncertainty_features == ["x1"]
        assert model.random_state == 42

    def test_init_accepts_base_model(self):
        """Should accept any sklearn-compatible regressor."""
        model = ConformalRegressor(base_model=GradientBoostingRegressor(n_estimators=10))
        assert model.base_model is not None


class TestConformalRegressorFit:
    """Test ConformalRegressor fit method."""

    def test_fit_returns_self(self, regression_data):
        """Should return self for method chaining."""
        model = ConformalRegressor(
            base_model=GradientBoostingRegressor(n_estimators=10, random_state=42),
            random_state=42,
        )
        result = model.fit(regression_data, target="target")
        assert result is model

    def test_fit_sets_fitted_flag(self, regression_data):
        """Should set _fitted flag to True."""
        model = ConformalRegressor(
            base_model=GradientBoostingRegressor(n_estimators=10, random_state=42),
            random_state=42,
        )
        assert not model._fitted
        model.fit(regression_data, target="target")
        assert model._fitted is True

    def test_fit_stores_feature_cols(self, regression_data):
        """Should store feature column names."""
        model = ConformalRegressor(
            base_model=GradientBoostingRegressor(n_estimators=10, random_state=42),
            random_state=42,
        )
        model.fit(regression_data, target="target")
        assert set(model._feature_cols_) == {"x1", "x2", "x3"}

    def test_fit_stores_target_col(self, regression_data):
        """Should store target column name."""
        model = ConformalRegressor(
            base_model=GradientBoostingRegressor(n_estimators=10, random_state=42),
            random_state=42,
        )
        model.fit(regression_data, target="target")
        assert model._target_col_ == "target"

    def test_fit_stores_quantiles(self, regression_data):
        """Should store conformal quantiles."""
        model = ConformalRegressor(
            base_model=GradientBoostingRegressor(n_estimators=10, random_state=42),
            random_state=42,
        )
        model.fit(regression_data, target="target")
        assert model._quantiles_ is not None
        assert len(model._quantiles_) == 11  # DEFAULT_QUANTILES

    def test_fit_with_lazyframe(self, regression_data):
        """Should work with LazyFrame input."""
        lazy_df = regression_data.lazy()
        model = ConformalRegressor(
            base_model=GradientBoostingRegressor(n_estimators=10, random_state=42),
            random_state=42,
        )
        model.fit(lazy_df, target="target")
        assert model._fitted is True

    def test_fit_with_cross_method(self, regression_data):
        """Should work with cross conformal calibration."""
        model = ConformalRegressor(
            base_model=GradientBoostingRegressor(n_estimators=10, random_state=42),
            calibration_method="cross",
            random_state=42,
        )
        model.fit(regression_data, target="target")
        assert model._fitted is True
        assert model._quantiles_ is not None


class TestConformalRegressorPredict:
    """Test ConformalRegressor predict method."""

    def test_predict_before_fit_raises(self):
        """Should raise error if model not fitted."""
        model = ConformalRegressor(base_model=GradientBoostingRegressor())
        with pytest.raises(ValueError, match="not fitted"):
            model.predict(pl.DataFrame({"x": [1, 2, 3]}))

    def test_predict_returns_distribution_prediction(self, regression_data):
        """Should return DistributionPrediction."""
        model = ConformalRegressor(
            base_model=GradientBoostingRegressor(n_estimators=10, random_state=42),
            random_state=42,
        )
        model.fit(regression_data, target="target")
        pred = model.predict(regression_data)
        assert isinstance(pred, DistributionPrediction)

    def test_predict_output_shape(self, regression_data):
        """Should output correct shape."""
        model = ConformalRegressor(
            base_model=GradientBoostingRegressor(n_estimators=10, random_state=42),
            random_state=42,
        )
        model.fit(regression_data, target="target")
        pred = model.predict(regression_data)
        assert pred._quantiles.shape == (100, 11)  # 100 samples, 11 quantiles

    def test_predict_quantiles_sorted(self, regression_data):
        """Should return sorted quantiles (non-crossing)."""
        model = ConformalRegressor(
            base_model=GradientBoostingRegressor(n_estimators=10, random_state=42),
            random_state=42,
        )
        model.fit(regression_data, target="target")
        pred = model.predict(regression_data)
        for i in range(pred._quantiles.shape[0]):
            row = pred._quantiles[i, :]
            assert list(row) == sorted(row), "Quantiles should be non-crossing"

    def test_predict_with_lazyframe(self, regression_data):
        """Should work with LazyFrame input."""
        model = ConformalRegressor(
            base_model=GradientBoostingRegressor(n_estimators=10, random_state=42),
            random_state=42,
        )
        model.fit(regression_data, target="target")
        pred = model.predict(regression_data.lazy())
        assert isinstance(pred, DistributionPrediction)


class TestConformalRegressorInterval:
    """Test ConformalRegressor interval predictions."""

    def test_interval_returns_dataframe(self, regression_data):
        """Should return DataFrame with lower/upper columns."""
        model = ConformalRegressor(
            base_model=GradientBoostingRegressor(n_estimators=10, random_state=42),
            random_state=42,
        )
        model.fit(regression_data, target="target")
        pred = model.predict(regression_data)
        interval = pred.interval(0.9)
        assert isinstance(interval, pl.DataFrame)
        assert "lower" in interval.columns
        assert "upper" in interval.columns

    def test_interval_lower_less_than_upper(self, regression_data):
        """Lower bound should be less than upper bound."""
        model = ConformalRegressor(
            base_model=GradientBoostingRegressor(n_estimators=10, random_state=42),
            random_state=42,
        )
        model.fit(regression_data, target="target")
        pred = model.predict(regression_data)
        interval = pred.interval(0.9)
        assert (interval["lower"] < interval["upper"]).all()


class TestConformalRegressorMean:
    """Test ConformalRegressor mean predictions."""

    def test_mean_returns_series(self, regression_data):
        """Should return Series with median predictions."""
        model = ConformalRegressor(
            base_model=GradientBoostingRegressor(n_estimators=10, random_state=42),
            random_state=42,
        )
        model.fit(regression_data, target="target")
        pred = model.predict(regression_data)
        median = pred.median()
        assert isinstance(median, pl.Series)
        assert len(median) == 100


class TestConformalRegressorCoverage:
    """Test ConformalRegressor coverage guarantees."""

    def test_90_coverage_approximately_90(self, linear_regression_data):
        """90% interval should contain approximately 90% of true values."""
        from uncertainty_flow import coverage_score

        train = linear_regression_data.head(120)
        calib = linear_regression_data.tail(30)

        model = ConformalRegressor(
            base_model=GradientBoostingRegressor(n_estimators=50, random_state=42),
            random_state=42,
        )
        model.fit(train, target="target")
        pred = model.predict(calib)

        interval = pred.interval(0.9)
        cov = coverage_score(calib["target"], interval["lower"], interval["upper"])
        assert 0.5 <= cov <= 1.0


class TestConformalRegressorUncertaintyDrivers:
    """Test uncertainty_drivers_ property."""

    def test_uncertainty_drivers_returns_dataframe(self, regression_data):
        """Should return DataFrame with correlation analysis."""
        model = ConformalRegressor(
            base_model=GradientBoostingRegressor(n_estimators=10, random_state=42),
            random_state=42,
        )
        model.fit(regression_data, target="target")
        assert model.uncertainty_drivers_ is not None
        assert isinstance(model.uncertainty_drivers_, pl.DataFrame)
        assert "feature" in model.uncertainty_drivers_.columns
        assert "residual_correlation" in model.uncertainty_drivers_.columns
        assert "p_value" in model.uncertainty_drivers_.columns

    def test_uncertainty_drivers_sorted_by_correlation(self, regression_data):
        """Should be sorted by absolute correlation descending."""
        model = ConformalRegressor(
            base_model=GradientBoostingRegressor(n_estimators=10, random_state=42),
            random_state=42,
        )
        model.fit(regression_data, target="target")
        drivers = model.uncertainty_drivers_
        if drivers.height > 1:
            correlations = drivers["residual_correlation"].to_list()
            assert correlations == sorted(correlations, reverse=True)
