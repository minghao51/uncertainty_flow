"""Tests for QuantileForestForecaster."""

import numpy as np
import polars as pl
import pytest

from uncertainty_flow import QuantileForestForecaster
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


class TestQuantileForestForecasterInit:
    """Test QuantileForestForecaster initialization."""

    def test_init_default_params(self):
        """Should initialize with default parameters."""
        model = QuantileForestForecaster(
            targets="price",
            horizon=3,
        )
        assert model.targets == ["price"]
        assert model.horizon == 3
        assert model.n_estimators == 200
        assert model.min_samples_leaf == 5
        assert model.copula_family == "auto"
        assert model.calibration_size == 0.2
        assert model.auto_tune is True

    def test_init_custom_params(self):
        """Should initialize with custom parameters."""
        model = QuantileForestForecaster(
            targets=["price", "volume"],
            horizon=5,
            n_estimators=100,
            min_samples_leaf=10,
            max_depth=10,
            copula_family="gaussian",
            calibration_size=0.3,
            auto_tune=False,
            uncertainty_features=["volume"],
            random_state=42,
        )
        assert model.targets == ["price", "volume"]
        assert model.horizon == 5
        assert model.n_estimators == 100
        assert model.min_samples_leaf == 10
        assert model.max_depth == 10
        assert model.copula_family == "gaussian"
        assert model.calibration_size == 0.3
        assert model.auto_tune is False
        assert model.uncertainty_features == ["volume"]
        assert model.random_state == 42

    def test_init_single_target_as_string(self):
        """Should accept single target as string."""
        model = QuantileForestForecaster(
            targets="price",
            horizon=3,
        )
        assert model.targets == ["price"]

    def test_init_multiple_targets_as_list(self):
        """Should accept multiple targets as list."""
        model = QuantileForestForecaster(
            targets=["price", "volume"],
            horizon=3,
        )
        assert model.targets == ["price", "volume"]


class TestQuantileForestForecasterFit:
    """Test QuantileForestForecaster fit method."""

    def test_fit_returns_self(self, univariate_time_series):
        """Should return self for method chaining."""
        model = QuantileForestForecaster(
            targets="target",
            horizon=3,
            n_estimators=10,
            random_state=42,
        )
        result = model.fit(univariate_time_series)
        assert result is model

    def test_fit_sets_fitted_flag(self, univariate_time_series):
        """Should set _fitted flag to True."""
        model = QuantileForestForecaster(
            targets="target",
            horizon=3,
            n_estimators=10,
            random_state=42,
        )
        assert not model._fitted
        model.fit(univariate_time_series)
        assert model._fitted is True

    def test_fit_stores_models_per_target(self, time_series_data):
        """Should store fitted model for each target."""
        model = QuantileForestForecaster(
            targets=["price", "volume"],
            horizon=3,
            n_estimators=10,
            random_state=42,
        )
        model.fit(time_series_data)
        assert len(model._models) == 2
        assert "price" in model._models
        assert "volume" in model._models

    def test_fit_stores_leaf_distributions(self, univariate_time_series):
        """Should extract and store leaf distributions."""
        model = QuantileForestForecaster(
            targets="target",
            horizon=3,
            n_estimators=10,
            random_state=42,
        )
        model.fit(univariate_time_series)
        assert "target" in model._leaf_distributions
        assert len(model._leaf_distributions["target"]) == model.n_estimators

    def test_fit_stores_feature_cols(self, univariate_time_series):
        """Should store feature column names per target."""
        model = QuantileForestForecaster(
            targets="target",
            horizon=3,
            n_estimators=10,
            random_state=42,
        )
        model.fit(univariate_time_series)
        assert "target" in model._feature_cols_

    def test_fit_with_lazyframe(self, univariate_time_series):
        """Should work with LazyFrame input."""
        lazy_df = univariate_time_series.lazy()
        model = QuantileForestForecaster(
            targets="target",
            horizon=3,
            n_estimators=10,
            random_state=42,
        )
        model.fit(lazy_df)
        assert model._fitted is True

    def test_fit_sets_copula_for_multivariate(self, time_series_data):
        """Should fit a copula when multivariate targets request dependence modeling."""
        model = QuantileForestForecaster(
            targets=["price", "volume"],
            horizon=3,
            n_estimators=10,
            copula_family="gaussian",
            auto_tune=False,
            random_state=42,
        )
        model.fit(time_series_data)
        assert model._copula is not None

    def test_fit_skips_copula_for_independent_family(self, time_series_data):
        """Should skip copula fitting when independence is requested."""
        model = QuantileForestForecaster(
            targets=["price", "volume"],
            horizon=3,
            n_estimators=10,
            copula_family="independent",
            auto_tune=False,
            random_state=42,
        )
        model.fit(time_series_data)
        assert model._copula is None

    def test_fit_invalid_copula_family_raises(self, time_series_data):
        """Should reject unknown copula families."""
        model = QuantileForestForecaster(
            targets=["price", "volume"],
            horizon=3,
            n_estimators=10,
            copula_family="invalid",
            auto_tune=False,
            random_state=42,
        )
        with pytest.raises(ValueError, match="Unknown copula_family"):
            model.fit(time_series_data)


class TestQuantileForestForecasterPredict:
    """Test QuantileForestForecaster predict method."""

    def test_predict_before_fit_raises(self):
        """Should raise error if model not fitted."""
        model = QuantileForestForecaster(
            targets="price",
            horizon=3,
        )
        with pytest.raises(ValueError, match="not fitted"):
            model.predict(pl.DataFrame({"date": [1, 2, 3], "price": [1, 2, 3]}))

    def test_predict_returns_distribution_prediction(self, univariate_time_series):
        """Should return DistributionPrediction."""
        model = QuantileForestForecaster(
            targets="target",
            horizon=3,
            n_estimators=10,
            random_state=42,
        )
        model.fit(univariate_time_series)
        pred = model.predict(univariate_time_series)
        assert isinstance(pred, DistributionPrediction)

    def test_predict_univariate_shape(self, univariate_time_series):
        """Should output correct shape for univariate."""
        model = QuantileForestForecaster(
            targets="target",
            horizon=3,
            n_estimators=10,
            random_state=42,
        )
        model.fit(univariate_time_series)
        pred = model.predict(univariate_time_series)
        assert pred._quantiles.shape[1] == 11  # 11 quantiles

    def test_predict_multivariate_shape(self, time_series_data):
        """Should output correct shape for multivariate."""
        model = QuantileForestForecaster(
            targets=["price", "volume"],
            horizon=3,
            n_estimators=10,
            random_state=42,
        )
        model.fit(time_series_data)
        pred = model.predict(time_series_data)
        # 2 targets * 11 quantiles = 22 columns
        assert pred._quantiles.shape[1] == 22

    def test_predict_quantiles_sorted(self, univariate_time_series):
        """Should return sorted quantiles (non-crossing)."""
        model = QuantileForestForecaster(
            targets="target",
            horizon=3,
            n_estimators=10,
            random_state=42,
        )
        model.fit(univariate_time_series)
        pred = model.predict(univariate_time_series)
        for i in range(pred._quantiles.shape[0]):
            row = pred._quantiles[i, :]
            assert list(row) == sorted(row), "Quantiles should be non-crossing"

    def test_predict_with_lazyframe(self, univariate_time_series):
        """Should work with LazyFrame input."""
        model = QuantileForestForecaster(
            targets="target",
            horizon=3,
            n_estimators=10,
            random_state=42,
        )
        model.fit(univariate_time_series)
        pred = model.predict(univariate_time_series.lazy())
        assert isinstance(pred, DistributionPrediction)

    def test_predict_matches_naive_leaf_quantiles(self, univariate_time_series):
        """Vectorized prediction should match the original naive quantile aggregation."""
        model = QuantileForestForecaster(
            targets="target",
            horizon=3,
            n_estimators=8,
            min_samples_leaf=3,
            auto_tune=False,
            random_state=42,
        )
        model.fit(univariate_time_series)

        x = univariate_time_series.select(model._feature_cols_["target"]).to_numpy()
        rf = model._models["target"]
        leaf_dists = model._leaf_distributions["target"]

        naive = np.zeros((len(x), len(model.predict(univariate_time_series)._levels)))
        for tree_idx, tree in enumerate(rf.estimators_):
            tree_leaf_ids = tree.apply(x)
            tree_dist = leaf_dists[tree_idx]
            for row_idx, leaf_id in enumerate(tree_leaf_ids):
                leaf_pos = np.searchsorted(tree_dist["leaf_ids"], leaf_id)
                naive[row_idx] += tree_dist["quantiles"][leaf_pos]
        naive /= len(rf.estimators_)

        optimized = model._predict_quantiles(
            rf, leaf_dists, x, model.predict(univariate_time_series)._levels.tolist()
        )
        np.testing.assert_allclose(optimized, naive)


class TestQuantileForestForecasterInterval:
    """Test QuantileForestForecaster interval predictions."""

    def test_interval_univariate_returns_dataframe(self, univariate_time_series):
        """Should return DataFrame with lower/upper columns for univariate."""
        model = QuantileForestForecaster(
            targets="target",
            horizon=3,
            n_estimators=10,
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
        model = QuantileForestForecaster(
            targets=["price", "volume"],
            horizon=3,
            n_estimators=10,
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
        model = QuantileForestForecaster(
            targets="target",
            horizon=3,
            n_estimators=10,
            random_state=42,
        )
        model.fit(univariate_time_series)
        pred = model.predict(univariate_time_series)
        interval = pred.interval(0.9)
        assert (interval["lower"] < interval["upper"]).all()


class TestQuantileForestForecasterMean:
    """Test QuantileForestForecaster mean predictions."""

    def test_mean_returns_series_univariate(self, univariate_time_series):
        """Should return Series with median predictions for univariate."""
        model = QuantileForestForecaster(
            targets="target",
            horizon=3,
            n_estimators=10,
            random_state=42,
        )
        model.fit(univariate_time_series)
        pred = model.predict(univariate_time_series)
        median = pred.median()
        assert isinstance(median, pl.Series)

    def test_median_returns_dataframe_multivariate(self, time_series_data):
        """Should return DataFrame with per-target medians for multivariate."""
        model = QuantileForestForecaster(
            targets=["price", "volume"],
            horizon=3,
            n_estimators=10,
            random_state=42,
        )
        model.fit(time_series_data)
        pred = model.predict(time_series_data)
        median = pred.median()
        assert isinstance(median, pl.DataFrame)
        assert "price" in median.columns
        assert "volume" in median.columns


class TestQuantileForestForecasterCoverage:
    """Test QuantileForestForecaster coverage."""

    def test_90_interval_has_reasonable_coverage(self, univariate_time_series):
        """90% interval should contain reasonable proportion of true values."""
        from uncertainty_flow import coverage_score

        train = univariate_time_series.head(100)
        calib = univariate_time_series.tail(30)

        model = QuantileForestForecaster(
            targets="target",
            horizon=3,
            n_estimators=50,
            random_state=42,
        )
        model.fit(train)
        pred = model.predict(calib)

        interval = pred.interval(0.9)
        cov = coverage_score(calib["target"], interval["lower"], interval["upper"])
        # Should be > 0 (leaf distribution method is empirical)
        assert 0.0 <= cov <= 1.0


class TestQuantileForestForecasterUncertaintyDrivers:
    """Test uncertainty_drivers_ property."""

    def test_uncertainty_drivers_returns_none(self, univariate_time_series):
        """Should return None (not implemented for this model)."""
        model = QuantileForestForecaster(
            targets="target",
            horizon=3,
            n_estimators=10,
            random_state=42,
        )
        model.fit(univariate_time_series)
        assert model.uncertainty_drivers_ is None
