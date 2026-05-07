"""Tests for EnsembleBootstrapPI."""

import numpy as np
import polars as pl
import pytest
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from uncertainty_flow.utils.exceptions import ModelNotFittedError
from uncertainty_flow.wrappers import EnsembleBootstrapPI


@pytest.fixture
def small_df():
    np.random.seed(42)
    return pl.DataFrame(
        {
            "x1": np.random.randn(60),
            "x2": np.random.randn(60),
            "y": np.random.randn(60),
        }
    )


class TestEnsembleBootstrapPI:
    def test_fit_predict_univariate(self, small_df):
        model = EnsembleBootstrapPI(
            base_model=LinearRegression(),
            n_estimators=10,
            random_state=42,
        )
        model.fit(small_df, target="y")
        pred = model.predict(small_df)
        assert pred._n_samples == 60
        assert len(pred._targets) == 1
        interval = pred.interval(0.9)
        assert "lower" in interval.columns
        assert "upper" in interval.columns

    def test_update_after_predict(self, small_df):
        model = EnsembleBootstrapPI(
            base_model=LinearRegression(),
            n_estimators=10,
            random_state=42,
        )
        pred = model.fit(small_df, target="y").predict(small_df)
        n_before = len(model._scores)
        model.update(pred.median().to_numpy().ravel())
        assert len(model._scores) == n_before + len(small_df)

    def test_update_without_predict_raises(self, small_df):
        model = EnsembleBootstrapPI(
            base_model=LinearRegression(),
            n_estimators=10,
            random_state=42,
        )
        model.fit(small_df, target="y")
        with pytest.raises(RuntimeError, match="predict"):
            model.update(0.5)

    def test_invalid_params(self):
        with pytest.raises(ValueError, match="n_estimators"):
            EnsembleBootstrapPI(LinearRegression(), n_estimators=1)
        with pytest.raises(ValueError, match="coverage_target"):
            EnsembleBootstrapPI(LinearRegression(), coverage_target=1.5)
        with pytest.raises(ValueError, match="subsample_ratio"):
            EnsembleBootstrapPI(LinearRegression(), subsample_ratio=2.0)

    def test_not_fitted_error(self, small_df):
        model = EnsembleBootstrapPI(LinearRegression(), n_estimators=5)
        with pytest.raises(ModelNotFittedError):
            model.predict(small_df)

    def test_summary_decomposition(self, small_df):
        model = EnsembleBootstrapPI(
            base_model=GradientBoostingRegressor(random_state=42),
            n_estimators=10,
            random_state=42,
        )
        model.fit(small_df, target="y")
        pred = model.predict(small_df)
        summary = pred.summary()
        assert "median" in summary.columns
        assert "mean_width_90" in summary.columns
        assert summary.shape[0] == 1

    def test_batch_integration(self, small_df):
        model = EnsembleBootstrapPI(
            base_model=LinearRegression(),
            n_estimators=5,
            random_state=42,
        )
        model.fit(small_df, target="y")
        count = 0
        for chunk in model.predict_batch(small_df, batch_size=20):
            count += 1
            assert chunk._n_samples <= 20
        assert count == 3

    def test_fit_requires_target(self, small_df):
        model = EnsembleBootstrapPI(
            base_model=LinearRegression(),
            n_estimators=5,
            random_state=42,
        )
        with pytest.raises(Exception, match="target is required"):
            model.fit(small_df)

    def test_fit_missing_target_column(self):
        df = pl.DataFrame({"x1": [1.0, 2.0], "x2": [3.0, 4.0]})
        model = EnsembleBootstrapPI(
            base_model=LinearRegression(),
            n_estimators=5,
            random_state=42,
        )
        with pytest.raises(ValueError, match="not found"):
            model.fit(df, target="missing")

    def test_fit_no_features(self):
        df = pl.DataFrame({"y": [1.0, 2.0, 3.0]})
        model = EnsembleBootstrapPI(
            base_model=LinearRegression(),
            n_estimators=5,
            random_state=42,
        )
        with pytest.raises(ValueError, match="No feature columns"):
            model.fit(df, target="y")

    def test_update_single_scalar(self, small_df):
        model = EnsembleBootstrapPI(
            base_model=LinearRegression(),
            n_estimators=10,
            random_state=42,
        )
        model.fit(small_df, target="y")
        model.predict(small_df[:3])
        with pytest.raises(ValueError, match="must match"):
            model.update(0.5)

    def test_subsample_ratio(self):
        model = EnsembleBootstrapPI(
            base_model=LinearRegression(),
            n_estimators=5,
            subsample_ratio=0.5,
            random_state=42,
        )
        assert model.subsample_ratio == 0.5

    def test_update_length_mismatch_raises(self, small_df):
        model = EnsembleBootstrapPI(
            base_model=LinearRegression(),
            n_estimators=10,
            random_state=42,
        )
        model.fit(small_df, target="y")
        model.predict(small_df[:5])
        with pytest.raises(ValueError, match="must match"):
            model.update(np.array([1.0, 2.0]))

    def test_higher_coverage_target_gives_wider_intervals(self, small_df):
        low = EnsembleBootstrapPI(
            base_model=LinearRegression(),
            n_estimators=10,
            coverage_target=0.8,
            random_state=42,
        )
        high = EnsembleBootstrapPI(
            base_model=LinearRegression(),
            n_estimators=10,
            coverage_target=0.95,
            random_state=42,
        )
        low.fit(small_df, target="y")
        high.fit(small_df, target="y")
        low_pred = low.predict(small_df)
        high_pred = high.predict(small_df)
        low_width = (low_pred.interval(0.9)["upper"] - low_pred.interval(0.9)["lower"]).mean()
        high_width = (high_pred.interval(0.9)["upper"] - high_pred.interval(0.9)["lower"]).mean()
        assert high_width >= low_width
