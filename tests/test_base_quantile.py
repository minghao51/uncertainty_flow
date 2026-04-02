"""Tests for BaseQuantileNeuralNet abstract class."""

import numpy as np
import polars as pl
import pytest

from uncertainty_flow.models.base_quantile import BaseQuantileNeuralNet
from uncertainty_flow.utils.exceptions import ModelNotFittedError


class ConcreteQuantileNet(BaseQuantileNeuralNet):
    """Concrete implementation for testing."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._backend_fit_called = False
        self._backend_predict_called = False

    def _fit_backend(self, x, y, **kwargs):
        """Store that fit was called."""
        self._backend_fit_called = True
        # Store some dummy data
        self._n_features = x.shape[1]
        self._n_samples = x.shape[0]

    def _predict_backend(self, x):
        """Return dummy predictions."""
        self._backend_predict_called = True
        n_samples = x.shape[0]
        n_quantiles = len(self.quantile_levels)
        # Return dummy quantiles that increase with quantile level
        quantile_matrix = np.zeros((n_samples, n_quantiles))
        for i, q in enumerate(self.quantile_levels):
            quantile_matrix[:, i] = q
        return quantile_matrix


class TestBaseQuantileNeuralNet:
    """Test BaseQuantileNeuralNet functionality."""

    def test_init_default_params(self):
        """Should initialize with default parameters."""
        model = ConcreteQuantileNet()
        assert model.hidden_layer_sizes == (100, 50)
        assert len(model.quantile_levels) == 11
        assert model.random_state is None

    def test_init_custom_params(self):
        """Should accept custom parameters."""
        model = ConcreteQuantileNet(
            hidden_layer_sizes=(64, 32),
            quantile_levels=[0.1, 0.5, 0.9],
            random_state=42,
        )
        assert model.hidden_layer_sizes == (64, 32)
        assert model.quantile_levels == [0.1, 0.5, 0.9]
        assert model.random_state == 42

    def test_fit_returns_self(self):
        """fit should return self for method chaining."""
        model = ConcreteQuantileNet()
        df = pl.DataFrame(
            {
                "x1": [1, 2, 3, 4, 5],
                "x2": [2, 4, 6, 8, 10],
                "y": [1.5, 3.5, 5.5, 7.5, 9.5],
            }
        )
        result = model.fit(df, target="y")
        assert result is model

    def test_fit_with_polars(self):
        """fit should work with Polars DataFrame."""
        model = ConcreteQuantileNet()
        df = pl.DataFrame(
            {
                "x1": [1, 2, 3, 4, 5],
                "y": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )
        model.fit(df, target="y")
        assert model._fitted
        assert model._backend_fit_called
        assert model._feature_cols_ == ["x1"]

    def test_fit_with_lazyframe(self):
        """fit should work with LazyFrame."""
        model = ConcreteQuantileNet()
        df = pl.DataFrame(
            {
                "x1": [1, 2, 3, 4, 5],
                "y": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )
        lazy = df.lazy()
        model.fit(lazy, target="y")
        assert model._fitted
        assert model._backend_fit_called

    def test_fit_with_numpy(self):
        """fit should work with numpy arrays."""
        model = ConcreteQuantileNet()
        x = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        model.fit(x, y)
        assert model._fitted
        assert model._backend_fit_called
        assert model._feature_cols_ is None

    def test_fit_numpy_with_polars_target_raises_error(self):
        """fit should raise error if data is numpy but target is not."""
        model = ConcreteQuantileNet()
        x = np.array([[1, 2], [3, 4]])

        with pytest.raises(Exception):  # InvalidDataError
            model.fit(x, "y")

    def test_fit_polars_with_numpy_target_raises_error(self):
        """fit should raise error if data is Polars but target is numpy."""
        model = ConcreteQuantileNet()
        df = pl.DataFrame({"x": [1, 2]})
        y = np.array([1.0, 2.0])

        with pytest.raises(Exception):  # InvalidDataError
            model.fit(df, y)

    def test_predict_before_fit_raises_error(self):
        """predict should raise error if model not fitted."""
        model = ConcreteQuantileNet()
        df = pl.DataFrame({"x": [1, 2, 3]})

        with pytest.raises(ModelNotFittedError):
            model.predict(df)

    def test_predict_returns_distribution_prediction(self):
        """predict should return DistributionPrediction."""
        from uncertainty_flow.core.distribution import DistributionPrediction

        model = ConcreteQuantileNet()
        df = pl.DataFrame(
            {
                "x1": [1, 2, 3, 4, 5],
                "y": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )
        model.fit(df, target="y")
        pred = model.predict(df)

        assert isinstance(pred, DistributionPrediction)
        assert pred._quantiles.shape == (5, 11)  # 5 samples, 11 quantiles

    def test_predict_with_polars(self):
        """predict should work with Polars DataFrame."""
        from uncertainty_flow.core.distribution import DistributionPrediction

        model = ConcreteQuantileNet()
        df = pl.DataFrame(
            {
                "x1": [1, 2, 3, 4, 5],
                "y": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )
        model.fit(df, target="y")
        pred = model.predict(df)

        assert isinstance(pred, DistributionPrediction)
        assert model._backend_predict_called

    def test_predict_with_lazyframe(self):
        """predict should work with LazyFrame."""
        model = ConcreteQuantileNet()
        df = pl.DataFrame(
            {
                "x1": [1, 2, 3, 4, 5],
                "y": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )
        model.fit(df, target="y")
        pred = model.predict(df.lazy())

        assert pred._quantiles.shape == (5, 11)

    def test_predict_with_numpy(self):
        """predict should work with numpy array."""
        model = ConcreteQuantileNet()
        df = pl.DataFrame(
            {
                "x1": [1, 2, 3, 4, 5],
                "y": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )
        model.fit(df, target="y")
        x = np.array([[1], [2], [3], [4], [5]])
        pred = model.predict(x)

        assert pred._quantiles.shape == (5, 11)

    def test_predict_ensures_monotonicity(self):
        """predict should ensure quantiles are monotonically increasing."""
        model = ConcreteQuantileNet()
        df = pl.DataFrame(
            {
                "x1": [1, 2, 3, 4, 5],
                "y": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )
        model.fit(df, target="y")
        pred = model.predict(df)

        # Check that quantiles are sorted for each sample
        for i in range(len(df)):
            quantiles = pred._quantiles[i, :]
            assert all(quantiles[j] <= quantiles[j + 1] for j in range(len(quantiles) - 1))

    def test_predict_sorts_by_quantile_level(self):
        """predict should return quantiles sorted by level."""
        model = ConcreteQuantileNet(quantile_levels=[0.9, 0.1, 0.5])
        df = pl.DataFrame(
            {
                "x1": [1, 2, 3, 4, 5],
                "y": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )
        model.fit(df, target="y")
        pred = model.predict(df)

        # Quantile levels should be sorted
        assert list(pred._levels) == [0.1, 0.5, 0.9]


class TestAbstractMethods:
    """Test that abstract methods must be implemented."""

    def test_cannot_instantiate_base_class(self):
        """BaseQuantileNeuralNet cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseQuantileNeuralNet()

    def test_missing_fit_backend_raises_error(self):
        """Subclass without _fit_backend should raise TypeError."""

        class IncompleteModel(BaseQuantileNeuralNet):
            def _predict_backend(self, x):
                return np.array([[0.1, 0.5, 0.9]])

        with pytest.raises(TypeError):
            IncompleteModel()

    def test_missing_predict_backend_raises_error(self):
        """Subclass without _predict_backend should raise TypeError."""

        class IncompleteModel(BaseQuantileNeuralNet):
            def _fit_backend(self, x, y, **kwargs):
                pass

        with pytest.raises(TypeError):
            IncompleteModel()
