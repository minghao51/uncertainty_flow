"""Abstract base class for quantile neural network models."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler

from ..core.base import BaseUncertaintyModel
from ..core.distribution import DistributionPrediction
from ..core.types import DEFAULT_QUANTILES, PolarsInput
from ..utils.exceptions import error_invalid_data
from ..utils.polars_bridge import to_numpy

if TYPE_CHECKING:
    pass


class BaseQuantileNeuralNet(BaseUncertaintyModel):
    """
    Abstract base class for quantile neural network models.

    Provides common functionality for neural quantile regression models:
    - Data preparation (Polars and numpy support)
    - Feature scaling with StandardScaler
    - Monotonicity enforcement
    - Consistent fit/predict interface

    Subclasses must implement:
    - _fit_backend(x, y): Backend-specific training logic
    - _predict_backend(x): Backend-specific prediction logic
    """

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (100, 50),
        quantile_levels: list[float] | None = None,
        random_state: int | None = None,
    ):
        """
        Initialize the base quantile neural network.

        Args:
            hidden_layer_sizes: Tuple of hidden layer sizes.
            quantile_levels: Quantile levels to predict. Defaults to DEFAULT_QUANTILES.
            random_state: Random seed for reproducibility.
        """
        self.hidden_layer_sizes = hidden_layer_sizes
        self.quantile_levels = quantile_levels or DEFAULT_QUANTILES
        self.random_state = random_state

        # Fitted attributes
        self._fitted = False
        self._scaler_: StandardScaler
        self._feature_cols_: list[str] | None

    def fit(  # type: ignore[override]
        self,
        data: PolarsInput | np.ndarray,
        target: str | np.ndarray,
        **kwargs: Any,
    ) -> "BaseQuantileNeuralNet":
        """
        Fit the quantile neural network model.

        Args:
            data: Polars DataFrame, LazyFrame, or numpy array with features.
            target: Target column name (if Polars) or numpy array (if data is numpy).
            **kwargs: Additional model-specific parameters.

        Returns:
            self (for method chaining).

        Raises:
            InvalidDataError: If data and target types are mismatched.
        """
        x, y = self._prepare_data(data, target)

        # Fit scaler
        self._scaler_ = StandardScaler()
        x_scaled = self._scaler_.fit_transform(x)

        # Backend-specific training
        self._fit_backend(x_scaled, y, **kwargs)

        self._fitted = True
        return self

    def predict(self, data: PolarsInput | np.ndarray) -> DistributionPrediction:
        """
        Generate probabilistic predictions.

        Args:
            data: Polars DataFrame, LazyFrame, or numpy array with features.

        Returns:
            DistributionPrediction with quantile forecasts.

        Raises:
            ModelNotFittedError: If model has not been fitted.
        """
        from ..utils.exceptions import error_model_not_fitted

        if not self._fitted:
            error_model_not_fitted(self.__class__.__name__)

        x = self._prepare_predict_data(data)
        x_scaled = self._scaler_.transform(x)

        # Backend-specific prediction
        quantile_matrix = self._predict_backend(x_scaled)

        # Ensure monotonicity
        quantile_matrix = self._ensure_monotonicity(quantile_matrix)

        # Sort by quantile levels
        sorted_levels = np.array(self.quantile_levels)
        sorted_indices = np.argsort(sorted_levels)
        quantile_matrix = quantile_matrix[:, sorted_indices]
        sorted_levels_list = [self.quantile_levels[i] for i in sorted_indices]

        return DistributionPrediction(
            quantile_matrix=quantile_matrix,
            quantile_levels=sorted_levels_list,
            target_names=["y"],
        )

    def _prepare_data(
        self,
        data: PolarsInput | np.ndarray,
        target: str | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data from various input formats.

        Args:
            data: Polars DataFrame, LazyFrame, or numpy array with features.
            target: Target column name (if Polars) or numpy array (if data is numpy).

        Returns:
            Tuple of (x, y) as numpy arrays.

        Raises:
            InvalidDataError: If data and target types are mismatched.
        """
        if isinstance(data, np.ndarray):
            if not isinstance(target, np.ndarray):
                error_invalid_data("If data is numpy array, target must also be numpy array")
            x = data
            y = target.flatten() if isinstance(target, np.ndarray) else target
            self._feature_cols_ = None
        else:
            if isinstance(data, pl.LazyFrame):
                data = data.collect()
            if isinstance(target, np.ndarray):
                error_invalid_data("If data is Polars, target must be string column name")
            target_str = str(target)  # type: ignore[arg-type]  # target is str here
            self._feature_cols_ = [col for col in data.columns if col != target_str]
            assert self._feature_cols_ is not None
            x = to_numpy(data, self._feature_cols_)
            y = to_numpy(data, [target_str]).flatten()

        return x, y  # type: ignore[return-value]

    def _prepare_predict_data(self, data: PolarsInput | np.ndarray) -> np.ndarray:
        """
        Prepare prediction data from various input formats.

        Args:
            data: Polars DataFrame, LazyFrame, or numpy array with features.

        Returns:
            x as numpy array.
        """
        if isinstance(data, np.ndarray):
            x = data
        else:
            if isinstance(data, pl.LazyFrame):
                data = data.collect()
            if self._feature_cols_ is None:
                error_invalid_data("Feature columns not set. Call fit() first.")
            x = to_numpy(data, self._feature_cols_)  # type: ignore[arg-type]

        return x

    def _ensure_monotonicity(self, quantile_matrix: np.ndarray) -> np.ndarray:
        """
        Ensure quantile monotonicity by post-sorting each prediction.

        Args:
            quantile_matrix: Raw quantile predictions (n_samples, n_quantiles).

        Returns:
            Monotonic quantile matrix.
        """
        n_samples = len(quantile_matrix)
        for i in range(n_samples):
            sorted_indices = np.argsort(quantile_matrix[i, :])
            quantile_matrix[i, :] = quantile_matrix[i, sorted_indices]

        return quantile_matrix

    @abstractmethod
    def _fit_backend(
        self,
        x: np.ndarray,
        y: np.ndarray,
        **kwargs: Any,
    ) -> None:
        """
        Backend-specific training logic.

        Args:
            x: Scaled feature matrix.
            y: Target values.
            **kwargs: Additional backend-specific parameters.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError("Subclasses must implement _fit_backend")

    @abstractmethod
    def _predict_backend(self, x: np.ndarray) -> np.ndarray:
        """
        Backend-specific prediction logic.

        Args:
            x: Scaled feature matrix.

        Returns:
            Quantile matrix (n_samples, n_quantiles).

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError("Subclasses must implement _predict_backend")
