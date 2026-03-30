"""DeepQuantileNet - Multi-quantile MLP with shared trunk (sklearn backend)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.base import RegressorMixin
from sklearn.neural_network import MLPRegressor

from .base_quantile import BaseQuantileNeuralNet

if TYPE_CHECKING:
    pass


class DeepQuantileNet(BaseQuantileNeuralNet, RegressorMixin):
    """
    Multi-quantile neural network with shared trunk architecture (sklearn backend).

    Architecture:
        Input → Shared MLP Trunk → Hidden Features → [Linear Head Q0, Linear Head Q1, ...]

    The shared trunk is implemented by extracting the hidden layer representation
    from a median-trained MLP and using it as features for linear quantile heads.

    Coverage guarantee: ⚠️ Empirical only
    Non-crossing: ✅ (via post-prediction sorting)

    Examples:
        >>> from uncertainty_flow.models import DeepQuantileNet
        >>> import polars as pl
        >>> import numpy as np

        >>> np.random.seed(42)
        >>> df = pl.DataFrame({
        ...     "x1": np.random.randn(100),
        ...     "x2": np.random.randn(100),
        ...     "y": 2 * np.random.randn(100) + 5,
        ... })
        >>> model = DeepQuantileNet(
        ...     hidden_layer_sizes=(64, 32),
        ...     random_state=42,
        ... )
        >>> model.fit(df, target="y")
        >>> pred = model.predict(df)
        >>> pred.interval(0.9)
    """

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (100, 50),
        quantile_levels: list[float] | None = None,
        trunk_alpha: float = 0.0001,
        trunk_max_iter: int = 500,
        head_solver: str = "pinball",
        random_state: int | None = None,
    ):
        """
        Initialize DeepQuantileNet.

        Args:
            hidden_layer_sizes: Tuple of hidden layer sizes for the trunk MLP.
                E.g., (100, 50) means two hidden layers with 100 and 50 units.
            quantile_levels: Quantile levels to predict. Defaults to DEFAULT_QUANTILES.
            trunk_alpha: L2 regularization parameter for the trunk MLP.
            trunk_max_iter: Maximum iterations for the trunk MLP optimizer.
            head_solver: Solver for quantile heads. Currently only "pinball" supported.
            random_state: Random seed for reproducibility.
        """
        super().__init__(
            hidden_layer_sizes=hidden_layer_sizes,
            quantile_levels=quantile_levels,
            random_state=random_state,
        )
        self.trunk_alpha = trunk_alpha
        self.trunk_max_iter = trunk_max_iter
        self.head_solver = head_solver

    def _fit_backend(
        self,
        x: np.ndarray,
        y: np.ndarray,
        **kwargs: Any,
    ) -> None:
        """
        Fit the sklearn backend.

        Args:
            x: Scaled feature matrix.
            y: Target values.
            **kwargs: Additional parameters (unused).
        """
        self._trunk_ = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation="relu",
            solver="adam",
            alpha=self.trunk_alpha,
            max_iter=self.trunk_max_iter,
            random_state=self.random_state,
        )
        self._trunk_.fit(x, y)

        self._trunk_features_ = self._extract_trunk_features(x)

        self._head_coefs_: dict[float, np.ndarray] = {}
        self._head_intercepts_: dict[float, float] = {}

        for q in self.quantile_levels:
            head = LinearQuantileHead(solver=self.head_solver)
            head.fit(self._trunk_features_, y, quantile=q)
            assert head.coef_ is not None
            assert head.intercept_ is not None
            self._head_coefs_[q] = head.coef_
            self._head_intercepts_[q] = head.intercept_

    def _predict_backend(self, x: np.ndarray) -> np.ndarray:
        """
        Generate predictions using the sklearn backend.

        Args:
            x: Scaled feature matrix.

        Returns:
            Quantile matrix (n_samples, n_quantiles).
        """
        trunk_features = self._extract_trunk_features(x)

        n_samples = len(x)
        n_quantiles = len(self.quantile_levels)
        quantile_matrix = np.zeros((n_samples, n_quantiles))

        for q_idx, q in enumerate(self.quantile_levels):
            coef = self._head_coefs_[q]
            intercept = self._head_intercepts_[q]
            quantile_matrix[:, q_idx] = trunk_features @ coef + intercept

        return quantile_matrix

    def _extract_trunk_features(self, x: np.ndarray) -> np.ndarray:
        """
        Extract hidden layer features from the trunk MLP.

        Args:
            x: Scaled input features.

        Returns:
            Hidden layer activations.
        """
        activations = x
        for i in range(len(self._trunk_.coefs_) - 1):
            activations = np.dot(activations, self._trunk_.coefs_[i]) + self._trunk_.intercepts_[i]
            activations = self._relu(activations)
        return activations

    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation function."""
        return np.maximum(0, x)


class LinearQuantileHead:
    """
    Linear quantile regression head.

    Solves: min_w sum_i rho_q(y_i - x_i @ w) + alpha * ||w||^2

    where rho_q(u) = u * (q - I(u < 0)) is the pinball loss.
    """

    def __init__(self, solver: str = "pinball", alpha: float = 0.0001):
        self.solver = solver
        self.alpha = alpha
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        quantile: float = 0.5,
    ) -> "LinearQuantileHead":
        """
        Fit the linear quantile head.

        Uses iterative weighted least squares (scipy.optimize.minimize).

        Args:
            x: Feature matrix.
            y: Target values.
            quantile: Quantile level (0 < q < 1).

        Returns:
            self.
        """
        from scipy.optimize import minimize

        n_features = x.shape[1]

        def pinball_loss(w):
            residuals = y - (x @ w[:-1] + w[-1])
            weights = np.where(residuals < 0, quantile, 1 - quantile)
            loss = np.sum(weights * np.abs(residuals))
            penalty = self.alpha * np.sum(w**2)
            return loss + penalty

        w0 = np.zeros(n_features + 1)
        result = minimize(
            pinball_loss,
            w0,
            method="L-BFGS-B",
            options={"maxiter": 500},
        )

        self.coef_ = result.x[:-1]
        self.intercept_ = result.x[-1]

        return self
