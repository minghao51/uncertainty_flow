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
    Non-crossing: ✅ (via post-prediction sorting, or post-hoc projection
    when ``non_crossing_penalty > 0``)

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
        calibration_size: float = 0.2,
        trunk_alpha: float = 0.0001,
        trunk_max_iter: int = 500,
        head_solver: str = "pinball",
        non_crossing_penalty: float = 0.0,
        random_state: int | None = None,
    ):
        """
        Initialize DeepQuantileNet.

        Args:
            hidden_layer_sizes: Tuple of hidden layer sizes for the trunk MLP.
                E.g., (100, 50) means two hidden layers with 100 and 50 units.
            quantile_levels: Quantile levels to predict. Defaults to DEFAULT_QUANTILES.
            calibration_size: Fraction of data held out as calibration set (0-1).
            trunk_alpha: L2 regularization parameter for the trunk MLP.
            trunk_max_iter: Maximum iterations for the trunk MLP optimizer.
            head_solver: Solver for quantile heads. Currently only "pinball" supported.
            non_crossing_penalty: When > 0, adds a training-time penalty term
                λ Σ max(0, q_{i+1} - q_i)^2 to the joint pinball loss, encouraging
                monotone quantiles during fitting. Higher values enforce stricter
                monotonicity (default 0.0 = disabled).
            random_state: Random seed for reproducibility.
        """
        super().__init__(
            hidden_layer_sizes=hidden_layer_sizes,
            quantile_levels=quantile_levels,
            calibration_size=calibration_size,
            random_state=random_state,
        )
        self.trunk_alpha = trunk_alpha
        self.trunk_max_iter = trunk_max_iter
        self.head_solver = head_solver
        self.non_crossing_penalty = non_crossing_penalty

    def _fit_backend(
        self,
        x: np.ndarray,
        y: np.ndarray,
        **kwargs: Any,
    ) -> None:
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

        if self.non_crossing_penalty > 0:
            self._apply_non_crossing_training_penalty(x, y)

    def _apply_non_crossing_training_penalty(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> None:
        """Jointly refine all quantile heads with a non-crossing penalty.

        Minimises the sum of pinball losses across all quantiles plus a
        penalty term: λ * mean( Σ_i max(0, q_{i+1} - q_i)^2 ) where
        λ = ``self.non_crossing_penalty``.
        """
        from scipy.optimize import minimize

        trunk_features = self._extract_trunk_features(x)
        n_samples, n_features = trunk_features.shape
        n_quantiles = len(self.quantile_levels)
        lam = self.non_crossing_penalty

        # Flatten all head parameters into one vector:
        # [coef_q1 (n_features), intercept_q1, coef_q2 (n_features), intercept_q2, ...]
        def _pack() -> np.ndarray:
            parts = []
            for q in self.quantile_levels:
                parts.append(self._head_coefs_[q])
                parts.append(np.array([self._head_intercepts_[q]]))
            return np.concatenate(parts)

        def _unpack(params: np.ndarray) -> tuple[list[np.ndarray], list[float]]:
            coefs = []
            intercepts = []
            idx = 0
            for _ in self.quantile_levels:
                coefs.append(params[idx : idx + n_features])
                intercepts.append(float(params[idx + n_features]))
                idx += n_features + 1
            return coefs, intercepts

        def _joint_loss(params: np.ndarray) -> float:
            coefs, intercepts = _unpack(params)

            # Compute predictions for all quantiles
            preds = np.empty((n_samples, n_quantiles))
            for j in range(n_quantiles):
                preds[:, j] = trunk_features @ coefs[j] + intercepts[j]

            # Pinball loss for each quantile
            total_pinball = 0.0
            for j, q in enumerate(self.quantile_levels):
                residuals = y - preds[:, j]
                weights = np.where(residuals < 0, q, 1 - q)
                total_pinball += np.sum(weights * np.abs(residuals))

            # Non-crossing penalty: λ * Σ max(0, q_{j+1} - q_j)^2
            crossing_penalty = 0.0
            if n_quantiles > 1:
                diffs = preds[:, 1:] - preds[:, :-1]
                crossing_penalty = lam * np.sum(np.maximum(0, -diffs) ** 2)

            # Small L2 regularisation to keep parameters stable
            l2_penalty = 1e-6 * np.sum(params**2)

            return total_pinball + crossing_penalty + l2_penalty

        x0 = _pack()
        result = minimize(
            _joint_loss,
            x0,
            method="L-BFGS-B",
            options={"maxiter": 500},
        )

        coefs, intercepts = _unpack(result.x)
        for j, q in enumerate(self.quantile_levels):
            self._head_coefs_[q] = coefs[j]
            self._head_intercepts_[q] = intercepts[j]

    def _apply_non_crossing_projection(self, x: np.ndarray) -> None:
        """Post-hoc projection to enforce monotone quantiles on training data."""
        n_iters = max(1, int(self.non_crossing_penalty * 10))
        trunk_features = self._extract_trunk_features(x)

        for _ in range(n_iters):
            predictions = self._predict_backend_raw(trunk_features)
            crossing = np.diff(predictions, axis=1) < 0
            if not np.any(crossing):
                break

            for i in range(1, len(self.quantile_levels)):
                q_curr = self.quantile_levels[i]
                violations = predictions[:, i] < predictions[:, i - 1]
                if not np.any(violations):
                    continue

                correction = (
                    predictions[:, i - 1][violations] - predictions[:, i][violations]
                ) / 2.0
                feat_viol = trunk_features[violations]
                target_corr = np.zeros(feat_viol.shape[0])

                for j in range(feat_viol.shape[1]):
                    grad = feat_viol[:, j]
                    norm_sq = np.dot(grad, grad)
                    if norm_sq < 1e-12:
                        continue
                    step = np.dot(grad, correction - target_corr) / norm_sq
                    self._head_coefs_[q_curr][j] += step * 0.5
                    target_corr += step * 0.5 * grad

                self._head_intercepts_[q_curr] += float(np.mean(correction - target_corr))

    def _predict_backend_raw(self, trunk_features: np.ndarray) -> np.ndarray:
        coef_matrix = np.column_stack([self._head_coefs_[q] for q in self.quantile_levels])
        intercepts = np.array([self._head_intercepts_[q] for q in self.quantile_levels])
        return trunk_features @ coef_matrix + intercepts

    def _predict_backend(self, x: np.ndarray) -> np.ndarray:
        trunk_features = self._extract_trunk_features(x)
        coef_matrix = np.column_stack([self._head_coefs_[q] for q in self.quantile_levels])
        intercepts = np.array([self._head_intercepts_[q] for q in self.quantile_levels])
        return trunk_features @ coef_matrix + intercepts  # type: ignore[no-any-return]

    def _extract_trunk_features(self, x: np.ndarray) -> np.ndarray:
        """
        Extract hidden layer features from the trunk MLP.

        Args:
            x: Scaled input features.

        Returns:
            Hidden layer activations.
        """
        activations = x
        for coef, intercept in zip(self._trunk_.coefs_[:-1], self._trunk_.intercepts_[:-1]):
            activations = np.dot(activations, coef) + intercept
            activations = self._relu(activations)
        return activations

    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation function."""
        return np.maximum(0, x)  # type: ignore[no-any-return]


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
