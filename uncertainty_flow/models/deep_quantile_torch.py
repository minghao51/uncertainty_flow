"""DeepQuantileNetPyTorch - Multi-quantile MLP with PyTorch backend."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..core.types import PolarsInput
from ..utils.polars_bridge import to_numpy
from .base_quantile import BaseQuantileNeuralNet

if TYPE_CHECKING:
    pass


class QuantileNetTorch(nn.Module):
    """PyTorch module for multi-quantile regression with shared trunk."""

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: tuple[int, ...],
        n_quantiles: int,
        activation: str = "relu",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_quantiles = n_quantiles

        layers: list[nn.Module] = []
        prev_size = input_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            prev_size = size

        self.trunk = nn.Sequential(*layers)
        self.heads = nn.ModuleList([nn.Linear(prev_size, 1) for _ in range(n_quantiles)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.trunk(x)
        outputs = [head(features) for head in self.heads]
        return torch.cat(outputs, dim=1)

    def get_trunk_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.trunk(x)


class PinballLoss(nn.Module):
    """Pinball loss for quantile regression."""

    def __init__(self, quantile: float):
        super().__init__()
        self.quantile = quantile

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        errors = target.unsqueeze(1) - preds
        loss = torch.where(errors < 0, (self.quantile - 1) * errors, self.quantile * errors)
        return torch.mean(torch.abs(loss))


class MonotonicityLoss(nn.Module):
    """Penalty for quantile crossing violations."""

    def __init__(self):
        super().__init__()

    def forward(self, preds: torch.Tensor) -> torch.Tensor:
        n_samples, n_quantiles = preds.shape
        if n_quantiles <= 1:
            return torch.tensor(0.0, device=preds.device)

        diffs = preds[:, 1:] - preds[:, :-1]
        violations = torch.clamp(-diffs, min=0)
        return torch.mean(violations)


def pinball_loss_numpy(preds: np.ndarray, target: np.ndarray, quantile: float) -> float:
    """Compute pinball loss in numpy for evaluation."""
    errors = target.reshape(-1, 1) - preds
    loss = np.where(errors < 0, (quantile - 1) * errors, quantile * errors)
    return float(np.mean(np.abs(loss)))


class DeepQuantileNetTorch(BaseQuantileNeuralNet):
    """
    Multi-quantile neural network with PyTorch backend.

    Architecture:
        Input → Shared MLP Trunk → Hidden Features → [Head Q0, Head Q1, ...]

    Benefits over sklearn backend:
        - GPU acceleration
        - Monotonicity penalty for non-crossing at training time
        - Custom architectures
        - Foundation for transformer forecasters

    Coverage guarantee: ⚠️ Empirical only
    Non-crossing: ✅ (via post-sort or monotonicity_weight)

    Examples:
        >>> from uncertainty_flow.models import DeepQuantileNetTorch
        >>> import polars as pl
        >>> import numpy as np

        >>> np.random.seed(42)
        >>> df = pl.DataFrame({
        ...     "x1": np.random.randn(100),
        ...     "x2": np.random.randn(100),
        ...     "y": 2 * np.random.randn(100) + 5,
        ... })
        >>> model = DeepQuantileNetTorch(
        ...     hidden_layer_sizes=(64, 32),
        ...     device="cpu",
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
        n_estimators: int = 1,
        epochs: int = 100,
        batch_size: int = 64,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0,
        monotonicity_weight: float = 0.0,
        activation: str = "relu",
        device: str = "auto",
        random_state: int | None = None,
        verbose: bool = False,
    ):
        """
        Initialize DeepQuantileNetTorch.

        Args:
            hidden_layer_sizes: Tuple of hidden layer sizes for the trunk.
            quantile_levels: Quantile levels to predict. Defaults to DEFAULT_QUANTILES.
            n_estimators: Number of ensemble members (for uncertainty in training).
            epochs: Number of training epochs.
            batch_size: Batch size for training.
            learning_rate: Learning rate for optimizer.
            weight_decay: L2 regularization strength.
            monotonicity_weight: Weight for monotonicity penalty (0 = no penalty).
            activation: Activation function ('relu' or 'tanh').
            device: Device for training ('auto', 'cpu', 'cuda').
            random_state: Random seed for reproducibility.
            verbose: Whether to print training progress.
        """
        super().__init__(
            hidden_layer_sizes=hidden_layer_sizes,
            quantile_levels=quantile_levels,
            random_state=random_state,
        )
        self.n_estimators = n_estimators
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.monotonicity_weight = monotonicity_weight
        self.activation = activation
        self.device = self._resolve_device(device)
        self.verbose = verbose

    def _resolve_device(self, device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _set_seed(self, seed: int) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    def _fit_backend(
        self,
        x: np.ndarray,
        y: np.ndarray,
        **kwargs: Any,
    ) -> None:
        """
        Fit the PyTorch backend.

        Args:
            x: Scaled feature matrix.
            y: Target values.
            **kwargs: Additional parameters (unused).
        """
        x_scaled = x.astype(np.float32)

        self._models: list[QuantileNetTorch] = []
        self._train_losses_: list[list[float]] = []

        if self.random_state is not None:
            self._set_seed(self.random_state)

        for estimator_idx in range(self.n_estimators):
            model = QuantileNetTorch(
                input_dim=x_scaled.shape[1],
                hidden_sizes=self.hidden_layer_sizes,
                n_quantiles=len(self.quantile_levels),
                activation=self.activation,
            ).to(self.device)

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )

            train_losses = []
            dataset = TensorDataset(
                torch.from_numpy(x_scaled),
                torch.from_numpy(y.astype(np.float32)),
            )
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            monotonicity_fn = MonotonicityLoss() if self.monotonicity_weight > 0 else None

            for epoch in range(self.epochs):
                model.train()
                epoch_losses = []

                for batch_x, batch_y in loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)

                    optimizer.zero_grad()
                    preds = model(batch_x)

                    quantile_losses = []
                    for q_idx, q in enumerate(self.quantile_levels):
                        loss_fn = PinballLoss(quantile=q)
                        q_loss = loss_fn(preds[:, q_idx : q_idx + 1], batch_y)
                        quantile_losses.append(q_loss)

                    total_loss = sum(quantile_losses) / len(quantile_losses)

                    if monotonicity_fn is not None:
                        mono_loss = monotonicity_fn(preds)
                        total_loss = total_loss + self.monotonicity_weight * mono_loss

                    total_loss.backward()
                    optimizer.step()

                    epoch_losses.append(total_loss.item())

                avg_loss = np.mean(epoch_losses)
                train_losses.append(avg_loss)

                if self.verbose and (epoch + 1) % 10 == 0:
                    msg = (
                        f"  Estimator {estimator_idx + 1}/{self.n_estimators}, "
                        f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.4f}"
                    )
                    print(msg)

            self._models.append(model)
            self._train_losses_.append(train_losses)

    def _predict_backend(self, x: np.ndarray) -> np.ndarray:
        """
        Generate predictions using the PyTorch backend.

        Args:
            x: Scaled feature matrix.

        Returns:
            Quantile matrix (n_samples, n_quantiles).
        """
        x_scaled = x.astype(np.float32)

        dataset = TensorDataset(torch.from_numpy(x_scaled))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        all_preds: list[np.ndarray] = []

        for model in self._models:
            model.eval()
            preds_list = []

            with torch.no_grad():
                for (batch_x,) in loader:
                    batch_x = batch_x.to(self.device)
                    preds = model(batch_x)
                    preds_list.append(preds.cpu().numpy())

            all_preds.append(np.concatenate(preds_list, axis=0))

        ensemble_preds = np.mean(all_preds, axis=0)
        return ensemble_preds

    def pinball_scores(self, data: PolarsInput, target: str) -> dict[float, float]:
        """
        Compute pinball loss for each quantile level on given data.

        Args:
            data: Polars DataFrame with features and target.
            target: Target column name.

        Returns:
            Dictionary mapping quantile levels to pinball loss values.
        """
        if isinstance(data, pl.LazyFrame):
            data = data.collect()

        y = to_numpy(data, [target]).flatten()

        preds = self.predict(data)

        scores = {}
        for q_idx, q in enumerate(self.quantile_levels):
            q_preds = preds._quantiles[:, q_idx]
            scores[q] = pinball_loss_numpy(
                q_preds.reshape(-1, 1),
                y,
                q,
            )

        return scores
