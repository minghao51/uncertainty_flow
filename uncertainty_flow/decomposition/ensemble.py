"""Refit-based uncertainty decomposition using bootstrap ensembles."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
import polars as pl

from ..utils.exceptions import error_invalid_data
from ..utils.polars_bridge import materialize_lazyframe

if TYPE_CHECKING:
    from ..core.base import BaseUncertaintyModel
    from ..core.distribution import DistributionPrediction
    from ..core.types import PolarsInput, TargetSpec


def _point_prediction_matrix(prediction: "DistributionPrediction") -> np.ndarray:
    """Return point predictions as a 2D array with one column per target."""
    mean_value = prediction.mean()
    if isinstance(mean_value, pl.Series):
        return mean_value.to_numpy().reshape(-1, 1)
    return mean_value.to_numpy()


def _interval_width_matrix(
    prediction: "DistributionPrediction",
    confidence: float,
) -> np.ndarray:
    """Return interval widths as a 2D array with one column per target."""
    interval = prediction.interval(confidence)
    if len(prediction._targets) == 1:
        lower = interval["lower"].to_numpy()
        upper = interval["upper"].to_numpy()
        return (upper - lower).reshape(-1, 1)

    width_columns = []
    for target_name in prediction._targets:
        lower = interval[f"{target_name}_lower"].to_numpy()
        upper = interval[f"{target_name}_upper"].to_numpy()
        width_columns.append(upper - lower)
    return np.column_stack(width_columns)


class EnsembleDecomposition:
    """
    Decompose uncertainty into aleatoric and epistemic components by refitting.

    This workflow fits a bootstrap ensemble from `train_data` using `model_factory`,
    then evaluates all refit members on the same prediction frame:
    - Aleatoric: mean interval width across refit members
    - Epistemic: variance of ensemble point predictions across refit members

    Parameters
    ----------
    model_factory : Callable[[], BaseUncertaintyModel]
        Callable returning a fresh model instance for each bootstrap refit
    train_data : PolarsInput
        Training data used to refit bootstrap ensemble members
    target : str | list[str], optional
        Optional target passed into `fit()` for models that require it
    confidence : float, default=0.9
        Confidence level for interval width calculation
    n_bootstrap : int, default=5
        Number of bootstrap refits
    random_state : int, optional
        Random seed for reproducibility
    """

    def __init__(
        self,
        model_factory: Callable[[], "BaseUncertaintyModel"],
        train_data: "PolarsInput",
        target: "TargetSpec" | None = None,
        confidence: float = 0.9,
        n_bootstrap: int = 5,
        random_state: int | None = None,
    ):
        if not callable(model_factory):
            error_invalid_data("model_factory must be callable")
        if n_bootstrap < 1:
            error_invalid_data(f"n_bootstrap must be at least 1, got {n_bootstrap}")
        if not (0 < confidence < 1):
            error_invalid_data(f"confidence must be in (0, 1), got {confidence}")
        if train_data is None:
            error_invalid_data("train_data is required")

        self.model_factory = model_factory
        self.train_data = materialize_lazyframe(train_data)
        if self.train_data.is_empty():
            error_invalid_data("train_data must contain at least one row")

        self.target = target
        self.confidence = confidence
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)
        self._ensemble_models: list["BaseUncertaintyModel"] | None = None

    def _fit_ensemble(self) -> None:
        """Fit bootstrap-refit ensemble members once and reuse them."""
        if self._ensemble_models is not None:
            return

        n_rows = self.train_data.height
        ensemble_models: list["BaseUncertaintyModel"] = []

        for _ in range(self.n_bootstrap):
            bootstrap_indices = self._rng.choice(n_rows, size=n_rows, replace=True)
            bootstrap_data = self.train_data[bootstrap_indices]
            model = self.model_factory()

            if hasattr(model, "random_state") and self.random_state is not None:
                bootstrap_seed = int(self._rng.integers(0, np.iinfo(np.int32).max))
                try:
                    setattr(model, "random_state", bootstrap_seed)
                except Exception:
                    pass

            if self.target is None:
                model.fit(bootstrap_data)
            else:
                model.fit(bootstrap_data, target=self.target)
            ensemble_models.append(model)

        self._ensemble_models = ensemble_models

    def _predict_ensemble(self, data: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return stacked point predictions and interval widths."""
        self._fit_ensemble()
        assert self._ensemble_models is not None

        point_predictions = []
        interval_widths = []
        for model in self._ensemble_models:
            prediction = model.predict(data)
            point_predictions.append(_point_prediction_matrix(prediction))
            interval_widths.append(_interval_width_matrix(prediction, self.confidence))

        return np.stack(point_predictions, axis=0), np.stack(interval_widths, axis=0)

    def decompose(
        self,
        data: pl.DataFrame,
    ) -> dict[str, float]:
        """
        Decompose prediction uncertainty on an evaluation frame.

        Returns
        -------
        dict[str, float]
            Dictionary with `aleatoric`, `epistemic`, and `total`
        """
        if data.height == 0:
            error_invalid_data("Cannot decompose uncertainty on empty DataFrame")

        point_stack, width_stack = self._predict_ensemble(data)
        aleatoric_by_sample = width_stack.mean(axis=(0, 2))
        epistemic_by_sample = point_stack.var(axis=0).mean(axis=1)

        aleatoric = float(aleatoric_by_sample.mean())
        epistemic = float(epistemic_by_sample.mean())
        return {
            "aleatoric": aleatoric,
            "epistemic": epistemic,
            "total": aleatoric + epistemic,
        }

    def decompose_by_sample(
        self,
        data: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Decompose uncertainty for each sample in an evaluation frame.

        Returns
        -------
        pl.DataFrame
            DataFrame with columns `aleatoric`, `epistemic`, and `total`
        """
        if data.height == 0:
            error_invalid_data("Cannot decompose uncertainty on empty DataFrame")

        point_stack, width_stack = self._predict_ensemble(data)
        aleatoric = width_stack.mean(axis=(0, 2))
        epistemic = point_stack.var(axis=0).mean(axis=1)

        return pl.DataFrame(
            {
                "aleatoric": aleatoric,
                "epistemic": epistemic,
                "total": aleatoric + epistemic,
            }
        )

    def summary(self) -> dict[str, object]:
        """Return configuration summary for the refit ensemble workflow."""
        return {
            "confidence": self.confidence,
            "n_bootstrap": self.n_bootstrap,
            "random_state": self.random_state,
            "target": self.target,
            "refit_based": True,
        }
