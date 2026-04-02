"""Base classes for uncertainty quantification models."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import polars as pl

from .types import PolarsInput, TargetSpec

if TYPE_CHECKING:
    from .distribution import DistributionPrediction


class BaseUncertaintyModel(ABC):
    """
    Base class for all uncertainty quantification models.

    All models must implement fit() and predict() methods.
    Calibration reports are provided via default implementation.
    """

    @abstractmethod
    def fit(
        self,
        data: PolarsInput,
        target: TargetSpec | None = None,
        **kwargs,
    ) -> "BaseUncertaintyModel":
        """
        Fit the model to training data.

        Args:
            data: Polars DataFrame or LazyFrame with features and target
            target: Target column name(s) - optional for some models
            **kwargs: Additional model-specific parameters

        Returns:
            self (for method chaining)
        """
        ...

    @abstractmethod
    def predict(self, data: PolarsInput) -> "DistributionPrediction":
        """
        Generate probabilistic predictions.

        Args:
            data: Polars DataFrame or LazyFrame with features

        Returns:
            DistributionPrediction object with quantile predictions
        """
        ...

    def calibration_report(
        self,
        data: PolarsInput,
        target: TargetSpec | None = None,
        quantile_levels: list[float] | None = None,
    ) -> pl.DataFrame:
        """
        Generate calibration diagnostics.

        Default implementation - can be overridden by subclasses.

        Args:
            data: Validation data
            target: Target column name(s) - optional for some models
            quantile_levels: Quantile levels to evaluate (default: [0.8, 0.9, 0.95])

        Returns:
            Polars DataFrame with calibration metrics
        """
        # Lazy import to avoid circular dependency:
        # calibration_utils -> DistributionPrediction -> this module
        from ..utils.calibration_utils import calibration_report as _calibration_report

        # Collect lazyframe if needed
        if isinstance(data, pl.LazyFrame):
            data = data.collect()

        return _calibration_report(self, data, target, quantile_levels)  # type: ignore[arg-type]

    @property
    def uncertainty_drivers_(self) -> pl.DataFrame | None:
        """
        Return residual correlation analysis results.

        Returns None if model has not been fitted.

        Returns:
            Polars DataFrame with feature-residual correlations, or None
        """
        # Default implementation - subclasses should override
        return None
