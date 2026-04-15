"""Base classes for uncertainty quantification models."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from ..utils.polars_bridge import materialize_lazyframe
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
        data = materialize_lazyframe(data)

        return _calibration_report(self, data, target, quantile_levels)  # type: ignore[arg-type]

    def save(
        self,
        path: str | Path,
        include_metadata: bool = True,
    ) -> None:
        """
        Save the model to a .uf archive.

        Args:
            path: Output archive path.
            include_metadata: Whether to include extended metadata.
        """
        from ._persistence import save_model_archive

        self._metadata = save_model_archive(self, path, include_metadata=include_metadata)

    @classmethod
    def load(cls, path: str | Path) -> "BaseUncertaintyModel":
        """
        Load a model from a .uf archive.

        Args:
            path: Archive path produced by save().

        Returns:
            Loaded model instance.
        """
        from ._persistence import _class_path, load_model_archive

        model, _ = load_model_archive(path)
        if cls is not BaseUncertaintyModel and not isinstance(model, cls):
            raise TypeError(
                f"Loaded archive contains {_class_path(model)}, "
                f"which is not an instance of {_class_path(cls)}."
            )
        return model

    @property
    def metadata(self) -> dict | None:
        """
        Return persisted or derived metadata for the model.

        Returns None for fresh unfitted models with no persisted metadata.
        """
        cached_metadata = getattr(self, "_metadata", None)
        if cached_metadata is not None:
            return cached_metadata

        if not getattr(self, "_fitted", False):
            return None

        from ._persistence import build_metadata

        return build_metadata(self, include_metadata=True)

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
