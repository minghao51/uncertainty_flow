"""Base classes for uncertainty quantification models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl

from ..utils.polars_bridge import materialize_lazyframe
from .types import PolarsInput, TargetSpec

if TYPE_CHECKING:
    from .distribution import DistributionPrediction


def _base_calibration_report(
    model: BaseUncertaintyModel,
    data: pl.DataFrame,
    target: str | list[str] | None,
    quantile_levels: list[float] | None = None,
) -> pl.DataFrame:
    from ..utils.calibration_utils import calibration_report as _calibration_report

    return _calibration_report(model, data, target, quantile_levels)  # type: ignore[arg-type]


def _base_explain_interval_width(
    model: BaseUncertaintyModel,
    X: pl.DataFrame,  # noqa: N803
    background: pl.DataFrame | None = None,
    quantile_pairs: list[tuple[float, float]] | None = None,
) -> pl.DataFrame:
    from ..calibration.shap_values import uncertainty_shap

    return uncertainty_shap(model, X, background=background, quantile_pairs=quantile_pairs)


def _base_analyze_leverage(
    model: BaseUncertaintyModel,
    X: pl.DataFrame,  # noqa: N803
    **kwargs,
) -> pl.DataFrame:
    from ..analysis.leverage import FeatureLeverageAnalyzer

    analyzer = FeatureLeverageAnalyzer(model, **kwargs)
    return analyzer.analyze(X)


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
        **kwargs: Any,
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

    def predict_batch(
        self,
        data: PolarsInput,
        batch_size: int = 1000,
    ) -> Iterator["DistributionPrediction"]:
        """
        Generate probabilistic predictions in chunks.

        Default implementation slices the data into batches and yields a
        ``DistributionPrediction`` per batch.  Models with native batch
        / GPU support (e.g. torch) should override this.

        Args:
            data: Polars DataFrame or LazyFrame with features
            batch_size: Number of rows per batch (default 1000).

        Yields:
            DistributionPrediction for each chunk.
        """
        data = materialize_lazyframe(data)
        n = len(data)

        for start in range(0, n, batch_size):
            chunk = data[start : start + batch_size]
            yield self.predict(chunk)

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
        data = materialize_lazyframe(data)
        return _base_calibration_report(self, data, target, quantile_levels)

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
    def load(
        cls,
        path: str | Path,
        *,
        expected_archive_sha256: str | None = None,
    ) -> "BaseUncertaintyModel":
        """
        Load a model from a .uf archive.

        Args:
            path: Archive path produced by save().
            expected_archive_sha256: Optional SHA-256 hex digest expected for the archive.
                When provided, load() fails if the on-disk archive digest does not match.

        Returns:
            Loaded model instance.
        """
        from ._persistence import _class_path, load_model_archive

        model, _ = load_model_archive(path, expected_archive_sha256=expected_archive_sha256)
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
        return None

    def explain_interval_width(
        self,
        X: PolarsInput,  # noqa: N803
        background: PolarsInput | None = None,
        quantile_pairs: list[tuple[float, float]] | None = None,
    ) -> pl.DataFrame:
        """
        Compute SHAP values for quantile interval widths.

        Identifies which features drive prediction interval width.
        Thin wrapper around :func:`uncertainty_shap`.

        Subclasses with native feature importance (e.g. quantile forests)
        may override this with a faster implementation.

        Args:
            X: Feature DataFrame to explain.
            background: Background dataset for SHAP. Defaults to ``X[:100]``.
            quantile_pairs: ``(lower, upper)`` quantile pairs to analyse.
                Defaults to ``[(0.1, 0.9), (0.05, 0.95)]``.

        Returns:
            Polars DataFrame with SHAP attributions per feature.
        """
        X = materialize_lazyframe(X)  # noqa: N806
        if background is not None:
            background = materialize_lazyframe(background)

        return _base_explain_interval_width(
            self, X, background=background, quantile_pairs=quantile_pairs
        )

    def analyze_leverage(
        self,
        X: PolarsInput,  # noqa: N803
        **kwargs: Any,
    ) -> pl.DataFrame:
        """
        Analyze which features most influence prediction uncertainty.

        Thin wrapper around :class:`FeatureLeverageAnalyzer`.

        Args:
            X: Feature DataFrame for leverage analysis.
            **kwargs: Forwarded to ``FeatureLeverageAnalyzer`` (e.g.
                ``confidence``, ``n_perturbations``, ``random_state``).

        Returns:
            Polars DataFrame with leverage scores per feature.
        """
        X = materialize_lazyframe(X)  # noqa: N806
        return _base_analyze_leverage(self, X, **kwargs)
