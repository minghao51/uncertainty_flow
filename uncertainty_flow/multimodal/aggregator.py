"""CrossModalAggregator - combine predictions from multiple feature groups."""

import copy

import numpy as np
import polars as pl

from ..core.base import BaseUncertaintyModel
from ..core.distribution import DistributionPrediction
from ..core.types import PolarsInput, TargetSpec
from ..utils.exceptions import error_model_not_fitted

VALID_AGGREGATIONS = ("product", "copula", "independent")


class CrossModalAggregator(BaseUncertaintyModel):
    """Train per-group models and combine their predictions.

    Each feature group is trained independently using the same base model
    (cloned per group). Predictions are aggregated using the chosen strategy.

    Args:
        feature_groups: Mapping of group name to list of feature column names.
        aggregation: Aggregation strategy - one of "product", "copula",
            "independent".
        random_state: Random seed (forwarded to cloned models where supported).

    Examples:
        >>> from uncertainty_flow.multimodal import CrossModalAggregator
        >>> groups = {"numeric": ["x1", "x2"], "lag": ["lag_1"]}
        >>> agg = CrossModalAggregator(feature_groups=groups, aggregation="independent")
    """

    def __init__(
        self,
        feature_groups: dict[str, list[str]],
        aggregation: str = "product",
        random_state: int | None = None,
    ):
        if aggregation not in VALID_AGGREGATIONS:
            raise ValueError(
                f"Invalid aggregation '{aggregation}'. Must be one of {VALID_AGGREGATIONS}"
            )
        if not feature_groups:
            raise ValueError("feature_groups cannot be empty")

        self.feature_groups = feature_groups
        self.aggregation = aggregation
        self.random_state = random_state

        self._fitted = False
        self._group_models: dict = {}
        self._quantile_levels: list[float] | None = None
        self._target_name: str = ""

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        data: PolarsInput,
        target: TargetSpec | None = None,
        *,
        base_model=None,
        **kwargs,
    ) -> "CrossModalAggregator":
        """Fit a cloned base model for each feature group.

        Args:
            data: Polars DataFrame or LazyFrame with features and target.
            target: Target column name.
            base_model: An sklearn-compatible estimator (e.g. ConformalRegressor)
                to clone for each group. Required.
            **kwargs: Ignored.

        Returns:
            self

        Raises:
            ValueError: If base_model is not provided or target is missing.
        """
        if isinstance(data, pl.LazyFrame):
            data = data.collect()

        if base_model is None:
            raise ValueError("base_model is required for CrossModalAggregator.fit()")
        if target is None:
            raise ValueError("target is required for CrossModalAggregator.fit()")

        target_str = target if isinstance(target, str) else target[0]
        self._target_name = target_str

        self._group_models = {}
        for group_name, feature_cols in self.feature_groups.items():
            model_clone = copy.deepcopy(base_model)
            select_cols = feature_cols + [target_str]
            group_data = data.select(select_cols)
            model_clone.fit(group_data, target=target_str, **kwargs)
            self._group_models[group_name] = model_clone

            # Capture quantile levels from the first prediction
            if self._quantile_levels is None:
                sample_pred = model_clone.predict(group_data.head(2).select(feature_cols))
                self._quantile_levels = list(sample_pred._levels)

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, data: PolarsInput) -> DistributionPrediction:
        """Generate aggregated predictions across all feature groups.

        Args:
            data: Polars DataFrame or LazyFrame with features.

        Returns:
            DistributionPrediction with group_predictions populated.

        Raises:
            ModelNotFittedError: If called before fit().
        """
        if not self._fitted:
            error_model_not_fitted("CrossModalAggregator")

        if isinstance(data, pl.LazyFrame):
            data = data.collect()

        group_preds: dict[str, DistributionPrediction] = {}
        for group_name, feature_cols in self.feature_groups.items():
            model = self._group_models[group_name]
            group_data = data.select(feature_cols)
            group_preds[group_name] = model.predict(group_data)

        aggregated = self._aggregate(group_preds)

        return DistributionPrediction(
            quantile_matrix=aggregated,
            quantile_levels=self._quantile_levels,
            target_names=[self._target_name],
            group_predictions=group_preds,
        )

    # ------------------------------------------------------------------
    # Aggregation strategies
    # ------------------------------------------------------------------

    def _aggregate(self, group_preds: dict[str, DistributionPrediction]) -> np.ndarray:
        """Dispatch to the chosen aggregation strategy."""
        if self.aggregation == "product":
            return self._aggregate_product(group_preds)
        elif self.aggregation == "copula":
            return self._aggregate_independent(group_preds)
        else:  # independent
            return self._aggregate_independent(group_preds)

    @staticmethod
    def _aggregate_product(group_preds: dict[str, DistributionPrediction]) -> np.ndarray:
        """Product aggregation: average medians + average deviations, then sort."""
        matrices = [p._quantiles for p in group_preds.values()]
        n_samples = matrices[0].shape[0]
        n_quantiles = matrices[0].shape[1]

        # Find the median index (closest to 0.5)
        levels = list(group_preds.values())[0]._levels
        median_idx = int(np.argmin(np.abs(levels - 0.5)))

        medians = np.column_stack([m[:, median_idx] for m in matrices])
        avg_median = np.mean(medians, axis=1)

        # Average deviations from each group's median
        deviation_sum = np.zeros((n_samples, n_quantiles))
        for m in matrices:
            deviation_sum += m - m[:, median_idx : median_idx + 1]
        avg_deviations = deviation_sum / len(matrices)

        result = avg_median[:, np.newaxis] + avg_deviations

        # Sort each row to ensure non-crossing quantiles
        result = np.sort(result, axis=1)
        return result

    @staticmethod
    def _aggregate_independent(group_preds: dict[str, DistributionPrediction]) -> np.ndarray:
        """Independent aggregation: simple average of quantile matrices."""
        matrices = [p._quantiles for p in group_preds.values()]
        result = np.mean(matrices, axis=0)
        # Sort rows to maintain non-crossing property
        result = np.sort(result, axis=1)
        return result
