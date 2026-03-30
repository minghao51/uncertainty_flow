"""QuantileForestForecaster - Quantile Regression Forest for time series."""

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestRegressor

from ..core.base import BaseUncertaintyModel
from ..core.distribution import DistributionPrediction
from ..core.types import DEFAULT_QUANTILES, PolarsInput, TargetSpec
from ..utils.exceptions import error_model_not_fitted
from ..utils.polars_bridge import to_numpy
from ..utils.split import TemporalHoldoutSplit

if TYPE_CHECKING:
    pass


class QuantileForestForecaster(BaseUncertaintyModel):
    """
    Quantile Regression Forest for time series.

    Stores full leaf distributions to compute true quantiles
    (not just split conformal like wrappers).

    Coverage guarantee: ⚠️ Empirical only
    Non-crossing: ✅ (by leaf distribution construction)

    Examples:
        >>> from uncertainty_flow.models import QuantileForestForecaster
        >>> import polars as pl
        >>>
        >>> df = pl.DataFrame({
        ...     "date": range(10),
        ...     "price": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        ... })
        >>> model = QuantileForestForecaster(
        ...     targets="price",
        ...     horizon=3,
        ...     random_state=42,
        ... )
        >>> model.fit(df)
        >>> pred = model.predict(df)
    """

    def __init__(
        self,
        targets: str | list[str],
        horizon: int,
        n_estimators: int = 200,
        min_samples_leaf: int = 5,
        max_depth: int | None = None,
        copula_family: str = "auto",
        calibration_size: float = 0.2,
        uncertainty_features: list[str] | None = None,
        random_state: int | None = None,
    ):
        """
        Initialize QuantileForestForecaster.

        Args:
            targets: Target column name(s)
            horizon: Forecast horizon
            n_estimators: Number of trees in the forest
            min_samples_leaf: Minimum samples per leaf (controls distribution richness)
            max_depth: Maximum tree depth
            copula_family: (
                "auto" (BIC selection) or one of "gaussian", "clayton", "gumbel", "frank". "
                "Use "independent" for no inter-target correlation."
            )
            calibration_size: Fraction for calibration (from END)
            uncertainty_features: Optional hint for heteroscedastic features
            random_state: Random seed
        """
        self.targets = [targets] if isinstance(targets, str) else targets
        self.horizon = horizon
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.copula_family = copula_family
        self.calibration_size = calibration_size
        self.uncertainty_features = uncertainty_features
        self.random_state = random_state

        # Fitted attributes
        self._fitted = False
        self._models: dict[str, RandomForestRegressor] = {}
        self._leaf_distributions: dict[str, list] = {}
        self._feature_cols_: dict[str, list[str]] = {}
        self._uncertainty_drivers_: pl.DataFrame | None = None

    def fit(
        self,
        data: PolarsInput,
        target: TargetSpec | None = None,
        **kwargs,
    ) -> "QuantileForestForecaster":
        """
        Fit the quantile forest forecaster.

        Args:
            data: Polars DataFrame or LazyFrame with time series
            target: Target column name(s) - uses self.targets if not provided
            **kwargs: Additional parameters (unused)

        Returns:
            self (for method chaining)
        """
        # Materialize if needed
        if isinstance(data, pl.LazyFrame):
            data = data.collect()

        # Temporal split
        splitter = TemporalHoldoutSplit()
        train, calib = splitter.split(data, self.calibration_size)

        # Fit per target
        for target in self.targets:
            # Get feature columns
            feature_cols = [col for col in train.columns if col != target]
            self._feature_cols_[target] = feature_cols

            x_train = to_numpy(train, feature_cols)
            y_train = to_numpy(train, [target]).flatten()

            rf = RandomForestRegressor(
                n_estimators=self.n_estimators,
                min_samples_leaf=self.min_samples_leaf,
                max_depth=self.max_depth,
                random_state=self.random_state,
            )
            rf.fit(x_train, y_train)
            self._models[target] = rf

            self._leaf_distributions[target] = self._extract_leaf_distributions(
                rf, x_train, y_train
            )

        self._fitted = True
        return self

    def _extract_leaf_distributions(
        self,
        rf: RandomForestRegressor,
        x: np.ndarray,
        y: np.ndarray,
    ) -> list:
        """
        Extract training values that fall into each leaf.

        Args:
            rf: Fitted random forest
            X: Feature matrix
            y: Target values

        Returns:
            List of dicts mapping leaf_id to leaf values
        """
        distributions = []

        for tree in rf.estimators_:
            leaf_ids = tree.apply(x)
            unique_leaves = np.unique(leaf_ids)

            tree_dist = {}
            for leaf_id in unique_leaves:
                leaf_mask = leaf_ids == leaf_id
                tree_dist[int(leaf_id)] = y[leaf_mask]

            distributions.append(tree_dist)

        return distributions

    def _predict_quantiles(
        self,
        rf: RandomForestRegressor,
        leaf_dists: list,
        x: np.ndarray,
        quantile_levels: list[float],
    ) -> np.ndarray:
        """
        Predict quantiles from leaf distributions.

        Args:
            rf: Fitted random forest
            leaf_dists: Leaf distributions from training
            X: Feature matrix
            quantile_levels: Quantile levels to compute

        Returns:
            Quantile predictions shape (n_samples, n_quantiles)
        """
        n_samples = len(x)
        n_quantiles = len(quantile_levels)
        predictions = np.zeros((n_samples, n_quantiles))

        for tree_idx, tree in enumerate(rf.estimators_):
            leaf_ids = tree.apply(x)
            tree_dist = leaf_dists[tree_idx]

            for i in range(n_samples):
                leaf_id = int(leaf_ids[i])
                if leaf_id in tree_dist:
                    leaf_values = tree_dist[leaf_id]
                    # Compute quantiles for this leaf
                    for q_idx, q in enumerate(quantile_levels):
                        predictions[i, q_idx] += np.quantile(leaf_values, q)

        # Average across trees
        predictions /= len(rf.estimators_)

        return predictions

    def predict(self, data: PolarsInput) -> DistributionPrediction:
        """
        Generate probabilistic forecasts.

        Args:
            data: Polars DataFrame or LazyFrame

        Returns:
            DistributionPrediction with quantile forecasts
        """
        if not self._fitted:
            error_model_not_fitted("QuantileForestForecaster")

        # Materialize if needed
        if isinstance(data, pl.LazyFrame):
            data = data.collect()

        # Get predictions for each target
        all_quantiles = []

        for target in self.targets:
            x = to_numpy(data, self._feature_cols_[target])
            rf = self._models[target]
            leaf_dists = self._leaf_distributions[target]

            quantile_matrix = self._predict_quantiles(rf, leaf_dists, x, DEFAULT_QUANTILES)

            all_quantiles.append(quantile_matrix)

        # Stack for multivariate
        if len(self.targets) == 1:
            final_matrix = all_quantiles[0]
        else:
            final_matrix = np.column_stack(
                [
                    all_quantiles[t][:, i]
                    for t in range(len(self.targets))
                    for i in range(len(DEFAULT_QUANTILES))
                ]
            )

        return DistributionPrediction(
            quantile_matrix=final_matrix,
            quantile_levels=DEFAULT_QUANTILES,
            target_names=self.targets,
        )

    @property
    def uncertainty_drivers_(self) -> pl.DataFrame | None:
        """Return residual correlation analysis results."""
        return self._uncertainty_drivers_
