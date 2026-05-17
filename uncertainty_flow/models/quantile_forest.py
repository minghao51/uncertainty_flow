"""QuantileForestForecaster - Quantile Regression Forest for time series."""

import warnings

import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestRegressor

from ..core.base import BaseUncertaintyModel
from ..core.distribution import DistributionPrediction
from ..core.types import DEFAULT_QUANTILES, PolarsInput, TargetSpec
from ..multivariate.copula import COPULA_FAMILIES, BaseCopula, auto_select_copula
from ..utils.auto_tuning import (
    candidate_values,
    score_distribution_prediction,
    valid_calibration_candidates,
)
from ..utils.exceptions import InvalidDataError, ModelNotFittedError
from ..utils.polars_bridge import materialize_lazyframe, to_numpy
from ..utils.split import select_validation_plan


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
        auto_tune: bool = True,
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
            copula_family: "auto" (BIC selection) or one of "gaussian", "clayton",
                "gumbel", "frank". Use "independent" for no inter-target correlation.
            calibration_size: Fraction for calibration (from END)
            auto_tune: Whether to tune supported hyperparameters before final fit
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
        self.auto_tune = auto_tune
        self.uncertainty_features = uncertainty_features
        self.random_state = random_state

        # Fitted attributes
        self._fitted = False
        self._copula: BaseCopula | None = None
        self._models: dict[str, RandomForestRegressor] = {}
        self._leaf_distributions: dict[str, list] = {}
        self._quantile_levels_: np.ndarray | None = None
        self._feature_cols_: dict[str, list[str]] = {}
        self._uncertainty_drivers_: pl.DataFrame | None = None
        self.tuned_params_: dict[str, float | int] = {}

    def _resolve_quantile_levels(self) -> np.ndarray:
        """Return fit-time quantile levels, with backward-compatible fallback."""
        if self._quantile_levels_ is not None:
            return self._quantile_levels_

        fallback_levels = np.asarray(list(DEFAULT_QUANTILES), dtype=float)
        if self._leaf_distributions:
            first_target = next(iter(self._leaf_distributions))
            first_tree = self._leaf_distributions[first_target][0]
            if first_tree["quantiles"].shape[1] != len(fallback_levels):
                raise InvalidDataError(
                    "Current config quantile count does not match fitted leaf distributions. "
                    "Refit the model after setting the desired quantile configuration."
                )
        return fallback_levels

    def _auto_tune(self, data: pl.DataFrame) -> None:
        """Tune params using validation splits, with CV averaging when inner splits exist."""
        plan = select_validation_plan(
            data,
            task_type="time_series",
            random_state=self.random_state,
            holdout_fraction=0.2,
            hybrid_mode=False,
        )
        eval_splits = plan.inner_splits if plan.inner_splits else [plan.outer_split]
        tune_calibration_size = valid_calibration_candidates(
            len(eval_splits[0][0]), self.calibration_size, [0.25, 0.3]
        )[0]

        best_score = float("inf")
        best_params: dict[str, float | int] = {}

        for n_estimators in candidate_values(self.n_estimators, [20, 30, 50]):
            for min_samples_leaf in candidate_values(self.min_samples_leaf, [3, 5, 10]):
                split_scores: list[float] = []
                for split_train, split_val in eval_splits:
                    candidate = QuantileForestForecaster(
                        targets=self.targets,
                        horizon=self.horizon,
                        n_estimators=n_estimators,
                        min_samples_leaf=min_samples_leaf,
                        max_depth=self.max_depth,
                        copula_family=self.copula_family,
                        calibration_size=tune_calibration_size,
                        auto_tune=False,
                        uncertainty_features=self.uncertainty_features,
                        random_state=self.random_state,
                    )
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        candidate.fit(split_train)
                        pred = candidate.predict(split_val)
                    actuals = split_val.select(self.targets)
                    score = score_distribution_prediction(
                        pred, actuals, self.targets, confidence=0.9
                    )
                    split_scores.append(score)
                avg_score = float(np.mean(split_scores))
                if avg_score < best_score:
                    best_score = avg_score
                    best_params = {
                        "n_estimators": int(n_estimators),
                        "min_samples_leaf": int(min_samples_leaf),
                    }

        self.n_estimators = int(best_params.get("n_estimators", self.n_estimators))
        self.min_samples_leaf = int(best_params.get("min_samples_leaf", self.min_samples_leaf))
        self.tuned_params_ = best_params

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
        data = materialize_lazyframe(data)
        self._quantile_levels_ = np.asarray(list(DEFAULT_QUANTILES), dtype=float)

        if self.auto_tune:
            self._auto_tune(data)

        # Automatic validation split strategy selection
        plan = select_validation_plan(
            data,
            task_type="time_series",
            random_state=self.random_state,
            holdout_fraction=self.calibration_size,
        )
        train, calib = plan.outer_split
        residual_matrix = []

        for target in self.targets:
            feature_cols = [col for col in train.columns if col not in self.targets]
            self._feature_cols_[target] = feature_cols

            x_train = to_numpy(train, feature_cols)
            y_train = to_numpy(train, [target]).flatten()
            x_calib = to_numpy(calib, feature_cols)
            y_calib = to_numpy(calib, [target]).flatten()

            if not np.all(np.isfinite(x_train)):
                raise InvalidDataError("Feature matrix contains NaN or Inf values")
            if not np.all(np.isfinite(y_train)):
                raise InvalidDataError("Target vector contains NaN or Inf values")

            rf = RandomForestRegressor(
                n_estimators=self.n_estimators,
                min_samples_leaf=self.min_samples_leaf,
                max_depth=self.max_depth,
                random_state=self.random_state,
            )
            rf.fit(x_train, y_train)
            self._models[target] = rf

            self._leaf_distributions[target] = self._extract_leaf_distributions(
                rf, x_train, y_train, self._quantile_levels_
            )
            residual_matrix.append(y_calib - rf.predict(x_calib))

        if len(self.targets) > 1 and self.copula_family != "independent":
            stacked_residuals = np.column_stack(residual_matrix)

            if self.copula_family == "auto":
                selected = auto_select_copula(stacked_residuals)
            elif self.copula_family in COPULA_FAMILIES:
                selected = self.copula_family
            else:
                raise InvalidDataError(
                    f"Unknown copula_family: {self.copula_family}. "
                    f"Valid options: auto, gaussian, clayton, gumbel, frank, independent"
                )

            self._copula = COPULA_FAMILIES[selected]().fit(stacked_residuals)
        else:
            self._copula = None

        self._fitted = True
        return self

    def _extract_leaf_distributions(
        self,
        rf: RandomForestRegressor,
        x: np.ndarray,
        y: np.ndarray,
        quantile_levels: np.ndarray,
    ) -> list[dict[str, np.ndarray]]:
        """
        Extract training values that fall into each leaf.

        Args:
            rf: Fitted random forest
            X: Feature matrix
            y: Target values

        Returns:
            List of dicts mapping leaf_id to leaf values
        """
        distributions: list[dict[str, np.ndarray]] = []

        for tree in rf.estimators_:
            leaf_ids = tree.apply(x)
            unique_leaves, inverse = np.unique(leaf_ids, return_inverse=True)

            leaf_quantiles = np.zeros((len(unique_leaves), len(quantile_levels)))
            for leaf_idx in range(len(unique_leaves)):
                leaf_values = y[inverse == leaf_idx]
                leaf_quantiles[leaf_idx] = np.quantile(leaf_values, quantile_levels)

            distributions.append(
                {
                    "leaf_ids": unique_leaves.astype(int),
                    "quantiles": leaf_quantiles,
                }
            )

        return distributions

    def _predict_quantiles(
        self,
        rf: RandomForestRegressor,
        leaf_dists: list,
        x: np.ndarray,
        quantile_levels: list[float] | None = None,
    ) -> np.ndarray:
        """
        Predict quantiles from leaf distributions.

        Args:
            rf: Fitted random forest
            leaf_dists: Leaf distributions from training
            X: Feature matrix
            quantile_levels: Quantile levels to compute (unused, kept for API compat)

        Returns:
            Quantile predictions shape (n_samples, n_quantiles)
        """
        del quantile_levels
        quantile_count = leaf_dists[0]["quantiles"].shape[1]
        predictions = np.zeros((len(x), quantile_count))
        all_leaf_ids = rf.apply(x)

        for tree_idx, tree_leaf_ids in enumerate(all_leaf_ids.T):
            tree_dist = leaf_dists[tree_idx]
            positions = np.searchsorted(tree_dist["leaf_ids"], tree_leaf_ids)
            predictions += tree_dist["quantiles"][positions]

        predictions /= len(leaf_dists)

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
            raise ModelNotFittedError("QuantileForestForecaster")

        # Materialize if needed
        data = materialize_lazyframe(data)

        all_quantiles = []
        quantile_levels = self._resolve_quantile_levels()

        for target in self.targets:
            x = to_numpy(data, self._feature_cols_[target])
            rf = self._models[target]
            leaf_dists = self._leaf_distributions[target]

            quantile_matrix = self._predict_quantiles(rf, leaf_dists, x)
            if quantile_matrix.shape[1] != len(quantile_levels):
                raise InvalidDataError(
                    "Stored leaf distribution quantiles do not match configured quantile levels. "
                    "Refit the model to regenerate compatible quantiles."
                )

            all_quantiles.append(quantile_matrix)

        if len(self.targets) == 1:
            final_matrix = all_quantiles[0]
        else:
            final_matrix = np.column_stack(
                [
                    all_quantiles[t][:, i]
                    for t in range(len(self.targets))
                    for i in range(len(quantile_levels))
                ]
            )

        return DistributionPrediction(
            quantile_matrix=final_matrix,
            quantile_levels=quantile_levels.tolist(),
            target_names=self.targets,
            copula=self._copula,
        )

    @property
    def uncertainty_drivers_(self) -> pl.DataFrame | None:
        """Return residual correlation analysis results."""
        return self._uncertainty_drivers_
