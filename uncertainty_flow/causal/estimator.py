"""CausalUncertaintyEstimator - treatment effect estimation with uncertainty."""

from typing import TYPE_CHECKING

import numpy as np
from sklearn.linear_model import LogisticRegression

from ..core.base import BaseUncertaintyModel
from ..core.distribution import DistributionPrediction
from ..core.types import PolarsInput, TargetSpec
from ..utils.exceptions import ConfigurationError, error_model_not_fitted
from ..utils.polars_bridge import materialize_lazyframe, to_numpy

if TYPE_CHECKING:
    pass

VALID_METHODS = ("doubly_robust", "s_learner", "t_learner")

# Quantile levels used for causal predictions
_CAUSAL_QUANTILES = [0.1, 0.5, 0.9]


class CausalUncertaintyEstimator(BaseUncertaintyModel):
    """
    Estimate treatment effects with quantified uncertainty.

    Supports doubly robust, S-learner, and T-learner methods.
    Wraps a ConformalRegressor outcome model and (optionally) a propensity
    score model to produce CATE estimates, ATE with confidence intervals,
    and a DistributionPrediction with treatment_info.

    Examples:
        >>> from sklearn.ensemble import GradientBoostingRegressor
        >>> from uncertainty_flow.wrappers import ConformalRegressor
        >>> from uncertainty_flow.causal import CausalUncertaintyEstimator
        >>> import polars as pl
        >>> import numpy as np
        >>>
        >>> rng = np.random.default_rng(0)
        >>> n = 200
        >>> df = pl.DataFrame({
        ...     "x1": rng.standard_normal(n),
        ...     "x2": rng.standard_normal(n),
        ...     "treatment": rng.binomial(1, 0.5, n),
        ...     "outcome": np.zeros(n),
        ... })
        >>> # Y = 2*T + noise
        >>> outcome_vals = (
        ...     df["treatment"].to_numpy() * 2.0
        ...     + df["x1"].to_numpy()
        ...     + rng.standard_normal(n) * 0.5
        ... )
        >>> df = df.with_columns(outcome=pl.Series(outcome_vals))
        >>> base = GradientBoostingRegressor(n_estimators=10, random_state=42)
        >>> outcome = ConformalRegressor(base_model=base, random_state=42)
        >>> model = CausalUncertaintyEstimator(
        ...     outcome_model=outcome,
        ...     method="doubly_robust",
        ...     random_state=42,
        ... )
        >>> model.fit(df, target="outcome")
        >>> pred = model.predict(df)
        >>> ate = pred.average_treatment_effect()
    """

    def __init__(
        self,
        outcome_model,
        propensity_model=None,
        treatment_col: str = "treatment",
        method: str = "doubly_robust",
        random_state: int | None = None,
    ):
        """
        Initialize CausalUncertaintyEstimator.

        Args:
            outcome_model: A ConformalRegressor (or any model with
                .base_model attribute and .fit() / .predict() interface).
            propensity_model: Optional sklearn classifier for propensity
                scores. Defaults to LogisticRegression.
            treatment_col: Name of the binary treatment column.
            method: One of "doubly_robust", "s_learner", "t_learner".
            random_state: Random seed for reproducibility.

        Raises:
            ValueError: If method is not one of VALID_METHODS.
        """
        if method not in VALID_METHODS:
            raise ConfigurationError(f"Invalid method '{method}'. Must be one of {VALID_METHODS}")
        self.outcome_model = outcome_model
        self.propensity_model = propensity_model
        self.treatment_col = treatment_col
        self.method = method
        self.random_state = random_state
        self._fitted = False

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        data: PolarsInput,
        target: TargetSpec | None = None,
        **kwargs,
    ) -> "CausalUncertaintyEstimator":
        """
        Fit the causal model to training data.

        Args:
            data: Polars DataFrame or LazyFrame with features, treatment
                indicator, and outcome.
            target: Outcome column name.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            self (for method chaining).
        """
        data = materialize_lazyframe(data)

        if target is None:
            raise ConfigurationError("target is required for CausalUncertaintyEstimator")
        target_str = target if isinstance(target, str) else target[0]

        # Feature columns = everything except target and treatment
        self._target_col_ = target_str
        self._feature_cols_ = [
            col for col in data.columns if col != target_str and col != self.treatment_col
        ]

        feature_cols = self._feature_cols_
        t = data[self.treatment_col].to_numpy().astype(float)
        y = data[target_str].to_numpy().astype(float)
        x = to_numpy(data, feature_cols)

        if self.method == "doubly_robust":
            self._fit_doubly_robust(data, x, t, y, target_str, feature_cols)
        elif self.method == "s_learner":
            self._fit_s_learner(data, x, t, y, target_str, feature_cols)
        elif self.method == "t_learner":
            self._fit_t_learner(data, x, t, y, target_str, feature_cols)

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict(self, data: PolarsInput) -> DistributionPrediction:
        """
        Generate predictions with treatment effect estimates.

        Args:
            data: Polars DataFrame or LazyFrame with features and treatment.

        Returns:
            DistributionPrediction with treatment_info containing CATE, ATE,
            and confidence interval.
        """
        if not self._fitted:
            error_model_not_fitted("CausalUncertaintyEstimator")

        data = materialize_lazyframe(data)

        feature_cols = self._feature_cols_
        t = data[self.treatment_col].to_numpy().astype(float)
        y = data[self._target_col_].to_numpy().astype(float)
        x = to_numpy(data, feature_cols)

        if self.method == "doubly_robust":
            mu1, mu0 = self._predict_counterfactuals_dr(x, data, feature_cols)
        elif self.method == "s_learner":
            mu1, mu0 = self._predict_counterfactuals_sl(x, data, feature_cols)
        elif self.method == "t_learner":
            mu1, mu0 = self._predict_counterfactuals_tl(x)
        else:
            raise ConfigurationError(f"Unknown method: {self.method}")

        cate = mu1 - mu0

        if self.method == "doubly_robust":
            e = self._propensity_predict(x)
            e = np.clip(e, 0.01, 0.99)
            dr_scores = mu1 - mu0 + t * (y - mu1) / e - (1 - t) * (y - mu0) / (1 - e)
        else:
            dr_scores = cate

        ate = float(np.mean(dr_scores))
        se = float(np.std(dr_scores, ddof=1) / np.sqrt(len(dr_scores)))
        ate_ci = (ate - 1.96 * se, ate + 1.96 * se)

        # Build quantile matrix from CATE ± DR-score-based uncertainty
        cate_std = float(np.std(dr_scores, ddof=1))
        quantile_levels = _CAUSAL_QUANTILES
        z_scores = np.array([-1.2816, 0.0, 1.2816])  # normal quantiles for 0.1, 0.5, 0.9
        quantile_matrix = np.column_stack([cate + z * cate_std for z in z_scores])

        return DistributionPrediction(
            quantile_matrix=quantile_matrix,
            quantile_levels=quantile_levels,
            target_names=["treatment_effect"],
            treatment_info={
                "cate": cate,
                "treatment_col": self.treatment_col,
                "ate": ate,
                "ate_ci": ate_ci,
            },
        )

    # ------------------------------------------------------------------
    # Private helpers - fit
    # ------------------------------------------------------------------

    def _fit_doubly_robust(self, data, x, t, y, target_str, feature_cols):
        """Fit outcome model (features + treatment) and propensity model."""
        from sklearn.base import clone

        x_aug = np.column_stack([x, t])

        self._outcome_model_fitted = clone(self.outcome_model.base_model)
        self._outcome_model_fitted.fit(x_aug, y)

        if self.propensity_model is not None:
            self._propensity_fitted = clone(self.propensity_model)
        else:
            self._propensity_fitted = LogisticRegression(
                random_state=self.random_state, max_iter=1000
            )
        self._propensity_fitted.fit(x, t)

        e = self._propensity_fitted.predict_proba(x)[:, 1]
        e = np.clip(e, 0.01, 0.99)

        x_aug1 = np.column_stack([x, np.ones(len(t))])
        x_aug0 = np.column_stack([x, np.zeros(len(t))])
        self._mu1_train = self._outcome_model_fitted.predict(x_aug1)
        self._mu0_train = self._outcome_model_fitted.predict(x_aug0)

        self._dr_scores_train = (
            self._mu1_train
            - self._mu0_train
            + t * (y - self._mu1_train) / e
            - (1 - t) * (y - self._mu0_train) / (1 - e)
        )

    def _fit_s_learner(self, data, x, t, y, target_str, feature_cols):
        """Fit single outcome model with treatment as feature."""
        from sklearn.base import clone

        x_aug = np.column_stack([x, t])
        self._outcome_model_fitted = clone(self.outcome_model.base_model)
        self._outcome_model_fitted.fit(x_aug, y)

    def _fit_t_learner(self, data, x, t, y, target_str, feature_cols):
        """Fit separate outcome models for T=0 and T=1 groups."""
        from sklearn.base import clone

        mask1 = t == 1
        mask0 = t == 0

        self._model_t1 = clone(self.outcome_model.base_model)
        self._model_t0 = clone(self.outcome_model.base_model)

        self._model_t1.fit(x[mask1], y[mask1])
        self._model_t0.fit(x[mask0], y[mask0])

    # ------------------------------------------------------------------
    # Private helpers - predict
    # ------------------------------------------------------------------

    def _propensity_predict(self, x):
        """Return propensity scores P(T=1|X)."""
        return self._propensity_fitted.predict_proba(x)[:, 1]

    def _predict_counterfactuals_dr(self, x, data, feature_cols):
        """Compute mu1 and mu0 for doubly robust method."""
        x_aug1 = np.column_stack([x, np.ones(x.shape[0])])
        x_aug0 = np.column_stack([x, np.zeros(x.shape[0])])
        mu1 = self._outcome_model_fitted.predict(x_aug1)
        mu0 = self._outcome_model_fitted.predict(x_aug0)
        return mu1, mu0

    def _predict_counterfactuals_sl(self, x, data, feature_cols):
        """Compute mu1 and mu0 for S-learner."""
        x_aug1 = np.column_stack([x, np.ones(x.shape[0])])
        x_aug0 = np.column_stack([x, np.zeros(x.shape[0])])
        mu1 = self._outcome_model_fitted.predict(x_aug1)
        mu0 = self._outcome_model_fitted.predict(x_aug0)
        return mu1, mu0

    def _predict_counterfactuals_tl(self, x):
        """Compute mu1 and mu0 for T-learner."""
        mu1 = self._model_t1.predict(x)
        mu0 = self._model_t0.predict(x)
        return mu1, mu0
