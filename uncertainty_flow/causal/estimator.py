"""CausalUncertaintyEstimator - treatment effect estimation with uncertainty."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from ..core.base import BaseUncertaintyModel
from ..core.distribution import DistributionPrediction
from ..core.types import PolarsInput, TargetSpec
from ..utils.exceptions import ConfigurationError, ModelNotFittedError
from ..utils.polars_bridge import materialize_lazyframe, to_numpy, to_numpy_series

VALID_METHODS = ("doubly_robust", "s_learner", "t_learner")

# Quantile levels used for causal predictions
_CAUSAL_QUANTILES = [0.1, 0.5, 0.9]


class CausalUncertaintyEstimator(BaseUncertaintyModel):
    """
    Estimate treatment effects with quantified uncertainty.

    Supports doubly robust, S-learner, and T-learner methods.
    Wraps an outcome model and (optionally) a propensity score model to
    produce CATE estimates and prediction intervals on treatment effects.
    Outcome-dependent summary metrics (ATE / confidence interval) are
    computed via :meth:`evaluate`, not :meth:`predict`.

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
        >>> pred = model.predict(df.drop("outcome"))
        >>> metrics = model.evaluate(df)
        >>> ate = metrics["ate"]
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
            outcome_model: Any sklearn-like regressor with fit/predict.
                Doubly-robust mode currently does not support conformal
                wrappers as the direct outcome model.
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
        self._effect_std_: float = 0.0

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
        t = to_numpy_series(data[self.treatment_col]).astype(float)
        y = to_numpy_series(data[target_str]).astype(float)
        x = to_numpy(data, feature_cols)

        if self.method == "doubly_robust":
            self._fit_doubly_robust(data, x, t, y, target_str, feature_cols)
        elif self.method == "s_learner":
            self._fit_s_learner(data, x, t, y, target_str, feature_cols)
        elif self.method == "t_learner":
            self._fit_t_learner(data, x, t, y, target_str, feature_cols)

        if self.method == "doubly_robust":
            self._effect_std_ = float(np.std(self._dr_scores_train, ddof=1))
        else:
            self._effect_std_ = float(np.std(self._cate_train, ddof=1))

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
            DistributionPrediction with CATE-based uncertainty bands.
        """
        if not self._fitted:
            raise ModelNotFittedError("CausalUncertaintyEstimator")

        data = materialize_lazyframe(data)

        feature_cols = self._feature_cols_
        x = to_numpy(data, feature_cols)

        if self.method == "doubly_robust":
            mu1, mu0 = self._predict_counterfactuals_single_model(x, data, feature_cols)
        elif self.method == "s_learner":
            mu1, mu0 = self._predict_counterfactuals_single_model(x, data, feature_cols)
        elif self.method == "t_learner":
            mu1, mu0 = self._predict_counterfactuals_tl(x)
        else:
            raise ConfigurationError(f"Unknown method: {self.method}")

        cate = mu1 - mu0

        # Build quantile matrix from CATE ± DR-score-based uncertainty
        cate_std = float(self._effect_std_)
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
            },
        )

    def evaluate(self, data: PolarsInput) -> dict[str, float | tuple[float, float]]:
        """
        Compute outcome-dependent treatment-effect metrics on labeled data.

        Args:
            data: Data with features, treatment indicator, and observed outcome.

        Returns:
            Dictionary with ``ate`` and ``ate_ci``.
        """
        if not self._fitted:
            raise ModelNotFittedError("CausalUncertaintyEstimator")

        data = materialize_lazyframe(data)

        if self._target_col_ not in data.columns:
            raise ConfigurationError(
                f"evaluate() requires observed outcome column '{self._target_col_}'."
            )

        feature_cols = self._feature_cols_
        t = to_numpy_series(data[self.treatment_col]).astype(float)
        y = to_numpy_series(data[self._target_col_]).astype(float)
        x = to_numpy(data, feature_cols)

        if self.method == "doubly_robust":
            mu1, mu0 = self._predict_counterfactuals_single_model(x, data, feature_cols)
            e = self._propensity_predict(x)
            e = np.clip(e, 0.01, 0.99)
            scores = mu1 - mu0 + t * (y - mu1) / e - (1 - t) * (y - mu0) / (1 - e)
        elif self.method == "s_learner":
            mu1, mu0 = self._predict_counterfactuals_single_model(x, data, feature_cols)
            scores = mu1 - mu0
        elif self.method == "t_learner":
            mu1, mu0 = self._predict_counterfactuals_tl(x)
            scores = mu1 - mu0
        else:
            raise ConfigurationError(f"Unknown method: {self.method}")

        ate = float(np.mean(scores))
        se = float(np.std(scores, ddof=1) / np.sqrt(len(scores)))
        ate_ci = (ate - 1.96 * se, ate + 1.96 * se)
        return {"ate": ate, "ate_ci": ate_ci}

    # ------------------------------------------------------------------
    # Private helpers - fit
    # ------------------------------------------------------------------

    def _fit_doubly_robust(self, data, x, t, y, target_str, feature_cols):
        """Fit outcome model (features + treatment) and propensity model."""
        from sklearn.base import clone

        if hasattr(self.outcome_model, "base_model"):
            raise ConfigurationError(
                "doubly_robust currently does not support conformal wrapper outcome_model "
                "objects because conformalized DR intervals are not yet implemented. "
                "Pass a direct regressor outcome_model or use method='s_learner'/'t_learner'."
            )

        x_aug = np.column_stack([x, t])

        self._outcome_model_fitted = clone(self.outcome_model)
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
        base = (
            self.outcome_model.base_model
            if hasattr(self.outcome_model, "base_model")
            else self.outcome_model
        )
        self._outcome_model_fitted = clone(base)
        self._outcome_model_fitted.fit(x_aug, y)
        x_aug1 = np.column_stack([x, np.ones(len(t))])
        x_aug0 = np.column_stack([x, np.zeros(len(t))])
        self._cate_train = self._outcome_model_fitted.predict(
            x_aug1
        ) - self._outcome_model_fitted.predict(x_aug0)

    def _fit_t_learner(self, data, x, t, y, target_str, feature_cols):
        """Fit separate outcome models for T=0 and T=1 groups."""
        from sklearn.base import clone

        mask1 = t == 1
        mask0 = t == 0

        if not np.any(mask1) or not np.any(mask0):
            raise ConfigurationError(
                "t_learner requires at least one treated and one control sample"
            )

        base = (
            self.outcome_model.base_model
            if hasattr(self.outcome_model, "base_model")
            else self.outcome_model
        )

        self._model_t1 = clone(base)
        self._model_t0 = clone(base)

        self._model_t1.fit(x[mask1], y[mask1])
        self._model_t0.fit(x[mask0], y[mask0])
        self._cate_train = self._model_t1.predict(x) - self._model_t0.predict(x)

    # ------------------------------------------------------------------
    # Private helpers - predict
    # ------------------------------------------------------------------

    def _propensity_predict(self, x):
        """Return propensity scores P(T=1|X)."""
        return self._propensity_fitted.predict_proba(x)[:, 1]

    def _predict_counterfactuals_single_model(self, x, data, feature_cols):
        """Compute mu1 and mu0 for DR or S-learner (single outcome model)."""
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
