"""Bayesian quantile regression via NumPyro MCMC."""

from __future__ import annotations

import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import random
from numpyro.infer import MCMC, NUTS

from ..core.base import BaseUncertaintyModel
from ..core.distribution import DistributionPrediction
from ..core.types import PolarsInput, TargetSpec
from ..utils.exceptions import ConfigurationError, error_model_not_fitted
from ..utils.polars_bridge import materialize_lazyframe, to_numpy


class BayesianQuantileRegressor(BaseUncertaintyModel):
    """Bayesian quantile regression using NumPyro MCMC with horseshoe-like priors.

    Fits a Bayesian linear regression model via MCMC and extracts quantiles
    from the posterior predictive distribution.

    NumPyro/JAX are optional dependencies. This module will raise ImportError
    at import time if they are not installed.
    """

    def __init__(
        self,
        quantiles: list[float] | None = None,
        n_warmup: int = 500,
        n_samples: int = 1000,
        kernel: str = "nuts",
        prior_width: float = 1.0,
        random_state: int | None = None,
    ):
        """
        Initialize the Bayesian quantile regressor.

        Args:
            quantiles: Quantile levels to predict. Defaults to [0.1, 0.5, 0.9].
            n_warmup: Number of MCMC warmup (burn-in) samples.
            n_samples: Number of MCMC posterior samples.
            kernel: MCMC kernel type. Currently only "nuts" is supported.
            prior_width: Scale parameter for horseshoe-like priors.
            random_state: Random seed for reproducibility.
        """
        self.quantiles = quantiles or [0.1, 0.5, 0.9]
        self.n_warmup = n_warmup
        self.n_samples = n_samples
        self.kernel = kernel
        self.prior_width = prior_width
        self.random_state = random_state

        self._fitted = False
        self._feature_cols_: list[str] = []
        self._target_col_: str = ""
        self._posterior_samples_: np.ndarray | None = None
        self._n_features_: int = 0

    def _numpyro_model(self, x, y=None):
        """NumPyro probabilistic model with horseshoe-like priors.

        Args:
            x: Feature matrix (JAX array).
            y: Target values (JAX array), None during prediction.
        """
        n_features = x.shape[1]
        pw = self.prior_width

        tau = numpyro.sample("tau", dist.HalfCauchy(pw * 0.1))
        lam = numpyro.sample("lam", dist.HalfCauchy(pw), sample_shape=(n_features,))
        beta = numpyro.sample("beta", dist.Normal(0, tau * lam))
        sigma = numpyro.sample("sigma", dist.HalfNormal(pw))

        mu = x @ beta

        with numpyro.plate("data", x.shape[0]):
            numpyro.sample("obs", dist.Normal(mu, sigma), obs=y)

    def fit(
        self,
        data: PolarsInput,
        target: TargetSpec | None = None,
        **kwargs,
    ) -> "BayesianQuantileRegressor":
        """
        Fit the Bayesian model via MCMC.

        Args:
            data: Polars DataFrame or LazyFrame with features and target.
            target: Target column name. Required.
            **kwargs: Additional parameters (unused).

        Returns:
            self (for method chaining).

        Raises:
            ValueError: If target is not specified.
        """
        # Materialize LazyFrame
        data = materialize_lazyframe(data)

        if target is None:
            raise ConfigurationError("target must be specified for BayesianQuantileRegressor")

        target_col = target if isinstance(target, str) else target[0]
        self._target_col_ = target_col

        # Extract feature/target columns
        self._feature_cols_ = [c for c in data.columns if c != target_col]
        self._n_features_ = len(self._feature_cols_)

        # Convert to numpy
        x = to_numpy(data, self._feature_cols_)
        y = to_numpy(data, [target_col]).flatten()

        # Create JAX PRNGKey
        seed = self.random_state if self.random_state is not None else 0
        rng_key = random.PRNGKey(seed)

        # Create NUTS kernel
        nuts_kernel = NUTS(self._numpyro_model)

        # Run MCMC
        mcmc = MCMC(
            nuts_kernel,
            num_warmup=self.n_warmup,
            num_samples=self.n_samples,
        )
        mcmc.run(rng_key, x=x, y=y)

        # Convert JAX posterior samples to numpy
        samples = mcmc.get_samples()
        self._posterior_samples_ = np.column_stack([np.array(v) for v in samples.values()])

        self._fitted = True
        return self

    def predict(self, data: PolarsInput) -> DistributionPrediction:
        """
        Generate probabilistic predictions from posterior predictive distribution.

        Args:
            data: Polars DataFrame or LazyFrame with features.

        Returns:
            DistributionPrediction with quantile forecasts and posterior samples.

        Raises:
            ModelNotFittedError: If model has not been fitted.
        """
        if not self._fitted:
            error_model_not_fitted("BayesianQuantileRegressor")

        assert self._posterior_samples_ is not None

        # Materialize LazyFrame
        data = materialize_lazyframe(data)

        # Convert features to numpy
        x = to_numpy(data, self._feature_cols_)

        # Extract beta samples from posterior
        # Beta is stored as a single parameter with n_features columns
        # We need to find which columns in the posterior correspond to beta
        # The posterior is column-stacked from all sample groups.
        # For our model: tau (1), lam (n_features), beta (n_features), sigma (1)
        # Total columns: 1 + n_features + n_features + 1
        n = self._n_features_
        # Find the beta columns: they come after lam in the flattened posterior
        # We reconstruct by finding the beta parameter's shape
        # Since we column_stack all samples, we need to figure out which
        # columns are beta. The safest approach: re-extract from raw samples.
        # But we only stored the stacked array. Let's use a different approach.

        # We'll re-extract from the posterior_samples_ array.
        # The ordering of samples in the dict determines column ordering.
        # NumPyro dict ordering: tau, lam, beta, sigma (alphabetical or
        # declaration order). Actually, numpyro.get_samples() returns in
        # the order the parameters were defined: tau(1), lam(n_features),
        # beta(n_features), sigma(1).
        #
        # For lam: shape (n_samples, n_features) -> flattened to n_features cols
        # For beta: shape (n_samples, n_features) -> flattened to n_features cols
        # tau: shape (n_samples,) -> 1 col
        # sigma: shape (n_samples,) -> 1 col
        #
        # Column layout after np.column_stack:
        # [tau(1), lam(n_feat), beta(n_feat), sigma(1)]
        #
        # However, dict ordering in Python 3.7+ is insertion order.
        # numpyro sample order: tau, lam, beta, sigma (as declared).
        #
        # Column indices:
        # tau: 0
        # lam: 1 to n_features
        # beta: n_features+1 to 2*n_features
        # sigma: 2*n_features+1

        # Actually, let's reconsider. Each sample value from get_samples()
        # may be 1D or 2D. np.column_stack handles both:
        # - tau: shape (n_samples,) -> 1 column
        # - lam: shape (n_samples, n_features) -> n_features columns
        # - beta: shape (n_samples, n_features) -> n_features columns
        # - sigma: shape (n_samples,) -> 1 column
        # Total: 1 + n_features + n_features + 1

        n_beta_cols = n
        beta_start = 1 + n  # after tau(1) + lam(n)
        beta_end = beta_start + n_beta_cols
        beta_samples = self._posterior_samples_[:, beta_start:beta_end]
        # shape: (n_posterior, n_features)

        # Compute predictive matrix: (n_data, n_posterior)
        pred_matrix = x @ beta_samples.T

        # Compute quantiles from predictive distribution
        quantile_matrix = np.quantile(pred_matrix, self.quantiles, axis=1).T
        # shape: (n_data, n_quantiles)

        # Sort each row for monotonicity
        quantile_matrix = np.sort(quantile_matrix, axis=1)

        return DistributionPrediction(
            quantile_matrix=quantile_matrix,
            quantile_levels=self.quantiles,
            target_names=[self._target_col_],
            posterior=self._posterior_samples_,
        )
