"""Copula families for multivariate joint intervals."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from scipy import stats

from ..utils.exceptions import error_invalid_data, error_model_not_fitted

if TYPE_CHECKING:
    pass


CopulaFamily = str | type["BaseCopula"]


class BaseCopula:
    """Base class for copula families."""

    name: ClassVar[str]
    has_lower_tail: ClassVar[bool]
    has_upper_tail: ClassVar[bool]

    def __init__(self):
        self.theta_: float | None = None
        self.fitted_ = False

    @abstractmethod
    def fit(self, residuals: np.ndarray) -> "BaseCopula":
        """Fit copula on residual matrix."""
        ...

    @abstractmethod
    def sample(
        self,
        marginals: np.ndarray,
        n_samples: int = 1000,
    ) -> np.ndarray:
        """Generate joint samples from copula."""
        ...

    def _compute_bic(self, log_likelihood: float, n_params: int, n_samples: int) -> float:
        """Compute BIC for model selection."""
        return n_params * np.log(n_samples) - 2 * log_likelihood

    def log_likelihood(self, residuals: np.ndarray) -> float:
        """
        Compute log-likelihood for BIC calculation.

        Default implementation returns -inf. Subclasses should override
        with proper implementation for accurate BIC calculation.
        """
        return -np.inf


class GaussianCopula(BaseCopula):
    """
    Fit a Gaussian copula on residuals to model inter-target correlation.

    Captures linear dependence between targets.

    Examples:
        >>> import numpy as np
        >>> residuals = np.array([
        ...     [1, 10],
        ...     [2, 20],
        ...     [3, 30],
        ... ])
        >>> copula = GaussianCopula()
        >>> copula.fit(residuals)
        >>> print(copula.correlation_matrix_)
        [[1. 1.]
         [1. 1.]]
    """

    name = "gaussian"
    has_lower_tail = False
    has_upper_tail = False

    def __init__(self):
        super().__init__()
        self.correlation_matrix_: np.ndarray | None = None

    def fit(self, residuals: np.ndarray) -> "GaussianCopula":
        """
        Fit copula on residual matrix.

        Args:
            residuals: Residual matrix shape (n_samples, n_targets)

        Returns:
            self (for method chaining)
        """
        if residuals.ndim != 2:
            error_invalid_data(f"residuals must be 2D, got shape {residuals.shape}")

        self.correlation_matrix_ = np.corrcoef(residuals.T)
        self.fitted_ = True
        self.theta_ = 0.0

        return self

    def log_likelihood(self, residuals: np.ndarray) -> float:
        """Compute log-likelihood for BIC calculation."""
        if not self.fitted_:
            error_model_not_fitted("GaussianCopula")

        uniform = self._to_copula_space(residuals)

        try:
            cov = self.correlation_matrix_
            if cov is None:
                return -np.inf
            mvn = stats.multivariate_normal(cov=cov)
            ll = np.sum(mvn.logpdf(stats.norm.ppf(uniform)))
        except Exception:
            ll = -np.inf

        return float(ll)

    def _to_copula_space(self, residuals: np.ndarray) -> np.ndarray:
        """Transform residuals to uniform copula space via empirical CDF."""
        n_samples, n_targets = residuals.shape
        uniform = np.zeros_like(residuals)

        for t in range(n_targets):
            sorted_vals = np.sort(residuals[:, t])
            ranks = np.searchsorted(sorted_vals, residuals[:, t])
            uniform[:, t] = (ranks + 1) / (n_samples + 1)

        return uniform

    def sample(
        self,
        marginals: np.ndarray,
        n_samples: int = 1000,
    ) -> np.ndarray:
        """
        Generate joint samples from copula.

        Args:
            marginals: Quantile predictions for each target
                      shape (n_samples_input, n_targets, n_quantiles)
            n_samples: Number of Monte Carlo samples to generate

        Returns:
            Joint samples shape (n_samples, n_targets)
        """
        if not self.fitted_:
            error_model_not_fitted("GaussianCopula")

        n_samples_input, n_targets, n_quantiles = marginals.shape

        try:
            cov = self.correlation_matrix_
            if cov is None:
                error_model_not_fitted("GaussianCopula")
            mvn = stats.multivariate_normal(
                mean=np.zeros(n_targets),
                cov=cov,
            )
            normal_samples = mvn.rvs(size=n_samples)
        except Exception:
            # Fallback for singular correlation matrices
            cov = self.correlation_matrix_
            if cov is None:
                error_model_not_fitted("GaussianCopula")
            assert cov is not None
            normal_samples = np.random.multivariate_normal(
                mean=np.zeros(n_targets),
                cov=cov,
                size=n_samples,
            )

        uniform_samples = stats.norm.cdf(normal_samples)

        result = np.zeros((n_samples, n_targets))
        for t in range(n_targets):
            for i in range(n_samples):
                result[i, t] = np.interp(
                    uniform_samples[i, t],
                    np.linspace(0, 1, n_quantiles),
                    marginals[0, t, :],
                )

        return result

    def __repr__(self) -> str:
        if self.fitted_:
            assert self.correlation_matrix_ is not None
            n_targets = self.correlation_matrix_.shape[0]
            return f"GaussianCopula(n_targets={n_targets}, fitted=True)"
        return "GaussianCopula(fitted=False)"


class ClaytonCopula(BaseCopula):
    """
    Clayton copula for modeling lower tail dependence.

    Lower tail dependence: extreme low values tend to co-occur.
    Ideal for modeling simultaneous extreme low values (e.g., market crashes).

    Examples:
        >>> import numpy as np
        >>> residuals = np.array([
        ...     [-1, -2],
        ...     [-0.5, -1],
        ...     [0, 0],
        ...     [0.5, 1],
        ...     [1, 2],
        ... ])
        >>> copula = ClaytonCopula()
        >>> copula.fit(residuals)
        >>> copula.theta_ > 0  # theta > 0 indicates lower tail dependence
        True
    """

    name = "clayton"
    has_lower_tail = True
    has_upper_tail = False

    def fit(self, residuals: np.ndarray) -> "ClaytonCopula":
        """
        Fit Clayton copula via maximum likelihood.

        Args:
            residuals: Residual matrix shape (n_samples, 2) — bivariate only

        Returns:
            self (for method chaining)
        """
        if residuals.ndim != 2:
            error_invalid_data(f"residuals must be 2D, got shape {residuals.shape}")

        if residuals.shape[1] != 2:
            error_invalid_data(
                f"ClaytonCopula supports bivariate only, got {residuals.shape[1]} dimensions"
            )

        uniform = self._to_copula_space(residuals)
        u, v = uniform[:, 0], uniform[:, 1]

        def neg_log_likelihood(theta: float) -> float:
            if theta <= 0:
                return 1e10
            try:
                h = (u ** (-theta) + v ** (-theta) - 1) ** (-1 / theta)
                ll = (
                    -1 / theta * np.log(u ** (-theta) + v ** (-theta) - 1)
                    + (-theta - 1) * (np.log(u) + np.log(v))
                    - np.log(h)
                )
                if not np.isfinite(ll).all():
                    return 1e10
                return -np.sum(ll)
            except Exception:
                return 1e10

        from scipy.optimize import minimize_scalar

        result = minimize_scalar(neg_log_likelihood, bounds=(0.01, 10), method="bounded")
        self.theta_ = float(result.x)
        self.fitted_ = True

        return self

    def _to_copula_space(self, residuals: np.ndarray) -> np.ndarray:
        """Transform residuals to uniform copula space."""
        n_samples, n_targets = residuals.shape
        uniform = np.zeros_like(residuals)

        for t in range(n_targets):
            sorted_vals = np.sort(residuals[:, t])
            ranks = np.searchsorted(sorted_vals, residuals[:, t])
            uniform[:, t] = (ranks + 1) / (n_samples + 1)

        return uniform

    def log_likelihood(self, residuals: np.ndarray) -> float:
        """Compute log-likelihood for BIC calculation."""
        if not self.fitted_:
            error_model_not_fitted("ClaytonCopula")

        uniform = self._to_copula_space(residuals)
        u, v = uniform[:, 0], uniform[:, 1]
        theta = self.theta_
        assert theta is not None

        try:
            h = (u ** (-theta) + v ** (-theta) - 1) ** (-1 / theta)
            ll = (
                -1 / theta * np.log(u ** (-theta) + v ** (-theta) - 1)
                + (-theta - 1) * (np.log(u) + np.log(v))
                - np.log(h)
            )
            return float(np.sum(ll))
        except Exception:
            return -np.inf

    def sample(
        self,
        marginals: np.ndarray,
        n_samples: int = 1000,
    ) -> np.ndarray:
        """
        Generate joint samples from Clayton copula.

        Args:
            marginals: Quantile predictions for each target
                      shape (n_samples_input, n_targets, n_quantiles)
            n_samples: Number of Monte Carlo samples to generate

        Returns:
            Joint samples shape (n_samples, n_targets)
        """
        if not self.fitted_:
            error_model_not_fitted("ClaytonCopula")

        if marginals.shape[1] != 2:
            error_invalid_data(
                f"ClaytonCopula supports bivariate only, got {marginals.shape[1]} dimensions"
            )

        n_samples_input, n_targets, n_quantiles = marginals.shape
        theta = self.theta_
        assert theta is not None

        s1 = np.random.uniform(0, 1, n_samples)
        s2 = np.random.uniform(0, 1, n_samples)

        u = s1 ** (1 / theta)
        v = s2 ** (1 / theta) * (1 - (1 - s1) ** (1 / theta)) ** (-1 / theta)

        uniform_samples = np.column_stack([u, v])

        result = np.zeros((n_samples, n_targets))
        for t in range(n_targets):
            for i in range(n_samples):
                result[i, t] = np.interp(
                    uniform_samples[i, t],
                    np.linspace(0, 1, n_quantiles),
                    marginals[0, t, :],
                )

        return result

    def __repr__(self) -> str:
        if self.fitted_:
            return f"ClaytonCopula(theta={self.theta_:.3f}, fitted=True)"
        return "ClaytonCopula(fitted=False)"


class GumbelCopula(BaseCopula):
    """
    Gumbel copula for modeling upper tail dependence.

    Upper tail dependence: extreme high values tend to co-occur.
    Suited for simultaneous extreme high values (e.g., extreme rainfall).

    Examples:
        >>> import numpy as np
        >>> residuals = np.array([
        ...     [-1, -2],
        ...     [-0.5, -1],
        ...     [0, 0],
        ...     [0.5, 1],
        ...     [1, 2],
        ... ])
        >>> copula = GumbelCopula()
        >>> copula.fit(residuals)
        >>> copula.theta_ >= 1  # theta >= 1 indicates upper tail dependence
        True
    """

    name = "gumbel"
    has_lower_tail = False
    has_upper_tail = True

    def fit(self, residuals: np.ndarray) -> "GumbelCopula":
        """
        Fit Gumbel copula via maximum likelihood.

        Args:
            residuals: Residual matrix shape (n_samples, 2) — bivariate only

        Returns:
            self (for method chaining)
        """
        if residuals.ndim != 2:
            error_invalid_data(f"residuals must be 2D, got shape {residuals.shape}")

        if residuals.shape[1] != 2:
            error_invalid_data(
                f"GumbelCopula supports bivariate only, got {residuals.shape[1]} dimensions"
            )

        uniform = self._to_copula_space(residuals)
        u, v = uniform[:, 0], uniform[:, 1]

        def neg_log_likelihood(theta: float) -> float:
            if theta < 1:
                return 1e10
            try:
                t = (-np.log(u)) ** theta + (-np.log(v)) ** theta
                h = np.exp(-(t ** (1 / theta)))
                psi = -np.log(-np.log(u)) - np.log(-np.log(v))
                ll = t ** (1 / theta) + psi - np.log(h)
                if not np.isfinite(ll).all():
                    return 1e10
                return -np.sum(ll)
            except Exception:
                return 1e10

        from scipy.optimize import minimize_scalar

        result = minimize_scalar(neg_log_likelihood, bounds=(1.0, 10), method="bounded")
        self.theta_ = float(result.x)
        self.fitted_ = True

        return self

    def _to_copula_space(self, residuals: np.ndarray) -> np.ndarray:
        """Transform residuals to uniform copula space."""
        n_samples, n_targets = residuals.shape
        uniform = np.zeros_like(residuals)

        for t in range(n_targets):
            sorted_vals = np.sort(residuals[:, t])
            ranks = np.searchsorted(sorted_vals, residuals[:, t])
            uniform[:, t] = (ranks + 1) / (n_samples + 1)

        return uniform

    def log_likelihood(self, residuals: np.ndarray) -> float:
        """Compute log-likelihood for BIC calculation."""
        if not self.fitted_:
            error_model_not_fitted("GumbelCopula")

        uniform = self._to_copula_space(residuals)
        u, v = uniform[:, 0], uniform[:, 1]
        theta = self.theta_
        assert theta is not None

        try:
            t = (-np.log(u)) ** theta + (-np.log(v)) ** theta
            h = np.exp(-(t ** (1 / theta)))
            psi = -np.log(-np.log(u)) - np.log(-np.log(v))
            ll = t ** (1 / theta) + psi - np.log(h)
            return float(np.sum(ll))
        except Exception:
            return -np.inf

    def sample(
        self,
        marginals: np.ndarray,
        n_samples: int = 1000,
    ) -> np.ndarray:
        """
        Generate joint samples from Gumbel copula.

        Args:
            marginals: Quantile predictions for each target
                      shape (n_samples_input, n_targets, n_quantiles)
            n_samples: Number of Monte Carlo samples to generate

        Returns:
            Joint samples shape (n_samples, n_targets)
        """
        if not self.fitted_:
            error_model_not_fitted("GumbelCopula")

        if marginals.shape[1] != 2:
            error_invalid_data(
                f"GumbelCopula supports bivariate only, got {marginals.shape[1]} dimensions"
            )

        n_samples_input, n_targets, n_quantiles = marginals.shape
        theta = self.theta_
        assert theta is not None

        s1 = np.random.uniform(0, 1, n_samples)
        s2 = np.random.uniform(0, 1, n_samples)

        u = np.exp(-(((-np.log(s1)) ** theta + (-np.log(s2)) ** theta) ** (-1 / theta)))
        v = np.exp(
            -(((-np.log(s1)) ** theta + (-np.log(s2)) ** theta) ** (-1 / theta))
            * (np.log(s2) / np.log(s1))
        )

        uniform_samples = np.column_stack([u, v])

        result = np.zeros((n_samples, n_targets))
        for t in range(n_targets):
            for i in range(n_samples):
                result[i, t] = np.interp(
                    uniform_samples[i, t],
                    np.linspace(0, 1, n_quantiles),
                    marginals[0, t, :],
                )

        return result

    def __repr__(self) -> str:
        if self.fitted_:
            return f"GumbelCopula(theta={self.theta_:.3f}, fitted=True)"
        return "GumbelCopula(fitted=False)"


class FrankCopula(BaseCopula):
    """
    Frank copula for symmetric dependence.

    Features symmetric tail dependence (technically zero tail dependence in the limit).
    Used when extreme events are not more likely in one tail than the other.

    Examples:
        >>> import numpy as np
        >>> residuals = np.array([
        ...     [-1, -2],
        ...     [-0.5, -1],
        ...     [0, 0],
        ...     [0.5, 1],
        ...     [1, 2],
        ... ])
        >>> copula = FrankCopula()
        >>> copula.fit(residuals)
        >>> copula.theta_ != 0  # theta != 0 indicates dependence
        True
    """

    name = "frank"
    has_lower_tail = False
    has_upper_tail = False

    def fit(self, residuals: np.ndarray) -> "FrankCopula":
        """
        Fit Frank copula via maximum likelihood.

        Args:
            residuals: Residual matrix shape (n_samples, 2) — bivariate only

        Returns:
            self (for method chaining)
        """
        if residuals.ndim != 2:
            error_invalid_data(f"residuals must be 2D, got shape {residuals.shape}")

        if residuals.shape[1] != 2:
            error_invalid_data(
                f"FrankCopula supports bivariate only, got {residuals.shape[1]} dimensions"
            )

        uniform = self._to_copula_space(residuals)
        u, v = uniform[:, 0], uniform[:, 1]

        def neg_log_likelihood(theta: float) -> float:
            if theta == 0:
                return 1e10
            try:
                h = (
                    -np.log(
                        (np.exp(-theta * u) - 1) * (np.exp(-theta * v) - 1) / (np.exp(-theta) - 1)
                        + 1
                    )
                    / theta
                )
                ll = (
                    theta * (u + v - 1)
                    - np.log(np.exp(-theta * u) - 1)
                    - np.log(np.exp(-theta * v) - 1)
                    - np.log(np.exp(-theta) - 1)
                    + theta * h
                )
                if not np.isfinite(ll).all():
                    return 1e10
                return -np.sum(ll)
            except Exception:
                return 1e10

        from scipy.optimize import minimize_scalar

        result = minimize_scalar(neg_log_likelihood, bounds=(-30, 30), method="bounded")
        self.theta_ = float(result.x)
        self.fitted_ = True

        return self

    def _to_copula_space(self, residuals: np.ndarray) -> np.ndarray:
        """Transform residuals to uniform copula space."""
        n_samples, n_targets = residuals.shape
        uniform = np.zeros_like(residuals)

        for t in range(n_targets):
            sorted_vals = np.sort(residuals[:, t])
            ranks = np.searchsorted(sorted_vals, residuals[:, t])
            uniform[:, t] = (ranks + 1) / (n_samples + 1)

        return uniform

    def log_likelihood(self, residuals: np.ndarray) -> float:
        """Compute log-likelihood for BIC calculation."""
        if not self.fitted_:
            error_model_not_fitted("FrankCopula")

        uniform = self._to_copula_space(residuals)
        u, v = uniform[:, 0], uniform[:, 1]
        theta = self.theta_
        assert theta is not None

        try:
            h = (
                -np.log(
                    (np.exp(-theta * u) - 1) * (np.exp(-theta * v) - 1) / (np.exp(-theta) - 1) + 1
                )
                / theta
            )
            ll = (
                theta * (u + v - 1)
                - np.log(np.exp(-theta * u) - 1)
                - np.log(np.exp(-theta * v) - 1)
                - np.log(np.exp(-theta) - 1)
                + theta * h
            )
            return float(np.sum(ll))
        except Exception:
            return -np.inf

    def sample(
        self,
        marginals: np.ndarray,
        n_samples: int = 1000,
    ) -> np.ndarray:
        """
        Generate joint samples from Frank copula.

        Args:
            marginals: Quantile predictions for each target
                      shape (n_samples_input, n_targets, n_quantiles)
            n_samples: Number of Monte Carlo samples to generate

        Returns:
            Joint samples shape (n_samples, n_targets)
        """
        if not self.fitted_:
            error_model_not_fitted("FrankCopula")

        if marginals.shape[1] != 2:
            error_invalid_data(
                f"FrankCopula supports bivariate only, got {marginals.shape[1]} dimensions"
            )

        n_samples_input, n_targets, n_quantiles = marginals.shape
        theta = self.theta_
        assert theta is not None

        s1 = np.random.uniform(0, 1, n_samples)
        s2 = np.random.uniform(0, 1, n_samples)

        u = -np.log(1 - s1 * (1 - np.exp(-theta))) / theta
        v = -np.log(1 - s2 * (1 - np.exp(-theta))) / theta

        uniform_samples = np.column_stack([u, v])

        result = np.zeros((n_samples, n_targets))
        for t in range(n_targets):
            for i in range(n_samples):
                result[i, t] = np.interp(
                    uniform_samples[i, t],
                    np.linspace(0, 1, n_quantiles),
                    marginals[0, t, :],
                )

        return result

    def __repr__(self) -> str:
        if self.fitted_:
            return f"FrankCopula(theta={self.theta_:.3f}, fitted=True)"
        return "FrankCopula(fitted=False)"


COPULA_FAMILIES: dict[str, type[BaseCopula]] = {
    "gaussian": GaussianCopula,
    "clayton": ClaytonCopula,
    "gumbel": GumbelCopula,
    "frank": FrankCopula,
}


def auto_select_copula(
    residuals: np.ndarray,
    families: list[str] | None = None,
) -> str:
    """
    Select best copula family via BIC.

    Args:
        residuals: Residual matrix shape (n_samples, n_targets)
        families: List of copula families to consider.
                  Defaults to all available families.

    Returns:
        Name of selected copula family

    Examples:
        >>> import numpy as np
        >>> residuals = np.array([
        ...     [1, 10],
        ...     [2, 20],
        ...     [3, 30],
        ... ])
        >>> selected = auto_select_copula(residuals)
        >>> selected in ["gaussian", "clayton", "gumbel", "frank"]
        True
    """
    if families is None:
        families = list(COPULA_FAMILIES.keys())

    n_samples = residuals.shape[0]
    best_bic = np.inf
    best_family = "gaussian"

    for family_name in families:
        copula_cls = COPULA_FAMILIES[family_name]

        if residuals.shape[1] > 2 and family_name != "gaussian":
            continue

        try:
            copula = copula_cls()
            copula.fit(residuals)

            if family_name == "gaussian":
                ll = copula.log_likelihood(residuals)
            else:
                ll = copula.log_likelihood(residuals)

            n_params = 1 if family_name != "gaussian" else (residuals.shape[1] - 1) ** 2
            bic = n_params * np.log(n_samples) - 2 * ll

            if bic < best_bic:
                best_bic = bic
                best_family = family_name

        except Exception:
            continue

    return best_family
