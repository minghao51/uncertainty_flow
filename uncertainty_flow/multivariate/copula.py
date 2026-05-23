"""Copula families for multivariate joint intervals."""

from __future__ import annotations

import inspect
import warnings
from abc import abstractmethod
from enum import Enum
from typing import ClassVar, Literal

import numpy as np
from scipy import stats

from ..utils.exceptions import InvalidDataError, ModelNotFittedError, UncertaintyFlowWarning

_THIS_MODULE = __name__


def _warn(message: str, category: type = UncertaintyFlowWarning) -> None:
    """Emit a warning attributed to the caller outside this module."""
    frame = inspect.currentframe()
    try:
        level = 2
        caller_frame = frame.f_back if frame else None
        while caller_frame is not None and caller_frame.f_globals.get("__name__", "").startswith(
            _THIS_MODULE.rsplit(".", 1)[0]
        ):
            level += 1
            caller_frame = caller_frame.f_back
        warnings.warn(message, category, stacklevel=level)
    finally:
        del frame


class CopulaFamily(str, Enum):
    """Copula family options."""

    GAUSSIAN = "gaussian"
    CLAYTON = "clayton"
    GUMBEL = "gumbel"
    FRANK = "frank"
    AUTO = "auto"


# Backward compatibility
CopulaFamilyLiteral = Literal["gaussian", "clayton", "gumbel", "frank", "auto"]
CopulaFamilyUnion = str | type["BaseCopula"]


class BaseCopula:
    """Base class for copula families."""

    name: ClassVar[str]
    has_lower_tail: ClassVar[bool]
    has_upper_tail: ClassVar[bool]

    def __init__(self):
        self.theta_: float | None = None
        self.fitted_ = False

    def _validate_fitted(self) -> None:
        """
        Ensure model is fitted before accessing parameters.

        Raises:
            ModelNotFittedError: If the model has not been fitted yet.
        """
        if not self.fitted_ or self.theta_ is None:
            raise ModelNotFittedError(self.__class__.__name__)

    @abstractmethod
    def fit(self, residuals: np.ndarray) -> "BaseCopula":
        """Fit copula on residual matrix."""
        ...

    @abstractmethod
    def sample(
        self,
        marginals: np.ndarray,
        n_samples: int = 1000,
        quantile_levels: np.ndarray | None = None,
        random_state: int | np.random.Generator | None = None,
    ) -> np.ndarray:
        """Generate joint samples from copula."""
        ...

    def _compute_bic(self, log_likelihood: float, n_params: int, n_samples: int) -> float:
        """Compute BIC for model selection."""
        return float(n_params * np.log(n_samples) - 2 * log_likelihood)

    def log_likelihood(self, residuals: np.ndarray) -> float:
        """
        Compute log-likelihood for BIC calculation.

        Default implementation returns -inf. Subclasses should override
        with proper implementation for accurate BIC calculation.
        """
        return -np.inf

    def _to_copula_space(self, residuals: np.ndarray) -> np.ndarray:
        """Transform residuals to uniform copula space via empirical CDF."""
        n_samples, n_targets = residuals.shape
        uniform = np.zeros_like(residuals)
        for t in range(n_targets):
            sorted_vals = np.sort(residuals[:, t])
            ranks = np.searchsorted(sorted_vals, residuals[:, t])
            uniform[:, t] = (ranks + 1) / (n_samples + 1)
        return uniform


def _resolve_rng(
    random_state: int | np.random.Generator | None,
) -> np.random.Generator:
    """Normalize supported random state inputs."""
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


def _gumbel_conditional_cdf(v: np.ndarray, u: np.ndarray, theta: float) -> np.ndarray:
    """Compute C_{2|1}(v|u) for Gumbel copula."""
    eps = 1e-300
    lu = -np.log(np.clip(u, eps, 1.0))
    lv = -np.log(np.clip(v, eps, 1.0))
    a = lu**theta + lv**theta
    return (
        np.exp(-(a ** (1 / theta)))
        * a ** ((1 - theta) / theta)
        * lu ** (theta - 1)
        / np.clip(u, eps, 1.0)
    )


def _solve_gumbel_conditional(
    u: np.ndarray, w: np.ndarray, theta: float, max_iter: int = 50, tol: float = 1e-8
) -> np.ndarray:
    """Solve C_{2|1}(v|u) = w for v using vectorized bisection."""
    lo = np.full_like(u, 1e-10)
    hi = np.full_like(u, 1.0 - 1e-10)
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        cdf_val = _gumbel_conditional_cdf(mid, u, theta)
        lo = np.where(cdf_val < w, mid, lo)
        hi = np.where(cdf_val >= w, mid, hi)
        if np.max(hi - lo) < tol:
            break
    return (lo + hi) / 2


def _inverse_from_marginals(
    marginals: np.ndarray,
    uniform_samples: np.ndarray,
    quantile_levels: np.ndarray | None = None,
) -> np.ndarray:
    """Map copula uniforms to marginal samples for one or more input rows."""
    n_rows, _, n_quantiles = marginals.shape
    if quantile_levels is not None:
        levels = np.asarray(quantile_levels, dtype=float)
        if len(levels) != n_quantiles:
            raise ValueError(
                f"quantile_levels length ({len(levels)}) must match "
                f"marginals n_quantiles ({n_quantiles})"
            )
    else:
        _warn(
            "quantile_levels not provided to _inverse_from_marginals. "
            "Using evenly spaced levels which may not match actual prediction "
            "quantile levels. Pass quantile_levels explicitly. [UF-W008]"
        )
        levels = np.linspace(0, 1, n_quantiles)
    clipped = np.clip(uniform_samples, levels[0], levels[-1])

    lower_idx = np.searchsorted(levels, clipped, side="right") - 1
    lower_idx = np.clip(lower_idx, 0, n_quantiles - 2)
    upper_idx = lower_idx + 1

    row_idx = np.arange(n_rows)[:, None, None]
    target_idx = np.arange(marginals.shape[1])[None, None, :]

    lower_x = levels[lower_idx]
    upper_x = levels[upper_idx]
    lower_y = marginals[row_idx, target_idx, lower_idx]
    upper_y = marginals[row_idx, target_idx, upper_idx]

    denom = upper_x - lower_x
    denom[denom == 0] = 1.0
    weights = (clipped - lower_x) / denom
    samples = lower_y + weights * (upper_y - lower_y)

    if n_rows == 1:
        return samples[0]
    return samples


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
            raise InvalidDataError(f"residuals must be 2D, got shape {residuals.shape}")

        n_samples, n_targets = residuals.shape

        # Check for constant (zero-variance) columns before computing correlation
        constant_cols = []
        for t in range(n_targets):
            if np.all(residuals[:, t] == residuals[0, t]):
                constant_cols.append(t)

        if len(constant_cols) > 0:
            raise InvalidDataError(
                f"Target columns at indices {constant_cols} are constant (zero variance). "
                "Cannot compute correlation matrix. Check if target values vary across samples."
            )

        self.correlation_matrix_ = np.corrcoef(residuals.T)

        # Condition correlation matrix for numerical stability
        min_eigval = 1e-6
        try:
            eigenvals = np.linalg.eigvals(self.correlation_matrix_)
            if np.any(np.isnan(eigenvals)):
                raise InvalidDataError(
                    "Correlation matrix contains NaN values. "
                    "This may indicate zero-variance columns or invalid residuals."
                )
            if np.any(eigenvals < min_eigval):
                deficit = min_eigval - eigenvals[eigenvals < min_eigval].min()
                conditioning = np.eye(n_targets) * deficit
                self.correlation_matrix_ = self.correlation_matrix_ + conditioning

                try:
                    np.linalg.cholesky(self.correlation_matrix_)
                except np.linalg.LinAlgError:
                    raise InvalidDataError(
                        "Correlation matrix is too ill-conditioned to condition. "
                        "This may indicate very high correlation between targets."
                    )
                eigenvals = np.linalg.eigvals(self.correlation_matrix_)

            if np.any(eigenvals < min_eigval):
                raise InvalidDataError(
                    f"Correlation matrix is too ill-conditioned. "
                    f"Minimum eigenvalue: {np.min(eigenvals):.2e}, "
                    f"Threshold: {min_eigval:.2e}. "
                    f"This may indicate very high correlation between targets."
                )
        except np.linalg.LinAlgError as e:
            raise InvalidDataError(
                f"Failed to validate correlation matrix: {e}. "
                f"Check for linear dependencies in your data."
            )

        self.fitted_ = True
        self.theta_ = 0.0

        return self

    def log_likelihood(self, residuals: np.ndarray) -> float:
        """Compute log-likelihood for BIC calculation."""
        self._validate_fitted()

        uniform = self._to_copula_space(residuals)

        try:
            cov = self.correlation_matrix_
            if cov is None:
                return -np.inf
            mvn = stats.multivariate_normal(cov=cov)
            ll = np.sum(mvn.logpdf(stats.norm.ppf(uniform)))
        except (np.linalg.LinAlgError, ValueError):
            ll = -np.inf

        return float(ll)

    def sample(
        self,
        marginals: np.ndarray,
        n_samples: int = 1000,
        quantile_levels: np.ndarray | None = None,
        random_state: int | np.random.Generator | None = None,
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
        self._validate_fitted()

        n_samples_input, n_targets, _ = marginals.shape
        rng = _resolve_rng(random_state)

        try:
            cov = self.correlation_matrix_
            if cov is None:
                raise ModelNotFittedError("GaussianCopula")
            normal_samples = stats.multivariate_normal.rvs(
                mean=np.zeros(n_targets),
                cov=cov,
                size=n_samples_input * n_samples,
                random_state=rng,
            )
        except (np.linalg.LinAlgError, ValueError):
            # Fallback for singular correlation matrices
            cov = self.correlation_matrix_
            if cov is None:
                raise ModelNotFittedError("GaussianCopula")
            assert cov is not None
            normal_samples = rng.multivariate_normal(
                mean=np.zeros(n_targets),
                cov=cov,
                size=n_samples_input * n_samples,
            )

        normal_samples = np.asarray(normal_samples).reshape(n_samples_input, n_samples, n_targets)
        uniform_samples = stats.norm.cdf(normal_samples)
        return _inverse_from_marginals(marginals, uniform_samples, quantile_levels)

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
            raise InvalidDataError(f"residuals must be 2D, got shape {residuals.shape}")

        if residuals.shape[1] != 2:
            raise InvalidDataError(
                f"ClaytonCopula supports bivariate only, got {residuals.shape[1]} dimensions"
            )

        uniform = self._to_copula_space(residuals)
        u, v = uniform[:, 0], uniform[:, 1]

        def neg_log_likelihood(theta: float) -> float:
            if theta <= 0:
                return 1e10
            try:
                with np.errstate(divide="ignore", invalid="ignore"):
                    s = u ** (-theta) + v ** (-theta) - 1
                    ll = (
                        np.log(1 + theta)
                        + (-theta - 1) * (np.log(u) + np.log(v))
                        + (-1 / theta - 2) * np.log(s)
                    )
                if not np.isfinite(ll).all():
                    return 1e10
                return float(-np.sum(ll))
            except (ValueError, OverflowError, ZeroDivisionError):
                return 1e10

        from scipy.optimize import minimize_scalar

        result = minimize_scalar(neg_log_likelihood, bounds=(0.01, 10), method="bounded")
        self.theta_ = float(result.x)
        self.fitted_ = True

        return self

    def log_likelihood(self, residuals: np.ndarray) -> float:
        """Compute log-likelihood for BIC calculation."""
        self._validate_fitted()

        uniform = self._to_copula_space(residuals)
        u, v = uniform[:, 0], uniform[:, 1]
        theta = self.theta_
        assert theta is not None

        try:
            with np.errstate(divide="ignore", invalid="ignore"):
                s = u ** (-theta) + v ** (-theta) - 1
                ll = (
                    np.log(1 + theta)
                    + (-theta - 1) * (np.log(u) + np.log(v))
                    + (-1 / theta - 2) * np.log(s)
                )
            return float(np.sum(ll))
        except (ValueError, OverflowError, ZeroDivisionError):
            return -np.inf

    def sample(
        self,
        marginals: np.ndarray,
        n_samples: int = 1000,
        quantile_levels: np.ndarray | None = None,
        random_state: int | np.random.Generator | None = None,
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
        self._validate_fitted()

        if marginals.shape[1] != 2:
            raise InvalidDataError(
                f"ClaytonCopula supports bivariate only, got {marginals.shape[1]} dimensions"
            )

        n_samples_input, n_targets, _ = marginals.shape
        theta = self.theta_
        assert theta is not None
        rng = _resolve_rng(random_state)

        s1 = rng.uniform(0, 1, size=(n_samples_input, n_samples))
        s2 = rng.uniform(0, 1, size=(n_samples_input, n_samples))

        u = s1
        v = (s1 ** (-theta) * (s2 ** (-theta / (theta + 1)) - 1) + 1) ** (-1 / theta)

        uniform_samples = np.stack([u, v], axis=-1)
        return _inverse_from_marginals(marginals, uniform_samples, quantile_levels)

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
            raise InvalidDataError(f"residuals must be 2D, got shape {residuals.shape}")

        if residuals.shape[1] != 2:
            raise InvalidDataError(
                f"GumbelCopula supports bivariate only, got {residuals.shape[1]} dimensions"
            )

        uniform = self._to_copula_space(residuals)
        u, v = uniform[:, 0], uniform[:, 1]

        def neg_log_likelihood(theta: float) -> float:
            if theta < 1:
                return 1e10
            try:
                with np.errstate(divide="ignore", invalid="ignore"):
                    a = (-np.log(u)) ** theta + (-np.log(v)) ** theta
                    a_pow = a ** (1 / theta)
                    ll = (
                        -a_pow
                        + (1 / theta - 2) * np.log(a)
                        + (theta - 1) * (np.log(-np.log(u)) + np.log(-np.log(v)))
                        - np.log(u)
                        - np.log(v)
                        + np.log(a_pow + theta - 1)
                    )
                if not np.isfinite(ll).all():
                    return 1e10
                return float(-np.sum(ll))
            except (ValueError, OverflowError, ZeroDivisionError):
                return 1e10

        from scipy.optimize import minimize_scalar

        result = minimize_scalar(neg_log_likelihood, bounds=(1.0, 10), method="bounded")
        self.theta_ = float(result.x)
        self.fitted_ = True

        return self

    def log_likelihood(self, residuals: np.ndarray) -> float:
        """Compute log-likelihood for BIC calculation."""
        self._validate_fitted()

        uniform = self._to_copula_space(residuals)
        u, v = uniform[:, 0], uniform[:, 1]
        theta = self.theta_
        assert theta is not None

        try:
            with np.errstate(divide="ignore", invalid="ignore"):
                a = (-np.log(u)) ** theta + (-np.log(v)) ** theta
                a_pow = a ** (1 / theta)
                ll = (
                    -a_pow
                    + (1 / theta - 2) * np.log(a)
                    + (theta - 1) * (np.log(-np.log(u)) + np.log(-np.log(v)))
                    - np.log(u)
                    - np.log(v)
                    + np.log(a_pow + theta - 1)
                )
            return float(np.sum(ll))
        except (ValueError, OverflowError, ZeroDivisionError):
            return -np.inf

    def sample(
        self,
        marginals: np.ndarray,
        n_samples: int = 1000,
        quantile_levels: np.ndarray | None = None,
        random_state: int | np.random.Generator | None = None,
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
        self._validate_fitted()

        if marginals.shape[1] != 2:
            raise InvalidDataError(
                f"GumbelCopula supports bivariate only, got {marginals.shape[1]} dimensions"
            )

        n_samples_input, n_targets, _ = marginals.shape
        theta = self.theta_
        assert theta is not None
        rng = _resolve_rng(random_state)

        s1 = rng.uniform(0, 1, size=(n_samples_input, n_samples))
        s2 = rng.uniform(0, 1, size=(n_samples_input, n_samples))

        u = s1
        v = _solve_gumbel_conditional(s1, s2, theta)

        uniform_samples = np.stack([u, v], axis=-1)
        return _inverse_from_marginals(marginals, uniform_samples, quantile_levels)

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
            raise InvalidDataError(f"residuals must be 2D, got shape {residuals.shape}")

        if residuals.shape[1] != 2:
            raise InvalidDataError(
                f"FrankCopula supports bivariate only, got {residuals.shape[1]} dimensions"
            )

        uniform = self._to_copula_space(residuals)
        u, v = uniform[:, 0], uniform[:, 1]

        def neg_log_likelihood(theta: float) -> float:
            if theta == 0:
                return 1e10
            try:
                with np.errstate(divide="ignore", invalid="ignore"):
                    exp_neg_theta_u = np.exp(-theta * u)
                    exp_neg_theta_v = np.exp(-theta * v)
                    exp_neg_theta = np.exp(-theta)
                    numerator = (exp_neg_theta_u - 1) * (exp_neg_theta_v - 1)
                    h = -np.log(numerator / (exp_neg_theta - 1) + 1) / theta
                    ll = (
                        theta * (u + v - 1)
                        - np.log(exp_neg_theta_u - 1)
                        - np.log(exp_neg_theta_v - 1)
                        - np.log(exp_neg_theta - 1)
                        + theta * h
                    )
                    if not np.isfinite(ll).all():
                        return 1e10
                    return float(-np.sum(ll))
            except (ValueError, OverflowError, ZeroDivisionError):
                return 1e10

        from scipy.optimize import minimize_scalar

        result = minimize_scalar(neg_log_likelihood, bounds=(-30, 30), method="bounded")
        self.theta_ = float(result.x)
        self.fitted_ = True

        return self

    def log_likelihood(self, residuals: np.ndarray) -> float:
        """Compute log-likelihood for BIC calculation."""
        self._validate_fitted()

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
        except (ValueError, OverflowError, ZeroDivisionError):
            return -np.inf

    def sample(
        self,
        marginals: np.ndarray,
        n_samples: int = 1000,
        quantile_levels: np.ndarray | None = None,
        random_state: int | np.random.Generator | None = None,
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
        self._validate_fitted()

        if marginals.shape[1] != 2:
            raise InvalidDataError(
                f"FrankCopula supports bivariate only, got {marginals.shape[1]} dimensions"
            )

        n_samples_input, n_targets, _ = marginals.shape
        theta = self.theta_
        assert theta is not None
        rng = _resolve_rng(random_state)

        s1 = rng.uniform(0, 1, size=(n_samples_input, n_samples))
        s2 = rng.uniform(0, 1, size=(n_samples_input, n_samples))

        u = s1
        denom = np.exp(-theta * u) * (1 - s2) + s2
        v = -np.log(1 + s2 * (np.exp(-theta) - 1) / denom) / theta

        uniform_samples = np.stack([u, v], axis=-1)
        return _inverse_from_marginals(marginals, uniform_samples, quantile_levels)

    def __repr__(self) -> str:
        if self.fitted_:
            return f"FrankCopula(theta={self.theta_:.3f}, fitted=True)"
        return "FrankCopula(fitted=False)"


class PairwiseChainCopula(BaseCopula):
    """
    Chain copula for d-dimensional dependence (d >= 2).

    Decomposes a d-dimensional joint into d-1 bivariate copulas arranged in a
    chain (a simplified vine).  Each pair captures dependence between
    consecutive targets.  Supports tail dependence via Archimedean families.

    Examples:
        >>> import numpy as np
        >>> residuals = np.column_stack([
        ...     np.random.randn(50),
        ...     np.random.randn(50) + 0.5 * np.random.randn(50),
        ...     np.random.randn(50) + 0.3 * np.random.randn(50),
        ... ])
        >>> copula = PairwiseChainCopula()
        >>> copula.fit(residuals)
        >>> copula.fitted_
        True
    """

    name = "pairwise_chain"
    has_lower_tail = False
    has_upper_tail = False

    def __init__(self):
        super().__init__()
        self._pair_copulas: list[BaseCopula] = []
        self._n_targets: int = 0

    def fit(self, residuals: np.ndarray) -> PairwiseChainCopula:
        """
        Fit the chain by learning a bivariate copula for each consecutive pair.

        Args:
            residuals: (n_samples, d) residual matrix with d >= 2.

        Returns:
            self
        """
        if residuals.ndim != 2:
            raise InvalidDataError(f"residuals must be 2D, got shape {residuals.shape}")
        if residuals.shape[1] < 2:
            raise InvalidDataError("PairwiseChainCopula requires at least 2 targets")

        n_targets = residuals.shape[1]
        self._n_targets = n_targets
        self._pair_copulas = []

        for i in range(n_targets - 1):
            pair_data = residuals[:, [i, i + 1]]
            bivariate_families = [f for f in COPULA_FAMILIES if f != "pairwise_chain"]
            best_family = auto_select_copula(pair_data, families=bivariate_families)
            pair_copula = COPULA_FAMILIES[best_family]()
            pair_copula.fit(pair_data)
            self._pair_copulas.append(pair_copula)

            if pair_copula.has_lower_tail:
                self.has_lower_tail = True
            if pair_copula.has_upper_tail:
                self.has_upper_tail = True

        self.theta_ = 0.0
        self.fitted_ = True
        return self

    def log_likelihood(self, residuals: np.ndarray) -> float:
        """Sum of log-likelihoods across all pairs."""
        self._validate_fitted()
        total = 0.0
        for i, pc in enumerate(self._pair_copulas):
            pair_data = residuals[:, [i, i + 1]]
            total += pc.log_likelihood(pair_data)
        return total

    def sample(
        self,
        marginals: np.ndarray,
        n_samples: int = 1000,
        quantile_levels: np.ndarray | None = None,
        random_state: int | np.random.Generator | None = None,
    ) -> np.ndarray:
        """
        Sequential conditional sampling through the chain.

        Draws u_1 ~ Uniform, then for each subsequent target draws
        u_{i+1} conditional on u_i using the i-th pair copula.

        Args:
            marginals: (n_input_rows, d, n_quantiles) quantile predictions.
            n_samples: Monte Carlo samples per input row.
            quantile_levels: Quantile levels.
            random_state: Random seed.

        Returns:
            Mapped samples via _inverse_from_marginals.
        """
        self._validate_fitted()

        n_rows, d, n_q = marginals.shape
        rng = _resolve_rng(random_state)

        uniform_samples = np.empty((n_rows, n_samples, d))
        uniform_samples[:, :, 0] = rng.uniform(0, 1, size=(n_rows, n_samples))

        for i in range(d - 1):
            pair_copula = self._pair_copulas[i]
            prev_u = uniform_samples[:, :, i]
            w = rng.uniform(0, 1, size=(n_rows, n_samples))

            if isinstance(pair_copula, GaussianCopula):
                corr = pair_copula.correlation_matrix_
                if corr is not None:
                    rho = corr[0, 1]
                else:
                    rho = 0.0
                z_prev = stats.norm.ppf(np.clip(prev_u, 1e-10, 1 - 1e-10))
                z_w = stats.norm.ppf(np.clip(w, 1e-10, 1 - 1e-10))
                z_cond = rho * z_prev + np.sqrt(max(1 - rho**2, 1e-10)) * z_w
                uniform_samples[:, :, i + 1] = stats.norm.cdf(z_cond)
            elif isinstance(pair_copula, ClaytonCopula):
                theta = pair_copula.theta_ if pair_copula.theta_ is not None else 0.01
                a = prev_u ** (-theta) + w ** (-theta / (theta + 1)) - 1
                uniform_samples[:, :, i + 1] = np.where(a > 0, a ** (-1 / theta), 1e-10)
            elif isinstance(pair_copula, GumbelCopula):
                prev_clipped = np.clip(prev_u, 1e-10, 1 - 1e-10)
                theta_val = pair_copula.theta_ if pair_copula.theta_ is not None else 1.01
                uniform_samples[:, :, i + 1] = _solve_gumbel_conditional(
                    prev_clipped,
                    w,
                    theta_val,
                )
            elif isinstance(pair_copula, FrankCopula):
                theta = pair_copula.theta_ if pair_copula.theta_ is not None else 0.01
                u_clipped = np.clip(prev_u, 1e-10, 1 - 1e-10)
                denom = np.exp(-theta * u_clipped) * (1 - w) + w
                uniform_samples[:, :, i + 1] = -np.log(1 + w * (np.exp(-theta) - 1) / denom) / theta
            else:
                uniform_samples[:, :, i + 1] = w

        return _inverse_from_marginals(marginals, uniform_samples, quantile_levels)

    def __repr__(self) -> str:
        if self.fitted_:
            families = [type(pc).__name__ for pc in self._pair_copulas]
            return f"PairwiseChainCopula(pairs={families}, fitted=True)"
        return "PairwiseChainCopula(fitted=False)"


COPULA_FAMILIES: dict[str, type[BaseCopula]] = {
    "gaussian": GaussianCopula,
    "clayton": ClaytonCopula,
    "gumbel": GumbelCopula,
    "frank": FrankCopula,
    "pairwise_chain": PairwiseChainCopula,
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
    n_targets = residuals.shape[1]
    best_bic = np.inf
    best_family = "gaussian"
    any_fitted = False

    if n_targets > 2:
        _warn(
            f"Auto-selecting copula for {n_targets}D data. "
            f"Only Gaussian copula supports dimensions > 2. [UF-W006]"
        )

    for family_name in families:
        copula_cls = COPULA_FAMILIES[family_name]

        if n_targets > 2 and family_name in ("clayton", "gumbel", "frank"):
            continue

        try:
            copula = copula_cls()
            copula.fit(residuals)
            ll = copula.log_likelihood(residuals)

            if family_name == "gaussian":
                n_params = n_targets * (n_targets - 1) // 2
            elif family_name == "pairwise_chain":
                n_params = n_targets - 1
            else:
                n_params = 1
            bic = n_params * np.log(n_samples) - 2 * ll

            if bic < best_bic:
                best_bic = bic
                best_family = family_name
                any_fitted = True

        except (ValueError, OverflowError, np.linalg.LinAlgError):
            continue

    if not any_fitted:
        _warn(
            "All copula families failed to fit. Defaulting to 'gaussian'. "
            "Results may be unreliable. [UF-W007]"
        )

    return best_family
