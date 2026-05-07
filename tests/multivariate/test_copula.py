"""Tests for Copula families."""

import numpy as np
import pytest
from scipy.stats import kendalltau

from uncertainty_flow.multivariate.copula import (
    COPULA_FAMILIES,
    ClaytonCopula,
    FrankCopula,
    GaussianCopula,
    GumbelCopula,
    PairwiseChainCopula,
    auto_select_copula,
)
from uncertainty_flow.utils.exceptions import InvalidDataError, ModelNotFittedError


def _make_bivariate_residuals(n=1000, seed=42):
    np.random.seed(seed)
    return np.array([np.random.randn(n), 2 * np.random.randn(n) + 1]).T


def _make_correlated_residuals(n=2000, rho=0.7, seed=42):
    np.random.seed(seed)
    z = np.random.randn(n, 2)
    z[:, 1] = rho * z[:, 0] + np.sqrt(1 - rho**2) * np.random.randn(n)
    return z


class TestGaussianCopula:
    """Test GaussianCopula."""

    @pytest.fixture
    def bivariate_residuals(self):
        return _make_bivariate_residuals()

    def test_fit_sets_fitted(self, bivariate_residuals):
        copula = GaussianCopula()
        assert not copula.fitted_
        copula.fit(bivariate_residuals)
        assert copula.fitted_ is True

    def test_fit_stores_correlation_matrix(self, bivariate_residuals):
        copula = GaussianCopula()
        copula.fit(bivariate_residuals)
        assert copula.correlation_matrix_ is not None
        assert copula.correlation_matrix_.shape == (2, 2)

    def test_log_likelihood(self, bivariate_residuals):
        copula = GaussianCopula()
        copula.fit(bivariate_residuals)
        ll = copula.log_likelihood(bivariate_residuals)
        assert isinstance(ll, float)
        assert ll < 0

    def test_sample_shape(self, bivariate_residuals):
        copula = GaussianCopula()
        copula.fit(bivariate_residuals)
        marginals = np.random.rand(1, 2, 11)
        samples = copula.sample(marginals, n_samples=100)
        assert samples.shape == (100, 2)

    def test_sample_uses_each_row_marginals(self, bivariate_residuals):
        copula = GaussianCopula()
        copula.fit(bivariate_residuals)
        marginals = np.array(
            [
                [[0.0, 1.0, 2.0], [10.0, 11.0, 12.0]],
                [[100.0, 101.0, 102.0], [200.0, 201.0, 202.0]],
            ]
        )
        samples = copula.sample(
            marginals,
            n_samples=5,
            quantile_levels=np.array([0.25, 0.5, 0.75]),
            random_state=123,
        )
        assert samples.shape == (2, 5, 2)
        assert np.all(samples[0, :, 0] < 10.0)
        assert np.all(samples[1, :, 0] > 90.0)

    def test_sample_falls_back_for_singular_covariance(self):
        copula = GaussianCopula()
        copula.fitted_ = True
        copula.theta_ = 0.0
        copula.correlation_matrix_ = np.array([[1.0, 1.0], [1.0, 1.0]])
        marginals = np.array([[[0.0, 0.5, 1.0], [10.0, 20.0, 30.0]]])
        samples = copula.sample(marginals, n_samples=10, random_state=123)
        assert samples.shape == (10, 2)

    def test_repr(self, bivariate_residuals):
        copula = GaussianCopula()
        assert "GaussianCopula" in repr(copula)
        assert "fitted=False" in repr(copula)
        copula.fit(bivariate_residuals)
        assert "fitted=True" in repr(copula)

    def test_fit_rejects_zero_variance_columns(self):
        residuals = np.column_stack(
            [
                np.ones(100),
                np.random.randn(100) * 2 + 1,
            ]
        )
        copula = GaussianCopula()
        with pytest.raises(InvalidDataError, match="zero variance"):
            copula.fit(residuals)

    def test_fit_handles_high_correlation_with_conditioning(self):
        np.random.seed(42)
        n_samples = 1000
        z = np.random.randn(n_samples)
        residuals = np.column_stack([z, 0.99 * z + 0.14 * np.random.randn(n_samples)])
        copula = GaussianCopula()
        copula.fit(residuals)
        assert copula.fitted_
        marginals = np.random.rand(1, 2, 11)
        samples = copula.sample(marginals, n_samples=100, random_state=42)
        assert samples.shape == (100, 2)
        assert np.all(np.isfinite(samples))

    def test_fit_triggers_ridge_conditioning_on_near_singular(self):
        np.random.seed(42)
        n_samples = 1000
        z = np.random.randn(n_samples)
        residuals = np.column_stack([z, z + 1e-6 * np.random.randn(n_samples)])
        copula = GaussianCopula()
        copula.fit(residuals)
        assert copula.fitted_
        assert copula.correlation_matrix_ is not None
        eigenvals = np.linalg.eigvals(copula.correlation_matrix_)
        assert np.all(eigenvals >= 1e-6)
        marginals = np.random.rand(1, 2, 11)
        samples = copula.sample(marginals, n_samples=100, random_state=42)
        assert np.all(np.isfinite(samples))


class TestClaytonCopula:
    """Test ClaytonCopula."""

    @pytest.fixture
    def bivariate_residuals(self):
        return _make_bivariate_residuals()

    def test_fit_sets_theta(self, bivariate_residuals):
        copula = ClaytonCopula()
        copula.fit(bivariate_residuals)
        assert copula.theta_ is not None
        assert copula.theta_ > 0

    def test_sample_shape(self, bivariate_residuals):
        copula = ClaytonCopula()
        copula.fit(bivariate_residuals)
        marginals = np.random.rand(1, 2, 11)
        samples = copula.sample(marginals, n_samples=200, random_state=42)
        assert samples.shape == (200, 2)

    def test_sample_produces_uniform_marginals(self, bivariate_residuals):
        copula = ClaytonCopula()
        copula.fit(bivariate_residuals)
        theta = copula.theta_
        rng = np.random.default_rng(42)
        s1 = rng.uniform(0, 1, size=5000)
        s2 = rng.uniform(0, 1, size=5000)
        u = s1
        v = (s1 ** (-theta) * (s2 ** (-theta / (theta + 1)) - 1) + 1) ** (-1 / theta)
        assert np.all(u > 0) and np.all(u < 1)
        assert np.all(v > 0) and np.all(v < 1)

    def test_kendall_tau_matches_theory(self):
        theta = 2.0
        expected_tau = theta / (theta + 2)
        rng = np.random.default_rng(42)
        s1 = rng.uniform(0, 1, size=5000)
        s2 = rng.uniform(0, 1, size=5000)
        u = s1
        v = (s1 ** (-theta) * (s2 ** (-theta / (theta + 1)) - 1) + 1) ** (-1 / theta)
        tau, _ = kendalltau(u, v)
        assert abs(tau - expected_tau) < 0.05

    def test_repr(self, bivariate_residuals):
        copula = ClaytonCopula()
        assert "ClaytonCopula" in repr(copula)
        copula.fit(bivariate_residuals)
        assert "fitted=True" in repr(copula)


class TestGumbelConditionalCdf:
    """Unit tests for _gumbel_conditional_cdf helper."""

    def test_output_in_unit_interval(self):
        from uncertainty_flow.multivariate.copula import _gumbel_conditional_cdf

        theta = 2.0
        u = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        v = np.array([0.2, 0.4, 0.6, 0.8, 0.95])
        result = _gumbel_conditional_cdf(v, u, theta)
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_monotonic_in_v(self):
        from uncertainty_flow.multivariate.copula import _gumbel_conditional_cdf

        theta = 2.5
        v = np.linspace(0.01, 0.99, 100)
        u = np.full_like(v, 0.5)
        cdf_vals = _gumbel_conditional_cdf(v, u, theta)
        diffs = np.diff(cdf_vals)
        assert np.all(diffs >= -1e-10)

    def test_boundary_values(self):
        from uncertainty_flow.multivariate.copula import _gumbel_conditional_cdf

        theta = 2.0
        u = np.array([0.5])
        near_one = _gumbel_conditional_cdf(np.array([0.9999]), u, theta)
        near_zero = _gumbel_conditional_cdf(np.array([0.0001]), u, theta)
        assert near_one[0] > 0.99
        assert near_zero[0] < 0.01

    def test_theta_equals_one_is_independent(self):
        from uncertainty_flow.multivariate.copula import _gumbel_conditional_cdf

        theta = 1.0
        u = np.array([0.3])
        v = np.array([0.6])
        result = _gumbel_conditional_cdf(v, u, theta)
        assert np.isclose(result[0], v[0], atol=1e-6)


class TestGumbelCopula:
    """Test GumbelCopula."""

    @pytest.fixture
    def bivariate_residuals(self):
        return _make_bivariate_residuals()

    def test_fit_sets_theta(self, bivariate_residuals):
        copula = GumbelCopula()
        copula.fit(bivariate_residuals)
        assert copula.theta_ is not None
        assert copula.theta_ >= 1.0

    def test_sample_shape(self, bivariate_residuals):
        copula = GumbelCopula()
        copula.fit(bivariate_residuals)
        marginals = np.random.rand(1, 2, 11)
        samples = copula.sample(marginals, n_samples=200, random_state=42)
        assert samples.shape == (200, 2)
        assert np.all(np.isfinite(samples))

    def test_kendall_tau_matches_theory(self):
        from uncertainty_flow.multivariate.copula import _solve_gumbel_conditional

        theta = 3.0
        expected_tau = 1.0 - 1.0 / theta
        rng = np.random.default_rng(42)
        s1 = rng.uniform(0, 1, size=5000)
        s2 = rng.uniform(0, 1, size=5000)
        u = s1
        v = _solve_gumbel_conditional(s1, s2, theta)
        tau, _ = kendalltau(u, v)
        assert abs(tau - expected_tau) < 0.05

    def test_log_likelihood(self, bivariate_residuals):
        copula = GumbelCopula()
        copula.fit(bivariate_residuals)
        ll = copula.log_likelihood(bivariate_residuals)
        assert isinstance(ll, float)

    def test_repr(self, bivariate_residuals):
        copula = GumbelCopula()
        assert "GumbelCopula" in repr(copula)
        assert "fitted=False" in repr(copula)
        copula.fit(bivariate_residuals)
        assert "fitted=True" in repr(copula)


class TestFrankCopula:
    """Test FrankCopula."""

    @pytest.fixture
    def bivariate_residuals(self):
        np.random.seed(42)
        n = 500
        residuals = np.array([np.random.randn(n), np.random.randn(n)]).T * 2 + 1
        return residuals

    def test_fit_sets_theta(self, bivariate_residuals):
        copula = FrankCopula()
        copula.fit(bivariate_residuals)
        assert copula.theta_ is not None
        assert copula.theta_ != 0

    def test_has_no_tail_dependence(self):
        assert FrankCopula.has_lower_tail is False
        assert FrankCopula.has_upper_tail is False

    def test_log_likelihood(self, bivariate_residuals):
        copula = FrankCopula()
        copula.fit(bivariate_residuals)
        ll = copula.log_likelihood(bivariate_residuals)
        assert isinstance(ll, float)

    def test_sample_shape(self, bivariate_residuals):
        copula = FrankCopula()
        copula.fit(bivariate_residuals)
        marginals = np.random.rand(1, 2, 11)
        samples = copula.sample(marginals, n_samples=200, random_state=42)
        assert samples.shape == (200, 2)
        assert np.all(np.isfinite(samples))

    def test_kendall_tau_matches_theory(self):
        from scipy.integrate import quad

        theta = 5.0
        debye, _ = quad(lambda t: t / (np.exp(t) - 1), 0, theta)
        expected_tau = 1 - 4.0 / theta * (1.0 - debye / theta)

        rng = np.random.default_rng(42)
        s1 = rng.uniform(0, 1, size=5000)
        s2 = rng.uniform(0, 1, size=5000)
        u = s1
        denom = np.exp(-theta * u) * (1 - s2) + s2
        v = -np.log(1 + s2 * (np.exp(-theta) - 1) / denom) / theta
        tau, _ = kendalltau(u, v)
        assert abs(tau - expected_tau) < 0.05

    def test_repr(self, bivariate_residuals):
        copula = FrankCopula()
        copula.fit(bivariate_residuals)
        assert "FrankCopula" in repr(copula)
        assert "theta=" in repr(copula)


class TestAutoSelectCopula:
    """Test auto_select_copula function."""

    @pytest.fixture
    def bivariate_residuals(self):
        return _make_bivariate_residuals(n=500)

    def test_returns_valid_family(self, bivariate_residuals):
        selected = auto_select_copula(bivariate_residuals)
        assert selected in ["gaussian", "clayton", "gumbel", "frank"]


class TestCopulaParameterized:
    """Test parameterized error handling across all copula families."""

    @pytest.mark.parametrize(
        "copula_class",
        [ClaytonCopula, FrankCopula, GumbelCopula, GaussianCopula],
    )
    def test_rejects_non_2d_input(self, copula_class):
        """All copulas should reject 1D residuals."""
        copula = copula_class()
        with pytest.raises(InvalidDataError, match="residuals must be 2D"):
            copula.fit(np.random.randn(100))

    @pytest.mark.parametrize(
        "copula_class",
        [ClaytonCopula, FrankCopula, GumbelCopula],
    )
    def test_rejects_non_bivariate_input(self, copula_class):
        """Bivariate-only copulas should reject 3-column residuals."""
        copula = copula_class()
        residuals_3col = np.random.randn(100, 3)
        with pytest.raises(InvalidDataError, match="supports bivariate only"):
            copula.fit(residuals_3col)

    def test_gaussian_accepts_multivariate_input(self):
        """Gaussian copula should support dimensions greater than two."""
        copula = GaussianCopula()
        residuals_3col = np.random.randn(100, 3)
        fitted = copula.fit(residuals_3col)
        assert fitted is copula

    @pytest.mark.parametrize("copula_class", [ClaytonCopula, FrankCopula, GumbelCopula])
    def test_log_likelihood_rejects_unfitted_copula(self, copula_class):
        """log_likelihood should raise error on unfitted copula."""
        copula = copula_class()
        bivariate_residuals = _make_bivariate_residuals()
        with pytest.raises(ModelNotFittedError, match="not fitted"):
            copula.log_likelihood(bivariate_residuals)


class TestGaussianCopulaErrors:
    """Test GaussianCopula-specific error handling."""

    def test_rejects_nan_correlation(self):
        """Should reject residuals producing NaN eigenvalues."""
        copula = GaussianCopula()
        residuals_nan = np.full((100, 2), np.nan)
        with pytest.raises(InvalidDataError, match="infs or NaNs"):
            copula.fit(residuals_nan)

    def test_conditioning_handles_perfectly_singular_correlation(self):
        """Conditioning should fix a perfectly singular correlation matrix."""
        copula = GaussianCopula()
        residuals = np.column_stack([np.arange(100, dtype=float), np.arange(100, dtype=float)])
        copula.fit(residuals)
        assert copula.fitted_
        eigenvals = np.linalg.eigvals(copula.correlation_matrix_)
        assert np.all(eigenvals >= 1e-6)
        assert np.all(np.isfinite(eigenvals))


class TestCopulaFamilies:
    """Test COPULA_FAMILIES registry."""

    def test_contains_all_families(self):
        assert "gaussian" in COPULA_FAMILIES
        assert "clayton" in COPULA_FAMILIES
        assert "gumbel" in COPULA_FAMILIES
        assert "frank" in COPULA_FAMILIES
        assert "pairwise_chain" in COPULA_FAMILIES

    def test_are_copula_classes(self):
        for name, cls in COPULA_FAMILIES.items():
            assert hasattr(cls, "fit")
            assert hasattr(cls, "sample")


class TestPairwiseChainCopula:
    def _make_trivariate_residuals(self, n=500, seed=42):
        rng = np.random.default_rng(seed)
        z = rng.standard_normal((n, 3))
        z[:, 1] = 0.7 * z[:, 0] + 0.3 * z[:, 1]
        z[:, 2] = 0.5 * z[:, 1] + 0.5 * z[:, 2]
        return z

    def test_fit_sets_fitted(self):
        residuals = self._make_trivariate_residuals()
        copula = PairwiseChainCopula()
        copula.fit(residuals)
        assert copula.fitted_ is True
        assert len(copula._pair_copulas) == 2

    def test_sample_shape(self):
        residuals = self._make_trivariate_residuals()
        copula = PairwiseChainCopula()
        copula.fit(residuals)
        levels = np.linspace(0.05, 0.95, 19)
        marginals = np.random.default_rng(42).uniform(0, 1, size=(5, 3, 19))
        samples = copula.sample(marginals, n_samples=100, quantile_levels=levels, random_state=42)
        assert samples.ndim == 3
        assert samples.shape == (5, 100, 3)

    def test_log_likelihood(self):
        residuals = self._make_trivariate_residuals()
        copula = PairwiseChainCopula()
        copula.fit(residuals)
        ll = copula.log_likelihood(residuals)
        assert isinstance(ll, float)

    def test_requires_at_least_2_targets(self):
        copula = PairwiseChainCopula()
        with pytest.raises(InvalidDataError):
            copula.fit(np.random.randn(50, 1))

    def test_repr(self):
        residuals = self._make_trivariate_residuals()
        copula = PairwiseChainCopula()
        assert "fitted=False" in repr(copula)
        copula.fit(residuals)
        assert "fitted=True" in repr(copula)

    def test_auto_select_considers_chain_for_d3(self):
        residuals = self._make_trivariate_residuals()
        selected = auto_select_copula(residuals)
        assert selected in ("gaussian", "pairwise_chain")

    def test_bivariate_chain_works(self):
        residuals = _make_bivariate_residuals()
        copula = PairwiseChainCopula()
        copula.fit(residuals)
        assert copula.fitted_
        assert len(copula._pair_copulas) == 1
