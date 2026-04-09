"""Tests for Copula families."""

import numpy as np
import pytest

from uncertainty_flow.multivariate.copula import (
    COPULA_FAMILIES,
    ClaytonCopula,
    FrankCopula,
    GaussianCopula,
    GumbelCopula,
    auto_select_copula,
)


class TestGaussianCopula:
    """Test GaussianCopula."""

    @pytest.fixture
    def bivariate_residuals(self):
        """Create bivariate residuals with positive correlation."""
        np.random.seed(42)
        n = 1000
        residuals = np.array(
            [
                np.random.randn(n),
                2 * np.random.randn(n) + 1,
            ]
        ).T
        return residuals

    def test_fit_sets_fitted(self, bivariate_residuals):
        """Should set fitted flag after fit."""
        copula = GaussianCopula()
        assert not copula.fitted_
        copula.fit(bivariate_residuals)
        assert copula.fitted_ is True

    def test_fit_stores_correlation_matrix(self, bivariate_residuals):
        """Should store correlation matrix."""
        copula = GaussianCopula()
        copula.fit(bivariate_residuals)
        assert copula.correlation_matrix_ is not None
        assert copula.correlation_matrix_.shape == (2, 2)

    def test_log_likelihood(self, bivariate_residuals):
        """Should compute log-likelihood."""
        copula = GaussianCopula()
        copula.fit(bivariate_residuals)
        ll = copula.log_likelihood(bivariate_residuals)
        assert isinstance(ll, float)
        assert ll < 0  # Log-likelihood should be negative

    def test_sample_shape(self, bivariate_residuals):
        """Should generate samples with correct shape."""
        copula = GaussianCopula()
        copula.fit(bivariate_residuals)

        marginals = np.random.rand(1, 2, 11)
        samples = copula.sample(marginals, n_samples=100)

        assert samples.shape == (100, 2)

    def test_sample_falls_back_for_singular_covariance(self):
        """Should use the NumPy fallback when scipy rejects a singular covariance."""
        copula = GaussianCopula()
        copula.fitted_ = True
        copula.theta_ = 0.0
        copula.correlation_matrix_ = np.array([[1.0, 1.0], [1.0, 1.0]])

        marginals = np.array([[[0.0, 0.5, 1.0], [10.0, 20.0, 30.0]]])
        samples = copula.sample(marginals, n_samples=10)

        assert samples.shape == (10, 2)

    def test_repr(self, bivariate_residuals):
        """Should return meaningful repr."""
        copula = GaussianCopula()
        assert "GaussianCopula" in repr(copula)
        assert "fitted=False" in repr(copula)

        copula.fit(bivariate_residuals)
        assert "fitted=True" in repr(copula)


class TestClaytonCopula:
    """Test ClaytonCopula."""

    @pytest.fixture
    def bivariate_residuals(self):
        """Create bivariate residuals with lower tail dependence."""
        np.random.seed(42)
        n = 500
        z1 = np.random.randn(n)
        z2 = np.random.randn(n)
        lower_tail = np.exp(-np.abs(z1)) * np.exp(-np.abs(z2))
        residuals = np.column_stack(
            [
                -lower_tail * 3,
                -lower_tail * 3,
            ]
        )
        return residuals

    def test_fit_sets_theta(self, bivariate_residuals):
        """Should fit theta parameter."""
        copula = ClaytonCopula()
        copula.fit(bivariate_residuals)
        assert copula.theta_ is not None
        assert copula.theta_ > 0

    def test_has_lower_tail(self):
        """Should have lower tail dependence."""
        assert ClaytonCopula.has_lower_tail is True
        assert ClaytonCopula.has_upper_tail is False

    def test_log_likelihood(self, bivariate_residuals):
        """Should compute log-likelihood."""
        copula = ClaytonCopula()
        copula.fit(bivariate_residuals)
        ll = copula.log_likelihood(bivariate_residuals)
        assert isinstance(ll, float)

    def test_repr(self, bivariate_residuals):
        """Should return meaningful repr."""
        copula = ClaytonCopula()
        copula.fit(bivariate_residuals)
        assert "ClaytonCopula" in repr(copula)
        assert "theta=" in repr(copula)


class TestGumbelCopula:
    """Test GumbelCopula."""

    @pytest.fixture
    def bivariate_residuals(self):
        """Create bivariate residuals with upper tail dependence."""
        np.random.seed(42)
        n = 500
        z1 = np.random.randn(n)
        z2 = np.random.randn(n)
        upper_tail = np.exp(-np.abs(z1)) + np.exp(-np.abs(z2))
        residuals = np.column_stack(
            [
                upper_tail * 3,
                upper_tail * 3,
            ]
        )
        return residuals

    def test_fit_sets_theta(self, bivariate_residuals):
        """Should fit theta parameter >= 1."""
        copula = GumbelCopula()
        copula.fit(bivariate_residuals)
        assert copula.theta_ is not None
        assert copula.theta_ >= 1

    def test_has_upper_tail(self):
        """Should have upper tail dependence."""
        assert GumbelCopula.has_upper_tail is True
        assert GumbelCopula.has_lower_tail is False

    def test_log_likelihood(self, bivariate_residuals):
        """Should compute log-likelihood."""
        copula = GumbelCopula()
        copula.fit(bivariate_residuals)
        ll = copula.log_likelihood(bivariate_residuals)
        assert isinstance(ll, float)

    def test_repr(self, bivariate_residuals):
        """Should return meaningful repr."""
        copula = GumbelCopula()
        copula.fit(bivariate_residuals)
        assert "GumbelCopula" in repr(copula)
        assert "theta=" in repr(copula)


class TestFrankCopula:
    """Test FrankCopula."""

    @pytest.fixture
    def bivariate_residuals(self):
        """Create bivariate residuals with symmetric dependence."""
        np.random.seed(42)
        n = 500
        residuals = (
            np.array(
                [
                    np.random.randn(n),
                    np.random.randn(n),
                ]
            ).T
            * 2
            + 1
        )
        return residuals

    def test_fit_sets_theta(self, bivariate_residuals):
        """Should fit theta parameter."""
        copula = FrankCopula()
        copula.fit(bivariate_residuals)
        assert copula.theta_ is not None
        assert copula.theta_ != 0

    def test_has_no_tail_dependence(self):
        """Should not have tail dependence."""
        assert FrankCopula.has_lower_tail is False
        assert FrankCopula.has_upper_tail is False

    def test_log_likelihood(self, bivariate_residuals):
        """Should compute log-likelihood."""
        copula = FrankCopula()
        copula.fit(bivariate_residuals)
        ll = copula.log_likelihood(bivariate_residuals)
        assert isinstance(ll, float)

    def test_repr(self, bivariate_residuals):
        """Should return meaningful repr."""
        copula = FrankCopula()
        copula.fit(bivariate_residuals)
        assert "FrankCopula" in repr(copula)
        assert "theta=" in repr(copula)


class TestAutoSelectCopula:
    """Test auto_select_copula function."""

    @pytest.fixture
    def bivariate_residuals(self):
        """Create bivariate residuals with correlation."""
        np.random.seed(42)
        n = 500
        residuals = np.array(
            [
                np.random.randn(n),
                2 * np.random.randn(n) + 1,
            ]
        ).T
        return residuals

    def test_returns_valid_family(self, bivariate_residuals):
        """Should return a valid copula family name."""
        selected = auto_select_copula(bivariate_residuals)
        assert selected in ["gaussian", "clayton", "gumbel", "frank"]

    def test_respects_family_list(self, bivariate_residuals):
        """Should only consider specified families."""
        selected = auto_select_copula(bivariate_residuals, families=["gaussian"])
        assert selected == "gaussian"

    def test_handles_multivariate(self):
        """Should default to gaussian for multivariate."""
        np.random.seed(42)
        n = 100
        residuals = np.random.randn(n, 3)
        selected = auto_select_copula(residuals)
        assert selected == "gaussian"


class TestCopulaFamilies:
    """Test COPULA_FAMILIES registry."""

    def test_contains_all_families(self):
        """Should contain all copula families."""
        assert "gaussian" in COPULA_FAMILIES
        assert "clayton" in COPULA_FAMILIES
        assert "gumbel" in COPULA_FAMILIES
        assert "frank" in COPULA_FAMILIES

    def test_are_copula_classes(self):
        """Should map to BaseCopula subclasses."""
        for name, cls in COPULA_FAMILIES.items():
            assert hasattr(cls, "fit")
            assert hasattr(cls, "sample")
