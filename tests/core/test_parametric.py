"""Tests for parametric distribution fitting."""

import numpy as np
import pytest
from scipy import stats

from uncertainty_flow.core.parametric import (
    ParametricDistribution,
    _fit_family,
    _ks_distance,
    _refine_params,
    _refine_params_impl,
    fit_parametric,
)


class TestParametricDistribution:
    def _make_normal_quantiles(self, loc=0.0, scale=1.0, n_levels=19):
        levels = np.linspace(0.05, 0.95, n_levels)
        values = stats.norm.ppf(levels, loc=loc, scale=scale)
        return values, levels

    def test_normal_pdf_cdf_roundtrip(self):
        qv, ql = self._make_normal_quantiles()
        dist = fit_parametric(qv, ql, family="normal")
        x = np.linspace(-3, 3, 50)
        cdf_vals = dist.cdf(x)
        assert np.all(np.diff(cdf_vals) >= 0)
        assert cdf_vals[0] < 0.01
        assert cdf_vals[-1] > 0.99

    def test_ppf_roundtrip(self):
        qv, ql = self._make_normal_quantiles(loc=5.0, scale=2.0)
        dist = fit_parametric(qv, ql, family="normal")
        q_check = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        ppf_vals = dist.ppf(q_check)
        assert ppf_vals[2] == pytest.approx(5.0, abs=0.3)

    def test_sample_shape(self):
        qv, ql = self._make_normal_quantiles()
        dist = fit_parametric(qv, ql, family="normal")
        samples = dist.sample(100, random_state=42)
        assert samples.shape == (100,)

    def test_mean_and_variance(self):
        qv, ql = self._make_normal_quantiles(loc=10.0, scale=3.0)
        dist = fit_parametric(qv, ql, family="normal")
        assert dist.mean == pytest.approx(10.0, abs=0.5)
        assert dist.variance == pytest.approx(9.0, abs=2.0)

    def test_shape_params_returns_dict(self):
        qv, ql = self._make_normal_quantiles()
        dist = fit_parametric(qv, ql, family="normal")
        params = dist.shape_params
        assert "loc" in params
        assert "scale" in params

    def test_repr(self):
        qv, ql = self._make_normal_quantiles()
        dist = fit_parametric(qv, ql, family="normal")
        r = repr(dist)
        assert "normal" in r

    def test_logpdf_finite(self):
        qv, ql = self._make_normal_quantiles()
        dist = fit_parametric(qv, ql, family="normal")
        lp = dist.logpdf(np.array([0.0, 1.0]))
        assert np.all(np.isfinite(lp))

    def test_invalid_family_raises(self):
        with pytest.raises(ValueError, match="Unknown family"):
            ParametricDistribution("beta", {"a": 1, "b": 1})


class TestFitParametric:
    def test_auto_selects_best(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, size=1000)
        levels = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
        qv = np.quantile(data, levels)
        dist = fit_parametric(qv, levels, family="auto")
        assert dist.family in ParametricDistribution._FAMILIES

    def test_normal_fit(self):
        levels = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        qv = stats.norm.ppf(levels, loc=5, scale=2)
        dist = fit_parametric(qv, levels, family="normal")
        assert dist.family == "normal"
        assert dist._params["loc"] == pytest.approx(5.0, abs=0.5)
        assert dist._params["scale"] == pytest.approx(2.0, abs=0.5)

    def test_student_t_fit(self):
        levels = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        qv = stats.t.ppf(levels, df=5, loc=0, scale=1)
        dist = fit_parametric(qv, levels, family="student_t")
        assert dist.family == "student_t"
        assert "df" in dist._params

    def test_lognormal_fit(self):
        levels = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        qv = stats.lognorm.ppf(levels, s=0.5, scale=np.exp(1.0))
        dist = fit_parametric(qv, levels, family="lognormal")
        assert dist.family == "lognormal"
        assert dist._params["scale"] > 0

    def test_gamma_fit(self):
        levels = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        qv = stats.gamma.ppf(levels, a=2.0, scale=2.0)
        dist = fit_parametric(qv, levels, family="gamma")
        assert dist.family == "gamma"
        assert dist._params["a"] > 0

    def test_auto_near_normal_data_selects_normal(self):
        levels = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
        qv = stats.norm.ppf(levels, loc=0, scale=1)
        dist = fit_parametric(qv, levels, family="auto")
        assert dist.family == "normal"


class TestFitFamilyHelpers:
    def test_fit_family_normal(self):
        levels = np.array([0.1, 0.5, 0.9])
        qv = np.array([-1.28, 0.0, 1.28])
        params = _fit_family("normal", qv, levels)
        assert "loc" in params
        assert "scale" in params
        assert params["scale"] > 0

    def test_fit_family_gamma_positive_data(self):
        levels = np.array([0.1, 0.5, 0.9])
        qv = np.array([0.5, 2.0, 5.0])
        params = _fit_family("gamma", qv, levels)
        assert params["a"] > 0
        assert params["scale"] > 0

    def test_refine_params_improves_fit(self):
        levels = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        qv = stats.norm.ppf(levels, loc=10, scale=3)
        init = _fit_family("normal", qv, levels)
        refined = _refine_params("normal", init, qv, levels)
        ks_init = _ks_distance("normal", init, qv, levels)
        ks_refined = _ks_distance("normal", refined, qv, levels)
        assert ks_refined <= ks_init + 1e-8


class TestDistributionPredictionFitDistribution:
    def test_univariate_fit_distribution(self):
        from uncertainty_flow.core.distribution import DistributionPrediction

        levels = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
        qv = stats.norm.ppf(levels, loc=5, scale=2)
        q_matrix = np.tile(qv, (10, 1))
        pred = DistributionPrediction(q_matrix, levels, target_names=["y"])
        dist = pred.fit_distribution(family="normal")
        assert isinstance(dist, ParametricDistribution)
        assert dist.family == "normal"
        assert dist.mean == pytest.approx(5.0, abs=0.5)

    def test_univariate_fit_distribution_row_index(self):
        from uncertainty_flow.core.distribution import DistributionPrediction

        levels = [0.1, 0.5, 0.9]
        qv = stats.norm.ppf(levels, loc=0, scale=1)
        q_matrix = np.tile(qv, (5, 1))
        pred = DistributionPrediction(q_matrix, levels, target_names=["y"])
        dist = pred.fit_distribution(family="normal", row_index=0)
        assert isinstance(dist, ParametricDistribution)

    def test_multivariate_fit_distribution(self):
        from uncertainty_flow.core.distribution import DistributionPrediction

        levels = [0.1, 0.5, 0.9]
        qv1 = stats.norm.ppf(levels, loc=0, scale=1)
        qv2 = stats.norm.ppf(levels, loc=5, scale=2)
        q_matrix = np.column_stack([np.tile(qv1, (5, 1)), np.tile(qv2, (5, 1))])
        pred = DistributionPrediction(q_matrix, levels, target_names=["a", "b"])
        dists = pred.fit_distribution(family="normal")
        assert isinstance(dists, list)
        assert len(dists) == 2
        assert dists[0].mean == pytest.approx(0.0, abs=0.5)
        assert dists[1].mean == pytest.approx(5.0, abs=1.0)


class TestParametricEdgeCases:
    def test_optimization_fallback_returns_initial(self):
        qv = np.array([-1.0, 0.0, 1.0])
        ql = np.array([0.1, 0.5, 0.9])
        init = _fit_family("normal", qv, ql)
        refined = _refine_params_impl("normal", init, qv * 1e100, ql)
        assert isinstance(refined, dict)
        assert "loc" in refined

    def test_auto_fallback_when_all_families_fail(self):
        qv = np.array([np.nan, np.nan, np.nan])
        ql = np.array([0.1, 0.5, 0.9])
        dist = fit_parametric(qv, ql, family="auto")
        assert dist.family == "normal"

    def test_ks_distance_overflow_returns_inf(self):
        qv = np.array([1e308, 1e308, 1e308])
        ql = np.array([0.1, 0.5, 0.9])
        ks = _ks_distance("normal", {"loc": 1e308, "scale": 1e308}, qv, ql)
        assert ks == np.inf
