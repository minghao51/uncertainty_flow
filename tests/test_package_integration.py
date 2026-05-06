"""Tests for package-level integration of new modules."""

from importlib.util import find_spec

import numpy as np
import polars as pl
import pytest
from sklearn.ensemble import GradientBoostingRegressor

import uncertainty_flow as uf
import uncertainty_flow.models as uf_models
from uncertainty_flow.core.distribution import DistributionPrediction
from uncertainty_flow.wrappers.conformal import ConformalRegressor

NUMPYRO_AVAILABLE = find_spec("numpyro") is not None


class TestPackageImports:
    def test_causal_module_importable(self):
        from uncertainty_flow.causal import CausalUncertaintyEstimator

        assert CausalUncertaintyEstimator is not None

    def test_multimodal_module_importable(self):
        from uncertainty_flow.multimodal import CrossModalAggregator

        assert CrossModalAggregator is not None

    def test_bayesian_module_conditional(self):
        """BayesianQuantileRegressor should be in __all__ only if numpyro available."""
        if NUMPYRO_AVAILABLE:
            assert "BayesianQuantileRegressor" in uf.__all__
        else:
            assert "BayesianQuantileRegressor" not in uf.__all__

    def test_causal_in_top_level_all(self):
        assert "CausalUncertaintyEstimator" in uf.__all__

    def test_multimodal_in_top_level_all(self):
        assert "CrossModalAggregator" in uf.__all__

    def test_top_level_imports_work(self):
        assert uf.CausalUncertaintyEstimator is not None
        assert uf.CrossModalAggregator is not None

    def test_torch_model_export_matches_models_package(self):
        """Torch model export should stay aligned across package entrypoints."""
        assert hasattr(uf, "DeepQuantileNetTorch")
        assert hasattr(uf_models, "DeepQuantileNetTorch")

        torch_model_available = (
            uf.DeepQuantileNetTorch is not None and uf_models.DeepQuantileNetTorch is not None
        )

        if torch_model_available:
            assert uf.DeepQuantileNetTorch is uf_models.DeepQuantileNetTorch
            assert "DeepQuantileNetTorch" in uf.__all__
            assert "DeepQuantileNetTorch" in uf_models.__all__
        else:
            assert uf.DeepQuantileNetTorch is None
            assert uf_models.DeepQuantileNetTorch is None
            assert "DeepQuantileNetTorch" not in uf.__all__
            assert "DeepQuantileNetTorch" not in uf_models.__all__


class TestIntegrationSmoke:
    """Smoke tests to verify modules work end-to-end."""

    @pytest.mark.smoke
    def test_causal_smoke(self):
        np.random.seed(42)
        n = 200
        df = pl.DataFrame(
            {
                "x1": np.random.randn(n),
                "x2": np.random.randn(n),
                "treatment": np.random.binomial(1, 0.5, n),
                "outcome": np.random.randn(n) * 0.5 + 2.0 * np.random.binomial(1, 0.5, n),
            }
        )
        model = uf.CausalUncertaintyEstimator(
            outcome_model=ConformalRegressor(
                base_model=GradientBoostingRegressor(n_estimators=10, random_state=42),
                random_state=42,
            ),
            treatment_col="treatment",
            method="s_learner",
        )
        model.fit(df, target="outcome")
        pred = model.predict(df.drop("outcome"))
        assert isinstance(pred, DistributionPrediction)
        assert pred._quantiles.shape[0] == n

    @pytest.mark.smoke
    def test_multimodal_smoke(self):
        np.random.seed(42)
        n = 100
        df = pl.DataFrame(
            {
                "a": np.random.randn(n),
                "b": np.random.randn(n),
                "c": np.random.randn(n),
                "d": np.random.randn(n),
                "y": np.random.randn(n),
            }
        )
        model = uf.CrossModalAggregator(
            feature_groups={"g1": ["a", "b"], "g2": ["c", "d"]},
            aggregation="product",
            random_state=42,
        )
        base = ConformalRegressor(
            base_model=GradientBoostingRegressor(n_estimators=10, random_state=42),
            random_state=42,
        )
        model.fit(df, target="y", base_model=base)
        pred = model.predict(df)
        assert isinstance(pred, DistributionPrediction)
        groups = pred.group_uncertainty()
        assert "g1" in groups
        assert "g2" in groups

    @pytest.mark.smoke
    def test_conformal_regressor_smoke(self):
        np.random.seed(42)
        n = 100
        df = pl.DataFrame(
            {
                "x": np.random.randn(n),
                "y": np.random.randn(n) + np.random.randn(n),
            }
        )
        model = uf.ConformalRegressor(
            base_model=GradientBoostingRegressor(n_estimators=10, random_state=42),
            random_state=42,
        )
        model.fit(df, target="y")
        pred = model.predict(df)
        assert isinstance(pred, DistributionPrediction)
        assert pred._quantiles.shape[0] == n

    @pytest.mark.smoke
    def test_quantile_forest_smoke(self):
        np.random.seed(42)
        n = 120
        df = pl.DataFrame(
            {
                "date": range(n),
                "price": np.random.randn(n) + np.arange(n) * 0.5,
            }
        )
        model = uf.QuantileForestForecaster(targets="price", horizon=1, random_state=42)
        model.fit(df, target="price")
        pred = model.predict(df)
        assert isinstance(pred, DistributionPrediction)

    @pytest.mark.smoke
    def test_copula_smoke(self):
        from uncertainty_flow.multivariate.copula import GaussianCopula

        rng = np.random.default_rng(42)
        n = 100
        data = rng.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], size=n)
        copula = GaussianCopula()
        copula.fit(data)
        marginals = np.stack([np.sort(data[:, 0])[:3], np.sort(data[:, 1])[:3]]).reshape(1, 2, 3)
        samples = copula.sample(marginals, n_samples=50, random_state=42)
        assert samples.shape[0] > 0

    @pytest.mark.smoke
    def test_persistence_smoke(self, tmp_path):
        from uncertainty_flow.core.base import BaseUncertaintyModel

        model = uf.QuantileForestForecaster(targets="price", horizon=1, random_state=42)
        df = pl.DataFrame(
            {"date": range(120), "price": np.random.randn(120) + np.arange(120) * 0.5}
        )
        model.fit(df, target="price")
        path = tmp_path / "model.uf"
        model.save(str(path))
        loaded = BaseUncertaintyModel.load(str(path))
        assert isinstance(loaded, uf.QuantileForestForecaster)
        pred = loaded.predict(df)
        assert isinstance(pred, DistributionPrediction)
