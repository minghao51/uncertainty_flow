"""Tests for package-level integration of new modules."""

from importlib.util import find_spec

import numpy as np
import polars as pl
from sklearn.ensemble import GradientBoostingRegressor

import uncertainty_flow as uf
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


class TestIntegrationSmoke:
    """Smoke tests to verify modules work end-to-end."""

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
        )
        model.fit(df, target="outcome")
        pred = model.predict(df)
        assert isinstance(pred, DistributionPrediction)
        assert pred._quantiles.shape[0] == n

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
