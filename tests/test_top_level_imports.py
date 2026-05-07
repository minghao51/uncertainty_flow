import uncertainty_flow
from uncertainty_flow import (
    CausalUncertaintyEstimator,
    ConformalClassifier,
    ConformalForecaster,
    ConformalRegressor,
    CrossModalAggregator,
    EnsembleBootstrapPI,
    EnsembleDecomposition,
    FeatureLeverageAnalyzer,
    UncertaintyExplainer,
)


class TestTopLevelImports:
    def test_version(self):
        assert uncertainty_flow.__version__ == "0.5.0"

    def test_imports_available(self):
        assert uncertainty_flow.BaseUncertaintyModel is not None
        assert uncertainty_flow.DistributionPrediction is not None
        assert uncertainty_flow.DEFAULT_QUANTILES is not None
        assert uncertainty_flow.coverage_score is not None
        assert uncertainty_flow.pinball_loss is not None
        assert uncertainty_flow.winkler_score is not None
        assert uncertainty_flow.to_numpy is not None
        assert uncertainty_flow.RandomHoldoutSplit is not None
        assert uncertainty_flow.TemporalHoldoutSplit is not None

    def test_wrapper_imports(self):
        assert ConformalRegressor is not None
        assert ConformalForecaster is not None
        assert ConformalClassifier is not None
        assert EnsembleBootstrapPI is not None

    def test_model_imports(self):
        assert uncertainty_flow.DeepQuantileNet is not None
        assert uncertainty_flow.QuantileForestForecaster is not None

    def test_analysis_imports(self):
        assert FeatureLeverageAnalyzer is not None
        assert CausalUncertaintyEstimator is not None
        assert UncertaintyExplainer is not None
        assert EnsembleDecomposition is not None
        assert CrossModalAggregator is not None

    def test_risk_imports(self):
        assert uncertainty_flow.ConformalRiskControl is not None
        assert uncertainty_flow.asymmetric_loss is not None
        assert uncertainty_flow.financial_var is not None
        assert uncertainty_flow.inventory_cost is not None
        assert uncertainty_flow.threshold_penalty is not None

    def test_all_contains_expected(self):
        for name in [
            "BaseUncertaintyModel",
            "DistributionPrediction",
            "ConformalRegressor",
            "DeepQuantileNet",
            "ConformalRiskControl",
        ]:
            assert name in uncertainty_flow.__all__
