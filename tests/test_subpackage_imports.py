from uncertainty_flow.analysis import FeatureLeverageAnalyzer
from uncertainty_flow.calibration import RecalibratedModel
from uncertainty_flow.causal import CausalUncertaintyEstimator
from uncertainty_flow.core import (
    BaseUncertaintyModel,
    DistributionPrediction,
    PredictionSet,
    get_config,
    reset_config,
    set_config,
)
from uncertainty_flow.counterfactual import UncertaintyExplainer
from uncertainty_flow.decomposition import EnsembleDecomposition
from uncertainty_flow.models import DeepQuantileNet, QuantileForestForecaster
from uncertainty_flow.multimodal import CrossModalAggregator
from uncertainty_flow.multivariate import (
    ClaytonCopula,
    FrankCopula,
    GaussianCopula,
    GumbelCopula,
    PairwiseChainCopula,
    auto_select_copula,
)
from uncertainty_flow.risk import (
    ConformalRiskControl,
    asymmetric_loss,
    financial_var,
    inventory_cost,
    threshold_penalty,
)
from uncertainty_flow.utils import (
    RandomHoldoutSplit,
    TemporalHoldoutSplit,
    to_numpy,
)
from uncertainty_flow.wrappers import (
    AdaptiveConformalForecaster,
    ConformalClassifier,
    ConformalForecaster,
    ConformalRegressor,
    EnsembleBootstrapPI,
)


class TestSubpackageInitImports:
    def test_core_init(self):
        assert BaseUncertaintyModel is not None
        assert DistributionPrediction is not None
        assert PredictionSet is not None
        assert callable(get_config)
        assert callable(set_config)
        assert callable(reset_config)

    def test_models_init(self):
        assert DeepQuantileNet is not None
        assert QuantileForestForecaster is not None

    def test_wrappers_init(self):
        assert AdaptiveConformalForecaster is not None
        assert ConformalRegressor is not None
        assert ConformalForecaster is not None
        assert ConformalClassifier is not None
        assert EnsembleBootstrapPI is not None

    def test_multivariate_init(self):
        assert GaussianCopula is not None
        assert ClaytonCopula is not None
        assert GumbelCopula is not None
        assert FrankCopula is not None
        assert PairwiseChainCopula is not None
        assert auto_select_copula is not None

    def test_analysis_init(self):
        assert FeatureLeverageAnalyzer is not None

    def test_causal_init(self):
        assert CausalUncertaintyEstimator is not None

    def test_counterfactual_init(self):
        assert UncertaintyExplainer is not None

    def test_decomposition_init(self):
        assert EnsembleDecomposition is not None

    def test_multimodal_init(self):
        assert CrossModalAggregator is not None

    def test_calibration_init(self):
        assert RecalibratedModel is not None

    def test_risk_init(self):
        assert ConformalRiskControl is not None
        assert callable(asymmetric_loss)
        assert callable(financial_var)
        assert callable(inventory_cost)
        assert callable(threshold_penalty)

    def test_utils_init(self):
        assert to_numpy is not None
        assert RandomHoldoutSplit is not None
        assert TemporalHoldoutSplit is not None
