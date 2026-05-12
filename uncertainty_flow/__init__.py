"""uncertainty_flow: Probabilistic forecasting and uncertainty quantification."""

__version__ = "0.5.0"

from .core import (
    DEFAULT_QUANTILES,
    BaseUncertaintyModel,
    DistributionPrediction,
)
from .metrics import (
    coverage_score,
    pinball_loss,
    winkler_score,
)
from .models import DeepQuantileNet, QuantileForestForecaster

try:
    from .models.deep_quantile_torch import DeepQuantileNetTorch  # noqa: F401

    _torch_available = True
except ImportError:
    DeepQuantileNetTorch = None  # type: ignore[assignment]
    _torch_available = False

try:
    from .models.transformer_forecaster import TransformerForecaster  # noqa: F401

    _transformers_available = True
except ImportError:
    TransformerForecaster = None  # type: ignore[assignment]
    _transformers_available = False

from .utils import (
    RandomHoldoutSplit,
    TemporalHoldoutSplit,
    to_numpy,
    to_numpy_series,
)
from .wrappers import (
    AdaptiveConformalForecaster,
    ConformalClassifier,
    ConformalForecaster,
    ConformalRegressor,
    EnsembleBootstrapPI,
)

try:
    from .bayesian.numpyro_model import BayesianQuantileRegressor  # noqa: F401

    _numpyro_available = True
except ImportError:
    BayesianQuantileRegressor = None  # type: ignore[assignment]
    _numpyro_available = False

from .analysis import FeatureLeverageAnalyzer
from .causal import CausalUncertaintyEstimator
from .counterfactual import UncertaintyExplainer
from .decomposition import EnsembleDecomposition
from .multimodal import CrossModalAggregator
from .risk import (
    ConformalRiskControl,
    asymmetric_loss,
    financial_var,
    inventory_cost,
    threshold_penalty,
)

try:
    from .viz import launch_dashboard

    _viz_available = True
except ImportError:
    launch_dashboard = None  # type: ignore[assignment]
    _viz_available = False

__all__ = [
    "BaseUncertaintyModel",
    "DistributionPrediction",
    "DEFAULT_QUANTILES",
    "coverage_score",
    "pinball_loss",
    "winkler_score",
    "to_numpy",
    "to_numpy_series",
    "RandomHoldoutSplit",
    "TemporalHoldoutSplit",
    "ConformalRegressor",
    "ConformalForecaster",
    "ConformalClassifier",
    "AdaptiveConformalForecaster",
    "EnsembleBootstrapPI",
    "DeepQuantileNet",
    "QuantileForestForecaster",
    "CausalUncertaintyEstimator",
    "CrossModalAggregator",
    "FeatureLeverageAnalyzer",
    "EnsembleDecomposition",
    "ConformalRiskControl",
    "asymmetric_loss",
    "inventory_cost",
    "financial_var",
    "threshold_penalty",
    "UncertaintyExplainer",
]

if _torch_available:
    __all__.append("DeepQuantileNetTorch")

if _transformers_available:
    __all__.append("TransformerForecaster")

if _numpyro_available:
    __all__.append("BayesianQuantileRegressor")

if _viz_available:
    __all__.append("launch_dashboard")
