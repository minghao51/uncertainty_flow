"""uncertainty_flow: Probabilistic forecasting and uncertainty quantification."""

__version__ = "0.1.0"

# Core classes
from .core import (
    DEFAULT_QUANTILES,
    BaseUncertaintyModel,
    DistributionPrediction,
)

# Metrics
from .metrics import (
    coverage_score,
    pinball_loss,
    winkler_score,
)

# Models
from .models import DeepQuantileNet, QuantileForestForecaster

try:
    from .models import DeepQuantileNetTorch  # noqa: F401

    _torch_available = DeepQuantileNetTorch is not None
except ImportError:
    DeepQuantileNetTorch = None  # type: ignore[assignment]
    _torch_available = False

# Utilities
from .utils import (
    RandomHoldoutSplit,
    TemporalHoldoutSplit,
    to_numpy,
)

# Wrappers
from .wrappers import ConformalForecaster, ConformalRegressor

# Bayesian module (optional - requires numpyro)
try:
    from .bayesian import BayesianQuantileRegressor  # noqa: F401

    _numpyro_available = True
except ImportError:
    _numpyro_available = False

# Causal module (no extra deps)
# Analysis module (no extra deps)
from .analysis import FeatureLeverageAnalyzer
from .causal import CausalUncertaintyEstimator

# Counterfactual module (no extra deps)
from .counterfactual import UncertaintyExplainer

# Decomposition module (no extra deps)
from .decomposition import EnsembleDecomposition

# Multi-modal module (no extra deps)
from .multimodal import CrossModalAggregator

# Risk module (no extra deps)
from .risk import (
    ConformalRiskControl,
    asymmetric_loss,
    financial_var,
    inventory_cost,
    threshold_penalty,
)

# Visualization module (optional - requires streamlit)
try:
    from .viz import launch_dashboard

    _viz_available = True
except ImportError:
    _viz_available = False
    launch_dashboard = None  # type: ignore[assignment]

__all__ = [
    # Core
    "BaseUncertaintyModel",
    "DistributionPrediction",
    "DEFAULT_QUANTILES",
    # Metrics
    "pinball_loss",
    "winkler_score",
    "coverage_score",
    # Utilities
    "to_numpy",
    "RandomHoldoutSplit",
    "TemporalHoldoutSplit",
    # Wrappers
    "ConformalRegressor",
    "ConformalForecaster",
    # Models
    "DeepQuantileNet",
    "QuantileForestForecaster",
    # Causal
    "CausalUncertaintyEstimator",
    # Multi-Modal
    "CrossModalAggregator",
    # Analysis
    "FeatureLeverageAnalyzer",
    # Decomposition
    "EnsembleDecomposition",
    # Risk
    "ConformalRiskControl",
    "asymmetric_loss",
    "inventory_cost",
    "financial_var",
    "threshold_penalty",
    # Counterfactual
    "UncertaintyExplainer",
]

if _viz_available:
    __all__.append("launch_dashboard")

if _torch_available:
    __all__.append("DeepQuantileNetTorch")

if _numpyro_available:
    __all__.append("BayesianQuantileRegressor")
