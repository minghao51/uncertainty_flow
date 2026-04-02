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

    _torch_available = True
except ImportError:
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
from .causal import CausalUncertaintyEstimator

# Multi-modal module (no extra deps)
from .multimodal import CrossModalAggregator

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
]

if _torch_available:
    __all__.append("DeepQuantileNetTorch")

if _numpyro_available:
    __all__.append("BayesianQuantileRegressor")
