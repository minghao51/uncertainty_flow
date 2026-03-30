"""Native uncertainty quantification models."""

from .deep_quantile import DeepQuantileNet
from .quantile_forest import QuantileForestForecaster

# Torch models are optional - only import if torch is available
try:
    from .deep_quantile_torch import DeepQuantileNetTorch  # noqa: F401

    _torch_available = True
except ImportError:
    _torch_available = False

# Transformer models are optional - only import if chronos is available
try:
    from .transformer_forecaster import TransformerForecaster  # noqa: F401

    _transformers_available = True
except ImportError:
    _transformers_available = False

__all__ = ["DeepQuantileNet", "QuantileForestForecaster"]

if _torch_available:
    __all__.append("DeepQuantileNetTorch")

if _transformers_available:
    __all__.append("TransformerForecaster")
