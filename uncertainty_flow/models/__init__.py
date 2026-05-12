"""Native uncertainty quantification models."""

from .deep_quantile import DeepQuantileNet
from .quantile_forest import QuantileForestForecaster

CHRONOS_MODELS = {
    "chronos-bolt-tiny": "amazon/chronos-bolt-tiny",
    "chronos-bolt-small": "amazon/chronos-bolt-small",
    "chronos-bolt-base": "amazon/chronos-bolt-base",
}

# Torch models are optional - only import if torch is available
try:
    from .deep_quantile_torch import DeepQuantileNetTorch  # noqa: F401

    _torch_available = True
except ImportError:
    DeepQuantileNetTorch = None  # type: ignore[assignment]
    _torch_available = False

# Transformer models are optional - only import if chronos is available
try:
    from .transformer_forecaster import TransformerForecaster  # noqa: F401

    _transformers_available = True
except ImportError:
    _transformers_available = False

__all__ = ["CHRONOS_MODELS", "DeepQuantileNet", "QuantileForestForecaster"]

if _torch_available:
    __all__.append("DeepQuantileNetTorch")

if _transformers_available:
    __all__.append("TransformerForecaster")
