try:
    from .numpyro_model import BayesianQuantileRegressor  # noqa: F401

    _numpyro_available = True
except ImportError:
    _numpyro_available = False

__all__ = []
if _numpyro_available:
    __all__.append("BayesianQuantileRegressor")
