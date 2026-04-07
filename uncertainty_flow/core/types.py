"""Core type aliases and constants for uncertainty_flow."""

from enum import Enum
from typing import TYPE_CHECKING, Literal

import polars as pl

if TYPE_CHECKING:
    pass

# Import from configuration for single source of truth
from .config import get_config

# Backward compatibility alias - DEFAULT_QUANTILES now comes from config
# For new code, use get_config().default_quantiles instead
DEFAULT_QUANTILES = get_config().default_quantiles


class CalibrationMethod(str, Enum):
    """Calibration method options."""

    HOLDOUT = "holdout"
    CROSS = "cross"


class CorrelationMode(str, Enum):
    """Correlation mode options."""

    AUTO = "auto"
    INDEPENDENT = "independent"


# Backward compatibility: keep Literal types
CalibrationMethodLiteral = Literal["holdout", "cross"]
CorrelationModeLiteral = Literal["auto", "independent"]

# Supported input types
PolarsInput = pl.DataFrame | pl.LazyFrame

# Target specification
TargetSpec = str | list[str]
