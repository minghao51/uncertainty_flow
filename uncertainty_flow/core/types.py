"""Core type aliases and constants for uncertainty_flow."""

from enum import Enum
from typing import TYPE_CHECKING, Literal

import polars as pl

if TYPE_CHECKING:
    pass

# Default quantiles - lazily initialized on first access via get_config()
# For new code, use get_config().default_quantiles instead
DEFAULT_QUANTILES = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)


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
