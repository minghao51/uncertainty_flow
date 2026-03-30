"""Core type aliases and constants for uncertainty_flow."""

from typing import TYPE_CHECKING, Literal

import polars as pl

if TYPE_CHECKING:
    pass

# Import from configuration for single source of truth
from .config import get_config

# Backward compatibility alias - DEFAULT_QUANTILES now comes from config
# For new code, use get_config().default_quantiles instead
DEFAULT_QUANTILES = get_config().default_quantiles

# Calibration method types
CalibrationMethod = Literal["holdout", "cross"]
CorrelationMode = Literal["auto", "independent"]

# Supported input types
PolarsInput = pl.DataFrame | pl.LazyFrame

# Target specification
TargetSpec = str | list[str]
