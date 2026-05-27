"""Configuration management for uncertainty_flow using Pydantic settings."""

from __future__ import annotations

import threading

from pydantic import Field, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ..utils.exceptions import ConfigurationError, QuantileError


class QuantileConfig(BaseSettings):
    """Configuration for quantile levels and calibration thresholds.

    Supports environment variable overrides:
    - UNCERTAINTY_FLOW_DEFAULT_QUANTILES: Comma-separated list of quantiles
    - UNCERTAINTY_FLOW_MIN_CALIBRATION_SIZE: Minimum calibration set size
    - UNCERTAINTY_FLOW_WARN_CALIBRATION_SIZE: Warning threshold for calibration size
    """

    model_config = SettingsConfigDict(
        env_prefix="UNCERTAINTY_FLOW_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    default_quantiles: list[float] = Field(
        default=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
        description="Default quantile levels to predict.",
    )

    min_calibration_size: int = Field(
        default=20,
        description="Minimum number of samples required for calibration.",
        ge=1,
    )

    warn_calibration_size: int = Field(
        default=50,
        description="Threshold for warning about small calibration sets.",
        ge=1,
    )

    default_chronos_model: str = Field(
        default="chronos-bolt-small",
        description="Default Chronos model for TransformerForecaster.",
    )

    @field_validator("default_quantiles")
    @classmethod
    def validate_quantiles(cls, v: list[float]) -> list[float]:
        if not v:
            raise QuantileError("Quantile list cannot be empty")

        for q in v:
            if not 0 < q < 1:
                raise QuantileError(f"Quantile {q} must be in (0, 1)")

        unique_sorted = sorted(set(v))
        if len(unique_sorted) != len(v):
            import warnings

            warnings.warn(
                f"Duplicate quantiles detected. Using unique quantiles: {unique_sorted}",
                UserWarning,
                stacklevel=2,
            )
        return unique_sorted

    @field_validator("warn_calibration_size")
    @classmethod
    def warn_threshold_greater_than_min(cls, v: int, info: ValidationInfo) -> int:
        if "min_calibration_size" in info.data and v < info.data["min_calibration_size"]:
            raise ConfigurationError(
                f"warn_calibration_size ({v}) must be >= min_calibration_size "
                f"({info.data['min_calibration_size']})"
            )
        return v


_config: QuantileConfig | None = None
_config_lock = threading.Lock()


def get_config() -> QuantileConfig:
    """Get the global configuration instance, creating a default on first call."""
    global _config
    if _config is None:
        with _config_lock:
            if _config is None:
                _config = QuantileConfig()
    return _config


def set_config(config: QuantileConfig) -> None:
    """Set a custom global configuration."""
    global _config
    with _config_lock:
        _config = config


def reset_config() -> None:
    """Reset configuration to defaults."""
    global _config
    with _config_lock:
        _config = None
