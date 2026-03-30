"""Configuration management for uncertainty_flow using Pydantic settings."""

from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

CHRONOS_MODELS = {
    "chronos-2-small": "amazon/chronos-2-small",
    "chronos-2": "amazon/chronos-2",
    "chronos-2-tiny": "amazon/chronos-2-tiny",
}


class QuantileConfig(BaseSettings):
    """Configuration for quantile levels and calibration thresholds.

    Supports environment variable overrides:
    - UNCERTAINTY_FLOW_DEFAULT_QUANTILES: Comma-separated list of quantiles
    - UNCERTAINTY_FLOW_MIN_CALIBRATION_SIZE: Minimum calibration set size
    - UNCERTAINTY_FLOW_WARN_CALIBRATION_SIZE: Warning threshold for calibration size

    Examples:
        >>> from uncertainty_flow.core.config import get_config
        >>> config = get_config()
        >>> config.default_quantiles
        [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    """

    model_config = SettingsConfigDict(
        env_prefix="UNCERTAINTY_FLOW_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    default_quantiles: List[float] = Field(
        default=[
            0.05,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            0.95,
        ],
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
        default="chronos-2-small",
        description=(
            "Default Chronos model for TransformerForecaster. "
            "Options: chronos-2-small (20M, default), chronos-2 (710M), chronos-2-tiny (8M)."
        ),
    )

    @field_validator("default_quantiles")
    @classmethod
    def validate_quantiles(cls, v: List[float]) -> List[float]:
        """Validate that quantiles are in (0, 1) and sorted.

        Args:
            v: List of quantile levels

        Returns:
            Sorted list of unique quantile levels

        Raises:
            ValueError: If quantiles are invalid
        """
        if not v:
            raise ValueError("Quantile list cannot be empty")

        for q in v:
            if not 0 < q < 1:
                raise ValueError(f"Quantile {q} must be in (0, 1)")

        # Remove duplicates and sort
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
    def warn_threshold_greater_than_min(cls, v: int, info) -> int:
        """Ensure warning threshold is >= minimum size.

        Args:
            v: Warning threshold value
            info: Field validation info

        Returns:
            Validated warning threshold
        """
        if "min_calibration_size" in info.data and v < info.data["min_calibration_size"]:
            raise ValueError(
                f"warn_calibration_size ({v}) must be >= min_calibration_size "
                f"({info.data['min_calibration_size']})"
            )
        return v


# Global configuration instance
_config: QuantileConfig | None = None


def get_config() -> QuantileConfig:
    """Get the global configuration instance.

    Creates a default instance on first call.

    Returns:
        QuantileConfig: Global configuration

    Examples:
        >>> from uncertainty_flow.core.config import get_config
        >>> config = get_config()
        >>> print(config.default_quantiles)
    """
    global _config
    if _config is None:
        _config = QuantileConfig()
    return _config


def set_config(config: QuantileConfig) -> None:
    """Set a custom global configuration.

    Args:
        config: Custom configuration to use

    Examples:
        >>> from uncertainty_flow.core.config import QuantileConfig, set_config
        >>> custom = QuantileConfig(default_quantiles=[0.1, 0.5, 0.9])
        >>> set_config(custom)
    """
    global _config
    _config = config


def reset_config() -> None:
    """Reset configuration to defaults.

    Examples:
        >>> from uncertainty_flow.core.config import reset_config
        >>> reset_config()
    """
    global _config
    _config = None
