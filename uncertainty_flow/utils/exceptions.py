"""Exception hierarchy and error helpers for uncertainty_flow."""

import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


# ============================================================================
# Exception Hierarchy
# ============================================================================


class UncertaintyFlowError(ValueError):
    """Base error class for uncertainty_flow.

    All exceptions inherit from ValueError for backward compatibility.
    """

    def __init__(self, message: str, error_code: str | None = None):
        """Initialize error with optional error code.

        Args:
            message: Error message
            error_code: Optional error code (e.g., "UF-E001")
        """
        self.error_code = error_code
        if error_code:
            message = f"{message} [{error_code}]"
        super().__init__(message)


class ModelError(UncertaintyFlowError):
    """Base class for model-related errors."""

    pass


class ModelNotFittedError(ModelError):
    """Raised when a model method is called before fitting."""

    def __init__(self, model_name: str = "Model"):
        super().__init__(
            f"{model_name} not fitted. Call .fit() first.",
            error_code="UF-E002",
        )


class DataError(UncertaintyFlowError):
    """Base class for data-related errors."""

    pass


class InvalidDataError(DataError):
    """Raised when input data is invalid."""

    def __init__(self, reason: str):
        super().__init__(
            f"Invalid data: {reason}",
            error_code="UF-E003",
        )


class CalibrationError(UncertaintyFlowError):
    """Base class for calibration-related errors."""

    pass


class CalibrationSizeError(CalibrationError):
    """Raised when calibration set is too small."""

    def __init__(self, n_samples: int, min_size: int = 20):
        super().__init__(
            f"Calibration set too small ({n_samples} samples). Minimum is {min_size}.",
            error_code="UF-E001",
        )


class ConfigurationError(UncertaintyFlowError):
    """Base class for configuration-related errors."""

    pass


class QuantileError(ConfigurationError):
    """Raised when quantile configuration is invalid."""

    def __init__(self, reason: str):
        super().__init__(
            f"Invalid quantile configuration: {reason}",
            error_code="UF-E004",
        )


# ============================================================================
# Error Helper Functions
# ============================================================================


def error_model_not_fitted(model_name: str = "Model") -> None:
    """Raise ModelNotFittedError.

    Args:
        model_name: Name of the model class

    Raises:
        ModelNotFittedError: Always
    """
    raise ModelNotFittedError(model_name)


def error_invalid_data(reason: str) -> None:
    """Raise InvalidDataError.

    Args:
        reason: Explanation of why data is invalid

    Raises:
        InvalidDataError: Always
    """
    raise InvalidDataError(reason)


def error_calibration_too_small(n_samples: int, min_size: int = 20) -> None:
    """Raise CalibrationSizeError.

    Args:
        n_samples: Number of samples in calibration set
        min_size: Minimum required samples

    Raises:
        CalibrationSizeError: Always
    """
    raise CalibrationSizeError(n_samples, min_size)


def error_quantile_invalid(reason: str) -> None:
    """Raise QuantileError.

    Args:
        reason: Explanation of why quantile configuration is invalid

    Raises:
        QuantileError: Always
    """
    raise QuantileError(reason)


# ============================================================================
# Warnings (for backward compatibility)
# ============================================================================


class UncertaintyFlowWarning(UserWarning):
    """Base warning class for uncertainty_flow."""

    pass


def warn_calibration_size(n_samples: int, warn_threshold: int = 50) -> None:
    """
    UF-W001: Calibration set smaller than threshold.

    Args:
        n_samples: Number of samples in calibration set
        warn_threshold: Threshold for warning (default: 50)
    """
    warnings.warn(
        f"Calibration set has only {n_samples} samples. "
        f"Coverage guarantees may be unreliable. [UF-W001]",
        UncertaintyFlowWarning,
        stacklevel=3,
    )


def warn_quantile_crossing(pct: float) -> None:
    """
    UF-W002: Quantile crossing detected.

    Args:
        pct: Percentage of predictions with crossing
    """
    warnings.warn(
        f"Quantile crossing detected in {pct:.1f}% of predictions. "
        f"Post-sort applied. Consider re-evaluating base model quality. [UF-W002]",
        UncertaintyFlowWarning,
        stacklevel=3,
    )


def warn_coverage_gap(requested: float, achieved: float) -> None:
    """
    UF-W003: Coverage gap exceeds 5%.

    Args:
        requested: Requested coverage level
        achieved: Achieved coverage level
    """
    warnings.warn(
        f"Requested {requested} coverage but achieved {achieved:.2f}. "
        f"Model may be miscalibrated. [UF-W003]",
        UncertaintyFlowWarning,
        stacklevel=3,
    )


def warn_no_uncertainty_drivers() -> None:
    """UF-W004: No significant uncertainty drivers found."""
    warnings.warn(
        "Residual correlation analysis found no significant drivers. "
        "Intervals may be uniformly conservative. [UF-W004]",
        UncertaintyFlowWarning,
        stacklevel=3,
    )


def warn_lazyframe_materialized(reason: str) -> None:
    """
    UF-W005: LazyFrame collected earlier than expected.

    Args:
        reason: Reason for materialization
    """
    warnings.warn(
        f"LazyFrame collected earlier than expected due to {reason}. "
        f"Consider restructuring upstream pipeline. [UF-W005]",
        UncertaintyFlowWarning,
        stacklevel=3,
    )
