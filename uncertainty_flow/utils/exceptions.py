"""Exception hierarchy for uncertainty_flow."""


class UncertaintyFlowError(Exception):
    """Base for all library errors."""

    def __init__(self, message: str, error_code: str | None = None):
        self.error_code = error_code
        if error_code:
            message = f"{message} [{error_code}]"
        super().__init__(message)


class RecoverableError(UncertaintyFlowError):
    """Errors where retry/recovery is possible."""


class NonRecoverableError(UncertaintyFlowError):
    """Critical errors that cannot be programmatically bypassed."""


class ModelError(NonRecoverableError):
    """Base class for model-related errors."""


class ModelNotFittedError(ModelError):
    """Raised when a model method is called before fitting."""

    def __init__(self, model_name: str = "Model"):
        super().__init__(
            f"{model_name} not fitted. Call .fit() first.",
            error_code="UF-E002",
        )


class DataError(NonRecoverableError):
    """Base class for data-related errors."""


class InvalidDataError(DataError):
    """Raised when input data is invalid."""

    def __init__(self, reason: str):
        super().__init__(f"Invalid data: {reason}", error_code="UF-E003")


class CalibrationError(NonRecoverableError):
    """Base class for calibration-related errors."""


class CalibrationSizeError(CalibrationError):
    """Raised when calibration set is too small."""

    def __init__(self, n_samples: int, min_size: int = 20):
        super().__init__(
            f"Calibration set too small ({n_samples} samples). Minimum is {min_size}.",
            error_code="UF-E001",
        )


class ConfigurationError(NonRecoverableError):
    """Base class for configuration-related errors."""


class QuantileError(ConfigurationError):
    """Raised when quantile configuration is invalid."""

    def __init__(self, reason: str):
        super().__init__(
            f"Invalid quantile configuration: {reason}",
            error_code="UF-E004",
        )


class UncertaintyFlowWarning(UserWarning):
    """Base warning class for uncertainty_flow."""


RECOVERABLE_EXCEPTIONS: tuple[type[Exception], ...] = (
    RecoverableError,
    ConnectionError,
    TimeoutError,
    OSError,
)
