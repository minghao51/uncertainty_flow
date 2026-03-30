"""Tests for exception hierarchy and error helpers."""

import pytest

from uncertainty_flow.utils.exceptions import (
    CalibrationError,
    CalibrationSizeError,
    ConfigurationError,
    DataError,
    InvalidDataError,
    ModelError,
    ModelNotFittedError,
    QuantileError,
    UncertaintyFlowError,
    error_calibration_too_small,
    error_invalid_data,
    error_model_not_fitted,
    error_quantile_invalid,
)


class TestExceptionHierarchy:
    """Test exception inheritance structure."""

    def test_base_error_inherits_from_value_error(self):
        """UncertaintyFlowError should inherit from ValueError."""
        assert issubclass(UncertaintyFlowError, ValueError)

    def test_model_error_hierarchy(self):
        """ModelError should inherit from UncertaintyFlowError."""
        assert issubclass(ModelError, UncertaintyFlowError)
        assert issubclass(ModelNotFittedError, ModelError)

    def test_data_error_hierarchy(self):
        """DataError should inherit from UncertaintyFlowError."""
        assert issubclass(DataError, UncertaintyFlowError)
        assert issubclass(InvalidDataError, DataError)

    def test_calibration_error_hierarchy(self):
        """CalibrationError should inherit from UncertaintyFlowError."""
        assert issubclass(CalibrationError, UncertaintyFlowError)
        assert issubclass(CalibrationSizeError, CalibrationError)

    def test_configuration_error_hierarchy(self):
        """ConfigurationError should inherit from UncertaintyFlowError."""
        assert issubclass(ConfigurationError, UncertaintyFlowError)
        assert issubclass(QuantileError, ConfigurationError)


class TestErrorCodes:
    """Test that exceptions include error codes."""

    def test_model_not_fitted_error_code(self):
        """ModelNotFittedError should include error code."""
        error = ModelNotFittedError("TestModel")
        assert "UF-E002" in str(error)
        assert "TestModel" in str(error)

    def test_invalid_data_error_code(self):
        """InvalidDataError should include error code."""
        error = InvalidDataError("test reason")
        assert "UF-E003" in str(error)
        assert "test reason" in str(error)

    def test_calibration_size_error_code(self):
        """CalibrationSizeError should include error code."""
        error = CalibrationSizeError(10, 20)
        assert "UF-E001" in str(error)
        assert "10" in str(error)
        assert "20" in str(error)

    def test_quantile_error_code(self):
        """QuantileError should include error code."""
        error = QuantileError("test reason")
        assert "UF-E004" in str(error)
        assert "test reason" in str(error)


class TestErrorHelpers:
    """Test error helper functions."""

    def test_error_model_not_fitted(self):
        """error_model_not_fitted should raise ModelNotFittedError."""
        with pytest.raises(ModelNotFittedError) as exc_info:
            error_model_not_fitted("TestModel")
        assert "TestModel" in str(exc_info.value)
        assert "UF-E002" in str(exc_info.value)

    def test_error_invalid_data(self):
        """error_invalid_data should raise InvalidDataError."""
        with pytest.raises(InvalidDataError) as exc_info:
            error_invalid_data("test reason")
        assert "test reason" in str(exc_info.value)
        assert "UF-E003" in str(exc_info.value)

    def test_error_calibration_too_small(self):
        """error_calibration_too_small should raise CalibrationSizeError."""
        with pytest.raises(CalibrationSizeError) as exc_info:
            error_calibration_too_small(10)
        assert "10" in str(exc_info.value)
        assert "UF-E001" in str(exc_info.value)

    def test_error_calibration_too_small_custom_min(self):
        """error_calibration_too_small should accept custom min_size."""
        with pytest.raises(CalibrationSizeError) as exc_info:
            error_calibration_too_small(5, min_size=10)
        assert "5" in str(exc_info.value)
        assert "10" in str(exc_info.value)

    def test_error_quantile_invalid(self):
        """error_quantile_invalid should raise QuantileError."""
        with pytest.raises(QuantileError) as exc_info:
            error_quantile_invalid("test reason")
        assert "test reason" in str(exc_info.value)
        assert "UF-E004" in str(exc_info.value)


class TestBackwardCompatibility:
    """Test that exceptions can be caught as ValueError."""

    def test_model_not_fitted_is_value_error(self):
        """ModelNotFittedError should be catchable as ValueError."""
        with pytest.raises(ValueError):
            error_model_not_fitted("TestModel")

    def test_invalid_data_is_value_error(self):
        """InvalidDataError should be catchable as ValueError."""
        with pytest.raises(ValueError):
            error_invalid_data("test reason")

    def test_calibration_size_is_value_error(self):
        """CalibrationSizeError should be catchable as ValueError."""
        with pytest.raises(ValueError):
            error_calibration_too_small(10)
