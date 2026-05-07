"""Tests for exception hierarchy and error helpers."""

import warnings

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
    UncertaintyFlowWarning,
    error_calibration_too_small,
    error_invalid_data,
    error_model_not_fitted,
    error_quantile_invalid,
    warn_calibration_size,
    warn_copula_auto_selection_ndim,
    warn_coverage_gap,
    warn_lazyframe_materialized,
    warn_no_uncertainty_drivers,
    warn_quantile_crossing,
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


class TestErrorCodesExtended:
    def test_error_code_attribute(self):
        err = UncertaintyFlowError("test", error_code="UF-E099")
        assert err.error_code == "UF-E099"
        assert "UF-E099" in str(err)

    def test_error_code_none(self):
        err = UncertaintyFlowError("test")
        assert err.error_code is None

    def test_model_not_fitted_default_name(self):
        err = ModelNotFittedError()
        assert "Model" in str(err)

    def test_invalid_data_error_message(self):
        err = InvalidDataError("bad shape")
        assert "bad shape" in str(err)
        assert "Invalid data" in str(err)


class TestWarnings:
    def test_warn_calibration_size(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_calibration_size(10, warn_threshold=50)
            assert len(w) == 1
            assert issubclass(w[0].category, UncertaintyFlowWarning)
            assert "10" in str(w[0].message)
            assert "UF-W001" in str(w[0].message)

    def test_warn_quantile_crossing(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_quantile_crossing(15.3)
            assert len(w) == 1
            assert "15.3" in str(w[0].message)
            assert "UF-W002" in str(w[0].message)

    def test_warn_coverage_gap(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_coverage_gap(0.9, 0.82)
            assert len(w) == 1
            assert "0.9" in str(w[0].message)
            assert "UF-W003" in str(w[0].message)

    def test_warn_no_uncertainty_drivers(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_no_uncertainty_drivers()
            assert len(w) == 1
            assert "UF-W004" in str(w[0].message)

    def test_warn_lazyframe_materialized(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_lazyframe_materialized("test reason")
            assert len(w) == 1
            assert "UF-W005" in str(w[0].message)

    def test_warn_copula_auto_selection_ndim(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_copula_auto_selection_ndim(5)
            assert len(w) == 1
            assert "5" in str(w[0].message)
            assert "UF-W006" in str(w[0].message)
