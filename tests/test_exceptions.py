"""Tests for exception hierarchy."""

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
)


class TestExceptionHierarchy:
    def test_base_error_inherits_from_value_error(self):
        assert issubclass(UncertaintyFlowError, ValueError)

    def test_model_error_hierarchy(self):
        assert issubclass(ModelError, UncertaintyFlowError)
        assert issubclass(ModelNotFittedError, ModelError)

    def test_data_error_hierarchy(self):
        assert issubclass(DataError, UncertaintyFlowError)
        assert issubclass(InvalidDataError, DataError)

    def test_calibration_error_hierarchy(self):
        assert issubclass(CalibrationError, UncertaintyFlowError)
        assert issubclass(CalibrationSizeError, CalibrationError)

    def test_configuration_error_hierarchy(self):
        assert issubclass(ConfigurationError, UncertaintyFlowError)
        assert issubclass(QuantileError, ConfigurationError)


class TestErrorCodes:
    def test_model_not_fitted_error_code(self):
        error = ModelNotFittedError("TestModel")
        assert "UF-E002" in str(error)
        assert "TestModel" in str(error)

    def test_invalid_data_error_code(self):
        error = InvalidDataError("test reason")
        assert "UF-E003" in str(error)
        assert "test reason" in str(error)

    def test_calibration_size_error_code(self):
        error = CalibrationSizeError(10, 20)
        assert "UF-E001" in str(error)
        assert "10" in str(error)
        assert "20" in str(error)

    def test_quantile_error_code(self):
        error = QuantileError("test reason")
        assert "UF-E004" in str(error)
        assert "test reason" in str(error)


class TestErrorRaising:
    def test_raise_model_not_fitted(self):
        with pytest.raises(ModelNotFittedError) as exc_info:
            raise ModelNotFittedError("TestModel")
        assert "TestModel" in str(exc_info.value)
        assert "UF-E002" in str(exc_info.value)

    def test_raise_invalid_data(self):
        with pytest.raises(InvalidDataError) as exc_info:
            raise InvalidDataError("test reason")
        assert "test reason" in str(exc_info.value)
        assert "UF-E003" in str(exc_info.value)

    def test_raise_calibration_too_small(self):
        with pytest.raises(CalibrationSizeError) as exc_info:
            raise CalibrationSizeError(10)
        assert "10" in str(exc_info.value)
        assert "UF-E001" in str(exc_info.value)

    def test_raise_calibration_too_small_custom_min(self):
        with pytest.raises(CalibrationSizeError) as exc_info:
            raise CalibrationSizeError(5, min_size=10)
        assert "5" in str(exc_info.value)
        assert "10" in str(exc_info.value)

    def test_raise_quantile_invalid(self):
        with pytest.raises(QuantileError) as exc_info:
            raise QuantileError("test reason")
        assert "test reason" in str(exc_info.value)
        assert "UF-E004" in str(exc_info.value)


class TestBackwardCompatibility:
    def test_model_not_fitted_is_value_error(self):
        with pytest.raises(ValueError):
            raise ModelNotFittedError("TestModel")

    def test_invalid_data_is_value_error(self):
        with pytest.raises(ValueError):
            raise InvalidDataError("test reason")

    def test_calibration_size_is_value_error(self):
        with pytest.raises(ValueError):
            raise CalibrationSizeError(10)


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
            warnings.warn(
                "Calibration set has only 10 samples. "
                "Coverage guarantees may be unreliable. [UF-W001]",
                UncertaintyFlowWarning,
                stacklevel=2,
            )
            assert len(w) == 1
            assert issubclass(w[0].category, UncertaintyFlowWarning)
            assert "10" in str(w[0].message)
            assert "UF-W001" in str(w[0].message)

    def test_warn_quantile_crossing(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warnings.warn(
                "Quantile crossing detected in 15.3% of predictions. "
                "Post-sort applied. Consider re-evaluating base model quality. [UF-W002]",
                UncertaintyFlowWarning,
                stacklevel=2,
            )
            assert len(w) == 1
            assert "15.3" in str(w[0].message)
            assert "UF-W002" in str(w[0].message)

    def test_warn_coverage_gap(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warnings.warn(
                "Requested 0.9 coverage but achieved 0.82. Model may be miscalibrated. [UF-W003]",
                UncertaintyFlowWarning,
                stacklevel=2,
            )
            assert len(w) == 1
            assert "0.9" in str(w[0].message)
            assert "UF-W003" in str(w[0].message)

    def test_warn_no_uncertainty_drivers(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warnings.warn(
                "Residual correlation analysis found no significant drivers. "
                "Intervals may be uniformly conservative. [UF-W004]",
                UncertaintyFlowWarning,
                stacklevel=2,
            )
            assert len(w) == 1
            assert "UF-W004" in str(w[0].message)

    def test_warn_lazyframe_materialized(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warnings.warn(
                "LazyFrame collected earlier than expected due to test reason. "
                "Consider restructuring upstream pipeline. [UF-W005]",
                UncertaintyFlowWarning,
                stacklevel=2,
            )
            assert len(w) == 1
            assert "UF-W005" in str(w[0].message)

    def test_warn_copula_auto_selection_ndim(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warnings.warn(
                "Auto-selecting copula for 5D data. "
                "Only Gaussian copula supports dimensions > 2. [UF-W006]",
                UncertaintyFlowWarning,
                stacklevel=2,
            )
            assert len(w) == 1
            assert "5" in str(w[0].message)
            assert "UF-W006" in str(w[0].message)
