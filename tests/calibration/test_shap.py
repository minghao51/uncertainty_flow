"""Tests for SHAP-based uncertainty attribution."""

import polars as pl
import pytest


class TestUncertaintyShapInterface:
    """Test uncertainty_shap interface."""

    def test_import_from_calibration(self):
        """Should import from calibration module."""
        from uncertainty_flow.calibration import uncertainty_shap

        assert callable(uncertainty_shap)

    def test_function_signature(self):
        """Should have correct function signature."""
        import inspect

        from uncertainty_flow.calibration.shap_values import uncertainty_shap

        sig = inspect.signature(uncertainty_shap)
        params = sig.parameters

        assert "model" in params
        assert "X" in params
        assert "background" in params
        assert "quantile_pairs" in params

    def test_background_defaults_to_none(self):
        """Background should default to None per roadmap spec."""
        import inspect

        from uncertainty_flow.calibration.shap_values import uncertainty_shap

        sig = inspect.signature(uncertainty_shap)
        default = sig.parameters["background"].default

        assert default is None

    def test_quantile_pairs_defaults_to_none(self):
        """Quantile pairs should default to None."""
        import inspect

        from uncertainty_flow.calibration.shap_values import uncertainty_shap

        sig = inspect.signature(uncertainty_shap)
        default = sig.parameters["quantile_pairs"].default

        assert default is None


class TestUncertaintyShapErrorHandling:
    """Test error handling in uncertainty_shap."""

    def test_raises_import_error_when_shap_not_available(self):
        """Should raise ImportError if shap is not installed."""
        import importlib.util

        if importlib.util.find_spec("shap") is not None:
            pytest.skip("shap is installed")

        from uncertainty_flow.calibration.shap_values import uncertainty_shap

        model = object()
        X = pl.DataFrame({"a": [1, 2, 3]})  # noqa: N806

        with pytest.raises(ImportError, match="shap is required"):
            uncertainty_shap(model, X)


class TestShapValuesAccessibility:
    """Test that shap_values module is accessible."""

    def test_module_exists(self):
        """Module should be importable."""
        from uncertainty_flow.calibration import shap_values

        assert hasattr(shap_values, "uncertainty_shap")

    def test_in_calibration_all(self):
        """Should be exported in calibration.__all__."""
        from uncertainty_flow.calibration import __all__

        assert "uncertainty_shap" in __all__
