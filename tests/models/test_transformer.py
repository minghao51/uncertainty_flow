"""Tests for TransformerForecaster."""

import polars as pl
import pytest

from uncertainty_flow.core.config import CHRONOS_MODELS, get_config


class TestTransformerForecasterInit:
    """Test TransformerForecaster initialization."""

    def test_init_default_params(self):
        """Should initialize with default parameters."""
        from uncertainty_flow.models import TransformerForecaster

        model = TransformerForecaster(target="value")
        assert model.target == "value"
        assert model.horizon == 24
        assert model.model_name == get_config().default_chronos_model
        assert model.calibration_size == 0.2
        assert model.device == "auto"
        assert model.random_state is None

    def test_init_custom_params(self):
        """Should initialize with custom parameters."""
        from uncertainty_flow.models import TransformerForecaster

        model = TransformerForecaster(
            target="price",
            horizon=48,
            model_name="chronos-2",
            calibration_size=0.3,
            device="cpu",
            random_state=42,
        )
        assert model.target == "price"
        assert model.horizon == 48
        assert model.model_name == "chronos-2"
        assert model.calibration_size == 0.3
        assert model.device == "cpu"
        assert model.random_state == 42

    def test_init_invalid_model_name(self):
        """Should raise error for invalid model name."""
        from uncertainty_flow.models import TransformerForecaster

        with pytest.raises(ValueError, match="Unknown model_name"):
            TransformerForecaster(target="y", model_name="invalid-model")

    def test_chronos_models_config(self):
        """Should have valid Chronos model mappings."""
        assert "chronos-2-small" in CHRONOS_MODELS
        assert "chronos-2" in CHRONOS_MODELS
        assert "chronos-2-tiny" in CHRONOS_MODELS
        assert CHRONOS_MODELS["chronos-2-small"] == "amazon/chronos-2-small"
        assert CHRONOS_MODELS["chronos-2"] == "amazon/chronos-2"

    def test_uncertainty_features_param(self):
        """Should accept uncertainty_features parameter."""
        from uncertainty_flow.models import TransformerForecaster

        model = TransformerForecaster(
            target="y",
            uncertainty_features=["feature1", "feature2"],
        )
        assert model.uncertainty_features == ["feature1", "feature2"]


class TestTransformerForecasterUnfitted:
    """Test TransformerForecaster before fitting."""

    def test_predict_raises_when_not_fitted(self):
        """Should raise error when predict called before fit."""
        from uncertainty_flow.models import TransformerForecaster
        from uncertainty_flow.utils.exceptions import ModelNotFittedError

        model = TransformerForecaster(target="value")
        df = pl.DataFrame({"value": [1, 2, 3]})

        with pytest.raises(ModelNotFittedError, match="not fitted"):
            model.predict(df)

    def test_uncertainty_drivers_returns_none_before_fit(self):
        """Should return None for uncertainty_drivers_ before fit."""
        from uncertainty_flow.models import TransformerForecaster

        model = TransformerForecaster(target="value")
        assert model.uncertainty_drivers_ is None


class TestTransformerForecasterInterface:
    """Test TransformerForecaster interface compliance."""

    def test_is_base_uncertainty_model(self):
        """Should inherit from BaseUncertaintyModel."""
        from uncertainty_flow.core.base import BaseUncertaintyModel
        from uncertainty_flow.models import TransformerForecaster

        assert issubclass(TransformerForecaster, BaseUncertaintyModel)

    def test_has_fit_and_predict_methods(self):
        """Should have fit and predict methods."""
        from uncertainty_flow.models import TransformerForecaster

        model = TransformerForecaster(target="y")
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")
        assert callable(model.fit)
        assert callable(model.predict)

    def test_has_uncertainty_drivers_property(self):
        """Should have uncertainty_drivers_ property."""
        from uncertainty_flow.models import TransformerForecaster

        model = TransformerForecaster(target="y")
        assert hasattr(model, "uncertainty_drivers_")
