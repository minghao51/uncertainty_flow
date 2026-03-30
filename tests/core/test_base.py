"""Tests for base module."""

import pytest

from uncertainty_flow.core.base import BaseUncertaintyModel


class TestBaseUncertaintyModel:
    """Test BaseUncertaintyModel ABC."""

    def test_cannot_instantiate_base_class(self):
        """BaseUncertaintyModel should not be instantiable."""
        with pytest.raises(TypeError):
            BaseUncertaintyModel()  # type: ignore

    def test_requires_fit_method(self):
        """Subclass must implement fit method."""

        class IncompleteModel(BaseUncertaintyModel):
            def predict(self, data):
                pass  # pragma: no cover

        with pytest.raises(TypeError):
            IncompleteModel()

    def test_requires_predict_method(self):
        """Subclass must implement predict method."""

        class IncompleteModel(BaseUncertaintyModel):
            def fit(self, data, target, **kwargs):
                return self  # pragma: no cover

        with pytest.raises(TypeError):
            IncompleteModel()

    def test_concrete_subclass_works(self):
        """A concrete subclass with both methods should be instantiable."""
        from uncertainty_flow.core.distribution import DistributionPrediction

        class DummyModel(BaseUncertaintyModel):
            def fit(self, data, target, **kwargs):
                self._fitted = True
                return self

            def predict(self, data):
                import numpy as np

                return DistributionPrediction(
                    quantile_matrix=np.zeros((len(data), 11)),
                    quantile_levels=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
                    target_names=["target"],
                )

        model = DummyModel()
        assert isinstance(model, BaseUncertaintyModel)

    def test_uncertainty_drivers_default(self):
        """uncertainty_drivers_ should return None by default."""

        class DummyModel(BaseUncertaintyModel):
            def fit(self, data, target, **kwargs):
                return self

            def predict(self, data):
                pass  # pragma: no cover

        model = DummyModel()
        assert model.uncertainty_drivers_ is None
