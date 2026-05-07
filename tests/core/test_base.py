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

    def test_metadata_default_none_for_unfitted_model(self):
        """metadata should default to None before fitting or loading."""

        class DummyModel(BaseUncertaintyModel):
            def fit(self, data, target, **kwargs):
                return self

            def predict(self, data):
                pass  # pragma: no cover

        model = DummyModel()
        assert model.metadata is None

    def test_analyze_leverage_method_exists(self):
        """analyze_leverage should be a method on BaseUncertaintyModel."""

        class DummyModel(BaseUncertaintyModel):
            def fit(self, data, target, **kwargs):
                return self

            def predict(self, data):
                pass  # pragma: no cover

        model = DummyModel()
        assert hasattr(model, "analyze_leverage")
        assert callable(model.analyze_leverage)

    def test_explain_interval_width_method_exists(self):
        """explain_interval_width should be a method on BaseUncertaintyModel."""

        class DummyModel(BaseUncertaintyModel):
            def fit(self, data, target, **kwargs):
                return self

            def predict(self, data):
                pass  # pragma: no cover

        model = DummyModel()
        assert hasattr(model, "explain_interval_width")
        assert callable(model.explain_interval_width)

    def test_predict_batch_yields_chunks(self):
        import numpy as np
        import polars as pl

        from uncertainty_flow.core.distribution import DistributionPrediction

        class DummyModel(BaseUncertaintyModel):
            def fit(self, data, target, **kwargs):
                self._fitted = True
                return self

            def predict(self, data):
                n = len(data)
                return DistributionPrediction(
                    quantile_matrix=np.zeros((n, 3)),
                    quantile_levels=[0.1, 0.5, 0.9],
                    target_names=["y"],
                )

        model = DummyModel()
        model._fitted = True
        df = pl.DataFrame({"x": np.arange(25)})

        chunks = list(model.predict_batch(df, batch_size=10))
        assert len(chunks) == 3
        assert chunks[0]._n_samples == 10
        assert chunks[1]._n_samples == 10
        assert chunks[2]._n_samples == 5

    def test_predict_batch_default_size(self):
        import numpy as np
        import polars as pl

        from uncertainty_flow.core.distribution import DistributionPrediction

        class DummyModel(BaseUncertaintyModel):
            def fit(self, data, target, **kwargs):
                self._fitted = True
                return self

            def predict(self, data):
                n = len(data)
                return DistributionPrediction(
                    quantile_matrix=np.zeros((n, 3)),
                    quantile_levels=[0.1, 0.5, 0.9],
                    target_names=["y"],
                )

        model = DummyModel()
        model._fitted = True
        df = pl.DataFrame({"x": np.arange(500)})

        chunks = list(model.predict_batch(df))
        assert len(chunks) == 1
        assert chunks[0]._n_samples == 500

    def test_metadata_fitted_model(self):
        class DummyModel(BaseUncertaintyModel):
            def fit(self, data, target, **kwargs):
                self._fitted = True
                return self

            def predict(self, data):
                pass  # pragma: no cover

        model = DummyModel()
        model._fitted = True
        meta = model.metadata
        assert isinstance(meta, dict)
        assert "class_path" in meta

    def test_metadata_with_cached_value(self):
        class DummyModel(BaseUncertaintyModel):
            def fit(self, data, target, **kwargs):
                return self

            def predict(self, data):
                pass  # pragma: no cover

        model = DummyModel()
        model._metadata = {"custom": True}
        assert model.metadata == {"custom": True}

    def test_uncertainty_drivers_returns_none(self):
        class DummyModel(BaseUncertaintyModel):
            def fit(self, data, target, **kwargs):
                return self

            def predict(self, data):
                pass  # pragma: no cover

        model = DummyModel()
        assert model.uncertainty_drivers_ is None

    def test_calibration_report_delegates(self):
        import numpy as np
        import polars as pl

        from uncertainty_flow.core.distribution import DistributionPrediction

        class DummyModel(BaseUncertaintyModel):
            def fit(self, data, target, **kwargs):
                self._fitted = True
                return self

            def predict(self, data):
                n = len(data)
                return DistributionPrediction(
                    quantile_matrix=np.zeros((n, 3)),
                    quantile_levels=[0.1, 0.5, 0.9],
                    target_names=["y"],
                )

        model = DummyModel()
        model._fitted = True
        df = pl.DataFrame({"x": np.arange(20).astype(float), "y": np.arange(20).astype(float)})
        report = model.calibration_report(df, target="y")
        assert isinstance(report, pl.DataFrame)
