"""Tests for model persistence helpers."""

from __future__ import annotations

import json
import zipfile

import numpy as np
import polars as pl
import pytest
from sklearn.ensemble import GradientBoostingRegressor

from uncertainty_flow.core.base import BaseUncertaintyModel
from uncertainty_flow.models import DeepQuantileNet, QuantileForestForecaster
from uncertainty_flow.wrappers import ConformalForecaster, ConformalRegressor


@pytest.fixture
def tabular_data():
    """Create a small tabular dataset."""
    rng = np.random.default_rng(42)
    n = 120
    return pl.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "target": 2.0 * rng.normal(size=n) + 5.0,
        }
    )


@pytest.fixture
def time_series_data():
    """Create a small multivariate time series dataset."""
    rng = np.random.default_rng(42)
    n = 140
    return pl.DataFrame(
        {
            "date": np.arange(n),
            "price": 10.0 + np.linspace(0, 3, n) + rng.normal(scale=0.2, size=n),
            "volume": 100.0 + np.linspace(0, 12, n) + rng.normal(scale=1.0, size=n),
        }
    )


class TestBasePersistenceContract:
    """Persistence behavior on the base class surface."""

    def test_concrete_subclass_exposes_save_and_load(self):
        """Concrete models should inherit the persistence contract."""
        model = QuantileForestForecaster(targets="price", horizon=3, auto_tune=False)
        assert callable(model.save)
        assert callable(model.load)

    def test_unfitted_model_metadata_defaults_to_none(self):
        """Fresh unfitted models should not claim persisted metadata."""
        model = ConformalRegressor(base_model=GradientBoostingRegressor(random_state=42))
        assert model.metadata is None


class TestModelRoundTrip:
    """Round-trip save/load coverage for core fitted models."""

    def test_conformal_regressor_round_trip(self, tabular_data, tmp_path):
        """ConformalRegressor predictions should match after load."""
        model = ConformalRegressor(
            base_model=GradientBoostingRegressor(n_estimators=10, random_state=42),
            auto_tune=False,
            random_state=42,
        )
        model.fit(tabular_data, target="target")
        original = model.predict(tabular_data)

        archive = tmp_path / "conformal_regressor.uf"
        model.save(archive)
        loaded = ConformalRegressor.load(archive)
        restored = loaded.predict(tabular_data)

        np.testing.assert_allclose(original._quantiles, restored._quantiles)
        assert loaded.metadata is not None
        assert loaded.metadata["class_path"].endswith("ConformalRegressor")
        assert loaded.metadata["fitted"] is True

    def test_conformal_forecaster_round_trip(self, time_series_data, tmp_path):
        """ConformalForecaster should preserve multivariate predictions and sampling."""
        model = ConformalForecaster(
            base_model=GradientBoostingRegressor(n_estimators=10, random_state=42),
            horizon=3,
            targets=["price", "volume"],
            copula_family="gaussian",
            auto_tune=False,
            random_state=42,
        )
        model.fit(time_series_data)
        original = model.predict(time_series_data)

        archive = tmp_path / "conformal_forecaster.uf"
        model.save(archive)
        loaded = ConformalForecaster.load(archive)
        restored = loaded.predict(time_series_data)

        np.testing.assert_allclose(original._quantiles, restored._quantiles)
        assert (
            restored.sample(4, random_state=123).shape == original.sample(4, random_state=123).shape
        )

    def test_quantile_forest_round_trip(self, time_series_data, tmp_path):
        """QuantileForestForecaster should preserve predictions and copula-backed sampling."""
        model = QuantileForestForecaster(
            targets=["price", "volume"],
            horizon=3,
            n_estimators=12,
            copula_family="gaussian",
            auto_tune=False,
            random_state=42,
        )
        model.fit(time_series_data)
        original = model.predict(time_series_data)

        archive = tmp_path / "quantile_forest.uf"
        model.save(archive)
        loaded = QuantileForestForecaster.load(archive)
        restored = loaded.predict(time_series_data)

        np.testing.assert_allclose(original._quantiles, restored._quantiles)
        assert restored.sample(3, random_state=123).columns == ["sample_id", "price", "volume"]

    def test_deep_quantile_round_trip(self, tabular_data, tmp_path):
        """DeepQuantileNet should round-trip through the shared persistence layer."""
        model = DeepQuantileNet(
            hidden_layer_sizes=(8,),
            trunk_max_iter=20,
            random_state=42,
        )
        model.fit(tabular_data, target="target")
        original = model.predict(tabular_data)

        archive = tmp_path / "deep_quantile.uf"
        model.save(archive)
        loaded = DeepQuantileNet.load(archive)
        restored = loaded.predict(tabular_data)

        np.testing.assert_allclose(original._quantiles, restored._quantiles, atol=1e-8)

    def test_save_without_extended_metadata_still_loads(self, tabular_data, tmp_path):
        """Minimal metadata archives should still be readable."""
        model = ConformalRegressor(
            base_model=GradientBoostingRegressor(n_estimators=10, random_state=42),
            auto_tune=False,
            random_state=42,
        )
        model.fit(tabular_data, target="target")

        archive = tmp_path / "minimal.uf"
        model.save(archive, include_metadata=False)
        loaded = BaseUncertaintyModel.load(archive)

        assert loaded.metadata is not None
        assert loaded.metadata == {
            "class_path": "uncertainty_flow.wrappers.conformal.ConformalRegressor",
            "fitted": True,
            "format_version": 1,
        }


class TestPersistenceFailures:
    """Failure modes for malformed or mismatched archives."""

    def test_load_missing_path_raises(self, tmp_path):
        """Loading a missing archive should fail clearly."""
        with pytest.raises(FileNotFoundError, match="Model archive not found"):
            BaseUncertaintyModel.load(tmp_path / "missing.uf")

    def test_load_missing_model_payload_raises(self, tmp_path):
        """Archives without the pickled model payload should be rejected."""
        archive = tmp_path / "missing_model.uf"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("metadata.json", json.dumps({"class_path": "x", "fitted": False}))

        with pytest.raises(ValueError, match="missing required payload 'model.pkl'"):
            BaseUncertaintyModel.load(archive)

    def test_load_missing_metadata_payload_raises(self, tmp_path):
        """Archives without metadata.json should be rejected."""
        archive = tmp_path / "missing_metadata.uf"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("model.pkl", b"not-used")

        with pytest.raises(ValueError, match="missing required payload 'metadata.json'"):
            BaseUncertaintyModel.load(archive)

    def test_load_corrupted_pickle_raises(self, tmp_path):
        """Corrupted pickles should raise a clear archive error."""
        archive = tmp_path / "corrupted.uf"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("metadata.json", json.dumps({"class_path": "x", "fitted": False}))
            zf.writestr("model.pkl", b"not-a-pickle")

        with pytest.raises(ValueError, match="failed to deserialize model payload"):
            BaseUncertaintyModel.load(archive)

    def test_wrong_class_load_raises(self, tabular_data, tmp_path):
        """Subclass load should enforce the expected archive type."""
        model = ConformalRegressor(
            base_model=GradientBoostingRegressor(n_estimators=10, random_state=42),
            auto_tune=False,
            random_state=42,
        )
        model.fit(tabular_data, target="target")
        archive = tmp_path / "wrong_class.uf"
        model.save(archive)

        with pytest.raises(TypeError, match="not an instance"):
            QuantileForestForecaster.load(archive)
