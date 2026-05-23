"""Tests for model persistence helpers."""

from __future__ import annotations

import json
import types
import zipfile

import numpy as np
import polars as pl
import pytest
from sklearn.ensemble import GradientBoostingRegressor

from uncertainty_flow.core._persistence import (
    _library_version,
    _quantile_levels,
    _safe_version,
    _target_names,
    _warn_version_mismatches,
    compute_archive_sha256,
)
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
        assert "model_payload_sha256" in loaded.metadata

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
            zf.writestr(
                "metadata.json",
                json.dumps({"format_version": 1, "class_path": "x", "fitted": False}),
            )

        with pytest.raises(ValueError, match="missing required payload 'model.pkl'"):
            BaseUncertaintyModel.load(archive)

    def test_load_missing_metadata_payload_raises(self, tmp_path):
        """Archives without metadata.json should be rejected."""
        archive = tmp_path / "missing_metadata.uf"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("model.pkl", b"not-used")
            # metadata.json intentionally missing

        with pytest.raises(ValueError, match="missing required payload 'metadata.json'"):
            BaseUncertaintyModel.load(archive)

    def test_load_corrupted_pickle_raises(self, tmp_path):
        """Corrupted pickles should raise a clear archive error."""
        archive = tmp_path / "corrupted.uf"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr(
                "metadata.json",
                json.dumps({"format_version": 1, "class_path": "x", "fitted": False}),
            )
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

    def test_payload_checksum_mismatch_raises(self, tabular_data, tmp_path):
        """Tampered model payload should be rejected before deserialization."""
        model = ConformalRegressor(
            base_model=GradientBoostingRegressor(n_estimators=10, random_state=42),
            auto_tune=False,
            random_state=42,
        )
        model.fit(tabular_data, target="target")
        archive = tmp_path / "tampered.uf"
        model.save(archive)

        with zipfile.ZipFile(archive, "a") as zf:
            zf.writestr("model.pkl", b"tampered-model-payload")

        with pytest.raises(ValueError, match="checksum mismatch"):
            BaseUncertaintyModel.load(archive)

    def test_expected_archive_sha256_mismatch_raises(self, tabular_data, tmp_path):
        """Expected archive hash pinning should reject mismatched files."""
        model = ConformalRegressor(
            base_model=GradientBoostingRegressor(n_estimators=10, random_state=42),
            auto_tune=False,
            random_state=42,
        )
        model.fit(tabular_data, target="target")
        archive = tmp_path / "hash_pin.uf"
        model.save(archive)

        actual_hash = compute_archive_sha256(archive)
        assert len(actual_hash) == 64

        with pytest.raises(ValueError, match="Archive SHA-256 mismatch"):
            BaseUncertaintyModel.load(archive, expected_archive_sha256="0" * 64)

    def test_invalid_json_metadata_raises(self, tmp_path):
        """Archives with non-JSON metadata should be rejected."""
        archive = tmp_path / "bad_json.uf"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("metadata.json", b"not-json-at-all")
            zf.writestr("model.pkl", b"not-used")

        with pytest.raises(ValueError, match="metadata.json is not valid JSON"):
            BaseUncertaintyModel.load(archive)

    def test_missing_format_version_raises(self, tmp_path):
        """Archives without format_version should be rejected."""
        archive = tmp_path / "no_version.uf"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("metadata.json", json.dumps({"class_path": "x", "fitted": False}))
            zf.writestr("model.pkl", b"not-used")

        with pytest.raises(ValueError, match="missing required 'format_version'"):
            BaseUncertaintyModel.load(archive)

    def test_unsupported_format_version_raises(self, tmp_path):
        """Archives with unsupported format_version should be rejected."""
        archive = tmp_path / "bad_version.uf"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr(
                "metadata.json",
                json.dumps({"format_version": 99, "class_path": "x", "fitted": False}),
            )
            zf.writestr("model.pkl", b"not-used")

        with pytest.raises(ValueError, match="Unsupported archive format version"):
            BaseUncertaintyModel.load(archive)

    def test_oversized_archive_raises(self, tabular_data, tmp_path, monkeypatch):
        """Archives exceeding max size should be rejected."""
        model = ConformalRegressor(
            base_model=GradientBoostingRegressor(n_estimators=10, random_state=42),
            auto_tune=False,
            random_state=42,
        )
        model.fit(tabular_data, target="target")
        archive = tmp_path / "oversized.uf"
        model.save(archive)

        monkeypatch.setattr("uncertainty_flow.core._persistence.MAX_ARCHIVE_SIZE_BYTES", 1)

        with pytest.raises(ValueError, match="Model archive too large"):
            BaseUncertaintyModel.load(archive)


class TestPersistenceHelpers:
    """Direct tests for internal persistence helpers."""

    def test_safe_version_known_package(self):
        v = _safe_version("numpy")
        assert v is not None
        assert len(v) > 0

    def test_safe_version_unknown_package(self):
        assert _safe_version("non-existent-package-xyz") is None

    def test_library_version_returns_string(self):
        v = _library_version()
        assert isinstance(v, str)
        assert len(v) > 0

    def test_compute_archive_sha256(self, tmp_path):
        p = tmp_path / "test.bin"
        p.write_bytes(b"hello")
        h = compute_archive_sha256(p)
        assert len(h) == 64
        assert isinstance(h, str)

    def test_target_names_from_targets_list(self):
        model = types.SimpleNamespace(targets=["a", "b"])
        assert _target_names(model) == ["a", "b"]

    def test_target_names_from_target_col(self):
        model = types.SimpleNamespace(targets=None, _target_col_="price")
        assert _target_names(model) == ["price"]

    def test_target_names_from_target(self):
        model = types.SimpleNamespace(targets=None, target="sales")
        assert _target_names(model) == ["sales"]

    def test_target_names_returns_none(self):
        model = types.SimpleNamespace()
        assert _target_names(model) is None

    def test_quantile_levels_from_attribute(self):
        model = types.SimpleNamespace(quantile_levels=[0.1, 0.5, 0.9])
        assert _quantile_levels(model) == [0.1, 0.5, 0.9]

    def test_quantile_levels_default_quantiles(self):
        model = types.SimpleNamespace(_quantiles_=True)
        levels = _quantile_levels(model)
        assert levels is not None
        assert all(0 < q < 1 for q in levels)

    def test_quantile_levels_returns_none(self):
        model = types.SimpleNamespace()
        assert _quantile_levels(model) is None

    def test_warn_version_mismatches_no_warning(self, recwarn):
        _warn_version_mismatches({"dependencies": {"numpy": _safe_version("numpy")}})
        assert len(recwarn) == 0

    def test_warn_version_mismatches_with_warning(self, recwarn):
        _warn_version_mismatches({"dependencies": {"numpy": "0.0.1"}})
        assert len(recwarn) >= 1

    def test_warn_version_mismatches_unknown_dep_skipped(self, recwarn):
        _warn_version_mismatches({"dependencies": {"unknown_lib": "1.0.0"}})
        assert len(recwarn) == 0

    def test_warn_version_mismatches_non_dict_deps_skipped(self, recwarn):
        _warn_version_mismatches({"dependencies": None})
        assert len(recwarn) == 0


class TestHMACSigning:
    """Tests for HMAC-SHA256 signing and verification."""

    def test_hmac_not_stored_by_default(self, tabular_data, tmp_path):
        model = ConformalRegressor(
            base_model=GradientBoostingRegressor(n_estimators=10, random_state=42),
            auto_tune=False,
            random_state=42,
        )
        model.fit(tabular_data, target="target")
        archive = tmp_path / "default.uf"
        model.save(archive)

        loaded = BaseUncertaintyModel.load(archive)
        assert "model_payload_hmac" not in loaded.metadata
        assert "model_payload_sha256" in loaded.metadata

    def test_hmac_stored_when_enabled_with_key(self, tabular_data, tmp_path):
        model = ConformalRegressor(
            base_model=GradientBoostingRegressor(n_estimators=10, random_state=42),
            auto_tune=False,
            random_state=42,
        )
        model.fit(tabular_data, target="target")
        archive = tmp_path / "signed.uf"
        key = b"test-secret-key-32-bytes-long!!!!!"

        from uncertainty_flow.core._persistence import save_model_archive

        meta = save_model_archive(model, archive, hmac_sign=True, signing_key=key)
        assert "model_payload_hmac" in meta

    def test_hmac_requires_explicit_key(self, tabular_data, tmp_path):
        model = ConformalRegressor(
            base_model=GradientBoostingRegressor(n_estimators=10, random_state=42),
            auto_tune=False,
            random_state=42,
        )
        model.fit(tabular_data, target="target")
        archive = tmp_path / "no_key.uf"

        from uncertainty_flow.core._persistence import save_model_archive

        with pytest.raises(ValueError, match="hmac_sign=True requires an explicit signing_key"):
            save_model_archive(model, archive, hmac_sign=True, signing_key=None)

    def test_tampered_payload_rejected_with_hmac_key(self, tabular_data, tmp_path):
        model = ConformalRegressor(
            base_model=GradientBoostingRegressor(n_estimators=10, random_state=42),
            auto_tune=False,
            random_state=42,
        )
        model.fit(tabular_data, target="target")
        archive = tmp_path / "tampered_hmac.uf"
        key = b"test-secret-key-32-bytes-long!!!!!"

        from uncertainty_flow.core._persistence import load_model_archive, save_model_archive

        save_model_archive(model, archive, hmac_sign=True, signing_key=key)

        with zipfile.ZipFile(archive) as zf:
            meta = json.loads(zf.read("metadata.json"))
        meta.pop("model_payload_sha256", None)

        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("model.pkl", b"tampered-payload")
            zf.writestr("metadata.json", json.dumps(meta))

        with pytest.raises(ValueError, match="HMAC signature verification failed"):
            load_model_archive(archive, signing_key=key)

    def test_load_warns_when_key_provided_but_no_hmac(self, tabular_data, tmp_path):
        model = ConformalRegressor(
            base_model=GradientBoostingRegressor(n_estimators=10, random_state=42),
            auto_tune=False,
            random_state=42,
        )
        model.fit(tabular_data, target="target")
        archive = tmp_path / "no_hmac.uf"
        model.save(archive)

        from uncertainty_flow.core._persistence import load_model_archive

        with pytest.warns(UserWarning, match="no HMAC signature"):
            load_model_archive(archive, signing_key=b"some-key")

    def test_valid_hmac_loads_successfully(self, tabular_data, tmp_path):
        model = ConformalRegressor(
            base_model=GradientBoostingRegressor(n_estimators=10, random_state=42),
            auto_tune=False,
            random_state=42,
        )
        model.fit(tabular_data, target="target")
        archive = tmp_path / "valid_hmac.uf"
        key = b"test-secret-key-32-bytes-long!!!!!"

        from uncertainty_flow.core._persistence import load_model_archive, save_model_archive

        save_model_archive(model, archive, hmac_sign=True, signing_key=key)
        loaded, meta = load_model_archive(archive, signing_key=key)
        assert "model_payload_hmac" in meta
        assert loaded.metadata is not None or hasattr(loaded, "_metadata")
