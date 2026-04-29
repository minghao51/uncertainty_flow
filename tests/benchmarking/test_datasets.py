"""Tests for uncertainty_flow.benchmarking.datasets."""

from types import SimpleNamespace

import polars as pl
import pyarrow as pa
import pytest

from uncertainty_flow.benchmarking.datasets import (
    AVAILABLE_DATASETS,
    DATASETS_DIR,
    DatasetInfo,
    load_dataset,
    load_local_dataset,
)
from uncertainty_flow.utils.exceptions import ConfigurationError


class TestDatasetRevisionPinning:
    """Security tests for HF revision pinning."""

    def test_rejects_unpinned_remote_dataset(self):
        """Remote dataset loading should fail without a revision pin."""
        with pytest.raises(ConfigurationError, match="pinned HuggingFace revision"):
            load_dataset("weather")

    def test_accepts_explicit_revision_suffix(self, monkeypatch):
        """Dataset name suffix @revision should be passed to HF loader."""
        captured: dict[str, str] = {}

        def _fake_load_dataset(path, *args, **kwargs):
            del args
            captured["path"] = path
            captured["revision"] = kwargs["revision"]
            arrow = pa.table({"OT": [1.0, 2.0], "feature": [10.0, 20.0]})
            return SimpleNamespace(data=SimpleNamespace(table=arrow))

        monkeypatch.setitem(
            __import__("sys").modules,
            "datasets",
            SimpleNamespace(load_dataset=_fake_load_dataset),
        )

        df, ds_info = load_dataset("owner/example@deadbeef", split="train")

        assert isinstance(df, pl.DataFrame)
        assert ds_info.hf_path == "owner/example"
        assert captured["path"] == "owner/example"
        assert captured["revision"] == "deadbeef"


_LOCAL_DATASETS = [
    ("concrete", "strength"),
    ("wine_quality", "quality"),
    ("energy_efficiency", "y1"),
    ("synthetic_heteroscedastic", "y"),
    ("synthetic_regime_switching", "y"),
    ("synthetic_heavy_tailed", "y"),
    ("financial_volatility", "OT_diff"),
    ("fraud", "isFraud"),
    ("synthetic_multivariate", "y1"),
]


def _parquet_exists(name: str) -> bool:
    ds_info = AVAILABLE_DATASETS[name]
    return (DATASETS_DIR / f"{ds_info.name}.parquet").exists()


@pytest.mark.parametrize("name,target", _LOCAL_DATASETS)
def test_local_dataset_loads_with_target(name: str, target: str):
    if not _parquet_exists(name):
        pytest.skip(f"{name}.parquet not present locally")
    df, ds_info = load_local_dataset(name)
    assert target in df.columns
    assert len(df) > 0
    assert isinstance(ds_info, DatasetInfo)
    assert ds_info.is_local is True


@pytest.mark.parametrize("name,target", _LOCAL_DATASETS)
def test_local_dataset_has_multiple_numeric_columns(name: str, target: str):
    if not _parquet_exists(name):
        pytest.skip(f"{name}.parquet not present locally")
    df, _ = load_local_dataset(name)
    numeric = [c for c in df.columns if df[c].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)]
    assert len(numeric) >= 2, f"{name} should have >= 2 numeric columns, got {df.columns}"


@pytest.mark.parametrize("name,target", _LOCAL_DATASETS)
def test_local_dataset_no_nulls_in_target(name: str, target: str):
    if not _parquet_exists(name):
        pytest.skip(f"{name}.parquet not present locally")
    df, _ = load_local_dataset(name)
    assert df[target].null_count() == 0, f"{name} target column '{target}' has nulls"


def test_energy_efficiency_has_dual_targets():
    if not _parquet_exists("energy_efficiency"):
        pytest.skip("energy_efficiency.parquet not present locally")
    df, _ = load_local_dataset("energy_efficiency")
    assert "y1" in df.columns
    assert "y2" in df.columns


def test_synthetic_regime_switching_has_regime_column():
    if not _parquet_exists("synthetic_regime_switching"):
        pytest.skip("synthetic_regime_switching.parquet not present locally")
    df, _ = load_local_dataset("synthetic_regime_switching")
    assert "regime" in df.columns
    assert set(df["regime"].unique().to_list()).issubset({0, 1})


def test_registry_entries_consistent():
    for name, target in _LOCAL_DATASETS:
        ds_info = AVAILABLE_DATASETS[name]
        assert ds_info.default_target == target
        assert ds_info.is_local is True
        assert ds_info.hf_path == ""


def test_load_local_dataset_rejects_unknown():
    with pytest.raises(ConfigurationError, match="not in available datasets"):
        load_local_dataset("nonexistent_dataset_xyz")


def test_synthetic_multivariate_has_triple_targets():
    if not _parquet_exists("synthetic_multivariate"):
        pytest.skip("synthetic_multivariate.parquet not present locally")
    df, _ = load_local_dataset("synthetic_multivariate")
    for t in ("y1", "y2", "y3"):
        assert t in df.columns


def test_fraud_has_class_imbalance():
    if not _parquet_exists("fraud"):
        pytest.skip("fraud.parquet not present locally")
    df, _ = load_local_dataset("fraud")
    fraud_rate = df["isFraud"].sum() / len(df)
    assert 0.001 < fraud_rate < 0.05, f"Expected rare fraud rate, got {fraud_rate:.4f}"


def test_financial_volatility_has_volatility_features():
    if not _parquet_exists("financial_volatility"):
        pytest.skip("financial_volatility.parquet not present locally")
    df, _ = load_local_dataset("financial_volatility")
    vol_cols = [c for c in df.columns if "vol" in c.lower()]
    assert len(vol_cols) >= 4, f"Expected volatility features, got columns: {df.columns}"
