import polars as pl
import pytest

from uncertainty_flow.benchmarking.datasets import (
    AVAILABLE_DATASETS,
    DatasetInfo,
    get_numeric_cols,
    list_datasets,
    list_datasets_by_domain,
)
from uncertainty_flow.utils.exceptions import ConfigurationError


class TestListDatasets:
    def test_returns_sorted_list(self):
        datasets = list_datasets()
        names = [d.name for d in datasets]
        assert names == sorted(names)

    def test_all_entries_are_dataset_info(self):
        for d in list_datasets():
            assert isinstance(d, DatasetInfo)

    def test_includes_known_datasets(self):
        names = {d.name for d in list_datasets()}
        assert "weather" in names
        assert "concrete" in names


class TestListDatasetsByDomain:
    def test_groups_by_domain(self):
        by_domain = list_datasets_by_domain()
        assert "Climate" in by_domain
        assert "Finance" in by_domain
        assert all(isinstance(v, list) for v in by_domain.values())

    def test_all_entries_valid(self):
        for domain, datasets in list_datasets_by_domain().items():
            for d in datasets:
                assert isinstance(d, DatasetInfo)
                assert d.domain == domain


class TestAvailableDatasets:
    def test_registry_not_empty(self):
        assert len(AVAILABLE_DATASETS) > 10

    def test_local_datasets_have_empty_hf_path(self):
        local = [d for d in AVAILABLE_DATASETS.values() if d.is_local]
        for d in local:
            assert d.hf_path == ""

    def test_remote_datasets_have_hf_path(self):
        remote = [d for d in AVAILABLE_DATASETS.values() if not d.is_local]
        for d in remote:
            assert d.hf_path != ""


class TestGetNumericCols:
    def test_filters_string_columns(self):
        df = pl.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"], "c": [3, 4]})
        assert get_numeric_cols(df) == ["a", "c"]

    def test_all_numeric(self):
        df = pl.DataFrame({"x": [1.0], "y": [2.0]})
        assert get_numeric_cols(df) == ["x", "y"]


class TestLoadDatasetValidation:
    def test_unknown_dataset_raises(self):
        with pytest.raises(ConfigurationError, match="not found"):
            from uncertainty_flow.benchmarking.datasets import load_dataset

            load_dataset("nonexistent_xyz_dataset")
