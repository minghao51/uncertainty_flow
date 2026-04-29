"""Tests for CLI commands."""

import json
import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from uncertainty_flow.cli import cli

runner = CliRunner()

try:
    import datasets  # noqa: F401

    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False


@pytest.fixture
def tmp_path():
    """Create a temporary directory for file operations."""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


class TestListDatasets:
    """Test list-datasets command."""

    def test_lists_all_datasets(self):
        """Should list all available datasets."""
        result = runner.invoke(cli, ["list-datasets"])
        assert result.exit_code == 0
        assert "Available datasets" in result.output
        assert "Available datasets" in result.output

    def test_filters_by_domain(self):
        """Should filter by domain."""
        result = runner.invoke(cli, ["list-datasets", "--domain", "Energy"])
        assert result.exit_code == 0
        assert "Energy" in result.output
        assert "Finance" not in result.output

    def test_invalid_domain_shows_available(self):
        """Should show available domains on invalid domain."""
        result = runner.invoke(cli, ["list-datasets", "--domain", "InvalidDomain"])
        assert result.exit_code == 1
        assert "Available domains:" in result.output


class TestDownloadDataset:
    """Test download-dataset command."""

    @pytest.mark.network
    @pytest.mark.skipif(
        not DATASETS_AVAILABLE, reason="datasets optional dependency not installed"
    )
    def test_downloads_dataset(self, tmp_path, monkeypatch):
        """Should download dataset to cache."""
        monkeypatch.setenv("UNCERTAINTY_FLOW_HF_REVISION", "main")
        result = runner.invoke(cli, ["download-dataset", "weather", "--cache-dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "Dataset saved to:" in result.output
        assert tmp_path.exists()

    def test_invalid_dataset_shows_error(self):
        """Should error on invalid dataset name."""
        result = runner.invoke(cli, ["download-dataset", "invalid-dataset"])
        assert result.exit_code == 1
        assert "Error downloading" in result.output


class TestBenchmark:
    """Test benchmark command."""

    @pytest.mark.integration
    @pytest.mark.optional
    @pytest.mark.skipif(
        not DATASETS_AVAILABLE, reason="datasets optional dependency not installed"
    )
    def test_runs_benchmark_with_all_models(self, tmp_path, monkeypatch):
        """Should run benchmark with all models."""
        monkeypatch.setenv("UNCERTAINTY_FLOW_HF_REVISION", "main")
        output = tmp_path / "results.json"
        result = runner.invoke(
            cli,
            [
                "benchmark",
                "--dataset",
                "weather",
                "--model",
                "all",
                "--no-auto-tune",
                "--samples",
                "200",
                "--output",
                str(output),
            ],
        )
        assert result.exit_code == 0
        assert output.exists()

        # Verify JSON structure
        with open(output) as f:
            data = json.load(f)
            assert "dataset" in data
            assert "models" in data

    @pytest.mark.integration
    @pytest.mark.optional
    @pytest.mark.skipif(
        not DATASETS_AVAILABLE, reason="datasets optional dependency not installed"
    )
    def test_runs_benchmark_with_specific_models(self, tmp_path, monkeypatch):
        """Should run benchmark with specific models."""
        monkeypatch.setenv("UNCERTAINTY_FLOW_HF_REVISION", "main")
        output = tmp_path / "results.json"
        result = runner.invoke(
            cli,
            [
                "benchmark",
                "--dataset",
                "weather",
                "--model",
                "quantile-forest,conformal-regressor",
                "--no-auto-tune",
                "--samples",
                "200",
                "--output",
                str(output),
            ],
        )
        assert result.exit_code == 0

    def test_invalid_model_shows_valid_options(self):
        """Should show valid model names on invalid model."""
        result = runner.invoke(
            cli,
            ["benchmark", "--dataset", "weather", "--model", "invalid-model"],
        )
        assert result.exit_code == 1
        assert "Valid options:" in result.output
        assert "quantile-forest" in result.output


class TestTune:
    """Test tune command."""

    @pytest.mark.integration
    @pytest.mark.optional
    @pytest.mark.skipif(
        not DATASETS_AVAILABLE, reason="datasets optional dependency not installed"
    )
    def test_runs_tuning_with_all_models(self, tmp_path):
        """Should run auto-tuning and save results."""
        output = tmp_path / "tuned.json"
        result = runner.invoke(
            cli,
            [
                "tune",
                "--dataset",
                "weather",
                "--model",
                "all",
                "--n-samples",
                "100",
                "--output",
                str(output),
            ],
        )
        assert result.exit_code == 0
        assert output.exists()

        # Verify JSON structure
        with open(output) as f:
            data = json.load(f)
            assert "dataset" in data
            assert "results" in data

    @pytest.mark.integration
    @pytest.mark.optional
    @pytest.mark.skipif(
        not DATASETS_AVAILABLE, reason="datasets optional dependency not installed"
    )
    def test_runs_tuning_with_specific_model(self, tmp_path):
        """Should run tuning on specific model."""
        output = tmp_path / "tuned.json"
        result = runner.invoke(
            cli,
            [
                "tune",
                "--dataset",
                "weather",
                "--model",
                "quantile-forest",
                "--n-samples",
                "100",
                "--output",
                str(output),
            ],
        )
        assert result.exit_code == 0
