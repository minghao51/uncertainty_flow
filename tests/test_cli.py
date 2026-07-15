"""Tests for CLI commands."""

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import polars as pl
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
    @pytest.mark.skipif(not DATASETS_AVAILABLE, reason="datasets optional dependency not installed")
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
    @pytest.mark.skipif(not DATASETS_AVAILABLE, reason="datasets optional dependency not installed")
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
            assert "manifest" in data
            assert "model_results" in data

    @pytest.mark.integration
    @pytest.mark.optional
    @pytest.mark.skipif(not DATASETS_AVAILABLE, reason="datasets optional dependency not installed")
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

    def test_pipeline_native_result_is_rendered_without_legacy_runner(self, tmp_path, monkeypatch):
        """The package benchmark command emits the canonical typed result shape."""

        frame = pl.DataFrame(
            {
                "feature": [float(i % 5) for i in range(150)],
                "y": [float(i) + 0.25 for i in range(150)],
            }
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "uncertainty_flow.cli.load_dataset",
            lambda *args, **kwargs: (
                frame,
                SimpleNamespace(domain="Synthetic", default_target="y"),
            ),
        )

        output = tmp_path / "typed-result.json"
        result = runner.invoke(
            cli,
            [
                "benchmark",
                "--dataset",
                "fixture",
                "--model",
                "conformal-regressor",
                "--no-auto-tune",
                "--samples",
                "150",
                "--output",
                str(output),
            ],
        )

        assert result.exit_code == 0, result.output
        payload = json.loads(output.read_text())
        assert "model_results" in payload
        assert "models" not in payload
        assert payload["model_results"][0]["provider"] == "conformal-regressor"

    def test_output_only_flags_select_the_named_format(self, tmp_path, monkeypatch):
        frame = pl.DataFrame(
            {
                "feature": [float(i % 5) for i in range(150)],
                "y": [float(i) for i in range(150)],
            }
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "uncertainty_flow.cli.load_dataset",
            lambda *args, **kwargs: (
                frame,
                SimpleNamespace(domain="Synthetic", default_target="y"),
            ),
        )

        json_result = runner.invoke(
            cli,
            [
                "benchmark",
                "--dataset",
                "fixture-json",
                "--model",
                "conformal-regressor",
                "--no-auto-tune",
                "--output",
                "json-result",
                "--json-only",
            ],
        )
        csv_result = runner.invoke(
            cli,
            [
                "benchmark",
                "--dataset",
                "fixture-csv",
                "--model",
                "conformal-regressor",
                "--no-auto-tune",
                "--output",
                "csv-result",
                "--csv-only",
            ],
        )
        invalid = runner.invoke(
            cli,
            [
                "benchmark",
                "--dataset",
                "fixture",
                "--json-only",
                "--csv-only",
            ],
        )

        assert json_result.exit_code == 0, json_result.output
        assert (tmp_path / "json-result.json").is_file()
        assert not (tmp_path / "json-result.csv").exists()
        assert csv_result.exit_code == 0, csv_result.output
        assert (tmp_path / "csv-result.csv").is_file()
        assert not (tmp_path / "csv-result.json").exists()
        assert invalid.exit_code == 2
        assert "mutually exclusive" in invalid.output


class TestTune:
    """Test tune command."""

    @pytest.mark.integration
    @pytest.mark.optional
    @pytest.mark.skipif(not DATASETS_AVAILABLE, reason="datasets optional dependency not installed")
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
    @pytest.mark.skipif(not DATASETS_AVAILABLE, reason="datasets optional dependency not installed")
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
