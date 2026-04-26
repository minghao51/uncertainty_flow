"""Integration tests for uncertainty_flow.benchmarking module."""

import json
import tempfile
from pathlib import Path

import pytest

from uncertainty_flow.benchmarking.runner import BenchmarkRunner


@pytest.fixture
def sample_benchmark_config():
    """Create a valid benchmark configuration."""
    return {
        "dataset_name": "weather",
        "n_samples": 100,
        "horizon": 3,
        "n_estimators": 30,
        "target_column": "price",
        "auto_tune": False,
        "target_coverage": 0.9,
        "tune_samples": 100,
        "tune_timeout": 300,
    }


class TestBenchmarkRunner:
    """Test BenchmarkRunner integration."""

    def test_load_data_weather(self, sample_benchmark_config):
        """Should load weather dataset."""
        from uncertainty_flow.benchmarking.datasets import load_dataset

        runner = BenchmarkRunner(sample_benchmark_config())
        runner.load_data()

        assert runner.df is not None
        assert len(runner.df) == 100

    def test_load_data_resolves_target(self, sample_benchmark_config):
        """Should auto-resolve target from DatasetInfo."""
        runner = BenchmarkRunner(sample_benchmark_config())
        runner.load_data()

        assert runner.target in ["price", "temperature", "humidity"]

    def test_run_single_model(self, sample_benchmark_config):
        """Should run a single model benchmark."""
        runner = BenchmarkRunner(sample_benchmark_config())
        runner.load_data()

        result = runner.run_model("quantile-forest")
        assert result.model_name == "quantile-forest"
        assert len(result.metrics) > 0

    def test_run_all_models(self, sample_benchmark_config):
        """Should run all models."""
        runner = BenchmarkRunner(sample_benchmark_config())
        runner.load_data()

        results = runner.run_all()
        assert len(results) > 0

    def test_save_results_json(self, sample_benchmark_config, tmp_path):
        """Should save results to JSON file."""
        runner = BenchmarkRunner(sample_benchmark_config())
        runner.load_data()
        runner.run_all()

        output_file = tmp_path / "results.json"
        runner.save_results(output_file)

        assert output_file.exists()
        with open(output_file) as f:
            data = json.load(f)
            assert "dataset" in data
            assert "results" in data

    def test_auto_tune_integration(self, sample_benchmark_config):
        """Should integrate with auto_tune_model."""
        runner = BenchmarkRunner(
            {**sample_benchmark_config(), "auto_tune": True}
        )
        runner.load_data()

        assert "_tuning_cache" in runner._tuning_cache

        result = runner.run_model("quantile-forest")
        assert result.best_params is not None
        assert "tuned" in result.model_name
