"""Integration tests for uncertainty_flow.benchmarking.runner."""

import json

import polars as pl
import pytest

from uncertainty_flow.benchmarking.runner import BenchmarkConfig, BenchmarkRunner


def _dataset_info(name: str):
    return type(
        "DatasetInfo",
        (),
        {"name": name, "domain": "Test", "default_target": "OT"},
    )()


@pytest.fixture
def sample_benchmark_config() -> BenchmarkConfig:
    """Create a valid benchmark configuration."""
    return BenchmarkConfig(
        dataset_name="weather",
        n_samples=200,
        horizon=3,
        n_estimators=10,
        target_column="OT",
        auto_tune=False,
        target_coverage=0.9,
        tune_samples=100,
        tune_timeout=30,
        test_size=0.2,
    )


@pytest.fixture
def mock_dataset() -> pl.DataFrame:
    """Local deterministic dataset to avoid network in tests."""
    n = 240
    return pl.DataFrame(
        {
            "date": list(range(n)),
            "feature": [float(i % 7) for i in range(n)],
            "OT": [100.0 + i * 0.3 for i in range(n)],
        }
    )


class TestBenchmarkRunner:
    """Test BenchmarkRunner integration."""

    def test_load_data_weather(self, sample_benchmark_config, mock_dataset, monkeypatch):
        """Should load dataset through loader hook."""

        def _fake_load_dataset(name, n_samples=None, split="train", **kwargs):
            del split, kwargs
            df = mock_dataset
            if n_samples is not None:
                df = df.head(n_samples)
            return df, _dataset_info(name)

        monkeypatch.setattr("uncertainty_flow.benchmarking.runner.load_dataset", _fake_load_dataset)

        runner = BenchmarkRunner(sample_benchmark_config)
        runner.load_data()

        assert runner.df is not None
        assert len(runner.df) == 200
        assert runner.target == "OT"

    def test_run_single_model(self, sample_benchmark_config, mock_dataset, monkeypatch):
        """Should run a single model benchmark."""

        def _fake_load_dataset(name, n_samples=None, split="train", **kwargs):
            del split, kwargs
            df = mock_dataset
            if n_samples is not None:
                df = df.head(n_samples)
            return df, _dataset_info(name)

        monkeypatch.setattr("uncertainty_flow.benchmarking.runner.load_dataset", _fake_load_dataset)

        runner = BenchmarkRunner(sample_benchmark_config)
        runner.load_data()

        result = runner.run_model("quantile-forest")
        assert result.model_name == "quantile-forest"
        assert 0 <= result.coverage_90 <= 1

    def test_run_all_models(self, sample_benchmark_config, mock_dataset, monkeypatch):
        """Should run all registered models."""

        def _fake_load_dataset(name, n_samples=None, split="train", **kwargs):
            del split, kwargs
            df = mock_dataset
            if n_samples is not None:
                df = df.head(n_samples)
            return df, _dataset_info(name)

        monkeypatch.setattr("uncertainty_flow.benchmarking.runner.load_dataset", _fake_load_dataset)

        runner = BenchmarkRunner(sample_benchmark_config)
        runner.load_data()

        result = runner.run_all(model_names=["quantile-forest", "conformal-regressor"])
        assert len(result.models) > 0

    def test_save_results_json(self, sample_benchmark_config, mock_dataset, monkeypatch, tmp_path):
        """Should save results to JSON file."""

        def _fake_load_dataset(name, n_samples=None, split="train", **kwargs):
            del split, kwargs
            df = mock_dataset
            if n_samples is not None:
                df = df.head(n_samples)
            return df, _dataset_info(name)

        monkeypatch.setattr("uncertainty_flow.benchmarking.runner.load_dataset", _fake_load_dataset)

        runner = BenchmarkRunner(sample_benchmark_config)
        runner.load_data()
        runner.run_all(model_names=["quantile-forest"])

        output_file = tmp_path / "results.json"
        runner.save_json(output_file)

        assert output_file.exists()
        with open(output_file, encoding="utf-8") as handle:
            data = json.load(handle)
        assert "metadata" in data
        assert "results" in data
        assert "models" in data
        assert data["models"] == data["results"]
        assert "test_split_type" in data["results"][0]
        assert "validation_split_type" in data["results"][0]

    def test_tuning_uses_train_split_only(self, sample_benchmark_config, mock_dataset, monkeypatch):
        seen = {"rows": None}

        def _fake_load_dataset(name, n_samples=None, split="train", **kwargs):
            del split, kwargs
            df = mock_dataset
            if n_samples is not None:
                df = df.head(n_samples)
            return df, _dataset_info(name)

        def _fake_auto_tune_model(model_name, df, target, horizon, config):
            del model_name, target, horizon, config
            seen["rows"] = len(df)
            return type("TuneResult", (), {"best_params": {"n_estimators": 10}})()

        monkeypatch.setattr("uncertainty_flow.benchmarking.runner.load_dataset", _fake_load_dataset)
        monkeypatch.setattr(
            "uncertainty_flow.benchmarking.runner.auto_tune_model", _fake_auto_tune_model
        )

        sample_benchmark_config.auto_tune = True
        sample_benchmark_config.test_size = 0.2
        runner = BenchmarkRunner(sample_benchmark_config)
        runner.load_data()
        runner.run_model("quantile-forest")

        assert seen["rows"] == int(len(runner.df) * (1 - sample_benchmark_config.test_size))
