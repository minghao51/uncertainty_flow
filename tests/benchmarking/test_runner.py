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

        monkeypatch.setattr("uncertainty_flow.benchmarking.flow.load_dataset", _fake_load_dataset)

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

        monkeypatch.setattr("uncertainty_flow.benchmarking.flow.load_dataset", _fake_load_dataset)

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

        monkeypatch.setattr("uncertainty_flow.benchmarking.flow.load_dataset", _fake_load_dataset)

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

        monkeypatch.setattr("uncertainty_flow.benchmarking.flow.load_dataset", _fake_load_dataset)

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
        assert "models" not in data
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

        monkeypatch.setattr("uncertainty_flow.benchmarking.flow.load_dataset", _fake_load_dataset)
        monkeypatch.setattr(
            "uncertainty_flow.benchmarking.flow.auto_tune_model", _fake_auto_tune_model
        )

        sample_benchmark_config.auto_tune = True
        sample_benchmark_config.test_size = 0.2
        runner = BenchmarkRunner(sample_benchmark_config)
        runner.load_data()
        runner.run_model("quantile-forest")

        n_total = len(runner.df)
        n_test = int(n_total * sample_benchmark_config.test_size)
        n_non_test = n_total - n_test
        n_tune = max(1, int(n_non_test * sample_benchmark_config.tune_size))
        assert seen["rows"] == n_tune
        assert seen["rows"] < n_non_test


class TestTrainTestSplit:
    """Tests for BenchmarkFlow._train_test_split 3-way split."""

    @staticmethod
    def _make_flow(test_size=0.2, tune_size=0.2):
        from uncertainty_flow.benchmarking.configs import BenchmarkConfig
        from uncertainty_flow.benchmarking.flow import BenchmarkFlow

        config = BenchmarkConfig(
            dataset_name="test",
            test_size=test_size,
            tune_size=tune_size,
            auto_tune=False,
        )
        return BenchmarkFlow(config=config, providers={}, class_registry={})

    def test_three_way_split_sizes(self):
        df = pl.DataFrame({"x": list(range(100)), "y": list(range(100))})
        flow = self._make_flow(test_size=0.2, tune_size=0.25)
        tune_df, train_df, test_df = flow._train_test_split(df)

        assert len(test_df) == 20
        assert len(tune_df) + len(train_df) == 80
        assert len(tune_df) == 20
        assert len(train_df) == 60

    def test_tune_and_train_are_disjoint(self):
        df = pl.DataFrame({"x": list(range(100)), "y": list(range(100))})
        flow = self._make_flow(test_size=0.2, tune_size=0.25)
        tune_df, train_df, test_df = flow._train_test_split(df)

        tune_vals = set(tune_df["x"].to_list())
        train_vals = set(train_df["x"].to_list())
        test_vals = set(test_df["x"].to_list())

        assert tune_vals.isdisjoint(train_vals)
        assert tune_vals.isdisjoint(test_vals)
        assert train_vals.isdisjoint(test_vals)

    def test_preserves_ordering(self):
        df = pl.DataFrame({"x": list(range(100))})
        flow = self._make_flow(test_size=0.2, tune_size=0.25)
        tune_df, train_df, test_df = flow._train_test_split(df)

        assert tune_df["x"].to_list() == list(range(20))
        assert train_df["x"].to_list() == list(range(20, 80))
        assert test_df["x"].to_list() == list(range(80, 100))

    def test_raises_on_too_few_rows(self):
        df = pl.DataFrame({"x": [1], "y": [2]})
        flow = self._make_flow()
        with pytest.raises(Exception, match="at least 3 rows"):
            flow._train_test_split(df)

    def test_tune_size_clamped_when_too_large(self):
        df = pl.DataFrame({"x": list(range(10)), "y": list(range(10))})
        flow = self._make_flow(test_size=0.2, tune_size=0.99)
        tune_df, train_df, test_df = flow._train_test_split(df)

        assert len(tune_df) >= 1
        assert len(train_df) >= 1
        assert len(tune_df) + len(train_df) + len(test_df) == 10
