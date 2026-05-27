import json

import polars as pl

from uncertainty_flow.benchmarking.results import BenchmarkResult, ModelResult
from uncertainty_flow.benchmarking.sinks import ResultSink


def _sample_model_result(name: str = "test-model") -> ModelResult:
    return ModelResult(
        model_name=name,
        coverage_90=0.89,
        coverage_80=0.82,
        sharpness_90=1.5,
        sharpness_80=1.0,
        winkler_90=3.2,
        winkler_80=2.1,
        pinball_loss=0.4,
        train_time_sec=0.5,
        n_samples=100,
        tuned_params={"n_estimators": 30},
        was_tuned=True,
    )


def _sample_benchmark_result() -> BenchmarkResult:
    return BenchmarkResult(
        run_id="abc123",
        timestamp="2025-01-01T00:00:00",
        dataset_name="weather",
        dataset_domain="Climate",
        n_samples=1000,
        horizon=3,
        models=[_sample_model_result()],
        errors=[],
    )


class TestResultSinkToDict:
    def test_none_result_returns_empty(self):
        sink = ResultSink(result=None, test_size=0.2, auto_tune=True, target_coverage=0.9)
        d = sink.to_dict()
        assert d["metadata"] == {}
        assert d["results"] == []

    def test_with_result(self):
        result = _sample_benchmark_result()
        sink = ResultSink(result=result, test_size=0.2, auto_tune=True, target_coverage=0.9)
        d = sink.to_dict()
        assert d["dataset"] == "weather"
        assert d["metadata"]["run_id"] == "abc123"
        assert d["metadata"]["test_size"] == 0.2
        assert len(d["results"]) == 1
        assert d["results"][0]["model"] == "test-model"
        assert d["errors"] == []


class TestResultSinkSaveJson:
    def test_saves_valid_json(self, tmp_path):
        result = _sample_benchmark_result()
        sink = ResultSink(result=result, test_size=0.2, auto_tune=True, target_coverage=0.9)
        path = tmp_path / "results.json"
        sink.save_json(path)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["dataset"] == "weather"

    def test_none_result_saves_empty(self, tmp_path):
        sink = ResultSink(result=None, test_size=0.2, auto_tune=True, target_coverage=0.9)
        path = tmp_path / "empty.json"
        sink.save_json(path)
        data = json.loads(path.read_text())
        assert data["results"] == []


class TestResultSinkSaveCsv:
    def test_saves_csv(self, tmp_path):
        result = _sample_benchmark_result()
        sink = ResultSink(result=result, test_size=0.2, auto_tune=True, target_coverage=0.9)
        path = tmp_path / "results.csv"
        sink.save_csv(path)
        assert path.exists()
        df = pl.read_csv(path)
        assert "model" in df.columns
        assert len(df) == 1

    def test_none_result_creates_no_file(self, tmp_path):
        sink = ResultSink(result=None, test_size=0.2, auto_tune=True, target_coverage=0.9)
        path = tmp_path / "empty.csv"
        sink.save_csv(path)
        assert not path.exists()


class TestResultSinkRow:
    def test_row_contains_all_model_fields(self):
        r = _sample_model_result()
        row = ResultSink._row(r)
        assert row["model"] == "test-model"
        assert row["coverage_90"] == 0.89
        assert row["was_tuned"] is True
        assert row["tuned_params"] == {"n_estimators": 30}
        assert "validation_coverage_90" in row
        assert "test_split_type" in row
