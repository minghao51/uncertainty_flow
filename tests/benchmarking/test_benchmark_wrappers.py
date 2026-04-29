"""Tests for benchmark model wrappers in run_benchmarks.py."""

import numpy as np
import polars as pl
import pytest

from uncertainty_flow.benchmarking.runner import BenchmarkConfig


@pytest.fixture
def sample_df():
    rng = np.random.default_rng(42)
    n = 100
    return pl.DataFrame({
        "feature1": rng.standard_normal(n),
        "feature2": rng.standard_normal(n),
        "target": rng.standard_normal(n) * 0.5 + 1.0,
    })


@pytest.fixture
def config():
    return BenchmarkConfig(
        dataset_name="test",
        n_samples=100,
        horizon=3,
        random_state=42,
    )


class TestBaselineModels:
    def test_linear_regression(self, sample_df, config):
        from benchmarks.run_benchmarks import _register_baselines

        _register_baselines()
        from uncertainty_flow.benchmarking.runner import MODEL_REGISTRY

        cls = MODEL_REGISTRY["linear-regression"]
        model = cls(config)
        model.fit(sample_df, "target")
        pred = model.predict(sample_df)
        interval = pred.interval(0.9)
        assert len(interval) == len(sample_df)
        assert model.train_time > 0

    def test_ridge_regression(self, sample_df, config):
        from benchmarks.run_benchmarks import _register_baselines

        _register_baselines()
        from uncertainty_flow.benchmarking.runner import MODEL_REGISTRY

        cls = MODEL_REGISTRY["ridge-regression"]
        model = cls(config)
        model.fit(sample_df, "target")
        pred = model.predict(sample_df)
        interval = pred.interval(0.9)
        assert len(interval) == len(sample_df)

    def test_random_forest(self, sample_df, config):
        from benchmarks.run_benchmarks import _register_baselines

        _register_baselines()
        from uncertainty_flow.benchmarking.runner import MODEL_REGISTRY

        cls = MODEL_REGISTRY["random-forest"]
        model = cls(config)
        model.fit(sample_df, "target")
        pred = model.predict(sample_df)
        interval = pred.interval(0.9)
        assert len(interval) == len(sample_df)

    def test_naive_forecast(self, sample_df, config):
        from benchmarks.run_benchmarks import _register_baselines

        _register_baselines()
        from uncertainty_flow.benchmarking.runner import MODEL_REGISTRY

        cls = MODEL_REGISTRY["naive-forecast"]
        model = cls(config)
        model.fit(sample_df, "target")
        pred = model.predict(sample_df)
        interval = pred.interval(0.9)
        assert len(interval) == len(sample_df)

    def test_moving_average(self, sample_df, config):
        from benchmarks.run_benchmarks import _register_baselines

        _register_baselines()
        from uncertainty_flow.benchmarking.runner import MODEL_REGISTRY

        cls = MODEL_REGISTRY["moving-average"]
        model = cls(config)
        model.fit(sample_df, "target")
        pred = model.predict(sample_df)
        interval = pred.interval(0.9)
        assert len(interval) == len(sample_df)


class TestPointPrediction:
    def test_get_point_prediction_distribution(self, sample_df, config):
        from benchmarks.run_benchmarks import _get_point_prediction, _register_baselines

        _register_baselines()
        from uncertainty_flow.benchmarking.runner import MODEL_REGISTRY

        cls = MODEL_REGISTRY["linear-regression"]
        model = cls(config)
        model.fit(sample_df, "target")
        pred = model.predict(sample_df)
        point = _get_point_prediction(pred)
        assert isinstance(point, np.ndarray)
        assert len(point) == len(sample_df)

    def test_get_point_prediction_simple(self):
        from benchmarks.run_benchmarks import SimpleDistributionPrediction, _get_point_prediction

        simple = SimpleDistributionPrediction(
            lower_90=np.array([1.0, 2.0]),
            upper_90=np.array([3.0, 4.0]),
            lower_80=np.array([1.5, 2.5]),
            upper_80=np.array([2.5, 3.5]),
        )
        point = _get_point_prediction(simple)
        np.testing.assert_array_equal(point, np.array([2.0, 3.0]))
