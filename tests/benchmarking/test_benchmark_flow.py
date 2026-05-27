import numpy as np
import polars as pl
import pytest

from uncertainty_flow.benchmarking.configs import BenchmarkConfig
from uncertainty_flow.benchmarking.flow import BenchmarkFlow, LoadedDataset, _compute_fold_metrics
from uncertainty_flow.benchmarking.providers import get_default_providers
from uncertainty_flow.core.distribution import DistributionPrediction
from uncertainty_flow.utils.exceptions import DataError


def _sample_df(n: int = 100, seed: int = 42) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    return pl.DataFrame(
        {
            "date": np.arange(n),
            "price": 10 + np.arange(n) * 0.5 + rng.standard_normal(n) * 0.5,
        }
    )


def _make_pred(n: int = 20, seed: int = 0) -> DistributionPrediction:
    rng = np.random.default_rng(seed)
    levels = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    raw = rng.standard_normal((n, len(levels)))
    qm = np.sort(raw, axis=1)
    return DistributionPrediction(
        quantile_matrix=qm,
        quantile_levels=levels,
        target_names=["price"],
    )


class TestComputeFoldMetrics:
    def test_returns_expected_keys(self):
        pred = _make_pred(20)
        rng = np.random.default_rng(0)
        y_true = rng.standard_normal(20)
        metrics = _compute_fold_metrics(pred, y_true)
        assert "cov_90" in metrics
        assert "cov_80" in metrics
        assert "wink_90" in metrics
        assert "wink_80" in metrics
        assert "sharp_90" in metrics
        assert "sharp_80" in metrics
        assert "pinball" in metrics

    def test_coverage_in_range(self):
        pred = _make_pred(20)
        rng = np.random.default_rng(1)
        y_true = rng.standard_normal(20)
        metrics = _compute_fold_metrics(pred, y_true)
        assert 0.0 <= metrics["cov_90"] <= 1.0
        assert 0.0 <= metrics["cov_80"] <= 1.0


class TestLoadedDataset:
    def test_fields(self):
        from uncertainty_flow.benchmarking.datasets import DatasetInfo

        df = _sample_df(50)
        info = DatasetInfo(
            name="test",
            hf_path="",
            subset=None,
            domain="Test",
            description="test",
            default_target="price",
            is_local=True,
        )
        loaded = LoadedDataset(df=df, ds_info=info, target="price")
        assert len(loaded.df) == 50
        assert loaded.target == "price"


class TestBenchmarkFlowTrainTestSplit:
    def test_split_sizes(self):
        cfg = BenchmarkConfig(
            dataset_name="weather",
            n_samples=100,
            test_size=0.2,
            tune_size=0.2,
            auto_tune=False,
        )
        providers = get_default_providers()
        flow = BenchmarkFlow(config=cfg, providers=providers, class_registry={})
        df = _sample_df(100)
        tune, train, test = flow._train_test_split(df)
        assert len(test) == 20
        assert len(tune) > 0
        assert len(train) > 0
        assert len(tune) + len(train) + len(test) == 100

    def test_small_dataset_raises(self):
        cfg = BenchmarkConfig(
            dataset_name="weather",
            n_samples=3,
            test_size=0.33,
            auto_tune=False,
        )
        providers = get_default_providers()
        flow = BenchmarkFlow(config=cfg, providers=providers, class_registry={})
        df = _sample_df(3)
        with pytest.raises(DataError):
            flow._train_test_split(df)


class TestBenchmarkFlowTuningAttr:
    def test_none_returns_none(self):
        assert BenchmarkFlow._tuning_attr(None, "coverage_90") is None

    def test_missing_attr_returns_none(self):
        class Obj:
            pass

        assert BenchmarkFlow._tuning_attr(Obj(), "coverage_90") is None

    def test_present_attr_returns_value(self):
        class Obj:
            coverage_90 = 0.85

        assert BenchmarkFlow._tuning_attr(Obj(), "coverage_90") == 0.85

    def test_tuning_attr_str(self):
        assert BenchmarkFlow._tuning_attr_str(None, "x") is None

        class Obj:
            x = "hello"

        assert BenchmarkFlow._tuning_attr_str(Obj(), "x") == "hello"

    def test_tuning_attr_int(self):
        assert BenchmarkFlow._tuning_attr_int(None, "n") is None

        class Obj:
            n = 5

        assert BenchmarkFlow._tuning_attr_int(Obj(), "n") == 5
