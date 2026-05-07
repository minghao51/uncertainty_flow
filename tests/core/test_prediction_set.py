import numpy as np
import polars as pl
import pytest

from uncertainty_flow.core.prediction_set import PredictionSet


@pytest.fixture
def pred_set():
    return PredictionSet(
        class_sets=[["a", "b"], ["a"], ["b", "c"]],
        class_names=["a", "b", "c"],
        probabilities=np.array([[0.4, 0.35, 0.25], [0.7, 0.2, 0.1], [0.1, 0.5, 0.4]]),
        coverage_target=0.9,
        threshold=0.6,
    )


class TestPredictionSet:
    def test_set_single_index(self, pred_set):
        assert pred_set.set(0) == ["a", "b"]

    def test_set_all(self, pred_set):
        all_sets = pred_set.set()
        assert len(all_sets) == 3

    def test_coverage_property(self, pred_set):
        assert pred_set.coverage == 0.9

    def test_size_property(self, pred_set):
        expected = np.mean([2, 1, 2])
        assert pred_set.size == pytest.approx(expected)

    def test_size_by_sample(self, pred_set):
        assert pred_set.size_by_sample() == [2, 1, 2]

    def test_probabilities(self, pred_set):
        df = pred_set.probabilities()
        assert isinstance(df, pl.DataFrame)
        assert df.shape == (3, 3)
        assert df.columns == ["class_a", "class_b", "class_c"]

    def test_summary(self, pred_set):
        df = pred_set.summary()
        assert isinstance(df, pl.DataFrame)
        assert df.shape[0] == 1
        assert "coverage_target" in df.columns
        assert "avg_set_size" in df.columns
        assert "n_samples" in df.columns
        assert "n_classes" in df.columns
        assert df["coverage_target"][0] == 0.9

    def test_repr(self, pred_set):
        r = repr(pred_set)
        assert "PredictionSet" in r
        assert "n_samples=3" in r
        assert "n_classes=3" in r

    def test_n_samples(self, pred_set):
        assert pred_set._n_samples == 3

    def test_n_classes(self, pred_set):
        assert pred_set._n_classes == 3
