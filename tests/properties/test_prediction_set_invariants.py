import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from uncertainty_flow.core.prediction_set import PredictionSet
from uncertainty_flow.utils.exceptions import InvalidDataError


def _make_prediction_set(
    n_classes: int, n_samples: int, coverage_target: float, threshold: float, seed: int = 0
) -> PredictionSet:
    rng = np.random.default_rng(seed)
    class_names = [f"c{i}" for i in range(n_classes)]
    probs = rng.dirichlet(np.ones(n_classes), size=n_samples)
    class_sets = []
    for row in probs:
        order = np.argsort(-row)
        n_pick = rng.integers(1, n_classes + 1)
        class_sets.append([class_names[i] for i in order[:n_pick]])
    return PredictionSet(
        class_sets=class_sets,
        class_names=class_names,
        probabilities=probs,
        coverage_target=coverage_target,
        threshold=threshold,
    )


@given(
    n_classes=st.integers(2, 10),
    n_samples=st.integers(1, 20),
    coverage_target=st.floats(0.01, 0.99, allow_nan=False, allow_infinity=False),
    threshold=st.floats(0.0, 2.0, allow_nan=False, allow_infinity=False),
)
@settings(deadline=None, max_examples=40)
def test_prediction_set_valid_construction(n_classes, n_samples, coverage_target, threshold):
    ps = _make_prediction_set(n_classes, n_samples, coverage_target, threshold)
    assert ps.size > 0
    assert ps.coverage == coverage_target
    sets = ps.prediction_sets()
    assert len(sets) == n_samples


@given(
    n_classes=st.integers(2, 8),
    n_samples=st.integers(1, 10),
)
@settings(deadline=None, max_examples=30)
def test_prediction_set_probabilities_sum_to_one(n_classes, n_samples):
    ps = _make_prediction_set(n_classes, n_samples, 0.9, 0.5)
    prob_df = ps.probabilities()
    sums = prob_df.to_numpy().sum(axis=1)
    np.testing.assert_allclose(sums, 1.0, atol=1e-10)


@given(
    n_classes=st.integers(2, 8),
    n_samples=st.integers(1, 15),
)
@settings(deadline=None, max_examples=30)
def test_prediction_set_sizes_in_range(n_classes, n_samples):
    ps = _make_prediction_set(n_classes, n_samples, 0.9, 0.5)
    sizes = ps.size_by_sample()
    for s in sizes:
        assert 1 <= s <= n_classes


@given(n_classes=st.integers(2, 6))
@settings(deadline=None, max_examples=20)
def test_prediction_set_summary_shape(n_classes):
    ps = _make_prediction_set(n_classes, 10, 0.9, 0.5)
    summary = ps.summary()
    assert len(summary) == 1
    assert "coverage_target" in summary.columns
    assert "avg_set_size" in summary.columns
    assert "n_samples" in summary.columns
    assert "n_classes" in summary.columns


def test_prediction_set_rejects_empty_class_sets():
    with pytest.raises(InvalidDataError, match="non-empty"):
        PredictionSet(
            class_sets=[],
            class_names=["a", "b"],
            probabilities=np.array([[0.5, 0.5]]),
            coverage_target=0.9,
            threshold=0.5,
        )


def test_prediction_set_rejects_nan_probs():
    with pytest.raises(InvalidDataError, match="NaN"):
        PredictionSet(
            class_sets=[["a"]],
            class_names=["a", "b"],
            probabilities=np.array([[float("nan"), 0.5]]),
            coverage_target=0.9,
            threshold=0.5,
        )


def test_prediction_set_rejects_bad_coverage():
    with pytest.raises(InvalidDataError, match="coverage_target"):
        PredictionSet(
            class_sets=[["a"]],
            class_names=["a", "b"],
            probabilities=np.array([[0.5, 0.5]]),
            coverage_target=0.0,
            threshold=0.5,
        )


def test_prediction_set_rejects_negative_threshold():
    with pytest.raises(InvalidDataError, match="non-negative"):
        PredictionSet(
            class_sets=[["a"]],
            class_names=["a", "b"],
            probabilities=np.array([[0.5, 0.5]]),
            coverage_target=0.9,
            threshold=-0.1,
        )


def test_prediction_set_rejects_mismatched_lengths():
    with pytest.raises(InvalidDataError, match="class_sets length"):
        PredictionSet(
            class_sets=[["a"], ["b"]],
            class_names=["a", "b"],
            probabilities=np.array([[0.5, 0.5]]),
            coverage_target=0.9,
            threshold=0.5,
        )
