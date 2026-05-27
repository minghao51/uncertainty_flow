import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from uncertainty_flow.core.distribution import DistributionPrediction


def _make_prediction(
    n: int, quantile_levels: list[float], target_names: list[str] | None = None
) -> DistributionPrediction:
    if target_names is None:
        target_names = ["y"]
    n_targets = len(target_names)
    n_quantiles = len(quantile_levels)
    rng = np.random.default_rng(0)
    raw = rng.standard_normal((n, n_quantiles))
    quantile_matrix = np.sort(raw, axis=1)
    if n_targets > 1:
        parts = [quantile_matrix]
        for _ in range(n_targets - 1):
            extra = rng.standard_normal((n, n_quantiles))
            parts.append(np.sort(extra, axis=1))
        quantile_matrix = np.hstack(parts)
    return DistributionPrediction(
        quantile_matrix=quantile_matrix,
        quantile_levels=quantile_levels,
        target_names=target_names,
    )


@st.composite
def sorted_floats(draw, min_size=3, max_size=10, min_val=0.01, max_val=0.99):
    vals = draw(
        st.lists(
            st.floats(min_val, max_val, allow_nan=False, allow_infinity=False),
            min_size=min_size,
            max_size=max_size,
        )
    )
    unique = sorted(set(round(v, 3) for v in vals if round(v, 3) > 0 and round(v, 3) < 1))
    if len(unique) < 3:
        unique = [0.1, 0.25, 0.5, 0.75, 0.9]
    return unique


@given(
    n=st.integers(5, 50),
    quantile_levels=sorted_floats(min_size=3, max_size=10),
)
@settings(deadline=None, max_examples=30)
def test_quantile_monotonicity(n, quantile_levels):
    pred = _make_prediction(n, quantile_levels)
    df = pred.quantile(quantile_levels)
    arr = df.to_numpy()
    for i in range(n):
        for j in range(len(quantile_levels) - 1):
            assert arr[i, j] <= arr[i, j + 1] + 1e-10


@given(
    n=st.integers(5, 50),
    confidence=st.floats(0.1, 0.99, allow_nan=False, allow_infinity=False),
    quantile_levels=sorted_floats(min_size=4, max_size=10),
)
@settings(deadline=None, max_examples=30)
def test_interval_width_positive(n, confidence, quantile_levels):
    pred = _make_prediction(n, quantile_levels)
    interval_df = pred.interval(confidence)
    lower = interval_df["lower"].to_numpy()
    upper = interval_df["upper"].to_numpy()
    assert np.all(upper - lower >= -1e-10)


@given(
    n=st.integers(5, 50),
    confidence=st.floats(0.1, 0.99, allow_nan=False, allow_infinity=False),
    quantile_levels=sorted_floats(min_size=4, max_size=10),
)
@settings(deadline=None, max_examples=30)
def test_coverage_in_bounds(n, confidence, quantile_levels):
    from uncertainty_flow.metrics.coverage import coverage_score

    pred = _make_prediction(n, quantile_levels)
    interval_df = pred.interval(confidence)
    lower = interval_df["lower"].to_numpy()
    upper = interval_df["upper"].to_numpy()
    rng = np.random.default_rng(0)
    y_true = rng.standard_normal(n)
    cov = coverage_score(y_true, lower, upper)
    assert 0.0 <= cov <= 1.0


@given(
    n=st.integers(5, 50),
    quantile_levels=sorted_floats(min_size=3, max_size=10),
)
@settings(deadline=None, max_examples=30)
def test_median_between_extremes(n, quantile_levels):
    pred = _make_prediction(n, quantile_levels)
    median_s = pred.median()
    if hasattr(median_s, "to_numpy"):
        median = median_s.to_numpy()
    else:
        median = median_s.to_numpy()
    lower_df = pred.quantile(quantile_levels[0])
    upper_df = pred.quantile(quantile_levels[-1])
    lower = lower_df.to_numpy().ravel()
    upper = upper_df.to_numpy().ravel()
    assert np.all(median >= lower - 1e-10)
    assert np.all(median <= upper + 1e-10)


@given(
    n=st.integers(5, 30),
    n_targets=st.integers(2, 3),
    quantile_levels=sorted_floats(min_size=3, max_size=7),
)
@settings(deadline=None, max_examples=20)
def test_multivariate_interval_width_positive(n, n_targets, quantile_levels):
    target_names = [f"t{i}" for i in range(n_targets)]
    pred = _make_prediction(n, quantile_levels, target_names=target_names)
    interval_df = pred.interval(0.9)
    for t in target_names:
        lower = interval_df[f"{t}_lower"].to_numpy()
        upper = interval_df[f"{t}_upper"].to_numpy()
        assert np.all(upper - lower >= -1e-10)
