import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from uncertainty_flow.core.distribution import DistributionPrediction
from uncertainty_flow.core.distribution_scoring import crps_score, log_score


def _make_prediction_and_truth(
    n: int, n_quantiles: int, seed: int = 0
) -> tuple[DistributionPrediction, np.ndarray]:
    rng = np.random.default_rng(seed)
    quantile_levels = sorted(rng.uniform(0.05, 0.95, size=n_quantiles).tolist())
    raw = rng.standard_normal((n, n_quantiles))
    quantile_matrix = np.sort(raw, axis=1)
    pred = DistributionPrediction(
        quantile_matrix=quantile_matrix,
        quantile_levels=quantile_levels,
        target_names=["y"],
    )
    y_true = rng.standard_normal(n)
    return pred, y_true


@given(
    n=st.integers(10, 60),
    n_quantiles=st.integers(4, 15),
    seed=st.integers(0, 100),
)
@settings(deadline=None, max_examples=40)
def test_crps_non_negative(n, n_quantiles, seed):
    pred, y_true = _make_prediction_and_truth(n, n_quantiles, seed)
    score = crps_score(pred, y_true)
    assert score >= -1e-10


@given(
    n=st.integers(10, 50),
    n_quantiles=st.integers(4, 12),
    seed=st.integers(0, 50),
)
@settings(deadline=None, max_examples=30)
def test_log_score_real_valued(n, n_quantiles, seed):
    pred, y_true = _make_prediction_and_truth(n, n_quantiles, seed)
    score = log_score(pred, y_true)
    assert np.isfinite(score)


@given(
    n=st.integers(10, 40),
    n_quantiles=st.integers(4, 10),
    seed=st.integers(0, 30),
)
@settings(deadline=None, max_examples=20)
def test_crps_zero_for_perfect_prediction(n, n_quantiles, seed):
    rng = np.random.default_rng(seed)
    quantile_levels = sorted(rng.uniform(0.05, 0.95, size=n_quantiles).tolist())
    y_true = rng.standard_normal(n)
    quantile_matrix = np.tile(y_true[:, None], (1, n_quantiles))
    pred = DistributionPrediction(
        quantile_matrix=quantile_matrix,
        quantile_levels=quantile_levels,
        target_names=["y"],
    )
    score = crps_score(pred, y_true)
    assert score < 0.1


@given(
    n=st.integers(10, 50),
    n_quantiles=st.integers(4, 12),
    seed=st.integers(0, 50),
)
@settings(deadline=None, max_examples=30)
def test_energy_score_non_negative(n, n_quantiles, seed):
    from uncertainty_flow.core.distribution_scoring import energy_score

    pred, y_true = _make_prediction_and_truth(n, n_quantiles, seed)
    target_names = ["t0", "t1"]
    rng = np.random.default_rng(seed)
    quantile_levels = sorted(rng.uniform(0.05, 0.95, size=n_quantiles).tolist())
    raw0 = rng.standard_normal((n, n_quantiles))
    raw1 = rng.standard_normal((n, n_quantiles))
    quantile_matrix = np.hstack([np.sort(raw0, axis=1), np.sort(raw1, axis=1)])
    pred_multi = DistributionPrediction(
        quantile_matrix=quantile_matrix,
        quantile_levels=quantile_levels,
        target_names=target_names,
    )
    y_multi = np.column_stack([rng.standard_normal(n), rng.standard_normal(n)])
    score = energy_score(pred_multi, y_multi, n_samples=50, random_state=seed)
    assert np.isfinite(score)
